// Engine benchmarks: throughput plus HONEST per-event latency percentiles.
//
//   engine_bench ticks [n_ticks]                 tick replay (BarBuilder), synthetic ticks
//   engine_bench ticks-tape <file.tape>          tick replay over a recorded binary tape
//   engine_bench backtest <corpus.bin> [repeats] backtest hot loop over a binary corpus
//
// Methodology (stated because it matters):
// * Throughput is measured over the whole batch with two clock reads — ground truth.
// * Per-event latency times each event individually with steady_clock, so every sample
//   includes ~one clock-pair overhead (reported as clock granularity); percentiles are computed
//   EXACTLY by sorting all samples, not from a lossy histogram.
// * Backtest per-event latency = one full decision (bar search + gates + quote + fill scan +
//   settle) on a single-prediction view; the batch loop is what production uses, the per-event
//   numbers show the distribution shape (fill-scan length dominates the tail).
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "pmlab_engine/bars.hpp"
#include "pmlab_engine/corpus_io.hpp"
#include "pmlab_engine/engine.hpp"
#include "pmlab_engine/tape.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double now_ns_diff(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::nano>(b - a).count();
}

void print_percentiles(std::vector<double>& ns) {
  std::sort(ns.begin(), ns.end());
  const auto pct = [&](double p) { return ns[static_cast<std::size_t>(p * (ns.size() - 1))]; };
  double sum = 0;
  for (const double v : ns) sum += v;
  std::printf("    per-event ns: mean %.0f  p50 %.0f  p90 %.0f  p99 %.0f  p99.9 %.0f  max %.0f\n",
              sum / static_cast<double>(ns.size()), pct(0.50), pct(0.90), pct(0.99), pct(0.999),
              ns.back());
}

// Smallest positive difference two clock reads can show. On Apple Silicon steady_clock ticks
// at 24MHz (~41.7ns), so per-event samples are quantized to multiples of it; percentiles at or
// below this are "faster than the clock can see", not zero cost.
double clock_granularity_ns() {
  double best = 1e18;
  auto prev = Clock::now();
  for (int i = 0; i < 100000; ++i) {
    const auto t = Clock::now();
    const double d = now_ns_diff(prev, t);
    if (d > 0 && d < best) best = d;
    prev = t;
  }
  return best;
}

std::vector<pmlab::BookTick> synthetic_ticks(std::size_t n) {
  std::vector<pmlab::BookTick> ticks;
  ticks.reserve(n);
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  std::int64_t ts = 1'700'000'000;
  for (std::size_t i = 0; i < n; ++i) {
    ts += static_cast<std::int64_t>(next() % 3);  // bursty: several ticks per second
    const double bid = static_cast<double>(next() % 95) / 100.0;
    ticks.push_back({ts, bid, bid + static_cast<double>(1 + next() % 5) / 100.0});
  }
  return ticks;
}

int bench_ticks(std::span<const pmlab::BookTick> ticks) {
  const std::int64_t start = ticks.front().ts;
  const std::int64_t end = ticks.back().ts;

  // Throughput: whole replay, two clock reads.
  const auto t0 = Clock::now();
  const auto bars = pmlab::assemble_bars(ticks, start, end);
  const auto t1 = Clock::now();
  const double s = now_ns_diff(t0, t1) / 1e9;
  std::printf("  replay: %zu ticks -> %zu bars in %.3fs  (%.2fM ticks/s)\n", ticks.size(),
              bars.size(), s, static_cast<double>(ticks.size()) / s / 1e6);

  // Per-event latency: BarBuilder::add per tick, close_minute at boundaries (as live capture
  // would call it).
  pmlab::BarBuilder builder;
  std::vector<double> ns;
  ns.reserve(ticks.size());
  std::int64_t boundary = pmlab::bucket_end(start);
  for (const auto& tick : ticks) {
    while (tick.ts > boundary) {
      builder.close_minute(boundary);
      boundary += 60;
    }
    const auto a = Clock::now();
    builder.add(tick);
    const auto b = Clock::now();
    ns.push_back(now_ns_diff(a, b));
  }
  print_percentiles(ns);
  std::printf("    clock granularity: %.0f ns (samples are quantized to it)\n", clock_granularity_ns());
  return 0;
}

int bench_backtest(const char* corpus_path, long repeats) {
  const pmlab::Corpus corpus = pmlab::read_corpus(corpus_path);
  const std::size_t n = corpus.n_predictions();
  pmlab::Results results(n);
  const auto out = results.outputs_view();
  const auto bars = corpus.bars_view();
  const auto preds = corpus.predictions_view();

  std::vector<double> run_us(static_cast<std::size_t>(repeats));
  for (auto& sample : run_us) {
    const auto t0 = Clock::now();
    pmlab::run_backtest(bars, preds, corpus.params, out);
    const auto t1 = Clock::now();
    sample = now_ns_diff(t0, t1) / 1e3;
  }
  std::sort(run_us.begin(), run_us.end());
  const double median_us = run_us[run_us.size() / 2];
  std::printf("  hot loop: %zu markets, %zu bars -> median %.0f us over %ld runs "
              "(%.2fM markets/s)\n",
              n, corpus.bar_ts.size(), median_us, repeats,
              static_cast<double>(n) / median_us);

  // Per-event latency: one decision at a time through single-row views.
  pmlab::Results one_out(1);
  const auto one = one_out.outputs_view();
  std::vector<double> ns;
  ns.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const pmlab::PredictionsView row{
        preds.decision_ts.subspan(i, 1), preds.model_p.subspan(i, 1), preds.y.subspan(i, 1),
        preds.bar_group.subspan(i, 1),   preds.trade_ok.subspan(i, 1),
        preds.has_maker_fee.subspan(i, 1), preds.risk_technical.subspan(i, 1)};
    const auto a = Clock::now();
    pmlab::run_backtest(bars, row, corpus.params, one);
    const auto b = Clock::now();
    ns.push_back(now_ns_diff(a, b));
  }
  print_percentiles(ns);
  std::printf("    clock granularity: %.0f ns (samples are quantized to it)\n", clock_granularity_ns());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc >= 2 ? argv[1] : "";
    if (mode == "ticks") {
      const std::size_t n =
          argc >= 3 ? static_cast<std::size_t>(std::strtoll(argv[2], nullptr, 10)) : 10'000'000;
      std::printf("ticks (synthetic, n=%zu)\n", n);
      const auto ticks = synthetic_ticks(n);
      return bench_ticks(ticks);
    }
    if (mode == "ticks-tape" && argc >= 3) {
      const pmlab::MappedTape tape(argv[2]);
      std::printf("ticks (tape %s)\n", argv[2]);
      // TapeTick and BookTick share layout on purpose; view the mmap directly.
      static_assert(sizeof(pmlab::TapeTick) == sizeof(pmlab::BookTick));
      const auto* p = reinterpret_cast<const pmlab::BookTick*>(tape.ticks().data());
      return bench_ticks({p, tape.ticks().size()});
    }
    if (mode == "backtest" && argc >= 3) {
      const long repeats = argc >= 4 ? std::strtol(argv[3], nullptr, 10) : 200;
      std::printf("backtest (%s)\n", argv[2]);
      return bench_backtest(argv[2], repeats);
    }
    std::fprintf(stderr,
                 "usage: %s ticks [n] | ticks-tape <file.tape> | backtest <corpus.bin> [reps]\n",
                 argv[0]);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "engine_bench: %s\n", e.what());
    return 1;
  }
}
