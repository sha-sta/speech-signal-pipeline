// Enforces the "no allocations in the hot loop" claim with a counting global operator new.
// If someone adds a std::string, a vector copy, or an accidental by-value capture to
// run_backtest, this fails rather than silently degrading the latency story.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "pmlab_engine/bars.hpp"
#include "pmlab_engine/corpus_io.hpp"
#include "pmlab_engine/engine.hpp"

namespace {

std::atomic<long> g_alloc_count{0};
std::atomic<bool> g_tracking{false};

class AllocScope {
 public:
  AllocScope() {
    g_alloc_count.store(0, std::memory_order_relaxed);
    g_tracking.store(true, std::memory_order_relaxed);
  }
  ~AllocScope() { g_tracking.store(false, std::memory_order_relaxed); }
  long allocations() const { return g_alloc_count.load(std::memory_order_relaxed); }
};

}  // namespace

// Binary-wide replacements; they only count while an AllocScope is live.
void* operator new(std::size_t size) {
  if (g_tracking.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void* p = std::malloc(size)) {
    return p;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

pmlab::Corpus synthetic_corpus(int n_markets, int bars_per_market) {
  pmlab::Corpus c;
  c.params = pmlab::BacktestParams{0.15, 0.60, 0.05, 0.01, 30.0 * 86400.0, 1.0};
  std::uint64_t state = 88172645463325252ULL;  // xorshift; deterministic, no <random> allocs
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int m = 0; m < n_markets; ++m) {
    const std::int64_t t0 = 1'700'000'000 + static_cast<std::int64_t>(next() % 365) * 86'400;
    c.offsets.push_back(static_cast<std::int64_t>(c.bar_ts.size()));
    c.counts.push_back(bars_per_market);
    for (int k = 0; k < bars_per_market; ++k) {
      const double bid = static_cast<double>(next() % 90) / 100.0;
      const double ask = bid + static_cast<double>(1 + next() % 5) / 100.0;
      c.bar_ts.push_back(static_cast<double>(t0 + static_cast<std::int64_t>(k) * 86'400));
      c.bar_bid_close.push_back(bid);
      c.bar_ask_close.push_back(ask);
      c.bar_price_low.push_back(bid - static_cast<double>(next() % 5) / 100.0);
      c.bar_price_high.push_back(ask + static_cast<double>(next() % 5) / 100.0);
    }
    c.decision_ts.push_back(static_cast<double>(t0 + 2 * 86'400));
    c.model_p.push_back(static_cast<double>(next() % 100) / 100.0);
    c.y.push_back(static_cast<std::int32_t>(next() % 2));
    c.bar_group.push_back(m);
    c.trade_ok.push_back(1);
    c.has_maker_fee.push_back(1);
    c.risk_technical.push_back(static_cast<std::uint8_t>(next() % 2));
  }
  return c;
}

TEST(Allocations, BacktestHotLoopAllocatesNothing) {
  const auto corpus = synthetic_corpus(2000, 50);
  pmlab::Results results(corpus.n_predictions());
  const auto out = results.outputs_view();
  const auto bars = corpus.bars_view();
  const auto preds = corpus.predictions_view();

  const AllocScope scope;
  pmlab::run_backtest(bars, preds, corpus.params, out);
  EXPECT_EQ(scope.allocations(), 0);
}

TEST(Allocations, BarBuilderSteadyStateIsAllocationFree) {
  pmlab::BarBuilder builder;
  builder.add({1, 0.40, 0.44});  // bucket + hash table exist after the first tick

  const AllocScope scope;
  for (int i = 2; i < 60; ++i) {
    builder.add({i, 0.40 + (i % 3) * 0.01, 0.44 + (i % 3) * 0.01});
  }
  EXPECT_EQ(scope.allocations(), 0) << "per-tick add() must not allocate within a minute";
}

}  // namespace
