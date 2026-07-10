#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pmlab {

// Port of src/pmlab/probe/bars.py: book-tick -> 1-min OHLC bar assembly with period-END
// timestamps. Semantics must match the Python BarBuilder bit for bit (enforced by the bars
// parity gate): per-bucket accumulators (out-of-order-safe across the minute boundary), NaN
// sides skipped in the OHLC push, silent minutes carry forward the previous EMITTED bar's
// close (never the standing quote), and nothing is emitted before the first real bar.

struct BookTick {
  std::int64_t ts;  // unix seconds the quote was observed
  double yes_bid;
  double yes_ask;
};

struct Bar {
  std::int64_t ts;  // period-END: covers (ts - period, ts]
  double bid_open, bid_high, bid_low, bid_close;
  double ask_open, ask_high, ask_low, ask_close;
};

// Period-END of the bucket a tick at ts belongs to: maps (S - period, S] -> S; a tick exactly
// on a boundary belongs to that boundary's bar. Uses floor division like Python (C++ integer
// division truncates toward zero instead, which differs for ts <= 0).
std::int64_t bucket_end(std::int64_t ts, std::int64_t period_s = 60);

class BarBuilder {
 public:
  explicit BarBuilder(std::int64_t period_s = 60) : period_s_(period_s) {}

  // Fold a tick into its minute's accumulator (per-bucket; closing S never disturbs S+60).
  void add(const BookTick& tick);

  // Completed bar for the minute ending at s, or nullopt if no real bar has ever completed.
  // A silent minute is flat at the previous emitted bar's close.
  std::optional<Bar> close_minute(std::int64_t s);

  // Most recent (yes_bid, yes_ask) seen — the standing quote (includes post-S ticks).
  std::pair<double, double> last_tob() const { return {last_bid_, last_ask_}; }

 private:
  struct Agg {
    double open, high, low, close;
    static Agg start(double x) { return Agg{x, x, x, x}; }
    void push(double x);
  };

  static Bar row(std::int64_t s, const Agg& bid, const Agg& ask);

  static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  std::int64_t period_s_;
  std::unordered_map<std::int64_t, std::pair<Agg, Agg>> aggs_;  // bucket period-END -> (bid, ask)
  std::optional<std::pair<double, double>> last_close_;         // last emitted REAL bar's closes
  double last_bid_ = kNaN;
  double last_ask_ = kNaN;
};

// Batch replay: every 1-min bar in [start_ts, end_ts] from a full tick log. Ticks are
// stable-sorted by ts internally (mirrors Python's sorted); equal-ts input order decides the
// close. Bars are emitted per period-END boundary once a real bar exists.
std::vector<Bar> assemble_bars(std::span<const BookTick> ticks, std::int64_t start_ts,
                               std::int64_t end_ts, std::int64_t period_s = 60);

}  // namespace pmlab
