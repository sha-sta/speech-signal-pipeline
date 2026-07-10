#pragma once

#include <cstdint>
#include <span>

namespace pmlab {

// Column views over one contiguous bar table, grouped by market: group g's bars occupy
// [offsets[g], offsets[g] + counts[g]) and are ascending in ts. The Python wrapper prepares the
// grouping exactly like backtest.py's _bars_by_ticker, so ordering parity is inherited from
// pandas rather than re-derived here.
struct BarsView {
  std::span<const double> ts;
  std::span<const double> bid_close;
  std::span<const double> ask_close;
  std::span<const double> price_low;
  std::span<const double> price_high;
  std::span<const std::int64_t> offsets;
  std::span<const std::int64_t> counts;
};

// One row per prediction. Strings and pandas-isms are resolved by the caller into plain
// numbers: bar_group indexes offsets/counts (-1 when the market has no bars); trade_ok is the
// precomputed split gate (split == trade_split, or all-1 when trade_split is None);
// has_maker_fee reproduces _has_maker_fee (series lookup with charge-by-default);
// risk_technical is resolution_risk in {medium, high}. decision_ts is NaN when missing.
struct PredictionsView {
  std::span<const double> decision_ts;
  std::span<const double> model_p;
  std::span<const std::int32_t> y;
  std::span<const std::int64_t> bar_group;
  std::span<const std::uint8_t> trade_ok;
  std::span<const std::uint8_t> has_maker_fee;
  std::span<const std::uint8_t> risk_technical;
};

// Mirrors BacktestParams in src/pmlab/study/backtest.py (same illustrative defaults).
// max_staleness is in seconds here; the Python wrapper converts from days.
struct BacktestParams {
  double band_lo = 0.15;
  double band_hi = 0.60;
  double theta = 0.05;
  double tick = 0.01;
  double max_staleness_s = 7.0 * 86400.0;
  double contracts = 1.0;
};

// Output columns, preallocated by the caller to n_predictions and fully overwritten by
// run_backtest (untraded rows get the same defaults the Python row dict starts with).
// side: -1 = none, 0 = yes, 1 = no. fill_ts is NaN when unfilled (wrapper casts to Int64).
struct BacktestOutputs {
  std::span<double> market_prob;
  std::span<double> quote_price;
  std::span<double> fill_ts;
  std::span<double> fill_price;
  std::span<double> fee;
  std::span<double> contracts;
  std::span<double> stake;
  std::span<double> payout;
  std::span<double> pnl;
  std::span<std::int8_t> side;
  std::span<std::uint8_t> traded;
  std::span<std::uint8_t> filled;
  std::span<std::uint8_t> decided_by_technicality;
};

// The hot loop. Pure function of its inputs, no allocation, no I/O; must reproduce
// run_backtest in src/pmlab/study/backtest.py bit for bit (enforced by the pytest parity gate).
void run_backtest(const BarsView& bars, const PredictionsView& preds,
                  const BacktestParams& params, const BacktestOutputs& out);

}  // namespace pmlab
