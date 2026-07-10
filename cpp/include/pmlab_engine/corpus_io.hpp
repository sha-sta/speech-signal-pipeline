#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "pmlab_engine/engine.hpp"

namespace pmlab {

// Owning storage for one backtest corpus (bars + predictions + params), the on-disk twin of
// the views in engine.hpp. Format: little-endian, fixed-width, no strings — a Python exporter
// writes it with numpy tofile and this reader loads it with plain fstream reads. Layout:
//
//   magic   u64  "PMCORP01"
//   n_bars  u64, n_groups u64, n_preds u64
//   params  f64 x6 (band_lo, band_hi, theta, tick, max_staleness_s, contracts)
//   bars    f64 x n_bars, five arrays in order: ts, bid_close, ask_close, price_low, price_high
//   groups  i64 x n_groups x2: offsets, counts
//   preds   f64 decision_ts, f64 model_p, i32 y, i64 bar_group, u8 trade_ok, u8 has_maker_fee,
//           u8 risk_technical (each n_preds long, in that order)
struct Corpus {
  BacktestParams params;
  std::vector<double> bar_ts, bar_bid_close, bar_ask_close, bar_price_low, bar_price_high;
  std::vector<std::int64_t> offsets, counts;
  std::vector<double> decision_ts, model_p;
  std::vector<std::int32_t> y;
  std::vector<std::int64_t> bar_group;
  std::vector<std::uint8_t> trade_ok, has_maker_fee, risk_technical;

  BarsView bars_view() const;
  PredictionsView predictions_view() const;
  std::size_t n_predictions() const { return decision_ts.size(); }
};

// Owning storage for the engine's output columns, sized on construction.
struct Results {
  std::vector<double> market_prob, quote_price, fill_ts, fill_price;
  std::vector<double> fee, contracts, stake, payout, pnl;
  std::vector<std::int8_t> side;
  std::vector<std::uint8_t> traded, filled, decided_by_technicality;

  explicit Results(std::size_t n);
  BacktestOutputs outputs_view();
};

// Throw std::runtime_error on a missing/truncated/wrong-magic file.
Corpus read_corpus(const std::filesystem::path& path);
void write_corpus(const Corpus& corpus, const std::filesystem::path& path);

// Results file: magic "PMRES001", u64 n, then the columns in Results declaration order.
void write_results(const Results& results, const std::filesystem::path& path);
Results read_results(const std::filesystem::path& path);

}  // namespace pmlab
