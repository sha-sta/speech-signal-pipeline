// End-to-end engine tests mirroring the run_backtest cases in tests/test_study_backtest.py.
// Full-scale cross-language parity is the pytest gate; these keep the C++ side self-checking.
#include "pmlab_engine/engine.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <vector>

#include "pmlab_engine/corpus_io.hpp"

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// One market, its bars as (ts, bid_close, ask_close, price_low, price_high) rows, one
// prediction. Mirrors the _candles/_features helpers in the Python test file.
struct Case {
  std::vector<std::array<double, 5>> bars;
  double decision_ts = 1000.0;
  double model_p = 0.9;
  std::int32_t y = 1;
  std::uint8_t trade_ok = 1;
  std::uint8_t has_fee = 1;
  std::uint8_t risk_technical = 0;
  bool no_bars = false;
};

pmlab::Corpus build(const Case& c) {
  pmlab::Corpus corpus;
  // Params mirror _PARAMS in the Python tests: trade_split None, 30-day staleness.
  corpus.params = pmlab::BacktestParams{0.15, 0.60, 0.05, 0.01, 30.0 * 86400.0, 1.0};
  for (const auto& b : c.bars) {
    corpus.bar_ts.push_back(b[0]);
    corpus.bar_bid_close.push_back(b[1]);
    corpus.bar_ask_close.push_back(b[2]);
    corpus.bar_price_low.push_back(b[3]);
    corpus.bar_price_high.push_back(b[4]);
  }
  corpus.offsets = {0};
  corpus.counts = {static_cast<std::int64_t>(c.bars.size())};
  corpus.decision_ts = {c.decision_ts};
  corpus.model_p = {c.model_p};
  corpus.y = {c.y};
  corpus.bar_group = {c.no_bars ? std::int64_t{-1} : std::int64_t{0}};
  corpus.trade_ok = {c.trade_ok};
  corpus.has_maker_fee = {c.has_fee};
  corpus.risk_technical = {c.risk_technical};
  return corpus;
}

pmlab::Results run(const pmlab::Corpus& corpus) {
  pmlab::Results results(corpus.n_predictions());
  const auto out = results.outputs_view();
  pmlab::run_backtest(corpus.bars_view(), corpus.predictions_view(), corpus.params, out);
  return results;
}

TEST(Engine, BuyYesFillsAndWins) {
  // decision bar at ts=900 (mid 0.40, in band); later bar trades down through 0.39
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.35, 0.40, 0.34, 0.41}}}));
  EXPECT_EQ(r.traded[0], 1);
  EXPECT_EQ(r.filled[0], 1);
  EXPECT_EQ(r.side[0], 0);  // yes
  EXPECT_EQ(r.quote_price[0], 0.39);
  EXPECT_EQ(r.fill_ts[0], 2000.0);
  EXPECT_EQ(r.fee[0], 0.01);
  EXPECT_NEAR(r.pnl[0], 0.60, 1e-12);  // 1 - 0.39 - 0.01
}

TEST(Engine, EdgeButNoTradeThroughIsUnfilled) {
  // later bar never trades down to the 0.39 resting bid
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.41, 0.45, 0.41, 0.50}}}));
  EXPECT_EQ(r.traded[0], 1);
  EXPECT_EQ(r.filled[0], 0);
  EXPECT_EQ(r.pnl[0], 0.0);
  EXPECT_EQ(r.stake[0], 0.0);
  EXPECT_TRUE(std::isnan(r.fill_ts[0]));
}

TEST(Engine, OutOfBandNotTraded) {
  const auto r = run(build({.bars = {{900, 0.68, 0.72, 0.70, 0.72}, {2000, 0.60, 0.66, 0.55, 0.66}},
                            .model_p = 0.95}));
  EXPECT_NEAR(r.market_prob[0], 0.70, 1e-12);
  EXPECT_EQ(r.traded[0], 0);
}

TEST(Engine, BelowThetaNotTraded) {
  // |0.43 - 0.40| = 0.03 <= theta 0.05
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.30, 0.35, 0.30, 0.42}},
                            .model_p = 0.43}));
  EXPECT_EQ(r.traded[0], 0);
}

TEST(Engine, StalePriceNotTraded) {
  // only bar is ~40 days before the decision (staleness = 30 days) -> untradable
  const auto r = run(build({.bars = {{1000000.0 - 40 * 86400.0, 0.38, 0.42, 0.40, 0.42}},
                            .decision_ts = 1000000.0}));
  EXPECT_TRUE(std::isnan(r.market_prob[0]));
  EXPECT_EQ(r.traded[0], 0);
}

TEST(Engine, SellYesFillsAndWins) {
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.38, 0.42}, {2000, 0.45, 0.55, 0.45, 0.55}},
                            .model_p = 0.1, .y = 0}));
  EXPECT_EQ(r.side[0], 1);  // no
  EXPECT_EQ(r.filled[0], 1);
  EXPECT_EQ(r.quote_price[0], 0.41);
  EXPECT_NEAR(r.pnl[0], 0.40, 1e-12);
}

TEST(Engine, DecidedByTechnicalityFlag) {
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.35, 0.40, 0.34, 0.41}},
                            .risk_technical = 1}));
  EXPECT_EQ(r.filled[0], 1);
  EXPECT_EQ(r.decided_by_technicality[0], 1);
}

TEST(Engine, TechnicalityFlagStaysFalseWhenUnfilled) {
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.41, 0.45, 0.41, 0.50}},
                            .risk_technical = 1}));
  EXPECT_EQ(r.filled[0], 0);
  EXPECT_EQ(r.decided_by_technicality[0], 0);  // Python only sets it on the fill update
}

TEST(Engine, SplitGateBlocksTrade) {
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.35, 0.40, 0.34, 0.41}},
                            .trade_ok = 0}));
  EXPECT_NEAR(r.market_prob[0], 0.40, 1e-12);  // priced, but not traded
  EXPECT_EQ(r.traded[0], 0);
}

TEST(Engine, LockedBookPlacesNoOrder) {
  const auto r = run(build({.bars = {{900, 0.40, 0.41, 0.40, 0.42}, {2000, 0.30, 0.35, 0.30, 0.42}}}));
  EXPECT_EQ(r.traded[0], 0);
  EXPECT_TRUE(std::isnan(r.quote_price[0]));
}

TEST(Engine, NoBarsGroup) {
  const auto r = run(build({.bars = {}, .no_bars = true}));
  EXPECT_TRUE(std::isnan(r.market_prob[0]));
  EXPECT_EQ(r.traded[0], 0);
}

TEST(Engine, NanDecisionTsNotPriced) {
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}}, .decision_ts = kNaN}));
  EXPECT_TRUE(std::isnan(r.market_prob[0]));
  EXPECT_EQ(r.traded[0], 0);
}

TEST(Engine, NoFeeSeries) {
  const auto r = run(build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.35, 0.40, 0.34, 0.41}},
                            .has_fee = 0}));
  EXPECT_EQ(r.filled[0], 1);
  EXPECT_EQ(r.fee[0], 0.0);
  EXPECT_NEAR(r.pnl[0], 0.61, 1e-12);  // 1 - 0.39, fee-free
}

TEST(CorpusIo, RoundTrip) {
  const auto corpus = build({.bars = {{900, 0.38, 0.42, 0.40, 0.42}, {2000, 0.35, 0.40, 0.34, 0.41}}});
  const auto dir = std::filesystem::temp_directory_path();
  const auto corpus_path = dir / "pmlab_engine_test_corpus.bin";
  const auto results_path = dir / "pmlab_engine_test_results.bin";

  pmlab::write_corpus(corpus, corpus_path);
  const auto loaded = pmlab::read_corpus(corpus_path);
  EXPECT_EQ(loaded.bar_ts, corpus.bar_ts);
  EXPECT_EQ(loaded.counts, corpus.counts);
  EXPECT_EQ(loaded.params.theta, corpus.params.theta);

  auto results = run(loaded);
  pmlab::write_results(results, results_path);
  const auto results2 = pmlab::read_results(results_path);
  EXPECT_EQ(results2.pnl, results.pnl);
  EXPECT_EQ(results2.filled, results.filled);

  std::filesystem::remove(corpus_path);
  std::filesystem::remove(results_path);
}

}  // namespace
