// Mirrors of the pure-mechanics unit tests in tests/test_study_backtest.py, so a semantics
// drift shows up as the same failure in both languages. The exhaustive cross-language check
// is test_golden.cpp.
#include "pmlab_engine/mechanics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

using pmlab::maker_fee;
using pmlab::maker_quote;
using pmlab::settle;
using pmlab::Side;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

TEST(MakerFee, CeilsToCent) {
  // 0.0175 * 1 * 0.4 * 0.6 = 0.0042 -> ceil to $0.01
  EXPECT_DOUBLE_EQ(maker_fee(0.4, 1.0, true), 0.01);
  // larger size scales then ceils: 0.0175 * 100 * 0.5 * 0.5 = 0.4375 -> 0.44
  EXPECT_DOUBLE_EQ(maker_fee(0.5, 100.0, true), 0.44);
}

TEST(MakerFee, ZeroWhenNoMakerFeeOrNoContracts) {
  EXPECT_EQ(maker_fee(0.4, 1.0, false), 0.0);
  EXPECT_EQ(maker_fee(0.4, 0.0, true), 0.0);
}

TEST(MakerQuote, ImprovesBestSide) {
  EXPECT_EQ(maker_quote(Side::kYes, 0.38, 0.42, 0.01), 0.39);  // improve the bid
  EXPECT_EQ(maker_quote(Side::kNo, 0.38, 0.42, 0.01), 0.41);   // improve the ask
}

TEST(MakerQuote, NoneOnLockedBook) {
  // one-tick improvement would cross a 1-tick-wide book
  EXPECT_EQ(maker_quote(Side::kYes, 0.40, 0.41, 0.01), std::nullopt);
  EXPECT_EQ(maker_quote(Side::kNo, 0.40, 0.41, 0.01), std::nullopt);
}

TEST(MakerQuote, NoneOnNanQuote) {
  // NaN fails every comparison, exactly like the Python `q < ask` / `q > bid` guards.
  EXPECT_EQ(maker_quote(Side::kYes, 0.38, kNaN, 0.01), std::nullopt);
  EXPECT_EQ(maker_quote(Side::kNo, kNaN, 0.42, 0.01), std::nullopt);
  EXPECT_EQ(maker_quote(Side::kYes, kNaN, 0.42, 0.01), std::nullopt);
}

TEST(Settle, YesSide) {
  const auto win = settle(Side::kYes, 0.39, 1, 0.01);
  EXPECT_EQ(win.payout, 1.0);
  EXPECT_EQ(win.stake, 0.39);
  EXPECT_NEAR(win.pnl, 0.60, 1e-12);
  const auto lose = settle(Side::kYes, 0.39, 0, 0.01);
  EXPECT_EQ(lose.payout, 0.0);
  EXPECT_NEAR(lose.pnl, -0.40, 1e-12);
}

TEST(Settle, NoSide) {
  // sold yes at 0.41 (= bought no at 0.59); settles No -> win
  const auto win = settle(Side::kNo, 0.41, 0, 0.01);
  EXPECT_EQ(win.payout, 1.0);
  EXPECT_NEAR(win.stake, 0.59, 1e-12);
  EXPECT_NEAR(win.pnl, 0.40, 1e-12);
  const auto lose = settle(Side::kNo, 0.41, 1, 0.01);
  EXPECT_NEAR(lose.pnl, -0.60, 1e-12);
}

}  // namespace
