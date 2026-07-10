#pragma once

#include <optional>

namespace pmlab {

// Ports of the pure trade mechanics in src/pmlab/study/backtest.py. Every function must match
// the Python reference bit for bit; the golden-file suite in tests/ enforces it. Parameter
// defaults mirror the Python module constants (illustrative, not tuned settings).

inline constexpr double kMakerFeeRate = 0.0175;

enum class Side : signed char { kYes, kNo };

// Kalshi maker fee: ceil_to_cent(rate * contracts * p * (1 - p)); 0 when the series has no
// maker fee or contracts <= 0. `price` is the fill price (yes-equivalent probability).
double maker_fee(double price, double contracts, bool has_maker_fee, double rate = kMakerFeeRate);

// Resting maker price one tick better than the best quote, never crossing the book.
// Buy yes: improve the bid; sell yes: improve the ask. nullopt when a one-tick improvement
// would lock or cross (including any NaN quote, which fails every comparison, as in Python).
std::optional<double> maker_quote(Side side, double yes_bid, double yes_ask, double tick);

struct Settlement {
  double payout;
  double stake;
  double pnl;
};

// Per-contract settlement held to resolution: pnl = payout - stake - fee.
// Yes: stake = fill, payout = 1 if y == 1. No (sold yes): stake = 1 - fill, payout = 1 if y == 0.
Settlement settle(Side side, double fill_price, int y, double fee);

}  // namespace pmlab
