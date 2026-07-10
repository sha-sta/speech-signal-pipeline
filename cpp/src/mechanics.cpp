#include "pmlab_engine/mechanics.hpp"

#include <cmath>

#include "pmlab_engine/pyround.hpp"

namespace pmlab {

double maker_fee(double price, double contracts, bool has_maker_fee, double rate) {
  if (!has_maker_fee || contracts <= 0.0) {
    return 0.0;
  }
  const double raw = rate * contracts * price * (1.0 - price);
  return std::ceil(raw * 100.0) / 100.0;
}

std::optional<double> maker_quote(Side side, double yes_bid, double yes_ask, double tick) {
  if (side == Side::kYes) {
    const double q = pyround2(yes_bid + tick);
    if (q < yes_ask) {
      return q;
    }
    return std::nullopt;
  }
  const double q = pyround2(yes_ask - tick);
  if (q > yes_bid) {
    return q;
  }
  return std::nullopt;
}

Settlement settle(Side side, double fill_price, int y, double fee) {
  double stake = 0.0;
  double payout = 0.0;
  if (side == Side::kYes) {
    stake = fill_price;
    payout = (y == 1) ? 1.0 : 0.0;
  } else {
    stake = 1.0 - fill_price;
    payout = (y == 0) ? 1.0 : 0.0;
  }
  return Settlement{payout, stake, payout - stake - fee};
}

}  // namespace pmlab
