#include "pmlab_engine/engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "pmlab_engine/mechanics.hpp"

namespace pmlab {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Index of the last bar at/before decision_ts with a fresh, valid two-sided quote, else -1.
// Port of _decision_bar: searchsorted(side="right") - 1, staleness, finite un-crossed quote,
// mid strictly inside (0, 1).
std::int64_t decision_bar(std::span<const double> ts, std::span<const double> bid,
                          std::span<const double> ask, double decision_ts,
                          double max_staleness_s) {
  if (ts.empty() || !std::isfinite(decision_ts)) {
    return -1;
  }
  const auto it = std::upper_bound(ts.begin(), ts.end(), decision_ts);
  const std::int64_t idx = static_cast<std::int64_t>(it - ts.begin()) - 1;
  if (idx < 0 || (decision_ts - ts[static_cast<std::size_t>(idx)]) > max_staleness_s) {
    return -1;
  }
  const double b = bid[static_cast<std::size_t>(idx)];
  const double a = ask[static_cast<std::size_t>(idx)];
  if (!(std::isfinite(b) && std::isfinite(a)) || a < b) {
    return -1;
  }
  const double mid = (b + a) / 2.0;
  if (!(0.0 < mid && mid < 1.0)) {
    return -1;
  }
  return idx;
}

// First later-bar ts whose trade print crosses the resting quote, else nullopt. Port of _fills:
// only bars strictly after after_idx count; NaN prints fail both comparisons and never fill.
std::optional<double> first_fill_ts(std::span<const double> ts, std::span<const double> price_low,
                                    std::span<const double> price_high, Side side, double quote,
                                    std::int64_t after_idx) {
  for (std::size_t j = static_cast<std::size_t>(after_idx) + 1; j < ts.size(); ++j) {
    const bool hit = side == Side::kYes ? price_low[j] <= quote : price_high[j] >= quote;
    if (hit) {
      return ts[j];
    }
  }
  return std::nullopt;
}

}  // namespace

void run_backtest(const BarsView& bars, const PredictionsView& preds,
                  const BacktestParams& params, const BacktestOutputs& out) {
  const std::size_t n = preds.decision_ts.size();
  for (std::size_t i = 0; i < n; ++i) {
    // Row defaults, identical to the Python row dict before any pricing.
    out.market_prob[i] = kNaN;
    out.quote_price[i] = kNaN;
    out.fill_ts[i] = kNaN;
    out.fill_price[i] = kNaN;
    out.fee[i] = 0.0;
    out.contracts[i] = 0.0;
    out.stake[i] = 0.0;
    out.payout[i] = 0.0;
    out.pnl[i] = 0.0;
    out.side[i] = -1;
    out.traded[i] = 0;
    out.filled[i] = 0;
    out.decided_by_technicality[i] = 0;

    const std::int64_t g = preds.bar_group[i];
    if (g < 0) {
      continue;  // market has no bars at all
    }
    const auto off = static_cast<std::size_t>(bars.offsets[static_cast<std::size_t>(g)]);
    const auto cnt = static_cast<std::size_t>(bars.counts[static_cast<std::size_t>(g)]);
    const auto ts = bars.ts.subspan(off, cnt);
    const auto bid = bars.bid_close.subspan(off, cnt);
    const auto ask = bars.ask_close.subspan(off, cnt);

    const std::int64_t idx =
        decision_bar(ts, bid, ask, preds.decision_ts[i], params.max_staleness_s);
    if (idx < 0) {
      continue;
    }
    const double b = bid[static_cast<std::size_t>(idx)];
    const double a = ask[static_cast<std::size_t>(idx)];
    const double market_prob = (b + a) / 2.0;
    out.market_prob[i] = market_prob;

    const double edge = preds.model_p[i] - market_prob;
    const bool in_band = params.band_lo <= market_prob && market_prob <= params.band_hi;
    if (!(preds.trade_ok[i] != 0 && in_band && std::fabs(edge) > params.theta)) {
      continue;
    }

    const Side side = edge > 0 ? Side::kYes : Side::kNo;
    const std::optional<double> quote = maker_quote(side, b, a, params.tick);
    if (!quote.has_value()) {
      continue;  // locked/crossed book: no maker improvement possible
    }

    out.traded[i] = 1;
    out.side[i] = side == Side::kYes ? 0 : 1;
    out.quote_price[i] = *quote;

    const std::optional<double> fill_ts = first_fill_ts(
        ts, bars.price_low.subspan(off, cnt), bars.price_high.subspan(off, cnt), side, *quote, idx);
    if (!fill_ts.has_value()) {
      continue;
    }

    const double fee = maker_fee(*quote, params.contracts, preds.has_maker_fee[i] != 0);
    const Settlement s = settle(side, *quote, preds.y[i], fee);
    out.filled[i] = 1;
    out.fill_ts[i] = *fill_ts;
    out.fill_price[i] = *quote;
    out.fee[i] = fee;
    out.contracts[i] = params.contracts;
    out.stake[i] = s.stake * params.contracts;
    out.payout[i] = s.payout * params.contracts;
    out.pnl[i] = s.pnl * params.contracts;
    out.decided_by_technicality[i] = preds.risk_technical[i];
  }
}

}  // namespace pmlab
