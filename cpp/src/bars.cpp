#include "pmlab_engine/bars.hpp"

#include <algorithm>
#include <cmath>

namespace pmlab {
namespace {

// Floor division (Python //). C++ / truncates toward zero; they differ on negatives.
std::int64_t floordiv(std::int64_t a, std::int64_t b) {
  const std::int64_t q = a / b;
  return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

}  // namespace

std::int64_t bucket_end(std::int64_t ts, std::int64_t period_s) {
  return (floordiv(ts - 1, period_s) + 1) * period_s;
}

void BarBuilder::Agg::push(double x) {
  if (!std::isfinite(x)) {
    return;
  }
  if (!std::isfinite(open)) {
    open = high = low = close = x;
    return;
  }
  high = std::max(high, x);
  low = std::min(low, x);
  close = x;
}

void BarBuilder::add(const BookTick& tick) {
  const std::int64_t end = bucket_end(tick.ts, period_s_);
  auto [it, inserted] = aggs_.try_emplace(end, Agg::start(tick.yes_bid), Agg::start(tick.yes_ask));
  if (!inserted) {
    it->second.first.push(tick.yes_bid);
    it->second.second.push(tick.yes_ask);
  }
  if (std::isfinite(tick.yes_bid)) {
    last_bid_ = tick.yes_bid;
  }
  if (std::isfinite(tick.yes_ask)) {
    last_ask_ = tick.yes_ask;
  }
}

std::optional<Bar> BarBuilder::close_minute(std::int64_t s) {
  if (const auto it = aggs_.find(s); it != aggs_.end()) {
    const auto [bid, ask] = it->second;
    aggs_.erase(it);
    last_close_ = {bid.close, ask.close};
    return row(s, bid, ask);
  }
  if (last_close_.has_value()) {
    return row(s, Agg::start(last_close_->first), Agg::start(last_close_->second));
  }
  return std::nullopt;
}

Bar BarBuilder::row(std::int64_t s, const Agg& bid, const Agg& ask) {
  return Bar{s,        bid.open, bid.high, bid.low, bid.close,
             ask.open, ask.high, ask.low,  ask.close};
}

std::vector<Bar> assemble_bars(std::span<const BookTick> ticks, std::int64_t start_ts,
                               std::int64_t end_ts, std::int64_t period_s) {
  std::vector<BookTick> ordered(ticks.begin(), ticks.end());
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const BookTick& a, const BookTick& b) { return a.ts < b.ts; });

  BarBuilder builder(period_s);
  const std::int64_t first_end = bucket_end(start_ts, period_s);
  const std::int64_t last_end = bucket_end(end_ts, period_s);
  std::vector<Bar> rows;
  if (last_end >= first_end) {
    rows.reserve(static_cast<std::size_t>((last_end - first_end) / period_s + 1));
  }
  std::size_t i = 0;
  for (std::int64_t s = first_end; s <= last_end; s += period_s) {
    while (i < ordered.size() && bucket_end(ordered[i].ts, period_s) <= s) {
      builder.add(ordered[i]);
      ++i;
    }
    if (const auto bar = builder.close_minute(s); bar.has_value()) {
      rows.push_back(*bar);
    }
  }
  return rows;
}

}  // namespace pmlab
