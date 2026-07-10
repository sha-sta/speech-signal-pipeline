// Ports of the semantics pinned in src/pmlab/probe/bars.py (and tests/test_probe_bars.py);
// exhaustive cross-language equality is the bars parity gate in tests/test_engine_bars_parity.py.
#include "pmlab_engine/bars.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using pmlab::assemble_bars;
using pmlab::Bar;
using pmlab::BarBuilder;
using pmlab::BookTick;
using pmlab::bucket_end;

TEST(BucketEnd, BoundaryTickBelongsToItsBoundary) {
  EXPECT_EQ(bucket_end(60), 60);   // exactly on a boundary -> that bar
  EXPECT_EQ(bucket_end(61), 120);  // one past -> next bar
  EXPECT_EQ(bucket_end(119), 120);
  EXPECT_EQ(bucket_end(1), 60);
}

TEST(BucketEnd, FloorDivisionOnNonPositiveTs) {
  // Python: ((ts - 1) // 60 + 1) * 60 with floor division.
  EXPECT_EQ(bucket_end(0), 0);
  EXPECT_EQ(bucket_end(-1), 0);
  EXPECT_EQ(bucket_end(-60), -60);
  EXPECT_EQ(bucket_end(-61), -60);
}

TEST(BarBuilder, OhlcWithinMinute) {
  BarBuilder b;
  b.add({10, 0.40, 0.44});
  b.add({20, 0.42, 0.46});
  b.add({30, 0.39, 0.43});
  const auto bar = b.close_minute(60);
  ASSERT_TRUE(bar.has_value());
  EXPECT_EQ(bar->bid_open, 0.40);
  EXPECT_EQ(bar->bid_high, 0.42);
  EXPECT_EQ(bar->bid_low, 0.39);
  EXPECT_EQ(bar->bid_close, 0.39);
  EXPECT_EQ(bar->ask_close, 0.43);
}

TEST(BarBuilder, SilentMinuteCarriesForwardEmittedClose) {
  BarBuilder b;
  b.add({10, 0.40, 0.44});
  ASSERT_TRUE(b.close_minute(60).has_value());
  const auto bar = b.close_minute(120);  // no ticks in (60, 120]
  ASSERT_TRUE(bar.has_value());
  EXPECT_EQ(bar->bid_open, 0.40);
  EXPECT_EQ(bar->bid_high, 0.40);
  EXPECT_EQ(bar->bid_low, 0.40);
  EXPECT_EQ(bar->bid_close, 0.40);
}

TEST(BarBuilder, CarryForwardIgnoresStandingQuoteFromLaterMinute) {
  BarBuilder b;
  b.add({10, 0.40, 0.44});
  // A tick for minute 120 arrives BEFORE close_minute(60) runs (the live race).
  b.add({70, 0.55, 0.60});
  const auto bar60 = b.close_minute(60);
  ASSERT_TRUE(bar60.has_value());
  EXPECT_EQ(bar60->bid_close, 0.40);  // closing 60 must not see the post-60 tick
  const auto bar120 = b.close_minute(120);
  ASSERT_TRUE(bar120.has_value());
  EXPECT_EQ(bar120->bid_close, 0.55);
  // The standing quote DOES include the later tick.
  EXPECT_EQ(b.last_tob().first, 0.55);
}

TEST(BarBuilder, NoBarBeforeFirstRealBar) {
  BarBuilder b;
  EXPECT_FALSE(b.close_minute(60).has_value());
  EXPECT_FALSE(b.close_minute(120).has_value());
}

TEST(BarBuilder, NanSidesAreSkippedInPush) {
  BarBuilder b;
  b.add({10, kNaN, 0.44});
  b.add({20, 0.41, kNaN});
  const auto bar = b.close_minute(60);
  ASSERT_TRUE(bar.has_value());
  // bid started NaN, first finite push seeds all four fields
  EXPECT_EQ(bar->bid_open, 0.41);
  EXPECT_EQ(bar->bid_close, 0.41);
  EXPECT_EQ(bar->ask_open, 0.44);
  EXPECT_EQ(bar->ask_close, 0.44);  // NaN push left the ask agg untouched
}

TEST(AssembleBars, SortsAndEmitsEveryBoundaryOnceRealBarExists) {
  const std::vector<BookTick> ticks = {{130, 0.5, 0.55}, {10, 0.4, 0.44}, {20, 0.41, 0.45}};
  const auto bars = assemble_bars(ticks, 10, 130);
  ASSERT_EQ(bars.size(), 3u);  // minutes 60, 120 (silent, carried), 180 (bucket_end(130))
  EXPECT_EQ(bars[0].ts, 60);
  EXPECT_EQ(bars[0].bid_close, 0.41);
  EXPECT_EQ(bars[1].ts, 120);
  EXPECT_EQ(bars[1].bid_close, 0.41);  // carried forward
  EXPECT_EQ(bars[2].ts, 180);
  EXPECT_EQ(bars[2].bid_close, 0.5);
}

TEST(AssembleBars, OrphanBucketsBeforeStartAreNeverEmitted) {
  // start_ts inside the log: ticks from earlier minutes fold into buckets that never close.
  const std::vector<BookTick> ticks = {{10, 0.4, 0.44}, {70, 0.5, 0.54}};
  const auto bars = assemble_bars(ticks, 70, 120);
  ASSERT_EQ(bars.size(), 1u);  // minute 60's bucket absorbed the early tick but never closes
  EXPECT_EQ(bars[0].ts, 120);
  EXPECT_EQ(bars[0].bid_close, 0.5);
  EXPECT_EQ(bars[0].bid_open, 0.5);  // the orphan tick's 0.4 must not leak into bar 120
}

TEST(AssembleBars, EmptyLog) {
  EXPECT_TRUE(assemble_bars({}, 0, 600).empty());
}

}  // namespace
