#include "pmlab_engine/tape.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

using pmlab::MappedTape;
using pmlab::TapeTick;
using pmlab::write_tape;

std::filesystem::path tmp(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}

TEST(Tape, RoundTripThroughMmap) {
  const std::vector<TapeTick> ticks = {{100, 0.40, 0.44}, {101, 0.41, 0.45}, {160, 0.39, 0.43}};
  const auto path = tmp("pmlab_engine_test.tape");
  write_tape(ticks, path);

  const MappedTape tape(path);
  ASSERT_EQ(tape.ticks().size(), ticks.size());
  EXPECT_EQ(tape.ticks()[0].ts, 100);
  EXPECT_EQ(tape.ticks()[2].yes_ask, 0.43);
  std::filesystem::remove(path);
}

TEST(Tape, MoveTransfersOwnership) {
  const std::vector<TapeTick> ticks = {{100, 0.40, 0.44}};
  const auto path = tmp("pmlab_engine_test_move.tape");
  write_tape(ticks, path);

  MappedTape a(path);
  const MappedTape b(std::move(a));
  ASSERT_EQ(b.ticks().size(), 1u);
  EXPECT_EQ(b.ticks()[0].yes_bid, 0.40);
  std::filesystem::remove(path);
}

TEST(Tape, RejectsBadMagicAndTruncation) {
  const auto path = tmp("pmlab_engine_test_bad.tape");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "NOTATAPE" << std::string(8, '\0');
  }
  EXPECT_THROW(MappedTape{path}, std::runtime_error);
  {
    // right magic, claimed count larger than the payload
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "PMTAPE01";
    const std::uint64_t n = 1000;
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  }
  EXPECT_THROW(MappedTape{path}, std::runtime_error);
  std::filesystem::remove(path);
}

}  // namespace
