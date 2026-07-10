#include "pmlab_engine/version.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(Version, ReportsCurrentVersion) {
  EXPECT_EQ(std::string(pmlab::engine_version()), "0.1");
}
