add_executable(engine_tests
  test_version.cpp
)
target_link_libraries(engine_tests
  PRIVATE pmlab_engine pmlab_engine_warnings GTest::gtest_main)

include(GoogleTest)
# PRE_TEST: discover at ctest time, not build time — a build-time run of an ASan binary trips
# macOS's malloc nano zone unless MallocNanoZone=0 is exported (see workflow / README).
gtest_discover_tests(engine_tests DISCOVERY_MODE PRE_TEST)
