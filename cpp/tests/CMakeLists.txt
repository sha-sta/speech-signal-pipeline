add_executable(engine_tests
  test_version.cpp
  test_mechanics.cpp
  test_golden.cpp
  test_engine.cpp
  test_bars.cpp
  test_tape.cpp
)
target_link_libraries(engine_tests
  PRIVATE pmlab_engine pmlab_engine_warnings GTest::gtest_main)

# The allocation-counting tests replace global operator new/delete, which confuses Linux GCC
# ASan's alloc-dealloc matching for the ENTIRE binary (interceptor-tagged allocations freed via
# our replacement). Keep them in their own executable and relax only that check, only here, so
# engine_tests keeps full sanitizer strictness.
add_executable(engine_alloc_tests test_allocations.cpp)
target_link_libraries(engine_alloc_tests
  PRIVATE pmlab_engine pmlab_engine_warnings GTest::gtest_main)

# Golden files: CPython computes expected values at build time (stdlib only, deterministic).
# Without an interpreter the golden tests skip; the pytest parity gate still covers parity.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  set(_golden_dir ${CMAKE_CURRENT_BINARY_DIR}/golden)
  set(_golden_script ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/gen_golden_mechanics.py)
  add_custom_command(
    OUTPUT ${_golden_dir}/pyround2.txt ${_golden_dir}/mechanics.txt
    COMMAND ${Python3_EXECUTABLE} ${_golden_script} ${_golden_dir}
    DEPENDS ${_golden_script}
    COMMENT "Generating mechanics golden files with ${Python3_EXECUTABLE}")
  add_custom_target(engine_golden_files
    DEPENDS ${_golden_dir}/pyround2.txt ${_golden_dir}/mechanics.txt)
  add_dependencies(engine_tests engine_golden_files)
  target_compile_definitions(engine_tests PRIVATE PMLAB_GOLDEN_DIR="${_golden_dir}")
endif()

include(GoogleTest)
# PRE_TEST: discover at ctest time, not build time — a build-time run of an ASan binary trips
# macOS's malloc nano zone unless MallocNanoZone=0 is exported (see workflow / README).
gtest_discover_tests(engine_tests DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(engine_alloc_tests DISCOVERY_MODE PRE_TEST
  PROPERTIES ENVIRONMENT "ASAN_OPTIONS=alloc_dealloc_mismatch=0")
