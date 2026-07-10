cmake_minimum_required(VERSION 3.24)
project(pmlab_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PMLAB_ENGINE_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(PMLAB_ENGINE_TESTS "Build unit tests" ON)

add_library(pmlab_engine_warnings INTERFACE)
target_compile_options(pmlab_engine_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

add_library(pmlab_engine_sanitizers INTERFACE)
if(PMLAB_ENGINE_SANITIZE)
  target_compile_options(pmlab_engine_sanitizers INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(pmlab_engine_sanitizers INTERFACE -fsanitize=address,undefined)
endif()

add_library(pmlab_engine STATIC
  src/version.cpp
)
target_include_directories(pmlab_engine PUBLIC include)
target_link_libraries(pmlab_engine
  PRIVATE pmlab_engine_warnings
  PUBLIC pmlab_engine_sanitizers)

if(PMLAB_ENGINE_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  )
  # Keep gtest out of our warning/install surface.
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()
