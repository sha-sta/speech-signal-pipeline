#pragma once

namespace pmlab {

// Engine version, bumped per milestone PR. The Python side reports this from the binding so a
// parity failure can always be tied to an exact engine build.
inline constexpr int kEngineVersionMajor = 0;
inline constexpr int kEngineVersionMinor = 1;

const char* engine_version();

}  // namespace pmlab
