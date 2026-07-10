#pragma once

namespace pmlab {

// CPython's round(x, 2), bit for bit.
//
// Python rounds the EXACT binary value of x to 2 decimal digits (correct rounding, ties to
// even on the exact halfway cases, which only occur for dyadic rationals like 0.125), then
// returns the nearest double. That is not std::round(x * 100) / 100 — naive scaling introduces
// a second rounding step that disagrees on ~1 in 1e3 realistic price values.
//
// Implementation: an exact integer fast path — ax = m * 2^-d with m < 2^53, so
// ax*100 = (m*100) / 2^d exactly in a u64; floor, compare the remainder against half (true
// ties-to-even), then one IEEE division q/100 gives the correctly rounded double. Values whose
// scaled integer exceeds 2^53 fall back to the snprintf("%.2f") + strtod round trip (CPython's
// own fallback implementation of round()). The claim is enforced by a generated golden-file
// fuzz suite (scripts/gen_golden_mechanics.py) that replays Python's answers bit for bit.
double pyround2(double x);

}  // namespace pmlab
