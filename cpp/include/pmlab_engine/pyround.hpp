#pragma once

namespace pmlab {

// CPython's round(x, 2), bit for bit.
//
// Python rounds the EXACT binary value of x to 2 decimal digits (correct rounding, ties to
// even on the exact halfway cases, which only occur for dyadic rationals like 0.125), then
// returns the nearest double. That is not std::round(x * 100) / 100 — naive scaling introduces
// a second rounding step that disagrees on ~1 in 1e3 realistic price values.
//
// Implementation: snprintf("%.2f") + strtod round trip. This is literally CPython's own
// fallback implementation of round() (Python/floatobject.c, double_round without short-float
// repr), and macOS/glibc printf both do correctly-rounded decimal conversion. The claim is
// enforced by a generated golden-file fuzz suite (scripts/gen_golden_mechanics.py) that
// replays Python's answers against this function.
double pyround2(double x);

}  // namespace pmlab
