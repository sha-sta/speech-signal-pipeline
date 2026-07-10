#include "pmlab_engine/pyround.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace pmlab {
namespace {

// Reference path: snprintf + strtod, CPython's own fallback implementation of round(). Exact
// but ~300ns a call — the profiler showed it (dtoa + locale locks) eating a third of the hot
// loop, which is why the integer fast path below exists.
double pyround2_dtoa(double x) {
  // %.2f of the largest double is sign + 309 integer digits + '.' + 2 + NUL.
  char buf[336];
  const int n = std::snprintf(buf, sizeof(buf), "%.2f", x);
  assert(n > 0 && static_cast<unsigned>(n) < sizeof(buf));
  (void)n;
  return std::strtod(buf, nullptr);
}

}  // namespace

double pyround2(double x) {
  if (!std::isfinite(x)) {
    return x;  // Python: round(nan, 2) is nan, round(inf, 2) is inf
  }
  const bool neg = std::signbit(x);
  const double ax = std::fabs(x);

  // Decompose exactly: ax = m * 2^(-d) with integer m < 2^53. frexp/ldexp are exact.
  int e2 = 0;
  const double frac = std::frexp(ax, &e2);  // ax = frac * 2^e2, frac in [0.5, 1) or 0
  const auto m = static_cast<std::uint64_t>(std::ldexp(frac, 53));  // integer, < 2^53
  const int d = 53 - e2;
  if (d <= 0) {
    return x;  // spacing >= 1: ax is an integer-valued double, unchanged by round(x, 2)
  }

  // ax * 100 == (m * 100) / 2^d exactly; m * 100 < 2^60 always fits u64.
  const std::uint64_t n = m * 100;
  if (d >= 61) {
    // n < 2^60 <= 2^(d-1): the value is under half a cent; no tie is representable.
    return neg ? -0.0 : 0.0;
  }
  const std::uint64_t half = std::uint64_t{1} << (d - 1);
  std::uint64_t q = n >> d;                                  // floor(ax * 100)
  const std::uint64_t r = n & ((std::uint64_t{1} << d) - 1);  // exact remainder
  if (r > half || (r == half && (q & 1U))) {
    ++q;  // round up, ties (exact halfway in binary) to even — Python semantics
  }
  if (q >= (std::uint64_t{1} << 53)) {
    return pyround2_dtoa(x);  // double(q) would round: defer to the exact string path
  }
  // Correct single rounding: q is exact in double, IEEE division rounds q/100 to nearest.
  const double res = static_cast<double>(q) / 100.0;
  return neg ? -res : res;
}

}  // namespace pmlab
