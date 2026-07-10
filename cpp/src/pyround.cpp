#include "pmlab_engine/pyround.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace pmlab {

double pyround2(double x) {
  if (!std::isfinite(x)) {
    return x;  // Python: round(nan, 2) is nan, round(inf, 2) is inf
  }
  // %.2f of the largest double is sign + 309 integer digits + '.' + 2 + NUL.
  char buf[336];
  const int n = std::snprintf(buf, sizeof(buf), "%.2f", x);
  assert(n > 0 && static_cast<unsigned>(n) < sizeof(buf));
  (void)n;
  return std::strtod(buf, nullptr);
}

}  // namespace pmlab
