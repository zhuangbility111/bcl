#pragma once

#include <limits>
#include <cstdio>
#include <cmath>

namespace {
template <typename T>
bool is_equal_fp(T a, T b, T epsilon = 128 * std::numeric_limits<T>::epsilon()) {
  // printf("Calling is equal_fp... %lf %lf\n", a, b);
  if (a == b) {
    return true;
  }

  T abs_th = std::numeric_limits<T>::min();

  T diff = std::abs(a - b);

  T norm = std::min(std::abs(a) + std::abs(b), std::numeric_limits<T>::max());
  return diff < std::max(abs_th, epsilon * norm);
}
}

template <typename T, typename U>
bool is_equal(T x, U y) {
  return x == y;
}

inline bool is_equal(float x, float y) {
  return is_equal_fp(x, y);
}

inline bool is_equal(double x, double y) {
  return is_equal_fp(x, y);
}
