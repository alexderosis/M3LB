// N^3 complex FFT for the post-processing tools (spectra_vti.cpp), the same code as in
// spectral3d_ref.cpp: radix 2, so N must be a power of two; the N^2 lines of each direction
// are shared among NT threads (std::thread, no OpenMP); forward unnormalised, inverse divided
// by N^3; index (z*N + y)*N + x, x fastest -- the layout of the .vti frames. It does NOT do
// real-to-complex transforms (a real field is transformed as complex, twice the memory).
#pragma once
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>
#include <set>
using cd = std::complex<double>;
using vec = std::vector<cd>;

static int NT = 8;
static void pfor(size_t n, const std::function<void(size_t, size_t)>& fn) {
  int nt = std::min<size_t>(NT, n);
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t) {
    size_t b = n * t / nt, e = n * (t + 1) / nt;
    th.emplace_back([=, &fn] { fn(b, e); });
  }
  for (auto& x : th) x.join();
}

struct FFT3 {
  int N; std::vector<cd> tw;
  explicit FFT3(int n) : N(n), tw(n / 2) { for (int k = 0; k < n / 2; ++k) tw[k] = std::polar(1.0, -2.0 * M_PI * k / n); }
  void line(cd* a, size_t stride, bool inv, cd* tmp) const {
    for (int i = 0; i < N; ++i) tmp[i] = a[(size_t)i * stride];
    for (int i = 1, j = 0; i < N; ++i) {
      int bit = N >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) std::swap(tmp[i], tmp[j]);
    }
    for (int len = 2; len <= N; len <<= 1) {
      int step = N / len;
      for (int i = 0; i < N; i += len)
        for (int k = 0; k < len / 2; ++k) {
          cd w = tw[(size_t)k * step];
          if (inv) w = std::conj(w);
          cd u = tmp[i + k], v = tmp[i + k + len / 2] * w;
          tmp[i + k] = u + v;
          tmp[i + k + len / 2] = u - v;
        }
    }
    for (int i = 0; i < N; ++i) a[(size_t)i * stride] = tmp[i];
  }
  // index = (iz*N + iy)*N + ix ; forward unnormalised, inverse normalised by 1/N^3
  void operator()(vec& A, bool inv) const {
    const size_t n2 = (size_t)N * N;
    pfor(n2, [&](size_t b, size_t e) { std::vector<cd> tmp(N); for (size_t l = b; l < e; ++l) line(&A[l * N], 1, inv, tmp.data()); });
    pfor(n2, [&](size_t b, size_t e) { std::vector<cd> tmp(N);
      for (size_t l = b; l < e; ++l) { size_t iz = l / N, ix = l % N; line(&A[iz * n2 + ix], N, inv, tmp.data()); } });
    pfor(n2, [&](size_t b, size_t e) { std::vector<cd> tmp(N);
      for (size_t l = b; l < e; ++l) line(&A[l], n2, inv, tmp.data()); });
    if (inv) { double s = 1.0 / ((double)N * N * N); pfor(n2 * N, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) A[i] *= s; }); }
  }
};

