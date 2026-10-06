// Independent pseudo-spectral reference for the 3D Orszag-Tang MHD vortex of De Rosis PRE 2017 (Sec. III D),
// in the physical units used by M3LB's validation/orszag_tang_3d.cpp and by OT3D_RR:
//   u = [-2 sin y, 2 sin x, 0],  b = 0.8 [-2 sin 2y + sin z, 2 sin x + sin z, sin x + sin y],  nu = eta = 2 pi / Re,
//   periodic cube [0, 2 pi)^3.
// Incompressible MHD (rho = 1, mu0 = 1):  du/dt = P[-div(u u - b b)] + nu lap u,  db/dt = curl(u x b) + eta lap b,
// Ns^3 grid, 2/3 de-aliasing, integrating-factor RK4 (exact diffusion), radix-2 FFT, threaded.
// Output times: the union of the logarithmic sample times of the LBM runs on the grids M (as in M3LB / OT3D_RR),
// plus, for every M, the peak current / vorticity obtained by the LBM's second-order central differences on the exact field
// sampled (M <= Ns) or spectrally interpolated (M > Ns) on the M^3 lattice.
//   usage: spectral3d_ref Ns Re tmax dt_max M1,M2,... out.txt [Ma=0.034] [threads] [tstop] [dump steps k1,k2 of lattice M1] [dump prefix]
//   environment: PROBES=n VTI=k SPEC_PREFIX=p  (sample at M3LB GPU probe times; spectra at its frame times)
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

struct Solver {
  int N; size_t n3; FFT3 fft; double nu;
  std::vector<double> kv; std::vector<char> mask; std::vector<double> kx, ky, kz, k2;
  vec uh[3], bh[3];
  double hcache = -1; std::vector<double> Eh, Ef;
  vec ph[3], pb[3], T[6], E[3];
  Solver(int n, double nu_) : N(n), n3((size_t)n * n * n), fft(n), nu(nu_) {
    kv.resize(N); for (int i = 0; i < N; ++i) kv[i] = (i <= N / 2) ? i : i - N;
    kx.resize(n3); ky.resize(n3); kz.resize(n3); k2.resize(n3); mask.resize(n3);
    const int kmax = N / 3;
    for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
      size_t id = ((size_t)iz * N + iy) * N + ix;
      kx[id] = kv[ix]; ky[id] = kv[iy]; kz[id] = kv[iz];
      k2[id] = kx[id] * kx[id] + ky[id] * ky[id] + kz[id] * kz[id];
      mask[id] = (std::abs(kv[ix]) <= kmax && std::abs(kv[iy]) <= kmax && std::abs(kv[iz]) <= kmax);
    }
    for (int c = 0; c < 3; ++c) { uh[c].assign(n3, 0); bh[c].assign(n3, 0); ph[c].assign(n3, 0); pb[c].assign(n3, 0); E[c].assign(n3, 0); }
    for (int c = 0; c < 6; ++c) T[c].assign(n3, 0);
  }
  void init() {
    vec u[3], b[3];
    for (int c = 0; c < 3; ++c) { u[c].assign(n3, 0); b[c].assign(n3, 0); }
    for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
      size_t id = ((size_t)iz * N + iy) * N + ix;
      double X = 2 * M_PI * ix / N, Y = 2 * M_PI * iy / N, Z = 2 * M_PI * iz / N;
      u[0][id] = -2 * sin(Y); u[1][id] = 2 * sin(X); u[2][id] = 0;
      b[0][id] = 0.8 * (-2 * sin(2 * Y) + sin(Z)); b[1][id] = 0.8 * (2 * sin(X) + sin(Z)); b[2][id] = 0.8 * (sin(X) + sin(Y));
    }
    for (int c = 0; c < 3; ++c) { fft(u[c], false); fft(b[c], false); uh[c] = u[c]; bh[c] = b[c]; }
    for (size_t i = 0; i < n3; ++i) if (!mask[i]) for (int c = 0; c < 3; ++c) { uh[c][i] = 0; bh[c][i] = 0; }
  }
  // nonlinear terms for the state (au, ab) (Fourier space), result in (nu_, nb_)
  void nonlinear(const vec* au, const vec* ab, vec* nu_, vec* nb_) {
    for (int c = 0; c < 3; ++c) { ph[c] = au[c]; pb[c] = ab[c]; fft(ph[c], true); fft(pb[c], true); }
    pfor(n3, [&](size_t bgn, size_t end) {
      for (size_t i = bgn; i < end; ++i) {
        double u0 = ph[0][i].real(), u1 = ph[1][i].real(), u2 = ph[2][i].real();
        double b0 = pb[0][i].real(), b1 = pb[1][i].real(), b2 = pb[2][i].real();
        T[0][i] = u0 * u0 - b0 * b0; T[1][i] = u1 * u1 - b1 * b1; T[2][i] = u2 * u2 - b2 * b2;
        T[3][i] = u0 * u1 - b0 * b1; T[4][i] = u0 * u2 - b0 * b2; T[5][i] = u1 * u2 - b1 * b2;
        E[0][i] = u1 * b2 - u2 * b1; E[1][i] = u2 * b0 - u0 * b2; E[2][i] = u0 * b1 - u1 * b0;
      }
    });
    for (int c = 0; c < 6; ++c) fft(T[c], false);
    for (int c = 0; c < 3; ++c) fft(E[c], false);
    const cd I(0, 1);
    pfor(n3, [&](size_t bgn, size_t end) {
      for (size_t i = bgn; i < end; ++i) {
        if (!mask[i]) { for (int c = 0; c < 3; ++c) { nu_[c][i] = 0; nb_[c][i] = 0; } continue; }
        const double a = kx[i], b = ky[i], c = kz[i];
        cd f0 = -I * (a * T[0][i] + b * T[3][i] + c * T[4][i]);
        cd f1 = -I * (a * T[3][i] + b * T[1][i] + c * T[5][i]);
        cd f2 = -I * (a * T[4][i] + b * T[5][i] + c * T[2][i]);
        if (k2[i] > 0) { cd d = (a * f0 + b * f1 + c * f2) / k2[i]; f0 -= a * d; f1 -= b * d; f2 -= c * d; }
        nu_[0][i] = f0; nu_[1][i] = f1; nu_[2][i] = f2;
        nb_[0][i] = I * (b * E[2][i] - c * E[1][i]);
        nb_[1][i] = I * (c * E[0][i] - a * E[2][i]);
        nb_[2][i] = I * (a * E[1][i] - b * E[0][i]);
      }
    });
  }
  void setfactors(double h) {
    if (h == hcache) return;
    hcache = h; Eh.resize(n3); Ef.resize(n3);
    pfor(n3, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) { Eh[i] = exp(-nu * k2[i] * h / 2); Ef[i] = exp(-nu * k2[i] * h); } });
  }
  void step(double h) {  // integrating-factor RK4
    setfactors(h);
    vec au[3], ab[3], Au[3], Ab[3], Bu[3], Bb[3], Cu[3], Cb[3], Du[3], Db[3];
    for (int c = 0; c < 3; ++c) for (vec* v : {&au[c], &ab[c], &Au[c], &Ab[c], &Bu[c], &Bb[c], &Cu[c], &Cb[c], &Du[c], &Db[c]}) v->assign(n3, 0);
    nonlinear(uh, bh, Au, Ab);
    pfor(n3, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) for (int c = 0; c < 3; ++c) {
      au[c][i] = Eh[i] * (uh[c][i] + 0.5 * h * Au[c][i]); ab[c][i] = Eh[i] * (bh[c][i] + 0.5 * h * Ab[c][i]); } });
    nonlinear(au, ab, Bu, Bb);
    pfor(n3, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) for (int c = 0; c < 3; ++c) {
      au[c][i] = Eh[i] * uh[c][i] + 0.5 * h * Bu[c][i]; ab[c][i] = Eh[i] * bh[c][i] + 0.5 * h * Bb[c][i]; } });
    nonlinear(au, ab, Cu, Cb);
    pfor(n3, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) for (int c = 0; c < 3; ++c) {
      au[c][i] = Ef[i] * uh[c][i] + h * Eh[i] * Cu[c][i]; ab[c][i] = Ef[i] * bh[c][i] + h * Eh[i] * Cb[c][i]; } });
    nonlinear(au, ab, Du, Db);
    pfor(n3, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) for (int c = 0; c < 3; ++c) {
      uh[c][i] = Ef[i] * uh[c][i] + h / 6. * (Ef[i] * Au[c][i] + 2. * Eh[i] * (Bu[c][i] + Cu[c][i]) + Du[c][i]);
      bh[c][i] = Ef[i] * bh[c][i] + h / 6. * (Ef[i] * Ab[c][i] + 2. * Eh[i] * (Bb[c][i] + Cb[c][i]) + Db[c][i]); } });
  }
};

// physical fields on the M^3 grid: point samples of the Ns-grid fields (M <= Ns, Ns % M == 0) or spectral interpolation (M > Ns)
static void fields_on_grid(const Solver& S, int M, std::vector<double>* uf, std::vector<double>* bf) {
  const int N = S.N; const size_t m3 = (size_t)M * M * M;
  if (M <= N) {
    const int s = N / M;
    for (int c = 0; c < 3; ++c) {
      vec ut = S.uh[c], bt = S.bh[c];
      S.fft(ut, true); S.fft(bt, true);
      uf[c].assign(m3, 0); bf[c].assign(m3, 0);
      for (int iz = 0; iz < M; ++iz) for (int iy = 0; iy < M; ++iy) for (int ix = 0; ix < M; ++ix) {
        size_t id = ((size_t)iz * M + iy) * M + ix, jd = ((size_t)(iz * s) * N + iy * s) * N + ix * s;
        uf[c][id] = ut[jd].real(); bf[c][id] = bt[jd].real();
      }
    }
  } else {
    FFT3 fm(M);
    const double sc = (double)m3 / (double)S.n3;
    for (int c = 0; c < 3; ++c) {
      vec up(m3, 0), bp(m3, 0);
      for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
        if (ix == N / 2 || iy == N / 2 || iz == N / 2) continue;
        int jx = (ix < N / 2) ? ix : M - (N - ix), jy = (iy < N / 2) ? iy : M - (N - iy), jz = (iz < N / 2) ? iz : M - (N - iz);
        size_t id = ((size_t)iz * N + iy) * N + ix, jd = ((size_t)jz * M + jy) * M + jx;
        up[jd] = S.uh[c][id] * sc; bp[jd] = S.bh[c][id] * sc;
      }
      fm(up, true); fm(bp, true);
      uf[c].resize(m3); bf[c].resize(m3);
      for (size_t i = 0; i < m3; ++i) { uf[c][i] = up[i].real(); bf[c][i] = bp[i].real(); }
    }
  }
}

// lattice central-difference peak of |curl f| on the M^3 grid, in physical units (dx = 2 pi / M)
static double curl_fd_max(const std::vector<double>* f, int M) {
  const double idx2 = (double)M / (4. * M_PI);
  double mx = 0;
  auto id = [&](int x, int y, int z) { return ((size_t)((z % M + M) % M) * M + ((y % M + M) % M)) * M + ((x % M + M) % M); };
  for (int z = 0; z < M; ++z) for (int y = 0; y < M; ++y) for (int x = 0; x < M; ++x) {
    double jx = idx2 * ((f[2][id(x, y + 1, z)] - f[2][id(x, y - 1, z)]) - (f[1][id(x, y, z + 1)] - f[1][id(x, y, z - 1)]));
    double jy = idx2 * ((f[0][id(x, y, z + 1)] - f[0][id(x, y, z - 1)]) - (f[2][id(x + 1, y, z)] - f[2][id(x - 1, y, z)]));
    double jz = idx2 * ((f[1][id(x + 1, y, z)] - f[1][id(x - 1, y, z)]) - (f[0][id(x, y + 1, z)] - f[0][id(x, y - 1, z)]));
    mx = std::max(mx, std::sqrt(jx * jx + jy * jy + jz * jz));
  }
  return mx;
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: spectral3d Ns Re tmax dt_max M1,M2,... out.txt [Ma=0.034] [threads]\n"); return 1; }
  const int Ns = atoi(argv[1]); const double Re = atof(argv[2]), tmax = atof(argv[3]), dtmax = atof(argv[4]);
  std::vector<int> Ms; { std::string s = argv[5]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); Ms.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
  const std::string outp = argv[6];
  const double Ma = (argc > 7) ? atof(argv[7]) : 0.034;
  if (argc > 8) NT = atoi(argv[8]);
  const double tstop = (argc > 9) ? atof(argv[9]) : 1e30;
  // optional field dumps at LBM steps k (of the first lattice M): argv[10] = "k1,k2,...", argv[11] = file prefix
  std::vector<double> dumpt; std::string dprefix = "dump";
  if (argc > 11) {
    std::string s = argv[10]; size_t p = 0;
    const double dtM0 = (Ma / (2. * std::sqrt(2.) * std::sqrt(3.))) * (2 * M_PI / Ms[0]);
    while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); dumpt.push_back(atol(s.substr(p, q - p).c_str()) * dtM0); p = q + 1; }
    dprefix = argv[11];
  }
  const double nu = 2 * M_PI / Re;
  const double v0 = Ma / (2. * std::sqrt(2.) * std::sqrt(3.));

  // output times: union of the logarithmic LBM sample times (t = k dt_M, as in M3LB / OT3D_RR)
  std::set<double> times; times.insert(0.0);
  // PROBES=n (environment): sample at the probe times of M3LB's GPU orszag_tang driver instead (for every M:
  // T = floor(tmax/dt_M), probe = T/n, samples at the multiples of probe and at T); VTI=k: energy spectra at every
  // k-th probe, i.e. at the times of the driver's .vti frames, written to SPEC_PREFIX_M{M}_t{t}.txt.
  const int env_probes = getenv("PROBES") ? atoi(getenv("PROBES")) : 0;
  const int env_vti = getenv("VTI") ? atoi(getenv("VTI")) : 0;
  const std::string spec_prefix = getenv("SPEC_PREFIX") ? getenv("SPEC_PREFIX") : "spec";
  std::vector<std::pair<double, int>> spec_times;      // (t, M) at which to write spectra
  if (env_probes > 0) {
    for (int M : Ms) {
      const double dtM = (Ma / (2. * std::sqrt(2.) * std::sqrt(3.))) * (2 * M_PI / M);
      const long T = (long)(tmax / dtM);
      const long probe = (T / env_probes) ? T / env_probes : 1;
      int probe_no = 0;
      for (long t = 1; t <= T; ++t) {
        if (t % probe == 0 || t == T) {
          ++probe_no;
          times.insert((double)t * dtM);
          if (env_vti > 0 && probe_no % env_vti == 0) spec_times.push_back({(double)t * dtM, M});
        }
      }
    }
    if (env_vti > 0) for (int M : Ms) spec_times.push_back({0.0, M});
  }
  for (int M : (env_probes > 0 ? std::vector<int>() : Ms)) {
    const double dtM = v0 * (2 * M_PI / M);
    const long T = (long)(tmax / dtM);
    std::vector<long> ks;
    const double t0log = 0.05; const int nlog = 60;
    for (int i = 0; i < nlog; ++i) { double t = t0log * std::pow(tmax / t0log, (double)i / (nlog - 1)); long k = (long)(t / dtM); if (k > 0 && k <= T && (ks.empty() || k != ks.back())) ks.push_back(k); }
    if (ks.empty() || ks.back() != T) ks.push_back(T);
    for (long k : ks) times.insert((double)k * dtM);
  }
  for (double td : dumpt) times.insert(td);
  Solver S(Ns, nu);
  S.init();
  FILE* fo = fopen(outp.c_str(), "w");
  fprintf(fo, "# Ns=%d Re=%g nu=%.8g dt_max=%g  columns: t Gamma Ek Em Jmax Wmax", Ns, Re, nu, dtmax);
  for (int M : Ms) fprintf(fo, " Jfd%d Wfd%d", M, M);
  fprintf(fo, " tailJ\n");
  double E0 = -1, t = 0;
  for (double tt : times) {
    if (tt > tstop + 1e-12) break;
    if (tt > t) {
      const long n = (long)std::ceil((tt - t) / dtmax - 1e-12);
      const double h = (tt - t) / n;
      for (long s = 0; s < n; ++s) S.step(h);
      t = tt;
    }
    // diagnostics
    vec u[3], b[3], w[3], j[3];
    for (int c = 0; c < 3; ++c) { u[c] = S.uh[c]; b[c] = S.bh[c]; }
    const cd I(0, 1);
    for (int c = 0; c < 3; ++c) { w[c].assign(S.n3, 0); j[c].assign(S.n3, 0); }
    for (size_t i = 0; i < S.n3; ++i) {
      const double a = S.kx[i], bb = S.ky[i], cc = S.kz[i];
      w[0][i] = I * (bb * S.uh[2][i] - cc * S.uh[1][i]); w[1][i] = I * (cc * S.uh[0][i] - a * S.uh[2][i]); w[2][i] = I * (a * S.uh[1][i] - bb * S.uh[0][i]);
      j[0][i] = I * (bb * S.bh[2][i] - cc * S.bh[1][i]); j[1][i] = I * (cc * S.bh[0][i] - a * S.bh[2][i]); j[2][i] = I * (a * S.bh[1][i] - bb * S.bh[0][i]);
    }
    for (int c = 0; c < 3; ++c) { S.fft(u[c], true); S.fft(b[c], true); S.fft(w[c], true); S.fft(j[c], true); }
    double eu = 0, eb = 0, jm = 0, wm = 0;
    for (size_t i = 0; i < S.n3; ++i) {
      eu += std::pow(u[0][i].real(), 2) + std::pow(u[1][i].real(), 2) + std::pow(u[2][i].real(), 2);
      eb += std::pow(b[0][i].real(), 2) + std::pow(b[1][i].real(), 2) + std::pow(b[2][i].real(), 2);
      jm = std::max(jm, std::sqrt(std::pow(j[0][i].real(), 2) + std::pow(j[1][i].real(), 2) + std::pow(j[2][i].real(), 2)));
      wm = std::max(wm, std::sqrt(std::pow(w[0][i].real(), 2) + std::pow(w[1][i].real(), 2) + std::pow(w[2][i].real(), 2)));
    }
    eu /= S.n3; eb /= S.n3;
    // spectral tail of the current: share of |j_hat|^2 in the modes closest to the de-aliasing cutoff
    double tail = 0, tot = 0;
    {
      const int kmax = Ns / 3;
      for (size_t i = 0; i < S.n3; ++i) {
        const double a = S.kx[i], bb = S.ky[i], cc = S.kz[i];
        cd jx = I * (bb * S.bh[2][i] - cc * S.bh[1][i]), jy = I * (cc * S.bh[0][i] - a * S.bh[2][i]), jz = I * (a * S.bh[1][i] - bb * S.bh[0][i]);
        const double e2 = std::norm(jx) + std::norm(jy) + std::norm(jz);
        tot += e2;
        const double km = std::max(std::abs(a), std::max(std::abs(bb), std::abs(cc)));
        if (km >= kmax - 3) tail += e2;
      }
    }
    if (E0 < 0) E0 = eu;
    fprintf(fo, "%.10f %.12e %.12e %.12e %.12e %.12e", t, eu / E0, 0.5 * eu, 0.5 * eb, jm, wm);
    for (int M : Ms) {
      std::vector<double> uf[3], bf[3];
      fields_on_grid(S, M, uf, bf);
      fprintf(fo, " %.12e %.12e", curl_fd_max(bf, M), curl_fd_max(uf, M));
    }
    fprintf(fo, " %.4e\n", tot > 0 ? tail / tot : 0.0); fflush(fo);
    for (const auto& st : spec_times) if (std::abs(st.first - t) < 1e-12) {
      const int nsh = int(std::ceil(std::sqrt(3.) * S.N / 2)) + 2;
      std::vector<double> Eu(nsh, 0.), Eb(nsh, 0.);
      const double nrm = 1. / ((double)S.n3 * (double)S.n3);
      for (size_t i = 0; i < S.n3; ++i) {
        const int k = (int)(std::sqrt(S.k2[i]) + 0.5);
        for (int c = 0; c < 3; ++c) { Eu[k] += 0.5 * std::norm(S.uh[c][i]) * nrm; Eb[k] += 0.5 * std::norm(S.bh[c][i]) * nrm; }
      }
      char nm[512];
      snprintf(nm, sizeof nm, "%s_M%d_t%.6f.txt", spec_prefix.c_str(), st.second, t);
      FILE* fs = fopen(nm, "w");
      fprintf(fs, "# pseudo-spectral Ns=%d Re=%g t=%.10f (probe time of the M=%d lattice); columns: k E_u(k) E_b(k)\n", S.N, Re, t, st.second);
      for (int k = 0; k < nsh; ++k) fprintf(fs, "%d %.10e %.10e\n", k, Eu[k], Eb[k]);
      fclose(fs);
    }
    for (double td : dumpt) if (std::abs(td - t) < 1e-12) {
      const int M = Ms[0];
      std::vector<double> uf[3], bf[3];
      fields_on_grid(S, M, uf, bf);
      const double idx2 = (double)M / (4. * M_PI);
      auto id = [&](int x, int y, int z) { return ((size_t)((z % M + M) % M) * M + ((y % M + M) % M)) * M + ((x % M + M) % M); };
      std::vector<float> jcd((size_t)M * M * M), jex(S.n3);
      for (int z = 0; z < M; ++z) for (int y = 0; y < M; ++y) for (int x = 0; x < M; ++x) {
        const std::vector<double>* f = bf;
        double jx = idx2 * ((f[2][id(x, y + 1, z)] - f[2][id(x, y - 1, z)]) - (f[1][id(x, y, z + 1)] - f[1][id(x, y, z - 1)]));
        double jy = idx2 * ((f[0][id(x, y, z + 1)] - f[0][id(x, y, z - 1)]) - (f[2][id(x + 1, y, z)] - f[2][id(x - 1, y, z)]));
        double jz = idx2 * ((f[1][id(x + 1, y, z)] - f[1][id(x - 1, y, z)]) - (f[0][id(x, y + 1, z)] - f[0][id(x, y - 1, z)]));
        jcd[id(x, y, z)] = (float)std::sqrt(jx * jx + jy * jy + jz * jz);
      }
      for (size_t i = 0; i < S.n3; ++i) jex[i] = (float)std::sqrt(std::pow(j[0][i].real(), 2) + std::pow(j[1][i].real(), 2) + std::pow(j[2][i].real(), 2));
      char nm[512];
      snprintf(nm, sizeof nm, "%s_jcd_M%d_t%.6f.bin", dprefix.c_str(), M, t);
      FILE* fd = fopen(nm, "wb"); fwrite(jcd.data(), 4, jcd.size(), fd); fclose(fd);
      snprintf(nm, sizeof nm, "%s_jex_Ns%d_t%.6f.bin", dprefix.c_str(), S.N, t);
      fd = fopen(nm, "wb"); fwrite(jex.data(), 4, jex.size(), fd); fclose(fd);
      fprintf(stderr, "dumped fields at t = %.6f\n", t);
    }
    fprintf(stderr, "t = %.4f  Gamma = %.8f  Jmax = %.6f  Wmax = %.6f\n", t, eu / E0, jm, wm);
  }
  fclose(fo);
  return 0;
}
