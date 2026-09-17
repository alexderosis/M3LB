//==============================================================================
//  Decaying MHD turbulence confined in a SPHERE, by volume penalisation.
//
//  The CUDA twin of ../demonstrator/mhd_sphere.cpp. The two share no headers and
//  are supposed to agree; where they do not, one of them is wrong. That is not
//  a slogan here -- the sibling port, GPU/src/mhd_decay.cu, shipped with four
//  uninitialised pointers that made its penalisation a silent no-op, and the
//  only instrument that found it was the twin.
//
//  WHY THIS EXISTS RATHER THAN A KOKKOS+CUDA BUILD OF THE PARENT. The parent
//  runs on MhdCentralMomentsShifted<D3Q27>, which tests/frame_check.sh measures
//  at a 752-byte stack frame with four register-indexed arrays -- second only to
//  ColourGradient. On a CPU that is L1-resident and free; in device code it is
//  per-thread LOCAL memory, off-chip and uncoalesced, and this tree has measured
//  that mechanism at 47x. So the parent is not fit for a device as it stands,
//  and this codebase -- which is D3Q27 natively and writes its collisions out --
//  is the shorter road to a fine grid.
//
//  THAT CLAIM IS NOT PROVED FOR *THIS* SIDE. GPU/'s own to_moments indexes
//  k[mi(a,b,c)] with runtime a, b, c too, and frame_check reads host assembly,
//  so it cannot answer for nvcc. Build with -DLBM_PTXAS_VERBOSE=ON and read the
//  registers and spills per kernel before believing this operator is cheap on a
//  device. The honest state is: the parent is measured bad, this one is
//  unmeasured.
//
//  BOTH WALLS, AND THEY ARE DIFFERENT PHYSICS. -mwall cond (default) penalises
//  toward B.n = 0, the PERFECT CONDUCTOR of Neffaa's Eq. (2) -- a conductor
//  coated inside with insulant, so the field may not penetrate while the
//  tangential component is free. -mwall insul penalises toward B = 0, the
//  INSULATOR, and lets flux leave through the wall. The parent measured the
//  contrast at N = 96, Re = 500, t/T_e = 2: E_b 2.3x lower for the insulator,
//  <j^2> 2.4x lower, E_u/E_B 0.29 against 0.22, i.e. the selective decay is
//  weaker rather than absent. Not a numerical switch.
//
//  Bn/B IS THE WRONG DIAGNOSTIC FOR THE INSULATOR and B_sh is here because of
//  it. For a conductor |B.n|/|B| on the shell measures the condition; for an
//  insulator |B| itself goes to zero there, the ratio becomes 0/0 and reports
//  round-off -- the parent measured 0.0013 against 0.32 for exactly that
//  reason, which looks like failure and is not. B_sh, |B| on the shell over the
//  interior r.m.s., is O(1) for a conductor and collapses three decades for an
//  insulator, and is well conditioned for both.
//
//  THE INITIAL CONDITION IS SOLENOIDAL BY CONSTRUCTION, not approximately: both
//  fields are the discrete curl of a random vector potential, and div(curl) of
//  commuting central differences vanishes identically. div_b at t = 0 must read
//  machine zero and anything else is the measurement -- it is printed on the
//  first line for that reason. The mode sum is O(N^3 n_modes) with no FFT
//  anywhere in this tree, so the trigonometry is table-driven per axis and the
//  shell is capped at kmax.
//
//  THE PENALISATION IS EXPLICIT and must be rebuilt from the CURRENT fields
//  every step; refreshed on the probe interval it would be a thousand-step-old
//  velocity and would enforce nothing. What it buys is that u and b are
//  penalised through the SAME chi, so the two boundaries are co-located by
//  construction with no half-cell mismatch to price.
//
//  WHAT THIS IS NOT. There is no published reference for a sphere, so nothing
//  here is a validation and it prints no PASS/FAIL. And the magnetic
//  penalisation INJECTS div B -- the parent measured rms(div b)/rms(|j|) at
//  0.19-0.28 at N = 96 and found it identical for both walls, so it is not a
//  property of the B.n projection and reshaping the source will not fix it.
//  Read |J| magnitude; do not read J direction.
//
//  -mwall cond HAS A RESOLUTION CEILING BETWEEN N = 96 AND N = 128, AND THE TWO
//  PARAMETERS THAT POSTPONE IT DO NOT CURE IT. Measured 2026-09-17 on a T4, at
//  Re = 500, kmax 6, k0 4, matched seed, two turnovers (32N steps), one flag
//  apart from an insulating run that survives every N:
//
//      N     omega    outcome                blow-up
//      96    1.912    survives 2 T_e         --
//      128   1.884    non-finite             step 448   t/Te 0.219
//      160   1.857    non-finite             step 460   t/Te 0.180
//      192   1.831    non-finite             step 432   t/Te 0.141
//      224   1.806    non-finite             step 420   t/Te 0.117
//
//  THE BLOW-UP STEP IS FIXED AND THE BLOW-UP TIME IS NOT -- 440 +- 20 steps at
//  every N while t/Te falls monotonically. The timescale belongs to the lattice,
//  not to the flow. It is NOT omega -> 2, and the ladder runs the wrong way to
//  read it that way: nu = u0 2R/Re grows with N, so omega FALLS from 1.912 to
//  1.806 across it and a relaxation instability would have taken the coarse
//  grids first. And it is not a defect of this port: at N = 96 this file agrees
//  with ../demonstrator/mhd_sphere.cpp -- no shared headers -- to -0.008 % on
//  E_u and -1.6 % on E_b at t/Te = 2, with Bn/B equal to four decimals at every
//  sampled point.
//
//  epsm IS THE STIFFNESS KNOB AND smooth IS NOT A CONTROL. At N = 128, epsm =
//  0.5/1/2 fail at step 208/288/448 and epsm = 4/8 survive -- monotonic, and the
//  epsilon >~ dt bound of the penalisation, with the threshold no longer O(1).
//  smooth = 1/1.333/2/2.667 gives fail/fail/survive/fail, which is not monotonic
//  and must not be read as one. NEITHER TRANSFERS: at N = 224, smooth = 2,
//  smooth = 2.5 and epsm = 4 still go non-finite at steps 448, 544 and 672, and
//  by then epsm = 4 has degraded Bn/B from 0.0005 to 0.0141 -- it is no longer
//  the boundary condition being asked for. Matching the layer as a FRACTION of R
//  does not help either (smooth = 1.333 at N = 128 is the same 2.6 % of R as
//  smooth = 1 at N = 96, and dies at step 480), so what matters is an absolute
//  number of cells. THE MECHANISM IS NOT DIAGNOSED. Series and the full sweep in
//  results/N_mhd_sphere/cond_ladder/.
//
//  So: -mwall insul is the wall to use above N = 96, and it is the one the
//  N = 224 series was run with. A conducting run above the ceiling is not a
//  result no matter how good its first two hundred steps look.
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/ehd.cuh"          // Field: the host/device-neutral N-element array
#include "lbm/vti.cuh"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Three components, strided by NT. mhd_decay's pair are the 2-D versions of
// these and are deliberately not reused: this file shares no headers with it
// either, and a functor that silently zeroed u_z would be invisible here.
struct FieldInit {
  const Real* a; int nx, ny; long NT;
  LBM_HD Macro operator()(int x, int y, int z) const {
    const long n = long(x) + long(nx) * (long(y) + long(ny) * long(z));
    Macro m; m.rho = Real(1);
    m.ux = a[n]; m.uy = a[n + NT]; m.uz = a[n + 2 * NT];
    return m;
  }
};
struct FieldBInit {
  const Real* b; int nx, ny; long NT;
  LBM_HD void operator()(int x, int y, int z, Real B[3]) const {
    const long n = long(x) + long(nx) * (long(y) + long(ny) * long(z));
    B[0] = b[n]; B[1] = b[n + NT]; B[2] = b[n + 2 * NT];
  }
};
struct ZeroU {
  LBM_HD void operator()(int, int, int, Real u[3]) const { u[0] = u[1] = u[2] = Real(0); }
};

//------------------------------------------------------------------------------
struct Opts {
  int N = 64;
  double rfac = 0.40, u0 = 0.05, Re = 200.0, prm = 1.0, alf = 1.0, cost = 0.0;
  double eps = 2.0, epsm = 2.0, smooth = 1.0, k0 = 3.0;
  int kmax = 4;
  // 99 to MATCH THE PARENT (demonstrator/mhd_sphere.cpp). The two share the RNG
  // and the IC construction, so a matched seed makes the twin comparison real;
  // mhd_decay shipped with 99 here and 12345 there and compared unrelated
  // realisations for it.
  std::uint64_t seed = 99;
  std::size_t steps = 4096, probe = 256, vti = 0, dump = 0, dumpvol = 0;
  Op op = Op::CentralMoments;
  std::string mwall = "cond";
};

// xorshift, so the initial condition does not depend on the host's RNG. Same
// generator as the parent: a matched seed is what makes the twin comparison a
// comparison rather than two unrelated realisations.
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t sd) : s(0x9E3779B97F4A7C15ULL ^ sd) { (*this)(); }
  double operator()() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return double(s >> 11) * (1.0 / 9007199254740992.0);
  }
};

//------------------------------------------------------------------------------
// A solenoidal random field: the discrete curl of a vector potential built from
// the integer shell 1 <= |k| <= kmax. The trigonometry is assembled from
// per-axis tables so the inner loop has no transcendental call, which is what
// makes the O(N^3 n_modes) sum affordable without an FFT.
//------------------------------------------------------------------------------
static void solenoidal(int N, int kmax, double k0, Rng& rng,
                       std::vector<double> v[3]) {
  const std::size_t NN = std::size_t(N) * N * N;
  std::vector<double> A[3];
  for (int a = 0; a < 3; ++a) { A[a].assign(NN, 0.0); v[a].assign(NN, 0.0); }
  std::vector<double> cx(N), sx(N), cy(N), sy(N), cz(N), sz(N);
  const double dl = 2.0 * M_PI / double(N);

  for (int kx = -kmax; kx <= kmax; ++kx)
    for (int ky = -kmax; ky <= kmax; ++ky)
      for (int kz = -kmax; kz <= kmax; ++kz) {
        const double k2 = double(kx * kx + ky * ky + kz * kz);
        if (k2 < 0.5 || k2 > double(kmax * kmax)) continue;
        if (kx < 0 || (kx == 0 && ky < 0) || (kx == 0 && ky == 0 && kz < 0)) continue;
        const double k = std::sqrt(k2);
        const double amp = std::sqrt(k2 * k2 * std::exp(-2.0 * k2 / (k0 * k0))) / (k * k);
        for (int i = 0; i < N; ++i) {
          cx[i] = std::cos(dl * kx * i); sx[i] = std::sin(dl * kx * i);
          cy[i] = std::cos(dl * ky * i); sy[i] = std::sin(dl * ky * i);
          cz[i] = std::cos(dl * kz * i); sz[i] = std::sin(dl * kz * i);
        }
        for (int a = 0; a < 3; ++a) {
          const double ph = 2.0 * M_PI * rng(), cph = std::cos(ph), sph = std::sin(ph);
          const double aa = amp * (2.0 * rng() - 1.0);
          double* Aa = A[a].data();
          for (int z = 0; z < N; ++z)
            for (int y = 0; y < N; ++y) {
              const double cyz = cy[y] * cz[z] - sy[y] * sz[z];
              const double syz = sy[y] * cz[z] + cy[y] * sz[z];
              const double p = aa * (cyz * cph - syz * sph);
              const double q = aa * (syz * cph + cyz * sph);
              double* row = Aa + (std::size_t(z) * N + y) * N;
              for (int x = 0; x < N; ++x) row[x] += p * cx[x] - q * sx[x];
            }
        }
      }

  auto ix = [N](int x, int y, int z) { return (std::size_t(z) * N + y) * N + x; };
  auto w = [N](int i) { return (i % N + N) % N; };
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const std::size_t n = ix(x, y, z);
        const std::size_t xp = ix(w(x + 1), y, z), xm = ix(w(x - 1), y, z);
        const std::size_t yp = ix(x, w(y + 1), z), ym = ix(x, w(y - 1), z);
        const std::size_t zp = ix(x, y, w(z + 1)), zm = ix(x, y, w(z - 1));
        v[0][n] = 0.5 * (A[2][yp] - A[2][ym]) - 0.5 * (A[1][zp] - A[1][zm]);
        v[1][n] = 0.5 * (A[0][zp] - A[0][zm]) - 0.5 * (A[2][xp] - A[2][xm]);
        v[2][n] = 0.5 * (A[1][xp] - A[1][xm]) - 0.5 * (A[0][yp] - A[0][ym]);
      }
}

//------------------------------------------------------------------------------
// The penalisation pass: F = -chi u / eps, and S = -chi (B.n) n / eps_m for a
// conductor or -chi B / eps_m for an insulator. Both from the CURRENT fields.
//
// EVERY MEMBER IS INITIALISED. The sibling driver declared these with none, the
// call site set most of them, and the four it missed were read every step --
// penalisation fluid AND magnetic was a silent no-op while the bulk energies
// stayed within a few percent of the parent, so the only symptom was the wall
// diagnostic. nullptr defaults plus the check at the call site turn the next
// omission into a refusal at step 0.
//------------------------------------------------------------------------------
struct PenParams {
  const Real *chi = nullptr, *nx_ = nullptr, *ny_ = nullptr, *nz_ = nullptr;
  const Real *ux = nullptr, *uy = nullptr, *uz = nullptr;
  const Real *bx = nullptr, *by = nullptr, *bz = nullptr;
  Real *Fx = nullptr, *Fy = nullptr, *Fz = nullptr;
  Real *Sx = nullptr, *Sy = nullptr, *Sz = nullptr;
  Real ie = Real(0), iem = Real(0);
  int insul = 0;
  long N = 0;
};

LBM_HD LBM_INLINE void penalise_node(const PenParams& p, long n) {
  const Real c = p.chi[n];
  p.Fx[n] = -c * p.ux[n] * p.ie;
  p.Fy[n] = -c * p.uy[n] * p.ie;
  p.Fz[n] = -c * p.uz[n] * p.ie;
  if (p.insul) {
    p.Sx[n] = -c * p.bx[n] * p.iem;
    p.Sy[n] = -c * p.by[n] * p.iem;
    p.Sz[n] = -c * p.bz[n] * p.iem;
  } else {
    const Real bn = p.bx[n] * p.nx_[n] + p.by[n] * p.ny_[n] + p.bz[n] * p.nz_[n];
    p.Sx[n] = -c * bn * p.nx_[n] * p.iem;
    p.Sy[n] = -c * bn * p.ny_[n] * p.iem;
    p.Sz[n] = -c * bn * p.nz_[n] * p.iem;
  }
}

#if defined(__CUDACC__)
__global__ void penalise_kernel(PenParams p) {
  const long n = blockIdx.x * long(blockDim.x) + threadIdx.x;
  if (n < p.N) penalise_node(p, n);
}
#endif

static void penalise(const PenParams& p) {
#if defined(__CUDACC__)
  constexpr int B = 128;
  penalise_kernel<<<int((p.N + B - 1) / B), B>>>(p);
#else
  for (long n = 0; n < p.N; ++n) penalise_node(p, n);
#endif
}

//------------------------------------------------------------------------------
// Field dumps for the animation pipeline, in the format validation/FieldDump.hpp
// writes and results/N_mhd_sphere/render_*.py reads: a slice is int32 nx, int32
// ny then nx*ny float32; a volume is int32 nx, ny, nz then the floats, x fastest.
//
// WRITTEN OUT HERE rather than including FieldDump.hpp, for the same reason
// vti.cuh is: GPU/ shares no headers with the Kokkos side. The FORMAT is shared
// on purpose, though -- it is what lets one renderer read both codebases, and a
// format agreed by two independent writers is worth more than one they inherit.
//
// THE SLICES ARE TINY AND THE VOLUME IS NOT: at N = 256 a slice is 256 kB and
// jvol is 67 MB. A hundred-frame animation is 26 MB of slices against 6.7 GB of
// volume, so dump the volume on a coarser cadence than the slices when the run
// is large -- -dumpvol takes its own interval for that.
//------------------------------------------------------------------------------
static void write_raw2(const std::string& path, int nx, int ny,
                       const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}
static void write_raw3(const std::string& path, int nx, int ny, int nz,
                       const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny, c = nz;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(&c), sizeof c);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

//------------------------------------------------------------------------------
struct Diag {
  double eu = 0, eb = 0, ratio = 0, hc = 0, ens = 0, j2 = 0, cosjb = 0;
  double bmean = 0, bmax = 0, divb = 0, bn = 0, bshell = 0, mass = 0;
  bool finite = true;
};

int main(int argc, char** argv) {
  Opts o;
  bool probe_given = false;
  auto next = [&](int& i) { return (i + 1 < argc) ? argv[++i] : (char*)"0"; };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "-n")      o.N = std::atoi(next(i));
    else if (a == "-rfac")   o.rfac = std::atof(next(i));
    else if (a == "-u")      o.u0 = std::atof(next(i));
    else if (a == "-re")     o.Re = std::atof(next(i));
    else if (a == "-prm")    o.prm = std::atof(next(i));
    else if (a == "-alf")    o.alf = std::atof(next(i));
    else if (a == "-cost")   o.cost = std::atof(next(i));
    else if (a == "-eps")    o.eps = std::atof(next(i));
    else if (a == "-epsm")   o.epsm = std::atof(next(i));
    else if (a == "-smooth") o.smooth = std::atof(next(i));
    else if (a == "-k0")     o.k0 = std::atof(next(i));
    else if (a == "-kmax")   o.kmax = std::atoi(next(i));
    else if (a == "-seed")   o.seed = std::strtoull(next(i), nullptr, 10);
    else if (a == "-steps")  o.steps = std::size_t(std::atoll(next(i)));
    else if (a == "-probe")  { o.probe = std::size_t(std::atoll(next(i))); probe_given = true; }
    else if (a == "-vti")    o.vti = std::size_t(std::atoll(next(i)));
    else if (a == "-dump")   o.dump = std::size_t(std::atoll(next(i)));
    else if (a == "-dumpvol") o.dumpvol = std::size_t(std::atoll(next(i)));
    else if (a == "-mwall")  o.mwall = next(i);
    else if (a == "-op") {
      const std::string s = next(i);
      o.op = (s == "bgk") ? Op::BGK : (s == "trt") ? Op::TRT : Op::CentralMoments;
    } else if (a == "-h" || a == "--help") {
      std::printf("mhd_sphere [-n N] [-rfac f] [-re Re] [-prm Pm] [-alf a] [-u u0]\n"
                  "  [-cost c] [-eps e] [-epsm e] [-smooth w] [-k0 k] [-kmax k]\n"
                  "  [-seed s] [-steps n] [-probe n] [-vti n] [-mwall cond|insul]\n"
                  "  [-op cm|bgk|trt]\n");
      return 0;
    }
  }
  if (o.mwall != "cond" && o.mwall != "insul") {
    std::fprintf(stderr, "-mwall must be cond or insul, not '%s'\n", o.mwall.c_str());
    return 2;
  }
  // A FRAME MUST LAND ON A PROBE -- meta.txt and the .pvd carry that probe's
  // energies and time -- but that is a DIVISIBILITY requirement and not a
  // licence to overwrite -probe. The first version snapped probe = dump and
  // silently threw away the diagnostic resolution that was asked for: -probe 256
  // with -vti 2560 produced three rows instead of twenty, and the time series
  // was the poorer for it with nothing said. Now -probe only defaults to the
  // frame interval when it was not given, and a cadence that cannot line up is
  // refused by name rather than quietly adjusted.
  if (!o.dumpvol) o.dumpvol = o.dump;
  if (!probe_given) {
    if (o.dump)      o.probe = o.dump;
    else if (o.vti)  o.probe = o.vti;
  } else {
    auto must_divide = [&](const char* nm, std::size_t iv) {
      if (iv && o.probe && iv % o.probe != 0) {
        std::fprintf(stderr,
            "%s %zu is not a multiple of -probe %zu. A frame has to land on a "
            "probe, because meta.txt and the .pvd carry that probe's energies "
            "and time. Pick a frame interval that divides by the probe (or drop "
            "-probe and it will follow the frames).\n", nm, iv, o.probe);
        std::exit(2);
      }
    };
    must_divide("-dump", o.dump);
    must_divide("-dumpvol", o.dumpvol);
    must_divide("-vti", o.vti);
  }

  const int N = o.N;
  const long NT = long(N) * N * N;
  const double R = o.rfac * double(N), c = 0.5 * double(N - 1);
  const double nu = o.u0 * 2.0 * R / o.Re, eta = nu / o.prm;
  const double T_e = 2.0 * R / o.u0;
  const bool insul = (o.mwall == "insul");

  auto id  = [&](int x, int y, int z) { return (long(z) * N + y) * N + x; };
  auto rad = [&](int x, int y, int z) {
    const double dx = x - c, dy = y - c, dz = z - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

  //---- the initial condition, on the host ---------------------------------------
  Rng rng(o.seed);
  std::vector<double> hu[3], hb[3];
  solenoidal(N, o.kmax, o.k0, rng, hu);
  solenoidal(N, o.kmax, o.k0, rng, hb);
  {
    double nin = 0, su[3] = {0, 0, 0}, sb[3] = {0, 0, 0};
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          if (rad(x, y, z) <= R) {
            const std::size_t k = std::size_t(id(x, y, z));
            nin += 1;
            for (int a = 0; a < 3; ++a) { su[a] += hu[a][k]; sb[a] += hb[a][k]; }
          }
    for (int a = 0; a < 3; ++a) { su[a] /= nin; sb[a] /= nin; }
    for (std::size_t k = 0; k < std::size_t(NT); ++k)
      for (int a = 0; a < 3; ++a) { hu[a][k] -= su[a]; hb[a][k] -= sb[a]; }

    double uu = 0, ub = 0;
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          if (rad(x, y, z) <= R) {
            const std::size_t k = std::size_t(id(x, y, z));
            for (int a = 0; a < 3; ++a) { uu += hu[a][k] * hu[a][k]; ub += hu[a][k] * hb[a][k]; }
          }
    const double proj = (uu > 0) ? ub / uu : 0.0;
    for (std::size_t k = 0; k < std::size_t(NT); ++k)
      for (int a = 0; a < 3; ++a) hb[a][k] -= proj * hu[a][k];

    double e0u = 0, e0b = 0;
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          if (rad(x, y, z) <= R) {
            const std::size_t k = std::size_t(id(x, y, z));
            for (int a = 0; a < 3; ++a) { e0u += hu[a][k] * hu[a][k]; e0b += hb[a][k] * hb[a][k]; }
          }
    // r.m.s. of the MAGNITUDE, not per component: u0 is the scale that enters
    // nu = u0 2R/Re and the Mach number, and a /3 here would make the flow
    // sqrt(3) faster than asked for.
    const double gu = o.u0 / std::sqrt(e0u / nin);
    const double gb = (o.u0 / std::sqrt(o.alf)) / std::sqrt(e0b / nin);
    for (std::size_t k = 0; k < std::size_t(NT); ++k)
      for (int a = 0; a < 3; ++a) { hu[a][k] *= gu; hb[a][k] *= gb; }
    if (o.cost != 0.0) {
      const double ct = o.cost, st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
      const double s = 1.0 / std::sqrt(o.alf);   // |b|/|u| after the normalisation
      for (std::size_t k = 0; k < std::size_t(NT); ++k)
        for (int a = 0; a < 3; ++a) hb[a][k] = ct * s * hu[a][k] + st * hb[a][k];
    }
  }

  //---- solvers ------------------------------------------------------------------
  backend::Magnetic mag(N, N, N, Real(eta));
  backend::Fluid    fl (N, N, N, o.op, Real(nu));
  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  Field chi(NT), pnx(NT), pny(NT), pnz(NT);
  Field Fx(NT), Fy(NT), Fz(NT), Sx(NT), Sy(NT), Sz(NT);
  {
    const std::size_t NTS = std::size_t(NT);
    std::vector<Real> h(NTS), hx(NTS), hy(NTS), hz(NTS);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const std::size_t n = std::size_t(id(x, y, z));
          const double r = rad(x, y, z);
          h[n] = Real(0.5 * (1.0 + std::tanh((r - R) / o.smooth)));
          const double ir = (r > 1e-9) ? 1.0 / r : 0.0;
          hx[n] = Real((x - c) * ir); hy[n] = Real((y - c) * ir); hz[n] = Real((z - c) * ir);
        }
    chi.from_host(h); pnx.from_host(hx); pny.from_host(hy); pnz.from_host(hz);
  }
  BodyForce bf; bf.Fx = Fx.data(); bf.Fy = Fy.data(); bf.Fz = Fz.data();
  fl.set_force(bf, ForceField);
  mag.set_source(Sx.data(), Sy.data(), Sz.data());

  Field u0f(3 * NT), b0f(3 * NT);
  {
    std::vector<Real> a(std::size_t(3 * NT)), b(std::size_t(3 * NT));
    for (std::size_t k = 0; k < std::size_t(NT); ++k)
      for (int d = 0; d < 3; ++d) {
        a[k + std::size_t(d) * std::size_t(NT)] = Real(hu[d][k]);
        b[k + std::size_t(d) * std::size_t(NT)] = Real(hb[d][k]);
      }
    u0f.from_host(a); b0f.from_host(b);
  }
  fl.initialise_with(FieldInit{u0f.data(), N, N, NT});
  mag.initialise_with(FieldBInit{b0f.data(), N, N, NT}, ZeroU{});

  PenParams pp;
  pp.chi = chi.data();
  pp.nx_ = pnx.data(); pp.ny_ = pny.data(); pp.nz_ = pnz.data();
  pp.ux = fl.ux_device(); pp.uy = fl.uy_device(); pp.uz = fl.uz_device();
  pp.bx = mag.Bx_device(); pp.by = mag.By_device(); pp.bz = mag.Bz_device();
  pp.Fx = Fx.data(); pp.Fy = Fy.data(); pp.Fz = Fz.data();
  pp.Sx = Sx.data(); pp.Sy = Sy.data(); pp.Sz = Sz.data();
  pp.ie = Real(1.0 / o.eps); pp.iem = Real(1.0 / o.epsm);
  pp.insul = insul ? 1 : 0; pp.N = NT;
  if (!pp.ux || !pp.uy || !pp.uz || !pp.bx || !pp.by || !pp.bz) {
    std::fprintf(stderr, "penalisation has no field to read -- refusing to run\n");
    return 1;
  }

  std::printf("Confined decaying MHD in a SPHERE, volume-penalised   CUDA twin\n");
  std::printf("precision %s   D3Q27 fluid + D3Q7 magnetic   operator %s\n",
              sizeof(Real) == 8 ? "FP64" : "FP32",
              o.op == Op::BGK ? "BGK" : o.op == Op::TRT ? "TRT" : "CM");
  std::printf("  N %d   R %.2f (rfac %.2f)   Re %.0f   Pm %.2f   u0 %.4f\n",
              N, R, o.rfac, o.Re, o.prm, o.u0);
  std::printf("  nu %.6e   eta %.6e   magnetic wall: %s (%s)\n", nu, eta,
              insul ? "INSULATING" : "perfectly conducting",
              insul ? "B = 0" : "B.n = 0");
  std::printf("  eps %.2f / %.2f   smooth %.2f   IC kmax %d  k0 %.1f  seed %llu\n",
              o.eps, o.epsm, o.smooth, o.kmax, o.k0, (unsigned long long)o.seed);
  std::printf("  one turnover 2R/u0 = %.0f steps; running %zu (%.1f turnovers)\n\n",
              T_e, o.steps, double(o.steps) / T_e);
  std::printf("   step   t/Te      E_u        E_b      E_u/E_b   H_c    enstr"
              "     <j^2>    cos(J,B)  |<b>|    max|b|   dv/cl    Bn/B   B_sh   dmass\n");

  std::vector<std::pair<double, std::string>> pvd;
  std::FILE* meta = nullptr;
  int dframe = 0;
  if (o.dump) {
    meta = std::fopen("anim_frames/meta.txt", "w");
    if (!meta) {
      std::fprintf(stderr, "cannot open anim_frames/meta.txt -- does the "
                           "directory exist?\n");
      return 1;
    }
    std::fprintf(meta, "N %d\nR %.6f\nTe %.6f\n", N, R, T_e);
  }
  std::vector<Real> hr, hux, huy, huz, hbx, hby, hbz;
  double mass0 = 0;
  int vframe = 0;

  for (std::size_t t = 0; t <= o.steps; ++t) {
    if (t % o.probe == 0 || t == o.steps) {
      fl.macroscopic_to_host(hr, hux, huy, huz);
      mag.field_to_host(hbx, hby, hbz);

      Diag g;
      double nin = 0, nbulk = 0, nshell = 0, jb = 0, bb = 0, jj = 0, mb[3] = {0, 0, 0};
      auto w = [N](int i) { return (i % N + N) % N; };
      for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
          for (int x = 0; x < N; ++x) {
            const std::size_t n = std::size_t(id(x, y, z));
            const double r = rad(x, y, z);
            const double a = hux[n], b = huy[n], e = huz[n];
            if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(e) ||
                !std::isfinite(double(hr[n]))) g.finite = false;
            g.mass += double(hr[n]);
            const double bxv = hbx[n], byv = hby[n], bzv = hbz[n];
            if (r <= R) {
              nin += 1;
              g.eu += 0.5 * (a * a + b * b + e * e);
              g.eb += 0.5 * (bxv * bxv + byv * byv + bzv * bzv);
              g.hc += a * bxv + b * byv + e * bzv;
              g.bmax = std::max(g.bmax, std::sqrt(bxv * bxv + byv * byv + bzv * bzv));
              mb[0] += bxv; mb[1] += byv; mb[2] += bzv;
            }
            if (r <= R - 2.0) {
              nbulk += 1;
              const std::size_t xp = std::size_t(id(w(x + 1), y, z)), xm = std::size_t(id(w(x - 1), y, z));
              const std::size_t yp = std::size_t(id(x, w(y + 1), z)), ym = std::size_t(id(x, w(y - 1), z));
              const std::size_t zp = std::size_t(id(x, y, w(z + 1))), zm = std::size_t(id(x, y, w(z - 1)));
              const double wx = 0.5 * (huz[yp] - huz[ym]) - 0.5 * (huy[zp] - huy[zm]);
              const double wy = 0.5 * (hux[zp] - hux[zm]) - 0.5 * (huz[xp] - huz[xm]);
              const double wz = 0.5 * (huy[xp] - huy[xm]) - 0.5 * (hux[yp] - hux[ym]);
              g.ens += 0.5 * (wx * wx + wy * wy + wz * wz);
              const double jx = 0.5 * (hbz[yp] - hbz[ym]) - 0.5 * (hby[zp] - hby[zm]);
              const double jy = 0.5 * (hbx[zp] - hbx[zm]) - 0.5 * (hbz[xp] - hbz[xm]);
              const double jz = 0.5 * (hby[xp] - hby[xm]) - 0.5 * (hbx[yp] - hbx[ym]);
              g.j2 += jx * jx + jy * jy + jz * jz;
              jj += jx * jx + jy * jy + jz * jz;
              bb += bxv * bxv + byv * byv + bzv * bzv;
              jb += jx * bxv + jy * byv + jz * bzv;
              const double db = 0.5 * (hbx[xp] - hbx[xm]) + 0.5 * (hby[yp] - hby[ym])
                              + 0.5 * (hbz[zp] - hbz[zm]);
              g.divb += db * db;
            }
            if (r > R - 1.0 && r < R + 1.0 && r > 1e-9) {
              const double bm = std::sqrt(bxv * bxv + byv * byv + bzv * bzv);
              if (bm > 1e-30) {
                g.bn += std::fabs((bxv * (x - c) + byv * (y - c) + bzv * (z - c)) / r) / bm;
                g.bshell += bm;
                nshell += 1;
              }
            }
          }
      if (nin > 0) {
        g.eu /= nin; g.eb /= nin; g.hc /= nin;
        g.bmean = std::sqrt(mb[0] * mb[0] + mb[1] * mb[1] + mb[2] * mb[2]) / nin;
      }
      if (nbulk > 0) { g.ens /= nbulk; g.j2 /= nbulk; g.divb /= nbulk; }
      g.divb = (g.j2 > 1e-300) ? std::sqrt(g.divb / g.j2) : 0.0;
      if (nshell > 0) {
        g.bn /= nshell;
        const double brms = (g.eb > 0) ? std::sqrt(2.0 * g.eb) : 0.0;
        g.bshell = (brms > 1e-30) ? g.bshell / nshell / brms : 0.0;
      }
      g.ratio = (g.eb > 0) ? g.eu / g.eb : 0.0;
      const double hn = std::sqrt(4.0 * g.eu * g.eb);
      g.hc = (hn > 1e-30) ? g.hc / hn : 0.0;
      g.cosjb = (jj > 0 && bb > 0) ? jb / std::sqrt(jj * bb) : 0.0;
      if (t == 0) mass0 = g.mass;

      std::printf("  %6zu %6.3f %10.3e %10.3e %8.3f %7.4f %9.3e %9.3e %8.4f"
                  " %8.2e %8.2e %8.2e %7.4f %6.3f %9.2e\n",
                  t, double(t) / T_e, g.eu, g.eb, g.ratio, g.hc, g.ens, g.j2,
                  g.cosjb, g.bmean, g.bmax, g.divb, g.bn, g.bshell,
                  mass0 != 0 ? (g.mass - mass0) / mass0 : 0.0);
      std::fflush(stdout);

      // t % dump, not just `o.dump`: the block runs inside the PROBE, and a
      // probe is now allowed to be finer than a frame. Without this the slices
      // came out once per probe while the volume came out once per dumpvol --
      // 7 slice frames against the 3 asked for, with meta.txt numbering them as
      // if they were the frames.
      if (o.dump && t % o.dump == 0) {
        const int zc = N / 2;
        std::vector<float> su(std::size_t(N) * N), sb(std::size_t(N) * N),
                           sj(std::size_t(N) * N);
        std::vector<float> vol;
        const bool wantvol = (t % o.dumpvol == 0);
        if (wantvol) vol.resize(std::size_t(NT));
        for (int z = 0; z < N; ++z)
          for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
              const std::size_t n = std::size_t(id(x, y, z));
              const std::size_t xp = std::size_t(id(w(x + 1), y, z)), xm = std::size_t(id(w(x - 1), y, z));
              const std::size_t yp = std::size_t(id(x, w(y + 1), z)), ym = std::size_t(id(x, w(y - 1), z));
              const std::size_t zp = std::size_t(id(x, y, w(z + 1))), zm = std::size_t(id(x, y, w(z - 1)));
              const double jx = 0.5 * (hbz[yp] - hbz[ym]) - 0.5 * (hby[zp] - hby[zm]);
              const double jy = 0.5 * (hbx[zp] - hbx[zm]) - 0.5 * (hbz[xp] - hbz[xm]);
              const double jz = 0.5 * (hby[xp] - hby[xm]) - 0.5 * (hbx[yp] - hbx[ym]);
              const double jm = std::sqrt(jx * jx + jy * jy + jz * jz);
              if (wantvol) vol[n] = float(jm);
              if (z == zc) {
                const std::size_t k = std::size_t(y) * N + x;
                su[k] = float(std::sqrt(hux[n] * hux[n] + huy[n] * huy[n] + huz[n] * huz[n]));
                sb[k] = float(std::sqrt(hbx[n] * hbx[n] + hby[n] * hby[n] + hbz[n] * hbz[n]));
                sj[k] = float(jm);
              }
            }
        char nm[96];
        std::snprintf(nm, sizeof nm, "anim_frames/umag_%04d.raw", dframe);
        write_raw2(nm, N, N, su);
        std::snprintf(nm, sizeof nm, "anim_frames/bmag_%04d.raw", dframe);
        write_raw2(nm, N, N, sb);
        std::snprintf(nm, sizeof nm, "anim_frames/jmag_%04d.raw", dframe);
        write_raw2(nm, N, N, sj);
        if (wantvol) {
          std::snprintf(nm, sizeof nm, "anim_frames/jvol_%04d.raw", dframe);
          write_raw3(nm, N, N, N, vol);
        }
        if (meta) {
          std::fprintf(meta, "frame %d %.6f %.8e %.8e\n", dframe,
                       double(t) / T_e, g.eu, g.eb);
          std::fflush(meta);
        }
        ++dframe;
      }

      if (o.vti && t % o.vti == 0) {
        const std::size_t np = std::size_t(NT);
        VtiArray am{"Jmag", 1, {}}, ar{"rho", 1, {}}, ac{"chi", 1, {}};
        VtiArray au{"u", 3, {}}, ab{"b", 3, {}}, aj{"J", 3, {}};
        am.data.reserve(np); ar.data.reserve(np); ac.data.reserve(np);
        au.data.reserve(3 * np); ab.data.reserve(3 * np); aj.data.reserve(3 * np);
        std::vector<Real> hc; chi.to_host(hc);
        for (int z = 0; z < N; ++z)
          for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
              const std::size_t n = std::size_t(id(x, y, z));
              const std::size_t xp = std::size_t(id(w(x + 1), y, z)), xm = std::size_t(id(w(x - 1), y, z));
              const std::size_t yp = std::size_t(id(x, w(y + 1), z)), ym = std::size_t(id(x, w(y - 1), z));
              const std::size_t zp = std::size_t(id(x, y, w(z + 1))), zm = std::size_t(id(x, y, w(z - 1)));
              const double jx = 0.5 * (hbz[yp] - hbz[ym]) - 0.5 * (hby[zp] - hby[zm]);
              const double jy = 0.5 * (hbx[zp] - hbx[zm]) - 0.5 * (hbz[xp] - hbz[xm]);
              const double jz = 0.5 * (hby[xp] - hby[xm]) - 0.5 * (hbx[yp] - hbx[ym]);
              au.data.push_back(float(hux[n])); au.data.push_back(float(huy[n]));
              au.data.push_back(float(huz[n]));
              ab.data.push_back(float(hbx[n])); ab.data.push_back(float(hby[n]));
              ab.data.push_back(float(hbz[n]));
              aj.data.push_back(float(jx)); aj.data.push_back(float(jy));
              aj.data.push_back(float(jz));
              am.data.push_back(float(std::sqrt(jx * jx + jy * jy + jz * jz)));
              ar.data.push_back(float(hr[n]));
              ac.data.push_back(float(hc[n]));
            }
        char nm[64];
        std::snprintf(nm, sizeof nm, "sphere_%04d.vti", vframe);
        if (write_vti_bin(std::string("vti/") + nm, N, N, N, {am, ar, ac, au, ab, aj})) {
          pvd.emplace_back(double(t) / T_e, nm);
          std::printf("  wrote vti/%s\n", nm);
        }
        ++vframe;
      }
      if (!g.finite) { std::printf("\n  *** NON-FINITE at step %zu ***\n", t); break; }
    }
    if (t < o.steps) {
      // Explicit, so it must see u(t) and B(t) rather than the last probe's.
      fl.refresh_velocity();
      mag.compute_field();
      penalise(pp);
      fl.step();
      mag.step();
    }
  }
  if (meta) { std::fprintf(meta, "frames %d\n", dframe); std::fclose(meta); }
  if (o.dump)
    std::printf("\n  wrote %d frames to anim_frames/  (render with "
                "results/N_mhd_sphere/render_slices.py and render_volume.py)\n",
                dframe);
  if (!pvd.empty()) {
    write_pvd("vti/sphere.pvd", pvd);
    std::printf("\n  ParaView: open vti/sphere.pvd  (%zu frames, t/Te 0 to %.2f)\n"
                "  Threshold on chi < 0.5 to clip to the sphere; colour by Jmag.\n",
                pvd.size(), pvd.back().first);
  }
  return 0;
}
