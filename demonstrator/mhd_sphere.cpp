//==============================================================================
//  Decaying MHD turbulence confined in a SPHERE, by volume penalisation.
//
//  The three-dimensional sibling of demonstrator/mhd_decay.cpp, which does the
//  same thing in a disc and reproduces Neffaa, Bos & Schneider (2008). Read that
//  banner first: the penalisation argument, the chi/eps convention and the
//  measured offset of the effective wall are all established there and are not
//  repeated. What follows is only what CHANGES in three dimensions.
//
//  WHAT THE OPERATOR MADE POSSIBLE. MhdCentralMoments<D3Q27> static_asserts
//  against forcing, and penalisation IS the confining wall -- it enters as a
//  body force -- so before 4102b6b this case could only have run on MhdBGK and
//  its stability floor. MhdCentralMomentsShifted<D3Q27, FieldGuo> is the
//  default here. In the shifted basis K_F is first-order only, so the operator
//  writes k[mi(1,0,0)] = a/2 and nothing else; that write IS the complete
//  sixth-order Hermite source on D3Q27, proved in
//  MATLAB/d3q27_shifted_force6.py and asserted in validation/forcing_cm.cpp.
//  -op bgk is kept as the control, not as a fallback.
//
//  THREE THINGS DO NOT CARRY OVER FROM THE DISC.
//
//  1. THE IDEAL INVARIANT. In 2-D the mean square vector potential A = 1/2 int
//     a^2 is an ideal invariant and mhd_decay.cpp carries `a` as an advected
//     D2Q5 scalar. In 3-D the potential is a VECTOR and its evolution is
//     dA/dt = u x B - eta J + grad phi, which is NOT advection-diffusion. Three
//     advected scalars would therefore solve the wrong equation, not merely fix
//     a gauge -- so that diagnostic is DROPPED rather than ported. What replaces
//     it is cos(J,B) = <J.B>/sqrt(<|J|^2><|B|^2>), the alignment that measures
//     relaxation toward a force-free state. It needs only the curl this file
//     already computes, it is gauge-free, and it does not inherit div b drift
//     the way a reconstructed potential would. Magnetic helicity int A.B is the
//     quantity one would really want and is NOT computed: it needs a Coulomb
//     gauge inversion, there is no Poisson solver in src/, and on a sphere the
//     gauge fix is its own problem. Stated as a gap, not papered over.
//
//  2. THE INITIAL CONDITION. The disc builds u from one scalar stream function
//     and b from one scalar potential. In 3-D both need a VECTOR potential and
//     the fields are their curls. Solenoidality is then exact rather than
//     approximate, for the same reason it is in 2-D: the discrete curl uses
//     central differences, and div(curl) of commuting central differences
//     vanishes identically. div_b at t = 0 must read machine zero and anything
//     else is the measurement -- it is printed on the first line for that
//     reason.
//
//  3. THE COST OF THE MODE SUM. The direct sum is O(N^3 x n_modes), which at
//     N = 321 would be 500x the 2-D work, and there is no FFT anywhere in this
//     tree. It is kept affordable two ways: the mode set is the integer shell
//     1 <= |k| <= kmax with kmax DEFAULT 4 (about 134 half-space modes), and the
//     trigonometry is done from precomputed per-axis tables so the inner loop
//     has no transcendental call. Measured: 0.6 s at N = 64, kmax = 4. Raising
//     kmax to 8 costs 8x that. This is the reason N is capped in practice, not
//     memory -- 64^3 is 120 MB of an 8 GB machine and 128^3 is 0.9 GB.
//
//  THE MAGNETIC PENALISATION NEEDS ALL THREE SOURCE COMPONENTS, and that is a
//  trap rather than a detail. MagneticSolver::set_source takes three views and
//  MagneticSolver's run_step guards on the FIRST one only, while NC = L::D = 3
//  on D3Q7 -- so a null sz is dereferenced rather than skipped. The disc gets
//  away with passing View1D<Real>() as the third argument because NC = 2 there.
//  All three are allocated here unconditionally. The guard has been added to
//  the solver as well, so the next case cannot repeat it.
//
//  WHAT THE MAGNETIC PENALISATION COSTS: div b, AND IT IS NOT SMALL. Measured
//  at N = 64, Re = 200, one turnover, as rms(div b)/rms(|j|) on the SAME central
//  stencil -- the disc's dv/cl column, so the two are comparable:
//
//      penalisation      dv/cl     Bn/B      E_u
//      both             3.93e-01   0.0018   2.746e-05
//      fluid only       8.68e-03   0.4869   1.656e-05
//      magnetic only    3.14e-01   0.0088   1.099e-04
//      neither          1.10e-02   0.4708   6.247e-05
//
//  The magnetic source carries ALL of it and the fluid penalisation carries
//  none: -chi (B.n) n / eps_m is not solenoidal, so it injects divergence by
//  construction. The disc reaches 7.3e-02 on the same diagnostic at the same Re,
//  so three dimensions cost about 5x, and it does NOT refine away -- 3.99e-01 at
//  N = 48 against 3.93e-01 at N = 64, flat.
//
//  eps_m IS NOT THE KNOB, and believing it was cost a wrong conclusion here.
//  Sweeping it at N = 64 gives dv/cl 3.93e-01 / 3.89e-01 / 3.70e-01 / 3.25e-01
//  at eps_m = 2 / 5 / 20 / 100 while Bn/B goes 0.0018 / 0.0064 / 0.0279 /
//  0.1017 -- a 50x weakening moves the condition being imposed by 56x and the
//  divergence by 17%. The reason is that div b SATURATES: eps_m sets the rate at
//  which the source injects it, but the level is set by where injection balances
//  the resistive decay of the error, and that balance barely moves. So the sweep
//  LOOKS like an isolation test and is not one; only turning each penalisation
//  off separately -- the 2x2 above -- identifies the source. A parameter sweep
//  over a saturating quantity measures the approach, not the cause.
//
//  This is a stated limitation, not a solved problem. Nothing here cleans the
//  divergence: there is no projection, no Powell source, no constrained
//  transport, and the tree has no Poisson solver to build the first with. And it
//  GROWS -- 0.345 / 0.393 / 0.426 / 0.502 at t/T_e = 0.5 / 1 / 2 / 4 -- so the
//  longer the run, the less a b-derived quantity is worth. Read any of them
//  knowing that half the field's gradient is spurious by four turnovers.
//
//  WHAT A RUN LOOKS LIKE (N = 64, Re = 200, Pm = 1, alf = 1, four turnovers,
//  4 threads, 2 min; the whole table is in results/N_mhd_sphere/):
//
//      t/T_e     E_u        E_b     E_u/E_b   cos(J,B)   |<b>|     Bn/B
//      0.0    1.250e-03  1.250e-03   1.000     0.118    7.8e-17   0.5029
//      1.0    2.746e-05  8.850e-05   0.310     0.231    1.7e-03   0.0018
//      2.0    3.527e-06  3.087e-05   0.114     0.222    2.5e-03   0.0011
//      4.0    3.547e-07  1.496e-05   0.024     0.170    3.2e-03   0.0008
//
//  SELECTIVE DECAY IS THE RESULT, and it is the right physics rather than a
//  bug: over four turnovers E_u falls by 3500x and E_b by only 84x, so E_u/E_B
//  goes 1.000 -> 0.024. The magnetic field survives what the velocity field does
//  not, which is what decaying MHD is supposed to do and what the disc shows in
//  two dimensions.
//
//  TWO THINGS IN THAT TABLE ARE NOT ENDORSED. cos(J,B) rises to 0.231 at one
//  turnover and then FALLS to 0.170 -- not the monotone approach to a force-free
//  state one would want to claim, and with dv/cl at 0.43-0.50 over that stretch
//  it cannot be separated from the divergence error. Do not read it as a Taylor
//  relaxation without a run that controls div b. And |<b>| grows from 8e-17 to
//  3.2e-03: the same mean-field drift the disc documents and has not explained,
//  now reproduced in three dimensions, which at least rules out anything
//  peculiar to two.
//
//  THE CONDITION IS B.n = 0, NOT B = 0 -- a perfect conductor coated in
//  insulant, the reference's Eq. (2). The source is -chi (B.n) n / eps_m and it
//  damps the NORMAL component alone; writing -chi B / eps_m instead would impose
//  the insulating wall, which is a different physical problem. `Bn/B` on the
//  shell is printed so the condition can be seen to be enforced rather than
//  assumed.
//
//  WHAT THIS CASE IS NOT. There is no published reference for a sphere, so
//  nothing here is a validation -- this is a demonstrator and it prints no
//  PASS/FAIL. The Reynolds number reachable at a coarse N is far below the
//  disc campaign's 1000: at N = 64 and rfac = 0.40, Re = 1000 would put tau at
//  0.5077, which is on the stability floor. Re = 200 (tau = 0.538) is the
//  default and is a MECHANISM check -- does the sphere confine the flow, does
//  B.n go to zero on the shell, does the energy decay cleanly, does cos(J,B)
//  rise. A Reynolds-matched run needs a bigger grid and is not this.
//
//  Two further gaps, named rather than hidden. The cross-helicity mixing here
//  is a plain Gram-Schmidt plus a two-field rotation to a target cos(theta),
//  NOT the disc's Eq. (11) construction with its bisection. And there is no
//  sharp-wall control: validation/mhd_pipe.cpp shows the 3-D wallcode that one
//  would need, and comparing the two boundary treatments is the obvious next
//  measurement, exactly as the disc's banner does for the circle.
//==============================================================================
#include "Campaign.hpp"
#include "FieldDump.hpp"
#include "collision/MhdBGK.hpp"
#include "collision/MhdCentralMomentsShifted.hpp"
#include "io/VtiWriter.hpp"
#include "io/VtpWriter.hpp"
#include "solver/MagneticSolver.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace lbm;

using FL = D3Q27;
using ML = D3Q7;

using CollS  = MhdCentralMomentsShifted<FL, FieldGuo>;
using CollB  = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations, FieldGuo>;

static constexpr double PI = 3.14159265358979323846;

//------------------------------------------------------------------------------
struct Opts {
  Index N = 64;
  double rfac = 0.40, u0 = 0.05, Re = 200.0, prm = 1.0, alf = 1.0, cost = 0.0;
  double eps = 2.0, epsm = 2.0, smooth = 1.0, k0 = 3.0;
  int kmax = 4;
  unsigned seed = 99;
  std::size_t steps = 20000, probe = 500, dumpevery = 0, vtievery = 0;
  std::string op = "cms";
};

// xorshift, so the initial condition does not depend on the host's RNG
struct Rng {
  unsigned long long s;
  explicit Rng(unsigned sd) : s(0x9E3779B97F4A7C15ULL ^ sd) { (*this)(); }
  double operator()() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return double(s >> 11) * (1.0 / 9007199254740992.0);
  }
};

//------------------------------------------------------------------------------
// A solenoidal random field on an N^3 periodic grid: a vector potential built
// from the integer mode shell 1 <= |k| <= kmax, then its discrete curl.
//
// The trigonometry is table-driven. For mode (kx,ky,kz) and phase p,
//   cos(kx X + ky Y + kz Z + p)
// is assembled from cos/sin(kx X), cos/sin(ky Y), cos/sin(kz Z) by two angle
// additions, so the innermost loop is multiplies and adds only. That is what
// makes the O(N^3 n_modes) sum affordable without an FFT.
//------------------------------------------------------------------------------
static void solenoidal(Index N, int kmax, double k0, Rng& rng,
                       std::vector<double> v[3]) {
  const std::size_t NN = static_cast<std::size_t>(N) * N * N;
  std::vector<double> A[3];
  for (int a = 0; a < 3; ++a) { A[a].assign(NN, 0.0); v[a].assign(NN, 0.0); }

  std::vector<double> cx(N), sx(N), cy(N), sy(N), cz(N), sz(N);
  const double dl = 2.0 * PI / double(N);

  for (int kx = -kmax; kx <= kmax; ++kx)
    for (int ky = -kmax; ky <= kmax; ++ky)
      for (int kz = -kmax; kz <= kmax; ++kz) {
        const double k2 = double(kx * kx + ky * ky + kz * kz);
        if (k2 < 0.5 || k2 > double(kmax * kmax)) continue;
        // half space only: the conjugate half is what makes the sum real
        if (kx < 0 || (kx == 0 && ky < 0) || (kx == 0 && ky == 0 && kz < 0)) continue;
        const double k = std::sqrt(k2);
        // E(k) ~ k^4 exp(-2 (k/k0)^2); A ~ u/k, so amp ~ sqrt(E)/k
        const double amp = std::sqrt(k2 * k2 * std::exp(-2.0 * k2 / (k0 * k0))) / (k * k);

        for (Index i = 0; i < N; ++i) {
          cx[i] = std::cos(dl * kx * double(i)); sx[i] = std::sin(dl * kx * double(i));
          cy[i] = std::cos(dl * ky * double(i)); sy[i] = std::sin(dl * ky * double(i));
          cz[i] = std::cos(dl * kz * double(i)); sz[i] = std::sin(dl * kz * double(i));
        }
        for (int a = 0; a < 3; ++a) {
          const double ph = 2.0 * PI * rng(), cph = std::cos(ph), sph = std::sin(ph);
          const double aa = amp * (2.0 * rng() - 1.0);
          double* Aa = A[a].data();
          for (Index z = 0; z < N; ++z)
            for (Index y = 0; y < N; ++y) {
              // cos(kxX+kyY+kzZ+ph) = cos(S)cos(ph) - sin(S)sin(ph), built up
              const double cyz = cy[y] * cz[z] - sy[y] * sz[z];   // cos(kyY+kzZ)
              const double syz = sy[y] * cz[z] + cy[y] * sz[z];   // sin(kyY+kzZ)
              const double p = aa * (cyz * cph - syz * sph);
              const double q = aa * (syz * cph + cyz * sph);
              double* row = Aa + (static_cast<std::size_t>(z) * N + y) * N;
              for (Index x = 0; x < N; ++x) row[x] += p * cx[x] - q * sx[x];
            }
        }
      }

  // v = curl A, central differences on the periodic grid. div v is then zero
  // identically, because mixed central differences commute.
  auto id = [N](Index x, Index y, Index z) {
    return (static_cast<std::size_t>(z) * N + y) * N + x;
  };
  auto w = [N](Index i) { return (i % N + N) % N; };
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const std::size_t n = id(x, y, z);
        const std::size_t xp = id(w(x + 1), y, z), xm = id(w(x - 1), y, z);
        const std::size_t yp = id(x, w(y + 1), z), ym = id(x, w(y - 1), z);
        const std::size_t zp = id(x, y, w(z + 1)), zm = id(x, y, w(z - 1));
        v[0][n] = 0.5 * (A[2][yp] - A[2][ym]) - 0.5 * (A[1][zp] - A[1][zm]);
        v[1][n] = 0.5 * (A[0][zp] - A[0][zm]) - 0.5 * (A[2][xp] - A[2][xm]);
        v[2][n] = 0.5 * (A[1][xp] - A[1][xm]) - 0.5 * (A[0][yp] - A[0][ym]);
      }
}

//------------------------------------------------------------------------------
// Field dumps for the animation. Three mid-plane slices and the |J| volume --
// the slices because they are cheap and readable, the volume because current
// SHEETS are the thing worth seeing in 3-D MHD and a single plane cuts through
// them at an arbitrary angle.
//
// The exterior is dumped as-is rather than masked: the renderer knows R and is
// the right place to decide what to show, and a file that has already thrown
// the penalised region away cannot be used to check that it really is at rest.
//------------------------------------------------------------------------------
template <class FS, class MS>
static void dump(FS& fl, MS& mag, const Domain& d, Index N, int frame) {
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto uz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto bz = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());

  auto w = [N](Index i) { return (i % N + N) % N; };
  auto jmag = [&](Index x, Index y, Index z) {
    const Index xp = d.id(w(x + 1), y, z), xm = d.id(w(x - 1), y, z);
    const Index yp = d.id(x, w(y + 1), z), ym = d.id(x, w(y - 1), z);
    const Index zp = d.id(x, y, w(z + 1)), zm = d.id(x, y, w(z - 1));
    const double jx = 0.5 * (double(bz(yp)) - double(bz(ym)))
                    - 0.5 * (double(by(zp)) - double(by(zm)));
    const double jy = 0.5 * (double(bx(zp)) - double(bx(zm)))
                    - 0.5 * (double(bz(xp)) - double(bz(xm)));
    const double jz = 0.5 * (double(by(xp)) - double(by(xm)))
                    - 0.5 * (double(bx(yp)) - double(bx(ym)));
    return std::sqrt(jx * jx + jy * jy + jz * jz);
  };

  char p[256];
  const Index zc = N / 2;
  std::snprintf(p, sizeof p, "results/N_mhd_sphere/anim_frames/umag_%04d.raw", frame);
  figdump::scalar_slice(p, N, N, [&](Index x, Index y) {
    const Index n = d.id(x, y, zc);
    return std::sqrt(double(ux(n)) * double(ux(n)) + double(uy(n)) * double(uy(n))
                   + double(uz(n)) * double(uz(n)));
  });
  std::snprintf(p, sizeof p, "results/N_mhd_sphere/anim_frames/bmag_%04d.raw", frame);
  figdump::scalar_slice(p, N, N, [&](Index x, Index y) {
    const Index n = d.id(x, y, zc);
    return std::sqrt(double(bx(n)) * double(bx(n)) + double(by(n)) * double(by(n))
                   + double(bz(n)) * double(bz(n)));
  });
  std::snprintf(p, sizeof p, "results/N_mhd_sphere/anim_frames/jmag_%04d.raw", frame);
  figdump::scalar_slice(p, N, N, [&](Index x, Index y) { return jmag(x, y, zc); });
  std::snprintf(p, sizeof p, "results/N_mhd_sphere/anim_frames/jvol_%04d.raw", frame);
  figdump::scalar_volume(p, N, N, N, jmag);
}


//------------------------------------------------------------------------------
// ParaView output: one .vti per frame plus a .pvd carrying the real times.
//
// SIX ARRAYS, and chi is not padding. Without the penalisation indicator there
// is no way to clip to the sphere inside ParaView, and every rendering would
// include the dead exterior -- which is 73 % of the box at rfac = 0.40, so it
// would dominate any volume rendering or isosurface. Threshold on chi < 0.5 and
// everything downstream sees the fluid only.
//
// J is written as a VECTOR as well as its magnitude: |J| is what you colour by,
// but the vector is what streamlines and the J.B alignment need, and
// recomputing a curl inside ParaView on data that already has a divergence
// problem would only add to it.
//
// 12.6 MB per frame at N = 64. That is the reason -vti has its own interval
// rather than following -dump: 129 frames would be 1.6 GB.
//------------------------------------------------------------------------------
template <class FS, class MS>
static std::string write_frame_vti(FS& fl, MS& mag, View1D<Real> chi,
                                   const Domain& d, Index N, int frame) {
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto uz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
  auto rh = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto bz = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());
  auto ch = Kokkos::create_mirror_view_and_copy(HostSpace{}, chi);

  const std::size_t np = std::size_t(N) * N * N;
  VtiArray au{"u", 3, {}}, ab{"b", 3, {}}, aj{"J", 3, {}};
  VtiArray am{"Jmag", 1, {}}, ar{"rho", 1, {}}, ac{"chi", 1, {}};
  au.data.reserve(3 * np); ab.data.reserve(3 * np); aj.data.reserve(3 * np);
  am.data.reserve(np); ar.data.reserve(np); ac.data.reserve(np);

  auto w = [N](Index i) { return (i % N + N) % N; };
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y, z);
        au.data.push_back(float(ux(n))); au.data.push_back(float(uy(n)));
        au.data.push_back(float(uz(n)));
        ab.data.push_back(float(bx(n))); ab.data.push_back(float(by(n)));
        ab.data.push_back(float(bz(n)));
        const Index xp = d.id(w(x + 1), y, z), xm = d.id(w(x - 1), y, z);
        const Index yp = d.id(x, w(y + 1), z), ym = d.id(x, w(y - 1), z);
        const Index zp = d.id(x, y, w(z + 1)), zm = d.id(x, y, w(z - 1));
        const double jx = 0.5 * (double(bz(yp)) - double(bz(ym)))
                        - 0.5 * (double(by(zp)) - double(by(zm)));
        const double jy = 0.5 * (double(bx(zp)) - double(bx(zm)))
                        - 0.5 * (double(bz(xp)) - double(bz(xm)));
        const double jz = 0.5 * (double(by(xp)) - double(by(xm)))
                        - 0.5 * (double(bx(yp)) - double(bx(ym)));
        aj.data.push_back(float(jx)); aj.data.push_back(float(jy));
        aj.data.push_back(float(jz));
        am.data.push_back(float(std::sqrt(jx * jx + jy * jy + jz * jz)));
        ar.data.push_back(float(rh(n)));
        ac.data.push_back(float(ch(n)));
      }

  char nm[64];
  std::snprintf(nm, sizeof nm, "sphere_%04d.vti", frame);
  write_vti_bin(std::string("results/N_mhd_sphere/vti/") + nm, N, N, N,
                {am, ar, ac, au, ab, aj});
  std::printf("  wrote results/N_mhd_sphere/vti/%s\n", nm);
  return nm;
}

//------------------------------------------------------------------------------
struct Diag {
  double eu, eb, ratio, hc, ens, j2, cosjb, bmean, bmax, divb, bn, mass;
  bool finite;
};

template <class FS, class MS>
static Diag measure(FS& fl, MS& mag, const Domain& d, Index N, double R) {
  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto uz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
  auto rh = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto bz = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());

  const double c = 0.5 * double(N - 1);
  Diag g{}; g.finite = true;
  double nin = 0, nbulk = 0, nshell = 0;
  double jb = 0, bb = 0, jj = 0, mb[3] = {0, 0, 0};

  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const double dx = double(x) - c, dy = double(y) - c, dz = double(z) - c;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        const Index n = d.id(x, y, z);
        const double a = double(ux(n)), b = double(uy(n)), e = double(uz(n));
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(e) ||
            !std::isfinite(double(rh(n)))) g.finite = false;
        g.mass += double(rh(n));
        if (r <= R) {
          nin += 1;
          g.eu += 0.5 * (a * a + b * b + e * e);
          const double bxv = double(bx(n)), byv = double(by(n)), bzv = double(bz(n));
          g.eb += 0.5 * (bxv * bxv + byv * byv + bzv * bzv);
          g.hc += a * bxv + b * byv + e * bzv;
          g.bmax = std::max(g.bmax, std::sqrt(bxv * bxv + byv * byv + bzv * bzv));
          mb[0] += bxv; mb[1] += byv; mb[2] += bzv;
        }
        // derivative quantities need a full stencil inside the body
        if (r <= R - 2.0 && x > 0 && y > 0 && z > 0 && x < N - 1 && y < N - 1 && z < N - 1) {
          nbulk += 1;
          const Index xp = d.id(x + 1, y, z), xm = d.id(x - 1, y, z);
          const Index yp = d.id(x, y + 1, z), ym = d.id(x, y - 1, z);
          const Index zp = d.id(x, y, z + 1), zm = d.id(x, y, z - 1);
          const double wx = 0.5 * (double(uz(yp)) - double(uz(ym)))
                          - 0.5 * (double(uy(zp)) - double(uy(zm)));
          const double wy = 0.5 * (double(ux(zp)) - double(ux(zm)))
                          - 0.5 * (double(uz(xp)) - double(uz(xm)));
          const double wz = 0.5 * (double(uy(xp)) - double(uy(xm)))
                          - 0.5 * (double(ux(yp)) - double(ux(ym)));
          g.ens += 0.5 * (wx * wx + wy * wy + wz * wz);
          const double jx = 0.5 * (double(bz(yp)) - double(bz(ym)))
                          - 0.5 * (double(by(zp)) - double(by(zm)));
          const double jy = 0.5 * (double(bx(zp)) - double(bx(zm)))
                          - 0.5 * (double(bz(xp)) - double(bz(xm)));
          const double jz = 0.5 * (double(by(xp)) - double(by(xm)))
                          - 0.5 * (double(bx(yp)) - double(bx(ym)));
          g.j2 += jx * jx + jy * jy + jz * jz;
          jj += jx * jx + jy * jy + jz * jz;
          const double bxv = double(bx(n)), byv = double(by(n)), bzv = double(bz(n));
          bb += bxv * bxv + byv * byv + bzv * bzv;
          jb += jx * bxv + jy * byv + jz * bzv;
          // div b as a RATIO to the curl on the SAME stencil: an absolute
          // divergence is meaningless without the field's own gradient scale,
          // and this is the disc's dv/cl column.
          const double db = 0.5 * (double(bx(xp)) - double(bx(xm)))
                          + 0.5 * (double(by(yp)) - double(by(ym)))
                          + 0.5 * (double(bz(zp)) - double(bz(zm)));
          g.divb += db * db;
        }
        // the condition being imposed, measured on the shell it acts on
        if (r > R - 1.0 && r < R + 1.0 && r > 1e-9) {
          const double bxv = double(bx(n)), byv = double(by(n)), bzv = double(bz(n));
          const double bm = std::sqrt(bxv * bxv + byv * byv + bzv * bzv);
          if (bm > 1e-30) {
            g.bn += std::abs((bxv * dx + byv * dy + bzv * dz) / r) / bm;
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
  if (nshell > 0) g.bn /= nshell;
  g.ratio = (g.eb > 0) ? g.eu / g.eb : 0.0;
  const double hnorm = std::sqrt(4.0 * g.eu * g.eb);
  g.hc = (hnorm > 1e-30) ? g.hc / hnorm : 0.0;
  g.cosjb = (jj > 0 && bb > 0) ? jb / std::sqrt(jj * bb) : 0.0;
  return g;
}

//------------------------------------------------------------------------------
template <class C>
static int run(const Opts& o) {
  const Index N = o.N;
  const double R = o.rfac * double(N);
  const double nu = o.u0 * 2.0 * R / o.Re;
  const double eta = nu / o.prm;
  const std::size_t NN = static_cast<std::size_t>(N) * N * N;

  Domain d(N, N, N, false, false, false);

  MagneticBGK<ML> mc;
  mc.omega = MagneticBGK<ML>::omega_from_resistivity(Real(eta));
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);

  C fc;
  fc.omega = C::omega_from_viscosity(Real(nu));
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, C> fl(d, fc);

  // ---- the initial condition, on the host --------------------------------
  Rng rng(o.seed);
  std::vector<double> u[3], b[3];
  solenoidal(N, o.kmax, o.k0, rng, u);
  solenoidal(N, o.kmax, o.k0, rng, b);

  const double cc = 0.5 * double(N - 1);
  auto inside = [&](std::size_t n) {
    const Index x = Index(n % N), y = Index((n / N) % N), z = Index(n / (N * N));
    const double dx = double(x) - cc, dy = double(y) - cc, dz = double(z) - cc;
    return dx * dx + dy * dy + dz * dz <= R * R;
  };
  // zero mean and r.m.s., both over the SPHERE -- the exterior is about to be
  // damped to rest and has no business setting the normalisation
  double nin = 0, su[3] = {0, 0, 0}, sb[3] = {0, 0, 0};
  for (std::size_t n = 0; n < NN; ++n)
    if (inside(n)) { nin += 1; for (int a = 0; a < 3; ++a) { su[a] += u[a][n]; sb[a] += b[a][n]; } }
  for (int a = 0; a < 3; ++a) { su[a] /= nin; sb[a] /= nin; }
  for (std::size_t n = 0; n < NN; ++n)
    for (int a = 0; a < 3; ++a) { u[a][n] -= su[a]; b[a][n] -= sb[a]; }

  // cross-helicity: Gram-Schmidt to zero, then rotate to the target cos(theta)
  double uu = 0, ub = 0;
  for (std::size_t n = 0; n < NN; ++n)
    if (inside(n))
      for (int a = 0; a < 3; ++a) { uu += u[a][n] * u[a][n]; ub += u[a][n] * b[a][n]; }
  const double proj = (uu > 0) ? ub / uu : 0.0;
  for (std::size_t n = 0; n < NN; ++n)
    for (int a = 0; a < 3; ++a) b[a][n] -= proj * u[a][n];

  double e0u = 0, e0b = 0;
  for (std::size_t n = 0; n < NN; ++n)
    if (inside(n))
      for (int a = 0; a < 3; ++a) { e0u += u[a][n] * u[a][n]; e0b += b[a][n] * b[a][n]; }
  // r.m.s. of the MAGNITUDE, not per component -- u0 is the velocity scale that
  // enters nu = u0 2R/Re and the Mach number, and dividing by 3 here would make
  // the flow sqrt(3) faster than asked for: Re 1.7x high and Ma 0.15 where the
  // tree's rule is 0.05. Matches demonstrator/mhd_decay.cpp:602.
  const double su0 = std::sqrt(e0u / nin), sb0 = std::sqrt(e0b / nin);
  const double gu = o.u0 / (su0 > 0 ? su0 : 1.0);
  const double gb = o.u0 / std::sqrt(o.alf) / (sb0 > 0 ? sb0 : 1.0);
  for (std::size_t n = 0; n < NN; ++n)
    for (int a = 0; a < 3; ++a) { u[a][n] *= gu; b[a][n] *= gb; }
  if (o.cost != 0.0) {
    // b <- ct (|b|/|u|) u + st b, with <u.b> = 0 going in. This keeps |b| and
    // therefore E_u/E_B exactly, and delivers cos(theta) = ct exactly:
    // |b|/|u| = 1/sqrt(alf) after the normalisation just above.
    const double ct = o.cost, st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
    const double s = 1.0 / std::sqrt(o.alf);
    for (std::size_t n = 0; n < NN; ++n)
      for (int a = 0; a < 3; ++a) b[a][n] = ct * s * u[a][n] + st * b[a][n];
  }

  View1D<Real> Ux("u0x", d.n_padded), Uy("u0y", d.n_padded), Uz("u0z", d.n_padded);
  View1D<Real> Bx0("b0x", d.n_padded), By0("b0y", d.n_padded), Bz0("b0z", d.n_padded);
  {
    auto hux = Kokkos::create_mirror_view(Ux); auto huy = Kokkos::create_mirror_view(Uy);
    auto huz = Kokkos::create_mirror_view(Uz); auto hbx = Kokkos::create_mirror_view(Bx0);
    auto hby = Kokkos::create_mirror_view(By0); auto hbz = Kokkos::create_mirror_view(Bz0);
    for (Index z = 0; z < N; ++z)
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const std::size_t m = (static_cast<std::size_t>(z) * N + y) * N + x;
          const Index n = d.id(x, y, z);
          hux(n) = Real(u[0][m]); huy(n) = Real(u[1][m]); huz(n) = Real(u[2][m]);
          hbx(n) = Real(b[0][m]); hby(n) = Real(b[1][m]); hbz(n) = Real(b[2][m]);
        }
    Kokkos::deep_copy(Ux, hux); Kokkos::deep_copy(Uy, huy); Kokkos::deep_copy(Uz, huz);
    Kokkos::deep_copy(Bx0, hbx); Kokkos::deep_copy(By0, hby); Kokkos::deep_copy(Bz0, hbz);
  }

  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    return FlowState{Real(1), Ux(n), Uy(n), Uz(n)};
  });
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Kokkos::Array<Real, 3> v; v[0] = Bx0(n); v[1] = By0(n); v[2] = Bz0(n); return v;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // ---- the penalisation fields -------------------------------------------
  // chi marks the EXTERIOR (1 outside the sphere), which is the opposite sense
  // to PenalisedBody's shapes -- those mark a body, and a vessel is the
  // complement of one. Written inline for that reason, as the disc does.
  View1D<Real> chi("chi", d.n_padded);
  View1D<Real> fx("fx", d.n_padded), fy("fy", d.n_padded), fz("fz", d.n_padded);
  View1D<Real> nx("nx", d.n_padded), ny("ny", d.n_padded), nz("nz", d.n_padded);
  View1D<Real> sx("sx", d.n_padded), sy("sy", d.n_padded), sz("sz", d.n_padded);
  {
    auto hc = Kokkos::create_mirror_view(chi);
    auto hnx = Kokkos::create_mirror_view(nx); auto hny = Kokkos::create_mirror_view(ny);
    auto hnz = Kokkos::create_mirror_view(nz);
    for (Index z = 0; z < N; ++z)
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          const Index n = d.id(x, y, z);
          const double dx = double(x) - cc, dy = double(y) - cc, dz = double(z) - cc;
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          hc(n) = Real(0.5 * (1.0 + std::tanh((r - R) / o.smooth)));
          const double ir = (r > 1e-9) ? 1.0 / r : 0.0;
          hnx(n) = Real(dx * ir); hny(n) = Real(dy * ir); hnz(n) = Real(dz * ir);
        }
    Kokkos::deep_copy(chi, hc);
    Kokkos::deep_copy(nx, hnx); Kokkos::deep_copy(ny, hny); Kokkos::deep_copy(nz, hnz);
  }
  fl.collision().forcing.Ex = fx;
  fl.collision().forcing.Ey = fy;
  fl.collision().forcing.Ez = fz;
  mag.set_source(sx, sy, sz);          // all three: NC = 3 on D3Q7, see banner

  const double Te = 2.0 * R / o.u0;
  std::printf("Confined decaying MHD in a SPHERE, volume-penalised   %s + %s\n",
              C::name, MagneticBGK<ML>::name);
  std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
  std::printf("  N = %d   R = %.1f (rfac %.2f)   Re = %.0f   Pm = %.2f   u0 = %.4f\n",
              int(N), R, o.rfac, o.Re, o.prm, o.u0);
  std::printf("  nu = %.6e   eta = %.6e   tau = %.6f   tau_mag = %.6f\n",
              nu, eta, 1.0 / double(fc.omega), 1.0 / double(mc.omega));
  std::printf("  penalisation eps = %.2f  eps_m = %.2f  smooth = %.2f;"
              "  IC kmax = %d  k0 = %.1f  seed %u\n", o.eps, o.epsm, o.smooth,
              o.kmax, o.k0, o.seed);
  std::printf("  one turnover 2R/u0 = %.0f steps; running %zu (%.1f turnovers)\n\n",
              Te, o.steps, double(o.steps) / Te);
  std::printf("   step   t/Te      E_u        E_b      E_u/E_b   H_c    enstr"
              "     <j^2>    cos(J,B)  |<b>|    max|b|   dv/cl    Bn/B   dmass\n");

  std::error_code ec;
  std::filesystem::create_directories("results/N_mhd_sphere", ec);
  std::FILE* mf = nullptr;
  if (o.dumpevery) {
    std::filesystem::create_directories("results/N_mhd_sphere/anim_frames", ec);
    mf = std::fopen("results/N_mhd_sphere/anim_frames/meta.txt", "w");
    if (mf) std::fprintf(mf, "N %d\nR %.6f\nTe %.6f\n", int(N), R, Te);
  }
  int frame = 0, vframe = 0;
  std::vector<std::pair<double, std::string>> pvd;
  if (o.vtievery) {
    std::filesystem::create_directories("results/N_mhd_sphere/vti", ec);
    // The container surface, once: the geometry does not move, and a per-frame
    // copy would be N identical files inviting someone to animate it.
    std::vector<float> spts; std::vector<std::int32_t> stris;
    const double c = 0.5 * double(N - 1);
    icosphere(3, c, c, c, R, spts, stris);
    write_vtp_triangles("results/N_mhd_sphere/vti/sphere_surface.vtp", spts, stris);
    std::printf("  wrote results/N_mhd_sphere/vti/sphere_surface.vtp"
                "  (%zu points, %zu triangles, r = %.3f)\n",
                spts.size() / 3, stris.size() / 3, R);
  }

  std::FILE* f = campaign::open_out("N_mhd_sphere",
                          "sphere_n" + std::to_string(int(N)) + "_re" +
                          std::to_string(int(o.Re)), "d3q27", o.op.c_str());
  if (f) std::fprintf(f, "# step t/Te E_u E_b ratio H_c enstrophy j2 cosJB bmean bmax divb BnB dmass\n");

  double mass0 = 0;
  for (std::size_t t = 0; t <= o.steps; ++t) {
    if (t % o.probe == 0 || t == o.steps) {
      const Diag g = measure(fl, mag, d, N, R);
      if (t == 0) mass0 = g.mass;
      const double dm = (mass0 != 0) ? (g.mass - mass0) / mass0 : 0.0;
      std::printf("  %6zu %6.3f %10.3e %10.3e %8.3f %7.4f %9.3e %9.3e %8.4f"
                  " %8.2e %8.2e %8.2e %7.4f %9.2e\n",
                  t, double(t) / Te, g.eu, g.eb, g.ratio, g.hc, g.ens, g.j2,
                  g.cosjb, g.bmean, g.bmax, g.divb, g.bn, dm);
      std::fflush(stdout);
      if (f) {
        std::fprintf(f, "%zu %.6f %.8e %.8e %.6f %.6f %.6e %.6e %.6f %.6e %.6e %.6e %.6f %.6e\n",
                     t, double(t) / Te, g.eu, g.eb, g.ratio, g.hc, g.ens, g.j2,
                     g.cosjb, g.bmean, g.bmax, g.divb, g.bn, dm);
        std::fflush(f);
      }
      if (o.dumpevery && t % o.dumpevery == 0) {
        dump(fl, mag, d, N, frame);
        if (mf) { std::fprintf(mf, "frame %d %.6f %.8e %.8e\n",
                               frame, double(t) / Te, g.eu, g.eb); std::fflush(mf); }
        ++frame;
      }
      if (o.vtievery && t % o.vtievery == 0)
        pvd.emplace_back(double(t) / Te,
                         write_frame_vti(fl, mag, chi, d, N, vframe++));
      if (!g.finite) { std::printf("\n  *** NON-FINITE at step %zu ***\n", t); break; }
    }
    if (t < o.steps) {
      // F = -chi u / eps and S = -chi (B.n) n / eps_m are EXPLICIT, so they must
      // see u(t) and B(t) rather than the previous step's.
      fl.compute_macroscopic();
      mag.compute_field();
      auto ux = fl.ux(); auto uy = fl.uy(); auto uz = fl.uz();
      auto bx = mag.Bx(); auto by = mag.By(); auto bz = mag.Bz();
      auto ch = chi;
      auto ax = fx; auto ay = fy; auto az = fz;
      auto mx = nx; auto my = ny; auto mz = nz;
      auto qx = sx; auto qy = sy; auto qz = sz;
      const Real ie = Real(1.0 / o.eps), iem = Real(1.0 / o.epsm);
      Kokkos::parallel_for("pen", Kokkos::RangePolicy<ExecSpace>(0, d.n_padded),
        KOKKOS_LAMBDA(Index n) {
          const Real c = ch(n);
          ax(n) = -c * ux(n) * ie;
          ay(n) = -c * uy(n) * ie;
          az(n) = -c * uz(n) * ie;
          const Real bn = bx(n) * mx(n) + by(n) * my(n) + bz(n) * mz(n);
          qx(n) = -c * bn * mx(n) * iem;
          qy(n) = -c * bn * my(n) * iem;
          qz(n) = -c * bn * mz(n) * iem;
        });
      mag.compute_field();
      fl.step(true);
      mag.step(true);
    }
  }
  if (f) std::fclose(f);
  if (mf) { std::fprintf(mf, "frames %d\n", frame); std::fclose(mf); }
  if (!pvd.empty()) {
    write_pvd("results/N_mhd_sphere/vti/sphere.pvd", pvd);
    std::printf("\n  ParaView: open results/N_mhd_sphere/vti/sphere.pvd"
                "  (%zu frames, t/Te 0 to %.2f)\n"
                "  Threshold on chi < 0.5 to clip to the sphere; colour by Jmag.\n",
                pvd.size(), pvd.back().first);
  }
  return 0;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Opts o;
    for (int i = 1; i < argc; ++i) {
      auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-n"))      { if (i + 1 < argc) o.N = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-rfac"))   next(o.rfac);
      else if (!std::strcmp(argv[i], "-u"))      next(o.u0);
      else if (!std::strcmp(argv[i], "-re"))     next(o.Re);
      else if (!std::strcmp(argv[i], "-prm"))    next(o.prm);
      else if (!std::strcmp(argv[i], "-alf"))    next(o.alf);
      else if (!std::strcmp(argv[i], "-cost"))   next(o.cost);
      else if (!std::strcmp(argv[i], "-eps"))    next(o.eps);
      else if (!std::strcmp(argv[i], "-epsm"))   next(o.epsm);
      else if (!std::strcmp(argv[i], "-smooth")) next(o.smooth);
      else if (!std::strcmp(argv[i], "-k0"))     next(o.k0);
      else if (!std::strcmp(argv[i], "-kmax"))   { if (i + 1 < argc) o.kmax = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-seed"))   { if (i + 1 < argc) o.seed = unsigned(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-steps"))  { if (i + 1 < argc) o.steps = std::size_t(std::atoll(argv[++i])); }
      else if (!std::strcmp(argv[i], "-probe"))  { if (i + 1 < argc) o.probe = std::size_t(std::atoll(argv[++i])); }
      else if (!std::strcmp(argv[i], "-dump"))   { if (i + 1 < argc) o.dumpevery = std::size_t(std::atoll(argv[++i])); }
      else if (!std::strcmp(argv[i], "-vti"))    { if (i + 1 < argc) o.vtievery = std::size_t(std::atoll(argv[++i])); }
      else if (!std::strcmp(argv[i], "-op"))     { if (i + 1 < argc) o.op = argv[++i]; }
    }
    // The dump lives inside the probe block because meta.txt carries that
    // probe's energies, so the two cadences must coincide or frames go missing
    // silently. Snapped rather than documented, because "-dump must divide
    // -probe" is exactly the kind of rule nobody reads.
    if (o.dumpevery) o.probe = o.dumpevery;
    if (o.vtievery && !o.dumpevery) o.probe = o.vtievery;
    if (o.op == "bgk") rc = run<CollB>(o);
    else               rc = run<CollS>(o);
  }
  Kokkos::finalize();
  return rc;
}
