//==============================================================================
//  Taylor-Green MHD in a CLOSED box -- the free-slip / no-slip / hydrodynamic
//  trio of the confined-MHD plan, with the diagnostics and the output it needs.
//
//  THE PROBLEM. Pouquet, Lee, Brachet, Mininni & Rosenberg (GAFD 104, 115,
//  2010) decay Taylor-Green MHD in a periodic 2pi box whose symmetries make the
//  fundamental box [0, pi]^3 a closed box with impermeable, FREE-SLIP walls:
//
//     v   = (sin x cos y cos z, -cos x sin y cos z, 0)                    (3)
//     B_C = b0 (sin2x cos2y cos2z, cos2x sin2y cos2z, -2 cos2x cos2y sin2z)
//     B_I = b0 (cos x sin y sin z, sin x cos y sin z, -2 sin x sin y cos z) (5-7)
//     B_A = b0 (cos2x sin2y sin2z, -sin2x cos2y sin2z, 0)
//
//  B_C has the CONDUCTING parity on the faces of [0, pi]^3 (B.n = 0) and B_I,
//  B_A the pseudo-vacuum one (B x n = 0) -- the paper's "conducting" and
//  "insulating" boxes. v0 and b0 give E_V = E_M = 0.125 at t = 0, so b0 = 1/sqrt3
//  for B_C and B_I and 1 for B_A. This file solves that box DIRECTLY, with
//  on-node walls, and then does the one thing a periodic code cannot: makes the
//  walls no-slip.
//
//  THE RUNS, one flag apart (every other parameter identical):
//
//     A   -vwall slip   -mwall cond   the control. SpecNode + the conducting
//                                     parity wall make it an EXACT restriction
//                                     of the periodic run, transient included
//                                     (validation/mhd_parity_wall, 2e-14).
//     B   -vwall noslip -mwall cond   the experiment: RegWall on all six faces.
//     C   -vwall noslip -ic hydro     the same box with no field at all.
//     B'  -mwall pv -ic tgi           the insulating-parity variant of B.
//     P   -geom periodic              the 2pi box itself, for the identity and
//                                     for comparing against the paper directly.
//
//  UNITS. Everything printed is in the paper's units: v0 = 1, the length unit is
//  1/h cells with h = pi/(N-1), and nu = eta = 1/Re, so Re = 1000 IS their C2
//  (nu = 1e-3). The lattice velocity of v0 is -u0 (default 0.05, Ma = 0.087),
//  hence one step is dt = u0 h of their time, and
//
//     nu_lat = u0 / (h Re),   eta_lat = nu_lat / Pm.
//
//  The box has N nodes a side with walls ON nodes 0 and N-1; the periodic twin
//  has 2(N-1). Either way the node spacing is the same h.
//
//  THE REFERENCE NUMBERS this is compared against at Re = 1000 (their C2,
//  Table 1): min E_M/E_V = 0.35 (the conducting flow's ratio falls below one
//  after a transient and never returns) and the first maximum of
//  Omega_M/Omega_V after its initial fall, printed as "2." -- one figure. Their
//  run is spectral at 128^3. Their SKEWNESS is not a reference: the paper calls
//  it "the normalized third-order moment of the velocity field" without a
//  formula, and its Fig. 10 reaches ~2 by t ~ 0.5, which no velocity-DERIVATIVE
//  skewness of this initial condition can do. The column here is the
//  longitudinal derivative skewness, direction-averaged, sign making turbulence
//  positive -- a diagnostic of this code, not a comparison with theirs.
//
//  DIAGNOSTICS (series.dat, one row per probe). Averages use trapezoidal weights
//  in the box -- half on a face, a quarter on an edge, an eighth at a corner --
//  because that makes a box average of a mirror-symmetric field EQUAL the
//  periodic average, not merely close to it. Derivatives at a wall node use the
//  extension the wall itself implies: the mirror for a free-slip velocity and
//  for the magnetic parity (so the free-slip box reproduces the periodic
//  diagnostics too), and a one-sided second-order stencil for a no-slip
//  velocity, whose natural extension is not a mirror. Columns: energies, cross
//  helicity, enstrophies, total dissipation eps = 2 nu Omega_V + 2 eta Omega_M,
//  the two ratios the reference quotes, j_max and w_max, the skewness, div b
//  (global and in the wall layer, both over rms |j|), the mass drift, the peak
//  lattice speed, the share of dissipation within 1, 2 and 4 x Re^-1/2 of the
//  walls, and the mean wall shear stress. That share is read off the
//  wall-distance profile with the band edge interpolated (within_cells, and why
//  a count of whole node layers is wrong).
//
//  OUTPUT, all under -out (default results/P_tg_mhd/<tag>/):
//     series.dat              the time series; tools/plot_tg_mhd.py plots it
//     profile.dat             the dissipation against wall distance, layer by
//                             layer, one row per probe
//     anim_frames/            mid-plane slices (umag, bmag, jmag, wmag) and,
//                             with -dumpvol, the |J| volume -- the format
//                             results/N_mhd_sphere/render_slices.py and
//                             render_volume.py read; meta.txt says R 0 (box)
//     vti/tg_NNNN.vti, .pvd   ParaView: u, b, omega, J, |J|, |omega|, rho, on
//                             the real [0, pi] coordinates
//     raw/fields_NNNN.raw     full-resolution float32 volumes of u, b, rho, for
//                             analysis a picture cannot do (spectra, sheet
//                             thickness, helicity); layout in write_raw_fields
//
//  WHAT THIS IS NOT. A spectral-quality run: at a given N the LB box resolves
//  less than the paper's grid, and the free-slip box against the paper is a
//  comparison at their printed precision. The no-slip walls have no reference
//  at all -- that is the point of them. RegWall's 3-D corners fall back to
//  rho = 1 and the box loses a few 1e-4 of its mass under a field in the first
//  hundreds of steps; series.dat carries the drift so it is never hidden.
//
//    usage: tg_mhd [-n N] [-geom box|periodic] [-vwall noslip|slip]
//                  [-mwall cond|pv] [-ic tgc|tgi|tga|hydro] [-re R] [-pm P]
//                  [-u0 U] [-tmax T] [-probe dt] [-vti dt] [-dump dt]
//                  [-dumpvol] [-volstride S] [-raw dt] [-out DIR] [-force]
//==============================================================================
#include "Campaign.hpp"
#include "boundary/Specular.hpp"
#include "collision/MagneticBGK.hpp"
#include "collision/MhdCentralMomentsShifted.hpp"
#include "io/VtiWriter.hpp"
#include "solver/MagneticSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace lbm;

using FL = D3Q27;
using ML = D3Q7;
// The SHIFTED-basis operator (phi_2 = C^2 - cs^2), the tree's default MHD
// central moments since 91c9404 and the one GPU/ implements. The monomial
// MhdCentralMoments shares its equilibrium but relaxes the non-equilibrium
// moments differently, and the twin drivers first disagreed by exactly that:
// identical to 5.7e-14 after one step, 7.7e-5 apart after two, the gap shaped
// like u itself (measured 2026-09-25, periodic TG, N = 17).
using FC = MhdCentralMomentsShifted<FL>;
using MC = MagneticBGK<ML>;
using FluidT = FluidSolver<FL, EsotericPull<FL>, FC>;
using MagT   = MagneticSolver<ML, EsotericPull<ML>, MC>;

namespace {

constexpr double PI = 3.14159265358979323846;

enum class ICKind { TGC, TGI, TGA, Hydro };

struct Opts {
  Index N = 65;
  bool periodic = false, noslip = true, conducting = true, force = false;
  ICKind ic = ICKind::TGC;
  double re = 500.0, pm = 1.0, u0 = 0.05, tmax = 10.0;
  double probe = 0.05, vti = 0.0, dump = 0.0, raw = 0.0;
  bool dumpvol = false;
  int volstride = 1;
  int threads = 0;              // host diagnostics; 0 = SLURM allocation or machine
  std::string out;
};

// The initial state at interior node (x, y, z). Device-callable: it seeds the
// populations and the field on whatever backend the solvers run.
// A NO-SLIP box's wall nodes are seeded AT REST: the wall's value is part of
// the wall condition, so run B differs from run A only by the flag, and GPU/'s
// twin seeds its RegWall nodes the same way. The interior field is identical.
struct Init {
  ICKind ic;
  double h, u0, b0;   // b0 in LATTICE units
  bool rest_walls = false;
  Index nb = 0;       // box nodes a side, for the wall test
  KOKKOS_INLINE_FUNCTION void at(Index x, Index y, Index z, double u[3], double b[3]) const {
    const double X = h * double(x), Y = h * double(y), Z = h * double(z);
    u[0] =  u0 * Kokkos::sin(X) * Kokkos::cos(Y) * Kokkos::cos(Z);
    u[1] = -u0 * Kokkos::cos(X) * Kokkos::sin(Y) * Kokkos::cos(Z);
    u[2] = 0.0;
    if (rest_walls && (x == 0 || y == 0 || z == 0 || x == nb - 1 || y == nb - 1 || z == nb - 1))
      u[0] = u[1] = 0.0;
    b[0] = b[1] = b[2] = 0.0;
    if (ic == ICKind::TGC) {
      b[0] =  b0 * Kokkos::sin(2 * X) * Kokkos::cos(2 * Y) * Kokkos::cos(2 * Z);
      b[1] =  b0 * Kokkos::cos(2 * X) * Kokkos::sin(2 * Y) * Kokkos::cos(2 * Z);
      b[2] = -2 * b0 * Kokkos::cos(2 * X) * Kokkos::cos(2 * Y) * Kokkos::sin(2 * Z);
    } else if (ic == ICKind::TGI) {
      b[0] =  b0 * Kokkos::cos(X) * Kokkos::sin(Y) * Kokkos::sin(Z);
      b[1] =  b0 * Kokkos::sin(X) * Kokkos::cos(Y) * Kokkos::sin(Z);
      b[2] = -2 * b0 * Kokkos::sin(X) * Kokkos::sin(Y) * Kokkos::cos(Z);
    } else if (ic == ICKind::TGA) {
      b[0] =  b0 * Kokkos::cos(2 * X) * Kokkos::sin(2 * Y) * Kokkos::sin(2 * Z);
      b[1] = -b0 * Kokkos::sin(2 * X) * Kokkos::cos(2 * Y) * Kokkos::sin(2 * Z);
    }
  }
};

std::uint8_t box_faces(Index x, Index y, Index z, Index N) {
  std::uint8_t m = SpecNone;
  if (x == 0) m |= SpecXm; else if (x == N - 1) m |= SpecXp;
  if (y == 0) m |= SpecYm; else if (y == N - 1) m |= SpecYp;
  if (z == 0) m |= SpecZm; else if (z == N - 1) m |= SpecZp;
  return m;
}

std::uint8_t box_normal(Index x, Index y, Index z, Index N) {
  int nf = 0;
  std::uint8_t c = NrmNone;
  if (x == 0) { ++nf; c = NrmXm; } else if (x == N - 1) { ++nf; c = NrmXp; }
  if (y == 0) { ++nf; c = NrmYm; } else if (y == N - 1) { ++nf; c = NrmYp; }
  if (z == 0) { ++nf; c = NrmZm; } else if (z == N - 1) { ++nf; c = NrmZp; }
  return nf >= 2 ? std::uint8_t(NrmCorner) : c;
}

//------------------------------------------------------------------------------
//  Host view of the seven fields, in the paper's units, with the derivative
//  rule each wall implies. Built once per probe; on a host backend the mirrors
//  are the device views themselves, so this costs no copy.
//------------------------------------------------------------------------------
enum Ext : int { ExtPeriodic, ExtEven, ExtOdd, ExtOneSided };

using HostView = decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, View1D<Real>()));

struct HostFields {
  const Domain* dom = nullptr;
  Index L = 0;                 // nodes a side of the domain
  bool periodic = false, noslip = true, conducting = true;
  double su = 1.0;             // lattice -> paper units for u and b (1/u0)
  HostView f[7];               // ux uy uz bx by bz rho

  double v(int c, Index x, Index y, Index z) const {
    if (periodic) { x = (x + L) % L; y = (y + L) % L; z = (z + L) % L; }
    return double(f[c](dom->id(x, y, z))) * (c < 6 ? su : 1.0);
  }
  int ext(int c, int k) const {
    if (periodic) return ExtPeriodic;
    if (c < 3) return noslip ? ExtOneSided : (c == k ? ExtOdd : ExtEven);
    if (c < 6) {
      const bool normal = (c - 3 == k);
      return (normal == conducting) ? ExtOdd : ExtEven;
    }
    return noslip ? ExtOneSided : ExtEven;
  }
  // d f_c / d x_k at a node, per CELL (multiply by 1/h for the paper's units).
  double d(int c, int k, Index x, Index y, Index z) const {
    const Index p[3] = {x, y, z};
    auto at = [&](Index off) {
      Index q[3] = {p[0], p[1], p[2]};
      q[k] += off;
      return v(c, q[0], q[1], q[2]);
    };
    if (periodic || (p[k] > 0 && p[k] < L - 1)) return 0.5 * (at(1) - at(-1));
    const int e = ext(c, k);
    const double s = (p[k] == 0) ? 1.0 : -1.0;          // step INTO the box
    if (e == ExtOneSided)
      return s * (-1.5 * at(0) + 2.0 * at(Index(s)) - 0.5 * at(Index(2 * s)));
    // Mirror ghost: f(-1) = +/- f(+1) about the wall node.
    const double in = at(Index(s)), ghost = (e == ExtEven) ? in : -in;
    return s * 0.5 * (in - ghost);
  }
};

// THE PROBE IS PARALLEL OVER z -- the same split as GPU/src/tg_mhd.cu, whose
// banner at Acc says why. Each thread sums its own z-slab and the slabs merge in
// thread order, so the per-node arithmetic is untouched and only the rounding of
// the sums depends on the thread count. std::thread, not Kokkos: these are host
// mirrors walked with lambdas that do their own index arithmetic.
struct Acc {
  double W = 0, ev = 0, em = 0, hc = 0, w2 = 0, j2 = 0, d2[3] = {0, 0, 0}, d3[3] = {0, 0, 0};
  double div2 = 0, eps = 0, divw = 0, ndivw = 0, j2w = 0, tw = 0, ntw = 0;
  double mass = 0, jmax = 0, wmax = 0, umax = 0;
  bool finite = true;
  // Per wall-distance layer k -- the nodes k cells from the nearest wall -- the
  // weighted dissipation and the weight itself; f_w and profile.dat come from
  // these (within_cells says how).
  std::vector<double> lay_e, lay_w;
  explicit Acc(std::size_t nlay = 0) : lay_e(nlay, 0.0), lay_w(nlay, 0.0) {}
  void add(const Acc& a) {
    W += a.W; ev += a.ev; em += a.em; hc += a.hc; w2 += a.w2; j2 += a.j2;
    for (int k = 0; k < 3; ++k) { d2[k] += a.d2[k]; d3[k] += a.d3[k]; }
    div2 += a.div2; eps += a.eps; divw += a.divw; ndivw += a.ndivw; j2w += a.j2w;
    tw += a.tw; ntw += a.ntw; mass += a.mass;
    jmax = std::max(jmax, a.jmax); wmax = std::max(wmax, a.wmax); umax = std::max(umax, a.umax);
    finite = finite && a.finite;
    for (std::size_t k = 0; k < lay_e.size(); ++k) { lay_e[k] += a.lay_e[k]; lay_w[k] += a.lay_w[k]; }
  }
};

int diag_threads(int requested) {
  if (requested > 0) return requested;
  if (const char* s = std::getenv("SLURM_CPUS_PER_TASK")) {
    const int n = std::atoi(s);
    if (n > 0) return n;
  }
  const unsigned hc = std::thread::hardware_concurrency();
  return int(std::min(16u, std::max(1u, hc)));
}

template <class Fn>
void parallel_z(Index L, int nt, Fn fn) {
  nt = std::max(1, std::min<int>(nt, int(L)));
  if (nt == 1) { fn(Index(0), L, 0); return; }
  std::vector<std::thread> th;
  th.reserve(std::size_t(nt));
  for (int t = 0; t < nt; ++t)
    th.emplace_back(fn, Index((L * t) / nt), Index((L * (t + 1)) / nt), t);
  for (auto& x : th) x.join();
}

// THE WALL SHARE f_w IS READ OFF THE WALL-DISTANCE PROFILE, NOT COUNTED IN WHOLE
// NODE LAYERS. With the trapezoidal weights, layer k is EXACTLY the slab
// [(k - 1/2) h, (k + 1/2) h] of the box in volume -- the wall layer is [0, h/2] --
// so the dissipation within a distance d of the walls is known at d = (k + 1/2) h
// and is interpolated linearly in between: a band edge inside a layer takes the
// fraction of that layer it covers. The count this replaced, every node with
// k h < m delta, measured a band of (floor(m delta / h) + 1/2) h instead of
// m delta -- 0.91 delta at N = 384 and 1.07 delta at N = 512 for Re = 1000 -- and
// in the first CSF3 ladder (2026-09-28) fw1 differed by 8-25 % between the two
// grids of a rung while the peak dissipation agreed to 1 %: the gate was
// measuring where the band edge fell between nodes, not the flow. r is the
// distance in cells; e[k] the layer sums. The same function as GPU/src/tg_mhd.cu.
double within_cells(const std::vector<double>& e, double r) {
  if (r <= 0.0 || e.empty()) return 0.0;
  if (r < 0.5) return e[0] * (r / 0.5);
  const std::size_t k = std::size_t(std::floor(r + 0.5));   // r in [k - 1/2, k + 1/2)
  double c = 0.0;
  for (std::size_t j = 0; j < std::min(k, e.size()); ++j) c += e[j];
  return k < e.size() ? c + e[k] * (r - (double(k) - 0.5)) : c;
}

struct Diag {
  double t = 0, ev = 0, em = 0, hc = 0, omv = 0, omm = 0, eps = 0;
  double jmax = 0, wmax = 0, skew = 0, divb = 0, divb_wall = 0, mass = 0;
  double umax_lat = 0, fw[3] = {0, 0, 0}, tauw = 0;
  std::vector<double> prof, vol;   // each layer's share of eps and of the volume
  bool finite = true;
};

//------------------------------------------------------------------------------
//  Output writers
//------------------------------------------------------------------------------
void write_raw2(const std::string& path, int nx, int ny, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
}

void write_raw3(const std::string& path, int nx, int ny, int nz, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny, c = nz;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(&c), sizeof c);
  o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
}

// raw/fields_NNNN.raw -- the full state for offline analysis.
//   char[8]  "TGMHDRAW"
//   int32    nx, ny, nz, nvar (= 7)
//   float64  t (paper units), h (cell size, paper units)
//   nvar blocks of nx*ny*nz float32, x fastest: ux uy uz bx by bz rho
// u and b are in the paper's units; rho is the lattice density (mean 1).
void write_raw_fields(const std::string& path, const HostFields& H, double t, double h) {
  std::ofstream o(path, std::ios::binary);
  o.write("TGMHDRAW", 8);
  const std::int32_t n = std::int32_t(H.L), nv = 7;
  for (int k = 0; k < 3; ++k) o.write(reinterpret_cast<const char*>(&n), sizeof n);
  o.write(reinterpret_cast<const char*>(&nv), sizeof nv);
  o.write(reinterpret_cast<const char*>(&t), sizeof t);
  o.write(reinterpret_cast<const char*>(&h), sizeof h);
  std::vector<float> blk(std::size_t(H.L) * H.L * H.L);
  for (int c = 0; c < 7; ++c) {
    std::size_t m = 0;
    for (Index z = 0; z < H.L; ++z)
      for (Index y = 0; y < H.L; ++y)
        for (Index x = 0; x < H.L; ++x) blk[m++] = float(H.v(c, x, y, z));
    o.write(reinterpret_cast<const char*>(blk.data()),
            std::streamsize(blk.size() * sizeof(float)));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Opts o;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
      if      (a == "-n")        o.N = std::atoi(next());
      else if (a == "-geom")     o.periodic = std::string(next()) == "periodic";
      else if (a == "-vwall")    o.noslip = std::string(next()) != "slip";
      else if (a == "-mwall")    o.conducting = std::string(next()) != "pv";
      else if (a == "-ic") {
        const std::string s = next();
        o.ic = s == "tgi" ? ICKind::TGI : s == "tga" ? ICKind::TGA
             : s == "hydro" ? ICKind::Hydro : ICKind::TGC;
      }
      else if (a == "-re")       o.re = std::atof(next());
      else if (a == "-pm")       o.pm = std::atof(next());
      else if (a == "-u0")       o.u0 = std::atof(next());
      else if (a == "-tmax")     o.tmax = std::atof(next());
      else if (a == "-probe")    o.probe = std::atof(next());
      else if (a == "-vti")      o.vti = std::atof(next());
      else if (a == "-dump")     o.dump = std::atof(next());
      else if (a == "-dumpvol")  o.dumpvol = true;
      else if (a == "-volstride") o.volstride = std::max(1, std::atoi(next()));
      else if (a == "-raw")      o.raw = std::atof(next());
      else if (a == "-out")      o.out = next();
      else if (a == "-force")    o.force = true;
      else if (a == "-threads")  o.threads = std::atoi(next());
    }

    const bool hydro = (o.ic == ICKind::Hydro);
    // A field that violates the magnetic wall is not an initial condition for
    // that box. Refused rather than warned about: the wall would impose its
    // parity in the first step and the run would start from a different field.
    const bool ic_conducting = (o.ic == ICKind::TGC);
    if (!hydro && !o.periodic && ic_conducting != o.conducting && !o.force) {
      std::fprintf(stderr, "tg_mhd: -ic %s has the %s parity but -mwall is %s; the "
                   "wall would rewrite the field in the first step. Use -mwall %s, "
                   "or -force to do it anyway.\n",
                   ic_conducting ? "tgc" : "tgi/tga", ic_conducting ? "conducting" : "pseudo-vacuum",
                   o.conducting ? "cond" : "pv", ic_conducting ? "cond" : "pv");
      Kokkos::finalize();
      return 2;
    }

    const Index N = o.N;
    const Index L = o.periodic ? 2 * (N - 1) : N;
    const double h = PI / double(N - 1);
    const double nu_lat = o.u0 / (h * o.re), eta_lat = nu_lat / o.pm;
    const double dt = o.u0 * h;                            // paper time per step
    const std::size_t T = std::size_t(std::llround(o.tmax / dt));
    const double b0_tg = hydro ? 0.0 : (o.ic == ICKind::TGA ? 1.0 : 1.0 / std::sqrt(3.0));
    const double tauf = 3.0 * nu_lat + 0.5, taum = 4.0 * eta_lat + 0.5;

    const char* icn = o.ic == ICKind::TGC ? "tgc" : o.ic == ICKind::TGI ? "tgi"
                    : o.ic == ICKind::TGA ? "tga" : "hydro";
    char tag[160];
    if (o.periodic)
      std::snprintf(tag, sizeof tag, "%s_periodic_n%lld_re%g", icn, (long long)N, o.re);
    else
      std::snprintf(tag, sizeof tag, "%s_%s_%s_n%lld_re%g", icn, o.noslip ? "noslip" : "slip",
                    hydro ? "nofield" : (o.conducting ? "cond" : "pv"), (long long)N, o.re);
    if (o.out.empty()) o.out = std::string("results/P_tg_mhd/") + tag;
    std::error_code ec;
    std::filesystem::create_directories(o.out, ec);
    if (ec) {
      std::fprintf(stderr, "tg_mhd: cannot create %s: %s\n", o.out.c_str(), ec.message().c_str());
      Kokkos::finalize();
      return 2;
    }

    std::printf("Taylor-Green MHD, %s   D3Q27 fluid (central moments) + D3Q7 field\n",
                o.periodic ? "PERIODIC 2pi box" : "CLOSED box [0,pi]^3, on-node walls");
    if (!o.periodic)
      std::printf("  walls: velocity %s, field %s\n",
                  o.noslip ? "NO-SLIP (RegWall)" : "free-slip (SpecNode)",
                  hydro ? "none (B = 0)" : (o.conducting ? "CONDUCTING parity" : "pseudo-vacuum parity"));
    std::printf("  ic %s   N = %lld (%lld^3 nodes)   Re = %g   Pm = %g   u0 = %g (Ma %.3f)\n",
                icn, (long long)N, (long long)L, o.re, o.pm, o.u0, o.u0 * std::sqrt(3.0));
    std::printf("  nu_lat = %.4e  tau_f = %.5f   eta_lat = %.4e  tau_m = %.5f\n",
                nu_lat, tauf, eta_lat, taum);
    std::printf("  dt = %.4e  ->  %zu steps to t = %g   ->  %s\n", dt, T, o.tmax, o.out.c_str());
    if (tauf - 0.5 < 0.002)
      std::printf("  NOTE: tau_f - 1/2 = %.2e is below the 2e-3 every established MHD run in "
                  "this tree sat at.\n", tauf - 0.5);
    if (o.noslip && !o.periodic && tauf - 0.5 < 0.012)
      std::printf("  NOTE: tau_f - 1/2 = %.2e is at RegWall's measured floor (0.006-0.012 at "
                  "u0 = 0.02-0.04, rising with Mach). Expect it to diverge at the wall.\n",
                  tauf - 0.5);

    Domain d(L, L, L, o.periodic, o.periodic, o.periodic);
    MC mc;
    mc.omega = MC::omega_from_resistivity(Real(eta_lat));
    MagT mag(d, mc);
    FC fc;
    fc.omega = FC::omega_from_viscosity(Real(nu_lat));
    fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
    FluidT fl(d, fc);
    fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
    if (!o.periodic) {
      if (o.noslip) {
        // FD edge closure is incompatible with a Lorentz force (FluidSolver).
        fl.set_fd_corners(false);
        fl.set_regularized_walls([&](Index x, Index y, Index z) {
          FluidT::WallSpec w;
          w.normal = box_normal(x, y, z, N);
          return w;
        });
      } else {
        fl.set_specular_nodes([&](Index x, Index y, Index z) -> std::uint8_t {
          return box_faces(x, y, z, N);
        });
      }
      if (!hydro)
        mag.set_parity_walls([&](Index x, Index y, Index z) -> std::uint8_t {
                               return box_faces(x, y, z, N);
                             },
                             o.conducting ? MagParity::Conducting : MagParity::PseudoVacuum);
    }

    const Init init{o.ic, h, o.u0, o.u0 * b0_tg, o.noslip && !o.periodic, N};
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      double u[3], b[3];
      init.at(px - d.hx, py - d.hy, pz - d.hz, u, b);
      return FlowState{Real(1), Real(u[0]), Real(u[1]), Real(u[2])};
    });
    // The MHD equilibrium, Maxwell stress included -- see mhd_parity_wall.cpp
    // for why seed_value alone is not enough and what this does and does not fix.
    static_assert(std::is_same_v<FC::Storage, RawPopulations>, "the seed writes raw populations");
    fl.seed_populations(KOKKOS_LAMBDA(Index n, int i) {
      Index px, py, pz; d.coords(n, px, py, pz);
      double u[3], b[3];
      init.at(px - d.hx, py - d.hy, pz - d.hz, u, b);
      const Real ur[3] = {Real(u[0]), Real(u[1]), Real(u[2])};
      const Real br[3] = {Real(b[0]), Real(b[1]), Real(b[2])};
      Real fe[FL::Q];
      fc.equilibrium(fe, Real(1), ur, br);
      return fe[i];
    });
    // The field at its equilibrium WITH the initial velocity, as GPU/ seeds it.
    mag.initialize_field(
        KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; d.coords(n, px, py, pz);
          double u[3], b[3];
          init.at(px - d.hx, py - d.hy, pz - d.hz, u, b);
          Kokkos::Array<Real, 3> bb;
          bb[0] = Real(b[0]); bb[1] = Real(b[1]); bb[2] = Real(b[2]);
          return bb;
        },
        KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; d.coords(n, px, py, pz);
          double u[3], b[3];
          init.at(px - d.hx, py - d.hy, pz - d.hz, u, b);
          Kokkos::Array<Real, 3> uu;
          uu[0] = Real(u[0]); uu[1] = Real(u[1]); uu[2] = Real(u[2]);
          return uu;
        });
    mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

    // ---- host fields and the probe ------------------------------------------
    HostFields H;
    H.dom = &d; H.L = L; H.periodic = o.periodic; H.noslip = o.noslip;
    H.conducting = o.conducting; H.su = 1.0 / o.u0;
    auto refresh = [&]() {
      fl.compute_macroscopic();
      if (!hydro) mag.compute_field();
      H.f[0] = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      H.f[1] = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      H.f[2] = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
      H.f[3] = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
      H.f[4] = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
      H.f[5] = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());
      H.f[6] = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
    };

    const double nu_tg = 1.0 / o.re, eta_tg = nu_tg / o.pm, ih = 1.0 / h;
    const double delta = 1.0 / std::sqrt(o.re);           // Re^-1/2 layer, paper units
    // The initial mass from the seeded state (rho = 1 everywhere), not from a
    // macroscopic pass: before the first step RegWall's edge and corner nodes
    // are not yet evaluated and read zero.
    const double mass0 = double(L) * double(L) * double(L);

    const int nthreads = diag_threads(o.threads);
    auto measure_slab = [&](Index z0, Index z1, Acc& a) {
      for (Index z = z0; z < z1; ++z)
        for (Index y = 0; y < L; ++y)
          for (Index x = 0; x < L; ++x) {
            const Index p[3] = {x, y, z};
            double w = 1.0;
            Index dw = L;                                  // cells to the nearest wall
            if (!o.periodic)
              for (int k = 0; k < 3; ++k) {
                if (p[k] == 0 || p[k] == L - 1) w *= 0.5;
                dw = std::min<Index>(dw, std::min<Index>(p[k], L - 1 - p[k]));
              }
            else
              for (int k = 0; k < 3; ++k) {
                // distance to the nearest symmetry plane of the fundamental box
                const Index m = p[k] % (N - 1);
                dw = std::min<Index>(dw, std::min<Index>(m, N - 1 - m));
              }
            const double u[3] = {H.v(0, x, y, z), H.v(1, x, y, z), H.v(2, x, y, z)};
            const double b[3] = {H.v(3, x, y, z), H.v(4, x, y, z), H.v(5, x, y, z)};
            const double rho = H.v(6, x, y, z);
            for (int c = 0; c < 3; ++c)
              if (!std::isfinite(u[c]) || !std::isfinite(b[c])) a.finite = false;
            a.mass += rho;
            a.umax = std::max(a.umax, o.u0 * std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]));
            // gradients, paper units
            double du[3][3], db[3][3];
            for (int c = 0; c < 3; ++c)
              for (int k = 0; k < 3; ++k) {
                du[c][k] = H.d(c, k, x, y, z) * ih;
                db[c][k] = hydro ? 0.0 : H.d(3 + c, k, x, y, z) * ih;
              }
            const double wv[3] = {du[2][1] - du[1][2], du[0][2] - du[2][0], du[1][0] - du[0][1]};
            const double jv[3] = {db[2][1] - db[1][2], db[0][2] - db[2][0], db[1][0] - db[0][1]};
            const double w2 = wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2];
            const double j2 = jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2];
            const double dvb = db[0][0] + db[1][1] + db[2][2];
            const double eloc = nu_tg * w2 + eta_tg * j2;
            a.W += w;
            a.ev += w * 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            a.em += w * 0.5 * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
            a.hc += w * (u[0] * b[0] + u[1] * b[1] + u[2] * b[2]);
            a.w2 += w * w2;  a.j2 += w * j2;  a.div2 += w * dvb * dvb;  a.eps += w * eloc;
            for (int k = 0; k < 3; ++k) {
              a.d2[k] += w * du[k][k] * du[k][k];
              a.d3[k] += w * du[k][k] * du[k][k] * du[k][k];
            }
            const std::size_t lay = std::min(std::size_t(dw), a.lay_e.size() - 1);
            a.lay_e[lay] += w * eloc;
            a.lay_w[lay] += w;
            if (dw <= 1) { a.divw += w * dvb * dvb; a.j2w += w * j2; a.ndivw += w; }
            a.jmax = std::max(a.jmax, std::sqrt(j2));
            a.wmax = std::max(a.wmax, std::sqrt(w2));
            // wall shear stress: |d u_t / d n| on the wall nodes of a no-slip box
            if (!o.periodic && o.noslip)
              for (int k = 0; k < 3; ++k)
                if (p[k] == 0 || p[k] == L - 1) {
                  double st = 0;
                  for (int c = 0; c < 3; ++c) if (c != k) st += du[c][k] * du[c][k];
                  a.tw += nu_tg * std::sqrt(st);
                  a.ntw += 1;
                }
          }
    };

    // Wall-distance layers: 0 .. (N - 1)/2 cells, in the box and in the twin alike.
    const std::size_t nlay = std::size_t((N - 1) / 2 + 1);
    auto measure = [&](double t) {
      std::vector<Acc> part(static_cast<std::size_t>(nthreads), Acc(nlay));
      parallel_z(L, nthreads, [&](Index z0, Index z1, int tid) {
        measure_slab(z0, z1, part[static_cast<std::size_t>(tid)]);
      });
      Acc a(nlay);
      for (const Acc& q : part) a.add(q);
      Diag g; g.t = t;
      g.finite = a.finite;
      g.umax_lat = a.umax; g.jmax = a.jmax; g.wmax = a.wmax;
      const double W = a.W;
      g.ev = a.ev / W; g.em = a.em / W; g.hc = a.hc / W;
      g.omv = 0.5 * a.w2 / W; g.omm = 0.5 * a.j2 / W;
      g.eps = 2.0 * nu_tg * g.omv + 2.0 * eta_tg * g.omm;
      // Direction-averaged moments FIRST, then the ratio. Averaging three ratios
      // instead divides round-off by round-off wherever one component of u is
      // identically zero -- u_z at t = 0 -- and read 4.8 for a field whose third
      // moment vanishes by symmetry.
      const double m2 = (a.d2[0] + a.d2[1] + a.d2[2]) / (3.0 * W);
      const double m3 = (a.d3[0] + a.d3[1] + a.d3[2]) / (3.0 * W);
      g.skew = (m2 > 0) ? -m3 / std::pow(m2, 1.5) : 0.0;
      g.divb = (a.j2 > 0) ? std::sqrt(a.div2 / a.j2) : 0.0;
      g.divb_wall = (a.j2w > 0 && a.ndivw > 0) ? std::sqrt(a.divw / a.j2w) : 0.0;
      double etot = 0.0;
      for (double e : a.lay_e) etot += e;
      for (int m = 0; m < 3; ++m)
        g.fw[m] = (etot > 0) ? within_cells(a.lay_e, double(1 << m) * delta / h) / etot : 0.0;
      g.prof.resize(nlay); g.vol.resize(nlay);
      for (std::size_t k = 0; k < nlay; ++k) {
        g.prof[k] = (etot > 0) ? a.lay_e[k] / etot : 0.0;
        g.vol[k] = a.lay_w[k] / W;
      }
      g.tauw = (a.ntw > 0) ? a.tw / a.ntw : 0.0;
      g.mass = (t > 0) ? a.mass / mass0 - 1.0 : 0.0;   // exact by construction at t = 0
      return g;
    };

    // ---- output bookkeeping ---------------------------------------------------
    auto every = [&](double dtp) -> std::size_t {
      return dtp > 0 ? std::max<std::size_t>(1, std::size_t(std::llround(dtp / dt))) : 0;
    };
    const std::size_t kp = every(o.probe), kv = every(o.vti), kd = every(o.dump),
                      kr = every(o.raw);
    std::FILE* ser = std::fopen((o.out + "/series.dat").c_str(), "w");
    if (!ser) { std::fprintf(stderr, "tg_mhd: cannot write series.dat\n"); Kokkos::finalize(); return 2; }
    // fw=profile marks f_w read off the wall-distance profile (within_cells);
    // series without it counted whole node layers, and tools/tg_mhd_ladder.py
    // corrects those.
    std::fprintf(ser, "# tg_mhd (Kokkos %s)  %s  N=%lld L=%lld Re=%g Pm=%g u0=%g nu_lat=%.6e"
                      " eta_lat=%.6e tau_f=%.6f tau_m=%.6f dt=%.6e fw=profile\n",
                 sizeof(Real) == 4 ? "FP32" : "FP64",
                 tag, (long long)N, (long long)L, o.re, o.pm, o.u0, nu_lat, eta_lat, tauf, taum, dt);
    std::fprintf(ser, "# t E_V E_M E_T H_C Omega_V Omega_M eps EM/EV OmM/OmV j_max w_max"
                      " skew divb/j divb_wall/j mass_drift umax_lat fw1 fw2 fw4 tau_wall\n");
    // profile.dat: the dissipation against wall distance, one row per probe --
    // what f_w is read from, and a figure of its own.
    std::FILE* pro = std::fopen((o.out + "/profile.dat").c_str(), "w");
    if (!pro) { std::fprintf(stderr, "tg_mhd: cannot write profile.dat\n"); Kokkos::finalize(); return 2; }
    std::fprintf(pro, "# tg_mhd (Kokkos %s)  %s  N=%lld Re=%g h=%.8e delta=%.8e delta/h=%.6f layers=%zu\n",
                 sizeof(Real) == 4 ? "FP32" : "FP64", tag, (long long)N, o.re, h, delta, delta / h, nlay);
    std::fprintf(pro, "# layer k: the nodes k cells from the nearest wall, i.e. the slab"
                      " [(k-1/2)h, (k+1/2)h] ([0, h/2] for k = 0)\n");
    std::fprintf(pro, "# row 'vol': each layer's share of the volume; then per probe: t and"
                      " each layer's share of eps\n");
    bool pro_vol = false;

    std::FILE* meta = nullptr;
    int dframe = 0, vframe = 0, rframe = 0;
    std::vector<std::pair<double, std::string>> pvd;
    if (kd) {
      std::filesystem::create_directories(o.out + "/anim_frames", ec);
      meta = std::fopen((o.out + "/anim_frames/meta.txt").c_str(), "w");
      // R 0: no sphere to mask or draw -- render the whole slice (box mode).
      if (meta) std::fprintf(meta, "N %lld\nR 0\nTe 1.0\ncase tg_mhd %s\n", (long long)L, tag);
      // The grid and the |J| volume's stride, so render_volume.py captions the
      // real N rather than the reduced volume's -- it cannot infer a stride that
      // does not divide N (129 against 65).
      if (meta && o.dumpvol) {
        const long long S = o.volstride, Mr = (L + S - 1) / S;
        std::fprintf(meta, "nx %lld\nny %lld\nnz %lld\nvstride %lld\nvnx %lld\nvny %lld\nvnz %lld\n",
                     (long long)L, (long long)L, (long long)L, S, Mr, Mr, Mr);
      }
    }
    if (kv) std::filesystem::create_directories(o.out + "/vti", ec);
    if (kr) std::filesystem::create_directories(o.out + "/raw", ec);

    double e0 = 0, em0 = 0;
    const Index zc = (N - 1) / 2;                          // the z = pi/2 plane
    auto frames = [&](std::size_t k, double t, const Diag& g) {
      const std::size_t np = std::size_t(L) * L * L;
      // |J| and |omega| everywhere, paper units, shared by all three writers.
      std::vector<float> jm, wm;
      std::vector<float> jv3, wv3;
      const bool need_vec = kv && k % kv == 0;
      if ((kd && k % kd == 0) || need_vec) {
        jm.resize(np); wm.resize(np);
        if (need_vec) { jv3.resize(3 * np); wv3.resize(3 * np); }
        parallel_z(L, nthreads, [&](Index z0, Index z1, int) {
        for (Index z = z0; z < z1; ++z)
          for (Index y = 0; y < L; ++y)
            for (Index x = 0; x < L; ++x) {
              const std::size_t m = (std::size_t(z) * L + y) * L + x;
              double du[3][3], db[3][3];
              for (int c = 0; c < 3; ++c)
                for (int kk = 0; kk < 3; ++kk) {
                  du[c][kk] = H.d(c, kk, x, y, z) * ih;
                  db[c][kk] = hydro ? 0.0 : H.d(3 + c, kk, x, y, z) * ih;
                }
              const double wv[3] = {du[2][1] - du[1][2], du[0][2] - du[2][0], du[1][0] - du[0][1]};
              const double jv[3] = {db[2][1] - db[1][2], db[0][2] - db[2][0], db[1][0] - db[0][1]};
              jm[m] = float(std::sqrt(jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2]));
              wm[m] = float(std::sqrt(wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]));
              if (need_vec)
                for (int c = 0; c < 3; ++c) { jv3[3 * m + c] = float(jv[c]); wv3[3 * m + c] = float(wv[c]); }
            }
        });
      }
      if (kd && k % kd == 0) {
        std::vector<float> su(std::size_t(L) * L), sb(su.size()), sj(su.size()), sw(su.size());
        for (Index y = 0; y < L; ++y)
          for (Index x = 0; x < L; ++x) {
            const std::size_t m2 = std::size_t(y) * L + x;
            const std::size_t m3 = (std::size_t(zc) * L + y) * L + x;
            const double u[3] = {H.v(0, x, y, zc), H.v(1, x, y, zc), H.v(2, x, y, zc)};
            const double b[3] = {H.v(3, x, y, zc), H.v(4, x, y, zc), H.v(5, x, y, zc)};
            su[m2] = float(std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]));
            sb[m2] = float(std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]));
            sj[m2] = jm[m3];
            sw[m2] = wm[m3];
          }
        char nm[64];
        const std::string af = o.out + "/anim_frames/";
        std::snprintf(nm, sizeof nm, "umag_%04d.raw", dframe); write_raw2(af + nm, int(L), int(L), su);
        std::snprintf(nm, sizeof nm, "bmag_%04d.raw", dframe); write_raw2(af + nm, int(L), int(L), sb);
        std::snprintf(nm, sizeof nm, "jmag_%04d.raw", dframe); write_raw2(af + nm, int(L), int(L), sj);
        std::snprintf(nm, sizeof nm, "wmag_%04d.raw", dframe); write_raw2(af + nm, int(L), int(L), sw);
        if (o.dumpvol) {
          // BLOCK MAXIMUM, not mean, for the reason orszag_tang.cu gives: the
          // renderer draws a max-intensity projection and a mean would dim the
          // one-cell sheets the picture is of.
          const int S = o.volstride, Mr = int((L + S - 1) / S);
          std::vector<float> red(std::size_t(Mr) * Mr * Mr, 0.0f);
          for (Index z = 0; z < L; ++z)
            for (Index y = 0; y < L; ++y)
              for (Index x = 0; x < L; ++x) {
                const std::size_t r3 = (std::size_t(z / S) * Mr + std::size_t(y / S)) * Mr + std::size_t(x / S);
                red[r3] = std::max(red[r3], jm[(std::size_t(z) * L + y) * L + x]);
              }
          std::snprintf(nm, sizeof nm, "jvol_%04d.raw", dframe);
          write_raw3(af + nm, Mr, Mr, Mr, red);
        }
        if (meta) {
          std::fprintf(meta, "frame %d %.6f %.8e %.8e\n", dframe, t,
                       e0 > 0 ? g.ev / e0 : 0.0, e0 > 0 ? g.em / e0 : 0.0);
          std::fflush(meta);
        }
        ++dframe;
      }
      if (need_vec) {
        std::vector<VtiArray> arr;
        VtiArray au{"u", 3, {}}, ab{"b", 3, {}}, aw{"omega", 3, {}}, aj{"J", 3, {}};
        VtiArray ajm{"Jmag", 1, {}}, awm{"Wmag", 1, {}}, ar{"rho", 1, {}};
        au.data.resize(3 * np); ab.data.resize(3 * np); ar.data.resize(np);
        std::size_t m = 0;
        for (Index z = 0; z < L; ++z)
          for (Index y = 0; y < L; ++y)
            for (Index x = 0; x < L; ++x, ++m) {
              for (int c = 0; c < 3; ++c) {
                au.data[3 * m + c] = float(H.v(c, x, y, z));
                ab.data[3 * m + c] = float(H.v(3 + c, x, y, z));
              }
              ar.data[m] = float(H.v(6, x, y, z));
            }
        aj.data = std::move(jv3); aw.data = std::move(wv3);
        ajm.data = jm; awm.data = wm;
        arr.push_back(std::move(ajm)); arr.push_back(std::move(au)); arr.push_back(std::move(ab));
        arr.push_back(std::move(aj)); arr.push_back(std::move(aw)); arr.push_back(std::move(awm));
        arr.push_back(std::move(ar));
        char nm[64];
        std::snprintf(nm, sizeof nm, "tg_%04d.vti", vframe++);
        write_vti_bin(o.out + "/vti/" + nm, L, L, L, arr, h);
        pvd.emplace_back(t, nm);
      }
      if (kr && k % kr == 0) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "fields_%04d.raw", rframe++);
        write_raw_fields(o.out + "/raw/" + nm, H, t, h);
      }
    };

    std::printf("\n  %7s %11s %11s %9s %9s %9s %9s %9s %9s %10s\n", "t", "E_V", "E_M", "EM/EV",
                "OmM/OmV", "eps", "j_max", "divb/j", "mass", "fw(1d)");
    const auto wall0 = std::chrono::steady_clock::now();
    // "First maximum after the initial time" (Pouquet et al., Table 1 note):
    // the conducting field starts with Omega_M/Omega_V ~ 4 -- its wavenumber is
    // twice the velocity's -- and FALLS, so the quantity is the first peak after
    // that initial descent. Phase 0 descends to a minimum, phase 1 climbs to the
    // peak, phase 2 has it.
    // The dissipation peak is the TURBULENT maximum, not the largest eps in the
    // run: a no-slip box starts impulsively -- the Taylor-Green velocity does not
    // vanish on the walls -- and its eps is largest at t = 0 (fw1 = 0.91 there,
    // against 0.15 at the t = 2 peak, N = 129, Re = 300). So the maximum is taken
    // only once eps has risen 5 % above its running minimum: at once for free
    // slip, after the start has decayed for no slip. tools/tg_mhd_ladder.py reads
    // the peak the same way.
    double first_ommv_max = 0, prev_ommv = 0, emev_min = 1e300, eps_max = 0, t_epsmax = 0,
           skew_at_epsmax = 0, skew_max = -1e300, eps_runmin = 1e300, t_lastprobe = 0;
    bool eps_armed = false;
    int ommv_phase = 0, nprobe = 0;
    double t_diag = 0.0;             // wall time in the host copies, probes and writers
    for (std::size_t k = 0; k <= T; ++k) {
      const bool probe = kp && (k % kp == 0 || k == T);
      const bool outp = (kv && k % kv == 0) || (kd && k % kd == 0) || (kr && k % kr == 0);
      if (probe || outp) {
        const auto d0 = std::chrono::steady_clock::now();
        refresh();
        ++nprobe;
        const double t = double(k) * dt;
        const Diag g = measure(t);
        if (k == 0) { e0 = g.ev + g.em; em0 = g.em; }
        if (!g.finite) {
          std::printf("  DIVERGED at t = %.4f (step %zu)\n", t, k);
          std::fprintf(ser, "# DIVERGED at t = %.6f\n", t);
          rc = 1;
          break;
        }
        if (probe) {
          const double emev = g.ev > 0 ? g.em / g.ev : 0.0;
          const double ommv = g.omv > 0 ? g.omm / g.omv : 0.0;
          if (k > 0) {
            emev_min = std::min(emev_min, emev);
            if (ommv_phase == 0 && ommv > prev_ommv) ommv_phase = 1;
            else if (ommv_phase == 1 && ommv < prev_ommv) { first_ommv_max = prev_ommv; ommv_phase = 2; }
            skew_max = std::max(skew_max, g.skew);
          }
          if (!eps_armed) {
            if (g.eps < eps_runmin) eps_runmin = g.eps;
            else if (g.eps > 1.05 * eps_runmin) eps_armed = true;
          }
          if (eps_armed && g.eps > eps_max) { eps_max = g.eps; t_epsmax = t; skew_at_epsmax = g.skew; }
          t_lastprobe = t;
          prev_ommv = ommv;
          std::fprintf(ser, "%.6f %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.6e %.6e %.6e %.6e"
                            " %.6e %.4e %.4e %+.4e %.5f %.5e %.5e %.5e %.5e\n",
                       t, g.ev, g.em, g.ev + g.em, g.hc, g.omv, g.omm, g.eps, emev, ommv,
                       g.jmax, g.wmax, g.skew, g.divb, g.divb_wall, g.mass, g.umax_lat,
                       g.fw[0], g.fw[1], g.fw[2], g.tauw);
          std::fflush(ser);
          if (!pro_vol) {
            std::fprintf(pro, "# vol");
            for (double v : g.vol) std::fprintf(pro, " %.6e", v);
            std::fprintf(pro, "\n");
            pro_vol = true;
          }
          std::fprintf(pro, "%.6f", t);
          for (double s : g.prof) std::fprintf(pro, " %.5e", s);
          std::fprintf(pro, "\n");
          std::fflush(pro);
          if (kp && (k % (kp * 10) == 0 || k == T))
            std::printf("  %7.3f %11.5e %11.5e %9.4f %9.4f %9.3e %9.3f %9.2e %+9.1e %10.4f\n",
                        t, g.ev, g.em, emev, ommv, g.eps, g.jmax, g.divb, g.mass, g.fw[0]);
          std::fflush(stdout);
        }
        if (outp) frames(k, t, g);
        t_diag += std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
      }
      if (k < T) {
        if (!hydro) mag.compute_field();
        fl.step(true);
        if (!hydro) mag.step(true);
      }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    std::fclose(ser);
    std::fclose(pro);
    if (meta) { std::fprintf(meta, "frames %d\n", dframe); std::fclose(meta); }
    if (!pvd.empty()) write_pvd(o.out + "/vti/tg.pvd", pvd);

    std::printf("\n  %.1f s   %.2f MLUPS (fluid nodes x steps; field and probes included)\n", secs,
                double(L) * L * L * double(T) / secs * 1e-6);
    std::printf("  host diagnostics: %d probes, %.1f s = %.0f %% of the run on %d thread(s);"
                " %.2f MLUPS without them\n",
                nprobe, t_diag, 100.0 * t_diag / secs, nthreads,
                double(L) * L * L * double(T) / std::max(secs - t_diag, 1e-9) * 1e-6);
    if (!eps_armed)
      std::printf("  no dissipation peak: eps fell throughout the run; skewness max %.3f\n", skew_max);
    else if (t_epsmax >= t_lastprobe)
      std::printf("  dissipation peak not reached: eps = %.4e still rising at t = %.3f; skewness max %.3f\n",
                  eps_max, t_epsmax, skew_max);
    else
      std::printf("  peak dissipation eps = %.4e at t = %.3f (past any impulsive start);"
                  " skewness there %.3f, max %.3f\n", eps_max, t_epsmax, skew_at_epsmax, skew_max);
    if (!hydro) {
      std::printf("  min E_M/E_V = %.3f   first max of Omega_M/Omega_V after its initial fall = %s\n",
                  emev_min, ommv_phase == 2 ? std::to_string(first_ommv_max).c_str()
                                            : "not reached");
      if (o.ic == ICKind::TGC && std::abs(o.re - 1000.0) < 1e-9)
        std::printf("  Pouquet et al. (2010) C2, nu = eta = 1e-3: min E_M/E_V = 0.35, first max"
                    " Omega_M/Omega_V = 2\n");
    }
    if (dframe) std::printf("  %d frame(s) in %s/anim_frames (render_slices.py, render_volume.py)\n",
                            dframe, o.out.c_str());
    if (!pvd.empty()) std::printf("  ParaView: open %s/vti/tg.pvd (%zu frames)\n", o.out.c_str(), pvd.size());
    if (rframe) std::printf("  %d raw field file(s) in %s/raw\n", rframe, o.out.c_str());
    (void)em0;
  }
  Kokkos::finalize();
  return rc;
}
