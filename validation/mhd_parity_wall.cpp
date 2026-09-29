//==============================================================================
//  The ON-NODE PARITY WALL for Dellar's magnetic distribution, checked as an
//  IDENTITY, against the parity it claims to impose, and against an exact
//  decay rate.
//
//  WHAT IS UNDER TEST. MagneticSolver::set_parity_walls mirrors the UNKNOWN
//  populations of each component of B with the sign that component's parity
//  gives it (mirror_unknowns_parity, boundary/Specular.hpp):
//
//      conducting     B_n odd,  B_t even   ->  B.n = 0,  d_n B_t = 0
//      pseudo-vacuum  B_n even, B_t odd    ->  B x n = 0, d_n B_n = 0
//
//  and then collides the node like any other. It exists because the tree had
//  no general conducting wall: MagNeumann is dB/dn = 0 on EVERY component,
//  which is right for Hunt's duct (the induced field is tangential there) and
//  leaves B.n free in a closed box -- where it goes non-finite in sixty steps.
//
//  FOUR CHECKS, and each one is there because the others cannot see something.
//
//  1. THE MOMENT IDENTITIES, on a random population vector, for faces, edges and
//     a corner, both parities. Along every masked axis the pair (+k, -k) must
//     either cancel (the component the plane flips) or carry no flux (the one
//     it keeps). Plus the fixed point: a state that already has the parity must
//     come back bit for bit, which rules out a wall that rescales or clamps.
//
//  2. THE IDENTITY. The Taylor-Green MHD fields of Lee et al. (PRE 78, 066401,
//     2008) and Pouquet et al. (GAFD 104, 115, 2010) are symmetric about the six
//     faces of [0, pi]^3. Their "conducting" field (b ~ sin2x cos2y cos2z, ...)
//     has exactly the conducting parity there and their "insulating" one
//     (b ~ cos x sin y sin z, ...) exactly the pseudo-vacuum parity. So a box of
//     N nodes a side, walls ON nodes 0 and N-1, closed by SpecNode for the fluid
//     and the parity wall for the field, is not an approximation to the
//     periodic 2pi box of 2(N-1) nodes -- it is the same problem, and must
//     reproduce it node for node to ROUND-OFF, transient included. A wall that
//     was only nearly the mirror would still converge; it would fail this.
//
//  3. THE NO-SLIP BOX, which has no periodic twin. RegWall for the fluid and the
//     conducting parity wall for the field: B.n on every wall node must stay at
//     round-off although the velocity there is now pinned rather than mirrored,
//     and the total mass is reported, because RegWall's 3-D corners carry no
//     density stencil (they fall back to rho = 1; see set_regularized_walls).
//
//  4. AN EXACT DECAY RATE, in that no-slip box. B = B0 (sin x cos y,
//     -cos x sin y, 0) meets the conducting condition on all six faces, its
//     Lorentz force is the pure gradient B0^2 grad(psi^2) with psi = sin x sin y,
//     so with the balancing pressure the fluid stays at rest and B decays at
//     exactly eta k^2, k^2 = 2 (pi/(N-1))^2 per step. Measured at second order,
//     2.00 and 2.00 over N = 17/33/65 (-full). What it establishes is that the
//     no-slip fluid wall leaves a magnetic eigenmode alone; the magnetic wall's
//     exactness is check 2's, since with the fluid at rest this box is the
//     periodic one again. It needs a SMALL amplitude, and why is written at the
//     call: at b0 = 0.04 a compressible pressure flow contaminated the rate and
//     produced an apparent fourth order that turned negative at the third point.
//
//  WHAT THIS DOES NOT TEST. The true insulator, which matches B to a potential
//  field outside and is nonlocal. Oblique or curved walls: the mirror exists
//  only on axis-aligned planes. And any flow statistic: that is the business of
//  demonstrator/tg_mhd.cpp.
//
//    usage: mhd_parity_wall [-n N] [-steps K] [-full]
//==============================================================================
#include "Campaign.hpp"
#include "boundary/Specular.hpp"
#include "collision/MagneticBGK.hpp"
#include "collision/MhdCentralMomentsShifted.hpp"
#include "solver/MagneticSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

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

constexpr double PI = 3.14159265358979323846;

//==============================================================================
//  1. THE MOMENT IDENTITIES
//==============================================================================
static int moments(std::uint8_t faces, bool conducting, const char* name) {
  int fails = 0;
  double worst = 0.0, untouched = 0.0;
  for (int a = 0; a < 3; ++a) {
    double g[ML::Q], g0[ML::Q];
    unsigned st = 2654435761u * unsigned(a + 1) + unsigned(faces);
    for (int i = 0; i < ML::Q; ++i) {
      st = st * 1664525u + 1013904223u;
      g0[i] = g[i] = double(st >> 8) / double(1u << 24) - 0.5;
    }
    mirror_unknowns_parity<ML>(g, faces, a, conducting);
    for (int k = 0; k < 3; ++k) {
      const int fs = face_sign(faces, k);
      if (fs == 0) continue;
      int ip = -1, im = -1;
      for (int i = 0; i < ML::Q; ++i) {
        if (cvel<ML>(i, k) == 1 && cvel<ML>(i, (k + 1) % 3) == 0 &&
            cvel<ML>(i, (k + 2) % 3) == 0) ip = i;
        if (cvel<ML>(i, k) == -1 && cvel<ML>(i, (k + 1) % 3) == 0 &&
            cvel<ML>(i, (k + 2) % 3) == 0) im = i;
      }
      const bool flipped = ((a == k) == conducting);
      const double r = flipped ? g[ip] + g[im] : g[ip] - g[im];
      worst = std::max(worst, std::abs(r));
    }
    // Every direction that did NOT point in from outside must be untouched.
    for (int i = 0; i < ML::Q; ++i) {
      bool in = false;
      for (int k = 0; k < 3; ++k) {
        const int fs = face_sign(faces, k);
        if (fs != 0 && cvel<ML>(i, k) * fs < 0) in = true;
      }
      if (!in) untouched = std::max(untouched, std::abs(g[i] - g0[i]));
    }
  }

  // The fixed point: an equilibrium that already HAS the parity on every
  // masked plane (the flipped components zero there, u_n = 0) must come back
  // unchanged. B and u are chosen as generic as that allows.
  double B[3] = {0.031, -0.027, 0.019}, u[3] = {0.021, 0.013, -0.017};
  for (int k = 0; k < 3; ++k) {
    if (face_sign(faces, k) == 0) continue;
    u[k] = 0.0;
    for (int a = 0; a < 3; ++a)
      if ((a == k) == conducting) B[a] = 0.0;
  }
  const Real Br[3] = {Real(B[0]), Real(B[1]), Real(B[2])};
  const Real ur[3] = {Real(u[0]), Real(u[1]), Real(u[2])};
  double fixed = 0.0;
  for (int a = 0; a < 3; ++a) {
    double g[ML::Q], ge[ML::Q];
    for (int i = 0; i < ML::Q; ++i) ge[i] = g[i] = double(MC::eq(i, a, Br, ur));
    mirror_unknowns_parity<ML>(g, faces, a, conducting);
    for (int i = 0; i < ML::Q; ++i) fixed = std::max(fixed, std::abs(g[i] - ge[i]));
  }

  const bool ok = worst < 1e-15 && untouched == 0.0 && fixed < 1e-17;
  if (!ok) ++fails;
  std::printf("    %-13s %-4s  pair identity %.1e  untouched %.1e  fixed point %.1e  %s\n",
              name, conducting ? "cond" : "pv", worst, untouched, fixed,
              ok ? "ok" : "FAIL");
  return fails;
}

//==============================================================================
//  The coupled runs
//==============================================================================
enum class IC { TGC, TGI, PSI };

struct Run {
  Index N = 13;             // box nodes a side; planes ON nodes 0 and N-1
  bool periodic = false;    // the 2(N-1) periodic box the box mirrors
  bool noslip = false;      // box only: RegWall rather than SpecNode
  bool conducting = true;   // magnetic parity of the box
  IC ic = IC::TGC;
  double u0 = 0.04;         // lattice velocity of the unit TG amplitude
  double b0 = 0.04;         // lattice field amplitude (PSI) or of the unit b0
  double nu = 0.01;         // nu = eta, lattice units
  std::size_t steps = 150;
  std::size_t every = 0;    // record E_M every `every` steps (0 = never)
};

struct Out {
  std::vector<double> f;    // ux uy uz bx by bz rho on box nodes 0..N-1
  std::vector<double> em;   // magnetic energy history (if every > 0)
  double bn_wall = 0.0;     // max |B.n| on the box's wall nodes, final state
  double mass0 = 0.0, mass1 = 0.0;
  double umax = 0.0, bmax = 0.0;
  bool finite = true;
};

// The initial state at interior node (x, y, z), TG units h per cell. The TG
// velocity is Brachet's; the two TG fields are Pouquet et al. (2010): B_C
// (their "conducting" case, wavenumber 2) and IMTG (Eqs. 5-7). PSI is the exact
// decaying mode of check 4, with the pressure that balances its Lorentz force.
struct Init {
  IC ic;
  double h, u0, b0;
  KOKKOS_INLINE_FUNCTION void at(Index x, Index y, Index z, double& rho, double u[3],
                                 double b[3]) const {
    const double X = h * double(x), Y = h * double(y), Z = h * double(z);
    rho = 1.0;
    u[0] = u[1] = u[2] = 0.0;
    b[0] = b[1] = b[2] = 0.0;
    if (ic == IC::PSI) {
      const double ps = Kokkos::sin(X) * Kokkos::sin(Y);
      rho  = 1.0 + b0 * b0 * (ps * ps - 0.25) * 3.0;      // p = b0^2 psi^2, cs2 = 1/3
      b[0] =  b0 * Kokkos::sin(X) * Kokkos::cos(Y);
      b[1] = -b0 * Kokkos::cos(X) * Kokkos::sin(Y);
      return;
    }
    u[0] =  u0 * Kokkos::sin(X) * Kokkos::cos(Y) * Kokkos::cos(Z);
    u[1] = -u0 * Kokkos::cos(X) * Kokkos::sin(Y) * Kokkos::cos(Z);
    if (ic == IC::TGC) {
      b[0] =  b0 * Kokkos::sin(2 * X) * Kokkos::cos(2 * Y) * Kokkos::cos(2 * Z);
      b[1] =  b0 * Kokkos::cos(2 * X) * Kokkos::sin(2 * Y) * Kokkos::cos(2 * Z);
      b[2] = -2 * b0 * Kokkos::cos(2 * X) * Kokkos::cos(2 * Y) * Kokkos::sin(2 * Z);
    } else {
      b[0] =  b0 * Kokkos::cos(X) * Kokkos::sin(Y) * Kokkos::sin(Z);
      b[1] =  b0 * Kokkos::sin(X) * Kokkos::cos(Y) * Kokkos::sin(Z);
      b[2] = -2 * b0 * Kokkos::sin(X) * Kokkos::sin(Y) * Kokkos::cos(Z);
    }
  }
};

static std::uint8_t box_faces(Index x, Index y, Index z, Index N) {
  std::uint8_t m = SpecNone;
  if (x == 0) m |= SpecXm; else if (x == N - 1) m |= SpecXp;
  if (y == 0) m |= SpecYm; else if (y == N - 1) m |= SpecYp;
  if (z == 0) m |= SpecZm; else if (z == N - 1) m |= SpecZp;
  return m;
}

static std::uint8_t box_normal(Index x, Index y, Index z, Index N) {
  int nf = 0;
  std::uint8_t c = NrmNone;
  if (x == 0)     { ++nf; c = NrmXm; } else if (x == N - 1) { ++nf; c = NrmXp; }
  if (y == 0)     { ++nf; c = NrmYm; } else if (y == N - 1) { ++nf; c = NrmYp; }
  if (z == 0)     { ++nf; c = NrmZm; } else if (z == N - 1) { ++nf; c = NrmZp; }
  if (nf >= 2) return NrmCorner;
  return c;
}

template <class F>
static double sum_box(Index N, F&& f) {
  double s = 0.0;
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) s += f(x, y, z);
  return s;
}

static Out run(const Run& r) {
  const Index N = r.N;
  const Index L = r.periodic ? 2 * (N - 1) : N;
  Domain d(L, L, L, r.periodic, r.periodic, r.periodic);

  MC mc;
  mc.omega = MC::omega_from_resistivity(Real(r.nu));
  MagT mag(d, mc);

  FC fc;
  fc.omega = FC::omega_from_viscosity(Real(r.nu));
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidT fl(d, fc);
  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });

  if (!r.periodic) {
    if (r.noslip) {
      fl.set_fd_corners(false);   // FD edge closure is incompatible with a Lorentz force
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
    mag.set_parity_walls([&](Index x, Index y, Index z) -> std::uint8_t {
                           return box_faces(x, y, z, N);
                         },
                         r.conducting ? MagParity::Conducting : MagParity::PseudoVacuum);
  }

  const Init init{r.ic, PI / double(N - 1), r.u0, r.b0};
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    double rho, u[3], b[3];
    init.at(px - d.hx, py - d.hy, pz - d.hz, rho, u, b);
    return FlowState{Real(rho), Real(u[0]), Real(u[1]), Real(u[2])};
  });
  // SEED THE MHD EQUILIBRIUM, not the hydrodynamic one initialize_field uses.
  // Collision::seed_value is static and never sees B, so every coupled case
  // otherwise starts with the whole Maxwell stress sitting in the
  // non-equilibrium part, to relax over the first few steps. Seeding through
  // the operator's own equilibrium() starts it where it belongs.
  //
  // WHAT IT DOES NOT DO, measured 2026-09-25 because it was the hypothesis that
  // motivated it: it does not remove the no-slip box's mass loss under a field.
  // That was -3.5e-4 over 150 steps at N = 13 with this seed and -3.3e-4 with
  // the hydrodynamic one (monomial operator), against +9e-5 with no field at
  // all -- so it belongs to RegWall with a Lorentz force, not to the initial
  // condition. It reads -3.7e-4 on the shifted operator used now. Check 3
  // bounds it rather than pretending it away.
  static_assert(std::is_same_v<FC::Storage, RawPopulations>,
                "the seed below writes raw populations");
  fl.seed_populations(KOKKOS_LAMBDA(Index n, int i) {
    Index px, py, pz; d.coords(n, px, py, pz);
    double rho, u[3], b[3];
    init.at(px - d.hx, py - d.hy, pz - d.hz, rho, u, b);
    const Real ur[3] = {Real(u[0]), Real(u[1]), Real(u[2])};
    const Real br[3] = {Real(b[0]), Real(b[1]), Real(b[2])};
    Real fe[FL::Q];
    fc.equilibrium(fe, Real(rho), ur, br);
    return fe[i];
  });
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    double rho, u[3], b[3];
    init.at(px - d.hx, py - d.hy, pz - d.hz, rho, u, b);
    Kokkos::Array<Real, 3> bb;
    bb[0] = Real(b[0]); bb[1] = Real(b[1]); bb[2] = Real(b[2]);
    return bb;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  Out o;
  auto energy_m = [&]() {
    mag.compute_field();
    auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    auto bz = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());
    return sum_box(N, [&](Index x, Index y, Index z) {
      const Index n = d.id(x, y, z);
      return 0.5 * (double(bx(n)) * bx(n) + double(by(n)) * by(n) + double(bz(n)) * bz(n));
    });
  };
  // The initial mass from the SEEDED state, not from compute_macroscopic():
  // before the first step the macro kernel skips RegWall edge and corner nodes
  // and they read zero, which once made this box look 6.8 % heavier after 150
  // steps than it was.
  auto mass = [&]() {
    fl.compute_macroscopic();
    auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
    return sum_box(L, [&](Index x, Index y, Index z) { return double(hr(d.id(x, y, z))); });
  };
  o.mass0 = sum_box(L, [&](Index x, Index y, Index z) {
    double rho, u[3], b[3];
    init.at(x, y, z, rho, u, b);
    return rho;
  });
  if (r.every) o.em.push_back(energy_m());
  for (std::size_t t = 1; t <= r.steps; ++t) {
    mag.compute_field();
    fl.step(true);
    mag.step(true);
    if (r.every && t % r.every == 0) o.em.push_back(energy_m());
  }
  o.mass1 = mass();

  fl.compute_macroscopic();
  mag.compute_field();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto uz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uz());
  auto rh = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
  auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
  auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
  auto bz = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bz());
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y, z);
        const double v[7] = {double(ux(n)), double(uy(n)), double(uz(n)),
                             double(bx(n)), double(by(n)), double(bz(n)), double(rh(n))};
        for (double q : v) { o.f.push_back(q); if (!std::isfinite(q)) o.finite = false; }
        o.umax = std::max(o.umax, std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
        o.bmax = std::max(o.bmax, std::sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]));
        // B.n on the walls: the component normal to every face the node is on.
        if (x == 0 || x == N - 1) o.bn_wall = std::max(o.bn_wall, std::abs(v[3]));
        if (y == 0 || y == N - 1) o.bn_wall = std::max(o.bn_wall, std::abs(v[4]));
        if (z == 0 || z == N - 1) o.bn_wall = std::max(o.bn_wall, std::abs(v[5]));
      }
  return o;
}

// B x n on the walls, for the pseudo-vacuum box: the tangential components.
static double bt_wall(const Out& o, Index N) {
  double w = 0.0;
  for (Index z = 0; z < N; ++z)
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const std::size_t i = 7 * ((std::size_t(z) * N + y) * N + x);
        const double b[3] = {o.f[i + 3], o.f[i + 4], o.f[i + 5]};
        for (int k = 0; k < 3; ++k) {
          const Index c = (k == 0) ? x : (k == 1) ? y : z;
          if (c != 0 && c != N - 1) continue;
          for (int a = 0; a < 3; ++a) if (a != k) w = std::max(w, std::abs(b[a]));
        }
      }
  return w;
}

// rms(div b) by distance from the nearest wall, for the nodes whose central
// stencil stays inside the box (distance >= 1), normalised by rms |curl b| over
// the same nodes. The profile is what shows a wall EXCESS; a single number
// would average it away, which is how the sphere's div b first hid.
static std::vector<double> divb_profile(const Out& o, Index N, double* jrms) {
  auto B = [&](Index x, Index y, Index z, int a) {
    return o.f[7 * ((std::size_t(z) * N + y) * N + x) + 3 + a];
  };
  const Index dmax = (N - 1) / 2;
  std::vector<double> s(std::size_t(dmax) + 1, 0.0), c(std::size_t(dmax) + 1, 0.0);
  double j2 = 0.0, nj = 0.0;
  for (Index z = 1; z < N - 1; ++z)
    for (Index y = 1; y < N - 1; ++y)
      for (Index x = 1; x < N - 1; ++x) {
        const Index dw = std::min({x, y, z, N - 1 - x, N - 1 - y, N - 1 - z});
        const double dv = 0.5 * (B(x + 1, y, z, 0) - B(x - 1, y, z, 0))
                        + 0.5 * (B(x, y + 1, z, 1) - B(x, y - 1, z, 1))
                        + 0.5 * (B(x, y, z + 1, 2) - B(x, y, z - 1, 2));
        const double jx = 0.5 * (B(x, y + 1, z, 2) - B(x, y - 1, z, 2))
                        - 0.5 * (B(x, y, z + 1, 1) - B(x, y, z - 1, 1));
        const double jy = 0.5 * (B(x, y, z + 1, 0) - B(x, y, z - 1, 0))
                        - 0.5 * (B(x + 1, y, z, 2) - B(x - 1, y, z, 2));
        const double jz = 0.5 * (B(x + 1, y, z, 1) - B(x - 1, y, z, 1))
                        - 0.5 * (B(x, y + 1, z, 0) - B(x, y - 1, z, 0));
        s[std::size_t(dw)] += dv * dv;
        c[std::size_t(dw)] += 1.0;
        j2 += jx * jx + jy * jy + jz * jz;
        nj += 1.0;
      }
  *jrms = std::sqrt(j2 / std::max(nj, 1.0));
  for (std::size_t k = 0; k < s.size(); ++k)
    s[k] = c[k] > 0 ? std::sqrt(s[k] / c[k]) / std::max(*jrms, 1e-300) : 0.0;
  return s;
}

static double compare(const Out& a, const Out& b, int c0, int c1, double* scale) {
  double w = 0.0, sc = 0.0;
  for (std::size_t i = 0; i < a.f.size(); i += 7)
    for (int c = c0; c < c1; ++c) {
      w = std::max(w, std::abs(a.f[i + c] - b.f[i + c]));
      sc = std::max(sc, std::abs(a.f[i + c]));
    }
  *scale = sc;
  return w;
}

//==============================================================================
static int check_identity(Index N, std::size_t steps, IC ic, bool conducting,
                          const char* label) {
  Run r;
  r.N = N; r.steps = steps; r.ic = ic; r.conducting = conducting;
  r.u0 = 0.04;
  // The TG field amplitudes that make E_V = E_M at t = 0 (Pouquet et al.):
  // b0 = 1/sqrt(3) for both of these fields when v0 = 1.
  r.b0 = 0.04 / std::sqrt(3.0);
  Run p = r; p.periodic = true;
  const Out ob = run(r), op = run(p);
  double su, sb;
  const double du = compare(op, ob, 0, 3, &su), db = compare(op, ob, 3, 6, &sb);
  double sr;
  const double dr = compare(op, ob, 6, 7, &sr);
  const double wall = conducting ? ob.bn_wall : bt_wall(ob, N);
  // Structure guard: two decayed fields agree whatever the wall does.
  const bool alive = su > 0.25 * r.u0 && sb > 0.1 * r.b0;
  const double tol = 1e-12;
  const bool ok = ob.finite && op.finite && alive && du / su < tol && db / sb < tol &&
                  dr < 10 * tol && wall < 1e-14 * std::max(sb, 1e-300) + 1e-17;
  std::printf("  %s  box %lld^3 vs periodic %lld^3, %zu steps\n", label, (long long)N,
              (long long)(2 * (N - 1)), steps);
  std::printf("    u   worst %.3e  (relative %.2e)\n", du, du / su);
  std::printf("    b   worst %.3e  (relative %.2e)\n", db, db / sb);
  std::printf("    rho worst %.3e\n", dr);
  std::printf("    %s on the wall nodes: %.2e   field still |u| %.4f |b| %.4f   %s\n",
              conducting ? "B.n  " : "B x n", wall, su, sb, ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}

static int check_noslip(Index N, std::size_t steps) {
  Run r;
  r.N = N; r.steps = steps; r.noslip = true; r.conducting = true; r.ic = IC::TGC;
  r.u0 = 0.04; r.b0 = 0.04 / std::sqrt(3.0);
  const Out o = run(r);
  double jr;
  const std::vector<double> pr = divb_profile(o, N, &jr);
  // The periodic twin of the SAME initial condition, as the reference level for
  // div b: no wall at all. Compared on the same interior nodes.
  Run p = r; p.periodic = true; p.noslip = false;
  const Out op = run(p);
  double jp;
  const std::vector<double> pp = divb_profile(op, N, &jp);
  // No wall EXCESS of div b: the layer against the wall must not stand out from
  // the box's own layers. The periodic twin is printed beside it as the level
  // the bulk scheme reaches with no wall at all -- a loose reference, since
  // no-slip changes the flow.
  double avg = 0.0;
  for (std::size_t k = 1; k < pr.size(); ++k) avg += pr[k];
  avg /= double(pr.size() - 1);
  const bool no_excess = pr[1] < 2.0 * avg;
  // RegWall is not mass conserving (CLAUDE.md), and under a Lorentz force this
  // box loses mass early: bounded here at the level measured, so a change that
  // made it worse would show.
  const double drift = (o.mass1 - o.mass0) / o.mass0;
  const bool ok = o.finite && o.bn_wall < 1e-14 && no_excess && std::abs(drift) < 1e-3;
  std::printf("  no-slip box %lld^3 (RegWall + conducting parity), TG-C, %zu steps\n",
              (long long)N, steps);
  std::printf("    B.n on the wall nodes %.2e   mass drift %+.3e (bound 1e-3)   |u| %.4f"
              " |b| %.4f   %s\n", o.bn_wall, drift, o.umax, o.bmax, ok ? "ok" : "FAIL");
  std::printf("    div b at the wall layer / box average: %.2f (must stay below 2)\n",
              pr[1] / avg);
  std::printf("    rms(div b)/rms|j| by distance from the nearest wall"
              "  (no-slip box | periodic twin)\n");
  for (std::size_t k = 1; k < pr.size(); ++k)
    std::printf("      d = %2zu   %.3e | %.3e\n", k, pr[k], pp[k]);
  return ok ? 0 : 1;
}

static int check_decay(bool full) {
  // tau_m = 1.0 on D3Q7 (cs2 = 1/4): eta = 0.125. Held FIXED across N -- the
  // diffusive ladder -- so the per-step rate falls as 1/N^2 and its relative
  // error isolates the spatial order.
  //
  // Measured: errors +1.61e-3, +4.02e-4, +1.00e-4 at N = 17, 33, 65, i.e. order
  // 2.00 and 2.00 -- the bulk scheme's own order, so the no-slip box costs the
  // magnetic field nothing. It cannot show the wall's own order: with the fluid
  // at rest and the field mirror-symmetric the box IS the periodic box again,
  // which is check 2's business.
  //
  // A TRAP THIS CHECK FELL INTO FIRST, worth keeping as a warning about orders.
  // At b0 = 0.04 the first two points gave order 4.32 (tau_m = 0.8) and 4.54
  // (tau_m = 1.0), which I first put down to tau_m sitting near the diffusion
  // "magic" value -- wrong, since 1.0 is nowhere near it -- and the third point
  // then went BACKWARDS (order -2.33). The cause was the compressible flow
  // described below, whose contribution does not shrink with N: at N = 33 it
  // happened to cancel most of the O(1/N^2) error, which read as super-
  // convergence. An apparent order well above the scheme's is a symptom, not a
  // bonus; a third point is what exposes it.
  const double eta = 0.125;
  std::vector<Index> Ns = {17, 33};
  if (full) Ns.push_back(65);
  std::vector<double> err;
  std::printf("  exact decay, no-slip box, B = b0 (sin x cos y, -cos x sin y, 0), eta = %.3f\n",
              eta);
  int fails = 0;
  for (Index N : Ns) {
    const double k2 = 2.0 * std::pow(PI / double(N - 1), 2);
    const double exact = 2.0 * eta * k2;        // E_M decays at twice B's rate
    // One e-fold of B, sampled 40 times, fitted over the second half so the
    // start-up acoustic adjustment is not in the fit.
    const std::size_t T = std::size_t(1.0 / (eta * k2));
    Run r;
    r.N = N; r.noslip = true; r.conducting = true; r.ic = IC::PSI;
    // SMALL AMPLITUDE, and it is the physics rather than a precaution. The
    // exact solution is the incompressible one, where the pressure re-balances
    // the decaying magnetic pressure instantly and u stays zero. Here that
    // needs a density change, hence a flow, and its induction term u k B
    // competes with eta k^2 B at a ratio that does NOT fall with N in the
    // diffusive ladder. At b0 = 0.04 it dominated by N = 65 (error -2.4e-4
    // against +4.9e-5 at N = 33, order -2.33). The coupling scales as b0^2 and
    // the decay itself is linear in B, so a small b0 isolates the diffusion:
    // at 1e-3 the fluid stays at |u|/b0 ~ 1e-5 and the order is 2.00.
    r.nu = eta; r.b0 = 1e-3; r.u0 = 0.0;
    r.every = std::max<std::size_t>(1, T / 40);
    r.steps = r.every * 40;
    const Out o = run(r);
    double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (std::size_t i = o.em.size() / 2; i < o.em.size(); ++i) {
      const double t = double(i * r.every), y = std::log(o.em[i]);
      sx += t; sy += y; sxx += t * t; sxy += t * y; n += 1;
    }
    const double rate = -(n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double rel = (rate - exact) / exact;
    err.push_back(std::abs(rel));
    const bool ok = o.finite && o.bn_wall < 1e-14 && o.umax < 0.05 * r.b0;
    if (!ok) ++fails;
    std::printf("    N = %3lld   rate %.6e  exact %.6e  error %+.3e   |u|/b0 %.1e"
                "   B.n wall %.1e  %s\n",
                (long long)N, rate, exact, rel, o.umax / r.b0, o.bn_wall,
                ok ? "ok" : "FAIL");
  }
  for (std::size_t i = 1; i < err.size(); ++i) {
    const double order = std::log(err[i - 1] / err[i]) /
                         std::log(double(Ns[i] - 1) / double(Ns[i - 1] - 1));
    const bool ok = order > 1.8;
    if (!ok) ++fails;
    std::printf("    order %lld -> %lld : %.2f  %s\n", (long long)Ns[i - 1],
                (long long)Ns[i], order, ok ? "ok" : "FAIL (need > 1.8)");
  }
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    Index N = 13;
    std::size_t steps = 150;
    bool full = false;
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::atoi(argv[++i]);
      else if (!std::strcmp(argv[i], "-steps") && i + 1 < argc)
        steps = std::size_t(std::atoll(argv[++i]));
      else if (!std::strcmp(argv[i], "-full")) full = true;
    }
    std::printf("On-node parity walls for the magnetic field (D3Q27 fluid, D3Q7 field,"
                " central moments)\n\n");

    std::printf("1. moment identities, random populations\n");
    struct M { std::uint8_t f; const char* n; };
    const M masks[] = {{SpecXm, "face -x"}, {SpecXp, "face +x"}, {SpecYm, "face -y"},
                       {SpecZp, "face +z"}, {std::uint8_t(SpecXm | SpecYp), "edge -x+y"},
                       {std::uint8_t(SpecXp | SpecZm), "edge +x-z"},
                       {std::uint8_t(SpecXm | SpecYm | SpecZm), "corner"}};
    for (const M& m : masks)
      for (int c = 0; c < 2; ++c) fails += moments(m.f, c == 0, m.n);

    std::printf("\n2. identity against the periodic box the wall mirrors\n");
    fails += check_identity(N, steps, IC::TGC, true,  "conducting,    TG-C field");
    fails += check_identity(N, steps, IC::TGI, false, "pseudo-vacuum, TG-I field");

    std::printf("\n3. no-slip box\n");
    fails += check_noslip(N, steps);

    std::printf("\n4. exact decay rate\n");
    fails += check_decay(full);

    std::printf("\n%s\n", fails ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return fails ? 1 : 0;
}
