//==============================================================================
//  Decaying MHD turbulence in a CONFINED CIRCULAR domain -- Neffaa, Bos &
//  Schneider, Phys. Fluids 20, 075104 (2008), "The decay of magnetohydrodynamic
//  turbulence in confined domains".
//
//  THIS IS A DEMONSTRATOR AND NOT A VALIDATION CASE. The reason is in the
//  reference itself and is worth stating rather than glossing: the initial
//  condition is a RANDOM-PHASE field, the results are ensemble statements about
//  decay regimes, and the sharp numbers are fitted exponents. None of that is
//  something a single run can be right or wrong against. What the paper gives
//  that IS checkable is qualitative and structural -- which invariants survive
//  the decay, which quantity grows, and whether the wall's electrical character
//  changes the outcome -- and that is what this file shows.
//
//  WHY IT IS WORTH BUILDING ANYWAY. Every MHD case in this tree before it was a
//  steady flow in a flat or smooth duct with an IMPOSED external field:
//  hartmann, shercliff, hunt, mhd_pipe. This one has no imposed field at all --
//  B is entirely self-generated and entirely decaying -- it is unsteady from
//  the first step, and its wall is curved. It is the only case here in which
//  the magnetic boundary condition acts on a field that the boundary condition
//  itself is shaping, rather than on a small induced correction to an applied
//  one.
//
//  THE SETUP. A disc of radius R inside a square box. No forcing, no applied
//  field, no through-flow: u and b are seeded from random-phase fields banded
//  around a wavenumber k0 and everything after that is decay. The fluid takes
//  regularised ON-NODE walls with the normal derived per node from the geometry
//  and NrmCorner at the staircase corners -- the pairing validation/mhd_pipe.cpp
//  established -- so the magnetic condition sits on the same plane.
//
//      Re = u0 (2R) / nu        on the initial r.m.s. velocity
//      Pr_m = nu / eta          1 by default, as in the reference
//
//  -walls CHOOSES THE PHYSICS, and it is the comparison the paper is built on:
//
//    insul  (default) INSULATING. The induced field has nowhere to go, so
//           B = 0 on the wall: Dirichlet, Dellar's moment condition.
//    cond   PERFECTLY CONDUCTING, dB/dn = 0: Neumann, the condition
//           validation/shercliff.cpp added for Hunt's duct. IT DOES NOT WORK
//           HERE, and the rest of this paragraph is why, because the failure
//           narrows what MagNeumann may be used for.
//    hunt   A CONTROL, square only: Neumann on the two y-faces and Dirichlet
//           on the two x-faces, which is Hunt's own arrangement.
//
//  AN ALL-NEUMANN CLOSED BOUNDARY IS THE THING THAT FAILS, NOT THE STENCIL.
//  `cond` goes non-finite inside sixty steps, and it does it at the GRID SCALE:
//  between step 40 and step 60 the magnetic energy rises by a factor 263, max|b|
//  goes from 0.14 to 19.2, and the magnetic integral scale L_b collapses to one
//  cell. Three explanations were tested and all three are wrong:
//
//    * "the staircase" -- on a disc, 14.6 % of Neumann nodes read another WALL
//      node one step inward and 6.1 % two steps inward, because the derived
//      axis can be nearly tangential, so part of the boundary copies from a
//      boundary that is copying. Plausible, and FALSIFIED by -shape square,
//      where the same figures are 0.9 % and 0.9 % and it fails identically.
//    * "omega_mag -> 2" -- the operator is BGK on D2Q5 and at omega = 1.95 an
//      odd-even mode is nearly undamped, which this tree has recorded for the
//      scalar twice. FALSIFIED by sweeping Pr_m: 1.949, 1.901, 1.769 and 1.586
//      all fail, and identically.
//    * "the field level is unpinned" -- true, and it IS visible (see below),
//      but far too slow: a level drift cannot produce a factor 263 in twenty
//      steps at the grid scale.
//
//  WHAT DOES HOLD IS THE CONTROL. -walls hunt -- Neumann on ONE opposing pair
//  of faces with Dirichlet on the other -- decays cleanly and tracks the
//  insulating run to two figures over two turnovers. So MagNeumann is sound and
//  it is the CLOSURE that is not: the condition needs at least one Dirichlet
//  boundary somewhere on the domain, and Hunt's duct has two. Its contract is
//  narrower than the name "perfectly conducting wall" suggests.
//
//  The level drift is real and is worth watching even in the cases that work:
//  under -walls hunt the domain-mean |<b>| climbs from 1.1e-06 to 1.5e-04 over
//  two turnovers while max|b| FALLS from 0.115 to 0.0129. The mean is only
//  partly pinned, and the |<b>| column exists so that it is visible rather than
//  hidden inside an energy that is falling anyway.
//
//  IT DRIFTS UNDER THE INSULATING WALL TOO, and that one is NOT explained by an
//  unpinned level, because b = 0 everywhere on the boundary pins it completely.
//  In the continuum the disc average is exactly zero: b = curl(a z), so
//  integral b dA = contour integral of a t ds, and b = 0 on the wall makes a
//  constant there, which kills it. The run does not reproduce that. |<b>| grows
//  from 9.7e-07 to 8.5e-04 over the first three turnovers, i.e. 0.5 % of the
//  initial peak |b|, and thereafter decays so slowly -- a factor 1.51 over
//  80000 steps, which is eta k^2 for a mode about 330 cells across, i.e. the
//  whole domain -- that by the end of the run it is 33 % of max|b|. So it is
//  harmless early and contaminates the late field, which is exactly the part
//  the animation shows.
//
//  IT IS NOT THE STAIRCASE, and -shape square says so more sharply than a
//  single number would. A square's wall nodes sit on exact flat faces, and it
//  does start slower -- 7.3e-05 against the disc's 2.9e-04 at t = 4000, and
//  3.9e-04 against 7.4e-04 at t = 8000, a factor 4 falling to 2. But it does
//  not stop lower: it PLATEAUS at 7.4e-04 against the disc's 8.5e-04, within
//  15 %. So the staircase sets how fast the drift arrives and not how big it
//  gets, which rules it out as the cause.
//
//  The remaining candidate is the module's own known limitation on div(B)
//  preservation, since a field with a small divergence need not have zero mean
//  even with b = 0 on the whole boundary. NOT ESTABLISHED; the divergence was
//  not measured here, and that is the next thing to measure rather than a
//  conclusion.
//
//  WHAT TO WATCH, AND WHY THESE AND NOT THE ENERGY ALONE. Two-dimensional MHD
//  has three ideal invariants -- total energy, cross-helicity u.b, and the
//  mean-square magnetic potential -- and the whole subject of a decay study is
//  that they do NOT decay at the same rate. Energy goes first; what survives
//  sets the state the flow relaxes into. So the run reports:
//
//      E_kin, E_mag      and their ratio, which is the Alfven ratio
//      H_c = <u.b>       an ideal invariant; its normalised form is the one
//                        the paper tracks, because an ABSOLUTE cross-helicity
//                        falling while the energies fall faster is not a
//                        statement about alignment at all
//      enstrophy, <j^2>  the dissipative quantities, one per field
//      L_u, L_b          integral scales, sqrt(E/enstrophy) and sqrt(E/<j^2>).
//                        THESE ARE THE POINT: selective decay is the statement
//                        that the scales GROW while the energies fall, and a
//                        length that grows is the one diagnostic that cannot be
//                        mistaken for the flow simply dying.
//
//  A DECAY HAS NO STEADY STATE, so nothing here converges and no residual is
//  reported. What bounds the run is not the energy but the DIVERGENCE, below.
//
//  WHAT THE RUN GIVES, AND THE LIMIT ON HOW MUCH OF IT TO BELIEVE. Over 29
//  turnovers at N = 321, Re = 1000, insulating walls: the total energy falls by
//  a factor 6000, E_kin/E_mag falls from 1 to 0.018, and the magnetic integral
//  scale grows 5.5x, from 11.0 to 60.0 cells, while the kinetic one barely
//  moves (11.4 to 16.3). Mass holds to 1.1e-07. That is the shape of selective
//  decay, and the last two columns are the reason not to quote the 5.5x bare.
//
//  div b GROWS FROM MACHINE ZERO TO PARITY WITH THE FIELD'S OWN CURL, and this
//  case is the first in the tree to measure that in a long UNFORCED run.
//  Orszag-Tang is the only other case that exercises the property and it is
//  short; there the divergence sits at truncation level and converges away at
//  order 1.65. Here, measured against rms|curl b| -- the field's own gradient
//  scale, which is the same normalisation Orszag-Tang uses:
//
//      t/T_e     0     1.7    2.1    4.9    9.7   19.4   29.1
//      div/curl  8e-16 0.097  0.145  0.354  0.625  0.976  1.04
//
//  It crosses 10 % at 1.7 turnovers and reaches ONE by twenty. No divergence
//  cleaning is implemented; that is a standing limitation of the module and
//  this is what it costs on a long run.
//
//  SO SPLIT WHAT SURVIVES IT FROM WHAT DOES NOT, rather than discounting the
//  whole result. L_b = sqrt(2 E_mag / <j^2>) has E_mag in its NUMERATOR, and a
//  spurious gradient field contributes to E_mag while contributing nothing to
//  <j^2>: so L_b is inflated once the divergence is appreciable, and the 5.5x
//  is an upper bound. Inside the window where div/curl stays under 10 % the
//  growth is 1.26x (10.96 -> 13.77), or 1.6x measured from the minimum at
//  0.35 turnovers, and THAT is the number this case can defend.
//
//  The current density is the part that is immune. j = curl b is blind to any
//  gradient field by construction, so the animation's late-time picture -- the
//  current organising into a single domain-scale structure where it began as
//  banded k = 4 noise -- is evidence that does not depend on the divergence at
//  all. The qualitative result stands on that; the number does not.
//
//  WHAT IS NOT BOUNDED. div/curl compares GRADIENTS, so it does not give the
//  ENERGY fraction carried by the spurious part: that ratio also carries
//  (L_spurious / L_b)^2, and the spurious field's own length scale was not
//  measured. If it sits at the grid scale the energy contamination is far
//  smaller than parity suggests; if it sits at the domain scale it is not.
//  Measuring it needs one more derivative, or a Helmholtz decomposition, and
//  is the next thing to do rather than a conclusion.
//
//  PENALISATION AGAINST THE SHARP WALL: A FEW PER CENT, AND THE WALL IS NOT
//  WHAT THIS PROBLEM MEASURES. -wall pen is the reference's own method and
//  -wall reg is this tree's; run from the SAME initial condition with the SAME
//  operator (BGK, forced by the constraint above) at Re = 400, N = 321, matched
//  step for step over 10.4 turnovers, penalisation differs by:
//
//      t/T_e        1.4    2.8    4.2    5.5    6.9    8.3    9.7
//      E_kin      +0.17  +0.90  +0.10  -1.39  -2.84  -3.94  -4.86  %
//      E_mag      -0.74  -1.14  -1.20  -1.42  -1.38  -1.02  -0.46  %
//      L_b        +0.23  +0.26  -0.15   0.00  +0.47  +1.12  +1.61  %
//      bulk div   +5.51  +2.21  +0.31  +0.29  +1.06  +2.03  +2.44  %
//
//  Everything magnetic agrees to under 2 % throughout. The one systematic
//  departure is the KINETIC energy, which drifts monotonically to -4.9 %: the
//  penalisation layer is a sink for u and nothing else, and its bite grows as
//  the structures grow toward the wall. That is the expected signature and it
//  is small.
//
//  WHY THIS DOES NOT CONTRADICT stokes_disc'S 4.5x. That case measures the
//  decay rate of a single mode that FILLS the disc, so it is directly
//  sensitive to where the wall is -- and penalisation puts it at R - 0.431
//  cells against a staircase's R - 0.096. Here the integral scales run 11 to
//  55 cells against R = 144.5, a fifth of the radius at best, so the decay is
//  interior-dominated and the boundary is barely involved in the numbers being
//  reported. Both statements are true of the same method: penalisation is 4.5x
//  worse at LOCATING a wall and within a few per cent on the BULK statistics of
//  a problem that is not wall-dominated. The lesson is about the diagnostic,
//  not the method -- if the wall treatment is what you want to discriminate,
//  measure it on a case whose answer depends on the wall, and stokes_disc is
//  that case.
//
//  WHAT PENALISATION COSTS AND WHAT IT BUYS HERE. It runs about 2x slower per
//  step, because F = -chi u / eps is explicit and must be refreshed from the
//  current velocity every step -- an extra macroscopic pass and a domain
//  kernel. What it buys is that u and b are penalised through the SAME chi, so
//  the two boundaries are co-located by construction and there is no half-cell
//  mismatch to price. It also removes the divergence diagnostic's wall ring
//  entirely: with no solid cells both div columns read 4.18e-16 at t = 0 rather
//  than the sharp wall's 2.57e-03, which is an independent confirmation of what
//  that floor was.
//
//  THE DIVERGENCE DIAGNOSTIC HAS A FLOOR AND THE FLOOR IS THE WALL RING, which
//  is why two of them are reported. The initial b is built from a potential by
//  central differences, for which div b cancels IDENTICALLY cell by cell, so
//  t = 0 must read machine zero and anything else is the measurement. Excluding
//  only cells with a solid neighbour reads 2.57e-03 there; excluding cells with
//  any solid within three reads 8.20e-16. The difference is the cells one step
//  inside the wall, whose stencils reach wall nodes that the Dirichlet
//  condition has overwritten. The bulk column is the honest one, and it is
//  about HALF the naive one throughout -- so a single-ring exclusion would have
//  overstated this by a factor of two at every point above.
//
//  -op bgk IS UNSTABLE AT THE DEFAULT Re, AND AN EARLIER VERSION OF THIS NOTE
//  UNDERSTATED IT. It said BGK needed "a comfortable tau" and would not get one
//  "at a small N", implying the default N = 321 was safe. It is not: measured,
//  BGK goes non-finite inside one turnover at Re = 1000 (tau = 0.5433) at the
//  full N = 321, not merely at N = 97 (tau = 0.510). It survives at Re = 600
//  (tau = 0.5722) and Re = 400 (tau = 0.6083). Central moments are the default
//  for that reason, and they run the default Re without trouble -- the same
//  ordering validation/square_cylinder.cpp measured at tau = 0.512.
//
//  This matters beyond a stability note, because -wall pen needs a FORCING
//  POLICY and MhdCentralMoments carries a hard-coded NoForcing: penalisation is
//  therefore only available under BGK, and so any penalisation run must drop to
//  Re <= 600. The wall comparison below is run at Re = 400 with BGK on both
//  sides for exactly that reason.
//
//  THE SEED IS A BRANCH, NOT A TRANSIENT, and this tree has been caught by that
//  before (CLAUDE.md, ehd_cavity: two decades of seed amplitude gave the same
//  saturated answer while a zero seed gave a different flow entirely). Here the
//  seed IS the initial condition rather than a perturbation to one, so there is
//  no question of seed-independence to check -- but -seed selects the random
//  stream, and a claim about a decay exponent that moves between streams is a
//  claim about one realisation. Run several before believing any of it.
//==============================================================================
#include "Campaign.hpp"
#include "collision/MhdBGK.hpp"
#include "collision/MhdCentralMoments.hpp"
#include "solver/MagneticSolver.hpp"

#include <type_traits>

using namespace lbm;
using namespace campaign;

using FL = D2Q9;
using ML = D2Q5;
using CollB = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations>;
using CollC = MhdCentralMoments<FL, true>;
// PENALISATION NEEDS A FORCING POLICY, AND ONLY MhdBGK HAS ONE.
// MhdCentralMoments carries a hard-coded NoForcing, so -wall pen forces BGK --
// which is why the comparison below runs BGK on BOTH sides. Comparing a
// penalised BGK run against a central-moment wall run would vary the collision
// operator and the boundary together, and this tree's first rule of measurement
// is to change one thing.
using CollP = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations, FieldGuo>;
// The central-moment operator gained a forcing term precisely so that
// penalisation would not be stuck with BGK's stability floor. It is the default
// for -wall pen, and it is what lifts the reachable Reynolds number: BGK dies
// above tau ~ 0.55, central moments do not.
using CollCP = MhdCentralMoments<FL, true, FieldGuo>;

// Both operators expose a `forcing` member but only MhdBGK names its policy, so
// the trait is written on the member rather than on a typedef.
template <class C>
inline constexpr bool penalisable =
    std::is_same_v<std::decay_t<decltype(std::declval<C&>().forcing)>, FieldGuo>;

//------------------------------------------------------------------------------
// A deterministic random stream. std::mt19937 would do, but a named 64-bit
// xorshift makes the initial condition reproducible across standard libraries
// as well as across runs, which matters when the whole case is one realisation
// of a random field.
//------------------------------------------------------------------------------
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  std::uint64_t next() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
  }
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
};

static void put_be(std::FILE* f, float v) {
  unsigned char* p = reinterpret_cast<unsigned char*>(&v);
  const unsigned char b[4] = {p[3], p[2], p[1], p[0]};
  std::fwrite(b, 1, 4, f);
}

//------------------------------------------------------------------------------
// VORTICITY AND CURRENT ARE THE PAIR. In hydrodynamic 2D turbulence the
// vorticity is the whole picture; in MHD the current sheets are where the
// magnetic energy actually goes, and a frame showing only omega shows half the
// flow. Both stencils are zeroed wherever they would reach outside the disc.
//------------------------------------------------------------------------------
template <class Solid>
static void write_vtk(const std::string& path, Index n, const Domain& d,
                      const std::vector<double>& ux, const std::vector<double>& uy,
                      const std::vector<double>& bx, const std::vector<double>& by,
                      Solid solid) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { std::printf("  cannot open %s\n", path.c_str()); return; }
  std::fprintf(f, "# vtk DataFile Version 3.0\nmhd decay\nBINARY\n"
                  "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n"
                  "ORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA %lld\n",
               int(n), int(n), (long long)(std::size_t(n) * std::size_t(n)));
  auto at = [&](Index x, Index y) { return std::size_t(d.id(x, y, 0)); };
  auto ok = [&](Index x, Index y) {
    return x > 0 && y > 0 && x < n - 1 && y < n - 1 && !solid(x, y) &&
           !solid(x + 1, y) && !solid(x - 1, y) && !solid(x, y + 1) && !solid(x, y - 1);
  };
  std::fprintf(f, "SCALARS vorticity float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < n; ++y)
    for (Index x = 0; x < n; ++x)
      put_be(f, ok(x, y) ? float(0.5 * (uy[at(x + 1, y)] - uy[at(x - 1, y)]) -
                                 0.5 * (ux[at(x, y + 1)] - ux[at(x, y - 1)]))
                         : 0.0f);
  std::fprintf(f, "\nSCALARS current float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < n; ++y)
    for (Index x = 0; x < n; ++x)
      put_be(f, ok(x, y) ? float(0.5 * (by[at(x + 1, y)] - by[at(x - 1, y)]) -
                                 0.5 * (bx[at(x, y + 1)] - bx[at(x, y - 1)]))
                         : 0.0f);
  std::fprintf(f, "\nSCALARS bmag float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < n; ++y)
    for (Index x = 0; x < n; ++x) {
      const double p = bx[at(x, y)], q = by[at(x, y)];
      put_be(f, solid(x, y) ? 0.0f : float(std::sqrt(p * p + q * q)));
    }
  std::fprintf(f, "\nSCALARS solid float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < n; ++y)
    for (Index x = 0; x < n; ++x) put_be(f, solid(x, y) ? 1.0f : 0.0f);
  std::fprintf(f, "\nVECTORS velocity float\n");
  for (Index y = 0; y < n; ++y)
    for (Index x = 0; x < n; ++x) {
      const bool sd = solid(x, y);
      put_be(f, sd ? 0.0f : float(ux[at(x, y)]));
      put_be(f, sd ? 0.0f : float(uy[at(x, y)]));
      put_be(f, 0.0f);
    }
  std::fclose(f);
}

struct Opts {
  // The defaults ARE the configuration the reported runs used. A default that
  // nobody ran is a trap: this case's relaxation time follows the resolution,
  // since nu = u0 (2R) / Re and R scales with N, so a smaller default box is
  // not merely coarser -- it sits closer to the stability floor.
  Index N = 321;
  double rfac = 0.45, u0 = 0.05, Re = 1000.0, prm = 1.0, alf = 1.0;
  int k0 = 4, kw = 1;
  std::uint64_t seed = 12345;
  std::size_t steps = 170000, probe = 2000, vtkevery = 0, binevery = 0;
  double eps = 2.0, smooth = 1.0, epsm = 2.0;
  int regime = 0;                 // 0 = this file's own IC; 1..4 = Neffaa Table I
  std::string vtkdir, bindir, walls = "insul", shape = "disc", wall = "reg";
};

template <class C>
static int run(const Opts& o) {
  const Index N = o.N;
  const double R = o.rfac * double(N);
  const double c = 0.5 * double(N) - 0.5;
  auto rad = [&](Index x, Index y) {
    const double dx = double(x) - c, dy = double(y) - c;
    return std::sqrt(dx * dx + dy * dy);
  };
  // -shape square IS A CONTROL, not a second physics case. See the conducting
  // wall's entry in the banner: a square domain's wall nodes all sit on flat
  // faces, so the Neumann stencil reads genuine interior nodes exactly as it
  // does in Hunt's duct, and running the same initial condition in both shapes
  // is what separates "the Neumann condition is broken" from "the Neumann
  // condition cannot be built on a staircase".
  const bool square = (o.shape == "square");
  const Index m0i = Index(std::lround(c - R)), m1i = Index(std::lround(c + R));
  auto solid = [&](Index x, Index y) {
    if (square) return x < m0i || x > m1i || y < m0i || y > m1i;
    return rad(x, y) > R;
  };

  //--------------------------------------------------------------------------
  // THE INITIAL CONDITION, BUILT FROM POTENTIALS AND NOT FROM THE FIELDS.
  //
  // u = (d psi/dy, -d psi/dx) and b = (d a/dy, -d a/dx) with CENTRAL
  // differences are divergence-free at the DISCRETE level, not merely in the
  // continuum, because the mixed second differences commute exactly. Seeding u
  // and b directly from random Fourier modes and projecting afterwards would
  // leave a divergence at round-off at best, and this tree already carries a
  // known limitation about div B preservation -- starting with a violation of
  // it would make that limitation unmeasurable here.
  //
  // THE TAPER (1 - (r/R)^2)^2 vanishes together with its first derivative at
  // r = R, so BOTH potentials and BOTH fields go to zero at the wall. For the
  // insulating wall that is exactly the boundary condition; for the conducting
  // one it is not (dB/dn = 0 is), so that case starts from a field the wall
  // will immediately relax -- stated rather than hidden, since the first few
  // turnover times of the `cond` run are that relaxation and not decay.
  //--------------------------------------------------------------------------
  std::vector<double> hux(std::size_t(N) * N, 0.0), huy(hux), hbx(hux), hby(hux);
  {
    Rng rng(o.seed);
    struct Mode { double kx, ky, pp, pa, ap, aa; };
    std::vector<Mode> modes;
    // NEFFAA'S SPECTRUM, Eq. (6): E(k) ~ k / [g + (k/k0)]^4 with g = 0.98 and
    // k0 = (3/4) sqrt(2) pi, which is k^-3 at large k. The amplitude PER MODE
    // follows from it rather than being the spectrum itself: a shell of radius
    // k in two dimensions holds O(k) modes, so |u_k|^2 ~ E(k)/k, and since u is
    // the curl of psi, |psi_k| = |u_k| / k. Hence
    //     |psi_k| ~ sqrt(E(k)/k) / k = 1 / ( k [g + (k/k0)]^2 ).
    // Getting that chain wrong is silent: the field still looks like turbulence
    // and simply has the wrong spectrum.
    const double g_ne = 0.98, k0_ne = 0.75 * std::sqrt(2.0) * M_PI;
    const int kmax = o.regime ? 24 : o.k0 + o.kw;
    for (int kx = -kmax; kx <= kmax; ++kx)
      for (int ky = 0; ky <= kmax; ++ky) {
        if (kx == 0 && ky == 0) continue;
        const double k = std::sqrt(double(kx * kx + ky * ky));
        double amp;
        if (o.regime) {
          if (k > double(kmax)) continue;
          const double d = g_ne + k / k0_ne;
          amp = 1.0 / (k * d * d);
        } else {
          if (k < double(o.k0 - o.kw) || k > double(o.k0 + o.kw)) continue;
          amp = 1.0;
        }
        modes.push_back(Mode{double(kx), double(ky),
                             2.0 * M_PI * rng.uniform(), 2.0 * M_PI * rng.uniform(),
                             amp, amp});
      }
    std::vector<double> psi(std::size_t(N) * N, 0.0), apot(psi);
    const double L = double(N);
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        double p = 0, a = 0;
        for (const Mode& m : modes) {
          const double ph = 2.0 * M_PI * (m.kx * double(x) + m.ky * double(y)) / L;
          p += m.ap * std::sin(ph + m.pp);
          a += m.aa * std::sin(ph + m.pa);
        }
        const double r = rad(x, y);
        const double tt = (r < R) ? (1.0 - (r / R) * (r / R)) : 0.0;
        const double tap = tt * tt;
        psi[std::size_t(y) * N + x]  = p * tap;
        apot[std::size_t(y) * N + x] = a * tap;
      }
    auto at = [&](Index x, Index y) { return std::size_t(y) * N + x; };
    for (Index y = 1; y < N - 1; ++y)
      for (Index x = 1; x < N - 1; ++x) {
        hux[at(x, y)] =  0.5 * (psi[at(x, y + 1)] - psi[at(x, y - 1)]);
        huy[at(x, y)] = -0.5 * (psi[at(x + 1, y)] - psi[at(x - 1, y)]);
        hbx[at(x, y)] =  0.5 * (apot[at(x, y + 1)] - apot[at(x, y - 1)]);
        hby[at(x, y)] = -0.5 * (apot[at(x + 1, y)] - apot[at(x - 1, y)]);
      }
    // Normalise to the requested r.m.s. velocity and Alfven ratio. Matching the
    // R.M.S. and not the maximum: a random field's maximum is one lucky cell
    // and is not reproducible between streams, so normalising by it would make
    // -seed change the Reynolds number as well as the realisation.
    double su = 0, sb = 0; std::size_t cnt = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        if (solid(x, y)) continue;
        su += hux[at(x, y)] * hux[at(x, y)] + huy[at(x, y)] * huy[at(x, y)];
        sb += hbx[at(x, y)] * hbx[at(x, y)] + hby[at(x, y)] * hby[at(x, y)];
        ++cnt;
      }
    const double urms = std::sqrt(su / double(cnt)), brms = std::sqrt(sb / double(cnt));
    const double fu = o.u0 / std::max(urms, 1e-300);
    // alf = E_kin / E_mag at t = 0.
    const double fb = (o.u0 / std::sqrt(std::max(o.alf, 1e-300))) / std::max(brms, 1e-300);
    for (std::size_t k = 0; k < hux.size(); ++k) {
      hux[k] *= fu; huy[k] *= fu; hbx[k] *= fb; hby[k] *= fb;
    }
  }

  const Real nu  = Real(o.u0 * 2.0 * R / o.Re);
  const Real eta = Real(double(nu) / o.prm);
  Domain d(N, N, 1, false, false, false);
  MagneticBGK<ML> mc; mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
  C fc;
  fc.omega = C::omega_from_viscosity(nu);
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, C> fl(d, fc);

  // A wall node is a fluid node with at least one axis neighbour outside; more
  // than one axis leaving is a staircase corner, where a single normal cannot
  // describe the unknown set and NrmCorner builds it geometrically.
  auto wallcode = [&](Index x, Index y) -> std::uint8_t {
    if (solid(x, y)) return NrmNone;
    int k = 0; std::uint8_t only = NrmNone;
    if (solid(x - 1, y)) { ++k; only = NrmXm; }
    if (solid(x + 1, y)) { ++k; only = NrmXp; }
    if (solid(x, y - 1)) { ++k; only = NrmYm; }
    if (solid(x, y + 1)) { ++k; only = NrmYp; }
    if (k == 0) return NrmNone;
    return (k == 1) ? only : std::uint8_t(NrmCorner);
  };
  //--------------------------------------------------------------------------
  // -wall pen: VOLUME PENALISATION, the reference's own method, and it replaces
  // BOTH boundaries at once.
  //
  // There is no solid cell, no wall node and no magnetic wall condition. A
  // smooth indicator chi = (1 + tanh((r - R)/w))/2 marks the exterior, the
  // fluid gains F = -chi u / eps through FieldGuo and the induction equation
  // gains -chi B / eps_m through MagneticSolver::set_penalisation. The disc
  // boundary is then a LAYER rather than a surface.
  //
  // ONE THING IT GETS RIGHT THAT THE SHARP PAIRING CANNOT. u and b are
  // penalised through the SAME chi, so the two boundaries are co-located by
  // construction -- there is no half-cell mismatch to price, because there is
  // no surface for them to disagree about. That is a real advantage and it is
  // presumably why the reference does it this way.
  //
  // WHAT IT COSTS is measured in validation/stokes_disc.cpp on the exact
  // azimuthal decay mode: the staircase behaves like a circle of radius
  // R - 0.096 cells, penalisation like R - 0.431, and both offsets are FIXED to
  // within 1 % over a 4x refinement -- so penalisation is not a different
  // convergence class, it is the same first-order fixed offset 4.5x larger.
  //--------------------------------------------------------------------------
  const bool pen = (o.wall == "pen");
  View1D<Real> pchi, pfx, pfy, pnx, pny, psx, psy;
  if (pen) {
    pchi = View1D<Real>("chi", d.n_padded);
    auto h = Kokkos::create_mirror_view(pchi);
    for (Index n = 0; n < d.n_padded; ++n) h(n) = Real(0);
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x)
        h(d.id(x, y, 0)) =
            Real(0.5 * (1.0 + std::tanh((rad(x, y) - R) / o.smooth)));
    Kokkos::deep_copy(pchi, h);
    if constexpr (penalisable<C>) {
      pfx = View1D<Real>("pfx", d.n_padded);
      pfy = View1D<Real>("pfy", d.n_padded);
      fl.collision().forcing.Ex = pfx;
      fl.collision().forcing.Ey = pfy;
    }
    // Outward unit normal of the disc, precomputed. The magnetic penalisation
    // needs it every step and it never changes.
    pnx = View1D<Real>("pnx", d.n_padded);
    pny = View1D<Real>("pny", d.n_padded);
    psx = View1D<Real>("psx", d.n_padded);
    psy = View1D<Real>("psy", d.n_padded);
    auto hnx = Kokkos::create_mirror_view(pnx), hny = Kokkos::create_mirror_view(pny);
    for (Index n = 0; n < d.n_padded; ++n) { hnx(n) = Real(0); hny(n) = Real(0); }
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const double dx = double(x) - c, dy = double(y) - c;
        const double r = std::max(std::sqrt(dx * dx + dy * dy), 1e-12);
        hnx(d.id(x, y, 0)) = Real(dx / r); hny(d.id(x, y, 0)) = Real(dy / r);
      }
    Kokkos::deep_copy(pnx, hnx); Kokkos::deep_copy(pny, hny);
    mag.set_source(psx, psy, View1D<Real>());
  }

  fl.set_geometry([&](Index x, Index y, Index) -> CellType {
    if (pen) return Fluid;                  // no solid at all under penalisation
    return solid(x, y) ? Solid : Fluid;
  });
  using WS = typename decltype(fl)::WallSpec;
  if (!pen)
    fl.set_regularized_walls([&](Index x, Index y, Index) -> WS {
      const std::uint8_t w = wallcode(x, y);
      if (w == NrmNone) return WS{};
      return WS{w, Real(0), Real(0), Real(0)};
    });
  // The FD corner stress knows nothing about the Lorentz force -- shercliff.cpp
  // measured it leaking mass linearly at a duct edge. A staircase circle is
  // nothing but corners.
  fl.set_fd_corners(false);

  const bool cond = (o.walls == "cond");
  // -walls hunt IS THE SECOND CONTROL. `cond` puts Neumann on the WHOLE closed
  // boundary; Hunt's duct -- the case that validated MagNeumann -- puts it on
  // one opposing PAIR of walls and Dirichlet on the other. If the pair is
  // stable and the closure is not, then what fails is the all-Neumann closure
  // and not the stencil. Square only: it names faces.
  const bool hunt = (o.walls == "hunt");
  if (hunt && !square) {
    // A face-named condition on a staircase names nothing. Refuse rather than
    // resolve it: a disc has no "x-faces", and silently treating the nodes with
    // a horizontal neighbour outside as one would produce a boundary split
    // along a diagonal that nobody chose.
    std::printf("  -walls hunt names opposing FACES and needs -shape square; "
                "a disc has none. Refusing.\n");
    return 1;
  }
  if (!pen) mag.set_geometry([&](Index x, Index y, Index) { return solid(x, y); });
  using WB = typename decltype(mag)::WallB;
  if (!pen)
  mag.set_moment_walls([&](Index x, Index y, Index) -> WB {
    if (wallcode(x, y) == NrmNone) return WB{};
    if (hunt) {
      // Neumann on the two y-faces, Dirichlet on the two x-faces. A corner
      // belongs to the Dirichlet pair, as Hunt's own side walls do.
      const bool xface = solid(x - 1, y) || solid(x + 1, y);
      if (!xface) { WB w; w.is_wall = true; w.neumann = true; return w; }
      return WB{true, Real(0), Real(0), Real(0)};
    }
    if (cond) { WB w; w.is_wall = true; w.neumann = true; return w; }  // face derived
    return WB{true, Real(0), Real(0), Real(0)};                        // insulating
  });

  View1D<Real> vx("ic_ux", d.n_padded), vy("ic_uy", d.n_padded),
               wx("ic_bx", d.n_padded), wy("ic_by", d.n_padded);
  {
    auto h1 = Kokkos::create_mirror_view(vx), h2 = Kokkos::create_mirror_view(vy),
         h3 = Kokkos::create_mirror_view(wx), h4 = Kokkos::create_mirror_view(wy);
    for (Index n = 0; n < d.n_padded; ++n) { h1(n) = h2(n) = h3(n) = h4(n) = Real(0); }
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y, 0);
        if (solid(x, y)) continue;
        h1(n) = Real(hux[std::size_t(y) * N + x]);
        h2(n) = Real(huy[std::size_t(y) * N + x]);
        h3(n) = Real(hbx[std::size_t(y) * N + x]);
        h4(n) = Real(hby[std::size_t(y) * N + x]);
      }
    Kokkos::deep_copy(vx, h1); Kokkos::deep_copy(vy, h2);
    Kokkos::deep_copy(wx, h3); Kokkos::deep_copy(wy, h4);
  }
  mag.initialize_field(KOKKOS_LAMBDA(Index n) {
    Kokkos::Array<Real, 3> b; b[0] = wx(n); b[1] = wy(n); b[2] = Real(0);
    return b;
  });
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    return FlowState{Real(1), vx(n), vy(n), Real(0)};
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  const double T_e = 2.0 * R / o.u0;     // one large-eddy turnover
  std::size_t nfluid = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) if (!solid(x, y)) ++nfluid;
  std::printf("  N %d   %s, half-width %.1f (%.0f%% of the box)   %zu fluid nodes\n",
              int(N), square ? "SQUARE" : "disc", R, 100.0 * o.rfac, nfluid);
  // HOW MUCH OF THE WALL IS SELF-REFERENTIAL. A Neumann node reads its field
  // one and two steps inward along a derived axis; on a staircase that axis can
  // be nearly tangential and the nodes it reads can themselves be wall nodes,
  // so a fraction of the boundary is copying from a boundary that is copying.
  // Counted, not assumed -- and zero on a square, which is the control.
  if (o.walls == "cond" || o.walls == "hunt") {
    const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::size_t tot = 0, ch1 = 0, ch2 = 0;
    auto iswall = [&](Index x, Index y) {
      return !solid(x, y) && (solid(x - 1, y) || solid(x + 1, y) ||
                              solid(x, y - 1) || solid(x, y + 1));
    };
    for (Index y = 1; y < N - 1; ++y)
      for (Index x = 1; x < N - 1; ++x) {
        if (!iswall(x, y)) continue;
        ++tot;
        for (int k = 0; k < 4; ++k) {
          const Index qx = x + dirs[k][0], qy = y + dirs[k][1];
          if (!solid(qx, qy)) continue;
          const Index a1x = x - dirs[k][0], a1y = y - dirs[k][1];
          const Index a2x = x - 2 * dirs[k][0], a2y = y - 2 * dirs[k][1];
          if (solid(a1x, a1y) || solid(a2x, a2y)) continue;
          if (iswall(a1x, a1y)) ++ch1;
          if (iswall(a2x, a2y)) ++ch2;
          break;
        }
      }
    std::printf("  Neumann stencil: %zu wall nodes, %zu (%.1f %%) read a WALL "
                "node one step in, %zu (%.1f %%) two steps in\n",
                tot, ch1, tot ? 100.0 * double(ch1) / double(tot) : 0.0,
                ch2, tot ? 100.0 * double(ch2) / double(tot) : 0.0);
  }
  // REPORT THE DISPATCHED TYPE, NOT THE REQUESTED ONE. The banner line above
  // is printed from the -op string, which is what was ASKED for; this one is
  // C::name and C's actual forcing policy, which is what the template was
  // instantiated with. They can differ -- -wall pen overrides -op -- and a log
  // that records the request rather than the fact cannot answer "what did this
  // run actually use?" afterwards.
  std::printf("  OPERATOR (dispatched) %s   fluid forcing %s   magnetic source %s\n",
              C::name, penalisable<C> ? "FieldGuo (penalisation ACTIVE)" : "none",
              (o.wall == "pen") ? "penalised B.n" : "none");
  std::printf("  Re %.0f   Pr_m %.2f   nu %.5f (tau %.4f)   eta %.5f "
              "(omega_mag %.4f)\n", o.Re, o.prm, double(nu),
              1.0 / double(fc.omega), double(eta),
              double(MagneticBGK<ML>::omega_from_resistivity(eta)));
  if (pen)
    std::printf("  wall PENALISATION   eps %.2f (fluid)  eps_m %.2f (magnetic)  "
                "chi width %.2f cells   no solid, no wall condition\n",
                o.eps, o.epsm, o.smooth);
  if (o.regime)
    std::printf("  NEFFAA REGIME %d (Table I)   spectrum E(k) ~ k/[0.98 + "
                "k/((3/4)sqrt(2)pi)]^4   r = (19/20)pi in a 2pi box\n", o.regime);
  std::printf("  u_rms %.4f (Ma %.4f)   %s   E_kin/E_mag %.3f   "
              "walls %s   seed %llu\n", o.u0, o.u0 / 0.5773502692,
              o.regime ? "k^-3 spectrum" : "banded k0",
              o.alf, pen  ? "penalised, B.n = 0 (perfect conductor)"
                    : cond ? "perfectly conducting (Neumann, all faces)"
                    : hunt ? "Neumann on y-faces, Dirichlet on x-faces"
                           : "insulating (Dirichlet)",
              (unsigned long long)o.seed);
  std::printf("  one turnover 2R/u0 = %.0f steps; running %zu (%.1f turnovers)\n\n",
              T_e, o.steps, double(o.steps) / T_e);
  std::printf("  %8s %8s %11s %11s %9s %11s %11s %7s %7s %8s %11s %11s %9s %9s %10s\n",
              "step", "t/T_e", "E_kin", "E_mag", "E_k/E_m", "enstrophy",
              "<j^2>", "L_u", "L_b", "H_c", "|<b>|", "max|b|", "dv/cl", "bulk",
              "Bn/B|wall");
  std::printf("  %s\n", std::string(164, '-').c_str());

  // A DEEP-INTERIOR MASK, because the divergence diagnostic below has a floor
  // and the floor is the wall ring. The initial b is built from a potential by
  // central differences, for which div b is EXACTLY zero cell by cell -- the
  // mixed second differences cancel identically -- so any nonzero reading at
  // t = 0 is the measurement and not the field. It reads 2.57e-03 of the curl,
  // and the reason is that the wall nodes' b is overwritten by the Dirichlet
  // condition while the cells ONE step inside them still read those nodes in
  // their stencil. Excluding a single ring is not enough. So the run reports
  // two divergences: one over every cell with four fluid neighbours, and one
  // over cells with no solid within three, which is the bulk. If the two agree
  // the divergence is a bulk property; if only the first grows it is a wall
  // ring, and the distinction decides whether a late-time field is physical.
  std::vector<std::uint8_t> deep(std::size_t(N) * N, 0);
  for (Index y = 3; y < N - 3; ++y)
    for (Index x = 3; x < N - 3; ++x) {
      bool ok = true;
      for (Index dy = -3; dy <= 3 && ok; ++dy)
        for (Index dx = -3; dx <= 3 && ok; ++dx)
          if (solid(x + dx, y + dy)) ok = false;
      deep[std::size_t(y) * N + x] = ok ? 1 : 0;
    }
  const Real m0 = fl.total_mass();
  double E0 = 0;
  std::size_t vframe = 0, bframe = 0;
  int status = 0;
  for (std::size_t t = 0; t <= o.steps; ++t) {
    // THE DUMP AND THE DIAGNOSTIC ARE ON SEPARATE CLOCKS, and the first version
    // nested one inside the other. A frame then needed t to be a multiple of
    // BOTH intervals, so -probe 5000 -binevery 3050 dumped on multiples of
    // LCM(5000, 3050) = 305000 -- the first frame and the last, out of a
    // hundred asked for, with no error and a perfectly healthy log. Anything
    // that silently produces 1 % of what was requested is worth the extra
    // predicate.
    const bool want_probe = (t % o.probe == 0);
    const bool want_bin   = o.binevery && (t % o.binevery == 0);
    const bool want_vtk   = o.vtkevery && (t % o.vtkevery == 0);
    if (want_probe || want_bin || want_vtk) {
      fl.compute_macroscopic(); mag.compute_field();
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
      auto hq = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
      // TWO COUNTS, NOT ONE. The energies are averaged over every fluid cell,
      // including the wall ring where u = b = 0 and the average is meant to
      // include them. The derivative stencils cannot run on that ring, so the
      // enstrophy and <j^2> sums cover fewer cells -- and dividing those by the
      // fluid count instead of their own puts a constant bias into BOTH
      // integral scales (1.4 % of the cells here, so 0.7 % in a length). It
      // would cancel out of the growth ratio that this case is about, which is
      // exactly why it would never have been noticed.
      double ek = 0, em = 0, en = 0, jj = 0, hc = 0, dvb = 0;
      double dvb2 = 0, jj2 = 0;
      std::size_t cnt = 0, dcnt = 0, kcnt = 0;
      // THE DOMAIN-MEAN FIELD, because a Neumann condition does not constrain
      // it. MagneticSolver.hpp says so where MagOutXp is documented ("a pure
      // Neumann condition does not constrain the field level at all"), and a
      // CLOSED boundary that is Neumann everywhere has no other boundary to
      // pin it -- so the constant mode is free. If that is what kills the
      // conducting run, the mean will run away while the fluctuation energy
      // does not, and these two columns separate the two.
      double mbx = 0, mby = 0, bmax = 0;
      // IS THE BOUNDARY CONDITION ACTUALLY BEING IMPOSED? The penalised wall
      // claims B.n = 0, and a differential test ("delete it and see if the
      // answer moves") shows only that the term does SOMETHING. This is the
      // direct statement: the largest normal component of B on the boundary
      // ring, as a fraction of the largest |B| there. It should be small under
      // -wall pen and is not constrained at all without it.
      double bn_max = 0, bt_max = 0;
      bool fin = true;
      for (Index y = 0; y < N; ++y)
        for (Index x = 0; x < N; ++x) {
          if (solid(x, y)) continue;
          const Index n = d.id(x, y, 0);
          const double a = double(hu(n)), b = double(hv(n));
          const double p = double(hp(n)), q = double(hq(n));
          if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(p) ||
              !std::isfinite(q)) { fin = false; continue; }
          ek += 0.5 * (a * a + b * b);
          em += 0.5 * (p * p + q * q);
          hc += a * p + b * q;
          mbx += p; mby += q;
          bmax = std::max(bmax, std::sqrt(p * p + q * q));
          ++cnt;
          {
            const double rr = rad(x, y);
            if (rr > R - 2.0 && rr < R + 2.0) {
              const double dx = (double(x) - c) / std::max(rr, 1e-12);
              const double dy = (double(y) - c) / std::max(rr, 1e-12);
              bn_max = std::max(bn_max, std::abs(p * dx + q * dy));
              bt_max = std::max(bt_max, std::sqrt(p * p + q * q));
            }
          }
          // The derivative stencils need a full interior neighbourhood; the
          // wall ring is excluded from BOTH so the two lengths below are
          // measured over the same set of cells.
          if (x == 0 || y == 0 || x == N - 1 || y == N - 1) continue;
          if (solid(x + 1, y) || solid(x - 1, y) || solid(x, y + 1) || solid(x, y - 1))
            continue;
          const double w = 0.5 * (double(hv(d.id(x + 1, y, 0))) - double(hv(d.id(x - 1, y, 0)))) -
                           0.5 * (double(hu(d.id(x, y + 1, 0))) - double(hu(d.id(x, y - 1, 0))));
          const double j = 0.5 * (double(hq(d.id(x + 1, y, 0))) - double(hq(d.id(x - 1, y, 0)))) -
                           0.5 * (double(hp(d.id(x, y + 1, 0))) - double(hp(d.id(x, y - 1, 0))));
          // div b ON THE SAME STENCIL AS THE CURL, which is what makes the
          // ratio below meaningful: both are central differences over the same
          // neighbours, so div/|curl| compares the spurious part of the field's
          // gradient against its real part rather than against a scale pulled
          // from somewhere else. It is the measurement the |<b>| drift needs --
          // a field with a small divergence need NOT have zero mean even with
          // b = 0 on the whole boundary, which is the one explanation of that
          // drift still standing after the staircase was ruled out.
          const double dv = 0.5 * (double(hp(d.id(x + 1, y, 0))) - double(hp(d.id(x - 1, y, 0)))) +
                            0.5 * (double(hq(d.id(x, y + 1, 0))) - double(hq(d.id(x, y - 1, 0))));
          if (std::isfinite(w)) en += w * w;
          if (std::isfinite(j)) jj += j * j;
          if (std::isfinite(dv)) dvb += dv * dv;
          ++dcnt;
          if (deep[std::size_t(y) * N + x]) {
            if (std::isfinite(dv)) dvb2 += dv * dv;
            if (std::isfinite(j))  jj2  += j * j;
            ++kcnt;
          }
        }
      const double inv = cnt ? 1.0 / double(cnt) : 0.0;
      const double dinv = dcnt ? 1.0 / double(dcnt) : 0.0;
      ek *= inv; em *= inv; hc *= inv; mbx *= inv; mby *= inv;
      en *= dinv; jj *= dinv; dvb *= dinv;
      const double kinv = kcnt ? 1.0 / double(kcnt) : 0.0;
      dvb2 *= kinv; jj2 *= kinv;
      if (t == 0) E0 = ek + em;
      // SELECTIVE DECAY IS A STATEMENT ABOUT LENGTHS, so report lengths. A
      // falling energy alone is consistent with the flow simply dying at every
      // scale at once; a length that GROWS while the energy falls is not.
      const double Lu = en > 0 ? std::sqrt(2.0 * ek / en) : 0.0;
      const double Lb = jj > 0 ? std::sqrt(2.0 * em / jj) : 0.0;
      // NORMALISED cross-helicity. The absolute one falls because both energies
      // do; what the alignment does is only visible after dividing it out.
      const double hcn = (ek + em) > 0 ? hc / (2.0 * std::sqrt(ek * em) + 1e-300) : 0.0;
      if (want_probe) {
        std::printf("  %8zu %8.2f %11.4e %11.4e %9.4f %11.4e %11.4e %7.2f %7.2f %+8.4f "
                    "%11.4e %11.4e %9.2e %9.2e %10.3e%s\n",
                    t, double(t) / T_e, ek, em, em > 0 ? ek / em : 0.0, en, jj, Lu, Lb,
                    hcn, std::sqrt(mbx * mbx + mby * mby), bmax,
                    jj > 0 ? std::sqrt(dvb / jj) : 0.0,
                    jj2 > 0 ? std::sqrt(dvb2 / jj2) : 0.0,
                    bt_max > 0 ? bn_max / bt_max : 0.0,
                    fin ? "" : "   NON-FINITE");
        std::fflush(stdout);
      }
      if (!fin) { status = 1; break; }
      // A COMPACT FRAME for the animation, in doc/fig/mkpng.py's format: two
      // little-endian int32 then N*N float32. Solid cells and the cells whose
      // stencil reaches one carry a NaN so the renderer paints the wall instead
      // of a ring of false shear.
      if (want_bin) {
        auto emit = [&](const char* tag, bool current) {
          char nm[512];
          // NAME THE FILE AFTER WHAT PRODUCED IT. This used o.walls -- the
          // magnetic wall option -- which under -wall pen is not used at all,
          // so a penalised run wrote frames labelled "insul". A dump that
          // misstates its own provenance is worse than an unlabelled one.
          std::snprintf(nm, sizeof nm, "%s/decay_%s%s_%s%04zu.bin", o.bindir.c_str(),
                        (o.wall == "pen") ? "pen" : o.walls.c_str(),
                        o.regime ? (o.regime == 1 ? "1" : o.regime == 2 ? "2"
                                  : o.regime == 3 ? "3" : "4") : "",
                        tag, bframe);
          std::FILE* f = std::fopen(nm, "wb");
          if (!f) return;
          const std::int32_t w = std::int32_t(N);
          std::fwrite(&w, 4, 1, f); std::fwrite(&w, 4, 1, f);
          for (Index y = 0; y < N; ++y)
            for (Index x = 0; x < N; ++x) {
              float v;
              if (x == 0 || y == 0 || x == N - 1 || y == N - 1 || solid(x, y) ||
                  solid(x + 1, y) || solid(x - 1, y) || solid(x, y + 1) ||
                  solid(x, y - 1))
                v = std::nanf("");
              else if (current)
                v = float(0.5 * (double(hq(d.id(x + 1, y, 0))) - double(hq(d.id(x - 1, y, 0)))) -
                          0.5 * (double(hp(d.id(x, y + 1, 0))) - double(hp(d.id(x, y - 1, 0)))));
              else
                v = float(0.5 * (double(hv(d.id(x + 1, y, 0))) - double(hv(d.id(x - 1, y, 0)))) -
                          0.5 * (double(hu(d.id(x, y + 1, 0))) - double(hu(d.id(x, y - 1, 0)))));
              std::fwrite(&v, 4, 1, f);
            }
          std::fclose(f);
        };
        emit("w", false); emit("j", true);
        ++bframe;
      }
      if (want_vtk) {
        std::vector<double> a1(std::size_t(d.n_padded)), a2(a1), a3(a1), a4(a1);
        for (Index n = 0; n < d.n_padded; ++n) {
          a1[n] = double(hu(n)); a2[n] = double(hv(n));
          a3[n] = double(hp(n)); a4[n] = double(hq(n));
        }
        char nm[512];
        std::snprintf(nm, sizeof nm, "%s/decay_%s_%04zu.vtk", o.vtkdir.c_str(),
                      (o.wall == "pen") ? "pen" : o.walls.c_str(), vframe++);
        write_vtk(nm, N, d, a1, a2, a3, a4, solid);
      }
    }
    if (t < o.steps) {
      // REFRESH THE PENALISATION FORCE FROM THE CURRENT VELOCITY, every step.
      // F = -chi u / eps is explicit, so it must see u(t) and not u at the last
      // diagnostic -- a force refreshed on the probe interval would be a
      // thousand-step-old velocity and would not enforce anything.
      if (pen) {
        fl.compute_macroscopic(); mag.compute_field();
        auto ux = fl.ux(); auto uy = fl.uy();
        auto bx = mag.Bx(); auto by = mag.By();
        auto fx = pfx; auto fy = pfy; auto ch = pchi;
        auto nx2 = pnx; auto ny2 = pny; auto sx = psx; auto sy = psy;
        const Real ie = Real(1.0 / o.eps), iem = Real(1.0 / o.epsm);
        const bool fluidpen = penalisable<C>;
        Kokkos::parallel_for("pen", Kokkos::RangePolicy<ExecSpace>(0, d.n_padded),
          KOKKOS_LAMBDA(Index n) {
            if (fluidpen && fx.data()) {
              fx(n) = -ch(n) * ux(n) * ie;
              fy(n) = -ch(n) * uy(n) * ie;
            }
            // B.n = 0, NOT B = 0: damp the NORMAL component only, which is the
            // perfect conductor of the reference's Eq. (2) with B_0 = B_par.
            const Real bn = bx(n) * nx2(n) + by(n) * ny2(n);
            sx(n) = -ch(n) * bn * nx2(n) * iem;
            sy(n) = -ch(n) * bn * ny2(n) * iem;
          });
      }
      mag.compute_field(); fl.step(true); mag.step(true);
    }
  }
  const double mrel = double(fl.total_mass()) / double(m0) - 1.0;
  std::printf("\n  initial total energy %.4e;  dmass %+.3e\n", E0, mrel);
  if (vframe) std::printf("  wrote %zu VTK frames to %s\n", vframe, o.vtkdir.c_str());
  if (bframe) std::printf("  wrote %zu binary frames to %s\n", bframe, o.bindir.c_str());
  return status;
}

//------------------------------------------------------------------------------
// -fcheck: THE FORCE AGAINST AN IDENTITY IT CANNOT GET WRONG.
//
// MhdCentralMoments gained a forcing term so that -wall pen could use it, and a
// new force in a collision operator that has never reproduced a known force is
// not usable. The identity is exact and needs no profile, no viscosity and no
// boundary: in a fully PERIODIC box a uniform body force F is the only thing
// that changes momentum, so after T steps
//
//     rho * u  =  F * (T + 1/2)
//
// exactly, for every T. THE HALF STEP IS THE POINT AND IT IS NOT A FUDGE. The
// stored populations carry F*T after T collisions, and `macroscopic` reports
// the Guo-shifted velocity, which adds F/(2 rho) on top -- so the reported
// momentum leads the streamed momentum by exactly half a force step. Writing
// F*T here instead reads as a 1e-3 relative error at T = 500 and would be
// blamed on the new operator; it is the convention, and this note exists
// because that is precisely the mistake this check caught on its first run.
//
// What the check actually establishes is the agreement of the two operators:
// MhdBGK's Guo term is long validated, the central-moment one is new, and they
// must produce the same momentum to every digit.
//------------------------------------------------------------------------------
template <class C>
static int force_check(const char* tag, int& status) {
  const Index N = 32;
  const Real F = Real(1e-6), nu = Real(0.1);
  const std::size_t T = 500;
  Domain d(N, N, 1, true, true, true);            // fully periodic: no walls
  MagneticBGK<ML> mc; mc.omega = MagneticBGK<ML>::omega_from_resistivity(nu);
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
  C fc;
  fc.omega = C::omega_from_viscosity(nu);
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  View1D<Real> fx("fx", d.n_padded), fy("fy", d.n_padded);
  {
    auto h1 = Kokkos::create_mirror_view(fx), h2 = Kokkos::create_mirror_view(fy);
    for (Index n = 0; n < d.n_padded; ++n) { h1(n) = F; h2(n) = Real(0); }
    Kokkos::deep_copy(fx, h1); Kokkos::deep_copy(fy, h2);
  }
  fc.forcing.Ex = fx; fc.forcing.Ey = fy;
  FluidSolver<FL, EsotericPull<FL>, C> fl(d, fc);
  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  mag.initialize_field(KOKKOS_LAMBDA(Index) {
    Kokkos::Array<Real, 3> b; b[0] = b[1] = b[2] = Real(0); return b;
  });
  fl.initialize(Real(1));
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());
  for (std::size_t t = 0; t < T; ++t) { mag.compute_field(); fl.step(true); mag.step(true); }
  fl.compute_macroscopic();
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
  double p = 0; std::size_t cnt = 0;
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const Index n = d.id(x, y, 0);
      p += double(hr(n)) * double(hu(n)); ++cnt;
    }
  p /= double(cnt);
  const double want = double(F) * (double(T) + 0.5);
  const double rel = std::abs(p - want) / want;
  std::printf("    %-24s rho*u %-14.8e  want %-14.8e  rel %8.2e  %s\n",
              tag, p, want, rel, rel < 1e-9 ? "PASS" : "FAIL");
  if (!(rel < 1e-9)) status = 1;
  return status;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Opts o; std::string op = "cm"; bool fcheck = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-n"     && i + 1 < argc) o.N = Index(std::atol(argv[++i]));
      if (a == "-rfac"  && i + 1 < argc) o.rfac = std::atof(argv[++i]);
      if (a == "-u"     && i + 1 < argc) o.u0 = std::atof(argv[++i]);
      if (a == "-re"    && i + 1 < argc) o.Re = std::atof(argv[++i]);
      if (a == "-prm"   && i + 1 < argc) o.prm = std::atof(argv[++i]);
      if (a == "-alf"   && i + 1 < argc) o.alf = std::atof(argv[++i]);
      if (a == "-k0"    && i + 1 < argc) o.k0 = std::atoi(argv[++i]);
      if (a == "-kw"    && i + 1 < argc) o.kw = std::atoi(argv[++i]);
      if (a == "-seed"  && i + 1 < argc) o.seed = std::strtoull(argv[++i], nullptr, 10);
      if (a == "-steps" && i + 1 < argc) o.steps = std::size_t(std::atol(argv[++i]));
      if (a == "-probe" && i + 1 < argc) o.probe = std::size_t(std::atol(argv[++i]));
      if (a == "-vtk"   && i + 1 < argc) o.vtkdir = argv[++i];
      if (a == "-vtkevery" && i + 1 < argc) o.vtkevery = std::size_t(std::atol(argv[++i]));
      if (a == "-bin"   && i + 1 < argc) o.bindir = argv[++i];
      if (a == "-binevery" && i + 1 < argc) o.binevery = std::size_t(std::atol(argv[++i]));
      if (a == "-walls" && i + 1 < argc) o.walls = argv[++i];
      if (a == "-shape" && i + 1 < argc) o.shape = argv[++i];
      if (a == "-wall"  && i + 1 < argc) o.wall = argv[++i];
      if (a == "-eps"   && i + 1 < argc) o.eps = std::atof(argv[++i]);
      if (a == "-epsm"  && i + 1 < argc) o.epsm = std::atof(argv[++i]);
      if (a == "-smooth" && i + 1 < argc) o.smooth = std::atof(argv[++i]);
      if (a == "-regime" && i + 1 < argc) {
        // Neffaa Table I. Selecting a regime sets the geometry, the spectrum
        // and the energy ratio together, because they are one case and picking
        // them apart is how a run ends up being of no published thing at all.
        o.regime = std::atoi(argv[++i]);
        o.rfac = 19.0 / 40.0;         // r = (19/20) pi in a box of 2 pi
        o.wall = "pen";               // the reference's method
        if (o.regime == 1) o.alf = 0.3;
        if (o.regime == 2) o.alf = 1.9e4;
        if (o.regime == 3) o.alf = 1.3;
        if (o.regime == 4) o.alf = 1.0;
      }
      if (a == "-op"    && i + 1 < argc) op = argv[++i];
      if (a == "-fcheck") fcheck = true;
    }
    if (fcheck) {
      std::printf("FORCE CHECK  periodic box, uniform F, 500 steps; rho*u must "
                  "equal F*T exactly\n");
      force_check<MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations,
                         FieldGuo>>("MhdBGK + FieldGuo", status);
      force_check<MhdCentralMoments<FL, true, FieldGuo>>(
          "MhdCM + FieldGuo (new)", status);
      Kokkos::finalize();
      return status;
    }
    std::printf("Decaying MHD turbulence in a confined disc -- Neffaa, Bos & "
                "Schneider (2008)   DEMONSTRATOR\n");
    std::printf("backend %s   precision %s   %s fluid + %s magnetic   operator %s\n\n",
                ExecSpace::name(), precision_name(), FL::name, ML::name,
                op == "cm" ? "MhdCM" : "MhdBGK");
    // -wall pen selects the forcing-capable operator regardless of -op, and
    // says so rather than silently overriding.
    if (o.wall == "pen") {
      status = (op == "bgk") ? run<CollP>(o) : run<CollCP>(o);
    } else {
      status = (op == "bgk") ? run<CollB>(o) : run<CollC>(o);
    }
  }
  Kokkos::finalize();
  return status;
}
