#pragma once
//==============================================================================
//  NIANZHI'S PASSIVE MIXER -- THE FORWARD EVALUATOR.
//
//  P1 of doc/mixer_adjoint_plan.md. Given a design field gamma in [0, 1] on the
//  design box (1 = fluid), this solves the steady flow, then the steady
//  concentration in that flow, and reports the mixing and the pressure drop.
//  The adjoint (P2-P4) will linearise exactly this, so every choice here is one
//  the plan already argued has a tractable transpose.
//
//  GEOMETRY. A square duct of side L, in units of L along x: an inlet section
//  len_in, the design box len_des, an outlet section len_out (1, 2.5, 1 by
//  default). The design box spans the whole width and the height H from the
//  floor. Walls are halfway bounce-back, so the duct is ny = nz = L + 2 with the
//  wall planes at 0.5 and L + 0.5 and fluid nodes 1..L ("ny = H + 2" in
//  CLAUDE.md). The inlet is the node plane x = 0, the outlet x = nx - 1, and
//  both are ON-node:
//
//    fluid   RegWall/NrmXm carrying the fully developed square-duct profile
//            (topopt/DuctFlow.hpp), RegWall/NrmOutXp at rho = 1 -- the pair
//            validation/channel3d.cpp uses;
//    scalar  ScalarMoment (Dellar's on-node Dirichlet) carrying the inlet
//            split, ScalarOutflow, and ScalarAdiabatic on the wall shell, which
//            is the scalar's halfway zero-flux wall on the fluid's wall plane.
//
//  The design box stops a full section short of both faces, so the drag never
//  reaches a RegWall node. (FluidSolver's banner records a body-force slip at
//  RegWall nodes; for a central-moment operator the wall's force path is not
//  even taken -- has_forcing needs a ForcingPolicy member MomentCollision does
//  not declare -- so the precaution here is geometric, not a reliance on it.)
//
//  `walls = false` drops the wall shell and makes y and z periodic; with
//  nz = 1 and a uniform inlet this is the plug-flow mixing layer that
//  validation/mixer_forward.cpp checks against the erf solution. It is a mode
//  of THIS class rather than a separate setup so that the check exercises the
//  evaluator's own inlet, outlet and scalar code.
//
//  THE DESIGN FIELD AND THE DRAG. gamma is one value per lattice cell of the
//  design box, mapped to DarcyGuo's per-node drag by RAMP,
//
//      A = A_max q (1 - gamma) / (q + gamma),   A_max = nu / (Da L^2)   (rho0 = 1).
//
//  A_max is DERIVED from the Darcy number every time, never stored as a lattice
//  number: nu = U L / Re moves with L, and holding A fixed while nu moves is the
//  melt_pool mistake recorded in CLAUDE.md. DarcyGuo is the implicit drag, so
//  every A is stable; its no-slip plane sits about 0.8 cells inside the solid
//  and moves with tau (Forcing.hpp), which is why a final design must be
//  re-evaluated with real Solid cells (plan, P6). Inside the drag region the
//  discrete steady state also carries an ODD-EVEN layer in the velocity next to
//  every solid surface (measured in mixer_forward's slab check: it decays by a
//  factor -0.42 per cell at tau = 0.62, A = 15.6, and -0.19 at tau = 0.8,
//  A = 6.25), the lattice's stand-in for a Brinkman layer far thinner than a
//  cell. A solid feature a few cells thick is all layer.
//
//  SHARP MODE (spec.sharp, 2026-10-23, written to run Nianzhi's own OpenLB case
//  from its STL: demonstrator/mixer_stl.cpp). A design cell with gamma < 0.5 is
//  a REAL wall -- Solid for the fluid, ScalarAdiabatic for the concentration,
//  i.e. halfway bounce-back for both, as in that case -- and the drag is zero
//  everywhere. It is the re-evaluation with real Solid cells the plan asks of a
//  final design (P6). It has no adjoint, since the topology is discrete, and c
//  does not diffuse into the solid. Off by default; nothing else changes then.
//    A real wall brings back the divergence problem the correction (below) was
//  written for, at the obstacles' EDGES. Measured on Nianzhi's mixer1.stl at
//  Re = 2, Pe = 2000 (2026-10-23): the D3Q7 face-flux divergence of the steady
//  velocity, with the bounce-back closure, peaks at 45 D per step at L = 20 and
//  17 D at L = 40, always on a wall-adjacent node at an obstacle's top edge
//  (a developed duct has none: its velocity is parallel to every wall). At
//  L = 20 the CORRECTED run is unstable -- c grows x1.17 per 900 steps after
//  three passes, fix in [-7.8e-3, 6.6e-3] -- while the uncorrected one is
//  bounded but reaches c = -11; at L = 40 the corrected run converges with c in
//  [-0.66, 1.04] (0.9% of nodes outside [-0.05, 1.05], mostly the interface
//  pushed obliquely across the grid). Why the correction destabilises is NOT
//  diagnosed; ScalarTRT's ghosts relax at only 0.002 per step here, which is
//  the first suspect. Do not run sharp mode at that Pe coarser than L = 40.
//
//  OPERATORS. CentralMoments<D3Q27, DarcyGuo> on raw populations, omega from
//  nu = U L / Re, omega_bulk = 1 unless told otherwise (set, not inherited --
//  the rb_high_ra drivers disagreed on exactly this). The scalar operator is a
//  template parameter and DEFAULTS TO ScalarTRT<D3Q7> AT Lambda = 1/4, omega
//  from D = U L / Pe. It was ScalarRegularised until the first two-grid run
//  diverged (2026-10-15): at Pe = 1e3, L = 40 that operator is linearly
//  unstable for |u| > 0.052, below the empty duct's own centreline speed. The
//  numbers and the scan are in collision/ScalarTRT.hpp's banner; the
//  regularised instance is kept (MixerEvaluatorReg) so the difference can be
//  measured rather than remembered. Both rates come from the operators' own
//  omega_from_*, which read each lattice's cs2; D3Q7's is 1/4.
//
//  THE UNIFORM-PRESERVING CORRECTION. The scalar is conservative: it keeps a
//  uniform c uniform only where u is divergence-free in the D3Q7 stencil, and
//  next to a drag that varies over a cell it is not. On the L = 16 grooves the
//  D3Q7 central divergence of u reaches 0.145 U per cell, and exceeds |u|
//  itself at near-stagnant drag nodes (310% at the node of largest c). A
//  UNIFORM inlet came out between 0.70 and 1.44 at L = 40, with a false outlet
//  Phi of 6.7e-5. Before every concentration solve, calibrate_uniform()
//  measures, on its own solver, what one step does to a uniform field, and
//  ScalarSolver::set_uniform_correction undoes it. A uniform inlet then stays
//  uniform to the CALIBRATION'S accuracy, not to round-off: an error d in the
//  factor acts as a source for a residence time, and the calibration's 1e-13
//  shows as 2.3e-10 in c at L = 16 (mixer_forward's anchor 6). It is exact only
//  in that sense, and only for UNIFORM fields: with two streams the converged
//  field still overshoots -- c = 1.73 at L = 16, inside a near-stagnant drag
//  region (anchor 5), 1.093 at L = 40 and 1.002 at L = 60 (anchor 4, the
//  grooves; with the correction the maximum sits at a grey edge, A ~ 0.5, not
//  in the solid).
//
//  WHAT IT CHANGES, MEASURED ON CONVERGED RUNS (2026-10-16, grooves, L = 16,
//  24, 32, 40): with it Phi = 0.17443, 0.17416, 0.17423, 0.17436 -- grid-
//  independent to 0.12% over L = 24..40; without it 0.15209, 0.14934, 0.14872,
//  0.14868, ALSO grid-converged, to a limit 15% lower. The difference is
//  mostly not the drag. It is COMPRESSIBILITY: at U = 0.025 (Ma_max 0.175) the
//  density falls ~8.5% from the inlet to the mixing plane; the fluid conserves
//  rho u, so u is not solenoidal even where the flow is smooth, and the
//  conservative scalar is DILUTED by the density ratio -- cbar = 0.461 where
//  conservation of an incompressible species gives 0.5 -- and Phi by its
//  square: (cbar_on / cbar_off)^2 equals Phi_on / Phi_off to 0.03% at L >= 32.
//  The rest, the drag regions, is 1.2% at L = 16, 0.17% at 24, 0.03% at 32 and
//  40: a discretisation error that refines away. (Advecting the scalar with
//  rho u / rho_0 would remove the dilution by itself.) The correction costs the
//  scalar's exact conservation where it acts, and the adjoint must
//  differentiate through its calibration (doc/mixer_adjoint_plan.md, sec. 10).
//
//  WHAT IS MEASURED.
//    mixing   on the plane x_m, a few cells upstream of the outlet, the
//             flux-weighted variance Phi = sum u (c - cbar)^2 / sum u with
//             cbar = sum u c / sum u, and M = 1 - sqrt(Phi / (cbar (1 - cbar))).
//    pressure Dp = cs2 (<rho>_1 - <rho>_2) between the middles of the inlet and
//             outlet sections, area-averaged over fluid nodes; with walls also
//             Dp / Dp0, Dp0 the EXACT empty-duct drop for the same mass flux,
//             K nu jbar (x2 - x1) / L^2. That ratio is dimensionless and is the
//             number to compare across grids.
//    limits   max |u| and the Mach number it gives, the relative density drop
//             between the pressure planes, and the range of c. They are
//             reported because the mean-velocity Mach number hides them: the
//             first two-grid run had a density drop of 8% and a peak |u| of four
//             times the mean.
//
//  CONVERGENCE is judged on the whole field over an interval, never per step
//  (see the convergence-criteria note this tree keeps): every `probe` steps the
//  relative L2 change r of the field is turned into e_hat = (r / probe) T_relax,
//  a bound on the remaining relative error for relaxation on the time T_relax.
//    flow           T_relax = L^2 / nu. After it stops, one extra step measures
//                   r_odd, the change over a single step: a period-2 component
//                   is invisible to probes that are all an even number of steps
//                   apart.
//    concentration  T_relax is the larger of two passes through the duct and,
//                   when the design holds solid, 4 l^2 / (pi^2 D) with l the
//                   design box's LONGEST side: what the flow does not reach
//                   relaxes by diffusion alone, and a pocket closed at one end
//                   relaxes FOUR times slower than one open at both -- a design
//                   can make one along x, y or z. (It was the height H until a
//                   review asked what a dead-end slot along x would do; the cap
//                   on the run still uses H.) Phi
//                   itself must meet the same e_hat, because Phi is what is
//                   reported. The run HALTS if c leaves [-1, 2] -- a DIVERGENCE
//                   guard. It was [-0.05, 1.05] until the transient from a seed
//                   was measured overshooting 5% with nothing unstable (mixer_
//                   forward, anchors 5 and 6); whether the CONVERGED field keeps
//                   the maximum principle is the anchors' question, not a halt's.
//                   The concentration is (re)seeded at the equilibrium OF THE
//                   CONVERGED FLOW, not at rest: a rest seed starts every flux at
//                   zero against an equilibrium flux c u, and near omega = 2
//                   that mismatch is over-relaxed, flipping sign every step.
//
//  WHAT IT DOES NOT DO.
//    * No gradient. That is P2-P4.
//    * D is uniform, so c diffuses through Brinkman solid exactly as through
//      stagnant fluid. A real wall is impermeable; whether the difference
//      matters is measured in P6, and the per-node rate (omega_of) is the fix.
//    * No compressibility correction: at fixed U the density drop does not
//      refine away (channel3d.cpp's banner), so it is reported, not removed.
//    * No GPU/ twin: GPU/ has no fluid outlet.
//    * FP64 only. The convergence tolerances the anchors need are below FP32's
//      reach, and the adjoint will be checked against finite differences.
//==============================================================================
#include "collision/MomentCollision.hpp"   // pulls collision/BGK.hpp, which defines Macro
#include "collision/ScalarRegularised.hpp"
#include "collision/ScalarTRT.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"
#include "topopt/DuctFlow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace lbm {
namespace topopt {

using MixFluidColl = CentralMoments<D3Q27, DarcyGuo, RawPopulations>;
using MixFluid     = FluidSolver<D3Q27, EsotericPull<D3Q27>, MixFluidColl>;

// RAMP interpolation of the drag: 0 at gamma = 1 (fluid), 1 at gamma = 0.
inline double ramp(double g, double q) { return q * (1.0 - g) / (q + g); }

struct MixerSpec {
  Index  L = 40;              // fluid nodes across the duct
  double len_in = 1.0, len_des = 2.5, len_out = 1.0;   // section lengths, units of L
  double H  = 0.5;            // design-box height from the floor, units of L
  double U  = 0.025;          // MEAN inlet velocity, lattice units
  double Re = 10.0;           // U L / nu
  double Pe = 1000.0;         // U L / D
  double Da = 1e-5;           // nu / (A_max L^2)
  double q  = 0.1;            // RAMP parameter
  bool   walls = true;        // false: no wall shell, periodic in y and z
  Index  nz_open = 1;         // nz when walls == false
  bool   developed = true;    // inlet: fully developed duct profile, else uniform
  bool   band = false;        // inlet split: c = 1 above y = L/2, or in the middle half
  double s0 = 1.0;            // erf width of the inlet split, in cells
  double c_uniform = -1.0;    // >= 0: a UNIFORM inlet at this value instead of a split --
                              // the exact answer is then c = c_uniform everywhere
  double omega_bulk = 1.0;
  double Lambda = 0.25;       // the scalar TRT's Lambda (ignored by other operators)
  bool   uniform_fix = true;  // calibrate ScalarSolver's uniform-preserving correction
  bool   sharp = false;       // design cells with gamma < 0.5 are REAL walls (Solid /
                              // ScalarAdiabatic, halfway bounce-back) and there is no drag
};

// Refuse a spec the geometry cannot hold. -H 1.5 used to write the drag field
// past the end of its View, silently.
inline void validate(const MixerSpec& s) {
  auto bad = [](const char* why) {
    std::printf("MixerSpec: %s\n", why);
    std::abort();
  };
  if (s.L < 4) bad("L must be at least 4");
  if (!(s.H >= 0.0 && s.H <= 1.0)) bad("H is a fraction of L and must lie in [0, 1]");
  if (!(s.len_in >= 0.5 && s.len_out >= 0.5 && s.len_des >= 0.0))
    bad("the inlet and outlet sections must be at least L/2 long, the design box >= 0");
  if (!(s.U > 0.0 && s.Re > 0.0 && s.Pe > 0.0 && s.Da > 0.0 && s.q > 0.0 && s.s0 > 0.0))
    bad("U, Re, Pe, Da, q and s0 must be positive");
  if (!s.walls && s.nz_open < 1) bad("nz_open must be at least 1");
  if (!s.walls && s.developed)
    bad("a developed duct inlet needs the duct: walls = false takes a uniform inlet");
}

struct Convergence {
  std::size_t steps = 0;
  double resid = NAN;         // relative L2 change over the last probe interval
  double e_hat = NAN;         // (resid / probe) * T_relax
  double e_phi = NAN;         // the same for Phi (concentration only)
  double r_odd = NAN;         // flow only: relative change over ONE step at the end
  double cmin = NAN, cmax = NAN;   // concentration only, at the last probe
  double T_relax = NAN;
  std::size_t probe = 0;
  bool converged = false;
  bool finite = true;
  bool bounded = true;        // concentration only: c stayed in run_scalar's halt band
  bool calibrated = true;     // concentration only: the uniform correction converged
  double secs = 0.0;
};

// The calibration of the uniform-preserving correction (see calibrate_uniform).
struct Calibration {
  std::size_t steps = 0;
  double change = NAN;        // max |change of the loaded field| over the last probe
  double fix_min = NAN, fix_max = NAN;   // the factor, over bulk nodes
  bool converged = false;
  double secs = 0.0;
};

struct MixerResult {
  double Phi = NAN, M = NAN, cbar = NAN;
  double rho1 = NAN, rho2 = NAN, j1 = NAN, j2 = NAN;
  double dp = NAN, dp0 = NAN, dp_ratio = NAN;
  double umax = NAN, Ma_max = NAN;  // max |u| over fluid nodes, and |u|/cs
  double drho = NAN;                // (rho1 - rho2) / rho2
  double cmin = NAN, cmax = NAN;    // over the concentration's bulk nodes
};

// Everything that follows from the spec alone.
struct MixerGeom {
  Index L, nx, ny, nz;
  Index y0, y1, z0, z1;       // fluid cross-section, inclusive
  Index x_d0, x_d1, nzd;      // design box: x in [x_d0, x_d1), z in [z0, z0 + nzd)
  Index x_p1, x_p2, x_m;      // pressure planes, mixing plane
  double U, nu, D, A_max, K;
  bool walls;

  explicit MixerGeom(const MixerSpec& s)
      : L(s.L), walls(s.walls) {
    validate(s);
    const Index nin = Index(std::lround(s.len_in * double(L)));
    const Index ndes = Index(std::lround(s.len_des * double(L)));
    const Index nout = Index(std::lround(s.len_out * double(L)));
    nx = nin + ndes + nout + 1;
    ny = walls ? L + 2 : L;
    nz = walls ? L + 2 : s.nz_open;
    y0 = walls ? 1 : 0;  y1 = walls ? L : L - 1;
    z0 = walls ? 1 : 0;  z1 = walls ? L : nz - 1;
    x_d0 = nin;  x_d1 = nin + ndes;
    nzd = walls ? Index(std::lround(s.H * double(L))) : 0;
    x_p1 = nin / 2;
    x_p2 = x_d1 + (nx - 1 - x_d1) / 2;
    x_m  = nx - 1 - std::max<Index>(4, L / 8);
    U  = s.U;
    nu = s.U * double(L) / s.Re;
    D  = s.U * double(L) / s.Pe;
    A_max = nu / (s.Da * double(L) * double(L));
    K = duct_K();
  }
  // Distance of node y (or z) from the y = 0 (z = 0) wall plane, in cells.
  double eta(Index y) const { return walls ? double(y) - 0.5 : double(y) + 0.5; }
  bool is_wall(Index y, Index z) const {
    return walls && (y == 0 || y == ny - 1 || z == 0 || z == nz - 1);
  }
  bool in_design(Index x, Index y, Index z) const {
    return x >= x_d0 && x < x_d1 && y >= y0 && y <= y1 && z >= z0 && z < z0 + nzd;
  }
  Index n_design() const { return (x_d1 - x_d0) * (y1 - y0 + 1) * nzd; }
};

// The inlet split, as the fully smoothed erf step it is imposed as. `band`
// sums periodic images, which is what the open (walls == false) box has.
inline double inlet_c(double eta, double L, double s, bool band) {
  if (!band) return 0.5 * (1.0 + std::erf((eta - 0.5 * L) / (2.0 * s)));
  double c = 0.0;
  for (int k = -2; k <= 2; ++k)
    c += 0.5 * (std::erf((eta - 0.25 * L + k * L) / (2.0 * s)) -
                std::erf((eta - 0.75 * L + k * L) / (2.0 * s)));
  return c;
}

template <class SColl>
class MixerEvaluatorT {
 public:
  using ScalarColl = SColl;
  using Scalar = ScalarSolver<D3Q7, EsotericPull<D3Q7>, SColl>;

  MixerSpec sp;
  MixerGeom g;
  Domain dom;
  View1D<Real> A;
  MixFluid fluid;
  Scalar scalar;
  Scalar calib_scalar;               // the calibration's own solver (calibrate_uniform)
  bool calib_seeded = false;         // its state persists: later calibrations warm-start
  std::vector<double> gamma;         // design box, x slowest then y then z
  bool has_solid = false;
  double umax_over_mean = NAN;       // from the duct series, for printing
  std::vector<Index> fnodes;         // fluid nodes (Fluid or RegWall), host list
  std::vector<Index> snodes;         // ScalarBulk nodes, host list
  View1D<Real> ctab;                 // inlet split by y, the scalar's seed
  View1D<Real> fix;                  // the uniform-preserving correction, per node
  Calibration calib;                 // how the last calibration went

  explicit MixerEvaluatorT(const MixerSpec& s)
      : sp(s), g(s), dom(g.nx, g.ny, g.nz, false, !s.walls, !s.walls),
        A("mixer_A", dom.n_padded),
        fluid(dom, make_fluid_coll(g, s, A)),
        scalar(dom, make_scalar_coll(g, s)),
        calib_scalar(dom, make_scalar_coll(g, s)),
        gamma(std::size_t(g.n_design()), 1.0) {
    if (s.walls && s.developed) umax_over_mean = duct_self_check();
    build_fluid();
    build_scalar();
  }

  static MixFluidColl make_fluid_coll(const MixerGeom& g, const MixerSpec& s,
                                      View1D<Real> A) {
    MixFluidColl c;
    c.omega = MixFluidColl::omega_from_viscosity(Real(g.nu));
    c.omega_bulk = Real(s.omega_bulk);
    c.forcing = DarcyGuo{};
    c.forcing.A = A;
    return c;
  }
  static SColl make_scalar_coll(const MixerGeom& g, const MixerSpec& s) {
    SColl c;
    c.omega = SColl::omega_from_diffusivity(Real(g.D));
    c.T_ref = Real(0);
    if constexpr (requires { c.Lambda; }) c.Lambda = Real(s.Lambda);
    return c;
  }
  double omega_fluid() const { return MixFluidColl::omega_from_viscosity(Real(g.nu)); }
  double omega_scalar() const { return SColl::omega_from_diffusivity(Real(g.D)); }

  // Inlet velocity at node (y, z), lattice units.
  double inlet_u(Index y, Index z) const {
    if (!sp.developed) return g.U;
    return g.U * duct_u_over_mean(g.eta(y) / double(g.L), g.eta(z) / double(g.L), g.K);
  }

  // A real wall inside the design box (sharp mode only).
  bool solid_at(Index x, Index y, Index z) const {
    if (!sp.sharp || !g.in_design(x, y, z)) return false;
    const Index nyd = g.y1 - g.y0 + 1;
    return gamma[std::size_t(((x - g.x_d0) * nyd + (y - g.y0)) * g.nzd + (z - g.z0))] < 0.5;
  }

  void build_fluid() {
    const MixerGeom gg = g;
    fluid.set_geometry([&](Index x, Index y, Index z) -> CellType {
      return gg.is_wall(y, z) || solid_at(x, y, z) ? Solid : Fluid;
    });
    // Inlet velocities on the host once: the series costs ~600 terms a node.
    std::vector<double> uin(std::size_t(g.ny * g.nz), 0.0);
    for (Index z = g.z0; z <= g.z1; ++z)
      for (Index y = g.y0; y <= g.y1; ++y) uin[std::size_t(z * g.ny + y)] = inlet_u(y, z);
    using WS = typename MixFluid::WallSpec;
    fluid.set_regularized_walls([&](Index x, Index y, Index z) -> WS {
      if (gg.is_wall(y, z)) return WS{};          // never promote a Solid cell
      if (x == 0)
        return WS{NrmXm, Real(uin[std::size_t(z * gg.ny + y)]), Real(0), Real(0), Real(1)};
      if (x == gg.nx - 1) return WS{NrmOutXp, Real(0), Real(0), Real(0), Real(1)};
      return WS{};
    });
    fluid.set_outflow_order(1);

    // Seed the fully developed state: the inlet profile at every station and
    // the exact linear pressure drop measured back from the outlet's rho = 1.
    // From rest the first diffusive time is spent building what is known.
    View1D<Real> utab("mixer_uin", std::size_t(g.ny * g.nz));
    auto hu = Kokkos::create_mirror_view(utab);
    for (std::size_t i = 0; i < uin.size(); ++i) hu(i) = Real(uin[i]);
    Kokkos::deep_copy(utab, hu);
    const double G = (sp.walls && sp.developed) ? g.K * g.nu * g.U / (double(g.L) * g.L) : 0.0;
    const Real drdx = Real(G / double(cs2<D3Q27, Real>()));
    const Domain d = dom;
    const Index ny = g.ny, nxl = g.nx;
    fluid.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index x = px - d.hx, y = py - d.hy, z = pz - d.hz;
      const Real r = Real(1) + drdx * Real(nxl - 1 - x);
      return FlowState{r, utab(std::size_t(z * ny + y)), Real(0), Real(0)};
    });

    fnodes.clear();
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fluid.flags());
    for (Index z = 0; z < g.nz; ++z)
      for (Index y = 0; y < g.ny; ++y)
        for (Index x = 0; x < g.nx; ++x) {
          const Index n = dom.id(x, y, z);
          if (hf(n) == Fluid || hf(n) == RegWall) fnodes.push_back(n);
        }
  }

  void build_scalar() {
    const MixerGeom gg = g;
    auto geometry = [&](Index x, Index y, Index z) -> std::uint8_t {
      if (gg.is_wall(y, z) || solid_at(x, y, z)) return ScalarAdiabatic;
      if (x == 0) return ScalarMoment;
      if (x == gg.nx - 1) return ScalarOutflow;
      return ScalarBulk;
    };
    scalar.set_geometry(geometry);
    // The calibration's solver: same geometry and velocity, inlet held at 1.
    calib_scalar.set_geometry(geometry);
    calib_scalar.set_wall_values([&](Index x, Index, Index) -> Real {
      return x == 0 ? Real(1) : Real(0);
    });
    calib_scalar.finalize_geometry();
    calib_scalar.set_velocity(fluid.ux(), fluid.uy(), fluid.uz());
    std::vector<double> cin(std::size_t(g.ny), 0.0);
    for (Index y = 0; y < g.ny; ++y)
      cin[std::size_t(y)] = sp.c_uniform >= 0.0 ? sp.c_uniform
                                                : inlet_c(g.eta(y), double(g.L), sp.s0, sp.band);
    scalar.set_wall_values([&](Index x, Index y, Index) -> Real {
      return x == 0 ? Real(cin[std::size_t(y)]) : Real(0);
    });
    scalar.finalize_geometry();
    scalar.set_velocity(fluid.ux(), fluid.uy(), fluid.uz());

    // Start from the unmixed state: the inlet split carried down the duct.
    ctab = View1D<Real>("mixer_cin", std::size_t(g.ny));
    auto hc = Kokkos::create_mirror_view(ctab);
    for (Index y = 0; y < g.ny; ++y) hc(std::size_t(y)) = Real(cin[std::size_t(y)]);
    Kokkos::deep_copy(ctab, hc);
    seed_scalar(false);

    snodes.clear();
    auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalar.flags());
    for (Index z = 0; z < g.nz; ++z)
      for (Index y = 0; y < g.ny; ++y)
        for (Index x = 0; x < g.nx; ++x) {
          const Index n = dom.id(x, y, z);
          if (hs(n) == ScalarBulk) snodes.push_back(n);
        }
  }

  // Seed the concentration at the unmixed state, at rest or -- once the flow is
  // known -- at the equilibrium of that flow (ScalarSolver's
  // initialize_field_in_flow). The seeder visits HALO nodes too, whose y lies
  // one outside [0, ny); clamp rather than read past the table.
  void seed_scalar(bool in_flow) {
    const Domain d = dom;
    const Index nyc = g.ny;
    auto tab = ctab;
    auto fn = KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      Index y = py - d.hy;
      y = y < 0 ? Index(0) : (y >= nyc ? nyc - 1 : y);
      return tab(std::size_t(y));
    };
    if (in_flow) scalar.initialize_field_in_flow(fn);
    else         scalar.initialize_field(fn);
  }

  //----------------------------------------------------------------------------
  // The design. gamma is indexed ((i * ny_d) + j) * nzd + k over the design box,
  // i.e. a C-order (nxd, L, nzd) array -- the layout a float32 .npy of that
  // shape has. Values are clamped to [0, 1]; a non-finite value aborts, since
  // clamping a NaN would silently make that cell solid.
  //----------------------------------------------------------------------------
  void set_design(const std::vector<double>& gam) {
    if (gam.size() != gamma.size()) {
      std::printf("MixerEvaluator::set_design: %zu values for a design box of %zu cells\n",
                  gam.size(), gamma.size());
      std::abort();
    }
    std::size_t bad = 0;
    for (double v : gam) if (!std::isfinite(v)) ++bad;
    if (bad) {
      std::printf("MixerEvaluator::set_design: %zu non-finite value(s) in the design\n", bad);
      std::abort();
    }
    gamma = gam;
    for (double& v : gamma) v = std::min(1.0, std::max(0.0, v));
    apply_design();
  }
  // gfn(xi, eta, zeta) in units of L from the design box's upstream end, the
  // y = 0 wall plane and the floor -- so one function samples any resolution.
  template <class Fn>
  void set_design_fn(Fn gfn) {
    const Index nyd = g.y1 - g.y0 + 1;
    std::vector<double> gam(gamma.size());
    std::size_t m = 0;
    for (Index i = 0; i < g.x_d1 - g.x_d0; ++i)
      for (Index j = 0; j < nyd; ++j)
        for (Index k = 0; k < g.nzd; ++k, ++m)
          gam[m] = gfn((double(i) + 0.5) / double(g.L), g.eta(g.y0 + j) / double(g.L),
                       g.eta(g.z0 + k) / double(g.L));
    set_design(gam);
  }
  // Sharp mode leaves A at zero and rebuilds both geometries instead, which
  // re-seeds the flow (developed profile) and the concentration (unmixed), and
  // re-seeds the calibration, whose solver's geometry has just changed.
  void apply_design() {
    auto h = Kokkos::create_mirror_view(A);
    for (std::size_t n = 0; n < h.extent(0); ++n) h(n) = Real(0);
    const Index nyd = g.y1 - g.y0 + 1;
    has_solid = false;
    std::size_t m = 0;
    for (Index i = 0; i < g.x_d1 - g.x_d0; ++i)
      for (Index j = 0; j < nyd; ++j)
        for (Index k = 0; k < g.nzd; ++k, ++m) {
          if (!sp.sharp)
            h(dom.id(g.x_d0 + i, g.y0 + j, g.z0 + k)) = Real(g.A_max * ramp(gamma[m], sp.q));
          if (gamma[m] < 0.5) has_solid = true;
        }
    Kokkos::deep_copy(A, h);
    if (sp.sharp) {
      build_fluid();
      build_scalar();
      calib_seeded = false;
    }
  }

  // Does fluid (gamma >= 0.5, and everything outside the box) connect the inlet
  // plane to the outlet plane? A blocked design does not crash; it returns a
  // mixing number that means nothing, so the driver asks first.
  bool connected() const {
    const Index nyd = g.y1 - g.y0 + 1;
    auto open = [&](Index x, Index y, Index z) {
      if (g.is_wall(y, z)) return false;
      if (!g.in_design(x, y, z)) return true;
      const std::size_t m = std::size_t(((x - g.x_d0) * nyd + (y - g.y0)) * g.nzd + (z - g.z0));
      return gamma[m] >= 0.5;
    };
    std::vector<char> seen(std::size_t(g.nx * g.ny * g.nz), 0);
    std::vector<std::size_t> stack;
    auto key = [&](Index x, Index y, Index z) { return std::size_t((z * g.ny + y) * g.nx + x); };
    for (Index z = g.z0; z <= g.z1; ++z)
      for (Index y = g.y0; y <= g.y1; ++y)
        if (open(0, y, z)) { seen[key(0, y, z)] = 1; stack.push_back(key(0, y, z)); }
    const std::size_t sx = std::size_t(g.nx), sy = std::size_t(g.ny);
    while (!stack.empty()) {
      const std::size_t k = stack.back(); stack.pop_back();
      const Index x = Index(k % sx), y = Index((k / sx) % sy), z = Index(k / (sx * sy));
      if (x == g.nx - 1) return true;
      const Index nb[6][3] = {{x+1,y,z},{x-1,y,z},{x,y+1,z},{x,y-1,z},{x,y,z+1},{x,y,z-1}};
      for (const auto& c : nb) {
        Index cx = c[0], cy = c[1], cz = c[2];
        if (!g.walls) { cy = (cy + g.ny) % g.ny; cz = (cz + g.nz) % g.nz; }
        if (cx < 0 || cx >= g.nx || cy < 0 || cy >= g.ny || cz < 0 || cz >= g.nz) continue;
        if (seen[key(cx, cy, cz)] || !open(cx, cy, cz)) continue;
        seen[key(cx, cy, cz)] = 1; stack.push_back(key(cx, cy, cz));
      }
    }
    return false;
  }

  //----------------------------------------------------------------------------
  // Host copies, for measurements and the anchors.
  //----------------------------------------------------------------------------
  template <class V> static std::vector<double> to_host(const V& v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    std::vector<double> out(h.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = double(h(i));
    return out;
  }
  std::vector<double> host_rho() { fluid.compute_macroscopic(); return to_host(fluid.rho()); }
  std::vector<double> host_ux()  { fluid.compute_macroscopic(); return to_host(fluid.ux()); }
  std::vector<double> host_uy()  { fluid.compute_macroscopic(); return to_host(fluid.uy()); }
  std::vector<double> host_uz()  { fluid.compute_macroscopic(); return to_host(fluid.uz()); }
  std::vector<double> host_c()   { scalar.compute_field(); return to_host(scalar.temperature()); }

  // The velocity at the fluid nodes, packed (ux, uy, uz) per node.
  std::vector<double> fluid_velocity() {
    fluid.compute_macroscopic();
    const auto ux = to_host(fluid.ux()), uy = to_host(fluid.uy()), uz = to_host(fluid.uz());
    std::vector<double> v(3 * fnodes.size());
    for (std::size_t m = 0; m < fnodes.size(); ++m) {
      const Index n = fnodes[m];
      v[3 * m] = ux[n]; v[3 * m + 1] = uy[n]; v[3 * m + 2] = uz[n];
    }
    return v;
  }
  static double rel_change(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t m = 0; m < a.size(); ++m) {
      const double e = a[m] - b[m]; num += e * e; den += a[m] * a[m];
    }
    return std::sqrt(num / std::max(den, 1e-300));
  }

  //----------------------------------------------------------------------------
  // Steady flow. tol bounds e_hat; max_T caps the run in units of T_relax.
  //----------------------------------------------------------------------------
  Convergence run_flow(double tol, double max_T = 30.0, std::size_t probe = 0,
                       int verbose = 0) {
    Convergence cv;
    cv.T_relax = double(g.L) * double(g.L) / g.nu;
    cv.probe = probe ? probe : std::size_t(std::clamp(cv.T_relax / 50.0, 100.0, 2000.0));
    const std::size_t cap = std::size_t(max_T * cv.T_relax);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> prev, cur;
    int nprobe = 0;
    while (cv.steps < cap) {
      for (std::size_t k = 0; k < cv.probe; ++k) fluid.step();
      cv.steps += cv.probe;
      cur = fluid_velocity();
      double den = 0.0;
      for (double v : cur) den += v * v;
      if (!std::isfinite(den)) { cv.finite = false; break; }
      if (!prev.empty()) {
        cv.resid = rel_change(cur, prev);
        cv.e_hat = cv.resid / double(cv.probe) * cv.T_relax;
        if (verbose && (++nprobe % verbose == 0))
          std::printf("      flow   step %8zu  t/T = %6.3f  r = %.3e  e_hat = %.3e\n",
                      cv.steps, double(cv.steps) / cv.T_relax, cv.resid, cv.e_hat);
        if (cv.e_hat < tol) { cv.converged = true; break; }
      }
      prev.swap(cur);
    }
    if (cv.finite) {
      // One odd step: the change over it is the period-2 content of the state.
      const std::vector<double> last = cv.converged ? cur : prev;
      fluid.step();
      cv.r_odd = rel_change(fluid_velocity(), last);
      fluid.step();
      cv.steps += 2;
    }
    cv.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fluid.compute_macroscopic();      // leave the steady velocity in the views
    return cv;
  }

  //----------------------------------------------------------------------------
  // THE UNIFORM-PRESERVING CORRECTION, calibrated for the CURRENT velocity.
  //
  // Hold c = 1 at every bulk node -- after each step the arriving concentration
  // f is topped back up to 1 by a source 1 - f injected at rest -- with the
  // inlet at 1, until f stops changing: the fluxes and ghosts reach the scheme's
  // own steady state for a uniform field. f(x) is then exactly what the
  // velocity's discrete divergence does to a uniform field in one step, and
  // fix = 1/f - 1 (ScalarSolver::set_uniform_correction) undoes it. For any
  // uniform inlet the corrected run reproduces this state scaled, so the field
  // stays uniform to the calibration's accuracy (2.3e-10 at L = 16:
  // validation/mixer_forward.cpp's anchor 6). The calibration is local -- c is pinned, so nothing has to cross the
  // duct -- and converges at the rate the flux and the ghosts relax, a few
  // hundred steps per decade near omega = 2.
  //
  // It depends on the velocity, so the ADJOINT must differentiate through it
  // (doc/mixer_adjoint_plan.md, section 10). It is linear in the populations,
  // so that is one more adjoint solve of the same kind.
  //----------------------------------------------------------------------------
  Calibration calibrate_uniform(double tol = 1e-13, std::size_t max_steps = 200000,
                                std::size_t probe = 100) {
    Calibration cb;
    const auto t0 = std::chrono::steady_clock::now();
    fluid.compute_macroscopic();
    // Its own solver, so the field being solved for is never touched -- a
    // warm start of `scalar` survives a calibration. The calibration itself
    // warm-starts from the previous flow's calibrated state after the first.
    if (!calib_seeded) {
      calib_scalar.initialize_field_in_flow(KOKKOS_LAMBDA(Index) { return Real(1); });
      calib_seeded = true;
    }
    auto field = calib_scalar.temperature();
    auto flags = calib_scalar.flags();
    std::vector<double> prev;
    while (cb.steps < max_steps) {
      for (std::size_t k = 0; k < probe; ++k) {
        calib_scalar.step();
        calib_scalar.compute_field();    // f: what each node loads next step
        calib_scalar.add_source(KOKKOS_LAMBDA(Index n) {
          return flags(n) == ScalarBulk ? Real(1) - field(n) : Real(0);
        });
      }
      cb.steps += probe;
      // field still holds f, computed before this probe's last source.
      const auto f = to_host(field);
      std::vector<double> cur(snodes.size());
      for (std::size_t m = 0; m < snodes.size(); ++m) cur[m] = f[snodes[m]];
      if (!prev.empty()) {
        double d = 0.0;
        for (std::size_t m = 0; m < cur.size(); ++m) d = std::max(d, std::fabs(cur[m] - prev[m]));
        cb.change = d;
        if (!std::isfinite(d)) break;
        if (d < tol) { cb.converged = true; prev.swap(cur); break; }
      }
      prev.swap(cur);
    }
    // calib_scalar is left topped up: that IS the pinned steady state, and the
    // next calibration (the next design's flow) continues from it.
    // fix = 1/f - 1 at bulk nodes, 0 elsewhere, switched on for the real solve.
    fix = View1D<Real>("mixer_fix", dom.n_padded);
    auto h = Kokkos::create_mirror_view(fix);
    for (std::size_t n = 0; n < h.extent(0); ++n) h(n) = Real(0);
    cb.fix_min = 1e300; cb.fix_max = -1e300;
    for (std::size_t m = 0; m < snodes.size(); ++m) {
      const double fv = 1.0 / prev[m] - 1.0;
      h(snodes[m]) = Real(fv);
      cb.fix_min = std::min(cb.fix_min, fv); cb.fix_max = std::max(cb.fix_max, fv);
    }
    Kokkos::deep_copy(fix, h);
    scalar.set_uniform_correction(fix);
    cb.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    calib = cb;
    return cb;
  }

  //----------------------------------------------------------------------------
  // Steady concentration in the CURRENT velocity field, which is frozen: the
  // fluid is not stepped here, and the scalar reads its views every step.
  //----------------------------------------------------------------------------
  // The time e_hat assumes. With solid in the design, a fluid pocket the flow
  // does not reach relaxes by diffusion alone, as 4 l^2/(pi^2 D) for a pocket of
  // length l closed at one end. A design can make one as long as the box's
  // LONGEST side -- along x (len_des L), y (L) or z (H) -- so that is the bound.
  // It costs a few probes when the decay is fast (the residual falls
  // geometrically) and is what keeps a slow pocket from passing as converged.
  double scalar_T_relax() const {
    double T = 2.0 * double(g.nx) / g.U;
    if (has_solid) {
      const double l = double(std::max({g.x_d1 - g.x_d0, g.y1 - g.y0 + 1, g.nzd}));
      const double pi = 3.14159265358979323846;
      T = std::max(T, 4.0 * l * l / (pi * pi * g.D));
    }
    return T;
  }
  // The run's cap: the floor-to-top diffusive time of the box, as before. The
  // bound above can be days at L = 40; a design that needs it is reported as
  // not converged rather than run that long.
  double scalar_T_cap() const {
    double T = 2.0 * double(g.nx) / g.U;
    if (has_solid) {
      const double h = double(g.nzd), pi = 3.14159265358979323846;
      T = std::max(T, 4.0 * h * h / (pi * pi * g.D));
    }
    return T;
  }
  Convergence run_scalar(double tol, double max_T = 30.0, std::size_t probe = 0,
                         int verbose = 0, std::vector<double>* phi_hist = nullptr,
                         bool reseed = true, double band = 1.0) {
    Convergence cv;
    fluid.compute_macroscopic();
    if (sp.uniform_fix) cv.calibrated = calibrate_uniform().converged;
    else                scalar.clear_uniform_correction();
    // reseed = false continues from whatever `scalar` holds -- a warm start, which
    // the calibration (on its own solver) no longer disturbs.
    if (reseed) seed_scalar(true);
    cv.T_relax = scalar_T_relax();
    cv.probe = probe ? probe
                     : std::size_t(std::clamp(2.0 * double(g.nx) / g.U / 20.0, 100.0, 2000.0));
    const std::size_t cap = std::size_t(max_T * scalar_T_cap());
    const auto ux = to_host(fluid.ux());
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> prev;
    double phi_prev = NAN;
    int nprobe = 0;
    while (cv.steps < cap) {
      for (std::size_t k = 0; k < cv.probe; ++k) scalar.step();
      cv.steps += cv.probe;
      const auto c = host_c();
      std::vector<double> cur(snodes.size());
      double den = 0.0, lo = 1e300, hi = -1e300;
      for (std::size_t m = 0; m < snodes.size(); ++m) {
        cur[m] = c[snodes[m]]; den += cur[m] * cur[m];
        lo = std::min(lo, cur[m]); hi = std::max(hi, cur[m]);
      }
      cv.cmin = lo; cv.cmax = hi;
      if (!std::isfinite(den)) { cv.finite = false; break; }
      if (lo < -band || hi > 1.0 + band) { cv.bounded = false; break; }
      const double phi = mixing(ux, c).Phi;
      if (phi_hist) phi_hist->push_back(phi);
      if (!prev.empty()) {
        cv.resid = rel_change(cur, prev);
        cv.e_hat = cv.resid / double(cv.probe) * cv.T_relax;
        // Relative to Phi, with an absolute floor: a uniform inlet's exact Phi is
        // zero, and round-off over round-off would never converge. 1e-14 is far
        // below any variance worth reporting (Phi <= cbar (1 - cbar) <= 1/4).
        cv.e_phi = std::fabs(phi - phi_prev) / std::max(std::fabs(phi), 1e-14) /
                   double(cv.probe) * cv.T_relax;
        if (verbose && (++nprobe % verbose == 0))
          std::printf("      scalar step %8zu  t/T = %6.3f  r = %.3e  e_hat = %.3e"
                      "  Phi = %.6e  e_phi = %.3e  c in [%.4f, %.4f]\n", cv.steps,
                      double(cv.steps) / cv.T_relax, cv.resid, cv.e_hat, phi, cv.e_phi, lo, hi);
        if (cv.e_hat < tol && cv.e_phi < tol) { cv.converged = true; break; }
      }
      prev.swap(cur);
      phi_prev = phi;
    }
    if (!cv.calibrated) cv.converged = false;   // a result on an unconverged correction is not one
    cv.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return cv;
  }

  //----------------------------------------------------------------------------
  // Measurements.
  //----------------------------------------------------------------------------
  MixerResult mixing(const std::vector<double>& ux, const std::vector<double>& c) const {
    MixerResult r;
    double sw = 0.0, swc = 0.0;
    for (Index z = g.z0; z <= g.z1; ++z)
      for (Index y = g.y0; y <= g.y1; ++y) {
        const Index n = dom.id(g.x_m, y, z);
        sw += ux[n]; swc += ux[n] * c[n];
      }
    r.cbar = swc / sw;
    double sv = 0.0;
    for (Index z = g.z0; z <= g.z1; ++z)
      for (Index y = g.y0; y <= g.y1; ++y) {
        const Index n = dom.id(g.x_m, y, z);
        const double e = c[n] - r.cbar; sv += ux[n] * e * e;
      }
    r.Phi = sv / sw;
    r.M = 1.0 - std::sqrt(r.Phi / (r.cbar * (1.0 - r.cbar)));
    return r;
  }
  // Area means of rho and rho u_x over the fluid nodes of plane x.
  void plane_means(const std::vector<double>& rho, const std::vector<double>& ux, Index x,
                   double& rbar, double& jbar) const {
    double sr = 0.0, sj = 0.0, cnt = 0.0;
    for (Index z = g.z0; z <= g.z1; ++z)
      for (Index y = g.y0; y <= g.y1; ++y) {
        const Index n = dom.id(x, y, z);
        sr += rho[n]; sj += rho[n] * ux[n]; cnt += 1.0;
      }
    rbar = sr / cnt; jbar = sj / cnt;
  }
  MixerResult measure() {
    const auto rho = host_rho(), ux = host_ux(), uy = host_uy(), uz = host_uz(), c = host_c();
    MixerResult r = mixing(ux, c);
    plane_means(rho, ux, g.x_p1, r.rho1, r.j1);
    plane_means(rho, ux, g.x_p2, r.rho2, r.j2);
    r.dp = double(cs2<D3Q27, Real>()) * (r.rho1 - r.rho2);
    r.drho = (r.rho1 - r.rho2) / r.rho2;
    if (g.walls) {
      r.dp0 = g.K * g.nu * 0.5 * (r.j1 + r.j2) * double(g.x_p2 - g.x_p1) /
              (double(g.L) * g.L);
      r.dp_ratio = r.dp / r.dp0;
    }
    double u2 = 0.0;
    for (Index n : fnodes) u2 = std::max(u2, ux[n] * ux[n] + uy[n] * uy[n] + uz[n] * uz[n]);
    r.umax = std::sqrt(u2);
    r.Ma_max = r.umax / std::sqrt(double(cs2<D3Q27, Real>()));
    double lo = 1e300, hi = -1e300;
    for (Index n : snodes) { lo = std::min(lo, c[n]); hi = std::max(hi, c[n]); }
    r.cmin = lo; r.cmax = hi;
    return r;
  }

  void print_spec(const char* indent = "  ") const {
    const double tau = 1.0 / omega_fluid(), wc = omega_scalar();
    std::printf("%sduct          L = %lld   grid %lld x %lld x %lld   design box %lld x %lld x %lld%s\n",
                indent, (long long)g.L, (long long)g.nx, (long long)g.ny, (long long)g.nz,
                (long long)(g.x_d1 - g.x_d0), (long long)(g.y1 - g.y0 + 1), (long long)g.nzd,
                g.walls ? "" : "   [no walls: periodic y, z]");
    std::printf("%sflow          U = %.5g  Ma = %.4f  Re = %.4g  nu = %.5g  tau = %.4f  omega_bulk = %.3g\n",
                indent, g.U, g.U * std::sqrt(3.0), sp.Re, g.nu, tau, sp.omega_bulk);
    std::printf("%sconcentration %s  Pe = %.4g  D = %.4e  omega_c = %.5f", indent, SColl::name,
                sp.Pe, g.D, wc);
    if constexpr (requires(SColl s) { s.Lambda; }) std::printf("  Lambda = %.4g", sp.Lambda);
    std::printf("  cell Pe = U/D = %.3g  layer after one pass ~ %.2f cells\n", g.U / g.D,
                std::sqrt(g.D * double(g.nx - 1) / g.U));
    if (sp.sharp)
      std::printf("%sdesign        SHARP: gamma < 0.5 is a halfway bounce-back wall for both "
                  "fields, no drag\n", indent);
    else
      std::printf("%sdesign        Da = %.3g  A_max = %.5g  RAMP q = %.3g  per-step drag factor"
                  " (2-A)/(2+A) = %.3f\n", indent, sp.Da, g.A_max, sp.q,
                  (2.0 - g.A_max) / (2.0 + g.A_max));
    std::printf("%splanes        pressure x = %lld, %lld   mixing x = %lld   (outlet x = %lld)\n",
                indent, (long long)g.x_p1, (long long)g.x_p2, (long long)g.x_m,
                (long long)(g.nx - 1));
  }
};

// The evaluator the plan uses, and the one it replaced, kept for comparison.
using MixScalarColl     = ScalarTRT<D3Q7>;
using MixerEvaluator    = MixerEvaluatorT<ScalarTRT<D3Q7>>;
using MixerEvaluatorReg = MixerEvaluatorT<ScalarRegularised<D3Q7>>;

}  // namespace topopt
}  // namespace lbm
