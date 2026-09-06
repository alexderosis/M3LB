//==============================================================================
//  Electroconvection in a CLOSED SQUARE CAVITY -- Sec. 3.2.3.
//
//  Patnaik, Skillen & De Rosis, Eng. Comput. 41:4977-5002 (2025), Sec. 3.2.3,
//  which reproduces the setup of Zhang et al. The same four equations as
//  validation/ehd_electroconvection.cpp,
//
//      grad^2 phi = -q/eps,   E = -grad phi,
//      d_t q + div[q(u + K E)] = D grad^2 q,
//      d_t u + u.grad u = -grad p/rho + nu grad^2 u + qE/rho,
//
//  in a square (A = 1, Lx == Ly) with EVERY wall closed. The reported quantity
//  is not a peak velocity but an electric Nusselt number, and the interesting
//  regime is high T -- Fig. 9 shows T = 250, 1000, 5000, 10 000.
//
//  ===================== WHY THIS IS A SEPARATE FILE =========================
//  Three things change from Sec. 3.2.1, and each of them changes what the case
//  can be trusted to say.
//
//  1. THE SIDE WALLS ARE REAL. Sec. 3.2.1 has free-slip sides, and this tree
//     runs it in a DOUBLED PERIODIC BOX precisely so that there is no lateral
//     boundary to discretise (see that file's banner, and CLAUDE.md's
//     "removing a boundary can beat discretising it"). Sec. 3.2.3 is
//
//         u = 0,   d_x q = 0,   d_x phi = 0     at x = 0, Lx,
//
//     which is NOT a mirror -- u = 0 is not u.n = 0 -- so the trick does not
//     apply and the lateral scalar wall must be built. The measurement that
//     made yesterday's electroconvection banner therefore governs this file:
//     omega_q -> 2 and a bounce-back scalar wall rings.
//
//  2. THE DIFFUSION NUMBER IS TIED TO T. Sec. 3.2 quotes alpha = 1e-4, and
//     Sec. 3.2.3 additionally fixes Sc = 1e3. Those are the same statement
//     only at T = 1000, and the identity says which one is the rule:
//
//         alpha = D/(K dphi),  Sc = nu/D,  nu = eps dphi/(K T),  eps = M^2 K^2
//         =>  alpha = eps/(K^2 T Sc) = M^2/(T Sc).
//
//     M = 10, Sc = 1e3 gives alpha = 0.1/T, i.e. 1e-4 at T = 1000 exactly.
//     Eq. (81) writes the diffusive flux with the coefficient M^2/(T Sc)
//     rather than with alpha, which settles it: Sc is the invariant across the
//     sweep and alpha is not. So alpha FOLLOWS T here, and `-alpha` overrides
//     only for a deliberate experiment.
//
//     The consequence is that high T is a low-diffusion problem: alpha = 1e-5
//     at T = 10 000, so omega_q sits at 1.9994 and the ghost-mode question
//     above is at its worst exactly where the physics is most interesting.
//
//  3. THE ANSWER IS AN INTEGRAL, NOT A PEAK. Eq. (81)-(82),
//
//         Ie = (1/V) int_V [ q (u_y + K E_y) - (M^2/(T Sc)) d_y q ] dV,
//         Ne = Ie / I0,
//
//     with I0 the same integral in the HYDROSTATIC state at the same forcing.
//     An integral is a far more forgiving diagnostic than a peak -- and that
//     cuts both ways, because it also hides a badly resolved boundary layer.
//
//  ===================== I0 IS MEASURED, NOT ASSUMED =========================
//  Every run computes its own I0 by solving the same problem with the Coulomb
//  force switched off and the seed removed, on the SAME grid, with the SAME
//  stencils and the SAME walls. That is deliberate: Ne is a ratio, so any
//  discretisation error common to the two states cancels out of it, and the
//  number that survives is the enhancement due to motion rather than the sum
//  of that and a grid error. Taking I0 from a formula instead would put the
//  grid error into Ne undivided.
//
//  The formula is still printed, as a CHECK on the numerical I0 rather than as
//  its source. In the hydrostatic steady state at D = 0 the current is uniform,
//  j = K q E with dE/dy = q/eps, so
//
//      E(y)^2 = E0^2 + 2 j y/(K eps),   E0 = j/(K q0),
//      int_0^H E dy = dphi,
//
//  and one bisection in j closes it. The two agree to O(alpha) plus the grid
//  error, and the run prints both so that a disagreement is visible rather
//  than absorbed.
//
//  ===================== THE SCALAR SIDE WALL, AND ITS ONE OPTION =============
//  d_x q = 0 and d_x phi = 0 want an on-node zero-flux condition. This tree
//  has two candidates and only one of them can be used here:
//
//    ScalarAdiabatic -- halfway bounce-back. It is UNUSABLE in this case, and
//      not for the ringing reason. `temperature()` REPORTS ZERO at an adiabatic
//      node (CLAUDE.md lists this trap; ScalarSolver's field_kernel does it
//      deliberately, because the node is a ghost outside the fluid and the
//      insulated plane is half a cell away). This case DIFFERENTIATES phi and
//      INTEGRATES q, both over the full domain including the wall column, so a
//      column of structural zeros would corrupt E_y at the wall, the Coulomb
//      force there, and Ie itself. A wall that is invisible to the diagnostic
//      is not an option that can be measured -- it is a wrong answer -- so it
//      is not offered.
//
//    ScalarOutflow -- on-node, zero-gradient, and it reports the real value.
//      Used for both fields. In Sec. 3.2.1 this bled charge (q/q0 reached
//      -0.165) and was rejected; the reason it is sound HERE is the difference
//      in the wall itself. ScalarOutflow prescribes the node's populations as
//      equilibrium at the donor's value and THE NODE'S OWN VELOCITY, so it
//      passes whatever normal flux that velocity carries. At a Sec. 3.2.1
//      symmetry plane the flow slides ALONG the plane and the drift has a
//      normal component; here u = 0 and E_x = 0 on the wall, so the normal
//      drift is identically zero and the condition degenerates to exactly
//      d_x q = 0 with no flux through it. The physical boundary is stronger,
//      so the same numerical device is safe. Every run reports the charge
//      bounds, which is the instrument that would catch it if this were wrong.
//
//  ===================== E AT THE WALLS ======================================
//  E_x is set to ZERO in the two side columns rather than differenced there.
//  That IS the boundary condition -- d_x phi = 0 -- and it is exact, whereas a
//  one-sided stencil applied to a copied value gives 0.5(phi_1 - phi_2), i.e.
//  the boundary condition plus a truncation error of the interior gradient.
//  E_y uses the same one-sided second-order stencils at the plates as the
//  hydrostatic case, for the reason CLAUDE.md records: E is a DERIVATIVE of the
//  field carrying the boundary value, so on-node plates are what let the
//  stencil see the imposed potential at all.
//
//  ===================== THE FORCE IS ZEROED ON THE WALL NODES ================
//  Regularised walls are fluid nodes: they collide, and they would be forced.
//  CLAUDE.md's zhou_thermal entry is the same trap wearing a different field
//  (a Boussinesq force applied along a RegWall reading T = 0). Here q and E are
//  both correct at the wall, so the force there would be the RIGHT force -- but
//  it is applied at a node whose velocity is PRESCRIBED to zero by the
//  regularised closure, so it cannot do anything except fight that closure.
//  F = 0 on the four boundary lines; the interior is untouched.
//
//  ===================== WHAT LIMITS THE GRID, AND IT IS NOT MEMORY ===========
//  The paper uses 500 x 500 and this file defaults to 129 x 129, so the gap has
//  to be stated in units of the physics rather than apologised for.
//
//  The velocity scale is NOT u0 at high T. u0 = K dphi/H is the DRIFT speed;
//  the free-fall speed of a charge layer of strength q0 falling through its own
//  field is u_f = sqrt(q0 E H/rho) = (dphi/H) M K sqrt(C), i.e.
//
//      u_f / u0 = M sqrt(C) = 31.6      at M = 10, C = 10,
//
//  independent of T. At T = 190 viscosity holds u_max/u0 to 3.7 (Re = T/M^2 =
//  1.9); by T = 10 000, Re = 100 and a good fraction of u_f is available. Two
//  consequences, and they pull against each other:
//
//    * u0 MUST FALL AS T RISES, because the Mach number is set by u_max, not
//      by u0. `-u0` defaults to 5e-3 and every run prints its own peak Mach.
//    * THE CELL REYNOLDS NUMBER IS THE REAL CEILING. nu = M^2 u0 H/T, so
//
//          Re_cell = u_max / nu = (u_max/u0) T / (M^2 H),
//
//      which is INDEPENDENT of u0 -- lowering u0 buys Mach number and buys
//      nothing at all here. At u_max/u0 = 15 that is 0.15 T/H: about 12 at
//      T = 10 000 on H = 128, and about 3 on the paper's H = 499. So 500^2 is
//      not a comfort margin in Sec. 3.2.3, it is what the case needs, and a
//      129^2 run at T = 10 000 is an under-resolved run that must be labelled
//      one. Every run prints Re_cell from its own measured u_max.
//
//  ===================== THE SEED ============================================
//  Eqs. (17)-(20) start from rest with q = 0. The Rayleigh-Benard work in this
//  tree established both halves of the lesson: a deterministic symmetric code
//  will hold a symmetric state indefinitely, and a seed in the wrong PLACE
//  decays instead of growing (CLAUDE.md, "a cold start's seed has to go where
//  the gradient is"). The gradient here is the injection layer at the bottom
//  plate, so the seed is a non-negative charge perturbation there, decaying
//  over H/8.
//
//  It is BROADBAND -- four lateral modes at fixed, deliberately incommensurate
//  phases -- and that is a change from Sec. 3.2.1, where a single mode was
//  right because A = 0.614 selects one. A square cavity at high T has no
//  preferred wavelength to select, and Fig. 9 is a picture of ASYMMETRIC
//  plumes; seeding one mode would be choosing the pattern in advance. Non-
//  negative for the usual reason: q must stay in [0, q0] and an initial state
//  outside the bound it is judged by is not a starting point.
//
//  ===================== WHAT WAS MEASURED (2026-09-05/06) ======================
//  N = 129, u0 = 5e-3, 40 t0 with the last 15 averaged, UNSEEDED -- which is
//  the reference's own protocol, Eqs. (17)-(20) starting from exactly zero:
//
//      T        250    500   1000   1500   3000   5000  10000
//      Ne      1.000  1.595  1.696  1.751  1.979  2.597  3.381
//      Fig. 8   1.03   1.75   1.88   1.93   2.35   2.80   3.20
//      diff    -2.9%  -8.9%  -9.8%  -9.3% -15.8%  -7.2%  +5.7%
//      Ma      .001   .030   .048   .054   .087   .138   .163
//      Re_cell  0.0    0.1    0.4    0.7    2.3    6.2   14.7
//
//  The SHAPE is reproduced -- hydrostatic below onset, a steep rise, a shallow
//  plateau, renewed growth -- with kappa = +0.35 over 1500 -> 1e4 against
//  Fig. 8's own points at 0.27 and its drawn guide line at 0.50. The LEVEL sits
//  9-13 % low wherever the grid is trustworthy.
//
//  The seeded sweep, kept because it is what produced the wrong reading first:
//
//      T        250    500   1000   1500   3000   5000  10000
//      Ne      1.392  1.547  1.680  1.765  2.019  2.802  3.358
//      Ma      .017   .031   .044   .065   .078   .136   .183
//      Re_cell  0.0    0.1    0.4    0.9    2.1    6.1   16.5
//
//  READ THE LAST TWO COLUMNS AGAINST THE LAST TWO ROWS BEFORE READING THEM
//  AGAINST Fig. 8. T = 5000 lands within 0.1 % of the reference and T = 10000
//  within 4.9 %, and those are the only two runs in the table that break this
//  tree's own rules -- Ma 0.136 and 0.183 against a 0.087 guideline, cell
//  Reynolds 6.1 and 16.5 where the reference's H = 499 gives 1.6 and 4.2. Every
//  WELL-RESOLVED point sits 9-14 % low. When the badly-behaved runs agree and
//  the well-behaved ones do not, the agreement is the thing to distrust.
//
//  THE RESOLUTION LADDER SETTLED IT. At T = 5000: 2.8558 at N = 81, 2.8017 at
//  N = 129, 2.4903 at N = 201 (2.4366 unseeded), with Re_cell 8.5, 6.1, 3.6.
//  Monotonically falling, so N = 129 was a coarse grid PASSING THROUGH the
//  published value on its way down; the converged deficit is about 13 %, which
//  is what the well-resolved mid-range already showed. Halving u0 at N = 129
//  moves 2.8017 to 2.7104, so compressibility is a 3 % term and not the cause.
//
//  THE SEED CHOOSES A BRANCH. This is a subcritical bifurcation and the usual
//  seed check does not see it:
//
//      T = 250    amp 1e-2   amp 1e-4    amp 0     Fig. 8
//      Ne           1.3922     1.2400   1.0003       1.03
//      u_max/u0      2.000      1.990    0.078
//
//  Two decades of seed give the same saturated amplitude -- which is exactly
//  what "the seed sets the transient, not the answer" looks like -- and zero
//  seed gives no convection at all. Eqs. (17)-(20) start from exactly zero, so
//  the reference's only seed is round-off, and `-amp 0` is the protocol that
//  reproduces it. At T = 500 the unseeded run is also STEADY (rms 0.0007
//  against 0.107) and closer to the reference, 1.6583 against 1.75: the seed
//  was not merely triggering the instability there, it was selecting a
//  different and worse attractor. CLAUDE.md carries both rules.
//
//  SO THE DEFICIT IS MEASURED AND NOT EXPLAINED. Resolution and Mach number are
//  both excluded as the whole cause, the seed is excluded, and the wall columns
//  contribute under 1 % (every run prints the interior-only Ne beside the full
//  one). No third candidate has been tested. The reference's 500^2 grid is what
//  Re_cell says T >= 5000 actually needs and is about five hours per T here.
//
//  ===================== WHAT THIS IS CHECKED AGAINST =========================
//  Fig. 8 is a PLOT, not a table, so this case cannot be scored to a digit. It
//  checks the three statements the text makes about that plot:
//
//      T < 250      substantially hydrostatic, Ne ~ 1
//      T <~ 1500    saturation, kappa ~ 0 in Ne ~ T^kappa
//      T > 1500     renewed growth, kappa ~ 1/2
//
//  and it fits kappa over whatever T are given, so the exponent is measured
//  rather than asserted. It is a DEMONSTRATOR-grade case in the second CMake
//  list: a single T at the default grid is minutes, and a sweep is hours.
//
//    usage: ehd_cavity [-n N] [-t T[,T,...]] [-u0 U] [-c C] [-m M] [-sc SC]
//                      [-alpha A] [-beta B] [-pois K] [-tf N] [-tfh N]
//                      [-tavg N]
//                      [-amp A] [-side spec|out|adia|per]
//                      [-dump PREFIX] [-dumpn K] [-dumpevery K] [-lat 2d|3d] [-watch]
//                      [--kokkos-num-threads=4]
//==============================================================================
#include "collision/ChargeCentralMoments.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "boundary/Regularized.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"
#include "FieldDump.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
//  Same stack as Sec. 3.2.1, and deliberately so: if the cavity needed a
//  different operator from the channel, one of the two would be wrong.
//------------------------------------------------------------------------------
template <class FL, class SL> struct Stack {
  using FluidOp   = MomentCollision<FL, FieldGuo, ShiftedPopulations, true>;
  using ChargeOp  = ChargeCentralMoments<FL>;
  using PotOp     = ScalarBGK<SL>;
  using FluidSol  = FluidSolver<FL, EsotericPull<FL>, FluidOp>;
  using ChargeSol = ScalarSolver<FL, EsotericPull<FL>, ChargeOp>;
  using PotSol    = ScalarSolver<SL, EsotericPull<SL>, PotOp>;
};

struct Opts {
  Index n = 129;              // N x N; H = n - 1, the paper's is 500
  double u0 = 5e-3;           // see the banner: u_max is 10-20x this at high T
  double C = 10.0, M = 10.0;  // Sec. 3.2
  double Sc = 1e3;            // Sec. 3.2.3; alpha = M^2/(T Sc) follows
  double alpha = -1.0;        // -alpha: override the Sc rule
  double beta = 0.3;          // Poisson pseudo-diffusivity
  int    pois = 1;            // Poisson sub-steps per LB step
  double tf = 30.0;           // total, in t0
  double tfh = 40.0;          // cap for the hydrostatic reference, in t0
  double tavg = 10.0;         // averaging window at the END, in t0
  double amp = 1e-2;          // seed, in units of q0
  // LATERAL SCALAR WALL. `per` and `adia` are DIAGNOSTICS, not options:
  // periodic is not the paper's problem at all, and adiabatic reports ZERO
  // at its own node so it corrupts E and Ie. They exist to attribute a
  // failure to the wall rather than to the operator.
  int    side = 3;            // 3 specular (on-node zero flux) -- the default
                              // 0 out, 1 adiabatic, 2 periodic: diagnostics
  bool   watch = false;
  bool   d3 = false;
  int    dumpn = 20;          // -dump: write one frame every dumpn probes
  // FRAMES FOR AN ANIMATION NEED THEIR OWN CADENCE. The probe interval is
  // t0/20, and at high T that is roughly ONE EDDY TURNOVER -- H/u_max is
  // t0/(u_max/u0), which is t0/21 at T = 10^4. A film sampled once per
  // turnover does not show the turnover. `-dumpevery K` writes a frame
  // every K steps regardless of the probes, so the sampling can be set
  // from the flow's own timescale rather than from the diagnostic's.
  std::size_t dumpevery = 0;  // 0 = follow the probes
  std::string dump;
};

struct Out {
  double Ie = 0;              // volume-averaged vertical charge flux
  double Ie_int = 0;          // the same, over INTERIOR cells only
  double Ne = 0, Ne_rms = 0;  // Ie/I0, time-averaged over the last tavg
  double Ne_int = 0;          // the same from Ie_int/I0_int
  double umax = 0;            // normalised by u0
  double recell = 0;          // u_max_lattice / nu
  double qlo = 0, qhi = 0;
  double worst = 0, t_worst = 0;
  double drift = 0;
  double tend = 0;
  int    nacc = 0;
  bool   finite = true;
};

//------------------------------------------------------------------------------
//  The D = 0 hydrostatic current, by bisection. A CHECK on the numerical I0,
//  not a substitute for it -- see the banner.
//------------------------------------------------------------------------------
static double analytic_current(double K, double eps, double q0, double dphi,
                               double H) {
  auto integral = [&](double j) {
    const double E0 = j / (K * q0);
    const double a  = 2.0 * j / (K * eps);
    return (2.0 / (3.0 * a)) * (std::pow(E0 * E0 + a * H, 1.5) - E0 * E0 * E0);
  };
  double lo = 1e-300, hi = 1.0;
  for (int i = 0; i < 200 && integral(hi) < dphi; ++i) hi *= 2.0;
  for (int i = 0; i < 300; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (integral(mid) < dphi) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

//------------------------------------------------------------------------------
//  One cavity run. `hydro` freezes the fluid and drops the seed, which is the
//  reference state for Ne; otherwise the Coulomb force is live and I0 is the
//  divisor.
//------------------------------------------------------------------------------
template <class FL, class SL>
static Out solve(const Opts& o, double Tel, bool hydro, double I0, double I0i,
                 bool verbose) {
  using S         = Stack<FL, SL>;
  using FluidOp   = typename S::FluidOp;
  using ChargeOp  = typename S::ChargeOp;
  using PotOp     = typename S::PotOp;
  using FluidSol  = typename S::FluidSol;
  using ChargeSol = typename S::ChargeSol;
  using PotSol    = typename S::PotSol;

  const Index ny = o.n;
  const Index H  = ny - 1;                       // on-node plates at 0 and H
  const Index nx = o.n;
  const Index nz = 1;

  const double dphi = 1.0, rho0 = 1.0;
  const double u0   = o.u0;
  const double Kmob = u0 * double(H) / dphi;
  const double eps  = o.M * o.M * Kmob * Kmob * rho0;
  const double nu   = eps * dphi / (Kmob * Tel);
  const double q0   = eps * o.C * dphi / (double(H) * double(H));
  const double alph = o.alpha > 0.0 ? o.alpha : o.M * o.M / (Tel * o.Sc);
  const double Dq   = alph * Kmob * dphi;
  const double t0   = double(H) / u0;

  Out r;
  if (verbose) {
    std::printf("  %s  T = %g   %lld x %lld  (H = %lld)   u0 = %.4g\n",
                hydro ? "reference (hydrostatic, F = 0)" : "cavity",
                Tel, (long long)nx, (long long)ny, (long long)H, u0);
    std::printf("    K = %.5g  eps = %.5g  nu = %.5g (tau = %.5f)  q0 = %.5g\n",
                Kmob, eps, nu, 3.0 * nu + 0.5, q0);
    std::printf("    alpha = %.4g %s  D = %.4g (omega_q = %.6f)   Re = T/M^2 = %g"
                "   t0 = %.0f steps\n", alph,
                o.alpha > 0.0 ? "(-alpha)" : "= M^2/(T Sc)", Dq,
                double(ChargeOp::omega_from_diffusivity(Real(Dq))),
                Tel / (o.M * o.M), t0);
    std::fflush(stdout);
  }

  Domain d(nx, ny, nz, /*periodic x*/ o.side == 2, false, true);

  // ---- the fluid: a closed box, on-node walls, NrmCorner at the corners ----
  View1D<Real> Fx("Fx", d.n_padded), Fy("Fy", d.n_padded), Fz("Fz", d.n_padded);
  FluidOp fcoll;
  fcoll.omega = FluidOp::omega_from_viscosity(Real(nu));
  fcoll.omega_bulk = Real(1);
  fcoll.forcing.Ex = Fx;  fcoll.forcing.Ey = Fy;  fcoll.forcing.Ez = Fz;
  FluidSol fl(d, fcoll);
  fl.set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
  using WS = typename FluidSol::WallSpec;
  fl.set_regularized_walls([&](Index x, Index y, Index) -> WS {
    const bool xl = (x == 0) && o.side != 2, xr = (x == nx - 1) && o.side != 2;
    const bool yb = (y == 0), yt = (y == ny - 1);
    if ((xl || xr) && (yb || yt))
      return WS{NrmCorner, Real(0), Real(0), Real(0)};
    if (yb) return WS{NrmYm, Real(0), Real(0), Real(0)};
    if (yt) return WS{NrmYp, Real(0), Real(0), Real(0)};
    if (xl) return WS{NrmXm, Real(0), Real(0), Real(0)};
    if (xr) return WS{NrmXp, Real(0), Real(0), Real(0)};
    return WS{};
  });
  fl.initialize(Real(rho0));

  // ---- the charge ---------------------------------------------------------
  ChargeOp ccoll;
  ccoll.omega = ChargeOp::omega_from_diffusivity(Real(Dq));
  ChargeSol chg(d, ccoll);
  chg.set_geometry([&](Index x, Index y, Index) -> ScalarCell {
    if (y == 0)      return ScalarMoment;                 // injector, Eq. (12)
    if (y == ny - 1) return ScalarOutflow;                // d_y q = 0, Eq. (13)
    if (o.side != 2 && (x == 0 || x == nx - 1))
      return o.side == 3 ? ScalarSpecular
           : o.side == 1 ? ScalarAdiabatic : ScalarOutflow;   // d_x q = 0
    return ScalarBulk;
  });
  chg.set_wall_values([&](Index, Index, Index) -> Real { return Real(q0); });

  // ---- the potential ------------------------------------------------------
  PotOp pcoll;
  pcoll.omega = PotOp::omega_from_diffusivity(Real(o.beta));
  pcoll.T_ref = Real(0);
  PotSol pot(d, pcoll);
  pot.set_geometry([&](Index x, Index y, Index) -> ScalarCell {
    if (y == 0 || y == ny - 1) return ScalarMoment;       // phi0 = 1, phi1 = 0
    if (o.side != 2 && (x == 0 || x == nx - 1))
      return o.side == 3 ? ScalarSpecular
           : o.side == 1 ? ScalarAdiabatic : ScalarOutflow;   // d_x phi = 0
    return ScalarBulk;
  });
  pot.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? Real(dphi) : Real(0);
  });

  if (o.side == 3) {
    auto nrm = [&](Index x, Index, Index) -> std::uint8_t {
      if (x == 0)      return NrmXm;
      if (x == nx - 1) return NrmXp;
      return NrmNone;
    };
    chg.set_specular_walls(nrm);
    pot.set_specular_walls(nrm);
  }

  // ---- initial state ------------------------------------------------------
  const Index Hc = H, nxc = nx;
  const double dp = dphi;
  pot.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Index y = py - d.hy;
    const double yy = double(y) < 0.0 ? 0.0
                    : (double(y) > double(Hc) ? double(Hc) : double(y));
    return Real(dp * (1.0 - yy / double(Hc)));
  });
  // The seed. Four modes, phases chosen once and fixed, each contributing at
  // most a quarter -- so the sum is in [0, 1] and q stays non-negative.
  const Real ampq = Real((hydro ? 0.0 : o.amp) * q0);
  const double dec = double(H) / 8.0;
  chg.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    const Index x = px - d.hx, y = py - d.hy;
    if (y <= 0 || y >= Hc) return Real(0);
    const double xs = (double(x) + 0.5) / double(nxc);
    double lat = 0.0;
    const double ph[4] = {0.0, 1.1, 2.3, 0.7};
    for (int m = 0; m < 4; ++m)
      lat += 0.25 * 0.5 * (1.0 + Kokkos::cos(M_PI * double(m + 1) * xs + ph[m]));
    return Real(double(ampq) * lat * Kokkos::exp(-double(y) / dec));
  });
  pot.finalize_geometry();
  chg.finalize_geometry();
  pot.compute_field();
  chg.compute_field();
  fl.compute_macroscopic();

  View1D<Real> kx("kx", d.n_padded), ky("ky", d.n_padded), kz("kz", d.n_padded);
  View1D<Real> qprev("qprev", d.n_padded);
  chg.set_velocity(kx, ky, kz);

  auto phi = pot.temperature();
  auto qf  = chg.temperature();
  auto ux  = fl.ux();
  auto uy  = fl.uy();

  const int NR = 20;
  const std::size_t probe = std::size_t(t0 / NR) ? std::size_t(t0 / NR) : 1;
  // The reference gets its own clock. It stops on a converged Ie, not on the
  // main run's tf -- tying the two together silently truncates I0 when a
  // short cavity run is asked for, and every Ne then inherits the error.
  const std::size_t steps = std::size_t((hydro ? o.tfh : o.tf) * t0);
  const double tavg0 = o.tf - o.tavg;
  double sNe = 0.0, sNe2 = 0.0, sNi = 0.0, ring[NR] = {0};
  int nprobe = 0, frame = 0;
  const bool isH = hydro;
  const int  sidec = o.side;
  const int npois = o.pois;

  for (std::size_t t = 0; t < steps; ++t) {
    if (!isH) fl.compute_macroscopic();

    const double Km = Kmob;
    Kokkos::parallel_for("E_drift_force", Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index y = py - d.hy, x = px - d.hx;
      kx(n) = ky(n) = kz(n) = Real(0);
      Fx(n) = Fy(n) = Fz(n) = Real(0);
      if (y < 0 || y > Hc || x < 0 || x >= nxc) return;
      double Ey;
      if (y == 0)
        Ey = -(-1.5 * double(phi(n)) + 2.0 * double(phi(d.id(x, y + 1, 0)))
               - 0.5 * double(phi(d.id(x, y + 2, 0))));
      else if (y == Hc)
        Ey = -(1.5 * double(phi(n)) - 2.0 * double(phi(d.id(x, y - 1, 0)))
               + 0.5 * double(phi(d.id(x, y - 2, 0))));
      else
        Ey = -0.5 * (double(phi(d.id(x, y + 1, 0))) - double(phi(d.id(x, y - 1, 0))));
      // d_x phi = 0 IS the side-wall condition, so E_x there is exactly zero.
      const double Ex = (sidec != 2 && (x == 0 || x == nxc - 1))
                      ? 0.0
                      : -0.5 * (double(phi(d.id((x + 1) % nxc, y, 0)))
                                - double(phi(d.id((x - 1 + nxc) % nxc, y, 0))));
      const double qn = double(qf(n));
      const bool wall = (sidec != 2 && (x == 0 || x == nxc - 1))
                        || y == 0 || y == Hc;
      if (!wall && !isH) {           // see the banner: a RegWall must not be forced
        Fx(n) = Real(qn * Ex);
        Fy(n) = Real(qn * Ey);
      }
      kx(n) = Real(Km * Ex + (isH ? 0.0 : double(ux(n))));
      ky(n) = Real(Km * Ey + (isH ? 0.0 : double(uy(n))));
    });
    Kokkos::fence();

    const Real bo = Real(o.beta / eps);
    for (int k = 0; k < npois; ++k) {
      pot.add_source(KOKKOS_LAMBDA(Index n) {
        return bo * (Real(1.5) * qf(n) - Real(0.5) * qprev(n));
      });
      pot.step();
    }
    Kokkos::deep_copy(qprev, qf);

    chg.step();
    if (!isH) fl.step();
    pot.compute_field();
    chg.compute_field();

    // ---- frames on their own clock ----------------------------------------
    if (!o.dump.empty() && !isH && o.dumpevery && (t + 1) % o.dumpevery == 0) {
      fl.compute_macroscopic();
      auto dux = Kokkos::create_mirror_view_and_copy(HostSpace{}, ux);
      auto duy = Kokkos::create_mirror_view_and_copy(HostSpace{}, uy);
      auto dq  = Kokkos::create_mirror_view_and_copy(HostSpace{}, qf);
      char tag[32];
      std::snprintf(tag, sizeof tag, "_%04d.bin", frame++);
      figdump::scalar_slice(o.dump + "_q" + tag, nx, ny, [&](Index xi, Index y) {
        return double(dq(d.id(xi, y, 0))) / q0;
      });
      figdump::scalar_slice(o.dump + "_u" + tag, nx, ny, [&](Index xi, Index y) {
        const Index m = d.id(xi, y, 0);
        return std::sqrt(double(dux(m)) * double(dux(m)) +
                         double(duy(m)) * double(duy(m))) / u0;
      });
    }

    if ((t + 1) % probe == 0 || t + 1 == steps) {
      if (!isH) fl.compute_macroscopic();
      auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, ux);
      auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, uy);
      auto hq  = Kokkos::create_mirror_view_and_copy(HostSpace{}, qf);
      auto hky = Kokkos::create_mirror_view_and_copy(HostSpace{}, ky);
      double peak = 0.0, qlo = 1e300, qhi = -1e300, sflux = 0.0, sint = 0.0;
      long nbad = 0, ncell = 0, nint = 0;
      for (Index y = 0; y <= H; ++y)
        for (Index x = 0; x < nx; ++x) {
          const Index n = d.id(x, y, 0);
          const double a = double(hux(n)), b = double(huy(n)), q = double(hq(n));
          if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(q)) { ++nbad; continue; }
          const double s = std::sqrt(a * a + b * b);
          if (s > peak) peak = s;
          if (q < qlo) qlo = q;
          if (q > qhi) qhi = q;
          const double dr = std::abs(double(hky(n)));
          if (dr > r.drift) r.drift = dr;
          // Eq. (81), with d_y q by the same second-order stencils as E.
          double dqdy;
          if (y == 0)
            dqdy = -1.5 * q + 2.0 * double(hq(d.id(x, y + 1, 0)))
                            - 0.5 * double(hq(d.id(x, y + 2, 0)));
          else if (y == H)
            dqdy =  1.5 * q - 2.0 * double(hq(d.id(x, y - 1, 0)))
                            + 0.5 * double(hq(d.id(x, y - 2, 0)));
          else
            dqdy = 0.5 * (double(hq(d.id(x, y + 1, 0)))
                          - double(hq(d.id(x, y - 1, 0))));
          const double fl_ = q * double(hky(n)) - Dq * dqdy;
          sflux += fl_;  ++ncell;
          // The same integral with the four boundary LINES removed. The
          // difference between the two is the wall columns' contribution,
          // and it is reported rather than chosen between: an on-node
          // zero-gradient wall is only first-order accurate, so its cells
          // carry an O(1) local error that a volume average dilutes as
          // 2/N and nothing else removes. MEASURED by shrinking N -- see
          // the banner's table.
          if (x > 0 && x < nx - 1 && y > 0 && y < H) { sint += fl_; ++nint; }
        }
      if (nbad) {
        r.finite = false;
        std::printf("      NON-FINITE at t/t0 = %.2f\n", double(t + 1) / t0);
        break;
      }
      r.Ie = sflux / double(ncell);
      r.Ie_int = sint / double(nint);
      r.umax = peak / u0;  r.qlo = qlo / q0;  r.qhi = qhi / q0;
      r.tend = double(t + 1) / t0;
      r.recell = peak / nu;
      const double exc = (r.qlo < 0.0) ? -r.qlo : (r.qhi > 1.0 ? r.qhi - 1.0 : 0.0);
      if (exc > r.worst) { r.worst = exc; r.t_worst = r.tend; }

      const double ne = I0 > 0.0 ? r.Ie / I0 : 0.0;
      if (!isH && r.tend >= tavg0) {
        sNe += ne;  sNe2 += ne * ne;  ++r.nacc;
        sNi += I0i > 0.0 ? r.Ie_int / I0i : 0.0;
      }

      if (!o.dump.empty() && !isH && !o.dumpevery && (nprobe % o.dumpn) == 0) {
        char tag[32];
        std::snprintf(tag, sizeof tag, "_%04d.bin", frame++);
        figdump::scalar_slice(o.dump + "_q" + tag, nx, ny, [&](Index xi, Index y) {
          return double(hq(d.id(xi, y, 0))) / q0;
        });
        figdump::scalar_slice(o.dump + "_u" + tag, nx, ny, [&](Index xi, Index y) {
          const Index m = d.id(xi, y, 0);
          return std::sqrt(double(hux(m)) * double(hux(m)) +
                           double(huy(m)) * double(huy(m))) / u0;
        });
      }

      if (o.watch)
        std::printf("      t/t0 %7.2f   u_max/u0 = %8.3f   Ie = %.6e   Ne = %7.4f"
                    "   q/q0 [%7.4f, %7.4f]   Re_c = %5.1f\n", r.tend, r.umax,
                    r.Ie, ne, r.qlo, r.qhi, r.recell);

      // The hydrostatic reference has a steady state and stops at it; the
      // cavity at high T does not, and runs the clock out on purpose.
      const double ago = ring[nprobe % NR];
      ring[nprobe % NR] = r.Ie;
      ++nprobe;
      if (isH && nprobe > NR && std::abs(r.Ie - ago) < 1e-6 * std::abs(r.Ie)) break;
      std::fflush(stdout);
    }
  }
  if (!isH && r.nacc) {
    r.Ne = sNe / r.nacc;
    const double v = sNe2 / r.nacc - r.Ne * r.Ne;
    r.Ne_rms = v > 0.0 ? std::sqrt(v) : 0.0;
    r.Ne_int = sNi / r.nacc;
  }
  return r;
}

//------------------------------------------------------------------------------
template <class FL, class SL>
static void run_one(const Opts& o, double Tel, double& NeOut, double& umaxOut,
                    double& recOut) {
  const Index H = o.n - 1;
  const double Kmob = o.u0 * double(H);
  const double eps  = o.M * o.M * Kmob * Kmob;
  const double q0   = eps * o.C / (double(H) * double(H));

  const Out h = solve<FL, SL>(o, Tel, /*hydro*/true, 0.0, 0.0, true);
  const double ja = analytic_current(Kmob, eps, q0, 1.0, double(H));
  std::printf("    I0 = %.6e  (D=0 analytic %.6e, %+.2f %%)   interior-only"
              " %.6e (%+.2f %%)   converged at t/t0 = %.2f\n\n", h.Ie, ja,
              100.0 * (h.Ie - ja) / ja, h.Ie_int,
              100.0 * (h.Ie_int - ja) / ja, h.tend);
  std::fflush(stdout);

  const Out f = solve<FL, SL>(o, Tel, /*hydro*/false, h.Ie, h.Ie_int, true);
  std::printf("    u_max/u0 = %.3f   (Ma = %.3f, Re_cell = %.1f)   peak drift"
              " |u + K E| = %.4f\n", f.umax, f.umax * o.u0 * std::sqrt(3.0),
              f.recell, f.drift);
  std::printf("    Ne = %.4f +/- %.4f  (%d samples over the last %.0f t0)"
              "   interior-only Ne = %.4f (%+.2f %%)\n", f.Ne, f.Ne_rms,
              f.nacc, o.tavg, f.Ne_int,
              f.Ne > 0.0 ? 100.0 * (f.Ne_int - f.Ne) / f.Ne : 0.0);
  std::printf("    q/q0 in [%.4f, %.4f]", f.qlo, f.qhi);
  if (f.worst > 0.0)
    std::printf("   worst excursion %.4f q0 at t/t0 = %.2f%s\n", f.worst,
                f.t_worst, (f.qlo >= -1e-9 && f.qhi <= 1.0 + 1e-9)
                           ? "  (recovered)" : "  (STILL OUT)");
  else
    std::printf("   stayed inside [0, q0] throughout\n");
  if (f.recell > 5.0)
    std::printf("    UNDER-RESOLVED: Re_cell = %.1f. The paper's H = 499 gives"
                " %.1f at this T.\n", f.recell, f.recell * double(o.n - 1) / 499.0);
  std::printf("\n");
  std::fflush(stdout);
  NeOut = f.finite ? f.Ne : std::nan("");
  umaxOut = f.umax;
  recOut = f.recell;
}

int main(int argc, char** argv) {
  Opts o;
  std::vector<double> Ts;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "-n"     && i + 1 < argc) o.n     = Index(std::atol(argv[++i]));
    else if (a == "-u0"    && i + 1 < argc) o.u0    = std::atof(argv[++i]);
    else if (a == "-c"     && i + 1 < argc) o.C     = std::atof(argv[++i]);
    else if (a == "-m"     && i + 1 < argc) o.M     = std::atof(argv[++i]);
    else if (a == "-sc"    && i + 1 < argc) o.Sc    = std::atof(argv[++i]);
    else if (a == "-alpha" && i + 1 < argc) o.alpha = std::atof(argv[++i]);
    else if (a == "-beta"  && i + 1 < argc) o.beta  = std::atof(argv[++i]);
    else if (a == "-pois"  && i + 1 < argc) o.pois  = std::atoi(argv[++i]);
    else if (a == "-tf"    && i + 1 < argc) o.tf    = std::atof(argv[++i]);
    else if (a == "-tfh"   && i + 1 < argc) o.tfh   = std::atof(argv[++i]);
    else if (a == "-tavg"  && i + 1 < argc) o.tavg  = std::atof(argv[++i]);
    else if (a == "-amp"   && i + 1 < argc) o.amp   = std::atof(argv[++i]);
    else if (a == "-dump"  && i + 1 < argc) o.dump  = argv[++i];
    else if (a == "-dumpn" && i + 1 < argc) o.dumpn = std::max(1, std::atoi(argv[++i]));
    else if (a == "-dumpevery" && i + 1 < argc)
      o.dumpevery = std::size_t(std::atol(argv[++i]));
    else if (a == "-lat"   && i + 1 < argc) o.d3 = (std::string(argv[++i]) == "3d");
    else if (a == "-side"  && i + 1 < argc) {
      const std::string v = argv[++i];
      o.side = (v == "adia") ? 1 : (v == "per") ? 2 : (v == "out") ? 0 : 3;
    }
    else if (a == "-watch")                 o.watch = true;
    else if (a == "-t"     && i + 1 < argc) {
      std::string s = argv[++i];
      std::size_t p = 0;
      while (p < s.size()) {
        const std::size_t c = s.find(',', p);
        Ts.push_back(std::atof(s.substr(p, c - p).c_str()));
        if (c == std::string::npos) break;
        p = c + 1;
      }
    }
  }
  if (Ts.empty()) Ts.push_back(1000.0);

  Kokkos::initialize(argc, argv);
  {
    std::printf("Closed square EHD cavity -- Patnaik, Skillen & De Rosis (2025)"
                " Sec. 3.2.3\n");
    std::printf("  %s fluid + charge / %s potential   %s   C = %g, M = %g,"
                " Sc = %g\n\n", o.d3 ? "D3Q27" : "D2Q9", o.d3 ? "D3Q7" : "D2Q5",
                sizeof(Real) == 4 ? "FP32" : "FP64", o.C, o.M, o.Sc);

    std::vector<double> Ne(Ts.size()), um(Ts.size()), rc(Ts.size());
    for (std::size_t i = 0; i < Ts.size(); ++i) {
      if (o.d3) run_one<D3Q27, D3Q7>(o, Ts[i], Ne[i], um[i], rc[i]);
      else      run_one<D2Q9,  D2Q5>(o, Ts[i], Ne[i], um[i], rc[i]);
    }

    // ================= WHAT THIS IS COMPARED AGAINST ====================
    // Fig. 8 is a PLOT, not a table, so these are DIGITISED FROM IT BY EYE off
    // a log-log axis. That is an honest +/- 0.05 on the plateau and +/- 0.1 at
    // the top, and it is why nothing here is scored: a number read off a figure
    // cannot be a pass/fail threshold. It is still worth carrying, because the
    // SHAPE -- hydrostatic, rise, plateau, renewed growth -- is what the case
    // is really being asked to reproduce, and a shape can be checked against a
    // figure when a digit cannot.
    struct Ref { double T, Ne; };
    static const Ref FIG8[] = {
      {250, 1.03}, {300, 1.50}, {500, 1.75}, {1000, 1.88}, {1500, 1.93},
      {2000, 1.97}, {3000, 2.35}, {5000, 2.80}, {10000, 3.20},
    };
    std::printf("  ---- summary -------------------------------------------\n");
    std::printf("      T        Ne     u_max/u0   Re_cell   Fig. 8 (digitised)\n");
    for (std::size_t i = 0; i < Ts.size(); ++i) {
      double ref = 0.0;
      for (const Ref& r : FIG8)
        if (std::abs(r.T - Ts[i]) < 0.02 * r.T) ref = r.Ne;
      std::printf("  %7.0f   %7.4f   %8.3f   %7.1f", Ts[i], Ne[i], um[i], rc[i]);
      if (ref > 0.0) std::printf("   %5.2f  (%+6.1f %%)", ref,
                                 100.0 * (Ne[i] - ref) / ref);
      else           std::printf("       --            ");
      std::printf("%s\n", rc[i] > 5.0 ? "  under-resolved" : "");
    }
    // kappa in Ne ~ T^kappa, over consecutive pairs. Fig. 8's text claims
    // kappa ~ 0 up to T ~ 1500 and kappa ~ 1/2 beyond it.
    if (Ts.size() > 1) {
      // THE PAPER'S OWN DATA IS SHALLOWER THAN THE LINE DRAWN THROUGH IT. The
      // text reads kappa ~ 1/2 above T ~ 1500, and the blue guide in Fig. 8
      // does have slope 0.505 when its endpoints are digitised. The PLOTTED
      // POINTS over the same range do not: 1.93 at T = 1500 to 3.20 at 10^4 is
      // kappa = 0.27, and 1.97 at 2000 to 3.20 at 10^4 is 0.30. So a run here
      // has two different things it might be asked to reproduce, and they
      // differ by a factor of about 1.8. Both are printed rather than one
      // chosen.
      std::printf("\n      T range            kappa in Ne ~ T^kappa"
                  "   (Fig. 8: line 0.50, its own points 0.27-0.30 above"
                  " T = 1500)\n");
      for (std::size_t i = 1; i < Ts.size(); ++i)
        if (Ne[i] > 0.0 && Ne[i - 1] > 0.0)
          std::printf("  %7.0f -> %-7.0f    %+.3f\n", Ts[i - 1], Ts[i],
                      std::log(Ne[i] / Ne[i - 1]) / std::log(Ts[i] / Ts[i - 1]));
    }
    std::fflush(stdout);
  }
  Kokkos::finalize();
  return 0;
}
