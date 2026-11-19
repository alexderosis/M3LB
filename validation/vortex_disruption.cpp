//==============================================================================
//  Disruption of a single vortex by a weak magnetic field.
//
//  One shielded Gaussian vortex at the centre of a periodic box, in a uniform
//  field along x. The vortex winds the field up, the wound field pushes back,
//  and above some field strength the vortex does not survive. The question the
//  case exists for is WHERE that happens as the magnetic Reynolds number rises:
//
//      M^2 Rm ~ 1          Mak, Griffiths & Hughes, Phys. Rev. Fluids 2,
//                          113701 (2017)
//      M^2 Rm^(2/3) ~ 1    Gilbert, Mason & Tobias, J. Fluid Mech. 791, 568
//                          (2016), in Mak's notation
//
//  Both arguments are written for ONE vortex, which is why the layout is one
//  vortex and not an array.
//
//  THE SETUP IS THE AUTHOR'S OWN, as his standalone Kokkos code of June 2026
//  defines it (VortexDisruption/Vortex/Swap/KOKKOS/src/common.hpp), in lattice
//  units with R0 the core radius in cells and eps the peak speed:
//
//      u_theta = eps (r / R0) exp[(1 - r^2 / R0^2) / 2]     peak eps at r = R0
//      b       = (M eps, 0)                                 uniform, along x
//      box     = (nr R0)^2, periodic, nr = 20, centre at ((n - 1)/2, (n - 1)/2)
//      Re = eps R0 / nu,   Rm = eps R0 / eta,   M = B0 / eps
//      time in R0 / eps;   defaults Re = Rm = 500, M = 0.05, R0 = 30, t = 50
//
//  The density is seeded with the vortex's own pressure,
//  rho = exp[-(e eps^2 / 2 cs^2) exp(-r^2 / R0^2)], so the run does not start
//  by radiating it as sound. HIS EARLIER FILE (Vortex/main.cpp) IS NOT THIS
//  SETUP: there the profile has no factor e^(1/2), so the peak speed is 0.607
//  eps and Re, Rm and M mean something else by that factor; the box is 20 R0 by
//  10 R0; and the field is held at its background value on the box edges. A
//  number from one is not comparable with a number from the other.
//
//  THE VORTEX CARRIES ITS OWN EXACT CHECK.  Its vorticity is the Laplacian of a
//  Gaussian, the flow is axisymmetric so the nonlinear term vanishes, and
//  viscosity therefore only widens it: sigma^2 = R0^2 + 2 nu t with the
//  amplitude falling as (R0 / sigma)^4. The projection of u on the initial
//  pattern is then
//
//      coh(t) = <u . u0> / <u0 . u0> = (1 + t / Re)^(-2)        (t in R0 / eps)
//
//  exactly, in an unbounded plane; in this box the images are at exp(-50). A
//  kinematic run measures the viscosity the scheme delivers from it and prints
//  PASS or FAIL. With the field on, `surv` = coh / (1 + t/Re)^(-2) is what the
//  field has taken beyond viscosity: 1 for an untouched vortex, no threshold,
//  no twin hydrodynamic run. `ang` is the angular momentum about the centre
//  over its initial value, which viscosity alone conserves.
//
//  M = 0 IS KINEMATIC, NOT FIELD-FREE.  -M 0 transports a field of lattice
//  strength eps and gives the fluid no force: flux expulsion with no feedback.
//
//  THE FIELD is the tree's MHD: Dellar's vector distribution for B, with the
//  Maxwell stress in the fluid's equilibrium. A flux-function formulation was
//  tried beside it and removed; the measured section says why.
//
//  LATTICES.  -lat d3q27 (the default) is D3Q27 + D3Q7 with nz = 1, the pairing
//  this tree takes for new work and the only one GPU/ has. -lat d2q9 is
//  D2Q9 + D2Q5, there because a sweep on a laptop has to be affordable. They
//  are NOT the same discretisation of the field (cs^2 = 1/4 against 1/3;
//  CLAUDE.md, "two-dimensional means nz = 1"). The run prints which one it
//  used; say it beside any number quoted from it. Only the D3Q27 form of the
//  fluid operator exposes equilibrium(), so only there is the fluid seeded
//  with the Maxwell stress; on D2Q9 that stress -- uniform here,
//  of relative size M^2 -- starts in the non-equilibrium part.
//
//  WHAT TO READ FIRST: THE LAYER IN CELLS, AND THE `osc` AND `divb` COLUMNS.
//  The wound-up sheet is R0 Rm^(-1/3) thick at its peak and the layer of the
//  expelled state R0 Rm^(-1/2); both are printed in cells at start-up, and the
//  second is the one a grid has to carry. `osc` is the odd-even share of the
//  current, <(j(x+1) - j(x))^2> / 4 <j^2> over both directions: (k h)^2 / 4 for
//  a resolved structure, 1 for a checkerboard. `divb` is max|div b| in the unit
//  of jmax, and it is not cleaned: a divergence error is a
//  monopole, and monopoles reconnect (GPU/src/mhd_jet.cu argues it).
//
//  COLUMNS.  t; EK, the kinetic energy over its initial value; B2 = <|b|^2> /
//  B0^2 over the box; bmax = max|b| / B0; coh, surv and ang as above; wmax and
//  jmax, max|vorticity| R0 / eps and max|j| R0 / B0; Lb = (<|b|^2> / <j^2>)^(1/2)
//  in R0; divb; osc; the peak Mach number; the mean density; and two measures
//  built from velocity GRADIENTS, each against the exact vortex at the same
//  time: ens, the enstrophy over the reference's, and dow, the author's Delta
//  after Mak (the comment in measure() defines both). The vorticity, j
//  and div b are second-order central differences of the reported fields.
//
//  MEASURED 2026-11-19, on a laptop (Threads, FP64, D2Q9 + D2Q5, eps = 0.05,
//  R0 = 30, t = 50 unless said).
//
//  THE EXACT CHECK HOLDS: viscosity delivered / set = 0.9990 at Re = 500, surv
//  within 2e-4 of 1 throughout, the angular momentum at 0.9995. Kinematic
//  expulsion peaks at bmax = 8.49 B0 at t = 15, against Rm^(1/3) = 7.94.
//
//  THE AUTHOR'S CASE, Re = Rm = 500, M = 0.05 (M^2 Rm = 1.25), IS
//  GRID-CONVERGED:
//
//      R0          15        20        30        45
//      surv(50)    0.71193   0.71181   0.71170   0.71165
//      peak bmax   7.841     7.815     7.801     7.795    (B0, at t = 13-13.5)
//      worst osc   0.037     0.021     0.010     0.004
//
//  The last two steps, each 1.5 in R0, move surv by 1.1e-4 and 0.5e-4: a ratio
//  of 2.2 against 2.25 for second order. The angular momentum stays within
//  7e-4 of its initial value while the field takes 29 % of the projection: it
//  takes the vortex's shape and energy, not its angular momentum.
//  NOTE WHAT CONVERGED AND WHAT DID NOT NEED TO: the expelled layer is 0.7 to
//  2.0 cells on these grids, which the line printed at start-up calls
//  unresolved, and surv agrees to four digits anyway. It is an integral over
//  the vortex. bmax moves in the third digit and the current is the grid's:
//  quote those with R0 beside them.
//
//  Re = Rm = 2000, M = 0.05 (M^2 Rm = 5), ONE GRID (R0 = 30):
//  surv(50) = 0.471, peak field 11.65 B0 at t = 17 against Rm^(1/3) = 12.6,
//  worst divb / jmax 1.8 %, worst osc 0.022. Not checked on a second grid.
//
//  THE MAP, 2026-11-19 (results/Q_vortex_disruption, tools/vortex_disruption_map.py).
//  Pm = 1, R0 = 20, Rm = 250, 500, 1000, 2000, 4000, 8000, six fields at each on
//  the ladder M^2 Rm = 0.3125 .. 10 in factors of two, each run to
//  t = 12 Rm^(1/3). 33 of 36 ran. Rm = 8000 at M^2 Rm = 2.5, 5 and 10 went
//  NON-FINITE at t = 64.5, 68.5 and 44: this solver's limit on this grid, reached
//  at the top row only. Two observables, both 0 for an untouched vortex:
//  D = 1 - surv, a measure of the MEAN flow, and Delta = dow, built from
//  velocity GRADIENTS.
//
//  READ EARLY, BOTH FOLLOW M^2 Rm^(2/3), NOT M^2 Rm.  At t = 3 Rm^(1/3), about
//  the time of the peak field, along the three weakest rows of the ladder:
//
//      M^2 Rm    Rm = 250  500     1000    2000    4000    8000     ~ Rm^q
//      D   0.3125  0.0729  0.0561  0.0436  0.0342  0.0269  0.0213   q = -0.355
//          0.625   0.1439  0.1107  0.0862  0.0676  0.0532  0.0420   q = -0.354
//          1.25    0.2790  0.2161  0.1688  0.1328  0.1049  0.0838   q = -0.347
//      Delta 0.3125  0.1091  0.0802  0.0603  0.0464  0.0360  0.0281   q = -0.389
//          0.625   0.2123  0.1571  0.1185  0.0912  0.0707  0.0551   q = -0.388
//          1.25    0.3999  0.3014  0.2297  0.1778  0.1386  0.1088   q = -0.375
//
//  A dependence on M^2 Rm alone would make each row flat; on M^2 Rm^(2/3) it
//  gives q = -1/3 while the observable is small, and D / (M^2 Rm^(2/3)) is 1.33
//  to 1.47 over all eighteen. So the gradient measure does NOT single out
//  M^2 Rm at this stage: the guess that the two published laws belong to two
//  kinds of measure was wrong. THESE ARE GRID-CONVERGED: R0 = 30 against 20
//  moves D by at most 0.75 % and Delta by at most 1.0 % (Rm = 4000 and 8000,
//  M^2 Rm = 1.25, and Rm 8000 at 0.3125 for D). The peak field of the weakest
//  row is 6.52 / 8.33 / 10.55 / 13.57 / 17.36 / 22.20 B0, 1.03 to 1.11 times
//  Rm^(1/3), and moves 0.5-2.5 % with the grid.
//  ens IS NOT CONVERGED AND IS NOT QUOTED: 1 - ens differs by 15 to 130 %
//  between the two grids, and changes sign. Total enstrophy lives in the thin
//  sheets, which neither grid carries.
//
//  THE READING TIME CHANGES THE EXPONENT, so no single exponent is claimed for
//  the early stage either: the same rows give q = -0.47 to -0.52 at a fixed
//  t = 50, where a higher Rm is read at an earlier phase.
//
//  THE VORTEX IS DESTROYED AT A FIXED M^2 Rm.  Taking Delta = 0.8, reached at
//  any time in the run, as "nearly destroyed", and with the ladder refined
//  between M^2 Rm = 1.25 and 2.5 to a factor 1.09 in M (three more fields at
//  each Rm, R0 = 20), the field that does it is
//
//      Rm             250      500      1000     2000     4000
//      M_c            0.0859   0.0673   0.0414   0.0297   0.0223     ~ Rm^-0.506
//      M_c^2 Rm       1.84     2.26     1.72     1.77     2.00
//      M_c^2 Rm^2/3   0.293    0.285    0.171    0.140    0.125
//
//  interpolated in log M between the two runs that bracket it. If each M_c lay
//  anywhere between its two runs the exponent could move by 0.038 at most, so
//  -1/3 is excluded for this level and -1/2 is not. Read at t = 12 Rm^(1/3)
//  instead of at the largest value: -0.504. Rm = 8000 is left out: the runs
//  that would bracket it are the three that went non-finite.
//
//  THE EXPONENT DEPENDS ON THE LEVEL CALLED A THRESHOLD, AND THAT IS THE TWO
//  STAGES SHOWING.  For Delta = 0.3 .. 0.9 in steps of 0.1 it is
//
//      -0.40  -0.41  -0.40  -0.43  -0.45  -0.51  -0.53
//
//  (allowed to move by 0.15 at the first two, 0.11 at the next two, 0.04-0.06
//  at the last three), and 1 - surv gives -0.38 at 0.3 and -0.48 to -0.51 from
//  0.6 up. A low level is reached partly by the wind-up; only a high one needs
//  the second stage. "-1/2" is therefore a statement about DESTRUCTION, levels
//  of 0.8 and above, and not about a threshold in general.
//
//  THE SECOND GRID, on the coarse ladder only (the two rows a factor 1.41
//  apart, R0 = 30 against 20, results/.../threshold_r030): a single run of this
//  stage is not reproducible to better than a few per cent -- Delta differs
//  between the grids by under 1 % at Rm = 250 and 500 and by up to 7 % at
//  Rm = 1000 and 2000, where the stage is chaotic -- but the threshold is
//  steadier than the runs: M_c moves by at most 3 %, and the exponent read from
//  those two rows is -0.498 and -0.506. That spacing allowed 0.15 either way,
//  so the agreement of the two grids was real and the value was not yet
//  established; the refined ladder is what establishes it, and the refined
//  ladder has been run on ONE grid.
//
//  THE SECOND STAGE IS A TRIPOLE, AND IT IS THE FIELD'S DOING.  At Re = Rm =
//  2000, M^2 Rm = 2.5 the vortex is axisymmetric to t = 50 (surv 0.713), is an
//  elongated core with two satellites by t = 76-88, and is gone by t = 120
//  (surv -0.05). Two controls, since a vortex with no net circulation is the
//  kind that goes tripolar by itself:
//
//      no Lorentz force, ellipticity 1e-2 (-ellip)   surv(151) = 0.9968
//      no Lorentz force, ellipticity 1e-3            surv(151) = 0.9986
//      field removed at t = 50 (-off 50)             dips to 0.589 at t = 82.5,
//                                                    recovers, 0.673 at t = 151
//      field kept                                    -0.05 from t = 120
//
//  An m = 2 seed without the field grows too slowly to matter, and the vortex
//  the field has left behind at t = 50 wobbles and survives once the field is
//  taken away. So the destruction needs the field to keep acting.
//
//  Re AND Rm SEPARATED (results/Q_vortex_disruption/pm_fixRe1000, pm_fixRm1000).
//  The map moved them together. Two slices through its point Re = Rm = 1000
//  move one at a time, over 250, 500, 1000, 2000, 4000 (Pm from 0.25 to 4), six
//  fields each -- M^2 Rm = 0.625 and the five across the threshold -- R0 = 20.
//  48 new runs, none non-finite.
//
//                                        Re = 1000,        Rm = 1000,
//                                        Rm varied         Re varied
//      early Delta ~ X^q   (0.625 row)   q = -0.318        q = -0.090
//      early 1 - surv ~ X^q              q = -0.324        q = -0.040
//      M_c at Delta = 0.8 ~ X^p          p = -0.467        p = -0.069
//      M_c at Delta = 0.9 ~ X^p          p = -0.519        p = -0.013 (Re >= 1000)
//      M_c^2 Rm at 0.8, along X          1.44 1.32 1.72    2.37 2.01 1.72
//                                        1.51 1.69         1.67 1.61
//
//  ("early" is t = 3 Rm^(1/3); the thresholds are of the largest Delta in the
//  run, and the ladder allows each p 0.04, or 0.06 where only three values
//  cross.) BOTH SCALINGS ARE Rm's. With Re held still the wind-up exponent is
//  -0.28 to -0.32 over the six rows and the destruction exponent -0.47 to
//  -0.52; with Rm held still neither measure moves by more than Re^-0.09. The
//  two slices add up to the map: -0.318 and -0.090 against its -0.388 for the
//  early Delta, -0.467 and -0.069 against its -0.506 for the threshold. So a
//  little of each exponent quoted from the map at Pm = 1 was the viscosity's.
//  What is left of the viscosity is at the bottom: at Re = 250 and 500 the
//  vortex needs a field 19 and 10 % stronger than at Re >= 1000, where
//  M_c^2 Rm is 1.6 to 1.7.
//
//  WHAT ALL THIS SUGGESTS, STATED AS A SUGGESTION: the wind-up brakes the vortex
//  in proportion to M^2 Rm^(2/3), and whether the vortex is then destroyed is
//  decided by M^2 Rm, near 2. The two published scalings would then describe
//  two stages of one process. It rests on a factor of 16 in Rm and in Re, on
//  one layout, with the refined ladder and both slices on one grid.
//
//  A FLUX-FUNCTION FORMULATION WAS TRIED AND REMOVED THE SAME DAY.  In two
//  dimensions b = z x grad(A - B0 y), so div b is zero identically; A was
//  carried by ScalarSolver with the regularised operator and the source B0 v
//  (plus A div u, since a lattice scalar is transported conservatively and A is
//  advected), and the Lorentz force went to the fluid through Guo. It was built
//  to cure a div B failure and it was the less robust of the two:
//
//      force                          outcome at the author's case
//      -j grad A, five-point j        non-finite at t = 19
//      div(b b - b^2/2), central      reaches t = 50, surv 0.7130; at
//                                     Re = Rm = 2000, non-finite at t = 12.5
//
//  Each tracked the runs above to three digits until its own failure began
//  (surv 0.84519 against 0.84407 at t = 15), with div b at 1e-14 throughout:
//  ZERO DIVERGENCE BOUGHT NO STABILITY. The second force agreed on surv, EK and
//  ang to 0.2 % at t = 50 and was WRONG IN THE FIELD from t = 32 -- a mode a
//  few cells long, osc 0.17-0.23 against 0.003, peak current 300-550 against
//  15-20, peak field 10-27 B0 against 4.3-4.9. An integral that agrees is not
//  evidence that the field does. Advancing A with the velocity after the flow's
//  step rather than before changed nothing, in either force, to four digits in
//  surv. The mechanism is NOT diagnosed. Suspected, for the first force only:
//  the five-point j sees a checkerboard in A that the central gradient cannot,
//  and a centred advection carries the resulting checkerboard in u with the
//  wrong sign.
//
//  NOT DONE, AND NOT CLAIMED.  dow follows the author's definition of Delta, not
//  Mak's exactly: the region is fixed by the reference vortex, not by the
//  largest connected patch of the run's own field, so it is not comparable
//  number for number with that paper's figures. The refined ladder has not
//  been run on a second grid, nor have the two slices in Pm, which pass through
//  one point only. Nothing above Rm = 4000 is usable for the second stage. No walls and no held
//  far field: the box is periodic. Nothing in this banner is a measurement of a
//  regime boundary or of a scaling in Rm; the measured section below says what
//  has been run.
//==============================================================================
#include "Mhd.hpp"
#include "collision/MhdCentralMomentsShifted.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace lbm;

namespace {

struct Opts {
  Index R0 = 30;             // core radius, cells
  int nr = 20;               // box side, in R0
  double Re = 500.0;
  double Rm = -1.0;          // < 0: follow Re (Pm = 1)
  double M = 0.05;           // 0 = kinematic
  double tmax = 50.0;        // in R0 / eps
  double u0 = 0.05;          // eps, the peak speed in lattice units
  double probe = 0.5;
  double wbulk = 1.0;
  int frames = 0;            // -frames K: vorticity and j dumped at K + 1 times
  double ellip = 0.0;        // -ellip E: an m = 2 seed on the vortex (see vortex())
  double off = -1.0;         // -off T: the field is removed at time T (see run())
  std::string lat = "d3q27";
  std::string out = ".";
};

// What a run is, once its options are read.
struct Run {
  Index N = 0;
  double R0 = 0, dt = 0, B0 = 0, Rm = 0, nu = 0, eta = 0, xc = 0;
  bool kinematic = false;
  std::size_t n_max = 0, stride = 0, fstride = 0;
};

Run plan(const Opts& o) {
  Run r;
  r.N = Index(o.nr) * o.R0;
  r.R0 = double(o.R0);
  r.xc = 0.5 * double(r.N - 1);
  r.dt = o.u0 / r.R0;                      // R0/eps per step
  r.Rm = (o.Rm > 0) ? o.Rm : o.Re;
  r.nu  = o.u0 * r.R0 / o.Re;
  r.eta = o.u0 * r.R0 / r.Rm;
  r.kinematic = !(o.M > 0);
  r.B0 = r.kinematic ? o.u0 : o.M * o.u0;  // lattice field strength
  r.n_max  = std::size_t(std::ceil(o.tmax / r.dt));
  r.stride = std::max<std::size_t>(1, std::size_t(std::llround(o.probe / r.dt)));
  r.fstride = (o.frames > 0) ? std::max<std::size_t>(1, r.n_max / std::size_t(o.frames)) : 0;
  return r;
}

// The vortex, from its stream function psi = -U R0 exp[(1 - s2)/2] with
// s2 = [(1 + e) dx^2 + (1 - e) dy^2] / R0^2, and the density that balances it.
// At e = 0 that is u_theta = U (r/R0) exp[(1 - r^2/R0^2)/2]. A non-zero e makes
// the vortex slightly elliptical -- an m = 2 seed -- and, taken through psi,
// leaves the velocity divergence-free. dx, dy are offsets from the centre in
// cells.
KOKKOS_INLINE_FUNCTION
void vortex(Real dx, Real dy, Real R0, Real U, Real ics, Real e, Real& u, Real& v, Real& rho) {
  const Real s2 = ((Real(1) + e) * dx * dx + (Real(1) - e) * dy * dy) / (R0 * R0);
  const Real g = U / R0 * Kokkos::exp(Real(0.5) * (Real(1) - s2));
  u = -g * (Real(1) - e) * dy;
  v =  g * (Real(1) + e) * dx;
  rho = Kokkos::exp(-Real(0.5) * ics * U * U * Kokkos::exp(Real(1) - s2));
}

void announce(const Opts& o, const Run& r, const char* pairing, double tauf, double taum) {
  const double sheet = r.R0 * std::cbrt(1.0 / r.Rm);
  const double layer = r.R0 / std::sqrt(r.Rm);
  std::printf("Disruption of a single vortex by a weak magnetic field\n");
  std::printf("backend %s   precision %s   lattices %s\n",
              ExecSpace::name(), precision_name(), pairing);
  std::printf("  R0 = %d cells   box %d x %d (%d R0)   Re = %.6g   Rm = %.6g   Pm = %.4g"
              "   M = %.6g%s\n", int(o.R0), int(r.N), int(r.N), o.nr, o.Re, r.Rm,
              r.Rm / o.Re, o.M, r.kinematic ? "   (KINEMATIC: no Lorentz force)" : "");
  if (o.ellip != 0.0)
    std::printf("  SEEDED: ellipticity %g on the initial vortex (an m = 2 perturbation)\n", o.ellip);
  if (o.off > 0)
    std::printf("  THE FIELD IS REMOVED AT t = %g: B is set to zero there and stays zero\n", o.off);
  std::printf("  M^2 Rm = %.4g   M^2 Rm^(2/3) = %.4g   M Rm = %.4g\n",
              o.M * o.M * r.Rm, o.M * o.M * std::cbrt(r.Rm * r.Rm), o.M * r.Rm);
  std::printf("  eps = %.4g (Ma = %.4f)   dt = %.6e   t_max = %.6g (%zu steps)\n",
              o.u0, o.u0 * std::sqrt(3.0), r.dt, o.tmax, r.n_max);
  std::printf("  nu = %.6e   tau_f - 1/2 = %.3e   eta = %.6e   tau_m - 1/2 = %.3e\n",
              r.nu, tauf - 0.5, r.eta, taum - 0.5);
  std::printf("  Re_cell = %.1f   Rm_cell = %.1f   (Re / R0: only cells move these)\n",
              o.u0 / r.nu, o.u0 / r.eta);
  std::printf("  wound sheet R0 Rm^(-1/3) = %.1f cells;   expelled layer R0 Rm^(-1/2)"
              " = %.1f cells\n", sheet, layer);
  std::printf("  probe every %zu steps (%.4f)%s\n\n", r.stride, double(r.stride) * r.dt,
              r.fstride ? "   frames on" : "");
}

struct Probe {
  double ek, b2, bmax, coh, ang, wmax, jmax, Lb, divb, osc, ma, mass, ens, dow;
  bool finite;
};

// One probe's fields on the host, row-major (y N + x), in LATTICE units, with
// the initial pattern beside them. Everything the probe reports is computed
// from these and from nothing else.
struct Host {
  Index N = 0;
  std::vector<double> ux, uy, rho, bx, by, ut, vt;
  Host(Index n, double R0, double xc, double U) : N(n) {
    const std::size_t s = std::size_t(n) * std::size_t(n);
    ux.assign(s, 0.0); uy.assign(s, 0.0); rho.assign(s, 0.0);
    bx.assign(s, 0.0); by.assign(s, 0.0); ut.assign(s, 0.0); vt.assign(s, 0.0);
    for (Index y = 0; y < n; ++y)
      for (Index x = 0; x < n; ++x) {
        const double dx = double(x) - xc, dy = double(y) - xc;
        const double g = U / R0 * std::exp(0.5 * (1.0 - (dx * dx + dy * dy) / (R0 * R0)));
        ut[at(x, y)] = -g * dy;
        vt[at(x, y)] =  g * dx;
      }
  }
  std::size_t at(Index x, Index y) const {
    return std::size_t(((y + N) % N) * N + ((x + N) % N));
  }
};

//------------------------------------------------------------------------------
// Lengths in R0, speeds in eps, the field in B0. If wf / jf are given they
// receive the vorticity and the current.
//------------------------------------------------------------------------------
Probe measure(const Host& H, const Opts& o, const Run& r, double t,
              std::vector<float>* wf = nullptr, std::vector<float>* jf = nullptr) {
  const Index N = H.N;
  const double iu = 1.0 / o.u0, ib = 1.0 / r.B0;
  const double idu = iu * r.R0, idb = ib * r.R0;
  if (wf) wf->assign(std::size_t(N * N), 0.0f);
  if (jf) jf->assign(std::size_t(N * N), 0.0f);

  double su2 = 0, sb2 = 0, sj2 = 0, sdot = 0, s00 = 0, sang = 0, sang0 = 0;
  double smass = 0, sdj = 0, wmax = 0, jmax = 0, dmax = 0, umax = 0, bmax = 0;
  bool finite = true;
  auto J = [&](Index x, Index y) {
    return (0.5 * (H.by[H.at(x + 1, y)] - H.by[H.at(x - 1, y)]) -
            0.5 * (H.bx[H.at(x, y + 1)] - H.bx[H.at(x, y - 1)])) * idb;
  };
  for (Index y = 0; y < N; ++y)
    for (Index x = 0; x < N; ++x) {
      const std::size_t k = H.at(x, y);
      const double u = H.ux[k], v = H.uy[k], a = H.bx[k], b = H.by[k];
      const double w = (0.5 * (H.uy[H.at(x + 1, y)] - H.uy[H.at(x - 1, y)]) -
                        0.5 * (H.ux[H.at(x, y + 1)] - H.ux[H.at(x, y - 1)])) * idu;
      const double j = J(x, y);
      const double dv = (0.5 * (H.bx[H.at(x + 1, y)] - H.bx[H.at(x - 1, y)]) +
                         0.5 * (H.by[H.at(x, y + 1)] - H.by[H.at(x, y - 1)])) * idb;
      if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(a) ||
          !std::isfinite(b) || !std::isfinite(w) || !std::isfinite(j))
        finite = false;
      const double jx = J(x + 1, y) - j, jy = J(x, y + 1) - j;
      sdj += 0.5 * (jx * jx + jy * jy);
      const double dx = double(x) - r.xc, dy = double(y) - r.xc;
      su2 += u * u + v * v;
      sb2 += a * a + b * b;
      sj2 += j * j;
      sdot += u * H.ut[k] + v * H.vt[k];
      s00  += H.ut[k] * H.ut[k] + H.vt[k] * H.vt[k];
      sang  += dx * v - dy * u;
      sang0 += dx * H.vt[k] - dy * H.ut[k];
      smass += H.rho[k];
      wmax = std::max(wmax, std::abs(w));
      jmax = std::max(jmax, std::abs(j));
      dmax = std::max(dmax, std::abs(dv));
      umax = std::max(umax, std::hypot(u, v));
      bmax = std::max(bmax, std::hypot(a, b));
      if (wf) (*wf)[k] = float(w);
      if (jf) (*jf)[k] = float(j);
    }
  const double c = 1.0 / (double(N) * double(N));
  Probe p;
  p.ek   = 0.5 * su2 * iu * iu * c;
  p.b2   = sb2 * ib * ib * c;
  p.bmax = bmax * ib;
  p.coh  = sdot / s00;
  p.ang  = sang / sang0;
  p.wmax = wmax; p.jmax = jmax; p.divb = dmax;
  // A uniform field has no current: j is round-off there, and Lb and osc are
  // reported as 0 (standing for "infinite" and "undefined") rather than as noise.
  const bool has_j = sj2 > 1e-24 * sb2 * ib * ib;
  p.Lb   = has_j ? std::sqrt(sb2 * ib * ib / sj2) : 0.0;
  p.osc  = has_j ? sdj / (4.0 * sj2) : 0.0;
  p.ma   = umax * std::sqrt(3.0);
  p.mass = smass * c;

  // ---- the two measures built from velocity GRADIENTS -----------------------
  // Both compare the run with the EXACT viscous vortex at the same time,
  // u_theta = U e^(1/2) (r/R0) (R0/s)^4 exp(-r^2 / 2 s^2), s^2 = R0^2 (1 + 2t/Re),
  // differenced on this grid with the stencil used for the run, so that what
  // the scheme does to a gradient cancels and a kinematic run reads ens = 1 and
  // dow = 0 up to its own error.
  //
  //   ens   the enstrophy over the reference's.
  //   dow   the author's Delta, after Mak, Griffiths & Hughes: with
  //         W = (u_x - v_y)^2 + (v_x + u_y)^2 - (v_x - u_y)^2 the Okubo-Weiss
  //         field, and A the region where the REFERENCE is rotation-dominated,
  //         W_ref < -0.2 rms(W_ref) over the box,
  //             dow = 1 - sum_A W / sum_A W_ref.
  //         His code takes the reference from a separate hydrodynamic run at
  //         one fixed time; here it is the exact vortex at the time of the probe.
  {
    const double s2 = r.R0 * r.R0 * (1.0 + 2.0 * t / o.Re);
    const double amp = o.u0 * std::exp(0.5) / r.R0 * (r.R0 * r.R0 / s2) * (r.R0 * r.R0 / s2);
    std::vector<double> ur(H.ux.size()), vr(H.ux.size());
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const double dx = double(x) - r.xc, dy = double(y) - r.xc;
        const double g = amp * std::exp(-0.5 * (dx * dx + dy * dy) / s2);
        ur[H.at(x, y)] = -g * dy;
        vr[H.at(x, y)] =  g * dx;
      }
    auto ow = [&](const std::vector<double>& U, const std::vector<double>& V, Index x, Index y,
                  double& w2) {
      const double ux_ = 0.5 * (U[H.at(x + 1, y)] - U[H.at(x - 1, y)]);
      const double uy_ = 0.5 * (U[H.at(x, y + 1)] - U[H.at(x, y - 1)]);
      const double vx_ = 0.5 * (V[H.at(x + 1, y)] - V[H.at(x - 1, y)]);
      const double vy_ = 0.5 * (V[H.at(x, y + 1)] - V[H.at(x, y - 1)]);
      const double ns = ux_ - vy_, ss = vx_ + uy_, vz = vx_ - uy_;
      w2 = vz * vz;
      return ns * ns + ss * ss - vz * vz;
    };
    std::vector<double> Wr(H.ux.size());
    double zr = 0, z = 0, rms = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        double w2;
        Wr[H.at(x, y)] = ow(ur, vr, x, y, w2);
        zr += w2;
        rms += Wr[H.at(x, y)] * Wr[H.at(x, y)];
      }
    const double thr = -0.2 * std::sqrt(rms * c);
    double num = 0, den = 0;
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        double w2;
        const double W = ow(H.ux, H.uy, x, y, w2);
        z += w2;
        if (Wr[H.at(x, y)] < thr) { num += W; den += Wr[H.at(x, y)]; }
      }
    p.ens = (zr > 0) ? z / zr : 0.0;
    p.dow = (den < 0) ? 1.0 - num / den : 0.0;
  }
  p.finite = finite;
  return p;
}

template <class FS>
void pull_fluid(FS& fl, Host& H) {
  const Domain& d = fl.domain();
  fl.compute_macroscopic();
  auto ux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
  auto uy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
  auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.rho());
  for (Index y = 0; y < H.N; ++y)
    for (Index x = 0; x < H.N; ++x) {
      const Index n = d.id(x, y);
      const std::size_t k = H.at(x, y);
      H.ux[k] = double(ux(n)); H.uy[k] = double(uy(n)); H.rho[k] = double(hr(n));
    }
}

void write_field(const std::string& path, Index N, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const int nx = int(N), ny = int(N);
  o.write(reinterpret_cast<const char*>(&nx), sizeof nx);
  o.write(reinterpret_cast<const char*>(&ny), sizeof ny);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

//------------------------------------------------------------------------------
// The loop, the probes, the series and the verdict. step() advances one time
// step; pull(H) fills the host fields.
//------------------------------------------------------------------------------
template <class Step, class Pull>
int march(const Opts& o, const Run& r, const char* pairing, Step step, Pull pull) {
  const Index N = r.N;
  std::filesystem::create_directories(o.out);
  std::ofstream series(o.out + "/series.dat");
  series << "# vortex_disruption  lattices " << pairing << "  R0 " << o.R0 << "  nr "
         << o.nr << "  Re " << o.Re << "  Rm " << r.Rm << "  M " << o.M
         << (r.kinematic ? " (kinematic)" : "") << "  eps " << o.u0
         << "\n# t EK B2 bmax coh surv ang wmax jmax Lb divb osc Ma mass ens dow\n";
  series.precision(10);

  Host H(N, r.R0, r.xc, o.u0);
  std::vector<float> wf, jf;
  int frame = 0;
  double t_now = 0.0;
  auto dump = [&]() {
    char fn[512];
    pull(H);
    measure(H, o, r, t_now, &wf, &jf);
    std::snprintf(fn, sizeof fn, "%s/w_%04d.bin", o.out.c_str(), frame);
    write_field(fn, N, wf);
    std::snprintf(fn, sizeof fn, "%s/j_%04d.bin", o.out.c_str(), frame);
    write_field(fn, N, jf);
    ++frame;
  };
  auto exact = [&](double t) { return 1.0 / ((1.0 + t / o.Re) * (1.0 + t / o.Re)); };

  std::printf("%-7s %-9s %-9s %-8s %-9s %-9s %-9s %-8s %-9s %-8s %-9s %-9s %-8s %-7s\n",
              "t", "EK", "B2", "bmax", "coh", "surv", "dow", "ens", "ang", "wmax", "jmax",
              "divb", "osc", "Ma");
  std::printf("%s\n", std::string(130, '-').c_str());
  double ek0 = 1.0;
  auto report = [&](double t, const Probe& p) {
    const double sv = p.coh / exact(t);
    std::printf("%-7.2f %-9.6f %-9.5f %-8.3f %-9.6f %-9.6f %-9.6f %-8.5f %-9.6f %-8.4f "
                "%-9.3f %-9.2e %-8.5f %-7.4f\n",
                t, p.ek / ek0, p.b2, p.bmax, p.coh, sv, p.dow, p.ens, p.ang, p.wmax, p.jmax,
                p.divb, p.osc, p.ma);
    std::fflush(stdout);
    series << t << ' ' << p.ek / ek0 << ' ' << p.b2 << ' ' << p.bmax << ' ' << p.coh
           << ' ' << sv << ' ' << p.ang << ' ' << p.wmax << ' ' << p.jmax << ' ' << p.Lb
           << ' ' << p.divb << ' ' << p.osc << ' ' << p.ma << ' ' << p.mass << ' '
           << p.ens << ' ' << p.dow << '\n';
    series.flush();
  };

  pull(H);
  const Probe p0 = measure(H, o, r, 0.0);
  ek0 = p0.ek;
  report(0.0, p0);
  if (r.fstride) dump();

  double bmax_peak = p0.bmax, t_peak = 0.0, osc_max = 0.0, div_worst = 0.0;
  double surv_min = 1.0, t_surv = 0.0, blow_t = -1.0, t_last = 0.0;
  Probe last = p0;
  bool told = false;
  const auto w0 = std::chrono::steady_clock::now();
  for (std::size_t t = 0; t < r.n_max; ++t) {
    step();
    t_now = double(t + 1) * r.dt;
    if (r.fstride && (t + 1) % r.fstride == 0) dump();
    if ((t + 1) % r.stride == 0 || t + 1 == r.n_max) {
      const double tt = double(t + 1) * r.dt;
      pull(H);
      const Probe p = measure(H, o, r, tt);
      report(tt, p);
      if (!p.finite || p.ma > 0.5) { blow_t = tt; break; }
      last = p; t_last = tt;
      if (p.bmax > bmax_peak) { bmax_peak = p.bmax; t_peak = tt; }
      const double sv = p.coh / exact(tt);
      if (sv < surv_min) { surv_min = sv; t_surv = tt; }
      osc_max = std::max(osc_max, p.osc);
      if (p.jmax > 0) div_worst = std::max(div_worst, p.divb / p.jmax);
      if (!told) {
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - w0).count();
        std::printf("  [%.1f MLUPS so far; the whole run is about %.1f min]\n",
                    double(t + 1) * double(N) * double(N) / el * 1e-6,
                    el * double(r.n_max) / double(t + 1) / 60.0);
        told = true;
      }
    }
  }
  const double el = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - w0).count();

  // The summary is of the last FINITE probe: a blown-up row is reported above
  // and is not a measurement.
  std::printf("\n");
  if (blow_t > 0)
    std::printf("  BLOW-UP at t = %.3f (non-finite field or Ma > 0.5); "
                "last usable probe t = %.3f\n", blow_t, t_last);
  else
    std::printf("  ran to t = %.3f\n", t_last);
  std::printf("  peak field bmax = %.3f B0 at t = %.2f   (Rm^(1/3) = %.2f)\n",
              bmax_peak, t_peak, std::cbrt(r.Rm));
  std::printf("  survival at the last probe = %.5f   lowest = %.5f at t = %.2f   "
              "angular momentum = %.5f\n", last.coh / exact(t_last), surv_min, t_surv,
              last.ang);
  std::printf("  worst osc = %.5f   (resolved: << 1; checkerboard: 1)   "
              "worst divb / jmax = %.2e\n", osc_max, div_worst);
  std::printf("  mass drift %.3e   wall clock %.1f s\n", last.mass / p0.mass - 1.0, el);

  int rc = (blow_t > 0) ? 1 : 0;
  if (r.kinematic && o.ellip != 0.0)
    std::printf("  no exact check: the vortex was seeded with an ellipticity of %g\n", o.ellip);
  if (r.kinematic && o.ellip == 0.0 && blow_t < 0 && t_last > 0) {
    // coh = (1 + nu t / R0^2)^(-2) exactly, so the viscosity delivered follows
    // from the projection alone. The projection, not the energy, because sound
    // left over from the start sits in EK and is orthogonal to the pattern.
    const double ratio = o.Re * (1.0 / std::sqrt(last.coh) - 1.0) / t_last;
    const bool ok = std::abs(ratio - 1.0) < 0.05;
    std::printf("  viscosity delivered / viscosity set = %.4f   (exact: coh = "
                "(1 + t/Re)^-2; gate 5%%)   %s\n", ratio, ok ? "PASS" : "FAIL");
    if (!ok) rc = 1;
  }
  return rc;
}

//------------------------------------------------------------------------------
// The solvers: Dellar's distribution for B, shifted central moments for the
// fluid with the Maxwell stress in its equilibrium.
//------------------------------------------------------------------------------
template <class FLt, class MLt>
int run(const Opts& o, const char* pairing) {
  using FC = MhdCentralMomentsShifted<FLt>;
  using MC = MagneticBGK<MLt>;
  static_assert(std::is_same_v<typename FC::Storage, RawPopulations>,
                "the Maxwell-stress seed writes raw populations");
  const Run r = plan(o);
  const Index N = r.N;

  MC mc;
  mc.omega = MC::omega_from_resistivity(Real(r.eta));
  FC fc;
  fc.omega = FC::omega_from_viscosity(Real(r.nu));
  fc.omega_bulk = Real(o.wbulk);
  announce(o, r, pairing, 1.0 / double(fc.omega), 1.0 / double(mc.omega));

  Domain d(N, N, 1, true, true, true);
  MagneticSolver<MLt, EsotericPull<MLt>, MC> mag(d, mc);
  // The fluid's view of the field. Kinematic: a zero view, so no Lorentz force.
  View1D<Real> off("B_off", d.n_padded);
  if (r.kinematic) { fc.Bx = off; fc.By = off; fc.Bz = off; }
  else             { fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz(); }
  FluidSolver<FLt, EsotericPull<FLt>, FC> fl(d, fc);

  const Real R0 = Real(r.R0), U = Real(o.u0), Bl = Real(r.B0), xc = Real(r.xc);
  const Real el = Real(o.ellip);
  const Real ics = inv_cs2<FLt, Real>();
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; d.coords(n, px, py, pz);
    FlowState s;
    vortex(Real(px - d.hx) - xc, Real(py - d.hy) - xc, R0, U, ics, el, s.ux, s.uy, s.rho);
    s.uz = Real(0);
    return s;
  });
  constexpr bool has_eq =
      requires(const FC& c, Real* fe, const Real* v) { c.equilibrium(fe, Real(1), v, v); };
  if constexpr (has_eq) {
    if (!r.kinematic)
      fl.seed_populations(KOKKOS_LAMBDA(Index n, int i) {
        Index px, py, pz; d.coords(n, px, py, pz);
        Real ur[3] = {Real(0), Real(0), Real(0)}, rho;
        vortex(Real(px - d.hx) - xc, Real(py - d.hy) - xc, R0, U, ics, el, ur[0], ur[1], rho);
        const Real br[3] = {Bl, Real(0), Real(0)};
        Real fe[FLt::Q];
        fc.equilibrium(fe, rho, ur, br);
        return fe[i];
      });
  }
  // The field at its equilibrium WITH the initial velocity (MagneticSolver.hpp).
  mag.initialize_field(
      KOKKOS_LAMBDA(Index) {
        Kokkos::Array<Real, 3> bb;
        bb[0] = Bl; bb[1] = Real(0); bb[2] = Real(0);
        return bb;
      },
      KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        Kokkos::Array<Real, 3> uu;
        Real rho;
        vortex(Real(px - d.hx) - xc, Real(py - d.hy) - xc, R0, U, ics, el, uu[0], uu[1], rho);
        uu[2] = Real(0);
        return uu;
      });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // -off T deletes the field at time T: its populations are re-seeded at B = 0,
  // and the induction equation is linear, so it stays zero. What the vortex does
  // AFTER that it does without any Lorentz force -- the test that separates an
  // instability of the vortex the field has left behind from the field's own
  // continuing push.
  std::size_t n_done = 0;
  const std::size_t n_off = (o.off > 0) ? std::size_t(std::llround(o.off / r.dt)) : 0;
  auto step = [&]() {
    if (n_off && n_done == n_off)
      mag.initialize_field(KOKKOS_LAMBDA(Index) {
        Kokkos::Array<Real, 3> bb;
        bb[0] = Real(0); bb[1] = Real(0); bb[2] = Real(0);
        return bb;
      });
    mag.compute_field(); fl.step(true); mag.step(true);
    ++n_done;
  };
  auto pull = [&](Host& H) {
    pull_fluid(fl, H);
    mag.compute_field();
    auto bx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    auto by = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    for (Index y = 0; y < N; ++y)
      for (Index x = 0; x < N; ++x) {
        const Index n = d.id(x, y);
        H.bx[H.at(x, y)] = double(bx(n)); H.by[H.at(x, y)] = double(by(n));
      }
  };
  return march(o, r, pairing, step, pull);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Opts o;
    bool bad = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&](double& x) { if (i + 1 < argc) x = std::atof(argv[++i]); else bad = true; };
      if (a == "-r0" && i + 1 < argc)           o.R0 = std::atoi(argv[++i]);
      else if (a == "-nr" && i + 1 < argc)      o.nr = std::atoi(argv[++i]);
      else if (a == "-re")                      val(o.Re);
      else if (a == "-rm")                      val(o.Rm);
      else if (a == "-M")                       val(o.M);
      else if (a == "-tmax")                    val(o.tmax);
      else if (a == "-eps")                     val(o.u0);
      else if (a == "-probe")                   val(o.probe);
      else if (a == "-wb")                      val(o.wbulk);
      else if (a == "-frames" && i + 1 < argc)  o.frames = std::atoi(argv[++i]);
      else if (a == "-ellip")                   val(o.ellip);
      else if (a == "-off")                     val(o.off);
      else if (a == "-lat" && i + 1 < argc)     o.lat = argv[++i];
      else if (a == "-out" && i + 1 < argc)     o.out = argv[++i];
      else if (a.rfind("--kokkos", 0) == 0)     continue;   // Kokkos's own
      else { std::fprintf(stderr, "\nERROR: unknown or incomplete argument %s\n", a.c_str()); bad = true; }
    }
    if (o.lat != "d3q27" && o.lat != "d2q9") {
      std::fprintf(stderr, "\nERROR: unknown -lat %s   (lattices: d3q27 d2q9)\n", o.lat.c_str());
      bad = true;
    }
    if (o.R0 < 4 || o.nr < 6 || o.Re <= 0 || o.u0 <= 0 || o.tmax <= 0 || o.probe <= 0 ||
        o.M < 0) {
      std::fprintf(stderr, "\nERROR: need -r0 >= 4, -nr >= 6, positive -re -eps -tmax "
                           "-probe, -M >= 0\n");
      bad = true;
    }
    if (bad) {
      std::fprintf(stderr,
          "usage: vortex_disruption [-r0 CELLS] [-nr BOX_IN_R0] [-re Re] [-rm Rm] [-M M]\n"
          "                         [-tmax T] [-eps U] [-probe DT] [-wb W] [-frames K]\n"
          "                         [-ellip E] [-off T]\n"
          "                         [-lat d3q27|d2q9] [-out DIR]\n"
          "NOTHING WAS RUN.\n\n");
      rc = 1;
    } else {
      rc = (o.lat == "d3q27") ? run<D3Q27, D3Q7>(o, "D3Q27 + D3Q7 (nz = 1)")
                              : run<D2Q9, D2Q5>(o, "D2Q9 + D2Q5");
    }
  }
  Kokkos::finalize();
  return rc;
}
