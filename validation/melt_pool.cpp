//==============================================================================
//  Conduction-mode single-track melt pool, Ti-6Al-4V bare plate, against the
//  analytic moving-distributed-source solution of Eagar & Tsai (1983).
//
//  A laser of power P and absorptivity A, Gaussian in the plane, travels along
//  +x at constant speed v over the free surface. The substrate is STATIC and
//  the SOURCE MOVES, so u == 0 identically: no fluid lattice is allocated and
//  the LBM solves pure conduction of the total enthalpy H = e(T) + La f_l on a
//  single scalar lattice. That is deliberate. The alternative -- a frame moving
//  with the beam, with the material advected at -v -- would put u_lat of order
//  0.2 into a D3Q7 scalar whose equilibrium is first order in u, and would
//  spend the one regime in which EnthalpyBGK's Chapman-Enskog recovery is EXACT
//  rather than to advection-diffusion order (its banner, at the top of
//  collision/EnthalpyBGK.hpp). The cost is a longer domain, and cells are cheap.
//
//  THIS IS A PREDICTION, NOT A FIT, AND THAT IS THE ENTIRE POINT. Every input
//  is either a process setting or a measured material property with a named
//  source printed at startup. No constant is adjusted to improve agreement with
//  anything. The absorptivity -- the only input that is neither -- is run as the
//  measured band 0.27 / 0.33 / 0.36 and reported as a band, so that if
//  agreement ever required an A outside it, that is a model defect to be
//  printed and not a value to be adopted. Contrast validation/keyhole.cpp,
//  which is a port of a model with thirteen fitted constants that does not
//  survive grid refinement; this case exists because that one cannot predict.
//
//  THREE TIERS OF COMPARISON, AND THEY ARE NOT THE SAME KIND OF CLAIM.
//
//   (a) La = 0, cp_s = cp_l, k_s = k_l. EnthalpyBGK is then population-for-
//       population ScalarBGK (pinned in tests/test_enthalpy.cpp block 1) and
//       the LBM solves EXACTLY the equation Eagar & Tsai solve. This row is a
//       validation against an analytic solution, with a convergence order.
//   (b) Latent heat and the solid/liquid property split switched back on. The
//       change in width, depth and length is a MEASUREMENT WITH NO ANALYTIC
//       REFERENCE -- neither Rosenthal nor Eagar & Tsai contains latent heat.
//       It is reported as an increment against (a) and never as agreement.
//   (c) Against published single-track dimensions: a COMPARISON, weaker than
//       either, because the experiment carries its own uncertainty.
//   (d) `-flow`: MARANGONI + A MUSHY-ZONE SINK + ADVECTION OF H BY u. Off by
//       default, so every number above is unchanged. There is NO analytic
//       reference for this tier either -- it is an INCREMENT against (a)/(b)
//       and is reported, never asserted. The two terms are validated
//       SEPARATELY and beforehand, which is the only reason this tier means
//       anything: validation/marangoni.cpp puts the surface shear against the
//       exact parallel-flow profile, and validation/mushy_sink.cpp puts the
//       sink against the exact reduced channel and measures its stability
//       bound.
//
//       MEASURED 2026-09-22 at TWO resolutions, dgamma/dT = -2.6e-4 N/(m K),
//       mu = 3.25e-3 Pa s, A_sink = 0.8:
//
//        RE-MEASURED 2026-09-22 with the VERIFIED constants (Mohr et al. 2020,
//        d(gamma)/dT = -1.9e-4 N/(m K), mu = 4.0e-3 Pa s). The earlier table
//        ran -2.6e-4 and 3.25e-3, which made the Marangoni forcing 1.68x too
//        strong -- forcing goes as |d(gamma)/dT|/mu and both were wrong the
//        same way. dx = 4 um, against a conduction row of 2w 96.651, d 23.888,
//        w/d 4.05:
//
//          d(gamma)/dT    2w (um)   d (um)   w/d    peak u (m/s)
//          -1.4e-4        102.323   21.084   4.85      0.844
//          -1.9e-4        103.898   20.609   5.04      1.034      <- measured
//          -2.4e-4        104.724   19.930   5.25      1.204
//
//        so the increment at the measured value is +7.5 % on width and -13.7 %
//        on depth, NOT the +9.4 % and -18.3 % this banner reported with the
//        unverified constants. The band is the measurement's own +/- 0.5e-4,
//        which is +/-26 % on the coefficient and moves w/d from 4.85 to 5.25 --
//        so the aspect ratio is predicted to about +/-4 %, and that is the
//        honest precision of this tier.
//
//        The older two-grid table, kept because the GRID conclusion from it
//        still stands (the effect survives halving dx and is therefore physical
//        rather than numerical), was run at the wrong coefficient:
//        dx    quantity   conduction   + Marangoni   change   Ma    rho excursion
//        4 um    2w        96.651 um   105.736 um    + 9.4 %
//                d         23.888       19.519       -18.3 %  0.264    17.6 %
//        2 um    2w       100.911      107.221       + 6.3 %
//                d         25.916       20.552       -20.7 %  0.190     7.0 %
//
//       Wider and shallower is the correct SIGN for outward thermocapillary
//       flow at dgamma/dT < 0, and it is the direction a conduction-only model
//       structurally cannot produce. Peak speed 1.39 m/s at dx = 4, 2.00 m/s
//       at dx = 2.
//
//       TWO RESOLUTIONS MATTER MORE THAN ONE HERE, because the increment is
//       the whole claim and a single grid cannot separate it from a
//       discretisation artefact. The sign and the magnitude survive halving dx
//       (+9.4/+6.3 % on width, -18.3/-20.7 % on depth), so the effect is
//       physical rather than numerical. It is NOT converged -- the two rows
//       differ by about 3 points on width -- so the increment is quotable as
//       "order 10 % wider, order 20 % shallower" and no further.
//
//       AND THE COMPRESSIBILITY PROBLEM IMPROVES UNDER REFINEMENT, which was
//       not obvious: dt goes as dx^2 while dx goes as dx, so the LATTICE
//       velocity falls as dx even though the physical speed rose. Ma 0.264 ->
//       0.190 and the density excursion 17.6 -> 7.0 %. So refinement buys
//       accuracy twice here, and the Ma < 0.1 this tier wants is about one
//       further halving away -- at 8x the cost of the dx = 2 run.
//
//       THREE THINGS MAKE THIS COARSE, AND ALL THREE ARE OPEN.
//        1. Ma = 0.26 and the density excursion is 17.6 %. dt is set by the
//           THERMAL problem and is far too long for the flow, so the
//           compressibility error is not small. The number above is an
//           increment, not a prediction, and this is the main reason.
//        2. The fluid's free surface is HALF A CELL below the scalar's. The
//           scalar's zero-flux plane is the top face of cell nz-1; SpecWall's
//           mirror would be half a cell outside the last fluid node, i.e. at
//           nz, outside the array -- so within one Domain the two surfaces
//           cannot coincide. SpecNode is used instead, which collides (and so
//           can carry the stress) but sits on the node.
//        3. TIER (d) IS SLOW, AND THE COST IS THE FLUID -- NOT, AS AN EARLIER
//           VERSION OF THIS BANNER CLAIMED, THE FORCE LOOP. The force field was
//           rebuilt on the HOST with six mirror copies per step; it now runs as
//           one Kokkos::parallel_for over the enthalpy and velocity Views, with
//           byte-identical output (105.736 / 19.519 / 149.476 um and peak
//           |u| = 1.5229e-01 before and after). Measured at dx = 4 um on four
//           threads:
//
//               scalar alone        3.74 s
//               + coupled flow     43.63 s    the FLUID adds 10.7x the scalar
//               host force loop    50.15 s    the loop was 13 % of the total
//
//           So the move to the device is worth 1.15x here and is the right
//           implementation -- on an actual device those mirror copies are
//           off-chip traffic rather than a no-op, which is where it would
//           matter -- but it was NEVER what blocked dx = 2 um. What blocks that
//           is arithmetic: a D3Q27 fluid carries 27 populations per node
//           against the scalar's 7, and dx = 2 um is 32x the work (8x cells,
//           4x steps). Estimated 23 minutes from the dx = 4 run; MEASURED 31
//           minutes, so the estimate was 35 % light -- the extra is memory
//           traffic at 4.6 M nodes carrying 27 fluid populations beside 7
//           scalar ones. Slow, not impossible.
//        4. d(gamma)/dT IS NOW VERIFIED, AND THE VALUE THIS CASE USED WAS WRONG.
//           Mohr et al. (2020) measured Ti64 containerless by electromagnetic
//           levitation on the ISS and fit
//               gamma(T) = (1.493 +/- 0.008) - (1.9 +/- 0.5)e-4 (T - 1933 K) N/m
//           so d(gamma)/dT = -1.9e-4 N/(m K) at the liquidus. This case ran
//           -2.6e-4, which is 37 % high and OUTSIDE the measured band. The
//           viscosity was 3.25e-3 Pa s against a measured 4e-3, 19 % low.
//           Marangoni forcing goes as |d(gamma)/dT| / mu, so the two errors
//           compounded and the forcing was 1.68x too strong.
//           A third figure, -1.38e-3 N/(m K), turns up in secondary sources; it
//           is 7x Mohr's, larger in magnitude than iron's -4.3e-4, and arrived
//           without a traceable citation. It is not used.
//           THE SIGN IS SETTLED FOR CLEAN MATERIAL: negative, measured in
//           microgravity with no crucible. The surfactant sign flip remains
//           real for sulfur- or oxygen-bearing melts -- it is established for
//           Fe, Ni, Cu and Ag -- and would INVERT the aspect-ratio change, so
//           a contaminated feedstock is still outside what this case models.
//           The +/-0.5e-4 uncertainty is +/-26 % and is a band to sweep with
//           -dgdt, not a number to quote alone.
//
//   (e) `-evap`: EVAPORATIVE COOLING. Hertz-Knudsen mass flux with a
//       Clausius-Clapeyron vapour pressure, q_evap = m_dot L_v subtracted from
//       the surface flux in the SAME source term the beam enters, because above
//       the boiling point the two are comparable and splitting them lets the
//       surface overshoot between them. NOT CLAMPED: keyhole.cpp clamps this
//       identical feedback and its own banner records that the clamp became the
//       integrator, so here the balance is REPORTED instead.
//
//       THIS CASE AND validation/keyhole.cpp DISAGREE ABOUT L_v BY 3.4x, AND AN
//       EARLIER VERSION OF THIS BANNER CLAIMED THEY WERE MATCHED. They are not.
//       keyhole.cpp:230 declares Lv = 8.86e6 J/kg as its struct default, but
//       line 280 OVERRIDES it to 2.6e6 for the Cunningham configuration, and
//       that override is what feeds its Clausius-Clapeyron exponent (line 586)
//       and its evaporative cooling (line 597). The default was matched here;
//       the value that case actually runs was not.
//         This one is right: L_v(Ti) = 425 kJ/mol / 0.047867 kg/mol = 8.879e6
//       J/kg, so 8.86e6 is the physical latent heat of vaporisation and 2.6e6
//       is a "table 1" figure inherited from Muhammad, Rogers & Li (2013) --
//       plausibly an EFFECTIVE value for a model carrying thirteen fitted
//       constants rather than a measured one.
//         It matters more than a 3.4x usually would, because L_v enters the
//       vapour pressure as exp(L_v/R_s (1/T_b - 1/T)) -- in the EXPONENT -- so
//       the disagreement GROWS with temperature. Measured, P_v in Pa:
//
//           T (K)     L_v = 8.86e6     L_v = 2.6e6     ratio
//            3400        1.49e5           1.13e5        1.3
//            3500        2.29e5           1.29e5        1.8
//            3800        7.22e5           1.80e5        4.0
//            4200        2.59e6           2.62e5        9.9
//            4800        1.18e7           4.10e5       28.9
//
//       (An earlier draft of this paragraph said "orders of magnitude at
//       3500 K". It is 1.8x at 3500 K. The arithmetic was run because the
//       sentence was about to be committed without it.)
//         So the two cases agree closely just above boiling, where keyhole
//       spends most of its time, and diverge hard in the regime tier (f)
//       reaches: at 1200 W this case's surface is 5162 K, where the gap is
//       nearly 30x. Evaporation rates are NOT comparable between the two cases
//       at high power, and nothing there should be read across.
//
//       MEASURED 2026-09-22, dx = 4 um. THE POOL BARELY MOVES AND THE SURFACE
//       TEMPERATURE IS TRANSFORMED, which is the opposite of what was expected:
//
//         tier              2w (um)   d (um)   peak surface T   q_evap/q_peak
//         conduction         96.651   23.888      4710 K             --
//         + evap             96.422   23.369      3238 K           0.065
//         + flow            105.736   19.519      2913 K             --
//         + flow + evap     105.533   19.430      2902 K           0.011
//
//       Evaporative cooling is a THERMOSTAT. q_evap rises exponentially in T,
//       so the surface self-limits just BELOW the boiling point -- 3238 K
//       against 3315 -- and at that point consumes only 6.5 % of the beam. It
//       costs 0.2 % of the width and 2.2 % of the depth. So the conduction
//       model's 4710 K was the unphysical part, not its pool size, and the two
//       failures were never the same size.
//       With Marangoni also on, convection has already removed most of the
//       heat and evaporation has little left to do: 1.1 % of the beam.
//
//       AND THAT UNDOES THE KEYHOLE CRITERION'S VERDICT AT THIS POINT. The
//       criterion reports the ANALYTIC reference's peak, 4981 K, and condemns
//       tier (c). The reference has neither evaporation nor convection. With
//       both, the SIMULATED surface is 2902 K, 400 K below boiling. The
//       operating point is admissible; it was the conduction-only criterion
//       that was not. The criterion now says which problem it bounds.
//
//       WHAT THIS TIER DOES NOT DO: no mass loss and no recession -- the
//       surface is flat and stays flat -- and no recoil pressure. If q_evap
//       ever exceeds the beam the case says so and says that the missing mass
//       loss makes the reported pool a lower bound on the error. beta_r = 0.18
//       is the one non-measured number, the conventional Anisimov
//       retro-diffusion value, and -betar sweeps it.
//
//   (f) `-recede`: THE EVAPORATED MASS ACTUALLY LEAVES, at v = m_dot/rho_liquid.
//       No recoil, no melt ejection, NO FITTED CONSTANTS -- validation/keyhole.cpp's
//       recession has thirteen. Only liquid evaporates; evaporating solid would
//       be sublimation, which is not this process. Cells above the receding
//       surface are voided and their enthalpy goes with them.
//
//       MEASURED 2026-09-22, dx = 4 um, v = 700 mm/s:
//
//         P (W)   peak surface T   recession    removed
//           75        3238 K         0.34 um       8 ng
//          150        3966 K         6.76 um     313 ng
//          400        4461 K        34.22 um    2326 ng
//          800        4736 K        75.94 um    5947 ng
//         1200        5162 K       117.68 um    9596 ng
//
//       16x the power buys 1.59x the temperature and 1200x the removal, so the
//       beam does go mostly into vaporising rather than heating. THE FLATNESS
//       WAS OVERSTATED ONCE: an earlier sweep read the peak at layer nz-1 and
//       reported 4115 / 4197 / 4216 K at 400 / 800 / 1200 W, from which
//       "3x the power, 101 K" was concluded. nz-1 is VOID once the surface has
//       receded and a voided cell inverts to the solidus. The true rise over
//       that range is 701 K. Commit af33a99 retracts it.
//
//       ANYTHING MEANING "THE SURFACE" MUST ASK THE COLUMN, and three sites in
//       this file did not -- the pool extraction, the frame dump and the tier
//       (e) report. They failed DIFFERENTLY, which is why it took three finds:
//       a nonsense L (41.7 um against 190.6), a false uniform trail in a
//       picture, and a PLAUSIBLE temperature that was only caught because the
//       figure and the case's own report disagreed by 758 K. The plausible one
//       is the expensive kind.
//
//       Grid sensitivity, 1200 W: 117.68 um at dx = 4 against 115.66 at dx = 2,
//       1.7 % apart -- against validation/keyhole.cpp's 12-16 % over the same
//       refinement. A recession with no fitted constants is far less grid
//       sensitive than a calibrated one, which is the point of having built it.
//
//       NOT MODELLED: no recoil pressure and no melt ejection, so the removal
//       is a LOWER BOUND on what a real keyhole loses. beta_r = 0.18 is the one
//       non-measured constant; -betar sweeps it.
//
//  STAGE 4 -- THE EXPERIMENTAL COMPARISON -- IS BLOCKED, AND THE TWO REASONS
//  ARE WORTH MORE THAN A WEAK COMPARISON WOULD HAVE BEEN. Searched 2026-09-21.
//
//   1. THE BARE-PLATE SINGLE-TRACK BENCHMARK IS FOR IN625, NOT THIS ALLOY.
//      NIST AM-Bench AMB2018-02 -- the well-characterised bare-plate laser
//      track set with published width, depth and length -- is Nickel Alloy 625.
//      This case cites A-AMB2022-01 for its BEAM SIZE only, which is legitimate
//      and is a different dataset from the dimensional one. So there is nothing
//      here to compare a Ti-6Al-4V pool against, and comparing it against an
//      IN625 benchmark would be comparing two alloys.
//      NOTE FOR WHOEVER DOES THE IN625 ROUTE: AMB2018-02's COMMANDED powers
//      were 150/195/195 W, the laser calibration was found to be wrong, and the
//      TRUE powers are 137.9 W at 400 mm/s and 179.2 W at 800 and 1200 mm/s.
//      Quoting the commanded value would be exactly the "a wrong constant is
//      still a consistent simulation" failure this tree records.
//
//   2. d(gamma)/dT FOR TI-6AL-4V IS NOT VERIFIED HERE, SO THE BAND CANNOT BE
//      SET. The -2.6e-4 N/(m K) in Opts is a plausible clean-alloy figure and
//      is NOT traced to a primary source. The surfactant sign flip is firmly
//      established for Fe, Ni, Cu and Ag -- sulfur and oxygen bonds break as T
//      rises, so d(gamma)/dT is POSITIVE below a critical temperature and
//      negative above, with a surface-tension maximum at the crossing -- but it
//      was NOT established for this alloy in what was searched. Since the sign
//      decides whether the pool goes wider/shallower or narrower/deeper, tier
//      (d)'s -18.3 % depth change is unquotable until it is settled.
//
//  WHAT STAGE 4 DID SETTLE: A SUB-KEYHOLE OPERATING POINT, and it exposes a
//  tension the earlier tiers hid. Scanning the reference at A = 0.33 and the
//  122.5 um spot, against a 3315 K boiling point:
//
//      P (W)  v (mm/s)  peak T   2w (um)  d (um)   verdict
//        75      700     4980     102.98   26.07   keyholing (this case's default)
//        60     1000     3564      80.24   14.73   keyholing
//        50     1000     3019      68.70   10.67   SUB-KEYHOLE, 296 K margin
//        60     1500     3061      67.82    9.57   SUB-KEYHOLE
//
//  SUB-KEYHOLE MEANS SHALLOW, AND SHALLOW MEANS BADLY RESOLVED. Every pool deep
//  enough to resolve comfortably at dx = 2-4 um is above the boiling point; the
//  legal points are about 10 um deep, i.e. FIVE cells at dx = 2 um and eleven
//  at dx = 1 um. So a physically coherent tier (c) or (d) run needs dx <= 1 um,
//  which is 36.9 M nodes for the conduction case alone and more with a D3Q27
//  fluid beside it. That is the real cost of making this case physical, and it
//  is a resolution problem rather than a modelling one.
//  (Two rows of that scan printed a zero pool with a peak above the liquidus;
//  that is a golden-section bracket failing on a very small melt region in the
//  scan script, not a physical result, and it is why only the rows with a
//  resolved envelope are quoted.)
//
//       And one bug this tier already caught, recorded because it was silent:
//       the surface force was first written into a SpecWall ghost, which does
//       not collide, so the whole coupled flow read u = 0 with a visibly
//       non-zero force field. "Delete the coupling and see whether the answer
//       moves" found it -- the coupled and conduction runs agreed to every
//       digit, which is the signature of a term that is not contributing.
//
//  THE REFERENCE IS CHECKED BEFORE THE SOLVER RUNS, because the reference IS
//  the claim here and a reference wrong in the sixth digit validates the wrong
//  answer. Five identities, all cheap, all printed:
//    R1  static peak at v = 0 against A P /(2 k sigma sqrt(2 pi))
//    R2  static surface field against (peak) exp(-u) I_0(u), four radii
//    R3  -k dT/dz at the surface against q''(x,y), at v = 0, 0.7, 2.0 m/s --
//        the only check that the reference delivers exactly A P ONCE at v != 0,
//        and the strongest available guard on the surface image factor
//    R5  the sigma -> 0 Rosenthal limit, sigma halved six times: the ERROR
//        RATIO must tend to 4. Asserting the ratio rather than a single small-
//        sigma value is what catches a wrong prefactor, which would leave a
//        non-vanishing offset that one point hides inside a tolerance.
//    R6  lateral symmetry of the field
//  Measured 2026-09-21 in a standalone host build: R1 and R2 at 2.2e-16, R3 at
//  6.1e-05 / 1.1e-04 / 1.8e-04 (the one-sided difference step, not the
//  reference), R5 ratios 3.10, 3.78, 3.95, 3.99.
//
//  POOL DIMENSIONS ARE ENVELOPES OVER xi, NEVER A SLICE, and this is the trap
//  most worth stating. The xi at which the pool is widest is not the xi at
//  which it is deepest -- measured here at -24.58 and -55.98 um -- so a single
//  y-z slice through the beam centre reads 2w = 95.28 and d = 16.72 um against
//  the envelope's 102.98 and 26.07: -7.5 % and -35.9 %, silently, and worse at
//  higher Peclet. A solidified track's transverse section shows the envelope.
//
//  WHAT THIS DOES NOT DO, and each one is a real limitation rather than a
//  caveat for form's sake:
//   - NO FLUID FLOW. No Marangoni, so no thermocapillary correction to the
//     aspect ratio, which in reality widens and shallows a conduction pool.
//     This is the largest omission and it is not small.
//   - CONSTANT PROPERTIES, because Eagar & Tsai assume them. cp is the enthalpy
//     mean over 293-1878 K and k the Kirchhoff mean over the same range; the
//     room-temperature k = 6.7 W/(m K) that circulates for this alloy
//     over-predicts depth by about 30 % and length by about 133 %.
//   - NO POWDER LAYER. A bare polished plate, which is what the beam
//     measurement and the analytic solution both assume.
//   - NO VAPORISATION and no recoil. AN EARLIER VERSION OF THIS BANNER CLAIMED
//     THE OPERATING POINT WAS CHOSEN TO SIT BELOW THE KEYHOLE THRESHOLD AND
//     THAT THE CRITERION WAS PRINTED. NEITHER WAS TRUE: no criterion was
//     printed, and the reference's own peak surface temperature at
//     P = 75 W, v = 0.7 m/s, A = 0.33 is about 4600 K against a boiling point
//     near 3315 K -- some 1300 K INTO vaporisation. The criterion is printed
//     now, and it FAILS, deliberately and visibly.
//     What that costs, precisely: NOTHING for tier (a), because tier (a) is a
//     comparison of two solutions of the SAME linear conduction problem and
//     neither side contains vaporisation -- it validates the solver, and a
//     solver validation does not require the boundary data to be physically
//     attainable. It invalidates tier (c): these dimensions must NOT be
//     compared against a real single track at this power and speed, because a
//     real one would be keyholing and this model cannot represent that. A
//     sub-keyhole operating point for tier (c) has to be chosen separately and
//     has not been.
//   - NO SOLIDIFICATION MICROSTRUCTURE, no residual stress, no track geometry
//     beyond the melt isotherm.
//
//  NOTHING IN THIS BANNER IS A SIMULATION RESULT YET. The reference numbers
//  above are measured; the solver's own agreement, its convergence order and
//  the latent-heat increment belong here once the rows below have been run.
//==============================================================================
#include "collision/EnthalpyBGK.hpp"
#include "collision/EnthalpyRegularised.hpp"
#include "collision/ScalarBGK.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"
#include "collision/BGK.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "solver/FluidSolver.hpp"

#include "NpyDump.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace lbm;

namespace {

int failures = 0, checks = 0;

void verdict(const char* what, double got, double want, double tol,
             const char* unit = "") {
  ++checks;
  const bool ok = std::abs(got - want) <= tol;
  if (!ok) ++failures;
  std::printf("  %-50s %13.6f %-4s (want %.6f +/- %.6f)   %s\n",
              what, got, unit, want, tol, ok ? "PASS" : "FAIL");
}

//------------------------------------------------------------------------------
//  THE ANALYTIC REFERENCE. double throughout, always, even in an FP32 build:
//  the integrand reaches 1e-155 in the far field and FP32 flushes that to zero,
//  turning a smooth decay into a step. The reference has no reason to share the
//  solver's precision.
//------------------------------------------------------------------------------
namespace et {

const double GLX[10] = {
 0.0765265211334973,0.2277858511416451,0.3737060887154195,0.5108670019508271,
 0.6360536807265150,0.7463319064601508,0.8391169718222188,0.9122344282513259,
 0.9639719272779138,0.9931285991850949};
const double GLW[10] = {
 0.1527533871307258,0.1491729864726037,0.1420961093183820,0.1316886384491766,
 0.1181945319615184,0.1019301198172404,0.0832767415767048,0.0626720483341091,
 0.0406014298003869,0.0176140071391521};

struct P { double xs, ys, zs, Pe, phimax; };

inline double f(double phi, const P& p) {
  const double c = std::cos(phi), s = std::sin(phi);
  if (c <= 0.0 || s <= 0.0) return 0.0;
  // Written so the finite quantity stays finite: forming tan^2 phi and
  // squaring it overflows near phi = pi/2 where the integrand is O(1).
  const double A = p.xs * c + p.Pe * s * s / c;
  const double ct = c / s;
  const double e = -0.5 * (A * A + p.ys * p.ys * c * c) - 0.5 * p.zs * p.zs * ct * ct;
  return (e > -700.0) ? std::exp(e) : 0.0;
}

double gl20(const P& p, double a, double b, int m) {
  const double h = (b - a) / m;
  double sum = 0.0;
  for (int k = 0; k < m; ++k) {
    const double mid = a + (k + 0.5) * h, hw = 0.5 * h;
    for (int i = 0; i < 10; ++i)
      sum += GLW[i] * (f(mid + hw * GLX[i], p) + f(mid - hw * GLX[i], p));
  }
  return sum * 0.5 * h;
}

// GRADED BREAKPOINTS, NOT UNIFORM PANELS. The exp(-z*^2 cot^2 phi / 2) factor
// rises 0 -> 1 in a layer of relative width ~z* at the START of the interval,
// so uniform panels resolve it only at about 1/z* panels: 256 at z* = 0.02,
// 32768 at z* = 8e-5. A fixed rule therefore fails silently at exactly the
// near-surface points this case must evaluate. The per-segment tolerance also
// carries an absolute floor referred to the total, because a purely relative
// one can never be met by a segment worth 1e-303 of the answer.
double integral(const P& p, double tol = 1e-11) {
  std::vector<double> bp{0.0};
  if (p.zs != 0.0) {
    const double az = std::abs(p.zs);
    for (double c : {40.,20.,10.,5.,2.,1.,0.5,0.2,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7}) {
      const double phi = std::atan(az / c);
      if (phi > 1e-300 && phi < p.phimax) bp.push_back(phi);
    }
  }
  bp.push_back(p.phimax);
  std::sort(bp.begin(), bp.end());
  bp.erase(std::unique(bp.begin(), bp.end()), bp.end());
  const int nseg = int(bp.size()) - 1;
  if (nseg <= 0) return 0.0;

  double est = 0.0;
  for (int i = 0; i < nseg; ++i) est += gl20(p, bp[i], bp[i + 1], 4);
  const double flr = tol * std::abs(est) / nseg;

  double total = 0.0;
  for (int i = 0; i < nseg; ++i) {
    int m = 4;
    double prev = gl20(p, bp[i], bp[i + 1], m), cur = prev;
    for (int it = 0; it < 14; ++it) {
      m *= 2; cur = gl20(p, bp[i], bp[i + 1], m);
      if (cur == 0.0 || std::abs(cur - prev) <= std::max(tol * std::abs(cur), flr)) break;
      prev = cur;
    }
    total += cur;
  }
  return total;
}

// Modified Bessel I0, for the R2 closed form only.
double i0(double x) {
  if (x < 14.0) {
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 80; ++k) {
      t *= (x / (2.0 * k)) * (x / (2.0 * k)); s += t;
      if (t < 1e-18 * s) break;
    }
    return s;
  }
  double s = 1.0, a = 1.0;
  for (int k = 1; k < 16; ++k) {
    a *= (2.0 * k - 1.0) * (2.0 * k - 1.0) / (8.0 * x * k); s += a;
    if (std::abs(a) < 1e-17 * s) break;
  }
  return std::exp(x) / std::sqrt(2.0 * M_PI * x) * s;
}

struct Field {
  double AP, k, sigma, alpha, v, T0;
  double prefac() const { return AP / (std::sqrt(2.0) * std::pow(M_PI, 1.5) * k * sigma); }
  double Pe() const { return v * sigma / (2.0 * alpha); }
  double t0() const { return sigma * sigma / (2.0 * alpha); }
  // t > 0: the beam switched on at t = 0, which is the initial-value problem
  // the LBM actually solves -- the substitution u = tan^2 phi maps tau in [0,t]
  // to phi in [0, atan(sqrt(t/t0))], so the SAME integrand with a moved
  // endpoint is exact at finite time. That removes the quasi-steady assumption
  // from the primary comparison, which is what makes it affordable.
  // t <= 0: the quasi-steady limit, phimax = pi/2.
  double T(double xi, double y, double z, double t = -1.0) const {
    P p{xi / sigma, y / sigma, z / sigma, Pe(),
        t > 0.0 ? std::atan(std::sqrt(t / t0())) : 0.5 * M_PI};
    return T0 + prefac() * integral(p);
  }
};

// --- pool dimensions: ENVELOPES over xi ---
// `off` offsets the coordinate NOT being searched, because the simulation does
// not sample the planes y = 0 and z = 0: the halfway bounce-back closure puts
// the top cell CENTRE at z = dx/2 (Domain is non-periodic in z and every node
// is ScalarBulk, so Esoteric Pull reflects on the halfway plane), and the width
// is therefore read dx/2 BELOW the free surface. Comparing a width read at
// z = dx/2 against the reference at z = 0 is worth -2.84 um at dx = 4 um and
// -1.39 um at dx = 2 um -- first order, and the largest single term in what
// used to look like a 6 % model deficit. Measured 2026-09-21.
double outer_root(const Field& c, double xi, double Tiso, int axis, double h0, double t = -1.0,
                  double off = 0.0) {
  auto g = [&](double s) {
    return (axis == 0 ? c.T(xi, s, off, t) : c.T(xi, off, s, t)) - Tiso;
  };
  if (g(0.0) <= 0.0) return 0.0;
  double lo = 0.0, hi = h0;
  while (g(hi) > 0.0) { lo = hi; hi *= 2.0; if (hi > 1.0) return hi; }
  for (int i = 0; i < 80; ++i) { const double m = 0.5 * (lo + hi); (g(m) > 0.0 ? lo : hi) = m; }
  return 0.5 * (lo + hi);
}
double envelope(const Field& c, double Tiso, int axis, double a, double b, double* at, double t = -1.0,
                double off = 0.0) {
  const double gr = 0.6180339887498949;
  double x1 = b - gr * (b - a), x2 = a + gr * (b - a);
  double f1 = outer_root(c, x1, Tiso, axis, c.sigma, t, off), f2 = outer_root(c, x2, Tiso, axis, c.sigma, t, off);
  for (int i = 0; i < 90; ++i) {
    if (f1 < f2) { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (b - a); f2 = outer_root(c, x2, Tiso, axis, c.sigma, t, off); }
    else         { b = x2; x2 = x1; f2 = f1; x1 = b - gr * (b - a); f1 = outer_root(c, x1, Tiso, axis, c.sigma, t, off); }
  }
  if (at) *at = 0.5 * (x1 + x2);
  return std::max(f1, f2);
}
struct Pool { double w = 0, d = 0, L = 0, xi_w = 0, xi_d = 0; };
Pool pool(const Field& c, double Tiso, double t = -1.0, double wz = 0.0) {
  auto Tc = [&](double xi) { return c.T(xi, 0, wz, t) - Tiso; };
  double xhot = 0.0, best = Tc(0.0);
  for (double s = -12.0; s <= 4.0; s += 0.05) {
    const double xx = s * c.sigma, vv = Tc(xx);
    if (vv > best) { best = vv; xhot = xx; }
  }
  Pool P;
  if (best <= 0.0) return P;
  double lo = xhot, hi = xhot + c.sigma;
  while (Tc(hi) > 0.0) hi += c.sigma;
  for (int i = 0; i < 90; ++i) { const double m = 0.5 * (lo + hi); (Tc(m) > 0.0 ? lo : hi) = m; }
  const double xf = 0.5 * (lo + hi);
  lo = xhot; hi = xhot - c.sigma;
  while (Tc(hi) > 0.0) hi -= c.sigma;
  for (int i = 0; i < 90; ++i) { const double m = 0.5 * (lo + hi); (Tc(m) > 0.0 ? lo : hi) = m; }
  const double xr = 0.5 * (lo + hi);
  P.L = xf - xr;
  P.w = envelope(c, Tiso, 0, xr, xf, &P.xi_w, t, wz);   // on the sampled plane
  P.d = envelope(c, Tiso, 1, xr, xf, &P.xi_d, t);        // depth is measured from z = 0
  return P;
}

}  // namespace et

//------------------------------------------------------------------------------
// Material. Ti-6Al-4V, "Set M". Every line prints its own source, because this
// tree has one project whose manuscript admits its property table "was not
// traced to one single primary source" -- and that table gives +15 % width,
// +30 % depth and +133 % length against this one.
//------------------------------------------------------------------------------
struct Mat {
  double rho   = 4420.0;    // kg/m^3   Mills (2002); NIST/Ghosh 4428, 0.2 % apart
  double cp_s  = 690.0;     // J/(kg K) ENTHALPY MEAN 293-1878 K of the linear law
                            //          between Mills 546 @ 298 K and Cezairliyan
                            //          837 @ 1878 K. NOT the room-temperature value.
  double cp_l  = 831.0;     // J/(kg K) Mills (2002)
  double k_s   = 17.4;      // W/(m K)  KIRCHHOFF MEAN over 293-1878 K of the
                            //          Ghosh/NIST k(T) table (6.85 -> 27.5).
                            //          The 6.7 that circulates is a 300 K datum.
  double k_l   = 33.4;      // W/(m K)  Mills (2002)
  double T_s   = 1878.0;    // K        solidus,  Mills / NIST agree
  double T_l   = 1928.0;    // K        liquidus
  double L_f   = 2.86e5;    // J/kg     Mills. CONTESTED: Ghosh/NIST 3.65e5 (+28 %)
  double T_0   = 293.0;     // K        ambient, the datum both means integrate from
  double rc_s() const { return rho * cp_s; }
  double rc_l() const { return rho * cp_l; }
  double alpha_s() const { return k_s / rc_s(); }
  double alpha_l() const { return k_l / rc_l(); }
};

// Evaporation constants. L_v = 8.86e6 J/kg, T_b = 3315 K, R_s = R/M_Ti with
// M = 0.047867 kg/mol, P_0 = 1 atm.
//
// NOT MATCHED TO validation/keyhole.cpp, although an earlier comment here said
// they were. That case overrides its own Lv default to 2.6e6 (keyhole.cpp:280)
// and runs the override, so the two differ by 3.4x in the CLAUSIUS-CLAPEYRON
// EXPONENT. 8.86e6 is the physical value -- 425 kJ/mol over 47.867 g/mol --
// and is kept; see this file's banner for why the two are not comparable.
struct Evap {
  double Lv = 8.86e6;                  // J/kg   latent heat of vaporisation
  double Tb = 3315.0;                  // K      boiling point at P_0
  double Rs = 8.314 / 0.047867;        // J/(kg K)
  double P0 = 101325.0;                // Pa
  double beta_r = 0.18;                // retro-diffusion: the fraction of
                                       // vapour that recondenses. THE ONE
                                       // NON-MEASURED NUMBER HERE -- see the
                                       // banner; 0.18 is the conventional
                                       // Anisimov value and -betar sweeps it.
};

struct Opts {
  double P = 75.0;          // W
  double v = 0.700;         // m/s
  double spot = 122.5;      // um, 1/e^2 DIAMETER (NIST A-AMB2022-01)
  double A = 0.33;          // absorptivity; band [0.27, 0.36]
  double dx = 4.0e-6;       // m
  double Lx = 480e-6, Ly = 320e-6, Lz = 240e-6;
  double track = -1.0;      // m of travel; < 0 = fill the domain
  bool   la0 = false;       // tier (a): latent heat and the property split OFF
  bool   bgk = false;       // EnthalpyBGK instead of the regularised default
  bool   conv = false;      // the grid ladder
  int    probe = 0;         // analytic field comparison every N steps
  std::string frames;       // directory for animation frames, empty = off
  int    fevery = 0;        // frame interval in steps; 0 = aim for ~150 frames

  // ---- TIER (d): the coupled flow. Off by default, so every number this case
  // ---- already reports is unchanged and the validated conduction row stays
  // ---- exactly what validation/melt_pool.cpp was validated as.
  bool   flow  = false;     // Marangoni + mushy sink + advection of H by u
  // VERIFIED 2026-09-22 against Mohr et al. (2020), "Precise Measurements of
  // Thermophysical Properties of Liquid Ti-6Al-4V (Ti64) Alloy On Board the
  // International Space Station", Adv. Eng. Mater. -- containerless
  // electromagnetic levitation in microgravity, which fits
  //     gamma(T) = (1.493 +/- 0.008) - (1.9 +/- 0.5)e-4 (T - 1933 K)  N/m
  // The reference temperature 1933 K is this alloy's liquidus, so the
  // coefficient applies directly here.
  double dgdT  = -1.9e-4;   // N/(m K)  Mohr et al. (2020). Band +/- 0.5e-4,
                            //          which is +/-26 % -- sweep it with -dgdt.
  double mu_l  = 4.0e-3;    // Pa s     Mohr et al. (2020), at the liquidus
  double A_lat = 0.8;       // mushy sink strength at f_l = 0, IN LATTICE UNITS.
                            //          validation/mushy_sink.cpp measures the
                            //          bound at 1.0 and recommends <= 0.8.

  // ---- TIER (e): evaporative COOLING. Off by default.
  bool   evap  = false;     // subtract m_dot * L_v from the surface flux
  double betar = 0.18;      // retro-diffusion coefficient
  bool   recede = false;    // TIER (f): let the evaporated mass actually leave
  double rho_l = 4130.0;    // kg/m^3 liquid density at Tm, for m_dot -> v_rec
};

}  // namespace

//==============================================================================
template <class L, class Coll>
int run(const Opts& o, const Mat& m) {
  const double sigma = o.spot * 1e-6 * 0.25;      // 1/e^2 diameter -> sigma = a/2
  const double a_beam = 2.0 * sigma;              // 1/e^2 radius
  const double dx = o.dx;
  // dt from a fixed lattice diffusivity on the LIQUID (the larger alpha, so it
  // is tau_max that is pinned), chosen so the beam advances exactly 1/K cells
  // per step with K an integer. K then doubles exactly under refinement, which
  // makes the sub-cell beam phase periodic with period K.
  const int    K  = int(std::llround(dx / (o.v * (dx * dx / m.alpha_l() * 0.249817))));
  const double dt = dx / (o.v * K);
  const double Dls = m.alpha_s() * dt / (dx * dx);
  const double Dll = m.alpha_l() * dt / (dx * dx);

  const Index nx = Index(std::llround(o.Lx / dx));
  const Index ny = Index(std::llround(o.Ly / dx));
  const Index nz = Index(std::llround(o.Lz / dx));
  Domain d(nx, ny, nz, false, false, false);

  // The gauge: divide every volumetric quantity by (rho c)_s, so H is in
  // kelvin and cp_s == 1 exactly. T_s and T_l stay literal kelvin.
  PhaseChange pc;
  pc.T_s = Real(m.T_s); pc.T_l = Real(m.T_l);
  pc.cp_s = Real(1);
  pc.cp_l = Real(o.la0 ? 1.0 : m.rc_l() / m.rc_s());
  pc.k_s  = Real(Dls * 1.0);
  pc.k_l  = Real(o.la0 ? Dls * 1.0 : Dll * double(pc.cp_l));
  pc.La   = Real(o.la0 ? 1e-30 : m.L_f / m.cp_s);
  pc.E_datum = Real(0);

  Coll coll;
  coll.set_material(pc);
  coll.T_ref = pc.enthalpy_of(Real(m.T_0), Real(0));   // an ENTHALPY despite the name

  ScalarSolver<L, EsotericPull<L>, Coll> s(d, coll);
  s.set_geometry([](Index, Index, Index) -> std::uint8_t { return ScalarBulk; });
  s.finalize_geometry();     // silent if omitted; see validation/stefan.cpp
  s.initialize(coll.T_ref);

  // ---- TIER (d): the fluid. D3Q27 is not a choice -- every Navier-Stokes
  // ---- operator static_asserts supports_navier_stokes, false on D3Q7, so a
  // ---- 3-D fluid on the scalar's own lattice is a compile error. The scalar
  // ---- stays D3Q7; CLAUDE.md's pairing note applies and the cost is memory
  // ---- traffic, 27 populations per node against 7.
  using LF   = D3Q27;
  using FColl = BGK<LF, SecondOrderEquilibrium<LF>, FieldGuo, ShiftedPopulations>;
  const double nu_phys = o.mu_l / m.rho;
  const double nu_lat  = nu_phys * dt / (dx * dx);
  // a physical force density [N/m^3] -> lattice, with rho_lat = 1
  const double F_to_lat = dt * dt / (dx * m.rho);

  View1D<Real> Fx, Fy, Fz;
  std::unique_ptr<FluidSolver<LF, EsotericPull<LF>, FColl>> fs;
  if (o.flow) {
    Fx = View1D<Real>("Fx", d.n_padded);
    Fy = View1D<Real>("Fy", d.n_padded);
    Fz = View1D<Real>("Fz", d.n_padded);
    FColl fc;
    fc.omega = FColl::omega_from_viscosity(Real(nu_lat));
    fc.forcing = FieldGuo{};
    fc.forcing.Ex = Fx; fc.forcing.Ey = Fy; fc.forcing.Ez = Fz;
    fs = std::make_unique<FluidSolver<LF, EsotericPull<LF>, FColl>>(d, fc);
    // THE TWO SURFACES MUST COINCIDE. The scalar's zero-flux plane is the top
    // FACE of the last cell (implicit halfway bounce-back); SpecWall's mirror
    // is half a cell outside the last fluid node, i.e. the same plane.
    // set_specular_nodes would put the fluid surface half a cell off the
    // thermal one -- validation/marangoni.cpp is sized around exactly this.
    fs->set_geometry([&](Index, Index, Index) -> CellType { return Fluid; });
    // ON-NODE, NOT THE GHOST, AND THIS IS A COMPROMISE THAT COST A BUG.
    // SpecWall is a ghost: it does not collide, so a body force written into it
    // is DISCARDED -- which is exactly what happened, and the whole coupled
    // flow read u = 0 with the force field visibly non-zero. Its mirror also
    // sits half a cell OUTSIDE the last fluid node, so landing it on the
    // scalar's surface (the top face of cell nz-1) would need a ghost at nz,
    // outside the array. Within one Domain the two surfaces cannot both sit on
    // that plane.
    // SpecNode puts the plane ON node nz-1, which DOES collide and can carry
    // the Marangoni stress. The cost is that the fluid's free surface is then
    // half a cell below the scalar's. That offset is recorded rather than
    // hidden, and it is the first thing to fix if this tier is ever promoted
    // past "measured increment".
    fs->set_specular_nodes([&](Index, Index, Index z) -> std::uint8_t {
      return (z == nz - 1) ? SpecZp : SpecNone;
    });
    fs->initialize(Real(1));
    s.set_velocity(fs->ux(), fs->uy(), fs->uz());
    std::printf("\n  TIER (d): COUPLED FLOW ON.  mu = %.4e Pa s  nu = %.4e m^2/s\n",
                o.mu_l, nu_phys);
    std::printf("    nu_lat = %.6f  tau_f = %.6f   d(gamma)/dT = %.3e N/(m K)"
                "   A_sink = %.2f\n", nu_lat,
                1.0 / double(FColl::omega_from_viscosity(Real(nu_lat))),
                o.dgdT, o.A_lat);
    if (nu_lat < 0.02)
      std::printf("    WARNING: tau_f is close to the 1/2 floor; the fluid is "
                  "under-relaxed at this dx and dt.\n");
  }

  // Seed self-check: invert the seeded enthalpy and require ambient back. A
  // gauge error is otherwise invisible -- it produces a plausible, converging,
  // wrong run, which is exactly what it did in validation/keyhole.cpp.
  {
    // coll.material() and NOT the local pc: set_material takes its argument BY
    // VALUE and normalises its own copy, so `pc` here still carries the
    // defaults H_l_ = 1, dTm_ = 0. Inverting through it puts every T above the
    // default liquidus 49 K low in this gauge and makes `T >= T_l` select the
    // 1879 K contour instead of 1928 K. PhaseChange::ready() exists and says
    // so; nothing was asking it. validation/keyhole.cpp has the same defect.
    const PhaseChange& pn = coll.material();
    if (!pn.ready()) {
      std::printf("melt_pool: PhaseChange not normalised. NOTHING WAS RUN.\n");
      return 1;
    }
    Real fl, T, E, dEdT; pn.invert(coll.T_ref, fl, T, E, dEdT);
    if (std::abs(double(T) - m.T_0) > 1e-6) {
      std::printf("melt_pool: seeded H inverts to %.3f K, not ambient %.3f K. "
                  "NOTHING WAS RUN.\n", double(T), m.T_0);
      return 1;
    }
    // THE AMBIENT CHECK ALONE IS BLIND, and that is how the un-normalised copy
    // survived: at ambient H < 0 the SOLID branch reads only T_s and cp_s,
    // which the caller sets directly. The band edges are what exercise H_l_ and
    // dTm_, so round-trip both -- the liquidus is the contour every pool
    // dimension in this file is defined by.
    Real f2, T2, E2, c2;
    pn.invert(pn.enthalpy_of(Real(m.T_l), Real(1)), f2, T2, E2, c2);
    verdict("seed: liquidus enthalpy inverts to T_l", double(T2), m.T_l, 1e-3, "K");
    pn.invert(pn.enthalpy_of(Real(m.T_s), Real(0)), f2, T2, E2, c2);
    verdict("seed: solidus enthalpy inverts to T_s", double(T2), m.T_s, 1e-3, "K");
    if (failures) {
      std::printf("melt_pool: the phase-change map does not round-trip its own "
                  "band edges. NOTHING WAS RUN.\n");
      return 1;
    }
  }

  const double x0 = 1.6 * a_beam;
  const double travel = o.track > 0 ? o.track : (o.Lx - 2.0 * x0);
  const long steps = long(std::ceil(travel / (o.v * dt)));
  // THE BEAM AXIS GOES ON A CELL CENTRE, NOT ON 0.5*Ly. Every rung of this
  // ladder has ny even (Ly/dx = 80, 160, 320), so 0.5*Ly falls on the FACE
  // between cells ny/2-1 and ny/2, and the column j = ny/2 that the depth and
  // length are read along then sits dx/2 off-axis while w_sim's datum -- the
  // index ny/2, not its centre -- is short by the same dx/2. Both are first
  // order in dx and both bias the reported dimension LOW. Moving the beam is
  // the fix that REMOVES the error at every rung rather than correcting it by a
  // half cell afterwards; the analytic reference is unchanged, because it is
  // symmetric about its own axis wherever that axis is put.
  const double yc = (double(ny / 2) + 0.5) * dx;
  const double q_peak = 2.0 * o.A * o.P / (M_PI * a_beam * a_beam);
  const double flux_to_K = dt / (m.rc_s() * dx);
  const double t_end_for_peak = double(steps) * dt;

  // k_s on BOTH tiers and not a choice: in tier (a) pc.k_s == pc.k_l so the
// LBM's diffusivity is alpha_s everywhere, and in tier (b) there is no
// analytic reference to select a k for. A ternary here read as though a
// conductivity were switched somewhere; it never was.
  et::Field ref{o.A * o.P, m.k_s, sigma,
                m.alpha_s(), o.v, m.T_0};

  std::printf("\nmelt_pool: %s  %s  %s\n", Coll::name, L::name,
              o.la0 ? "TIER (a): La = 0, matched properties -- the VALIDATED row"
                    : "TIER (b): latent heat ON -- a measured increment, NOT validated");
  std::printf("  P = %.1f W   v = %.0f mm/s   1/e^2 diameter = %.1f um -> sigma = %.4f um"
              "   A = %.3f\n", o.P, o.v * 1e3, o.spot, sigma * 1e6, o.A);
  std::printf("  grid %dx%dx%d = %.2f M nodes   dx = %.2f um   dt = %.6e s   K = %d   steps = %ld\n",
              int(nx), int(ny), int(nz), double(nx) * ny * nz / 1e6, dx * 1e6, dt, K, steps);
  std::printf("  alpha_s = %.6e  alpha_l = %.6e m^2/s   cp_l = %.6f  La = %.4f K\n",
              m.alpha_s(), m.alpha_l(), double(pc.cp_l), double(pc.La));
  // The LIQUID row must be the rate the run actually uses: under -la0 the
  // property split is off and the liquid relaxes at Dls, so printing Dll there
  // overstated the liquid lattice diffusivity by 59 % in the very line a reader
  // would quote as the stability margin.
  const double Dll_used = o.la0 ? Dls : Dll;
  std::printf("  D_lat  s %.6f  l %.6f    tau  s %.6f  l %.6f    cs2 = %.4f%s\n",
              Dls, Dll_used, 1.0 / double(Coll::omega_from_diffusivity(Real(Dls))),
              1.0 / double(Coll::omega_from_diffusivity(Real(Dll_used))),
              double(cs2<L, Real>()), o.la0 ? "   (-la0: liquid == solid)" : "");
  std::printf("  T_ref = %.4f (an enthalpy)   rate_from_solver = %d   Pe = %.6f\n",
              double(coll.T_ref), int(coll.rate_from_solver()), ref.Pe());
  {
    et::Field rq = ref;
    double Tpk = m.T_0;
    for (double s2 = -6.0; s2 <= 2.0; s2 += 0.02)
      Tpk = std::max(Tpk, rq.T(s2 * sigma, 0.0, 0.0, t_end_for_peak));
    const double T_boil = 3315.0;   // Ti-6Al-4V, approximate; Mills (2002)
    // THIS CRITERION IS THE ANALYTIC REFERENCE'S PEAK, AND THE REFERENCE HAS
    // NEITHER EVAPORATION NOR CONVECTION -- so on its own it CONDEMNS operating
    // points that the fuller model permits. Measured at this point: the
    // reference says 4981 K, the simulation with evaporative cooling reaches
    // 3238 K and with Marangoni as well 2902 K, both BELOW boiling. Read the
    // tier (e) report at the end of the run for the simulated number; this line
    // bounds the CONDUCTION problem and nothing more.
    std::printf("  KEYHOLE CRITERION (conduction reference, no evaporation or "
                "convection):\n    peak surface T = %.0f K against a boiling point "
                "near %.0f K -- %s\n", Tpk, T_boil,
                Tpk < T_boil ? "below, conduction mode"
                             : "ABOVE for the REFERENCE. Tier (a) is unaffected "
                               "(both sides solve the same linear problem). Whether "
                               "tier (c) is invalid depends on the SIMULATED peak -- "
                               "run -evap and read the tier (e) report.");
  }
  std::printf("  q_peak = %.4e W/m^2   flux_to_K = %.4e   peak dH/step = %.2f K (%.2f %% of T_s-T_0)\n",
              q_peak, flux_to_K, q_peak * flux_to_K,
              100.0 * q_peak * flux_to_K / (m.T_s - m.T_0));

  //---- the reference's own identities, before the solver runs ----
  std::printf("\nreference self-checks:\n");
  { et::Field r0 = ref; r0.v = 0.0;
    verdict("R1 static peak / closed form", (r0.T(0,0,0) - m.T_0) /
            (o.A*o.P/(2*m.k_s*sigma*std::sqrt(2*M_PI))), 1.0, 1e-12);
    const double pk = o.A*o.P/(2*m.k_s*sigma*std::sqrt(2*M_PI));
    double w2 = 0;
    for (double rs : {0.5,1.0,2.0,3.0}) {
      const double u = rs*rs/4.0;
      w2 = std::max(w2, std::abs((r0.T(0,rs*sigma,0)-m.T_0)/(pk*std::exp(-u)*et::i0(u)) - 1.0));
    }
    verdict("R2 static surface field, worst of four radii", w2, 0.0, 1e-10);
    double w3 = 0;
    for (double vv : {0.0,0.7,2.0}) {
      et::Field rv = ref; rv.v = vv;
      for (double xs : {0.0,0.5,-0.5,1.0,-1.0}) {
        const double xi = xs*sigma, h = 1e-9;
        const double dTdz = (rv.T(xi,0,h) - rv.T(xi,0,2*h)) / (-h);
        const double q = q_peak*std::exp(-2*xi*xi/(a_beam*a_beam));
        w3 = std::max(w3, std::abs((-m.k_s*dTdz)/q - 1.0));
      }
    }
    verdict("R3 surface flux delivers A*P, worst over v and x", w3, 0.0, 1e-3);
    verdict("R6 lateral symmetry", ref.T(-30e-6,40e-6,0), ref.T(-30e-6,-40e-6,0), 1e-9);
  }

  //---- the analytic pool. TWO of them, and comparing against the wrong one is
  //---- the mistake this block exists to prevent: the quasi-steady pool is the
  //---- t -> infinity limit, and a run of a couple of thermal times
  //---- (sigma^2/alpha = %.4f ms here) has not reached it.
  const double t_end = double(steps) * dt;
  const double t_therm = sigma * sigma / m.alpha_s();
  // THREE pools, and only one of them is the denominator of a verdict:
  //   pa  -- finite time, on the planes the SIMULATION samples. The comparison.
  //   ppz -- finite time, on z = 0. The physical pool a metallograph would cut.
  //   pqs -- the quasi-steady limit, for context only.
  const et::Pool pa   = et::pool(ref, m.T_l, t_end, dx * 0.5);
  const et::Pool ppz  = et::pool(ref, m.T_l, t_end);
  const et::Pool pqs  = et::pool(ref, m.T_l);
  std::printf("\nanalytic pool at the RUN's own time t = %.4f ms = %.2f thermal times:\n",
              t_end * 1e3, t_end / t_therm);
  std::printf("  on the SAMPLED planes (z = dx/2 for width and length, z = 0 datum for depth):\n"
              "    2w = %.3f um   d = %.3f um   L = %.3f um   (xi_w = %.2f, xi_d = %.2f um)\n",
              2*pa.w*1e6, pa.d*1e6, pa.L*1e6, pa.xi_w*1e6, pa.xi_d*1e6);
  std::printf("  on z = 0 (the physical pool, NOT the comparison):\n"
              "    2w = %.3f um   d = %.3f um   L = %.3f um\n",
              2*ppz.w*1e6, ppz.d*1e6, ppz.L*1e6);
  std::printf("  quasi-steady limit: 2w = %.3f um   d = %.3f um   L = %.3f um\n",
              2*pqs.w*1e6, pqs.d*1e6, pqs.L*1e6);
  std::printf("  a centre SLICE of the limit reads 2w = %.2f, d = %.2f um -- "
              "the envelope is not a slice\n",
              2*et::outer_root(ref,0.0,m.T_l,0,sigma)*1e6,
              et::outer_root(ref,0.0,m.T_l,1,sigma)*1e6);
  std::printf("  depth resolved by %.1f cells at this dx -- the depth, not the beam, "
              "is the resolution constraint\n", pqs.d / dx);

  //---- run ----
  const Real Href = coll.T_ref;
  const PhaseChange pcv = coll.material();   // NOT pc -- see the seed check above
  double worst_probe = 0.0; int nprobe = 0;

  // ONE EXTRACTION, USED BY BOTH THE FRAME TRACE AND THE REPORTED NUMBER.
  // Duplicating it would let the animation's last frame disagree with the
  // verdict printed underneath it, which is the kind of gap a reader cannot
  // see. `colmap`, when non-null, also receives the per-column isotherm depth,
  // which is what the 3-D panel draws as the pool's lower surface.
  // WITH RECESSION THE TOP LAYER IS PARTLY VOID, so every scan starts at the
  // COLUMN'S OWN SURFACE rather than at nz-1. Reading nz-1 regardless made the
  // isotherm search run through vacuum and reported L = 41.7 um against a true
  // 150 -- the width and depth were wrong too, less visibly. Depths stay
  // measured from the ORIGINAL surface, which is what a metallograph sees.
  std::vector<Index> hsurf(std::size_t(nx * ny), nz - 1);
  auto surf_of = [&](Index i, Index j) { return hsurf[std::size_t(i * ny + j)]; };
  auto extract = [&](auto& hf, double& w_out, double& d_out, double& L_out,
                     Index& iw_out, Index& id_out, std::vector<float>* colmap) {
    auto Tat = [&](Index i, Index j, Index kk) {
      Real fl, T, E, dE; pcv.invert(hf(d.id(i, j, kk)), fl, T, E, dE); return double(T);
    };
    w_out = d_out = L_out = 0; iw_out = id_out = 0;
    for (Index i = 0; i < nx; ++i) {
      double wv = 0;
      for (Index j = ny / 2; j + 1 < ny; ++j) {
        const double a1 = Tat(i, j, surf_of(i, j)),
                     b1 = Tat(i, j + 1, surf_of(i, j + 1));
        if (a1 >= m.T_l && b1 < m.T_l) {
          wv = (double(j - ny / 2) + (a1 - m.T_l) / (a1 - b1)) * dx; break;
        }
      }
      if (wv > w_out) { w_out = wv; iw_out = i; }
      double dv = 0;
      for (Index kk = surf_of(i, ny / 2); kk > 0; --kk) {
        const double a1 = Tat(i, ny / 2, kk), b1 = Tat(i, ny / 2, kk - 1);
        if (a1 >= m.T_l && b1 < m.T_l) {
          // +0.5: cell nz-1's CENTRE is dx/2 below the free surface (halfway
          // closure), and a depth is measured from the surface. The probe
          // already used this convention; this line did not, and the
          // disagreement read as a 7.7 % model deficit at dx = 4 um.
          dv = (double(nz - 1 - kk) + 0.5 + (a1 - m.T_l) / (a1 - b1)) * dx; break;
        }
      }
      if (dv > d_out) { d_out = dv; id_out = i; }
    }
    // L IS INTERPOLATED AT BOTH ENDS. As a cell COUNT it was quantised to dx --
    // an admissible run could only ever print 152 or 156 um at dx = 4 against
    // an analytic 153.8, a window of one whole cell (+/-1.29 %) -- so its
    // former 0.9 % "agreement" carried no information and could not be
    // laddered. The old sentinel also conflated "unset" with index 0 and
    // reported L = dx for a run with no melt; both went with the cell count.
    Index i1 = -1, i2 = -1;
    for (Index i = 0; i < nx; ++i)
      if (Tat(i, ny / 2, surf_of(i, ny / 2)) >= m.T_l) { if (i1 < 0) i1 = i; i2 = i; }
    if (i1 > 0 && i2 + 1 < nx) {
      const double a0 = Tat(i1 - 1, ny / 2, surf_of(i1 - 1, ny / 2)),
                   b0 = Tat(i1,     ny / 2, surf_of(i1,     ny / 2));
      const double a2 = Tat(i2,     ny / 2, surf_of(i2,     ny / 2)),
                   b2 = Tat(i2 + 1, ny / 2, surf_of(i2 + 1, ny / 2));
      const double xlo = (double(i1 - 1) + (m.T_l - a0) / (b0 - a0)) * dx;
      const double xhi = (double(i2)     + (a2 - m.T_l) / (a2 - b2)) * dx;
      L_out = xhi - xlo;
    }
    if (colmap) {
      colmap->clear();
      for (Index i = 0; i < nx; ++i)
        for (Index j = 0; j < ny; ++j) {
          double dv = 0;
          for (Index kk = surf_of(i, j); kk > 0; --kk) {
            const double a1 = Tat(i, j, kk), b1 = Tat(i, j, kk - 1);
            if (a1 >= m.T_l && b1 < m.T_l) {
              dv = (double(nz - 1 - kk) + 0.5 + (a1 - m.T_l) / (a1 - b1)) * dx; break;
            }
          }
          colmap->push_back(float(dv));
        }
    }
  };

  // ---- animation frames. The panels deliberately differ from
  // ---- tools/render_keyhole.py's, because this case has no receding surface
  // ---- and no keyhole: the surface is FLAT and u == 0 identically. What it has
  // ---- instead, and what the keyhole case does not, is an analytic reference,
  // ---- so the trace panel carries the pool's three dimensions against Eagar &
  // ---- Tsai rather than a depth with nothing to check it.
  const int fev = o.frames.empty() ? 0
                : (o.fevery > 0 ? o.fevery : int(std::max<long>(1, steps / 150)));
  std::vector<float> fr_Ttop, fr_Txz, fr_pool, fr_t, fr_w, fr_d, fr_L, fr_xb;
  // TIER (d) only: the velocity the pool is actually stirred by. Dumped in
  // PHYSICAL units (m/s), because a quiver in lattice units is unreadable and
  // the whole point of the panel is that these are real speeds.
  std::vector<float> fr_usx, fr_usy, fr_ucx, fr_ucz;
  // TIER (f): the surface itself, in micrometres BELOW the original plane, so
  // the render can draw the material leaving rather than infer it.
  std::vector<float> fr_rec, fr_removed;   // um below datum; ng total
  const Index jc = ny / 2;

  // ---- TIER (f): A RECEDING SURFACE. The evaporated mass actually leaves, at
  // ---- v_rec = m_dot / rho_liquid -- no recoil, no melt ejection, no fitted
  // ---- constant, just the Hertz-Knudsen flux tier (e) already computes
  // ---- divided by a density. validation/keyhole.cpp's recession is driven by
  // ---- recoil pressure with thirteen fitted constants; this one has none.
  const Index ncol = nx * ny;
  View1D<Index> Hs("Hs", ncol);
  View1D<Real>  rec("rec", ncol);
  Kokkos::deep_copy(Hs, Index(nz - 1));
  auto flags_v = s.flags();

  const Evap ev;
  const bool do_evap = o.evap;
  const bool do_rec = o.recede;
  const double rho_liq = o.rho_l;
  const Index nyc = ny;
  // THE ENERGY THAT LEAVES. E_evap is the enthalpy the vapour carries off,
  // accumulated from the SAME m_dot the source term subtracts; E_removed is the
  // enthalpy of the cells recession voids, taken at the moment of voiding.
  // Together with the field they must reconstruct A*P*t.
  double E_evap = 0.0, E_removed = 0.0;
  const double ev_Lv = ev.Lv, ev_Tb = ev.Tb, ev_Rs = ev.Rs, ev_P0 = ev.P0;
  const double ev_br = o.betar;
  const PhaseChange pcs = pcv;
  auto field_now = s.temperature();

  // THE EVAPORATION NEEDS A CURRENT SURFACE TEMPERATURE, AND NOTHING ELSE WAS
  // COMPUTING ONE. compute_field() is called inside the loop only when -probe
  // or -flow is on; without them the field the source lambda reads is whatever
  // the last call left, so the evaporation was evaluated against a stale -- at
  // the start, ambient -- temperature and lost almost all of its magnitude. The
  // first -evap run read -0.35 % on width where it should have been large.
  // Found by noticing that -evap alone moved the pool far less than -flow -evap
  // did, which is the wrong way round: evaporation is a bigger term than
  // Marangoni at this surface temperature.
  double worst_evap_frac = 0.0;
  for (long it = 0; it < steps; ++it) {
    if (do_evap) {
      s.compute_field();
      // Same field, same m_dot, same surface as the source term below, so this
      // accounts for exactly what the source removed rather than for a second
      // estimate of it.
      auto Th_e = s.temperature();
      auto Hs_e = Hs;
      const PhaseChange pce = pcv;
      const double eLv2 = ev.Lv, eTb2 = ev.Tb, eRs2 = ev.Rs, eP02 = ev.P0;
      const double ebr2 = o.betar, dx2 = dx, dt2 = dt;
      const Index nyE = ny, nzE = nz;
      const bool recE = do_rec;
      double eStep = 0.0;
      Kokkos::parallel_reduce("melt_pool_evap_energy", ncol,
        KOKKOS_LAMBDA(Index c, double& acc) {
          const Index i = c / nyE, j = c % nyE;
          const Index h = recE ? Hs_e(c) : nzE - 1;
          Real fl, T, E, dE; pce.invert(Th_e(d.id(i, j, h)), fl, T, E, dE);
          const double Ts = Kokkos::fmax(double(T), 1.0);
          const double Pv = eP02 * Kokkos::exp(Kokkos::fmin(
              (eLv2 / eRs2) * (1.0 / eTb2 - 1.0 / Ts), 60.0));
          const double md = (1.0 - ebr2) * Pv *
                            Kokkos::sqrt(1.0 / (2.0 * M_PI * eRs2 * Ts));
          acc += md * eLv2 * dx2 * dx2 * dt2;
        }, eStep);
      E_evap += eStep;
    }
    const double xb = x0 + o.v * (double(it) + 0.5) * dt;   // MIDPOINT of this step
    const double cut = 3.0 * a_beam;
    s.add_source(KOKKOS_LAMBDA(Index n) -> Real {
      Index px, py, pz; d.coords(n, px, py, pz);
      if (!d.is_interior(px, py, pz)) return Real(0);
      const Index i = px - d.hx, j = py - d.hy, kk = pz - d.hz;
      // the CURRENT surface of this column, which recedes under -recede and is
      // the fixed top layer otherwise.
      if (kk != (do_rec ? Hs(i * nyc + j) : nz - 1)) return Real(0);
      const double xm = (double(i) + 0.5) * dx - xb;
      const double ym = (double(j) + 0.5) * dx - yc;
      const double r2 = xm * xm + ym * ym;
      const double q_in = (r2 > cut * cut)
                        ? 0.0
                        : q_peak * Kokkos::exp(-2.0 * r2 / (a_beam * a_beam));
      if (!do_evap) return Real(q_in * flux_to_K);

      // ---- TIER (e): EVAPORATIVE COOLING ----------------------------------
      // Hertz-Knudsen with a Clausius-Clapeyron vapour pressure:
      //     P_v(T) = P_0 exp[ (L_v/R_s) (1/T_b - 1/T) ]
      //     m_dot  = (1 - beta_r) P_v sqrt( 1 / (2 pi R_s T) )
      //     q_evap = m_dot L_v                                   [W/m^2]
      // It is a SINK on the same surface the beam heats, so it enters the same
      // source term rather than as a separate pass -- that matters, because
      // above the boiling point q_evap is comparable to the beam and applying
      // the two at different points in the step lets the surface overshoot.
      Real flv, Tlv, Elv, dElv;
      pcs.invert(field_now(n), flv, Tlv, Elv, dElv);
      const double Tsurf = Kokkos::fmax(double(Tlv), 1.0);
      const double expo = Kokkos::fmin((ev_Lv / ev_Rs) * (1.0 / ev_Tb - 1.0 / Tsurf),
                                       60.0);
      const double Pv = ev_P0 * Kokkos::exp(expo);
      const double mdot = (1.0 - ev_br) * Pv *
                          Kokkos::sqrt(1.0 / (2.0 * M_PI * ev_Rs * Tsurf));
      // NOT CLAMPED, deliberately. validation/keyhole.cpp clamps this feedback
      // and its own banner records that the clamp became the integrator. Here
      // the balance is reported instead: if q_evap exceeds the beam the case
      // says so, and that is a statement about the operating point rather than
      // a number to be suppressed.
      return Real((q_in - mdot * ev_Lv) * flux_to_K);
    });
    if (o.flow) {
      // ---- the force field, rebuilt each step, ON THE DEVICE ---------------
      // Marangoni: tau = (dgamma/dT) grad_s T on the free surface, applied as a
      // BODY FORCE tau/dx in the top cell. validation/marangoni.cpp validates
      // exactly this device against the exact profile and measures what it
      // costs: the surface velocity is FIRST order, the interior second, so a
      // pool dimension read off an isotherm away from the surface is unharmed.
      //
      // Mushy sink: F = -A u with A = A_lat eps (1-f_l)^2/(f_l^3+eps) in
      // lattice units. validation/mushy_sink.cpp measures the stability bound
      // at A = 1.000 (a two-step recurrence, NOT the textbook A < 2) and the
      // residual leakage floor at about 2 % of u_max. THE eps IS NOT
      // DECORATION: A(0) = C/eps, so naming C "the strength at f_l = 0" makes
      // it 1/eps too large -- 800 against a bound of 1, and the run went NaN.
      //
      // ONE KERNEL, NO MIRRORS. This was a host loop with six device-host
      // copies per step, which was tolerable at dx = 4 um and not at dx = 2.
      // PhaseChange::invert is KOKKOS_INLINE_FUNCTION, so the phase-change
      // inverse runs on the device as happily as on the host; the host version
      // existed only because it was easier to get right first.
      s.compute_field();
      auto Th = s.temperature();
      auto ux = fs->ux(); auto uy = fs->uy(); auto uz = fs->uz();
      auto fxv = Fx, fyv = Fy, fzv = Fz;
      const PhaseChange pcd = pcv;
      const double eps = 1e-3, dgdT = o.dgdT, Alat = o.A_lat;
      const double dxl = dx, Fscale = F_to_lat;
      const Index nxl = nx, nyl = ny, nzl = nz;
      Kokkos::parallel_for("melt_pool_force", d.n_padded, KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) return;
        const Index i = px - d.hx, j = py - d.hy, kk = pz - d.hz;
        auto Tof = [&](Index a, Index b, Index c) {
          Real fl, T, E, dE; pcd.invert(Th(d.id(a, b, c)), fl, T, E, dE);
          return double(T);
        };
        Real flr, Tr, Er, dEr;
        pcd.invert(Th(n), flr, Tr, Er, dEr);
        const double fl = double(flr);
        double fx = 0, fy = 0, fz = 0;
        if (kk == nzl - 1 && fl > 0.0) {
          // central differences on the surface plane; one-sided at the rim.
          // ONLY WHERE THERE IS LIQUID: a surface tension gradient on solid is
          // not a stress, and applying it there would drive the solid.
          const Index ip = (i + 1 < nxl) ? i + 1 : i, im = (i > 0) ? i - 1 : i;
          const Index jp = (j + 1 < nyl) ? j + 1 : j, jm = (j > 0) ? j - 1 : j;
          const double dTdx = (Tof(ip, j, kk) - Tof(im, j, kk)) /
                              (double(ip - im) * dxl);
          const double dTdy = (Tof(i, jp, kk) - Tof(i, jm, kk)) /
                              (double(jp - jm) * dxl);
          fx = fl * dgdT * dTdx / dxl * Fscale;
          fy = fl * dgdT * dTdy / dxl * Fscale;
        }
        const double A = Alat * eps * (1.0 - fl) * (1.0 - fl) /
                         (fl * fl * fl + eps);
        fx -= A * double(ux(n));
        fy -= A * double(uy(n));
        fz -= A * double(uz(n));
        fxv(n) = Real(fx); fyv(n) = Real(fy); fzv(n) = Real(fz);
      });
      fs->step();
      fs->compute_macroscopic();
    }

    s.step();

    if (do_rec) {
      // v_rec = m_dot / rho_liquid, integrated explicitly. The increment is NOT
      // capped: validation/keyhole.cpp caps this and its banner records that the
      // cap became the integrator, so here an excursion is reported at the end
      // instead of being quietly bounded.
      s.compute_field();
      auto Th2 = s.temperature();
      auto Hs2 = Hs; auto rec2 = rec;
      const PhaseChange pcr = pcv;
      const double eLv = ev.Lv, eTb = ev.Tb, eRs = ev.Rs, eP0 = ev.P0, ebr = o.betar;
      const double dtl = dt, dxl2 = dx, rl = rho_liq;
      const Index nzc2 = nz, nyc2 = ny;
      Kokkos::parallel_for("melt_pool_recede", ncol, KOKKOS_LAMBDA(Index c) {
        const Index i = c / nyc2, j = c % nyc2;
        const Index h = Hs2(c);
        Real fl, T, E, dE; pcr.invert(Th2(d.id(i, j, h)), fl, T, E, dE);
        const double Ts = Kokkos::fmax(double(T), 1.0);
        const double Pv = eP0 * Kokkos::exp(Kokkos::fmin(
            (eLv / eRs) * (1.0 / eTb - 1.0 / Ts), 60.0));
        const double md = (1.0 - ebr) * Pv *
                          Kokkos::sqrt(1.0 / (2.0 * M_PI * eRs * Ts));
        // ONLY LIQUID LEAVES. Evaporating solid would be sublimation, which is
        // not this process and not modelled.
        rec2(c) += Real(double(fl) * (md / rl) * dtl / dxl2);
        Index nh = Index(Kokkos::round(double(nzc2 - 1) - double(rec2(c))));
        Hs2(c) = Kokkos::min(Kokkos::max(nh, Index(0)), Index(nzc2 - 1));
      });
      // what is above the surface is gone: excluded, and its enthalpy with it
      auto fv = flags_v; auto Hs3 = Hs; auto Th3 = s.temperature();
      const Index nyc3 = ny;
      // A CELL THAT LEAVES TAKES ITS ENTHALPY WITH IT, and that has to be
      // counted at the moment it goes -- afterwards it is unrecoverable. It is
      // also written to AMBIENT rather than to zero: H = 0 is not ambient in
      // this gauge (ambient is Href = T_0 - T_s), so zeroing made every voided
      // cell read as +1585 K of enthalpy to anything that summed the field.
      const Real Href_v = Href;
      const double rcs = m.rc_s(), dxv = dx;
      double eGone = 0.0;
      Kokkos::parallel_reduce("melt_pool_void", d.n_padded,
        KOKKOS_LAMBDA(Index n, double& acc) {
          Index px, py, pz; d.coords(n, px, py, pz);
          if (!d.is_interior(px, py, pz)) return;
          const Index i = px - d.hx, j = py - d.hy, kk = pz - d.hz;
          const bool out = kk > Hs3(i * nyc3 + j);
          const bool was_bulk = fv(n) != std::uint8_t(ScalarExcluded);
          fv(n) = out ? std::uint8_t(ScalarExcluded) : std::uint8_t(ScalarBulk);
          if (out) {
            if (was_bulk)
              acc += (double(Th3(n)) - double(Href_v)) * rcs * dxv * dxv * dxv;
            Th3(n) = Href_v;
          }
        }, eGone);
      E_removed += eGone;
    }

    if (o.probe && (it + 1) % o.probe == 0) {
      s.compute_field();
      auto h = Kokkos::create_mirror_view(s.temperature());
      Kokkos::deep_copy(h, s.temperature());
      const double t = double(it + 1) * dt;
      const double xbn = x0 + o.v * t;
      double num = 0, den = 0;
      for (Index i = 0; i < nx; ++i)
        // A FIXED PHYSICAL DEPTH, not a fixed number of cells: `nz - 12`
        // samples 48 um at dx = 4 and 24 um at dx = 2, so the L2 it reports is
        // not the same quantity at two rungs and cannot be laddered.
        for (Index kk = nz - 1; kk >= nz - Index(std::llround(48e-6 / dx)) && kk >= 0; --kk) {
          const double xi = (double(i) + 0.5) * dx - xbn;
          if (std::abs(xi) > 6 * sigma) continue;
          const double z = (double(nz - 1 - kk) + 0.5) * dx;
          Real fl, T, E, dE; pcv.invert(h(d.id(i, ny / 2, kk)), fl, T, E, dE);
          const double Ta = ref.T(xi, 0.0, z, t);
          num += (double(T) - Ta) * (double(T) - Ta);
          den += (Ta - m.T_0) * (Ta - m.T_0);
        }
      if (den > 0) { worst_probe = std::max(worst_probe, std::sqrt(num / den)); ++nprobe; }
    }

    if (fev && (it + 1) % fev == 0) {
      s.compute_field();
      auto hf = Kokkos::create_mirror_view(s.temperature());
      Kokkos::deep_copy(hf, s.temperature());
      auto Tof = [&](Index i, Index j, Index kk) {
        Real fl, T, E, dE; pcv.invert(hf(d.id(i, j, kk)), fl, T, E, dE); return float(T);
      };
      // THE COLUMN'S OWN SURFACE, not nz-1. Reading the fixed top layer under
      // recession samples VOIDED cells, which invert to the solidus and paint a
      // uniform false trail behind the pool.
      for (Index i = 0; i < nx; ++i)
        for (Index j = 0; j < ny; ++j)
          fr_Ttop.push_back(Tof(i, j, o.recede ? hsurf[std::size_t(i * ny + j)]
                                               : nz - 1));
      for (Index i = 0; i < nx; ++i)
        for (Index kk = 0; kk < nz; ++kk) fr_Txz.push_back(Tof(i, jc, kk));
      if (o.recede) {
        auto hh = Kokkos::create_mirror_view_and_copy(HostSpace{}, Hs);
        for (Index c = 0; c < nx * ny; ++c) hsurf[std::size_t(c)] = hh(c);
      }
      std::vector<float> colmap;
      double wf, df, Lf; Index a, b;
      extract(hf, wf, df, Lf, a, b, &colmap);
      fr_pool.insert(fr_pool.end(), colmap.begin(), colmap.end());
      fr_t.push_back(float(double(it + 1) * dt));
      fr_w.push_back(float(2.0 * wf)); fr_d.push_back(float(df)); fr_L.push_back(float(Lf));
      fr_xb.push_back(float(x0 + o.v * double(it + 1) * dt));
      if (o.recede) {
        auto hrc = Kokkos::create_mirror_view_and_copy(HostSpace{}, rec);
        double vol = 0;
        for (Index i = 0; i < nx; ++i)
          for (Index j = 0; j < ny; ++j) {
            const double r = double(hrc(i * ny + j));
            fr_rec.push_back(float(r * dx * 1e6));
            vol += r * dx * dx * dx;
          }
        fr_removed.push_back(float(vol * o.rho_l * 1e12));   // ng
      }
      if (o.flow) {
        fs->compute_macroscopic();
        auto qx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->ux());
        auto qy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->uy());
        auto qz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->uz());
        const double to_ms = dx / dt;
        for (Index i = 0; i < nx; ++i)
          for (Index j = 0; j < ny; ++j) {
            const Index n = d.id(i, j, nz - 1);
            fr_usx.push_back(float(double(qx(n)) * to_ms));
            fr_usy.push_back(float(double(qy(n)) * to_ms));
          }
        for (Index i = 0; i < nx; ++i)
          for (Index kk = 0; kk < nz; ++kk) {
            const Index n = d.id(i, jc, kk);
            fr_ucx.push_back(float(double(qx(n)) * to_ms));
            fr_ucz.push_back(float(double(qz(n)) * to_ms));
          }
      }
    }
  }
  s.compute_field();

  //---- ENERGY BALANCE. The dimensions converge at FIRST order (measured), and
  //---- there are only two candidates: the source amplitude is wrong, or its
  //---- amplitude is right and its spatial distribution is smeared over the top
  //---- cell instead of sitting on z = 0. This separates them, and it needs no
  //---- reference: sum the enthalpy the field actually holds against A*P*t.
  //---- In this gauge H is in kelvin and the volumetric heat capacity is
  //---- (rho c)_s exactly, so the energy in a cell is (H - H_amb) (rho c)_s dx^3.
  {
    auto Th = s.temperature();
    auto fl_e = s.flags();
    double sumH = 0.0;
    Kokkos::parallel_reduce("melt_pool_energy", Th.extent(0),
      KOKKOS_LAMBDA(const Index n, double& acc) {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) return;
        if (fl_e(n) == std::uint8_t(ScalarExcluded)) return;   // it is gone
        acc += double(Th(n)) - double(Href);
      }, sumH);
    const double E_field = sumH * m.rc_s() * dx * dx * dx;
    const double E_in    = o.A * o.P * t_end;
    std::printf("\nenergy balance (no reference needed, so it separates an amplitude "
                "error from a shape error):\n");
    std::printf("  deposited A*P*t = %.6e J   field holds %.6e J   ratio %.6f\n",
                E_in, E_field, E_field / E_in);
    // THE BALANCE ONLY CLOSES WITHOUT EVAPORATION, and asserting it regardless
    // turned a correct run into a FAIL: at 115 W with -recede the field holds
    // 0.9023 of A*P*t, and the missing 9.8 % is exactly what evaporation
    // carried off (plus, under -recede, the enthalpy of the cells that left).
    // The check is a CLOSED-SYSTEM statement and tiers (e) and (f) open the
    // system, so it is reported rather than asserted there.
    //
    // The guard, where it does apply, is deliberately LOOSE and the reason is
    // stated rather than hidden in a number: the analytic sum over cells of
    // q_peak exp(-2r^2/a^2) dx^2 dt is EXACTLY A*P dt, so a gross miss would be
    // an amplitude or footprint error, and that is what 5 % catches. The
    // residual is 1.19 % at dx = 4 um and 0.61 % at dx = 2 um -- ratio 1.95, so
    // FIRST ORDER and vanishing, not a leak. Measured 2026-09-21.
    //
    // THIS IS NOW DONE. The evaporated energy is accumulated from the SAME
    // m_dot the source subtracts, and the enthalpy of each voided cell is taken
    // at the moment it goes, so the three terms must reconstruct A*P*t.
    // Measured 2026-09-22, 115 W, -recede:
    //
    //     dx      field   evaporated   removed    closure
    //     4 um    0.892     0.071       0.027     0.990105
    //     2 um      --        --          --      0.994942
    //
    // and -evap alone at dx = 4 closes to 0.990068 -- the same value by a
    // different split (0.909 + 0.081 + 0), which is the check working.
    // The residual is FIRST ORDER and vanishing: 0.9895 % -> 0.5058 %, ratio
    // 1.96, the same rate as the closed-system residual (1.19 % -> 0.61 %,
    // ratio 1.95). So the evaporation bookkeeping introduces NO error of its
    // own; what is left is the surface flux smeared over the top cell, which
    // this file already measures elsewhere.
    if (o.evap) {
      // THE BALANCE IS CLOSED RATHER THAN EXCUSED. Everything the beam put in
      // is either still in the field, carried off as vapour, or left with a
      // cell recession voided. If those three do not reconstruct A*P*t the
      // evaporation bookkeeping is wrong, and no other check in this file
      // would notice.
      const double E_tot = E_field + E_evap + E_removed;
      std::printf("    + evaporated %.6e J (%.1f %%)   + removed with the voided "
                  "cells %.6e J (%.1f %%)\n", E_evap, 100.0 * E_evap / E_in,
                  E_removed, 100.0 * E_removed / E_in);
      std::printf("    field + evaporated + removed = %.6e J\n", E_tot);
      verdict("(field + evaporated + removed) / A*P*t",
              E_tot / E_in, 1.0, 0.05);
    } else {
      verdict("energy in field / A*P*t (1st order, see comment)",
              E_field / E_in, 1.0, 0.05);
    }
  }

  if (o.flow) {
    fs->compute_macroscopic();
    auto hux = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->ux());
    auto huy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->uy());
    auto huz = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->uz());
    double um = 0, usurf = 0;
    for (Index i = 0; i < nx; ++i)
      for (Index j = 0; j < ny; ++j)
        for (Index kk = 0; kk < nz; ++kk) {
          const Index n = d.id(i, j, kk);
          const double q = std::sqrt(double(hux(n)) * double(hux(n)) +
                                     double(huy(n)) * double(huy(n)) +
                                     double(huz(n)) * double(huz(n)));
          um = std::max(um, q);
          if (kk == nz - 1) usurf = std::max(usurf, q);
        }
    double rmin = 1e30, rmax = 0;
    auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs->rho());
    for (Index i = 0; i < nx; ++i)
      for (Index j = 0; j < ny; ++j)
        for (Index kk = 0; kk < nz; ++kk) {
          const double r = double(hr(d.id(i, j, kk)));
          rmin = std::min(rmin, r); rmax = std::max(rmax, r);
        }
    const double Ma = um / std::sqrt(double(cs2<LF, Real>()));
    std::printf("\n  TIER (d) flow report:\n");
    std::printf("    peak |u| = %.4e lattice = %.4f m/s   surface peak = %.4e\n",
                um, um * dx / dt, usurf);
    std::printf("    Mach = %.4f   rho in [%.4f, %.4f] (%.1f %% excursion)\n",
                Ma, rmin, rmax, 100.0 * (rmax - rmin));
    if (Ma > 0.1)
      std::printf("    WARNING: Ma > 0.1. The compressibility error grows as Ma^2 and\n"
                  "    the density excursion above is its symptom. This tier is a\n"
                  "    MEASURED INCREMENT, and at this Mach number it is a coarse one:\n"
                  "    dt is set by the THERMAL problem and is too long for the flow.\n");
    if (um * dx / dt <= 1e-3)
      std::printf("    THE FLOW IS NOT MOVING -- check that the surface force is not\n"
                  "    being written into a non-colliding node.\n");
  }

  // Reported by tier (e) and carried into the frame meta, so the figure can
  // state a number belonging to THIS run rather than one transcribed from
  // another. The render used to hardcode "2 % of the depth, 1400 K of surface
  // T" from a 115 W run, which was wrong at every other operating point.
  double Tmax_surf = 0, qfrac = 0;
  if (o.recede) {
    auto hh2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, Hs);
    for (Index c = 0; c < nx * ny; ++c) hsurf[std::size_t(c)] = hh2(c);
    auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, rec);
    double rmax = 0, vol = 0; long ncols = 0;
    for (Index c = 0; c < ncol; ++c) {
      const double r = double(hr(c));
      rmax = std::max(rmax, r);
      vol += r * dx * dx * dx;
      if (r > 0.5) ++ncols;
    }
    std::printf("\n  TIER (f) recession report:\n");
    std::printf("    deepest column receded %.3f um = %.2f cells\n",
                rmax * dx * 1e6, rmax);
    std::printf("    material removed %.4e um^3 = %.3f ng   (%ld of %ld columns "
                "lost >= half a cell)\n", vol * 1e18,
                vol * o.rho_l * 1e12, ncols, long(ncol));
    if (rmax < 1.0)
      std::printf("    LESS THAN ONE CELL. At this operating point the removal is\n"
                  "    sub-grid and the recession is not resolved -- raise the power\n"
                  "    until the beam outruns the evaporative thermostat, or refine.\n");
  }
  if (o.evap) {
    // The balance, reported and not clamped. validation/keyhole.cpp clamps this
    // same feedback and its banner records that the clamp became the
    // integrator; here the ratio is printed, because q_evap > q_in is a
    // statement about the OPERATING POINT and not a number to suppress.
    s.compute_field();
    auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    const Evap ev2;
    Tmax_surf = 0; qfrac = 0;
    double& Tmax = Tmax_surf; double& fr = qfrac;
    for (Index i = 0; i < nx; ++i)
      for (Index j = 0; j < ny; ++j) {
        // THE COLUMN'S OWN SURFACE. This read nz-1 and was the THIRD place in
        // this file with that defect -- after the pool extraction and the frame
        // dump -- so under recession it sampled voided cells and under-reported
        // the peak surface temperature by 758 K (4050 against 4808 at 1200 W).
        // The pattern: anything that means "the surface" must ask the column,
        // because tier (f) moved it.
        Real fl, T, E, dE;
        pcv.invert(he(d.id(i, j, o.recede ? hsurf[std::size_t(i * ny + j)] : nz - 1)),
                   fl, T, E, dE);
        const double Ts = std::max(double(T), 1.0);
        Tmax = std::max(Tmax, Ts);
        const double Pv = ev2.P0 * std::exp(std::min((ev2.Lv / ev2.Rs) *
                          (1.0 / ev2.Tb - 1.0 / Ts), 60.0));
        const double md = (1.0 - o.betar) * Pv *
                          std::sqrt(1.0 / (2.0 * M_PI * ev2.Rs * Ts));
        fr = std::max(fr, md * ev2.Lv / q_peak);
      }
    std::printf("\n  TIER (e) evaporation report:\n");
    std::printf("    simulated peak surface T = %.0f K   (boiling %.0f K)\n",
                Tmax, ev2.Tb);
    std::printf("    worst q_evap / q_peak = %.4f   beta_r = %.2f\n", fr, o.betar);
    if (fr > 1.0)
      std::printf("    q_evap EXCEEDS the beam: at this operating point the surface\n"
                  "    cannot be in a quasi-steady balance without mass loss, which\n"
                  "    this tier does NOT model (the surface is flat and does not\n"
                  "    recede). Treat the pool below as a lower bound on the error.\n");
  }

  //---- extract the pool from the simulated field, as an envelope ----
  auto h = Kokkos::create_mirror_view(s.temperature());
  Kokkos::deep_copy(h, s.temperature());
  if (o.recede) {
    auto hh = Kokkos::create_mirror_view_and_copy(HostSpace{}, Hs);
    for (Index c = 0; c < nx * ny; ++c) hsurf[std::size_t(c)] = hh(c);
  }
  double w_sim = 0, d_sim = 0, L_sim = 0; Index iw = 0, id_ = 0;
  extract(h, w_sim, d_sim, L_sim, iw, id_, nullptr);

  std::printf("\nsimulated pool (envelope over x, sub-cell on the isotherm):\n");
  std::printf("  2w = %.3f um   d = %.3f um   L = %.3f um\n", 2*w_sim*1e6, d_sim*1e6, L_sim*1e6);
  std::printf("  widest at i = %d, deepest at i = %d  (they differ: the envelope is real)\n",
              int(iw), int(id_));
  if (nprobe)
    std::printf("  worst relative L2 of T against the FINITE-TIME analytic field, "
                "%d probes: %.4f %%\n", nprobe, 100.0 * worst_probe);

  std::printf("\nacceptance:\n");
  if (o.la0) {
    // THE TOLERANCE SCALES WITH dx, BECAUSE THE ERROR DOES. The old +/-10 %
    // and +/-15 % were fixed numbers "derived from the cell size, not
    // measured", and a fixed tolerance on a FIRST-ORDER quantity asserts the
    // mesh rather than the scheme: it passes trivially when the grid is fine
    // and it would have passed the 6 % systematic bias this case carried before
    // the datum errors were found.
    //
    // Measured tier (a) deviations, 2026-09-22:
    //     dx = 4 um   2w 1.33 %   d 1.01 %
    //     dx = 2 um   2w 0.66 %   d 0.51 %
    // i.e. 0.33 %/um and 0.25 %/um, first order (ratios 2.02 and 1.98). The
    // tolerances are those slopes x 1.5, so the margin is the same at every
    // rung instead of growing as the grid refines.
    //
    // THE FLOOR IS NOT COSMETIC. Below about dx = 0.6 um the dx-scaled bound
    // would fall under the other errors this case carries -- the analytic
    // quadrature at 1e-11, the sub-cell isotherm interpolation, the envelope's
    // discrete search over i -- and the case would start failing on those
    // rather than on the scheme. 0.3 % is that floor.
    //
    // FITTED ON dx = 4 AND 2, AND CHECKED AT dx = 8, WHERE IT PART-HOLDS:
    //
    //     dx     2w dev   d dev    tol      margin (2w, d)
    //     8 um   2.89 %   1.53 %   4.0/3.2   1.38, 2.10
    //     4 um   1.33 %   1.02 %   2.0/1.6   1.50, 1.57
    //     2 um   0.67 %   0.51 %   1.0/0.8   1.50, 1.57
    //
    // The WIDTH is first order across all three (ratios 2.17 and 2.00). The
    // DEPTH is not, above dx = 4: it grows only 1.50x from 4 to 8 um, because
    // the pool is 3.3 cells deep there and outside the asymptotic range. So the
    // 2w margin narrows to 1.38 at dx = 8 rather than holding at 1.5, and the
    // bound is calibrated for dx <= 4 um. Coarser than that it still passes,
    // with less room than designed.
    //
    // WHAT IT WOULD HAVE CAUGHT: the 6.4 % width and 5.9 % depth bias this case
    // carried before the datum errors were found fails at 2.0 % and 1.6 %. The
    // old +/-10 % and +/-15 % passed it, which is how it survived.
    const double tol_w = std::max(0.005 * (dx * 1e6), 0.003);
    const double tol_d = std::max(0.004 * (dx * 1e6), 0.003);
    verdict("2w / analytic", 2*w_sim / (2*pa.w), 1.0, tol_w);
    verdict("d  / analytic", d_sim / pa.d, 1.0, tol_d);
    std::printf("  tolerances are dx-scaled: %.2f %% and %.2f %% at dx = %.2f um\n"
                "  (measured slope x 1.5, floor 0.3 %%). A FIXED tolerance here "
                "would assert the mesh,\n  not the scheme -- it passes trivially "
                "as dx falls.\n",
                100.0 * tol_w, 100.0 * tol_d, dx * 1e6);
  } else {
    std::printf("  TIER (b): no analytic reference contains latent heat. The numbers above\n"
                "  are an INCREMENT against the -la0 row and are reported, not asserted.\n");
  }
  if (fev && !fr_t.empty()) {
    const std::size_t F = fr_t.size();
    const std::string D = o.frames + "/";
    write_npy(D + "T_top.npy", fr_Ttop, {F, std::size_t(nx), std::size_t(ny)});
    write_npy(D + "T_xz.npy",  fr_Txz,  {F, std::size_t(nx), std::size_t(nz)});
    write_npy(D + "pool.npy",  fr_pool, {F, std::size_t(nx), std::size_t(ny)});
    write_npy(D + "t.npy",     fr_t,    {F});
    write_npy(D + "w2.npy",    fr_w,    {F});
    write_npy(D + "d.npy",     fr_d,    {F});
    write_npy(D + "L.npy",     fr_L,    {F});
    write_npy(D + "xb.npy",    fr_xb,   {F});
    // The analytic pool goes in the meta so the trace panel can draw what the
    // simulation is being CHECKED AGAINST rather than only what it did. These
    // are the sampled-plane values -- the same denominators the verdict uses.
    const std::vector<float> meta{
        float(dx), float(dt), float(nx), float(ny), float(nz), float(jc),
        float(m.T_s), float(m.T_l), float(m.T_0), float(o.P), float(o.v),
        float(o.spot), float(o.A), float(2 * pa.w), float(pa.d), float(pa.L),
        float(t_therm), float(o.la0 ? 1 : 0), float(steps),
        float(o.flow ? 1 : 0), float(o.evap ? 1 : 0), float(o.dgdT),
        float(o.recede ? 1 : 0), float(o.rho_l),
        float(Tmax_surf), float(qfrac), float(o.betar)};
    write_npy(D + "meta.npy", meta, {meta.size()});
    if (o.recede && !fr_rec.empty()) {
      write_npy(D + "recess.npy",  fr_rec,     {F, std::size_t(nx), std::size_t(ny)});
      write_npy(D + "removed.npy", fr_removed, {F});
    }
    if (o.flow && !fr_usx.empty()) {
      write_npy(D + "us_x.npy", fr_usx, {F, std::size_t(nx), std::size_t(ny)});
      write_npy(D + "us_y.npy", fr_usy, {F, std::size_t(nx), std::size_t(ny)});
      write_npy(D + "uc_x.npy", fr_ucx, {F, std::size_t(nx), std::size_t(nz)});
      write_npy(D + "uc_z.npy", fr_ucz, {F, std::size_t(nx), std::size_t(nz)});
      std::printf("    + velocity (m/s): us_x, us_y on the surface; uc_x, uc_z "
                  "on the centreline\n");
    }
    std::printf("\n  frames -> %s (%zu frames, %dx%d top, %dx%d centreline, "
                "every %d steps)\n", o.frames.c_str(), F, int(nx), int(ny),
                int(nx), int(nz), fev);
    std::printf("    meta.npy = [dx, dt, nx, ny, nz, y_centre, T_s, T_l, T_0, P, v, "
                "spot_um, A, 2w_an, d_an, L_an, t_therm, la0, steps, flow, evap, dgdT, "
                "recede, rho_l, T_surf_max, q_evap_frac, beta_r]\n");
  }

  std::printf("\n[melt_pool] %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  Opts o; Mat m;
  for (int i = 1; i < argc; ++i) {
    auto nx = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-P"))     nx(o.P);
    else if (!std::strcmp(argv[i], "-v"))     { double t; nx(t); o.v = t * 1e-3; }
    else if (!std::strcmp(argv[i], "-spot"))  nx(o.spot);
    else if (!std::strcmp(argv[i], "-A"))     nx(o.A);
    else if (!std::strcmp(argv[i], "-dx"))    { double t; nx(t); o.dx = t * 1e-6; }
    else if (!std::strcmp(argv[i], "-Lx"))    { double t; nx(t); o.Lx = t * 1e-6; }
    else if (!std::strcmp(argv[i], "-Ly"))    { double t; nx(t); o.Ly = t * 1e-6; }
    else if (!std::strcmp(argv[i], "-Lz"))    { double t; nx(t); o.Lz = t * 1e-6; }
    else if (!std::strcmp(argv[i], "-track")) { double t; nx(t); o.track = t * 1e-6; }
    else if (!std::strcmp(argv[i], "-probe")) { double t; nx(t); o.probe = int(t); }
    else if (!std::strcmp(argv[i], "-flow"))  o.flow = true;
    else if (!std::strcmp(argv[i], "-evap"))  o.evap = true;
    else if (!std::strcmp(argv[i], "-recede")) { o.evap = true; o.recede = true; }
    else if (!std::strcmp(argv[i], "-betar")) nx(o.betar);
    else if (!std::strcmp(argv[i], "-dgdt"))  nx(o.dgdT);
    else if (!std::strcmp(argv[i], "-mu"))    nx(o.mu_l);
    else if (!std::strcmp(argv[i], "-asink")) nx(o.A_lat);
    else if (!std::strcmp(argv[i], "-frames")) { if (i + 1 < argc) o.frames = argv[++i]; }
    else if (!std::strcmp(argv[i], "-fevery")) { double t; nx(t); o.fevery = int(t); }
    else if (!std::strcmp(argv[i], "-la0"))   o.la0 = true;
    else if (!std::strcmp(argv[i], "-bgk"))   o.bgk = true;
    else if (!std::strncmp(argv[i], "--kokkos", 8)) {}
    else std::fprintf(stderr, "melt_pool: unknown option %s\n", argv[i]);
  }
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("conduction-mode melt pool vs Eagar & Tsai (1983)\n");
    std::printf("backend %s   precision %s\n",
                Kokkos::DefaultExecutionSpace::name(), precision_name());
    if (o.bgk) rc = run<D3Q7, EnthalpyBGK<D3Q7>>(o, m);
    else       rc = run<D3Q7, EnthalpyRegularised<D3Q7>>(o, m);
  }
  Kokkos::finalize();
  return rc;
}
