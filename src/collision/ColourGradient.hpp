#pragma once
//==============================================================================
//  Colour-gradient two-component flow, with nonorthogonal central moments.
//
//  De Rosis, Huang & Coreixas, "Universal formulation of central-moments-based
//  lattice Boltzmann method with external forcing for the simulation of
//  multiphysics phenomena", Phys. Fluids 31, 117102 (2019), APPENDIX D.
//  Equation numbers of the form (Dn) are that appendix's, and it is what this
//  file implements as of 2026-09-19. D3Q27 only.
//
//  IT USED TO IMPLEMENT Saito, De Rosis, Festuccia, Kaneko, Abe & Koyama,
//  Phys. Rev. E 98, 013305 (2018), whose equation numbers survive in a few
//  places below where the two papers agree and the older one states it better.
//  Three things changed, and only the first is a change of physics:
//
//    * THE EQUILIBRIUM. Eq. (D5) is the COMPLETE sixth-order Hermite set where
//      Saito's Eq. (18) truncates at third order in u. Measured, it changes
//      almost nothing in a static droplet -- see the ladder under RestTerm
//      below, where the two agree to four or five significant figures at every
//      density ratio -- because the difference is exactly the truncation
//      residuals, -rho ux^2 uy^2 at order 4 and -10 rho ux^2 uy^2 uz^2 at order
//      6, and those are ~1e-20 at the spurious velocities a droplet reaches.
//      What it buys is not accuracy there but the collapse of the closed form:
//      see the closed-form banner.
//    * THE REST TERM's alpha. Eq. (D6) interpolates ONE alpha linearly, Eq.
//      (D13). That is a switch here, not a decision, and the default is the
//      other reading -- for a measured reason given in full at RestTerm.
//    * THE PERTURBATION COEFFICIENT. Eq. (D14) is colour-blind with A/2, giving
//      sigma = 2 A tau / 9; Saito's Eq. (30)-(32) give sigma = 4 A tau / 9 for
//      the same symbol. That is a normalisation of A and not a different surface
//      tension. Reach A through A_from_sigma() and it never matters.
//
//  A SECOND ROUTE TO TWO-PHASE FLOW, and a genuinely different one from the
//  phase field of PhaseFieldSolver.hpp. There the interface is a field with its
//  own advection equation and its own distribution set; here there is no
//  interface equation at all. Two distributions f_i^r and f_i^b are carried, the
//  interface is wherever both are non-zero, and it is held together by a
//  RECOLOURING step that redistributes the colour-blind post-collision state
//  back into red and blue along the colour gradient. Nothing diffuses and
//  nothing is reinitialised: segregation is algebraic.
//
//  Which one to use is not settled by this file. The phase field has a
//  conservative advection equation and a prescribed interface width; the colour
//  gradient has neither, and its interface width is an outcome. What the colour
//  gradient is supposed to have instead is the density ratio -- Saito et al.
//  report 1000 in their own static tests, where the pressure form of
//  MultiphasePotentialBGK.hpp is fighting the conditioning problem of its own
//  banner by ratio 100.
//
//  THIS IMPLEMENTATION IS SUB-PERCENT TO gamma = 20 AND COMES APART AT 1000.
//  validation/static_droplet at 48^3, R = 16, tau = 1, CONVERGED (see below for
//  what that costs), Laplace error against the ratio:
//
//      gamma        1       10       20      100          1000
//      error     0.88 %   0.68 %   0.46 %   3.21 %   droplet comes apart
//
//  At gamma = 1000 it is not slow convergence: the measured Laplace jump crosses
//  ZERO (+4.84e-3, +1.20e-3, -2.34e-4 at 8000/16000/32000 steps) while the
//  interface widens from 4.92 to 5.47 cells. A droplet with positive surface
//  tension cannot do that, so no number is quotable there and the published 1000
//  is not reproduced here. The cause is not diagnosed.
//
//  THE STEP COUNT SCALES WITH GAMMA, AND THAT IS THE TRAP. Relaxation slows as
//  the ratio grows while tau is held at 1, so a ladder quoted at ONE step count
//  across a gamma sweep measures the transient and not the model. Measured:
//  gamma <= 20 is converged by 8000 steps (0.89/0.71/0.50 there against
//  0.88/0.68/0.46 at 32000, so 0.04 points of drift), but gamma = 100 reads
//  2.12 % at 8000, 3.11 % at 16000 and 3.21 % at 32000. An earlier version of
//  this banner quoted 493 % and 3871 % at gamma = 100 and 1000 from a 1500-step
//  run and called them converged-enough to characterise the model. They were
//  pure transient. Check the series is flat before reading a number off it.
//
//  THE THREE SUBOPERATORS, Eq. (D2):
//
//      Omega_i^k = (Omega_i^k)^(3) [ (Omega_i)^(1) + (Omega_i)^(2) ],
//
//  read right to left: collide the colour-blind f_i = f_i^r + f_i^b, perturb it
//  to make surface tension, then recolour. Only the third carries a colour
//  index, which is the whole economy of the scheme -- the expensive part, the
//  central-moment collision, is done ONCE for both fluids.
//
//  (1) THE SINGLE-PHASE COLLISION is a general MRT in central moments,
//  Eq. (D3), with the relaxation matrix Lambda of that paper's Eq. (17),
//
//      K = diag[s0, s1 x3, s2v x5, s2b, s3 x7, s4 x6, s5 x3, s6],
//
//  and s0 = s1 = 0, s2b = s3 = s4 = s5 = s6 = 1. That grouping is ALREADY what
//  MomentCollision.hpp does: five second-order moments at the shear rate, the
//  trace on its own at the bulk rate, everything above second order straight to
//  equilibrium. The paper reaches it through De Rosis's nonorthogonal basis and
//  this code reaches it through a Hermite product basis, and above second order
//  the two cannot disagree -- when every moment of a subspace relaxes at the
//  same rate, any basis of that subspace gives the same operator. At second
//  order the split into trace and traceless deviatoric is basis-independent for
//  the same reason. So ProductBasis is reused unchanged and only the
//  EQUILIBRIUM is new.
//
//  THE EQUILIBRIUM IS NOT THE PRODUCT FORM, THOUGH IT IS BUILT FROM IT. Eq. (D5)
//  is the product form with its rest weight w_i replaced by phi_i, so
//  MomentCollision.hpp's eq_moment() -- where every central moment above order 0
//  collapses to zero -- accounts for only part of it. Reusing it would silently
//  drop the rest difference, which is the entire density ratio. The equilibrium,
//  that difference and the perturbation are evaluated in CLOSED FORM below, one
//  slot at a time; the first version built them as POPULATIONS and transformed
//  them, which was exact and was also the largest stack frame in this tree. See
//  the closed-form banner for what that cost and what removing it bought.
//
//  THE DENSITY RATIO LIVES IN THE REST WEIGHT. phi_i of Eq. (D6) replaces w_i in
//  the rest term of the equilibrium, and carries a free parameter alpha:
//
//      phi_0 = alpha,   phi_{|c|^2=1} = 2(1-alpha)/19,
//      phi_{|c|^2=2} = (1-alpha)/38,   phi_{|c|^2=3} = (1-alpha)/152,
//
//  which sums to one for any alpha, and whose second moment is
//  cs^2 = 9(1-alpha)/19. The sound speed is therefore a PROPERTY OF THE PHASE.
//  Pressure continuity across the interface, p = rho cs^2, then fixes the
//  density ratio without a single density appearing in the collision, Eq. (D11):
//
//      gamma = rho_r^0 / rho_b^0 = (1 - alpha_b) / (1 - alpha_r).
//
//  alpha_b = 8/27 recovers cs^2 = 1/3 exactly for the blue fluid, which is the
//  paper's choice and the one that makes the blue phase an ordinary lattice.
//  Note what this does NOT do: it does not make cs^2 = 1/3 anywhere else, so the
//  standard lattice sound speed is not available as a global constant here and
//  nothing in this file assumes it. ProductBasis's cs^2 is a basis constant, not
//  a physical one, and the two are deliberately not the same number.
//
//  Phi_i of Eq. (D7) is the last term of the equilibrium and the one that is
//  easiest to drop by accident. It restores Galilean invariance when the density
//  varies, through the second moment, using
//
//      G = (1/48) [ u (x) grad rho + (u (x) grad rho)^T ].
//
//  It is the same physics that ViscousInterfaceForce.hpp adds to the phase-field
//  module as an explicit body force nu (grad u + grad u^T) . grad rho -- the same
//  missing stress, entered by a different door. Here it costs no extra field
//  beyond grad rho, because it goes into the equilibrium rather than into F.
//
//  (2) THE PERTURBATION makes the surface tension, Eq. (D14):
//
//      (Omega_i)^(2) = (A/2) |grad phi| [ w_i (c_i . n)^2 - B_i ],
//
//  colour-blind -- Eq. (D14) carries no colour index k, so no doubling is owed
//  and the coefficient is A/2 exactly as printed. The capillary stress of a flat
//  interface then integrates to
//
//      sigma = integral (S_tt - S_nn) dn = (A tau / 9) * delta(phi),
//
//  and phi runs from -1 to +1, so delta(phi) = 2 and sigma = 2 A tau / 9.
//
//  THIS FILE USED TO CARRY A RATHER THAN A/2, on the argument that Saito et
//  al.'s Eq. (30) is written PER COLOUR and that the two halves add when it is
//  applied to the colour-blind population -- which gives their Eq. (32),
//  sigma = 4 A tau / 9. Both readings are self-consistent; they are different
//  normalisations of the symbol A for the same physical sigma, and nothing
//  computed changes as long as A and sigma are converted with the same pair.
//  That is what A_from_sigma() and sigma_from_A() are for, and why every driver
//  here reaches A through them. The three that set A by hand had it doubled on
//  2026-09-19 so that their surface tension did not move.
//
//  n is the unit colour gradient and B_i the lattice constants of Eq. (D15). The
//  operator conserves mass and momentum exactly, and it acts only ACROSS the
//  interface: sum_i w_i (c_i.n)^4 = 3 cs^4 = 1/3 cancels sum_i B_i (c_i.n)^2 =
//  1/3 identically, so the NORMAL component of the capillary stress vanishes for
//  every direction of n, while the tangential component does not. Both are
//  direction-independent because D3Q27's fourth moment is isotropic.
//
//  (3) THE RECOLOURING, Eqs. (D16)-(D17), splits the post-collision f_i back:
//
//      f_i^r = (rho_r/rho) f_i + beta (rho_r rho_b / rho^2) cos(theta_i) f_i^eq(rho,0),
//      f_i^b = (rho_b/rho) f_i - beta (rho_r rho_b / rho^2) cos(theta_i) f_i^eq(rho,0),
//
//  with cos(theta_i) the cosine of the angle between c_i and grad phi. The two
//  sum to f_i identically, so mass and momentum survive it whatever beta is; all
//  beta does is push colour up the gradient. beta = 0.7 is the paper's value and
//  the largest that keeps the interface smooth. f_i^eq(rho,0) is simply rho phi_i
//  -- at zero velocity every other term of Eq. (18) vanishes, Phi_i included.
//
//  WHAT THIS OPERATOR DOES NOT DO. It does not stream, it does not own the two
//  distributions, and it does not compute a gradient: with Esoteric Pull the
//  neighbour populations during a fused kernel are a mixture of two time levels
//  and a gradient taken from them is meaningless. grad phi and grad rho are node
//  FIELDS, computed by ColourGradientSolver in a separate pass, for the same
//  reason PhaseFieldSolver.hpp gives at length.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/ProductBasis.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "core/Types.hpp"
#include "forcing/Forcing.hpp"
#include "lattice/Lattices.hpp"
#include "memory/Storage.hpp"

#include <utility>   // integer_sequence -- the moment loop is unrolled, see collide

namespace lbm {

template <class L>
struct ColourGradient {
  using Lattice = L;
  using Storage = RawPopulations;      // two colours; a w_i shift fits neither
  using Basis   = ProductBasis<L>;
  static constexpr const char* name = "ColourGradientCM";
  static constexpr int D  = L::D;
  static constexpr int NM = Basis::NM;

  static_assert(L::D == 3 && L::Q == 27,
                "the colour-gradient model of Saito et al. (2018) is derived for "
                "D3Q27; phi_i, B_i and sigma = 2 A tau / 9 are all lattice-specific.");
  static_assert(Basis::enabled, "no product basis for this lattice.");
  // The basis's exponent accessors MUST be usable in a constant expression.
  // relax_high_one<N> reads them at compile time precisely so the moment array
  // stays out of memory; dropping constexpr from one of them would not fail to
  // compile, it would put this operator back on a runtime-indexed 432-byte
  // table. See the closed-form banner below and MomentCollision.hpp.
  static_assert(Basis::p_of(NM - 1) >= 0 && Basis::q_of(NM - 1) >= 0 &&
                Basis::r_of(NM - 1) >= 0 && Basis::order(0) == 0,
                "the moment basis's p_of/q_of/r_of/order must be constexpr.");

  //---- parameters ------------------------------------------------------------
  Real alpha_r = Real(8) / Real(27);   // rest weights; gamma = (1-ab)/(1-ar)
  Real alpha_b = Real(8) / Real(27);
  Real nu_r = Real(1) / Real(6);       // kinematic viscosities, harmonic-mixed
  Real nu_b = Real(1) / Real(6);
  Real A = Real(0);                    // interfacial tension, Eq. (D14); sigma = 2 A tau / 9
  Real beta = Real(0.7);               // recolouring sharpness
  Real omega_bulk = Real(1);           // s2b
  Real bx = 0, by = 0, bz = 0;         // body force per unit mass
  // The density the body force is measured against: F = (rho - rho_ref) b.
  // Zero gives the plain rho b. In a FULLY PERIODIC box rho b injects net
  // momentum every step -- there is no wall for the hydrostatic pressure
  // gradient to push against and no boundary to absorb it, so the whole fluid
  // simply falls. Setting rho_ref to the domain mean removes the mean force and
  // leaves the buoyancy difference, which is the part a Rayleigh-Taylor problem
  // is actually about. It is not a Boussinesq approximation: the full density
  // difference is retained, only its mean is subtracted.
  Real rho_ref = 0;

  // The initial density of each PURE phase, Eq. (D10). Carried explicitly rather
  // than derived from alpha, even though equal pressure ties them together as
  // rho^0 = const / (1 - alpha): the derived form inverts easily and does so
  // SILENTLY. Getting it upside down leaves both pure phases at phi = +-1 --
  // every bulk check still passes -- and corrupts only the interface, where the
  // weighting between the two colours decides alpha and therefore the local
  // pressure. Measured at a ratio of 10 it dropped the droplet's density from
  // 10 to 7.95 and put the Laplace tension out by 91%; at 100 it went to NaN.
  Real rho_r0 = Real(1);
  Real rho_b0 = Real(1);

  //----------------------------------------------------------------------------
  //  WHICH ALPHA THE REST TERM SEES, AND WHY IT IS A SWITCH RATHER THAN A
  //  DECISION.
  //
  //  Eq. (D6) writes phi_i(alpha-bar) with the single interpolated alpha-bar of
  //  Eq. (D13). Eq. (D12) writes the pressure as `rho (cs^k)^2 = rho_k 9(1-alpha
  //  -bar)/19` -- carrying BOTH rho and rho_k, and calling cs^k "the speed of
  //  sound of the fluid k". Those two cannot both be right, and the difference
  //  is not cosmetic: it is the interface pressure at a density ratio.
  //
  //  MEASURED, in closed form. With a symmetric interface profile the midpoint
  //  has phi = 0, so alpha-bar = (alpha_r + alpha_b)/2; cs^2 is LINEAR in alpha,
  //  so cs^2(alpha-bar) = (cs_r^2 + cs_b^2)/2, and with cs_r^2 = cs_b^2/gamma and
  //  rho = (gamma + 1)/2 the AlphaBar pressure there is
  //
  //      p_mid / p_bulk = (gamma + 1)^2 / (4 gamma),
  //
  //  i.e. 1.00, 3.03, 25.50 and 250.50 at gamma = 1, 10, 100 and 1000. That is a
  //  pressure spike a couple of cells wide sitting inside a field whose bulk
  //  value is 1/3, and it is what makes the static droplet return NaN at
  //  gamma >= 100. PerColour has no such term: sum_k rho_k cs_k^2 is continuous
  //  by construction, because Eq. (D11) is derived by balancing exactly that.
  //
  //  THEY ARE THE SAME FUNCTIONAL FORM, AND THAT IS THE USEFUL WAY TO SEE IT.
  //  phi_i is AFFINE in alpha -- its rest slot is alpha and every other slot is
  //  a fixed multiple of (1 - alpha) -- from which, identically, population by
  //  population,
  //
  //      sum_k rho_k phi_i(alpha_k)  ==  rho phi_i(alpha_P),
  //      alpha_P = 1 - 19 P / (9 rho),      P = sum_k rho_k cs_k^2.
  //
  //  So PerColour IS Eq. (D6)'s own single-alpha form, with alpha chosen to make
  //  the pressure continuous. The entire difference between the two readings is
  //  THE INTERPOLATION RULE FOR ALPHA -- linear in the order parameter, Eq.
  //  (D13), against alpha_P -- and nothing else: not the equilibrium, not the
  //  collision, not the perturbation. tests/test_colour_gradient.cpp block 2
  //  pins that identity at 2.6e-15.
  //
  //  PERCOLOUR IS THE DEFAULT, on the measurement rather than on the reading.
  //  validation/static_droplet at 48^3, R = 16, tau = 1, sigma held fixed,
  //  CONVERGED (32000 steps; see the top banner on why that matters), Laplace
  //  error against gamma:
  //
  //      gamma         1        10       20      100     1000
  //      PerColour   0.88 %   0.68 %   0.46 %   3.21 %   comes apart
  //      AlphaBar    0.88 %   3.34 %   1.51 %   NaN      NaN
  //
  //  Identical at gamma = 1, because there the two readings are algebraically
  //  the same; about FIVE TIMES the error at gamma = 10; gone at 100.
  //
  //  THE EQUILIBRIUM ORDER IS NOT WHAT IS AT STAKE HERE. Holding Eq. (D5) fixed
  //  and swapping only this reading reproduces Saito et al.'s THIRD-ORDER ladder
  //  to four or five significant figures at every ratio (0.23 / 0.95 / 493.49 /
  //  3871.32 against 0.23 / 0.95 / 493.50 / 3871.32 at a matched 1500 steps).
  //  The difference between the two equilibria is exactly the truncation
  //  residuals, -rho ux^2 uy^2 at order 4 and -10 rho ux^2 uy^2 uz^2 at order 6,
  //  and at a droplet's spurious velocity those are ~1e-20.
  //
  //  AlphaBar is kept, exposed, and NOT hidden behind a comment, because the gap
  //  between the two IS the measurement and because Eq. (D6) as printed is what
  //  it implements. Choose it to reproduce the paper literally; expect the spike.
  //----------------------------------------------------------------------------
  enum class RestTerm { AlphaBar, PerColour };
  RestTerm rest = RestTerm::PerColour;

  //----------------------------------------------------------------------------
  // rho_r and rho_b recovered from (rho, phi). EXACT, not an approximation:
  // phi = (a - b)/(a + b) with a = rho_r/rho_r0 and b = rho_b/rho_b0 fixes the
  // ratio a:b, and rho fixes the scale. The collision is handed rho and phi
  // rather than the two densities -- Eq. (D5) is a function of rho -- so this is
  // what PerColour needs to reach its own alphas.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void split_colours(Real rho, Real p, Real& rr, Real& rb) const {
    const Real ha = Real(0.5) * (Real(1) + p), hb = Real(0.5) * (Real(1) - p);
    const Real den = ha * rho_r0 + hb * rho_b0;
    const Real sc  = (den > Real(0)) ? rho / den : Real(0);
    rr = sc * ha * rho_r0;
    rb = sc * hb * rho_b0;
  }

  //---- node fields, written by the solver ------------------------------------
  View1D<const Real> phi;              // order parameter, Eq. (D10)
  View1D<const Real> Gx, Gy, Gz;       // grad phi
  View1D<const Real> Rx, Ry, Rz;       // grad rho, for Phi_i of Eq. (D7)

  //----------------------------------------------------------------------------
  // sigma = (2/9) A tau, Eq. (D14), and its inverse. tau = 1/s2v is the SHEAR
  // relaxation time of the mixture, so a viscosity ratio makes sigma depend on
  // where the interface sits; the paper's tests use a matched tau and so does
  // the validation case.
  //----------------------------------------------------------------------------
  static Real A_from_sigma(Real sigma, Real tau) {
    return Real(9) * sigma / (Real(2) * tau);
  }
  static Real sigma_from_A(Real a, Real tau) {
    return Real(2) * a * tau / Real(9);
  }
  // The coefficient the colour-blind perturbation carries: A/2, exactly as
  // Eq. (D14) writes it. Exposed so the test asserts the relation rather than
  // reproducing the constant, and so a driver never has to know it.
  static constexpr Real perturbation_coefficient(Real a) { return Real(0.5) * a; }

  //----------------------------------------------------------------------------
  // The order parameter, Eq. (D10). +1 in pure red, -1 in pure blue, 0 where the
  // two are present in equal PROPORTION of their own bulk densities -- which is
  // not the same as equal mass, and is the whole point of dividing by rho^0.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real order_parameter(Real rr, Real rb) const {
    const Real a = rr / rho_r0, b = rb / rho_b0;
    const Real s = a + b;
    return (s > Real(0)) ? (a - b) / s : Real(0);
  }

  // cs^2 = 9(1-alpha)/19, the second moment of phi_i. NOT the lattice constant.
  KOKKOS_INLINE_FUNCTION static Real cs2_of_alpha(Real a) {
    return Real(9) * (Real(1) - a) / Real(19);
  }
  // gamma = (1 - alpha_b)/(1 - alpha_r), Eq. (D11), inverted for alpha_r.
  static Real alpha_r_from_ratio(Real gamma, Real ab) {
    return Real(1) - (Real(1) - ab) / gamma;
  }

  //----------------------------------------------------------------------------
  // phi_i, Eq. (D6). Keyed off |c_i|^2, which on this lattice is 0, 1, 2 or 3.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION static Real phi_i(int i, Real a) {
    const int q = cvel<L>(i, 0) * cvel<L>(i, 0) + cvel<L>(i, 1) * cvel<L>(i, 1)
                + cvel<L>(i, 2) * cvel<L>(i, 2);
    const Real c = Real(1) - a;
    if (q == 0) return a;
    if (q == 1) return Real(2) * c / Real(19);
    if (q == 2) return c / Real(38);
    return c / Real(152);
  }

  // B_i, Eq. (D15). sum_i B_i = 1/3, which is what makes the perturbation
  // conserve mass against sum_i w_i (c_i.n)^2 = 1/3.
  KOKKOS_INLINE_FUNCTION static Real B_i(int i) {
    const int q = cvel<L>(i, 0) * cvel<L>(i, 0) + cvel<L>(i, 1) * cvel<L>(i, 1)
                + cvel<L>(i, 2) * cvel<L>(i, 2);
    if (q == 0) return Real(-10) / Real(27);
    if (q == 1) return Real(2) / Real(27);
    if (q == 2) return Real(1) / Real(54);
    return Real(1) / Real(216);
  }

  // The coefficient of (G : c_i x c_i) in Phi_i, Eq. (D7).
  KOKKOS_INLINE_FUNCTION static Real Phi_coeff(int i) {
    const int q = cvel<L>(i, 0) * cvel<L>(i, 0) + cvel<L>(i, 1) * cvel<L>(i, 1)
                + cvel<L>(i, 2) * cvel<L>(i, 2);
    if (q == 0) return Real(0);
    if (q == 1) return Real(16);
    if (q == 2) return Real(4);
    return Real(1);
  }

  //----------------------------------------------------------------------------
  // Interpolants across the interface. alpha and the viscosity are mixed
  // differently on purpose: alpha linearly, Eq. (D13), and the viscosity
  // HARMONICALLY, Eq. (D9). The harmonic mean is not a stylistic choice -- it is
  // what keeps the shear stress continuous across an interface with a viscosity
  // ratio, and a linear mean there produces a jump in stress that shows up as a
  // spurious current.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real alpha_at(Real p) const {
    return Real(0.5) * ((Real(1) + p) * alpha_r + (Real(1) - p) * alpha_b);
  }
  KOKKOS_INLINE_FUNCTION Real nu_at(Real p) const {
    const Real inv = Real(0.5) * ((Real(1) + p) / nu_r + (Real(1) - p) / nu_b);
    return Real(1) / inv;
  }
  //----------------------------------------------------------------------------
  // The shear rate, Eq. (16): nu = (c^2/3)(1/s2v - 1/2). THE 1/3 IS THE
  // LATTICE'S, not the phase's, and the distinction is the single most
  // consequential detail in this operator.
  //
  // cs^2 appears in two unrelated roles here and they are different numbers.
  // The EQUATION OF STATE is p = rho cs^2(alpha) with cs^2(alpha) = 9(1-alpha)/19,
  // because the rest weight phi_i carries the density ratio and the
  // equilibrium's trace follows it. The VISCOUS STRESS does not: it comes from
  // the non-equilibrium part, whose second moment is governed by the standard
  // weights w_i in the velocity-dependent terms of Eq. (18), and those carry
  // cs^2 = 1/3 in every phase.
  //
  // Using the phase's cs^2 here instead was measured, and it fails in the way
  // that is hardest to attribute: at a density ratio of 1000 it drives omega to
  // 1.992 in the middle of the interface -- inside the stability limit by a
  // hair, wrong by a factor of 500 in the viscosity, and the droplet returns
  // NaN a few hundred steps later with nothing in between to point at it.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real omega_at(Real p) const {
    return Real(1) / (nu_at(p) * inv_cs2<L, Real>() + Real(0.5));
  }
  static Real viscosity_from_tau(Real tau) {
    return cs2<L, Real>() * (tau - Real(0.5));
  }

  //----------------------------------------------------------------------------
  //  THE REST TERM, Eq. (D5)'s `rho phi_i`, ON A SINGLE INTERPOLATED ALPHA.
  //
  //  alpha-bar is Eq. (D13), linear in the order parameter, and it is the ONLY
  //  alpha the equilibrium sees. Its pure-phase limits are alpha_r and alpha_b,
  //  and Eq. (D11) -- rho_r^0 (1 - alpha_r) = rho_b^0 (1 - alpha_b) -- makes the
  //  pressure rho cs^2(alpha-bar) of Eq. (D12) match at BOTH ends of the
  //  interface. What happens between the ends is an interpolation question, not
  //  an identity, and it is measured rather than asserted: see the density-ratio
  //  ladder in the banner at the top of this file.
  //
  //  THIS REPLACED A PER-COLOUR READING, sum_k rho_k phi_i(alpha_k), which is
  //  what Saito et al.'s Eq. (18) was taken to mean here until 2026-09-19. That
  //  reading has a trace of sum_k rho_k cs_k^2, continuous by construction; this
  //  one does not have that guarantee, and the earlier version of this file
  //  recorded a 25-fold pressure spike two cells wide at gamma = 100 with it.
  //  THAT MEASUREMENT WAS TAKEN UNDER THE THIRD-ORDER EQUILIBRIUM and does not
  //  transfer: the equilibrium below is the full sixth-order Hermite set, whose
  //  central moments carry no velocity dependence at all, so the interface state
  //  the spike was measured in is not the state this operator reaches. It is
  //  re-measured, not assumed.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real rest_term(int i, Real rho, Real p) const {
    if (rest == RestTerm::PerColour) {
      Real rr, rb;  split_colours(rho, p, rr, rb);
      return rr * phi_i(i, alpha_r) + rb * phi_i(i, alpha_b);
    }
    return rho * phi_i(i, alpha_at(p));
  }

  //----------------------------------------------------------------------------
  // The pressure, Eq. (D12): p = rho cs^2(alpha-bar). This is the trace of the
  // rest term, and the only place the density ratio enters the collision.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real pressure(Real rho, Real p) const {
    if (rest == RestTerm::PerColour) {
      Real rr, rb;  split_colours(rho, p, rr, rb);
      return rr * cs2_of_alpha(alpha_r) + rb * cs2_of_alpha(alpha_b);
    }
    return rho * cs2_of_alpha(alpha_at(p));
  }

  //----------------------------------------------------------------------------
  // Delta = rho [ cs^2(alpha-bar) - cs^2 ], the rest term measured against the
  // LATTICE weights rather than against nothing -- see the closed-form banner.
  // It is the single number by which this equilibrium differs from the ordinary
  // product-form one, and it is zero exactly when alpha-bar = 8/27.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real delta_of(Real rho, Real p) const {
    return pressure(rho, p) - rho * cs2<L, Real>();
  }

  // f_i^eq(rho, u = 0): every other term of Eq. (D5) carries a u or a G, and G
  // vanishes with u, so the rest term is the whole equilibrium at rest. This is
  // what the recolouring of Eqs. (D16)-(D17) is written against.
  KOKKOS_INLINE_FUNCTION Real eq_at_rest(int i, Real rho, Real p) const {
    return rest_term(i, rho, p);
  }

  //----------------------------------------------------------------------------
  // The two colours' shares of f_i^eq(rho, u = 0), which sum to eq_at_rest()
  // EXACTLY under either reading. The solver seeds with this rather than with
  // its own copy of the formula, because a seed that disagrees with the
  // collision about which alpha is in force puts the interface out of
  // equilibrium on step 0 and looks exactly like a bad model. That is not
  // hypothetical: it is the bug that produced this file's first, wrong,
  // attribution of the gamma = 10 divergence on 2026-09-19.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void seed_at_rest(int i, Real rho_r, Real rho_b, Real p, Real& sr, Real& sb) const {
    if (rest == RestTerm::PerColour) {
      sr = rho_r * phi_i(i, alpha_r);
      sb = rho_b * phi_i(i, alpha_b);
    } else {
      const Real w = phi_i(i, alpha_at(p));
      sr = rho_r * w;
      sb = rho_b * w;
    }
  }

  //----------------------------------------------------------------------------
  //  THE EQUILIBRIUM, Eq. (D5) with the Phi_i of Eq. (D7).
  //
  //  SIXTH-ORDER HERMITE, which is the whole difference from Saito et al.'s
  //  Eq. (18) and is not a refinement of it. Their equilibrium truncates at
  //  third order in u; Eq. (D5) carries the complete set, and the paper's own
  //  statement of what that buys is that "the equilibrium state is fully
  //  Galilean invariant since it does not show any dependence on the fluid
  //  velocity" -- i.e. its central moments are constants, not polynomials in u.
  //
  //  IT IS EVALUATED AS A PRODUCT, NOT AS SIX HERMITE TENSORS. On D3Q27 the
  //  complete sixth-order series is identically the factorised product form,
  //  verified in exact rational arithmetic in MATLAB/D3Q27_CM.m and implemented
  //  as ProductFormPhi<D3Q27> -- three multiplies instead of six tensors, the
  //  same polynomial bit for bit. So Eq. (D5) is
  //
  //      f_i^eq = rho w_i Pi_i(u)  +  rho [ phi_i(alpha-bar) - w_i ]  +  Phi_i,
  //
  //  reading the bracket of Eq. (D5) as the product form with its own `1` taken
  //  back out and phi_i put in its place. Nothing is approximated by writing it
  //  this way, and the second term is what carries the density ratio.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void equilibrium(Real fe[L::Q], Real rho, Real p, const Real u[3],
                   const Real G[3][3], Real nubar, Real udrho) const {
    for (int i = 0; i < L::Q; ++i) {
      Real e = ProductFormEquilibrium<L>::eq(i, rho, u[0], u[1], u[2])
             + rest_term(i, rho, p) - rho * weight<L, Real>(i);
      // Phi_i, Eq. (D7): the rest slot balances the rest, so sum_i Phi_i = 0.
      const Real k = Phi_coeff(i);
      if (k == Real(0)) {
        e += Real(-3) * nubar * udrho;
      } else {
        const Real c[3] = {Real(cvel<L>(i, 0)), Real(cvel<L>(i, 1)),
                           Real(cvel<L>(i, 2))};
        Real gcc = Real(0);
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) gcc += G[a][b] * c[a] * c[b];
        e += k * nubar * gcc;
      }
      fe[i] = e;
    }
  }

  //============================================================================
  //  CLOSED-FORM CENTRAL MOMENTS, AND WHY EQ. (D5) MAKES THEM SHORT.
  //
  //  The first version of this operator built the equilibrium and the
  //  perturbation as POPULATIONS and transformed both. That is exact by
  //  construction and was the right way to get it correct; it also cost three
  //  forward transforms per node and three live 27-arrays, and it made this the
  //  largest stack frame in the tree -- 1200 bytes at FP64, measured by
  //  tests/frame_check.sh. On a CPU that frame is L1-resident and costs little;
  //  in DEVICE code it is per-thread LOCAL memory, off-chip, with every
  //  subscript uncoalesced, and GPU/ measured that mechanism here at 47x.
  //
  //  WHAT EQ. (D5) CHANGES. Under Saito et al.'s third-order equilibrium the
  //  closed form needed a total-order coefficient table and a second separable
  //  product to carry the truncation residuals -- terms like -rho ux^2 uy^2 at
  //  order 4 and -10 rho ux^2 uy^2 uz^2 at order 6. Eq. (D5) is the COMPLETE
  //  sixth-order Hermite set, which on D3Q27 is the product form, and the
  //  product form's central moments in this shifted basis are
  //
  //      k = rho  at slot 0,  and EXACTLY ZERO everywhere else.
  //
  //  (ProductBasis's own banner records that, checked symbolically.) Every
  //  residual therefore vanishes and the table, the second product and their
  //  per-axis factors were deleted along with them. This is the paper's
  //  "fully Galilean invariant ... does not show any dependence on the fluid
  //  velocity", read as a statement about the moment vector.
  //
  //  WHAT IS LEFT IS TWO SOURCES AND A CONSTANT.
  //
  //  (a) THE REST DIFFERENCE, rho [phi_i(alpha-bar) - w_i]. Both weight sets sum
  //      to one, so this has NO zeroth moment, which is why it never disturbs
  //      the density. Its raw moments are isotropic and GEOMETRIC in the number
  //      of squared indices -- sum_i phi_i c_x^2 = cs^2(alpha), then c_x^2 c_y^2
  //      gives cs^2(alpha)/3 and c_x^2 c_y^2 c_z^2 gives cs^2(alpha)/9, and the
  //      same with cs^2 = 1/3 for w_i -- so with
  //
  //          Delta = rho [ cs^2(alpha-bar) - 1/3 ]
  //
  //      its raw moments are T_ab = Delta delta_ab, Q = Delta/3, R = 0,
  //      H = Delta/9, and every odd one vanishes. That is EXACTLY the shape of
  //      Phi_i and of the perturbation, so it needs no machinery of its own: it
  //      is added into the same Source and shifted by the same source_high().
  //
  //  (b) Phi_i of Eq. (D7) AND the perturbation of Eq. (D14), unchanged in
  //      structure: every ODD raw moment vanishes, leaving a second-order tensor
  //      T, a fourth-order pair (Q, R) and a sixth-order scalar H.
  //      source_high() is the shift of exactly such a source, derived once over
  //      exact rationals and used for all three. Its 17 cases are
  //      machine-generated rather than transcribed, because 27 formulas copied
  //      by eye is precisely how a silent error gets in.
  //
  //  SO THE WHOLE EQUILIBRIUM IS:
  //
  //      order 0    rho
  //      order 1    0
  //      order 2    Delta delta_ab + E_ab          (no u-dependence at all)
  //      order >=3  source_high<N>(Delta + Phi)
  //
  //  and at alpha-bar = 8/27, Delta = 0 and the classical result -- equilibrium
  //  central moments vanish above order 0 -- comes back as the special case.
  //
  //  NOTHING HERE IS ASSUMED FROM eq_moment(). MomentCollision.hpp's version
  //  collapses to rho delta_N0 for the product form alone; Eq. (D5) is the
  //  product form PLUS the rest difference, so reusing it would silently drop
  //  Delta -- the entire density ratio. tests/test_colour_gradient.cpp block 6
  //  keeps the population path and asserts the two agree, which is what makes
  //  the replacement safe rather than merely plausible.
  //
  //  EVERY SLOT INDEX BELOW IS A TEMPLATE PARAMETER, for the reason
  //  MomentCollision.hpp's eq_moment() banner gives at length: Basis::p_of is a
  //  lookup in a 432-byte table, and a runtime index forces that table into
  //  memory, the p it returns then indexes the factors, and k[n] follows them.
  //  The dispatch is `if constexpr` rather than a switch on a constant, because
  //  the failure this rewrite exists to undo was relying on a compiler to do
  //  something it was never obliged to do.
  //============================================================================

  // The raw moments of a source whose odd moments all vanish. Both Phi_i and the
  // perturbation are of this kind, so one struct and one shift serve both.
  struct Source {
    Real Txx, Tyy, Tzz, Txy, Txz, Tyz;   // order 2
    Real Qxy, Qxz, Qyz;                  // order 4, the (2,2,0) family
    Real Rx, Ry, Rz;                     // order 4, the (2,1,1) family
    Real H;                              // order 6
  };

  //----------------------------------------------------------------------------
  // The central moment of such a source at slot N, for order >= 3 only -- order
  // 0 and 1 vanish identically and order 2 is just T, so the caller reads those
  // off directly. MACHINE-GENERATED; see the banner above.
  //----------------------------------------------------------------------------
  template <int N>
  KOKKOS_INLINE_FUNCTION
  static Real source_high(const Source& s, Real ux, Real uy, Real uz) {
    [[maybe_unused]] const Real ux2 = ux * ux, uy2 = uy * uy, uz2 = uz * uz;
    if constexpr (N == 5) {         // (0,1,2)
      return -Real(2)*s.Tyz*uz - s.Tzz*uy ;
    } else if constexpr (N == 7) {  // (0,2,1)
      return -Real(2)*s.Tyz*uy - s.Tyy*uz ;
    } else if constexpr (N == 8) {  // (0,2,2)
      return s.Qyz - (Real(1)/Real(3))*s.Tzz - (Real(1)/Real(3))*s.Tyy +
             Real(4)*s.Tyz*uy*uz + s.Tzz*uy2 + s.Tyy*uz2 ;
    } else if constexpr (N == 11) { // (1,0,2)
      return -Real(2)*s.Txz*uz - s.Tzz*ux ;
    } else if constexpr (N == 13) { // (1,1,1)
      return -s.Tyz*ux - s.Txz*uy - s.Txy*uz ;
    } else if constexpr (N == 14) { // (1,1,2)
      return s.Rz - (Real(1)/Real(3))*s.Txy + Real(2)*s.Tyz*ux*uz +
             Real(2)*s.Txz*uy*uz + s.Txy*uz2 + s.Tzz*ux*uy ;
    } else if constexpr (N == 15) { // (1,2,0)
      return -Real(2)*s.Txy*uy - s.Tyy*ux ;
    } else if constexpr (N == 16) { // (1,2,1)
      return s.Ry - (Real(1)/Real(3))*s.Txz + Real(2)*s.Tyz*ux*uy + s.Txz*uy2 +
             Real(2)*s.Txy*uy*uz + s.Tyy*ux*uz ;
    } else if constexpr (N == 17) { // (1,2,2)
      return -Real(2)*s.Rz*uy - Real(2)*s.Ry*uz - s.Qyz*ux +
             (Real(2)/Real(3))*s.Txz*uz + (Real(2)/Real(3))*s.Txy*uy +
             (Real(1)/Real(3))*s.Tzz*ux + (Real(1)/Real(3))*s.Tyy*ux -
             Real(4)*s.Tyz*ux*uy*uz - Real(2)*s.Txz*uy2*uz - Real(2)*s.Txy*uy*uz2 -
             s.Tzz*ux*uy2 - s.Tyy*ux*uz2 ;
    } else if constexpr (N == 19) { // (2,0,1)
      return -Real(2)*s.Txz*ux - s.Txx*uz ;
    } else if constexpr (N == 20) { // (2,0,2)
      return s.Qxz - (Real(1)/Real(3))*s.Tzz - (Real(1)/Real(3))*s.Txx +
             Real(4)*s.Txz*ux*uz + s.Tzz*ux2 + s.Txx*uz2 ;
    } else if constexpr (N == 21) { // (2,1,0)
      return -Real(2)*s.Txy*ux - s.Txx*uy ;
    } else if constexpr (N == 22) { // (2,1,1)
      return s.Rx - (Real(1)/Real(3))*s.Tyz + s.Tyz*ux2 + Real(2)*s.Txz*ux*uy +
             Real(2)*s.Txy*ux*uz + s.Txx*uy*uz ;
    } else if constexpr (N == 23) { // (2,1,2)
      return -Real(2)*s.Rz*ux - Real(2)*s.Rx*uz - s.Qxz*uy +
             (Real(2)/Real(3))*s.Tyz*uz + (Real(2)/Real(3))*s.Txy*ux +
             (Real(1)/Real(3))*s.Tzz*uy + (Real(1)/Real(3))*s.Txx*uy -
             Real(2)*s.Tyz*ux2*uz - Real(4)*s.Txz*ux*uy*uz - Real(2)*s.Txy*ux*uz2 -
             s.Tzz*ux2*uy - s.Txx*uy*uz2 ;
    } else if constexpr (N == 24) { // (2,2,0)
      return s.Qxy - (Real(1)/Real(3))*s.Tyy - (Real(1)/Real(3))*s.Txx +
             Real(4)*s.Txy*ux*uy + s.Tyy*ux2 + s.Txx*uy2 ;
    } else if constexpr (N == 25) { // (2,2,1)
      return -Real(2)*s.Ry*ux - Real(2)*s.Rx*uy - s.Qxy*uz +
             (Real(2)/Real(3))*s.Tyz*uy + (Real(2)/Real(3))*s.Txz*ux +
             (Real(1)/Real(3))*s.Tyy*uz + (Real(1)/Real(3))*s.Txx*uz -
             Real(2)*s.Tyz*ux2*uy - Real(2)*s.Txz*ux*uy2 - Real(4)*s.Txy*ux*uy*uz -
             s.Tyy*ux2*uz - s.Txx*uy2*uz ;
    } else if constexpr (N == 26) { // (2,2,2)
      return s.H - (Real(1)/Real(3))*s.Qyz - (Real(1)/Real(3))*s.Qxz -
             (Real(1)/Real(3))*s.Qxy + (Real(1)/Real(9))*s.Tzz +
             (Real(1)/Real(9))*s.Tyy + (Real(1)/Real(9))*s.Txx + Real(4)*s.Rz*ux*uy
             + Real(4)*s.Ry*ux*uz + Real(4)*s.Rx*uy*uz + s.Qyz*ux2 + s.Qxz*uy2 +
             s.Qxy*uz2 - (Real(4)/Real(3))*s.Tyz*uy*uz -
             (Real(4)/Real(3))*s.Txz*ux*uz - (Real(4)/Real(3))*s.Txy*ux*uy -
             (Real(1)/Real(3))*s.Tzz*uy2 - (Real(1)/Real(3))*s.Tzz*ux2 -
             (Real(1)/Real(3))*s.Tyy*uz2 - (Real(1)/Real(3))*s.Tyy*ux2 -
             (Real(1)/Real(3))*s.Txx*uz2 - (Real(1)/Real(3))*s.Txx*uy2 +
             Real(4)*s.Tyz*ux2*uy*uz + Real(4)*s.Txz*ux*uy2*uz +
             Real(4)*s.Txy*ux*uy*uz2 + s.Tzz*ux2*uy2 + s.Tyy*ux2*uz2 +
             s.Txx*uy2*uz2 ;
    } else {
      return Real(0);
    }
  }

  // The moment slots as a compile-time list, and the fold over them. Same
  // construction as MomentCollision.hpp: no lambda, because nvcc rejects an
  // extended lambda inside a generic one, and no recursion, so there is no
  // inlining depth for a compiler to give up on.
  using Moments = std::make_integer_sequence<int, NM>;

  template <int N>
  KOKKOS_INLINE_FUNCTION
  static void relax_high_one(Real k[NM], const Source& s, Real ux, Real uy,
                             Real uz) {
    // s3 = s4 = s5 = s6 = 1, and the product form contributes NOTHING above
    // order 0, so the post-collision moment is the shifted source and nothing
    // else. Under the third-order equilibrium this line carried two more
    // separable products; Eq. (D5) removed them.
    if constexpr (Basis::order(N) >= 3) k[N] = source_high<N>(s, ux, uy, uz);
  }
  template <int... N>
  KOKKOS_INLINE_FUNCTION
  static void relax_high(Real k[NM], const Source& s, Real ux, Real uy, Real uz,
                         std::integer_sequence<int, N...>) {
    (relax_high_one<N>(k, s, ux, uy, uz), ...);
  }

  //----------------------------------------------------------------------------
  // Suboperators (1) and (2), on the colour-blind populations.
  //
  // `rho` and `u` are the mixture's, `p` the order parameter, `n` the node.
  // f is overwritten with the post-collision, post-perturbation state; the
  // caller recolours it.
  //
  // ONE transform in, one out. The equilibrium and the perturbation never become
  // populations and never become moment arrays; they are evaluated per slot from
  // the closed forms above.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void collide(Real f[L::Q], Real rho, const Real u[3], Real p,
               Index n) const {
    const Real nubar = nu_at(p);
    const Real omega = omega_at(p);
    const Real ux = u[0], uy = u[1], uz = u[2];
    const Real dr[3] = {Rx(n), Ry(n), Rz(n)};
    const Real g[3]  = {Gx(n), Gy(n), Gz(n)};

    // ---- Phi_i, Eq. (D7), through its raw moments. 48 G_ab = u_a dr_b +
    // u_b dr_a, so G is never formed either.
    const Real udr = ux * dr[0] + uy * dr[1] + uz * dr[2];
    const Real Exx = nubar * (Real(2) * ux * dr[0] + udr);
    const Real Eyy = nubar * (Real(2) * uy * dr[1] + udr);
    const Real Ezz = nubar * (Real(2) * uz * dr[2] + udr);
    const Real Exy = nubar * (ux * dr[1] + uy * dr[0]);
    const Real Exz = nubar * (ux * dr[2] + uz * dr[0]);
    const Real Eyz = nubar * (uy * dr[2] + uz * dr[1]);
    const Real t3  = nubar * udr / Real(3);

    // ---- (2) THE PERTURBATION, Eq. (D14), through its raw moments.
    //
    // THE COEFFICIENT IS A/2, colour-blind, exactly as Eq. (D14) writes it --
    // that equation carries no colour index k, so no doubling is owed. It gives
    // sigma = 2 A tau / 9, which is HALF what Saito et al.'s Eq. (32) calls
    // sigma for the same symbol A. That is a normalisation of A and not a
    // different surface tension: every driver here reaches A through
    // A_from_sigma(), so the physics is carried by sigma and A never has to be
    // compared across the two papers. A driver that sets A by hand does have to
    // be, and the three that did were rescaled on 2026-09-19.
    //
    // Its second moment is (A/9) |grad phi| (n_a n_b - delta_ab) with n the
    // unit colour gradient, so the NORMAL component vanishes identically for
    // every direction of n and the tangential one does not. Writing it as
    // g_a g_b / |g| keeps the unit vector implicit and costs one reciprocal.
    Real Pxx = Real(0), Pyy = Real(0), Pzz = Real(0);
    Real Pxy = Real(0), Pxz = Real(0), Pyz = Real(0);
    Real Qxy = Real(0), Qxz = Real(0), Qyz = Real(0);
    Real Rx_ = Real(0), Ry_ = Real(0), Rz_ = Real(0);
    const Real gm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    if (gm2 > Real(1e-24) && A != Real(0)) {
      const Real gm  = Kokkos::sqrt(gm2);
      const Real inv = Real(1) / gm;
      const Real ca  = A / Real(9);             // = (2/9) * (A/2), Eq. (D14)
      const Real cq  = A / Real(27) * inv;      // = (2/27) * (A/2)
      Pxx = ca * (g[0] * g[0] * inv - gm);
      Pyy = ca * (g[1] * g[1] * inv - gm);
      Pzz = ca * (g[2] * g[2] * inv - gm);
      Pxy = ca * g[0] * g[1] * inv;
      Pxz = ca * g[0] * g[2] * inv;
      Pyz = ca * g[1] * g[2] * inv;
      Qxy = -cq * g[2] * g[2];
      Qxz = -cq * g[1] * g[1];
      Qyz = -cq * g[0] * g[0];
      Rx_ =  cq * g[1] * g[2];
      Ry_ =  cq * g[0] * g[2];
      Rz_ =  cq * g[0] * g[1];
    }

    // ---- (0) THE REST DIFFERENCE, rho [phi_i(alpha-bar) - w_i]: the only part
    // of Eq. (D5) the product form does not already carry. Isotropic, even, and
    // GEOMETRIC in the number of squared indices, so one scalar describes it.
    const Real Delta = delta_of(rho, p);

    // All three sources have the same shape and above second order are only
    // ever needed added together.
    Source s;
    s.Txx = Delta + Exx + Pxx;  s.Tyy = Delta + Eyy + Pyy;
    s.Tzz = Delta + Ezz + Pzz;
    s.Txy = Exy + Pxy;  s.Txz = Exz + Pxz;  s.Tyz = Eyz + Pyz;
    const Real d3 = Delta / Real(3);
    s.Qxy = d3 + nubar * (Real(2) * (ux * dr[0] + uy * dr[1]) / Real(3)) + t3 + Qxy;
    s.Qxz = d3 + nubar * (Real(2) * (ux * dr[0] + uz * dr[2]) / Real(3)) + t3 + Qxz;
    s.Qyz = d3 + nubar * (Real(2) * (uy * dr[1] + uz * dr[2]) / Real(3)) + t3 + Qyz;
    s.Rx  = nubar * (uy * dr[2] + uz * dr[1]) / Real(3) + Rx_;   // no Delta: the
    s.Ry  = nubar * (ux * dr[2] + uz * dr[0]) / Real(3) + Ry_;   // (2,1,1) raw
    s.Rz  = nubar * (ux * dr[1] + uy * dr[0]) / Real(3) + Rz_;   // moment is odd
    s.H   = Delta / Real(9) + t3;                // the perturbation has none

    // ---- (1) the central-moment collision. The only transform in the operator.
    const Real ub[3] = {ux, uy, uz};
    Real k[NM];
    Basis::template to_moments<true>(f, ub, k);

    // order 1: conserved. s0 = s1 = 0 in Lambda means the collision leaves
    // them alone, and BOTH sources have a vanishing first central moment, so the
    // body force is the only thing that moves them.
    const Real fw = rho - rho_ref;
    k[i1(0)] += fw * bx;
    k[i1(1)] += fw * by;
    k[i1(2)] += fw * bz;

    // order 2: trace at s2b, deviatoric and shear at s2v. The perturbation is
    // NOT relaxed -- it is a source, and the (1 - s/2) factor that a Guo force
    // carries does not apply to it: Eq. (39) defines the capillary stress as
    // -tau sum_i Omega^(2) c_i c_i, which is the UNRELAXED second moment
    // integrated over one relaxation time. Applying (1 - s/2) here would put
    // sigma out by that factor, and the static droplet would report it.
    {
      // The equilibrium's second central moment, closed form: Delta delta_ab
      // plus Phi_i's own, and NO u-dependence -- the product form contributes
      // nothing here. Under the third-order equilibrium this slot needed a
      // cancellation between two u^2 terms to come out constant; under Eq. (D5)
      // it simply is.
      const Real e[3] = {Delta + Exx, Delta + Eyy, Delta + Ezz};
      const Real q[3] = {Pxx, Pyy, Pzz};               // perturbation
      Real d[3];
      Real tr = 0, tre = 0, trq = 0;
      for (int a2 = 0; a2 < D; ++a2) {
        d[a2] = k[i2d(a2)];
        tr += d[a2];  tre += e[a2];  trq += q[a2];
      }
      const Real invD = Real(1) / Real(D);
      const Real tr_post = (Real(1) - omega_bulk) * tr + omega_bulk * tre + trq;
      for (int a2 = 0; a2 < D; ++a2)
        k[i2d(a2)] = (Real(1) - omega) * (d[a2] - tr * invD)
                   + omega * (e[a2] - tre * invD)
                   + (q[a2] - trq * invD) + tr_post * invD;
      k[i2s(0, 1)] = (Real(1) - omega) * k[i2s(0, 1)] + omega * Exy + Pxy;
      k[i2s(0, 2)] = (Real(1) - omega) * k[i2s(0, 2)] + omega * Exz + Pxz;
      k[i2s(1, 2)] = (Real(1) - omega) * k[i2s(1, 2)] + omega * Eyz + Pyz;
    }

    // order >= 3: s3 = s4 = s5 = s6 = 1, so straight to equilibrium plus source
    // -- which under Eq. (D5) is the shifted source and nothing else. Unrolled
    // over compile-time slot indices (see the banner).
    relax_high(k, s, ux, uy, uz, Moments{});

    Basis::template to_populations<true>(k, ub, f);
  }

  //----------------------------------------------------------------------------
  // Suboperator (3), Eqs. (D16)-(D17). The two outputs sum to f_i identically, so
  // this cannot lose mass or momentum however wrong beta is.
  //----------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void recolour(const Real f[L::Q], Real rho_r, Real rho_b, Real p, Index n,
                Real fr[L::Q], Real fb[L::Q]) const {
    const Real rho = rho_r + rho_b;
    const Real inv = (rho > Real(0)) ? Real(1) / rho : Real(0);
    const Real mix = beta * rho_r * rho_b * inv * inv;

    const Real g[3] = {Gx(n), Gy(n), Gz(n)};
    const Real gm2  = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    const Real gm   = (gm2 > Real(1e-24)) ? Kokkos::sqrt(gm2) : Real(0);

    for (int i = 0; i < L::Q; ++i) {
      Real cosine = Real(0);
      if (gm > Real(0)) {
        const Real cx = Real(cvel<L>(i, 0)), cy = Real(cvel<L>(i, 1)),
                   cz = Real(cvel<L>(i, 2));
        const Real cm2 = cx * cx + cy * cy + cz * cz;
        if (cm2 > Real(0))
          cosine = (cx * g[0] + cy * g[1] + cz * g[2])
                 / (Kokkos::sqrt(cm2) * gm);
      }
      const Real split = mix * cosine * eq_at_rest(i, rho, p);
      fr[i] = rho_r * inv * f[i] + split;
      fb[i] = rho_b * inv * f[i] - split;
    }
  }

  //----------------------------------------------------------------------------
  // Moment slots, located through the basis rather than hardcoded.
  //----------------------------------------------------------------------------
  static constexpr int i1(int a) {
    return Basis::index_of(a == 0, a == 1, a == 2);
  }
  static constexpr int i2d(int a) {
    return Basis::index_of(2 * (a == 0), 2 * (a == 1), 2 * (a == 2));
  }
  static constexpr int i2s(int a, int b) {
    return Basis::index_of((a == 0 || b == 0), (a == 1 || b == 1),
                           (a == 2 || b == 2));
  }
};

}  // namespace lbm
