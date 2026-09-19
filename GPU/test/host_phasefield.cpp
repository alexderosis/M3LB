//==============================================================================
//  Phase-field checks for the CUDA port, compiled and run WITHOUT a GPU.
//
//  THE SHARPEST CHECK ON THIS MODEL IS THAT THE INTERFACE DOES NOT MOVE OR
//  SPREAD. The conservative Allen-Cahn form exists so that the tanh profile of
//  width W is an exact equilibrium: for it, |grad phi| = (4/W) phi (1 - phi)
//  identically, so the diffusive and anti-diffusive fluxes cancel term for
//  term. Section 1 asserts that identity on the profile, section 4 runs it and
//  measures the width after 600 steps.
//
//  That is not a formality. The parent recorded a bug here that is loud in its
//  effect and silent in its cause: with a Guo prefactor and a 1/cs2 the source
//  comes out a factor M/cs2 too small, the anti-diffusion cannot balance the
//  diffusion, and W = 4 became 14.1 in 600 steps -- against the sqrt(4 M t) = 11
//  of pure diffusion. Nothing else in the model looks wrong while that happens.
//
//  Build:  c++ -std=c++17 -O2 -I../include host_phasefield.cpp -o host_phasefield
//==============================================================================
#include "lbm/phasefield.cuh"
#include "lbm/hostsim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbm;

// The lattice these checks are written against. Named once, so the
// central-moment blocks below can say D3Q27 and mean it.
using PL = D3Q7;

static int failures = 0;
static void check(bool ok, const char* what, double detail = 0.0) {
  if (ok) std::printf("  PASS  %s\n", what);
  else  { std::printf("  FAIL  %s   (%.3e)\n", what, detail); ++failures; }
}
static double worst(double a, double b) { return std::fabs(a) > std::fabs(b) ? a : b; }

//==============================================================================
//  THE PHI CONSERVATION BOUND, AND WHY IT IS 4 ULP/STEP RATHER THAN 1.
//
//  phi conservation is EXACT in the algorithm: sum_i S_i = 0, and streaming and
//  collision are both conservative. So any measured drift is round-off, and the
//  bound exists to separate round-off from a real leak. It is normalised by the
//  working precision's ulp so that ONE bound serves FP32 and FP64.
//
//  It used to be one ulp/step, and that was calibrated in FP32 on D3Q7. Built in
//  FP64 the D3Q27 blocks reported 4.45 (BGK) and 5.51 (CM) and failed. Measured
//  rather than adjusted, and two separate things were wrong.
//
//  FIRST, MOST OF THAT WAS THE INSTRUMENT. total_phase() summed Q*N slots
//  sequentially -- 62208 of them on D3Q27 -- and in FP64 the reduction's own
//  error dominated the drift it was measuring. Compensating it (hostsim.hpp)
//  took D3Q27 BGK from 4.451 to 0.351 ulp/step, i.e. seven eighths of the
//  "drift" was the sum. In FP32 the two agree to three digits, which is exactly
//  why this only surfaced in FP64: there the algorithm's round-off is a hundred
//  thousand times larger than the reduction's and hides it.
//
//  SECOND, THE DRIFT DOES NOT ACCUMULATE, so a per-step figure is an ENVELOPE
//  and not a rate. Total relative drift, D3Q27 BGK, FP64, over 600 / 1200 /
//  2400 / 9600 steps: -4.6e-14, -1.0e-13, -3.3e-13, -4.4e-13 -- sixty-four
//  times the steps for ten times the drift. In FP32 it stops dead: -5.0419e-05
//  at 600 and -8.8306e-05 at both 2400 AND 9600, to every digit, because the
//  increments fall below an ulp of the accumulated total. Dividing a saturating
//  drift by the window therefore makes the bound STRICTER the shorter the
//  window, which is backwards; the same run reads 18.3 ulp/step at 150 steps and
//  0.46 at 9600.
//
//  THE MEASURED ENVELOPE, over both 600-step windows, both precisions, D3Q7 BGK
//  and D3Q27 BGK and CM: worst 0.813 (FP32, D3Q27 CM) and 0.415 (FP64, D3Q27
//  BGK); over longer windows worst 0.866 (FP64, D3Q27 BGK, 1200->2400). Four
//  gives a factor 4.6 over the worst of those -- enough that this is not a
//  flaky test -- while still bounding phi to 4 x 600 x ulp = 5.3e-13 relative
//  over 600 FP64 steps.
//
//  WHAT IT DOES NOT COVER. It is not a bound on a leak smaller than round-off:
//  a defect losing under ~1e-15 of phi per step is invisible here at any
//  precision, and no test at this precision can see it. It says nothing about
//  conservation under geometry -- every case here is periodic, and a wall slot
//  holds populations in flight, which is why total_phase() sums populations and
//  not the phi field. And the envelope was measured on ONE configuration (the
//  flat interface, W = 4, mobility 0.05); a case with far more interface cells
//  per unit volume has not been measured. What it does catch is any leak worth
//  the name: a loss of 1e-12 of phi per step is 4545 ulp/step in FP64, three
//  orders over the bound.
//==============================================================================
static const double PHI_ULP_BOUND = 4.0;
static double ulp_of_real() { return (sizeof(Real) == 4) ? 1.19e-7 : 2.2e-16; }

// Zeroth moment of a 27-population array. The anti-diffusion source must have
// none, in any basis and at any shift velocity -- that is what makes phi
// conserved rather than merely nearly conserved.
static double ku0(const Real S[27]) {
  double s = 0;
  for (int i = 0; i < 27; ++i) s += double(S[i]);
  return s;
}

int main() {
  std::printf("Phase-field checks, host build, Real = %s\n",
              sizeof(Real) == 4 ? "float" : "double");
  const double eps = (sizeof(Real) == 4) ? 2e-5 : 1e-12;

  //===========================================================================
  std::printf("\n1. the tanh profile is an exact equilibrium\n\n");
  //===========================================================================
  {
    // For phi = 1/2 [1 + tanh(2x/W)],  dphi/dx = (2/W) phi (1 - phi) * 2 = ...
    // precisely |grad phi| = (4/W) phi (1 - phi). Check it analytically first,
    // then check that anti_diffusion() returns grad phi itself on that profile,
    // which is the cheapest possible test of the function.
    PhaseModel pm;  pm.width = Real(4);  pm.omega = Real(0.8);
    double wid = 0, wa = 0;
    for (double x = -6; x <= 6; x += 0.5) {
      const double ph = 0.5 * (1.0 + std::tanh(2.0 * x / 4.0));
      const double dp = (1.0 / 4.0) / std::cosh(2.0 * x / 4.0)
                                   / std::cosh(2.0 * x / 4.0);
      const double theta = (4.0 / 4.0) * ph * (1.0 - ph);
      wid = worst(wid, dp - theta);
      const Real G[3] = {Real(dp), Real(0), Real(0)};
      Real A[3];
      pm.anti_diffusion(Real(ph), G, A);
      wa = worst(wa, double(A[0]) - dp);
    }
    check(std::fabs(wid) < 1e-12, "|grad phi| == (4/W) phi (1-phi) on the profile", wid);
    check(std::fabs(wa) < 1e-6, "so A = theta n IS grad phi there", wa);
  }

  //===========================================================================
  std::printf("\n2. the source conserves phi, and the equilibrium carries it\n\n");
  //===========================================================================
  {
    PhaseModel pm;  pm.width = Real(4);  pm.omega = Real(0.7);
    // sum_i S_i = 0 with S_i = pref w_i (c_i . A). That is the "conservative"
    // in conservative Allen-Cahn: phi survives the source exactly.
    const Real A[3] = {Real(0.3), Real(-0.2), Real(0.1)};
    double s = 0;
    for (int i = 0; i < PL::Q; ++i) {
      const Real cA = Real(PL::cx(i)) * A[0]
                    + Real(PL::cy(i)) * A[1]
                    + Real(PL::cz(i)) * A[2];
      s += PL::w(i) * double(cA);
    }
    check(std::fabs(s) < eps, "sum_i S_i = 0, so phi is conserved exactly", s);

    // and its first moment is A itself, which is what the Chapman-Enskog
    // matching in the banner assumed.
    double m[3] = {0, 0, 0};
    for (int i = 0; i < PL::Q; ++i) {
      const Real cA = Real(PL::cx(i)) * A[0]
                    + Real(PL::cy(i)) * A[1]
                    + Real(PL::cz(i)) * A[2];
      const double si = PL::w(i) * double(cA) * PL::inv_cs2();
      m[0] += si * PL::cx(i);
      m[1] += si * PL::cy(i);
      m[2] += si * PL::cz(i);
    }
    double wm = 0;
    for (int a = 0; a < 3; ++a) wm = worst(wm, m[a] - double(A[a]));
    check(std::fabs(wm) < 1e-6, "sum_i c_i S_i / cs2 = A", wm);

    // g^eq: zeroth moment phi, first moment phi u.
    const Real ph = Real(0.37);
    const Real u[3] = {Real(0.03), Real(-0.02), Real(0.015)};
    double z = 0, p1[3] = {0, 0, 0};
    for (int i = 0; i < PL::Q; ++i) {
      const double e = PhaseModel::eq<PL>(i, ph, u[0], u[1], u[2]);
      z += e;
      p1[0] += e * PL::cx(i);
      p1[1] += e * PL::cy(i);
      p1[2] += e * PL::cz(i);
    }
    check(std::fabs(z - double(ph)) < 1e-6, "sum g^eq = phi", z - double(ph));
    double wp = 0;
    for (int a = 0; a < 3; ++a) wp = worst(wp, p1[a] - double(ph) * double(u[a]));
    check(std::fabs(wp) < 1e-6, "sum c g^eq = phi u", wp);

    // mobility round trip, on D3Q7's cs2 = 1/4 and not the fluid's 1/3.
    const Real om = PhaseModel::omega_from_mobility<PL>(Real(0.05));
    check(std::fabs(double(PhaseModel::mobility_from_omega<PL>(om)) - 0.05) < 1e-6,
          "M = cs2 (1/omega - 1/2) inverts, with cs2 = 1/4",
          double(PhaseModel::mobility_from_omega<PL>(om)) - 0.05);
  }

  //===========================================================================
  std::printf("\n3. the fluid equilibrium carries p~ and u, not rho u\n\n");
  //===========================================================================
  {
    const Real pt = Real(0.37);
    const Real u[3] = {Real(0.05), Real(-0.03), Real(0.02)};
    double z = 0, m1[3] = {0, 0, 0};
    for (int i = 0; i < 27; ++i) {
      const double e = FluidLattice::w(i) * (double(pt) + pf_phi_eq(i, u[0], u[1], u[2]));
      z += e;
      m1[0] += e * FluidLattice::cx(i);
      m1[1] += e * FluidLattice::cy(i);
      m1[2] += e * FluidLattice::cz(i);
    }
    check(std::fabs(z - double(pt)) < 1e-6, "sum f^eq = p~ exactly (Phi sums to zero)",
          z - double(pt));
    double wm = 0;
    for (int a = 0; a < 3; ++a) wm = worst(wm, m1[a] - double(u[a]));
    check(std::fabs(wm) < 1e-6, "sum c f^eq = u, NOT rho u", wm);

    // The interpolant is clamped, so rho stays between the two phases even when
    // phi overshoots -- a large enough undershoot would otherwise make rho
    // negative and 1/rho unbounded.
    MultiphaseModel fm;
    fm.rho_L = Real(1);  fm.rho_H = Real(100);
    fm.mu_L = Real(0.01);  fm.mu_H = Real(0.05);
    bool inrange = true;
    for (double p : {-0.7, -0.05, 0.0, 0.5, 1.0, 1.05, 1.9}) {
      const double r = double(fm.local(Real(p)).rho);
      if (r < 1.0 - 1e-9 || r > 100.0 + 1e-9) inrange = false;
    }
    check(inrange, "rho stays inside [rho_L, rho_H] for phi outside [0,1]");
    check(std::fabs(double(fm.drho_dphi()) - 99.0) < 1e-6, "drho/dphi = (rho_H-rho_L)/(phi_H-phi_L)",
          double(fm.drho_dphi()) - 99.0);
    check(!fm.constant_reference(), "local-rho normalisation is the default");
    fm.rho_0 = Real(1);
    check(fm.constant_reference(), "a positive rho_0 selects the constant one");
  }

  //===========================================================================
  std::printf("\n4. THE ONE THAT MATTERS: a flat interface must hold its width\n\n");
  //===========================================================================
  {
    const int nx = 64, ny = 6, nz = 6;
    const double W = 4.0;
    host::PhaseField pf(nx, ny, nz);
    pf.phase.width = Real(W);
    pf.phase.omega = Real(PhaseModel::omega_from_mobility<PL>(Real(0.05)));
    // No surface tension and no density ratio: the fluid must stay quiescent so
    // this measures the phase field alone.
    pf.fluid.rho_L = Real(1);  pf.fluid.rho_H = Real(1);
    pf.fluid.mu_L = Real(0.1); pf.fluid.mu_H = Real(0.1);
    pf.fluid.beta = Real(0);   pf.fluid.kappa = Real(0);

    const double xa = 20.0, xb = 44.0;
    pf.initialise_with([&](int x, int, int, Real& ph, Real& pt) {
      const double s = 0.5 * (std::tanh(2.0 * (x - xa) / W)
                            - std::tanh(2.0 * (x - xb) / W));
      ph = Real(s);  pt = Real(0);
    });

    auto width_now = [&]() {
      std::vector<Real> phi;
      pf.field_to_host(pf.phi_device(), phi);
      // W = 1 / max|dphi/dx| for the tanh profile, measured near the left face.
      double dmax = 0;
      for (int x = 10; x < 32; ++x) {
        const long a = node_id(x - 1, ny / 2, nz / 2, nx, ny);
        const long b = node_id(x + 1, ny / 2, nz / 2, nx, ny);
        dmax = std::max(dmax, std::fabs(0.5 * (double(phi[b]) - double(phi[a]))));
      }
      return 1.0 / dmax;
    };

    const double w0 = width_now();
    const double m0 = pf.total_phase();
    for (int t = 0; t < 600; ++t) pf.step();
    const double w1 = width_now();
    const double m1 = pf.total_phase();
    for (int t = 0; t < 600; ++t) pf.step();
    const double m2 = pf.total_phase();

    std::printf("        (W seeded %.2f, measured %.2f -> %.2f after 600 steps)\n",
                W, w0, w1);
    check(std::fabs(w1 - w0) / w0 < 0.05,
          "the interface neither spreads nor sharpens", (w1 - w0) / w0);
    // Pure diffusion at this mobility would reach sqrt(4 M t) = 11, so this is
    // a wide margin against the failure mode, not a tight one against noise.
    check(w1 < 6.0, "and is nowhere near the sqrt(4 M t) = 11 of pure diffusion", w1);
    // See PHI_ULP_BOUND above for the whole argument. Two windows rather than
    // one so that a leak, which grows, is separated from round-off, which
    // saturates. Measured in FP32 over 600 / 1200 / 2400 / 9600 steps the total
    // drift runs -3.4e-05, -6.7e-05, -1.0e-04, -1.4e-04: it grows while the
    // seeded profile relaxes onto the discrete equilibrium and then nearly
    // stops. In FP64 it stays at 1e-15 to 1e-13 throughout. Neither is a leak;
    // the FP32 figure is the price of the precision and is worth knowing before
    // a long run.
    const double d1 = (m1 - m0) / m0, d2 = (m2 - m1) / m0;
    const double ulp = ulp_of_real();
    check(std::fabs(d1) / 600.0 < PHI_ULP_BOUND * ulp,
          "phi conserved to within the round-off envelope",
          std::fabs(d1) / 600.0 / ulp);
    check(std::fabs(d2) / 600.0 < PHI_ULP_BOUND * ulp,
          "  ... and still so over the next 600",
          std::fabs(d2) / 600.0 / ulp);
    std::printf("        (total phi %.10f -> %.10f -> %.10f;"
                " %.2f then %.2f ulp/step)\n", m0, m1, m2,
                std::fabs(d1) / 600.0 / ulp, std::fabs(d2) / 600.0 / ulp);

    double umax = 0;
    std::vector<Real> ux, uy, uz;
    pf.field_to_host(pf.ux_device(), ux);
    pf.field_to_host(pf.uy_device(), uy);
    pf.field_to_host(pf.uz_device(), uz);
    for (long n = 0; n < pf.nodes(); ++n)
      umax = std::max(umax, std::sqrt(double(ux[n])*double(ux[n])
                                    + double(uy[n])*double(uy[n])
                                    + double(uz[n])*double(uz[n])));
    check(umax < 1e-6, "and the fluid stayed quiescent", umax);
  }

  //===========================================================================
  std::printf("\n5. a droplet at a density ratio stays a droplet\n\n");
  //===========================================================================
  {
    const int n = 32;
    const double R = 8.0, W = 4.0, sigma = 1e-3, gamma = 10.0;
    host::PhaseField pf(n, n, n);
    pf.phase.width = Real(W);
    pf.phase.omega = Real(PhaseModel::omega_from_mobility<PL>(Real(0.05)));
    pf.fluid.rho_L = Real(1);  pf.fluid.rho_H = Real(gamma);
    pf.fluid.mu_L = Real(0.01); pf.fluid.mu_H = Real(0.1);
    pf.fluid.beta  = Real(MultiphaseModel::beta_from_sigma(Real(sigma), Real(W)));
    pf.fluid.kappa = Real(MultiphaseModel::kappa_from_sigma(Real(sigma), Real(W)));
    pf.enable_viscous_force(true);

    const double c = 0.5 * n;
    pf.initialise_with([&](int x, int y, int z, Real& ph, Real& pt) {
      const double dx = x - c, dy = y - c, dz = z - c;
      const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
      ph = Real(0.5 * (1.0 - std::tanh(2.0 * (r - R) / W)));
      pt = Real(0);              // the gauge; see the banner
    });

    const double m0 = pf.total_phase();
    bool finite = true;
    for (int t = 0; t < 400 && finite; ++t) {
      pf.step();
      if (t % 100 == 99) {
        std::vector<Real> phi;
        pf.field_to_host(pf.phi_device(), phi);
        for (long k = 0; k < pf.nodes(); ++k)
          if (!std::isfinite(double(phi[k]))) { finite = false; break; }
      }
    }
    const double m1 = pf.total_phase();
    std::vector<Real> phi, ux, uy, uz;
    pf.field_to_host(pf.phi_device(), phi);
    pf.field_to_host(pf.ux_device(), ux);
    pf.field_to_host(pf.uy_device(), uy);
    pf.field_to_host(pf.uz_device(), uz);
    const double pc = double(phi[node_id(n/2, n/2, n/2, n, n)]);
    const double pe = double(phi[node_id(0, 0, 0, n, n)]);
    double umax = 0;
    for (long k = 0; k < pf.nodes(); ++k)
      umax = std::max(umax, std::sqrt(double(ux[k])*double(ux[k])
                                    + double(uy[k])*double(uy[k])
                                    + double(uz[k])*double(uz[k])));

    check(finite, "400 steps at a density ratio of 10 stay finite");
    const double tol = (sizeof(Real) == 4) ? 2e-5 : 1e-12;
    check(std::fabs(m1 - m0) < tol * std::fabs(m0),
          "phi conserved through the coupled step", (m1 - m0) / m0);
    check(pc > 0.9, "heavy phase at the centre", pc);
    check(pe < 0.1, "light phase in the corner", pe);
    std::printf("        (phi centre %.4f, corner %.4f)\n", pc, pe);
    check(umax < 0.05, "spurious current stays small", umax);
    std::printf("        (max |u| = %.4e)\n", umax);
  }

  //===========================================================================
  std::printf("\n6. CENTRAL MOMENTS: the phase field on D3Q27\n\n");
  //===========================================================================
  {
    const Real om = Real(1.4);
    const Real phi0 = Real(0.37);
    const Real A[3] = {Real(0.021), Real(-0.013), Real(0.007)};
    constexpr double cs2 = 1.0 / 3.0;

    // ---- the u = 0 equilibrium is w_i phi, and it is a fixed point ----------
    //
    // Not an assumption: inverse-transforming k^eq = (phi, 0, ..., 0) at u = 0
    // gives the 1D triple (1/6, 2/3, 1/6) phi on each axis, whose product IS
    // the D3Q27 weight set. So the seeder's w_i phi is the CM equilibrium too,
    // and the same initial condition serves both operators.
    {
      Real h[27];
      for (int i = 0; i < 27; ++i) h[i] = D3Q27::w(i) * phi0;
      const Real u[3] = {Real(0), Real(0), Real(0)};
      const Real Z[3] = {Real(0), Real(0), Real(0)};
      Real g[27];
      for (int i = 0; i < 27; ++i) g[i] = h[i];
      collide_phase_cm(g, phi0, u, Z, om);
      double e = 0;
      for (int i = 0; i < 27; ++i) e = worst(e, double(g[i]) - double(h[i]));
      check(std::fabs(e) < 1e-6, "k^eq = (phi, 0, ..., 0): w_i phi is a fixed point", e);
    }

    // ---- phi is conserved at any u and any source --------------------------
    {
      const Real u[3] = {Real(0.05), Real(-0.03), Real(0.02)};
      Real h[27];
      for (int i = 0; i < 27; ++i)
        h[i] = D3Q27::w(i) * phi0 * Real(1.0 + 0.05 * std::sin(double(i)));
      double s0 = 0;
      for (int i = 0; i < 27; ++i) s0 += double(h[i]);
      collide_phase_cm(h, Real(s0), u, A, om);
      double s1 = 0;
      for (int i = 0; i < 27; ++i) s1 += double(h[i]);
      check(std::fabs(s1 - s0) < 1e-6, "phi conserved by the CM collision", s1 - s0);
    }

    // ---- the source delivers (1 - omega/2) cs2 A into the first moment -----
    //
    // THE cs2 IS THE POINT. Writing A without it makes the anti-diffusion three
    // times too weak on a cs2 = 1/3 lattice, which neither blows up nor looks
    // wrong -- the interface simply spreads.
    {
      const Real u[3] = {Real(0.04), Real(-0.02), Real(0.01)};
      Real h[27];
      for (int i = 0; i < 27; ++i)
        h[i] = D3Q27::w(i) * phi0 * Real(1.0 + 0.05 * std::cos(double(i)));
      Real kpre[27];
      to_moments(h, u, kpre);
      const double k1pre[3] = {double(kpre[mi(1,0,0)]), double(kpre[mi(0,1,0)]),
                               double(kpre[mi(0,0,1)])};
      double ph = 0;
      for (int i = 0; i < 27; ++i) ph += double(h[i]);
      collide_phase_cm(h, Real(ph), u, A, om);
      Real kpost[27];
      to_moments(h, u, kpost);
      const double keep = 1.0 - double(om);
      const double pref = (1.0 - 0.5 * double(om)) * cs2;
      double e = 0;
      for (int a = 0; a < 3; ++a) {
        const int slot = (a == 0) ? mi(1,0,0) : (a == 1) ? mi(0,1,0) : mi(0,0,1);
        e = worst(e, double(kpost[slot]) - (keep * k1pre[a] + pref * double(A[a])));
      }
      check(std::fabs(e) < 1e-6, "k*_1 = (1-omega) k_1 + (1-omega/2) cs2 A", e);
    }

    // ---- A PUBLISHED MOMENT LIST BELONGS TO A BASIS, checked BOTH ways -----
    //
    // The paper's Eq. (61) lists nine nonzero source entries -- three at first
    // order and six at third -- in the MONOMIAL central moments. core.cuh's
    // transform is SHIFTED, phi_2 = C^2 - cs2, and in that basis the six
    // third-order ones are identically zero: the (a,a,b) slot contributes
    // cs4 A_b - cs2 * cs2 A_b = 0.
    //
    // The trap is live in both directions. Adding the six here double-counts
    // them; deleting them from a monomial implementation loses them; NEITHER
    // CRASHES. So both halves are pinned.
    {
      const Real sp = Real(1) - Real(0.5) * om;
      Real S[27];
      for (int i = 0; i < 27; ++i) {
        const Real cA = Real(D3Q27::cx(i)) * A[0] + Real(D3Q27::cy(i)) * A[1]
                      + Real(D3Q27::cz(i)) * A[2];
        S[i] = sp * D3Q27::w(i) * cA;
      }

      // MONOMIAL, at rest: how many slots does the same source occupy?
      int nz_mono = 0;
      double worst_first = 0, worst_third = 0;
      for (int n = 0; n < 27; ++n) {
        const int px = p_of(n), qy = q_of(n), rz = r_of(n);
        double M = 0;
        for (int i = 0; i < 27; ++i) {
          double t = double(S[i]);
          for (int k = 0; k < px; ++k) t *= double(D3Q27::cx(i));
          for (int k = 0; k < qy; ++k) t *= double(D3Q27::cy(i));
          for (int k = 0; k < rz; ++k) t *= double(D3Q27::cz(i));
          M += t;
        }
        if (std::fabs(M) > 1e-9) ++nz_mono;
        if (px + qy + rz == 1) {
          const int a = (px == 1) ? 0 : (qy == 1) ? 1 : 2;
          worst_first = worst(worst_first, M - double(sp) * cs2 * double(A[a]));
        }
        if (px == 2 && qy == 1 && rz == 0)
          worst_third = worst(worst_third, M - double(sp) * cs2 * cs2 * double(A[1]));
      }
      check(nz_mono == 12,
            "monomial: the source occupies 12 slots (their 9 + 3 D3Q27 fifth-order)",
            double(nz_mono));
      check(std::fabs(worst_first) < 1e-9, "monomial: R_a = (1-omega/2) cs2 A_a",
            worst_first);
      check(std::fabs(worst_third) < 1e-9, "monomial: R_(2,1,0) = (1-omega/2) cs4 A_y",
            worst_third);

      // SHIFTED, at rest: only the three first-order slots survive.
      const Real z[3] = {Real(0), Real(0), Real(0)};
      Real k[27];
      to_moments(S, z, k);
      double leak = 0;
      int nz_shift = 0;
      for (int n = 0; n < 27; ++n) {
        const bool first = (n == mi(1,0,0) || n == mi(0,1,0) || n == mi(0,0,1));
        if (std::fabs(double(k[n])) > 1e-9) ++nz_shift;
        if (!first) leak = worst(leak, double(k[n]));
      }
      check(nz_shift == 3, "shifted: the SAME source occupies 3 slots, not 12",
            double(nz_shift));
      check(std::fabs(leak) < 1e-9,
            "shifted: every slot outside the first three cancels exactly", leak);

      // AND THE SPARSITY IS A REST-FRAME PROPERTY -- measured as a MAGNITUDE,
      // not as a count of nonzeros.
      //
      // Counting is the wrong statistic and it took a wrong answer to see why:
      // 26 of the 27 shifted slots are nonzero at a general u, but the smallest
      // of them is 4e-7 of the largest, which is below the FP32 floor, so the
      // count reads 25 or 27 depending on the threshold and tells you nothing
      // about how much has been dropped. The ratio
      //
      //     ||k outside the first three|| / ||k in the first three||
      //
      // has no threshold in it, is the same in both precisions, and IS the size
      // of the approximation. Measured in FP64, exactly linear in |u|:
      //
      //     at rest    5.1e-17        |u| ~ 0.03   6.8e-02
      //     |u| ~ 0.01 2.3e-02        |u| ~ 0.06   1.4e-01
      //                               |u| ~ 0.10   2.3e-01
      //
      // So at a lattice velocity of 0.1 the part of the source this truncation
      // discards is 23% of the part it keeps. Their drivers add the same
      // u-independent source and drop the same 23%: a shared approximation,
      // named as one rather than presented as an identity.
      auto tail_ratio = [&](const Real uu[3]) {
        Real kk[27];
        to_moments(S, uu, kk);
        double in = 0, out = 0;
        for (int n = 0; n < 27; ++n) {
          const double v = double(kk[n]) * double(kk[n]);
          if (p_of(n) + q_of(n) + r_of(n) == 1) in += v; else out += v;
        }
        return std::sqrt(out / in);
      };
      const Real ua[3] = {Real(0.03), Real(-0.021), Real(0.015)};
      const Real ub[3] = {Real(0.06), Real(-0.042), Real(0.030)};
      const double r0 = tail_ratio(z), r1 = tail_ratio(ua), r2 = tail_ratio(ub);
      check(r0 < 1e-6, "at rest the source is EXACTLY three slots", r0);
      check(r1 > 0.03 && r1 < 0.12,
            "at |u| ~ 0.04 the truncated part is a few per cent of the kept part", r1);
      check(std::fabs(r2 / r1 - 2.0) < 0.05,
            "and it grows LINEARLY in |u|, so it is O(|u| A) as claimed", r2 / r1);
      check(std::fabs(double(ku0(S))) < 1e-9, "  ... but sum_i S_i is still zero",
            double(ku0(S)));
      std::printf("        (tail/kept: %.1e at rest, %.3f at |u|~0.04, %.3f at ~0.08)\n",
                  r0, r1, r2);
    }

    // ---- BGK and CM agree where they must: the conserved moments -----------
    {
      const Real u[3] = {Real(0.03), Real(-0.02), Real(0.01)};
      Real hb[27], hc[27];
      for (int i = 0; i < 27; ++i) {
        hb[i] = D3Q27::w(i) * phi0 * Real(1.0 + 0.03 * std::sin(2.0 * double(i)));
        hc[i] = hb[i];
      }
      double ph = 0;
      for (int i = 0; i < 27; ++i) ph += double(hb[i]);

      const Real pref = Real(1) - Real(0.5) * om;
      for (int i = 0; i < 27; ++i) {
        const Real cA = Real(D3Q27::cx(i)) * A[0] + Real(D3Q27::cy(i)) * A[1]
                      + Real(D3Q27::cz(i)) * A[2];
        hb[i] += om * (PhaseModel::eq<D3Q27>(i, Real(ph), u[0], u[1], u[2]) - hb[i])
               + pref * D3Q27::w(i) * cA;
      }
      collide_phase_cm(hc, Real(ph), u, A, om);

      double d0 = 0, d1 = 0;
      for (int i = 0; i < 27; ++i) d0 += double(hb[i]) - double(hc[i]);
      for (int a = 0; a < 3; ++a) {
        double mb = 0, mc = 0;
        for (int i = 0; i < 27; ++i) {
          const double c = (a == 0) ? D3Q27::cx(i)
                         : (a == 1) ? D3Q27::cy(i) : D3Q27::cz(i);
          mb += double(hb[i]) * c;  mc += double(hc[i]) * c;
        }
        d1 = worst(d1, mb - mc);
      }
      check(std::fabs(d0) < 1e-6, "BGK and CM agree in the zeroth moment", d0);
      check(std::fabs(d1) < 1e-6, "BGK and CM agree in the first moment", d1);
      std::printf("        (they differ above first order -- that IS the operator)\n");
    }
  }

  //===========================================================================
  std::printf("\n7. CENTRAL MOMENTS: the multiphase fluid\n\n");
  //===========================================================================
  {
    const Real om = Real(1.5), omb = Real(1.0);
    const Real pt = Real(0.013);
    const Real rho = Real(3.7);
    const Real u[3] = {Real(0.05), Real(-0.02), Real(0.03)};

    // The equilibrium of THIS operator is the inverse transform of k_eq, not
    // the truncated Eq. (10). Build it that way and it must be a fixed point.
    Real Aw[3][3];
    mp_weight_factors(u, Aw);
    Real keq[27], feqm[27];
    mp_all_eq(pt, Aw, keq, std::make_integer_sequence<int, 27>{});
    to_populations(keq, u, feqm);

    {
      Real f[27];
      for (int i = 0; i < 27; ++i) f[i] = feqm[i];
      const Real Z[3] = {Real(0), Real(0), Real(0)};
      collide_multiphase_cm(f, pt, u, Z, rho, om, omb);
      double e = 0;
      for (int i = 0; i < 27; ++i) e = worst(e, double(f[i]) - double(feqm[i]));
      check(std::fabs(e) < 1e-6, "the equilibrium is a fixed point", e);
      double z = 0;
      for (int i = 0; i < 27; ++i) z += double(feqm[i]);
      check(std::fabs(z - double(pt)) < 1e-6, "and its zeroth moment is p~ exactly",
            z - double(pt));
    }

    // p~ conserved, and the momentum picks up EXACTLY F/(2 rho).
    //
    // In this model u = sum_i c_i f_i + F/(2 rho), so a post-collision raw first
    // moment of u + F/(2 rho) is what makes the NEXT step read u + F/rho -- the
    // right increment for du/dt = ... + F/rho. Getting the factor wrong is a
    // viscosity-scale error that reads like a bad boundary.
    {
      const Real F[3] = {Real(2e-4), Real(-1e-4), Real(3e-4)};
      Real f[27];
      for (int i = 0; i < 27; ++i)
        f[i] = feqm[i] + Real(1e-3) * D3Q27::w(i) * Real(std::sin(3.0 * double(i)));
      double s0 = 0;
      for (int i = 0; i < 27; ++i) s0 += double(f[i]);
      collide_multiphase_cm(f, Real(s0), u, F, rho, om, omb);
      double s1 = 0, M[3] = {0, 0, 0};
      for (int i = 0; i < 27; ++i) {
        s1 += double(f[i]);
        M[0] += double(f[i]) * D3Q27::cx(i);
        M[1] += double(f[i]) * D3Q27::cy(i);
        M[2] += double(f[i]) * D3Q27::cz(i);
      }
      check(std::fabs(s1 - s0) < 1e-6, "p~ conserved", s1 - s0);
      double e = 0;
      for (int a = 0; a < 3; ++a)
        e = worst(e, M[a] - (double(u[a]) + 0.5 * double(F[a]) / double(rho)));
      check(std::fabs(e) < 1e-6, "sum_i c_i f_i = u + F/(2 rho) after collision", e);
    }
  }

  //===========================================================================
  std::printf("\n8. A FLAT INTERFACE ON D3Q27, BGK AND CENTRAL MOMENTS\n\n");
  //===========================================================================
  //
  // Section 4 is this case on D3Q7 BGK. Running it again on the product lattice
  // with each operator is the integration test for everything above: the
  // templated lattice, the 27-slot phase storage, the CM collision and the
  // dispatch. All three must hold the width, and each must conserve phi to
  // round-off.
  {
    const int nx = 64, ny = 6, nz = 6;
    const double W = 4.0;
    const double xa = 20.0, xb = 44.0;

    auto run = [&](PhaseOp op, const char* name) {
      host::PhaseField<D3Q27> pf(nx, ny, nz);
      pf.phase.width = Real(W);
      pf.set_mobility(Real(0.05));      // cs2 = 1/3 here, NOT the D3Q7 1/4
      pf.set_phase_op(op);
      pf.fluid.rho_L = Real(1);  pf.fluid.rho_H = Real(1);
      pf.fluid.mu_L = Real(0.1); pf.fluid.mu_H = Real(0.1);
      pf.fluid.beta = Real(0);   pf.fluid.kappa = Real(0);
      pf.initialise_with([&](int x, int, int, Real& ph, Real& p_t) {
        ph = Real(0.5 * (std::tanh(2.0 * (x - xa) / W)
                       - std::tanh(2.0 * (x - xb) / W)));
        p_t = Real(0);
      });

      auto width_now = [&]() {
        std::vector<Real> phi;
        pf.field_to_host(pf.phi_device(), phi);
        double dmax = 0;
        for (int x = 10; x < 32; ++x) {
          const long a = node_id(x - 1, ny / 2, nz / 2, nx, ny);
          const long b = node_id(x + 1, ny / 2, nz / 2, nx, ny);
          dmax = std::max(dmax, std::fabs(0.5 * (double(phi[b]) - double(phi[a]))));
        }
        return 1.0 / dmax;
      };

      const double w0 = width_now(), m0 = pf.total_phase();
      for (int t = 0; t < 600; ++t) pf.step();
      const double w1 = width_now(), m1 = pf.total_phase();
      // A SECOND WINDOW, as section 4 has. One window cannot tell a leak from
      // round-off: both look like a number. A leak grows across windows and
      // round-off need not, and this block was the one measuring only the first.
      for (int t = 0; t < 600; ++t) pf.step();
      const double m2 = pf.total_phase();

      char buf[160];
      std::snprintf(buf, sizeof buf, "D3Q27 %s: the interface holds its width", name);
      check(std::fabs(w1 - w0) / w0 < 0.05, buf, (w1 - w0) / w0);
      const double ulp = ulp_of_real();
      const double e1 = std::fabs((m1 - m0) / m0) / 600.0 / ulp;
      const double e2 = std::fabs((m2 - m1) / m0) / 600.0 / ulp;
      std::snprintf(buf, sizeof buf,
                    "D3Q27 %s: phi conserved to within the round-off envelope", name);
      check(e1 < PHI_ULP_BOUND, buf, e1);
      std::snprintf(buf, sizeof buf,
                    "D3Q27 %s:   ... and still so over the next 600", name);
      check(e2 < PHI_ULP_BOUND, buf, e2);
      std::printf("        (%s: W %.2f -> %.2f;  %.2f then %.2f ulp/step)\n",
                  name, w0, w1, e1, e2);
      return w1;
    };

    const double wb = run(PhaseOp::BGK, "BGK");
    const double wc = run(PhaseOp::CentralMoments, "CM ");
    // Two different operators need not agree exactly; they must agree on the
    // physics, which here is the equilibrium profile width.
    check(std::fabs(wb - wc) / wb < 0.05,
          "and the two operators agree on it to 5%", (wb - wc) / wb);
  }

  //===========================================================================
  std::printf("\n9. THE ZERO-FLUX WALL, WHICH NOTHING HERE USED TO TOUCH\n\n");
  //===========================================================================
  //
  // Every other block in this file is periodic in all three directions and says
  // so, which left PhaseWall and PhaseExcluded executed by NO test in GPU/ --
  // host or device. That is exactly the gap the parent records: its own
  // validation/phase_wall.cpp banner says "Every other phase-field check in
  // this tree is periodic in every direction ... so PhaseWall was exercised by
  // no test at all. This is the gap that let the bug below live." The bug it
  // means was found only by diffing phi node-by-node against THIS port, so the
  // port inheriting the same blind spot is worth closing.
  //
  // THE CASE, ported from the parent so the two are comparable: a closed box
  // holding ONE phase uniformly at rest. phi = 1 everywhere, PhaseWall on the
  // two y planes, no interface anywhere. A uniform field is an EXACT stationary
  // solution of the conservative Allen-Cahn equation -- grad phi is zero, so the
  // diffusive and anti-diffusive fluxes both vanish identically -- and a
  // zero-flux wall cannot change that. So phi must stay 1 at every bulk node,
  // forever, to round-off. The tolerance below is round-off, not an accuracy
  // claim.
  //
  // WALL NODES ARE EXCLUDED FROM THE CHECK, not asserted about. A PhaseWall node
  // never collides, so its slots hold whatever a neighbour emitted toward it one
  // step ago; pf_field_node leaves its phi at the initialised value rather than
  // giving it a meaningless one. What phi() reports there is not a physical
  // value, the same trap the parent's ScalarSolver::temperature() carries.
  //
  // THE NORMAL-VELOCITY CASE IS PRINTED AND NOT ASSERTED. A uniform velocity
  // normal to an impermeable wall is not a well-posed input -- the wall blocks a
  // flux the interior keeps supplying -- so phi genuinely has nowhere to go. It
  // is NOT a ringing, which an early draft of the parent's banner called it:
  // measured, it is a monotone, antisymmetric, mass-conserving pile-up, phi
  // draining from the upstream wall into the downstream one, linear in v0
  // (error/v0 = -40.3, -40.0, -39.6, -38.6 over v0 = 0.0005 .. 0.004) and
  // saturating in time as diffusion comes into balance. Both codebases produce
  // it identically, to every printed digit.
  {
    const int nx = 8, ny = 32, nz = 8;
    const long N = long(nx) * ny * nz;
    const std::size_t NN = std::size_t(N);

    // THE PASSES ARE DRIVEN DIRECTLY, not through host::PhaseField, and that is
    // forced rather than chosen. The parent's PhaseFieldSolver is a standalone
    // solver whose velocity is an INPUT, so its phase_wall.cpp can prescribe a
    // uniform u. This port fuses the fluid and the phase field into one step(),
    // and the fluid pass writes the velocity array before the phase pass reads
    // it -- so a hand-written velocity is erased in the same step. Driving
    // pf_field_node / pf_derivatives_node / pf_phase_node here is the only way
    // to reach the port's own arithmetic on the parent's own input, and it is
    // the same thing sections 6 and 7 do with the free functions.
    std::vector<Real> h(std::size_t(27 * N), Real(0));
    std::vector<Real> fld(std::size_t(5 * N), Real(0));      // phi, gx, gy, gz, lap
    const std::uint8_t bulk = PhaseBulk, wall = PhaseWall;
    std::vector<std::uint8_t> pfl(NN, bulk);
    for (long n = 0; n < N; ++n) {
      int x, y, z;
      coords(n, nx, ny, x, y, z);
      if (y == 0 || y == ny - 1) pfl[std::size_t(n)] = wall;
    }

    auto run = [&](PhaseOp op, const char* name, int steps, double omega,
                   double u0, double v0) {
      std::fill(h.begin(), h.end(), Real(0));
      std::fill(fld.begin(), fld.end(), Real(0));
      std::vector<Real> ux(NN, Real(u0)), uy(NN, Real(v0)), uz(NN, Real(0));
      for (long n = 0; n < N; ++n) {
        int x, y, z;
        coords(n, nx, ny, x, y, z);
        Real e0[27];
        for (int i = 0; i < 27; ++i) e0[i] = D3Q27::w(i) * Real(1);   // eq at phi=1, u=0
        init_scatter<0, D3Q27>(h.data(), N, x, y, z, nx, ny, nz, e0);
        fld[std::size_t(n)] = Real(1);
      }
      PfParamsT<D3Q27> p;
      p.h = h.data();
      p.phi = &fld[0];
      p.gx = &fld[std::size_t(N)];      p.gy = &fld[std::size_t(2 * N)];
      p.gz = &fld[std::size_t(3 * N)];  p.lap = &fld[std::size_t(4 * N)];
      p.ux = ux.data();  p.uy = uy.data();  p.uz = uz.data();
      p.pflags = pfl.data();
      p.nx = nx;  p.ny = ny;  p.nz = nz;
      p.pm.width = Real(4);
      p.pm.omega = Real(omega);
      p.pop = op;

      for (long i = 0; i < N; ++i) pf_derivatives_node<true>(p, i);
      for (int t = 0; t < steps; ++t) {
        if (t % 2 == 0) {
          for (long i = 0; i < N; ++i) pf_field_node<0, true>(p, N, i);
          for (long i = 0; i < N; ++i) pf_derivatives_node<true>(p, i);
          for (long i = 0; i < N; ++i) pf_phase_node<0, true>(p, N, i);
        } else {
          for (long i = 0; i < N; ++i) pf_field_node<1, true>(p, N, i);
          for (long i = 0; i < N; ++i) pf_derivatives_node<true>(p, i);
          for (long i = 0; i < N; ++i) pf_phase_node<1, true>(p, N, i);
        }
      }
      if (steps % 2 == 0) for (long i = 0; i < N; ++i) pf_field_node<0, true>(p, N, i);
      else                for (long i = 0; i < N; ++i) pf_field_node<1, true>(p, N, i);

      double worst = 0;
      int worst_y = 0;
      for (long n = 0; n < N; ++n) {
        int x, y, z;
        coords(n, nx, ny, x, y, z);
        if (y == 0 || y == ny - 1) continue;              // bulk only
        const double e = std::fabs(double(fld[std::size_t(n)]) - 1.0);
        if (e > worst) { worst = e; worst_y = y; }
      }
      std::printf("        %-26s max|phi - 1| = %.3e  at y = %d\n",
                  name, worst, worst_y);
      return worst;
    };

    // A uniform field is EXACT for this scheme, so this is round-off. FP32
    // carries ~1e-7 per operation and 1000 steps of it, hence the split bar --
    // the same one the parent's phase_wall.cpp uses.
    const double tol = (sizeof(Real) == 4) ? 1e-4 : 1e-10;
    const double a = run(PhaseOp::BGK,            "BGK,  omega 1.0",  100, 1.0, 0.0, 0.0);
    const double b = run(PhaseOp::CentralMoments, "CM,   omega 1.0",  100, 1.0, 0.0, 0.0);
    const double c = run(PhaseOp::CentralMoments, "CM,   omega 1.8",  100, 1.8, 0.0, 0.0);
    const double e = run(PhaseOp::CentralMoments, "CM,   1000 steps", 1000, 1.0, 0.0, 0.0);
    const double f = run(PhaseOp::CentralMoments, "CM,   u tangential 0.02",
                         100, 1.0, 0.02, 0.0);
    check(a < tol, "D3Q27 BGK: a uniform phase holds against a zero-flux wall", a);
    check(b < tol, "D3Q27 CM:  the same", b);
    check(c < tol, "D3Q27 CM:  and at omega = 1.8", c);
    check(e < tol, "D3Q27 CM:  and over 1000 steps", e);
    check(f < tol, "D3Q27 CM:  and under a TANGENTIAL uniform flow", f);

    // AND THE CROSS-CODE NUMBER, which is the point of having two codebases.
    // The parent's validation/phase_wall.cpp prints 7.915e-02 on this exact
    // input in FP64. It is NOT asserted as physics -- the input is not well
    // posed -- but the two implementations agreeing on it IS assertable, and it
    // is the only quantitative tie between the two wall implementations.
    const double g = run(PhaseOp::CentralMoments, "CM,   u NORMAL 0.002", 100, 1.0,
                         0.0, 0.002);
    if (sizeof(Real) == 8) {
      const double parent = 7.915e-02;
      check(std::fabs(g - parent) / parent < 1e-3,
            "and the NORMAL-velocity pile-up matches the parent's 7.915e-02",
            (g - parent) / parent);
    }
    std::printf("        (the last row is a pile-up, not a ringing: phi drains from the\n"
                "         upstream wall into the downstream one, linear in v0 and\n"
                "         saturating in time, with the population sum conserved.)\n");
  }

  //===========================================================================
  std::printf("\n10. F_nu IS BUILT FROM u(t), NOT u(t-1)\n\n");
  //===========================================================================
  //
  // A GUARD ON THE PASS ORDER, because the order is the whole content and
  // nothing else in this file can see it. Until 2026-09-19 the macroscopic pass
  // ran only for a pre-fluid hook and ran AFTER the viscous pass, so F_nu was
  // built from the previous step's velocity. Moving it ahead changed a droplet
  // in a shear layer by 0.02 % to 0.12 %, growing with the density ratio -- real
  // but far too small to notice by eye, which is why it wants an assertion
  // rather than a printed number.
  //
  // THE TEST IS DIFFERENTIAL, and it has to be. Start from rest and step ONCE.
  // With the correct order the macroscopic pass runs first, so the velocity
  // array holds u(t) -- which is still zero at t = 0, and F_nu with it. That
  // tells us nothing. So instead: write a velocity field by hand, zero F_nu,
  // and run the single pass pair. If the macro pass runs FIRST it overwrites the
  // hand-written velocity with u from the populations (zero), and F_nu comes out
  // zero; if it runs SECOND, F_nu is built from the hand-written field and comes
  // out non-zero. The two orders therefore differ by everything, not by 0.1 %.
  {
    const int n = 16;
    const long N = long(n) * n * n;
    std::vector<Real> f(std::size_t(27 * N), Real(0));
    std::vector<Real> h(std::size_t(27 * N), Real(0));
    std::vector<Real> fld(std::size_t(15 * N), Real(0));

    const double R = 5.0, W = 4.0, c = 0.5 * n;
    for (long id = 0; id < N; ++id) {
      int x, y, z;
      coords(id, n, n, x, y, z);
      const double dx = x - c, dy = y - c, dz = z - c;
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      const Real ph = Real(0.5 * (1.0 - std::tanh(2.0 * (r - R) / W)));
      Real g0[27];
      for (int i = 0; i < 27; ++i) g0[i] = Real(0);
      init_scatter<0, FluidLattice>(f.data(), N, x, y, z, n, n, n, g0);
      Real e0[27];
      for (int i = 0; i < 27; ++i) e0[i] = FluidLattice::w(i) * ph;
      init_scatter<0, D3Q27>(h.data(), N, x, y, z, n, n, n, e0);
      fld[std::size_t(id)] = ph;
    }

    PfParamsT<D3Q27> p;
    p.f = f.data();  p.h = h.data();
    p.phi = &fld[0];
    p.gx = &fld[std::size_t(N)];      p.gy = &fld[std::size_t(2 * N)];
    p.gz = &fld[std::size_t(3 * N)];  p.lap = &fld[std::size_t(4 * N)];
    p.ux = &fld[std::size_t(5 * N)];  p.uy = &fld[std::size_t(6 * N)];
    p.uz = &fld[std::size_t(7 * N)];  p.pt = &fld[std::size_t(8 * N)];
    p.vx = &fld[std::size_t(9 * N)];  p.vy = &fld[std::size_t(10 * N)];
    p.vz = &fld[std::size_t(11 * N)];
    p.nx = n;  p.ny = n;  p.nz = n;
    p.fm.rho_L = Real(1);  p.fm.rho_H = Real(10);
    p.fm.mu_L = Real(0.05);  p.fm.mu_H = Real(0.5);

    for (long i = 0; i < N; ++i) pf_derivatives_node<false>(p, i);

    // A shear the populations do NOT carry: grad u is large at the interface,
    // which is the only place F_nu lives. The populations are at rest, so the
    // macroscopic pass will erase this if it runs first -- which is the point.
    const double A = 0.05, k = 2.0 * 3.14159265358979323846 / n;
    auto write_shear = [&]() {
      for (long id = 0; id < N; ++id) {
        int x, y, z;
        coords(id, n, n, x, y, z);
        p.ux[id] = Real(A * std::sin(k * y));
        p.uy[id] = Real(0);  p.uz[id] = Real(0);
      }
    };
    auto fnu_max = [&]() {
      double m = 0;
      for (long i = 0; i < N; ++i)
        m = std::fmax(m, std::sqrt(double(p.vx[i]) * double(p.vx[i])
                                 + double(p.vy[i]) * double(p.vy[i])
                                 + double(p.vz[i]) * double(p.vz[i])));
      return m;
    };
    auto clear_fnu = [&]() {
      for (long i = 0; i < N; ++i) { p.vx[i] = Real(0); p.vy[i] = Real(0); p.vz[i] = Real(0); }
    };

    // ORDER AS SHIPPED: macro, then viscous. The macro pass replaces the shear
    // with u from the populations, which is zero, so F_nu must vanish.
    write_shear();  clear_fnu();
    for (long i = 0; i < N; ++i) pf_macro_node<0, false>(p, N, i);
    for (long i = 0; i < N; ++i) pf_viscous_node<false>(p, i);
    const double with_macro_first = fnu_max();

    // THE OLD ORDER, for contrast: viscous first, so F_nu is built from a
    // velocity field the populations never had.
    write_shear();  clear_fnu();
    for (long i = 0; i < N; ++i) pf_viscous_node<false>(p, i);
    const double with_viscous_first = fnu_max();

    std::printf("        max|F_nu| with macro first  = %.6e  (u from the populations)\n",
                with_macro_first);
    std::printf("        max|F_nu| with viscous first = %.6e  (u from the stale array)\n",
                with_viscous_first);
    check(with_macro_first < 1e-14,
          "macro-before-viscous: F_nu sees u(t), which is zero here",
          with_macro_first);
    check(with_viscous_first > 1e-6,
          "and the old order would have seen the stale field instead",
          with_viscous_first);
  }

  std::printf("\n[phasefield] %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
