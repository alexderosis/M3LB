//==============================================================================
//  The colour-gradient operator, checked against its own algebra.
//
//  Saito et al., Phys. Rev. E 98, 013305 (2018). No flow, no time stepping: this
//  file establishes the identities the model rests on, each of which can be
//  wrong in a way a running simulation would absorb rather than report.
//
//   1. phi_i is a weight set. It must sum to one for ANY alpha, and its second
//      moment must be 9(1-alpha)/19 -- that number IS the sound speed, and the
//      density ratio is obtained from it through p = rho cs^2 and nothing else.
//      Get it wrong and the ratio is wrong by exactly the same factor, silently,
//      because every other test still passes.
//   2. The equilibrium carries rho and rho u and nothing else in its first two
//      moments, at third order in u and with the Galilean term Phi_i present.
//      Phi_i is the term most easily dropped, and dropping it changes nothing
//      that is checked anywhere else in a static test.
//   3. The perturbation conserves mass and momentum, and acts only on the second
//      moment. sum_i B_i = 1/3 against sum_i w_i (c.n)^2 = 1/3 is the identity
//      that makes it so, and it holds for EVERY direction of n. The same block
//      closes on the capillary stress against sigma = 2 A tau / 9, Eq. (D14) --
//      the only closed form the model offers for the interfacial tension, which
//      validation/static_droplet.cpp measures the same number the hard way. (It
//      was listed as a seventh block here for a while; it has always lived in
//      this one.)
//   4. Recolouring is a partition: f^r + f^b = f identically, for any beta and
//      any gradient. Mass and momentum cannot be lost there whatever else is
//      wrong, and this says so.
//   5. Equilibrium is a fixed point. With no perturbation, no force and no
//      density gradient, collision must return the equilibrium unchanged at any
//      relaxation rate.
//   6. The closed-form central moments reproduce the population transform they
//      replaced, over states that carry a density ratio, a perturbation, a
//      density gradient and a body force. This is the only block that can catch
//      a wrong coefficient in a single high-order moment slot: every identity
//      above is a sum over i, and one slot's error survives most of them.
//==============================================================================
#include "Check.hpp"
#include "collision/ColourGradient.hpp"
#include "grid/Domain.hpp"
#include "core/Types.hpp"

#include <cmath>

using namespace lbm;
using L  = D3Q27;
using CG = ColourGradient<L>;

static Real TOL() { return sizeof(Real) == 4 ? Real(2e-5) : Real(1e-12); }

// A reproducible uniform on [0,1). A fixed stream rather than <random>, so a
// failure here is the same failure on every machine and can be bisected.
static double urand(unsigned long long& st) {
  st = st * 6364136223846793005ULL + 1442695040888963407ULL;
  return double((st >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
}

//==============================================================================
//  THE IMPLEMENTATION THE CLOSED FORM REPLACED, kept here as the reference.
//
//  collide() used to build the equilibrium and the perturbation as POPULATIONS
//  and transform both -- exact by construction, and the largest stack frame in
//  the tree (tests/frame_check.sh: 1200 bytes at FP64, with a live moment loop).
//  It now evaluates their central moments in closed form. Block 6 asserts the
//  two agree, which is the only thing that makes the replacement safe: nothing
//  else in this file would notice a wrong coefficient in one of the 17 generated
//  high-order slots, because every identity above is a SUM over i and a single
//  slot's error survives most of them.
//
//  This is deliberately a transcription of the old code rather than a tidied
//  version of it. If it is ever "simplified" to share machinery with the operator
//  it checks, it stops being an independent path and this block stops testing
//  anything.
//==============================================================================
static void reference_collide(const CG& cg, Real f[L::Q], Real rho,
                              const Real u[3], Real p, Index n) {
  using B = CG::Basis;
  const Real nubar = cg.nu_at(p);
  const Real omega = cg.omega_at(p);

  const Real dr[3] = {cg.Rx(n), cg.Ry(n), cg.Rz(n)};
  Real G[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      G[i][j] = (u[i] * dr[j] + u[j] * dr[i]) / Real(48);
  const Real udrho = u[0] * dr[0] + u[1] * dr[1] + u[2] * dr[2];

  Real fe[L::Q];
  cg.equilibrium(fe, rho, p, u, G, nubar, udrho);

  const Real g[3] = {cg.Gx(n), cg.Gy(n), cg.Gz(n)};
  const Real gm2  = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
  Real pert[L::Q];
  if (gm2 > Real(1e-24) && cg.A != Real(0)) {
    const Real gm = Real(std::sqrt(double(gm2))), invm = Real(1) / gm;
    const Real nh[3] = {g[0] * invm, g[1] * invm, g[2] * invm};
    for (int i = 0; i < L::Q; ++i) {
      const Real cn = Real(cvel<L>(i, 0)) * nh[0] + Real(cvel<L>(i, 1)) * nh[1]
                    + Real(cvel<L>(i, 2)) * nh[2];
      // Eq. (D14)'s A/2, reached through the operator's own accessor so that the
      // reference cannot drift from it silently.
      pert[i] = CG::perturbation_coefficient(cg.A) * gm
              * (weight<L, Real>(i) * cn * cn - CG::B_i(i));
    }
  } else {
    for (int i = 0; i < L::Q; ++i) pert[i] = Real(0);
  }

  const Real ub[3] = {u[0], u[1], u[2]};
  Real k[CG::NM], ke[CG::NM], kp[CG::NM];
  B::to_moments<true>(f,    ub, k);
  B::to_moments<true>(fe,   ub, ke);
  B::to_moments<true>(pert, ub, kp);

  const Real fw = rho - cg.rho_ref;
  const Real F[3] = {fw * cg.bx, fw * cg.by, fw * cg.bz};
  for (int a = 0; a < 3; ++a) k[CG::i1(a)] += F[a] + kp[CG::i1(a)];
  {
    Real d[3], e[3], q[3];
    Real tr = 0, tre = 0, trq = 0;
    for (int a = 0; a < 3; ++a) {
      d[a] = k[CG::i2d(a)];  e[a] = ke[CG::i2d(a)];  q[a] = kp[CG::i2d(a)];
      tr += d[a];  tre += e[a];  trq += q[a];
    }
    const Real invD = Real(1) / Real(3);
    const Real tr_post = (Real(1) - cg.omega_bulk) * tr + cg.omega_bulk * tre + trq;
    for (int a = 0; a < 3; ++a)
      k[CG::i2d(a)] = (Real(1) - omega) * (d[a] - tr * invD)
                    + omega * (e[a] - tre * invD)
                    + (q[a] - trq * invD) + tr_post * invD;
    for (int a = 0; a < 3; ++a)
      for (int b = a + 1; b < 3; ++b) {
        const int id = CG::i2s(a, b);
        k[id] = (Real(1) - omega) * k[id] + omega * ke[id] + kp[id];
      }
  }
  for (int m = 0; m < CG::NM; ++m)
    if (B::order(m) >= 3) k[m] = ke[m] + kp[m];
  B::to_populations<true>(k, ub, f);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Real tol = TOL();

    //--------------------------------------------------------------------------
    std::printf("\n1. phi_i is a weight set, and its second moment is the sound speed\n");
    {
      const double alphas[4] = {8.0 / 27.0, 0.2, 0.5, 0.9};
      for (int k = 0; k < 4; ++k) {
        const Real a = Real(alphas[k]);
        double s = 0, m2[3] = {0, 0, 0}, off = 0, odd = 0;
        for (int i = 0; i < L::Q; ++i) {
          const double p = double(CG::phi_i(i, a));
          s += p;
          for (int d = 0; d < 3; ++d) m2[d] += p * cvel<L>(i, d) * cvel<L>(i, d);
          off += p * cvel<L>(i, 0) * cvel<L>(i, 1);
          odd += p * cvel<L>(i, 0);
        }
        const double cs2 = double(CG::cs2_of_alpha(a));
        char b[128];
        std::snprintf(b, sizeof b, "sum phi_i = 1            (alpha = %.4f)", alphas[k]);
        check::near(s, 1.0, double(tol), b);
        std::snprintf(b, sizeof b, "second moment = 9(1-a)/19 (alpha = %.4f)", alphas[k]);
        check::near(m2[0], cs2, double(tol), b);
        check::near(m2[1], cs2, double(tol), "  ... isotropic in y");
        check::near(m2[2], cs2, double(tol), "  ... isotropic in z");
        check::near(off, 0.0, double(tol), "  ... no off-diagonal part");
        check::near(odd, 0.0, double(tol), "  ... odd moments vanish");
      }
      // alpha_b = 8/27 is the paper's choice because it makes the blue fluid an
      // ordinary lattice; if that stops being true the density ratio moves.
      check::near(double(CG::cs2_of_alpha(Real(8) / Real(27))), 1.0 / 3.0,
                  double(tol), "alpha = 8/27 gives cs^2 = 1/3 exactly");
      // and the ratio relation of Eq. (25) inverts consistently
      for (double g : {1.0, 10.0, 100.0, 1000.0}) {
        const Real ab = Real(8) / Real(27);
        const Real ar = CG::alpha_r_from_ratio(Real(g), ab);
        const double back = double((Real(1) - ab) / (Real(1) - ar));
        char b[128];
        std::snprintf(b, sizeof b, "gamma = (1-ab)/(1-ar) inverts   (gamma = %.0f)", g);
        check::near(back, g, g * 1e-12, b);
      }
    }

    //--------------------------------------------------------------------------
    std::printf("\n2. the equilibrium carries rho and rho u, Phi_i included\n");
    {
      CG cg;
      cg.alpha_r = Real(0.1);  cg.alpha_b = Real(8) / Real(27);
      cg.rho_r0 = Real(1);     cg.rho_b0 = Real(1);
      const Real rho = Real(1.7);
      const Real rr_ = Real(1.1), rb_ = rho - rr_;
      const Real u[3] = {Real(0.11), Real(-0.07), Real(0.05)};
      const Real dr[3] = {Real(0.3), Real(-0.2), Real(0.13)};   // grad rho
      Real G[3][3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) G[i][j] = (u[i] * dr[j] + u[j] * dr[i]) / Real(48);
      const Real udr = u[0] * dr[0] + u[1] * dr[1] + u[2] * dr[2];
      const Real nub = Real(0.13);

      Real fe[L::Q];
      cg.equilibrium(fe, rr_ + rb_, cg.order_parameter(rr_, rb_), u, G, nub, udr);
      double s = 0, mx = 0, my = 0, mz = 0;
      for (int i = 0; i < L::Q; ++i) {
        s += double(fe[i]);
        mx += double(fe[i]) * cvel<L>(i, 0);
        my += double(fe[i]) * cvel<L>(i, 1);
        mz += double(fe[i]) * cvel<L>(i, 2);
      }
      check::near(s, double(rho), double(tol) * 10, "sum f^eq = rho");
      check::near(mx, double(rho * u[0]), double(tol) * 10, "sum c_x f^eq = rho u_x");
      check::near(my, double(rho * u[1]), double(tol) * 10, "sum c_y f^eq = rho u_y");
      check::near(mz, double(rho * u[2]), double(tol) * 10, "sum c_z f^eq = rho u_z");

      // Phi_i on its own must be invisible to both conserved moments, which is
      // what lets it correct the STRESS without disturbing anything else.
      Real fz[L::Q];
      const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      cg.equilibrium(fz, rr_ + rb_, cg.order_parameter(rr_, rb_), u, zeroG, Real(0), Real(0));
      double ds = 0, dm = 0, dstress = 0;
      for (int i = 0; i < L::Q; ++i) {
        const double d = double(fe[i]) - double(fz[i]);
        ds += d;  dm += d * cvel<L>(i, 0);
        dstress += d * cvel<L>(i, 0) * cvel<L>(i, 1);
      }
      check::near(ds, 0.0, double(tol) * 10, "Phi_i changes no density");
      check::near(dm, 0.0, double(tol) * 10, "Phi_i changes no momentum");
      check::ok(std::fabs(dstress) > 1e-6,
                "Phi_i DOES change the stress (it would be dead code otherwise)");
      std::printf("        (its xy stress contribution is %.4e)\n", dstress);

      // THE PRESSURE, Eq. (D12): p = rho cs^2(alpha-bar), the trace of the
      // second moment at rest. This is where the density ratio enters, and it
      // is the ONE quantity that distinguishes Eq. (D6)'s single interpolated
      // alpha from the per-colour reading sum_k rho_k phi_i(alpha_k) that this
      // file carried until 2026-09-19.
      //
      // AND IT CANNOT DISTINGUISH THEM HERE, which is the point of the second
      // case below. cs^2 is LINEAR in alpha and alpha-bar is linear in the
      // order parameter, so when rho_r^0 = rho_b^0 the order parameter is
      // (rho_r - rho_b)/rho exactly, rho(1 +- phi)/2 collapses to rho_r and
      // rho_b, and the two readings are ALGEBRAICALLY IDENTICAL -- not close,
      // equal. A test run only at a matched initial density would pass under
      // either reading and prove nothing about which one is implemented.
      auto trace_at_rest = [&](const CG& m, Real rr, Real rb) {
        Real fq[L::Q];
        const Real u0[3] = {0, 0, 0};
        m.equilibrium(fq, rr + rb, m.order_parameter(rr, rb), u0, zeroG,
                      Real(0), Real(0));
        double t = 0;
        for (int i = 0; i < L::Q; ++i)
          t += double(fq[i]) * cvel<L>(i, 0) * cvel<L>(i, 0);
        return t;
      };
      const double per_colour = double(rr_ * CG::cs2_of_alpha(cg.alpha_r)
                                     + rb_ * CG::cs2_of_alpha(cg.alpha_b));
      const double bar = double((rr_ + rb_)
                    * CG::cs2_of_alpha(cg.alpha_at(cg.order_parameter(rr_, rb_))));
      check::near(trace_at_rest(cg, rr_, rb_), bar, double(tol) * 10,
                  "p = rho cs^2(alpha) at rest, Eq. (D12)");
      check::near(per_colour, bar, double(tol) * 10,
                  "  ... and at rho_r^0 = rho_b^0 BOTH readings give it (see below)");

      // THE TWO READINGS ARE THE SAME FUNCTIONAL FORM, AND DIFFER ONLY IN HOW
      // ALPHA IS INTERPOLATED. That is the whole of the Eq. (D6) question and it
      // is an identity, not an approximation:
      //
      //     sum_k rho_k phi_i(alpha_k)  ==  rho phi_i(alpha_P)
      //     with   alpha_P = 1 - 19 P / (9 rho),   P = sum_k rho_k cs_k^2,
      //
      // because phi_i is AFFINE in alpha -- its rest slot is alpha and every
      // other slot is a fixed multiple of (1 - alpha) -- and both sides then
      // match term by term through sum_k rho_k (1 - alpha_k) = 19 P / 9.
      //
      // So PerColour IS the single-alpha reading, with alpha chosen to make the
      // pressure continuous. Eq. (D13) instead interpolates alpha LINEARLY in
      // the order parameter, and that single choice is what produces the
      // (gamma+1)^2/(4 gamma) interface pressure spike. Nothing else about the
      // two readings differs -- not the equilibrium order, not the collision.
      {
        CG cp;
        cp.alpha_b = Real(8) / Real(27);
        cp.alpha_r = CG::alpha_r_from_ratio(Real(100), cp.alpha_b);
        cp.rho_r0 = Real(100);  cp.rho_b0 = Real(1);
        const Real rr2 = Real(40), rb2 = Real(0.6);
        Real rr_s, rb_s;
        cp.rest = CG::RestTerm::PerColour;
        cp.split_colours(rr2 + rb2, cp.order_parameter(rr2, rb2), rr_s, rb_s);
        const double P = double(rr_s * CG::cs2_of_alpha(cp.alpha_r)
                              + rb_s * CG::cs2_of_alpha(cp.alpha_b));
        const double rho = double(rr2 + rb2);
        const Real aP = Real(1.0 - 19.0 * P / (9.0 * rho));
        double worst = 0;
        for (int i = 0; i < L::Q; ++i)
          worst = std::max(worst,
                    std::fabs(double(cp.rest_term(i, rr2 + rb2,
                                                  cp.order_parameter(rr2, rb2)))
                            - rho * double(CG::phi_i(i, aP))));
        check::near(worst, 0.0, double(tol) * 100,
                    "per-colour rest term == rho phi_i(alpha_P), population by population");
        std::printf("        (so the readings differ ONLY in the rule for alpha:"
                    " linear in phi, or alpha_P)\n");
      }

      // At a real density ratio they part company, and THIS is the case that
      // says which one the operator implements.
      {
        CG cr;
        cr.alpha_b = Real(8) / Real(27);
        cr.alpha_r = CG::alpha_r_from_ratio(Real(10), cr.alpha_b);
        cr.rho_r0 = Real(10);  cr.rho_b0 = Real(1);       // gamma = 10
        const Real rr2 = Real(4.0), rb2 = Real(0.6);      // mid-interface
        const double pc = double(rr2 * CG::cs2_of_alpha(cr.alpha_r)
                               + rb2 * CG::cs2_of_alpha(cr.alpha_b));
        const double ab = double((rr2 + rb2)
                    * CG::cs2_of_alpha(cr.alpha_at(cr.order_parameter(rr2, rb2))));
        check::ok(std::fabs(pc / ab - 1.0) > 1e-3,
                  "at gamma = 10 the two readings genuinely differ");
        std::printf("        (alpha-bar gives %.8f, per colour %.8f, %+.2f%%)\n",
                    ab, pc, 100.0 * (pc / ab - 1.0));
        // Both are reachable and each gives ITS OWN answer -- asserted by name
        // rather than through the default, so that flipping the default cannot
        // silently turn this into a test of the other reading.
        cr.rest = CG::RestTerm::AlphaBar;
        check::near(trace_at_rest(cr, rr2, rb2), ab, double(tol) * 10,
                    "  ... RestTerm::AlphaBar gives Eq. (D6)'s rho cs^2(alpha-bar)");
        cr.rest = CG::RestTerm::PerColour;
        check::near(trace_at_rest(cr, rr2, rb2), pc, double(tol) * 10,
                    "  ... RestTerm::PerColour gives the continuous sum_k rho_k cs_k^2");
        // The bulk pressure is the SAME on both sides of a gamma = 10 interface
        // under PerColour, and that is the property AlphaBar loses.
        const double p_red  = double(cr.pressure(Real(10), Real(+1)));
        const double p_blue = double(cr.pressure(Real(1),  Real(-1)));
        check::near(p_red, p_blue, double(tol) * 10,
                    "  ... and PerColour's pressure matches across the interface");
        cr.rest = CG::RestTerm::AlphaBar;
        const double spike = double(cr.pressure(Real(5.5), Real(0)))
                           / double(CG::cs2_of_alpha(cr.alpha_b));
        check::near(spike, 121.0 / 40.0, 1e-9,
                    "  ... while AlphaBar's midpoint pressure is (g+1)^2/(4g) = 3.025x");
      }
    }

    //--------------------------------------------------------------------------
    std::printf("\n3. the perturbation is a pure second-moment source\n");
    {
      double sb = 0;
      for (int i = 0; i < L::Q; ++i) sb += double(CG::B_i(i));
      check::near(sb, 1.0 / 3.0, double(tol), "sum B_i = 1/3");

      // For EVERY direction of n, not just the axes: the cancellation is
      // sum w_i (c.n)^2 = cs^2 = 1/3 and it is direction independent.
      const double dirs[5][3] = {{1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                 {1, 1, 1}, {0.37, -0.62, 0.19}};
      double worst_m = 0, worst_p = 0, worst_n = 0, least_s = 1e30;
      for (int k = 0; k < 5; ++k) {
        const double nm = std::sqrt(dirs[k][0] * dirs[k][0] + dirs[k][1] * dirs[k][1]
                                  + dirs[k][2] * dirs[k][2]);
        const double nh[3] = {dirs[k][0] / nm, dirs[k][1] / nm, dirs[k][2] / nm};
        // a tangent to n, for the stress component that must NOT vanish
        double t[3] = {-nh[1], nh[0], 0};
        double tm = std::sqrt(t[0] * t[0] + t[1] * t[1]);
        if (tm < 1e-8) { t[0] = 0; t[1] = -nh[2]; t[2] = nh[1]; tm = std::sqrt(t[1]*t[1]+t[2]*t[2]); }
        for (int d = 0; d < 3; ++d) t[d] /= tm;
        double s = 0, p[3] = {0, 0, 0}, snn = 0, stt = 0;
        for (int i = 0; i < L::Q; ++i) {
          const double cn = cvel<L>(i, 0) * nh[0] + cvel<L>(i, 1) * nh[1]
                          + cvel<L>(i, 2) * nh[2];
          const double ct = cvel<L>(i, 0) * t[0] + cvel<L>(i, 1) * t[1]
                          + cvel<L>(i, 2) * t[2];
          const double o = double(weight<L, Real>(i)) * cn * cn - double(CG::B_i(i));
          s += o;
          for (int d = 0; d < 3; ++d) p[d] += o * cvel<L>(i, d);
          snn += o * cn * cn;
          stt += o * ct * ct;
        }
        worst_m = std::max(worst_m, std::fabs(s));
        for (int d = 0; d < 3; ++d) worst_p = std::max(worst_p, std::fabs(p[d]));
        worst_n = std::max(worst_n, std::fabs(snn));
        least_s = std::min(least_s, std::fabs(stt));
      }
      check::near(worst_m, 0.0, double(tol), "conserves mass, every direction of n");
      check::near(worst_p, 0.0, double(tol), "conserves momentum, every direction");
      check::near(worst_n, 0.0, double(tol),
                  "normal capillary stress vanishes, every direction of n");
      check::ok(least_s > 1e-2, "but the tangential one does not");

      // sigma = 2 A tau / 9, Eq. (D14)'s normalisation, against the capillary
      // stress integral,
      // S = -tau sum_i Omega^(2) c_i c_i. For a flat interface with normal n the
      // tension is the difference between the stress along n and across it.
      const double A = 8e-4, tau = 1.0, gm = 1.0;
      double snn = 0, stt = 0;
      for (int i = 0; i < L::Q; ++i) {
        const double cn = cvel<L>(i, 2);
        const double o = double(CG::perturbation_coefficient(Real(A))) * gm
                       * (double(weight<L, Real>(i)) * cn * cn - double(CG::B_i(i)));
        snn += -tau * o * cvel<L>(i, 2) * cvel<L>(i, 2);
        stt += -tau * o * cvel<L>(i, 0) * cvel<L>(i, 0);
      }
      // sigma is the integral of (S_tt - S_nn) THROUGH the interface. The
      // integrand is proportional to |grad phi|, whose integral across the
      // interface is the jump in phi -- and phi runs -1 to +1, so that is 2.
      const double dphi = 2.0;
      const double sigma = (stt - snn) / gm * dphi;
      const double sig_th = double(CG::sigma_from_A(Real(A), Real(tau)));
      check::near(sigma, sig_th, 1e-15,
                  "capillary stress gives sigma = 2 A tau / 9, Eq. (D14)");
      std::printf("        (sigma = %.8e, 2 A tau / 9 = %.8e)\n", sigma, sig_th);
    }

    //--------------------------------------------------------------------------
    std::printf("\n4. recolouring is a partition\n");
    {
      Domain d(4, 4, 4, true, true, true);
      CG cg;
      cg.alpha_r = Real(0.1);  cg.alpha_b = Real(8) / Real(27);
      cg.rho_r0 = Real(1);     cg.rho_b0 = Real(1);
      cg.beta = Real(0.7);
      View1D<Real> gx("gx", d.n_padded), gy("gy", d.n_padded), gz("gz", d.n_padded);
      auto hx = Kokkos::create_mirror_view(gx);
      auto hy = Kokkos::create_mirror_view(gy);
      auto hz = Kokkos::create_mirror_view(gz);
      for (Index n = 0; n < d.n_padded; ++n) {
        hx(n) = Real(0.31); hy(n) = Real(-0.12); hz(n) = Real(0.44);
      }
      Kokkos::deep_copy(gx, hx); Kokkos::deep_copy(gy, hy); Kokkos::deep_copy(gz, hz);
      cg.Gx = gx; cg.Gy = gy; cg.Gz = gz;

      Real f[L::Q], fr[L::Q], fb[L::Q];
      for (int i = 0; i < L::Q; ++i)
        f[i] = Real(0.03) * Real((i * 7) % 11 - 5) + weight<L, Real>(i);
      const Real rr = Real(1.3), rb = Real(0.4);
      cg.recolour(f, rr, rb, Real(0.2), 0, fr, fb);
      double worst = 0, ms = 0, px = 0;
      for (int i = 0; i < L::Q; ++i) {
        worst = std::max(worst, std::fabs(double(fr[i]) + double(fb[i]) - double(f[i])));
        ms += double(fr[i]) + double(fb[i]);
        px += (double(fr[i]) + double(fb[i])) * cvel<L>(i, 0);
      }
      check::near(worst, 0.0, double(tol), "f^r + f^b = f, population by population");
      double fs = 0, fx = 0;
      for (int i = 0; i < L::Q; ++i) { fs += double(f[i]); fx += double(f[i]) * cvel<L>(i, 0); }
      check::near(ms, fs, double(tol), "so mass survives it");
      check::near(px, fx, double(tol), "and momentum survives it");
      // and it does something: colour is pushed along the gradient
      double split = 0;
      for (int i = 0; i < L::Q; ++i)
        split = std::max(split, std::fabs(double(fr[i]) - rr / (rr + rb) * double(f[i])));
      check::ok(split > 1e-4, "but it is not the identity: colour moves up grad phi");
    }

    //--------------------------------------------------------------------------
    std::printf("\n5. equilibrium is a fixed point\n");
    {
      Domain d(4, 4, 4, true, true, true);
      CG cg;
      cg.alpha_r = Real(0.15);  cg.alpha_b = Real(8) / Real(27);
      cg.nu_r = Real(0.05);     cg.nu_b = Real(0.05);
      cg.rho_r0 = Real(1);      cg.rho_b0 = Real(1);
      cg.A = Real(0);           // no perturbation
      View1D<Real> z("z", d.n_padded);
      cg.Gx = z; cg.Gy = z; cg.Gz = z;   // no colour gradient
      cg.Rx = z; cg.Ry = z; cg.Rz = z;   // no density gradient

      const Real rho = Real(1.4);
      const Real u[3] = {Real(0.09), Real(-0.06), Real(0.03)};
      for (double p : {-1.0, -0.3, 0.0, 0.5, 1.0}) {
        const Real pp = Real(p);
        const Real rr2 = rho * Real(0.5) * (Real(1) + pp);
        const Real rb2 = rho - rr2;
        const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        Real fe[L::Q], f[L::Q];
        cg.equilibrium(fe, rr2 + rb2, pp, u, zeroG, Real(0), Real(0));
        for (int i = 0; i < L::Q; ++i) f[i] = fe[i];
        cg.collide(f, rr2 + rb2, u, pp, 0);
        double worst = 0;
        for (int i = 0; i < L::Q; ++i)
          worst = std::max(worst, std::fabs(double(f[i]) - double(fe[i])));
        char b[128];
        std::snprintf(b, sizeof b, "collision leaves f^eq alone   (phi = %+.1f)", p);
        check::near(worst, 0.0, double(tol) * 100, b);
      }
      // ... and the collision conserves mass and momentum away from equilibrium
      Real f[L::Q];
      const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      const Real rr3 = rho * Real(0.6), rb3 = rho - rr3;
      cg.equilibrium(f, rr3 + rb3, cg.order_parameter(rr3, rb3), u, zeroG, Real(0), Real(0));
      for (int i = 0; i < L::Q; ++i) f[i] *= Real(1) + Real(0.05) * Real((i * 5) % 7 - 3);
      double s0 = 0, p0[3] = {0, 0, 0};
      for (int i = 0; i < L::Q; ++i) {
        s0 += double(f[i]);
        for (int dd = 0; dd < 3; ++dd) p0[dd] += double(f[i]) * cvel<L>(i, dd);
      }
      cg.collide(f, rr3 + rb3, u, cg.order_parameter(rr3, rb3), 0);
      double s1 = 0, p1[3] = {0, 0, 0};
      for (int i = 0; i < L::Q; ++i) {
        s1 += double(f[i]);
        for (int dd = 0; dd < 3; ++dd) p1[dd] += double(f[i]) * cvel<L>(i, dd);
      }
      check::near(s1, s0, double(tol) * 100, "collision conserves mass off equilibrium");
      check::near(p1[0], p0[0], double(tol) * 100, "  ... and x momentum");
      check::near(p1[1], p0[1], double(tol) * 100, "  ... and y momentum");
      check::near(p1[2], p0[2], double(tol) * 100, "  ... and z momentum");
    }

    //--------------------------------------------------------------------------
    std::printf("\n6. the closed form against the transform it replaced\n");
    //
    // collide() used to build the equilibrium and the perturbation as
    // POPULATIONS and transform both, which is exact by construction. It now
    // evaluates their central moments in closed form instead, to get rid of two
    // live 27-arrays and two of the three forward transforms. That is only worth
    // doing if the two agree, so the old path lives in reference_collide() above
    // and this block asserts it -- over states that exercise every branch the
    // closed form collapsed: a density ratio (alpha_r != alpha_b, so S != 0), a
    // live perturbation, a density gradient feeding Phi_i, a body force, and the
    // |grad phi| = 0 case that switches the perturbation off entirely.
    //
    // The two paths share nothing but the input: the reference goes through
    // Eq. (18) as written and three transforms, the closed form through the
    // derived moments and one. A transcription error in any of the 17 generated
    // slots shows up here and nowhere else.
    {
      Domain d(4, 4, 4, true, true, true);
      View1D<Real> gx("gx", d.n_padded), gy("gy", d.n_padded), gz("gz", d.n_padded);
      View1D<Real> rx("rx", d.n_padded), ry("ry", d.n_padded), rz("rz", d.n_padded);
      auto hg = Kokkos::create_mirror_view(gx);
      auto hgy = Kokkos::create_mirror_view(gy);
      auto hgz = Kokkos::create_mirror_view(gz);
      auto hr = Kokkos::create_mirror_view(rx);
      auto hry = Kokkos::create_mirror_view(ry);
      auto hrz = Kokkos::create_mirror_view(rz);

      unsigned long long st = 0x2018013305ULL;   // the paper's volume and page
      double worst_rel = 0;
      int states = 0, nonzero = 0;
      for (int trial = 0; trial < 60; ++trial) {
        CG cg;
        cg.alpha_b = Real(8) / Real(27);
        // gamma = (1 - alpha_b)/(1 - alpha_r) swept from 1 to about 30, so S = 0
        // and S != 0 are both covered -- at a matched ratio the rest term's two
        // separable products collapse into one and half the derivation is idle.
        const double gamma = 1.0 + 29.0 * urand(st);
        cg.alpha_r = CG::alpha_r_from_ratio(Real(gamma), cg.alpha_b);
        cg.rho_r0 = Real(gamma);  cg.rho_b0 = Real(1);
        cg.nu_r = Real(0.02 + 0.3 * urand(st));
        cg.nu_b = Real(0.02 + 0.3 * urand(st));
        cg.omega_bulk = Real(0.4 + 1.4 * urand(st));
        cg.beta = Real(0.7);
        cg.A = (trial % 5 == 0) ? Real(0) : Real(0.002 + 0.02 * urand(st));
        cg.bx = Real(1e-5 * (2 * urand(st) - 1));
        cg.by = Real(1e-5 * (2 * urand(st) - 1));
        cg.bz = Real(1e-5 * (2 * urand(st) - 1));
        cg.rho_ref = (trial % 3 == 0) ? Real(0) : Real(0.5 + urand(st));

        // Every fifth state has NO colour gradient, which is the branch that
        // switches the perturbation off -- the bulk, where most nodes live.
        const bool flat = (trial % 5 == 1);
        const Real gv[3] = {flat ? Real(0) : Real(0.4 * (2 * urand(st) - 1)),
                            flat ? Real(0) : Real(0.4 * (2 * urand(st) - 1)),
                            flat ? Real(0) : Real(0.4 * (2 * urand(st) - 1))};
        const Real rv[3] = {Real(0.3 * (2 * urand(st) - 1)),
                            Real(0.3 * (2 * urand(st) - 1)),
                            Real(0.3 * (2 * urand(st) - 1))};
        hg(0) = gv[0]; hgy(0) = gv[1]; hgz(0) = gv[2];
        hr(0) = rv[0]; hry(0) = rv[1]; hrz(0) = rv[2];
        Kokkos::deep_copy(gx, hg); Kokkos::deep_copy(gy, hgy);
        Kokkos::deep_copy(gz, hgz);
        Kokkos::deep_copy(rx, hr); Kokkos::deep_copy(ry, hry);
        Kokkos::deep_copy(rz, hrz);
        cg.Gx = gx; cg.Gy = gy; cg.Gz = gz;
        cg.Rx = rx; cg.Ry = ry; cg.Rz = rz;

        const Real rr = Real(0.2 + gamma * urand(st));
        const Real rb = Real(0.2 + urand(st));
        const Real u[3] = {Real(0.1 * (2 * urand(st) - 1)),
                           Real(0.1 * (2 * urand(st) - 1)),
                           Real(0.1 * (2 * urand(st) - 1))};
        const Real pp = cg.order_parameter(rr, rb);

        Real f0[L::Q], f1[L::Q], f2[L::Q];
        const Real zeroG[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        cg.equilibrium(f0, rr + rb, pp, u, zeroG, Real(0), Real(0));
        // Away from equilibrium, or the non-equilibrium slots the collision
        // relaxes would all be zero and most of the operator would go untested.
        for (int i = 0; i < L::Q; ++i) {
          f0[i] *= Real(1) + Real(0.1 * (2 * urand(st) - 1));
          f1[i] = f0[i];  f2[i] = f0[i];
        }

        cg.collide(f1, rr + rb, u, pp, 0);
        reference_collide(cg, f2, rr + rb, u, pp, 0);

        double scale = 0, err = 0;
        for (int i = 0; i < L::Q; ++i) {
          scale = std::max(scale, std::fabs(double(f2[i])));
          err   = std::max(err, std::fabs(double(f1[i]) - double(f2[i])));
        }
        if (scale > 0) worst_rel = std::max(worst_rel, err / scale);
        ++states;
        if (!flat && cg.A != Real(0)) ++nonzero;
      }
      // Relative, because the populations themselves run over three decades once
      // a density ratio is on. FP64 lands near 1e-15; the FP32 bound is what a
      // different summation order costs at 24 bits, not a property of the model.
      const double rtol = (sizeof(Real) == 4) ? 3e-5 : 1e-13;
      char b[160];
      std::snprintf(b, sizeof b,
                    "closed form == transform, %d states (%d with perturbation)",
                    states, nonzero);
      check::near(worst_rel, 0.0, rtol, b);
    }
  }
  const int rc = check::report("colour_gradient");
  Kokkos::finalize();
  return rc;
}
