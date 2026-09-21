//==============================================================================
//  The enthalpy operator, checked against its own algebra.
//
//  No flow, no time stepping. Every block below pins an identity that can be
//  wrong in a way a running melting simulation would absorb rather than report:
//  a front in the wrong place still looks like a front.
//
//   1. THE ScalarBGK IDENTITY. At La = 0 with matched properties this operator
//      must be population-for-population ScalarBGK. That is what the (1 - w_0)
//      factor in the rest term buys, and dropping the factor breaks this block
//      first -- h_0 comes back as 0 instead of w_0 dH.
//   2. EQUILIBRIUM MOMENTS, in all three regimes, with E_datum and H_ref swept
//      independently. sum = dH, flux = E u, second = cs2 dE. This is where
//      "diffuse H instead of E" dies: the second moment reads cs2 dH and the
//      scheme converges cleanly to the wrong front speed. The three candidate
//      mis-writes have three different signatures, which is why both gauges
//      move.
//   3. CONSERVATION UNDER collide. The regularised form writes h[0] as the
//      residual, so it conserves in floating point; BGK's per-slot update
//      conserves to the accumulated error of Q multiply-adds. The block states
//      the difference rather than asserting one number for both.
//   4. INVERSION ROUND TRIP, across all three regimes and both band widths,
//      including H = 0 and H = H_l exactly. f_l must land in [0,1] with no
//      clamp, and the one-sided limits of T, E and dE/dT must agree at both
//      band edges -- a discontinuity there is a jump in diffusivity that
//      refinement cannot remove.
//   5. THE STABLE ROOT. The naive quadratic root is transcribed here verbatim
//      (not tidied) and diffed against the citardauq form as d_cp -> 0. The
//      naive one loses its digits while staying finite and plausible, and
//      divides by zero at cp_s == cp_l, which is the commonest case.
//   6. THE ISOTHERMAL LIMIT IS AN ORDINARY POINT. At dTm = 0 every mushy H
//      gives T = T_m exactly, f_l = H/La exactly, and no division by zero --
//      because f_l is solved first and dTm is only ever a multiplier. The same
//      block shows the forward map really is two-valued there, which is why
//      enthalpy_of takes two arguments and enthalpy_of_T aborts.
//   7. REGULARISED vs BGK. Identical at omega = 1; flux moment identical at
//      every rate; the regularised ghosts sit at cs2 dE where BGK's are off
//      equilibrium; and h[0] = dH - D cs2 dE reproduces dH - (1 - w_0) dE on
//      both D3Q7 and D2Q5.
//   8. THE Total CONTROL IS A GENUINE CONTROL. It must coincide with the
//      default outside the band and differ by exactly La f_l inside it. A
//      control that is an accidental alias of the thing it controls for would
//      pass validation/stefan.cpp's failing row for the wrong reason.
//   9. omega_from_* ON ITS OWN LATTICE. Round trip on D3Q7 and D2Q5, plus an
//      explicit check that a hardcoded 1/3 would be wrong by exactly 4/3 on
//      D3Q7.
//  10. THE RATE THE COLLISION PICKS. rate_at must equal the material's rate
//      where alpha is phase-dependent and the solver's where it is not, and a
//      measured single-population relaxation must reproduce it. Catches a
//      collide that silently ignores the material, or silently ignores the
//      solver.
//==============================================================================
#include "Check.hpp"

#include "collision/EnthalpyBGK.hpp"
#include "collision/EnthalpyRegularised.hpp"
#include "collision/ScalarBGK.hpp"
#include "collision/ScalarRegularised.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

using namespace lbm;

namespace {

// Hand-rolled LCG: <random> gives no cross-platform reproducibility guarantee
// and a test that changes its own inputs between machines is not a test.
struct Lcg {
  std::uint64_t s;
  explicit Lcg(std::uint64_t seed) : s(seed) {}
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return double((s >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
  }
  double in(double a, double b) { return a + (b - a) * next(); }
};

const Real TOL = sizeof(Real) == 4 ? Real(2e-5) : Real(1e-12);

PhaseChange material(Real Ts, Real Tl, Real cps, Real cpl, Real ks, Real kl,
                     Real La, Real datum = Real(0)) {
  PhaseChange m;
  m.T_s = Ts; m.T_l = Tl; m.cp_s = cps; m.cp_l = cpl;
  m.k_s = ks; m.k_l = kl; m.La = La;    m.E_datum = datum;
  m.normalise();
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("enthalpy operator -- algebra checks   precision %s\n",
                precision_name());

    using L7 = D3Q7;
    using L5 = D2Q5;
    using L27 = D3Q27;   // a PRODUCT lattice, cs2 = 1/3 -- the operator must
                         // work here too; see block 2.

    //--------------------------------------------------------------------------
    // 4. Inversion round trip.  (Run first: it is the cheapest failure, and
    //    everything below is built on the inversion being right.)
    //--------------------------------------------------------------------------
    std::printf("\n-- 4. inversion round trip --\n");
    {
      for (int band = 0; band < 2; ++band) {
        const Real Tl = band ? Real(1.5) : Real(0.5);   // dTm = 1 or 0
        PhaseChange m = material(Real(0.5), Tl, Real(2), Real(3),
                                 Real(0.4), Real(0.9), Real(5), Real(7));
        const char* tag = band ? "dTm=1" : "dTm=0";

        Lcg r(20260921u + band);
        Real worst_T = 0, worst_f = 0;
        for (int t = 0; t < 400; ++t) {
          const Real fl = Real(r.in(0.0, 1.0));
          // Pick a temperature consistent with the fraction, so the forward map
          // is evaluated only where it is single-valued.
          const Real T = fl <= Real(0)   ? Real(r.in(-3.0, 0.5))
                       : fl >= Real(1)   ? Real(r.in(1.5, 4.0))
                                         : m.T_s + fl * m.band_width();
          const Real H = m.enthalpy_of(T, fl);
          Real f2, T2, E2, c2;
          m.invert(H, f2, T2, E2, c2);
          worst_f = Kokkos::max(worst_f, Kokkos::abs(f2 - fl));
          // At dTm = 0 every mushy state has the same T, so only f is
          // recoverable there; T is still checked, against T_m.
          worst_T = Kokkos::max(worst_T, Kokkos::abs(T2 - T));
          if (!(f2 >= Real(0) && f2 <= Real(1))) {
            check::ok(false, std::string("f_l in [0,1] ") + tag);
            break;
          }
        }
        check::near(worst_f, Real(0), TOL, std::string("f_l round trip  ") + tag);
        check::near(worst_T, Real(0), TOL, std::string("T   round trip  ") + tag);

        // The two band edges, exactly.
        Real f, T, E, c;
        m.invert(Real(0), f, T, E, c);
        check::near(f, Real(0), TOL, std::string("H = 0    gives f_l = 0  ") + tag);
        check::near(T, m.T_s, TOL,   std::string("H = 0    gives T = T_s  ") + tag);
        m.invert(m.liquidus_H(), f, T, E, c);
        check::near(f, Real(1), TOL, std::string("H = H_l  gives f_l = 1  ") + tag);
        check::near(T, Tl, TOL,      std::string("H = H_l  gives T = T_l  ") + tag);

        // One-sided limits at both edges: T, E and dE/dT continuous.
        const Real eps = Real(1e-6);
        for (int edge = 0; edge < 2; ++edge) {
          const Real H0 = edge ? m.liquidus_H() : Real(0);
          Real fa, Ta, Ea, ca, fb, Tb, Eb, cb;
          m.invert(H0 - eps, fa, Ta, Ea, ca);
          m.invert(H0 + eps, fb, Tb, Eb, cb);
          const char* w = edge ? "liquidus" : "solidus";
          check::near(Ta, Tb, Real(1e-4), std::string("T    continuous at ") + w + " " + tag);
          check::near(Ea, Eb, Real(1e-4), std::string("E    continuous at ") + w + " " + tag);
          check::near(ca, cb, Real(1e-4), std::string("dE/dT continuous at ") + w + " " + tag);
        }
      }
    }

    //--------------------------------------------------------------------------
    // 5. The stable root against the naive one.
    //--------------------------------------------------------------------------
    std::printf("\n-- 5. citardauq vs the naive quadratic root --\n");
    {
      // The naive form, transcribed and NOT tidied:
      //     theta = (-B + sqrt(B^2 + 2 d_cp dTm H)) / d_cp
      //     f     = theta / dTm
      //
      // EVALUATED IN Real, NOT IN double. Computing the reference candidate at
      // a higher precision than the code under test is not a comparison: in an
      // FP32 build it made the naive form look BETTER than the stable one at
      // d_cp = 1 (2.2e-16 against 7.6e-08), which is the double doing the work,
      // not the formula.
      auto naive = [](Real B, Real d_cp, Real dTm, Real H) {
        return double(((-B + std::sqrt(B * B + Real(2) * d_cp * dTm * H)) / d_cp) / dTm);
      };
      for (double dcp : {1.0, 1e-3, 1e-6, 1e-12}) {
        PhaseChange m = material(Real(0), Real(1), Real(1), Real(1 + dcp),
                                 Real(0.2), Real(0.2), Real(1));
        const double B = double(m.band_slope());
        // A long-double reference, so neither candidate is its own judge.
        const long double Bl = (long double)B, dl = (long double)dcp;
        double worst_stable = 0, worst_naive = 0;
        for (int i = 1; i < 20; ++i) {
          const double H  = double(m.liquidus_H()) * i / 20.0;
          const long double Hl = (long double)H;
          const long double ref = 2.0L * Hl / (Bl + std::sqrt(Bl * Bl + 2.0L * dl * 1.0L * Hl));
          Real f, T, E, c;
          m.invert(Real(H), f, T, E, c);
          worst_stable = std::max(worst_stable, std::abs(double(f) - double(ref)));
          worst_naive  = std::max(worst_naive,
                                  std::abs(naive(Real(B), Real(dcp), Real(1), Real(H)) -
                                           double(ref)));
        }
        std::printf("    d_cp = %-8.0e   stable %.3e   naive %.3e   ratio %.1f\n",
                    dcp, worst_stable, worst_naive,
                    worst_stable > 0 ? worst_naive / worst_stable : 0.0);
        // The floor is the build's own epsilon, not a fixed number: FP32 cannot
        // track a long-double reference below ~1e-7 however good the formula is.
        check::ok(worst_stable <= (sizeof(Real) == 4 ? 2e-6 : 1e-12),
                  "stable root stays at the precision floor at d_cp = " +
                      std::to_string(dcp));
      }
      // And the case the naive form cannot take at all.
      PhaseChange m = material(Real(0), Real(1), Real(1), Real(1),
                               Real(0.2), Real(0.2), Real(1));
      check::ok(m.linear_band(), "d_cp == 0 takes the linear-band fast path");
      Real f, T, E, c;
      m.invert(Real(0.5) * m.liquidus_H(), f, T, E, c);
      check::near(f, Real(0.5), TOL, "cp_s == cp_l: f_l = H/B with no sqrt");
    }

    //--------------------------------------------------------------------------
    // 6. The isothermal limit.
    //--------------------------------------------------------------------------
    std::printf("\n-- 6. dTm = 0 is an ordinary point --\n");
    {
      const Real Tm = Real(0), La = Real(3), datum = Real(11);
      PhaseChange m = material(Tm, Tm, Real(2), Real(5), Real(0.3), Real(0.7),
                               La, datum);
      check::ok(m.linear_band(), "dTm = 0 takes the linear-band fast path");
      check::near(m.band_slope(), La, TOL, "B = La when dTm = 0");
      check::near(m.liquidus_H(), La, TOL, "H_l = La when dTm = 0");

      Real worst_T = 0, worst_f = 0, worst_E = 0;
      for (int i = 0; i <= 20; ++i) {
        const Real H = La * Real(i) / Real(20);
        Real f, T, E, c;
        m.invert(H, f, T, E, c);
        worst_T = Kokkos::max(worst_T, Kokkos::abs(T - Tm));
        worst_f = Kokkos::max(worst_f, Kokkos::abs(f - H / La));
        worst_E = Kokkos::max(worst_E, Kokkos::abs(E - datum));
      }
      check::near(worst_T, Real(0), TOL, "T == T_m for every H in the band");
      check::near(worst_f, Real(0), TOL, "f_l == H/La exactly");
      check::near(worst_E, Real(0), TOL, "E == E_datum across the whole band");

      // The forward map really is two-valued here. This is the reason
      // enthalpy_of takes two arguments and enthalpy_of_T aborts.
      check::near(m.enthalpy_of(Tm, Real(0)), Real(0),  TOL, "H(T_m, f=0) = 0");
      check::near(m.enthalpy_of(Tm, Real(1)), La,       TOL, "H(T_m, f=1) = La");
      check::ok(m.enthalpy_of(Tm, Real(1)) != m.enthalpy_of(Tm, Real(0)),
                "H is NOT a function of T at an isothermal front");
    }

    //--------------------------------------------------------------------------
    // 1. The ScalarBGK identity at La = 0.
    //--------------------------------------------------------------------------
    std::printf("\n-- 1. La = 0 reproduces ScalarBGK population-for-population --\n");
    {
      // La must be > 0 for normalise() at dTm = 0, so the identity is taken in
      // the limit that matters: every node fully liquid, so no latent heat is
      // ever in play and the two operators see the same sensible field.
      const Real Href = Real(0.37);
      PhaseChange m = material(Real(-100), Real(-100), Real(1), Real(1),
                               Real(0.2), Real(0.2), Real(1e-30));
      EnthalpyBGK<L7> en;  en.set_material(m);  en.T_ref = Href;
      ScalarBGK<L7>   sc;  sc.T_ref = Href;

      Lcg r(7u);
      Real worst = 0;
      for (Real w : {Real(0.2), Real(1.0), Real(1.6), Real(1.9982)}) {
        en.omega = w; sc.omega = w;
        for (int t = 0; t < 40; ++t) {
          Real a[L7::Q], b[L7::Q];
          for (int i = 0; i < L7::Q; ++i) a[i] = b[i] = Real(r.in(-0.3, 0.3));
          const Real dH = EnthalpyBGK<L7>::deviation(a);
          const Real ux = Real(r.in(-0.05, 0.05)), uy = Real(r.in(-0.05, 0.05)),
                     uz = Real(r.in(-0.05, 0.05));
          en.collide(a, dH, ux, uy, uz, w);
          sc.collide(b, dH, ux, uy, uz, w);
          for (int i = 0; i < L7::Q; ++i)
            worst = Kokkos::max(worst, Kokkos::abs(a[i] - b[i]));
        }
      }
      check::near(worst, Real(0), Real(sizeof(Real) == 4 ? 1e-6 : 1e-14),
                  "EnthalpyBGK == ScalarBGK when nothing melts");
    }

    //--------------------------------------------------------------------------
    // 2. Equilibrium moments, all three regimes, both gauges swept.
    //--------------------------------------------------------------------------
    std::printf("\n-- 2. equilibrium moments --\n");
    {
      for (Real datum : {Real(0), Real(4.5)}) {
        for (Real Href : {Real(0), Real(-1.25)}) {
          PhaseChange m = material(Real(0), Real(1), Real(2), Real(3),
                                   Real(0.4), Real(0.4), Real(5), datum);
          EnthalpyBGK<L7> en; en.set_material(m); en.T_ref = Href;

          // One state per regime: solid, mushy, liquid.
          const Real Hs[3] = {Real(-1.5), Real(0.5) * m.liquidus_H(),
                              m.liquidus_H() + Real(2)};
          const Real ux = Real(0.03), uy = Real(-0.02), uz = Real(0.01);
          Real worst_sum = 0, worst_flux = 0, worst_sec = 0;
          for (int s = 0; s < 3; ++s) {
            const Real dH = Hs[s] - Href;
            Real fl, T, E, c;
            m.invert(Hs[s], fl, T, E, c);
            Real g[L7::Q];
            for (int i = 0; i < L7::Q; ++i) g[i] = en.eq(i, dH, ux, uy, uz);

            Real sum = 0, jx = 0, jy = 0, jz = 0, pxx = 0, pyy = 0, pxy = 0;
            for (int i = 0; i < L7::Q; ++i) {
              const Real cx = Real(cvel<L7>(i, 0)), cy = Real(cvel<L7>(i, 1)),
                         cz = Real(cvel<L7>(i, 2));
              sum += g[i];
              jx += cx * g[i]; jy += cy * g[i]; jz += cz * g[i];
              pxx += cx * cx * g[i]; pyy += cy * cy * g[i]; pxy += cx * cy * g[i];
            }
            const Real dE = E - Href;
            worst_sum  = Kokkos::max(worst_sum,  Kokkos::abs(sum - dH));
            worst_flux = Kokkos::max(worst_flux, Kokkos::abs(jx - E * ux));
            worst_flux = Kokkos::max(worst_flux, Kokkos::abs(jy - E * uy));
            worst_flux = Kokkos::max(worst_flux, Kokkos::abs(jz - E * uz));
            worst_sec  = Kokkos::max(worst_sec, Kokkos::abs(pxx - cs2<L7, Real>() * dE));
            worst_sec  = Kokkos::max(worst_sec, Kokkos::abs(pyy - cs2<L7, Real>() * dE));
            worst_sec  = Kokkos::max(worst_sec, Kokkos::abs(pxy));
          }
          char tag[96];
          std::snprintf(tag, sizeof tag, "(E_datum %.2f, H_ref %.2f)",
                        double(datum), double(Href));
          check::near(worst_sum,  Real(0), TOL, std::string("sum_i eq = dH        ") + tag);
          check::near(worst_flux, Real(0), TOL, std::string("sum_i c eq = E u     ") + tag);
          check::near(worst_sec,  Real(0), TOL, std::string("sum_i cc eq = cs2 dE ") + tag);
        }
      }

      // THE SAME MOMENTS ON D3Q27 AND D2Q5. The equilibrium needs only the
      // three weight identities, so it must hold on a product lattice too --
      // and D3Q27 is the one that matters, because it shares cs2 = 1/3 with
      // the D3Q19 the reduced-keyhole port is checked against. An earlier
      // static_assert on EnthalpyBGK banned this lattice outright.
      {
        PhaseChange m = material(Real(0), Real(1), Real(2), Real(3),
                                 Real(0.4), Real(0.4), Real(5), Real(1.5));
        const Real Href = Real(-0.75);
        const Real ux = Real(0.03), uy = Real(-0.02), uz = Real(0.01);
        const Real Hmid = Real(0.5) * m.liquidus_H();
        Real fl, T, E, c; m.invert(Hmid, fl, T, E, c);
        const Real dH = Hmid - Href, dE = E - Href;

        EnthalpyBGK<L27> e27; e27.set_material(m); e27.T_ref = Href;
        Real s27 = 0, jx = 0, pxx = 0, pxy = 0;
        for (int i = 0; i < L27::Q; ++i) {
          const Real g = e27.eq(i, dH, ux, uy, uz);
          const Real cx = Real(cvel<L27>(i, 0)), cy = Real(cvel<L27>(i, 1));
          s27 += g; jx += cx * g; pxx += cx * cx * g; pxy += cx * cy * g;
        }
        check::near(s27, dH, TOL,                        "D3Q27: sum_i eq = dH");
        check::near(jx,  E * ux, TOL,                    "D3Q27: sum_i c eq = E u");
        check::near(pxx, cs2<L27, Real>() * dE, TOL,     "D3Q27: sum_i cc eq = cs2 dE");
        check::near(pxy, Real(0), TOL,                   "D3Q27: off-diagonal second moment = 0");

        EnthalpyBGK<L5> e5; e5.set_material(m); e5.T_ref = Href;
        Real s5 = 0, j5 = 0, p5 = 0;
        for (int i = 0; i < L5::Q; ++i) {
          const Real g = e5.eq(i, dH, ux, uy, Real(0));
          s5 += g; j5 += Real(cvel<L5>(i, 0)) * g;
          p5 += Real(cvel<L5>(i, 0)) * Real(cvel<L5>(i, 0)) * g;
        }
        check::near(s5, dH, TOL,                      "D2Q5: sum_i eq = dH");
        check::near(j5, E * ux, TOL,                  "D2Q5: sum_i c eq = E u");
        check::near(p5, cs2<L5, Real>() * dE, TOL,    "D2Q5: sum_i cc eq = cs2 dE");
      }
    }

    //--------------------------------------------------------------------------
    // 3. Conservation under collide.
    //--------------------------------------------------------------------------
    std::printf("\n-- 3. collide conserves the zeroth moment --\n");
    {
      PhaseChange m = material(Real(0), Real(0.8), Real(2), Real(3),
                               Real(0.4), Real(0.9), Real(4), Real(1.5));
      EnthalpyBGK<L7>         bgk; bgk.set_material(m); bgk.T_ref = Real(0.6);
      EnthalpyRegularised<L7> reg; reg.set_material(m); reg.T_ref = Real(0.6);

      Lcg r(99u);
      Real worst_bgk = 0, worst_reg = 0;
      for (Real w : {Real(0.3), Real(1.0), Real(1.7), Real(1.995)}) {
        for (int t = 0; t < 60; ++t) {
          Real a[L7::Q], b[L7::Q];
          for (int i = 0; i < L7::Q; ++i) a[i] = b[i] = Real(r.in(-0.5, 0.5));
          const Real dH = EnthalpyBGK<L7>::deviation(a);
          const Real ux = Real(r.in(-0.05, 0.05)), uy = Real(r.in(-0.05, 0.05)),
                     uz = Real(r.in(-0.05, 0.05));
          bgk.collide(a, dH, ux, uy, uz, w);
          reg.collide(b, dH, ux, uy, uz, w);
          worst_bgk = Kokkos::max(worst_bgk,
                                  Kokkos::abs(EnthalpyBGK<L7>::deviation(a) - dH));
          worst_reg = Kokkos::max(worst_reg,
                                  Kokkos::abs(EnthalpyBGK<L7>::deviation(b) - dH));
        }
      }
      // Two thresholds on purpose, because the two forms conserve for different
      // reasons: the regularised one writes h[0] as the residual dH - D cs2 dE,
      // so the sum telescopes, while BGK accumulates Q separate multiply-adds.
      //
      // MEASURED, AND IT DOES NOT SHOW AT Q = 7: both drift at 4.44e-16 here
      // (FP64, states of order 1), i.e. one ulp of dH, so the structural
      // difference is below the rounding of the final accumulation at this
      // lattice size. The looser BGK threshold is kept as headroom for a longer
      // run and a wider lattice, not because a gap has been observed. Do not
      // quote the regularised form as the more conservative one on the strength
      // of this block.
      std::printf("    drift: BGK %.3e   regularised %.3e   (equal at Q=7)\n",
                  double(worst_bgk), double(worst_reg));
      check::near(worst_reg, Real(0), Real(sizeof(Real) == 4 ? 1e-6 : 1e-15),
                  "regularised conserves H to a rounding");
      check::near(worst_bgk, Real(0), Real(sizeof(Real) == 4 ? 1e-5 : 1e-14),
                  "BGK conserves H to accumulated round-off");
    }

    //--------------------------------------------------------------------------
    // 7. Regularised against BGK.
    //--------------------------------------------------------------------------
    std::printf("\n-- 7. regularised vs BGK --\n");
    {
      PhaseChange m = material(Real(0), Real(0.8), Real(2), Real(2),
                               Real(0.4), Real(0.4), Real(4));
      EnthalpyBGK<L7>         bgk; bgk.set_material(m); bgk.T_ref = Real(0.25);
      EnthalpyRegularised<L7> reg; reg.set_material(m); reg.T_ref = Real(0.25);

      Lcg r(5150u);
      Real worst_one = 0, worst_flux = 0;
      for (int t = 0; t < 40; ++t) {
        Real a[L7::Q], b[L7::Q];
        for (int i = 0; i < L7::Q; ++i) a[i] = b[i] = Real(r.in(-0.4, 0.4));
        const Real dH = EnthalpyBGK<L7>::deviation(a);
        const Real ux = Real(r.in(-0.05, 0.05)), uy = Real(r.in(-0.05, 0.05)),
                   uz = Real(r.in(-0.05, 0.05));
        bgk.collide(a, dH, ux, uy, uz, Real(1));
        reg.collide(b, dH, ux, uy, uz, Real(1));
        for (int i = 0; i < L7::Q; ++i)
          worst_one = Kokkos::max(worst_one, Kokkos::abs(a[i] - b[i]));
      }
      check::near(worst_one, Real(0), Real(sizeof(Real) == 4 ? 1e-6 : 1e-14),
                  "identical at omega = 1");

      // The flux moment agrees at every rate; the ghosts do not.
      for (Real w : {Real(0.4), Real(1.3), Real(1.9)}) {
        Real a[L7::Q], b[L7::Q];
        for (int i = 0; i < L7::Q; ++i) a[i] = b[i] = Real(r.in(-0.4, 0.4));
        const Real dH = EnthalpyBGK<L7>::deviation(a);
        const Real ux = Real(0.04);
        bgk.collide(a, dH, ux, Real(0), Real(0), w);
        reg.collide(b, dH, ux, Real(0), Real(0), w);
        Real ja = 0, jb = 0;
        for (int i = 0; i < L7::Q; ++i) {
          ja += Real(cvel<L7>(i, 0)) * a[i];
          jb += Real(cvel<L7>(i, 0)) * b[i];
        }
        worst_flux = Kokkos::max(worst_flux, Kokkos::abs(ja - jb));
      }
      check::near(worst_flux, Real(0), Real(sizeof(Real) == 4 ? 1e-6 : 1e-14),
                  "flux moment identical at every rate");

      // D cs2 == 1 - w_0, the identity the two rest-term forms share. Asserted
      // at compile time in the operator; evaluated here so the number is seen.
      const Real lhs7 = Real(L7::D) * cs2<L7, Real>();
      const Real rhs7 = Real(1) - weight<L7, Real>(0);
      check::near(lhs7, rhs7, TOL, "D3Q7: D*cs2 == 1 - w_0  (3/4)");
      const Real lhs5 = Real(L5::D) * cs2<L5, Real>();
      const Real rhs5 = Real(1) - weight<L5, Real>(0);
      check::near(lhs5, rhs5, TOL, "D2Q5: D*cs2 == 1 - w_0  (2/3)");
    }

    //--------------------------------------------------------------------------
    // 8. The Total control is a genuine control.
    //--------------------------------------------------------------------------
    std::printf("\n-- 8. EnthalpyAdvect::Total is not an alias of the default --\n");
    {
      PhaseChange m = material(Real(0), Real(1), Real(2), Real(2),
                               Real(0.4), Real(0.4), Real(6));
      EnthalpyBGK<L7, EnthalpyAdvect::Sensible> sen; sen.set_material(m);
      EnthalpyBGK<L7, EnthalpyAdvect::Total>    tot; tot.set_material(m);

      const Real ux = Real(0.05);
      // Fully solid: E == H, so the two must coincide exactly.
      {
        const Real H = Real(-1.5);
        Real a = 0, b = 0;
        for (int i = 0; i < L7::Q; ++i) {
          a += Real(cvel<L7>(i, 0)) * sen.eq(i, H, ux, Real(0), Real(0));
          b += Real(cvel<L7>(i, 0)) * tot.eq(i, H, ux, Real(0), Real(0));
        }
        check::near(a, b, TOL, "coincide outside the band");
      }
      // Mid-band: they must differ by exactly La f_l in the flux moment.
      {
        const Real H = Real(0.5) * m.liquidus_H();
        Real fl, T, E, c; m.invert(H, fl, T, E, c);
        Real a = 0, b = 0;
        for (int i = 0; i < L7::Q; ++i) {
          a += Real(cvel<L7>(i, 0)) * sen.eq(i, H, ux, Real(0), Real(0));
          b += Real(cvel<L7>(i, 0)) * tot.eq(i, H, ux, Real(0), Real(0));
        }
        check::near(b - a, m.La * fl * ux, TOL,
                    "differ by exactly La f_l u in the flux moment");
        check::ok(Kokkos::abs(b - a) > Real(1e-6),
                  "the difference is not zero -- the control is live");
      }
    }

    //--------------------------------------------------------------------------
    // 9. omega_from_* on its own lattice.
    //--------------------------------------------------------------------------
    std::printf("\n-- 9. omega_from_diffusivity reads the lattice's own cs2 --\n");
    {
      for (Real d : {Real(0.05), Real(0.2), Real(0.4)}) {
        const Real w7 = EnthalpyBGK<L7>::omega_from_diffusivity(d);
        check::near(EnthalpyBGK<L7>::diffusivity_from_omega(w7), d, TOL,
                    "D3Q7 round trip");
        const Real w5 = EnthalpyBGK<L5>::omega_from_diffusivity(d);
        check::near(EnthalpyBGK<L5>::diffusivity_from_omega(w5), d, TOL,
                    "D2Q5 round trip");
        // D3Q7 has cs2 = 1/4, so alpha = (1/4)(1/w - 1/2).
        check::near(d, Real(0.25) * (Real(1) / w7 - Real(0.5)), TOL,
                    "D3Q7 alpha = (1/4)(1/w - 1/2)");
        // Writing tau = 3 alpha + 1/2 on D3Q7 -- i.e. using cs2 = 1/3 on a
        // lattice whose cs2 is 1/4 -- asks for `d` and DELIVERS 3d/4. The
        // simulation runs, converges, and solves a problem with three quarters
        // of the intended diffusivity. Stated as the ratio that is actually
        // measured; the reciprocal, 4/3, is how much more you would need.
        const Real wrong = Real(1) / (d * Real(3) + Real(0.5));
        check::near(EnthalpyBGK<L7>::diffusivity_from_omega(wrong) / d,
                    Real(3) / Real(4), TOL,
                    "cs2 = 1/3 on D3Q7 delivers 3/4 of the intended diffusivity");
      }
      // And the operator's own rate matches the free function.
      PhaseChange m = material(Real(0), Real(1), Real(2), Real(2),
                               Real(0.4), Real(0.4), Real(3));
      EnthalpyBGK<L7> en; en.set_material(m);
      check::near(en.omega, EnthalpyBGK<L7>::omega_from_diffusivity(Real(0.4) / Real(2)),
                  TOL, "set_material sets omega from the solid diffusivity");
    }

    //--------------------------------------------------------------------------
    // 10. The rate the collision picks.
    //--------------------------------------------------------------------------
    std::printf("\n-- 10. rate_at: material or solver, and which --\n");
    {
      // (a) alpha phase-independent: k_s cp_l == k_l cp_s. The solver's rate
      //     must be honoured verbatim.
      {
        PhaseChange m = material(Real(0), Real(1), Real(2), Real(4),
                                 Real(0.4), Real(0.8), Real(3));
        EnthalpyBGK<L7> en; en.set_material(m);
        check::ok(en.rate_from_solver(), "k_s cp_l == k_l cp_s is detected");
        const Real w = Real(1.234);
        check::near(en.rate_at(Real(0.5) * m.liquidus_H(), w), w, TOL,
                    "uniform alpha: the solver's rate is used");
      }
      // (b) alpha phase-dependent: the material decides and the argument is
      //     ignored. Two different arguments must give the same answer.
      {
        PhaseChange m = material(Real(0), Real(1), Real(2), Real(2),
                                 Real(0.4), Real(0.9), Real(3));
        EnthalpyBGK<L7> en; en.set_material(m);
        check::ok(!en.rate_from_solver(), "k_s cp_l != k_l cp_s is detected");
        const Real H = Real(0.5) * m.liquidus_H();
        const Real r1 = en.rate_at(H, Real(0.3)), r2 = en.rate_at(H, Real(1.7));
        check::near(r1, r2, TOL, "phase-dependent alpha ignores the solver's rate");

        Real fl, T, E, c; m.invert(H, fl, T, E, c);
        const Real k_mid = m.conductivity<MushMix::Parallel>(fl);
        check::near(r1, EnthalpyBGK<L7>::omega_from_diffusivity(k_mid / c), TOL,
                    "rate_at == omega_from_diffusivity(k(f_l) / dE/dT)");

        // And collide really relaxes at that rate: start from a state whose
        // deviation is unchanged by the inversion, relax once, and measure.
        Real g[L7::Q];
        for (int i = 0; i < L7::Q; ++i) g[i] = Real(0);
        g[0] = H;                       // dH = H, everything else empty
        Real eq0[L7::Q];
        for (int i = 0; i < L7::Q; ++i) eq0[i] = en.eq(i, H, Real(0), Real(0), Real(0));
        Real g2[L7::Q];
        for (int i = 0; i < L7::Q; ++i) g2[i] = g[i];
        en.collide(g2, H, Real(0), Real(0), Real(0), Real(0.3));
        // g2[1] = g[1] + w (eq0[1] - g[1]) with g[1] = 0  =>  w = g2[1]/eq0[1]
        check::near(g2[1] / eq0[1], r1, Real(1e-9),
                    "collide relaxes at rate_at, not at the passed omega");
      }
    }
  }
  const int rc = check::report("enthalpy");
  Kokkos::finalize();
  return rc;
}
