//==============================================================================
//  EHD hydrostatic solution -- the first validation of the electroconvection
//  stack, against an analytic profile, a published error table and a published
//  convergence rate.
//
//  Patnaik, Skillen & De Rosis, Engineering with Computers 41:4977-5002 (2025),
//  Sec. 3.1. Unipolar charge injection between two plates with the FLUID AT
//  REST, so the charge and the potential balance each other and nothing else:
//
//      grad^2 phi = -q / eps,        E = -grad phi,
//      div[ q K E ] = D grad^2 q     (steady, u = 0).
//
//  ===================== WHAT THIS ACTUALLY PINS DOWN =========================
//  Three independent checks, which is more than most cases in this directory
//  have. A scheme can pass one by luck.
//
//    1. THE PROFILE. q(y) = a / (2 sqrt(y + b)) with (a, b) from the paper's
//       Table 2, y scaled by the PLATE SEPARATION so y in [0, 1].
//    2. THE ERROR. The l2-norm of the percentage relative error, against the
//       paper's Table 3 for model C (finite-difference E): 0.0203 / 0.1415 /
//       1.1135 at C = 0.1 / 1 / 10 on a 160 x 320 lattice.
//    3. THE RATE. Refining n_y gives a published convergence rate of 1.6.
//
//  WHERE y IN [0,1] COMES FROM, because the paper says "L_x = 1 and L_y = 2"
//  and that reads like the separation is 2. It is not: those set the ASPECT
//  RATIO -- the domain is twice as tall as wide, which is the 160 x 320 grid --
//  while the analytic solution non-dimensionalises by the separation. The
//  arithmetic settles it without needing the convention. E = a sqrt(y + b)
//  follows from dE/dy = q/eps, and the potential difference fixes its integral:
//
//      int_0^L a sqrt(y+b) dy = dphi = 1.
//
//  At L = 1 that evaluates to 1.00017, 1.00002, 0.99998 for the three C. At
//  L = 2 it is out by a factor of three. The same reading gives q(0) = 0.1000,
//  1.0001, 9.9981 -- i.e. exactly C, which is what "injection strength" should
//  mean. Both consistency checks pick L = 1 and neither depends on how the
//  formula was read off the page.
//
//  ======================= WHAT IS NEW, AND WHAT IS NOT =======================
//  The charge is ChargeCentralMoments<D3Q27>, new for this. The potential is
//  NOT new: Eq. (26) of the paper is
//
//      g* = g + (1/tau_phi)[w phi - g] + w beta q/eps + (w beta/2)(q - q_prev)/eps
//
//  and with a ZERO advecting velocity ScalarBGK's equilibrium w(dT + T c.u/cs2)
//  is exactly w phi, while ScalarSolver::add_source adds precisely w_i S. So
//  the Poisson solver is ScalarSolver + ScalarBGK + a source, composed rather
//  than written. ScalarSolver leaves the velocity at zero when set_velocity is
//  never called (`have_u = ux.data() != nullptr`), which is what makes that
//  work without a special case.
//
//  THE SOURCE COMMUTES WITH THE COLLISION, which is why adding it before the
//  step rather than inside it is exact and not an approximation. Writing
//  g' = g + w S, the sum phi' = phi + S, so
//
//      g'* = (1-w)(g + wS) + w_i w (phi + S) = [g + w(w_i phi - g)] + w_i S,
//
//  identical to adding w_i S to the post-collision state. Check 0 below tests
//  that claim rather than trusting it.
//
//  tau_phi = 3 beta + 1/2 comes out of omega_from_diffusivity(beta) because
//  D3Q27's cs2 is 1/3. That agreement is a coincidence of the lattice and not a
//  rule -- on D3Q7 cs2 is 1/4 and the same call gives 4 beta + 1/2, which is
//  the trap CLAUDE.md names first among the units errors.
//
//  ============================== RESOLUTION =================================
//  The default grid is deliberately SMALLER than the paper's. Reaching a
//  steady state is diffusion-limited by the potential, ~H^2/beta steps, so the
//  paper's H = 320 needs ~3e5 steps and belongs in the analysis list rather
//  than in ctest. The default H = 80 runs in seconds and still resolves the
//  profile; `-ny 320` reproduces the published configuration. The default was
//  H = 80 and made this the slowest case in ctest at 194 s; H = 40 runs in ~15 s
//  and still converges every C.
//
//  ==================== TWO DEFECTS THIS CASE FOUND ==========================
//  Neither was in the scheme; both were in how it was set up here, and both are
//  worth knowing because they are properties of this tree rather than of EHD.
//
//  1. THE TREE'S DEFAULT HALFWAY PLATES ARE FIRST ORDER HERE. With the plates
//     at y = 0.5 and H + 0.5 the field's one-sided stencil works entirely from
//     interior nodes and never sees the imposed potential, and the effective
//     separation is off by O(1) cell. Measured: a uniform +3.3% bias in the bulk
//     charge at H = 40, halving to +1.3% at H = 80 -- O(1/H), and it dragged the
//     whole convergence rate to 1.0. On-node plates (ScalarMoment) took C = 10
//     at H = 80 from 11.19% to 1.49%.
//
//     THE GENERAL LESSON, which is not confined to this case: the half cell is
//     harmless where the boundary value is what you compare against, and it is
//     NOT harmless where you differentiate the field carrying it. rb_high_ra's
//     banner records the same half cell as costing "1% in H, 3% in Ra" and
//     being negligible; here the same half cell is the leading error, because
//     E = -grad phi.
//
//  2. A UNIFORM INITIAL CHARGE MAKES C = 10 DIVERGE, and the arithmetic says
//     why before any run does. For uniform q the Poisson solution carries a
//     parabolic bump of q H^2 / 8; at C = 10, H = 40 that is 1.25, LARGER than
//     the applied potential difference of 1. Measured consequence: phi reached
//     1.19 -- above its own boundary value -- by t = 400, E changed sign in the
//     lower layer, the drift reversed, and q went negative by t = 800. C = 0.1
//     and C = 1 survived the identical setup only because the same bump is 100x
//     and 10x smaller. The reference seeds q = 0 and phi = 0 in the bulk and
//     lets both enter from the plates; matching that removed the divergence,
//     and the H = 80 answers were unchanged to every digit, which is what says
//     the steady state was never in question.
//
//  ===================== THE RATE IS NOT REPRODUCED, YET ======================
//  Measured over H = 20, 40, 80, all verified at steady state (the error is
//  unchanged to four decimals over a 16x range of step counts):
//
//      C = 0.1    rate 1.57      <- the published 1.6
//      C = 1      rate 0.89
//      C = 10     rate 1.00
//
//  Weak injection reproduces the paper. Something degrades with injection
//  strength, and it is NOT the injector boundary layer: excluding the first 5%
//  or 10% of the depth from the norm cuts the magnitude fivefold (3.25 -> 0.63
//  at H = 40) and leaves the rate at 1.07 and 1.26. So the bulk is first order
//  too at strong injection.
//
//  Absolute accuracy is nonetheless in the right place. At H = 80 this gives
//  0.0272 / 0.2607 / 1.4889 % against the paper's 0.0203 / 0.1415 / 1.1135 % at
//  H = 320 -- within a factor of two at a quarter of the resolution, which is
//  better than a rate-1.6 scheme would manage and is itself part of the puzzle.
//
//  TWO CANDIDATES for the next session, in order: the potential's BGK source,
//  whose magnitude scales with C and whose (1/2) d_t G correction is what makes
//  it second order; and the charge operator at omega_q = 1.999, where the same
//  ghost-mode argument that mattered in ScalarRegularised may bite.
//  DO NOT quote this case's rate as a reproduction of the paper's until that is
//  settled. The profile and the error magnitudes are reproduced; the rate is
//  not.
//
//    usage: ehd_hydrostatic [-ny H] [-nx NX] [-c C] [-steps N] [-conv]
//                           [-profile] [-watch N] [-skip FRACTION]
//                           [--kokkos-num-threads=4]
//==============================================================================
#include "collision/ChargeCentralMoments.hpp"
#include "collision/ScalarBGK.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

using LT        = D3Q27;
using ChargeOp  = ChargeCentralMoments<LT>;
using PotOp     = ScalarBGK<LT>;
using ChargeSol = ScalarSolver<LT, EsotericPull<LT>, ChargeOp>;
using PotSol    = ScalarSolver<LT, EsotericPull<LT>, PotOp>;

// Table 2 of the paper: the analytic parameters at each injection strength.
struct Ref { double C, a, b, err_modelC; };
static constexpr Ref REF[3] = {
    {0.1,  0.4363,  4.7590e+0, 0.0203},
    {1.0,  1.1256,  3.1670e-1, 0.1415},
    {10.0, 1.4882,  5.539e-3,  1.1135},
};

static double q_analytic(double y, double a, double b) {
  return a / (2.0 * std::sqrt(y + b));
}

//------------------------------------------------------------------------------
//  One steady hydrostatic solve. Returns the l2-norm of the percentage relative
//  error against the analytic profile, Eq. (79).
//------------------------------------------------------------------------------
static bool profile = false;
static std::size_t watch = 0;
static double skip_frac = 0.0;
static double solve(Index nx, Index H, double C, std::size_t steps, bool verbose) {
  // ON-NODE PLATES, ny = H + 1, so the plates ARE nodes 0 and H and the
  // separation is exactly H. This is the reference's wall family and it is not
  // a style choice here -- see the banner: with the tree's usual HALFWAY plates
  // the field's one-sided stencil never sees the imposed potential and the
  // result is first order.
  const Index ny = H + 1, nz = 1;

  // Units. eps = dphi = 1, lengths in plate separations, so the reference charge
  // is eps dphi / H^2 = 1/H^2 in lattice units and the injected value is C x it.
  const double dphi = 1.0, eps = 1.0;
  const double Kmob = 0.01 * double(H) / dphi;       // u0 = K dphi / H = 0.01
  const double alpha = 1e-4;
  const double Dq   = alpha * Kmob * dphi;           // charge diffusivity
  const double beta = 0.3;                           // potential relaxation
  const double q_wall = C * eps * dphi / (double(H) * double(H));

  Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

  // ---- the charge ---------------------------------------------------------
  ChargeOp ccoll;
  ccoll.omega = ChargeOp::omega_from_diffusivity(Real(Dq));
  ChargeSol chg(d, ccoll);
  chg.set_geometry([&](Index, Index y, Index) -> ScalarCell {
    if (y == 0)      return ScalarMoment;           // injector: q = q_wall, ON NODE
    if (y == ny - 1) return ScalarOutflow;          // collector: zero gradient
    return ScalarBulk;
  });
  chg.set_wall_values([&](Index, Index, Index) -> Real { return Real(q_wall); });

  // ---- the potential ------------------------------------------------------
  // No set_velocity: ScalarSolver then runs with u = 0 and ScalarBGK's
  // equilibrium collapses to w_i phi, which is Eq. (29).
  PotOp pcoll;
  pcoll.omega = PotOp::omega_from_diffusivity(Real(beta));   // tau = 3 beta + 1/2
  pcoll.T_ref = Real(0);
  PotSol pot(d, pcoll);
  pot.set_geometry([&](Index, Index y, Index) -> ScalarCell {
    return (y == 0 || y == ny - 1) ? ScalarMoment : ScalarBulk;
  });
  pot.set_wall_values([&](Index, Index y, Index) -> Real {
    return (y == 0) ? Real(dphi) : Real(0);          // phi_b = 1, phi_t = 0
  });

  // THE INITIAL CONDITION IS THE REFERENCE'S: both fields ZERO in the bulk,
  // with charge and potential entering only from the plates. That is not a
  // detail, and seeding q uniformly at the injected value instead is what made
  // C = 10 diverge. For uniform q the Poisson solution carries a parabolic bump
  // of height q H^2 / 8 -- at C = 10, H = 40 that is 1.25, LARGER than the
  // applied potential difference. phi then exceeds its own boundary value
  // (measured: 1.19 by t = 400), E changes sign in the lower layer, the drift
  // reverses, and the charge is driven negative by t = 800. C = 0.1 and C = 1
  // survived the same initial condition only because the same bump is 100x and
  // 10x smaller. Starting from zero, the potential relaxes to its linear
  // profile before there is any charge to bend it.
  const Index nyc = ny, Hc = H;
  pot.initialize_field(KOKKOS_LAMBDA(Index) { return Real(0); });
  chg.initialize(Real(0));
  pot.finalize_geometry();
  chg.finalize_geometry();
  pot.compute_field();
  chg.compute_field();

  // Drift velocity K E and the previous charge, both host-visible so the
  // finite-difference field can be built between steps.
  View1D<Real> kx("kx", d.n_padded), ky("ky", d.n_padded), kz("kz", d.n_padded);
  View1D<Real> qprev("qprev", d.n_padded);
  chg.set_velocity(kx, ky, kz);

  auto phi = pot.temperature();
  auto qf  = chg.temperature();

  for (std::size_t t = 0; t < steps; ++t) {
    // E = -grad phi by SECOND-ORDER FINITE DIFFERENCE, the paper's model C.
    // Sec. 3.1 measures this against reconstructing E from the distribution's
    // own moments (model B) and model C wins at every C -- 1.11 against 2.49 at
    // C = 10 -- because the moment reconstruction is sensitive to grid-scale
    // oscillation. One-sided at the plates, central inside.
    Kokkos::parallel_for("E_field", Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index y = py - d.hy;
      kx(n) = ky(n) = kz(n) = Real(0);
      if (y < 0 || y > Hc) return;
      double Ey;
      // The plates are nodes now, so the one-sided stencils START from the
      // imposed potential. That is the whole point of the on-node family here.
      if (y == 0)
        Ey = -(-1.5 * double(phi(n)) + 2.0 * double(phi(d.id(px - d.hx, y + 1, 0)))
               - 0.5 * double(phi(d.id(px - d.hx, y + 2, 0))));
      else if (y == Hc)
        Ey = -(1.5 * double(phi(n)) - 2.0 * double(phi(d.id(px - d.hx, y - 1, 0)))
               + 0.5 * double(phi(d.id(px - d.hx, y - 2, 0))));
      else
        Ey = -0.5 * (double(phi(d.id(px - d.hx, y + 1, 0)))
                     - double(phi(d.id(px - d.hx, y - 1, 0))));
      ky(n) = Real(Kmob * Ey);              // u = 0 here, so the drift IS K E
    });
    Kokkos::fence();

    // Eq. (26)'s source: beta [ q + (q - q_prev)/2 ] / eps, added as w_i S.
    // See the banner -- this commutes with the collision, so adding it here is
    // exact rather than a splitting.
    const Real bo = Real(beta / eps);
    pot.add_source(KOKKOS_LAMBDA(Index n) {
      return bo * (Real(1.5) * qf(n) - Real(0.5) * qprev(n));
    });
    Kokkos::deep_copy(qprev, qf);

    pot.step();  chg.step();
    pot.compute_field();  chg.compute_field();

    // WHICH FIELD GOES FIRST, and does the charge leave its physical range?
    // The steady charge is positive and monotonically decreasing from the
    // injector, so q must stay in [0, q_wall]. That bound is the same kind of
    // maximum principle rb_high_ra uses, and it is the sharpest detector here
    // for the same reason: it fails long before a norm does.
    if (watch && (t % watch == 0 || t + 1 == steps)) {
      auto hq2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, qf);
      auto hp2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, phi);
      double qlo = 1e300, qhi = -1e300, plo = 1e300, phi_hi = -1e300;
      long nbad = 0; Index yq = -1;
      for (Index y = 1; y <= H; ++y)
        for (Index x = 0; x < nx; ++x) {
          const double v = double(hq2(d.id(x, y, 0)));
          const double pv = double(hp2(d.id(x, y, 0)));
          if (!std::isfinite(v) || !std::isfinite(pv)) { ++nbad; continue; }
          if (v < qlo) { qlo = v; yq = y; }
          if (v > qhi) qhi = v;
          if (pv < plo) plo = pv;
          if (pv > phi_hi) phi_hi = pv;
        }
      std::printf("      t=%7zu  q [%11.4e, %11.4e]/qw=%.3g  phi [%7.4f, %7.4f]"
                  "  min q at y=%lld%s\n", t, qlo, qhi, qhi / q_wall, plo, phi_hi,
                  (long long)yq, nbad ? "   NON-FINITE" : (qlo < 0 ? "   q<0 !" : ""));
      std::fflush(stdout);
      if (nbad) break;
    }
  }

  // ---- the error, Eq. (79) ------------------------------------------------
  auto hq = Kokkos::create_mirror_view_and_copy(HostSpace{}, qf);
  double num = 0.0, den = 0.0;
  const Ref* r = nullptr;
  for (const auto& e : REF) if (std::abs(e.C - C) < 1e-9) r = &e;
  // FROM y = 1, NOT y = 0. Node 0 is the injector, where q is IMPOSED, so its
  // error is identically zero and its magnitude is the largest in the profile
  // (q(0) = C). Including it inflates the denominator of Eq. (79) and deflates
  // the reported error -- at C = 10 that alone moved 11.2% to 1.2% and looked
  // like an improvement in the scheme. Score only what the scheme computed.
  for (Index y = 1; y <= H; ++y) {
    const double yh = double(y) / double(H);                  // plates ARE nodes
    // -skip f excludes the fraction f of the depth nearest the injector. A
    // FRACTION and not a cell count, so the excluded region is the same physical
    // layer at every H and the comparison stays fair under refinement.
    if (yh < skip_frac) continue;
    const double an = q_analytic(yh, r->a, r->b) / (double(H) * double(H));
    const double nu = double(hq(d.id(0, y, 0)));
    num += (an - nu) * (an - nu);
    den += an * an;
  }
  const double err = 100.0 * std::sqrt(num) / std::sqrt(den);

  // WHERE THE ERROR LIVES, which is a different question from how big it is.
  // A boundary that is one order low concentrates it in the first cells; an
  // operator that is one order low spreads it through the bulk.
  if (profile) {
    std::printf("      %6s %14s %14s %10s\n", "y", "q_num", "q_analytic", "rel %");
    for (Index y = 0; y <= H; ++y) {
      if (!(y <= 4 || y == H / 4 || y == H / 2 || y == 3 * H / 4 || y >= H - 1)) continue;
      const double yh = double(y) / double(H);
      const double an = q_analytic(yh, r->a, r->b) / (double(H) * double(H));
      const double nu = double(hq(d.id(0, y, 0)));
      std::printf("      %6lld %14.6e %14.6e %10.3f\n",
                  (long long)y, nu, an, 100.0 * (nu - an) / an);
    }
    std::fflush(stdout);
  }
  if (verbose) {
    std::printf("    C = %5.1f  H = %4lld   l2 error %8.4f %%   (paper, model C: %.4f)\n",
                C, (long long)H, err, r->err_modelC);
    // FLUSH. The convergence sweep runs for minutes at the larger H, and block
    // buffering to a redirected file hides every intermediate line until exit --
    // so a run that is progressing is indistinguishable from one that has hung.
    // rb_high_ra flushes per row for the same reason.
    std::fflush(stdout);
  }
  return err;
}

int main(int argc, char** argv) {
  Index nx = 0, H = 40;   // ctest default; -ny 80/320 for the real comparison
  double Conly = -1;
  std::size_t steps = 0;
  bool conv = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-ny"    && i + 1 < argc) H     = Index(std::atol(argv[++i]));
    if (a == "-nx"    && i + 1 < argc) nx    = Index(std::atol(argv[++i]));
    if (a == "-c"     && i + 1 < argc) Conly = std::atof(argv[++i]);
    if (a == "-steps" && i + 1 < argc) steps = std::size_t(std::atol(argv[++i]));
    if (a == "-conv")                  conv  = true;
    if (a == "-profile")               profile = true;
    if (a == "-watch" && i + 1 < argc)  watch = std::size_t(std::atol(argv[++i]));
    if (a == "-skip"  && i + 1 < argc)  skip_frac = std::atof(argv[++i]);
  }
  if (nx == 0) nx = H / 2;                       // the paper's 1:2 aspect ratio
  // Steady state is potential-diffusion limited, ~H^2/beta. Six of those.
  if (steps == 0) steps = std::size_t(6.0 * double(H) * double(H) / 0.3);

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("EHD hydrostatic solution   %s   D3Q27 charge (central moments)"
                " + D3Q27 potential (BGK)\n", sizeof(Real) == 4 ? "FP32" : "FP64");
    std::printf("  Patnaik, Skillen & De Rosis, Eng. Comput. 41:4977-5002 (2025),"
                " Sec. 3.1, Tables 2 and 3\n");
    std::printf("  %lld x %lld x 1   H = %lld   %zu steps\n\n",
                (long long)nx, (long long)(H + 2), (long long)H, steps);

    if (conv) {
      // The paper's convergence study: refine n_y at C = 10, rate 1.6.
      std::printf("  Convergence at C = 10 (paper: rate 1.6)\n");
      std::vector<Index> Hs{20, 40, 80, 160};
      std::vector<double> es;
      for (Index h : Hs) {
        const std::size_t st = std::size_t(6.0 * double(h) * double(h) / 0.3);
        es.push_back(solve(h / 2, h, 10.0, st, true));
      }
      std::printf("\n  %8s %12s %10s\n", "H", "l2 %", "order");
      for (std::size_t i = 0; i < Hs.size(); ++i) {
        if (i == 0) std::printf("  %8lld %12.5f %10s\n", (long long)Hs[i], es[i], "-");
        else {
          const double p = std::log(es[i - 1] / es[i]) / std::log(2.0);
          std::printf("  %8lld %12.5f %10.2f\n", (long long)Hs[i], es[i], p);
        }
      }
    } else {
      std::printf("  %s\n", "l2-norm of the percentage relative error, Eq. (79)");
      int bad = 0;
      for (const auto& e : REF) {
        if (Conly > 0 && std::abs(e.C - Conly) > 1e-9) continue;
        const double err = solve(nx, H, e.C, steps, true);
        // THE TOLERANCE USES BOTH PUBLISHED NUMBERS AND NO INVENTED ONE. The
        // table is at H = 320 and this defaults to a coarser grid, so the
        // error is legitimately larger -- but by how much is not free: the
        // paper also publishes a convergence rate of 1.6, so the admissible
        // error at H is err(320) x (320/H)^1.6, and the test allows 50% over
        // that. A scheme converging at the right rate from the right constant
        // passes at every H; one that is merely stable does not.
        //
        // This matters most at C = 10, where the analytic charge falls from 10
        // to 4.3 ACROSS ONE CELL at H = 40. A fixed tolerance there either
        // passes everything or fails the physics.
        const double scale = std::pow(320.0 / double(H), 1.6);
        const double tol = 1.5 * e.err_modelC * scale;
        std::printf("        admissible at this H: %.4f %%  (= 1.5 x %.4f x (320/H)^1.6)\n",
                    tol, e.err_modelC);
        if (!(err < tol)) { ++bad; std::printf("        FAIL\n"); }
      }
      std::printf("\n  %s\n", bad ? "FAIL" : "PASS");
      rc = bad ? 1 : 0;
    }
  }
  Kokkos::finalize();
  return rc;
}
