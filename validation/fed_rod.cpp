//==============================================================================
//  A solid rod fed through a melt front, against its exact steady state -- the
//  first reference test of the ADVECTIVE half of the enthalpy operator, and the
//  case that decides which EnthalpyAdvect mode a moving solid needs.
//
//  THE PROBLEM. Material enters at x = 0 as solid at T_c and leaves at x = L as
//  liquid at T_h, at a uniform speed U: plug flow, as for a filament fed into a
//  hot end (demonstrator/mew_liquefier.cpp) or a strand through a caster. Both
//  ends are on-node Dirichlet walls (ScalarMoment), so the exact solution's
//  boundary points ARE nodes and no half cell is in play. In steady state the
//  total-enthalpy flux U H - k T' is a constant q; H(T) is piecewise linear, so
//  T is an exponential on each branch, and q is fixed by requiring the branches
//  to span L. At an isothermal front T' jumps by U La / k -- the Stefan
//  condition for a front the material CROSSES, rather than one that moves.
//  The reference is checked against itself before it judges anything: a second
//  derivation, closed form at an isothermal front, agrees with the shooting
//  solver to 1.1e-13 cells.
//
//  WHY THE DEFAULT OPERATOR CANNOT DO THIS, AND WHY IT IS NOT A DISCRETISATION
//  ERROR. EnthalpyAdvect::Sensible advects E, so in steady state it solves
//  div(E u) = div(k grad T), which has no La in it: the latent heat leaves the
//  problem. Row 3 measures exactly that. Material -- advect H, diffuse E -- is
//  the model for a moving solid; Total, which also DIFFUSES H, is wrong too.
//
//  MEASURED 2026-11-03, FP64, D3Q7, Threads backend. Pe = U L / alpha = 4,
//  tau = 1 (alpha = 1/8), T_c = -1, T_h = +1, La/c = 1 (Ste = 1), diffusive
//  scaling over L = 40, 80, 160, 320.
//
//      row                              value
//      0. La = 0, advective half        T rms 1.69e-4, 4.25e-5, 1.07e-5;
//                                       orders 1.99, 2.00. Material == Sensible
//                                       at every node, exactly
//      1. isothermal front, Material    EVERY NODE IN ITS EXACT PHASE, NONE
//                                       MUSHY, at every L: the liquid integral
//                                       is the exact one plus the sub-cell
//                                       offset x_f - ceil(x_f) + 1/2 (0.49,
//                                       0.48, 0.47, 0.43 cells), to 0.00e+00.
//                                       T rms first order (1.03, 1.05, 1.11)
//         uniqueness, L = 80            seeded upstream or downstream of the
//                                       exact front: l identical, T to 3.2e-12
//      2. band [-0.1, 0.1], Material    |dl| <= 0.026 cells at every L;
//                                       T rms 6.1e-4, 3.0e-4, 8.0e-5, 1.5e-5
//      3. controls, L = 80              Sensible: the La = 0 front to 5.5e-5
//                                       cells, the real one missed by 5.48
//                                       cells (68 %). Total: missed by 5.84.
//                                       Material: 0.48, its row-1 offset
//
//  THE ISOTHERMAL WINDOW CLOSES WHEN THE MATERIAL MOVES. CLAUDE.md and
//  doc/m3lb.tex record that at u = 0 an isothermal front has a window of steady
//  link positions about a cell wide, reached according to the direction the
//  front came from. With material crossing the front the steady state is
//  unique and sits on the exact node partition. It is still first order: the
//  front is resolved to the node and no better.
//
//  NO ORDER IS QUOTED FOR THE BAND. Its apparent rates climb 1.03 -> 1.89 ->
//  2.44 (T rms) and 0.87 -> 1.91 -> 2.98 (dl/l) as the band widens from 1.4 to
//  11 cells; a rate above the scheme's is a symptom (CLAUDE.md), here most
//  likely the band coming into resolution. The gates are the bound and that
//  refinement never stalls (each doubling at least halves T rms). Every gate
//  in this file was set after the first run and is a regression guard on what
//  the code does, except rows R, 1 and 3, which are derived.
//
//  FP32. Rows R, 1 and 3 pass unchanged. Rows 0 and 2 STAGNATE on their finest
//  grids instead of converging: the slowest mode moves H by about
//  A * 13.9 alpha / L^2 per step, 7e-5 A at L = 160, which falls below FP32's
//  resolution of H once A < ~1e-3, so the field freezes with that error in it.
//  Row 0's FP32 error therefore GROWS with L (1.3e-4, 1.1e-4, 6.0e-4), and row
//  2's |dl| reads 0.023, 0.029, 0.043, 0.25 cells. Under FP32 the band gate
//  covers L <= 80 and no order is gated.
//
//  WHAT THIS DOES NOT TEST. Equal properties in both phases, which the exact
//  solution assumes. A uniform velocity: no shear, no div u, no liquid moving
//  past a solid. One dimension. Coupling the front to a flow -- a solid at the
//  feed speed beside a liquid sheared by a wall -- needs a momentum sink with a
//  target velocity, and DarcyGuo drags to rest.
//==============================================================================
#include "collision/EnthalpyBGK.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

int failures = 0;
int checks   = 0;

void verdict_le(const char* what, double got, double ceil_, const char* unit = "") {
  ++checks;
  const bool ok = got <= ceil_;
  if (!ok) ++failures;
  std::printf("  %-58s %11.4e %-5s (want <= %.2e)   %s\n", what, got, unit, ceil_,
              ok ? "PASS" : "FAIL");
}
void verdict_ge(const char* what, double got, double floor_, const char* unit = "") {
  ++checks;
  const bool ok = got >= floor_;
  if (!ok) ++failures;
  std::printf("  %-58s %11.4e %-5s (want >= %.2e)   %s\n", what, got, unit, floor_,
              ok ? "PASS" : "FAIL");
}

//------------------------------------------------------------------------------
// The exact steady state, for EQUAL properties (cp_s = cp_l = c, k_s = k_l = k)
// and a linear band. With x along the flow, steady conservation of the total
// enthalpy flux is
//
//     U H(T) - k T' = q,       q constant,
//
// and H(T) is piecewise linear -- c (T - T_s) in the solid, (c + La/dT)(T - T_s)
// across the band, c (T - T_s) + La in the liquid -- so on each branch
// k T' = U c_b (T - T*) and T is an exponential. q is the one unknown: the
// branch lengths must sum to L with T(0) = T_c and T(L) = T_h, and that length
// is monotone in q on (-inf, U H(T_c)), so bisection finds it. At an isothermal
// front the band has zero length and T' jumps by U La / k there, which is the
// Stefan condition for a front the material crosses.
//------------------------------------------------------------------------------
struct Exact {
  double Tc, Th, Ts, Tl, c, k, La, U, L;
  struct Branch { double T0, Tstar, a, x0, x1; };
  std::vector<Branch> br;
  double q = 0.0;

  double length(double qq, std::vector<Branch>* out) const {
    double x = 0.0;
    bool finite = true;
    auto add = [&](double T0, double T1, double h0, double cb) {
      if (T1 <= T0 || !finite) return;
      const double Tst = T0 + (qq - U * h0) / (U * cb);
      const double a   = U * cb / k;
      if (T0 - Tst <= 0.0) { finite = false; return; }
      const double dx = std::log((T1 - Tst) / (T0 - Tst)) / a;
      if (out) out->push_back({T0, Tst, a, x, x + dx});
      x += dx;
    };
    const double dT = Tl - Ts;
    add(Tc, Ts, c * (Tc - Ts), c);
    if (dT > 0.0) add(Ts, Tl, 0.0, c + La / dT);
    add(Tl, Th, c * dT + La, c);
    return finite ? x : 1e300;
  }

  void solve() {
    if (!(Tc < Ts && Th > Tl)) {
      std::printf("Exact: need T_c < T_s <= T_l < T_h\n");
      std::abort();
    }
    const double qhi = U * c * (Tc - Ts);       // length -> inf as q -> qhi-
    double lo = qhi - 1.0, hi = qhi;
    while (length(lo, nullptr) > L) lo = qhi - 2.0 * (qhi - lo);
    for (int it = 0; it < 300; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (length(mid, nullptr) > L) hi = mid; else lo = mid;
    }
    q = 0.5 * (lo + hi);
    br.clear();
    length(q, &br);
  }

  double T(double x) const {
    for (const Branch& b : br)
      if (x <= b.x1) return b.Tstar + (b.T0 - b.Tstar) * std::exp(b.a * (x - b.x0));
    return Th;
  }
  // Where the solid ends and the liquid begins. Equal at an isothermal front.
  double x_solidus() const { return br.front().x1; }
  double x_liquidus() const { return br.back().x0; }
  // int_0^L f_l dx -- liquid length plus the band's partial fraction.
  double liquid_integral() const {
    double s = L - x_liquidus();
    const double dT = Tl - Ts;
    if (dT > 0.0 && br.size() == 3) {
      const Branch& b = br[1];
      const double dx = b.x1 - b.x0;
      s += ((b.Tstar - Ts) * dx + (b.T0 - b.Tstar) * (std::exp(b.a * dx) - 1.0) / b.a) / dT;
    }
    return s;
  }
};

struct Result {
  double l_int = 0;       // sum_i w_i f_l,i  (trapezoid, ends on the nodes)
  double T_l2  = 0;       // rms |T - T_exact| / (T_h - T_c)
  double T_max = 0;       // max |T - T_exact| / (T_h - T_c)
  long   steps = 0;
  double last_change = 0; // max |dH| over the last interval / (H_h - H_c)
  std::vector<double> T;
};

//------------------------------------------------------------------------------
// One steady run. nx = L + 1 nodes with on-node (ScalarMoment) walls at both
// ends, so the exact solution's Dirichlet points ARE nodes and no half cell is
// in play. Diffusive scaling: alpha fixed by tau, U = Pe alpha / L.
//------------------------------------------------------------------------------
template <EnthalpyAdvect A>
Result run(int L, double Pe, double tau, double Tc, double Th, double Ts, double Tl,
           double La, const Exact& ex, double seed_front = 0.5) {
  using Coll = EnthalpyBGK<D3Q7, A>;
  const double alpha = Coll::diffusivity_from_omega(Real(1.0 / tau));
  const double U     = Pe * alpha / L;

  PhaseChange pc;
  pc.T_s = Real(Ts); pc.T_l = Real(Tl);
  pc.cp_s = pc.cp_l = Real(1);
  pc.k_s  = pc.k_l  = Real(alpha);
  pc.La   = Real(La);
  Coll coll; coll.set_material(pc);
  const PhaseChange m = coll.material();   // the NORMALISED copy -- CLAUDE.md

  const Real H_c = m.enthalpy_of(Real(Tc), Real(0));
  const Real H_h = m.enthalpy_of(Real(Th), Real(1));
  coll.T_ref = Real(0.5) * (H_c + H_h);

  Domain d(L + 1, 1, 1, false, true, true);
  ScalarSolver<D3Q7, EsotericPull<D3Q7>, Coll> s(d, coll);
  s.set_geometry([&](Index x, Index, Index) -> std::uint8_t {
    return (x == 0 || x == L) ? ScalarMoment : ScalarBulk;
  });
  s.set_wall_values([&](Index x, Index, Index) -> Real { return x == 0 ? H_c : H_h; });
  s.finalize_geometry();   // without it impose_moment() is a silent no-op

  View1D<Real> ux("ux", d.n_padded), uy("uy", d.n_padded), uz("uz", d.n_padded);
  Kokkos::deep_copy(ux, Real(U));
  Kokkos::deep_copy(uy, Real(0));
  Kokkos::deep_copy(uz, Real(0));
  s.set_velocity(ux, uy, uz);

  // Seed: piecewise-linear T with the band's middle at seed_front * L, at the
  // flow's own equilibrium. 0.5 is the straight line; a seed far downstream of
  // the exact front makes the front arrive from the OTHER side, which is the
  // uniqueness test (an isothermal front at u = 0 has a window of steady
  // states -- CLAUDE.md).
  const Domain dc = d;
  const double dTm = Tl - Ts;
  const double Tmid = 0.5 * (Ts + Tl), xm = seed_front * L;
  s.initialize_field_in_flow(KOKKOS_LAMBDA(Index n) -> Real {
    Index px, py, pz; dc.coords(n, px, py, pz);
    const double x = double(px - dc.hx);
    const double T = x <= xm ? Tc + (Tmid - Tc) * x / xm
                             : Tmid + (Th - Tmid) * (x - xm) / (double(L) - xm);
    double f = dTm > 0.0 ? (T - Ts) / dTm : (T > Ts ? 1.0 : 0.0);
    f = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
    return m.enthalpy_of(Real(T), Real(f));
  });

  auto read = [&](std::vector<double>& H) {
    s.compute_field();
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.temperature());
    H.resize(L + 1);
    for (int x = 0; x <= L; ++x) H[x] = double(h(d.id(x, 0, 0)));
  };

  // Steady when the whole field moves by less than tol over one interval of a
  // quarter flush time -- an interval criterion, not a per-step residual.
  const double tol   = sizeof(Real) == 4 ? 1e-6 : 1e-11;
  const long   chunk = std::max<long>(50, long(0.25 * L / U));
  const long   cap   = long(60.0 * double(L) * L / alpha);
  std::vector<double> Hp, Hn;
  read(Hp);
  Result r;
  for (long t = 0; t < cap; t += chunk) {
    for (long k = 0; k < chunk; ++k) s.step();
    read(Hn);
    double dmax = 0;
    for (int x = 0; x <= L; ++x) dmax = std::max(dmax, std::abs(Hn[x] - Hp[x]));
    r.last_change = dmax / double(H_h - H_c);
    r.steps = t + chunk;
    Hp.swap(Hn);
    if (r.last_change < tol) break;
  }

  r.T.resize(L + 1);
  double sl = 0, s2 = 0;
  for (int x = 0; x <= L; ++x) {
    Real fl, T, E, c;
    m.invert(Real(Hp[x]), fl, T, E, c);
    r.T[x] = double(T);
    const double w = (x == 0 || x == L) ? 0.5 : 1.0;
    sl += w * double(fl);
    const double e = (double(T) - ex.T(double(x))) / (Th - Tc);
    s2 += e * e;
    r.T_max = std::max(r.T_max, std::abs(e));
  }
  r.l_int = sl;
  r.T_l2  = std::sqrt(s2 / double(L + 1));
  return r;
}

Exact exact(int L, double Pe, double tau, double Tc, double Th, double Ts, double Tl,
            double La) {
  const double alpha = EnthalpyBGK<D3Q7>::diffusivity_from_omega(Real(1.0 / tau));
  Exact e{Tc, Th, Ts, Tl, 1.0, alpha, La, Pe * alpha / L, double(L), {}, 0.0};
  e.solve();
  return e;
}


// The isothermal front, derived a SECOND way: with theta = T - T_m and
// P = U / alpha, the two exponentials and the jump k [T'] = U La give
//     e^{Ps} [ theta_h / (e^{PL} - e^{Ps}) + theta_c / (e^{Ps} - 1) ] = La / c.
// Bisected on its own, it must land where the shooting solver put x_solidus.
double isothermal_front(double L, double P, double th_c, double th_h, double Lam) {
  auto F = [&](double s) {
    const double e = std::exp(P * s);
    return e * (th_h / (std::exp(P * L) - e) + th_c / (e - 1.0)) - Lam;
  };
  double lo = 1e-9 * L, hi = L * (1.0 - 1e-12);   // F: -inf at 0+, +inf at L-
  for (int it = 0; it < 300; ++it) {
    const double mid = 0.5 * (lo + hi);
    (F(mid) > 0.0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    for (int a = 1; a < argc; ++a) {
      std::printf("fed_rod: unknown option '%s'\n", argv[a]);
      Kokkos::finalize();
      return 2;
    }
    const bool fp32 = sizeof(Real) == 4;
    const double Pe = 4.0, tau = 1.0, Tc = -1.0, Th = 1.0, Lam = 1.0;
    const double alpha = EnthalpyBGK<D3Q7>::diffusivity_from_omega(Real(1.0 / tau));
    std::printf("fed_rod: D3Q7, EnthalpyBGK, on-node walls, precision %s\n",
                fp32 ? "float" : "double");
    std::printf("  Pe = U L / alpha = %.2f   tau = %.3f   alpha = %.5f   "
                "T_c = %.1f  T_h = %.1f   La/c = %.2f (Ste = 1)\n",
                Pe, tau, alpha, Tc, Th, Lam);

    const int NL = 4;
    const int Ls[NL] = {40, 80, 160, 320};

    // ---- The reference, checked against itself before it judges anything.
    std::printf("\n-- R. the exact solution, two derivations --\n");
    {
      Exact ex = exact(80, Pe, tau, Tc, Th, 0.0, 0.0, Lam);
      const double s2 = isothermal_front(80.0, ex.U / ex.k, Tc, Th, Lam);
      verdict_le("|x_f(shooting) - x_f(closed form)| [cells]",
                 std::abs(ex.x_solidus() - s2), 1e-9);
      verdict_le("|T(L) - T_h| from the branches", std::abs(ex.T(80.0 - 1e-12) - Th), 1e-9);
    }

    // ---- 0. No latent heat: the advective half against exp(Pe x / L). ----
    std::printf("\n-- 0. La = 0, band [-0.1, 0.1]: pure advection-diffusion --\n");
    std::printf("  %5s %8s %9s %12s %12s %14s\n", "L", "U", "steps", "T rms", "T max",
                "|Mat - Sens|");
    double e0[NL], dms_worst = 0;
    for (int k = 0; k < NL - 1; ++k) {
      const int L = Ls[k];
      Exact ex = exact(L, Pe, tau, Tc, Th, -0.1, 0.1, 0.0);
      Result rm = run<EnthalpyAdvect::Material>(L, Pe, tau, Tc, Th, -0.1, 0.1, 0.0, ex);
      Result rs = run<EnthalpyAdvect::Sensible>(L, Pe, tau, Tc, Th, -0.1, 0.1, 0.0, ex);
      double dms = 0;
      for (int x = 0; x <= L; ++x) dms = std::max(dms, std::abs(rm.T[x] - rs.T[x]));
      dms_worst = std::max(dms_worst, dms);
      std::printf("  %5d %8.5f %9ld %12.4e %12.4e %14.4e\n", L, Pe * alpha / L, rm.steps,
                  rm.T_l2, rm.T_max, dms);
      e0[k] = rm.T_l2;
    }
    verdict_le("La = 0: Material == Sensible, every node", dms_worst, 0.0);
    if (!fp32) {
      verdict_ge("La = 0: T order, 40 -> 80", std::log2(e0[0] / e0[1]), 1.9);
      verdict_ge("La = 0: T order, 80 -> 160", std::log2(e0[1] / e0[2]), 1.9);
    } else {
      std::printf("  (orders not gated in FP32)\n");
    }

    // ---- 1. Isothermal front, Material. ----
    std::printf("\n-- 1. Material, La/c = 1, isothermal T_m = 0 --\n");
    std::printf("  %5s %9s %12s %12s %12s %12s %12s %10s\n", "L", "steps", "x_f exact",
                "l_exact", "l_lbm", "dl [cells]", "node-phase", "T rms");
    {
      double worst_phase = 0, et[NL];
      for (int k = 0; k < NL; ++k) {
        const int L = Ls[k];
        Exact ex = exact(L, Pe, tau, Tc, Th, 0.0, 0.0, Lam);
        Result r = run<EnthalpyAdvect::Material>(L, Pe, tau, Tc, Th, 0.0, 0.0, Lam, ex);
        const double le = ex.liquid_integral(), xf = ex.x_solidus();
        // If every node is in its exact phase and none is mushy, the trapezoid
        // sum of f_l is (liquid nodes) - 1/2, which differs from L - x_f by
        // x_f - ceil(x_f) + 1/2: the sub-cell position of the exact front.
        const double dl_nodes = xf - std::ceil(xf) + 0.5;
        const double dl = r.l_int - le;
        worst_phase = std::max(worst_phase, std::abs(dl - dl_nodes));
        std::printf("  %5d %9ld %12.6f %12.6f %12.6f %12.6f %12.2e %10.3e\n", L, r.steps,
                    xf, le, r.l_int, dl, dl - dl_nodes, r.T_l2);
        et[k] = r.T_l2;
      }
      verdict_le("every node in its exact phase, none mushy [cells]", worst_phase,
                 fp32 ? 1e-3 : 1e-8);
      std::printf("  T rms orders  %.2f  %.2f  %.2f\n", std::log2(et[0] / et[1]),
                  std::log2(et[1] / et[2]), std::log2(et[2] / et[3]));

      // Uniqueness: the same steady state reached from the other side.
      Exact ex = exact(80, Pe, tau, Tc, Th, 0.0, 0.0, Lam);
      Result up = run<EnthalpyAdvect::Material>(80, Pe, tau, Tc, Th, 0.0, 0.0, Lam, ex, 0.5);
      Result dn = run<EnthalpyAdvect::Material>(80, Pe, tau, Tc, Th, 0.0, 0.0, Lam, ex, 0.98);
      double dT = 0;
      for (int x = 0; x <= 80; ++x) dT = std::max(dT, std::abs(up.T[x] - dn.T[x]));
      std::printf("  seeded upstream (front at 0.50 L): l = %.8f   downstream (0.98 L): l = %.8f\n",
                  up.l_int, dn.l_int);
      verdict_le("unique: |l(up) - l(down)| [cells]", std::abs(up.l_int - dn.l_int),
                 fp32 ? 1e-3 : 1e-8);
      verdict_le("unique: max |T(up) - T(down)|", dT, fp32 ? 1e-4 : 1e-9);
    }

    // ---- 2. Banded front, Material. ----
    std::printf("\n-- 2. Material, La/c = 1, band [-0.1, 0.1] --\n");
    std::printf("  %5s %9s %10s %10s %10s %12s %12s %12s\n", "L", "steps", "band [c]",
                "l_exact", "l_lbm", "dl [cells]", "dl/l", "T rms");
    {
      double el[NL], et[NL], dlc[NL];
      for (int k = 0; k < NL; ++k) {
        const int L = Ls[k];
        Exact ex = exact(L, Pe, tau, Tc, Th, -0.1, 0.1, Lam);
        Result r = run<EnthalpyAdvect::Material>(L, Pe, tau, Tc, Th, -0.1, 0.1, Lam, ex);
        const double le = ex.liquid_integral();
        std::printf("  %5d %9ld %10.3f %10.4f %10.4f %12.4e %12.4e %12.4e\n", L, r.steps,
                    ex.x_liquidus() - ex.x_solidus(), le, r.l_int, r.l_int - le,
                    (r.l_int - le) / le, r.T_l2);
        el[k] = std::abs(r.l_int - le) / le;
        et[k] = r.T_l2;
        dlc[k] = std::abs(r.l_int - le);
      }
      std::printf("  orders  dl/l %.2f %.2f %.2f   T rms %.2f %.2f %.2f\n",
                  std::log2(el[0] / el[1]), std::log2(el[1] / el[2]), std::log2(el[2] / el[3]),
                  std::log2(et[0] / et[1]), std::log2(et[1] / et[2]), std::log2(et[2] / et[3]));
      // FP32 STAGNATES on the finest rows rather than converging: the slowest
      // mode moves H by ~A * 13.9 alpha / L^2 per step (7e-5 A at L = 160), which
      // drops below FP32's resolution of H once A < ~1e-3, so the field freezes
      // with that error in it -- row 0's FP32 error GROWS with L for the same
      // reason. Measured in FP32: |dl| = 0.023, 0.029, 0.043, 0.25 cells. The gate
      // is kept on the rows above that floor.
      const int kmax = fp32 ? 2 : NL;
      double worst = 0;
      for (int k = 0; k < kmax; ++k) worst = std::max(worst, dlc[k]);
      if (fp32) std::printf("  (FP32: |dl| gated at L <= 80 only -- finer rows stagnate)\n");
      verdict_le(fp32 ? "band: |dl| at L <= 80 [cells]" : "band: |dl| at every L [cells]",
                 worst, 0.05);
      // NOT an order. The apparent rate climbs 1.03 -> 1.89 -> 2.44 as the band
      // goes from 1.4 to 11 cells, and a rate above the scheme's is a symptom
      // (CLAUDE.md). What is gated is that refinement never stalls.
      double worst_ratio = 1e30;
      for (int k = 0; k + 1 < NL; ++k) worst_ratio = std::min(worst_ratio, et[k] / et[k + 1]);
      if (!fp32) verdict_ge("band: T rms falls >= first order at every doubling", worst_ratio, 1.9);
    }

    // ---- 3. Controls at L = 80, isothermal. ----
    std::printf("\n-- 3. controls, L = 80, isothermal, La/c = 1 --\n");
    {
      const int L = 80;
      Exact ex1 = exact(L, Pe, tau, Tc, Th, 0.0, 0.0, Lam);
      Exact ex0 = exact(L, Pe, tau, Tc, Th, 0.0, 0.0, 0.0);
      Result rm = run<EnthalpyAdvect::Material>(L, Pe, tau, Tc, Th, 0.0, 0.0, Lam, ex1);
      Result rs = run<EnthalpyAdvect::Sensible>(L, Pe, tau, Tc, Th, 0.0, 0.0, Lam, ex1);
      Result rt = run<EnthalpyAdvect::Total>(L, Pe, tau, Tc, Th, 0.0, 0.0, Lam, ex1);
      const double l1 = ex1.liquid_integral(), l0 = ex0.liquid_integral();
      std::printf("  exact liquid length   La/c = 1: %.4f    La = 0: %.4f\n", l1, l0);
      std::printf("  %-26s %10s %14s %14s\n", "operator", "l_lbm", "dl [cells]",
                  "vs La=0 [c]");
      auto row = [&](const char* nm, const Result& r) {
        std::printf("  %-26s %10.4f %14.4f %14.4f\n", nm, r.l_int, r.l_int - l1, r.l_int - l0);
      };
      row("Material", rm);
      row("Sensible (the default)", rs);
      row("Total (control)", rt);
      // Sensible in steady state solves div(E u) = div(k grad T): La is gone,
      // so it must reproduce the La = 0 front -- to the same half-cell
      // quantisation as row 1 -- and miss the real one by l0 - l1.
      verdict_le("Sensible == the La = 0 answer [cells]", std::abs(rs.l_int - l0), 0.5);
      verdict_ge("Sensible misses the La/c = 1 front by [cells]", std::abs(rs.l_int - l1),
                 0.5 * (l0 - l1));
      verdict_ge("Total misses the La/c = 1 front by [cells]", std::abs(rt.l_int - l1),
                 0.5 * (l0 - l1));
    }

    std::printf("\n%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
