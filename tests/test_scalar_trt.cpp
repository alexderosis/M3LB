//==============================================================================
//  ScalarTRT -- the two-relaxation-time scalar collision (collision/ScalarTRT.hpp).
//
//  1. It IS the family its banner says: at Lambda = (1/w - 1/2)^2 it reproduces
//     ScalarBGK and at Lambda = (1/w - 1/2)/2 ScalarRegularised, population for
//     population, on D2Q5 and D3Q7, at three omegas and two storage references.
//     Two independent operators already in the tree pin the new one.
//  2. At Lambda = 1/4: the scalar is conserved, the flux relaxes at omega toward
//     T u, every ghost relaxes at omega_plus toward cs2 dT, and omega_plus
//     satisfies (1/w+ - 1/2)(1/w - 1/2) = 1/4.
//  3. Its purpose, as a DIFFERENTIAL test that can fail: a small random
//     perturbation on a periodic ring along x, at u = 0.1 and omega = 1.98413 --
//     the mixer's D = 1e-3. The ScalarRegularised control must GROW (its von
//     Neumann limit there is |u| = 0.052) and ScalarTRT must not. If the two
//     operators were the same, or the ghost rate were wrong, one half fails.
//==============================================================================
#include "Check.hpp"
#include "collision/ScalarRegularised.hpp"
#include "collision/ScalarTRT.hpp"
#include "core/Types.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace lbm;

static Real TOL() { return sizeof(Real) == 4 ? Real(5e-6) : Real(5e-14); }

template <class L>
static void offeq(Real h[L::Q], Real dT) {
  for (int i = 0; i < L::Q; ++i)
    h[i] = weight<L, Real>(i) * dT + Real(0.021) * Real((i * 5) % 4 - 1);
}

template <class L, class A, class B>
static void same_collision(const A& a, const B& b, Real w, const std::string& what) {
  Real ha[L::Q], hb[L::Q];
  offeq<L>(ha, Real(0.37));
  for (int i = 0; i < L::Q; ++i) hb[i] = ha[i];
  const Real dT = ScalarBGK<L>::deviation(ha);
  a.collide(ha, dT, Real(0.031), Real(-0.017), Real(0.009), w);
  b.collide(hb, dT, Real(0.031), Real(-0.017), Real(0.009), w);
  Real e = Real(0);
  for (int i = 0; i < L::Q; ++i) e = std::max(e, Real(std::fabs(ha[i] - hb[i])));
  check::near(e, Real(0), TOL(), what);
}

template <class L>
static void reductions() {
  for (double w : {1.2, 1.9, 1.98413})
    for (double Tr : {0.0, 0.5}) {
      const std::string n = std::string(L::name) + " w=" + std::to_string(w) +
                            " T_ref=" + std::to_string(Tr);
      const Real tm = Real(1) / Real(w) - Real(0.5);
      ScalarBGK<L> bgk;          bgk.omega = Real(w); bgk.T_ref = Real(Tr);
      ScalarRegularised<L> reg;  reg.omega = Real(w); reg.T_ref = Real(Tr);
      ScalarTRT<L> t1;           t1.omega = Real(w);  t1.T_ref = Real(Tr);  t1.Lambda = tm * tm;
      ScalarTRT<L> t2;           t2.omega = Real(w);  t2.T_ref = Real(Tr);  t2.Lambda = Real(0.5) * tm;
      same_collision<L>(t1, bgk, Real(w), n + ": TRT at (tau-1/2)^2 == ScalarBGK");
      same_collision<L>(t2, reg, Real(w), n + ": TRT at (tau-1/2)/2 == ScalarRegularised");
    }
}

template <class L>
static void quarter() {
  constexpr Real c2 = cs2<L, Real>();
  for (double wd : {1.2, 1.9, 1.98413}) {
    const Real w = Real(wd);
    const std::string n = std::string(L::name) + " Lambda=1/4 w=" + std::to_string(wd);
    ScalarTRT<L> t;  t.omega = w;  t.T_ref = Real(0.5);
    const Real wp = t.omega_plus(w);
    check::near((Real(1) / wp - Real(0.5)) * (Real(1) / w - Real(0.5)), Real(0.25), TOL(),
                n + ": (1/w+ - 1/2)(1/w - 1/2) = 1/4");
    Real h[L::Q];
    offeq<L>(h, Real(0.37));
    const Real dT = ScalarBGK<L>::deviation(h);
    const Real T = t.T_ref + dT;
    const Real u[3] = {Real(0.031), Real(-0.017), Real(0.009)};
    Real j0[3], m0[3];
    for (int a = 0; a < L::D; ++a) {
      j0[a] = h[2 * a + 1] - h[2 * a + 2];
      m0[a] = h[2 * a + 1] + h[2 * a + 2];
    }
    t.collide(h, dT, u[0], u[1], u[2], w);
    check::near(ScalarBGK<L>::deviation(h), dT, TOL(), n + ": scalar conserved");
    for (int a = 0; a < L::D; ++a) {
      const std::string ax = " axis " + std::to_string(a);
      check::near(h[2 * a + 1] - h[2 * a + 2], j0[a] + w * (T * u[a] - j0[a]), TOL(),
                  n + ": flux relaxed at omega" + ax);
      check::near(h[2 * a + 1] + h[2 * a + 2], m0[a] + wp * (c2 * dT - m0[a]), TOL(),
                  n + ": ghost relaxed at omega_plus" + ax);
    }
  }
}

// Max |dT| after `steps` over its initial value, for a D3Q7 field on a periodic
// ring of N cells along x with a uniform velocity u along x. Only the x pair
// moves; the y and z pairs and the rest population stay (the mode is uniform in
// y and z), which is exactly the 1-D reduction the von Neumann scan uses.
template <class Coll>
static double ring_growth(const Coll& c, Real u, int N, int steps) {
  using L = D3Q7;
  std::vector<Real> h(std::size_t(N) * L::Q);
  // A deterministic, broadband perturbation of the uniform state.
  unsigned s = 12345u;
  auto rnd = [&]() { s = s * 1103515245u + 12345u; return Real((s >> 8) & 0xFFFF) / Real(65535) - Real(0.5); };
  double first = 0.0;
  for (int x = 0; x < N; ++x) {
    const Real dT = Real(1e-6) * rnd();
    for (int i = 0; i < L::Q; ++i)
      h[std::size_t(x) * L::Q + i] = c.eq(i, dT, u, Real(0), Real(0)) + Real(1e-7) * rnd();
    Real t = Real(0);
    for (int i = 0; i < L::Q; ++i) t += h[std::size_t(x) * L::Q + i];
    first = std::max(first, std::fabs(double(t)));
  }
  std::vector<Real> nxt(h.size());
  double last = 0.0;
  for (int n = 0; n < steps; ++n) {
    for (int x = 0; x < N; ++x) {
      Real* p = &h[std::size_t(x) * L::Q];
      const Real dT = ScalarBGK<L>::deviation(p);
      c.collide(p, dT, u, Real(0), Real(0));
    }
    for (int x = 0; x < N; ++x)
      for (int i = 0; i < L::Q; ++i) {
        const int src = (x - cvel<L>(i, 0) + N) % N;    // pull along x only
        nxt[std::size_t(x) * L::Q + i] = h[std::size_t(src) * L::Q + i];
      }
    h.swap(nxt);
  }
  for (int x = 0; x < N; ++x) {
    Real t = Real(0);
    for (int i = 0; i < L::Q; ++i) t += h[std::size_t(x) * L::Q + i];
    last = std::max(last, std::fabs(double(t)));
  }
  return last / first;
}

static void stability() {
  const Real w = Real(1.98413), u = Real(0.1);
  ScalarRegularised<D3Q7> reg;  reg.omega = w;  reg.T_ref = Real(0.5);
  ScalarTRT<D3Q7> trt;          trt.omega = w;  trt.T_ref = Real(0.5);
  const double gr = ring_growth(reg, u, 64, 1500);
  const double gt = ring_growth(trt, u, 64, 1500);
  std::printf("  ring, u = 0.1, omega = 1.98413, 1500 steps: max|dT| grew x%.3e under "
              "ScalarRegularised, x%.3e under ScalarTRT\n", gr, gt);
  check::ok(gr > 1e3, "control: ScalarRegularised grows on the ring (its limit is |u| = 0.052)");
  check::ok(gt < 2.0, "ScalarTRT (Lambda = 1/4) stays bounded on the same ring");
}

int main() {
  reductions<D2Q5>();
  reductions<D3Q7>();
  quarter<D2Q5>();
  quarter<D3Q7>();
  stability();
  return check::report("scalar_trt");
}
