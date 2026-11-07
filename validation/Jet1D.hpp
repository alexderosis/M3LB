//==============================================================================
//  validation/Jet1D.hpp -- a steady, slender, axisymmetric viscous jet from its
//  start to a collector: the part of melt electrowriting that M3LB's
//  free-surface grids cannot resolve.
//
//  WHY IT EXISTS. An MEW jet is 10-20 um across (the fibres of Ju, Omar,
//  Mirihanage and Kilic) against the 25-50 um cells of demonstrator/mew_axi, so
//  the free surface follows the melt under the tip and nothing that leaves it
//  (CLAUDE.md, the MEW limitation). A slender jet is one-dimensional: its radius
//  changes over lengths far larger than itself, each cross-section is a uniform
//  stretch, and the jet is its speed u(z), tension N(z) and temperature T(z)
//  along the axis, z from its start (z = 0) to the collector (z = L).
//
//  THE EQUATIONS, with Q the volume flow the jet carries and a = sqrt(Q/(pi u))
//  its radius:
//      N  = 3 mu(z,T) Q u'/u + pi gamma a     Trouton's extensional viscosity
//                                             3 mu, plus the surface tension's
//                                             net pull on a cross-section
//      N' = rho Q u' - f(z,T) Q/u             axial momentum: inertia, and a body
//                                             force f per unit VOLUME
//      T' = -2 pi a h (T - T_air)/(rho c_p Q) a lumped cross-section losing heat
//                                             through its surface
//  The surface-tension term is the slender limit of sigma_zz = 3 mu u' - gamma/a
//  (the capillary pressure gamma/a less the radial viscous stress) over the area,
//  plus 2 pi a gamma around the perimeter. A jet carrying a CONVECTED surface
//  charge has I/u of it per length and feels (I/u) E per length, i.e. I E / Q per
//  unit volume: an electrified jet in a uniform field is exactly a jet under an
//  enhanced gravity, rho g + I E / Q, which is what makes the uniform-force
//  solutions of validation/jet1d exact for it.
//
//  BOUNDARY CONDITIONS. u(0) = u0 and T(0) = T0, and at the collector either the
//  speed it is drawn at, u(L) = u_L (solve_drawn), or no tension, N(L) = 0
//  (solve_free): a free landing. Its speed is the fastest the jet arrives on its
//  own; a collector slower than that has to push the jet back, i.e. compress it,
//  and a compressed thread buckles -- so the free-landing speed is this model's
//  estimate of the critical translation speed below which a laid fibre stops
//  being straight. The model says WHEN that happens, not what the fibre does
//  then.
//
//  NUMERICS. Shooting on the unknown N(0): classical RK4 over n equal steps in z,
//  and bisection on N(0) to adjacent doubles. The residual (u(L) - u_L, or N(L))
//  rises monotonically with N(0) -- more tension stretches the jet faster, which
//  thins it and weakens a body force per length (f Q/u) -- and a step that drives
//  u to zero or to overflow is read as the matching end of that monotone range, so
//  the bracket is never lost. Fourth order: validation/jet1d checks each term
//  against an exact solution of its own and the coupled set by self-convergence.
//
//  THE DRAGGED JET (solve_dragged, its own banner below) bends the same thread
//  behind a collector moving sideways, as a catenary landing tangentially.
//
//  WHAT IT DOES NOT DO. No bending stiffness anywhere -- the string limit -- so
//  no coiling (it says only when the collector would compress the jet) and no
//  heel at the landing. Steady only. The charge is convected, never conducted,
//  and the field E(z) is given, not solved with the jet; behind a moving
//  collector it is taken as vertical and a function of height alone. Newtonian:
//  no viscoelasticity.
//  No air drag. No crystallisation kinetics: a freezing jet is whatever viscosity
//  law the caller supplies. Always FP64, whatever LBM_PRECISION says -- it is
//  host-side and shares nothing with the lattice.
//==============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace jet1d {

constexpr double kPi = 3.14159265358979323846;

struct Jet {
  double Q = 0.0;       // volume flow the jet carries [m^3/s]
  double L = 0.0;       // from the jet's start to the collector [m]
  double rho = 0.0;     // density, for the inertia only [kg/m^3]; 0 = inertialess
  double gamma = 0.0;   // surface tension [N/m]; 0 = off
  double rho_cp = 0.0;  // volumetric heat capacity [J/m^3/K]; 0 = isothermal
  double T_air = 0.0;   // ambient temperature (any scale; only differences enter)
  std::function<double(double z, double T)> mu;           // viscosity [Pa s]; required
  std::function<double(double z, double T)> f;            // body force per unit volume along z [N/m^3]
  std::function<double(double a, double u, double T)> h;  // heat-transfer coefficient [W/m^2/K]
};

struct State {
  double u, N, T;
};

struct Profile {
  std::vector<double> z, u, N, T;
};

inline State rhs(const Jet& J, double z, const State& y) {
  const double a = std::sqrt(J.Q / (kPi * y.u));
  const double up = (y.N - kPi * J.gamma * a) * y.u / (3.0 * J.mu(z, y.T) * J.Q);
  const double F = J.f ? J.f(z, y.T) * J.Q / y.u : 0.0;  // per unit length
  const double Tp = (J.rho_cp > 0.0 && J.h)
                        ? -2.0 * kPi * a * J.h(a, y.u, y.T) * (y.T - J.T_air) / (J.rho_cp * J.Q)
                        : 0.0;
  return {up, J.rho * J.Q * up - F, Tp};
}

// RK4 from z = 0 to L in n steps. Returns 0 on success, -1 if the speed fell to
// zero or below, +1 if anything overflowed or went non-finite.
inline int integrate(const Jet& J, State y, int n, State& end, Profile* p = nullptr) {
  const double dz = J.L / n;
  auto at = [](const State& s, const State& k, double c) {
    return State{s.u + c * k.u, s.N + c * k.N, s.T + c * k.T};
  };
  auto bad = [](const State& s) {
    if (!(s.u > 0.0)) return std::isnan(s.u) ? 1 : -1;
    if (!std::isfinite(s.u) || !std::isfinite(s.N) || !std::isfinite(s.T)) return 1;
    return 0;
  };
  if (p) {
    p->z.assign(1, 0.0);
    p->u.assign(1, y.u);
    p->N.assign(1, y.N);
    p->T.assign(1, y.T);
  }
  for (int i = 0; i < n; ++i) {
    const double z = i * dz;
    const State k1 = rhs(J, z, y);
    const State y2 = at(y, k1, 0.5 * dz);
    if (int b = bad(y2)) return b;
    const State k2 = rhs(J, z + 0.5 * dz, y2);
    const State y3 = at(y, k2, 0.5 * dz);
    if (int b = bad(y3)) return b;
    const State k3 = rhs(J, z + 0.5 * dz, y3);
    const State y4 = at(y, k3, dz);
    if (int b = bad(y4)) return b;
    const State k4 = rhs(J, z + dz, y4);
    y.u += dz / 6.0 * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u);
    y.N += dz / 6.0 * (k1.N + 2.0 * k2.N + 2.0 * k3.N + k4.N);
    y.T += dz / 6.0 * (k1.T + 2.0 * k2.T + 2.0 * k3.T + k4.T);
    if (int b = bad(y)) return b;
    if (p) {
      p->z.push_back((i + 1) * dz);
      p->u.push_back(y.u);
      p->N.push_back(y.N);
      p->T.push_back(y.T);
    }
  }
  end = y;
  return 0;
}

struct Solution {
  bool ok = false;
  double N0 = 0.0;  // the tension at the start that meets the collector's condition
  Profile p;
};

namespace detail {

// Bisection on N0 for a residual that rises monotonically with it.
template <class Residual>
inline Solution shoot(const Jet& J, double u0, double T0, int n, double scale, Residual res) {
  Solution s;
  auto r = [&](double N0) {
    State e{};
    const int st = integrate(J, State{u0, N0, T0}, n, e);
    if (st < 0) return -std::numeric_limits<double>::infinity();
    if (st > 0) return std::numeric_limits<double>::infinity();
    return res(e);
  };
  double lo = -scale, hi = scale;
  int k = 0;
  while (r(lo) > 0.0 && k++ < 200) lo *= 2.0;
  k = 0;
  while (r(hi) < 0.0 && k++ < 200) hi *= 2.0;
  if (!(r(lo) <= 0.0 && r(hi) >= 0.0)) return s;
  for (int it = 0; it < 400; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break;  // adjacent doubles
    (r(mid) < 0.0 ? lo : hi) = mid;
  }
  s.N0 = 0.5 * (lo + hi);
  State e{};
  s.ok = integrate(J, State{u0, s.N0, T0}, n, e, &s.p) == 0;
  return s;
}

}  // namespace detail

// The tension scale the bracket starts from: Trouton's tension for a unit draw.
inline double tension_scale(const Jet& J, double u0, double T0) {
  return 3.0 * J.mu(0.0, T0) * J.Q / J.L + std::abs(J.gamma) * std::sqrt(kPi * J.Q / u0) + 1e-300;
}

// u(0) = u0, T(0) = T0, u(L) = uL: a jet drawn by the collector at uL.
inline Solution solve_drawn(const Jet& J, double u0, double T0, double uL, int n) {
  return detail::shoot(J, u0, T0, n, tension_scale(J, u0, T0) * std::max(1.0, std::log(uL / u0) + 1.0),
                       [uL](const State& e) { return e.u - uL; });
}

// u(0) = u0, T(0) = T0, N(L) = 0: a free landing.
inline Solution solve_free(const Jet& J, double u0, double T0, int n) {
  return detail::shoot(J, u0, T0, n, tension_scale(J, u0, T0), [](const State& e) { return e.N; });
}

//------------------------------------------------------------------------------
//  THE DRAGGED JET: the same thread behind a collector moving sideways at v_c,
//  in the plane of that motion (x along it, z down from the jet's start).
//
//  With P = N - rho Q u (tension less momentum flux) and t = (sin th, cos th)
//  the jet's direction, steady momentum is d(P t)/ds = -F z_hat: the force is
//  vertical, so the horizontal component H = P sin th is the SAME all along the
//  jet, and the vertical one V = P cos th falls by the pull, dV/ds = -F. The
//  collector only drags the jet -- it exerts no vertical force on it -- so V = 0
//  where the jet lands: the landing point is the lowest point of a catenary and
//  the jet meets the collector tangentially, at the collector's speed (u = v_c,
//  since the laid fibre then moves with it). In a real thread a short bending
//  layer, the heel, rounds that corner; this is the string limit, with no
//  bending stiffness anywhere.
//
//  Because V falls monotonically from V0 to 0, it is the independent variable:
//  RK4 in V over n steps, ds/dV = -1/F, and the landing is the fixed end V = 0
//  rather than an event to be located (which would cost the order). Unknowns H and
//  V0: for a given H, V0 is bisected until the landing is at z = L (the landing
//  falls lower with more V0); H is bisected until the landing speed is v_c (more
//  H stretches the jet more). At H = 0 the jet is vertical and lands freely with
//  P = 0, so a collector slower than that has no steady dragged jet:
//  solve_dragged returns ok = false and the free-landing speed. (With inertia
//  this free landing is P = 0, i.e. N = rho Q u, not solve_free's N = 0.)
//------------------------------------------------------------------------------
struct Path {
  std::vector<double> V, x, z, u, T, s;
};

struct Dragged {
  bool ok = false;      // false: the collector is slower than the free landing, U_free
  double U_free = 0.0;  // the landing speed at H = 0
  double H = 0.0;       // horizontal (tension - momentum flux): the collector's drag on the jet [N]
  double V0 = 0.0;      // vertical (tension - momentum flux) at the start [N]
  double lag = 0.0;     // horizontal distance from the start to the landing [m]
  double theta0 = 0.0;  // the jet's angle from the vertical at its start [rad]
  Path p;
};

namespace detail {

struct Y {
  double x, z, u, T, s;
};

inline Y drhs(const Jet& J, double H, double V, const Y& y) {
  // V >= 0 by construction. A last stage that rounding put at -1e-16 would, at
  // H = 0, flip cos(th) = V/P to -1 and send the jet UP for that stage: a
  // first-order error in where it lands (found by jet1d's H -> 0 check).
  V = std::max(V, 0.0);
  const double P = std::sqrt(H * H + V * V);
  const double a = std::sqrt(J.Q / (kPi * y.u));
  const double F = J.f(y.z, y.T) * J.Q / y.u;  // vertical pull per length, > 0
  const double N = P + J.rho * J.Q * y.u;
  const double dsdV = -1.0 / F;
  const double us = (N - kPi * J.gamma * a) * y.u / (3.0 * J.mu(y.z, y.T) * J.Q);
  const double Ts = (J.rho_cp > 0.0 && J.h)
                        ? -2.0 * kPi * a * J.h(a, y.u, y.T) * (y.T - J.T_air) / (J.rho_cp * J.Q)
                        : 0.0;
  const double sn = P > 0.0 ? H / P : 0.0, cs = P > 0.0 ? V / P : 1.0;
  return {sn * dsdV, cs * dsdV, us * dsdV, Ts * dsdV, dsdV};
}

// RK4 in V from V0 down to 0. 0 on success; +1 if anything left the finite
// positive range (read as "too much tension", the end that stretches without
// bound).
inline int integrate_V(const Jet& J, double H, double V0, double u0, double T0, int n, Y& end,
                       Path* p = nullptr) {
  Y y{0.0, 0.0, u0, T0, 0.0};
  const double dV = -V0 / n;
  auto at = [](const Y& a, const Y& k, double c) {
    return Y{a.x + c * k.x, a.z + c * k.z, a.u + c * k.u, a.T + c * k.T, a.s + c * k.s};
  };
  auto bad = [](const Y& a) {
    return !(a.u > 0.0) || !std::isfinite(a.u) || !std::isfinite(a.z) || !std::isfinite(a.x) ||
           !std::isfinite(a.T);
  };
  if (p) {
    p->V.assign(1, V0);
    p->x.assign(1, 0.0);
    p->z.assign(1, 0.0);
    p->u.assign(1, u0);
    p->T.assign(1, T0);
    p->s.assign(1, 0.0);
  }
  for (int i = 0; i < n; ++i) {
    // each stage's V from V0 directly, so the last one is exactly 0
    const double V = V0 * double(n - i) / n, Vh = V0 * (double(n - i) - 0.5) / n, V1 = V0 * double(n - i - 1) / n;
    const Y k1 = drhs(J, H, V, y);
    const Y y2 = at(y, k1, 0.5 * dV);
    if (bad(y2)) return 1;
    const Y k2 = drhs(J, H, Vh, y2);
    const Y y3 = at(y, k2, 0.5 * dV);
    if (bad(y3)) return 1;
    const Y k3 = drhs(J, H, Vh, y3);
    const Y y4 = at(y, k3, dV);
    if (bad(y4)) return 1;
    const Y k4 = drhs(J, H, V1, y4);
    y.x += dV / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
    y.z += dV / 6.0 * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z);
    y.u += dV / 6.0 * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u);
    y.T += dV / 6.0 * (k1.T + 2.0 * k2.T + 2.0 * k3.T + k4.T);
    y.s += dV / 6.0 * (k1.s + 2.0 * k2.s + 2.0 * k3.s + k4.s);
    if (bad(y)) return 1;
    if (p) {
      p->V.push_back(V1);
      p->x.push_back(y.x);
      p->z.push_back(y.z);
      p->u.push_back(y.u);
      p->T.push_back(y.T);
      p->s.push_back(y.s);
    }
  }
  end = y;
  return 0;
}

// The V0 that lands the jet at z = L for a given H (bisection; z at the landing
// rises with V0). Returns false if no bracket was found.
inline bool land_at_L(const Jet& J, double H, double u0, double T0, int n, double& V0, Y& end) {
  auto zend = [&](double v, Y& e) {
    if (integrate_V(J, H, v, u0, T0, n, e)) return std::numeric_limits<double>::infinity();
    return e.z;
  };
  Y e{};
  double lo = 0.0, hi = std::max(V0, 1e-300);
  int k = 0;
  while (zend(hi, e) < J.L && k++ < 400) {
    lo = hi;
    hi *= 2.0;
  }
  if (!(zend(hi, e) >= J.L)) return false;
  for (int it = 0; it < 200; ++it) {  // to 1e-14 of the bracket: past round-off in every check
    const double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi || hi - lo <= 1e-14 * hi) break;
    (zend(mid, e) < J.L ? lo : hi) = mid;
  }
  V0 = 0.5 * (lo + hi);
  return zend(V0, end) < std::numeric_limits<double>::infinity();
}

}  // namespace detail

// The jet dragged by a collector moving at vc: u(0) = u0, T(0) = T0, a tangential
// landing at z = L with u = vc. A first guess for the vertical tension speeds the
// bracket (any positive value works).
inline Dragged solve_dragged(const Jet& J, double u0, double T0, double vc, int n, double V0_guess = 0.0) {
  Dragged r;
  double V0 = V0_guess > 0.0 ? V0_guess : tension_scale(J, u0, T0);
  detail::Y e{};
  if (!detail::land_at_L(J, 0.0, u0, T0, n, V0, e)) return r;
  r.U_free = e.u;
  if (vc <= r.U_free) return r;  // slower than the free landing: compressed, no steady dragged jet
  auto uend = [&](double H, double& v0) {
    detail::Y ee{};
    if (!detail::land_at_L(J, H, u0, T0, n, v0, ee)) return std::numeric_limits<double>::infinity();
    return ee.u;
  };
  double lo = 0.0, hi = 0.1 * V0, v0 = V0;
  int k = 0;
  while (uend(hi, v0) < vc && k++ < 400) {
    lo = hi;
    hi *= 2.0;
  }
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi || hi - lo <= 1e-14 * hi) break;
    double vm = v0;
    (uend(mid, vm) < vc ? lo : hi) = mid;
    v0 = vm;
  }
  r.H = 0.5 * (lo + hi);
  if (!detail::land_at_L(J, r.H, u0, T0, n, v0, e)) return r;
  r.V0 = v0;
  r.ok = detail::integrate_V(J, r.H, r.V0, u0, T0, n, e, &r.p) == 0;
  r.lag = e.x;
  r.theta0 = std::atan2(r.H, r.V0);
  return r;
}

}  // namespace jet1d
