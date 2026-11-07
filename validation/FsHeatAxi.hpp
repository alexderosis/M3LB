#pragma once
//==============================================================================
//  HEAT IN THE r-z MELT: a temperature carried by FreeSurfaceSolver's
//  axisymmetric liquid, lost through its free surface, held by heated and cold
//  walls, and fed back as a VISCOSITY (FreeSurfaceSolver::omega_of) and a
//  FREEZING DRAG (FreeSurfaceSolver::drag_of).
//
//  WHY. A PCL melt leaves a 120 C nozzle into room air. Its viscosity is
//  Arrhenius, mu = mu0 exp[(Ea/R)(1/T - 1/T0)], so a column that has cooled
//  by 40 C is several times stiffer, and on the collector it crystallises.
//  Every isothermal run in this tree carries the nozzle's viscosity all the
//  way down; this is the piece that lets a cooling column be stiffer.
//
//  WHAT IS SOLVED, per radian, on the liquid only (Fluid, and Interface cells
//  with a fill above `phi_min`):
//      phi r dT/dt + phi r u . grad T = div(alpha phi r grad T) - h_l r L (T - T_air),
//  T intensive (one value per cell), phi the cell's fill, L the length of the
//  surface inside it. In lattice units: alpha in cells^2 per fluid step and
//  h_l = h dt / (rho c_p dx), so the surface term is a rate h_l L / phi.
//   * ADVECTION, upwind on the LBM's own cell velocity, from liquid
//     neighbours only (zero gradient towards gas; a wall upstream supplies its
//     temperature, which is how the piston feeds hot melt). First order.
//   * DIFFUSION, conservative in r: radial faces weighted r_c +- 1/2 (zero on
//     the axis, a symmetry face for free), axial faces r_c, each by the smaller
//     fill of its two cells; a wall half a cell away doubles its face. EXPLICIT
//     with sub-steps, because a point-implicit update slows every slow mode by
//     1 / (1 + dt d) -- 14 % over one field update at 50 um, measured in the
//     design arithmetic before a line was written -- and implicit only in a
//     cell so empty that its own explicit step would overshoot (theta-scheme
//     per cell, positivity guaranteed). No conduction into gas.
//   * SURFACE LOSS, implicit, on interface cells, through an area that is
//     sum |grad phi| r over the cells around them (see surface_area(): exact
//     for a vertical surface, +1.3-1.4 % for a circle or a sphere wherever it
//     sits; per-cell PLIC segments were 2-15 % short and erratic).
//     h is ONE number: a real drop's h depends on its size and on the air,
//     and the caller chooses it.
//   * NEW LIQUID CELLS take the fill-weighted mean of their liquid neighbours;
//     a cell that empties forgets its temperature.
//
//  WHAT IT DOES NOT DO.
//   - It is NOT conservative to round-off: T is intensive and advected in
//     non-conservative form, the free surface's own mass exchange is not
//     reproduced, and a cell's heat goes with its fill. The heat budget is a
//     diagnostic, not an invariant.
//   - No latent heat: crystallisation is a drag switched on below a
//     temperature band (write_drag), not a phase change with an enthalpy.
//   - No viscous heating and no heat in the gas; the air is a fixed T_air.
//  validation/axisym_heat checks it against exact answers: conduction in a
//  cylinder (Bessel), Newton cooling through a free surface, and steady
//  advection-diffusion between two walls.
//==============================================================================
#include "solver/FreeSurfaceSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lbm {

enum HeatRole : std::uint8_t { HeatFree = 0, HeatWall = 1, HeatInsulated = 2 };

// The fields the heat solver reads, on the host, indexed by Domain::id. A test
// can fill them by hand; fields_from() fills them from a FreeSurfaceSolver.
struct HeatFields {
  std::vector<std::uint8_t> flags;
  std::vector<double> fill, ux, uy;
};

template <class FS>
void fields_from(const FS& fs, const Domain& d, HeatFields& F) {
  const std::size_t N = std::size_t(d.n_padded);
  auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.uy());
  F.flags.resize(N); F.fill.resize(N); F.ux.resize(N); F.uy.resize(N);
  for (std::size_t n = 0; n < N; ++n) {
    F.flags[n] = hf(n); F.fill[n] = double(he(n)); F.ux[n] = double(hu(n)); F.uy[n] = double(hv(n));
  }
}

class FsHeatAxi {
 public:
  double alpha = 0;        // thermal diffusivity, cells^2 per fluid step
  double hloss = 0;        // h dt / (rho c_p dx), cells per fluid step
  double T_air = 25;       // the gas
  double phi_min = 1e-3;   // an interface cell below this fill carries no temperature

  explicit FsHeatAxi(const Domain& d)
      : d_(d), role_(N(), HeatFree), wall_(N(), 0.0), T_(N(), NAN), Tn_(N(), NAN), liq_(N(), 0),
        phi_(N(), 0.0) {
    if (d.nz != 1) Kokkos::abort("FsHeatAxi: r-z only (nz = 1)");
  }

  // Solid cells only: HeatWall holds `val`, HeatInsulated (and gas) passes nothing.
  template <class RoleFn, class ValFn>
  void set_roles(RoleFn fn, ValFn val) {
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        role_[n] = fn(x, y);
        wall_[n] = double(val(x, y));
      }
  }

  // Every liquid cell of F at T0(x, y).
  template <class Fn>
  void initialize(const HeatFields& F, Fn T0) {
    classify(F);
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        T_[n] = liq_[n] ? double(T0(x, y)) : NAN;
      }
  }

  // Advance by `steps` fluid steps in the fields F. Returns the sub-steps used.
  int update(const HeatFields& F, double steps) {
    std::vector<std::uint8_t> was = liq_;
    classify(F);
    adopt_new_cells(was);
    // sub-steps: explicit diffusion in full cells, and advection, both below 1/2
    double dmax = 0, umax = 0;
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (!liq_[n]) continue;
        umax = std::max(umax, std::fabs(F.ux[n]) + std::fabs(F.uy[n]));
        if (phi_[n] >= 0.5) dmax = std::max(dmax, diag(x, y) / phi_[n]);
      }
    const int nsub = std::max(1, int(std::ceil(steps * std::max(dmax, umax) / 0.45)));
    const double dt = steps / double(nsub);
    surface_area(F);
    for (int k = 0; k < nsub; ++k) substep(F, dt);
    return nsub;
  }

  // The melt's viscosity, Arrhenius about Tref and capped at nu_cap, into
  // fs.omega_of (allocated on first use). EaR = Ea / R_gas in kelvin.
  template <class FS>
  void write_omega(FS& fs, double nu0, double EaR, double Tref, double nu_cap) const {
    if (!fs.omega_of.data()) fs.omega_of = View1D<Real>("fs_omega_of", d_.n_padded);
    auto h = Kokkos::create_mirror_view(fs.omega_of);
    const double w0 = double(FS::omega_from_viscosity(Real(nu0)));
    for (Index n = 0; n < d_.n_padded; ++n) h(n) = Real(w0);
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (!liq_[n]) continue;
        const double nu = std::min(nu_cap, nu0 * viscosity_ratio(T_[n], EaR, Tref));
        h(n) = FS::omega_from_viscosity(Real(nu));
      }
    Kokkos::deep_copy(fs.omega_of, h);
  }
  static double viscosity_ratio(double T, double EaR, double Tref) {
    return std::exp(EaR * (1.0 / (T + 273.15) - 1.0 / (Tref + 273.15)));
  }

  // The freezing drag A s(T), s rising from 0 at Tl to 1 at Ts, into fs.drag_of.
  template <class FS>
  void write_drag(FS& fs, double A, double Ts, double Tl) const {
    if (!fs.drag_of.data()) fs.drag_of = View1D<Real>("fs_drag_of", d_.n_padded);
    auto h = Kokkos::create_mirror_view(fs.drag_of);
    for (Index n = 0; n < d_.n_padded; ++n) h(n) = Real(0);
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (!liq_[n]) continue;
        h(n) = Real(A * solid_fraction(T_[n], Ts, Tl));
      }
    Kokkos::deep_copy(fs.drag_of, h);
  }
  static double solid_fraction(double T, double Ts, double Tl) {
    return std::clamp((Tl - T) / std::max(1e-12, Tl - Ts), 0.0, 1.0);
  }

  double T(Index x, Index y) const { return T_[d_.id(x, y)]; }
  bool liquid(Index x, Index y) const { return liq_[d_.id(x, y)] != 0; }
  double phi(Index x, Index y) const { return phi_[d_.id(x, y)]; }
  // r-weighted heat sum(phi r T) and volume sum(phi r), per radian, cells^3
  void totals(double& heat, double& vol) const {
    heat = vol = 0;
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (!liq_[n]) continue;
        const double w = phi_[n] * (double(x) + 0.5);
        heat += w * T_[n];
        vol += w;
      }
  }
  double lost() const { return lost_; }   // heat through the surface so far (phi r T units)

 private:
  Domain d_;
  std::vector<std::uint8_t> role_;
  std::vector<double> wall_, T_, Tn_;
  std::vector<std::uint8_t> liq_;
  std::vector<double> phi_, area_;
  double lost_ = 0;

  // THE SURFACE, per radian, as sum |grad phi| r over every cell (central
  // differences; a solid reads as the cell itself, so a wall is not surface,
  // and the axis mirrors). For a vertical surface of fill f that sum is
  // EXACTLY R + f, the surface's own radius; for a circle or a sphere it is
  // 1.013-1.014 of the area at every size and position. Per-cell PLIC
  // segments were the first version and summed to 0.92-0.99 of a circle and
  // 0.85-0.93 of a sphere: where a surface runs nearly along a face, the
  // nearly full cell holds a lens of gas that PLIC turns into a corner
  // triangle (0.74 long where the arc is 1.0). What a fluid or gas cell holds
  // is handed in equal shares to the surface cells among its eight neighbours.
  void surface_area(const HeatFields& F) {
    area_.assign(N(), 0.0);
    auto ph = [&](Index x, Index y, double self) {
      if (x < 0) x = -x - 1;                               // the axis
      if (x >= d_.nx || y < 0 || y >= d_.ny) return self;
      const std::uint8_t f = F.flags[d_.id(x, y)];
      if (f == FsFluid) return 1.0;
      if (f == FsInterface) return std::clamp(F.fill[d_.id(x, y)], 0.0, 1.0);
      if (f == FsGas) return 0.0;
      return self;                                         // solid: no surface
    };
    auto surf = [&](Index x, Index y) {
      const Index j = d_.id(x, y);
      return liq_[j] && F.flags[j] == FsInterface;
    };
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const std::uint8_t f = F.flags[d_.id(x, y)];
        if (f != FsFluid && f != FsInterface && f != FsGas) continue;
        const double self = ph(x, y, 0.0);
        const double gx = 0.5 * (ph(x + 1, y, self) - ph(x - 1, y, self));
        const double gy = 0.5 * (ph(x, y + 1, self) - ph(x, y - 1, self));
        const double A = std::hypot(gx, gy) * (double(x) + 0.5);
        if (!(A > 0)) continue;
        if (surf(x, y)) { area_[d_.id(x, y)] += A; continue; }
        Index tx[8], ty[8];
        int m = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy) continue;
            const Index xn = x + dx, yn = y + dy;
            if (xn < 0 || xn >= d_.nx || yn < 0 || yn >= d_.ny) continue;
            if (surf(xn, yn)) { tx[m] = xn; ty[m] = yn; ++m; }
          }
        for (int k = 0; k < m; ++k) area_[d_.id(tx[k], ty[k])] += A / double(m);
      }
  }

  std::size_t N() const { return std::size_t(d_.n_padded); }

  void classify(const HeatFields& F) {
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        const std::uint8_t f = F.flags[n];
        const double p = (f == FsFluid) ? 1.0 : (f == FsInterface ? std::clamp(F.fill[n], 0.0, 1.0) : 0.0);
        liq_[n] = (f == FsFluid) || (f == FsInterface && p > phi_min);
        phi_[n] = liq_[n] ? p : 0.0;
      }
  }

  // neighbour k of (x, y): 0 E, 1 W, 2 N, 3 S; false outside the box
  bool nb(Index x, Index y, int k, Index& xn, Index& yn) const {
    xn = x + (k == 0 ? 1 : (k == 1 ? -1 : 0));
    yn = y + (k == 2 ? 1 : (k == 3 ? -1 : 0));
    return xn >= 0 && xn < d_.nx && yn >= 0 && yn < d_.ny;
  }
  double face_r(Index x, int k) const {
    const double r = double(x) + 0.5;
    return k == 0 ? r + 0.5 : (k == 1 ? r - 0.5 : r);
  }
  // the full-cell diagonal sum of the diffusion coefficients, for the step size
  double diag(Index x, Index y) const {
    double s = 0;
    for (int k = 0; k < 4; ++k) {
      Index xn, yn;
      if (!nb(x, y, k, xn, yn)) continue;
      const Index j = d_.id(xn, yn);
      const double w = liq_[j] ? 1.0 : (role_[j] == HeatWall && !is_fluid_cell(j) ? 2.0 : 0.0);
      s += alpha * face_r(x, k) * w;
    }
    return s / (double(x) + 0.5);
  }
  bool is_fluid_cell(Index j) const { return liq_[j] != 0; }

  void adopt_new_cells(const std::vector<std::uint8_t>& was) {
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (!liq_[n]) { T_[n] = NAN; continue; }
        if (was[n] && std::isfinite(T_[n])) continue;
        double s = 0, w = 0, sw = 0, ww = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy) continue;
            Index xn = x + dx, yn = y + dy;
            if (yn < 0 || yn >= d_.ny) continue;
            if (xn < 0) xn = -xn - 1;                // the mirror across the axis
            if (xn >= d_.nx) continue;
            const Index j = d_.id(xn, yn);
            if (was[j] && std::isfinite(T_[j])) {
              const double q = std::max(phi_[j], 1e-3) * (double(xn) + 0.5);
              s += q * T_[j]; w += q;
            } else if (role_[j] == HeatWall && !liq_[j]) {
              sw += wall_[j]; ww += 1;
            }
          }
        T_[n] = w > 0 ? s / w : (ww > 0 ? sw / ww : T_air);
      }
  }

  void substep(const HeatFields& F, double dt) {
    Tn_ = T_;
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (!liq_[n]) continue;
        const double r = double(x) + 0.5, pc = phi_[n], Tc = Tn_[n];
        // ---- advection, upwind, the cell's own velocity
        auto up = [&](int k) {           // the upstream value through face k
          Index xn, yn;
          if (k == 1 && x == 0) return Tc;               // the axis: T even
          if (!nb(x, y, k, xn, yn)) return Tc;
          const Index j = d_.id(xn, yn);
          if (liq_[j]) return Tn_[j];
          if (role_[j] == HeatWall) return wall_[j];
          return Tc;                                      // gas or insulated: zero gradient
        };
        const double u = F.ux[n], v = F.uy[n];
        const double adv = (u > 0 ? u * (Tc - up(1)) : u * (up(0) - Tc))
                         + (v > 0 ? v * (Tc - up(3)) : v * (up(2) - Tc));
        // ---- diffusion, conservative in r, theta per cell
        double sa = 0, sat = 0;
        for (int k = 0; k < 4; ++k) {
          Index xn, yn;
          if (!nb(x, y, k, xn, yn)) continue;
          const double rf = face_r(x, k);
          if (rf <= 0) continue;                          // the axis face
          const Index j = d_.id(xn, yn);
          double a = 0, Tj = 0;
          if (liq_[j]) { a = alpha * rf * std::min(pc, phi_[j]); Tj = Tn_[j]; }
          else if (role_[j] == HeatWall) { a = 2.0 * alpha * rf * pc; Tj = wall_[j]; }
          else continue;
          a /= r * pc;
          sa += a; sat += a * Tj;
        }
        // ---- surface loss through the area surface_area() gave this cell
        const double b = (hloss > 0 && area_[n] > 0) ? hloss * area_[n] / (r * pc) : 0.0;
        const double th = std::clamp(1.0 - 0.5 / std::max(1e-30, dt * sa), 0.0, 1.0);
        const double rhs = Tc - dt * adv + dt * (sat - (1.0 - th) * sa * Tc) + dt * b * T_air;
        const double Tnew = rhs / (1.0 + dt * (th * sa + b));
        lost_ += dt * b * (Tnew - T_air) * pc * r;
        T_[n] = Tnew;
      }
  }
};

}  // namespace lbm
