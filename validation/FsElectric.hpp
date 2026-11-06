#pragma once
//==============================================================================
//  The electric pull on a conducting free surface -- a FreeSurfaceSolver liquid
//  sitting in the field of the electrodes around it.
//
//  MEASURED (validation/fs_electric.cpp, 2026-11-03, FP64): the stress map on
//  a flat conductor is within 0.05 % of the exact linear pattern, the feedback
//  of a deformed conductor within 1.3 %, a flat surface in a uniform field stays
//  flat to 1e-29, and the live two-way coupling lands within 3 % of the exact
//  response once the free surface's own stiffness -- which a frozen-field run
//  measures -- is accounted for. The raw deflections are 9-17 % short, and that
//  is the free surface's floor at one-to-two-cell deflections, not this file's;
//  that case's banner says how it was separated.
//
//  THREE DIMENSIONS (2026-11-05). With a Domain of nz > 1 every loop runs over
//  z, the gradient gains d/dz, and the stress is the mean over a 3 x 3 x 3
//  block; a geometry callback may take (x, y, z), and a 2-D case keeps (x, y).
//  At nz = 1 the arithmetic is the old one in the old order: fs_electric's 2-D
//  rows and a 12 s field-on mew_nozzle rerun (250 series rows) reproduce bit
//  for bit. fs_electric's section C measures the new path: A1 turned so that
//  the field runs along z equals A1 to the last bit, and a conducting sphere in
//  a uniform field gets the exact field's amplitude within 0.13 % at a radius
//  of 10 cells, with a cell-by-cell scatter of ~7 % of the polar field that
//  does not shrink with the radius -- the staircase, seen from the next cell.
//  The host loops are single-threaded and run once per refresh. Measured in
//  mew_cone3d (80 x 80 x 136, a refresh every 20 fluid steps, 40 s simulated,
//  2026-11-06): the field -- those loops and the twenty potential steps between
//  them -- took 3,440 s against 18,298 s in the free surface (26 ms a step),
//  19 %. They grow with the box.
//
//  THE MODEL. The liquid is an equipotential conductor at the potential of the
//  electrode it touches (the nozzle), the gas is vacuum, and the field is
//  quasi-static. The traction on a conductor is then purely normal and purely
//  outward,
//
//      p_E = eps0 |E|^2 / 2,           E evaluated on the gas side,
//
//  so it is an outward pull, i.e. a REDUCTION of the gas pressure the free
//  surface sees. FreeSurfaceSolver::rho_G_of is exactly that door: its banner
//  argues that a non-uniform normal stress is a per-node gas density and
//  nothing else, and recoil.cpp measures it. Here
//
//      rho_G_of(n) = rho_G0 - p_E(n) / cs^2        at interface cells,
//
//  and surface tension, which the solver adds itself, superposes on it.
//
//  THE FIELD IS A SECOND SOLVER, NOT A NEW ONE. phi solves Laplace's equation
//  in the gas with Dirichlet values on every conductor, through ScalarSolver
//  and ScalarTRT -- CLAUDE.md's prescription for a potential. TRT at
//  Lambda = 1/4 has the steady state of ScalarBGK at tau = 1 whatever its own
//  rate, so the rate is spent on convergence: a large tau makes the iteration
//  a damped wave rather than a diffusion, and the slowest mode settles in O(L)
//  steps instead of O(L^2).
//
//  THE CONDUCTOR'S SURFACE IS A STAIRCASE. A free-surface interface cell is
//  conductor when its fill is >= 1/2, and anti-bounce-back then puts phi on the
//  half link, so the electrode the field sees can sit up to half a cell from
//  the liquid's surface. That is first order in dx/R; fs_electric.cpp measures
//  what it costs.
//
//  WHAT THIS DOES NOT DO.
//   - No free charge, no conductivity, no leaky dielectric. A polymer melt is
//     a poor conductor; treating the meniscus as charge-relaxed is the
//     classical Taylor-cone assumption and nothing more.
//   - No polarisation stress for the liquid as a dielectric.
//   - Liquid may be declared non-conducting by region (conduct_ok): a
//     conductor that reaches the grounded collector would short the gap.
//   - The field follows the surface with the potential solver's lag; the
//     caller chooses how many potential steps it spends per fluid step.
//==============================================================================
#include "collision/ScalarTRT.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FreeSurfaceSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace lbm {

// The fixed part of the electric geometry, set once by the case.
enum ElecRole : std::uint8_t {
  ElecFree      = 0,   // gas or liquid: decided by the free surface every refresh
  ElecElectrode = 1,   // held at its own value (nozzle metal, collector, a plate)
  ElecFrame     = 2,   // zero normal field: an open side of the box
};

template <class PL>
class FsElectric {
 public:
  using Coll = ScalarTRT<PL>;
  using Pot  = ScalarSolver<PL, EsotericPull<PL>, Coll>;

  // p_E = coef |grad phi|^2 in the FLUID's lattice pressure, phi in units of
  // the applied voltage and the gradient per cell.
  Real coef   = Real(0);
  Real rho_G0 = Real(1);
  Real liquid_value = Real(1);   // the potential of conducting liquid

  FsElectric(const Domain& d, Real tau_phi)
      : d_(d), pot_(d, make_coll(tau_phi)),
        role_(std::size_t(d.n_padded), ElecFree),
        value_(std::size_t(d.n_padded), Real(0)),
        ok_(std::size_t(d.n_padded), 1),
        cls_(std::size_t(d.n_padded), ScalarExcluded),
        phi_(std::size_t(d.n_padded), Real(0)),
        e2_(std::size_t(d.n_padded), Real(0)) {}

  // fn(x, y) -> ElecRole; val(x, y) -> the electrode's potential. A 3-D case
  // passes fn(x, y, z) and val(x, y, z); a 2-D one may keep two arguments.
  template <class RoleFn, class ValFn>
  void set_roles(RoleFn fn, ValFn val) {
    for (Index z = 0; z < d_.nz; ++z)
      for (Index y = 0; y < d_.ny; ++y)
        for (Index x = 0; x < d_.nx; ++x) {
          const Index n = d_.id(x, y, z);
          role_[n]  = at(fn, x, y, z);
          value_[n] = at(val, x, y, z);
        }
  }
  // fn(x, y[, z]) -> bool: may liquid here conduct? Default: everywhere.
  template <class Fn>
  void set_conduct_region(Fn fn) {
    for (Index z = 0; z < d_.nz; ++z)
      for (Index y = 0; y < d_.ny; ++y)
        for (Index x = 0; x < d_.nx; ++x) ok_[d_.id(x, y, z)] = at(fn, x, y, z) ? 1 : 0;
  }

  //----------------------------------------------------------------------------
  // Rebuild the field's geometry from the free surface. Only cells whose state
  // changed are touched in effect; a cell that stops being conductor keeps the
  // populations anti-bounce-back left it, whose sum is already close to the
  // conductor's value, so no reseed is needed.
  //----------------------------------------------------------------------------
  template <class FS>
  int refresh(const FS& fs) {
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
    auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
    int changed = 0;
    for (Index z = 0; z < d_.nz; ++z)
      for (Index y = 0; y < d_.ny; ++y)
        for (Index x = 0; x < d_.nx; ++x) {
          const Index n = d_.id(x, y, z);
          std::uint8_t c;
          if (role_[n] == ElecElectrode)      c = ScalarDirichlet;
          else if (role_[n] == ElecFrame)     c = ScalarAdiabatic;
          else {
            const std::uint8_t f = hf(n);
            const bool liquid = f == FsFluid || (f == FsInterface && he(n) >= Real(0.5));
            if (f == FsSolid)                 c = ScalarAdiabatic;   // an insulating wall
            else if (liquid && ok_[n])        c = ScalarDirichlet;
            else                              c = ScalarBulk;
          }
          if (c != cls_[n]) ++changed;
          cls_[n] = c;
        }
    const auto& cls = cls_;
    const auto& role = role_;
    const auto& val = value_;
    const Real lv = liquid_value;
    const Domain& dd = d_;
    pot_.set_geometry([&](Index x, Index y, Index z) -> std::uint8_t {
      return cls[dd.id(x, y, z)];
    });
    pot_.set_wall_values([&](Index x, Index y, Index z) -> Real {
      const Index n = dd.id(x, y, z);
      return role[n] == ElecElectrode ? val[n] : lv;
    });
    return changed;
  }

  void initialize(Real guess) { pot_.initialize(guess); }
  void relax(long k) { for (long i = 0; i < k; ++i) pot_.step(); }

  //----------------------------------------------------------------------------
  // |grad phi|^2 at every gas cell, then the stress at every interface cell as
  // the mean over the gas cells of its 3x3 block. Returns max |dphi| since the
  // previous call, the potential's own convergence monitor.
  //
  // THE GRADIENT KNOWS WHERE THE ELECTRODES ARE. Anti-bounce-back puts a
  // Dirichlet value on the HALF LINK, so beside a conductor the three points of
  // the difference are at s = -1/2 (the wall), 0 and 1, not at -1, 0, 1:
  //     phi'(0) = -4/3 phi_w + phi_0 + 1/3 phi_1,
  // exact for a quadratic. Reading the conductor cell's value as if it sat a
  // full cell away would misplace every surface field by half a cell.
  //----------------------------------------------------------------------------
  template <class FS>
  double apply(FS& fs) {
    pot_.compute_field();
    auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pot_.temperature());
    double dmax = 0;
    for (Index n = 0; n < d_.n_padded; ++n) {
      const Real v = (cls_[n] == ScalarDirichlet) ? wall_value(n) : hp(n);
      dmax = std::max(dmax, std::fabs(double(v - phi_[n])));
      phi_[n] = v;
    }
    const bool three = d_.nz > 1;
    for (Index z = 0; z < d_.nz; ++z)
      for (Index y = 0; y < d_.ny; ++y)
        for (Index x = 0; x < d_.nx; ++x) {
          const Index n = d_.id(x, y, z);
          if (cls_[n] != ScalarBulk) { e2_[n] = Real(0); continue; }
          const Real gx = deriv(x, y, z, 1, 0, 0), gy = deriv(x, y, z, 0, 1, 0);
          Real e = gx * gx + gy * gy;
          if (three) { const Real gz = deriv(x, y, z, 0, 0, 1); e += gz * gz; }
          e2_[n] = e;
        }

    if (!fs.rho_G_of.data()) fs.rho_G_of = View1D<Real>("rho_G_of", d_.n_padded);
    if (h_rg_.extent(0) != std::size_t(d_.n_padded))
      h_rg_ = Kokkos::create_mirror_view(fs.rho_G_of);
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
    constexpr Real ics = inv_cs2<typename FS::Lattice, Real>();
    max_pe_ = 0;
    for (Index n = 0; n < d_.n_padded; ++n) h_rg_(n) = rho_G0;
    // Every non-wall cell that touches the gas gets a value, not only today's
    // interface cells: rho_G_of is read at INTERFACE cells, and a cell that
    // becomes one between two calls would otherwise feel no pull at all until
    // the next. The first version wrote interface cells only, and a surface
    // that moved a cell between calls lost its stress there.
    // In 3-D the block is 3 x 3 x 3; in 2-D (nz = 1) it stays the 3 x 3 of the
    // plane, summed in the same order, so a 2-D case is unchanged bit for bit.
    const int zr = three ? 1 : 0;
    for (Index z = 0; z < d_.nz; ++z)
      for (Index y = 0; y < d_.ny; ++y)
        for (Index x = 0; x < d_.nx; ++x) {
          const Index n = d_.id(x, y, z);
          const std::uint8_t f = hf(n);
          if ((f != FsInterface && f != FsFluid && f != FsGas) || !ok_[n]) continue;
          Real s = 0; int c = 0;
          for (int dz = -zr; dz <= zr; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                Index m;
                if (!neighbour(x, y, z, dx, dy, dz, m)) continue;
                if (cls_[m] != ScalarBulk) continue;
                s += e2_[m]; ++c;
              }
          if (c == 0) continue;
          const Real pe = coef * s / Real(c);
          if (f == FsInterface) max_pe_ = std::max(max_pe_, double(pe));
          h_rg_(n) = rho_G0 - pe * ics;
        }
    Kokkos::deep_copy(fs.rho_G_of, h_rg_);
    return dmax;
  }

  double max_stress() const { return max_pe_; }
  Real phi(Index x, Index y, Index z = 0) const { return phi_[d_.id(x, y, z)]; }
  Real e2(Index x, Index y, Index z = 0) const { return e2_[d_.id(x, y, z)]; }
  std::uint8_t cell(Index x, Index y, Index z = 0) const { return cls_[d_.id(x, y, z)]; }
  Pot& potential() { return pot_; }

 private:
  // A geometry callback with (x, y, z) or, for a 2-D case, (x, y).
  template <class Fn>
  static auto at(Fn& fn, Index x, Index y, Index z) {
    if constexpr (std::is_invocable_v<Fn&, Index, Index, Index>) return fn(x, y, z);
    else return fn(x, y);
  }
  static Coll make_coll(Real tau) {
    Coll c;
    c.omega  = Real(1) / tau;
    c.Lambda = Real(0.25);
    c.T_ref  = Real(0.5);
    return c;
  }
  Real wall_value(Index n) const {
    return role_[n] == ElecElectrode ? value_[n] : liquid_value;
  }
  bool neighbour(Index x, Index y, Index z, int dx, int dy, int dz, Index& m) const {
    Index xx = x + dx, yy = y + dy, zz = z + dz;
    if (d_.periodic[0]) xx = (xx + d_.nx) % d_.nx;
    if (d_.periodic[1]) yy = (yy + d_.ny) % d_.ny;
    if (d_.periodic[2]) zz = (zz + d_.nz) % d_.nz;
    if (xx < 0 || xx >= d_.nx || yy < 0 || yy >= d_.ny || zz < 0 || zz >= d_.nz) return false;
    m = d_.id(xx, yy, zz);
    return true;
  }
  // d phi / d(axis) at bulk cell (x, y, z), electrode- and mirror-aware.
  Real deriv(Index x, Index y, Index z, int ax, int ay, int az) const {
    const Index n = d_.id(x, y, z);
    const Real p0 = phi_[n];
    Index l, r;
    const bool hl = neighbour(x, y, z, -ax, -ay, -az, l), hr = neighbour(x, y, z, ax, ay, az, r);
    const std::uint8_t cl = hl ? cls_[l] : std::uint8_t(ScalarAdiabatic);
    const std::uint8_t cr = hr ? cls_[r] : std::uint8_t(ScalarAdiabatic);
    const bool bl = cl == ScalarBulk, br = cr == ScalarBulk;
    const bool wl = cl == ScalarDirichlet, wr = cr == ScalarDirichlet;
    if (bl && br) return Real(0.5) * (phi_[r] - phi_[l]);
    if (wl && br) return -Real(4) / 3 * phi_[l] + p0 + phi_[r] / 3;
    if (bl && wr) return  Real(4) / 3 * phi_[r] - p0 - phi_[l] / 3;
    if (wl && wr) return phi_[r] - phi_[l];
    if (br) return Real(0.5) * (phi_[r] - p0);        // mirror on the left
    if (bl) return Real(0.5) * (p0 - phi_[l]);        // mirror on the right
    if (wl) return Real(2) * (p0 - phi_[l]);          // wall left, mirror right
    if (wr) return Real(2) * (phi_[r] - p0);
    return Real(0);
  }

  Domain d_;
  Pot pot_;
  std::vector<std::uint8_t> role_;
  std::vector<Real> value_;
  std::vector<std::uint8_t> ok_;
  std::vector<std::uint8_t> cls_;
  std::vector<Real> phi_, e2_;
  typename View1D<Real>::host_mirror_type h_rg_;
  double max_pe_ = 0;
};

}  // namespace lbm
