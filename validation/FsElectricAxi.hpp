#pragma once
//==============================================================================
//  The electric pull on a conducting free surface in r-z -- FreeSurfaceSolver's
//  axisymmetric mode -- and an OHMIC melt, which may touch the collector.
//
//  THE GEOMETRY IS FsElectric's, the solver is not. The roles are the same
//  (ElecElectrode held at its value, ElecFrame a zero-normal-field side, ElecFree
//  decided by the free surface every refresh), the conductor's surface is the
//  same staircase (an interface cell conducts when its fill is >= 1/2, and the
//  conductor's value sits on the half link), the stress is the same
//  p_E = eps0 |E|^2 / 2 on the gas side, written into rho_G_of. What differs:
//
//  THE POTENTIAL IS A FINITE-VOLUME SOLVE OF div(r grad phi) = 0, not an LBM.
//  Cell (i, j) has its centre at r = i + 1/2, as the fluid's does; its radial
//  faces carry the weights r = i + 1 and r = i (zero on the axis, which is then
//  a zero-flux face for free), its axial faces r = i + 1/2. A Dirichlet value on
//  a face half a cell away doubles that face's weight. Red-black SOR, warm-
//  started from the last refresh. Why not ScalarSolver as FsElectric does: the
//  axisymmetric Laplacian is a conductivity proportional to r, and an LBM with a
//  per-node rate (ScalarBGK's omega_of) would run near tau = 1/2 at the axis --
//  exactly where the cone's tip sits -- and would set the face conductivities by
//  its own interface rule rather than by the face's radius.
//
//  THE OHMIC MELT. A polymer melt conducts poorly but conducts: Murat's
//  estimate puts its charge-relaxation time eps/sigma at <= 0.03 s, far below
//  the drop's seconds, so on the drop's time scale the melt carries current
//  rather than sitting at one potential. With `ohmic` on, the melt's own
//  potential is solved, div(r grad phi) = 0 inside it, with the nozzle and the
//  collector as its electrodes and NO current through its free surface (air does
//  not conduct). Before the melt touches the collector the only electrode it
//  touches is the nozzle, so it IS the equipotential drop -- the same field to
//  round-off, measured. After contact the potential falls along the column
//  instead of shorting the gap, so the conductor cut-off FsElectric needs
//  (set_conduct_region, mew_nozzle's -lc) and its artificial edge are not
//  needed. The conductivity's VALUE drops out while it is uniform, which is
//  why Murat's unmeasured sigma does not matter here.
//
//  WHAT THIS DOES NOT DO.
//   - After contact the melt's surface carries a tangential field E_t, a
//     surface charge eps0 E_n and so a TANGENTIAL electric traction eps0 E_n E_t,
//     and the normal traction becomes eps0 (E_n^2 - E_t^2)/2 + eps_m E_t^2/2.
//     Only eps0 |E|^2 / 2 from the gas side is applied, as before contact. Both
//     omissions vanish before contact (E_t = 0 on an equipotential).
//   - No free charge in the air, so no corona: the field is unbounded where the
//     real air would break down (Peek, ~12 kV/mm for a 0.2-0.3 mm tip).
//   - A melt region touching no electrode keeps whatever potential it had (a
//     floating conductor's charge is not tracked).
//==============================================================================
#include "FsElectric.hpp"          // ElecRole
#include "solver/FreeSurfaceSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace lbm {

class FsElectricAxi {
 public:
  // p_E = coef |grad phi|^2 in the FLUID's lattice pressure, phi in units of
  // the applied voltage and the gradient per cell -- FsElectric's convention.
  Real coef   = Real(0);
  Real rho_G0 = Real(1);
  Real liquid_value = Real(1);   // the nozzle's potential; an equipotential melt sits at it
  bool ohmic  = false;           // solve the potential inside the melt
  double omega_sor = 1.9;

  explicit FsElectricAxi(const Domain& d)
      : d_(d), role_(std::size_t(d.n_padded), ElecFree), value_(std::size_t(d.n_padded), Real(0)),
        ok_(std::size_t(d.n_padded), 1), cls_(std::size_t(d.n_padded), CF),
        phi_(std::size_t(d.n_padded), 0.0), e2_(std::size_t(d.n_padded), 0.0) {
    if (d.nz != 1) Kokkos::abort("FsElectricAxi: r-z only (nz = 1)");
  }

  template <class RoleFn, class ValFn>
  void set_roles(RoleFn fn, ValFn val) {
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        role_[n] = fn(x, y);
        value_[n] = val(x, y);
      }
  }
  // fn(x, y) -> bool: may liquid here conduct? Default: everywhere.
  template <class Fn>
  void set_conduct_region(Fn fn) {
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) ok_[d_.id(x, y)] = fn(x, y) ? 1 : 0;
  }

  // Reclassify from the free surface. A cell that becomes melt or gas keeps its
  // last potential as the warm start. Returns how many cells changed class.
  template <class FS>
  int refresh(const FS& fs) {
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
    auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
    int changed = 0;
    touching_ = false;
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        std::uint8_t c;
        if (role_[n] == ElecElectrode)   c = CE;
        else if (role_[n] == ElecFrame)  c = CF;
        else {
          const std::uint8_t f = hf(n);
          const bool liquid = f == FsFluid || (f == FsInterface && he(n) >= Real(0.5));
          if (f == FsSolid || f == FsExcluded) c = CF;      // an insulating wall
          else if (liquid && ok_[n])           c = CM;
          else                                 c = CG;
        }
        if (c != cls_[n]) {
          ++changed;
          if (c == CE) phi_[n] = value_[n];
          else if (c == CM && !ohmic) phi_[n] = liquid_value;
        }
        cls_[n] = c;
      }
    // does the melt touch an electrode other than the one at liquid_value?
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        if (cls_[d_.id(x, y)] != CM) continue;
        for (int k = 0; k < 4; ++k) {
          Index m;
          if (!nb(x, y, k, m)) continue;
          if (cls_[m] == CE && value_[m] != liquid_value) touching_ = true;
        }
      }
    return changed;
  }

  // Red-black SOR until the largest update is below tol, at most max_sweeps.
  // Returns the sweeps used.
  int solve(int max_sweeps, double tol) {
    int s = 0;
    for (; s < max_sweeps; ++s) {
      double dmax = 0;
      for (int colour = 0; colour < 2; ++colour)
        for (Index y = 0; y < d_.ny; ++y)
          for (Index x = (y + colour) % 2; x < d_.nx; x += 2) {
            const Index n = d_.id(x, y);
            const std::uint8_t c = cls_[n];
            if (c != CG && !(c == CM && ohmic)) continue;
            double a = 0, b = 0;
            const double w[4] = {double(x) + 1.0, double(x), double(x) + 0.5, double(x) + 0.5};
            for (int k = 0; k < 4; ++k) {
              Index m;
              if (!nb(x, y, k, m)) continue;               // the axis, or outside: no flux
              const std::uint8_t cm = cls_[m];
              if (cm == CF) continue;                       // insulator: no flux
              if (c == CM && cm == CG) continue;            // melt: no current into the air
              const bool dir = (cm == CE) || (c == CG && cm == CM);
              const double ww = dir ? 2.0 * w[k] : w[k];    // a face value half a cell away
              const double pv = (cm == CM && !ohmic) ? double(liquid_value) : phi_[m];
              a += ww;  b += ww * pv;
            }
            if (a <= 0) continue;
            const double pn = (1.0 - omega_sor) * phi_[n] + omega_sor * b / a;
            dmax = std::max(dmax, std::fabs(pn - phi_[n]));
            phi_[n] = pn;
          }
      sweeps_total_ += 1;
      if (dmax < tol) { ++s; break; }
    }
    return s;
  }

  // |grad phi|^2 at gas cells, then the stress at every free cell as the mean
  // over the gas cells of its 3x3 block -- FsElectric::apply, in r-z.
  template <class FS>
  void apply(FS& fs) {
    max_e2_ = 0;
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        if (cls_[n] != CG) { e2_[n] = 0; continue; }
        const double gr = deriv(x, y, 0), gz = deriv(x, y, 1);
        e2_[n] = gr * gr + gz * gz;
        max_e2_ = std::max(max_e2_, e2_[n]);
      }
    if (!fs.rho_G_of.data()) fs.rho_G_of = View1D<Real>("rho_G_of", d_.n_padded);
    if (h_rg_.extent(0) != std::size_t(d_.n_padded)) h_rg_ = Kokkos::create_mirror_view(fs.rho_G_of);
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
    constexpr Real ics = inv_cs2<typename FS::Lattice, Real>();
    max_pe_ = 0;
    for (Index n = 0; n < d_.n_padded; ++n) h_rg_(n) = rho_G0;
    for (Index y = 0; y < d_.ny; ++y)
      for (Index x = 0; x < d_.nx; ++x) {
        const Index n = d_.id(x, y);
        const std::uint8_t f = hf(n);
        if ((f != FsInterface && f != FsFluid && f != FsGas) || !ok_[n]) continue;
        double s = 0; int c = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const Index xx = x + dx, yy = y + dy;
            if (xx < 0 || xx >= d_.nx || yy < 0 || yy >= d_.ny) continue;
            const Index m = d_.id(xx, yy);
            if (cls_[m] != CG) continue;
            s += e2_[m]; ++c;
          }
        if (c == 0) continue;
        const double pe = double(coef) * s / double(c);
        if (f == FsInterface) max_pe_ = std::max(max_pe_, pe);
        h_rg_(n) = Real(double(rho_G0) - pe * double(ics));
      }
    Kokkos::deep_copy(fs.rho_G_of, h_rg_);
  }

  double max_stress() const { return max_pe_; }
  double max_e2() const { return max_e2_; }
  bool touching() const { return touching_; }
  long sweeps() const { return sweeps_total_; }
  double phi(Index x, Index y) const { return phi_[d_.id(x, y)]; }
  double e2(Index x, Index y) const { return e2_[d_.id(x, y)]; }
  // 1 electrode, 2 insulator / frame, 3 gas, 4 melt
  std::uint8_t cell(Index x, Index y) const { return cls_[d_.id(x, y)]; }
  void initialize(double guess) {
    for (Index n = 0; n < d_.n_padded; ++n)
      if (cls_[n] == CG || (cls_[n] == CM && ohmic)) phi_[n] = guess;
  }

 private:
  enum : std::uint8_t { CE = 1, CF = 2, CG = 3, CM = 4 };
  // neighbour k = E, W, N, S of interior (x, y); false off the domain (the
  // axis on the W side of column 0, and any non-periodic edge)
  bool nb(Index x, Index y, int k, Index& m) const {
    Index xx = x + (k == 0 ? 1 : (k == 1 ? -1 : 0));
    Index yy = y + (k == 2 ? 1 : (k == 3 ? -1 : 0));
    if (d_.periodic[1]) yy = (yy + d_.ny) % d_.ny;
    if (xx < 0 || xx >= d_.nx || yy < 0 || yy >= d_.ny) return false;
    m = d_.id(xx, yy);
    return true;
  }
  // the potential a neighbour presents to a GAS cell: its own if gas, the
  // conductor's on the half link if melt or electrode
  double val(Index m) const {
    if (cls_[m] == CM && !ohmic) return double(liquid_value);
    return phi_[m];
  }
  // d phi / dr (ax = 0) or d phi / dz (ax = 1) at gas cell (x, y), per cell.
  // Dirichlet neighbours sit on the half link (phi'(0) = -4/3 phi_w + phi_0 +
  // 1/3 phi_1, exact for a quadratic), insulators and the axis are mirrors.
  double deriv(Index x, Index y, int ax) const {
    const Index n = d_.id(x, y);
    const double p0 = phi_[n];
    Index l, r;
    const bool hl = nb(x, y, ax == 0 ? 1 : 3, l), hr = nb(x, y, ax == 0 ? 0 : 2, r);
    const std::uint8_t cl = hl ? cls_[l] : std::uint8_t(CF), cr = hr ? cls_[r] : std::uint8_t(CF);
    const bool bl = cl == CG, br = cr == CG;
    const bool wl = cl == CE || cl == CM, wr = cr == CE || cr == CM;
    if (bl && br) return 0.5 * (phi_[r] - phi_[l]);
    if (wl && br) return -4.0 / 3.0 * val(l) + p0 + phi_[r] / 3.0;
    if (bl && wr) return 4.0 / 3.0 * val(r) - p0 - phi_[l] / 3.0;
    if (wl && wr) return val(r) - val(l);
    if (br) return 0.5 * (phi_[r] - p0);          // mirror on the left (the axis, an insulator)
    if (bl) return 0.5 * (p0 - phi_[l]);
    if (wl) return 2.0 * (p0 - val(l));
    if (wr) return 2.0 * (val(r) - p0);
    return 0.0;
  }

  Domain d_;
  std::vector<std::uint8_t> role_;
  std::vector<Real> value_;
  std::vector<std::uint8_t> ok_, cls_;
  std::vector<double> phi_, e2_;
  typename View1D<Real>::host_mirror_type h_rg_;
  double max_pe_ = 0, max_e2_ = 0;
  bool touching_ = false;
  long sweeps_total_ = 0;
};

}  // namespace lbm
