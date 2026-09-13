//==============================================================================
//  Laminar flow past a square cylinder in a plane channel -- Breuer, Bernsdorf,
//  Zeiser & Durst, Int. J. Heat and Fluid Flow 21 (2000) 186-196.
//
//  THE SETUP, taken from Sec. 3.1 and Fig. 1 of that paper:
//
//      square of side D, blockage B = D/H = 1/8, so the channel is H = 8D
//      channel length L = 50D, cylinder centred l = L/4 = 12.5D from the inlet
//      parabolic inlet of peak speed u_max, pressure outlet
//      bounce-back on the channel walls AND on the cylinder
//      Re = u_max D / nu          -- on the PEAK inlet speed, not the mean
//      C_d = F_x / (rho u_max^2 D / 2)
//
//  Re is on the peak of the parabola and the paper says so twice (Sec. 4 and
//  footnote 3). Using the mean instead, which is 2/3 of the peak for a plane
//  channel, would move every Reynolds number by 1.5x and every C_d with it.
//
//  WALL CONVENTION. Halfway bounce-back puts the wall plane half a cell outside
//  the last solid cell, so D SOLID CELLS give an obstacle of side exactly D and
//  H FLUID ROWS between two solid rows give a channel of height exactly H. That
//  is why the domain is ny = 8D + 2 and not 8D: the paper's "500 x 80" counts
//  its 80 control volumes across the channel, and 80 fluid rows plus the two
//  bounding wall rows is 82 lattice rows here. Getting this wrong changes the
//  blockage ratio, which the paper shows is the parameter the Strouhal number
//  is most sensitive to.
//
//  THE FORCE IS BY MOMENTUM EXCHANGE, and it needs nothing from the solver's
//  internals. For a link from fluid node n into a solid neighbour along c_i,
//  halfway bounce-back returns the emitted f_i as f_opp(i) one step later, so
//  the momentum handed to the wall is 2 c_i f_i^post-collision. The population
//  ARRIVING at n along opp(i) IS that f_i, which is exactly what
//  `gather_populations()` reports -- "the population travelling in direction i
//  that is about to arrive at node n", defined independently of how the
//  streaming scheme lays memory out. So
//
//      F = sum over boundary links of  2 c_i g(n, opp(i))
//
//  is correct under Esoteric Pull without knowing anything about Esoteric Pull.
//  Reaching into the raw array instead would have to know which of the two
//  slots of an opposite pair is live at this parity, which is the class of
//  mistake that cost this tree a first-cell gradient in GPU/'s Poisson source.
//  The reported force lags by one step; in a steady state that is nothing, and
//  every Re here is steady.
//
//  -selftest IS NOT OPTIONAL AND IS RUN FIRST. A new force evaluation that has
//  never reproduced a force it cannot get wrong is not a measurement. In a
//  periodic channel driven by a uniform body force G, steady state requires the
//  walls to carry the entire body force: sum of the wall force = G * (number of
//  fluid nodes), exactly, with no appeal to a profile or a viscosity. If the
//  momentum-exchange sum does not reproduce that, nothing downstream of it
//  means anything.
//
//  WHAT CAN AND CANNOT BE CHECKED AGAINST THE PAPER. It publishes C_d for
//  steady flow only as log-log FIGURES (Fig. 5a for FVM on three grids, Fig. 5b
//  for LBA against FVM on the finest) -- there is no table. Values read off
//  those axes are good to perhaps 5-10 %, and the paper's OWN two methods
//  differ by about 9 % at Re = 10, which it attributes to the LBA's resolution
//  of the boundary layer. So the comparison here is a comparison against a
//  digitised curve with a stated uncertainty, and a percentage agreement
//  quoted to three figures against it would be a fiction. The grid convergence
//  below is the part that stands on its own.
//==============================================================================
#include "Campaign.hpp"

using namespace lbm;
using namespace campaign;

using L   = D2Q9;
using Coll  = BGK<L, SecondOrderEquilibrium<L>, NoForcing, ShiftedPopulations>;
using CollF = BGK<L, SecondOrderEquilibrium<L>, Guo,       ShiftedPopulations>;
// The paper is lattice-BGK throughout, so BGK is the default and -op cm is a
// stated departure. It is not a cosmetic one: BGK at Re = 300 on the 2000x320
// grid sits at tau = 0.512 for u_max = 0.03 and DIVERGED at step 79999, as the
// wake became energetic rather than at startup. Central moments relax the ghost
// modes independently instead of at the single rate that makes BGK a near-
// reflection as tau -> 1/2, which is the whole reason this tree carries four
// operators.
using CM = CentralMoments<L, NoForcing, ShiftedPopulations>;

// UNSHIFT BEFORE TAKING A ONE-SIDED MOMENT. gather_populations() returns the
// STORED value, and shifted storage holds g_i = f_i - w_i. Summing c_i over
// every direction the weights cancel, because sum_i c_i w_i = 0 -- which is
// exactly why the shift is invisible in rho and u, and exactly why it is NOT
// invisible here: a momentum-exchange sum runs over the directions pointing
// into the wall and no further, and over that half the weights carry a real
// momentum. The error would have been a fixed offset proportional to the
// surface area, i.e. a plausible drag that does not vanish with the flow.
template <class C>
KOKKOS_INLINE_FUNCTION Real unshift(Real stored, int i) {
  if constexpr (C::Storage::shifted) return stored + weight<L, Real>(i);
  else                                  return stored;
}

// opp(i) is i+1 for odd i and i-1 for even, the ordering contract that
// Lattices.hpp static_asserts and that Esoteric Pull depends on.
static inline int opp_of(int i) { return i == 0 ? 0 : ((i % 2) ? i + 1 : i - 1); }

struct Link { Index n; int i; };   // fluid node n, direction i into the solid

//------------------------------------------------------------------------------
// Legacy binary VTK, STRUCTURED_POINTS. ParaView groups files named
// <base>_0000.vtk, _0001.vtk ... into a time series on its own, so there is no
// .pvd to keep in step with the data.
//
// BIG-ENDIAN, because the legacy binary format says so and does not carry a
// byte-order field to say otherwise. Writing native little-endian produces a
// file ParaView opens without complaint and renders as noise -- values around
// 1e-40 and 1e+38 -- which reads as a diverged simulation rather than as a
// broken writer.
//
// VORTICITY is what makes a von Karman street visible; speed alone shows the
// wake but not the vortices. It is a central difference on fluid neighbours and
// is set to zero wherever the stencil would reach into a solid, so the obstacle
// and the walls do not emit a spurious sheet.
//------------------------------------------------------------------------------
static void put_be(std::FILE* f, float v) {
  unsigned char* p = reinterpret_cast<unsigned char*>(&v);
  const unsigned char b[4] = {p[3], p[2], p[1], p[0]};
  std::fwrite(b, 1, 4, f);
}

template <class Solid>
static void write_vtk(const std::string& path, Index nx, Index ny, const Domain& d,
                      const std::vector<double>& ux, const std::vector<double>& uy,
                      Solid solid) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { std::printf("  cannot open %s\n", path.c_str()); return; }
  std::fprintf(f, "# vtk DataFile Version 3.0\nsquare cylinder\nBINARY\n"
                  "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n"
                  "ORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA %lld\n",
               int(nx), int(ny), (long long)(std::size_t(nx) * std::size_t(ny)));
  auto at = [&](Index x, Index y) { return std::size_t(d.id(x, y, 0)); };
  std::fprintf(f, "SCALARS vorticity float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      double w = 0;
      const bool edge = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
      if (!edge && !solid(x, y) && !solid(x + 1, y) && !solid(x - 1, y) &&
          !solid(x, y + 1) && !solid(x, y - 1))
        w = 0.5 * (uy[at(x + 1, y)] - uy[at(x - 1, y)]) -
            0.5 * (ux[at(x, y + 1)] - ux[at(x, y - 1)]);
      put_be(f, float(w));
    }
  std::fprintf(f, "\nSCALARS solid float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) put_be(f, solid(x, y) ? 1.0f : 0.0f);
  std::fprintf(f, "\nVECTORS velocity float\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const bool sd = solid(x, y);
      put_be(f, sd ? 0.0f : float(ux[at(x, y)]));
      put_be(f, sd ? 0.0f : float(uy[at(x, y)]));
      put_be(f, 0.0f);
    }
  std::fclose(f);
}


struct Opts {
  Real umax = Real(0.02);
  double seed = 0.0, ctol = 1e-6;
  std::size_t maxsteps = 400000, probe = 2000, transient = 0, avg = 0, vtkevery = 0;
  std::size_t vtkfrom = 0;
  std::string vtkdir;
  std::string outlet = "press";
};

//------------------------------------------------------------------------------
// One (grid, Reynolds) point. Templated on the collision operator rather than
// branching on it: the operator is a compile-time policy of FluidSolver, so a
// runtime branch would need both solvers instantiated at once. Same reason
// demonstrator/rb_high_ra keeps its Opts and run at namespace scope.
//------------------------------------------------------------------------------
template <class C>
void run_one(const Opts& o, Index D, double Re, int& status) {
      const Index Hc = 8 * D, ny = Hc + 2, nx = 50 * D;
      const Index x0 = 12 * D;                    // centre at 12.5D - 0.5
      const Index y0 = Index(3.5 * double(D)) + 1;  // centred on the channel
      const Real nu = Real(double(o.umax) * double(D) / Re);
      Domain d(nx, ny, 1, false, false, false);
      C coll;
      coll.omega = C::omega_from_viscosity(nu);
      FluidSolver<L, EsotericPull<L>, C> s(d, coll);
      auto is_cyl = [&](Index x, Index y) {
        return x >= x0 && x < x0 + D && y >= y0 && y < y0 + D;
      };
      s.set_geometry([&](Index x, Index y, Index) -> CellType {
        if (y == 0 || y == ny - 1) return Solid;
        if (is_cyl(x, y)) return Solid;
        if (x == 0 || x == nx - 1) return RegWall;
        return Fluid;
      });
      // Parabola vanishing on the two wall PLANES at y = 0.5 and y = H + 0.5.
      auto prof = [&](Index y) {
        const double yy = double(y);
        return 4.0 * double(o.umax) * (yy - 0.5) * (double(Hc) + 0.5 - yy) /
               (double(Hc) * double(Hc));
      };
      using WS = decltype(s)::WallSpec;
      s.set_regularized_walls([&](Index x, Index y, Index) -> WS {
        if (x == 0)      return WS{NrmXm, Real(prof(y)), Real(0), Real(0), Real(1)};
        // TWO OUTLETS, and at Re = 300 the choice is not cosmetic.
        //
        //   press  NrmOutXp -- constant back-pressure. rho imposed, normal
        //          velocity from the inverted closure, tangential zero-gradient
        //          from upstream. This is what the paper describes for its LBA
        //          and it is right for a steady flow, but it is REFLECTING: at
        //          Re = 300 the field went non-finite in a solid block spanning
        //          x = 1745..1999 over the FULL channel height, 255 columns
        //          anchored on the outlet, which is a boundary-column failure
        //          and not a wake one. The paper used a convective condition
        //          du/dt + u_conv du/dx = 0 for its FVM precisely so that
        //          "vortices can approach and pass the outflow boundary without
        //          significant disturbances or reflections".
        //
        //   eq     NrmOutEq -- the paper's own: rho imposed, all velocity
        //          components extrapolated from upstream, populations set to
        //          EQUILIBRIUM. Discards f^(1), so it is the most dissipative
        //          of the three and the only one measured to survive Re = 300.
        //
        //   free   NrmOutFree -- take the upstream neighbour's whole
        //          distribution and rescale it to the imposed density. That is
        //          zero-gradient on the SHAPE of the distribution and Dirichlet
        //          only on the pressure, so an arriving vortex passes through
        //          as a shape rather than being re-derived from moments.
        //
        // On a flat face this tree has measured `free` at constant rho as an
        // 8.4e-2 bulk mass source against 1.3e-6 for `press`, so it is the
        // worse outlet for a steady channel and is not the default. What it
        // costs HERE is measured below by running the steady points both ways.
        if (x == nx - 1) {
          if (o.outlet == "free")
            return WS{NrmOutFree, Real(0), Real(0), Real(0), Real(1)};
          if (o.outlet == "eq")
            return WS{NrmOutEq, Real(0), Real(0), Real(0), Real(1)};
          return WS{NrmOutXp, Real(0), Real(0), Real(0), Real(1)};
        }
        return WS{};
      });
      // Start from the developed channel rather than from rest: the inlet
      // profile is imposed exactly, so the channel never needs to develop and
      // only the wake does. Starting from rest instead spends the whole
      // H^2/nu diffusive time reproducing a profile that was already known,
      // which at Re = 1 is 128000 steps of nothing.
      s.initialize_field(KOKKOS_LAMBDA(Index n) {
        Index x, y, z; d.coords(n, x, y, z); (void)z;
        const bool solid = (y == 0 || y == ny - 1) ||
                           (x >= x0 && x < x0 + D && y >= y0 && y < y0 + D);
        const double u = solid ? 0.0
                               : 4.0 * double(o.umax) * (double(y) - 0.5) *
                                     (double(Hc) + 0.5 - double(y)) /
                                     (double(Hc) * double(Hc));
        // SYMMETRY-BREAKING SEED. The cylinder sits exactly on the channel
        // centreline, so the whole problem is mirror-symmetric about it and
        // the shedding mode is orthogonal to every symmetric state: from a
        // symmetric start the wake stays symmetric until round-off breaks it,
        // which takes far longer than the run. One transverse blob just
        // behind the cylinder removes that. It sets WHEN shedding starts, and
        // for a supercritical Hopf bifurcation not the amplitude it saturates
        // at -- but this tree has already been caught assuming that of a
        // SUBcritical one, so the check is to halve it and confirm the
        // averaged C_d and St do not move, not to assume.
        const double xs = double(x) - (double(x0) + 1.5 * double(D));
        const double ys = double(y) - (double(y0) + 0.5 * double(D));
        const double v = solid ? 0.0
                               : o.seed * double(o.umax) *
                                     std::exp(-(xs * xs + ys * ys) /
                                              (0.5 * double(D) * double(D)));
        return FlowState{Real(1), Real(u), Real(v), Real(0)};
      });
      const Real m0 = s.total_mass();
      // MASS SERVO, for the outlets that impose rho AND u.
      //
      // NrmOutXp derives the normal velocity from the inverted closure, so it
      // passes whatever flux the interior sends and holds mass to 2.6e-4 on its
      // own. NrmOutEq and NrmOutFree impose both, which over-determines the
      // boundary and makes it a source: measured 8.31e-2 and 8.36e-2 here, and
      // the drag normalised by rho = 1 then read 8.30 % and 8.35 % high --
      // matching the drift to three digits, so the whole discrepancy was the
      // density and none of it was the flow.
      //
      // The fix is the one demonstrator/aorta.cpp uses: close a loop on TOTAL
      // mass by moving the imposed outlet density, so the pressure at which
      // what leaves equals what enters is FOUND rather than assumed. Re-issuing
      // set_regularized_walls mid-run is safe -- verified elsewhere in this
      // tree by re-issuing with zero gain and reproducing the fixed outlet
      // exactly -- and is how the new density reaches the wall table. It costs
      // a domain sweep, so it runs on a coarse interval and never per step.
      const bool servo = (o.outlet != "press");
      double orho = 1.0;
      // One table write, not a domain sweep. Re-issuing set_regularized_walls
      // to move a single number rebuilt the geometry and the donor lists every
      // 500 steps and cost about 60% in wall clock.
      auto reissue = [&]() {
        s.set_wall_density(Real(orho));
        if (o.outlet == "free") s.set_outflow_density(Real(orho));
      };
      // Boundary links of the CYLINDER only -- the channel walls carry a real
      // force too and it is not the drag.
      std::vector<Link> links;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 1; x < nx - 1; ++x) {
          if (is_cyl(x, y)) continue;
          for (int i = 0; i < L::Q; ++i)
            if (is_cyl(x + cvel<L>(i, 0), y + cvel<L>(i, 1)))
              links.push_back(Link{d.id(x, y, 0), i});
        }
      auto forces = [&](double& Fx, double& Fy) {
        Fx = Fy = 0;
        auto g = s.gather_populations();
        auto hg = Kokkos::create_mirror_view_and_copy(HostSpace{}, g);
        for (const Link& lk : links) {
          const int io = opp_of(lk.i);
          const double f = double(unshift<C>(hg(lk.n, io), io));
          Fx += 2.0 * cvel<L>(lk.i, 0) * f;
          Fy += 2.0 * cvel<L>(lk.i, 1) * f;
        }
      };
      auto dump = [&](std::size_t idx) {
        if (o.vtkdir.empty()) return;
        s.compute_macroscopic();
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        std::vector<double> vx(std::size_t(d.n_padded)), vy(std::size_t(d.n_padded));
        for (Index n = 0; n < d.n_padded; ++n) { vx[n] = double(hx(n)); vy[n] = double(hy(n)); }
        char nm[512];
        std::snprintf(nm, sizeof nm, "%s/sq_re%03d_%04zu.vtk", o.vtkdir.c_str(), int(Re), idx);
        write_vtk(nm, nx, ny, d, vx, vy,
                  [&](Index x, Index y) { return y == 0 || y == ny - 1 || is_cyl(x, y); });
      };
      const double cd_norm = 0.5 * double(o.umax) * double(o.umax) * double(D);
      double cd = 0, prev = 0, rel = 1, cl = 0, Fx = 0, Fy = 0;
      std::size_t t = 0, vframe = 0;
      bool ok = true;
      if (o.transient == 0) {
        // STEADY protocol: stop when C_d stops moving.
        for (; t < o.maxsteps; ++t) {
          s.step();
          if (servo && (t + 1) % 500 == 0 && t > 2000) {
            const double err = double(s.total_mass()) / double(m0) - 1.0;
            double dr = 0.6 * err;                 // same gain as the aorta
            if (dr >  5e-4) dr =  5e-4;            // clamped; unclamped this
            if (dr < -5e-4) dr = -5e-4;            // loop goes bang-bang
            orho -= dr;
            if (orho < 0.9) orho = 0.9;
            if (orho > 1.1) orho = 1.1;
            reissue();
          }
          if ((t + 1) % o.probe) continue;
          prev = cd; forces(Fx, Fy); cd = Fx / cd_norm;
          if (!std::isfinite(cd)) { ok = false; break; }
          rel = (prev != 0) ? std::abs(cd - prev) / std::abs(cd) : 1.0;
          if (rel < o.ctol) break;
        }
        // MASS, because an outlet that imposes BOTH rho and u is a mass
        // source, and a drag normalised by rho = 1 then reads high by exactly
        // the density drift. Two different outlets came out 8.3 % above the
        // pressure outlet -- far too similar to be two separate physical
        // effects -- so print the one thing that would explain both at once.
        const double mrel = double(s.total_mass()) / double(m0) - 1.0;
        std::printf("  %4d %6.0f %7.4f %9.5f %9zu %10.4f %12.2e %9.4f %10.3e%s\n",
                    int(D), Re, double(nu), 1.0 / double(coll.omega), t + 1, cd, rel,
                    double(o.umax) / 0.5773502692, mrel, ok ? "" : "  NON-FINITE");
      } else {
        // UNSTEADY protocol. Above Re_crit ~ 60 the wake sheds and C_d never
        // stops moving, so a convergence test on it would run to the step cap
        // and report whatever phase it stopped in. Run a o.transient, then
        // TIME-AVERAGE over a window, and report the variation as well as the
        // mean -- an average without its amplitude hides whether the window
        // covered whole cycles.
        // A FLUSHED PROGRESS LINE, because the averaging window prints
        // nothing until it is finished and stdio block-buffers to a file: a
        // twenty-minute run was completely silent, and when it diverged there
        // was no way to see where. Check finiteness often enough to stop
        // promptly rather than grind out the rest of the o.transient on NaNs.
        // WATCH THE PRECURSOR, NOT THE CORPSE. Checking only for non-finite
        // values located nothing: NaN spreads about a cell a step, so by the
        // first check after onset 99.1 % of 644000 nodes were already bad and
        // the "first" one was simply the first in scan order. An instability
        // announces itself as a LOCAL velocity blow-up long before any NaN, so
        // track max|u| and where it is, and stop at a threshold while the field
        // is still finite and the location still means something.
        const double u_panic = 8.0 * double(o.umax);
        // DUMP THROUGH THE TRANSIENT, NOT ONLY THE AVERAGING WINDOW. Writing
        // frames only once the flow is deemed ready means a run that DIES
        // produces nothing at all -- which is precisely when a picture is worth
        // most. Four divergences here were diagnosed from three printed numbers
        // because there was no field to look at.
        for (; t < o.transient; ++t) {
          s.step();
          if (servo && (t + 1) % 500 == 0 && t > 2000) {
            const double err = double(s.total_mass()) / double(m0) - 1.0;
            double dr = 0.6 * err;                 // same gain as the aorta
            if (dr >  5e-4) dr =  5e-4;            // clamped; unclamped this
            if (dr < -5e-4) dr = -5e-4;            // loop goes bang-bang
            orho -= dr;
            if (orho < 0.9) orho = 0.9;
            if (orho > 1.1) orho = 1.1;
            reissue();
          }
          if (o.vtkevery && (t + 1) >= o.vtkfrom && (t + 1) % o.vtkevery == 0)
            dump(vframe++);
          if ((t + 1) % 1000 == 0) {
            s.compute_macroscopic();
            auto hx2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
            auto hy2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
            double um = 0; Index mx = 0, my = 0; bool fin = true;
            for (Index yy = 1; yy < ny - 1; ++yy)
              for (Index xx = 0; xx < nx; ++xx) {
                const Index m = d.id(xx, yy, 0);
                const double a = double(hx2(m)), b = double(hy2(m));
                if (!std::isfinite(a) || !std::isfinite(b)) { fin = false; continue; }
                const double sp = std::sqrt(a * a + b * b);
                if (sp > um) { um = sp; mx = xx; my = yy; }
              }
            if (!fin || um > u_panic) {
              std::printf("       BLOW-UP at step %zu: max|u| = %.4f at (%d,%d)%s\n",
                          t + 1, um, int(mx), int(my), fin ? "" : "  (NaN present)");
              std::printf("       cylinder x in [%d,%d], y in [%d,%d]; outlet x=%d; "
                          "channel y in [1,%d]\n",
                          int(x0), int(x0 + D - 1), int(y0), int(y0 + D - 1),
                          int(nx - 1), int(ny - 2));
              std::printf("       = %d past the cylinder's trailing edge, "
                          "%d short of the outlet, %d from the nearest wall\n",
                          int(mx) - int(x0 + D - 1), int(nx - 1) - int(mx),
                          int(std::min(my - 1, ny - 2 - my)));
              // The failing state itself, which is the one frame that shows
              // WHAT the blow-up looks like rather than where it is. NaNs
              // render as gaps in ParaView, so the hole is the answer.
              if (o.vtkevery) {
                dump(vframe++);
                std::printf("       wrote the failing field as frame %zu\n", vframe - 1);
              }
              std::fflush(stdout);
              ok = false; break;
            }
          }
          if ((t + 1) % 5000) continue;
          forces(Fx, Fy);
          if (!std::isfinite(Fx)) {
            // WHERE, not just when. Two different operators died within 13% of
            // the same step, so the cause is not the collision -- and the
            // candidates (inlet, outlet, cylinder corners) are told apart by
            // position and by nothing else. One sweep at the moment of death.
            std::printf("       NON-FINITE during the transient at step %zu\n", t + 1);
            s.compute_macroscopic();
            auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
            auto hx2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
            auto hy2 = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
            Index bx = -1, by = -1; std::size_t nbad = 0;
            double um = 0; Index mx = 0, my = 0;
            for (Index yy = 0; yy < ny; ++yy)
              for (Index xx = 0; xx < nx; ++xx) {
                const Index m = d.id(xx, yy, 0);
                const double a = double(hx2(m)), b = double(hy2(m)), r = double(hr(m));
                if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(r)) {
                  if (bx < 0) { bx = xx; by = yy; }
                  ++nbad;
                } else {
                  const double sp = std::sqrt(a * a + b * b);
                  if (sp > um) { um = sp; mx = xx; my = yy; }
                }
              }
            std::printf("       first non-finite node (%d,%d)   %zu bad of %lld\n",
                        int(bx), int(by), nbad,
                        (long long)(std::size_t(nx) * std::size_t(ny)));
            std::printf("       %d cells from the inlet, %d from the outlet, "
                        "%d past the cylinder; channel y in [1,%d]\n",
                        int(bx), int(nx - 1 - bx), int(bx - (x0 + D)), int(ny - 2));
            std::printf("       fastest finite |u| = %.4f at (%d,%d)   (u_max = %.4f)\n",
                        um, int(mx), int(my), double(o.umax));
            std::fflush(stdout);
            ok = false; break;
          }
          if ((t + 1) % 25000 == 0) {
            std::printf("       transient %zu/%zu   C_d %.4f   C_l %.4f\n",
                        t + 1, o.transient, Fx / cd_norm, Fy / cd_norm);
            std::fflush(stdout);
          }
        }
        std::vector<double> cds, cls; std::vector<std::size_t> ts;
        for (std::size_t k = 0; ok && k < o.avg; ++k, ++t) {
          s.step();
          if (servo && (t + 1) % 500 == 0 && t > 2000) {
            const double err = double(s.total_mass()) / double(m0) - 1.0;
            double dr = 0.6 * err;                 // same gain as the aorta
            if (dr >  5e-4) dr =  5e-4;            // clamped; unclamped this
            if (dr < -5e-4) dr = -5e-4;            // loop goes bang-bang
            orho -= dr;
            if (orho < 0.9) orho = 0.9;
            if (orho > 1.1) orho = 1.1;
            reissue();
          }
          if ((t + 1) % o.probe) continue;
          forces(Fx, Fy);
          if (!std::isfinite(Fx)) { ok = false; break; }
          cds.push_back(Fx / cd_norm); cls.push_back(Fy / cd_norm); ts.push_back(t + 1);
          if ((t + 1) % 25000 == 0) {
            std::printf("       averaging %zu/%zu   C_d %.4f   C_l %.4f\n",
                        t + 1 - o.transient, o.avg, Fx / cd_norm, Fy / cd_norm);
            std::fflush(stdout);
          }
          if (o.vtkevery && ((t + 1) % o.vtkevery == 0)) dump(vframe++);
        }
        double m = 0, lo = 1e30, hi = -1e30, lmin = 1e30, lmax = -1e30, lm = 0;
        for (std::size_t k = 0; k < cds.size(); ++k) {
          m += cds[k]; lm += cls[k];
          lo = std::min(lo, cds[k]); hi = std::max(hi, cds[k]);
          lmin = std::min(lmin, cls[k]); lmax = std::max(lmax, cls[k]);
        }
        if (!cds.empty()) { m /= double(cds.size()); lm /= double(cls.size()); }
        // Strouhal from UPWARD zero crossings of the lift about its own mean,
        // linearly interpolated. Counting peaks instead is noisier: the lift
        // is smooth at a crossing and flat at a peak.
        std::vector<double> cross;
        for (std::size_t k = 1; k < cls.size(); ++k)
          if (cls[k - 1] - lm < 0 && cls[k] - lm >= 0) {
            const double a = cls[k - 1] - lm, b = cls[k] - lm;
            cross.push_back(double(ts[k - 1]) + (0.0 - a) / (b - a) *
                                                     double(ts[k] - ts[k - 1]));
          }
        double St = 0;
        if (cross.size() >= 2) {
          const double period = (cross.back() - cross.front()) / double(cross.size() - 1);
          St = double(o.umax) > 0 ? double(D) / (period * double(o.umax)) : 0.0;
        }
        std::printf("  %4d %6.0f %7.4f %9.5f %9zu %10.4f %12s %9.4f%s\n",
                    int(D), Re, double(nu), 1.0 / double(coll.omega), t, m,
                    "unsteady", double(o.umax) / 0.5773502692, ok ? "" : "  NON-FINITE");
        std::printf("       time-averaged over %zu samples spanning %zu steps, "
                    "%.1f shedding cycles\n",
                    cds.size(), o.avg, cross.size() >= 2 ? double(cross.size() - 1) : 0.0);
        std::printf("       C_d  mean %.4f   min %.4f   max %.4f   max-min %.4f\n",
                    m, lo, hi, hi - lo);
        std::printf("       C_l  mean %.4f   min %.4f   max %.4f   max-min %.4f\n",
                    lm, lmin, lmax, lmax - lmin);
        std::printf("       Strouhal %.4f   (%zu upward lift crossings)\n",
                    St, cross.size());
        if (vframe) std::printf("       wrote %zu VTK frames to %s\n", vframe, o.vtkdir.c_str());
      }
      std::fflush(stdout);
      if (!ok) status = 1;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::vector<double> Res = {1.0, 10.0, 20.0, 50.0};
    std::vector<Index> Ds   = {10, 20};
    Real umax = Real(0.02);
    double seed = 0.0;          // transverse kick behind the cylinder
    std::size_t maxsteps = 400000, probe = 2000;
    double ctol = 1e-6;          // relative change in C_d over one probe window
    bool selftest_only = false;
    std::string op = "bgk", outlet = "press";
    std::size_t transient = 0, avg = 0, vtkevery = 0, vtkfrom = 0;
    std::string vtkdir;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-u" && i + 1 < argc) umax = Real(std::atof(argv[++i]));
      if (a == "-ctol" && i + 1 < argc) ctol = std::atof(argv[++i]);
      if (a == "-steps" && i + 1 < argc) maxsteps = std::size_t(std::atol(argv[++i]));
      if (a == "-probe" && i + 1 < argc) probe = std::size_t(std::atol(argv[++i]));
      if (a == "-selftest") selftest_only = true;
      if (a == "-transient" && i + 1 < argc) transient = std::size_t(std::atol(argv[++i]));
      if (a == "-avg" && i + 1 < argc) avg = std::size_t(std::atol(argv[++i]));
      if (a == "-vtk" && i + 1 < argc) vtkdir = argv[++i];
      if (a == "-vtkevery" && i + 1 < argc) vtkevery = std::size_t(std::atol(argv[++i]));
      if (a == "-vtkfrom" && i + 1 < argc) vtkfrom = std::size_t(std::atol(argv[++i]));
      if (a == "-seed" && i + 1 < argc) seed = std::atof(argv[++i]);
      if (a == "-op" && i + 1 < argc) op = argv[++i];
      if (a == "-outlet" && i + 1 < argc) outlet = argv[++i];
      if (a == "-res" && i + 1 < argc) {
        Res.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { Res.push_back(std::atof(t.c_str())); t.clear(); } else t += c; }
      }
      if (a == "-ds" && i + 1 < argc) {
        Ds.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { Ds.push_back(Index(std::atol(t.c_str()))); t.clear(); } else t += c; }
      }
    }
    std::printf("Square cylinder in a channel -- Breuer et al. (2000)   %s   %s\n",
                L::name, ExecSpace::name());
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    //--------------------------------------------------------------------------
    // SELF-TEST. Periodic channel, body force G, bounce-back walls. In steady
    // state the walls must carry exactly G * N_fluid; this is a statement about
    // momentum conservation and holds whatever the profile or the viscosity is.
    //--------------------------------------------------------------------------
    {
      const Index H = 32, nx = 8, ny = H + 2;
      const Real G = Real(1e-6), nu = Real(0.1);
      Domain d(nx, ny, 1, true, false, false);
      CollF coll;
      coll.omega = CollF::omega_from_viscosity(nu);
      coll.forcing = Guo{G, Real(0), Real(0)};
      FluidSolver<L, EsotericPull<L>, CollF> s(d, coll);
      s.set_geometry([&](Index x, Index y, Index) -> CellType {
        (void)x; return (y == 0 || y == ny - 1) ? Solid : Fluid;
      });
      s.initialize(Real(1));
      std::vector<Link> links;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 0; x < nx; ++x)
          for (int i = 0; i < L::Q; ++i) {
            const Index qy = y + cvel<L>(i, 1);
            if (qy != 0 && qy != ny - 1) continue;
            links.push_back(Link{d.id(x, y, 0), i});
          }
      const std::size_t nfluid = std::size_t(nx) * std::size_t(H);
      for (std::size_t t = 0; t < 40000; ++t) s.step();
      double Fx = 0, Fy = 0;
      {
        // Scoped so the gathered View dies before Kokkos::finalize -- an early
        // return past a live View is a deallocation after finalize, which
        // Kokkos reports at exit and which is easy to leave in.
        auto g = s.gather_populations();
        auto hg = Kokkos::create_mirror_view_and_copy(HostSpace{}, g);
        for (const Link& lk : links) {
          const int io = opp_of(lk.i);
          const double f = double(unshift<CollF>(hg(lk.n, io), io));
          Fx += 2.0 * cvel<L>(lk.i, 0) * f;
          Fy += 2.0 * cvel<L>(lk.i, 1) * f;
        }
      }
      const double want = double(G) * double(nfluid);
      // SIGN: the sum is the force ON THE SOLID, and it points WITH the flow --
      // the body force pushes the fluid along +x and the fluid drags the wall
      // along +x. So the drag on the cylinder below is +F_x directly, and this
      // test compares +F_x with G * N_fluid. Measured equal to nine figures.
      const double rel = std::abs(Fx - want) / want;
      std::printf("  SELF-TEST  momentum exchange against a known force\n");
      std::printf("    periodic channel H=%d, G=%.1e, %zu fluid nodes\n",
                  int(H), double(G), nfluid);
      std::printf("    wall force   %-14.8e   expected %-14.8e\n", Fx, want);
      std::printf("    transverse   %-14.8e   (must vanish by symmetry)\n", Fy);
      std::printf("    relative error %.3e   %s\n\n", rel, rel < 1e-6 ? "PASS" : "FAIL");
      if (!(rel < 1e-6)) status = 1;
    }
    // AFTER the block, so the solver and its Views are destroyed first. An
    // early return from inside it finalises Kokkos while they are still alive.
    if (selftest_only) { Kokkos::finalize(); return status; }

    //--------------------------------------------------------------------------
    // The paper's case.
    //--------------------------------------------------------------------------
    std::printf("  %4s %6s %7s %9s %9s %10s %12s %9s %10s\n",
                "D", "Re", "nu", "tau", "steps", "C_d", "dC_d/C_d", "Ma", "dmass");
    std::printf("  %s\n", std::string(88, '-').c_str());
    Opts o; o.umax = umax; o.seed = seed; o.ctol = ctol; o.maxsteps = maxsteps;
    o.probe = probe; o.transient = transient; o.avg = avg; o.vtkevery = vtkevery;
    o.vtkdir = vtkdir; o.vtkfrom = vtkfrom; o.outlet = outlet;
    for (const Index D : Ds) {
      for (const double Re : Res) {
        if (op == "cm") run_one<CM>(o, D, Re, status);
        else            run_one<Coll>(o, D, Re, status);
      }
      std::printf("\n");
    }
  }
  Kokkos::finalize();
  return status;
}
