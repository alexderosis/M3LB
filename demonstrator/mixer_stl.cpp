//==============================================================================
//  NIANZHI'S MIXER FROM ITS STL -- the OpenLB case (mixer3d.cpp), run in M3LB.
//
//  Nianzhi's own driver solves the steady flow (D3Q19 BGK), freezes it, and
//  solves a passive concentration in it (D3Q7 BGK), in a square duct whose
//  FLUID volume is an STL built from unit cubes with the obstacles as holes. It
//  reports a mixing index at six stations and a pressure drop. This runs the
//  same case with M3LB's evaluator (src/topopt/Mixer.hpp) in SHARP mode -- the
//  obstacles as real halfway bounce-back walls, as in OpenLB -- and reports
//  the same numbers in the same units.
//
//    python3 tools/stl_voxels.py mixer1.stl --out geom/mixer1
//    ./build/demonstrator/mixer_stl -geom geom/mixer1.npy -N 20 --kokkos-num-threads=4
//
//  WHAT IS THE SAME. The geometry, exactly (tools/stl_voxels.py checks the
//  voxel count against the STL's volume), the physical parameters and their
//  defaults (param:: in mixer3d.cpp), the fully developed duct profile at the
//  inlet (the exact series, topopt/DuctFlow.hpp), p = const at the outlet, a
//  sharp Heaviside split at the inlet, zero-flux walls, and the diagnostics:
//  M = 1 - sigma / sqrt(cbar (1 - cbar)) on node planes at the same distances
//  from the inlet, weighted by u_x over the points with u_x > 0 and by area
//  over every fluid point, and the area-averaged pressure drop between planes
//  20 um and 880 um from the inlet, against the empty duct's exact value.
//
//  WHAT IS NOT, AND WHY.
//    * The operators are M3LB's standing defaults: D3Q27 central moments for
//      the flow, D3Q7 ScalarTRT at Lambda = 1/4 for the concentration
//      (ScalarTRT's banner: at omega near 2 the regularised scalar is unstable
//      in this mixer and BGK rings).
//    * ONE timestep for both fields. OpenLB rescales the velocity onto a
//      second, larger AD timestep; the steady answer depends on Pe and the
//      geometry, not on dt, but the coupled-lattice trap in CLAUDE.md (two
//      lattices at two timesteps do not share a velocity unit) is avoided by
//      not having one. tau defaults to 0.875, which fixes u_lat; OpenLB's 0.933
//      is BGK's "magic" sqrt(3/16) + 1/2 for halfway bounce-back, which does
//      not carry over to a central-moment operator.
//    * The obstacles sit HALF A CELL UPSTREAM of the STL. The inlet is a node
//      plane at x = 0 and a voxel is a whole number of cells long, so every
//      x-face of the STL falls ON a node plane, where a halfway wall cannot sit.
//      Lattice cell i of the design box takes voxel floor(i / r), r cells per
//      voxel, which puts each x-face half a cell upstream: 2.5 um at N = 20.
//      The sizes are exact; y and z are exact (the duct walls are halfway).
//    * c = 1 on the UPPER half of the inlet (y > W/2) here and on the lower
//      half in mixer3d.cpp. The problem is linear and c = 1 solves it, so the
//      swapped field is exactly 1 - c: M is unchanged, cbar becomes 1 - cbar.
//    * Node planes, not a 200 x 200 interpolated sample. On a node plane the
//      area weighting is exact; the interpolation only resamples it.
//    * The uniform-preserving correction is on (the evaluator's default). It
//      undoes the scalar's response to a non-solenoidal discrete velocity; its
//      range is printed, and at this Mach number it should be small.
//
//  CONVERGENCE. The flow uses the evaluator's whole-field criterion on L^2/nu.
//  The concentration stops when M at every station has moved by less than
//  -tolM over one residence time (the duct length at the mean speed), after at
//  least -minres of them; the whole-field change over the same time is printed
//  beside it, because a dead pocket the stations do not see relaxes by
//  diffusion alone. A run that does not meet it prints NOT CONVERGED and
//  exits 1. This is a demonstrator: there is no exact answer to fail against.
//
//  THE EMPTY DUCT HAS ONE, AND IT SAYS HOW FAR TO TRUST M. With the interface
//  thin against W, c is an erf of width 2 sqrt(D x / u_i(z)) about y = W/2,
//  u_i the exact duct velocity on that plane, so Phi = 1/4 - sqrt(2/pi)
//  sqrt(D x) int sqrt(u_i) dz / Q. At Re_W = 2, Pe_W = 2000 that gives M_flux =
//  0.0297 / 0.0422 / 0.0603 / 0.0743 / 0.0864 / 0.0908 at the six stations.
//  Measured (2026-10-23, an all-fluid voxel file): N = 20 is 15 / 7.1 / 3.1 /
//  1.8 / 1.3 / 1.2 % low, N = 10 is 43 / 28 / 16 / 10 / 7.9 / 7.2 % low -- about
//  second order at the outlet, and NOT resolved at the first station, where
//  the interface is thinner than a cell. dp/dp_ref there is 1.008 at N = 20
//  (1.032 at N = 10): Nianzhi's planes are four cells from the boundaries.
//  On mixer1.stl, N = 10 is unusable (Mixer.hpp, sharp mode). N = 20 and 30
//  both converge: M_flux = 0.0246 / 0.0409 / 0.0642 / 0.0779 / 0.0908 / 0.0955
//  and 0.0276 / 0.0431 / 0.0654 / 0.0795 / 0.0920 / 0.0967, i.e. N = 20 sits
//  11 / 5.3 / 1.9 / 2.0 / 1.3 / 1.2 % below N = 30; dp = 17.41 and 17.33 Pa
//  (dp/dp_ref 2.846, 2.832); c in [-0.66, 1.04] and [-0.12, 1.01]. Two points
//  give no order, so no extrapolated value is quoted. Against the empty duct
//  the insert raises M at 880 um by about 6.5% (0.0967 vs the reference's
//  0.0908) for 2.8 times the pressure drop.
//==============================================================================
#include "Campaign.hpp"
#include "io/HeightField.hpp"
#include "topopt/Mixer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace lbm;
using namespace lbm::topopt;

namespace {

struct Station {
  double s;                     // distance from the inlet plane [m]
  Index x;                      // node plane
  double cbar_f = NAN, M_f = NAN, cbar_a = NAN, M_a = NAN;
  int n_f = 0, n_a = 0;
};

// M on one node plane, mixer3d.cpp's two weightings.
void measure_station(Station& st, const MixerEvaluator& ev, const std::vector<double>& c,
                     const std::vector<double>& ux, const std::vector<char>& is_fluid) {
  const MixerGeom& g = ev.g;
  double sw = 0, swc = 0, sa = 0, sac = 0;
  int nf = 0, na = 0;
  for (Index z = g.z0; z <= g.z1; ++z)
    for (Index y = g.y0; y <= g.y1; ++y) {
      const Index n = ev.dom.id(st.x, y, z);
      if (!is_fluid[std::size_t(n)]) continue;
      sa += 1.0; sac += c[n]; ++na;
      if (ux[n] > 0.0) { sw += ux[n]; swc += ux[n] * c[n]; ++nf; }
    }
  st.n_f = nf; st.n_a = na;
  st.cbar_f = swc / sw; st.cbar_a = sac / sa;
  double vf = 0, va = 0;
  for (Index z = g.z0; z <= g.z1; ++z)
    for (Index y = g.y0; y <= g.y1; ++y) {
      const Index n = ev.dom.id(st.x, y, z);
      if (!is_fluid[std::size_t(n)]) continue;
      va += (c[n] - st.cbar_a) * (c[n] - st.cbar_a);
      if (ux[n] > 0.0) vf += ux[n] * (c[n] - st.cbar_f) * (c[n] - st.cbar_f);
    }
  st.M_f = 1.0 - std::sqrt(vf / sw / (st.cbar_f * (1.0 - st.cbar_f)));
  st.M_a = 1.0 - std::sqrt(va / sa / (st.cbar_a * (1.0 - st.cbar_a)));
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    // mixer3d.cpp's param:: defaults, in SI.
    std::string geom, out;
    double voxel = 1e-5, charL = 100e-6, Uphys = 0.01, nu_phys = 1e-6, rho_phys = 1000.0,
           D_phys = 1e-9, tau = 0.875, s0 = 1e-9, tol = 1e-6, tolM = 1e-5, maxres = 40.0,
           minres = 3.0, p_in = 20e-6, p_out = 880e-6, halt = 1.0;
    int N = 20, verbose = 0;
    bool nofix = false, bad_arg = false;
    std::vector<double> st_s = {100e-6, 200e-6, 400e-6, 600e-6, 800e-6, 880e-6};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> const char* {
        if (i + 1 < argc) return argv[++i];
        std::fprintf(stderr, "mixer_stl: option '%s' needs a value\n", a.c_str());
        bad_arg = true;
        return "0";
      };
      if      (a == "-geom")   geom = next();
      else if (a == "-voxel")  voxel = std::atof(next());
      else if (a == "-charL")  charL = std::atof(next());
      else if (a == "-N")      N = std::atoi(next());
      else if (a == "-U")      Uphys = std::atof(next());
      else if (a == "-nu")     nu_phys = std::atof(next());
      else if (a == "-rho")    rho_phys = std::atof(next());
      else if (a == "-D")      D_phys = std::atof(next());
      else if (a == "-tau")    tau = std::atof(next());
      else if (a == "-s0")     s0 = std::atof(next());
      else if (a == "-tol")    tol = std::atof(next());
      else if (a == "-tolM")   tolM = std::atof(next());
      else if (a == "-maxres") maxres = std::atof(next());
      else if (a == "-minres") minres = std::atof(next());
      else if (a == "-halt")   halt = std::atof(next());   // diagnosis: the halt band
      else if (a == "-nofix")  nofix = true;
      else if (a == "-v")      verbose = std::atoi(next());
      else if (a == "-out")    out = next();
      else {
        std::fprintf(stderr, "mixer_stl: unknown option '%s'\n  options: -geom FILE.npy -voxel M "
                     "-charL M -N n -U M/S -nu M2/S -rho KG/M3 -D M2/S -tau T -s0 CELLS -tol E "
                     "-tolM E -maxres R -minres R -halt BAND -nofix -v K -out PREFIX\n", a.c_str());
        bad_arg = true;
      }
    }
    if (geom.empty() && !bad_arg) {
      std::fprintf(stderr, "mixer_stl: -geom FILE.npy is required (tools/stl_voxels.py)\n");
      bad_arg = true;
    }
    if (bad_arg) rc = 2;
    if (rc == 0) do {
      // The voxel array, (NX, NY, NZ), x slowest, 1 = fluid.
      std::vector<float> vox;
      std::size_t NX = 0, NY = 0, NZ = 0;
      try {
        std::ifstream f(geom, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + geom);
        const npy::Header h = npy::read_header(f, geom);
        if (h.shape.size() != 3) throw std::runtime_error(geom + ": expected a 3-D array");
        NX = h.shape[0]; NY = h.shape[1]; NZ = h.shape[2];
        vox = npy::read_f32_shaped(geom, h.shape);
      } catch (const std::exception& e) {
        std::printf("mixer_stl: %s\n", e.what());
        rc = 2;
        break;
      }
      auto V = [&](std::size_t i, std::size_t j, std::size_t k) {
        return vox[(i * NY + j) * NZ + k] >= 0.5f;
      };
      if (NY != NZ) {
        std::printf("mixer_stl: the duct is %zu x %zu voxels; the evaluator's is square\n", NY, NZ);
        rc = 2;
        break;
      }
      // The design box: the solids' x extent, the full width, floor to their top.
      std::size_t xs0 = NX, xs1 = 0, zs1 = 0, nsolid = 0;
      for (std::size_t i = 0; i < NX; ++i)
        for (std::size_t j = 0; j < NY; ++j)
          for (std::size_t k = 0; k < NZ; ++k)
            if (!V(i, j, k)) {
              ++nsolid;
              xs0 = std::min(xs0, i); xs1 = std::max(xs1, i + 1); zs1 = std::max(zs1, k + 1);
            }
      if (nsolid == 0) { xs0 = NX / 2; xs1 = NX / 2; zs1 = 0; }

      // Lattice: a voxel must be a whole number of cells, or the obstacles
      // change shape with N and a grid study measures that.
      const double dx = charL / double(N);
      const double r_d = voxel / dx;
      const Index r = Index(std::lround(r_d));
      if (r < 1 || std::fabs(r_d - double(r)) > 1e-9 * r_d) {
        std::printf("mixer_stl: a voxel is %.6g cells at N = %d; choose N so that it is a whole "
                    "number (voxel %.3g m, charL %.3g m)\n", r_d, N, voxel, charL);
        rc = 2;
        break;
      }
      const double W = double(NY) * voxel;              // duct side [m]
      MixerSpec sp;
      sp.L = Index(NY) * r;
      sp.len_in  = double(xs0) / double(NY);
      sp.len_des = double(xs1 - xs0) / double(NY);
      sp.len_out = double(NX - xs1) / double(NY);
      sp.H  = double(zs1) / double(NY);
      sp.Re = Uphys * W / nu_phys;
      sp.Pe = Uphys * W / D_phys;
      const double nu_lat = (tau - 0.5) / 3.0;
      sp.U  = nu_lat * sp.Re / double(sp.L);
      sp.s0 = s0;
      sp.sharp = true;
      sp.uniform_fix = !nofix;
      const double dt = sp.U * dx / Uphys;
      const double vconv = dx / dt;                     // m/s per lattice velocity

      MixerEvaluator ev(sp);
      const MixerGeom& g = ev.g;
      if (g.nx - 1 != Index(NX) * r || g.x_d1 - g.x_d0 != Index(xs1 - xs0) * r ||
          g.nzd != Index(zs1) * r) {
        std::printf("mixer_stl: the evaluator's box (%lld, %lld, %lld) is not the STL's "
                    "refined by %lld\n", (long long)(g.nx - 1), (long long)(g.x_d1 - g.x_d0),
                    (long long)g.nzd, (long long)r);
        rc = 2;
        break;
      }
      std::vector<double> gam(std::size_t(g.n_design()));
      {
        const Index nyd = g.y1 - g.y0 + 1;
        std::size_t m = 0;
        for (Index i = 0; i < g.x_d1 - g.x_d0; ++i)
          for (Index j = 0; j < nyd; ++j)
            for (Index k = 0; k < g.nzd; ++k, ++m)
              gam[m] = V(xs0 + std::size_t(i / r), std::size_t(j / r), std::size_t(k / r)) ? 1.0 : 0.0;
      }
      ev.set_design(gam);

      std::printf("MIXER FROM STL -- Nianzhi's OpenLB case (D3Q27 CM, D3Q7 %s, %s)\n",
                  MixScalarColl::name, sizeof(Real) == 8 ? "FP64" : "FP32");
      std::printf("  geometry      %s: %zu x %zu x %zu voxels of %.3g um, %zu solid; duct side "
                  "W = %.4g um, length %.4g um\n", geom.c_str(), NX, NY, NZ, voxel * 1e6, nsolid,
                  W * 1e6, double(NX) * voxel * 1e6);
      std::printf("                obstacles x in [%.4g, %.4g) um, z < %.4g um: sections %.3g + "
                  "%.3g + %.3g W, H = %.3g W\n", double(xs0) * voxel * 1e6,
                  double(xs1) * voxel * 1e6, double(zs1) * voxel * 1e6, sp.len_in, sp.len_des,
                  sp.len_out, sp.H);
      std::printf("  physics       U = %.4g m/s  nu = %.3g m^2/s  D = %.3g m^2/s  rho = %.4g kg/m^3\n",
                  Uphys, nu_phys, D_phys, rho_phys);
      std::printf("                Re_W = %.4g  Pe_W = %.4g  (on charL = %.3g um: Re = %.4g, Pe = %.4g)"
                  "  Sc = %.4g\n", sp.Re, sp.Pe, charL * 1e6, Uphys * charL / nu_phys,
                  Uphys * charL / D_phys, nu_phys / D_phys);
      std::printf("  lattice       N = %d per charL, dx = %.4g um, %lld cells per voxel; tau = %.4g "
                  "-> dt = %.4g s, u_lat = %.5g\n", N, dx * 1e6, (long long)r, tau, dt, sp.U);
      std::printf("                check: nu = %.4g, D = %.4g m^2/s back from the lattice\n",
                  g.nu * dx * dx / dt, g.D * dx * dx / dt);
      ev.print_spec("  ");
      const bool conn = ev.connected();
      std::printf("  inlet split   c = 1 for y > W/2 (mixer3d.cpp: y < W/2; M is invariant), "
                  "erf width s0 = %.3g cells\n", s0);
      std::printf("  connected     fluid path inlet -> outlet: %s\n",
                  conn ? "yes" : "NO -- the result below means nothing");

      // ---- stage 1: the flow --------------------------------------------------
      const Convergence cf = ev.run_flow(tol, 40.0, 0, verbose);
      std::printf("  flow          %zu steps (t/T = %.3f, T = L^2/nu; %.4g s)  e_hat %.2e%s  "
                  "r_odd %.1e  %.1f s\n", cf.steps, double(cf.steps) / cf.T_relax,
                  double(cf.steps) * dt, cf.e_hat, cf.converged ? "" : "  NOT CONVERGED",
                  cf.r_odd, cf.secs);
      if (!cf.finite) { std::printf("  flow went non-finite\n"); rc = 1; break; }

      // ---- stage 2: the concentration, with station monitoring ---------------
      const auto ux = ev.host_ux();
      std::vector<char> is_fluid(ux.size(), 0);
      for (Index n : ev.fnodes) is_fluid[std::size_t(n)] = 1;
      std::vector<Station> st;
      for (double s : st_s) {
        Station a; a.s = s;
        a.x = std::clamp<Index>(Index(std::lround(s / dx)), 1, g.nx - 2);
        st.push_back(a);
      }
      if (sp.uniform_fix) ev.calibrate_uniform();
      else                ev.scalar.clear_uniform_correction();
      ev.seed_scalar(true);
      const double t_res = double(g.nx - 1) / g.U;      // steps for one pass at the mean speed
      const int per_res = 8;
      const std::size_t probe = std::size_t(std::ceil(t_res / per_res));
      const std::size_t cap = std::size_t(maxres * t_res);
      std::vector<std::vector<double>> hist;            // per probe: M_f, M_a at every station
      std::vector<std::vector<double>> fields;          // the bulk field, one residence ago
      std::FILE* fs = out.empty() ? nullptr : std::fopen((out + "_series.dat").c_str(), "w");
      if (fs) {
        std::fprintf(fs, "# step  t[s]  t/t_res  then M_flux, M_area at stations (um):");
        for (const auto& a : st) std::fprintf(fs, " %.0f", a.s * 1e6);
        std::fprintf(fs, "\n");
      }
      std::size_t steps = 0;
      double dM = NAN, dfield = NAN, cmin = NAN, cmax = NAN;
      bool conv = false, finite = true, bounded = true;
      const auto t0 = std::chrono::steady_clock::now();
      while (steps < cap) {
        for (std::size_t k = 0; k < probe; ++k) ev.scalar.step();
        steps += probe;
        const auto c = ev.host_c();
        std::vector<double> bulk(ev.snodes.size());
        double lo = 1e300, hi = -1e300, den = 0;
        for (std::size_t m = 0; m < bulk.size(); ++m) {
          bulk[m] = c[ev.snodes[m]]; den += bulk[m] * bulk[m];
          lo = std::min(lo, bulk[m]); hi = std::max(hi, bulk[m]);
        }
        cmin = lo; cmax = hi;
        if (!std::isfinite(den)) { finite = false; break; }
        if (lo < -halt || hi > 1.0 + halt) { bounded = false; break; }
        std::vector<double> row;
        for (auto& a : st) { measure_station(a, ev, c, ux, is_fluid); row.push_back(a.M_f); row.push_back(a.M_a); }
        hist.push_back(row);
        fields.push_back(bulk);
        if (fs) {
          std::fprintf(fs, "%zu %.6e %.4f", steps, double(steps) * dt, double(steps) / t_res);
          for (double v : row) std::fprintf(fs, " %.8f", v);
          std::fprintf(fs, "\n");
          std::fflush(fs);
        }
        const bool full = hist.size() > std::size_t(per_res);
        if (full) {
          const auto& old = hist[hist.size() - 1 - per_res];
          dM = 0;
          for (std::size_t q = 0; q < row.size(); ++q) dM = std::max(dM, std::fabs(row[q] - old[q]));
          dfield = MixerEvaluator::rel_change(bulk, fields[fields.size() - 1 - per_res]);
          fields.erase(fields.begin());                 // keep one residence time of fields
        }
        if (verbose && hist.size() % std::size_t(verbose) == 0)
          std::printf("      scalar t/t_res = %6.3f  dM/t_res = %.2e  field change/t_res = %.2e"
                      "  M_flux(outlet) = %.6f  c in [%.4f, %.4f]\n", double(steps) / t_res,
                      dM, dfield, st.back().M_f, lo, hi);
        if (full && double(steps) >= minres * t_res && dM < tolM) { conv = true; break; }
      }
      if (fs) std::fclose(fs);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  concentration %zu steps (%.2f residence times, %.4g s)  dM/t_res %.2e  "
                  "field change/t_res %.2e%s  %.1f s\n", steps, double(steps) / t_res,
                  double(steps) * dt, dM, dfield, conv ? "" : "  NOT CONVERGED", secs);
      if (sp.uniform_fix)
        std::printf("  correction    calibration %s in %zu steps, fix in [%.3e, %.3e]\n",
                    ev.calib.converged ? "converged" : "NOT CONVERGED", ev.calib.steps,
                    ev.calib.fix_min, ev.calib.fix_max);
      if (!finite || !bounded)
        std::printf("  concentration HALTED: %s (c in [%.4f, %.4f])\n",
                    finite ? "c left the halt band" : "non-finite", cmin, cmax);
      const bool valid = conn && cf.converged && conv && finite && bounded &&
                         (!sp.uniform_fix || ev.calib.converged);

      // ---- the numbers mixer3d.cpp prints --------------------------------------
      std::printf("  MIXING%s  (node planes; c in [%.5f, %.5f] over the bulk)\n",
                  valid ? "" : " -- NOT A RESULT: see above", cmin, cmax);
      std::printf("    x from inlet [um]  plane | flux weighted: cbar     M        (pts) | "
                  "area weighted: cbar     M        (pts)\n");
      for (const auto& a : st)
        std::printf("    %8.1f        %5lld |               %.5f  %.6f (%4d) |               "
                    "%.5f  %.6f (%4d)\n", a.s * 1e6, (long long)a.x, a.cbar_f, a.M_f, a.n_f,
                    a.cbar_a, a.M_a, a.n_a);
      {
        const auto rho = ev.host_rho();
        const Index x1 = Index(std::lround(p_in / dx)), x2 = Index(std::lround(p_out / dx));
        double r1, j1, r2, j2;
        ev.plane_means(rho, ux, x1, r1, j1);
        ev.plane_means(rho, ux, x2, r2, j2);
        const double dp_lat = double(cs2<D3Q27, Real>()) * (r1 - r2);
        const double dp = dp_lat * rho_phys * vconv * vconv;
        const double dp_ref = g.K * nu_phys * rho_phys * Uphys * double(x2 - x1) * dx / (W * W);
        std::printf("  PRESSURE%s  planes %.0f and %.0f um (x = %lld, %lld): dp = %.5g Pa   "
                    "empty duct (exact) %.5g Pa   dp/dp_ref = %.4f\n", valid ? "" : " (NOT A RESULT)",
                    double(x1) * dx * 1e6, double(x2) * dx * 1e6, (long long)x1, (long long)x2,
                    dp, dp_ref, dp / dp_ref);
      }
      const MixerResult res = ev.measure();
      std::printf("  EVALUATOR     Phi = %.6e  M = %.6f at x = %lld;  Dp/Dp0 = %.5f (planes %lld, "
                  "%lld)\n", res.Phi, res.M, (long long)g.x_m, res.dp_ratio, (long long)g.x_p1,
                  (long long)g.x_p2);
      std::printf("  LIMITS        max|u| = %.4g (%.2f U = %.4g m/s)  Ma_max = %.4f  density drop "
                  "%.3f%%  cell Pe at max|u| = %.3g\n", res.umax, res.umax / sp.U,
                  res.umax * vconv, res.Ma_max, 100.0 * res.drho, res.umax / g.D);

      if (!out.empty()) {
        // Interior fields, shape (nz, ny, nx), x fastest. c is 0 at a wall node.
        const auto c = ev.host_c(), uy = ev.host_uy(), uz = ev.host_uz();
        std::vector<float> fc, fx, fy, fz;
        for (Index z = 0; z < g.nz; ++z)
          for (Index y = 0; y < g.ny; ++y)
            for (Index x = 0; x < g.nx; ++x) {
              const std::size_t n = std::size_t(ev.dom.id(x, y, z));
              const bool fl = is_fluid[n];
              fc.push_back(fl ? float(c[n]) : NAN);
              fx.push_back(fl ? float(ux[n]) : 0.f);
              fy.push_back(fl ? float(uy[n]) : 0.f);
              fz.push_back(fl ? float(uz[n]) : 0.f);
            }
        npy::write_f32(out + "_c.npy", fc, g.nz, g.ny, g.nx);
        npy::write_f32(out + "_ux.npy", fx, g.nz, g.ny, g.nx);
        npy::write_f32(out + "_uy.npy", fy, g.nz, g.ny, g.nx);
        npy::write_f32(out + "_uz.npy", fz, g.nz, g.ny, g.nx);
        std::printf("  wrote         %s_{c,ux,uy,uz}.npy (nz, ny, nx), %s_series.dat\n",
                    out.c_str(), out.c_str());
      }
      rc = valid ? 0 : 1;
    } while (false);
  }
  Kokkos::finalize();
  return rc;
}
