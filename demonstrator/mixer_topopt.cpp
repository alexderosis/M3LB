//==============================================================================
//  NIANZHI'S MIXER -- evaluate one design.
//
//  The command-line face of src/topopt/Mixer.hpp (P1 of
//  doc/mixer_adjoint_plan.md): read a design, solve the steady flow and the
//  steady concentration, print the mixing index and the pressure drop. P2-P4
//  add the gradient to this same driver; until then it evaluates only.
//
//  THE DESIGN is gamma in [0, 1] per lattice cell of the design box, 1 = fluid:
//
//    -design empty      no obstacle (the reference for Dp/Dp0 and M)
//    -design slab       the whole box solid
//    -design grooves    slanted grooves, the design of the two-grid check in
//                       validation/mixer_forward.cpp
//    -design FILE.npy   float32 ('<f4'), C order, shape EXACTLY (nxd, L, nzd)
//                       with x slowest -- nxd = round(len_des L), nzd = round(H L);
//                       the program prints the shape it expects and refuses any
//                       other, a permutation with the right count included
//
//  This is a demonstrator, not a validation case: the evaluator's accuracy is
//  established by validation/mixer_forward.cpp, and nothing here can fail but
//  a crash, a blocked design or a run that does not converge -- each of which
//  is reported, and the last two set a non-zero exit code.
//
//  Example, the plan's production grid (about 25 minutes on four threads):
//      ./build/demonstrator/mixer_topopt -L 40 -design grooves --kokkos-num-threads=4
//
//  The inlet split's erf width -s0 defaults to L/40 cells, as in the two-grid
//  check; -halt sets the concentration's divergence guard (default 1, i.e.
//  [-1, 2]). A run that does not converge, halts, or is blocked prints its
//  numbers labelled NOT A RESULT and exits 1.
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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    MixerSpec sp;
    std::string design = "grooves", out;
    // halt: the evaluator's own divergence guard, [-1, 2]. It was 0.05 here, which
    // stopped the documented example at its first probe on a transient.
    double tol = 1e-6, halt = 1.0;
    int verbose = 0;
    bool s0_given = false, bad_arg = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> const char* {
        if (i + 1 < argc) return argv[++i];
        std::fprintf(stderr, "mixer_topopt: option '%s' needs a value\n", a.c_str());
        bad_arg = true;
        return "0";
      };
      if      (a == "-L")      sp.L = std::atoi(next());
      else if (a == "-H")      sp.H = std::atof(next());
      else if (a == "-re")     sp.Re = std::atof(next());
      else if (a == "-pe")     sp.Pe = std::atof(next());
      else if (a == "-u")      sp.U = std::atof(next());
      else if (a == "-da")     sp.Da = std::atof(next());
      else if (a == "-q")      sp.q = std::atof(next());
      else if (a == "-s0")   { sp.s0 = std::atof(next()); s0_given = true; }
      else if (a == "-design") design = next();
      else if (a == "-tol")    tol = std::atof(next());
      else if (a == "-v")      verbose = std::atoi(next());
      else if (a == "-out")    out = next();
      else if (a == "-halt")   halt = std::atof(next());   // diagnosis: the halt band
      else if (a == "-cuniform") sp.c_uniform = std::atof(next());   // exact answer: c = this
      else if (a == "-nofix")  sp.uniform_fix = false;   // no uniform-preserving correction
      else {
        std::fprintf(stderr, "mixer_topopt: unknown option '%s'\n  options: -L -H -re -pe -u "
                     "-da -q -s0 -design empty|slab|grooves|FILE.npy -tol -v -out PREFIX -halt BAND "
                     "-cuniform C -nofix\n", a.c_str());
        bad_arg = true;
      }
    }
    // The inlet split's width scales with the grid unless given, as in the
    // two-grid check -- a grid study through this driver must not change it.
    if (!s0_given) sp.s0 = double(sp.L) / 40.0;
    if (bad_arg) rc = 2;
    if (rc == 0) do {
      MixerEvaluator ev(sp);
      const MixerGeom& g = ev.g;
      std::printf("MIXER -- forward evaluation (D3Q27 CM + DarcyGuo, D3Q7 %s, %s)\n", MixScalarColl::name,
                  sizeof(Real) == 8 ? "FP64" : "FP32: tolerances below 1e-6 are not reachable");
      ev.print_spec();
      const double pi = 3.14159265358979323846;
      if (design == "empty") {
        ev.set_design_fn([](double, double, double) { return 1.0; });
      } else if (design == "slab") {
        ev.set_design_fn([](double, double, double) { return 0.0; });
      } else if (design == "grooves") {
        ev.set_design_fn([=](double xi, double eta, double) {
          return 0.5 * (1.0 + std::tanh(-std::sin(2.0 * pi * (xi + eta) / 0.5) / 0.3));
        });
      } else {
        const std::vector<std::size_t> shape = {std::size_t(g.x_d1 - g.x_d0),
                                                std::size_t(g.y1 - g.y0 + 1), std::size_t(g.nzd)};
        std::printf("  design file   %s, expecting float32 shape (%zu, %zu, %zu)\n",
                    design.c_str(), shape[0], shape[1], shape[2]);
        std::vector<float> f;
        try {
          f = npy::read_f32_shaped(design, shape);    // the exact shape, not just the count
        } catch (const std::exception& e) {
          // Leave the block, so the evaluator's Views are freed BEFORE
          // Kokkos::finalize -- returning from here aborted the process.
          std::printf("  %s\n", e.what());
          rc = 2;
          break;
        }
        ev.set_design(std::vector<double>(f.begin(), f.end()));
      }
      const bool conn = ev.connected();
      std::printf("  design        '%s'   fluid path inlet -> outlet (gamma >= 0.5): %s\n",
                  design.c_str(), conn ? "yes" : "NO -- the result below means nothing");
      const Convergence cf = ev.run_flow(tol, 40.0, 0, verbose);
      std::printf("  flow          %zu steps (t/T = %.3f, T = L^2/nu)  e_hat %.2e%s  %.1f s\n",
                  cf.steps, double(cf.steps) / cf.T_relax, cf.e_hat,
                  cf.converged ? "" : "  NOT CONVERGED", cf.secs);
      const Convergence cs = ev.run_scalar(tol, 40.0, 0, verbose, nullptr, true, halt);
      std::printf("  concentration %zu steps (t/T = %.3f)  e_hat %.2e  e_phi %.2e%s  %.1f s\n",
                  cs.steps, double(cs.steps) / cs.T_relax, cs.e_hat, cs.e_phi,
                  cs.converged ? "" : "  NOT CONVERGED", cs.secs);
      if (sp.uniform_fix)
        std::printf("  correction    uniform-preserving fix: calibration %s in %zu steps (last "
                    "change %.1e), fix in [%.4f, %.4f], %.1f s\n",
                    ev.calib.converged ? "converged" : "NOT CONVERGED", ev.calib.steps,
                    ev.calib.change, ev.calib.fix_min, ev.calib.fix_max, ev.calib.secs);
      else
        std::printf("  correction    OFF (-nofix): a uniform field is not preserved in drag regions\n");
      if (!cs.bounded)
        std::printf("  concentration HALTED: c left [%.3g, %.3g] (range [%.4f, %.4f]) -- a "
                    "divergence or a broken boundary\n", -halt, 1.0 + halt, cs.cmin, cs.cmax);
      const bool valid = conn && cf.converged && cs.converged && cs.bounded;
      const MixerResult r = ev.measure();
      std::printf("  RESULT%s  Phi = %.6e   M = %.6f   cbar = %.6f\n",
                  valid ? "       " : " (NOT A RESULT: see above)", r.Phi, r.M, r.cbar);
      std::printf("                Dp = %.6e   Dp/Dp0 = %.5f   (Dp0: exact empty duct, same flux)\n",
                  r.dp, r.dp_ratio);
      std::printf("  LIMITS        max|u| = %.5f (%.2f U)  Ma_max = %.4f  density drop %.2f%%  "
                  "c in [%.5f, %.5f]  flow period-2 r_odd = %.2e\n", r.umax, r.umax / sp.U,
                  r.Ma_max, 100.0 * r.drho, r.cmin, r.cmax, cf.r_odd);
      {
        // Where the largest c sits: inside the drag region, or in open fluid.
        const auto c = ev.host_c(), ux = ev.host_ux(), uy = ev.host_uy(), uz = ev.host_uz();
        auto hA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ev.A);
        Index nm = ev.snodes.empty() ? 0 : ev.snodes[0];
        for (Index n : ev.snodes) if (c[n] > c[nm]) nm = n;
        Index px, py, pz; ev.dom.coords(nm, px, py, pz);
        std::printf("                max c at (%lld, %lld, %lld): A = %.4g (A_max %.4g)  |u| = %.3e U\n",
                    (long long)(px - ev.dom.hx), (long long)(py - ev.dom.hy),
                    (long long)(pz - ev.dom.hz), double(hA(nm)), g.A_max,
                    std::sqrt(ux[nm] * ux[nm] + uy[nm] * uy[nm] + uz[nm] * uz[nm]) / sp.U);
      }
      if (!out.empty()) {
        // The mixing plane, for plotting: c and u_x as (1, L, L) float32.
        const auto c = ev.host_c(), ux = ev.host_ux();
        std::vector<float> pc, pu;
        for (Index y = g.y0; y <= g.y1; ++y)
          for (Index z = g.z0; z <= g.z1; ++z) {
            const Index n = ev.dom.id(g.x_m, y, z);
            pc.push_back(float(c[std::size_t(n)])); pu.push_back(float(ux[std::size_t(n)]));
          }
        const Index ny = g.y1 - g.y0 + 1, nz = g.z1 - g.z0 + 1;
        npy::write_f32(out + "_plane_c.npy", pc, 1, ny, nz);
        npy::write_f32(out + "_plane_ux.npy", pu, 1, ny, nz);
        std::printf("  wrote         %s_plane_c.npy, %s_plane_ux.npy (y slowest)\n",
                    out.c_str(), out.c_str());
      }
      rc = valid ? 0 : 1;
    } while (false);
  }
  Kokkos::finalize();
  return rc;
}
