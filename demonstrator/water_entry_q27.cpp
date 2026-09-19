//==============================================================================
//  A tilted square falling into water, on D3Q27 + D3Q7 at nz = 1.
//
//  THIS IS demonstrator/water_entry.cpp EXPRESSED ON THE 3-D LATTICE PAIR, and
//  the only reason to have both is to see whether the answer survives the
//  translation. The physics, the scaling and the output columns are that case's;
//  read its banner for the impact sequence and for what a square (as against a
//  wedge) can and cannot be checked against.
//
//  WHY THIS FILE IS AWKWARD, STATED FIRST BECAUSE IT IS THE POINT.
//
//  A tilted square is a planar problem: three degrees of freedom, x, y and one
//  angle about z. The shape that expresses exactly that is `Rect`, and `Rect`
//  CANNOT BE USED HERE -- PenalisedBody carries
//
//      static_assert(L::D == 2 || Shape::three_d, ...)
//
//  which refuses a prism on a 3-D lattice, correctly, because on D3Q27 a Rect
//  would model an infinite prism in z with a solve carrying one angle. Changing
//  nz does not help: the assertion is on the LATTICE and the SHAPE, both known
//  at compile time, and nz is a runtime argument to Domain. Compile-checked at
//  nz = 3, not assumed.
//
//  So the only 3-D shape that can tilt is `Box`, which declares six_dof = true
//  and takes the full 6x6 with a quaternion pose. That solve has no per-axis
//  control: `free_rotation` is ONE BOOL covering all three axes,
//
//      if (free_rotation) { wx += dW[0];  wy += dW[1];  wz += dW[2]; }
//
//  so a Box here is free to rotate about x and y as well -- out of a domain one
//  cell deep and periodic in z, where such a rotation is not a motion but a
//  contradiction: a body periodic in z cannot tilt out of z.
//
//  THE CONSTRAINT, AND IT IS A CONSTRAINT RATHER THAN A MODEL. Between refresh6
//  and advance6 this driver sets wx = wy = 0 by hand. advance6 integrates the
//  quaternion from (wx, wy, wz), so zeroing two of them leaves a pose that only
//  ever turns about z, which is what a planar problem means. It is a hard
//  constraint applied after the solve, not a term in it: the 6x6 still computes
//  dW[0] and dW[1] from the measured torque and this file throws them away.
//
//  WHICH IS WHY THE OUT-OF-PLANE TORQUES ARE PRINTED. tx and ty are what the
//  constraint discards, and they are the diagnostic that says whether it is
//  benign. Against tz they should be small and of no particular sign -- they are
//  discretisation noise about a symmetry the lattice does not have exactly, the
//  same class of thing cube_entry measures when it releases a cube corner-down.
//  If they ever come out comparable to tz, this setup is not describing a planar
//  problem and the numbers should not be read as one.
//
//  AND THE CONSTRAINT TURNS OUT TO BE A NO-OP, WHICH IS NOT WHAT I EXPECTED.
//  This file was written expecting the Box path to be a poor imitation of Rect's
//  3x3. It is not, and -free is the flag that shows it: skipping the two zeroing
//  lines changes nothing to the printed digits. The reason is symmetry rather
//  than luck. With one cell in z and the body centred on it, every cell's z
//  offset from the body centre is zero, so r x F has no x or y component at all
//  and the 6x6 is handed tau_x = tau_y = 0. Measured over a full entry:
//
//      worst |tau_x,y| = 1.4e-16   against   worst |tau_z| = 1.9
//
//  a ratio of 7e-17, which is round-off and not smallness. The constraint is
//  kept as a GUARD rather than a correction -- it states the intent and would
//  catch a case where that symmetry stopped holding -- but it is not doing any
//  work here, and saying so is more useful than leaving the caution standing.
//
//  THE TWO LATTICE PAIRS AGREE, which is the only thing that makes this file
//  worth having. Against demonstrator/water_entry at matched parameters
//  (L = 16, ratio 100, Re 200, U = 0.04, xi = 4, M = 0.02, sigma = 1e-3,
//  rho_b = 0.9, drop 1.5 L, tilt 20 deg) -- so D2Q9 fluid + D2Q9 phase with Rect
//  and the validated 3x3, against D3Q27 fluid + D3Q7 phase with Box and the
//  constrained 6x6:
//
//      t U/L    y_c/L  (2-D / 3-D)     V/U  (2-D / 3-D)     tilt (2-D / 3-D)
//      1.000    1.8439 / 1.8439        -0.3109 / -0.3109    20.02 / 20.02
//      2.000    1.3801 / 1.3801        -0.6166 / -0.6165    20.01 / 20.01
//      3.000    0.6624 / 0.6628        -0.6157 / -0.6145    18.83 / 18.83
//      4.000    0.2780 / 0.2771        -0.3134 / -0.3154    18.34 / 18.39
//
//  Four significant figures through free fall and three through impact, across
//  two fluid lattices, two phase lattices with DIFFERENT cs2 (1/3 against 1/4),
//  two body shapes and two rigid-body solves. That is a real cross-check on all
//  of them, and it is the reason to keep both files rather than only the 2-D one.
//
//  THE SEQUENCE, at rho_b = 0.9 x water. Free fall to t U/L ~ 3, where F_y jumps
//  by a factor of 40 in one frame; deceleration; the deepest point at t ~ 7 with
//  y_c/L = -0.20; then the velocity reverses and it floats back, because 0.9 is
//  less than 1. The tilt falls 20 -> 18.3 on impact as the low corner is pushed
//  up, recovers slightly while submerged, and then falls again to 13 as it
//  rises. None of that has a closed form -- see water_entry.cpp on why a square
//  is not a wedge -- so it is a shape to read, not a number to quote.
//
//  Even so: validation/floating_body and demonstrator/water_entry use D2Q9 with
//  Rect and the validated 3x3, which remains the right tool for a planar problem.
//  Where the two disagree, believe the 2-D one.
//
//  THE LATTICE PAIR. D3Q27 for the fluid, which is forced -- every multiphase
//  operator asserts supports_navier_stokes and D3Q7 does not have it. D3Q7 for
//  the phase field, which is a free choice and the cheap one: 7 populations
//  against 27. Note it buys less than that ratio suggests, because
//  GradientLatticeOf<D3Q7> is D3Q27, so the gradient and Laplacian still cost a
//  27-neighbour gather. It also forces BGK on the phase field, since the
//  central-moment form needs a product basis and D3Q7 is not a product lattice.
//  The solver's collision argument is OMITTED below so that
//  DefaultPhaseCollision<D3Q7> resolves it, and the resolved name is printed.
//
//  MOBILITY ON D3Q7 IS NOT THE SAME NUMBER AS ON D3Q27. cs2 is 1/4 here and 1/3
//  there, so omega_from_mobility differs by 4/3 for the same M. That is exactly
//  the trap CLAUDE.md records against writing tau = 3 nu + 1/2 on D3Q7; using
//  the operator's own converter is what avoids it, and this file does.
//==============================================================================
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/PhaseFieldBGK.hpp"
#include "core/Types.hpp"
#include "grid/Domain.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/PenalisedBody.hpp"
#include "solver/PhaseFieldSolver.hpp"
#include "solver/ViscousInterfaceForce.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

// FieldDump's format, which demonstrator/render_rt reads: int32 nx, int32 ny,
// then nx*ny float32 row major. Identical to water_entry.cpp's, deliberately --
// the two cases share a renderer, so the frames have to be interchangeable.
template <class Get>
static void dump_field(const std::string& path, Index nx, Index ny, Get get) {
  std::vector<float> v(std::size_t(nx) * std::size_t(ny));
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x)
      v[std::size_t(y) * std::size_t(nx) + std::size_t(x)] = float(get(x, y));
  std::ofstream o(path, std::ios::binary);
  if (!o) { std::printf("  cannot write %s\n", path.c_str()); return; }
  const std::int32_t a = int(nx), b = int(ny);
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

using FL = D3Q27;
using PL = D3Q7;
using FColl    = MultiphaseCentralMoments<FL>;
using PColl    = DefaultPhaseCollision<PL>;          // PhaseFieldBGK<D3Q7>
using PhaseSlv = PhaseFieldSolver<PL, EsotericPull<PL>>;   // collision defaulted
using FluidSlv = FluidSolver<FL, EsotericPull<FL>, FColl>;
using Body     = PenalisedBody<FL, Box>;

struct Params {
  Index W = 24;
  double ratio = 100.0, Re = 200.0, U = 0.04, iw = 4.0, M = 0.02, sigma = 1e-3;
  double body_rho = 0.9, drop = 1.5, tmax = 6.0, theta = 20.0;
  int nframes = 12;
  bool unconstrained = false;    // -free: skip the planar constraint entirely
  std::string dump;              // -dump <dir>: frames for demonstrator/render_rt
};

// Box tracks its pose as a quaternion and deliberately does not track theta.
// For a pose that only ever turns about z, q = (cos(t/2), 0, 0, sin(t/2)), so
// the angle comes back exactly. Reported in degrees.
static double tilt_deg(const Quat& q) {
  return 2.0 * std::atan2(double(q.z), double(q.w)) * 180.0 / M_PI;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Params P;
    for (int i = 1; i < argc; ++i) {
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (!std::strcmp(argv[i], "-l"))     { if (i+1<argc) P.W = Index(std::atoi(argv[++i])); }
      else if (!std::strcmp(argv[i], "-ratio")) num(P.ratio);
      else if (!std::strcmp(argv[i], "-re"))    num(P.Re);
      else if (!std::strcmp(argv[i], "-u"))     num(P.U);
      else if (!std::strcmp(argv[i], "-iw"))    num(P.iw);
      else if (!std::strcmp(argv[i], "-m"))     num(P.M);
      else if (!std::strcmp(argv[i], "-sigma")) num(P.sigma);
      else if (!std::strcmp(argv[i], "-rhob"))  num(P.body_rho);
      else if (!std::strcmp(argv[i], "-drop"))  num(P.drop);
      else if (!std::strcmp(argv[i], "-tmax"))  num(P.tmax);
      else if (!std::strcmp(argv[i], "-theta")) num(P.theta);
      else if (!std::strcmp(argv[i], "-nframes")) { if (i+1<argc) P.nframes = std::atoi(argv[++i]); }
      else if (!std::strcmp(argv[i], "-free"))  { P.unconstrained = true; }
      else if (!std::strcmp(argv[i], "-dump"))  { if (i+1<argc) P.dump = argv[++i]; }
    }

    const Index L = P.W;
    const Index nx = 6 * L, ny = 7 * L, nz = 1;
    const double y_water = 4.0 * double(L);
    const double rho_l = 1.0, rho_h = P.ratio;
    const double U = P.U;
    const double g = U * U / (2.0 * P.drop * double(L));
    const double nu = double(L) * U / P.Re;
    const std::size_t nsteps = std::size_t(P.tmax * double(L) / U);

    std::printf("Tilted square into water   D3Q27 fluid + D3Q7 phase, nz = 1\n");
    std::printf("backend %s   precision %s\n", ExecSpace::name(), precision_name());
    std::printf("fluid %s   phase %s (defaulted)\n", FColl::name, PColl::name);
    std::printf("%dx%dx%d   L = %d   rho_H/rho_L = %.0f   Re = %.0f   nu = %.3e\n",
                int(nx), int(ny), int(nz), int(L), rho_h / rho_l, P.Re, nu);
    const double tauf = nu / (1.0 / 3.0);
    std::printf("impact U = %.4f   g = %.3e   drop = %.2f L   omega_f = %.6f\n",
                U, g, P.drop, 1.0 / (tauf + 0.5));
    std::printf("body rho = %.2f x water   release tilt = %.1f deg   %zu steps%s\n",
                P.body_rho, P.theta, nsteps,
                P.unconstrained ? "   [-free: PLANAR CONSTRAINT OFF]" : "");

    Domain d(nx, ny, nz, /*periodic x*/ true, /*y*/ false, /*z*/ true);

    PColl pc;
    pc.omega = PColl::omega_from_mobility(Real(P.M));   // reads D3Q7's cs2 = 1/4
    pc.width = Real(P.iw);
    std::printf("phase M = %.4f -> omega_phi = %.6f  (cs2 = %.4f on %s)\n\n",
                P.M, double(pc.omega), double(cs2<PL, Real>()), PL::name);
    PhaseSlv pf(d, pc);

    const Real yw = Real(y_water), iwr = Real(P.iw);
    const Index hy = d.hy;
    pf.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real y = Real(py - hy);
      return Real(0.5) * (Real(1) + Kokkos::tanh(Real(2) * (yw - y) / iwr));
    });

    ViscousInterfaceForce<FL> vf(d);
    Body body(d);
    // hz is deliberately LARGE against smooth. With one cell in z the indicator
    // must saturate there, or chi is scaled down everywhere and the body is only
    // partly solid: at hz = 0.5 and smooth = 1.5 the z factor is 0.66, not 1.
    body.shape = Box{Real(0.5 * double(nx)),
                     Real(y_water + P.drop * double(L) + 0.5 * double(L)),
                     Real(0),
                     Real(0.5 * double(L)), Real(0.5 * double(L)), Real(8),
                     Real(1.5)};
    body.shape.set_orientation(
        Quat::from_axis_angle(Real(0), Real(0), Real(1),
                              Real(P.theta * M_PI / 180.0)));
    body.by = Real(-g);

    FColl fc;
    fc.phi = pf.phi();
    fc.Gx = pf.grad_x();  fc.Gy = pf.grad_y();  fc.Gz = pf.grad_z();
    fc.Lap = pf.laplacian();
    fc.Vx = vf.x();  fc.Vy = vf.y();  fc.Vz = vf.z();
    fc.Ex = body.x(); fc.Ey = body.y(); fc.Ez = body.z();
    fc.rho_L = Real(rho_l);      fc.rho_H = Real(rho_h);
    fc.mu_L  = Real(rho_l * nu); fc.mu_H  = Real(rho_h * nu);
    fc.kappa = FColl::kappa_from_sigma(Real(P.sigma), Real(P.iw));
    fc.beta  = FColl::beta_from_sigma(Real(P.sigma), Real(P.iw));
    fc.by    = Real(-g);
    FluidSlv fl(d, fc);

    auto phiv = pf.phi();
    const Real rl = Real(rho_l), rh = Real(rho_h), gr = Real(g);
    fl.initialize_field(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real y = Real(py - hy);
      const Real dz = y - yw;
      const Real az = (dz < Real(0) ? -dz : dz) * Real(2) / iwr;
      const Real lnch = az + Kokkos::log(Real(1) + Kokkos::exp(Real(-2) * az))
                      - Real(0.6931471805599453);
      const Real I = rh * dz - (rh - rl) * Real(0.5) * (dz + Real(0.5) * iwr * lnch);
      const Real r = rl + phiv(n) * (rh - rl);
      return FlowState{(-gr * I) / (r / Real(3)), Real(0), Real(0), Real(0)};
    });

    // GEOMETRY AFTER THE SEED, and both solvers get it -- the same ordering
    // water_entry.cpp argues at length. Seeding p~ = 0 into a wall against a
    // hydrostatic neighbour is a pressure discontinuity that survives two steps.
    fl.set_geometry([&](Index, Index y, Index) -> CellType {
      return (y == 0 || y == ny - 1) ? Solid : Fluid;
    });
    pf.set_geometry([&](Index, Index y, Index) -> PhaseCell {
      return (y == 0 || y == ny - 1) ? PhaseWall : PhaseBulk;
    });

    pf.set_velocity(fl.ux(), fl.uy(), fl.uz());
    vf.set_velocity(fl.ux(), fl.uy(), fl.uz());
    vf.set_phase_gradient(pf.grad_x(), pf.grad_y(), pf.grad_z());
    body.set_velocity(fl.ux(), fl.uy(), fl.uz());

    // set_uniform_density6, NOT set_uniform_density: the 2-D one fills the mass
    // and leaves the inertia TENSOR at zero, which makes the angular half of the
    // 6x6 singular and produces a plausible tumble rather than a failure.
    body.set_uniform_density6(Real(P.body_rho * rho_h));
    std::printf("penalised volume %.1f cells (nominal %d x 1)   mass %.4e\n",
                double(body.penalised_volume()), int(L * L), double(body.mass));

    const std::size_t every = nsteps / std::size_t(P.nframes > 0 ? P.nframes : 1);
    std::printf("\n%-8s %-8s %-9s %-9s %-8s %-11s %-11s %-11s %-9s\n",
                "t U/L", "step", "y_c/L", "V/U", "tilt", "F_y", "tau_z",
                "|tau_x,y|", "max|u|");
    std::printf("%s\n", std::string(100, '-').c_str());

    auto phi_view = pf.phi();
    auto dens_of = KOKKOS_LAMBDA(Index n) {
      const Real q = phi_view(n);
      const Real c = q < Real(0) ? Real(0) : (q > Real(1) ? Real(1) : q);
      return Real(rho_l) + c * Real(rho_h - rho_l);
    };

    double worst_outplane = 0.0, worst_inplane = 0.0;
    int frame = 0;
    for (std::size_t step = 0; step <= nsteps; ++step) {
      pf.refresh();
      fl.compute_macroscopic();
      vf.refresh(fc);
      const auto R = body.refresh6(dens_of);

      const double oop = std::sqrt(double(R.tx) * double(R.tx)
                                 + double(R.ty) * double(R.ty));
      worst_outplane = std::fmax(worst_outplane, oop);
      worst_inplane  = std::fmax(worst_inplane, std::fabs(double(R.tz)));

      if (every && step % every == 0) {
        auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, pf.phi());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
        double umx = 0;  bool bad = false;
        for (Index y = 1; y < ny - 1; ++y)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y, 0);
            if (!std::isfinite(double(hp(n)))) bad = true;
            const double a = double(hu(n)), b = double(hv(n));
            umx = std::fmax(umx, std::sqrt(a * a + b * b));
          }
        std::printf("%-8.3f %-8zu %-9.4f %-9.4f %-8.2f %-11.4e %-11.4e %-11.4e %-9.3e\n",
                    double(step) * U / double(L), step,
                    (double(body.shape.cy) - y_water) / double(L),
                    double(body.vy) / U, tilt_deg(body.shape.q),
                    double(R.fy), double(R.tz), oop, umx);
        if (!P.dump.empty()) {
          char nm[512];
          auto at = [&](const char* f) {
            std::snprintf(nm, sizeof nm, "%s/rt_%04d_%s.bin",
                          P.dump.c_str(), frame, f);
            return std::string(nm);
          };
          dump_field(at("phi"), nx, ny, [&](Index x, Index y) { return hp(d.id(x, y, 0)); });
          dump_field(at("ux"),  nx, ny, [&](Index x, Index y) { return hu(d.id(x, y, 0)); });
          dump_field(at("uy"),  nx, ny, [&](Index x, Index y) { return hv(d.id(x, y, 0)); });
          // The body mask, so the renderer can composite the square. A Box is
          // a genuine 3-D shape, so chi takes three coordinates; z = 0 is the
          // one plane this domain has.
          const Box bs = body.shape;
          dump_field(at("body"), nx, ny, [&](Index x, Index y) {
            return bs.chi(Real(x), Real(y), Real(0));
          });
        }
        ++frame;
        if (bad) { std::printf("  DIVERGED\n"); status = 1; break; }
      }
      if (step == nsteps) break;

      fl.step(true);
      pf.step();

      // THE PLANAR CONSTRAINT. See the banner: the 6x6 has no per-axis control,
      // so the two out-of-plane components are zeroed here, after the solve and
      // before advance6 integrates the quaternion from them.
      //
      // MEASURED TO BE A NO-OP AT nz = 1, and -free is how that is reproduced:
      // it skips the two lines and nothing moves. The reason is symmetry rather
      // than luck -- with one cell in z and the body centred on it, every cell's
      // z offset from the centre is zero, so r x F has no x or y component and
      // the 6x6 is handed tau_x = tau_y = 0 to machine precision. The constraint
      // is kept as a GUARD, not as a correction: it makes the intent explicit
      // and it would catch any case where that symmetry stopped holding.
      if (!P.unconstrained) { body.wx = Real(0);  body.wy = Real(0); }
      body.advance6();

      const Real half = Real(std::sqrt(double(body.shape.hx * body.shape.hx +
                                              body.shape.hy * body.shape.hy)));
      if (body.shape.cy - half < Real(2)) {
        body.shape.cy = half + Real(2);
        if (body.vy < Real(0)) body.vy = Real(0);
      }
      if (!P.unconstrained) { body.shape.cz = Real(0);  body.vz = Real(0); }
    }

    std::printf("\nWHAT THE PLANAR CONSTRAINT DISCARDED, which is the number to\n"
                "read before believing any of the above:\n");
    std::printf("  worst |tau_x,y| = %.4e   against worst |tau_z| = %.4e"
                "   ratio %.2e\n", worst_outplane, worst_inplane,
                worst_inplane > 0 ? worst_outplane / worst_inplane : 0.0);
    std::printf("  A planar problem should give a ratio far below 1. If it does\n"
                "  not, this setup is not describing one -- use D2Q9 with Rect.\n");
    std::printf("\n%d frame(s)%s\n", frame,
                P.dump.empty() ? " (pass -dump <dir> to write fields for render_rt)"
                               : " written; render with demonstrator/render_rt");
  }
  Kokkos::finalize();
  return status;
}
