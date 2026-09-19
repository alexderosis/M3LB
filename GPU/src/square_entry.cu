//==============================================================================
//  A tilted square falling into water. The CUDA twin of the parent's
//  demonstrator/water_entry_q27.cpp, and a cross-check of the COUPLED stack.
//
//  WHY THIS CASE AND NOT ANOTHER. src/impact.cu already puts a penalised body
//  into a phase-field free surface, but it drops a SPHERE, whose chi is
//  invariant under rotation -- it sets free_rotation = false and says so,
//  because the roll equation there measures nothing. So until this file the
//  port had no case in which the body ROTATES in response to the fluid, and the
//  torque half of the penalisation coupling was untested against anything.
//  A square released tilted strikes one corner first, the reaction on that
//  corner is not through the centre, and it slaps flat. That is the sequence a
//  belly-flop makes painful and it is the one this file exists to exercise.
//
//  THE PARENT HAS TO WORK HARDER FOR THIS THAN WE DO, which is worth recording
//  because it is the only place the port is the more natural expression.
//  src/solver/PenalisedBody.hpp carries
//
//      static_assert(L::D == 2 || Shape::three_d, ...)
//
//  so Rect -- a prism, with the validated planar 3x3 in x, y and theta_z --
//  is REFUSED on D3Q27 there, and its D3Q27 driver has to use Box, take the
//  full 6x6, and then zero the two out-of-plane angular velocities by hand
//  between refresh6 and advance6. This tree has one fluid lattice, D3Q27, so a
//  prism is the only way it can express a planar body at all; Rect therefore
//  carries no three_d flag, there is no assertion, and the planar solve is
//  simply what a Rect gets. Same physics, no constraint to apply.
//
//  CROSS-CHECKED AGAINST THAT PARENT, which is the point of having both. At
//  L = 16, ratio 100, Re = 200, U = 0.04, xi = 4, M = 0.02, sigma = 1e-3,
//  rho_b = 0.9 x water, released 1.5 L above the surface at 20 degrees:
//
//      step   y_c/L (parent/here)  V/U (parent/here)   tilt (parent/here)
//      400    1.8439 / 1.8439      -0.3109 / -0.3101   20.02 / 20.02
//      800    1.3801 / 1.3800      -0.6165 / -0.6158   20.01 / 20.01
//      1200   0.6628 / 0.6627      -0.6145 / -0.6171   18.83 / 18.83
//      1600   0.2771 / 0.2769      -0.3154 / -0.3157   18.39 / 18.39
//      2400  -0.1501 / -0.1509     -0.0958 / -0.0962   18.83 / 18.79
//
//  Three to four significant figures through free fall, the impact spike
//  (F_y 3.0151 against 3.0133) and the rebound, with no shared headers. That
//  covers the phase field, the pressure-form fluid, the penalised body AND the
//  torque coupling in one comparison.
//
//  TWO FAILURES ON THE WAY TO IT, both recorded because both looked like
//  results rather than mistakes.
//
//   * SEED THE HYDROSTATIC PRESSURE. With p~ = 0 everywhere the square falls
//     straight through the water -- F_y stays around 1e-3 where it should reach
//     3.0, and the body free-falls at g past the surface as though the water
//     were not there. It does not diverge and it does not look broken; it looks
//     like a simulation of something. Penalisation has to push against a
//     pressure field, and with a uniform gauge there is none to push against.
//
//   * SET A GEOMETRY. neighbour() here is unconditionally periodic, so without
//     a floor and a lid the hydrostatic column wraps from the tank bottom to the
//     top of the air and the run is NON-FINITE inside 400 steps. The parent's
//     Domain would have given the non-periodic axis a halo; this tree has no
//     such thing, so the walls are not optional decoration.
//
//  Usage:  square_entry [-l 16] [-ratio 100] [-re 200] [-u 0.04] [-iw 4]
//                       [-m 0.02] [-sigma 1e-3] [-rhob 0.9] [-drop 1.5]
//                       [-theta 20] [-tmax 6] [-frames 7]
//==============================================================================
#include "lbm/backend.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

int main(int argc, char** argv) {
  int L = 16, frames = 7;
  double ratio = 100.0, Re = 200.0, U = 0.04, iw = 4.0, mob = 0.02;
  double sigma = 1e-3, rho_b = 0.9, drop = 1.5, tmax = 6.0, theta_deg = 20.0;
  for (int i = 1; i < argc; ++i) {
    auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-l"))      { if (i+1<argc) L = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-frames")) { if (i+1<argc) frames = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "-ratio"))  num(ratio);
    else if (!std::strcmp(argv[i], "-re"))     num(Re);
    else if (!std::strcmp(argv[i], "-u"))      num(U);
    else if (!std::strcmp(argv[i], "-iw"))     num(iw);
    else if (!std::strcmp(argv[i], "-m"))      num(mob);
    else if (!std::strcmp(argv[i], "-sigma"))  num(sigma);
    else if (!std::strcmp(argv[i], "-rhob"))   num(rho_b);
    else if (!std::strcmp(argv[i], "-drop"))   num(drop);
    else if (!std::strcmp(argv[i], "-tmax"))   num(tmax);
    else if (!std::strcmp(argv[i], "-theta"))  num(theta_deg);
  }

  const int nx = 6 * L, ny = 7 * L, nz = 1;
  const long N = long(nx) * ny * nz;
  const double y_water = 4.0 * L;
  const double nu = double(L) * U / Re;
  const double g  = U * U / (2.0 * drop * double(L));
  const double theta = theta_deg * M_PI / 180.0;
  const std::size_t nsteps = std::size_t(tmax * double(L) / U);

  const backend::DeviceInfo dev = backend::device_info();
  std::printf("Tilted square into water   D3Q27 fluid + D3Q7 phase, nz = 1\n");
  std::printf("device %s   precision %s\n", dev.name.c_str(),
              sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("%dx%dx%d   L = %d   rho_H/rho_L = %.0f   Re = %.0f   nu = %.3e\n",
              nx, ny, nz, L, ratio, Re, nu);
  std::printf("U = %.4f   g = %.3e   drop = %.2f L   tilt = %.1f deg   "
              "rho_b = %.2f x water   %zu steps\n\n",
              U, g, drop, theta_deg, rho_b, nsteps);

  backend::PhaseField pf(nx, ny, nz);      // D3Q7 phase, the default
  pf.phase.width = Real(iw);
  pf.set_mobility(Real(mob));              // through the solver, so cs2 = 1/4 is used
  pf.fluid.rho_L = Real(1);          pf.fluid.rho_H = Real(ratio);
  pf.fluid.mu_L  = Real(1.0 * nu);   pf.fluid.mu_H  = Real(ratio * nu);
  pf.fluid.beta  = MultiphaseModel::beta_from_sigma(Real(sigma), Real(iw));
  pf.fluid.kappa = MultiphaseModel::kappa_from_sigma(Real(sigma), Real(iw));
  pf.fluid.by    = Real(-g);
  pf.enable_viscous_force(true);

  // phi = 1 is water and phi = 0 air, with the pressure integrated through the
  // diffuse interface and the zero AT the surface. See the banner on why the
  // second half of this is not optional.
  pf.initialise_with([&](int, int y, int, Real& ph, Real& pt) {
    const double phv = 0.5 * (1.0 + std::tanh(2.0 * (y_water - double(y)) / iw));
    ph = Real(phv);
    const double dz = double(y) - y_water;              // positive upward, in air
    const double az = std::fabs(dz) * 2.0 / iw;
    const double lnch = az + std::log(1.0 + std::exp(-2.0 * az))
                      - 0.6931471805599453;
    const double I = ratio * dz - (ratio - 1.0) * 0.5 * (dz + 0.5 * iw * lnch);
    const double r = 1.0 + phv * (ratio - 1.0);
    pt = Real((-g * I) / (r / 3.0));
  });

  // Tank floor and lid, for both distributions. Not decoration: see the banner.
  {
    const std::size_t NN = std::size_t(N);
    const std::uint8_t pbulk = PhaseBulk, pwall = PhaseWall;
    const std::uint8_t ffluid = Fluid, fsolid = Solid;
    std::vector<std::uint8_t> pfl(NN, pbulk), ffl(NN, ffluid);
    for (long n = 0; n < N; ++n) {
      int x, y, z;  coords(n, nx, ny, x, y, z);
      if (y == 0 || y == ny - 1) {
        pfl[std::size_t(n)] = pwall;  ffl[std::size_t(n)] = fsolid;
      }
    }
    pf.set_geometry(pfl, ffl);
  }

  //---- the square -------------------------------------------------------------
  backend::Body<Rect> body(nx, ny, nz);
  body.shape.cx = Real(0.5 * double(nx));
  body.shape.cy = Real(y_water + drop * double(L) + 0.5 * double(L));
  body.shape.hx = Real(0.5 * double(L));
  body.shape.hy = Real(0.5 * double(L));
  body.shape.smooth = Real(1.5);
  body.shape.set_angle(Real(theta));       // Rect carries the tilt natively
  body.vx = Real(0);  body.vy = Real(0);  body.omega = Real(0);
  body.props.by = Real(-g);                // the SAME vector the collision gets
  // ON, unlike src/impact.cu's sphere, and that is the whole point of this case:
  // a square's chi is NOT invariant under rotation, so the roll equation has
  // something to measure and the impact torque is a result rather than zero.
  body.props.free_rotation = true;
  body.set_uniform_density(Real(rho_b * ratio));
  body.couple_velocity(pf.ux_device(), pf.uy_device(), pf.uz_device());

  // The same clamp the equation of state uses, and for the same reason: the
  // body's effective mass is m_b - integral(chi rho), and an unclamped rho lets
  // that bracket collapse or change sign, which is a runaway in Newton's
  // equation rather than in the flow.
  auto dens = [&](long n) {
    const Real q = pf.phi_device()[n];
    const Real c = q < Real(0) ? Real(0) : (q > Real(1) ? Real(1) : q);
    return Real(1) + c * Real(ratio - 1.0);
  };

  BodyReaction R{};
  pf.couple_external_force(body.fx(), body.fy(), body.fz());
  // THE BODY RUNS INSIDE pf.step(). Penalisation needs the window between the
  // macroscopic field and the collision; calling it around the step reads a u
  // one step stale, which inverts the sign of the measured deficit.
  pf.set_pre_fluid([&]{ R = body.refresh(dens); });

  {
    const auto m = body.indicator_moments();
    std::printf("penalised area from chi %.1f   nominal %d   (%+.2f %%)\n",
                m.area, L * L, 100.0 * (m.area - double(L * L)) / double(L * L));
    std::printf("body mass %.4e   displaced water %.4e\n\n",
                double(body.props.mass), ratio * double(L * L));
  }

  //---- march ------------------------------------------------------------------
  const std::size_t every =
      (frames > 1) ? std::max<std::size_t>(1, nsteps / std::size_t(frames - 1))
                   : nsteps + 1;
  std::printf("%-8s %-8s %-9s %-9s %-8s %-11s %-11s %-9s\n",
              "t U/L", "step", "y_c/L", "V/U", "tilt", "F_y", "torque", "max|u|");
  std::printf("--------------------------------------------------------"
              "--------------------------\n");

  std::vector<Real> phi, ux, uy;
  int status = 0;
  for (std::size_t step = 0; step <= nsteps; ++step) {
    if (step % every == 0 || step == nsteps) {
      backend::sync();
      pf.field_to_host(pf.phi_device(), phi);
      pf.field_to_host(pf.ux_device(),  ux);
      pf.field_to_host(pf.uy_device(),  uy);
      double umx = 0;  bool bad = false;
      for (long n = 0; n < N; ++n) {
        if (!std::isfinite(double(phi[std::size_t(n)]))) bad = true;
        const double a = double(ux[std::size_t(n)]), b = double(uy[std::size_t(n)]);
        umx = std::fmax(umx, std::sqrt(a * a + b * b));
      }
      std::printf("%-8.3f %-8zu %-9.4f %-9.4f %-8.2f %-11.4e %-11.4e %-9.3e%s\n",
                  double(step) * U / double(L), step,
                  (double(body.shape.cy) - y_water) / double(L),
                  double(body.vy) / U, double(body.shape.theta) * 180.0 / M_PI,
                  R.fy, R.torque, umx, bad ? "   NON-FINITE" : "");
      if (bad) { status = 1; break; }
    }
    if (step == nsteps) break;

    pf.step();
    body.advance();

    // A floor stop, not a contact model. The reach of a tilted square is its
    // diagonal, so the clamp uses that rather than the half-height.
    const Real half = Real(std::sqrt(double(body.shape.hx) * double(body.shape.hx)
                                   + double(body.shape.hy) * double(body.shape.hy)));
    if (body.shape.cy - half < Real(2)) {
      body.shape.cy = half + Real(2);
      if (body.vy < Real(0)) body.vy = Real(0);
    }
  }
  std::printf("\n%s\n", status ? "NON-FINITE -- see the two failures in the banner"
                               : "ran to completion.");
  return status;
}
