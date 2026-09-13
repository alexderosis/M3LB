//==============================================================================
//  Decay of the slowest viscous mode in a circular container -- an EXACT
//  Navier-Stokes solution, used to measure what a curved wall costs.
//
//  WHY THIS MODE. Neffaa, Bos & Schneider (Phys. Plasmas 15, 092304, 2008)
//  report that decaying MHD turbulence in a circular container ends in an
//  exponential viscous decay E ~ exp(-2 alpha nu t) with alpha "related to the
//  largest Stokes eigenvalue of the circle (alpha = 1.64)". That eigenvalue is
//  not a fitted number: the slowest-decaying mode of a disc with no-slip is the
//  purely AZIMUTHAL one,
//
//      u_theta(r, t) = U J1(beta r / R) exp(-nu beta^2 t / R^2),   u_r = 0,
//
//  with beta = j(1,1) = 3.8317059702 the first zero of J1, which makes
//  u_theta(R) = 0 automatically. Their r = 19 pi / 20 gives
//  alpha = (beta/R)^2 = 1.648, which is the 1.64 they quote.
//
//  AND IT IS EXACT, NOT A STOKES LIMIT. For a purely azimuthal field the
//  nonlinear term is u.grad u = -(u_theta^2 / r) r_hat, a centripetal
//  acceleration carried entirely by the pressure gradient; the theta-momentum
//  equation is then exactly linear,
//
//      du_theta/dt = nu (lap u_theta - u_theta / r^2),
//
//  and J1(beta r/R) is its eigenfunction. So this is a full Navier-Stokes
//  solution at any amplitude the Mach number allows, not a small-amplitude or
//  creeping-flow approximation, and any measured departure from the analytic
//  rate is the WALL or the lattice and nothing else.
//
//  WHAT IS BEING MEASURED. A circle is not a lattice direction, so the question
//  is what a curved no-slip wall costs. Two treatments, on the same exact
//  solution:
//
//    -wall bb   staircase halfway bounce-back: every cell outside R is Solid.
//               Sharp, free under Esoteric Pull, and geometrically wrong by up
//               to half a cell in a way that varies around the circumference.
//
//    -wall pen  volume penalisation, the method Neffaa et al. use: a body force
//               F = -chi(x) u / eps with chi a smoothed indicator of r > R,
//               applied through FieldGuo. Smooth, but the wall is spread over
//               the width of chi and the boundary is only approached as eps
//               falls -- and an explicit penalisation is stable only for
//               eps >~ dt, which on a lattice with dt = 1 means eps of order 1,
//               far above the 1e-3 a spectral code with dt = 5e-4 can use.
//
//  TAU IS FIXED AT THE MAGIC VALUE by default. For BGK the bounce-back wall
//  sits exactly halfway only when Lambda = (tau - 1/2)^2 = 3/16, i.e.
//  tau = 0.9330127; away from it the effective wall position drifts with
//  viscosity, which on a decay-rate measurement would masquerade as a wall
//  error that refines away differently. Fixing tau there removes the viscosity
//  dependence so that what is left is the STAIRCASE, which is the thing worth
//  measuring. -tau reintroduces it on purpose.
//==============================================================================
#include "Campaign.hpp"
#include "forcing/Forcing.hpp"

using namespace lbm;
using namespace campaign;

using L    = D2Q9;
using Coll = BGK<L, SecondOrderEquilibrium<L>, NoForcing,  ShiftedPopulations>;
using CollP = BGK<L, SecondOrderEquilibrium<L>, FieldGuo,  ShiftedPopulations>;

static constexpr double BETA = 3.8317059702075123;   // first zero of J1

// J1 by its ascending series. Needed only on [0, BETA], where twenty terms are
// already at round-off; libc++ does not ship the C++17 special functions, so
// std::cyl_bessel_j is not available on this platform.
static double J1(double x) {
  const double h = 0.5 * x, q = -0.25 * x * x;
  double term = h, sum = h;
  for (int k = 1; k < 40; ++k) {
    term *= q / (double(k) * double(k + 1));
    sum += term;
    if (std::abs(term) < 1e-18 * std::abs(sum)) break;
  }
  return sum;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::vector<Index> Ns = {64, 128, 256};
    double tau = 0.9330127, U = 0.01, eps = 2.0, smooth = 1.0, rfac = 0.45;
    std::size_t steps = 0, probe = 100;
    std::string wall = "bb";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-tau"   && i + 1 < argc) tau = std::atof(argv[++i]);
      if (a == "-u"     && i + 1 < argc) U = std::atof(argv[++i]);
      if (a == "-eps"   && i + 1 < argc) eps = std::atof(argv[++i]);
      if (a == "-smooth"&& i + 1 < argc) smooth = std::atof(argv[++i]);
      if (a == "-rfac"  && i + 1 < argc) rfac = std::atof(argv[++i]);
      if (a == "-steps" && i + 1 < argc) steps = std::size_t(std::atol(argv[++i]));
      if (a == "-probe" && i + 1 < argc) probe = std::size_t(std::atol(argv[++i]));
      if (a == "-wall"  && i + 1 < argc) wall = argv[++i];
      if (a == "-ns"    && i + 1 < argc) {
        Ns.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { if (!t.empty()) Ns.push_back(Index(std::atoi(t.c_str()))); t.clear(); } else t += c; }
      }
    }
    const double nu = (tau - 0.5) / 3.0;
    std::printf("Slowest viscous mode of a disc -- exact NSE solution   D2Q9 %s\n",
                ExecSpace::name());
    std::printf("  u_theta = U J1(beta r/R) exp(-nu beta^2 t / R^2),  beta = %.10f\n", BETA);
    std::printf("  wall %s   tau %.7f (nu %.6f)%s   U %.4f%s\n",
                wall.c_str(), tau, nu,
                std::abs(tau - 0.9330127) < 1e-9 ? " [magic, Lambda = 3/16]" : "",
                U, wall == "pen" ? "" : "");
    if (wall == "pen") std::printf("  penalisation eps %.3f   chi width %.2f cells\n", eps, smooth);
    std::printf("\n  %5s %8s %12s %14s %14s %9s %10s\n",
                "N", "R", "alpha exact", "alpha measured", "rel err", "profile", "steps");
    std::printf("  %s\n", std::string(80, '-').c_str());

    for (const Index N : Ns) {
      const double R = rfac * double(N);
      const double cx = 0.5 * double(N) - 0.5, cy = 0.5 * double(N) - 0.5;
      const double alpha_ex = BETA * BETA / (R * R);
      // Energy e-folds in R^2/(2 nu beta^2); run about six of those.
      const std::size_t T = steps ? steps
                                  : std::size_t(6.0 * R * R / (2.0 * nu * BETA * BETA));
      Domain d(N, N, 1, false, false, true);
      auto rad = [&](Index x, Index y) {
        const double dx = double(x) - cx, dy = double(y) - cy;
        return std::sqrt(dx * dx + dy * dy);
      };
      // The exact field at t = 0.
      auto exact = [&](Index x, Index y, double& ux, double& uy) {
        const double r = rad(x, y);
        if (r < 1e-12 || r > R) { ux = uy = 0; return; }
        const double ut = U * J1(BETA * r / R);
        ux = -ut * (double(y) - cy) / r;
        uy =  ut * (double(x) - cx) / r;
      };
      double fitted = NAN, perr = NAN; std::size_t taken = 0;

      auto measure = [&](auto& s, View1D<Real> fx, View1D<Real> fy) {
        std::vector<double> ts, les;
        auto energy_and_err = [&](double& E, double& relerr) {
          s.compute_macroscopic();
          auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
          auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
          // SHAPE, NOT AMPLITUDE. An eigenmode decays without changing form,
          // so the test is whether the NORMALISED field still matches J1 --
          // comparing the raw field against the t = 0 profile just measures how
          // far it has decayed and reads 0.95 for a perfectly healthy run,
          // which is what the first version of this did.
          E = 0; double Eex = 0;
          for (Index y = 0; y < N; ++y)
            for (Index x = 0; x < N; ++x) {
              const double r = rad(x, y);
              if (r > R) continue;                    // the fluid region, as Neffaa integrates it
              const Index n = d.id(x, y, 0);
              const double a = double(hx(n)), b = double(hy(n));
              if (!std::isfinite(a) || !std::isfinite(b)) { E = NAN; return; }
              E += a * a + b * b;
              double ex, ey; exact(x, y, ex, ey);
              Eex += ex * ex + ey * ey;
            }
          const double sc = (E > 0 && Eex > 0) ? std::sqrt(Eex / E) : 0.0;
          double num = 0;
          for (Index y = 0; y < N; ++y)
            for (Index x = 0; x < N; ++x) {
              const double r = rad(x, y);
              if (r > R) continue;
              const Index n = d.id(x, y, 0);
              double ex, ey; exact(x, y, ex, ey);
              const double a = double(hx(n)) * sc, b = double(hy(n)) * sc;
              num += (a - ex) * (a - ex) + (b - ey) * (b - ey);
            }
          relerr = std::sqrt(num / std::max(Eex, 1e-300));
        };
        for (std::size_t t = 0; t <= T; ++t) {
          if (t % probe == 0) {
            double E, re; energy_and_err(E, re);
            if (!std::isfinite(E) || E <= 0) { fitted = NAN; return; }
            ts.push_back(double(t)); les.push_back(std::log(E));
          }
          if (t == T) break;
          if (fx.data()) {                            // penalisation: refresh F = -chi u / eps
            s.compute_macroscopic();
            auto ux = s.ux(); auto uy = s.uy();
            const double Rc = R, cxc = cx, cyc = cy, sm = smooth, ie = 1.0 / eps;
            Kokkos::parallel_for("pen", Kokkos::RangePolicy<ExecSpace>(0, d.n_padded),
              KOKKOS_LAMBDA(Index n) {
                Index px, py, pz; d.coords(n, px, py, pz);
                const double dx = double(px) - cxc, dy = double(py) - cyc;
                const double r = Kokkos::sqrt(dx * dx + dy * dy);
                const Real chi = Real(0.5 * (1.0 + Kokkos::tanh((r - Rc) / sm)));
                fx(n) = Real(-chi * ux(n) * ie);
                fy(n) = Real(-chi * uy(n) * ie);
              });
            Kokkos::fence();
          }
          s.step();
        }
        // Fit ln E over the last two thirds: the first samples carry the
        // adjustment of a continuum field onto the lattice, which is not decay.
        const std::size_t k0 = ts.size() / 3;
        double sx = 0, sy = 0, sxx = 0, sxy = 0; std::size_t m = 0;
        for (std::size_t k = k0; k < ts.size(); ++k) {
          sx += ts[k]; sy += les[k]; sxx += ts[k] * ts[k]; sxy += ts[k] * les[k]; ++m;
        }
        const double slope = (double(m) * sxy - sx * sy) / (double(m) * sxx - sx * sx);
        fitted = -slope / (2.0 * nu);       // E ~ exp(-2 alpha nu t)
        taken = T;
        double E; energy_and_err(E, perr);
      };

      if (wall == "pen") {
        CollP coll; coll.omega = CollP::omega_from_viscosity(Real(nu));
        View1D<Real> fx("fx", d.n_padded), fy("fy", d.n_padded);
        coll.forcing.Ex = fx; coll.forcing.Ey = fy;
        FluidSolver<L, EsotericPull<L>, CollP> s(d, coll);
        s.initialize_field(KOKKOS_LAMBDA(Index n) {
          Index x, y, z; d.coords(n, x, y, z); (void)z;
          const double dx = double(x) - cx, dy = double(y) - cy;
          const double r = Kokkos::sqrt(dx * dx + dy * dy);
          double ux = 0, uy = 0;
          if (r > 1e-12 && r <= R) {
            const double ut = U * J1(BETA * r / R);
            ux = -ut * dy / r; uy = ut * dx / r;
          }
          return FlowState{Real(1), Real(ux), Real(uy), Real(0)};
        });
        measure(s, fx, fy);
      } else {
        Coll coll; coll.omega = Coll::omega_from_viscosity(Real(nu));
        FluidSolver<L, EsotericPull<L>, Coll> s(d, coll);
        s.set_geometry([&](Index x, Index y, Index) -> CellType {
          return rad(x, y) > R ? Solid : Fluid;
        });
        s.initialize_field(KOKKOS_LAMBDA(Index n) {
          Index x, y, z; d.coords(n, x, y, z); (void)z;
          const double dx = double(x) - cx, dy = double(y) - cy;
          const double r = Kokkos::sqrt(dx * dx + dy * dy);
          double ux = 0, uy = 0;
          if (r > 1e-12 && r <= R) {
            const double ut = U * J1(BETA * r / R);
            ux = -ut * dy / r; uy = ut * dx / r;
          }
          return FlowState{Real(1), Real(ux), Real(uy), Real(0)};
        });
        View1D<Real> none;
        measure(s, none, none);
      }
      const double rel = (fitted - alpha_ex) / alpha_ex;
      std::printf("  %5d %8.2f %12.6e %14.6e %+8.3f %% %9.2e %10zu\n",
                  int(N), R, alpha_ex, fitted, 100.0 * rel, perr, taken);
      std::fflush(stdout);
      if (!std::isfinite(fitted)) status = 1;
    }
  }
  Kokkos::finalize();
  return status;
}
