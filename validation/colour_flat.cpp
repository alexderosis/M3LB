//==============================================================================
//  The flat colour interface: what beta actually does to the interface width.
//
//  The colour-gradient model does not PRESCRIBE an interface width. The phase
//  field does -- W is an input to the conservative Allen-Cahn source, and
//  validation/phase_flat.cpp checks that the seeded tanh of width W is a
//  stationary solution. Here the width is an OUTCOME: a balance between the
//  recolouring step, which pushes colour up the gradient, and the diffusion
//  that collision and streaming apply to it. beta is the only knob on the
//  first half of that balance, and this file measures what it buys.
//
//  THE GEOMETRY IS A SLAB, periodic in every direction, at rest, with no
//  gravity and no prescribed velocity. That is deliberate: a droplet curves the
//  interface and mixes the width question with Laplace's law, and a stretched
//  interface (enan_rt_cg) mixes it with the flow. Here the equilibrium profile
//  is one-dimensional and flat, so anything that happens is the scheme.
//
//  ===================== WHAT IS MEASURED, AND WHY THREE THINGS ==============
//  WIDTH, by a least-squares fit of
//
//      phi(y) = tanh( 2 (y - y0) / W )
//
//  to the profile across ONE interface, over the cells where |phi| < 0.9. The
//  fit is on atanh(phi), which is linear in y, so it is a straight line fit and
//  has no starting guess to get wrong. Cells outside that band are excluded
//  because atanh saturates there and a single cell at |phi| = 0.999 would
//  otherwise dominate the residual.
//
//  SPURIOUS CURRENT, u_max over the whole box. The interface is at rest and the
//  exact answer is zero, so every bit of this is discretisation. It is reported
//  because beta trades against it: ColourGradient.hpp's banner records 0.7 as
//  "the largest that keeps the interface smooth", which is a statement about
//  this quantity and not about the width.
//
//  MASS DRIFT, and it is here as a NEGATIVE control rather than as a check.
//  The recolouring splits f_i into two parts that sum to f_i identically, so
//  mass and momentum survive it WHATEVER beta is -- see Eqs. (33)-(34). A beta
//  that ruins the interface will still conserve mass to round-off. So a clean
//  drift column proves nothing about beta, and it is printed precisely so that
//  it is not mistaken for evidence.
//
//  ===================== WHAT THIS DOES NOT DO ===============================
//  One density ratio and one viscosity per run: the question here is beta, and
//  a sweep that moved gamma at the same time could not attribute anything. No
//  claim that the fitted W is the continuum width of any particular model --
//  it is the width THIS discretisation settles on, which is the quantity a
//  caller actually gets.
//
//    usage: colour_flat [-n N] [-w W0] [-gamma G] [-tau T] [-a A]
//                       [-steps K] [-betas b1,b2,...] [-out FILE]
//==============================================================================
#include "collision/ColourGradient.hpp"
#include "core/Types.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ColourGradientSolver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

using L   = D3Q27;
using CG  = ColourGradient<L>;
using Slv = ColourGradientSolver<L, EsotericPull<L>, CG>;

struct Out {
  double W = 0, y0 = 0, resid = 0;
  double umax = 0, drift = 0, phi_lo = 0, phi_hi = 0;
  int    npts = 0;
  bool   ok = false;
};

static Out run(int N, double W0, double gamma, double A, double tau,
               double beta, std::size_t steps) {
  const double ab = 8.0 / 27.0;
  const double ar = 1.0 - (1.0 - ab) / gamma;
  const double rho_b0 = 1.0, rho_r0 = gamma;
  const double nu = (1.0 / 3.0) * (tau - 0.5);

  // Thin in x and z: the equilibrium is one-dimensional, so cells across the
  // interface are the only ones that carry information and the rest is cost.
  // Four is not one -- a single cell would make the x and z gradients exactly
  // zero by construction and could not show a scheme that breaks the symmetry.
  const Index nx = 4, ny = Index(N), nz = 4;
  Domain d(nx, ny, nz, true, true, true);

  CG cg;
  cg.alpha_r = Real(ar);   cg.alpha_b = Real(ab);
  cg.nu_r    = Real(nu);   cg.nu_b    = Real(nu);
  cg.A       = Real(A);
  cg.beta    = Real(beta);
  cg.omega_bulk = Real(1);
  cg.rho_r0  = Real(rho_r0);  cg.rho_b0 = Real(rho_b0);
  Slv s(d, cg);

  // A slab of red in the middle, so the box holds TWO interfaces and stays
  // periodic without a jump. The fit below uses the lower one only.
  const Real Wr = Real(W0), rr0 = Real(rho_r0), rb0 = Real(rho_b0);
  const Real q1 = Real(0.25 * double(ny)), q3 = Real(0.75 * double(ny));
  const Index hy = d.hy;
  const Domain dd = d;
  s.initialize(KOKKOS_LAMBDA(Index n) {
    Index px, py, pz; dd.coords(n, px, py, pz);
    const Real y = Real(py - hy);
    // +1 inside the slab, -1 outside; two tanh fronts back to back.
    const Real t = Kokkos::tanh(Real(2) * (y - q1) / Wr)
                 * Kokkos::tanh(Real(2) * (q3 - y) / Wr);
    return Slv::Colours{Real(0.5) * rr0 * (Real(1) + t),
                        Real(0.5) * rb0 * (Real(1) - t)};
  });

  const double m_r0 = double(s.total_red()), m_b0 = double(s.total_blue());
  for (std::size_t k = 0; k < steps; ++k) { s.refresh(); s.step(); }
  s.refresh();

  auto hp = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.phi());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto hw = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());

  Out o;
  bool bad = false;
  for (Index z = 0; z < nz; ++z)
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x) {
        const Index n = d.id(x, y, z);
        const double p = double(hp(n));
        if (!std::isfinite(p)) bad = true;
        o.phi_lo = std::min(o.phi_lo, p);
        o.phi_hi = std::max(o.phi_hi, p);
        const double a = double(hu(n)), b = double(hv(n)), c = double(hw(n));
        o.umax = std::max(o.umax, std::sqrt(a * a + b * b + c * c));
      }

  // The LOWER interface, phi rising through zero near y = ny/4. atanh(phi) is
  // linear in y with slope 2/W, so this is a straight-line least squares and
  // there is no initial guess to get wrong.
  double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
  for (Index y = 0; y < ny / 2; ++y) {
    const double p = double(hp(d.id(nx / 2, y, nz / 2)));
    if (!std::isfinite(p) || std::fabs(p) > 0.9) continue;
    const double t = std::atanh(p);
    const double yy = double(y);
    sx += yy; sy += t; sxx += yy * yy; sxy += yy * t; ++m;
  }
  o.npts = m;
  if (m >= 3) {
    const double den = double(m) * sxx - sx * sx;
    const double slope = (double(m) * sxy - sx * sy) / den;
    const double inter = (sy - slope * sx) / double(m);
    o.W  = 2.0 / slope;                       // phi = tanh(2 (y-y0)/W)
    o.y0 = -inter / slope;
    double r2 = 0;
    for (Index y = 0; y < ny / 2; ++y) {
      const double p = double(hp(d.id(nx / 2, y, nz / 2)));
      if (!std::isfinite(p) || std::fabs(p) > 0.9) continue;
      const double e = std::atanh(p) - (slope * double(y) + inter);
      r2 += e * e;
    }
    o.resid = std::sqrt(r2 / double(m));
  }

  const double m_r1 = double(s.total_red()), m_b1 = double(s.total_blue());
  o.drift = std::max(std::fabs(m_r1 / m_r0 - 1.0), std::fabs(m_b1 / m_b0 - 1.0));
  o.ok = !bad && std::isfinite(o.W) && o.W > 0 && m >= 3;
  return o;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    int N = 64;
    double W0 = 4, gamma = 1, A = 8e-4, tau = 1.0;
    std::size_t steps = 8000;
    std::string outf;
    std::vector<double> betas = {0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto num = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
      if      (a == "-n"     && i + 1 < argc) N = std::atoi(argv[++i]);
      else if (a == "-w")     num(W0);
      else if (a == "-gamma") num(gamma);
      else if (a == "-tau")   num(tau);
      else if (a == "-a")     num(A);
      else if (a == "-steps" && i + 1 < argc) steps = std::size_t(std::atol(argv[++i]));
      else if (a == "-out"   && i + 1 < argc) outf = argv[++i];
      else if (a == "-betas" && i + 1 < argc) {
        betas.clear();
        std::string v = argv[++i], tok;
        for (char ch : v + ",") {
          if (ch == ',') { if (!tok.empty()) betas.push_back(std::atof(tok.c_str())); tok.clear(); }
          else tok += ch;
        }
      }
    }

    std::printf("Colour gradient: the flat interface, and what beta does to its width\n");
    std::printf("  D3Q27  %d x %d x %d slab, periodic, at rest   gamma = %.0f   tau = %.2f"
                "   A = %.1e   seeded W0 = %.1f   %zu steps\n\n",
                4, N, 4, gamma, tau, A, W0, steps);
    std::printf("  %6s %9s %9s %8s %6s %12s %11s %s\n",
                "beta", "W", "W/W0", "y0", "pts", "u_max", "mass drift", "phi range");
    std::printf("  %s\n", std::string(88, '-').c_str());

    std::FILE* f = outf.empty() ? nullptr : std::fopen(outf.c_str(), "w");
    if (f) std::fprintf(f, "# beta W W_over_W0 umax drift phi_lo phi_hi resid\n");

    for (double b : betas) {
      const Out o = run(N, W0, gamma, A, tau, b, steps);
      std::printf("  %6.2f %9.4f %9.4f %8.2f %6d %12.3e %11.2e  [%.4f, %.4f]%s\n",
                  b, o.W, o.W / W0, o.y0, o.npts, o.umax, o.drift,
                  o.phi_lo, o.phi_hi, o.ok ? "" : "   FAILED");
      if (f) std::fprintf(f, "%g %g %g %g %g %g %g %g\n", b, o.W, o.W / W0,
                          o.umax, o.drift, o.phi_lo, o.phi_hi, o.resid);
      if (!o.ok) status = 1;
    }
    if (f) std::fclose(f);

    std::printf("\n  phi must stay inside [-1, 1]: outside it a colour has gone NEGATIVE,\n"
                "  which is the recolouring overshooting and not a width the caller can use.\n");
    std::printf("  Mass drift is a NEGATIVE control here -- Eqs. (33)-(34) sum to f_i\n"
                "  identically, so it stays clean at every beta, good or bad.\n");
  }
  Kokkos::finalize();
  return status;
}
