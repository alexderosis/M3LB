//==============================================================================
//  MELTING ICE IN WATER NEAR 4 C -- the three regimes of Weady, Tong, Zidovska
//  & Ristroph, PRL 128, 044502 (2022), in a 2-D cavity at REDUCED Rayleigh
//  number. A demonstrator: nothing here has an answer to be right against;
//  what it shows is whether the coupled model (EnthalpyRegularised + DarcyGuo +
//  the anomalous buoyancy) puts the flow and the melting where the paper does.
//
//  WHAT THE PAPER FINDS. Ice at 0 C in fresh water at T_inf: below ~5 C the
//  meltwater boundary layer RISES and the ice tapers from below (a downward
//  pinnacle); above ~7 C it SINKS and the ice tapers from above (an upright
//  one); in between, a thin rising layer next to the ice sits inside a broad
//  sinking one, the counterflow rolls up and carves scallops. The reason is the
//  density maximum: water at 0 C is as light as water at ~8 C, so the cold
//  layer is buoyant against a far field below 8 C and heavy against one above.
//
//  THE MODEL IS THEIRS, THE RAYLEIGH NUMBER IS NOT. Quadratic equation of state
//  rho = rho* [1 - beta (T - 4)^2] (their choice, not Gebhart-Mollendorf), Pr =
//  12, St = 0.05 T_inf (their simulation value), ice held at 0 C throughout
//  (a 0 C wall behind it, so it cannot conduct). Their Ra/T_inf^2 = 2.5e6 C^-2
//  needs ~1000 cells per ice height and ~1e7 steps -- weeks on this CPU, under an
//  hour on the H200 -- so the default here is 3.2e4, i.e. Ra = 1.0e6 at 5.6 C,
//  78x lower. The boundary layers then carry ~8 cells at H = 256. What survives
//  that reduction is the DIRECTION of the flow and of the taper, which the
//  density anomaly decides; the scallops need a shear-layer Reynolds number
//  that may not be reached, and whether they appear is the thing to look at.
//
//  GEOMETRY, and why it is a cavity rather than their suspended cylinder. A
//  planar slab against the left wall, water to the right, the right wall at
//  T_inf, top and bottom adiabatic, all halfway walls (the pair that survived
//  validation/ice_equilibrium.cpp; the on-node regularised wall diverged there
//  under this kind of force). A cavity has no far field -- the meltwater
//  collects at the top or bottom -- so run-to-run comparisons are the claim,
//  not absolute melt rates. Two-dimensional in the tree's sense: D3Q27 central
//  moments + D3Q7 EnthalpyRegularised at nz = 1 (omega_T ~ 1.94 at the
//  defaults, where the BGK scalar rings).
//
//  UNITS. Temperatures are carried in C. Fixing the free-fall velocity U_ff =
//  sqrt(g beta T_inf^2 H) in lattice units fixes kappa = U_ff H / sqrt(Ra Pr)
//  and nu = Pr kappa; w = g beta then follows from Ra. La = c_p T_inf / St =
//  1 / 0.05 = 20 in units of c_p C, independent of T_inf.
//
//  THE CAVITY HAS NO FAR FIELD, AND THAT DECIDES HOW LONG A RUN MEANS ANYTHING.
//  With St = 0.05 T_inf, melting one cell of ice takes the heat of ~3.6 cells
//  of water cooled to 0 C, so the bulk cools as the ice goes: at aspect 1 and
//  8 C the boundary layer turned from sinking to RISING once a third of the ice
//  had melted, and 5.6 C drifted into the 4 C behaviour. The run therefore
//  watches the bulk (liquid more than 20 cells beyond the front) and stops
//  once it is `-drift` (0.25 C) from T_inf; the box defaults to aspect 2 with
//  a 32-cell slab at H = 128 to buy melt before that happens.
//
//  MEASURED 2026-09-24, H = 128, aspect 2, Ra/T_inf^2 = 3.2e4 (78x below
//  Weady et al.), each run stopped at 0.25 C of drift. u_y / U_ff across the
//  layer at mid-height, by distance from the ice:
//
//    T_inf   1-8 cells (peak)     12-32 cells (peak)   flow          taper
//    4 C     +0.103 at 6-8        ~0                   rising        from below
//    5 C     +0.047 at 4          -0.025 at 32         COUNTERFLOW   from below
//    5.6 C   +0.039 at 4          -0.061 at 24         COUNTERFLOW   from below
//    6.5 C   ~0 (-0.002 at 1)     -0.082 at 12         sinking       from above
//    7 C     -0.100 at 8          sinking              sinking       from above
//    8 C     -0.123 at 8          sinking              sinking       from above
//
//  The three regimes appear in the paper's order and the 5.6 C structure is
//  the one it describes -- a thin rising layer inside a broad sinking one. At
//  this Ra the counterflow has NOT rolled up and no scallops form, and the
//  counterflow window reads ~5-6 C against the paper's ~5-7 C.
//
//  THE UPPER EDGE IS NOT RESOLUTION -- TESTED, AND THE HYPOTHESIS FAILED. At
//  6.5 C water between 0 and 1.5 C is still lighter than the far field, so a
//  rising sublayer thinner than a cell looked like the explanation. H = 256
//  (one cell now half as far from the ice) reads -0.0021 at one cell, the same
//  as H = 128, and sinks everywhere out to 48 cells (peak -0.084 at 24); taper
//  +0.051 at 11 % melted, 1334 s. What decides it is the momentum layer: at
//  Pr = 12 it is ~sqrt(12) = 3.5 thermal layers thick, so the thin light film
//  is dragged down by the thick dense layer outside it. At 5.6 C the light
//  part spans 0-2.4 C and wins near the wall; at 6.5 C it does not. The edge
//  therefore moves with Ra and geometry, and 6 C here against 7 C in the paper
//  at 78x their Ra, planar rather than axisymmetric, is not a discrepancy that
//  this case can settle.
//
//  usage:  ice_melting [-T 5.6] [-H 128] [-ra 3.2e4] [-drift 0.25] [-out dir]
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/EnthalpyRegularised.hpp"
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "FieldDump.hpp"
#include "forcing/Forcing.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/FluidSolver.hpp"
#include "solver/ScalarSolver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace lbm;

using FL = D3Q27;
using SL = D3Q7;
using FColl = CentralMoments<FL, DarcyGuo, ShiftedPopulations>;
using SColl = EnthalpyRegularised<SL>;

namespace {

struct Opts {
  double Tinf = 5.6;        // far-field (right wall) temperature, C
  Index  H = 128;           // cavity height = Rayleigh length, cells
  double aspect = 2.0;      // cavity width / H
  double ice0 = 0.125;      // initial ice thickness / width
  double drift = 0.25;      // stop when the bulk water has drifted this far from T_inf, C
  double ra_per_T2 = 3.2e4; // Ra / T_inf^2, C^-2 (Weady et al.: 2.5e6)
  double Pr = 12.0;
  double st_per_T = 0.05;   // St / T_inf, C^-1 (Weady et al.'s simulations)
  double u_ff = 0.05;       // free-fall velocity, lattice units
  double melt = 0.5;        // stop when this fraction of the ice has melted
  double Asol = 100.0;      // Darcy drag at f_l = 0 (DarcyGuo: any value is stable)
  std::string out;          // directory for field dumps; empty = none
  long probe = 5000;
};

}  // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    double v = 0;
    if      (!std::strcmp(argv[i], "-T"))    next(o.Tinf);
    else if (!std::strcmp(argv[i], "-H"))    { next(v); o.H = Index(v); }
    else if (!std::strcmp(argv[i], "-ra"))   next(o.ra_per_T2);
    else if (!std::strcmp(argv[i], "-melt")) next(o.melt);
    else if (!std::strcmp(argv[i], "-uff"))  next(o.u_ff);
    else if (!std::strcmp(argv[i], "-drift")) next(o.drift);
    else if (!std::strcmp(argv[i], "-aspect")) next(o.aspect);
    else if (!std::strcmp(argv[i], "-ice"))  next(o.ice0);
    else if (!std::strcmp(argv[i], "-out") && i + 1 < argc) o.out = argv[++i];
    else if (std::strncmp(argv[i], "--", 2) != 0)
      std::fprintf(stderr, "ice_melting: unknown option %s\n", argv[i]);
  }

  Kokkos::initialize(argc, argv);
  {
    const Index H = o.H, W = Index(std::lround(o.aspect * double(H)));
    const Index nx = W + 2, ny = H + 2;
    const double Ra = o.ra_per_T2 * o.Tinf * o.Tinf;
    const double kappa = o.u_ff * double(H) / std::sqrt(Ra * o.Pr);
    const double nu = o.Pr * kappa;
    const double w = Ra * nu * kappa / (o.Tinf * o.Tinf * double(H) * double(H) * double(H));
    const double La = 1.0 / o.st_per_T;
    const Index xice = Index(std::lround(o.ice0 * double(W)));   // last ice column

    std::printf("ice_melting: T_inf = %.2f C  Ra = %.3e  Pr = %.1f  St = %.3f  "
                "(Weady et al. Ra = %.3e)\n", o.Tinf, Ra, o.Pr, o.st_per_T * o.Tinf,
                2.5e6 * o.Tinf * o.Tinf);
    std::printf("  box %d x %d, ice %d columns  kappa %.3e  nu %.3e  tau_f %.4f  "
                "tau_T %.4f  U_ff %.3f  A_solid %.0f\n", int(W), int(H), int(xice),
                kappa, nu, 3.0 * nu + 0.5, 4.0 * kappa + 0.5, o.u_ff, o.Asol);
    std::printf("  backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    Domain d(nx, ny, 1, false, false, true);

    PhaseChange pc;
    pc.T_s = Real(0); pc.T_l = Real(0);
    pc.cp_s = Real(1); pc.cp_l = Real(1);
    pc.k_s = Real(kappa); pc.k_l = Real(kappa);
    pc.La = Real(La); pc.E_datum = Real(0);
    pc.normalise();

    SColl scoll;
    scoll.set_material(pc);
    const Real H_ice = pc.enthalpy_of(Real(0), Real(0));
    const Real H_wat = pc.enthalpy_of(Real(o.Tinf), Real(1));
    scoll.T_ref = Real(0.5) * (H_ice + H_wat);
    scoll.omega = SColl::omega_from_diffusivity(Real(kappa));
    ScalarSolver<SL, EsotericPull<SL>, SColl> th(d, scoll);
    th.set_geometry([&](Index x, Index y, Index) -> ScalarCell {
      if (x == 0 || x == nx - 1) return ScalarDirichlet;
      if (y == 0 || y == ny - 1) return ScalarAdiabatic;
      return ScalarBulk;
    });
    // Anti-bounce-back is handed the SENSIBLE enthalpy (CLAUDE.md): E = H - La f.
    const Real E_ice = H_ice, E_wat = H_wat - pc.La;
    th.set_wall_values([&](Index x, Index, Index) -> Real {
      return (x == 0) ? E_ice : E_wat;
    });
    th.finalize_geometry();
    const PhaseChange m = pc;
    const Real Tinf = Real(o.Tinf);
    th.initialize_field(KOKKOS_LAMBDA(Index n) -> Real {
      Index px, py, pz; d.coords(n, px, py, pz);
      const Index x = px - d.hx;
      return (x <= xice) ? m.enthalpy_of(Real(0), Real(0))
                         : m.enthalpy_of(Tinf, Real(1));
    });
    th.compute_field();

    View1D<Real> Ey("Ey", d.n_padded), Av("A", d.n_padded);
    FColl fcoll;
    fcoll.omega = FColl::omega_from_viscosity(Real(nu));
    fcoll.omega_bulk = Real(1);
    fcoll.forcing = DarcyGuo{};
    fcoll.forcing.Ey = Ey;
    fcoll.forcing.A = Av;
    FluidSolver<FL, EsotericPull<FL>, FColl> fl(d, fcoll);
    fl.set_geometry([&](Index x, Index y, Index) -> CellType {
      return (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) ? Solid : Fluid;
    });
    fl.initialize(Real(1));
    th.set_velocity(fl.ux(), fl.uy(), fl.uz());

    // Buoyancy relative to the far field, f_l-weighted so that the ice carries
    // none (it is held by the drag), and the Carman-Kozeny coefficient.
    auto couple = [&]() {
      th.compute_field();
      auto Hf = th.temperature();                 // an ENTHALPY
      auto ey = Ey; auto av = Av;
      const Real wl = Real(w), Ti = Real(o.Tinf), As = Real(o.Asol), eps = Real(1e-3);
      const PhaseChange mm = pc;
      Kokkos::parallel_for("ice_melt_couple", d.n_padded, KOKKOS_LAMBDA(Index n) {
        Real f, T, E, dE;
        mm.invert(Hf(n), f, T, E, dE);
        ey(n) = f * wl * ((T - Real(4)) * (T - Real(4)) - (Ti - Real(4)) * (Ti - Real(4)));
        av(n) = As * eps * (Real(1) - f) * (Real(1) - f) / (f * f * f + eps);
      });
    };

    // Ice thickness per row: the ice is attached to the left wall, so the
    // column-sum of (1 - f_l) IS the local thickness, to sub-cell precision.
    auto thickness = [&](std::vector<double>& t) {
      th.compute_field();
      auto hH = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
      t.assign(std::size_t(H), 0.0);
      for (Index y = 1; y <= H; ++y)
        for (Index x = 1; x <= W; ++x)
          t[std::size_t(y - 1)] += 1.0 - double(pc.liquid_fraction(hH(d.id(x, y))));
    };
    // Vertical velocity in the 8 cells beside the ice, averaged over the middle
    // half of the height: its sign is the direction of the meltwater layer.
    auto bl_velocity = [&](const std::vector<double>& t) {
      fl.compute_macroscopic();
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      double acc = 0; long cnt = 0;
      for (Index y = H / 4; y <= 3 * H / 4; ++y) {
        const Index x0 = Index(std::lround(t[std::size_t(y - 1)])) + 1;
        for (Index x = x0; x < std::min(x0 + 8, W + 1); ++x) {
          acc += double(hv(d.id(x, y))); ++cnt;
        }
      }
      return cnt ? acc / double(cnt) / o.u_ff : 0.0;
    };

    // THE CAVITY HAS NO FAR FIELD. Melting one cell of ice takes La / c_p =
    // 20 C of one cell of water, so the bulk cools as the ice goes and the
    // regime drifts with it: at 8 C and aspect 1 the boundary layer turned from
    // sinking to rising once a third of the ice had melted. So the bulk
    // temperature -- liquid nodes more than 20 cells beyond the local front --
    // is watched, and the run stops while it is still within `drift` of T_inf.
    auto bulk_T = [&](const std::vector<double>& t) {
      auto hH = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
      double s = 0; long c = 0;
      for (Index y = 1; y <= H; ++y) {
        const Index x0 = Index(t[std::size_t(y - 1)]) + 21;
        for (Index x = x0; x <= W; ++x) { s += double(pc.temperature_of(hH(d.id(x, y)))); ++c; }
      }
      return c ? s / double(c) : o.Tinf;
    };
    // u_y across the boundary layer at mid-height, at fixed distances from the
    // front: the counterflow regime is a thin rising layer inside a broad
    // sinking one, which a single strip average cannot show.
    const int dist[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48};
    auto bl_profile = [&](const std::vector<double>& t) {
      fl.compute_macroscopic();
      auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      std::printf("  u_y/U_ff across the layer at mid-height, by distance from the ice (cells):\n  ");
      for (int k : dist) std::printf(" %7d", k);
      std::printf("\n  ");
      for (int k : dist) {
        double s = 0; int c = 0;
        for (Index y = 7 * H / 16; y <= 9 * H / 16; ++y) {
          const Index x = Index(std::lround(t[std::size_t(y - 1)])) + k;
          if (x <= W) { s += double(hv(d.id(x, y))); ++c; }
        }
        std::printf(" %+7.4f", c ? s / c / o.u_ff : 0.0);
      }
      std::printf("\n");
    };

    std::vector<double> t0, t;
    thickness(t0);
    double vol0 = 0;
    for (double v : t0) vol0 += v;
    const double td = double(H) * double(H) / kappa;          // diffusive time
    std::printf("  %9s %9s %8s %9s %9s %9s %10s %8s\n", "step", "t/t_diff", "melted",
                "bottom", "middle", "top", "v_BL/U_ff", "T_bulk");
    const auto clock0 = std::chrono::steady_clock::now();
    long step = 0;
    const long cap = long(0.5 * td);                       // a fallback only
    bool finite = true;
    while (step < cap) {
      for (long k = 0; k < o.probe; ++k) { couple(); fl.step(true); th.step(); }
      step += o.probe;
      thickness(t);
      double vol = 0;
      for (double v : t) { vol += v; if (!std::isfinite(v)) finite = false; }
      if (!finite) { std::printf("  NON-FINITE at step %ld\n", step); break; }
      const double melted = 1.0 - vol / vol0;
      auto band = [&](double a, double b) {                 // mean over a height band
        double s = 0; int c = 0;
        for (Index y = Index(a * H); y < Index(b * H); ++y) { s += t[std::size_t(y)]; ++c; }
        return c ? s / c : 0.0;
      };
      const double Tb = bulk_T(t);
      std::printf("  %9ld %9.4f %8.3f %9.2f %9.2f %9.2f %+10.4f %8.3f\n", step,
                  double(step) / td, melted, band(0.0, 0.125), band(0.4375, 0.5625),
                  band(0.875, 1.0), bl_velocity(t), Tb);
      std::fflush(stdout);
      if (melted >= o.melt || band(0.0, 1.0) <= 1.0) break;
      if (std::abs(Tb - o.Tinf) > o.drift) {
        std::printf("  bulk has drifted %.3f C from T_inf: stopping here\n", Tb - o.Tinf);
        break;
      }
    }
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock0).count();

    // Final shape. Taper = bottom-quarter minus top-quarter thickness, in units
    // of the initial thickness; extrema of the (5-row smoothed) profile count
    // the waves.
    if (finite) {
      auto mean = [&](Index a, Index b) {
        double s = 0; for (Index y = a; y < b; ++y) s += t[std::size_t(y)];
        return s / double(b - a);
      };
      const double taper = (mean(0, H / 4) - mean(3 * H / 4, H)) / double(xice);
      std::vector<double> sm(std::size_t(H), 0.0);
      for (Index y = 0; y < H; ++y) {
        double s = 0; int c = 0;
        for (Index k = std::max<Index>(0, y - 2); k <= std::min<Index>(H - 1, y + 2); ++k) {
          s += t[std::size_t(k)]; ++c;
        }
        sm[std::size_t(y)] = s / c;
      }
      int extrema = 0;
      for (Index y = H / 16; y + 1 < H - H / 16; ++y) {
        const double a = sm[std::size_t(y) - 1], b = sm[std::size_t(y)], c = sm[std::size_t(y) + 1];
        if ((b > a && b > c) || (b < a && b < c)) ++extrema;
      }
      std::printf("\n");
      bl_profile(t);
      std::printf("  final profile (thickness in cells, bottom to top, every H/16):\n  ");
      for (Index y = H / 32; y < H; y += H / 16) std::printf(" %6.2f", t[std::size_t(y)]);
      std::printf("\n  taper (bottom - top) / initial = %+.3f   interior extrema %d   "
                  "%.0f s\n", taper, extrema, secs);
      std::printf("  %s\n", taper < -0.05 ? "THINNER BELOW: tapered from below, as for a "
                                           "downward pinnacle (T_inf < ~5 C)"
                          : taper > 0.05 ? "THINNER ABOVE: tapered from above, as for an "
                                           "upright pinnacle (T_inf > ~7 C)"
                                         : "NO NET TAPER");

      if (!o.out.empty()) {
        ::mkdir(o.out.c_str(), 0755);
        using namespace lbm::figdump;
        th.compute_field(); fl.compute_macroscopic();
        auto hH = Kokkos::create_mirror_view_and_copy(HostSpace{}, th.temperature());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
        char tag[32]; std::snprintf(tag, sizeof tag, "T%.1f", o.Tinf);
        const std::string base = o.out + "/" + tag;
        scalar_slice(base + "_temperature.bin", nx, ny, [&](Index x, Index y) {
          return double(pc.temperature_of(hH(d.id(x, y)))); });
        scalar_slice(base + "_liquid.bin", nx, ny, [&](Index x, Index y) {
          return double(pc.liquid_fraction(hH(d.id(x, y)))); });
        scalar_slice(base + "_uy.bin", nx, ny, [&](Index x, Index y) {
          return double(hv(d.id(x, y))); });
        scalar_slice(base + "_ux.bin", nx, ny, [&](Index x, Index y) {
          return double(hu(d.id(x, y))); });
        std::FILE* fp = std::fopen((base + "_profile.txt").c_str(), "w");
        if (fp) {
          for (Index y = 0; y < H; ++y) std::fprintf(fp, "%d %.6f %.6f\n", int(y + 1),
                                                     t0[std::size_t(y)], t[std::size_t(y)]);
          std::fclose(fp);
        }
        std::printf("  fields written under %s_*\n", base.c_str());
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
