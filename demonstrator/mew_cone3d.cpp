//==============================================================================
//  The melt-electrowriting nozzle in THREE dimensions: does the melt hanging
//  from a round nozzle form a Taylor cone? The 3-D companion of mew_nozzle.cpp,
//  whose banner argues the set-up, the lattice scaling and the 2-D findings;
//  this file repeats only what differs.
//
//  WHY 3-D AT ALL. A straight-sided planar wedge has no curvature to balance
//  eps0 E^2 / 2, so a planar model has no conical equilibrium (CLAUDE.md), and a
//  flat 2-D tip feels a weaker field than a round one. A Taylor cone needs the
//  axisymmetric geometry, and this tree has no axisymmetric solver.
//
//  THE MODEL. FreeSurfaceSolver<D3Q27> for the melt (central moments, surface
//  tension; height-function curvature by default, -nohf for the earlier pass),
//  FsElectric<D3Q7> for the field (3-D since 2026-11-05: validation/
//  fs_electric.cpp section C). A round nozzle -- bore radius R, tip face of
//  radius R + lip -- 5.5 mm above a grounded collector moving at 35 mm/s, fed
//  through the bore by a piston at the mean velocity Q / (pi R^2), in a square
//  box whose four sides are zero-field frames. The melt conducts down to -lc
//  below the tip and is an insulator below that, so it cannot short the gap;
//  the default, 5 mm, is the whole drop (Murat Kilic's conductivity estimate
//  relaxes charge in <= 0.03 s; mew_nozzle's banner). The lattice is scaled as
//  mew_nozzle scales it -- Ca of the feed and of the collector, Bo and
//  Lambda_E = eps0 V^2 / (gamma R) matched; Re and Oh are not.
//
//  THE DEFAULTS ARE THE PHYSICAL CASE AS FAR AS IT IS KNOWN: the corrected
//  feed 0.0208 mm^3/s, the measured 820 Pa s, 6.5 kV, the whole drop
//  conducting, a non-wetted nozzle (it reads as gas in the curvature, as in
//  mew_nozzle without -wet), 50 um cells (-R 4, 8 cells across the bore).
//
//  WHAT IT DOES NOT DO.
//   - It is not grid-converged and cannot be here: one 50 um run costs about
//     9 minutes per simulated second on this laptop, and 25 um would cost 32x.
//   - The jet of the experiment, 10-20 um, is far below the cells. Whatever
//     leaves the cone's tip is a thread at least two or three cells wide,
//     set by the grid, not by the physics.
//   - No air breakdown: fields above about 3 kV/mm are reported, not acted on.
//   - The geometric mass exchange that makes the 2-D free drop converge
//     (FreeSurfaceSolver::exchange_geometric) is 2-D only; this runs the
//     earlier exchange, whose ripples pin a relaxing drop harder the more
//     cells it spans. At 50 um the drop spans ten or so.
//   - Until the melt touches the collector the state is projected onto its
//     x-mirror-symmetric part, as in mew_nozzle; there is no y projection.
//
//  MEASURED (2026-11-06: the defaults to -tend 40, FP64, 4 threads, 6.0 h --
//  18,298 s in the fluid, 3,440 s in the field). NO CONE AT ANY FRAME. The
//  melt leaves the nozzle 0.55-0.60 mm wide (the neck column) -- the bore plus
//  four cells, as in 2-D on every grid -- and is drawn into a column that
//  narrows to about 0.45 mm just above a rounded end five rows long;
//  tools/render_cone3d.py draws that end at true scale against Taylor's cone.
//  The end comes down at 0.06 mm/s to 15 s, then 0.10 (20-24 s), 0.15
//  (26-28 s), 0.28 (28-30 s), 0.85 (31-32 s) and 2.4 mm/s over its last
//  0.6 mm (2.7 over the last 0.4), while the strongest field below the tip
//  goes from about 11 kV/mm at 20 s to 16 at 30 s and 35 at 32.4 s. It
//  passes 12 kV/mm at 22-24 s, about Peek's corona onset for a sphere of
//  the end's radius (11-13 kV/mm for 0.3-0.2 mm; an estimate): the run-away
//  happens where the model's air, which never breaks down, is least like air.
//  IT ENDS ON THE CONDUCTOR'S EDGE, AS THE 2-D -lc 5 RUN DID (mew_nozzle's
//  banner). -lc 5 in a 5.5 mm gap leaves the last 0.5 mm insulating, and the
//  end reaches it at 32.39 s (2-D at the same 820 Pa s: 40.18 s; the gap is
//  not attributed), overshoots to 0.33 mm and parks at 0.43-0.48 mm. The melt
//  piles up on that plane into a flat conducting disk, 0.42 mm across at
//  32.4 s and 1.02 mm at 37.4 s, and from about 38 s the disk's rim -- a
//  sharp-edged conductor 0.5 mm above ground -- throws thin sheets (fill ~0.2)
//  along the plane to the box walls, 2.7 mm from the axis by 39.8 s, with the
//  reported field rising to 76 kV/mm at their edges. NOTHING AFTER 32.4 s IS
//  THE MELT; the renderer marks it. 2-D hovered at this edge for 24.5 s before
//  touching; this run stopped at 40 s without contact, so a 3-D contact time
//  is not known. Any -lc short of the gap makes such an edge, and one that
//  reaches the collector shorts the gap: with the field on, contact is outside
//  this model.
//  THE FEED IS 3.45 % HIGH. The piston covers nbore = 52 cells where
//  pi R^2 = 50.27, and Un_l is set from the nominal area, so the melt enters
//  at 0.02159 mm^3/s (the slope of the gap volume over 5-40 s), not 0.0208.
//  The ledger counts nbore and stayed at 2.2e-3 to 4.2e-3. Scaling uin by
//  pi R^2 / nbore would fix it; not applied, so that the run above is
//  reproduced.
//
//  Usage: mew_cone3d [-R 4] [-Ucol 0.04] [-mu 820] [-V 6.5] [-lc 5] [-feed 0.0208]
//                    [-W 4] [-nofield] [-nohf] [-nosym] [-M 20] [-tend 20]
//                    [-tframe 0.5] [-out PREFIX]   (lengths in mm, V in kV)
//==============================================================================
#include "FsElectric.hpp"
#include "NpyDump.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using L  = D3Q27;
using PL = D3Q7;
using FS = FreeSurfaceSolver<L>;

namespace {

constexpr double kPi  = 3.14159265358979323846;
constexpr double kEps0 = 8.8541878128e-12;

struct Physical {
  double R    = 0.2e-3;      // bore radius: the 0.4 mm nozzle
  double lip  = 0.3e-3;      // nozzle wall, so the tip face is 1.0 mm across (NOMINAL)
  double gap  = 5.5e-3;      // nozzle-collector distance, Table 1
  double up   = 1.25e-3;     // nozzle length inside the box above its tip
  double W    = 4.0e-3;      // box width, both ways; the nozzle on its axis
  double mu   = 820.0;       // PCL at 120 C, measured (Murat Kilic, 2026-11-04)
  double gam  = 0.030;       // surface tension [N/m] (NOMINAL)
  double rho  = 1080.0;
  double g    = 9.81;
  double V    = 6.5e3;
  double Q    = 0.0208e-9;   // feed [m^3/s], Table 1 corrected
  double Uc   = 0.035;       // collector, 2100 mm/min
  double lc   = 5.0e-3;      // liquid within this of the tip conducts: the whole drop

  double Un()   const { return Q / (kPi * R * R); }
  double Ca_n() const { return mu * Un() / gam; }
  double Ca_c() const { return mu * Uc / gam; }
  double Bo()   const { return rho * g * R * R / gam; }
  double LamE() const { return kEps0 * V * V / (gam * R); }
  double tvc()  const { return mu * R / gam; }
};

struct Opts {
  int    R      = 4;         // cells across the bore radius
  double nu     = 0.5;       // lattice kinematic viscosity (tau = 2)
  double Ucol   = 0.04;      // collector speed in lattice units
  double tauphi = 10.0;
  int    M      = 20;        // fluid steps per field update
  double t_end  = 20.0;
  double t_out  = 0.05;
  double t_frame = 0.5;
  bool   field  = true;
  bool   hf     = true;
  bool   sym    = true;
  std::string out;
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Physical P;
    Opts o;
    for (int i = 1; i < argc; ++i) {
      auto num = [&]() {
        if (i + 1 >= argc) { std::printf("mew_cone3d: %s needs a value\n", argv[i]); std::exit(2); }
        return std::atof(argv[++i]);
      };
      const char* a = argv[i];
      if      (!std::strcmp(a, "-R"))       o.R = int(num());
      else if (!std::strcmp(a, "-Ucol"))    o.Ucol = num();
      else if (!std::strcmp(a, "-nu"))      o.nu = num();
      else if (!std::strcmp(a, "-M"))       o.M = int(num());
      else if (!std::strcmp(a, "-tend"))    o.t_end = num();
      else if (!std::strcmp(a, "-tframe"))  o.t_frame = num();
      else if (!std::strcmp(a, "-mu"))      P.mu = num();
      else if (!std::strcmp(a, "-V"))       P.V = num() * 1e3;
      else if (!std::strcmp(a, "-lc"))      P.lc = num() * 1e-3;
      else if (!std::strcmp(a, "-W"))       P.W = num() * 1e-3;
      else if (!std::strcmp(a, "-feed"))    P.Q = num() * 1e-9;
      else if (!std::strcmp(a, "-nofield")) o.field = false;
      else if (!std::strcmp(a, "-nohf"))    o.hf = false;
      else if (!std::strcmp(a, "-nosym"))   o.sym = false;
      else if (!std::strcmp(a, "-out") && i + 1 < argc) o.out = argv[++i];
      else {
        std::printf("mew_cone3d: unknown option '%s'. NOTHING WAS RUN.\n", a);
        Kokkos::finalize();
        return 2;
      }
    }

    //---- lattice units: mew_nozzle's scaling --------------------------------------
    const int    R    = o.R;
    const double dx   = P.R / R;
    const double ucap = o.Ucol / P.Ca_c();
    const double mu_l = o.nu;
    const double gam_l = ucap * mu_l;
    const double Un_l = P.Ca_n() * ucap;
    const double g_l  = P.Bo() * gam_l / double(R * R);
    const double C_l  = 0.5 * P.LamE() * gam_l * double(R);
    const double tvc_l = double(R) / ucap;
    const double dt   = P.tvc() / tvc_l;
    const double Re_c = o.Ucol * R / o.nu;
    const double Oh_l = std::sqrt(mu_l * mu_l / (gam_l * R));

    const Index nx   = Index(std::lround(P.W / dx)), ny = nx;
    const Index gapc = Index(std::lround(P.gap / dx));
    const Index ztip = 1 + gapc;                    // first nozzle row; face at ztip - 1/2
    const Index nz   = ztip + Index(std::lround(P.up / dx));
    const double xc  = 0.5 * double(nx);            // the axis, on a cell corner
    const Index lipc = Index(std::lround(P.lip / dx));
    const Index lcc  = Index(std::lround(P.lc / dx));
    const Index tipz = Index(std::lround(1.0e-3 / dx));
    const double Ro  = double(R + lipc);

    std::printf("mew_cone3d: D3Q27 free surface + D3Q7 TRT potential, precision %s\n",
                sizeof(Real) == 4 ? "float" : "double");
    std::printf("  PCL: mu %.0f Pa s  gamma %.3f N/m  rho %.0f   field %s\n", P.mu, P.gam, P.rho,
                o.field ? "ON" : "OFF");
    std::printf("  set-up: round nozzle, bore %.1f mm, tip face %.1f mm, gap %.1f mm, %.1f kV,"
                " collector %.0f mm/s, feed %.4g mm^3/s, box %.1f x %.1f mm\n", 2e3 * P.R,
                2e3 * (P.R + P.lip), P.gap * 1e3, P.V * 1e-3, P.Uc * 1e3, P.Q * 1e9, P.W * 1e3,
                P.W * 1e3);
    std::printf("  groups: Ca_feed %.3g  Ca_coll %.3g  Bo %.3g  Lambda_E %.3g  (U_bore %.4g mm/s,"
                " t_vc %.3g s)\n", P.Ca_n(), P.Ca_c(), P.Bo(), P.LamE(), P.Un() * 1e3, P.tvc());
    std::printf("  lattice: %lld x %lld x %lld, dx %.1f um, dt %.3e s, gamma %.3e, U_bore %.3e,"
                " U_coll %.3g, g %.2e, C_E %.3e\n", (long long)nx, (long long)ny, (long long)nz,
                dx * 1e6, dt, gam_l, Un_l, o.Ucol, g_l, C_l);
    std::printf("           Oh %.1f  Re_coll %.3f  Ma_coll %.3f   conducting %.1f mm below the tip"
                "   curvature %s   M %d\n", Oh_l, Re_c, o.Ucol / std::sqrt(1.0 / 3.0), P.lc * 1e3,
                o.hf ? "height functions" : "pass 4 (-nohf)", o.M);

    //---- geometry ----------------------------------------------------------------
    // radius of a cell's centre from the axis, in cells
    auto rad = [=](Index x, Index y) {
      const double X = double(x) + 0.5 - xc, Y = double(y) + 0.5 - xc;
      return std::sqrt(X * X + Y * Y);
    };
    auto is_lip = [=](Index x, Index y, Index z) {
      const double r = rad(x, y);
      return z >= ztip && r >= double(R) && r < Ro;
    };
    // The piston spans the bore and the lip's top row, as in 2-D: over the bore
    // alone each edge cell drew its diagonal feed links from a static lip cell.
    auto is_piston = [=](Index x, Index y, Index z) { return z == nz - 1 && rad(x, y) < Ro; };
    long nbore = 0;
    for (Index y = 0; y < ny; ++y)
      for (Index x = 0; x < nx; ++x) nbore += rad(x, y) < double(R) ? 1 : 0;

    Domain d(nx, ny, nz, false, false, false);
    FS fs(d);
    fs.coll.omega = FS::omega_from_viscosity(Real(o.nu));
    fs.set_gravity(Real(0), Real(0), Real(-g_l));
    fs.set_surface_tension(Real(gam_l));
    fs.set_height_function(o.hf);
    fs.set_geometry([&](Index x, Index y, Index z) -> FsCell {
      if (z == 0 || is_lip(x, y, z) || is_piston(x, y, z)) return FsSolid;
      return FsGas;
    });
    const Domain dd = d;
    const Index hx = d.hx, hy = d.hy, hz = d.hz;
    const Real xcr = Real(xc), Rr = Real(R);
    fs.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Real X = Real(px - hx) + Real(0.5) - xcr, Y = Real(py - hy) + Real(0.5) - xcr;
      const Index z = pz - hz;
      const bool bore = z >= ztip && z < nz - 1 && X * X + Y * Y < Rr * Rr;
      return typename FS::Seed{bore ? Real(1) : Real(0), Real(1)};
    });
    const Real ucol = Real(o.Ucol), uin = Real(Un_l), Ror = Real(Ro);
    fs.move_obstacle(
        KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; dd.coords(n, px, py, pz);
          const Real X = Real(px - hx) + Real(0.5) - xcr, Y = Real(py - hy) + Real(0.5) - xcr;
          const Index z = pz - hz;
          return z == 0 || (z == nz - 1 && X * X + Y * Y < Ror * Ror);
        },
        KOKKOS_LAMBDA(Index n, Real u[3]) {
          Index px, py, pz; dd.coords(n, px, py, pz);
          const Index z = pz - hz;
          u[0] = (z == 0) ? ucol : Real(0);
          u[1] = Real(0);
          u[2] = (z == 0) ? Real(0) : -uin;
        },
        Real(0), Real(0));

    FsElectric<PL> el(d, Real(o.tauphi));
    el.coef = o.field ? Real(C_l) : Real(0);
    el.set_roles(
        [&](Index x, Index y, Index z) -> ElecRole {
          if (z == 0 || is_lip(x, y, z) || is_piston(x, y, z)) return ElecElectrode;
          if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == nz - 1) return ElecFrame;
          return ElecFree;
        },
        [&](Index, Index, Index z) -> Real { return z == 0 ? Real(0) : Real(1); });
    el.set_conduct_region([&](Index, Index, Index z) { return z >= ztip - lcc; });
    el.refresh(fs);
    el.initialize(Real(0.5));
    el.relax(20000);
    el.apply(fs);

    //---- diagnostics ---------------------------------------------------------------
    const double v3 = dx * dx * dx * 1e9;          // cell volume in mm^3
    const double m0 = double(fs.total_mass());
    struct Diag {
      double t, total, tip, gap, zlow, neck, cx, cy, drained, ledger, pe, emax;
    };
    auto measure = [&](double t) {
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      Diag D{};
      D.t = t;
      Index zlow = nz;
      double neck = 0, sx = 0, sy = 0, sg = 0;
      for (Index z = 1; z < nz - 1; ++z)
        for (Index y = 0; y < ny; ++y)
          for (Index x = 0; x < nx; ++x) {
            const std::uint8_t f = hf(d.id(x, y, z));
            const double v = (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(d.id(x, y, z))) : 0.0);
            if (v <= 0) continue;
            D.total += v;
            if (z >= ztip) continue;
            D.gap += v;
            sx += v * (double(x) + 0.5 - xc); sy += v * (double(y) + 0.5 - xc); sg += v;
            if (z >= ztip - tipz) D.tip += v;
            if (rad(x, y) <= 3.0 * Ro && z < zlow) zlow = z;
            if (z == ztip - 1) neck += v;
          }
      D.zlow = (zlow < nz) ? (double(zlow) - 0.5) * dx * 1e3 : P.gap * 1e3;
      D.neck = 2.0 * std::sqrt(neck / kPi) * dx * 1e3;          // equivalent diameter, mm
      D.cx = sg > 0 ? sx / sg * dx * 1e3 : 0.0;
      D.cy = sg > 0 ? sy / sg * dx * 1e3 : 0.0;
      const double drained = double(fs.drained_mass());
      const double injected = double(uin) * double(nbore) * (t / dt);
      D.drained = drained * v3;
      D.ledger = (double(fs.total_mass()) + drained - m0 - injected) / std::max(1.0, injected);
      D.total *= v3; D.tip *= v3; D.gap *= v3;
      D.pe = el.max_stress() / (gam_l / R);
      double e2m = 0;
      for (Index z = 1; z < ztip - 1; ++z)
        for (Index y = 1; y < ny - 1; ++y)
          for (Index x = 1; x < nx - 1; ++x)
            if (el.cell(x, y, z) == ScalarBulk) e2m = std::max(e2m, double(el.e2(x, y, z)));
      D.emax = std::sqrt(e2m) * P.V / dx * 1e-6;                // kV/mm
      return D;
    };
    auto frame = [&](double t) {
      if (o.out.empty()) return;
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      const std::size_t N = std::size_t(nx * ny * nz);
      std::vector<float> fl(N), ph(N);
      for (Index z = 0; z < nz; ++z)
        for (Index y = 0; y < ny; ++y)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y, z);
            const std::uint8_t f = hf(n);
            const std::size_t k = std::size_t((z * ny + y) * nx + x);
            fl[k] = (f == FsSolid) ? -1.0f : (f == FsFluid ? 1.0f : (f == FsInterface ? float(he(n)) : 0.0f));
            ph[k] = float(el.phi(x, y, z));
          }
      char nm[64];
      std::snprintf(nm, sizeof nm, "_t%06.2f", t);
      const std::vector<std::size_t> shape{std::size_t(nz), std::size_t(ny), std::size_t(nx)};
      write_npy(o.out + nm + "_fill.npy", fl, shape);
      write_npy(o.out + nm + "_phi.npy", ph, shape);
    };

    FILE* fs_out = nullptr;
    if (!o.out.empty()) {
      fs_out = std::fopen((o.out + "_series.csv").c_str(), "w");
      if (fs_out) std::fprintf(fs_out, "t_s,melt_total_mm3,tip_zone_mm3,gap_mm3,z_low_mm,neck_mm,"
                               "cx_mm,cy_mm,drained_mm3,ledger,max_pe_over_gamma_R,max_E_kV_per_mm\n");
    }
    std::printf("\n  %8s %10s %10s %10s %7s %6s %7s %7s %9s %8s %7s %s\n", "t [s]", "melt mm^3",
                "tip mm^3", "gap mm^3", "z_low", "neck", "cx", "cy", "ledger", "pE/g/R",
                "E kV/mm", "wall s/s");

    const long nsteps = long(o.t_end / dt);
    const long sout = std::max<long>(o.M, long(o.t_out / dt) / o.M * o.M);
    const long sframe = std::max<long>(sout, long(o.t_frame / dt) / sout * sout);
    frame(0.0);
    bool contacted = false;
    double t_fluid = 0, t_field = 0;
    auto wall0 = std::chrono::steady_clock::now();
    double last_wall = 0, last_t = 0;
    for (long t = 0; t < nsteps; t += o.M) {
      auto c0 = std::chrono::steady_clock::now();
      for (int j = 0; j < o.M; ++j) fs.step();
      Kokkos::fence();
      auto c1 = std::chrono::steady_clock::now();
      // Melt the collector carries to a side wall leaves the box there, on the
      // collector only (mew_nozzle's banner says why).
      const Index zdr = Index(std::lround(0.25e-3 / dx));
      fs.drain(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; dd.coords(n, px, py, pz);
        const Index x = px - hx, y = py - hy, z = pz - hz;
        return (x < 3 || x >= nx - 3 || y < 3 || y >= ny - 3) && z <= zdr;
      });
      if (!contacted) {
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
        for (Index y = 0; y < ny && !contacted; ++y)
          for (Index x = 0; x < nx && !contacted; ++x) {
            const std::uint8_t f = hf(d.id(x, y, 1));
            contacted = (f == FsFluid || f == FsInterface);
          }
        if (contacted)
          std::printf("  first contact with the collector at t = %.3f s%s\n", (t + o.M) * dt,
                      o.sym ? ": mirror projection off" : "");
        else if (o.sym)
          fs.mirror_symmetrize_x(nx - 1);
      }
      auto c2 = std::chrono::steady_clock::now();
      if (o.field) {
        el.refresh(fs);
        el.relax(o.M);
        el.apply(fs);
      }
      auto c3 = std::chrono::steady_clock::now();
      t_fluid += std::chrono::duration<double>(c1 - c0).count();
      t_field += std::chrono::duration<double>(c3 - c2).count();
      const long ts = t + o.M;
      if (ts % sout == 0) {
        const Diag D = measure(ts * dt);
        if (!std::isfinite(D.total)) { std::printf("  NOT FINITE at t = %.3f s\n", ts * dt); rc = 1; break; }
        if (fs_out) std::fprintf(fs_out, "%.5f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f,%.5f,%.6f,%.3e,%.4f,%.4f\n",
                                 D.t, D.total, D.tip, D.gap, D.zlow, D.neck, D.cx, D.cy, D.drained,
                                 D.ledger, D.pe, D.emax);
        if (ts % sframe == 0) {
          const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
          const double rate = (wall - last_wall) / std::max(1e-9, D.t - last_t);
          last_wall = wall; last_t = D.t;
          std::printf("  %8.3f %10.5f %10.5f %10.5f %7.3f %6.3f %7.4f %7.4f %9.1e %8.3f %7.2f %6.0f\n",
                      D.t, D.total, D.tip, D.gap, D.zlow, D.neck, D.cx, D.cy, D.ledger, D.pe, D.emax,
                      rate);
          std::fflush(stdout);
          if (fs_out) std::fflush(fs_out);
          frame(ts * dt);
        }
      }
    }
    std::printf("  time in the fluid %.0f s, in the field %.0f s\n", t_fluid, t_field);
    if (fs_out) std::fclose(fs_out);
  }
  Kokkos::finalize();
  return rc;
}
