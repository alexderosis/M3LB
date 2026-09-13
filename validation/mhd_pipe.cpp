//==============================================================================
//  MHD flow in a CIRCULAR pipe with a transverse field -- a curved magnetic
//  wall, against a verified reference.
//
//  Gold's problem (R. R. Gold, J. Fluid Mech. 13, 505, 1962): fully developed
//  flow along x in a pipe of radius R, uniform field B0 along y, insulating
//  wall. Scaling lengths by R, u by F R^2/nu and b by that times sqrt(nu/eta),
//  with Ha = B0 R / sqrt(nu eta), the pair reduces to
//
//      lap u + Ha db/dy = -1 ,   lap b + Ha du/dy = 0 ,   u = b = 0 at r = 1.
//
//  WHY A CURVED WALL IS A DIFFERENT TEST, not a harder version of the duct.
//  The Hartmann layer thickness goes as 1/(Ha |cos psi|) with psi the local
//  angle between B and the wall normal, so around a circle the layer
//  CONTINUOUSLY thins and thickens, and at the two points where B runs tangent
//  to the wall it degenerates altogether and is replaced by a much thicker
//  layer of order 1/sqrt(Ha). validation/shercliff.cpp has those two regimes at
//  two SEPARATE wall pairs; here they vary continuously around one boundary,
//  and nothing flat can show that.
//
//  THE REFERENCE IS A POLAR FINITE-DIFFERENCE SOLVE, AND IT IS VERIFIED TWICE.
//  Gold's series is not transcribed here -- this tree has been bitten by
//  "a wrong constant is still a consistent simulation" -- so the reduced 2-D
//  system above was solved directly on a POLAR grid, where the circular
//  boundary is exact rather than staircased, and checked:
//
//    * at Ha = 0 it returns u(centre) = 0.2500000 against the analytic
//      Poiseuille (1-r^2)/4, and a flow rate of 0.1250000 against 1/8, to
//      every digit printed;
//    * for Ha > 0 it self-converges at second order, flow-rate error ratios
//      3.97 / 3.95 / 3.93 / 3.88 at Ha = 0 / 5 / 10 / 20 over 40x80 ->
//      80x160 -> 160x320.
//
//  The constants below are Richardson extrapolations of that ladder and are
//  good to about 1e-6 in u(centre).
//
//  WHAT IS BEING MEASURED, AND WHAT IS NOT. A lattice circle is a staircase,
//  and validation/stokes_disc.cpp already measured what that costs for a plain
//  no-slip wall: the staircase behaves like a perfect circle of radius
//  R - 0.096 cells, constant to 4 % over a 4x resolution range. So a finite
//  offset here is EXPECTED and is the thing being quantified; what would be a
//  defect is an offset that fails to shrink as 1/N, or one that grows with Ha.
//
//  BOTH FAMILIES ARE ON-NODE, deliberately. The magnetic Dirichlet condition
//  (Dellar) puts B on the grid point, so pairing it with halfway bounce-back
//  for the fluid would place the two boundaries half a cell apart -- which
//  validation/hartmann3d.cpp warns corrupts Ha itself. On a curved wall that
//  mismatch would vary around the circumference. So the fluid gets regularised
//  walls on the same nodes, with the normal derived per node from the geometry
//  and NrmCorner wherever more than one axis leaves the pipe.
//==============================================================================
#include "Campaign.hpp"
#include "collision/MhdBGK.hpp"
#include "solver/MagneticSolver.hpp"

using namespace lbm;
using namespace campaign;

using FL = D3Q27;
using ML = D3Q7;

struct Ref { double Ha, uc, Q; };
// Polar FD, Richardson-extrapolated; see the banner.
static const Ref REF[] = {
  {  0.0, 0.2500000, 0.1250000 },
  {  5.0, 0.1529994, 0.0896204 },
  { 10.0, 0.0893383, 0.0601036 },
  { 20.0, 0.0474300, 0.0353467 },
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::vector<Index> Ns = {41, 81};
    std::vector<double> Has = {0.0, 5.0, 10.0, 20.0};
    double nu_in = 0.05, prm = 1.0, umax_t = 0.02, tol = 1e-8, rfac = 0.45;
    std::size_t cap = 400000, probe = 500;
    std::string dumpdir;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-nu"   && i + 1 < argc) nu_in = std::atof(argv[++i]);
      if (a == "-prm"  && i + 1 < argc) prm = std::atof(argv[++i]);
      if (a == "-umax" && i + 1 < argc) umax_t = std::atof(argv[++i]);
      if (a == "-tol"  && i + 1 < argc) tol = std::atof(argv[++i]);
      if (a == "-rfac" && i + 1 < argc) rfac = std::atof(argv[++i]);
      if (a == "-cap"  && i + 1 < argc) cap = std::size_t(std::atol(argv[++i]));
      if (a == "-dump" && i + 1 < argc) dumpdir = argv[++i];
      if (a == "-ns"   && i + 1 < argc) {
        Ns.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { if (!t.empty()) Ns.push_back(Index(std::atol(t.c_str()))); t.clear(); } else t += c; }
      }
      if (a == "-has"  && i + 1 < argc) {
        Has.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { if (!t.empty()) Has.push_back(std::atof(t.c_str())); t.clear(); } else t += c; }
      }
    }
    std::printf("MHD circular pipe -- curved insulating wall, vs a polar FD reference   %s\n",
                ExecSpace::name());
    std::printf("  nu %.4f  Pr_m %.2f  target u_max %.4f\n\n", nu_in, prm, umax_t);
    std::printf("  %5s %7s %7s %8s %11s %9s %11s %9s %9s\n",
                "N", "R", "Ha", "d_Ha", "u_c/ref", "err %", "Q/ref", "err %", "steps");
    std::printf("  %s\n", std::string(88, '-').c_str());

    for (const Index N : Ns) {
      for (const double Ha : Has) {
        // A REFERENCE IS OPTIONAL. The tabulated values exist only at the four
        // Hartmann numbers the polar solve was run at, and the first version of
        // this SKIPPED any other -- correct for validation and useless for a
        // sweep, which silently dropped ten of twelve points of a figure run.
        // Without one the row still runs and still dumps; it just reports the
        // raw scaling instead of an error.
        const Ref* rf = nullptr;
        for (const Ref& q : REF) if (std::abs(q.Ha - Ha) < 1e-9) rf = &q;
        const Ref fallback{Ha, 0.25, 0.125};   // the Ha = 0 scaling, for the drive only
        const Ref* sc = rf ? rf : &fallback;

        const Index nx = 4;
        const double R = rfac * double(N);
        const double cy = 0.5 * double(N) - 0.5, cz = 0.5 * double(N) - 0.5;
        auto rad = [&](Index y, Index z) {
          const double dy = double(y) - cy, dz = double(z) - cz;
          return std::sqrt(dy * dy + dz * dz);
        };
        auto solid = [&](Index y, Index z) { return rad(y, z) > R; };
        const Real nu = Real(nu_in), eta = Real(nu_in / prm);
        const Real B0 = Real(Ha * std::sqrt(double(nu) * double(eta)) / R);
        // u = (F R^2/nu) u_hat, and u_hat peaks at rf->uc on the axis.
        const double Uref = umax_t / std::max(sc->uc, 1e-30);
        const Real F = Real(Uref * double(nu) / (R * R));

        using FluidColl = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations, Guo>;
        Domain d(nx, N, N, /*x*/ true, /*y*/ false, /*z*/ false);
        MagneticBGK<ML> mc; mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
        MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
        FluidColl fc;
        fc.omega = FluidColl::omega_from_viscosity(nu);
        fc.forcing = Guo{F, Real(0), Real(0)};
        fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
        FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

        // A wall node is a fluid node with at least one axis neighbour outside.
        auto wallcode = [&](Index y, Index z) -> std::uint8_t {
          if (solid(y, z)) return NrmNone;
          int n = 0; std::uint8_t only = NrmNone;
          if (solid(y - 1, z)) { ++n; only = NrmYm; }
          if (solid(y + 1, z)) { ++n; only = NrmYp; }
          if (solid(y, z - 1)) { ++n; only = NrmZm; }
          if (solid(y, z + 1)) { ++n; only = NrmZp; }
          if (n == 0) return NrmNone;
          // More than one axis leaving is a staircase corner; a single normal
          // cannot describe the unknown set there and NrmCorner builds it
          // geometrically, exactly as demonstrator/aorta.cpp does for its caps.
          return (n == 1) ? only : std::uint8_t(NrmCorner);
        };
        fl.set_geometry([&](Index, Index y, Index z) -> CellType {
          return solid(y, z) ? Solid : Fluid;
        });
        using WS = typename decltype(fl)::WallSpec;
        fl.set_regularized_walls([&](Index, Index y, Index z) -> WS {
          const std::uint8_t c = wallcode(y, z);
          if (c == NrmNone) return WS{};
          return WS{c, Real(0), Real(0), Real(0)};
        });
        // The FD corner stress knows nothing about the Lorentz force; see
        // validation/shercliff.cpp, where it leaked mass linearly at a duct
        // edge. A staircase circle is nothing but corners.
        fl.set_fd_corners(false);

        const Real B0c = B0;
        mag.set_geometry([&](Index, Index y, Index z) { return solid(y, z); });
        using WB = typename decltype(mag)::WallB;
        mag.set_moment_walls([&](Index, Index y, Index z) -> WB {
          if (wallcode(y, z) == NrmNone) return WB{};
          return WB{true, Real(0), B0c, Real(0)};    // insulating: induced b = 0
        });
        mag.initialize_field(KOKKOS_LAMBDA(Index) {
          Kokkos::Array<Real, 3> b; b[0] = Real(0); b[1] = B0c; b[2] = Real(0);
          return b;
        });
        mag.set_velocity(fl.ux(), fl.uy(), fl.uz());
        fl.initialize(Real(1));

        std::vector<double> prevf(std::size_t(N) * std::size_t(N), 0.0);
        std::size_t taken = 0; double res = 1; bool ok = true;
        for (std::size_t t = 0; t <= cap; ++t) {
          if (t % probe == 0) {
            fl.compute_macroscopic(); mag.compute_field();
            auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
            double num = 0, den = 0;
            for (Index z = 0; z < N; ++z)
              for (Index y = 0; y < N; ++y) {
                if (solid(y, z)) continue;
                const double v = double(hx(d.id(0, y, z)));
                if (!std::isfinite(v)) { ok = false; }
                const std::size_t k = std::size_t(z) * N + y;
                num += (v - prevf[k]) * (v - prevf[k]); den += v * v;
                prevf[k] = v;
              }
            if (!ok) break;
            res = std::sqrt(num / std::max(den, 1e-300));
            taken = t;
            if (t > 0 && res < tol) break;
          }
          if (t < cap) { mag.compute_field(); fl.step(true); mag.step(true); }
        }

        fl.compute_macroscopic();
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
        double sum = 0; std::size_t cnt = 0;
        for (Index z = 0; z < N; ++z)
          for (Index y = 0; y < N; ++y) {
            if (solid(y, z)) continue;
            sum += double(hx(d.id(0, y, z))); ++cnt;
          }
        const double uc = double(hx(d.id(0, Index(std::lround(cy)), Index(std::lround(cz)))));
        const double Qm = cnt ? sum / double(cnt) : 0.0;
        // GUARDED. Dereferencing rf unconditionally here is what segfaulted
        // the sweep: a Hartmann number outside the reference table leaves it
        // null, and the crash produced NO output at all because stdout is
        // block-buffered to a file and the buffer dies with the process. Exit
        // code 139 was the only evidence, which is why "the run stopped early"
        // read as a parsing problem for two rounds.
        const double ucr = rf ? uc / (Uref * rf->uc) : 0.0;
        const double Qr  = rf ? Qm / (Uref * rf->Q)  : 0.0;
        // The cross-section, for a figure: u and b on the (y,z) plane as raw
        // doubles behind a small header. A Ha sweep of these is the only way to
        // SEE what a curved magnetic wall does -- the Hartmann layer thinning
        // where B meets the wall head-on and degenerating into a much thicker
        // layer at the two points where B runs tangent to it.
        if (!dumpdir.empty()) {
          auto hb = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
          char nm[512];
          std::snprintf(nm, sizeof nm, "%s/pipe_n%03d_ha%03d.bin",
                        dumpdir.c_str(), int(N), int(std::lround(Ha)));
          if (std::FILE* fp = std::fopen(nm, "wb")) {
            const std::int32_t hdr[2] = {std::int32_t(N), std::int32_t(N)};
            const double meta[3] = {R, Ha, Uref};
            std::fwrite(hdr, sizeof hdr, 1, fp);
            std::fwrite(meta, sizeof meta, 1, fp);
            for (Index z = 0; z < N; ++z)
              for (Index y = 0; y < N; ++y) {
                const double v = solid(y, z) ? std::nan("") : double(hx(d.id(0, y, z)));
                std::fwrite(&v, sizeof v, 1, fp);
              }
            for (Index z = 0; z < N; ++z)
              for (Index y = 0; y < N; ++y) {
                const double v = solid(y, z) ? std::nan("")
                                             : double(hb(d.id(0, y, z)));
                std::fwrite(&v, sizeof v, 1, fp);
              }
            std::fclose(fp);
          }
        }
        if (rf)
          std::printf("  %5d %7.2f %7.1f %8.2f %11.5f %+8.2f %11.5f %+8.2f %9zu%s\n",
                      int(N), R, Ha, Ha > 0 ? R / Ha : 999.0,
                      ucr, 100.0 * (ucr - 1.0), Qr, 100.0 * (Qr - 1.0), taken,
                      ok ? "" : "  NON-FINITE");
        else
          std::printf("  %5d %7.2f %7.1f %8.2f %11.5f %8s %11.5f %8s %9zu%s\n",
                      int(N), R, Ha, Ha > 0 ? R / Ha : 999.0,
                      uc / Uref, "no ref", Qm / Uref, "no ref", taken,
                      ok ? "" : "  NON-FINITE");
        std::fflush(stdout);
        if (!ok) status = 1;
      }
      std::printf("\n");
    }
  }
  Kokkos::finalize();
  return status;
}
