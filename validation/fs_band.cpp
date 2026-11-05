//==============================================================================
// fs_band -- a straight band of liquid carried through a periodic box by a
// uniform body force, against its exact rigid motion.
//
// WHY THIS CASE EXISTS. validation/surface_tension -rate measured a free drop
// that relaxes too slowly, and the slowing grows with the drop's size in cells;
// FreeSurfaceSolver::height_fit showed the cause is cell-scale ripples in the
// fill, read as curvature noise. A drop cannot say where those ripples come
// from, because its fill is moved by the free surface's bookkeeping, its
// curvature and its pressure at once. This removes all but the bookkeeping.
//
// THE CASE. A band of liquid between two parallel lines of slope P/Q (-slope,
// 3/7 by default), in a periodic Qm x Pm box so that the band wraps onto
// itself, seeded with exact cell areas, at rest, and then a constant body force
// g along the band (-along, the default) or across it (-across). In a periodic
// box nothing holds the band, so the exact answer is free fall: uniform
// velocity g t, uniform pressure, and a band that keeps its shape exactly. Its
// vertical thickness T is then the same in every column, so the column sums of
// the fill must all equal T at every step. Their scatter about their mean is
// the ripple, and nothing physical can make one. Along the band the surface
// does not move at all -- the liquid slides along itself -- so every cell's fill
// should stay what it was. -sigma S [-hf [-hfit N]] adds surface tension, which
// a flat band does not feel (kappa = 0) unless it ripples, and -gstop N turns
// the force off so the band then slides steadily.
//
// MEASURED 2026-11-05 (FP64, nu = 1/6, slope 3/7, m = 16 unless said).
//  THE RIPPLES ARE THE MASS EXCHANGE. With no surface tension, sliding ALONG
//  itself the band's column sums scatter by 0.0106 cells after it has moved
//  0.1 cell and 0.042 after 0.4, then between 0.08 and 0.26 from 1.6 to 10
//  cells, with 72-93 % of it at wavelengths of four cells or less at first;
//  across it, 0.019 and 0.069. It is set by the distance slid in cells and not by the box: m = 8 / 16
//  / 32 give 0.0119 / 0.0106 / 0.0105 at 0.1 cell. And it needs uneven steps:
//  at slope 0 the scatter is exactly zero and at slope 1 it is 4e-15, where
//  every surface cell has the same neighbourhood; at slope 1/2, where steps and
//  flats alternate, it is a pure two-cell zigzag (0.16 by 1.6 cells). -predict
//  says why, cell by cell. With equilibrium populations the net flux on link i
//  is -2 w_i rho (c_i . u)/cs^2, and mass_exchange() weights it by 1 to a Fluid
//  cell, by the mean fill to an Interface cell and by 0 to Gas -- which sums to
//  zero for a surface cell only where its neighbourhood is symmetric about the
//  flow. Summed over the links and integrated over the slide, that formula,
//  from the initial state alone, predicts every interface cell's fill change
//  after 0.1 cell: correlation 0.9999 and 0.9998 with the solver's, measured
//  over predicted 1.0004 and 0.9994, at slopes 1/2 and 3/7 (192 and 320 cells).
//  The formula accounts for all of it, and no cell converts in that interval.
//  NOT THE CHOICE OF FILL: weighting a link by the upwind cell's fill instead of
//  the mean (FreeSurfaceSolver::exchange_upwind) makes it worse -- 0.36 against
//  0.19 after 10 cells along, and across the surface thickens from 320 to 1376
//  interface cells.
//  WHAT A DROP SEES. With the surface tension of validation/surface_tension's
//  R = 32 drop at Oh = 3 (9.645e-5) and that drop's surface-speed scale,
//  lambda eps R = 3e-5, the 3-column heights hold the column sums at 0.8-2.3e-3
//  and the curvature at 3e-4 to 1.1e-3 per cell, where a flat band has none --
//  the same order as that drop's own: 6e-4 of a cell of ripple, and 1.4e-3 per
//  cell of curvature scatter about the mode-2 fit (-rate's resid*R, 0.044 at
//  R = 32). Smoothed over 7 columns the curvature falls to
//  3-4e-4 but the fill keeps roughening, to 0.028 by 2.85 cells, because the
//  zigzag is no longer restored; pass 4 reads
//  the flat 3/7 band with 0.0135 of curvature even at rest. At rest with
//  surface tension the band holds to 1e-12. With the surface tension and the
//  speed both halved the noise starts at the same level and then DECAYS, to
//  7e-5 by 2.9 cells, so this case does not settle how the noise scales with a
//  drop's size.
//
//  THE FIX (FreeSurfaceSolver::exchange_geometric, -geom, with -donor and
//  -follow; 2026-11-05). Routing every diagonal link through faces and
//  weighting each face by how much of it the reconstructed line wets makes the
//  exchange exact for a planar surface; the two refinements switch cells on
//  where that line reaches and weight a face from the side the flux leaves.
//  With all three, over 10 cells, the column sums scatter by 1.4-3.8e-6 along
//  the band and 5e-6 to 9e-5 across it at slopes 1/2 and 3/7, and mass holds to
//  1.5e-15 along and 1.8e-10 across, where cells convert (m = 8, 1.6 cells:
//  tests/test_free_surface section 9 gates both). -geom alone fixes along
//  (1.8-3.5e-5) but not across (0.02-0.07, as the default).
//
// OUTPUT, every -every steps: the exact velocity and displacement, the mass
// drift, the number of interface cells, the column sums' scatter -- rms and
// largest -- with the share of its energy at wavelengths of four cells or less,
// and with -sigma the rms curvature over interface cells. A measurement, not a
// gate.
//==============================================================================

#include "core/Types.hpp"
#include "solver/FreeSurfaceSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L>;

namespace {

// Area of the cell [x0,x1] x [y0,y1] below the line y = s x + c. The integrand
// clamp(s x + c - y0, 0, y1 - y0) is linear between the points where the line
// crosses y0 and y1, so the trapezoid rule over those breakpoints is exact.
double below_line(double s, double c, double x0, double x1, double y0, double y1) {
  auto g = [&](double x) { return std::min(y1 - y0, std::max(0.0, s * x + c - y0)); };
  std::vector<double> br = {x0, x1};
  if (s != 0.0)
    for (double yb : {y0, y1}) {
      const double xb = (yb - c) / s;
      if (xb > x0 && xb < x1) br.push_back(xb);
    }
  std::sort(br.begin(), br.end());
  double a = 0.0;
  for (std::size_t i = 0; i + 1 < br.size(); ++i)
    a += 0.5 * (g(br[i]) + g(br[i + 1])) * (br[i + 1] - br[i]);
  return a;
}

// The band: liquid where y - s x - c1, taken modulo Ny, lies in (0, T). Its
// copies a period apart are disjoint, so the cell's fill is the sum over them.
double band_fill(double s, double c1, double T, double Ny, double x0, double x1,
                 double y0, double y1) {
  double a = 0.0;
  for (int k = -2; k <= 2; ++k)
    a += below_line(s, c1 + T + k * Ny, x0, x1, y0, y1) - below_line(s, c1 + k * Ny, x0, x1, y0, y1);
  return a;
}

struct Opt {
  int    m = 16;            // the box is Q m x P m (P = 0: Q m x 3m)
  int    P = 3, Q = 7;      // slope P/Q
  double g = 5e-8;          // body force, lattice units
  bool   across = false;    // force across the band rather than along it
  long   steps = 20000, every = 1000;
  double fo = -1.0;         // fill_offset; < 0 keeps the solver's default
  bool   nrec = false;      // normal_reconstruction
  bool   upwind = false;    // exchange_upwind
  bool   geom = false;      // set_exchange_geometric
  bool   donor = false;     // exchange_donor, with -geom
  bool   follow = false;    // surface_follows_line, with -geom
  double sigma = 0.0;       // surface tension; 0 = none (the default)
  bool   hf = false;        // height_function, with sigma
  int    hfit = 3;          // height_fit, with -hf
  long   gstop = 0;         // > 0: the force is switched off at this step
  bool   predict = false;   // compare each cell's fill change with the exchange formula
  const char* dump = nullptr;
};

struct Ripple { double rms = 0, max = 0, hik = 0, mean = 0; };

// The column sums' scatter about their mean, and the share of its energy at
// wavenumbers k >= pi/2 (wavelengths of four cells or less), by a direct DFT.
Ripple column_ripple(const std::vector<double>& S, double T) {
  Ripple r;
  const std::size_t N = S.size();
  for (double v : S) r.mean += v - T;
  r.mean /= double(N);
  std::vector<double> dv(N);
  for (std::size_t x = 0; x < N; ++x) {
    dv[x] = S[x] - T - r.mean;
    r.rms += dv[x] * dv[x];
    r.max = std::max(r.max, std::fabs(dv[x]));
  }
  r.rms = std::sqrt(r.rms / double(N));
  double all = 0.0, hi = 0.0;
  for (std::size_t k = 1; k <= N / 2; ++k) {
    double re = 0.0, im = 0.0;
    for (std::size_t x = 0; x < N; ++x) {
      const double ph = 2.0 * M_PI * double(k * x) / double(N);
      re += dv[x] * std::cos(ph);
      im -= dv[x] * std::sin(ph);
    }
    const double p = re * re + im * im;
    all += p;
    if (2.0 * M_PI * double(k) / double(N) >= 0.5 * M_PI - 1e-12) hi += p;
  }
  r.hik = all > 0.0 ? hi / all : 0.0;
  return r;
}

int run(const Opt& o) {
  const Index Nx = Index(o.Q * o.m), Ny = Index((o.P > 0 ? o.P : 3) * o.m);
  const double s = double(o.P) / double(o.Q);
  const double c1 = 0.2137 * double(Ny), T = 0.5 * double(Ny) + 0.31;
  const double nu = 1.0 / 6.0;
  const double tl = std::sqrt(double(o.P * o.P + o.Q * o.Q));
  const double dir[2] = {o.across ? -double(o.P) / tl : double(o.Q) / tl,
                         o.across ? double(o.Q) / tl : double(o.P) / tl};

  Domain d(Nx, Ny, 1, true, true, true);
  FS fs(d);
  fs.coll.omega = FS::omega_from_viscosity(Real(nu));
  fs.set_normal_reconstruction(o.nrec);
  fs.exchange_upwind = o.upwind;
  fs.set_exchange_geometric(o.geom);
  fs.exchange_donor = o.donor;
  fs.surface_follows_line = o.follow;
  if (o.fo >= 0.0) fs.fill_offset = Real(o.fo);
  fs.set_geometry([&](Index, Index, Index) -> FsCell { return FsGas; });

  std::vector<Real> fill(d.n_padded, Real(0));
  for (Index y = 0; y < Ny; ++y)
    for (Index x = 0; x < Nx; ++x) {
      const double a = band_fill(s, c1, T, double(Ny), double(x) - 0.5, double(x) + 0.5,
                                 double(y) - 0.5, double(y) + 0.5);
      fill[d.id(x, y)] = Real(a > 1.0 - 1e-9 ? 1.0 : (a < 1e-9 ? 0.0 : a));
    }
  Kokkos::View<Real*> fv("fill", d.n_padded);
  {
    auto h = Kokkos::create_mirror_view(fv);
    for (Index n = 0; n < d.n_padded; ++n) h(n) = fill[n];
    Kokkos::deep_copy(fv, h);
  }
  fs.initialize(KOKKOS_LAMBDA(Index n) { return FS::Seed{fv(n), Real(1)}; });
  fs.set_gravity(Real(o.g * dir[0]), Real(o.g * dir[1]));
  if (o.sigma > 0.0) {
    fs.set_surface_tension(Real(o.sigma));
    fs.set_height_function(o.hf);
    fs.set_height_fit(o.hfit);
  }
  // The exact motion: free fall at g until gstop, then a steady slide.
  auto u_at = [&](long t) { return o.g * double(o.gstop > 0 ? std::min(t, o.gstop) : t); };
  auto x_at = [&](long t) {
    const double ts = double(o.gstop > 0 ? std::min(t, o.gstop) : t);
    return 0.5 * o.g * ts * ts + (o.gstop > 0 && t > o.gstop ? o.g * double(o.gstop) * double(t - o.gstop) : 0.0);
  };

  std::printf("a liquid band in free fall: slope %d/%d, box %ld x %ld periodic, T = %.4f, nu = 1/6\n",
              o.P, o.Q, long(Nx), long(Ny), T);
  std::printf("backend %s   precision %s   force %.3e %s the band%s%s%s\n",
              Kokkos::DefaultExecutionSpace::name(), precision_name(), o.g,
              o.across ? "ACROSS" : "ALONG", o.nrec ? "   normal reconstruction ON" : "",
              o.upwind ? "   upwind exchange ON"
                       : (o.geom ? (o.follow ? (o.donor ? "   geometric exchange ON, donor, follows line"
                                                        : "   geometric exchange ON, follows line")
                                             : (o.donor ? "   geometric exchange ON, donor"
                                                        : "   geometric exchange ON"))
                                 : ""),
              o.fo >= 0.0 ? "   fill_offset set" : "");
  if (o.sigma > 0.0)
    std::printf("   surface tension %.3e, %s%s; the force stops at step %ld\n", o.sigma,
                o.hf ? "height functions" : "the default curvature",
                (o.hf && o.hfit != 3) ? ", smoothed" : "", o.gstop);
  std::printf("   column sums of the fill must all equal T; 'rms' and 'max' are their scatter\n"
              "   about their mean, 'k>=pi/2' its share at wavelengths of 4 cells or less;\n"
              "   'kappa rms' is the curvature over interface cells, exactly 0 for a flat band\n\n");
  std::printf("   %-8s %-10s %-10s %-11s %-6s %-10s %-10s %-8s %-10s\n", "step", "u exact", "moved",
              "mass drift", "iface", "rms", "max", "k>=pi/2", "kappa rms");

  const double m0 = double(fs.total_mass());
  // A COPY, not a mirror view: on a host backend create_mirror_view_and_copy
  // hands back the live array itself, and the "initial" state then follows the
  // run (the first version of -predict measured a change of exactly zero).
  auto f0 = Kokkos::create_mirror(HostSpace{}, fs.flags());
  auto e0 = Kokkos::create_mirror(HostSpace{}, fs.fill());
  Kokkos::deep_copy(f0, fs.flags());
  Kokkos::deep_copy(e0, fs.fill());
  bool finite = true;
  for (long t = 0; t <= o.steps; ++t) {
    if (t % o.every == 0 || t == o.steps) {
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      std::vector<double> S(Nx, 0.0);
      int iface = 0;
      for (Index x = 0; x < Nx; ++x)
        for (Index y = 0; y < Ny; ++y) {
          const Index n = d.id(x, y);
          const std::uint8_t f = hf(n);
          if (f == FsFluid) S[x] += 1.0;
          else if (f == FsInterface) { S[x] += double(he(n)); ++iface; }
        }
      const Ripple r = column_ripple(S, T);
      double krms = 0.0;
      if (o.sigma > 0.0) {
        fs.compute_curvature();                    // this state's, not last step's
        auto hk = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.kappa());
        for (Index n = 0; n < d.n_padded; ++n)
          if (hf(n) == FsInterface) krms += double(hk(n)) * double(hk(n));
        krms = iface ? std::sqrt(krms / double(iface)) : 0.0;
      }
      const double m = double(fs.total_mass());
      if (!std::isfinite(m)) finite = false;
      std::printf("   %-8ld %-10.3e %-10.4f %-11.3e %-6d %-10.3e %-10.3e %-8.3f %-10.3e\n", t, u_at(t),
                  x_at(t), (m - m0) / m0, iface, r.rms, r.max, r.hik, krms);
      if (!finite) break;
    }
    if (o.gstop > 0 && t == o.gstop) fs.set_gravity(Real(0), Real(0));
    if (t < o.steps) fs.step();
  }
  if (o.predict && finite) {
    // THE MECHANISM, cell by cell. With equilibrium populations the net flux on
    // link i is in - out = -2 w_i rho (c_i . u)/cs^2, and mass_exchange() weights it
    // by 1 to a Fluid cell, by the mean fill to an Interface cell and by 0 to Gas.
    // Summed over the links and integrated over the slide (u dt -> D t_hat), that
    // predicts every interface cell's fill change from the INITIAL state alone.
    // The measured change is the solver's; the two are compared over the cells
    // that were interface cells at the start.
    auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
    const double D = x_at(o.steps);
    const double ics = double(inv_cs2<L, Real>());
    double sp = 0, sm = 0, spm = 0;
    int nc = 0;
    for (Index y = 0; y < Ny; ++y)
      for (Index x = 0; x < Nx; ++x) {
        const Index n = d.id(x, y);
        if (f0(n) != FsInterface) continue;
        double dm = 0.0;
        for (int i = 1; i < L::Q; ++i) {
          const int cx = cvel<L>(i, 0), cy = cvel<L>(i, 1);
          const Index j = d.id((x + cx + Nx) % Nx, (y + cy + Ny) % Ny);
          if (f0(j) == FsGas) continue;
          const double W = f0(j) == FsFluid ? 1.0 : 0.5 * (double(e0(n)) + double(e0(j)));
          dm += W * (-2.0 * double(weight<L, Real>(i)) * (cx * dir[0] + cy * dir[1]) * ics) * D;
        }
        const double now = hf(n) == FsInterface ? double(he(n)) : (hf(n) == FsFluid ? 1.0 : 0.0);
        const double dmeas = now - double(e0(n));
        sp += dm * dm;  sm += dmeas * dmeas;  spm += dm * dmeas;  ++nc;
      }
    std::printf("\n   the mass exchange, cell by cell over %d interface cells, after moving %.4f:\n"
                "   rms predicted %.4e   rms measured %.4e   correlation %.4f   measured/predicted %.4f\n",
                nc, D, std::sqrt(sp / nc), std::sqrt(sm / nc), spm / std::sqrt(sp * sm), spm / sp);
  }
  if (o.dump) {
    auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
    auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
    std::FILE* fp = std::fopen(o.dump, "w");
    if (fp) {
      for (Index y = 0; y < Ny; ++y)
        for (Index x = 0; x < Nx; ++x) {
          const Index n = d.id(x, y);
          std::fprintf(fp, "%ld %ld %d %.17g\n", long(x), long(y), int(hf(n)), double(he(n)));
        }
      std::fclose(fp);
    }
  }
  std::printf("\n[fs_band] %s\n", finite ? "finished" : "NON-FINITE");
  return finite ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    if      (!std::strcmp(argv[i], "-m") && i + 1 < argc)     o.m = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-g") && i + 1 < argc)     o.g = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-slope") && i + 2 < argc) {
      o.P = std::atoi(argv[++i]);
      o.Q = std::atoi(argv[++i]);
    }
    else if (!std::strcmp(argv[i], "-across"))                o.across = true;
    else if (!std::strcmp(argv[i], "-along"))                 o.across = false;
    else if (!std::strcmp(argv[i], "-steps") && i + 1 < argc) o.steps = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "-every") && i + 1 < argc) o.every = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "-fo") && i + 1 < argc)    o.fo = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-nrec"))                  o.nrec = true;
    else if (!std::strcmp(argv[i], "-upwind"))                o.upwind = true;
    else if (!std::strcmp(argv[i], "-geom"))                  o.geom = true;
    else if (!std::strcmp(argv[i], "-donor"))                 o.donor = true;
    else if (!std::strcmp(argv[i], "-follow"))                o.follow = true;
    else if (!std::strcmp(argv[i], "-sigma") && i + 1 < argc) o.sigma = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-hf"))                    o.hf = true;
    else if (!std::strcmp(argv[i], "-hfit") && i + 1 < argc)  o.hfit = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-gstop") && i + 1 < argc) o.gstop = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "-predict"))               o.predict = true;
    else if (!std::strcmp(argv[i], "-dump") && i + 1 < argc)  o.dump = argv[++i];
    else if (!std::strncmp(argv[i], "--kokkos", 8)) {}
    else {
      std::fprintf(stderr, "fs_band: unknown option '%s'. NOTHING WAS RUN.\n"
                           "  -m M  -slope P Q  -g G  -along|-across  -steps N  -every K  -fo X  -nrec  -upwind  -geom [-donor] [-follow]\n"
                           "  -sigma S [-hf [-hfit N]]  -gstop N  -predict  -dump FILE\n",
                   argv[i]);
      return 2;
    }
  }
  if (o.m < 4 || o.steps < 0 || o.every < 1 || o.P < 0 || o.Q < 1 || o.P > o.Q || o.sigma < 0.0 ||
      ((o.hf || o.hfit != 3) && o.sigma <= 0.0) || (o.hfit != 3 && !o.hf) ||
      o.hfit < 3 || o.hfit > 9 || o.hfit % 2 == 0 || o.gstop < 0 ||
      (o.predict && (o.across || o.sigma > 0.0 || o.geom)) || (o.upwind && o.geom) ||
      ((o.donor || o.follow) && !o.geom)) {
    std::fprintf(stderr, "fs_band: need -m >= 4, -steps >= 0, -every >= 1, 0 <= P <= Q, Q >= 1;"
                         " -hf needs -sigma, -hfit an odd 3..9 and -hf; -predict is for -along"
                         " without -sigma."
                         " NOTHING WAS RUN.\n");
    return 2;
  }
  Kokkos::initialize(argc, argv);
  int rc = 0;
  { rc = run(o); }
  Kokkos::finalize();
  return rc;
}
