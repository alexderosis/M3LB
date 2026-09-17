//==============================================================================
//  Orszag-Tang vortex in three dimensions -- the case that tests div B.
//
//  Initial condition after De Rosis, Phys. Rev. E 95, 013310 (2017), Sec. III D,
//  taken there from Mininni, Pouquet and Montgomery, on a cubic periodic box of
//  side 2 pi discretised by M^3 points:
//
//      u = 2 v0 (-sin Y,  sin X,  0)
//      b = 0.8 b0 (-2 sin 2Y + sin Z,   2 sin X + sin Z,   sin X + sin Y)
//
//  with X = 2 pi x / M and so on. Both fields are solenoidal at t = 0: each
//  component of b is independent of its own coordinate, so div b vanishes term
//  by term.
//
//  WHY THIS CASE EXISTS HERE. It has no closed-form solution -- it steepens into
//  current sheets -- so it is not validated against a formula. It tests what a
//  formula cannot:
//
//    * DIV B PRESERVATION over a long nonlinear run. The wave cases cannot: in
//      those, div B is structurally zero and reports round-off whatever the
//      scheme does. Here the nonlinear dynamics makes every component depend on
//      every coordinate, so a scheme that generates monopoles will show it. The
//      antisymmetry of the induction equilibrium's first moment is what prevents
//      that, and this is where it is exercised.
//    * the energy budget, which must decay monotonically and never rise.
//    * self-convergence: refine M and the curves must approach a limit.
//
//  WHAT IS NOT CLAIMED. The paper's Figure 6 is a plot against a high-resolution
//  pseudospectral run whose values are not available numerically, so no
//  per-point comparison is possible and none is made. In particular this driver
//  does NOT assert when J_max peaks; it prints the history and leaves the
//  reading to whoever has the reference.
//
//  J_max IS PRINTED IN BOTH UNITS, AND THE DIMENSIONLESS ONE IS THE PAPER'S.
//  measure() forms the curl with central differences of spacing 1, so its J_max
//  is in LATTICE units and is ~1e-2 where the paper's figures are ~1e1.
//
//  THE CONVERSION, STATED AS THE DIMENSIONAL ARGUMENT RATHER THAN AS A FORMULA,
//  because the formula hides an assumption and the argument does not:
//
//      length     x_phys = x_lat * dx,      dx = 2 pi / M
//      velocity   u_phys = u_lat / v0,      v0 = |u|_rms(lattice) / 2
//      time       t = x / u   ->   dt = dx * v0
//      field      B_lat = v0 * B_Mininni  (both are 0.8 * [same shape], one
//                 carrying v0), so B converts by v0 as well -- NOT by b0
//      current    J = curl B   ->   J_phys = J_lat / (dx * v0) = J_lat / dt
//
//  The 2 in v0 = |u|_rms/2 is Mininni's own rms, a property of his printed IC:
//  <|u|^2> = 4 there, which is his stated E_V = 2. So this reads the velocity
//  scale off the FIELD.
//
//  v0 = Ma / (2 sqrt(2) sqrt(3)) below is how v0 is CHOSEN, not what it means.
//  It reads the scale off De Rosis's Ma = 0.034 under the assumption that Ma is
//  built on the PEAK speed 2 sqrt(2) v0; on the rms instead it would be
//  Ma / (2 sqrt(3)), larger by sqrt(2), and nu, dt and the J_max scale would all
//  move with it. The dimensional argument above stays correct either way; the
//  formula does not. Mininni himself has no Mach number -- he is incompressible
//  pseudospectral -- so the ambiguity belongs to the LBM paper, not to him.
//
//  Checked three ways: 6.7e-09 against the tracked M = 32 Kokkos series, which
//  carries both columns; the printed factor against the arithmetic done
//  separately (733.8323 at M = 32); and J_max(t=0) = 5.2174 here against the
//  same 5.2174 in that series, with no shared headers.
//
//  NOT corrected for b0. Mininni states E_V = E_M = 2, which pins b0/v0 =
//  1.020621, and this driver uses b0 = v0 -- so its magnetic field is 2 % weak
//  and E_M/E_V reads 0.96 (the t = 0 row shows E_u/E0 = 0.5102 where an exact
//  reproduction would show 0.5000). That is an IC error, NOT a scaling one:
//  J_max above is the right number for the field this run actually had.
//
//  Two parameters the paper does not pin down, stated here as assumptions rather
//  than readings: it says only that v0 and b0 "lead to a Mach number Ma ~ 0.034",
//  and Ma depends on which speed it is built on. Taken on the PEAK initial speed,
//  which for this field is 2 sqrt(2) v0, giving v0 = Ma / (2 sqrt(2) sqrt(3)).
//  And b0 = v0, the usual equipartition choice.
//
//    usage: orszag_tang [-m M] [-re RE] [-ma MA] [-tmax T] [-op bgk|cm]
//                       [-probes N] [-vti K] [-dump K] [-dumpvol K]
//                       [-volstride S]
//
//  -vti K writes a ParaView .vti every K-th PROBE, plus a .pvd time series, into
//  ./vti/. Tying frames to probes rather than to steps is deliberate: a frame
//  needs the six host fields a probe has already copied, and writing on its own
//  schedule would pay for a second copy of all of them. It also means the frame
//  times ARE the probe times, so -probes 40 -vti 1 over tmax = 4 puts a frame
//  every 0.1 without any divisibility trap for the caller to fall into.
//
//  THE DIRECTORY IS CREATED HERE. mhd_sphere.cu does not create its own and
//  merely warns per frame when the open fails, which can lose a whole run's
//  output to a missing mkdir. Not repeating that.
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/vti.cuh"

#include <algorithm>
#include <fstream>
#include <cstdint>

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct OtFluidInit {
  Real dl, v0;
  LBM_HD Macro operator()(int x, int y, int) const {
    const double X = double(dl) * double(x), Y = double(dl) * double(y);
    Macro m;
    m.rho = Real(1);
    m.ux  = Real(-2.0 * double(v0) * sin(Y));
    m.uy  = Real( 2.0 * double(v0) * sin(X));
    m.uz  = Real(0);
    return m;
  }
};

struct OtBInit {
  Real dl, b0;
  LBM_HD void operator()(int x, int y, int z, Real B[3]) const {
    const double X = double(dl) * double(x), Y = double(dl) * double(y),
                 Z = double(dl) * double(z);
    const double a = double(b0);
    B[0] = Real(a * (-2.0 * sin(2.0 * Y) + sin(Z)));
    B[1] = Real(a * ( 2.0 * sin(X)       + sin(Z)));
    B[2] = Real(a * ( sin(X) + sin(Y)));
  }
};

struct OtUInit {
  Real dl, v0;
  LBM_HD void operator()(int x, int y, int, Real u[3]) const {
    const double X = double(dl) * double(x), Y = double(dl) * double(y);
    u[0] = Real(-2.0 * double(v0) * sin(Y));
    u[1] = Real( 2.0 * double(v0) * sin(X));
    u[2] = Real(0);
  }
};

//------------------------------------------------------------------------------
// Diagnostics on the host: energies, max |curl B|, and max |div B| normalised by
// the field's own gradient scale k|B|, which is the only normalisation that
// stays meaningful as the grid refines.
//------------------------------------------------------------------------------
struct Diag { double eu, eb, jmax, divb, finite; };

static Diag measure(const std::vector<Real>& ux, const std::vector<Real>& uy,
                    const std::vector<Real>& uz, const std::vector<Real>& bx,
                    const std::vector<Real>& by, const std::vector<Real>& bz, int M) {
  auto id = [M](int x, int y, int z) {
    return std::size_t(node_id(((x % M) + M) % M, ((y % M) + M) % M,
                               ((z % M) + M) % M, M, M));
  };
  Diag d{0, 0, 0, 0, 1};
  double bscale = 0;
  for (int z = 0; z < M; ++z)
    for (int y = 0; y < M; ++y)
      for (int x = 0; x < M; ++x) {
        const std::size_t n = id(x, y, z);
        const double a = ux[n], b = uy[n], c = uz[n];
        const double p = bx[n], q = by[n], r = bz[n];
        if (!std::isfinite(a) || !std::isfinite(p)) { d.finite = 0; continue; }
        d.eu += 0.5 * (a * a + b * b + c * c);
        d.eb += 0.5 * (p * p + q * q + r * r);
        bscale = std::fmax(bscale, std::sqrt(p * p + q * q + r * r));

        // Central differences, spacing 1 in lattice units.
        const double dxbx = 0.5 * (double(bx[id(x + 1, y, z)]) - double(bx[id(x - 1, y, z)]));
        const double dyby = 0.5 * (double(by[id(x, y + 1, z)]) - double(by[id(x, y - 1, z)]));
        const double dzbz = 0.5 * (double(bz[id(x, y, z + 1)]) - double(bz[id(x, y, z - 1)]));
        d.divb = std::fmax(d.divb, std::fabs(dxbx + dyby + dzbz));

        const double jx = 0.5 * (double(bz[id(x, y + 1, z)]) - double(bz[id(x, y - 1, z)]))
                        - 0.5 * (double(by[id(x, y, z + 1)]) - double(by[id(x, y, z - 1)]));
        const double jy = 0.5 * (double(bx[id(x, y, z + 1)]) - double(bx[id(x, y, z - 1)]))
                        - 0.5 * (double(bz[id(x + 1, y, z)]) - double(bz[id(x - 1, y, z)]));
        const double jz = 0.5 * (double(by[id(x + 1, y, z)]) - double(by[id(x - 1, y, z)]))
                        - 0.5 * (double(bx[id(x, y + 1, z)]) - double(bx[id(x, y - 1, z)]));
        d.jmax = std::fmax(d.jmax, std::sqrt(jx * jx + jy * jy + jz * jz));
      }
  // Normalise div B by k |B|: the gradient the field itself carries.
  const double k = 2.0 * M_PI / M;
  if (bscale > 0) d.divb /= k * bscale;
  return d;
}

// Raw field dumps for the animation pipeline, byte-identical in format to
// mhd_sphere.cu's: int32 nx, ny [, nz] then nx*ny[*nz] float32, x fastest. The
// renderers in results/N_mhd_sphere/ read exactly this.
//
// meta.txt carries R = 0, which is the sentinel for "periodic box, no sphere".
// mhd_sphere writes a real radius there and the renderers mask to it and draw
// the boundary ring; this case has neither, so R = 0 tells them to render the
// whole slice and draw no ring. Writing 0 rather than omitting R matters: an
// ABSENT R makes render_slices.py assume 0.40 N from mhd_sphere's own default
// and silently cut the box down to a disc.
static void write_raw2(const std::string& path, int nx, int ny,
                       const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}
static void write_raw3(const std::string& path, int nx, int ny, int nz,
                       const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny, c = nz;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(&c), sizeof c);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

// |J| at every node, for the .vti. Same central differences and same periodic
// wrap as measure(); kept separate because measure() runs at every probe and
// this only runs when a frame is actually written.
static void jmag_field(const std::vector<Real>& bx, const std::vector<Real>& by,
                       const std::vector<Real>& bz, int M, std::vector<float>& j) {
  auto id = [M](int x, int y, int z) {
    return std::size_t(node_id(((x % M) + M) % M, ((y % M) + M) % M,
                               ((z % M) + M) % M, M, M));
  };
  j.assign(std::size_t(M) * std::size_t(M) * std::size_t(M), 0.0f);
  for (int z = 0; z < M; ++z)
    for (int y = 0; y < M; ++y)
      for (int x = 0; x < M; ++x) {
        const double jx = 0.5 * (double(bz[id(x, y + 1, z)]) - double(bz[id(x, y - 1, z)]))
                        - 0.5 * (double(by[id(x, y, z + 1)]) - double(by[id(x, y, z - 1)]));
        const double jy = 0.5 * (double(bx[id(x, y, z + 1)]) - double(bx[id(x, y, z - 1)]))
                        - 0.5 * (double(bz[id(x + 1, y, z)]) - double(bz[id(x - 1, y, z)]));
        const double jz = 0.5 * (double(by[id(x + 1, y, z)]) - double(by[id(x - 1, y, z)]))
                        - 0.5 * (double(bx[id(x, y + 1, z)]) - double(bx[id(x, y - 1, z)]));
        j[id(x, y, z)] = float(std::sqrt(jx * jx + jy * jy + jz * jz));
      }
}

// One frame: rho, u, b as vectors, and |J| as the scalar ParaView opens on.
static bool write_frame(int M, int idx, const std::vector<Real>& rho,
                        const std::vector<Real>& ux, const std::vector<Real>& uy,
                        const std::vector<Real>& uz, const std::vector<Real>& bx,
                        const std::vector<Real>& by, const std::vector<Real>& bz,
                        std::string& name) {
  const std::size_t np = std::size_t(M) * std::size_t(M) * std::size_t(M);
  std::vector<float> jm;
  jmag_field(bx, by, bz, M, jm);
  std::vector<float> fr(np), fu(3 * np), fb(3 * np);
  for (std::size_t n = 0; n < np; ++n) {
    fr[n] = float(rho[n]);
    fu[3 * n] = float(ux[n]); fu[3 * n + 1] = float(uy[n]); fu[3 * n + 2] = float(uz[n]);
    fb[3 * n] = float(bx[n]); fb[3 * n + 1] = float(by[n]); fb[3 * n + 2] = float(bz[n]);
  }
  std::vector<lbm::VtiArray> arr;
  arr.push_back({"Jmag", 1, std::move(jm)});
  arr.push_back({"rho", 1, std::move(fr)});
  arr.push_back({"u", 3, std::move(fu)});
  arr.push_back({"b", 3, std::move(fb)});
  char buf[64];
  std::snprintf(buf, sizeof buf, "ot3d_%04d.vti", idx);
  name = buf;
  return lbm::write_vti_bin(std::string("vti/") + name, M, M, M, arr);
}

// One animation frame: the three mid-plane slices the renderers expect, and
// optionally the |J| volume. Mid-plane is z = M/2, matching mhd_sphere.
// vstride > 1 reduces the |J| volume by BLOCK MAXIMUM to (M/vstride)^3 before
// writing. Necessary rather than decorative: at M = 288 a full volume frame is
// 95.6 MB, so 244 of them is 23 GB on disk and about three hours in
// render_volume.py, which is pure Python. At vstride 3 it is 3.5 MB and seven
// minutes.
//
// MAXIMUM, NOT MEAN, and the choice is the whole point. The renderer draws a
// MAX-intensity projection, so a block max is the same operator applied earlier
// and the current sheets survive it; a block mean would average a one-cell sheet
// against its neighbours and dim exactly the structure the picture is of. The
// last block on each axis is clipped rather than wrapped, so M need not divide
// by vstride.
static void dump_frame(int M, int idx, bool withvol, int vstride,
                       const std::vector<Real>& ux, const std::vector<Real>& uy,
                       const std::vector<Real>& uz, const std::vector<Real>& bx,
                       const std::vector<Real>& by, const std::vector<Real>& bz) {
  const std::size_t np = std::size_t(M) * std::size_t(M) * std::size_t(M);
  std::vector<float> jm;
  jmag_field(bx, by, bz, M, jm);
  const int z0 = M / 2;
  std::vector<float> su(std::size_t(M) * M), sb(std::size_t(M) * M), sj(std::size_t(M) * M);
  for (int y = 0; y < M; ++y)
    for (int x = 0; x < M; ++x) {
      const std::size_t n = std::size_t(node_id(x, y, z0, M, M));
      const std::size_t m = std::size_t(y) * M + x;
      su[m] = float(std::sqrt(double(ux[n]) * ux[n] + double(uy[n]) * uy[n] + double(uz[n]) * uz[n]));
      sb[m] = float(std::sqrt(double(bx[n]) * bx[n] + double(by[n]) * by[n] + double(bz[n]) * bz[n]));
      sj[m] = jm[n];
    }
  char nm[96];
  std::snprintf(nm, sizeof nm, "anim_frames/umag_%04d.raw", idx); write_raw2(nm, M, M, su);
  std::snprintf(nm, sizeof nm, "anim_frames/bmag_%04d.raw", idx); write_raw2(nm, M, M, sb);
  std::snprintf(nm, sizeof nm, "anim_frames/jmag_%04d.raw", idx); write_raw2(nm, M, M, sj);
  if (withvol) {
    std::snprintf(nm, sizeof nm, "anim_frames/jvol_%04d.raw", idx);
    if (vstride <= 1) {
      write_raw3(nm, M, M, M, jm);
    } else {
      const int Mr = (M + vstride - 1) / vstride;
      std::vector<float> red(std::size_t(Mr) * Mr * Mr, 0.0f);
      for (int z = 0; z < Mr; ++z)
        for (int y = 0; y < Mr; ++y)
          for (int x = 0; x < Mr; ++x) {
            float m = 0.0f;
            for (int dz = 0; dz < vstride; ++dz) {
              const int zz = z * vstride + dz; if (zz >= M) break;
              for (int dy = 0; dy < vstride; ++dy) {
                const int yy = y * vstride + dy; if (yy >= M) break;
                for (int dx = 0; dx < vstride; ++dx) {
                  const int xx = x * vstride + dx; if (xx >= M) break;
                  const float v = jm[std::size_t(node_id(xx, yy, zz, M, M))];
                  if (v > m) m = v;
                }
              }
            }
            red[(std::size_t(z) * Mr + y) * Mr + x] = m;
          }
      write_raw3(nm, Mr, Mr, Mr, red);
    }
  }
  (void)np;
}

int main(int argc, char** argv) {
  int M = 64, nprobe = 20;
  double Re = 100.0, Ma = 0.034, tmax = 4.0;
  int vti = 0, dump = 0, dvol = 0, vstride = 1;
  std::string op = "cm";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-m"    && i + 1 < argc) M    = std::atoi(argv[++i]);
    if (a == "-re"   && i + 1 < argc) Re   = std::atof(argv[++i]);
    if (a == "-ma"   && i + 1 < argc) Ma   = std::atof(argv[++i]);
    if (a == "-tmax" && i + 1 < argc) tmax = std::atof(argv[++i]);
    if (a == "-op"   && i + 1 < argc) op   = argv[++i];
    if (a == "-probes" && i + 1 < argc) nprobe = std::atoi(argv[++i]);
    if (a == "-vti"    && i + 1 < argc) vti    = std::atoi(argv[++i]);
    if (a == "-dump"   && i + 1 < argc) dump   = std::atoi(argv[++i]);
    if (a == "-dumpvol"&& i + 1 < argc) dvol   = std::atoi(argv[++i]);
    if (a == "-volstride" && i + 1 < argc) vstride = std::max(1, std::atoi(argv[++i]));
  }

  // Ma on the PEAK initial speed, 2 sqrt(2) v0 -- an assumption, not a reading.
  const double v0 = Ma / (2.0 * std::sqrt(2.0) * std::sqrt(3.0));
  const double b0 = 0.8 * v0;                  // equipartition, times the 0.8 of the IC
  const double nu = v0 * double(M) / Re;
  const double eta = nu;                       // Pr_m = 1
  const double dl = 2.0 * M_PI / double(M);
  const double dt = dl * v0;                   // one unit of t per 1/(dl v0) steps
  const std::size_t T = std::size_t(tmax / dt);
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  std::printf("Orszag-Tang 3D   %s   D3Q27 fluid / D3Q7 field   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  M = %d (%ld nodes)   Re = %.0f   Ma = %.3f   Pr_m = 1\n",
              M, long(M) * M * M, Re, Ma);
  std::printf("  v0 = %.6e   b0 = %.6e   nu = eta = %.6e (tau %.6f)\n",
              v0, b0, nu, 3.0 * nu + 0.5);
  std::printf("  t up to %.1f  (%zu steps)\n\n", tmax, T);

  backend::Magnetic mag(M, M, M, Real(eta));
  backend::Fluid    fl (M, M, M, which, Real(nu));

  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  fl.initialise_with(OtFluidInit{Real(dl), Real(v0)});
  mag.initialise_with(OtBInit{Real(dl), Real(b0)}, OtUInit{Real(dl), Real(v0)});

  std::vector<Real> rho, ux, uy, uz, bx, by, bz;
  auto sample = [&]() {
    fl.macroscopic_to_host(rho, ux, uy, uz);
    mag.field_to_host(bx, by, bz);
    return measure(ux, uy, uz, bx, by, bz, M);
  };

  const Diag d0 = sample();
  const double e0 = d0.eu + d0.eb;
  std::printf("  %8s %12s %12s %12s %14s %13s %11s\n", "t", "E/E0", "E_u/E0",
              "E_b/E0", "max|divB|/k|B|", "Jmax(lat)", "Jmax");
  std::printf("  %8.3f %12.6f %12.6f %12.6f %14.3e %13.4e %11.4f\n", 0.0, 1.0,
              d0.eu / e0, d0.eb / e0, d0.divb, d0.jmax, d0.jmax / dt);

  std::vector<std::pair<double, std::string>> pvd;
  int frame = 0, probe_no = 0;
  if (vti > 0) {
    ::mkdir("vti", 0755);                       // EEXIST is fine and expected
    std::string nm;
    if (!write_frame(M, frame, rho, ux, uy, uz, bx, by, bz, nm)) {
      std::fprintf(stderr, "cannot write into vti/ -- giving up rather than "
                           "running to completion with no output\n");
      return 2;
    }
    pvd.emplace_back(0.0, nm);
    ++frame;
  }

  std::FILE* meta = nullptr;
  int dframe = 0;
  if (dump > 0) {
    ::mkdir("anim_frames", 0755);
    meta = std::fopen("anim_frames/meta.txt", "w");
    if (!meta) {
      std::fprintf(stderr, "cannot open anim_frames/meta.txt -- giving up\n");
      return 2;
    }
    // R = 0 is the sentinel for "periodic box": render the whole slice, draw no
    // boundary ring. Te is the eddy time the frame clock is quoted in; here the
    // paper's own t IS that clock, so Te = 1 and t/Te = t.
    std::fprintf(meta, "N %d\nR 0\nTe 1.0\n", M);
    dump_frame(M, dframe, dvol > 0, vstride, ux, uy, uz, bx, by, bz);
    std::fprintf(meta, "frame %d %.6f %.8e %.8e\n", dframe, 0.0, d0.eu / e0, d0.eb / e0);
    std::fflush(meta);
    ++dframe;
  }

  // Each probe copies six full fields to the host and walks every node there to
  // form curl and div. At 512^3 that is 3.2 GB and 134M host iterations per
  // sample, which can cost more than the simulation. Lower -probes on large
  // grids; the summary lines do not depend on how many were taken.
  const std::size_t probe = (nprobe > 0 && T / std::size_t(nprobe)) ? T / std::size_t(nprobe) : 1;
  double worst_div = d0.divb, prev_e = 1.0, worst_rise = 0.0;
  const auto wall0 = std::chrono::steady_clock::now();

  for (std::size_t t = 1; t <= T; ++t) {
    mag.compute_field();
    fl.step();
    mag.step();
    if (t % probe == 0 || t == T) {
      const Diag d = sample();
      if (!d.finite) { std::printf("  DIVERGED at t = %.3f\n", double(t) * dt); return 1; }
      const double e = (d.eu + d.eb) / e0;
      worst_rise = std::fmax(worst_rise, e - prev_e);
      prev_e = e;
      worst_div = std::fmax(worst_div, d.divb);
      std::printf("  %8.3f %12.6f %12.6f %12.6f %14.3e %13.4e %11.4f\n",
                  double(t) * dt, e, d.eu / e0, d.eb / e0, d.divb,
                  d.jmax, d.jmax / dt);
      std::fflush(stdout);
      ++probe_no;
      if (vti > 0 && probe_no % vti == 0) {
        std::string nm;
        if (write_frame(M, frame, rho, ux, uy, uz, bx, by, bz, nm)) {
          pvd.emplace_back(double(t) * dt, nm);
          ++frame;
        }
      }
      if (dump > 0 && probe_no % dump == 0) {
        dump_frame(M, dframe, dvol > 0 && (probe_no % dvol == 0), vstride,
                   ux, uy, uz, bx, by, bz);
        if (meta) {
          std::fprintf(meta, "frame %d %.6f %.8e %.8e\n", dframe,
                       double(t) * dt, d.eu / e0, d.eb / e0);
          std::fflush(meta);
        }
        ++dframe;
      }
    }
  }

  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();
  if (meta) {
    std::fprintf(meta, "frames %d\n", dframe);
    std::fclose(meta);
    std::printf("\n  wrote %d frames to anim_frames/  (render with "
                "results/N_mhd_sphere/render_slices.py and render_volume.py)\n", dframe);
  }
  if (vti > 0) {
    lbm::write_pvd("vti/ot3d.pvd", pvd);
    std::printf("\n  ParaView: open vti/ot3d.pvd  (%d frames, t 0 to %.3f)\n"
                "  Colour by Jmag; u and b are there as vectors.\n",
                frame, pvd.empty() ? 0.0 : pvd.back().first);
  }
  std::printf("\n  Jmax columns: (lat) is lattice units, Jmax is the paper's --\n"
              "      Jmax = Jmax(lat) / dt,  dt = v0 * 2pi/M = %.6e  (x %.2f here).\n",
              dt, 1.0 / dt);
  std::printf("  worst max|div B| / k|B| over the run   %.3e\n", worst_div);
  std::printf("  largest rise in E_u + E_b between samples %.3e\n", worst_rise);
  std::printf("      Ideal incompressible MHD has dE/dt = -nu |grad u|^2 - eta |grad B|^2,\n");
  std::printf("      so this should be zero. It is not, at coarse M: the exchange between\n");
  std::printf("      kinetic and magnetic energy overshoots when the current sheets are\n");
  std::printf("      under-resolved, and E_b oscillates by a per cent or two while E_u\n");
  std::printf("      decays smoothly. Measured to t = 0.5, central moments: 1.10e-2 at\n");
  std::printf("      M = 12, 5.92e-3 at M = 24, and 0 at M = 32. A rise that does NOT\n");
  std::printf("      vanish as M grows is a different matter and would be a real fault.\n");
  std::printf("  %zu steps in %.2f s  ->  %.1f MLUPS (fluid nodes only)\n",
              T, sec, double(fl.nodes()) * double(T) / sec / 1e6);
  return 0;
}
