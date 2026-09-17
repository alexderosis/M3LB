//==============================================================================
//  Transitional magnetohydrodynamic jet -- Williams, De Rosis & Skillen (JFM,
//  under consideration), whose setup is Mak, Griffiths & Hughes, Phys. Rev.
//  Fluids 2, 113701 (2017).
//
//  A sech^2 jet in a fully periodic box carrying a uniform streamwise field.
//  The jet's sigma-mode instability rolls up into vortices, the vortices stretch
//  and fold the field, the folded field forms current sheets, and at t ~ 82 the
//  sheets go three-dimensional. The reference paper calls that last step SGTR
//  and its whole argument rests on WHEN it happens and HOW FAST the magnetic
//  energy grows into it, so those two curves -- not the fields -- are what this
//  driver exists to produce.
//
//  THE REFERENCE, IN ITS OWN UNITS.  L0 = 1, u0 = max sech^2 = 1, so
//
//      u(t=0) = (sech^2(x2), 0, 0) + (A sin(s x1) + a, A sin(s x1) + a, a) e^{-x2^2}
//      b(t=0) = (b0, 0, 0)
//
//  with s = 0.9 (the unstable mode), A = 1e-3, a ~ U(-1e-5, 1e-5), and a box
//  Lx1 = Lx3 = 4 pi / s, Lx2 = 20, x2 in [-10, 10].  Re = u0 L0 / nu = 3500,
//  Pm = nu / eta = 1, and S = L0 b0 / eta = 52.5 at t = 0, which fixes
//  b0 / u0 = S / (Re Pm) = 0.015.  The reference ran 512 x 1024 x 512 to t = 150.
//
//  THREE PLACES WHERE THIS RUN CANNOT BE THE SAME RUN, ALL STATED RATHER THAN
//  ABSORBED.
//
//  (1) THE GRID CANNOT BE ANISOTROPIC.  Theirs is: dx1 = dx3 = 0.02727 against
//      dx2 = 0.01953, a ratio of 1.40.  A uniform lattice has one spacing, so
//      matching x1 leaves us COARSER across the jet than they are -- at -nx 512
//      this box is 512 x 734 x 512, not 512 x 1024 x 512.  Nothing here can fix
//      that; it is a property of the method.
//
//  (2) THE INITIAL CONDITION IS MADE SOLENOIDAL BY CONSTRUCTION, NOT BY
//      PROJECTION.  Their u(t=0) is not divergence free -- d1 u1 alone is
//      A s cos(s x1) e^{-x2^2}, i.e. 9e-4, the same order as the perturbation
//      itself -- and they remove that with a Helmholtz-Hodge projection, which
//      needs an FFT.  GPU/ has no FFT and acquiring one to run the initial
//      condition once is the wrong trade, so the perturbation is instead built
//      as the CURL OF A VECTOR POTENTIAL and is divergence free exactly, under
//      the same central-difference operator measure() uses.  Choosing
//      Psi3 = (A/s) cos(s x1) e^{-x2^2} reproduces their CROSS-STREAM component
//      u2 = A sin(s x1) e^{-x2^2} term for term -- that is the component that
//      drives the shear instability -- and gives a streamwise partner of the
//      same amplitude (2A/s * max|x2 e^{-x2^2}| = 0.95 A) but a different shape,
//      cos(s x1) x2 e^{-x2^2} rather than sin(s x1) e^{-x2^2}.  The random part
//      is the curl of a hashed potential, so it is broadband and solenoidal
//      where theirs is broadband and projected.  SAME MODE, SAME AMPLITUDE,
//      DIFFERENT REALISATION.  Since seed amplitude enters an exponential
//      instability logarithmically, expect the onset time to shift by O(ln 2 /
//      growth rate) rather than to land on their t = 82; a shifted onset is
//      therefore NOT evidence of disagreement, and the shape of gamma(t) is.
//
//  (3) DIV B IS NOT CLEANED.  They run Brackbill-Barnes and hold d_i b_i below
//      1e-13.  Dellar's vector distribution has no cleaning stage: it preserves
//      div B through the antisymmetry of the induction equilibrium's first
//      moment and no better.  The Orszag-Tang run at Re = 3040 measured
//      max|div B| / k|B| = 5.24.  THIS MATTERS MORE HERE THAN THERE.  A div B
//      error is a magnetic monopole and monopoles reconnect, so on a case whose
//      entire subject is reconnection the scheme can reproduce the qualitative
//      story for the wrong reason.  The column is printed every probe and the
//      worst value at the end; read it before reading anything else.
//
//  WHAT THE COLUMNS ARE FOR.  gamma is their Eq. (3.1),
//
//      gamma = d/dt [ (1/2) int b_i b_i dOmega / int b0 b0 dOmega ],
//
//  which is half the growth rate of E_b/E_b(0) as written; E_b/E0 is printed
//  beside it so a different reading of that normalisation can be recovered
//  without rerunning.  jx, jy, jz are max|j_i| per component -- their Fig. 2
//  right -- and the transition they name is the moment jx and jy climb to meet
//  jz, which starts alone.  All four current columns are in the paper's units.
//
//  WHAT IS NOT CLAIMED.  Not a validation: the reference data is not published,
//  and at any resolution reachable here this is under-resolved relative to a
//  512 x 1024 x 512 sixth-order DNS -- k_max eta_K is about 0.25 even at
//  -nx 512, against the ~1.5 a DNS wants.  Spectra and stress PDFs are
//  deliberately NOT produced: ours would be measuring the scheme's dissipation
//  range, not the flow's.
//
//  PRECISION IS A SEED QUESTION HERE, AND THE MARGIN IS THREEFOLD, NOT NINE
//  DECADES.  The initial condition is solenoidal by construction, so the
//  startup line `max|div u|/k|u|` is a direct readout of the round-off noise
//  floor in u.  It reads 7.0e-15 in FP64 and 3.3e-6 in FP32.  Against a
//  deliberate seed of 1e-5 that is nine decades of headroom in FP64 and about
//  HALF A DECADE in FP32.
//
//  So FP32 does not erase the seed -- an earlier version of this banner said it
//  would, and the measurement above is what corrected it.  What FP32 does is
//  COMPETE with it: a broadband noise floor a factor of three below the seed,
//  re-injected every one of ~2.8e5 steps, while the sigma mode has to grow
//  exponentially out of that seed over t = 0 to 80.  Since ehd_cavity has
//  already measured that a seed here selects a BRANCH and not merely a
//  transient, expect an FP32 onset TIME to be unreliable rather than merely
//  shifted -- and onset time is what the reference's t = 82 is.
//
//  FP32 is therefore supported and is not silently wrong; it is a run whose
//  onset time should not be quoted against theirs.  The growth-rate SHAPE of
//  gamma(t), the current-component ordering, and the energy partition do not
//  depend on when the instability was seeded, and remain readable.  Both
//  precisions print the noise floor on startup, so the log says which run it is.
//==============================================================================
#include "lbm/backend.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include <sys/stat.h>

using namespace lbm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// A reproducible hash-based U(-1,1), evaluated per (cell, component). It has to
// be a FUNCTION of the wrapped index rather than a stored array, because the
// initialiser runs as a device functor and gets nothing but (x, y, z) -- and it
// has to wrap, or the random potential is discontinuous across the periodic
// seam and the curl of it puts a sheet of vorticity there.
//------------------------------------------------------------------------------
LBM_HD inline std::uint32_t mix32(std::uint32_t h) {
  h ^= h >> 16; h *= 0x7feb352dU;
  h ^= h >> 15; h *= 0x846ca68bU;
  h ^= h >> 16;
  return h;
}

LBM_HD inline double urand(int X, int Y, int Z, int c, std::uint32_t seed) {
  std::uint32_t h = mix32(std::uint32_t(X) + 0x9e3779b9U *
                   (std::uint32_t(Y) + 0x85ebca6bU *
                   (std::uint32_t(Z) + 0xc2b2ae35U * (std::uint32_t(c) + seed))));
  return double(h) * (2.0 / 4294967296.0) - 1.0;
}

//------------------------------------------------------------------------------
// The vector potential whose curl is the perturbation. Components 0,1,2 are
// Psi_x1, Psi_x2, Psi_x3; the deterministic sigma mode lives in Psi_x3 alone.
//
// The random amplitude carries a factor dl: the discrete curl of white noise of
// amplitude a has r.m.s. a/(dl sqrt 3), so a = seed * dl * sqrt(3) makes `seed`
// mean the r.m.s. of the resulting VELOCITY perturbation, which is the quantity
// the reference specifies. Without the dl the seed would scale with resolution.
//------------------------------------------------------------------------------
struct JetPsi {
  int nx, ny, nz;
  double dl, half, amp, sigma, seed_a;
  std::uint32_t seed;

  LBM_HD double operator()(int c, int x, int y, int z) const {
    const int X = ((x % nx) + nx) % nx;
    const int Y = ((y % ny) + ny) % ny;
    const int Z = ((z % nz) + nz) % nz;
    const double x1  = dl * double(X);
    const double x2  = -half + dl * double(Y);
    const double env = exp(-x2 * x2);
    double p = seed_a * urand(X, Y, Z, c, seed) * env;
    if (c == 2) p += (amp / sigma) * cos(sigma * x1) * env;
    return p;
  }
};

// curl Psi, central differences of spacing dl -- the same stencil measure() uses
// for div B, so the initial velocity divergence it would report is exactly zero.
LBM_HD inline void jet_curl(const JetPsi& P, int x, int y, int z, double d[3]) {
  const double h = 0.5 / P.dl;
  d[0] = h * (P(2, x, y + 1, z) - P(2, x, y - 1, z))
       - h * (P(1, x, y, z + 1) - P(1, x, y, z - 1));
  d[1] = h * (P(0, x, y, z + 1) - P(0, x, y, z - 1))
       - h * (P(2, x + 1, y, z) - P(2, x - 1, y, z));
  d[2] = h * (P(1, x + 1, y, z) - P(1, x - 1, y, z))
       - h * (P(0, x, y + 1, z) - P(0, x, y - 1, z));
}

LBM_HD inline double jet_base(const JetPsi& P, int y) {
  const int Y = ((y % P.ny) + P.ny) % P.ny;
  const double x2 = -P.half + P.dl * double(Y);
  const double c  = cosh(x2);
  return 1.0 / (c * c);                       // sech^2(x2)
}

struct JetFluidInit {
  JetPsi P; Real u0;
  LBM_HD Macro operator()(int x, int y, int z) const {
    double d[3]; jet_curl(P, x, y, z, d);
    Macro m;
    m.rho = Real(1);
    m.ux  = Real(double(u0) * (jet_base(P, y) + d[0]));
    m.uy  = Real(double(u0) * d[1]);
    m.uz  = Real(double(u0) * d[2]);
    return m;
  }
};

struct JetUInit {
  JetPsi P; Real u0;
  LBM_HD void operator()(int x, int y, int z, Real u[3]) const {
    double d[3]; jet_curl(P, x, y, z, d);
    u[0] = Real(double(u0) * (jet_base(P, y) + d[0]));
    u[1] = Real(double(u0) * d[1]);
    u[2] = Real(double(u0) * d[2]);
  }
};

struct JetBInit {
  Real b0;
  LBM_HD void operator()(int, int, int, Real B[3]) const {
    B[0] = b0; B[1] = Real(0); B[2] = Real(0);
  }
};

//------------------------------------------------------------------------------
// Diagnostics. Same construction as orszag_tang.cu's, generalised to a
// non-cubic box and extended with the per-component current maxima, which are
// the reference's Fig. 2 right and the marker it reads the 3D onset from.
//------------------------------------------------------------------------------
struct Diag { double eu, eb, jmax, jc[3], divb, divu, finite; };

static Diag measure(const std::vector<Real>& ux, const std::vector<Real>& uy,
                    const std::vector<Real>& uz, const std::vector<Real>& bx,
                    const std::vector<Real>& by, const std::vector<Real>& bz,
                    int nx, int ny, int nz, double dl) {
  auto id = [nx, ny, nz](int x, int y, int z) {
    return std::size_t(node_id(((x % nx) + nx) % nx, ((y % ny) + ny) % ny,
                               ((z % nz) + nz) % nz, nx, ny));
  };
  Diag d{0, 0, 0, {0, 0, 0}, 0, 0, 1};
  double bscale = 0, uscale = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t n = id(x, y, z);
        const double a = ux[n], b = uy[n], c = uz[n];
        const double p = bx[n], q = by[n], r = bz[n];
        if (!std::isfinite(a) || !std::isfinite(p)) { d.finite = 0; continue; }
        d.eu += 0.5 * (a * a + b * b + c * c);
        d.eb += 0.5 * (p * p + q * q + r * r);
        bscale = std::fmax(bscale, std::sqrt(p * p + q * q + r * r));
        uscale = std::fmax(uscale, std::sqrt(a * a + b * b + c * c));

        // div u, same stencil. At t = 0 this is the check that the vector
        // potential construction really did replace their FFT projection.
        const double dxux = 0.5 * (double(ux[id(x + 1, y, z)]) - double(ux[id(x - 1, y, z)]));
        const double dyuy = 0.5 * (double(uy[id(x, y + 1, z)]) - double(uy[id(x, y - 1, z)]));
        const double dzuz = 0.5 * (double(uz[id(x, y, z + 1)]) - double(uz[id(x, y, z - 1)]));
        d.divu = std::fmax(d.divu, std::fabs(dxux + dyuy + dzuz));

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
        d.jc[0] = std::fmax(d.jc[0], std::fabs(jx));
        d.jc[1] = std::fmax(d.jc[1], std::fabs(jy));
        d.jc[2] = std::fmax(d.jc[2], std::fabs(jz));
        d.jmax  = std::fmax(d.jmax, std::sqrt(jx * jx + jy * jy + jz * jz));
      }
  // Normalise div B by k|B| with k the box's own longest wavenumber, so the
  // number means the same thing as the Orszag-Tang driver's.
  const double k = 2.0 * M_PI / (double(nx) * dl);
  if (bscale > 0) d.divb /= k * bscale;
  if (uscale > 0) d.divu /= k * uscale;
  return d;
}

//------------------------------------------------------------------------------
// Mid-plane slices in the x1-x2 plane at z = nz/2 -- the reference's Fig. 1
// middle row. Format is mhd_sphere.cu's: int32 nx, ny then nx*ny float32.
//------------------------------------------------------------------------------
static void write_raw2(const std::string& path, int nx, int ny,
                       const std::vector<float>& a) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  const std::int32_t p = nx, q = ny;
  std::fwrite(&p, 4, 1, f); std::fwrite(&q, 4, 1, f);
  std::fwrite(a.data(), 4, a.size(), f);
  std::fclose(f);
}

static void dump_slices(int nx, int ny, int nz, int idx,
                        const std::vector<Real>& ux, const std::vector<Real>& uy,
                        const std::vector<Real>& uz, const std::vector<Real>& bx,
                        const std::vector<Real>& by, const std::vector<Real>& bz) {
  auto id = [nx, ny, nz](int x, int y, int z) {
    return std::size_t(node_id(((x % nx) + nx) % nx, ((y % ny) + ny) % ny,
                               ((z % nz) + nz) % nz, nx, ny));
  };
  const int z0 = nz / 2;
  std::vector<float> u(std::size_t(nx) * ny), b(u.size()), j(u.size());
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const std::size_t n = id(x, y, z0), s = std::size_t(y) * nx + x;
      u[s] = float(std::sqrt(double(ux[n]) * double(ux[n]) +
                             double(uy[n]) * double(uy[n]) +
                             double(uz[n]) * double(uz[n])));
      b[s] = float(std::sqrt(double(bx[n]) * double(bx[n]) +
                             double(by[n]) * double(by[n]) +
                             double(bz[n]) * double(bz[n])));
      const double jx = 0.5 * (double(bz[id(x, y + 1, z0)]) - double(bz[id(x, y - 1, z0)]))
                      - 0.5 * (double(by[id(x, y, z0 + 1)]) - double(by[id(x, y, z0 - 1)]));
      const double jy = 0.5 * (double(bx[id(x, y, z0 + 1)]) - double(bx[id(x, y, z0 - 1)]))
                      - 0.5 * (double(bz[id(x + 1, y, z0)]) - double(bz[id(x - 1, y, z0)]));
      const double jz = 0.5 * (double(by[id(x + 1, y, z0)]) - double(by[id(x - 1, y, z0)]))
                      - 0.5 * (double(bx[id(x, y + 1, z0)]) - double(bx[id(x, y - 1, z0)]));
      j[s] = float(std::sqrt(jx * jx + jy * jy + jz * jz));
    }
  char tag[32]; std::snprintf(tag, sizeof tag, "_%04d.raw", idx);
  write_raw2(std::string("anim_frames/umag") + tag, nx, ny, u);
  write_raw2(std::string("anim_frames/bmag") + tag, nx, ny, b);
  write_raw2(std::string("anim_frames/jmag") + tag, nx, ny, j);
}

int main(int argc, char** argv) {
  int nx = 256, nprobe = 300, dump = 0;
  double Re = 3500.0, Ma = 0.034, tmax = 150.0, Pm = 1.0, S = 52.5;
  double sigma = 0.9, amp = 1.0e-3, seedamp = 1.0e-5;
  unsigned seed = 20260917u;
  std::string op = "cm";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-nx"    && i + 1 < argc) nx      = std::atoi(argv[++i]);
    if (a == "-re"    && i + 1 < argc) Re      = std::atof(argv[++i]);
    if (a == "-ma"    && i + 1 < argc) Ma      = std::atof(argv[++i]);
    if (a == "-pm"    && i + 1 < argc) Pm      = std::atof(argv[++i]);
    if (a == "-s"     && i + 1 < argc) S       = std::atof(argv[++i]);
    if (a == "-tmax"  && i + 1 < argc) tmax    = std::atof(argv[++i]);
    if (a == "-sigma" && i + 1 < argc) sigma   = std::atof(argv[++i]);
    if (a == "-amp"   && i + 1 < argc) amp     = std::atof(argv[++i]);
    if (a == "-seedamp" && i + 1 < argc) seedamp = std::atof(argv[++i]);
    if (a == "-seed"  && i + 1 < argc) seed    = unsigned(std::atol(argv[++i]));
    if (a == "-probes"&& i + 1 < argc) nprobe  = std::atoi(argv[++i]);
    if (a == "-dump"  && i + 1 < argc) dump    = std::atoi(argv[++i]);
    if (a == "-op"    && i + 1 < argc) op      = argv[++i];
  }

  // Their box, and the isotropic lattice that comes closest to it.
  const double Lx = 4.0 * M_PI / sigma, Ly = 20.0;
  const double dl = Lx / double(nx);
  const int ny = 2 * int(std::lround(Ly / dl * 0.5));   // even: x2 = 0 on a node
  const int nz = nx;
  const double Ly_lat = double(ny) * dl;

  const double u0  = Ma / std::sqrt(3.0);      // Ma on the jet centreline speed
  const double N0  = 1.0 / dl;                 // cells per L0
  const double nu  = u0 * N0 / Re;
  const double eta = nu / Pm;
  const double b0  = (S / (Re * Pm)) * u0;     // b0/u0 = S eta/(u0 L0) = S/(Re Pm)
  const double dt  = u0 * dl;                  // one unit of their t per N0/u0 steps
  const std::size_t T = std::size_t(tmax / dt);
  const Op which = (op == "bgk") ? Op::BGK : Op::CentralMoments;

  std::printf("MHD jet (Williams, De Rosis & Skillen; Mak et al. 2017 setup)\n");
  std::printf("  %s   D3Q27 fluid / D3Q7 field   operator %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              which == Op::BGK ? "bgk" : "cm", sizeof(Real) == 4 ? "FP32" : "FP64");
  std::printf("  grid %d x %d x %d = %.1f M nodes   dl = %.6f   %.1f cells per L0\n",
              nx, ny, nz, double(nx) * ny * nz / 1e6, dl, N0);
  std::printf("  box  %.4f x %.4f x %.4f   (reference %.4f x %.1f x %.4f)\n",
              Lx, Ly_lat, Lx, Lx, Ly, Lx);
  std::printf("  Re = %.0f   Pm = %.2f   S(t=0) = %.1f   Ma = %.3f   sigma = %.2f\n",
              Re, Pm, S, Ma, sigma);
  std::printf("  u0 = %.6e   b0 = %.6e (b0/u0 = %.4f)\n", u0, b0, b0 / u0);
  std::printf("  nu = %.6e (tau %.6f)   eta = %.6e (tau_m %.6f)\n",
              nu, 3.0 * nu + 0.5, eta, 4.0 * eta + 0.5);
  std::printf("  perturbation A = %.1e, seed rms = %.1e, hash seed %u\n",
              amp, seedamp, seed);
  std::printf("  t up to %.1f  (%zu steps, dt = %.6e)\n\n", tmax, T, dt);

  if (3.0 * nu + 0.5 < 0.5002)
    std::printf("  NOTE: tau - 1/2 = %.2e. The Orszag-Tang runs this scheme is\n"
                "        established on sat at 2.0e-3. Nothing here has been\n"
                "        measured this close to the floor.\n\n", 3.0 * nu);

  backend::Magnetic mag(nx, ny, nz, Real(eta));
  backend::Fluid    fl (nx, ny, nz, which, Real(nu));

  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  const JetPsi P{nx, ny, nz, dl, 0.5 * Ly_lat, amp, sigma,
                 seedamp * dl * std::sqrt(3.0), seed};
  fl.initialise_with(JetFluidInit{P, Real(u0)});
  mag.initialise_with(JetBInit{Real(b0)}, JetUInit{P, Real(u0)});

  std::vector<Real> rho, ux, uy, uz, bx, by, bz;
  auto sample = [&]() {
    fl.macroscopic_to_host(rho, ux, uy, uz);
    mag.field_to_host(bx, by, bz);
    return measure(ux, uy, uz, bx, by, bz, nx, ny, nz, dl);
  };

  const Diag d0 = sample();
  const double e0 = d0.eu + d0.eb, eb0 = d0.eb;
  // The initial condition is built as a curl, so this is the claim being
  // checked, not a property being reported: it must be at round-off.
  std::printf("  initial max|div u|/k|u| = %.3e   max|div B|/k|B| = %.3e\n\n",
              d0.divu, d0.divb);
  std::printf("  %8s %11s %11s %11s %11s %11s %10s %10s %10s %12s\n",
              "t", "E/E0", "E_u/E0", "E_b/E0", "gamma", "Jmax", "jx", "jy", "jz",
              "max|divB|");
  std::printf("  %8.3f %11.6f %11.6f %11.6f %11s %11.4f %10.4f %10.4f %10.4f %12.3e\n",
              0.0, 1.0, d0.eu / e0, d0.eb / e0, "-", d0.jmax / dt,
              d0.jc[0] / dt, d0.jc[1] / dt, d0.jc[2] / dt, d0.divb);

  if (dump > 0) {
    ::mkdir("anim_frames", 0755);
    std::FILE* m = std::fopen("anim_frames/meta.txt", "w");
    // N for render_slices.py, which wants that key; nx/ny beside it because
    // this box is NOT square and the renderer was written for one that is.
    // The .raw headers carry the true dimensions and the loader reads them.
    // dt is here so a renderer can put the slices into the reference's units
    // without re-deriving u0 and dl: the .raw values are LATTICE |J|, and the
    // conversion is the same J(lat)/dt the J_max column uses.
    if (m) { std::fprintf(m, "N %d\nnx %d\nny %d\nR 0\nTe 1.0\ndt %.8e\n",
                          nx, nx, ny, dt);
             std::fclose(m); }
  }

  const std::size_t probe =
      (nprobe > 0 && T / std::size_t(nprobe)) ? T / std::size_t(nprobe) : 1;
  double worst_div = d0.divb, prev_eb = d0.eb / eb0, prev_t = 0.0;
  int probe_no = 0, dframe = 0;
  if (dump > 0) { dump_slices(nx, ny, nz, dframe, ux, uy, uz, bx, by, bz); ++dframe; }
  const auto wall0 = std::chrono::steady_clock::now();

  for (std::size_t t = 1; t <= T; ++t) {
    mag.compute_field();
    fl.step();
    mag.step();
    if (t % probe == 0 || t == T) {
      const Diag d = sample();
      const double now = double(t) * dt;
      if (!d.finite) { std::printf("  DIVERGED at t = %.3f\n", now); return 1; }
      const double ebn = d.eb / eb0;
      // Their Eq. (3.1): gamma = d/dt[ (1/2) int b.b / int b0.b0 ] = (1/2) d(E_b/E_b0)/dt.
      const double gamma = 0.5 * (ebn - prev_eb) / (now - prev_t);
      prev_eb = ebn; prev_t = now;
      worst_div = std::fmax(worst_div, d.divb);
      std::printf("  %8.3f %11.6f %11.6f %11.6f %11.4f %11.4f %10.4f %10.4f %10.4f %12.3e\n",
                  now, (d.eu + d.eb) / e0, d.eu / e0, d.eb / e0, gamma,
                  d.jmax / dt, d.jc[0] / dt, d.jc[1] / dt, d.jc[2] / dt, d.divb);
      std::fflush(stdout);
      ++probe_no;
      if (dump > 0 && probe_no % dump == 0) {
        dump_slices(nx, ny, nz, dframe, ux, uy, uz, bx, by, bz);
        ++dframe;
      }
    }
  }

  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall0).count();
  std::printf("\n  Currents are in the reference's units: J = J(lat)/dt, dt = u0*dl = %.6e.\n", dt);
  std::printf("  gamma is Eq. (3.1) as written, i.e. HALF d(E_b/E_b0)/dt; E_b/E0 is\n"
              "      beside it so the other reading of that normalisation costs no rerun.\n");
  std::printf("  worst max|div B| / k|B| over the run   %.3e\n", worst_div);
  std::printf("      Not cleaned. The reference holds div B below 1e-13 with\n"
              "      Brackbill-Barnes; this scheme has no cleaning stage at all. On a\n"
              "      reconnection problem that is the first number to read, not the last.\n");
  if (dump > 0)
    std::printf("  wrote %d mid-plane frames to anim_frames/ (%d x %d, x1-x2 at z = nz/2)\n",
                dframe, nx, ny);
  std::printf("  %zu steps in %.2f s  ->  %.1f MLUPS\n",
              T, sec, double(fl.nodes()) * double(T) / sec / 1e6);
  return 0;
}
