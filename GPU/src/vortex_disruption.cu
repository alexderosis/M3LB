//==============================================================================
//  Disruption of a single vortex by a weak magnetic field -- the GPU/ twin of
//  validation/vortex_disruption.cpp.
//
//  WHY IT EXISTS.  The parent's map in Rm and M ran on a laptop and stops at
//  Rm = 8000, where its three strongest fields went non-finite on 20 cells per
//  vortex radius. Pushing the map to Rm >= 1e4 with the vortex resolved needs
//  boxes of 1600^2 to 3200^2 for 10^5 to 10^6 steps, which is a GPU's work.
//
//  SAME CASE, SAME NUMBERS.  The setup, the exact viscous check, the
//  disruption measures and every column of series.dat are defined in the
//  parent's banner and are not repeated here: this file reproduces them on
//  D3Q27 + D3Q7 with nz = 1 -- the parent's -lat d3q27 -- and writes the same
//  series.dat, header included, so tools/vortex_disruption_map.py reads either.
//  The header carries one more token, `code GPU/ <native|host> <FP32|FP64>`.
//
//      vortex   u_theta = eps (r/R0) exp[(1 - r^2/R0^2)/2], rho balancing it,
//               centred at ((N-1)/2, (N-1)/2) of a periodic (nr R0)^2 box
//      field    b = (M eps, 0), uniform
//      fluid    shifted central moments with the Maxwell stress in the
//               equilibrium (GPU/'s Op::CentralMoments, the twin of the
//               parent's MhdCentralMomentsShifted); omega_bulk = 1
//      field    Dellar's vector distribution, BGK on D3Q7
//      seed     product-form equilibrium plus the Maxwell stress for the fluid,
//               the magnetic equilibrium WITH the initial velocity for B
//      order    compute B, step the fluid, step the field (the parent's)
//
//  M = 0 IS KINEMATIC, AND IT IS BUILT THE PARENT'S WAY.  The fluid is coupled
//  to three arrays of zeros, so it runs the same MHD operator as every other
//  run with B = 0, while the field is transported at lattice strength eps. Not
//  coupling it at all would switch the fluid to the non-MHD path, which is a
//  different operator and would break the twin comparison for no gain. And the
//  velocity the field is advected with exists only once the fluid has been
//  coupled: ux_device() is null before that (CLAUDE.md, "an opt-in buffer plus
//  a defensive null-check"), so the driver aborts if it ever sees a null.
//
//  WHAT THE PARENT HAS AND THIS DOES NOT: -lat d2q9 (GPU/ is D3Q27 only) and
//  -off. -ellip is kept, since it costs a parameter. -shifted is this file's
//  own: GPU/'s fluid can store f_i - w_i, which the parent's operator cannot,
//  and in FP32 that is the difference between resolving a weak Maxwell stress
//  and rounding it (PRECISION below).
//
//  CROSS-CHECKED AGAINST THE PARENT, 2026-11-19: this file's FP64 host build
//  against validation/vortex_disruption -lat d3q27 (Kokkos Threads, FP64), at
//  R0 = 10, a box of 20 R0, Re = Rm = 500, to t = 20, for M = 0.05 and M = 0.
//  ALL FIFTEEN COLUMNS OF series.dat AGREE TO EVERY ONE OF THEIR TEN PRINTED
//  DIGITS, at all 21 probes of both runs. The one difference is Delta at t = 0,
//  which is zero to round-off in both (1.0e-15 against 2.0e-15). The kinematic
//  exact check reads 1.0199 in both: R0 = 10 is coarse, and the gate is 5 %.
//  That is the same arithmetic run twice, so it certifies the port, not the
//  physics (CLAUDE.md, measurement discipline); the physics is the parent's,
//  measured in its banner. The host build ran at 4.5 MLUPS, serial.
//
//  PRECISION, MEASURED 2026-11-19 on the host builds (R0 = 20, a box of 10 R0
//  to fit a laptop, Pm = 1), FP64 against FP32 and FP32 with -shifted:
//
//      Re = Rm = 4000, M^2 Rm = 1.77      FP64      FP32      FP32-shifted
//        Delta at t = 3 Rm^(1/3) = 47.6   0.19214   0.19213   0.19214
//        Delta at t = 12 Rm^(1/3)         0.677     0.563     0.729
//        largest Delta in the run         0.698     0.580     0.729
//
//  THE WIND-UP IS SAFE IN FP32 AND THE SECOND STAGE IS NOT REPRODUCIBLE IN IT.
//  The early reading agrees to four digits, so the weak Maxwell stress is not
//  being lost to rounding. The second stage is chaotic, and rounding alone moves
//  a single trajectory by 8-17 % in Delta -- more than the 7 % the parent found
//  between two grids. It is scatter, not a bias (raw FP32 lands low, shifted
//  high), but a threshold read off single runs carries it. So: FP64 for any run
//  whose second stage is measured, or the stage treated as an ensemble.
//  -shifted does not cure it and costs nothing.
//
//      Re = Rm = 16000, M^2 Rm = 1.25, R0 = 20: NON-FINITE in all three, at
//      t = 97 (FP64), 131 (FP32) and 93 (shifted), the three already 15 % apart
//      in Delta by t = 75.6.
//
//  tau_f - 1/2 is 1.9e-4 there, below what this scheme survives; the parent's
//  map lost its strongest fields at Rm = 8000 on the same R0 (3.8e-4), and kept
//  every run at Rm = 4000 (7.5e-4). Scale R0 with Rm: R0 >= Rm / 200 keeps the
//  margin the map had. That is an inference from those two rungs, not a tested
//  bound.
//
//  COST AND COMPILE TIME.  Diagnostics run on the host every -probe time
//  units: each copies four fields and walks the box several times, a few
//  seconds at 3200^2 on a cluster CPU, so on large boxes raise -probe. The
//  general binary compiles every operator and force kind; this case uses
//  central moments and no force, so on a cluster build with
//  -DLBM_ONLY_OP=cm -DLBM_ONLY_FORCE=none.
//
//  THE DEVICE BUILD, 2026-11-19 (GPU/csf3/vortex_disruption_verify.sub, job
//  24933077 on a CSF3 A100; log and series in results/Q_vortex_disruption/
//  gpu_port/device_verify/). It compiles under nvcc with -DLBM_ONLY_OP=cm
//  -DLBM_ONLY_FORCE=none and reproduces the twin runs above: worst column
//  3.4e-15 (M = 0.05) and 4.6e-14 (M = 0) of its scale against the Kokkos
//  parent, Delta at t = 0 in both, and every other printed value identical --
//  to the host build in 671 of 672. FP64 throughput at the Rm = 16000 rung's
//  size (1600^2): 1599 MLUPS without probes, the probes 50 ns per node each,
//  so the map job's largest element (Rm = 32000, 3200^2) is ~2.2 h, not the
//  3.4 h its header estimated from tg_mhd's 1100 MLUPS.
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/ehd.cuh"          // Field: the host/device-neutral N-element array

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

// Everything a kernel sees is at file scope: nvcc restricts types from unnamed
// namespaces as template arguments of __global__ functions.

// The vortex and the density that balances it, in the same operations and the
// same order as the parent's vortex(), so the FP64 twins seed identical bits.
LBM_HD inline void vd_vortex(Real dx, Real dy, Real R0, Real U, Real e,
                             Real& u, Real& v, Real& rho) {
  const Real ics = Real(3);                     // 1/cs^2 on D3Q27
  const Real s2 = ((Real(1) + e) * dx * dx + (Real(1) - e) * dy * dy) / (R0 * R0);
  const Real g = U / R0 * exp(Real(0.5) * (Real(1) - s2));
  u = -g * (Real(1) - e) * dy;
  v =  g * (Real(1) + e) * dx;
  rho = exp(-Real(0.5) * ics * U * U * exp(Real(1) - s2));
}

struct VdPopulations {
  Real R0, U, xc, e, B0;
  bool stress;                                  // false: kinematic, no Maxwell stress
  LBM_HD void operator()(int x, int y, int, Real f[27]) const {
    Real u[3] = {Real(0), Real(0), Real(0)}, rho = Real(1);
    vd_vortex(Real(x) - xc, Real(y) - xc, R0, U, e, u[0], u[1], rho);
    product_equilibrium(rho, u, f);
    if (stress) {
      const Real B[3] = {B0, Real(0), Real(0)};
      for (int i = 0; i < 27; ++i) f[i] += maxwell(i, B);
    }
  }
};

struct VdB {
  Real B0;
  LBM_HD void operator()(int, int, int, Real B[3]) const {
    B[0] = B0; B[1] = Real(0); B[2] = Real(0);
  }
};

struct VdU {
  Real R0, U, xc, e;
  LBM_HD void operator()(int x, int y, int, Real u[3]) const {
    Real rho = Real(1);
    vd_vortex(Real(x) - xc, Real(y) - xc, R0, U, e, u[0], u[1], rho);
    u[2] = Real(0);
  }
};

struct VdOpts {
  int R0 = 30;               // core radius, cells
  int nr = 20;               // box side, in R0
  double Re = 500.0;
  double Rm = -1.0;          // < 0: follow Re (Pm = 1)
  double M = 0.05;           // 0 = kinematic
  double tmax = 50.0;        // in R0 / eps
  double u0 = 0.05;          // eps, the peak speed in lattice units
  double probe = 0.5;
  double wbulk = 1.0;
  double ellip = 0.0;        // an m = 2 seed, as the parent's -ellip
  int frames = 0;            // -frames K: vorticity and j dumped at K + 1 times
  bool shifted = false;      // -shifted: store f_i - w_i (see the banner)
  std::string out = ".";
};

struct VdRun {
  int N = 0;
  double R0 = 0, dt = 0, B0 = 0, Rm = 0, nu = 0, eta = 0, xc = 0;
  bool kinematic = false;
  std::size_t n_max = 0, stride = 0, fstride = 0;
};

static VdRun vd_plan(const VdOpts& o) {
  VdRun r;
  r.N = o.nr * o.R0;
  r.R0 = double(o.R0);
  r.xc = 0.5 * double(r.N - 1);
  r.dt = o.u0 / r.R0;                          // R0/eps per step
  r.Rm = (o.Rm > 0) ? o.Rm : o.Re;
  r.nu  = o.u0 * r.R0 / o.Re;
  r.eta = o.u0 * r.R0 / r.Rm;
  r.kinematic = !(o.M > 0);
  r.B0 = r.kinematic ? o.u0 : o.M * o.u0;
  r.n_max  = std::size_t(std::ceil(o.tmax / r.dt));
  r.stride = std::max<std::size_t>(1, std::size_t(std::llround(o.probe / r.dt)));
  r.fstride = (o.frames > 0) ? std::max<std::size_t>(1, r.n_max / std::size_t(o.frames)) : 0;
  return r;
}

// mkdir -p, without <filesystem>: a cluster's host compiler may predate it.
static void vd_mkdirs(const std::string& path) {
  std::string cur;
  for (std::size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!cur.empty()) mkdir(cur.c_str(), 0755);
    }
    if (i < path.size()) cur += path[i];
  }
}

//------------------------------------------------------------------------------
// The host fields of one probe, row-major (y N + x) -- which is node_id at
// nz = 1 -- in lattice units, with the initial pattern beside them. The parent's
// Host, unchanged.
//------------------------------------------------------------------------------
struct VdHost {
  int N = 0;
  std::vector<double> ux, uy, rho, bx, by, ut, vt;
  VdHost(int n, double R0, double xc, double U) : N(n) {
    const std::size_t s = std::size_t(n) * std::size_t(n);
    ux.assign(s, 0.0); uy.assign(s, 0.0); rho.assign(s, 0.0);
    bx.assign(s, 0.0); by.assign(s, 0.0); ut.assign(s, 0.0); vt.assign(s, 0.0);
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double dx = double(x) - xc, dy = double(y) - xc;
        const double g = U / R0 * std::exp(0.5 * (1.0 - (dx * dx + dy * dy) / (R0 * R0)));
        ut[at(x, y)] = -g * dy;
        vt[at(x, y)] =  g * dx;
      }
  }
  std::size_t at(int x, int y) const {
    return std::size_t(((y + N) % N) * N + ((x + N) % N));
  }
};

struct VdProbe {
  double ek, b2, bmax, coh, ang, wmax, jmax, Lb, divb, osc, ma, mass, ens, dow;
  bool finite;
};

// The parent's measure(), line for line: lengths in R0, speeds in eps, the field
// in B0. If wf / jf are given they receive the vorticity and the current.
static VdProbe vd_measure(const VdHost& H, const VdOpts& o, const VdRun& r, double t,
                          std::vector<float>* wf = nullptr, std::vector<float>* jf = nullptr) {
  const int N = H.N;
  const double iu = 1.0 / o.u0, ib = 1.0 / r.B0;
  const double idu = iu * r.R0, idb = ib * r.R0;
  if (wf) wf->assign(std::size_t(N) * std::size_t(N), 0.0f);
  if (jf) jf->assign(std::size_t(N) * std::size_t(N), 0.0f);

  double su2 = 0, sb2 = 0, sj2 = 0, sdot = 0, s00 = 0, sang = 0, sang0 = 0;
  double smass = 0, sdj = 0, wmax = 0, jmax = 0, dmax = 0, umax = 0, bmax = 0;
  bool finite = true;
  auto J = [&](int x, int y) {
    return (0.5 * (H.by[H.at(x + 1, y)] - H.by[H.at(x - 1, y)]) -
            0.5 * (H.bx[H.at(x, y + 1)] - H.bx[H.at(x, y - 1)])) * idb;
  };
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      const std::size_t k = H.at(x, y);
      const double u = H.ux[k], v = H.uy[k], a = H.bx[k], b = H.by[k];
      const double w = (0.5 * (H.uy[H.at(x + 1, y)] - H.uy[H.at(x - 1, y)]) -
                        0.5 * (H.ux[H.at(x, y + 1)] - H.ux[H.at(x, y - 1)])) * idu;
      const double j = J(x, y);
      const double dv = (0.5 * (H.bx[H.at(x + 1, y)] - H.bx[H.at(x - 1, y)]) +
                         0.5 * (H.by[H.at(x, y + 1)] - H.by[H.at(x, y - 1)])) * idb;
      if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(a) ||
          !std::isfinite(b) || !std::isfinite(w) || !std::isfinite(j))
        finite = false;
      const double jx = J(x + 1, y) - j, jy = J(x, y + 1) - j;
      sdj += 0.5 * (jx * jx + jy * jy);
      const double dx = double(x) - r.xc, dy = double(y) - r.xc;
      su2 += u * u + v * v;
      sb2 += a * a + b * b;
      sj2 += j * j;
      sdot += u * H.ut[k] + v * H.vt[k];
      s00  += H.ut[k] * H.ut[k] + H.vt[k] * H.vt[k];
      sang  += dx * v - dy * u;
      sang0 += dx * H.vt[k] - dy * H.ut[k];
      smass += H.rho[k];
      wmax = std::max(wmax, std::abs(w));
      jmax = std::max(jmax, std::abs(j));
      dmax = std::max(dmax, std::abs(dv));
      umax = std::max(umax, std::hypot(u, v));
      bmax = std::max(bmax, std::hypot(a, b));
      if (wf) (*wf)[k] = float(w);
      if (jf) (*jf)[k] = float(j);
    }
  const double c = 1.0 / (double(N) * double(N));
  VdProbe p;
  p.ek   = 0.5 * su2 * iu * iu * c;
  p.b2   = sb2 * ib * ib * c;
  p.bmax = bmax * ib;
  p.coh  = sdot / s00;
  p.ang  = sang / sang0;
  p.wmax = wmax; p.jmax = jmax; p.divb = dmax;
  const bool has_j = sj2 > 1e-24 * sb2 * ib * ib;
  p.Lb   = has_j ? std::sqrt(sb2 * ib * ib / sj2) : 0.0;
  p.osc  = has_j ? sdj / (4.0 * sj2) : 0.0;
  p.ma   = umax * std::sqrt(3.0);
  p.mass = smass * c;
  p.finite = finite;

  // The two gradient measures against the exact viscous vortex at time t,
  // differenced on this grid with the run's own stencil (the parent's comment
  // in measure() defines them).
  {
    const double s2 = r.R0 * r.R0 * (1.0 + 2.0 * t / o.Re);
    const double amp = o.u0 * std::exp(0.5) / r.R0 * (r.R0 * r.R0 / s2) * (r.R0 * r.R0 / s2);
    std::vector<double> ur(H.ux.size()), vr(H.ux.size());
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = double(x) - r.xc, dy = double(y) - r.xc;
        const double g = amp * std::exp(-0.5 * (dx * dx + dy * dy) / s2);
        ur[H.at(x, y)] = -g * dy;
        vr[H.at(x, y)] =  g * dx;
      }
    auto ow = [&](const std::vector<double>& U, const std::vector<double>& V, int x, int y,
                  double& w2) {
      const double ux_ = 0.5 * (U[H.at(x + 1, y)] - U[H.at(x - 1, y)]);
      const double uy_ = 0.5 * (U[H.at(x, y + 1)] - U[H.at(x, y - 1)]);
      const double vx_ = 0.5 * (V[H.at(x + 1, y)] - V[H.at(x - 1, y)]);
      const double vy_ = 0.5 * (V[H.at(x, y + 1)] - V[H.at(x, y - 1)]);
      const double ns = ux_ - vy_, ss = vx_ + uy_, vz = vx_ - uy_;
      w2 = vz * vz;
      return ns * ns + ss * ss - vz * vz;
    };
    std::vector<double> Wr(H.ux.size());
    double zr = 0, z = 0, rms = 0;
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        double w2;
        Wr[H.at(x, y)] = ow(ur, vr, x, y, w2);
        zr += w2;
        rms += Wr[H.at(x, y)] * Wr[H.at(x, y)];
      }
    const double thr = -0.2 * std::sqrt(rms * c);
    double num = 0, den = 0;
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        double w2;
        const double W = ow(H.ux, H.uy, x, y, w2);
        z += w2;
        if (Wr[H.at(x, y)] < thr) { num += W; den += Wr[H.at(x, y)]; }
      }
    p.ens = (zr > 0) ? z / zr : 0.0;
    p.dow = (den < 0) ? 1.0 - num / den : 0.0;
  }
  return p;
}

static void vd_write_field(const std::string& path, int N, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const int nx = N, ny = N;
  o.write(reinterpret_cast<const char*>(&nx), sizeof nx);
  o.write(reinterpret_cast<const char*>(&ny), sizeof ny);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

int main(int argc, char** argv) {
  VdOpts o;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](double& x) { if (i + 1 < argc) x = std::atof(argv[++i]); else bad = true; };
    if (a == "-r0" && i + 1 < argc)           o.R0 = std::atoi(argv[++i]);
    else if (a == "-nr" && i + 1 < argc)      o.nr = std::atoi(argv[++i]);
    else if (a == "-re")                      val(o.Re);
    else if (a == "-rm")                      val(o.Rm);
    else if (a == "-M")                       val(o.M);
    else if (a == "-tmax")                    val(o.tmax);
    else if (a == "-eps")                     val(o.u0);
    else if (a == "-probe")                   val(o.probe);
    else if (a == "-wb")                      val(o.wbulk);
    else if (a == "-ellip")                   val(o.ellip);
    else if (a == "-frames" && i + 1 < argc)  o.frames = std::atoi(argv[++i]);
    else if (a == "-shifted")                 o.shifted = true;
    else if (a == "-out" && i + 1 < argc)     o.out = argv[++i];
    else { std::fprintf(stderr, "\nERROR: unknown or incomplete argument %s\n", a.c_str()); bad = true; }
  }
  if (o.R0 < 4 || o.nr < 6 || o.Re <= 0 || o.u0 <= 0 || o.tmax <= 0 || o.probe <= 0 ||
      o.M < 0 || !(o.wbulk > 0 && o.wbulk < 2)) {
    std::fprintf(stderr, "\nERROR: need -r0 >= 4, -nr >= 6, positive -re -eps -tmax -probe, "
                         "-M >= 0, -wb in (0, 2)\n");
    bad = true;
  }
  if (bad) {
    std::fprintf(stderr,
        "usage: vortex_disruption [-r0 CELLS] [-nr BOX_IN_R0] [-re Re] [-rm Rm] [-M M]\n"
        "                         [-tmax T] [-eps U] [-probe DT] [-wb W] [-frames K]\n"
        "                         [-ellip E] [-shifted] [-out DIR]\n"
        "NOTHING WAS RUN.\n\n");
    return 1;
  }

  const VdRun r = vd_plan(o);
  const int N = r.N;
  const long NT = long(N) * long(N);
  const char* where = backend::on_device ? "native" : "host";
  const char* prec = sizeof(Real) == 4 ? (o.shifted ? "FP32-shifted" : "FP32")
                                       : (o.shifted ? "FP64-shifted" : "FP64");
  const std::string pairing = "D3Q27 + D3Q7 (nz = 1)";

  std::printf("Disruption of a single vortex by a weak magnetic field   GPU/ %s   %s\n",
              backend::on_device ? "CUDA native" : "HOST reference", prec);
  std::printf("lattices %s\n", pairing.c_str());
  std::printf("  R0 = %d cells   box %d x %d (%d R0)   Re = %.6g   Rm = %.6g   Pm = %.4g"
              "   M = %.6g%s\n", o.R0, N, N, o.nr, o.Re, r.Rm, r.Rm / o.Re, o.M,
              r.kinematic ? "   (KINEMATIC: no Lorentz force)" : "");
  if (o.ellip != 0.0)
    std::printf("  SEEDED: ellipticity %g on the initial vortex (an m = 2 perturbation)\n", o.ellip);
  std::printf("  M^2 Rm = %.4g   M^2 Rm^(2/3) = %.4g   M Rm = %.4g\n",
              o.M * o.M * r.Rm, o.M * o.M * std::cbrt(r.Rm * r.Rm), o.M * r.Rm);
  std::printf("  eps = %.4g (Ma = %.4f)   dt = %.6e   t_max = %.6g (%zu steps)\n",
              o.u0, o.u0 * std::sqrt(3.0), r.dt, o.tmax, r.n_max);
  // tau - 1/2 = nu / cs^2: 1/3 for the D3Q27 fluid, 1/4 for the D3Q7 field.
  std::printf("  nu = %.6e   tau_f - 1/2 = %.3e   eta = %.6e   tau_m - 1/2 = %.3e\n",
              r.nu, 3.0 * r.nu, r.eta, 4.0 * r.eta);
  std::printf("  Re_cell = %.1f   Rm_cell = %.1f   (Re / R0: only cells move these)\n",
              o.u0 / r.nu, o.u0 / r.eta);
  std::printf("  wound sheet R0 Rm^(-1/3) = %.1f cells;   expelled layer R0 Rm^(-1/2)"
              " = %.1f cells\n", r.R0 * std::cbrt(1.0 / r.Rm), r.R0 / std::sqrt(r.Rm));
  std::printf("  probe every %zu steps (%.4f)%s\n\n", r.stride, double(r.stride) * r.dt,
              r.fstride ? "   frames on" : "");

  //---- solvers ------------------------------------------------------------------
  backend::Magnetic mag(N, N, 1, Real(r.eta));
  backend::Fluid    fl (N, N, 1, Op::CentralMoments, Real(r.nu), Real(o.wbulk));
  fl.set_shifted(o.shifted);                    // before any seeding
  Field zx(NT), zy(NT), zz(NT);                 // the kinematic run's B = 0
  if (r.kinematic) fl.couple_magnetic(zx.data(), zy.data(), zz.data());
  else             fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  if (!fl.ux_device() || !fl.uy_device() || !fl.uz_device()) {
    std::fprintf(stderr, "vortex_disruption: the fluid has no velocity output after "
                         "couple_magnetic -- the field would never be advected. Aborting.\n");
    return 3;
  }
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  const Real R0r = Real(r.R0), Ur = Real(o.u0), xcr = Real(r.xc), er = Real(o.ellip);
  fl.seed_populations_with(VdPopulations{R0r, Ur, xcr, er, Real(r.B0), !r.kinematic});
  mag.initialise_with(VdB{Real(r.B0)}, VdU{R0r, Ur, xcr, er});

  std::vector<Real> hrho, hux, huy, huz, hbx, hby, hbz;
  auto pull = [&](VdHost& H) {
    fl.macroscopic_to_host(hrho, hux, huy, huz);
    mag.field_to_host(hbx, hby, hbz);
    for (std::size_t k = 0; k < std::size_t(NT); ++k) {
      H.ux[k] = double(hux[k]); H.uy[k] = double(huy[k]); H.rho[k] = double(hrho[k]);
      H.bx[k] = double(hbx[k]); H.by[k] = double(hby[k]);
    }
  };
  auto step = [&]() { mag.compute_field(); fl.step(); mag.step(); };

  //---- the loop, the probes, the series and the verdict (the parent's march) ----
  vd_mkdirs(o.out);
  std::ofstream series(o.out + "/series.dat");
  series << "# vortex_disruption  lattices " << pairing << "  R0 " << o.R0 << "  nr "
         << o.nr << "  Re " << o.Re << "  Rm " << r.Rm << "  M " << o.M
         << (r.kinematic ? " (kinematic)" : "") << "  eps " << o.u0
         << "  code GPU/ " << where << " " << prec
         << "\n# t EK B2 bmax coh surv ang wmax jmax Lb divb osc Ma mass ens dow\n";
  series.precision(10);

  VdHost H(N, r.R0, r.xc, o.u0);
  std::vector<float> wf, jf;
  int frame = 0;
  double t_now = 0.0;
  auto dump = [&]() {
    char fn[512];
    pull(H);
    vd_measure(H, o, r, t_now, &wf, &jf);
    std::snprintf(fn, sizeof fn, "%s/w_%04d.bin", o.out.c_str(), frame);
    vd_write_field(fn, N, wf);
    std::snprintf(fn, sizeof fn, "%s/j_%04d.bin", o.out.c_str(), frame);
    vd_write_field(fn, N, jf);
    ++frame;
  };
  auto exact = [&](double t) { return 1.0 / ((1.0 + t / o.Re) * (1.0 + t / o.Re)); };

  std::printf("%-7s %-9s %-9s %-8s %-9s %-9s %-9s %-8s %-9s %-8s %-9s %-9s %-8s %-7s\n",
              "t", "EK", "B2", "bmax", "coh", "surv", "dow", "ens", "ang", "wmax", "jmax",
              "divb", "osc", "Ma");
  std::printf("%s\n", std::string(130, '-').c_str());
  double ek0 = 1.0;
  auto report = [&](double t, const VdProbe& p) {
    const double sv = p.coh / exact(t);
    std::printf("%-7.2f %-9.6f %-9.5f %-8.3f %-9.6f %-9.6f %-9.6f %-8.5f %-9.6f %-8.4f "
                "%-9.3f %-9.2e %-8.5f %-7.4f\n",
                t, p.ek / ek0, p.b2, p.bmax, p.coh, sv, p.dow, p.ens, p.ang, p.wmax, p.jmax,
                p.divb, p.osc, p.ma);
    std::fflush(stdout);
    series << t << ' ' << p.ek / ek0 << ' ' << p.b2 << ' ' << p.bmax << ' ' << p.coh
           << ' ' << sv << ' ' << p.ang << ' ' << p.wmax << ' ' << p.jmax << ' ' << p.Lb
           << ' ' << p.divb << ' ' << p.osc << ' ' << p.ma << ' ' << p.mass << ' '
           << p.ens << ' ' << p.dow << '\n';
    series.flush();
  };

  pull(H);
  const VdProbe p0 = vd_measure(H, o, r, 0.0);
  ek0 = p0.ek;
  report(0.0, p0);
  if (r.fstride) dump();

  double bmax_peak = p0.bmax, t_peak = 0.0, osc_max = 0.0, div_worst = 0.0;
  double surv_min = 1.0, t_surv = 0.0, blow_t = -1.0, t_last = 0.0, probe_s = 0.0;
  VdProbe last = p0;
  bool told = false;
  backend::sync();
  const auto w0 = std::chrono::steady_clock::now();
  for (std::size_t t = 0; t < r.n_max; ++t) {
    step();
    t_now = double(t + 1) * r.dt;
    if (r.fstride && (t + 1) % r.fstride == 0) dump();
    if ((t + 1) % r.stride == 0 || t + 1 == r.n_max) {
      // Kernel launches are asynchronous: drain the queue before timing the
      // probe, so its share is the probe's and not the steps'.
      backend::sync();
      const auto q0 = std::chrono::steady_clock::now();
      const double tt = t_now;
      pull(H);
      const VdProbe p = vd_measure(H, o, r, tt);
      report(tt, p);
      probe_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
      if (!p.finite || p.ma > 0.5) { blow_t = tt; break; }
      last = p; t_last = tt;
      if (p.bmax > bmax_peak) { bmax_peak = p.bmax; t_peak = tt; }
      const double sv = p.coh / exact(tt);
      if (sv < surv_min) { surv_min = sv; t_surv = tt; }
      osc_max = std::max(osc_max, p.osc);
      if (p.jmax > 0) div_worst = std::max(div_worst, p.divb / p.jmax);
      if (!told) {
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - w0).count();
        std::printf("  [%.1f MLUPS so far, probes included; the whole run is about %.1f min]\n",
                    double(t + 1) * double(NT) / el * 1e-6,
                    el * double(r.n_max) / double(t + 1) / 60.0);
        told = true;
      }
    }
  }
  backend::sync();
  const double el = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - w0).count();
  const double steps_done = (blow_t > 0 ? blow_t : t_last) / r.dt;

  std::printf("\n");
  if (blow_t > 0)
    std::printf("  BLOW-UP at t = %.3f (non-finite field or Ma > 0.5); "
                "last usable probe t = %.3f\n", blow_t, t_last);
  else
    std::printf("  ran to t = %.3f\n", t_last);
  std::printf("  peak field bmax = %.3f B0 at t = %.2f   (Rm^(1/3) = %.2f)\n",
              bmax_peak, t_peak, std::cbrt(r.Rm));
  std::printf("  survival at the last probe = %.5f   lowest = %.5f at t = %.2f   "
              "angular momentum = %.5f\n", last.coh / exact(t_last), surv_min, t_surv,
              last.ang);
  std::printf("  worst osc = %.5f   (resolved: << 1; checkerboard: 1)   "
              "worst divb / jmax = %.2e\n", osc_max, div_worst);
  std::printf("  mass drift %.3e   wall clock %.1f s, of which probes %.1f s   "
              "%.1f MLUPS without them\n", last.mass / p0.mass - 1.0, el, probe_s,
              (el > probe_s) ? steps_done * double(NT) / (el - probe_s) * 1e-6 : 0.0);

  int rc = (blow_t > 0) ? 1 : 0;
  if (r.kinematic && o.ellip != 0.0)
    std::printf("  no exact check: the vortex was seeded with an ellipticity of %g\n", o.ellip);
  if (r.kinematic && o.ellip == 0.0 && blow_t < 0 && t_last > 0) {
    const double ratio = o.Re * (1.0 / std::sqrt(last.coh) - 1.0) / t_last;
    const bool ok = std::abs(ratio - 1.0) < 0.05;
    std::printf("  viscosity delivered / viscosity set = %.4f   (exact: coh = "
                "(1 + t/Re)^-2; gate 5%%)   %s\n", ratio, ok ? "PASS" : "FAIL");
    if (!ok) rc = 1;
  }
  return rc;
}
