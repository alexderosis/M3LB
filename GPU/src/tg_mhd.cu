//==============================================================================
//  Taylor-Green MHD in a CLOSED box -- the CUDA twin of
//  ../demonstrator/tg_mhd.cpp.
//
//  Read that banner first: the problem (Pouquet et al. 2010's Taylor-Green MHD
//  fields in their fundamental box [0, pi]^3), the A / B / C / B' / P runs one
//  flag apart, the units (v0 = 1, h = pi/(N-1), nu = eta = 1/Re, one step is
//  u0 h of their time), the diagnostics and the reference numbers are all
//  argued there and are not repeated. This file shares no headers with it and is
//  supposed to agree with it; where they do not, one of them is wrong.
//
//  WHAT IS THE SAME, deliberately, so that a disagreement means something:
//
//    * the initial state, node for node, INCLUDING the seed: the fluid at the
//      product-form equilibrium PLUS the Maxwell stress (seed_populations_with,
//      which exists for this), the field at its equilibrium WITH the initial
//      velocity, and a no-slip box's wall nodes at rest;
//    * the walls: SpecNode (free slip) or RegWall with the local edge closure
//      (no slip, set_fd_corners(false)), and the on-node parity wall for the
//      field (set_parity_walls, conducting or pseudo-vacuum);
//    * the diagnostics, formula for formula: trapezoidal weights in the box,
//      derivatives at a wall node closed by the extension that wall implies,
//      direction-averaged moments before the skewness ratio;
//    * the output formats: series.dat with the same columns, profile.dat (the
//      dissipation against wall distance, which f_w is read from) with its
//      viscous and Ohmic parts in profile_visc.dat and profile_ohm.dat, anim_frames/
//      for the renderers, vti/ on the real [0, pi] coordinates, raw/ volumes.
//
//  WHAT IS NOT: the operator is written out in closed form here and reached
//  through a moment transform in the parent, so the two agree to round-off in
//  FP64 at the start of a run and drift apart at the rate the flow amplifies
//  round-off after that. Compare early, and compare in FP64.
//
//  COST ON A DEVICE. Every probe copies seven full fields to the host and walks
//  every node there: at 512^3 that is 3.7 GB and ~10^9 host operations, which
//  can cost more than the steps between probes. Use a coarse -probe on large
//  grids; the time series needs tens of rows, not thousands.
//
//    usage: tg_mhd [-n N] [-geom box|periodic] [-vwall noslip|slip]
//                  [-mwall cond|pv] [-ic tgc|tgi|tga|hydro] [-re R] [-pm P]
//                  [-u0 U] [-tmax T] [-probe dt] [-vti dt] [-dump dt]
//                  [-dumpvol] [-volstride S] [-raw dt] [-out DIR] [-force]
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/vti.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/stat.h>

using namespace lbm;

namespace {

constexpr double PI = 3.14159265358979323846;

enum ICKind : int { TGC = 0, TGI = 1, TGA = 2, Hydro = 3 };

struct Opts {
  int N = 65;
  bool periodic = false, noslip = true, conducting = true, force = false;
  int ic = TGC;
  double re = 500.0, pm = 1.0, u0 = 0.05, tmax = 10.0;
  double probe = 0.05, vti = 0.0, dump = 0.0, raw = 0.0;
  bool dumpvol = false;
  int volstride = 1;
  int threads = 0;              // host diagnostics; 0 = SLURM allocation or machine
  std::string out;
};

// The initial state at node (x, y, z). Plain structs with LBM_HD operators, not
// lambdas: they are launched as kernel arguments, and nvcc's extended-lambda
// rules are the recurring trap this tree records.
// `pad` is the Solid shell's width in STORAGE coordinates (1 for a box, 0 for
// the periodic twin); `at` takes box coordinates and the functors below
// subtract it.
struct TGState {
  int ic;
  double h, u0, b0;      // b0 in LATTICE units
  bool rest_walls;
  int nb;
  int pad;
  LBM_HD void at(int x, int y, int z, double u[3], double b[3]) const {
    const double X = h * double(x), Y = h * double(y), Z = h * double(z);
    u[0] =  u0 * sin(X) * cos(Y) * cos(Z);
    u[1] = -u0 * cos(X) * sin(Y) * cos(Z);
    u[2] = 0.0;
    if (rest_walls && (x == 0 || y == 0 || z == 0 || x == nb - 1 || y == nb - 1 || z == nb - 1))
      u[0] = u[1] = 0.0;
    b[0] = b[1] = b[2] = 0.0;
    if (ic == TGC) {
      b[0] =  b0 * sin(2 * X) * cos(2 * Y) * cos(2 * Z);
      b[1] =  b0 * cos(2 * X) * sin(2 * Y) * cos(2 * Z);
      b[2] = -2 * b0 * cos(2 * X) * cos(2 * Y) * sin(2 * Z);
    } else if (ic == TGI) {
      b[0] =  b0 * cos(X) * sin(Y) * sin(Z);
      b[1] =  b0 * sin(X) * cos(Y) * sin(Z);
      b[2] = -2 * b0 * sin(X) * sin(Y) * cos(Z);
    } else if (ic == TGA) {
      b[0] =  b0 * cos(2 * X) * sin(2 * Y) * sin(2 * Z);
      b[1] = -b0 * sin(2 * X) * cos(2 * Y) * sin(2 * Z);
    }
  }
};

struct TGField {
  TGState s;
  LBM_HD void operator()(int x, int y, int z, Real B[3]) const {
    double u[3], b[3];
    s.at(x - s.pad, y - s.pad, z - s.pad, u, b);
    B[0] = Real(b[0]); B[1] = Real(b[1]); B[2] = Real(b[2]);
  }
};

struct TGVelocity {
  TGState s;
  LBM_HD void operator()(int x, int y, int z, Real U[3]) const {
    double u[3], b[3];
    s.at(x - s.pad, y - s.pad, z - s.pad, u, b);
    U[0] = Real(u[0]); U[1] = Real(u[1]); U[2] = Real(u[2]);
  }
};

// The MHD equilibrium the central-moment operator relaxes toward: product form
// plus the Maxwell stress. See the parent's seed for why initialise_with is not
// enough in a coupled case.
struct TGPopulations {
  TGState s;
  LBM_HD void operator()(int x, int y, int z, Real f[27]) const {
    double u[3], b[3];
    s.at(x - s.pad, y - s.pad, z - s.pad, u, b);
    const Real ur[3] = {Real(u[0]), Real(u[1]), Real(u[2])};
    const Real br[3] = {Real(b[0]), Real(b[1]), Real(b[2])};
    product_equilibrium(Real(1), ur, f);
    for (int i = 0; i < 27; ++i) f[i] += maxwell(i, br);
  }
};

std::uint8_t box_faces(int x, int y, int z, int N) {
  std::uint8_t m = SpecNone;
  if (x == 0) m |= SpecXm; else if (x == N - 1) m |= SpecXp;
  if (y == 0) m |= SpecYm; else if (y == N - 1) m |= SpecYp;
  if (z == 0) m |= SpecZm; else if (z == N - 1) m |= SpecZp;
  return m;
}

std::uint8_t box_normal(int x, int y, int z, int N) {
  int nf = 0;
  std::uint8_t c = NrmNone;
  if (x == 0) { ++nf; c = NrmXm; } else if (x == N - 1) { ++nf; c = NrmXp; }
  if (y == 0) { ++nf; c = NrmYm; } else if (y == N - 1) { ++nf; c = NrmYp; }
  if (z == 0) { ++nf; c = NrmZm; } else if (z == N - 1) { ++nf; c = NrmZp; }
  return nf >= 2 ? std::uint8_t(NrmCorner) : c;
}

// Host fields and the wall-implied derivative rule -- the parent's HostFields.
enum Ext : int { ExtPeriodic, ExtEven, ExtOdd, ExtOneSided };

struct HostFields {
  int L = 0;                  // LOGICAL extent: the box, or the periodic twin
  int LS = 0, pad = 0;        // storage extent and the Solid shell's width
  bool periodic = false, noslip = true, conducting = true;
  double su = 1.0;
  const std::vector<Real>* f[7] = {};     // ux uy uz bx by bz rho

  double v(int c, int x, int y, int z) const {
    if (periodic) { x = (x + L) % L; y = (y + L) % L; z = (z + L) % L; }
    return double((*f[c])[std::size_t(node_id(x + pad, y + pad, z + pad, LS, LS))]) *
           (c < 6 ? su : 1.0);
  }
  int ext(int c, int k) const {
    if (periodic) return ExtPeriodic;
    if (c < 3) return noslip ? ExtOneSided : (c == k ? ExtOdd : ExtEven);
    if (c < 6) {
      const bool normal = (c - 3 == k);
      return (normal == conducting) ? ExtOdd : ExtEven;
    }
    return noslip ? ExtOneSided : ExtEven;
  }
  double d(int c, int k, int x, int y, int z) const {
    const int p[3] = {x, y, z};
    auto at = [&](int off) {
      int q[3] = {p[0], p[1], p[2]};
      q[k] += off;
      return v(c, q[0], q[1], q[2]);
    };
    if (periodic || (p[k] > 0 && p[k] < L - 1)) return 0.5 * (at(1) - at(-1));
    const int e = ext(c, k);
    const int s = (p[k] == 0) ? 1 : -1;
    if (e == ExtOneSided)
      return double(s) * (-1.5 * at(0) + 2.0 * at(s) - 0.5 * at(2 * s));
    const double in = at(s), ghost = (e == ExtEven) ? in : -in;
    return double(s) * 0.5 * (in - ghost);
  }
};

// mkdir -p on POSIX ::mkdir, not std::filesystem: every GPU/ driver that has
// run on CSF3 creates its directories this way, and <filesystem> needs an extra
// -lstdc++fs with a GCC older than 9 -- a link error that would surface only on
// the cluster. EEXIST at any level is fine.
bool mkdir_p(const std::string& path) {
  std::string cur;
  for (std::size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0) {
        struct stat st;
        if (::stat(cur.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
      }
    }
    if (i < path.size()) cur += path[i];
  }
  return true;
}

// THE PROBE RUNS ON THE HOST, PARALLEL OVER z. Every probe copies the fields
// down and walks every node computing eighteen derivatives, and at N = 512 a
// serial pass of that is ~10^9 operations -- more than the steps between
// probes, so the diagnostics would have cost more than the run. Each thread sums
// its own z-slab into an Acc and the slabs merge in thread order: a given thread
// count gives a deterministic answer, and changing the count moves only the
// ROUNDING of the sums, never the arithmetic at a node. Plain std::thread rather
// than OpenMP, so nothing about the nvcc or Apple host toolchain changes.
struct Acc {
  double W = 0, ev = 0, em = 0, hc = 0, w2 = 0, j2 = 0, d2[3] = {0, 0, 0}, d3[3] = {0, 0, 0};
  double div2 = 0, eps = 0, divw = 0, ndivw = 0, j2w = 0, tw = 0, ntw = 0;
  double mass = 0, jmax = 0, wmax = 0, umax = 0;
  bool finite = true;
  // Per wall-distance layer k -- the nodes k cells from the nearest wall -- the
  // weighted dissipation, its viscous and Ohmic parts, and the weight itself;
  // f_w and the profile files come from these (within_cells says how).
  std::vector<double> lay_e, lay_w, lay_v, lay_m;
  explicit Acc(std::size_t nlay = 0)
      : lay_e(nlay, 0.0), lay_w(nlay, 0.0), lay_v(nlay, 0.0), lay_m(nlay, 0.0) {}
  void add(const Acc& a) {
    W += a.W; ev += a.ev; em += a.em; hc += a.hc; w2 += a.w2; j2 += a.j2;
    for (int k = 0; k < 3; ++k) { d2[k] += a.d2[k]; d3[k] += a.d3[k]; }
    div2 += a.div2; eps += a.eps; divw += a.divw; ndivw += a.ndivw; j2w += a.j2w;
    tw += a.tw; ntw += a.ntw; mass += a.mass;
    jmax = std::max(jmax, a.jmax); wmax = std::max(wmax, a.wmax); umax = std::max(umax, a.umax);
    finite = finite && a.finite;
    for (std::size_t k = 0; k < lay_e.size(); ++k) {
      lay_e[k] += a.lay_e[k]; lay_w[k] += a.lay_w[k]; lay_v[k] += a.lay_v[k]; lay_m[k] += a.lay_m[k];
    }
  }
};

// Threads for the host diagnostics: -threads if given, else the SLURM
// allocation (hardware_concurrency sees the whole node, not the job's cores,
// and oversubscribing a -c 4 job would be slower than serial), else the
// machine's, capped at 16.
int diag_threads(int requested) {
  if (requested > 0) return requested;
  if (const char* s = std::getenv("SLURM_CPUS_PER_TASK")) {
    const int n = std::atoi(s);
    if (n > 0) return n;
  }
  const unsigned hc = std::thread::hardware_concurrency();
  return int(std::min(16u, std::max(1u, hc)));
}

// fn(z0, z1, tid) over nt contiguous z-slabs; nt = 1 runs inline.
template <class Fn>
void parallel_z(int L, int nt, Fn fn) {
  nt = std::max(1, std::min(nt, L));
  if (nt == 1) { fn(0, L, 0); return; }
  std::vector<std::thread> th;
  th.reserve(std::size_t(nt));
  for (int t = 0; t < nt; ++t) th.emplace_back(fn, (L * t) / nt, (L * (t + 1)) / nt, t);
  for (auto& x : th) x.join();
}

// THE WALL SHARE f_w IS READ OFF THE WALL-DISTANCE PROFILE, NOT COUNTED IN WHOLE
// NODE LAYERS. With the trapezoidal weights, layer k is EXACTLY the slab
// [(k - 1/2) h, (k + 1/2) h] of the box in volume -- the wall layer is [0, h/2] --
// so the dissipation within a distance d of the walls is known at d = (k + 1/2) h
// and is interpolated linearly in between: a band edge inside a layer takes the
// fraction of that layer it covers. The count this replaced, every node with
// k h < m delta, measured a band of (floor(m delta / h) + 1/2) h instead of
// m delta -- 0.91 delta at N = 384 and 1.07 delta at N = 512 for Re = 1000 -- and
// in the first CSF3 ladder (2026-09-28) fw1 differed by 8-25 % between the two
// grids of a rung while the peak dissipation agreed to 1 %: the gate was
// measuring where the band edge fell between nodes, not the flow. r is the
// distance in cells; e[k] the layer sums.
double within_cells(const std::vector<double>& e, double r) {
  if (r <= 0.0 || e.empty()) return 0.0;
  if (r < 0.5) return e[0] * (r / 0.5);
  const std::size_t k = std::size_t(std::floor(r + 0.5));   // r in [k - 1/2, k + 1/2)
  double c = 0.0;
  for (std::size_t j = 0; j < std::min(k, e.size()); ++j) c += e[j];
  return k < e.size() ? c + e[k] * (r - (double(k) - 0.5)) : c;
}

struct Diag {
  double t = 0, ev = 0, em = 0, hc = 0, omv = 0, omm = 0, eps = 0;
  double jmax = 0, wmax = 0, skew = 0, divb = 0, divb_wall = 0, mass = 0;
  double umax_lat = 0, fw[3] = {0, 0, 0}, tauw = 0;
  std::vector<double> prof, vol;   // each layer's share of eps and of the volume
  std::vector<double> prof_v, prof_m;   // its viscous and Ohmic parts, shares of the TOTAL eps
  bool finite = true;
};

void write_raw2(const std::string& path, int nx, int ny, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
}

void write_raw3(const std::string& path, int nx, int ny, int nz, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny, c = nz;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(&c), sizeof c);
  o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
}

// Byte-identical to the parent's write_raw_fields: "TGMHDRAW", int32 nx ny nz
// nvar, float64 t h, then nvar float32 blocks (ux uy uz bx by bz rho).
void write_raw_fields(const std::string& path, const HostFields& H, double t, double h) {
  std::ofstream o(path, std::ios::binary);
  o.write("TGMHDRAW", 8);
  const std::int32_t n = std::int32_t(H.L), nv = 7;
  for (int k = 0; k < 3; ++k) o.write(reinterpret_cast<const char*>(&n), sizeof n);
  o.write(reinterpret_cast<const char*>(&nv), sizeof nv);
  o.write(reinterpret_cast<const char*>(&t), sizeof t);
  o.write(reinterpret_cast<const char*>(&h), sizeof h);
  std::vector<float> blk(std::size_t(H.L) * H.L * H.L);
  for (int c = 0; c < 7; ++c) {
    std::size_t m = 0;
    for (int z = 0; z < H.L; ++z)
      for (int y = 0; y < H.L; ++y)
        for (int x = 0; x < H.L; ++x) blk[m++] = float(H.v(c, x, y, z));
    o.write(reinterpret_cast<const char*>(blk.data()), std::streamsize(blk.size() * sizeof(float)));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if      (a == "-n")         o.N = std::atoi(next());
    else if (a == "-geom")      o.periodic = std::string(next()) == "periodic";
    else if (a == "-vwall")     o.noslip = std::string(next()) != "slip";
    else if (a == "-mwall")     o.conducting = std::string(next()) != "pv";
    else if (a == "-ic") {
      const std::string s = next();
      o.ic = s == "tgi" ? TGI : s == "tga" ? TGA : s == "hydro" ? Hydro : TGC;
    }
    else if (a == "-re")        o.re = std::atof(next());
    else if (a == "-pm")        o.pm = std::atof(next());
    else if (a == "-u0")        o.u0 = std::atof(next());
    else if (a == "-tmax")      o.tmax = std::atof(next());
    else if (a == "-probe")     o.probe = std::atof(next());
    else if (a == "-vti")       o.vti = std::atof(next());
    else if (a == "-dump")      o.dump = std::atof(next());
    else if (a == "-dumpvol")   o.dumpvol = true;
    else if (a == "-volstride") o.volstride = std::max(1, std::atoi(next()));
    else if (a == "-raw")       o.raw = std::atof(next());
    else if (a == "-out")       o.out = next();
    else if (a == "-force")     o.force = true;
    else if (a == "-threads")   o.threads = std::atoi(next());
  }

  const bool hydro = (o.ic == Hydro);
  const bool ic_conducting = (o.ic == TGC);
  if (!hydro && !o.periodic && ic_conducting != o.conducting && !o.force) {
    std::fprintf(stderr, "tg_mhd: -ic %s has the %s parity but -mwall is %s; the wall would "
                 "rewrite the field in the first step. Use -mwall %s, or -force.\n",
                 ic_conducting ? "tgc" : "tgi/tga", ic_conducting ? "conducting" : "pseudo-vacuum",
                 o.conducting ? "cond" : "pv", ic_conducting ? "cond" : "pv");
    return 2;
  }

  const int N = o.N;
  const int L = o.periodic ? 2 * (N - 1) : N;
  const long NP = long(L) * L * L;
  // THE BOX GETS A ONE-CELL SOLID SHELL in storage. This array wraps, and
  // RegWall finds its unknown directions by looking for Solid or Excluded
  // neighbours -- with the box flush against the array edge it found none and
  // built its wall stress from the opposite face (see build_reg_walls). The
  // mirrors (SpecNode, the parity wall) take their unknowns from a face mask and
  // never needed this; the shell costs them nothing.
  const int pad = o.periodic ? 0 : 1;
  const int LS = L + 2 * pad;
  const long NS = long(LS) * LS * LS;
  const double h = PI / double(N - 1);
  const double nu_lat = o.u0 / (h * o.re), eta_lat = nu_lat / o.pm;
  const double dt = o.u0 * h;
  const std::size_t T = std::size_t(std::llround(o.tmax / dt));
  const double b0_tg = hydro ? 0.0 : (o.ic == TGA ? 1.0 : 1.0 / std::sqrt(3.0));
  const double tauf = 3.0 * nu_lat + 0.5, taum = 4.0 * eta_lat + 0.5;

  const char* icn = o.ic == TGC ? "tgc" : o.ic == TGI ? "tgi" : o.ic == TGA ? "tga" : "hydro";
  char tag[160];
  if (o.periodic)
    std::snprintf(tag, sizeof tag, "%s_periodic_n%d_re%g", icn, N, o.re);
  else
    std::snprintf(tag, sizeof tag, "%s_%s_%s_n%d_re%g", icn, o.noslip ? "noslip" : "slip",
                  hydro ? "nofield" : (o.conducting ? "cond" : "pv"), N, o.re);
  if (o.out.empty()) o.out = std::string("tg_mhd_") + tag + (backend::on_device ? "_cuda" : "_host");
  if (!mkdir_p(o.out)) { std::fprintf(stderr, "tg_mhd: cannot create %s\n", o.out.c_str()); return 2; }

  std::printf("Taylor-Green MHD, %s   %s   %s\n",
              o.periodic ? "PERIODIC 2pi box" : "CLOSED box [0,pi]^3, on-node walls",
              backend::on_device ? "CUDA native" : "HOST reference",
              sizeof(Real) == 4 ? "FP32" : "FP64");
  if (!o.periodic)
    std::printf("  walls: velocity %s, field %s\n",
                o.noslip ? "NO-SLIP (RegWall)" : "free-slip (SpecNode)",
                hydro ? "none (B = 0)" : (o.conducting ? "CONDUCTING parity" : "pseudo-vacuum parity"));
  std::printf("  ic %s   N = %d (%d^3 nodes)   Re = %g   Pm = %g   u0 = %g (Ma %.3f)\n",
              icn, N, L, o.re, o.pm, o.u0, o.u0 * std::sqrt(3.0));
  std::printf("  nu_lat = %.4e  tau_f = %.5f   eta_lat = %.4e  tau_m = %.5f\n",
              nu_lat, tauf, eta_lat, taum);
  std::printf("  dt = %.4e  ->  %zu steps to t = %g   ->  %s\n", dt, T, o.tmax, o.out.c_str());
  if (o.noslip && !o.periodic && tauf - 0.5 < 0.012)
    std::printf("  NOTE: tau_f - 1/2 = %.2e is at RegWall's measured floor; expect trouble at "
                "the wall.\n", tauf - 0.5);

  backend::Magnetic mag(LS, LS, LS, Real(eta_lat));
  backend::Fluid    fl (LS, LS, LS, Op::CentralMoments, Real(nu_lat));
  if (!o.periodic) {
    // static_cast, NOT std::size_t(NS): the functional casts make this the most
    // vexing parse -- `faces` becomes a function declaration -- which this tree
    // records hitting five times before. The host build caught this one.
    std::vector<std::uint8_t> geom(static_cast<std::size_t>(NS),
                                   static_cast<std::uint8_t>(Solid));
    std::vector<std::uint8_t> faces(static_cast<std::size_t>(NS),
                                    static_cast<std::uint8_t>(SpecNone));
    std::vector<RegWallSpec> spec(static_cast<std::size_t>(NS));
    for (int z = 0; z < L; ++z)
      for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
          const std::size_t n = std::size_t(node_id(x + pad, y + pad, z + pad, LS, LS));
          geom[n] = Fluid;
          faces[n] = box_faces(x, y, z, N);
          spec[n].normal = box_normal(x, y, z, N);
        }
    fl.set_geometry(geom);
    mag.set_geometry(geom);
    if (o.noslip) {
      fl.set_fd_corners(false);      // FD edge closure leaks mass under a Lorentz force
      fl.set_regularized_walls(spec);
    } else {
      fl.set_specular_nodes(faces);
    }
    if (!hydro) mag.set_parity_walls(faces, o.conducting);
  }
  if (!hydro) {
    fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
    mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());
  }
  const TGState st{o.ic, h, o.u0, o.u0 * b0_tg, o.noslip && !o.periodic, N, pad};
  mag.initialise_with(TGField{st}, TGVelocity{st});
  fl.seed_populations_with(TGPopulations{st});

  std::vector<Real> rho, ux, uy, uz, bx, by, bz;
  HostFields H;
  H.L = L; H.LS = LS; H.pad = pad;
  H.periodic = o.periodic; H.noslip = o.noslip; H.conducting = o.conducting;
  H.su = 1.0 / o.u0;
  H.f[0] = &ux; H.f[1] = &uy; H.f[2] = &uz; H.f[3] = &bx; H.f[4] = &by; H.f[5] = &bz; H.f[6] = &rho;
  auto refresh = [&]() {
    fl.macroscopic_to_host(rho, ux, uy, uz);
    if (hydro) { bx.assign(std::size_t(NS), Real(0)); by = bx; bz = bx; }
    else       mag.field_to_host(bx, by, bz);
  };

  const double nu_tg = 1.0 / o.re, eta_tg = nu_tg / o.pm, ih = 1.0 / h;
  const double delta = 1.0 / std::sqrt(o.re);
  const double mass0 = double(NP);

  const int nthreads = diag_threads(o.threads);
  auto measure_slab = [&](int z0, int z1, Acc& a) {
    for (int z = z0; z < z1; ++z)
      for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
          const int p[3] = {x, y, z};
          double w = 1.0;
          int dw = L;
          if (!o.periodic)
            for (int k = 0; k < 3; ++k) {
              if (p[k] == 0 || p[k] == L - 1) w *= 0.5;
              dw = std::min(dw, std::min(p[k], L - 1 - p[k]));
            }
          else
            for (int k = 0; k < 3; ++k) {
              const int m = p[k] % (N - 1);
              dw = std::min(dw, std::min(m, N - 1 - m));
            }
          const double u[3] = {H.v(0, x, y, z), H.v(1, x, y, z), H.v(2, x, y, z)};
          const double b[3] = {H.v(3, x, y, z), H.v(4, x, y, z), H.v(5, x, y, z)};
          for (int c = 0; c < 3; ++c)
            if (!std::isfinite(u[c]) || !std::isfinite(b[c])) a.finite = false;
          a.mass += H.v(6, x, y, z);
          a.umax = std::max(a.umax, o.u0 * std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]));
          double du[3][3], db[3][3];
          for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) {
              du[c][k] = H.d(c, k, x, y, z) * ih;
              db[c][k] = hydro ? 0.0 : H.d(3 + c, k, x, y, z) * ih;
            }
          const double wv[3] = {du[2][1] - du[1][2], du[0][2] - du[2][0], du[1][0] - du[0][1]};
          const double jv[3] = {db[2][1] - db[1][2], db[0][2] - db[2][0], db[1][0] - db[0][1]};
          const double w2 = wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2];
          const double j2 = jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2];
          const double dvb = db[0][0] + db[1][1] + db[2][2];
          const double eloc = nu_tg * w2 + eta_tg * j2;
          a.W += w;
          a.ev += w * 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
          a.em += w * 0.5 * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
          a.hc += w * (u[0] * b[0] + u[1] * b[1] + u[2] * b[2]);
          a.w2 += w * w2;  a.j2 += w * j2;  a.div2 += w * dvb * dvb;  a.eps += w * eloc;
          for (int k = 0; k < 3; ++k) {
            a.d2[k] += w * du[k][k] * du[k][k];
            a.d3[k] += w * du[k][k] * du[k][k] * du[k][k];
          }
          const std::size_t lay = std::min(std::size_t(dw), a.lay_e.size() - 1);
          a.lay_e[lay] += w * eloc;
          a.lay_w[lay] += w;
          a.lay_v[lay] += w * (nu_tg * w2);     // eloc's viscous part
          a.lay_m[lay] += w * (eta_tg * j2);    // and its Ohmic part
          if (dw <= 1) { a.divw += w * dvb * dvb; a.j2w += w * j2; a.ndivw += w; }
          a.jmax = std::max(a.jmax, std::sqrt(j2));
          a.wmax = std::max(a.wmax, std::sqrt(w2));
          if (!o.periodic && o.noslip)
            for (int k = 0; k < 3; ++k)
              if (p[k] == 0 || p[k] == L - 1) {
                double sst = 0;
                for (int c = 0; c < 3; ++c) if (c != k) sst += du[c][k] * du[c][k];
                a.tw += nu_tg * std::sqrt(sst);
                a.ntw += 1;
              }
        }
  };

  // Wall-distance layers: 0 .. (N - 1)/2 cells, in the box and in the twin alike.
  const std::size_t nlay = std::size_t((N - 1) / 2 + 1);
  auto measure = [&](double t) {
    std::vector<Acc> part(static_cast<std::size_t>(nthreads), Acc(nlay));
    parallel_z(L, nthreads, [&](int z0, int z1, int tid) {
      measure_slab(z0, z1, part[static_cast<std::size_t>(tid)]);
    });
    Acc a(nlay);
    for (const Acc& q : part) a.add(q);
    Diag g; g.t = t;
    g.finite = a.finite;
    g.umax_lat = a.umax; g.jmax = a.jmax; g.wmax = a.wmax;
    const double W = a.W;
    g.ev = a.ev / W; g.em = a.em / W; g.hc = a.hc / W;
    g.omv = 0.5 * a.w2 / W; g.omm = 0.5 * a.j2 / W;
    g.eps = 2.0 * nu_tg * g.omv + 2.0 * eta_tg * g.omm;
    const double m2 = (a.d2[0] + a.d2[1] + a.d2[2]) / (3.0 * W);
    const double m3 = (a.d3[0] + a.d3[1] + a.d3[2]) / (3.0 * W);
    g.skew = (m2 > 0) ? -m3 / std::pow(m2, 1.5) : 0.0;
    g.divb = (a.j2 > 0) ? std::sqrt(a.div2 / a.j2) : 0.0;
    g.divb_wall = (a.j2w > 0 && a.ndivw > 0) ? std::sqrt(a.divw / a.j2w) : 0.0;
    double etot = 0.0;
    for (double e : a.lay_e) etot += e;
    for (int m = 0; m < 3; ++m)
      g.fw[m] = (etot > 0) ? within_cells(a.lay_e, double(1 << m) * delta / h) / etot : 0.0;
    g.prof.resize(nlay); g.vol.resize(nlay); g.prof_v.resize(nlay); g.prof_m.resize(nlay);
    for (std::size_t k = 0; k < nlay; ++k) {
      g.prof[k] = (etot > 0) ? a.lay_e[k] / etot : 0.0;
      g.vol[k] = a.lay_w[k] / W;
      g.prof_v[k] = (etot > 0) ? a.lay_v[k] / etot : 0.0;
      g.prof_m[k] = (etot > 0) ? a.lay_m[k] / etot : 0.0;
    }
    g.tauw = (a.ntw > 0) ? a.tw / a.ntw : 0.0;
    g.mass = (t > 0) ? a.mass / mass0 - 1.0 : 0.0;
    return g;
  };

  auto every = [&](double dtp) -> std::size_t {
    return dtp > 0 ? std::max<std::size_t>(1, std::size_t(std::llround(dtp / dt))) : 0;
  };
  const std::size_t kp = every(o.probe), kv = every(o.vti), kd = every(o.dump), kr = every(o.raw);
  std::FILE* ser = std::fopen((o.out + "/series.dat").c_str(), "w");
  if (!ser) { std::fprintf(stderr, "tg_mhd: cannot write series.dat\n"); return 2; }
  // fw=profile marks f_w read off the wall-distance profile (within_cells);
  // series without it counted whole node layers, and tools/tg_mhd_ladder.py
  // corrects those.
  std::fprintf(ser, "# tg_mhd (%s %s)  %s  N=%d L=%d Re=%g Pm=%g u0=%g nu_lat=%.6e eta_lat=%.6e"
                    " tau_f=%.6f tau_m=%.6f dt=%.6e fw=profile\n",
               backend::on_device ? "CUDA" : "host", sizeof(Real) == 4 ? "FP32" : "FP64",
               tag, N, L, o.re, o.pm, o.u0, nu_lat, eta_lat, tauf, taum, dt);
  std::fprintf(ser, "# t E_V E_M E_T H_C Omega_V Omega_M eps EM/EV OmM/OmV j_max w_max"
                    " skew divb/j divb_wall/j mass_drift umax_lat fw1 fw2 fw4 tau_wall\n");
  // profile.dat: the dissipation against wall distance, one row per probe --
  // what f_w is read from, and a figure of its own. profile_visc.dat and
  // profile_ohm.dat split it into its viscous (nu w^2) and Ohmic (eta j^2)
  // parts, each still a share of the TOTAL eps, so the two add up to
  // profile.dat: which one a wall layer is made of is what tells a no-slip
  // boundary layer from a current sheet on a conducting wall.
  const char* pro_name[3] = {"profile.dat", "profile_visc.dat", "profile_ohm.dat"};
  const char* pro_what[3] = {"share of eps", "VISCOUS dissipation as a share of the total eps",
                             "OHMIC dissipation as a share of the total eps"};
  std::FILE* pro[3] = {nullptr, nullptr, nullptr};
  for (int f = 0; f < 3; ++f) {
    pro[f] = std::fopen((o.out + "/" + pro_name[f]).c_str(), "w");
    if (!pro[f]) { std::fprintf(stderr, "tg_mhd: cannot write %s\n", pro_name[f]); return 2; }
    std::fprintf(pro[f], "# tg_mhd (%s %s)  %s  N=%d Re=%g h=%.8e delta=%.8e delta/h=%.6f layers=%zu\n",
                 backend::on_device ? "CUDA" : "host", sizeof(Real) == 4 ? "FP32" : "FP64",
                 tag, N, o.re, h, delta, delta / h, nlay);
    std::fprintf(pro[f], "# layer k: the nodes k cells from the nearest wall, i.e. the slab"
                         " [(k-1/2)h, (k+1/2)h] ([0, h/2] for k = 0)\n");
    std::fprintf(pro[f], "# row 'vol': each layer's share of the volume; then per probe: t and"
                         " each layer's %s\n", pro_what[f]);
  }
  bool pro_vol = false;

  std::FILE* meta = nullptr;
  int dframe = 0, vframe = 0, rframe = 0;
  std::vector<std::pair<double, std::string>> pvd;
  if (kd) {
    mkdir_p(o.out + "/anim_frames");
    meta = std::fopen((o.out + "/anim_frames/meta.txt").c_str(), "w");
    if (meta) std::fprintf(meta, "N %d\nR 0\nTe 1.0\ncase tg_mhd %s\n", L, tag);
    // Grid and |J| volume stride, as the parent writes them; see there.
    if (meta && o.dumpvol) {
      const int S = o.volstride, Mr = (L + S - 1) / S;
      std::fprintf(meta, "nx %d\nny %d\nnz %d\nvstride %d\nvnx %d\nvny %d\nvnz %d\n",
                   L, L, L, S, Mr, Mr, Mr);
    }
  }
  if (kv) mkdir_p(o.out + "/vti");
  if (kr) mkdir_p(o.out + "/raw");

  double e0 = 0;
  const int zc = (N - 1) / 2;
  auto frames = [&](std::size_t k, double t, const Diag& g) {
    const std::size_t np = std::size_t(NP);
    std::vector<float> jm, wm, jv3, wv3;
    const bool need_vec = kv && k % kv == 0;
    if ((kd && k % kd == 0) || need_vec) {
      jm.resize(np); wm.resize(np);
      if (need_vec) { jv3.resize(3 * np); wv3.resize(3 * np); }
      parallel_z(L, nthreads, [&](int z0, int z1, int) {
      for (int z = z0; z < z1; ++z)
        for (int y = 0; y < L; ++y)
          for (int x = 0; x < L; ++x) {
            const std::size_t m = (std::size_t(z) * L + y) * L + x;
            double du[3][3], db[3][3];
            for (int c = 0; c < 3; ++c)
              for (int kk = 0; kk < 3; ++kk) {
                du[c][kk] = H.d(c, kk, x, y, z) * ih;
                db[c][kk] = hydro ? 0.0 : H.d(3 + c, kk, x, y, z) * ih;
              }
            const double wv[3] = {du[2][1] - du[1][2], du[0][2] - du[2][0], du[1][0] - du[0][1]};
            const double jv[3] = {db[2][1] - db[1][2], db[0][2] - db[2][0], db[1][0] - db[0][1]};
            jm[m] = float(std::sqrt(jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2]));
            wm[m] = float(std::sqrt(wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]));
            if (need_vec)
              for (int c = 0; c < 3; ++c) { jv3[3 * m + c] = float(jv[c]); wv3[3 * m + c] = float(wv[c]); }
          }
      });
    }
    if (kd && k % kd == 0) {
      std::vector<float> su(std::size_t(L) * L), sb(su.size()), sj(su.size()), sw(su.size());
      for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
          const std::size_t m2 = std::size_t(y) * L + x;
          const std::size_t m3 = (std::size_t(zc) * L + y) * L + x;
          const double u[3] = {H.v(0, x, y, zc), H.v(1, x, y, zc), H.v(2, x, y, zc)};
          const double b[3] = {H.v(3, x, y, zc), H.v(4, x, y, zc), H.v(5, x, y, zc)};
          su[m2] = float(std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]));
          sb[m2] = float(std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]));
          sj[m2] = jm[m3];
          sw[m2] = wm[m3];
        }
      char nm[64];
      const std::string af = o.out + "/anim_frames/";
      std::snprintf(nm, sizeof nm, "umag_%04d.raw", dframe); write_raw2(af + nm, L, L, su);
      std::snprintf(nm, sizeof nm, "bmag_%04d.raw", dframe); write_raw2(af + nm, L, L, sb);
      std::snprintf(nm, sizeof nm, "jmag_%04d.raw", dframe); write_raw2(af + nm, L, L, sj);
      std::snprintf(nm, sizeof nm, "wmag_%04d.raw", dframe); write_raw2(af + nm, L, L, sw);
      if (o.dumpvol) {
        const int S = o.volstride, Mr = (L + S - 1) / S;
        std::vector<float> red(std::size_t(Mr) * Mr * Mr, 0.0f);
        for (int z = 0; z < L; ++z)
          for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x) {
              const std::size_t r3 = (std::size_t(z / S) * Mr + std::size_t(y / S)) * Mr + std::size_t(x / S);
              red[r3] = std::max(red[r3], jm[(std::size_t(z) * L + y) * L + x]);
            }
        std::snprintf(nm, sizeof nm, "jvol_%04d.raw", dframe);
        write_raw3(af + nm, Mr, Mr, Mr, red);
      }
      if (meta) {
        std::fprintf(meta, "frame %d %.6f %.8e %.8e\n", dframe, t,
                     e0 > 0 ? g.ev / e0 : 0.0, e0 > 0 ? g.em / e0 : 0.0);
        std::fflush(meta);
      }
      ++dframe;
    }
    if (need_vec) {
      std::vector<VtiArray> arr;
      VtiArray au{"u", 3, {}}, ab{"b", 3, {}}, aw{"omega", 3, {}}, aj{"J", 3, {}};
      VtiArray ajm{"Jmag", 1, {}}, awm{"Wmag", 1, {}}, ar{"rho", 1, {}};
      au.data.resize(3 * np); ab.data.resize(3 * np); ar.data.resize(np);
      std::size_t m = 0;
      for (int z = 0; z < L; ++z)
        for (int y = 0; y < L; ++y)
          for (int x = 0; x < L; ++x, ++m) {
            for (int c = 0; c < 3; ++c) {
              au.data[3 * m + c] = float(H.v(c, x, y, z));
              ab.data[3 * m + c] = float(H.v(3 + c, x, y, z));
            }
            ar.data[m] = float(H.v(6, x, y, z));
          }
      aj.data = std::move(jv3); aw.data = std::move(wv3);
      ajm.data = jm; awm.data = wm;
      arr.push_back(std::move(ajm)); arr.push_back(std::move(au)); arr.push_back(std::move(ab));
      arr.push_back(std::move(aj)); arr.push_back(std::move(aw)); arr.push_back(std::move(awm));
      arr.push_back(std::move(ar));
      char nm[64];
      std::snprintf(nm, sizeof nm, "tg_%04d.vti", vframe++);
      if (write_vti_bin(o.out + "/vti/" + nm, L, L, L, arr, h)) pvd.emplace_back(t, nm);
    }
    if (kr && k % kr == 0) {
      char nm[64];
      std::snprintf(nm, sizeof nm, "fields_%04d.raw", rframe++);
      write_raw_fields(o.out + "/raw/" + nm, H, t, h);
    }
  };

  std::printf("\n  %7s %11s %11s %9s %9s %9s %9s %9s %9s %10s\n", "t", "E_V", "E_M", "EM/EV",
              "OmM/OmV", "eps", "j_max", "divb/j", "mass", "fw(1d)");
  const auto wall0 = std::chrono::steady_clock::now();
  // The dissipation peak is the TURBULENT maximum, not the largest eps in the
  // run: a no-slip box starts impulsively -- the Taylor-Green velocity does not
  // vanish on the walls -- and its eps is largest at t = 0 (fw1 = 0.91 there,
  // against 0.15 at the t = 2 peak, N = 129, Re = 300). So the maximum is taken
  // only once eps has risen 5 % above its running minimum: at once for free
  // slip, after the start has decayed for no slip. tools/tg_mhd_ladder.py reads
  // the peak the same way, and so does the Kokkos twin.
  double first_ommv_max = 0, prev_ommv = 0, emev_min = 1e300, eps_max = 0, t_epsmax = 0,
         skew_at_epsmax = 0, skew_max = -1e300, eps_runmin = 1e300, t_lastprobe = 0;
  bool eps_armed = false;
  int ommv_phase = 0, rc = 0, nprobe = 0;
  double t_diag = 0.0;               // wall time in the host copies, probes and writers
  for (std::size_t k = 0; k <= T; ++k) {
    const bool probe = kp && (k % kp == 0 || k == T);
    const bool outp = (kv && k % kv == 0) || (kd && k % kd == 0) || (kr && k % kr == 0);
    if (probe || outp) {
      // Drain the queued steps FIRST: launches are asynchronous, so without this
      // the probe's first copy waits for them and the probe timer absorbs the
      // steps -- the first device runs read "100 % of the run" in probes at
      // N = 65 and "7.5 GLUPS without them" at N = 257, twice what an A100's
      // memory bandwidth allows. The total and its MLUPS were always right.
      backend::sync();
      const auto d0 = std::chrono::steady_clock::now();
      refresh();
      const double t = double(k) * dt;
      const Diag g = measure(t);
      ++nprobe;
      if (k == 0) e0 = g.ev + g.em;
      if (!g.finite) {
        std::printf("  DIVERGED at t = %.4f (step %zu)\n", t, k);
        std::fprintf(ser, "# DIVERGED at t = %.6f\n", t);
        rc = 1;
        break;
      }
      if (probe) {
        const double emev = g.ev > 0 ? g.em / g.ev : 0.0;
        const double ommv = g.omv > 0 ? g.omm / g.omv : 0.0;
        if (k > 0) {
          emev_min = std::min(emev_min, emev);
          if (ommv_phase == 0 && ommv > prev_ommv) ommv_phase = 1;
          else if (ommv_phase == 1 && ommv < prev_ommv) { first_ommv_max = prev_ommv; ommv_phase = 2; }
          skew_max = std::max(skew_max, g.skew);
        }
        if (!eps_armed) {
          if (g.eps < eps_runmin) eps_runmin = g.eps;
          else if (g.eps > 1.05 * eps_runmin) eps_armed = true;
        }
        if (eps_armed && g.eps > eps_max) { eps_max = g.eps; t_epsmax = t; skew_at_epsmax = g.skew; }
        t_lastprobe = t;
        prev_ommv = ommv;
        std::fprintf(ser, "%.6f %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.6e %.6e %.6e %.6e"
                          " %.6e %.4e %.4e %+.4e %.5f %.5e %.5e %.5e %.5e\n",
                     t, g.ev, g.em, g.ev + g.em, g.hc, g.omv, g.omm, g.eps, emev, ommv,
                     g.jmax, g.wmax, g.skew, g.divb, g.divb_wall, g.mass, g.umax_lat,
                     g.fw[0], g.fw[1], g.fw[2], g.tauw);
        std::fflush(ser);
        const std::vector<double>* pv[3] = {&g.prof, &g.prof_v, &g.prof_m};
        for (int f = 0; f < 3; ++f) {
          if (!pro_vol) {
            std::fprintf(pro[f], "# vol");
            for (double v : g.vol) std::fprintf(pro[f], " %.6e", v);
            std::fprintf(pro[f], "\n");
          }
          std::fprintf(pro[f], "%.6f", t);
          for (double s : *pv[f]) std::fprintf(pro[f], " %.5e", s);
          std::fprintf(pro[f], "\n");
          std::fflush(pro[f]);
        }
        pro_vol = true;
        if (k % (kp * 10) == 0 || k == T)
          std::printf("  %7.3f %11.5e %11.5e %9.4f %9.4f %9.3e %9.3f %9.2e %+9.1e %10.4f\n",
                      t, g.ev, g.em, emev, ommv, g.eps, g.jmax, g.divb, g.mass, g.fw[0]);
        std::fflush(stdout);
      }
      if (outp) frames(k, t, g);
      t_diag += std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
    }
    if (k < T) {
      if (!hydro) mag.compute_field();
      fl.step();
      if (!hydro) mag.step();
    }
  }
  backend::sync();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  std::fclose(ser);
  for (std::FILE* f : pro) std::fclose(f);
  if (meta) { std::fprintf(meta, "frames %d\n", dframe); std::fclose(meta); }
  if (!pvd.empty()) write_pvd(o.out + "/vti/tg.pvd", pvd);

  std::printf("\n  %.1f s   %.2f MLUPS (storage nodes x steps; field and probes included)\n",
              secs, double(NS) * double(T) / secs * 1e-6);
  std::printf("  host diagnostics: %d probes, %.1f s = %.0f %% of the run on %d thread(s);"
              " %.2f MLUPS without them\n",
              nprobe, t_diag, 100.0 * t_diag / secs, nthreads,
              double(NS) * double(T) / std::max(secs - t_diag, 1e-9) * 1e-6);
  if (!eps_armed)
    std::printf("  no dissipation peak: eps fell throughout the run; skewness max %.3f\n", skew_max);
  else if (t_epsmax >= t_lastprobe)
    std::printf("  dissipation peak not reached: eps = %.4e still rising at t = %.3f; skewness max %.3f\n",
                eps_max, t_epsmax, skew_max);
  else
    std::printf("  peak dissipation eps = %.4e at t = %.3f (past any impulsive start);"
                " skewness there %.3f, max %.3f\n", eps_max, t_epsmax, skew_at_epsmax, skew_max);
  if (!hydro) {
    std::printf("  min E_M/E_V = %.3f   first max of Omega_M/Omega_V after its initial fall = %s\n",
                emev_min, ommv_phase == 2 ? std::to_string(first_ommv_max).c_str() : "not reached");
    if (o.ic == TGC && std::fabs(o.re - 1000.0) < 1e-9)
      std::printf("  Pouquet et al. (2010) C2, nu = eta = 1e-3: min E_M/E_V = 0.35, first max"
                  " Omega_M/Omega_V = 2\n");
  }
  if (dframe) std::printf("  %d frame(s) in %s/anim_frames\n", dframe, o.out.c_str());
  if (!pvd.empty()) std::printf("  ParaView: open %s/vti/tg.pvd (%zu frames)\n", o.out.c_str(), pvd.size());
  if (rframe) std::printf("  %d raw field file(s) in %s/raw\n", rframe, o.out.c_str());
  return rc;
}
