//==============================================================================
//  Electroconvection in a CLOSED SQUARE CAVITY -- the CUDA twin.
//
//  Patnaik, Skillen & De Rosis, Eng. Comput. 41:4977-5002 (2025), Sec. 3.2.3,
//  and the port of the parent tree's validation/ehd_cavity.cpp. A square with
//  every wall closed,
//
//      u = 0,   d_x q = 0,   d_x phi = 0     at x = 0, Lx,
//
//  reported through an electric Nusselt number rather than a peak velocity.
//
//  ===================== WHAT THE PORT HAD TO ADD ============================
//  The fluid side needed nothing: D3Q27 central moments, ForceField and on-node
//  regularised walls with corners were all already here. The scalar side needed
//  four things, and they are in the headers rather than in this driver:
//
//    ScalarMoment    Dellar's on-node Dirichlet. NOT optional: E = -grad phi is
//                    a derivative of the field carrying the boundary value, and
//                    the parent measured halfway plates as first order here.
//    ScalarSpecular  the on-node zero-flux wall (specular.cuh). The closed
//                    square cannot dodge a lateral scalar wall the way the
//                    reference's Sec. 3.2.1 can, and both pre-existing
//                    conditions fail there -- see that banner for the table.
//    a source term   for the Poisson relaxation, which must reach specular
//                    nodes as well as bulk ones (scalar.cuh, add_source).
//    ChargeCM        the central-moment charge collision, in closed form
//                    rather than through a 27-moment transform (core.cuh).
//
//  The solvers are templated on their lattice, so the charge is
//  ScalarSolverT<D3Q27> and not a second implementation.
//
//  ===================== WHAT IT IS CHECKED AGAINST ==========================
//  Its Kokkos twin, first and mainly. The two codebases share no headers, so
//  agreement is evidence about the physics and disagreement is a bug in one of
//  them; that is the whole reason both exist. The parent's numbers, unseeded at
//  N = 129:
//
//      T        250    500   1000   1500   3000   5000  10000
//      Ne      1.000  1.595  1.696  1.751  1.979  2.597  3.381
//
//  and, below onset, Ne = 1 to four figures with the fluid at rest -- which is
//  the sharpest check available here, because numerator and denominator are
//  then literally the same computation.
//
//  Against the reference itself the parent sits 9-13 % low wherever the grid is
//  trustworthy, and that deficit is MEASURED AND NOT EXPLAINED. Resolution,
//  Mach number, the seed and the wall columns are all excluded. Do not read
//  agreement between these two codebases as agreement with the paper.
//
//  ===================== WHAT IT MEASURED ON A DEVICE (2026-09-06) ============
//  Tesla T4, FP32, cc 7.5, against the Kokkos twin at matched lattices.
//
//      N     T      Ne                 u_max/u0  Re_cell  wall clock
//      21    150    1.0060 +/- 0.0116     0.073     0.0   seconds
//     201   5000    2.4735 +/- 0.2945    11.926     3.0   3m28
//     501   5000    3.1010 +/- 0.4139    15.159     1.5   30m09
//
//  and I0 = 1.333878e-05 at N = 21 against the twin's 1.334004e-05 and the host
//  build's 1.333898e-05 -- three implementations inside 0.01 %, two of which
//  share no headers.
//
//  N = 501 IS THE REFERENCE'S OWN GRID and the first properly resolved point in
//  either tree: Re_cell = 1.5, where the paper's H = 499 wants about 1.6. It
//  gives Ne = 3.10 against Fig. 8's digitised 2.80, i.e. +10.8 %.
//
//  THAT CONTRADICTS WHAT THE COARSE LADDER SAID, and the coarse ladder was
//  wrong. The parent tree measured 2.8558, 2.8017, 2.4903 at N = 81, 129, 201
//  and concluded that refinement moves AWAY from the reference and that the
//  converged deficit was about 11 % low. Every one of those three points is
//  under-resolved -- Re_cell 8.5, 6.1, 3.6 -- and the first resolved one lands
//  10.8 % HIGH instead. The sequence is not monotonic, so "the converged
//  deficit is about 11 %" was an extrapolation from three points that had not
//  reached the regime they were being extrapolated into.
//
//  WHAT IS ACTUALLY CLAIMABLE, and no more: at the reference's grid and
//  Re_cell = 1.5, Ne = 3.10 +/- 0.41, and the paper's 2.80 sits inside that.
//  The r.m.s. is 13 % of the mean over only THREE t0 of averaging, so this is
//  consistent with the reference rather than converged to it. A 20-30 t0 window
//  is what would turn it into a measurement, and that is another two hours of
//  T4 time.
//
//  The lattice family is not the variable: CUDA at N = 201 on D3Q27/D3Q7 gives
//  2.4735 and the Kokkos twin at N = 201 on D2Q9/D2Q5 gives 2.4903, 0.7 %
//  apart. What changed between 201 and 501 is the resolution.
//
//    usage: ehd_cavity [-n N] [-t T] [-u0 U] [-c C] [-m M] [-sc SC]
//                      [-alpha A] [-beta B] [-tf N] [-tavg N] [-amp A]
//                      [-tfh N] [-watch]
//                      [-dump PREFIX] [-dumpn K]
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/ehd.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
//  A field plane in this tree's dump format: two int32 (nx, ny) then nx*ny
//  float32, row major. Deliberately the SAME format the Kokkos side writes, so
//  doc/fig/bin2vtk.py converts either one to ParaView's legacy VTK without
//  knowing which codebase produced it -- and so a device dump and a host dump
//  can be differenced directly.
//------------------------------------------------------------------------------
static void write_plane(const std::string& path, int nx, int ny,
                        const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  const std::int32_t a = nx, b = ny;
  o.write(reinterpret_cast<const char*>(&a), sizeof a);
  o.write(reinterpret_cast<const char*>(&b), sizeof b);
  o.write(reinterpret_cast<const char*>(v.data()),
          std::streamsize(v.size() * sizeof(float)));
}

struct RestInit {
  LBM_HD Macro operator()(int, int, int) const {
    return Macro{Real(1), Real(0), Real(0), Real(0), Real(1)};
  }
};

//------------------------------------------------------------------------------
//  THE INITIALISERS ARE STRUCTS, NOT LAMBDAS, and that is a device requirement
//  rather than a style choice. initialise_with instantiates a __global__
//  template on the functor's type, and the closure type of an ordinary host
//  lambda cannot be a template argument of a __global__ instantiation --
//  nvcc rejects it outright unless it is an extended (__device__) lambda. A
//  HOST BUILD ACCEPTS THE LAMBDA HAPPILY, because nothing there is __global__,
//  so this is exactly the class of error the host build cannot find and the
//  device build finds immediately. Every other driver here uses the same shape.
//------------------------------------------------------------------------------
struct PotentialInit {
  int H;
  // phi starts at its CHARGE-FREE solution, not at zero. Eq. (19) says zero,
  // but Poisson is elliptic: that is where a relaxation starts, not a state of
  // the system. Against phi = dphi on the plate, zero puts the whole potential
  // difference across one cell and the drift K E goes supersonic.
  LBM_HD Real operator()(int, int y, int) const {
    const double yy = y < 0 ? 0.0 : (y > H ? double(H) : double(y));
    return Real(1.0 - yy / double(H));
  }
};

struct ChargeInit {
  int H;
  Real amp;                          // in lattice units, already scaled by q0
  Real dec;                          // seed depth
  int nz = 1;                        // depth; > 1 switches the z modulation on
  // THE SEED MUST BREAK z-SYMMETRY OR THE DEPTH BUYS NOTHING. The equations
  // preserve z-independence EXACTLY, so a z-uniform initial condition leaves
  // u_z at round-off for all time and an nz-plane grid returns the same answer
  // as one plane, at nz times the cost. Two z modes rather than one, so the
  // flow picks its own rather than being handed one; the amplitudes sum to
  // 0.95 < 1 so the modulation stays POSITIVE and the seeded charge cannot
  // start outside [0, q0]. At nz = 1 it is switched off rather than evaluated
  // at z = 0, which keeps every existing run bit-identical.
  LBM_HD Real operator()(int x, int y, int z) const {
    if (y <= 0 || y >= H) return Real(0);
    double lat = 0.0;
    const double ph[4] = {0.0, 1.1, 2.3, 0.7};
    for (int m = 0; m < 4; ++m)
      lat += 0.25 * 0.5 * (1.0 + cos(M_PI * double(m + 1) * (double(x) + 0.5) / 64.0
                                     + ph[m]));
    const double zmod = (nz > 1)
        ? 1.0 + 0.60 * cos(2.0 * M_PI * double(z) / double(nz))
              + 0.35 * cos(6.0 * M_PI * double(z) / double(nz) + 0.7)
        : 1.0;
    return Real(double(amp) * lat * zmod * exp(-double(y) / double(dec)));
  }
};

struct Opts {
  int n = 129, nz = 1;
  double T = 1000, u0 = 5e-3, C = 10, M = 10, Sc = 1e3, alpha = -1, beta = 0.3;
  double tf = 30, tavg = 15, tfh = 12, amp = 0;
  bool watch = false, profile = false;
  int  dumpn = 0;             // -dumpn K: a frame every K probes
  std::string dump;           // -dump PREFIX
};

//------------------------------------------------------------------------------
// The D = 0 hydrostatic current, by bisection: a CHECK on the measured I0.
//------------------------------------------------------------------------------
static double analytic_current(double K, double eps, double q0, double dphi, double H) {
  auto integral = [&](double j) {
    const double E0 = j / (K * q0), a = 2.0 * j / (K * eps);
    return (2.0 / (3.0 * a)) * (std::pow(E0 * E0 + a * H, 1.5) - E0 * E0 * E0);
  };
  double lo = 1e-300, hi = 1.0;
  for (int i = 0; i < 200 && integral(hi) < dphi; ++i) hi *= 2.0;
  for (int i = 0; i < 300; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (integral(mid) < dphi) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

struct Out {
  double Ie = 0, Ne = 0, Ne_rms = 0, umax = 0, qlo = 0, qhi = 0;
  // THE WORST EXCURSION OVER THE WHOLE RUN, not the last probe's bounds. The
  // charge obeys a maximum principle -- div(u + K E) = K q/eps > 0, so the
  // non-conservative form carries -K q^2/eps, which vanishes at q = 0 and
  // leaves [0, q0] invariant -- so negative charge is the scheme failing and
  // nothing else. Reporting only the final probe hides a violation that
  // recovered: this run reported [0.0088, 1.0000] at t/t0 = 8 while its own
  // dumped frame at t/t0 = 7 held -0.0343. A run that FINISHED is not thereby
  // a run that stayed in bounds, which is the discipline rb_high_ra already
  // applies to temperature.
  double worst = 0, t_worst = 0;
  int nacc = 0;
};

//------------------------------------------------------------------------------
static Out solve(const Opts& o, bool hydro, double I0, bool verbose) {
  const int nx = o.n, ny = o.n, nz = o.nz, H = o.n - 1;
  const long N = long(nx) * ny * nz;
  const std::size_t NN = static_cast<std::size_t>(N);

  const double dphi = 1.0;
  const double K    = o.u0 * double(H) / dphi;
  const double eps  = o.M * o.M * K * K;
  const double nu   = eps * dphi / (K * o.T);
  const double q0   = eps * o.C * dphi / (double(H) * double(H));
  const double alph = o.alpha > 0 ? o.alpha : o.M * o.M / (o.T * o.Sc);
  const double Dq   = alph * K * dphi;
  const double t0   = double(H) / o.u0;

  if (verbose)
    std::printf("  %s  T = %g   %d x %d (H = %d)   K = %.4g  eps = %.4g  nu = %.5g"
                " (tau %.4f)\n            q0 = %.4g  alpha = %.4g  D = %.4g\n",
                hydro ? "reference (F = 0)" : "cavity          ", o.T, nx, ny, H,
                K, eps, nu, 3.0 * nu + 0.5, q0, alph, Dq);

  //---- geometry ---------------------------------------------------------------
  const std::uint8_t kFluid = Fluid;
  std::vector<std::uint8_t> geo(NN, kFluid);
  std::vector<RegWallSpec> spec(NN);
  std::vector<std::uint8_t> qf(NN, std::uint8_t(ScalarBulk)), pf(NN, std::uint8_t(ScalarBulk));
  std::vector<std::uint8_t> nrm(NN, std::uint8_t(NrmNone));
  std::vector<Real> qw(NN, Real(0)), pw(NN, Real(0));
  for (int z = 0; z < nz; ++z)
   for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const std::size_t id = std::size_t(node_id(x, y, z, nx, ny));
      const bool xl = (x == 0), xr = (x == nx - 1), yb = (y == 0), yt = (y == ny - 1);
      if ((xl || xr) && (yb || yt)) spec[id] = RegWallSpec{NrmCorner, 0, 0, 0};
      else if (yb) spec[id] = RegWallSpec{NrmYm, 0, 0, 0};
      else if (yt) spec[id] = RegWallSpec{NrmYp, 0, 0, 0};
      else if (xl) spec[id] = RegWallSpec{NrmXm, 0, 0, 0};
      else if (xr) spec[id] = RegWallSpec{NrmXp, 0, 0, 0};
      // charge: injector on the node, zero gradient at the collector, mirrors
      // on the sides. The plates win at the corners.
      if (yb)      { qf[id] = ScalarMoment;  qw[id] = Real(q0); }
      else if (yt) { qf[id] = ScalarOutflow; }
      else if (xl || xr) { qf[id] = ScalarSpecular; nrm[id] = xl ? NrmXm : NrmXp; }
      // potential: phi = dphi at the injector, 0 at the collector, mirrors.
      if (yb || yt) { pf[id] = ScalarMoment; pw[id] = yb ? Real(dphi) : Real(0); }
      else if (xl || xr) pf[id] = ScalarSpecular;
    }

  //---- solvers ----------------------------------------------------------------
  backend::Fluid fl(nx, ny, nz, Op::CentralMoments, Real(nu));
  fl.set_geometry(geo);
  fl.set_regularized_walls(spec);
  // WITHOUT THIS THE CHARGE NEVER SEES THE FLOW. The velocity field a coupled
  // solver advects with is opt-in here -- 12 bytes per node against 108 for the
  // populations, which an uncoupled run should not pay -- and ux_device()
  // returns NULL until it is asked for. That null then meets the defensive
  // `p.ux ? p.ux[n] : 0` in ehd_node and becomes a silent zero: the Coulomb
  // force still drives the fluid, the fluid still moves, and the charge is
  // simply never advected by it. The instability then grows on drift alone and
  // saturates at u_max/u0 = 0.578 where the Kokkos twin gives 11.583.
  fl.enable_velocity_output();

  backend::Charge chg(nx, ny, nz, Real(Dq), Real(0), ScalarOp::ChargeCM);
  chg.set_periodicity(false, false, true);
  chg.set_geometry(qf, qw);
  chg.set_specular_walls(nrm);

  backend::Scalar pot(nx, ny, nz, Real(o.beta), Real(0), ScalarOp::BGK);
  pot.set_periodicity(false, false, true);
  pot.set_geometry(pf, pw);
  pot.set_specular_walls(nrm);

  //---- coupling fields --------------------------------------------------------
  Field kx(N), ky(N), kz(N), Fx(N), Fy(N), Fz(N), src(N), qprev(N);
  BodyForce b; b.Fx = Fx.data(); b.Fy = Fy.data(); b.Fz = Fz.data();
  fl.set_force(b, ForceField);
  chg.advect_with(kx.data(), ky.data(), kz.data());

  fl.initialise_with(RestInit{});
  pot.initialise_with(PotentialInit{H});
  chg.initialise_with(ChargeInit{H, Real((hydro ? 0.0 : o.amp) * q0),
                                 Real(double(H) / 8.0), nz});

  EhdParams ep;
  ep.phi = pot.field_device(); ep.q = chg.field_device();
  ep.kx = kx.data(); ep.ky = ky.data(); ep.kz = kz.data();
  ep.Fx = Fx.data(); ep.Fy = Fy.data(); ep.Fz = Fz.data();
  ep.src = src.data(); ep.qprev = qprev.data();
  ep.nx = nx; ep.ny = ny; ep.nz = nz; ep.H = H;
  ep.K = Real(K); ep.eps = Real(eps); ep.beta = Real(o.beta);
  ep.sidewalls = true;

  //---- march ------------------------------------------------------------------
  const std::size_t steps = std::size_t((hydro ? o.tfh : o.tf) * t0);
  const std::size_t probe = std::size_t(t0 / 20) ? std::size_t(t0 / 20) : 1;
  Out r;
  double sNe = 0, sNe2 = 0, ring[20] = {0};
  int nprobe = 0, frame = 0;
  std::vector<Real> hq, hrho, hux, huy, huz, hky;
  const double tavg0 = o.tf - o.tavg;

  for (std::size_t t = 0; t < steps; ++t) {
    // A null velocity here means "hydrostatic reference", and it must mean ONLY
    // that -- see enable_velocity_output above for what it meant once.
    ep.ux = hydro ? nullptr : fl.ux_device();
    ep.uy = hydro ? nullptr : fl.uy_device();
    ep.uz = hydro ? nullptr : fl.uz_device();
    if (!hydro && (!ep.ux || !ep.uy || !ep.uz)) {
      std::printf("  the fluid is not publishing a velocity field; the charge "
                  "would advect on drift alone\n");
      std::abort();
    }
    ehd_pass(ep);
    pot.add_source(src.data());
    pot.step();
    chg.step();
    if (!hydro) fl.step();
    pot.compute_field();
    chg.compute_field();

    if ((t + 1) % probe == 0 || t + 1 == steps) {
      chg.field_to_host(hq);
      ky.to_host(hky);              // the VERTICAL drift, u_y + K E_y
      if (!hydro) fl.macroscopic_to_host(hrho, hux, huy, huz);
      double peak = 0, qlo = 1e300, qhi = -1e300, sflux = 0;
      long ncell = 0;
      for (int z = 0; z < nz; ++z)
       for (int y = 0; y <= H; ++y)
        for (int x = 0; x < nx; ++x) {
          const std::size_t n = std::size_t(node_id(x, y, z, nx, ny));
          const double q = double(hq[n]);
          if (!hydro) {
            const double a = double(hux[n]), c = double(huy[n]);
            peak = std::fmax(peak, std::sqrt(a * a + c * c));
          }
          qlo = std::fmin(qlo, q); qhi = std::fmax(qhi, q);
          double dqdy;
          const std::size_t up = std::size_t(node_id(x, y + 1 <= H ? y + 1 : H, z, nx, ny));
          const std::size_t dn = std::size_t(node_id(x, y - 1 >= 0 ? y - 1 : 0, z, nx, ny));
          if (y == 0)
            dqdy = -1.5 * q + 2.0 * double(hq[up])
                   - 0.5 * double(hq[std::size_t(node_id(x, 2, z, nx, ny))]);
          else if (y == H)
            dqdy = 1.5 * q - 2.0 * double(hq[dn])
                   + 0.5 * double(hq[std::size_t(node_id(x, H - 2, z, nx, ny))]);
          else
            dqdy = 0.5 * (double(hq[up]) - double(hq[dn]));
          sflux += q * double(hky[n]) - Dq * dqdy;
          ++ncell;
        }
      r.Ie = sflux / double(ncell);
      r.umax = peak / o.u0; r.qlo = qlo / q0; r.qhi = qhi / q0;
      const double tt = double(t + 1) / t0;
      const double exc = (r.qlo < 0.0) ? -r.qlo : (r.qhi > 1.0 ? r.qhi - 1.0 : 0.0);
      if (exc > r.worst) { r.worst = exc; r.t_worst = tt; }
      const double ne = I0 > 0 ? r.Ie / I0 : 0.0;
      if (!hydro && tt >= tavg0) { sNe += ne; sNe2 += ne * ne; ++r.nacc; }
      if (o.watch) {
        // THE LATERAL SPREAD OF q AT MID-HEIGHT is what has to grow: the base
        // state is x-uniform and its Coulomb force is balanced hydrostatically,
        // so a peak velocity alone cannot say whether the instability is
        // running or the seed is merely sitting there.
        double qmin = 1e300, qmax = -1e300, fmax = 0;
        std::vector<Real> hfy; Fy.to_host(hfy);
        for (int x = 0; x < nx; ++x) {
          const std::size_t m = std::size_t(node_id(x, H / 2, 0, nx, ny));
          qmin = std::fmin(qmin, double(hq[m])); qmax = std::fmax(qmax, double(hq[m]));
        }
        for (std::size_t m = 0; m < hfy.size(); ++m)
          fmax = std::fmax(fmax, std::fabs(double(hfy[m])));
        std::printf("      t/t0 %7.2f   u_max/u0 = %8.3f   Ie = %.6e   Ne = %7.4f"
                    "   q/q0 [%7.4f, %7.4f]   dq_x = %.3e   max|Fy| = %.3e\n",
                    tt, r.umax, r.Ie, ne, r.qlo, r.qhi, (qmax - qmin) / q0, fmax);
      }
      // ---- frames ---------------------------------------------------------
      // Written at probe time because the fields are already on the host there;
      // a separate cadence would mean a second transfer for no extra
      // information. q, |u| and phi, so ParaView can colour by any of them.
      if (!o.dump.empty() && !hydro && o.dumpn > 0 && (nprobe % o.dumpn) == 0) {
        std::vector<Real> hp;
        pot.field_to_host(hp);
        std::vector<float> f(std::size_t(nx) * std::size_t(ny));
        char tag[32];
        std::snprintf(tag, sizeof tag, "_%04d.bin", frame);
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            f[std::size_t(y) * std::size_t(nx) + std::size_t(x)] =
                float(double(hq[std::size_t(node_id(x, y, 0, nx, ny))]) / q0);
        write_plane(o.dump + "_q" + tag, nx, ny, f);
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            const std::size_t m = std::size_t(node_id(x, y, 0, nx, ny));
            f[std::size_t(y) * std::size_t(nx) + std::size_t(x)] =
                float(std::sqrt(double(hux[m]) * double(hux[m]) +
                                double(huy[m]) * double(huy[m])) / o.u0);
          }
        write_plane(o.dump + "_u" + tag, nx, ny, f);
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            f[std::size_t(y) * std::size_t(nx) + std::size_t(x)] =
                float(double(hp[std::size_t(node_id(x, y, 0, nx, ny))]));
        write_plane(o.dump + "_phi" + tag, nx, ny, f);
        ++frame;
      }

      const double ago = ring[nprobe % 20];
      ring[nprobe % 20] = r.Ie;
      ++nprobe;
      if (hydro && nprobe > 20 && std::fabs(r.Ie - ago) < 1e-6 * std::fabs(r.Ie)) break;
      std::fflush(stdout);
    }
  }
  if (o.profile) {
    std::vector<Real> hp;
    pot.field_to_host(hp);
    chg.field_to_host(hq);
    ky.to_host(hky);
    // Sampled at MID-X, not averaged over x: the two specular columns are 18 %
    // of an average at nx = 11, so an x-average hides where an error lives.
    std::printf("      y     q/q0        phi        K.Ey+u      phi(x=0)\n");
    for (int y = 0; y <= H; y += (H / 10 > 0 ? H / 10 : 1)) {
      const std::size_t m = std::size_t(node_id(nx / 2, y, 0, nx, ny));
      const std::size_t e = std::size_t(node_id(0, y, 0, nx, ny));
      std::printf("   %4d  %9.5f  %9.5f  %11.6f  %11.6f\n", y,
                  double(hq[m]) / q0, double(hp[m]), double(hky[m]), double(hp[e]));
    }
  }
  if (!hydro && r.nacc) {
    r.Ne = sNe / r.nacc;
    const double v = sNe2 / r.nacc - r.Ne * r.Ne;
    r.Ne_rms = v > 0 ? std::sqrt(v) : 0.0;
  }
  return r;
}

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "-n"     && i + 1 < argc) o.n     = std::atoi(argv[++i]);
    else if (a == "-nz"    && i + 1 < argc) o.nz    = std::atoi(argv[++i]);
    else if (a == "-t"     && i + 1 < argc) o.T     = std::atof(argv[++i]);
    else if (a == "-u0"    && i + 1 < argc) o.u0    = std::atof(argv[++i]);
    else if (a == "-c"     && i + 1 < argc) o.C     = std::atof(argv[++i]);
    else if (a == "-m"     && i + 1 < argc) o.M     = std::atof(argv[++i]);
    else if (a == "-sc"    && i + 1 < argc) o.Sc    = std::atof(argv[++i]);
    else if (a == "-alpha" && i + 1 < argc) o.alpha = std::atof(argv[++i]);
    else if (a == "-beta"  && i + 1 < argc) o.beta  = std::atof(argv[++i]);
    else if (a == "-tf"    && i + 1 < argc) o.tf    = std::atof(argv[++i]);
    else if (a == "-tavg"  && i + 1 < argc) o.tavg  = std::atof(argv[++i]);
    else if (a == "-tfh"   && i + 1 < argc) o.tfh   = std::atof(argv[++i]);
    else if (a == "-amp"   && i + 1 < argc) o.amp   = std::atof(argv[++i]);
    else if (a == "-watch")                 o.watch = true;
    else if (a == "-profile")               o.profile = true;
    else if (a == "-dump"  && i + 1 < argc) o.dump  = argv[++i];
    else if (a == "-dumpn" && i + 1 < argc) o.dumpn = std::atoi(argv[++i]);
  }

  std::printf("Closed square EHD cavity   %s   D3Q27 fluid + charge / D3Q7 potential"
              "   %s\n  C = %g  M = %g  Sc = %g   Sec. 3.2.3 of Patnaik et al. (2025)\n\n",
              backend::on_device ? "CUDA native" : "HOST reference",
              sizeof(Real) == 4 ? "FP32" : "FP64", o.C, o.M, o.Sc);

  const Out h = solve(o, true, 0.0, true);
  const int H = o.n - 1;
  const double K = o.u0 * double(H), eps = o.M * o.M * K * K;
  const double q0 = eps * o.C / (double(H) * double(H));
  const double ja = analytic_current(K, eps, q0, 1.0, double(H));
  std::printf("    I0 = %.6e   (D = 0 analytic %.6e, %+.2f %%)\n\n", h.Ie, ja,
              100.0 * (h.Ie - ja) / ja);

  const Out f = solve(o, false, h.Ie, true);
  std::printf("    u_max/u0 = %.3f   Ne = %.4f +/- %.4f  (%d samples)\n",
              f.umax, f.Ne, f.Ne_rms, f.nacc);
  const double eps_q = (sizeof(Real) == 4) ? 1e-5 : 1e-9;
  std::printf("    q/q0 in [%.4f, %.4f] at the end", f.qlo, f.qhi);
  if (f.worst > 0.0)
    std::printf("   worst excursion %.4f q0 at t/t0 = %.2f%s\n", f.worst, f.t_worst,
                // The tolerance has to be the PRECISION's, not a fixed 1e-9:
                // the injector holds q0 exactly, so q/q0 sits at 1 + a few ulp
                // in FP32 and a 1e-9 window calls every healthy run "STILL
                // OUT". A verdict line that cries wolf is one nobody reads.
                (f.qlo >= -eps_q && f.qhi <= 1.0 + eps_q) ? "  (recovered)"
                                                         : "  (STILL OUT)");
  else
    std::printf("   stayed inside [0, q0] throughout\n");
  return 0;
}
