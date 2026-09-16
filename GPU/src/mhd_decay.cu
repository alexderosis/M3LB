//==============================================================================
//  Decaying MHD turbulence in a confined disc, after
//
//    S. Neffaa, W. J. T. Bos & K. Schneider, "The decay of magnetohydrodynamic
//    turbulence in a confined domain", Phys. Plasmas 15, 092304 (2008).
//
//  The CUDA twin of ../demonstrator/mhd_decay.cpp. The two share no headers, and
//  the point of the port is that they should agree: where they do not, one of
//  them is wrong.
//
//  THIS IS QUASI-2D, AND THAT IS THE ONE STRUCTURAL DIFFERENCE FROM THE PARENT.
//  This codebase is D3Q27 + D3Q7 and has no D2Q9, so the 2-D problem runs as one
//  periodic cell in z. In a periodic direction one cell deep wrap(z +/- 1, 1) = z,
//  so a +/-z pair's neighbour IS the node; Esoteric Pull still writes slots i and
//  i+1, two distinct slots at one node, and the population simply stays put --
//  which is what streaming into yourself means. That makes it genuinely 2-D
//  rather than a thin slab. test/host_physics.cpp magnetic_thin() asserts it on
//  the MAGNETIC lattice specifically (nz = 1 against nz = 4, identical to
//  0.00e+00 after 200 steps), because every other magnetic driver here uses
//  nz = 4 and the claim had never been checked where this driver needs it.
//
//  So a disagreement with the parent has THREE candidate causes, not two: a port
//  bug, an implementation difference, or D2Q9 against D3Q27-in-one-cell. The
//  third is real and is not a defect of either side. Orders 0 to 2 of the two
//  lattices agree by construction; the fourth-order ghosts do not.
//
//  THE CROSS-CHECK AGAINST THE PARENT HAS BEEN RUN, AND IT DOES NOT FULLY PASS.
//  Measured 2026-09-16, -regime 1, N = 129, Re = 200, penalised, 600 steps
//  (t/T_e = 0.24), parent = demonstrator/mhd_decay.cpp -wall pen:
//
//                      parent      this      gap
//      E_kin        3.2853e-04  3.0758e-04    6 %
//      E_mag        1.3191e-03  1.2941e-03    2 %
//      enstrophy    1.7078e-05  1.4423e-05   16 %
//      H_c            +0.3055     -0.1164   SIGN
//      Bn/B|wall     4.755e-02   5.309e-01    11x
//
//  The energies agree to a few percent, which is the port working. The last two
//  rows are not explained. Bn/B starts COMPARABLE (8.554e-02 against 6.758e-02
//  at t = 0) and then the parent IMPROVES to 0.048 while this degrades to 0.53 --
//  so the magnetic penalisation is not holding B.n = 0 the way the parent's does,
//  and H_c, which is identical at t = 0 (+0.0438 both), has changed sign by
//  t/T_e = 0.24.
//
//  PRECISION IS ELIMINATED. Rebuilt with -DLBM_DOUBLE, the FP64 run reproduces
//  the FP32 one to EVERY PRINTED DIGIT on all ten columns, so a 32-bit residual
//  is not the cause of either gap. The three candidates left are a port bug, the
//  D3Q27-in-one-cell against D2Q9 difference named above (the fourth-order
//  ghosts differ, and Bn/B is a wall-layer quantity), and a differing definition
//  of the Bn/B shell between the two diagnostics. None has been tested.
//
//  AND A COMPARISON AT DEFAULT FLAGS IS MEANINGLESS, which cost a wrong reading
//  before this one. Without -regime the two build DIFFERENT SPECTRA: this file
//  hard-codes kmax = 5 while the parent uses its own k0 + kw band, so the
//  initial fields differ in enstrophy by 2.1x and in integral scale by 1.45x at
//  equal energy, and the CUDA run then decays 2.4x more slowly for a reason that
//  has nothing to do with the port. Compare under -regime or not at all.
//
//  THE WALL IS VOLUME PENALISATION, the reference's own method, and it replaces
//  both boundaries at once. There is no solid cell and no magnetic wall
//  condition: a smooth indicator chi = (1 + tanh((r - R)/w))/2 marks the
//  exterior, the fluid gains F = -chi u / eps and the induction equation gains
//  S = -chi (B.n) n / eps_m. B.n = 0, NOT B = 0 -- the reference's solid is "a
//  perfect conductor, coated inside with a thin layer of insulant", so the field
//  may not penetrate while the tangential component is free.
//
//  Both terms are EXPLICIT and must be rebuilt from the current fields every
//  step; a force refreshed on the probe interval would be a thousand-step-old
//  velocity and would enforce nothing. What penalisation buys is that u and b are
//  penalised through the SAME chi, so the two boundaries are co-located by
//  construction and there is no half-cell mismatch to price.
//
//  WHAT IS HERE AND WHAT IS NOT. The initial condition is the reference's Eq. (6)
//  spectrum with the Eq. (11) cross-helicity construction, so both columns of
//  Table I are imposed rather than drawn -- see the parent's banner for why that
//  matters and for the two departures it forces. The vector potential is NOT
//  carried here: the parent advects it as a D2Q5 scalar to measure A = (1/2)
//  int a^2, and porting that needs the same treatment on D3Q7 plus its own
//  verification. E/A is therefore absent from this driver's output, and the
//  columns that are present are the ones the two codebases can be compared on.
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/ehd.cuh"          // Field: the host/device-neutral N-element array

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// The same 64-bit LCG the parent uses, so a given -seed draws the same phases in
// both codebases. Reproducing the parent's realisation is the whole point: two
// implementations of one scheme on one field is a comparison, on two fields it
// is not.
//------------------------------------------------------------------------------
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed ? seed : 1) {}
  double uniform() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return double(s >> 11) * (1.0 / 9007199254740992.0);
  }
};

struct Opts {
  int N = 321, nz = 1, regime = 0, kmax = 0;
  double rfac = 19.0 / 40.0, u0 = 0.05, Re = 1000.0, prm = 1.0, alf = 1.0;
  double hct = 0.0, eps = 2.0, epsm = 2.0, smooth = 1.0;
  std::uint64_t seed = 99;
  std::size_t steps = 0, probe = 0;
  Op op = Op::CentralMoments;
};

// Read a host array on the device, for initialise_with.
struct FieldInit {
  const Real* a; int nx, ny;
  LBM_HD Macro operator()(int x, int y, int z) const {
    const long n = long(x) + long(nx) * (long(y) + long(ny) * long(z));
    Macro m; m.rho = Real(1);
    m.ux = a[n]; m.uy = a[n + long(nx) * long(ny)]; m.uz = Real(0);
    return m;
  }
};
struct FieldBInit {
  const Real* b; int nx, ny;
  LBM_HD void operator()(int x, int y, int z, Real B[3]) const {
    const long n = long(x) + long(nx) * (long(y) + long(ny) * long(z));
    B[0] = b[n]; B[1] = b[n + long(nx) * long(ny)]; B[2] = Real(0);
  }
};
struct ZeroU {
  LBM_HD void operator()(int, int, int, Real u[3]) const { u[0] = u[1] = u[2] = Real(0); }
};

//------------------------------------------------------------------------------
// The penalisation pass: F = -chi u / eps and S = -chi (B.n) n / eps_m, both from
// the CURRENT fields. Written as a plain loop over nodes so the host and device
// builds share it; on a device this is one kernel's worth of work per step and is
// the price penalisation charges over a sharp wall.
//------------------------------------------------------------------------------
struct PenParams {
  const Real *chi, *nx_, *ny_, *ux, *uy, *bx, *by;
  Real *Fx, *Fy, *Fz, *Sx, *Sy, *Sz;
  Real ie, iem;
  long N;
};

LBM_HD LBM_INLINE void penalise_node(const PenParams& p, long n) {
  const Real c = p.chi[n];
  p.Fx[n] = -c * p.ux[n] * p.ie;
  p.Fy[n] = -c * p.uy[n] * p.ie;
  p.Fz[n] = Real(0);
  // B.n = 0, NOT B = 0: damp the NORMAL component only.
  const Real bn = p.bx[n] * p.nx_[n] + p.by[n] * p.ny_[n];
  p.Sx[n] = -c * bn * p.nx_[n] * p.iem;
  p.Sy[n] = -c * bn * p.ny_[n] * p.iem;
  p.Sz[n] = Real(0);
}

#if defined(__CUDACC__)
__global__ void penalise_kernel(PenParams p) {
  const long n = blockIdx.x * long(blockDim.x) + threadIdx.x;
  if (n < p.N) penalise_node(p, n);
}
#endif

static void penalise(const PenParams& p) {
#if defined(__CUDACC__)
  const int B = 128;
  penalise_kernel<<<int((p.N + B - 1) / B), B>>>(p);
  LBM_CUDA_CHECK(cudaGetLastError());
#else
  for (long n = 0; n < p.N; ++n) penalise_node(p, n);
#endif
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return argv[++i]; };
    if (a == "-n"      && i + 1 < argc) o.N = std::atoi(next());
    else if (a == "-nz"     && i + 1 < argc) o.nz = std::atoi(next());
    else if (a == "-u"      && i + 1 < argc) o.u0 = std::atof(next());
    else if (a == "-re"     && i + 1 < argc) o.Re = std::atof(next());
    else if (a == "-prm"    && i + 1 < argc) o.prm = std::atof(next());
    else if (a == "-alf"    && i + 1 < argc) o.alf = std::atof(next());
    else if (a == "-seed"   && i + 1 < argc) o.seed = std::strtoull(next(), nullptr, 10);
    else if (a == "-steps"  && i + 1 < argc) o.steps = std::size_t(std::atol(next()));
    else if (a == "-probe"  && i + 1 < argc) o.probe = std::size_t(std::atol(next()));
    else if (a == "-eps"    && i + 1 < argc) o.eps = std::atof(next());
    else if (a == "-epsm"   && i + 1 < argc) o.epsm = std::atof(next());
    else if (a == "-kmax"   && i + 1 < argc) o.kmax = std::atoi(next());
    else if (a == "-op"     && i + 1 < argc) {
      const std::string s = next();
      o.op = (s == "bgk") ? Op::BGK : (s == "trt") ? Op::TRT : Op::CentralMoments;
    } else if (a == "-regime" && i + 1 < argc) {
      o.regime = std::atoi(next());
      if (o.regime < 1 || o.regime > 4) {
        std::printf("  -regime must be 1..4 (Neffaa Table I); got %d\n", o.regime);
        return 2;
      }
      // BOTH columns of Table I, because a regime is the PAIR (E_u/E_B, H_c).
      o.rfac = 19.0 / 40.0;
      if (o.regime == 1) { o.alf = 0.3;   o.hct = 0.012;  }
      if (o.regime == 2) { o.alf = 1.9e4; o.hct = 3.5e-5; }
      if (o.regime == 3) { o.alf = 1.3;   o.hct = 0.27;   }
      if (o.regime == 4) { o.alf = 1.0;   o.hct = 0.045;  }
    }
  }
  const int N = o.N, nz = o.nz;
  const double R = o.rfac * double(N), c = 0.5 * double(N - 1);
  const double nu = o.u0 * 2.0 * R / o.Re, eta = nu / o.prm;
  const double T_e = 2.0 * R / o.u0;
  if (!o.steps) o.steps = std::size_t(50.0 * T_e);
  if (!o.probe) o.probe = std::size_t(T_e);
  const long NT = long(N) * long(N) * long(nz);
  auto rad = [&](int x, int y) {
    const double dx = double(x) - c, dy = double(y) - c;
    return std::sqrt(dx * dx + dy * dy);
  };
  auto solid = [&](int x, int y) { return rad(x, y) > R; };
  auto id = [&](int x, int y, int z) { return long(x) + long(N) * (long(y) + long(N) * long(z)); };

  //---- the initial condition, on the host --------------------------------------
  // Neffaa's Eq. (6), E(k) ~ k / [g + k/k0]^4 with g = 0.98 and k0 = (3/4)sqrt(2)pi.
  // The amplitude PER MODE is not the spectrum: a 2-D shell holds O(k) modes, so
  // |u_k|^2 ~ E(k)/k, and u = curl psi gives |psi_k| ~ 1/(k [g + k/k0]^2).
  std::vector<double> hux(std::size_t(N) * N, 0.0), huy(hux), hbx(hux), hby(hux);
  {
    Rng rng(o.seed);
    struct Mode { double kx, ky, pp, pa, amp; };
    std::vector<Mode> modes;
    const double g_ne = 0.98, k0_ne = 0.75 * std::sqrt(2.0) * M_PI;
    const int kmax = o.regime ? (o.kmax > 0 ? o.kmax : 24) : 5;
    for (int kx = -kmax; kx <= kmax; ++kx)
      for (int ky = 0; ky <= kmax; ++ky) {
        if (kx == 0 && ky == 0) continue;
        const double k = std::sqrt(double(kx * kx + ky * ky));
        if (k > double(kmax)) continue;
        const double d = g_ne + k / k0_ne;
        modes.push_back(Mode{double(kx), double(ky),
                             2.0 * M_PI * rng.uniform(), 2.0 * M_PI * rng.uniform(),
                             1.0 / (k * d * d)});
      }
    std::vector<double> psi(std::size_t(N) * N, 0.0), apot(psi);
    const double L = double(N);
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        double p = 0, a = 0;
        for (const Mode& m : modes) {
          const double ph = 2.0 * M_PI * (m.kx * double(x) + m.ky * double(y)) / L;
          p += m.amp * std::sin(ph + m.pp);
          a += m.amp * std::sin(ph + m.pa);
        }
        const double r = rad(x, y);
        const double tt = (r < R) ? (1.0 - (r / R) * (r / R)) : 0.0;
        psi [std::size_t(y) * N + x] = p * tt * tt;
        apot[std::size_t(y) * N + x] = a * tt * tt;
      }
    auto at = [&](int x, int y) { return std::size_t(y) * N + x; };
    for (int y = 1; y < N - 1; ++y)
      for (int x = 1; x < N - 1; ++x) {
        hux[at(x, y)] =  0.5 * (psi [at(x, y + 1)] - psi [at(x, y - 1)]);
        huy[at(x, y)] = -0.5 * (psi [at(x + 1, y)] - psi [at(x - 1, y)]);
        hbx[at(x, y)] =  0.5 * (apot[at(x, y + 1)] - apot[at(x, y - 1)]);
        hby[at(x, y)] = -0.5 * (apot[at(x + 1, y)] - apot[at(x - 1, y)]);
      }

    // Eq. (11): impose the cross helicity rather than draw it. See the parent's
    // banner -- u_perp taken literally is not solenoidal, so the independent
    // solenoidal draw is used, and it is orthogonalised first because the two
    // fields share a taper and are correlated at +0.24 without it.
    if (o.regime && o.hct > 0.0) {
      double quu = 0, qww = 0, quw = 0;
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          if (solid(x, y)) continue;
          const double ux = hux[at(x, y)], uy = huy[at(x, y)];
          const double wx = hbx[at(x, y)], wy = hby[at(x, y)];
          quu += ux * ux + uy * uy; qww += wx * wx + wy * wy;
          quw += ux * wx + uy * wy;
        }
      const double proj = quw / (quu > 0 ? quu : 1e-300);
      qww = 0; quw = 0;
      for (std::size_t k = 0; k < hux.size(); ++k) {
        hbx[k] -= proj * hux[k]; hby[k] -= proj * huy[k];
      }
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          if (solid(x, y)) continue;
          const double wx = hbx[at(x, y)], wy = hby[at(x, y)];
          qww += wx * wx + wy * wy;
          quw += hux[at(x, y)] * wx + huy[at(x, y)] * wy;
        }
      const double su0 = std::sqrt(quu > 0 ? quu : 1e-300);
      const double sw0 = std::sqrt(qww > 0 ? qww : 1e-300);
      const double Quw = quw / (su0 * sw0);
      const double ctarget = o.hct / (0.5 * std::sqrt(o.alf > 0 ? o.alf : 1e-300));
      auto cos_of = [&](double beta) {
        const double hc = beta + (1.0 - beta) * Quw;
        const double eb = beta * beta + 2.0 * beta * (1.0 - beta) * Quw
                        + (1.0 - beta) * (1.0 - beta);
        return hc / std::sqrt(eb > 0 ? eb : 1e-300);
      };
      double lo = 0.0, hi = 1.0;
      for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        (cos_of(mid) < ctarget ? lo : hi) = mid;
      }
      const double beta = 0.5 * (lo + hi);
      for (std::size_t k = 0; k < hux.size(); ++k) {
        const double ux = hux[k] / su0, uy = huy[k] / su0;
        const double wx = hbx[k] / sw0, wy = hby[k] / sw0;
        hbx[k] = beta * ux + (1.0 - beta) * wx;
        hby[k] = beta * uy + (1.0 - beta) * wy;
      }
      std::printf("  Eq. (11) mixing: beta %.6f gives cos(theta) %.6f "
                  "against Table I's %.6f  (<u.w> = %+.4f)\n",
                  beta, cos_of(beta), ctarget, Quw);
    }

    // normalise to the requested r.m.s. velocity and Alfven ratio
    double su = 0, sb = 0; std::size_t cnt = 0;
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        if (solid(x, y)) continue;
        su += hux[at(x, y)] * hux[at(x, y)] + huy[at(x, y)] * huy[at(x, y)];
        sb += hbx[at(x, y)] * hbx[at(x, y)] + hby[at(x, y)] * hby[at(x, y)];
        ++cnt;
      }
    const double fu = o.u0 / std::sqrt(su / double(cnt));
    const double fb = (o.u0 / std::sqrt(o.alf)) / std::sqrt(sb / double(cnt));
    for (std::size_t k = 0; k < hux.size(); ++k) {
      hux[k] *= fu; huy[k] *= fu; hbx[k] *= fb; hby[k] *= fb;
    }
  }

  //---- solvers ------------------------------------------------------------------
  backend::Magnetic mag(N, N, nz, Real(eta));
  backend::Fluid    fl (N, N, nz, o.op, Real(nu));
  fl.couple_magnetic(mag.Bx_device(), mag.By_device(), mag.Bz_device());
  mag.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  Field chi(NT), pnx(NT), pny(NT), Fx(NT), Fy(NT), Fz(NT), Sx(NT), Sy(NT), Sz(NT);
  {
    const std::size_t NTS = static_cast<std::size_t>(NT);
    std::vector<Real> h(NTS), hx(NTS), hy(NTS);
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const long n = id(x, y, z);
          h[std::size_t(n)] = Real(0.5 * (1.0 + std::tanh((rad(x, y) - R) / o.smooth)));
          const double dx = double(x) - c, dy = double(y) - c;
          const double r = std::max(rad(x, y), 1e-12);
          hx[std::size_t(n)] = Real(dx / r);
          hy[std::size_t(n)] = Real(dy / r);
        }
    chi.from_host(h); pnx.from_host(hx); pny.from_host(hy);
  }
  BodyForce bf; bf.Fx = Fx.data(); bf.Fy = Fy.data(); bf.Fz = Fz.data();
  fl.set_force(bf, ForceField);
  mag.set_source(Sx.data(), Sy.data(), Sz.data());

  // upload the initial fields
  Field u0f(2 * NT), b0f(2 * NT);
  {
    std::vector<Real> hu(std::size_t(2 * NT), Real(0)), hb(std::size_t(2 * NT), Real(0));
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const long n = id(x, y, z);
          const std::size_t s = std::size_t(y) * N + x;
          hu[std::size_t(n)]      = Real(hux[s]);
          hu[std::size_t(n + NT)] = Real(huy[s]);
          hb[std::size_t(n)]      = Real(hbx[s]);
          hb[std::size_t(n + NT)] = Real(hby[s]);
        }
    u0f.from_host(hu); b0f.from_host(hb);
  }
  fl.initialise_with(FieldInit{u0f.data(), N, N});
  mag.initialise_with(FieldBInit{b0f.data(), N, N}, ZeroU{});

  // The velocity array must exist before the first penalisation pass reads it.
  // couple_magnetic allocates it, but say so rather than rely on that: a null
  // here would silently make the wall inert, which is what CLAUDE.md records
  // costing a wrong EHD answer.
  if (!fl.ux_device() || !fl.uy_device()) {
    std::printf("  the fluid is not publishing a velocity field; the "
                "penalisation would read nothing\n");
    return 1;
  }

  std::printf("Decaying MHD turbulence in a confined disc -- Neffaa, Bos & "
              "Schneider (2008)   CUDA twin\n");
  std::printf("precision %s   D3Q27 fluid + D3Q7 magnetic   nz %d (quasi-2D)   "
              "operator %s\n", sizeof(Real) == 8 ? "FP64" : "FP32", nz,
              o.op == Op::BGK ? "BGK" : o.op == Op::TRT ? "TRT" : "CM");
  if (o.regime) std::printf("  NEFFAA REGIME %d (Table I)   r = (19/20)pi in a 2pi box\n", o.regime);
  std::printf("  N %d   R %.2f   Re %.0f   Pm %.2f   nu %.5f   eta %.5f\n",
              N, R, o.Re, o.prm, nu, eta);
  std::printf("  u_rms %.4f (Ma %.4f)   E_kin/E_mag %.3f   penalised walls, "
              "B.n = 0   eps %.2f / %.2f   seed %llu\n",
              o.u0, o.u0 / 0.5773502692, o.alf, o.eps, o.epsm,
              (unsigned long long)o.seed);
  std::printf("  one turnover 2R/u0 = %.0f steps; running %zu (%.1f turnovers)\n\n",
              T_e, o.steps, double(o.steps) / T_e);
  std::printf("  %8s %8s %12s %12s %10s %12s %12s %8s %8s %9s %10s\n",
              "step", "t/T_e", "E_kin", "E_mag", "E_k/E_m", "enstrophy",
              "<j^2>", "L_u", "L_b", "H_c", "Bn/B|wall");
  std::printf("  %s\n", std::string(124, '-').c_str());

  PenParams pp;
  pp.chi = chi.data(); pp.nx_ = pnx.data(); pp.ny_ = pny.data();
  pp.Fx = Fx.data(); pp.Fy = Fy.data(); pp.Fz = Fz.data();
  pp.Sx = Sx.data(); pp.Sy = Sy.data(); pp.Sz = Sz.data();
  pp.ie = Real(1.0 / o.eps); pp.iem = Real(1.0 / o.epsm); pp.N = NT;

  std::vector<Real> hr, hu, hv, hw, bx, by, bz;
  for (std::size_t t = 0; t <= o.steps; ++t) {
    if (t % o.probe == 0 || t == o.steps) {
      fl.macroscopic_to_host(hr, hu, hv, hw);
      mag.field_to_host(bx, by, bz);
      double ek = 0, em = 0, hc = 0, en = 0, jj = 0, bn = 0, bt = 0;
      std::size_t cnt = 0, dcnt = 0;
      bool fin = true;
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          if (solid(x, y)) continue;
          const long n = id(x, y, 0);
          const double a = hu[std::size_t(n)], b = hv[std::size_t(n)];
          const double p = bx[std::size_t(n)], q = by[std::size_t(n)];
          if (!std::isfinite(a) || !std::isfinite(p)) { fin = false; continue; }
          ek += 0.5 * (a * a + b * b);
          em += 0.5 * (p * p + q * q);
          hc += a * p + b * q;
          ++cnt;
          const double rr = rad(x, y);
          if (rr > R - 2.0 && rr < R + 2.0) {
            const double dx = (double(x) - c) / std::max(rr, 1e-12);
            const double dy = (double(y) - c) / std::max(rr, 1e-12);
            bn = std::max(bn, std::fabs(p * dx + q * dy));
            bt = std::max(bt, std::sqrt(p * p + q * q));
          }
          if (x > 0 && y > 0 && x < N - 1 && y < N - 1 &&
              !solid(x + 1, y) && !solid(x - 1, y) && !solid(x, y + 1) && !solid(x, y - 1)) {
            const double w = 0.5 * (hv[std::size_t(id(x + 1, y, 0))] - hv[std::size_t(id(x - 1, y, 0))])
                           - 0.5 * (hu[std::size_t(id(x, y + 1, 0))] - hu[std::size_t(id(x, y - 1, 0))]);
            const double j = 0.5 * (by[std::size_t(id(x + 1, y, 0))] - by[std::size_t(id(x - 1, y, 0))])
                           - 0.5 * (bx[std::size_t(id(x, y + 1, 0))] - bx[std::size_t(id(x, y - 1, 0))]);
            en += w * w; jj += j * j; ++dcnt;
          }
        }
      const double inv = cnt ? 1.0 / double(cnt) : 0.0;
      const double dinv = dcnt ? 1.0 / double(dcnt) : 0.0;
      ek *= inv; em *= inv; hc *= inv; en *= dinv; jj *= dinv;
      std::printf("  %8zu %8.2f %12.4e %12.4e %10.4f %12.4e %12.4e %8.2f %8.2f %+9.4f %10.3e%s\n",
                  t, double(t) / T_e, ek, em, em > 0 ? ek / em : 0.0, en, jj,
                  en > 0 ? std::sqrt(2.0 * ek / en) : 0.0,
                  jj > 0 ? std::sqrt(2.0 * em / jj) : 0.0,
                  (ek > 0 && em > 0) ? hc / (2.0 * std::sqrt(ek * em)) : 0.0,
                  bt > 0 ? bn / bt : 0.0, fin ? "" : "   NON-FINITE");
      std::fflush(stdout);
      if (!fin) return 1;
    }
    if (t < o.steps) {
      // The penalisation is explicit and must see u(t) and B(t), not the values
      // at the last dump -- hence a refresh every step and not on the probe.
      fl.refresh_velocity();
      mag.compute_field();
      penalise(pp);
      fl.step();
      mag.step();
    }
  }
  return 0;
}
