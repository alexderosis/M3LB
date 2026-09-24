//==============================================================================
//  ICE ON WATER AT EQUILIBRIUM -- the GPU twin of the parent's
//  validation/ice_equilibrium.cpp, and the cross-check of this tree's enthalpy
//  scalar (enthalpy.cuh) and implicit drag (ForceDarcy, core.cuh) against it.
//
//  The same case, argued in full in the parent's banner: water below 4 C under
//  ice, stably stratified by the density anomaly, so the equilibrium is exact
//  two-slab conduction, h/H = k_i |T_t| / (k_i |T_t| + k_w T_b); a band [-d, d]
//  replaces that by its own exact state through the Kirchhoff transform. Water's
//  k_i/k_w = 3.944 and C_i/C_w = 0.4824, La = 2, the anomaly at the
//  Gebhart-Mollendorf q sized so the layer would sit at Ra = 5e3 if its sign
//  were wrong, and seeded. D3Q27 central moments + D3Q7 EnthalpyBGK, halfway
//  walls: bounce-back, and anti-bounce-back handed the SENSIBLE enthalpy.
//
//  WHAT THIS CASE IS FOR: THE TWO TREES MUST AGREE. They share no header --
//  the material law, its inversion, the collision, the drag and the coupling
//  pass are all written twice -- so agreement on the converged front and
//  profile is evidence that both are right, and disagreement is a bug in one.
//  The parent's numbers at H = 32 are printed beside this tree's.
//
//  usage: ice_equilibrium [-H 32] [-band d] [-case 0|1]
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/ehd.cuh"          // Field
#include "lbm/enthalpy.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

constexpr double kKi  = 2.25 / 0.5705;       // k_ice / k_water
constexpr double kCi  = 2027.0 / 4202.0;     // (rho c_p)_ice / (rho c_p)_water
constexpr double kLa  = 2.0;
constexpr double kAlw = 0.02, kNu = 0.02;
constexpr double kQ   = 1.894816, kTm = 4.029325;
constexpr double kAsol = 100.0, kEps = 1e-3, kRa = 5e3;

struct IceInit {                             // the parent's seed, node for node
  int H;
  double Tt, Tb, h0, kx;
  PhaseChange m;
  LBM_HD Real operator()(int x, int y, int) const {
    const double s = (double(y) - 0.5) / double(H);
    const double yf = 1.0 - h0;
    if (s >= yf) return m.enthalpy_of(Real(Tt * (s - yf) / (1.0 - yf)), Real(0));
    double T = Tb * (1.0 - s / yf);
    T += 0.02 * Tb * sin(kx * double(x)) * sin(M_PI * s / yf);
    return m.enthalpy_of(Real(T), Real(1));
  }
};

struct AtRest {
  LBM_HD Macro operator()(int, int, int) const { return Macro{Real(1), Real(0), Real(0), Real(0)}; }
};

struct Result { double h_fl, h_T, T_err, u_ratio; long steps; bool converged, finite; };

Result run(int H, double Tt, double Tb, double band) {
  const int nx = H, ny = H + 2, nz = 1;
  const long N = long(nx) * ny * nz;

  // The exact equilibrium (Kirchhoff), exactly as the parent computes it.
  const double kl = 1.0, ks = kKi, dk = kl - ks, bd = band;
  auto K = [&](double T) {
    auto bnd = [&](double t) { return ks * t + (bd > 0 ? dk / (4 * bd) * ((t + bd) * (t + bd) - bd * bd) : 0.0); };
    if (T >= bd)  return bnd(bd) + kl * (T - bd);
    if (T <= -bd) return bnd(-bd) + ks * (T + bd);
    return bnd(T);
  };
  const double q  = (K(Tb) - K(Tt)) / double(H);
  const double he = 1.0 - K(Tb) / q / double(H);
  auto T_exact = [&](double s) {
    const double target = K(Tb) - q * s;
    double lo = Tt, hi = Tb;
    for (int it = 0; it < 80; ++it) { const double mid = 0.5 * (lo + hi); (K(mid) > target ? hi : lo) = mid; }
    return 0.5 * (lo + hi);
  };
  const double he_iso = kKi * (-Tt) / (kKi * (-Tt) + Tb);
  const double dw = (1.0 - he_iso) * double(H);
  const double w = kRa * kNu * kAlw / (std::pow(Tb, kQ) * dw * dw * dw);
  const double u_ff = std::sqrt(w * std::pow(Tb, kQ) * dw);
  const double T0 = kTm - std::pow((std::pow(kTm, kQ + 1.0) - std::pow(kTm - Tb, kQ + 1.0)) /
                                   ((kQ + 1.0) * Tb), 1.0 / kQ);

  PhaseChange pc;
  pc.T_s = Real(-band); pc.T_l = Real(band);
  pc.cp_s = Real(kCi); pc.cp_l = Real(1);
  pc.k_s = Real(kKi * kAlw); pc.k_l = Real(kAlw);
  pc.La = Real(kLa); pc.E_datum = Real(0);
  pc.normalise();
  const Real H_bot = pc.enthalpy_of(Real(Tb), Real(1)), H_top = pc.enthalpy_of(Real(Tt), Real(0));

  backend::Fluid  fl(nx, ny, nz, Op::CentralMoments, Real(kNu));
  backend::Scalar sc(nx, ny, nz, Real(kAlw), Real(0.5) * (H_bot + H_top), ScalarOp::EnthalpyBGK);
  sc.set_material(pc);

  std::vector<std::uint8_t> ff(static_cast<std::size_t>(N), std::uint8_t(Fluid));
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(N), std::uint8_t(ScalarBulk));
  std::vector<Real> sw(static_cast<std::size_t>(N), Real(0));
  for (int x = 0; x < nx; ++x) {
    const std::size_t lo = std::size_t(node_id(x, 0, 0, nx, ny)), hi = std::size_t(node_id(x, ny - 1, 0, nx, ny));
    ff[lo] = Solid; ff[hi] = Solid;
    sf[lo] = ScalarDirichlet; sf[hi] = ScalarDirichlet;
    sw[lo] = H_bot - pc.La;   sw[hi] = H_top;          // the SENSIBLE enthalpy
  }
  fl.set_geometry(ff);
  sc.set_geometry(sf, sw);
  fl.enable_velocity_output();
  sc.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  Field Fx(N), Fy(N), Fz(N), Av(N);
  BodyForce b;
  b.Fx = Fx.data(); b.Fy = Fy.data(); b.Fz = Fz.data(); b.A = Av.data();
  fl.set_force(b, ForceDarcy);

  const double h0 = std::max(0.05, he - 0.15);
  fl.initialise_with(AtRest{});
  sc.initialise_with(IceInit{H, Tt, Tb, h0, 2.0 * M_PI / double(nx), pc});

  MeltParams mp;
  mp.H = sc.field_device(); mp.Fy = Fy.data(); mp.A = Av.data(); mp.pc = pc;
  mp.w = Real(w); mp.q = Real(kQ); mp.Tm = Real(kTm); mp.T0 = Real(T0);
  mp.A_solid = Real(kAsol); mp.eps = Real(kEps); mp.N = N;

  Result r{0, 0, 0, 0, 0, false, true};
  std::vector<Real> Hf, prev, rho, ux, uy, uz;
  sc.field_to_host(prev);
  const double span = double(H_bot - H_top);
  const long probe = 2000, cap = 4000000;
  for (long t = 0; t < cap; t += probe) {
    for (long k = 0; k < probe; ++k) { sc.compute_field(); melt_pass(mp); fl.step(); sc.step(); }
    r.steps = t + probe;
    sc.compute_field();
    sc.field_to_host(Hf);
    double dmax = 0;
    for (int y = 1; y <= H; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t n = std::size_t(node_id(x, y, 0, nx, ny));
        if (!std::isfinite(double(Hf[n]))) { r.finite = false; break; }
        dmax = std::max(dmax, std::fabs(double(Hf[n]) - double(prev[n])));
      }
    if (!r.finite) break;
    prev = Hf;
    // FP32 cannot resolve a 1e-10 change; its floor is ~1e-7 of the span.
    if (dmax / span < (sizeof(Real) == 4 ? 2e-6 : 1e-10)) { r.converged = true; break; }
  }

  // The same three measurements as the parent: f_l integral, T = 0 crossing,
  // pointwise T against the exact profile; and the residual velocity.
  fl.macroscopic_to_host(rho, ux, uy, uz);
  double liq = 0, y0sum = 0, terr = 0, umax = 0;
  for (int x = 0; x < nx; ++x) {
    double y0 = 0;
    for (int y = 1; y <= H; ++y) {
      const std::size_t n = std::size_t(node_id(x, y, 0, nx, ny));
      const double T = double(pc.temperature_of(Hf[n]));
      liq += double(pc.liquid_fraction(Hf[n]));
      terr = std::max(terr, std::fabs(T - T_exact(double(y) - 0.5)));
      umax = std::max(umax, std::hypot(double(ux[n]), double(uy[n])));
      if (y < H && y0 == 0) {
        const double T2 = double(pc.temperature_of(Hf[std::size_t(node_id(x, y + 1, 0, nx, ny))]));
        if (T > 0 && T2 <= 0) y0 = double(y) - 0.5 + T / (T - T2);
      }
    }
    y0sum += y0;
  }
  r.h_fl = 1.0 - liq / double(nx) / double(H);
  r.h_T = 1.0 - (y0sum / double(nx)) / double(H);
  r.T_err = terr / (Tb - Tt);
  r.u_ratio = umax / u_ff;
  (void)he;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  int H = 32, only = -1;
  double band = -1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-H") && i + 1 < argc) H = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-band") && i + 1 < argc) band = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "-case") && i + 1 < argc) only = std::atoi(argv[++i]);
  }
  const auto dev = backend::device_info();
  std::printf("ice_equilibrium (GPU/ twin): D3Q27 CM + D3Q7 EnthalpyBGK + ForceDarcy, %s, %s\n\n",
              dev.name.c_str(), sizeof(Real) == 8 ? "FP64" : "FP32");

  // The parent's measurements at H = 32 (validation/ice_equilibrium.cpp,
  // 2026-09-24): h from the T = 0 crossing and from the f_l integral.
  struct Case { double Tt, Tb; double par_T[2], par_fl[2]; };
  const Case cases[] = {{-2.0, 3.0, {0.700077, 0.709403}, {0.687500, 0.705184}},
                        {-1.0, 3.8, {0.477252, 0.487930}, {0.468750, 0.482453}}};
  const std::vector<double> bands = band >= 0 ? std::vector<double>{band}
                                              : std::vector<double>{0.0, 0.2};
  int rc = 0;
  std::printf("  %-10s %-5s %-12s %-12s %-9s %-12s %-12s %-9s %-9s %-9s %s\n", "T_t,T_b",
              "band", "h(T=0)", "parent", "diff", "h(f_l)", "parent", "diff", "T err",
              "u/u_ff", "steps");
  for (double bd : bands)
    for (int ci = 0; ci < 2; ++ci) {
      if (only >= 0 && ci != only) continue;
      const Case& c = cases[ci];
      const Result r = run(H, c.Tt, c.Tb, bd);
      const int bi = bd > 0 ? 1 : 0;
      const bool compare = (H == 32 && (bd == 0.0 || bd == 0.2));
      const double dT = compare ? r.h_T - c.par_T[bi] : 0, df = compare ? r.h_fl - c.par_fl[bi] : 0;
      // FP64: 1e-5 of H, two orders under a printed digit of the parent's table.
      // FP32: 2e-3 of H -- the front integrates rounding over ~1e5 steps.
      const double tol = sizeof(Real) == 4 ? 2e-3 : 1e-5;
      const bool ok = r.finite && r.converged && r.u_ratio < 1e-4 &&
                      (!compare || (std::fabs(dT) < tol && std::fabs(df) < tol));
      std::printf("  %+4.1f,%+4.1f %-5.2f %-12.9f %-12.6f %+9.1e %-12.9f %-12.6f %+9.1e %-9.2e "
                  "%-9.2e %ld %s\n", c.Tt, c.Tb, bd, r.h_T, compare ? c.par_T[bi] : NAN, dT,
                  r.h_fl, compare ? c.par_fl[bi] : NAN, df, r.T_err, r.u_ratio, r.steps,
                  ok ? "PASS" : "FAIL");
      if (!ok) rc = 1;
    }
  std::printf("\n  %s\n", rc ? "FAIL" : "PASS");
  return rc;
}
