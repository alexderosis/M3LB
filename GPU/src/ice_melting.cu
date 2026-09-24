//==============================================================================
//  MELTING ICE IN WATER NEAR 4 C, at Weady et al.'s Rayleigh number -- the GPU
//  twin of the parent's demonstrator/ice_melting.cpp, and the production case
//  for the 3-D scallop study.
//
//  The model is Weady, Tong, Zidovska & Ristroph, PRL 128, 044502 (2022):
//  quadratic density rho* [1 - beta (T - 4)^2], Pr = 12, St = 0.05 T_inf, ice
//  at 0 C throughout. Geometry as the parent demonstrator: a slab of ice
//  against a 0 C wall at x = 0, water held at T_inf on the far wall, top and
//  bottom adiabatic, halfway walls throughout -- and a PERIODIC SPAN in z, so
//  that with nz > 1 the counterflowing layer is free to break its 2-D rolls into
//  3-D cells. nz = 1 is the parent's 2-D case exactly (D3Q27 at nz = 1 IS D2Q9's
//  dynamics for the fluid).
//
//  THE PARENT RAN THIS AT 78x BELOW THEIR Ra, and found the regimes in the
//  paper's order -- rising at 4 C, counterflow at 5-5.6 C, sinking from 6.5 C --
//  with no scallops. The default here is their Ra/T_inf^2 = 2.5e6, which at
//  U_ff = 0.05 puts omega_T near 2: the scalar defaults to the REGULARISED
//  enthalpy collision for that reason (BGK rings there; the parent's CLAUDE.md).
//
//  THE CAVITY HAS NO FAR FIELD, as the parent found: melting a cell of ice
//  takes the heat of ~3.6 cells of water cooled to 0 C. The bulk (water more
//  than 20 cells beyond the front) is watched and the run stops at `-drift` C.
//
//  WHAT IT REPORTS. Per probe: melted fraction, ice thickness in bottom / middle
//  / top bands, the boundary-layer velocity, the bulk temperature, and the
//  SPANWISE r.m.s. of the thickness -- zero for a 2-D state, and the first
//  sign of 3-D cells when it grows. At the end, the thickness map h(y, z) as
//  text for plotting, and optionally VTI fields.
//
//  usage: ice_melting [-T 5.6] [-H 256] [-aspect 2] [-nz 1] [-ice 0.125]
//                     [-ra 2.5e6] [-uff 0.05] [-drift 0.25] [-melt 0.5]
//                     [-sop reg|bgk] [-probe 5000] [-out prefix]
//==============================================================================
#include "lbm/backend.cuh"
#include "lbm/ehd.cuh"          // Field
#include "lbm/enthalpy.cuh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace lbm;

namespace {

struct Opts {
  double Tinf = 5.6, aspect = 2.0, ice0 = 0.125, ra_per_T2 = 2.5e6, Pr = 12.0;
  double st_per_T = 0.05, u_ff = 0.05, drift = 0.25, melt = 0.5, Asol = 100.0;
  double noise = 1e-3;     // seed amplitude, C; 0 reproduces the parent's noiseless 2-D case
  double margin = 0.25;    // water above AND below the ice, as a fraction of its height H
  double sponge = 0.25;    // outer fraction of the width relaxed toward T_inf (0: off)
  double sigma = 0.02;     // sponge strength per step at the far wall
  int dump = 0;            // write h(y, z) every `dump` probes (0: only at the end)
  int H = 256, nz = 1;
  long probe = 5000;
  bool reg = true;
  std::string out;
};

// The initial state: ice at 0 C in x <= xice, water at T_inf beyond it with a
// deterministic 1e-3 C hash seed -- enough to break the spanwise symmetry that
// a 3-D instability has to break, and nothing a 2-D run can feel.
struct MeltInit {
  int xice, y0, y1;
  Real Tinf, amp;
  PhaseChange m;
  LBM_HD Real operator()(int x, int y, int z) const {
    if (x <= xice && y >= y0 && y <= y1) return m.enthalpy_of(Real(0), Real(0));
    std::uint32_t h = std::uint32_t(x) * 73856093u ^ std::uint32_t(y) * 19349663u ^
                      std::uint32_t(z) * 83492791u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    const Real noise = amp * (Real(h & 0xffffu) / Real(65535) - Real(0.5));
    return m.enthalpy_of(Tinf + noise, Real(1));
  }
};

struct AtRest {
  LBM_HD Macro operator()(int, int, int) const { return Macro{Real(1), Real(0), Real(0), Real(0)}; }
};

}  // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto d = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    auto n = [&](int& v) { if (i + 1 < argc) v = std::atoi(argv[++i]); };
    if      (!std::strcmp(argv[i], "-T"))      d(o.Tinf);
    else if (!std::strcmp(argv[i], "-H"))      n(o.H);
    else if (!std::strcmp(argv[i], "-nz"))     n(o.nz);
    else if (!std::strcmp(argv[i], "-aspect")) d(o.aspect);
    else if (!std::strcmp(argv[i], "-ice"))    d(o.ice0);
    else if (!std::strcmp(argv[i], "-ra"))     d(o.ra_per_T2);
    else if (!std::strcmp(argv[i], "-uff"))    d(o.u_ff);
    else if (!std::strcmp(argv[i], "-drift"))  d(o.drift);
    else if (!std::strcmp(argv[i], "-melt"))   d(o.melt);
    else if (!std::strcmp(argv[i], "-noise"))  d(o.noise);
    else if (!std::strcmp(argv[i], "-margin")) d(o.margin);
    else if (!std::strcmp(argv[i], "-sponge")) d(o.sponge);
    else if (!std::strcmp(argv[i], "-sigma"))  d(o.sigma);
    else if (!std::strcmp(argv[i], "-dump"))   n(o.dump);
    else if (!std::strcmp(argv[i], "-probe"))  { double v = 0; d(v); o.probe = long(v); }
    else if (!std::strcmp(argv[i], "-sop") && i + 1 < argc) o.reg = std::strcmp(argv[++i], "bgk") != 0;
    else if (!std::strcmp(argv[i], "-out") && i + 1 < argc) o.out = argv[++i];
    else { std::fprintf(stderr, "ice_melting: unknown option %s\n", argv[i]); return 2; }
  }

  // The ice occupies rows y0..y1 (H of them) against the x = 0 wall, with `margin`
  // of water above and below it, so its top and bottom are exposed like Weady
  // et al.'s suspended block rather than welded to the lids.
  const int H = o.H, W = int(std::lround(o.aspect * H)), mgn = int(std::lround(o.margin * H));
  const int nx = W + 2, ny = H + 2 * mgn + 2, nz = o.nz, y0 = mgn + 1, y1 = mgn + H;
  const int x_sponge = o.sponge > 0 ? W + 1 - int(std::lround(o.sponge * W)) : W + 1;
  const long N = long(nx) * ny * nz;
  const double Ra = o.ra_per_T2 * o.Tinf * o.Tinf;
  const double kappa = o.u_ff * double(H) / std::sqrt(Ra * o.Pr);
  const double nu = o.Pr * kappa;
  const double w = Ra * nu * kappa / (o.Tinf * o.Tinf * double(H) * double(H) * double(H));
  const int xice = int(std::lround(o.ice0 * double(W)));
  const auto dev = backend::device_info();
  std::printf("ice_melting (GPU/): T_inf %.2f C  Ra %.3e (Weady et al. %.3e)  Pr %.0f  St %.3f\n",
              o.Tinf, Ra, 2.5e6 * o.Tinf * o.Tinf, o.Pr, o.st_per_T * o.Tinf);
  std::printf("  box %d x %d x %d: ice %d x %d cells (rows %d..%d), sponge x >= %d (sigma %.3f)\n",
              W, ny - 2, nz, xice, H, y0, y1, x_sponge, o.sponge > 0 ? o.sigma : 0.0);
  std::printf("  kappa %.3e nu %.3e  tau_f %.4f tau_T %.4f  %s scalar  %s, %s\n\n", kappa, nu,
              3 * nu + 0.5, 4 * kappa + 0.5, o.reg ? "regularised" : "BGK", dev.name.c_str(),
              sizeof(Real) == 8 ? "FP64" : "FP32");

  PhaseChange pc;
  pc.T_s = pc.T_l = Real(0);
  pc.cp_s = pc.cp_l = Real(1);
  pc.k_s = pc.k_l = Real(kappa);
  pc.La = Real(1.0 / o.st_per_T); pc.E_datum = Real(0);
  pc.normalise();
  const Real H_ice = pc.enthalpy_of(Real(0), Real(0)), H_wat = pc.enthalpy_of(Real(o.Tinf), Real(1));

  backend::Fluid  fl(nx, ny, nz, Op::CentralMoments, Real(nu));
  backend::Scalar sc(nx, ny, nz, Real(kappa), Real(0.5) * (H_ice + H_wat),
                     o.reg ? ScalarOp::EnthalpyRegularised : ScalarOp::EnthalpyBGK);
  sc.set_material(pc);

  std::vector<std::uint8_t> ff(static_cast<std::size_t>(N), std::uint8_t(Fluid));
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(N), std::uint8_t(ScalarBulk));
  std::vector<Real> sw(static_cast<std::size_t>(N), Real(0));
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t c = std::size_t(node_id(x, y, z, nx, ny));
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) ff[c] = Solid;
        if (x == 0) {          // 0 C behind the ice only; the rest of that wall is adiabatic
          if (y >= y0 && y <= y1) { sf[c] = ScalarDirichlet; sw[c] = H_ice; }       // E = H at f = 0
          else sf[c] = ScalarAdiabatic;
        }
        else if (x == nx - 1)  { sf[c] = ScalarDirichlet; sw[c] = H_wat - pc.La; }  // SENSIBLE E
        else if (y == 0 || y == ny - 1) sf[c] = ScalarAdiabatic;
      }
  fl.set_geometry(ff);
  sc.set_geometry(sf, sw);
  fl.enable_velocity_output();
  sc.advect_with(fl.ux_device(), fl.uy_device(), fl.uz_device());

  Field Fx(N), Fy(N), Fz(N), Av(N), Sp(N);
  BodyForce b;
  b.Fx = Fx.data(); b.Fy = Fy.data(); b.Fz = Fz.data(); b.A = Av.data();
  fl.set_force(b, ForceDarcy);
  fl.initialise_with(AtRest{});
  sc.initialise_with(MeltInit{xice, y0, y1, Real(o.Tinf), Real(o.noise), pc});

  MeltParams mp;
  mp.H = sc.field_device(); mp.Fy = Fy.data(); mp.A = Av.data(); mp.pc = pc;
  mp.w = Real(w); mp.q = Real(2); mp.Tm = Real(4); mp.T0 = Real(o.Tinf);
  mp.A_solid = Real(o.Asol); mp.eps = Real(1e-3); mp.N = N;
  if (o.sponge > 0) {
    mp.S = Sp.data(); mp.H_inf = H_wat; mp.sigma = Real(o.sigma);
    mp.nx = nx; mp.x_sponge = x_sponge; mp.x_end = W;
  }

  // Thickness per (y, z) column: the ice is attached to the x = 0 wall, so the
  // column sum of (1 - f_l) IS the local thickness, to sub-cell precision.
  std::vector<Real> Hf, rho, ux, uy, uz;
  std::vector<double> th(std::size_t(H) * nz);
  auto write_thickness = [&](const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return;
    std::fprintf(fp, "# ice thickness h(y, z) in cells; rows = the %d ice rows bottom to top, "
                     "columns z = 0..%d\n", H, nz - 1);
    for (int y = 0; y < H; ++y) {
      for (int z = 0; z < nz; ++z) std::fprintf(fp, " %.5f", th[std::size_t(z) * H + std::size_t(y)]);
      std::fprintf(fp, "\n");
    }
    std::fclose(fp);
    std::printf("  wrote %s\n", path.c_str());
  };
  auto measure = [&](bool with_flow) {
    sc.compute_field();
    sc.field_to_host(Hf);
    if (with_flow) fl.macroscopic_to_host(rho, ux, uy, uz);
    double vol = 0, bulk = 0; long nb = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = y0; y <= y1; ++y) {
        double t = 0;
        for (int x = 1; x <= W; ++x)
          t += 1.0 - double(pc.liquid_fraction(Hf[std::size_t(node_id(x, y, z, nx, ny))]));
        th[std::size_t(z) * H + std::size_t(y - y0)] = t;
        vol += t;
        for (int x = int(t) + 21; x < x_sponge; ++x, ++nb)
          bulk += double(pc.temperature_of(Hf[std::size_t(node_id(x, y, z, nx, ny))]));
      }
    return std::make_pair(vol, nb ? bulk / double(nb) : o.Tinf);
  };
  auto band = [&](double a, double c) {
    double s = 0; long k = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = int(a * H); y < int(c * H); ++y, ++k) s += th[std::size_t(z) * H + std::size_t(y)];
    return k ? s / double(k) : 0.0;
  };
  auto span_rms = [&]() {                    // r.m.s. over z at fixed y, averaged over y
    if (nz == 1) return 0.0;
    double acc = 0;
    for (int y = 0; y < H; ++y) {
      double m = 0, m2 = 0;
      for (int z = 0; z < nz; ++z) { const double v = th[std::size_t(z) * H + std::size_t(y)]; m += v; m2 += v * v; }
      m /= nz; m2 /= nz;
      acc += std::sqrt(std::max(0.0, m2 - m * m));
    }
    return acc / double(H);
  };
  auto bl_velocity = [&]() {                 // 8 cells beside the ice, middle half of the height
    double s = 0; long k = 0;
    for (int z = 0; z < nz; ++z)
      for (int y = y0 + H / 4; y <= y0 + 3 * H / 4; ++y) {
        const int x0 = int(std::lround(th[std::size_t(z) * H + std::size_t(y - y0)])) + 1;
        for (int x = x0; x < std::min(x0 + 8, W + 1); ++x, ++k)
          s += double(uy[std::size_t(node_id(x, y, z, nx, ny))]);
      }
    return k ? s / double(k) / o.u_ff : 0.0;
  };

  const double vol0 = measure(false).first;
  const double td = double(H) * double(H) / kappa;
  std::printf("  %9s %9s %8s %8s %8s %8s %10s %8s %8s %9s\n", "step", "t/t_diff", "melted",
              "bottom", "middle", "top", "v_BL/U_ff", "T_bulk", "T_max", "span rms");
  const auto t0 = std::chrono::steady_clock::now();
  long step = 0;
  bool finite = true;
  const long cap = long(0.5 * td);
  while (step < cap) {
    for (long k = 0; k < o.probe; ++k) {
      sc.compute_field(); melt_pass(mp);
      if (mp.S) sc.add_source(Sp.data());      // the far field, before the scalar steps
      fl.step(); sc.step();
    }
    step += o.probe;
    const auto [vol, Tb] = measure(true);
    // THE MAXIMUM PRINCIPLE: no interior water should exceed the warm wall.
    // At omega_T ~ 1.99 it does, by ~1 % along the floor where the cold
    // current runs out -- in both trees (the parent never printed it).
    double tmax = -1e300;
    {
      int xm = 0, ym = 0;
      for (int z = 0; z < nz; ++z)
        for (int y = 1; y <= H; ++y)
          for (int x = 1; x <= W; ++x) {
            const double T = double(pc.temperature_of(Hf[std::size_t(node_id(x, y, z, nx, ny))]));
            if (T > tmax) { tmax = T; xm = x; ym = y; }
          }
      (void)xm; (void)ym;
    }
    if (!std::isfinite(vol) || !std::isfinite(Tb)) { std::printf("  NON-FINITE at step %ld\n", step); finite = false; break; }
    const double melted = 1.0 - vol / vol0;
    std::printf("  %9ld %9.4f %8.3f %8.2f %8.2f %8.2f %+10.4f %8.3f %8.3f %9.4f\n", step,
                double(step) / td, melted, band(0, 0.125), band(0.4375, 0.5625), band(0.875, 1.0),
                bl_velocity(), Tb, tmax, span_rms());
    if (o.dump > 0 && !o.out.empty() && (step / o.probe) % o.dump == 0) {
      char tag[48]; std::snprintf(tag, sizeof tag, "_thickness_%09ld.txt", step);
      write_thickness(o.out + tag);
    }
    std::fflush(stdout);
    if (melted >= o.melt || band(0, 1) <= 1.0) break;
    if (std::fabs(Tb - o.Tinf) > o.drift) {
      std::printf("  bulk has drifted %.3f C from T_inf: stopping here\n", Tb - o.Tinf);
      break;
    }
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("\n  %ld steps in %.1f s: %.1f MLUPS (fluid + scalar + coupling)\n", step, secs,
              double(N) * double(step) / secs / 1e6);
  if (finite) {
    const double taper = (band(0, 0.25) - band(0.75, 1.0)) / double(xice);
    std::printf("  taper (bottom - top) / initial = %+.3f   spanwise rms %.4f cells\n", taper, span_rms());
    if (!o.out.empty()) write_thickness(o.out + "_thickness.txt");
  }
  return finite ? 0 : 1;
}
