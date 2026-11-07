//==============================================================================
//  demonstrator/mew_jet -- the MEW jet below the Taylor cone as a slender viscous
//  thread (validation/Jet1D.hpp, validated by validation/jet1d): how fast it
//  lands on its own, against the current it carries.
//
//  WHY. The jet is 10-20 um across, below every free-surface grid this tree can
//  afford (demonstrator/mew_axi resolves the melt under the tip and nothing that
//  leaves it). Its free-landing speed -- the speed it reaches when nothing pulls
//  at its foot -- is this model's estimate of the CRITICAL TRANSLATION SPEED: a
//  collector slower than that must push the jet back, compressing it, and a
//  compressed thread buckles, so the laid fibre stops being straight. MEW
//  printers are run above that speed, and it can be measured by slowing the
//  collector until the lines start to wave.
//
//  THE SET-UP (Ju, Omar, Mirihanage and Kilic; NOMINAL where not measured).
//  PCL leaving the cone at 120 C with 820 Pa s (measured), Arrhenius below that
//  with 31.3 kJ/mol (literature, as in mew_axi), 0.030 N/m (nominal),
//  1080 kg/m^3, c_p 2000 J/kg/K (nominal). The jet carries the flow its fibres
//  show: a straight fibre d across laid at the collector's 35 mm/s carries
//  pi d^2 v / 4, which for the 20.5 um middle of the 19-22 um fibres is
//  0.0116 mm^3/s, 56 % of the 0.0208 mm^3/s feed. It starts at the cone's apex,
//  a0 = 50 um in radius (NOMINAL), 5.0 mm above the collector (the
//  5.5 mm gap less an assumed 0.5 mm cone). The electric pull on a convected
//  charge is I E / Q per unit volume (Jet1D.hpp's banner), with E the MEAN gap
//  field, 6.5 kV over 5.5 mm (NOMINAL: the field near the cone is stronger). The
//  current I is not measured and is the swept unknown. The jet loses heat to air
//  at 25 C through h = Nu k_air / (2a), Nu = 0.4 (NOMINAL: natural convection
//  from a wire this thin is a few tenths), k_air = 0.026 W/m/K. With Nu fixed the
//  loss per length, 2 pi a h = pi Nu k_air, does not depend on the radius, so the
//  temperature falls exponentially over rho c_p Q / (pi Nu k_air) however the
//  jet thins (exact; validation/jet1d's J8 checks it). No crystallisation in
//  flight: the jet cools at a few hundred kelvin a second (183 at the apex, 760
//  at most, nominal), and PCL's glass transition is far below the air, so it
//  stays a supercooled melt whose
//  viscosity keeps rising (Arrhenius, uncapped). Gravity, inertia and surface
//  tension are in, and negligible here.
//
//  WHAT IT SAYS (2026-11-07; data in results/mew_jet/, drawn by
//  tools/plot_mew_jet.py):
//   - THE JET THINS HOT AND EARLY. It cools over 0.76 mm (an e-fold), its
//     viscosity is 10 times its apex value 1.24 mm from the apex, and 81 % of its
//     stretching, ln(U/u0), is done in the first millimetre; it lands at 25.1 C.
//   - IT LANDS ON ITS OWN at 2.1, 6.8, 40 and 314 mm/s carrying 0.01, 0.1, 1 and
//     10 nA -- nearly in proportion to the current above 1 nA (7.9 and 9.1 times
//     per decade) -- and at the collector's 35 mm/s carrying 0.86 nA. So, IF the
//     free-landing speed is the critical translation speed, lines laid straight
//     at 35 mm/s mean a jet carrying less than about a nanoamp. Both ends of that
//     are measurable: the current with an ammeter in the collector's line, the
//     critical speed by slowing the collector until the lines wave.
//   - WHAT MOVES THAT CURRENT, each input changed alone (variants.txt): the field
//     enters only as the product I E, so twice the field halves it (0.43 nA);
//     Nu 0.2 / 0.8: 0.53 / 1.28 nA; an apex radius of 25 / 100 um: 0.57 / 1.05;
//     a jet 3.5 mm long: 1.37; the fan's 11.5 um fibres: 0.59; a jet starting at
//     90 / 60 C: 1.28 / 1.84; Ea 50 kJ/mol: 1.76. So 0.43 to 1.84 nA under
//     everything tried, and 0.14 nA if the jet stayed at 120 C: the cooling is
//     the largest single effect. 8000 steps give the same 0.8590 nA as 4000.
//   - At 0.859 nA the jet lands at 35.0 mm/s and 20.50 um across: the fibre, as
//     mass conservation requires, since the flow was taken from it -- a
//     consistency check, not a prediction.
//   - THE FIELD MEW_AXI COMPUTES BELOW THE DROP (-Efile, from
//     tools/mew_axi_axis_field.py) RAISES that current rather than lowering it,
//     although it is 12-19 kV/mm at the drop: 1.00 nA below the oozing drop at
//     60 s (3.5 mm up; 0.87 nA in a uniform field of the same voltage over the
//     same length), 0.99 nA at 20 s, 0.89 nA below the young drop at 1 s
//     (5.3 mm; 0.77 uniform). The tension at a point of the jet is the pull on
//     all of it BELOW that point, so a pull at the apex stretches nothing: what
//     thins the hot first millimetre is the pull on the long cold part below it,
//     and the r-z field puts its strength at the apex and is weaker than the mean
//     below (0.6-0.95 kV/mm at the collector).
//   - BEHIND THE MOVING COLLECTOR (-drag; Jet1D.hpp's solve_dragged, the string
//     limit), at the collector's 35 mm/s the jet lags 0.49 / 2.4 / 4.4 / 11.4 /
//     15.0 mm at 0.97 / 0.75 / 0.5 / 0.1 / 0.05 of the current that lands it there
//     on its own (mean field; 0.46 / 2.2 / 3.9 / 10.1 / 13.3 mm in the r-z field,
//     whose strong apex holds the jet's start nearer the vertical: 5.2 against
//     11.6 degrees at half that current). At half that current the lag rises
//     steeply just above the free-landing speed U: 0.42 / 1.34 / 2.39 / 5.05 /
//     8.22 mm at 1.02 / 1.1 / 1.25 / 2 / 4 U (mean field). So a straight line laid
//     well above the critical speed trails the cone by millimetres, and the lag
//     at a known speed is a third measurable, beside the current and the critical
//     speed.
//
//  WHAT IT DOES NOT DO: the jet's flow and current are inputs, not outputs (the
//  cone that sets them is below mew_axi's grid); behind the moving collector the
//  jet is a string -- no bending stiffness, so it lands tangentially, with no
//  heel, and nothing is said about the coiling below the critical speed; the field
//  is the mean gap field or mew_axi's on the axis, vertical, a function of height
//  alone, and not changed by the jet's charge; Newtonian -- the jet is stretched
//  at up to 22 1/s (nominal; 10 1/s held at 120 C), just past the 16 1/s the
//  measured curve reaches, and that curve is in shear, not in extension.
//
//  Usage: mew_jet [-dfib UM 20.5] [-vc MMS 35] [-L MM 5.0] [-a0 UM 50] [-T0 C 120]
//                 [-Tair C 25] [-Nu 0.4] [-kair 0.026] [-E KV_PER_MM 1.1818]
//                 [-mu 820] [-Ea 31.3] [-gamma 0.030] [-nocool] [-n 4000]
//                 [-I NA] [-Efile FILE] [-drag [-ndrag 800]] [-out FILE]
//         Without -I: a sweep over the current and the current at which the jet
//         lands at the collector's speed; -out writes the sweep on 41 currents
//         from 0.01 to 100 nA as CSV (I_nA, U_mm_s, d_um, T_C). With -I: that
//         current's landing, and -out writes its profile (z_mm, d_um, u_mm_s,
//         T_C, mu_Pa_s, N_uN). -T0 moves where the jet starts, not where the
//         viscosity was measured (120 C, Opts::Tref). -Efile takes E(z) below the
//         apex from tools/mew_axi_axis_field.py (and the jet's length from its
//         last row unless -L is given). -drag: the lag at -vc against the current
//         and, at -I (else half the current that lands the jet at -vc), against
//         the collector's speed; -out PREFIX writes PREFIX_lag_I.csv,
//         PREFIX_lag_vc.csv and PREFIX_path_<k>.csv (x_mm, z_mm, d_um, u_mm_s, T_C).
//  The data in results/mew_jet/: sweep_nominal.txt and variants.txt are this
//  program's printout (the variants are one flag each, as listed above in
//  WHAT IT SAYS); sweep_*.csv are -out of the sweeps (nominal, -nocool, -Nu 0.2,
//  -Nu 0.8, -a0 25, -a0 100, -T0 60, -Ea 50); profile_nominal.csv is -I 0.859
//  -out and profile_isothermal.csv -nocool -I 0.1404 -out. E_axis_heat_60s,
//  _heat_20s and _iso_1s.csv are tools/mew_axi_axis_field.py on mew_axi's frames
//  R4_heat_best_t059.95, _t019.66 and R4_iso_nu6_t000.98; drag_nominal* is -drag
//  -out, drag_heat60s* is -drag -Efile E_axis_heat_60s.csv -out (printouts in the
//  .txt beside them).
//==============================================================================
#include "../validation/Jet1D.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace jet1d;

namespace {

struct Opts {
  double dfib = 20.5e-6, vc = 35e-3, L = 5.0e-3, a0 = 50e-6, T0 = 120.0, Tair = 25.0;
  double Tref = 120.0;  // where mu0 was measured; NOT the jet's start, which -T0 moves
  double Nu = 0.4, kair = 0.026, E = 6.5e3 / 5.5e-3, mu0 = 820.0, Ea = 31.3e3;  // E in V/m
  double gamma = 0.030, rho = 1080.0, cp = 2000.0, g = 9.81;
  bool cool = true;
  int n = 4000;
  double I = -1.0;  // A; < 0: sweep
  std::string out;
  std::string Efile;                 // -Efile: E(z) along the axis, tools/mew_axi_axis_field.py
  std::vector<double> Ez_z, Ez_E;    // its table, z from the apex [m] and E [V/m]
  bool L_set = false;
  bool drag = false;  // -drag: the jet behind the moving collector (Jet1D.hpp's solve_dragged)
  int ndrag = 800;    // RK4 steps in the vertical tension for it
};

double Q_of(const Opts& o) { return kPi * o.dfib * o.dfib / 4.0 * o.vc; }

Jet make(const Opts& o, double I) {
  Jet J;
  J.Q = Q_of(o);
  J.L = o.L;
  J.rho = o.rho;
  J.gamma = o.gamma;
  J.T_air = o.Tair;
  const double EaR = o.Ea / 8.314462618, TrK = o.Tref + 273.15, mu0 = o.mu0;
  J.mu = [=](double, double T) { return mu0 * std::exp(EaR * (1.0 / (T + 273.15) - 1.0 / TrK)); };
  if (o.Ez_z.empty()) {
    const double fv = o.rho * o.g + I * o.E / J.Q;
    J.f = [fv](double, double) { return fv; };
  } else {  // the field below the melt from mew_axi, linear between its rows, flat beyond them
    const std::vector<double> zt = o.Ez_z, Et = o.Ez_E;
    const double rg = o.rho * o.g, IQ = I / J.Q;
    J.f = [zt, Et, rg, IQ](double z, double) {
      if (z <= zt.front()) return rg + IQ * Et.front();
      if (z >= zt.back()) return rg + IQ * Et.back();
      const std::size_t k = std::size_t(std::upper_bound(zt.begin(), zt.end(), z) - zt.begin());
      const double w = (z - zt[k - 1]) / (zt[k] - zt[k - 1]);
      return rg + IQ * ((1.0 - w) * Et[k - 1] + w * Et[k]);
    };
  }
  if (o.cool) {
    J.rho_cp = o.rho * o.cp;
    const double Nu = o.Nu, k = o.kair;
    J.h = [Nu, k](double a, double, double) { return Nu * k / (2.0 * a); };
  }
  return J;
}

struct Landing {
  bool ok = false;
  double U = 0, dL = 0, TL = 0, N0 = 0, z10 = -1, frac1mm = 0;
  Profile p;
};

Landing land(const Opts& o, double I) {
  const Jet J = make(o, I);
  const double u0 = J.Q / (kPi * o.a0 * o.a0);
  const Solution s = solve_free(J, u0, o.T0, o.n);
  Landing r;
  if (!s.ok) return r;
  r.ok = true;
  r.p = s.p;
  r.U = s.p.u.back();
  r.dL = 2.0 * std::sqrt(J.Q / (kPi * r.U));
  r.TL = s.p.T.back();
  r.N0 = s.N0;
  const double mu_start = J.mu(0.0, o.T0);
  for (std::size_t i = 0; i < s.p.z.size(); ++i)
    if (J.mu(s.p.z[i], s.p.T[i]) >= 10.0 * mu_start) { r.z10 = s.p.z[i]; break; }
  // the share of the jet's stretching, ln(U/u0), done within 1 mm of its start
  for (std::size_t i = 0; i < s.p.z.size(); ++i)
    if (s.p.z[i] >= 1e-3 - 1e-12) { r.frac1mm = std::log(s.p.u[i] / u0) / std::log(r.U / u0); break; }
  return r;
}

// The current at which the jet lands on its own at vc (bisection in log I); < 0 if none in 1e-4..1e3 nA.
double find_Istar(const Opts& o, double vc) {
  double lo = std::log(1e-13), hi = std::log(1e-6);
  const Landing rl = land(o, std::exp(lo)), rh = land(o, std::exp(hi));
  if (!(rl.ok && rh.ok && rl.U < vc && rh.U > vc)) return -1.0;
  for (int it = 0; it < 80; ++it) {
    const double mid = 0.5 * (lo + hi);
    const Landing r = land(o, std::exp(mid));
    (r.ok && r.U < vc ? lo : hi) = mid;
  }
  return std::exp(0.5 * (lo + hi));
}

void write_path(const std::string& name, const Path& p, double Q) {
  FILE* f = std::fopen(name.c_str(), "w");
  if (!f) { std::fprintf(stderr, "mew_jet: cannot write %s\n", name.c_str()); return; }
  std::fprintf(f, "x_mm,z_mm,d_um,u_mm_s,T_C\n");
  for (std::size_t i = 0; i < p.x.size(); ++i)
    std::fprintf(f, "%.6f,%.6f,%.5f,%.6f,%.4f\n", p.x[i] * 1e3, p.z[i] * 1e3, 2e6 * std::sqrt(Q / (kPi * p.u[i])),
                 p.u[i] * 1e3, p.T[i]);
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Opts o;
    for (int i = 1; i < argc; ++i) {
      const char* a = argv[i];
      auto num = [&]() {
        if (i + 1 >= argc) { std::fprintf(stderr, "mew_jet: %s needs a value\n", a); std::exit(2); }
        return std::atof(argv[++i]);
      };
      if (!std::strcmp(a, "-dfib")) o.dfib = num() * 1e-6;
      else if (!std::strcmp(a, "-vc")) o.vc = num() * 1e-3;
      else if (!std::strcmp(a, "-L")) { o.L = num() * 1e-3; o.L_set = true; }
      else if (!std::strcmp(a, "-Efile")) { if (i + 1 >= argc) { std::fprintf(stderr, "mew_jet: -Efile needs a file\n"); std::exit(2); } o.Efile = argv[++i]; }
      else if (!std::strcmp(a, "-a0")) o.a0 = num() * 1e-6;
      else if (!std::strcmp(a, "-T0")) o.T0 = num();
      else if (!std::strcmp(a, "-Tair")) o.Tair = num();
      else if (!std::strcmp(a, "-Nu")) o.Nu = num();
      else if (!std::strcmp(a, "-kair")) o.kair = num();
      else if (!std::strcmp(a, "-E")) o.E = num() * 1e6;  // kV/mm -> V/m
      else if (!std::strcmp(a, "-mu")) o.mu0 = num();
      else if (!std::strcmp(a, "-Ea")) o.Ea = num() * 1e3;
      else if (!std::strcmp(a, "-gamma")) o.gamma = num();
      else if (!std::strcmp(a, "-nocool")) o.cool = false;
      else if (!std::strcmp(a, "-n")) o.n = int(num());
      else if (!std::strcmp(a, "-I")) o.I = num() * 1e-9;
      else if (!std::strcmp(a, "-drag")) o.drag = true;
      else if (!std::strcmp(a, "-ndrag")) o.ndrag = int(num());
      else if (!std::strcmp(a, "-out")) { if (i + 1 >= argc) { std::fprintf(stderr, "mew_jet: -out needs a file\n"); std::exit(2); } o.out = argv[++i]; }
      else { std::fprintf(stderr, "mew_jet: unknown option %s\n", a); Kokkos::finalize(); return 2; }
    }
    if (!o.Efile.empty()) {
      FILE* f = std::fopen(o.Efile.c_str(), "r");
      if (!f) { std::fprintf(stderr, "mew_jet: cannot read %s\n", o.Efile.c_str()); Kokkos::finalize(); return 2; }
      char line[256];
      if (!std::fgets(line, sizeof line, f)) line[0] = 0;  // header
      double zm, e;
      while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "%lf,%lf", &zm, &e) == 2) { o.Ez_z.push_back(zm * 1e-3); o.Ez_E.push_back(e * 1e6); }
      std::fclose(f);
      if (o.Ez_z.size() < 2) { std::fprintf(stderr, "mew_jet: %s holds no field\n", o.Efile.c_str()); Kokkos::finalize(); return 2; }
      if (!o.L_set) o.L = o.Ez_z.back();  // the table ends at the collector
    }
    const double Q = Q_of(o), u0 = Q / (kPi * o.a0 * o.a0);
    const double lT = o.rho * o.cp * Q / (kPi * o.Nu * o.kair);
    std::printf("mew_jet: the MEW jet as a slender viscous thread (validation/Jet1D.hpp)\n");
    std::printf("  flow %.4f mm^3/s (a %.1f um fibre at %.0f mm/s; %.0f %% of 0.0208), from a %.0f um apex at %.2f mm/s,"
                " %.2f mm to the collector\n", Q * 1e9, o.dfib * 1e6, o.vc * 1e3, Q * 1e9 / 0.0208 * 100, o.a0 * 1e6,
                u0 * 1e3, o.L * 1e3);
    std::printf("  melt %.0f C at the apex; %.0f Pa s at %.0f C, Ea %.1f kJ/mol; gamma %.3f N/m\n", o.T0,
                o.mu0, o.Tref, o.Ea * 1e-3, o.gamma);
    if (o.Ez_z.empty()) std::printf("  field %.3f kV/mm, uniform\n", o.E * 1e-6);
    else {
      auto Eat = [&](double z) {
        const std::size_t k = std::min<std::size_t>(o.Ez_z.size() - 1, std::size_t(std::upper_bound(o.Ez_z.begin(), o.Ez_z.end(), z) - o.Ez_z.begin()));
        const double w = std::clamp((z - o.Ez_z[k - 1]) / (o.Ez_z[k] - o.Ez_z[k - 1]), 0.0, 1.0);
        return ((1.0 - w) * o.Ez_E[k - 1] + w * o.Ez_E[k]) * 1e-6;
      };
      std::printf("  field from %s: %.2f kV/mm at the apex, %.2f at 0.5 mm, %.2f at 1 mm, %.2f at the collector\n",
                  o.Efile.c_str(), o.Ez_E.front() * 1e-6, Eat(0.5e-3), Eat(1e-3), o.Ez_E.back() * 1e-6);
    }
    if (o.cool)
      std::printf("  air %.0f C, Nu %.2f, k_air %.3f W/m/K: the jet cools over %.3f mm (e-fold), viscosity x %.1f of %.0f C at the air\n",
                  o.Tair, o.Nu, o.kair, lT * 1e3,
                  std::exp(o.Ea / 8.314462618 * (1.0 / (o.Tair + 273.15) - 1.0 / (o.Tref + 273.15))), o.Tref);
    else
      std::printf("  isothermal at %.0f C (-nocool)\n", o.T0);

    if (o.drag) {
      // (a) the lag at the collector's speed for currents below the one that lands the jet there on
      // its own; (b) the lag against the collector's speed at one current (-I, else half of that one)
      const double Is = find_Istar(o, o.vc);
      if (Is < 0) { std::printf("  no current lands the jet at %.0f mm/s on its own\n", o.vc * 1e3); Kokkos::finalize(); return 1; }
      std::printf("\n  the jet lands on its own at the collector's %.0f mm/s with I* = %.4f nA\n", o.vc * 1e3, Is * 1e9);
      std::printf("\n  (a) at the collector's %.0f mm/s: the jet behind it, against the current\n", o.vc * 1e3);
      std::printf("     I/I*   I [nA]  free landing [mm/s]  lag [mm]  angle at the start [deg]  drag H [uN]  V0 [uN]\n");
      FILE* fa = nullptr;
      if (!o.out.empty()) {
        fa = std::fopen((o.out + "_lag_I.csv").c_str(), "w");
        if (fa) std::fprintf(fa, "I_over_Istar,I_nA,U_free_mm_s,lag_mm,theta0_deg,H_uN,V0_uN\n");
      }
      for (double fr : {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97}) {
        const Jet J = make(o, fr * Is);
        const Dragged d = solve_dragged(J, u0, o.T0, o.vc, o.ndrag);
        if (!d.ok) { std::printf("     %4.2f  (no dragged jet)\n", fr); continue; }
        std::printf("     %4.2f  %7.4f  %19.3f  %8.4f  %24.3f  %11.4f  %7.3f\n", fr, fr * Is * 1e9, d.U_free * 1e3,
                    d.lag * 1e3, d.theta0 * 180.0 / kPi, d.H * 1e6, d.V0 * 1e6);
        if (fa) std::fprintf(fa, "%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", fr, fr * Is * 1e9, d.U_free * 1e3, d.lag * 1e3,
                             d.theta0 * 180.0 / kPi, d.H * 1e6, d.V0 * 1e6);
      }
      if (fa) std::fclose(fa);
      const double I = o.I > 0 ? o.I : 0.5 * Is;
      const Jet J = make(o, I);
      const double U = solve_dragged(J, u0, o.T0, 1e9, o.ndrag).U_free;  // any vc above it returns U_free
      std::printf("\n  (b) at I = %.4f nA, which lands on its own at %.3f mm/s: the lag against the collector's speed\n",
                  I * 1e9, U * 1e3);
      std::printf("     v_c/U   v_c [mm/s]  lag [mm]  angle at the start [deg]  drag H [uN]  fibre [um]\n");
      FILE* fb = nullptr;
      if (!o.out.empty()) {
        fb = std::fopen((o.out + "_lag_vc.csv").c_str(), "w");
        if (fb) std::fprintf(fb, "vc_over_U,vc_mm_s,lag_mm,theta0_deg,H_uN,fibre_um\n");
      }
      int k = 0;
      for (double m : {1.02, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0}) {
        const double vc = m * U;
        const Dragged d = solve_dragged(J, u0, o.T0, vc, o.ndrag);
        if (!d.ok) { std::printf("     %5.2f  (no dragged jet)\n", m); continue; }
        const double fib = 2e6 * std::sqrt(J.Q / (kPi * vc));
        std::printf("     %5.2f  %10.3f  %8.4f  %24.3f  %11.4f  %10.2f\n", m, vc * 1e3, d.lag * 1e3,
                    d.theta0 * 180.0 / kPi, d.H * 1e6, fib);
        if (fb) std::fprintf(fb, "%.4f,%.6g,%.6g,%.6g,%.6g,%.6g\n", m, vc * 1e3, d.lag * 1e3, d.theta0 * 180.0 / kPi,
                             d.H * 1e6, fib);
        if (!o.out.empty()) write_path(o.out + "_path_" + std::to_string(k) + ".csv", d.p, J.Q);
        ++k;
      }
      if (fb) std::fclose(fb);
      if (!o.out.empty()) std::printf("  wrote %s_lag_I.csv, %s_lag_vc.csv and %s_path_*.csv\n", o.out.c_str(),
                                      o.out.c_str(), o.out.c_str());
    } else if (o.I < 0) {
      std::printf("\n   I [nA]  landing speed [mm/s]  diameter there [um]  T there [C]  viscosity x10 at [mm]"
                  "  stretching in 1st mm  N0 [uN]\n");
      for (double I : {0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0}) {
        const Landing r = land(o, I * 1e-9);
        if (!r.ok) { std::printf("  %7.2f  (no solution)\n", I); continue; }
        char z10[32];
        if (r.z10 < 0) std::snprintf(z10, sizeof z10, "never");
        else std::snprintf(z10, sizeof z10, "%.3f", r.z10 * 1e3);
        std::printf("  %7.2f  %20.3f  %19.2f  %11.1f  %21s  %20.2f  %7.3f\n", I, r.U * 1e3, r.dL * 1e6, r.TL, z10,
                    r.frac1mm, r.N0 * 1e6);
      }
      if (!o.out.empty()) {  // the sweep on a finer grid of currents, for tools/plot_mew_jet.py
        FILE* f = std::fopen(o.out.c_str(), "w");
        if (!f) { std::fprintf(stderr, "mew_jet: cannot write %s\n", o.out.c_str()); rc = 1; }
        else {
          std::fprintf(f, "I_nA,U_mm_s,d_um,T_C\n");
          for (int k = 0; k <= 40; ++k) {
            const double I = std::pow(10.0, -2.0 + 4.0 * k / 40.0);
            const Landing r = land(o, I * 1e-9);
            if (r.ok) std::fprintf(f, "%.6g,%.6g,%.6g,%.4f\n", I, r.U * 1e3, r.dL * 1e6, r.TL);
          }
          std::fclose(f);
          std::printf("  wrote %s\n", o.out.c_str());
        }
      }
      // the current at which the jet lands at the collector's speed: bisection in log I
      double lo = std::log(1e-13), hi = std::log(1e-6);
      const Landing rl = land(o, std::exp(lo)), rh = land(o, std::exp(hi));
      if (rl.ok && rh.ok && rl.U < o.vc && rh.U > o.vc) {
        for (int it = 0; it < 80; ++it) {
          const double mid = 0.5 * (lo + hi);
          const Landing r = land(o, std::exp(mid));
          (r.ok && r.U < o.vc ? lo : hi) = mid;
        }
        const Landing r = land(o, std::exp(0.5 * (lo + hi)));
        std::printf("\n  lands at the collector's %.0f mm/s with I = %.4f nA: straight lines need less, more makes them"
                    " wave (if the free-landing speed is the critical one)\n", o.vc * 1e3, std::exp(0.5 * (lo + hi)) * 1e9);
        char z10[48];
        if (r.z10 < 0) std::snprintf(z10, sizeof z10, "never 10x its apex value");
        else std::snprintf(z10, sizeof z10, "10x its apex value at %.3f mm", r.z10 * 1e3);
        std::printf("  there: viscosity %s, %.0f %% of the stretching in the first mm, landing at %.1f C\n", z10,
                    r.frac1mm * 100, r.TL);
      } else {
        std::printf("\n  no current between 1e-4 and 1e3 nA lands the jet at %.0f mm/s\n", o.vc * 1e3);
      }
    } else {
      const Landing r = land(o, o.I);
      if (!r.ok) { std::printf("  no solution\n"); rc = 1; }
      else {
        std::printf("\n  I = %.3f nA: lands at %.3f mm/s, %.2f um across, %.1f C\n", o.I * 1e9, r.U * 1e3, r.dL * 1e6, r.TL);
        if (!o.out.empty()) {
          FILE* f = std::fopen(o.out.c_str(), "w");
          if (!f) { std::fprintf(stderr, "mew_jet: cannot write %s\n", o.out.c_str()); rc = 1; }
          else {
            const Jet J = make(o, o.I);
            std::fprintf(f, "z_mm,d_um,u_mm_s,T_C,mu_Pa_s,N_uN\n");
            for (std::size_t i = 0; i < r.p.z.size(); i += std::max<std::size_t>(1, r.p.z.size() / 2000))
              std::fprintf(f, "%.6f,%.5f,%.6f,%.4f,%.6g,%.6f\n", r.p.z[i] * 1e3, 2e6 * std::sqrt(Q / (kPi * r.p.u[i])),
                           r.p.u[i] * 1e3, r.p.T[i], J.mu(r.p.z[i], r.p.T[i]), r.p.N[i] * 1e6);
            std::fclose(f);
            std::printf("  wrote %s\n", o.out.c_str());
          }
        }
      }
    }
  }
  Kokkos::finalize();
  return rc;
}
