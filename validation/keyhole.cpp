//==============================================================================
//  The reduced-order keyhole drilling model, ported from the Python at
//  ~/Desktop/02_Code_Projects/Laser/keyhole_validation, as a second
//  implementation.
//
//  Ti-6Al-4V, stationary beam, Cunningham et al., Science 363, 849-852 (2019).
//  A thermal-only lattice Boltzmann field with a per-column receding surface:
//  the laser heats the top cell of each column, evaporative recoil ejects melt,
//  the column recedes, and the reported "keyhole depth" is the largest
//  accumulated recession. There is no flow, no surface tension, no free-surface
//  hydrodynamics and no optics beyond a fitted absorptivity curve.
//
//  WHAT THIS CASE IS FOR, AND WHAT IT IS NOT FOR.
//
//  It exists to do two things the Python could not:
//    (1) be a SECOND IMPLEMENTATION, so the two can be diffed. This tree's
//        doctrine is that two independent codes agreeing where they overlap is
//        worth something and disagreement is a bug in one of them.
//    (2) run the GRID LADDER. The Python's own README records 15-60 % change at
//        a single 1.5x refinement and a third rung abandoned after hours of
//        swap thrashing on an 8 GB machine. Here a rung costs 10 s to 5 min,
//        because D3Q7 is 7 populations against D3Q19's 19, Esoteric Pull is
//        in place, and there are no per-step temporaries -- which was the
//        Python's specifically diagnosed problem.
//
//  IT DOES NOT VALIDATE THE PHYSICS, and no output of this case may be quoted
//  as if it did. The Python's agreement with Cunningham et al. is a fit with
//  THIRTEEN calibrated numbers against seven curves scored on two features
//  each: eta_min, eta_max, absorb_ramp_depth, two d_ignite(P) power laws, two
//  absorb_decay_depth(P) power laws, two per-case max_recession_per_step
//  overrides, Vm_max, T_cap and the alpha clips. Agreement between this port
//  and that Python proves the port. Both run the same arithmetic.
//
//  THE CLAMPS ARE NOT SAFETY VALVES, THEY ARE THE INTEGRATOR. Measured on the
//  Python itself (instrumented census, 95 um / 104 W, production resolution,
//  2026-09-21), fraction of IGNITED steps on which each bound at one node or
//  more:
//
//      T0 floor (300 K)             67.8 %   worst unclamped  -31334 K
//      max_recession_per_step       52.8 %   worst unclamped  2.715 cells (cap 0.5)
//      T_cap upper (3515 K)         20.0 %   worst unclamped   4345 K
//      Vm_max (15 m/s)               0.0 %   worst wanted      9.023 m/s
//      alpha and tau clips           0.0 %
//
//  Of the columns actually receding, 47.5 % are truncated by the recession cap.
//  The surface energy balance is forward Euler on a stiff evaporative feedback
//  and wants to reach minus thirty-one thousand kelvin; the two-sided clamp is
//  the only reason it looks stable. So three clamps decide the answer on a
//  fifth to two thirds of steps, and THIS CASE PRINTS ITS OWN CENSUS EVERY RUN
//  rather than leaving that to be rediscovered.
//
//  THE CAP DOES NOT SCALE UNDER REFINEMENT, AND THAT IS PROBABLY THE LADDER'S
//  REAL CONTENT. max_recession_per_step is in cells per step, so the physical
//  ceiling it imposes is cap*dx/dt. At fixed alpha_lattice, dt goes as dx^2, so
//  dx/dt goes as 1/dx: halving dx DOUBLES the physical speed limit. Two rungs
//  then run two different models and the ladder measures that rather than the
//  discretisation. (run_104W_2ms_fine.py:36-40 asserts the cap "correctly
//  scales itself under refinement", which is true in cells/step and false in
//  m/s.) This port therefore stores the cap as a VELOCITY and converts per
//  rung; `-cap cells` reproduces the Python's behaviour for comparison.
//
//  THE GAUSSIAN DELIVERS HALF THE NOMINAL POWER, ON PURPOSE. solver.py:172 is
//  q = (P/(pi rb^2)) exp(-2 r^2/rb^2), whose integral over the plane is P/2,
//  not P; the beam-normalised peak is 2P/(pi rb^2). Every fitted constant
//  absorbed that factor, so a port that "fixes" it doubles the delivered power
//  and invalidates the calibration. The bug is reproduced and the delivered
//  power is printed at startup so the choice is visible rather than hidden.
//
//  TWO DELIBERATE DEVIATIONS FROM THE REFERENCE, both conservative where the
//  Python is lossy, and NEITHER ISOLATED BY A CONTROL:
//    - the Python re-equilibrates all 19 populations at the surface node every
//      step (solver.py:332), discarding whatever non-equilibrium flux streaming
//      just delivered. This port adds the surface increment through
//      ScalarSolver::add_source and keeps the non-equilibrium part.
//    - a vacated cell is marked ScalarExcluded here; the Python resets its
//      populations to W*T0.
//  Both are named here because an unexplained gap against the Python is more
//  likely to come from these than from anything else.
//
//  THE LATTICE IS NOT THE PYTHON'S. D3Q19 was removed from this tree on
//  2026-09-18. -lat d3q27 shares its cs2 = 1/3 and is the closer twin;
//  -lat d3q7 has cs2 = 1/4 and is the cheaper one. The gap between them bounds
//  what the lattice change costs; it does not bracket D3Q19.
//
//  MEASURED 2026-09-21. THE CROSS-CHECK, 95 um / 104 W, production grid,
//  against the regenerated Python trace (the original .npz no longer exist):
//
//      lattice   final depth        first ignition      cap binding
//      D3Q27     134.089 um         0.5221 ms           48.5 %
//      D3Q7      134.219 um         0.6725 ms           48.5 %
//      Python    134.473 um         0.5145 ms           47.5 %
//
//  So the endpoint agrees to 0.29 % and the clamp statistics to a point. THE
//  TRAJECTORY DOES NOT: over the whole trace the D3Q27 port is rms 8.4 um from
//  the Python with a worst gap of -18.2 um at t = 0.79 ms. Five checkpoints
//  would have called this agreement; 53 samples show two codes reaching the
//  same endpoint by visibly different routes. The two deviations named above
//  are the leading suspects and neither has been isolated.
//
//  THE LATTICE COSTS 30 % ON THE IGNITION TIME. D3Q7 (cs2 = 1/4) ignites at
//  0.6725 ms against D3Q27's 0.5221 ms, with the PHYSICAL DIFFUSIVITY MATCHED
//  on both through omega_from_diffusivity. What differs is the truncation, and
//  this model's central observable is a threshold on a near-surface quantity,
//  which amplifies it. Use D3Q27 for the cross-check and D3Q7 for the ladder
//  (memory), and quote which one a figure used.
//
//  THE LADDER, AND WHAT IT SETTLES. Three rungs at a fixed 300x300x1500 um
//  domain, D3Q7, 2 ms, run in three cap modes. Depth at t = 2 ms:
//
//      dx (um)   fixed velocity   cells/step (Python)   uncapped (10 m/s)
//      5.000     129.96 um        129.96 um             136.40 um
//      3.333     116.07           119.95                120.04
//      2.500     109.31           112.25                112.16
//
//      ignition  0.563 -> 0.865   0.563 -> 0.826        0.533 -> 0.826 ms
//      cap binds 48.0 -> 28.9 %   48.0 -> 6.7 %         0 % throughout
//
//  Three findings, and the second and third contradict the obvious guess.
//
//  (1) THE MODEL IS NOT GRID-CONVERGED IN ANY MODE. Depth falls 12-16 % over a
//      2x refinement, monotone, not settling. The Python's README reports
//      15-60 % at a single 1.5x refinement and records that a third rung was
//      abandoned; that rung is 4.8 minutes here and confirms the trend.
//  (2) THE PYTHON'S RUNGS WERE NOT RUNNING THE SAME MODEL. A cells/step cap has
//      a physical ceiling cap*dx/dt which rises as 1/dx, so refining silently
//      raised it 1.66 -> 2.49 -> 3.32 m/s and the binding fraction collapsed
//      48.0 -> 6.7 %. At the finest rung that mode sits 0.08 % from the FULLY
//      UNCAPPED model: their refinement was removing their own clamp.
//  (3) BUT THE CLAMP IS NOT THE CAUSE OF THE GRID DEPENDENCE. Lifting it
//      entirely (0 of 32843 column-steps capped) still gives 136.4 -> 112.2 um.
//      Holding it at a fixed physical velocity still gives -15.9 %. The
//      sensitivity is genuine under-resolution; the clamp only confounded its
//      MEASUREMENT.
//
//  So d_ignite is not a resolution-independent physical constant: the ignition
//  time moves +54 % over a 2x refinement in every cap mode. That answers the
//  question the Python's README leaves open, and it is the one result here that
//  is about the MODEL rather than about the port. NOTHING IN THE LADDER IS
//  CONVERGED, so no rung of it may be quoted as a depth prediction.
//
//  -scheme cp REPRODUCES THE PYTHON'S APPARENT HEAT CAPACITY; -scheme enthalpy
//  USES THE TRANSPORTED ENTHALPY (collision/EnthalpyBGK.hpp). They are a real
//  switch, not a flag, because the apparent-cp form divides the surface
//  increment by the APPARENT capacity and the enthalpy form by the constant
//  one. Measured on the 1-D Neumann problem at this case's own lattice numbers,
//  the apparent-cp scheme's front error plateaus near +4.4 % and a 4x
//  refinement removes none of it, where the enthalpy scheme is at -0.03 %:
//  sum_i g_i = T conserves the integral of T, not of rho c_app(T) T. The cross
//  check against the Python must therefore run -scheme cp; -scheme enthalpy is
//  the better model and a DIFFERENT one.
//==============================================================================
#include "collision/EnthalpyBGK.hpp"
#include "collision/ScalarBGK.hpp"
#include "memory/EsotericPull.hpp"
#include "solver/ScalarSolver.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

namespace {

//------------------------------------------------------------------------------
// Material and process constants. Ti-6Al-4V, from the Python's case_runner.py.
// The Python's banner records that these were NOT traced to a single primary
// source; they are carried here unchanged so the two codes solve the same
// problem, and that provenance gap is the Python's to close, not this port's.
//------------------------------------------------------------------------------
struct Mat {
  double rho0    = 4430.0;      // kg/m^3, solid -- the bulk capacity density
  double cp      = 546.0;       // J/(kg K)
  double k       = 6.7;         // W/(m K)
  double Tm      = 1903.0;      // K
  double band    = 25.0;        // K, half-width: solidus 1878, liquidus 1928
  double Tb      = 3315.0;      // K
  double L_fus   = 2.86e5;      // J/kg
  double Lv      = 8.86e6;      // J/kg
  double Rs      = 8.314 / 0.047867;   // J/(kg K), R / M_Ti
  double rho_lq  = 4130.0;      // kg/m^3 at Tm
  double rho_slp = 0.4;         // kg/(m^3 K)
  double T0      = 300.0;       // K
  double P0      = 101325.0;    // Pa
  double h_conv  = 20.0;        // W/(m^2 K)
  double emis    = 0.4;
  double sigma   = 5.670374419e-8;
  double T_cap   = 3515.0;      // = Tb + 200, see the Python's README
  double Vm_max  = 15.0;        // m/s -- measured never to bind
  double eta_max = 1.0;
  double ramp    = 6.0e-6;      // m, absorb_ramp_depth
  double eff     = 1.0;         // removal_efficiency: absent from case_runner,
                                // so the solver's dict.get default applies
  // The ancestor model (Muhammad, Rogers & Li 2013) needs four things the
  // Cunningham descendant does not, and LACKS three clamps it has.
  double Peff      = 0.0;       // Pa, assist gas. P_i = 0 in the Cunningham
                                // case so it is identically zero there.
  double thickness = 0.0;       // m; > 0 saturates the reported depth
  double pulse     = 0.0;       // s; > 0 turns the beam off after this
  bool   clampT    = true;      // the two-sided T clamp. The ancestor has NO
                                // temperature clamp at all -- it was added for
                                // Ti, whose Lv/R is 3x steel's.
  double Ts() const { return Tm - band; }
  double Tl() const { return Tm + band; }
};

// Assist gas, eqs (6)-(9) of the 2013 paper: a choked nozzle's stagnation
// pressure scaled by the area the beam actually sees.
double assist_pressure(double gamma, double Pi, double rb, double dn, double zn) {
  const double Pc = std::pow(2.0 / (gamma + 1.0), gamma / (gamma - 1.0)) * Pi;
  const double Ae = M_PI * rb * rb, Ar = dn * M_PI * zn;
  return Pc * Ae / (Ae + Ar);
}

// SS316L, 150 um foil, single 100 W pulse, N2 assist gas. Muhammad, Rogers &
// Li, J. Phys. D 46, 095101 (2013), section 5.1 and tables 1 and 3.
//
// THIS IS THE ANCESTOR OF THE CUNNINGHAM MODEL AND IT IS SIMPLER. Absorptivity
// is the bare opening ramp on TOTAL depth: no ignition latch, no escape decay.
// And it carries NO temperature clamp, NO Vm_max and NO recession cap -- those
// three were added downstream because Ti-6Al-4V's Lv/R is about three times
// steel's, so the same overshoot that is harmless here diverges there.
Mat muhammad_material() {
  Mat m;
  m.rho0 = 7950.0; m.cp = 470.0; m.k = 20.0;
  m.Tm = 1723.0;  m.band = 25.0;
  m.Tb = 3100.0;
  m.L_fus = 2.5e5;              // NOT from the paper: numerical smoothing only
  m.Lv = 2.6e6;                 // table 1, used in Clausius-Clapeyron
  m.Rs = 8.314 / 0.055845;      // Fe
  m.rho_lq = 6881.0; m.rho_slp = 0.77;   // eq (10), SS316L-specific
  m.eff = 0.80;                 // calibrated, NOT from the paper
  m.ramp = 6.0e-6;
  m.Peff = assist_pressure(1.4, 6.0e5, 25.0e-6, 0.5e-3, 1.0e-3);
  m.thickness = 150.0e-6;
  m.pulse = 0.15e-3;
  m.clampT = false;             // the ancestor clamps nothing
  m.T_cap = 0.0; m.Vm_max = 0.0;
  return m;
}

struct Opts {
  double P       = 104.0;       // W
  double spot    = 95.0;        // um, 1/e^2 DIAMETER
  double dx      = 5.0e-6;      // m
  double t_max   = 2.0e-3;      // s
  double eta_min = 0.12;
  double d_ign   = -1.0;        // m; < 0 means "from the power law"
  double decay   = -1.0;        // m; < 0 means "from the power law"
  double a_lat   = 0.1667;      // alpha_lattice_target, sets dt
  double v_cap   = -1.0;        // m/s; < 0 means "0.5 cells/step at dx = 5 um"
  bool   cap_cells = false;     // true: cap fixed in cells/step, the Python's way
  double cap_val   = 0.5;       // the cells/step value when cap_cells is set
  double cap_free  = -1.0;      // m/s override for the uncapped counterfactual
  Index  nx = 0, ny = 0, nz = 0;   // 0 means "the Python's grid for this spot"
  bool   enthalpy = false;
  bool   d3q27    = false;
  int    every    = 0;          // trace interval in steps, 0 = off
  std::string ref;              // reference CSV to diff against
  std::string out;              // depth trace to write
  bool   muhammad = false;
};

// The shipped seven-case power laws (run_final_calibrated_v2.py). Both are
// fitted per spot size to each case's OWN observed transition time and final
// depth, so they are inputs and not predictions.
double d_ignite_law(double P, double spot_um) {
  return (spot_um < 120.0 ? 7.7858e4 * std::pow(P, -1.7475)
                          : 1.8752e7 * std::pow(P, -2.6438)) * 1e-6;
}
double decay_law(double P, double spot_um) {
  return (spot_um < 120.0 ? 2.5992e-4 * std::pow(P, 2.6397)
                          : 1.1536e-4 * std::pow(P, 2.7404)) * 1e-6;
}

//------------------------------------------------------------------------------
// A tiny CSV reader for the regenerated Python trace. Deliberately stdlib only:
// the reference is 41 KB of two columns and pulling a dependency in to read it
// would be worse than the twenty lines.
//------------------------------------------------------------------------------
struct Trace { std::vector<double> t, d; };

Trace read_trace(const std::string& path) {
  Trace tr;
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return tr;
  char line[512];
  while (std::fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    double a, b;
    if (std::sscanf(line, "%lf,%lf", &a, &b) == 2) { tr.t.push_back(a); tr.d.push_back(b); }
  }
  std::fclose(f);
  return tr;
}

double trace_at(const Trace& tr, double t) {
  if (tr.t.empty()) return 0.0;
  if (t <= tr.t.front()) return tr.d.front();
  if (t >= tr.t.back())  return tr.d.back();
  std::size_t i = 1;
  while (i < tr.t.size() && tr.t[i] < t) ++i;
  const double w = (t - tr.t[i - 1]) / (tr.t[i] - tr.t[i - 1]);
  return tr.d[i - 1] + w * (tr.d[i] - tr.d[i - 1]);
}

//==============================================================================
template <class L, class Coll, bool Enth>
int run(const Opts& o) {
  const Mat m = o.muhammad ? muhammad_material() : Mat{};
  const double dx    = o.dx;
  const double alpha = m.k / (m.rho0 * m.cp);
  const double dt    = o.a_lat * dx * dx / alpha;
  const long   steps = long(std::ceil(o.t_max / dt)) + 2;
  // The 2013 paper quotes a beam RADIUS (25 um); Cunningham quotes a spot
  // DIAMETER. -spot is the diameter in both, so the ancestor takes 50.
  const double rb    = o.spot * 1e-6 * 0.5;
  const double d_ign = o.d_ign >= 0 ? o.d_ign : d_ignite_law(o.P, o.spot);
  const double decay = o.decay >= 0 ? o.decay : decay_law(o.P, o.spot);

  // The cap. Stored as a velocity so that every rung of the ladder runs the
  // same model; -cap cells reproduces the Python, whose ceiling moves with dx.
  const double v_ref = 0.5 * 5.0e-6 / (o.a_lat * 25.0e-12 / alpha);  // 0.5 cells/step at dx = 5 um
  double v_cap = o.v_cap > 0 ? o.v_cap : v_ref;
  if (o.cap_free > 0) v_cap = o.cap_free;
  // <= 0 means no cap at all, which is what the ancestor has.
  const double cap_cells = o.muhammad ? 1e30
                         : (o.cap_cells ? o.cap_val : v_cap * dt / dx);

  const Index nx = o.nx ? o.nx : (o.spot < 120.0 ? 61 : 91);
  const Index ny = o.ny ? o.ny : nx;
  const Index nz = o.nz ? o.nz : 300;
  const Index ncol = nx * ny;

  Domain d(nx, ny, nz, false, false, false);
  Coll coll;
  const double a_lat_solid = alpha * dt / (dx * dx);

  // The enthalpy gauge: kelvin-equivalent, everything divided by rho0*cp, and
  // the datum chosen so that AMBIENT is H = 0. field_kernel writes 0 at an
  // excluded node, and in the naive gauge H = 0 is exactly the solidus -- a
  // void cell would read 1878 K. Depth is taken from `rec` regardless.
  const double La_k = m.L_fus / m.cp;
  PhaseChange pc;
  if constexpr (Enth) {
    pc.T_s = Real(m.Ts()); pc.T_l = Real(m.Tl());
    pc.cp_s = Real(1); pc.cp_l = Real(1);
    pc.k_s = Real(a_lat_solid); pc.k_l = Real(a_lat_solid);
    pc.La = Real(La_k); pc.E_datum = Real(m.Ts() - m.T0);
    coll.set_material(pc);
    coll.T_ref = Real(0);
  } else {
    coll.omega = Coll::omega_from_diffusivity(Real(a_lat_solid));
    coll.T_ref = Real(m.T0);
  }

  ScalarSolver<L, EsotericPull<L>, Coll> s(d, coll);
  s.set_geometry([&](Index, Index, Index) -> std::uint8_t { return ScalarBulk; });
  s.finalize_geometry();

  // The per-node relaxation field: only the apparent-cp path needs it, because
  // there alpha depends on the local temperature through c_app.
  View1D<Real> omega_of;
  if constexpr (!Enth) {
    omega_of = View1D<Real>("omega_of", d.n_padded);
    Kokkos::deep_copy(omega_of, Real(coll.omega));
    coll.omega_of = omega_of;
    s = ScalarSolver<L, EsotericPull<L>, Coll>(d, coll);
    s.set_geometry([&](Index, Index, Index) -> std::uint8_t { return ScalarBulk; });
    s.finalize_geometry();
  }

  const Real H_amb = Enth ? Real(0) : Real(m.T0);
  s.initialize(H_amb);

  // Column state.
  View1D<Index> Hs("Hs", ncol);
  View1D<Real>  rec("rec", ncol), dign("dign", ncol);
  View1D<char>  ign("ign", ncol);
  View1D<Real>  dTraw("dTraw", ncol), drec("drec", ncol), dHeff("dHeff", ncol);
  Kokkos::deep_copy(Hs, Index(nz - 1));
  // The ancestor has no ignition latch: eta is the opening ramp evaluated on
  // TOTAL depth from step one. Seeding the latch closed with dign = 0 makes the
  // same code path reproduce it exactly, rather than adding a second branch.
  if (o.muhammad) Kokkos::deep_copy(ign, char(1));

  // Clamp census counters, in the same shape the Python was instrumented with.
  View1D<long> cen("census", 8);   // 0 Vm_max, 1 Tcap_hi, 2 T0_lo, 3 rec_cap,
                                   // 4 columns receding, 5 columns capped,
                                   // 6 ignited steps, 7 alpha clip
  auto h_cen = Kokkos::create_mirror_view(cen);

  const double xc = 0.5 * nx * dx, yc = 0.5 * ny * dx;
  const double q_peak = o.P / (M_PI * rb * rb);      // HALF-POWER form, on purpose
  const double flux_to_K = dt / (m.rho0 * m.cp * dx);

  std::printf("\nkeyhole: %s  %s  P = %.0f W  spot = %.0f um\n",
              Enth ? "enthalpy" : "apparent-cp", L::name, o.P, o.spot);
  std::printf("  grid %dx%dx%d   dx = %.4g m   dt = %.6e s   steps = %ld\n",
              int(nx), int(ny), int(nz), dx, dt, steps);
  std::printf("  alpha = %.6e m^2/s   alpha_lat = %.6f   cs2 = %.4f   tau = %.6f\n",
              alpha, a_lat_solid, double(cs2<L, Real>()),
              1.0 / double(Coll::omega_from_diffusivity(Real(a_lat_solid))));
  std::printf("  eta_min = %.4f  d_ignite = %.4f um  decay = %.4f um\n",
              o.eta_min, d_ign * 1e6, decay * 1e6);
  std::printf("  beam: peak %.6e W/m^2, DELIVERED %.3f W = P/2 "
              "(the Python's missing factor of 2, reproduced)\n",
              q_peak, q_peak * M_PI * rb * rb / 2.0);
  if (cap_cells > 1e20)
    std::printf("  recession cap: NONE (the ancestor model carries no cap, "
                "no Vm_max and no temperature clamp)\n");
  else
    std::printf("  recession cap: %.6f cells/step = %.6f m/s   (%s)\n",
                cap_cells, cap_cells * dx / dt,
                o.cap_cells ? "fixed cells/step, the Python's way"
                            : "fixed velocity, so every rung runs one model");

  const Mat mm = m;
  const double etamin = o.eta_min, ramp = m.ramp, Tcap = m.T_cap;
  const bool cap_cells_mode = o.cap_cells;
  const double capc = cap_cells;
  const bool clampT = m.clampT;
  Trace ref = o.ref.empty() ? Trace{} : read_trace(o.ref);
  if (!o.ref.empty())
    std::printf("  reference: %s (%zu points)\n", o.ref.c_str(), ref.t.size());

  std::vector<double> hist_t, hist_d;

  for (long it = 0; it < steps; ++it) {
    // The ancestor fires a single finite pulse; the descendant is continuous.
    const bool beam_on = (m.pulse <= 0.0) || (double(it) * dt <= m.pulse);
    auto field = s.temperature();
    auto flags = s.flags();

    // (a)-(c) surface sample and surface physics, per column, from the
    // PRE-collision field -- which is what field() holds at the top of a step.
    Kokkos::parallel_for("surface", Range(0, ncol), KOKKOS_LAMBDA(Index c) {
      const Index x = c / ny, y = c % ny;
      const Index z = Hs(c);
      const Index ns = d.id(x, y, z);
      const Real Hn = field(ns);
      Real Ttop, fltop, cptop;
      if constexpr (Enth) {
        Real fl, T, E, dEdT; pc.invert(Hn, fl, T, E, dEdT);
        Ttop = T; fltop = fl; cptop = Real(mm.cp);
      } else {
        Ttop = Hn;
        fltop = Kokkos::min(Kokkos::max((Ttop - Real(mm.Ts())) /
                                        Real(2 * mm.band), Real(0)), Real(1));
        const Real th = (Ttop - Real(mm.Ts())) / Real(2 * mm.band);
        cptop = (Ttop > Real(mm.Ts()) && Ttop < Real(mm.Tl()))
                  ? Real(mm.cp + mm.L_fus * (M_PI / (2 * 2 * mm.band)) *
                                     Kokkos::sin(M_PI * th))
                  : Real(mm.cp);
      }

      // Absorptivity. depth is read BEFORE this step's increment (solver.py:276
      // against :345); the latch below reads it AFTER. The one-step lag between
      // them is deliberate and is reproduced.
      const double depth  = double(rec(c)) * dx;
      const double dsince = Kokkos::max(depth - double(dign(c)), 0.0);
      double eta = etamin;
      if (ign(c)) {
        const double fr = Kokkos::min(dsince / ramp, 1.0);
        eta = etamin + (mm.eta_max - etamin) * fr *
                       (decay > 0 ? Kokkos::exp(-dsince / decay) : 1.0);
      }
      const double xm = (double(x) + 0.5) * dx - xc;
      const double ym = (double(y) + 0.5) * dx - yc;
      const double r2 = xm * xm + ym * ym;
      const double q_las = beam_on ? q_peak * Kokkos::exp(-2.0 * r2 / (rb * rb)) * eta
                                   : 0.0;
      const double Tt = double(Ttop);
      const double q_loss = mm.h_conv * (Tt - mm.T0) +
                            mm.emis * mm.sigma * (Tt * Tt * Tt * Tt -
                                                  mm.T0 * mm.T0 * mm.T0 * mm.T0);
      const double expo = Kokkos::min(Kokkos::max(
          (mm.Lv / mm.Rs) * (1.0 / mm.Tb - 1.0 / Kokkos::max(Tt, 1.0)), -60.0), 60.0);
      const double Pvap = mm.P0 * Kokkos::exp(expo);
      const double Prec = Kokkos::max(Pvap - mm.P0, 0.0);
      const double rho_m = Kokkos::max(mm.rho_lq - mm.rho_slp * (Tt - mm.Tm), 1000.0);
      // P_eff is the assist gas and is ADDITIVE inside the square root, so it
      // is a floor on the ejection velocity: the ancestor recedes wherever the
      // surface is molten, not only above boiling.
      double Vm = Kokkos::sqrt(Kokkos::max(2.0 * (Prec + mm.Peff) / rho_m, 0.0)) * mm.eff;
      if (mm.Vm_max > 0.0 && Vm > mm.Vm_max) {
        Vm = mm.Vm_max; Kokkos::atomic_add(&cen(0), 1L);
      }
      const double q_rem = rho_m * Vm * double(fltop) * mm.Lv;
      const double q_net = q_las - q_loss - q_rem;
      dTraw(c) = Real(q_net * dt / (mm.rho0 * double(cptop) * dx));

      double dr = Vm * double(fltop) * dt / dx;
      if (dr > capc) { dr = capc; Kokkos::atomic_add(&cen(3), 1L);
                       Kokkos::atomic_add(&cen(5), 1L); }
      if (dr > 0.0) Kokkos::atomic_add(&cen(4), 1L);
      drec(c) = Real(dr);
    });

    // (b) the per-node relaxation field, apparent-cp path only.
    if constexpr (!Enth) {
      auto om = omega_of;
      const double al = a_lat_solid;
      Kokkos::parallel_for("omega", Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
        const Real T = field(n);
        const Real th = (T - Real(mm.Ts())) / Real(2 * mm.band);
        const Real capp = (T > Real(mm.Ts()) && T < Real(mm.Tl()))
                            ? Real(mm.cp + mm.L_fus * (M_PI / (2 * 2 * mm.band)) *
                                               Kokkos::sin(M_PI * th))
                            : Real(mm.cp);
        Real a = Real(al) * Real(mm.cp) / capp;
        a = Kokkos::min(Kokkos::max(a, Real(1.0e-3)), Real(0.3));
        om(n) = Real(1) / (a * inv_cs2<L, Real>() + Real(0.5));
      });
    }

    s.step();
    s.compute_field();

    // (f) the clamp, folded into the source. The Python clamps AFTER adding
    // (solver.py:330), so the clamp cannot be applied to the increment alone.
    auto f2 = s.temperature();
    Kokkos::parallel_for("clamp", Range(0, ncol), KOKKOS_LAMBDA(Index c) {
      const Index x = c / ny, y = c % ny;
      const Index ns = d.id(x, y, Hs(c));
      const Real Hp = f2(ns);
      Real Tp;
      if constexpr (Enth) { Real fl, T, E, dE; pc.invert(Hp, fl, T, E, dE); Tp = T + dTraw(c); }
      else                { Tp = Hp + dTraw(c); }
      Real Tc = Tp;
      if (clampT) {
        if (Tc > Real(Tcap))  { Tc = Real(Tcap);  Kokkos::atomic_add(&cen(1), 1L); }
        if (Tc < Real(mm.T0)) { Tc = Real(mm.T0); Kokkos::atomic_add(&cen(2), 1L); }
      }
      Real Hnew;
      if constexpr (Enth) {
        const Real fl = Kokkos::min(Kokkos::max((Tc - Real(mm.Ts())) /
                                                Real(2 * mm.band), Real(0)), Real(1));
        Hnew = pc.enthalpy_of(Tc, fl);
      } else {
        Hnew = Tc;
      }
      dHeff(c) = Hnew - Hp;
    });

    {
      auto Hsv = Hs; auto dH = dHeff; const Index nyl = ny;
      s.add_source(KOKKOS_LAMBDA(Index n) -> Real {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) return Real(0);
        const Index x = px - d.hx, y = py - d.hy, z = pz - d.hz;
        return (z == Hsv(x * nyl + y)) ? dH(x * nyl + y) : Real(0);
      });
      Kokkos::parallel_for("writeback", Range(0, ncol), KOKKOS_LAMBDA(Index c) {
        const Index x = c / nyl, y = c % nyl;
        const Index ns = d.id(x, y, Hs(c));
        f2(ns) += dH(c);
      });
    }

    // (g) recession, the one-way latch, and the geometry.
    const double dign_m = d_ign;
    Kokkos::parallel_for("recede", Range(0, ncol), KOKKOS_LAMBDA(Index c) {
      rec(c) += drec(c);
      // H = round((nz-1) - rec): the cell vanishes at rec = 0.5, not 1.0, and
      // the integer surface trails the float depth by up to half a cell BY
      // CONSTRUCTION. numpy rounds half-to-even and llround is half-away-from-
      // zero; the difference is one ulp of a tie and is not worth emulating,
      // but it is why this line is commented.
      Index h = Index(Kokkos::round(double(nz - 1) - double(rec(c))));
      Hs(c) = Kokkos::min(Kokkos::max(h, Index(0)), Index(nz - 1));
      const double dn = double(rec(c)) * dx;          // AFTER the increment
      if (!ign(c) && dn >= dign_m) { ign(c) = 1; dign(c) = Real(dn); }
    });

    {
      auto Hsv = Hs; const Index nyl = ny;
      Kokkos::parallel_for("geom", Range(0, d.n_padded), KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; d.coords(n, px, py, pz);
        if (!d.is_interior(px, py, pz)) return;
        const Index x = px - d.hx, y = py - d.hy, z = pz - d.hz;
        const bool out = z > Hsv(x * nyl + y);
        flags(n) = out ? std::uint8_t(ScalarExcluded) : std::uint8_t(ScalarBulk);
        if (out) f2(n) = Real(0);
      });
    }

    { long ig = 0;
      Kokkos::parallel_reduce("ig", Range(0, ncol),
        KOKKOS_LAMBDA(Index c, long& a) { a += ign(c) ? 1 : 0; }, ig);
      if (ig > 0) { h_cen(6) += 1; } }

    if (o.every && (it + 1) % o.every == 0) {
      Real mx = 0;
      Kokkos::parallel_reduce("mx", Range(0, ncol),
        KOKKOS_LAMBDA(Index c, Real& a) { a = Kokkos::max(a, rec(c)); },
        Kokkos::Max<Real>(mx));
      hist_t.push_back(double(it + 1) * dt);
      hist_d.push_back(double(mx) * dx);
    }
  }

  Real mx = 0;
  Kokkos::parallel_reduce("final", Range(0, ncol),
    KOKKOS_LAMBDA(Index c, Real& a) { a = Kokkos::max(a, rec(c)); }, Kokkos::Max<Real>(mx));
  double depth = double(mx) * dx;
  // penetration_depth_m() returns min(recession*dx, thickness): once the hole
  // breaks through, `recession` keeps accumulating but nothing physical is
  // happening, so the REPORTED depth saturates. The raw value is printed too,
  // because the gap between them says how long the run has been meaningless.
  const double depth_raw = depth;
  if (m.thickness > 0.0) depth = std::min(depth, m.thickness);

  Kokkos::deep_copy(h_cen, cen);
  const double ign_steps = double(h_cen(6) > 0 ? h_cen(6) : 1);
  std::printf("\n  final depth at t = %.4f ms : %.3f um", steps * dt * 1e3, depth * 1e6);
  if (m.thickness > 0.0 && depth_raw > depth)
    std::printf("   (raw %.3f um, saturated at the %.0f um foil)", depth_raw * 1e6,
                m.thickness * 1e6);
  std::printf("\n");
  std::printf("  clamp census (counts are node/column-steps over the whole run):\n");
  std::printf("    Vm_max              %10ld\n", h_cen(0));
  std::printf("    T_cap upper         %10ld\n", h_cen(1));
  std::printf("    T0 floor            %10ld\n", h_cen(2));
  std::printf("    recession cap       %10ld   of %ld receding column-steps (%.1f %%)\n",
              h_cen(3), h_cen(4), h_cen(4) ? 100.0 * double(h_cen(3)) / double(h_cen(4)) : 0.0);
  std::printf("    steps with >=1 ignited column %ld of %ld\n", h_cen(6), steps);
  std::printf("  first ignition at t = %.6f ms  (step %ld)\n",
              (steps - h_cen(6)) * dt * 1e3, steps - h_cen(6));

  if (!o.out.empty()) {
    std::FILE* f = std::fopen(o.out.c_str(), "w");
    if (f) {
      std::fprintf(f, "# %s %s  P=%.0f W spot=%.0f um dx=%.4g m steps=%ld\n",
                   Enth ? "enthalpy" : "apparent-cp", L::name, o.P, o.spot, dx, steps);
      std::fprintf(f, "t_s,depth_m\n");
      for (std::size_t i = 0; i < hist_t.size(); ++i) {
        double dv = hist_d[i];
        if (m.thickness > 0.0) dv = std::min(dv, m.thickness);
        std::fprintf(f, "%.9e,%.9e\n", hist_t[i], dv);
      }
      std::fclose(f);
      std::printf("  trace -> %s (%zu samples)\n", o.out.c_str(), hist_t.size());
    }
  }

  int fails = 0;
  if (!ref.t.empty()) {
    // The reference columns are t_s, depth_m -- SI, not microns.
    const double rd = trace_at(ref, steps * dt);
    std::printf("\n  cross-check against the Python:\n");
    std::printf("    final depth  port %.3f um   python %.3f um   diff %+.3f um (%+.3f %%)\n",
                depth * 1e6, rd * 1e6, (depth - rd) * 1e6,
                rd > 0 ? 100.0 * (depth / rd - 1.0) : 0.0);
    for (double tt : {0.25e-3, 0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3}) {
      const double pv = trace_at(ref, tt) * 1e6;
      double ov = 0;
      for (std::size_t i = 0; i < hist_t.size(); ++i)
        if (hist_t[i] <= tt) ov = hist_d[i] * 1e6;
      std::printf("    t = %.2f ms   port %8.3f um   python %8.3f um   diff %+7.3f um\n",
                  tt * 1e3, ov, pv, ov - pv);
    }
    // The whole trace, not five points: a port can hit every checkpoint and
    // still take a different route between them.
    double worst = 0.0, worst_t = 0.0, sum2 = 0.0;
    int n = 0;
    for (std::size_t i = 0; i < hist_t.size(); ++i) {
      const double pv = trace_at(ref, hist_t[i]) * 1e6;
      const double e  = hist_d[i] * 1e6 - pv;
      if (std::abs(e) > std::abs(worst)) { worst = e; worst_t = hist_t[i]; }
      sum2 += e * e; ++n;
    }
    if (n) std::printf("    over the whole trace (%d samples): rms %.3f um, "
                       "worst %+.3f um at t = %.4f ms\n",
                       n, std::sqrt(sum2 / n), worst, worst_t * 1e3);
    std::printf("    NOTE: agreement here proves the PORT, not the physics -- both\n"
                "          run the same 13-parameter fit, and the Python's own\n"
                "          calibration is not grid-converged.\n");
  }
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
    if      (!std::strcmp(argv[i], "-P"))      next(o.P);
    else if (!std::strcmp(argv[i], "-spot"))   next(o.spot);
    else if (!std::strcmp(argv[i], "-dx"))     { double v; next(v); o.dx = v * 1e-6; }
    else if (!std::strcmp(argv[i], "-t"))      { double v; next(v); o.t_max = v * 1e-3; }
    else if (!std::strcmp(argv[i], "-etamin")) next(o.eta_min);
    else if (!std::strcmp(argv[i], "-capfree")) next(o.cap_free);
    else if (!std::strcmp(argv[i], "-capcells")) {
      o.cap_cells = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') o.cap_val = std::atof(argv[++i]);
    }
    else if (!std::strcmp(argv[i], "-decay"))   { double v; next(v); o.decay = v * 1e-6; }
    else if (!std::strcmp(argv[i], "-dignite")) { double v; next(v); o.d_ign = v * 1e-6; }
    else if (!std::strcmp(argv[i], "-enthalpy")) o.enthalpy = true;
    else if (!std::strcmp(argv[i], "-d3q27"))  o.d3q27 = true;
    else if (!std::strcmp(argv[i], "-every"))  { double v; next(v); o.every = int(v); }
    else if (!std::strcmp(argv[i], "-nz"))     { double v; next(v); o.nz = Index(v); }
    else if (!std::strcmp(argv[i], "-n"))      { double v; next(v); o.nx = o.ny = Index(v); }
    else if (!std::strcmp(argv[i], "-ref"))    { if (i + 1 < argc) o.ref = argv[++i]; }
    else if (!std::strcmp(argv[i], "-out"))    { if (i + 1 < argc) o.out = argv[++i]; }
    else if (!std::strcmp(argv[i], "-muhammad")) {
      o.muhammad = true;
      o.P = 100.0; o.spot = 50.0;        // rb = 25 um, quoted as a RADIUS
      o.dx = 2.5e-6; o.t_max = 0.15e-3;
      o.nx = o.ny = 61; o.nz = 60;       // 150 um / 2.5 um, the exact foil
      o.eta_min = 0.12; o.d_ign = 0.0; o.decay = 0.0;
    }
    else if (!std::strncmp(argv[i], "--kokkos", 8)) { /* Kokkos' own flags */ }
    else { std::fprintf(stderr, "keyhole: unknown option %s\n", argv[i]); }
  }
  if (o.enthalpy && o.d3q27) {
    std::fprintf(stderr,
        "keyhole: -enthalpy -d3q27 is not available. EnthalpyRegularised's\n"
        "  rest-plus-axial-pairs contract is D2Q5/D3Q7 only, and the ladder's\n"
        "  memory budget rules D3Q27 out past the first rung anyway.\n"
        "  Valid: (default) cp+D3Q7, -d3q27, -enthalpy.  NOTHING WAS RUN.\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("reduced keyhole model, ported from the Python twin\n");
    std::printf("backend %s   precision %s\n",
                Kokkos::DefaultExecutionSpace::name(), precision_name());
    if (o.enthalpy)     rc = run<D3Q7,  EnthalpyBGK<D3Q7>, true >(o);
    else if (o.d3q27)   rc = run<D3Q27, ScalarBGK<D3Q27>,  false>(o);
    else                rc = run<D3Q7,  ScalarBGK<D3Q7>,   false>(o);
  }
  Kokkos::finalize();
  return rc;
}
