//==============================================================================
//  Flow through a voxelised patient-specific aorta -- arbitrary geometry.
//
//  Everything else in this suite runs on a geometry described by a formula. This
//  one loads a voxel array off disk and runs the same solver on it unchanged,
//  which is the point: `set_geometry` already takes an arbitrary predicate, so
//  supporting real geometry needs a reader, not a new solver.
//
//  GEOMETRY. src/io/VoxelGeometry.hpp reads the file the aorta project's
//  scripts/voxelize.py produces from the SimVascular surface mesh (case
//  0074_H_AO_H): 109 x 184 x 361 voxels at a pitch of 0.0616 cm, tagged
//  0 solid, 1 fluid, 2 inlet, 3 outlet. About 16% of the box is fluid.
//
//  D3Q27, not the D3Q19 the original project uses -- see the note on lattice
//  scope in doc/. The geometry is lattice-independent, so the only consequence
//  is that results are not directly comparable with that project's.
//
//  BOUNDARY CONDITIONS.
//    solid   halfway bounce-back, which is free under Esoteric Pull
//    inlet   regularised velocity wall, u = U * inward normal from the file
//    outlet  constant back-pressure, rho = 1
//
//  The inlet normal is stored in the file because the ascending-aorta cap is
//  oblique to the voxel axes; using a face normal instead would drive flow into
//  the wall. The outlet caps are left at a fixed pressure rather than a
//  prescribed flow split, so the division of flow between the branch vessels is
//  an OUTPUT of the simulation rather than an input.
//
//  WHAT IS AND IS NOT CHECKED. There is no analytic solution here, so this is
//  not an accuracy test and is not presented as one. What it verifies is that
//  the machinery holds up on a geometry with no symmetry: that the solver stays
//  finite, that mass is conserved to round-off once the flow is established,
//  and that what goes in at the inlet comes out at the outlets. That last one is
//  the useful check -- a leak through the bounce-back surface, or a mis-tagged
//  cap, shows up there and nowhere else.
//
//  WHAT THE OUTLET DOES WHEN THE DRIVE STOPS, MEASURED 2026-09-13. Under the
//  physiological waveform the inlet is at REST for 57% of the cycle, and
//  `NrmOutFree` then has no through-flow left to copy: it rescales a copied
//  distribution to an IMPOSED DENSITY, so in diastole that density is the only
//  thing driving the vessel. The descending aorta runs FASTER in diastole than
//  at peak systole -- p99 of |u| in the exit band 0.0543 against 0.0142 --
//  which momentum coasting after ejection cannot do. 100% of the top-0.1%
//  voxels sit within 2 voxels of an outlet cap (91% within 1) against 0% at
//  systole, and 80.5% of them belong to ONE cap, the 449-voxel descending-aorta
//  cap at z = 6. The old waveform never showed this because its diastolic floor
//  of 15% of peak kept flow moving through the caps at all times.
//
//  TWO EXPLANATIONS WERE TESTED AND KILLED, and both were plausible. An
//  ACOUSTIC ECHO: valve closure peaks at step 585, sound crosses 361 cells in
//  626 steps, and 585 + 626 lands at phase 0.807 -- which is where the peak is.
//  A band-by-band space-time map killed it: the maximum is PINNED at z ~ 10
//  from phase 0.70 to 0.95 and never propagates, and a wave moves. A BAD DONOR:
//  the rule in FluidSolver detects "outward" only as a neighbour that is Solid
//  or off-domain, and a NEIGHBOURING OUTLET NODE is neither -- the same blind
//  spot GPU/'s ehd_cavity had. But cap #3 resolves its true outward direction
//  for 419 of its 449 nodes (mean alignment +0.930, 6.7% badly aligned) against
//  cap #2's 10.0%, and cap #2 shows no artefact at all. The hole in that rule
//  is real and does not bite on this geometry.
//
//  THE PROBE THAT CONFIRMED IT ALSO FAILED AS A FIX. `-fragmin 33` demotes the
//  25 staircase fragment caps (1-32 voxels, 105 in total, all at z = 1..8) to
//  Solid, removing about 19% of the outlet area at that end. The diastolic exit
//  velocity went UP by a near-constant 11.8% at every diastolic phase (11.8,
//  11.9, 11.8, 11.8, 13.1% at phases 0.35 to 0.90) and by 1.0% at systole. That
//  is a FLUX-driven boundary: less area, the same imposed-density-driven flux,
//  higher speed. Real flow would have gone the other way. So `-fragmin` is a
//  diagnostic rather than a repair, and it is off by default.
//
//  WHAT STILL STANDS, AND WHAT DOES NOT. The contamination is confined to about
//  twenty voxels of each cap. At peak systole the z-bands holding the inlet
//  (z = 137) and the whole systolic core (z = 100-179) move by 0.6-2.9% when
//  those 105 voxels are removed, so systolic velocities in the ascending aorta
//  and arch are not outlet-driven. The FLOW SPLIT is the casualty: the branch
//  end at z ~ 330-350 moves by 16-23% from a change made 350 voxels away, so
//  the division of flow between the branch vessels -- called an OUTPUT rather
//  than an input above -- is a soft one, and 105 voxels of staircase debris
//  move it by about a fifth. Do not quote a branch flow fraction from this
//  geometry without saying that.
//==============================================================================
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "boundary/Regularized.hpp"
#include "io/VoxelGeometry.hpp"
#include "FieldDump.hpp"
#include "memory/EsotericPull.hpp"
#include "memory/Storage.hpp"
#include "solver/FluidSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lbm;

//------------------------------------------------------------------------------
// Pulsatile inlet waveforms, in lattice time units.
//
// TWO OF THEM, AND THE DIFFERENCE IS NOT COSMETIC. The return value multiplies
// the inlet velocity, so the DIRECTION is fixed by the cap normal and only the
// magnitude varies -- which is what lets the whole drive be a scale on the wall
// table rather than a per-node update. `set_wall_velocity_scale` multiplies the
// ORIGINAL table, so a negative scale reverses the inlet cleanly; that is what
// makes the diastolic backflow lobe below possible at all.
//
//   Wave::Smooth   the profile the aorta project drives its inlet with,
//                  reproduced unchanged so the two runs stay comparable: a
//                  cycle running between a diastolic floor at 15% of peak and a
//                  systolic peak of 1, with the upstroke sharpened by raising
//                  the sinusoid to the 1.5 power. Phase 0 is the diastolic
//                  minimum; the systolic peak falls at phase 0.5.
//
//   Wave::Physio   an aortic flow waveform. It is NOT a smoothed version of the
//                  above -- it differs in the three properties that decide what
//                  the flow does:
//
//                    ejection occupies 0.35 of the cycle, not all of it
//                      (0.30 s of a 0.857 s beat at 70 bpm)
//                    the peak sits at phase 0.13, not 0.50
//                    diastole is at REST, not at 15% of peak, with a small
//                      reverse lobe at valve closure
//
//                  Measured on the form below: peak at phase 0.1300, cycle
//                  mean 0.2069 of peak, reverse volume 2.40% of forward
//                  (physiological regurgitant fraction through closure is a few
//                  per cent). The smooth waveform's cycle mean is 0.5108 of
//                  peak -- 2.5x the flow for the same Doppler peak velocity, so
//                  the two do not describe the same cardiac output and their
//                  Reynolds numbers are not comparable at matched U.
//
//                  WHY THE SHAPE MATTERS AT HIGH WOMERSLEY. At alpha = 13 the
//                  core is inertia-dominated and follows the drive almost
//                  instantly, so the flow structure is set by the waveform's
//                  DERIVATIVE as much as its amplitude. The deceleration after
//                  peak systole is what produces separation and reverse flow
//                  near the wall; a drive that never leaves 15% of peak and
//                  decelerates over half a cycle does not produce it. Running
//                  this case for aortic haemodynamics with Wave::Smooth is the
//                  units error of the waveform: every dimensionless group can
//                  be right and the flow still be the wrong flow.
//
//                  The skew exponent is what puts the peak early: with
//                  x = phi/phi_s, sin(pi x^0.7) peaks at x = 0.372 rather than
//                  at 0.5. It is a shape fit to the three properties above, not
//                  a Fourier reconstruction of any one published trace.
//------------------------------------------------------------------------------
enum class Wave { Smooth, Physio };

static double inlet_profile(double t, double ramp, double period, Wave w) {
  const double ramp_factor = std::min(1.0, t / ramp);
  const double phase = std::fmod(t, period) / period;
  if (w == Wave::Smooth) {
    const double wave = 0.5 + 0.5 * std::sin(2.0 * M_PI * phase - M_PI / 2.0);
    return ramp_factor * (0.15 + 0.85 * std::pow(wave, 1.5));
  }
  const double phi_s = 0.35;   // ejection duration / cycle
  const double skew  = 0.70;   // < 1 moves the systolic peak early
  const double sharp = 1.30;   // exponent on the sine
  const double back  = 0.10;   // reverse lobe amplitude, fraction of peak
  const double back_w = 0.08;  // reverse lobe width, in phase
  double q;
  if (phase < phi_s) {
    q = std::pow(std::sin(M_PI * std::pow(phase / phi_s, skew)), sharp);
  } else if (phase < phi_s + back_w) {
    q = -back * std::sin(M_PI * (phase - phi_s) / back_w);
  } else {
    q = 0.0;
  }
  return ramp_factor * q;
}

// Cycle mean of a waveform, as a fraction of its peak. Used to turn a peak
// velocity into a cardiac output without assuming the shape.
static double wave_mean(Wave w, double period) {
  const int N = 20000;
  double s = 0;
  // ramp = 0 would divide; a ramp far below one step makes ramp_factor 1 for
  // every sample taken, which is what a cycle mean of the SHAPE needs.
  for (int i = 0; i < N; ++i)
    s += inlet_profile((double(i) + 0.5) * period / N, 1e-9, period, w);
  return s / N;
}

using L  = D3Q27;
using CM = CentralMoments<L, NoForcing, ShiftedPopulations>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    // Default geometry: $LBM_AORTA_GEOM, else a path relative to the working
    // directory; -geom overrides both. This was an absolute path into one
    // developer's home directory, which made the case unrunnable for anyone
    // else who checked the repository out.
    const char* env_geom = std::getenv("LBM_AORTA_GEOM");
    std::string path = env_geom ? env_geom : "data/geometry.bin";
    double Re = 200.0;
    Real U = Real(0.02);
    std::size_t steps = 20000, probe = 500;
    bool pulse = false;
    double period = 2000.0, ramp = 800.0;
    bool ramp_set = false;
    Wave wave = Wave::Smooth;
    // -physio inputs. Blood is Newtonian at 3.5 mPa s over 1060 kg/m^3, which
    // is the high-shear asymptote and the standard aortic value; the vessel is
    // well above the shear rate where rouleaux make that wrong.
    bool physio = false;
    double hr = 70.0;         // beats per minute
    double upeak = 100.0;     // peak systolic velocity, cm/s (clinical Doppler)
    double nu_blood = 0.0330; // cm^2/s
    double tau = 0.515;       // the lattice knob that is actually free
    double umax = 0.035;      // Mach clamp on the lattice peak inlet speed
    double beats = 0.0;       // run length, in beats; 0 = use -steps
    std::size_t fragmin = 0;  // demote outlet caps smaller than this to Solid
    std::size_t dumpfrom = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-geom"  && i + 1 < argc) path = argv[++i];
      if (a == "-re"    && i + 1 < argc) Re = std::atof(argv[++i]);
      if (a == "-u"     && i + 1 < argc) U = Real(std::atof(argv[++i]));
      if (a == "-steps" && i + 1 < argc) steps = std::size_t(std::atol(argv[++i]));
      if (a == "-probe" && i + 1 < argc) probe = std::size_t(std::atol(argv[++i]));
      if (a == "-pulse") pulse = true;
      if (a == "-period" && i + 1 < argc) period = std::atof(argv[++i]);
      if (a == "-ramp"   && i + 1 < argc) { ramp = std::atof(argv[++i]); ramp_set = true; }
      if (a == "-dumpfrom" && i + 1 < argc) dumpfrom = std::size_t(std::atol(argv[++i]));
      if (a == "-wave" && i + 1 < argc) {
        const std::string w = argv[++i];
        wave = (w == "physio") ? Wave::Physio : Wave::Smooth;
      }
      if (a == "-physio") { physio = true; pulse = true; wave = Wave::Physio; }
      if (a == "-hr"    && i + 1 < argc) hr    = std::atof(argv[++i]);
      if (a == "-upeak" && i + 1 < argc) upeak = std::atof(argv[++i]);
      if (a == "-tau"   && i + 1 < argc) tau   = std::atof(argv[++i]);
      if (a == "-umax"  && i + 1 < argc) umax  = std::atof(argv[++i]);
      if (a == "-beats" && i + 1 < argc) beats = std::atof(argv[++i]);
      if (a == "-fragmin" && i + 1 < argc) fragmin = std::size_t(std::atol(argv[++i]));
    }

    std::printf("Aorta: flow through a voxelised patient geometry   D3Q27, central moments\n");
    std::printf("backend %s   precision %s\n\n", ExecSpace::name(), precision_name());

    VoxelGeometry g;
    try {
      g = load_voxel_geometry(path);
    } catch (const std::exception& e) {
      std::printf("  %s\n", e.what());
      Kokkos::finalize();
      return 1;
    }
    report(g, "aorta");

    const std::size_t n_in  = g.count_of(VoxelGeometry::TagInlet);
    const std::size_t n_out = g.count_of(VoxelGeometry::TagOutlet);
    if (n_in == 0 || n_out == 0) {
      std::printf("  geometry has no inlet or no outlet; nothing to drive\n");
      Kokkos::finalize();
      return 1;
    }

    // Reynolds number on the inlet cap's equivalent diameter: the cap holds
    // n_in voxels, so its area is n_in in lattice units and D = 2 sqrt(A/pi).
    //
    // THAT DIAMETER IS 14% TOO LARGE, and it is kept anyway. A voxelised cap
    // oblique to the axes holds about A*max|n_i| voxels for a true planar area
    // A, so counting voxels OVERSTATES the area by 1/max|n_i| = 1/0.832 here.
    // The exact area is the magnitude of the summed exposed-face normals -- the
    // same identity the flux measurement below rests on, since a staircase and
    // the plane it approximates bound a closed region whose total vector area
    // is zero. Measured: 653.2 against 851 voxels, so D_true = 28.84 lattice
    // units (1.777 cm) against 32.92. The voxel-count form still sets nu from
    // -re, because every number recorded for this case was produced that way
    // and changing it silently would move them all by 14%; the true diameter is
    // what the -physio scaling below uses, and both are printed.
    const double Dlb = 2.0 * std::sqrt(double(n_in) / M_PI);
    double Ain = 0.0;
    {
      const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
      double A[3] = {0, 0, 0};
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x) {
            if (g.at(x, y, z) != VoxelGeometry::TagInlet) continue;
            for (int k = 0; k < 6; ++k) {
              const Index qx = x + dirs[k][0], qy = y + dirs[k][1], qz = z + dirs[k][2];
              const bool exposed =
                  (qx < 0 || qx >= g.nx || qy < 0 || qy >= g.ny ||
                   qz < 0 || qz >= g.nz) ||
                  g.at(qx, qy, qz) == VoxelGeometry::TagSolid;
              if (!exposed) continue;
              A[0] += dirs[k][0]; A[1] += dirs[k][1]; A[2] += dirs[k][2];
            }
          }
      Ain = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
    }
    const double Dtrue = 2.0 * std::sqrt(Ain / M_PI);

    // PHYSIOLOGICAL SCALING. One free lattice parameter, and it is NOT the
    // Reynolds number.
    //
    // dx is fixed by the geometry file -- it IS the voxel pitch. Choosing dt
    // then fixes everything: u_lat = u_phys dt/dx, nu_lat = nu_phys dt/dx^2,
    // T_lat = T_phys/dt, and both dimensionless groups come out at their
    // physiological values automatically. So a fully physiological run has NO
    // freedom left, and this is what it demands:
    //
    //     u_peak = 0.05 (the Mach ceiling) forces dt = 3.08e-5 s
    //     -> nu_lat = 2.68e-4, tau = 0.50080, 27825 steps per beat
    //
    // tau = 0.50080 is BELOW the tau_crit = 0.5034-0.5077 this tree measured
    // for its own regularised wall, and 28 beats at that period is 7.8e5 steps
    // -- about ten hours here. So the physiological point is not reachable, and
    // pretending otherwise by quoting Re and hoping is the failure mode this
    // file's siblings are full of.
    //
    // WHAT IS GIVEN UP, DELIBERATELY. tau is taken as the free parameter and
    // the WOMERSLEY number is matched exactly, because alpha is what makes a
    // pulsatile flow pulsatile: it sets how far the wall shear layer penetrates
    // in a beat and therefore the phase lag between drive and core. Re then
    // falls out of the Mach clamp and is short by a stated factor. The run is
    // dynamically similar in alpha and NOT in Re: near-wall oscillatory
    // structure and timing carry over, convective secondary flow does not --
    // which in an arch means the Dean number, and so the helicity, is low by
    // the same factor. Say that rather than showing a spiral and calling it
    // physiological.
    //
    // One invariant worth knowing before choosing a run length: the number of
    // BEATS in one viscous diffusion time is alpha^2/(2 pi), independent of tau
    // -- because T_lat and R^2/nu both scale as 1/nu. At alpha = 13.2 that is
    // 27.9 beats, and no choice of lattice parameters makes it cheaper in
    // beats. Only in steps.
    Real nu = Real(double(U) * Dlb / Re);
    if (physio) {
      const double dx = g.pitch;                       // cm, from the file
      const double Dphys = Dtrue * dx;                 // cm
      const double Tphys = 60.0 / hr;                  // s
      const double Rlb = 0.5 * Dtrue;
      const double alpha =
          0.5 * Dphys * std::sqrt(2.0 * M_PI / (Tphys * nu_blood));
      const double Rephys = upeak * Dphys / nu_blood;

      const double nulb = (tau - 0.5) / 3.0;           // D3Q27, cs2 = 1/3
      // alpha matched, then the period ROUNDED TO A MULTIPLE OF 20 so that a
      // probe interval can divide it exactly. Cycle-to-cycle comparison is the
      // only convergence test this case has -- the same phase in successive
      // beats, which needs samples that land on the same phases every beat --
      // and an exact divisor is what guarantees it. 1490 -> 1500 here, moving
      // alpha by 0.33% since alpha goes as 1/sqrt(T); the value printed below
      // is the one actually run, not the target.
      period = 2.0 * M_PI * Rlb * Rlb / (alpha * alpha * nulb);
      period = 20.0 * std::max(1.0, std::round(period / 20.0));
      const double alpha_run = Rlb * std::sqrt(2.0 * M_PI / (period * nulb));
      const double dt = Tphys / period;                // s per step
      const double u_ideal = upeak * dt / dx;          // what physiology wants
      const double u_use = std::min(umax, u_ideal);

      nu = Real(nulb);
      U = Real(u_use);
      Re = u_use * Dtrue / nulb;
      // TWO BEATS, not half of one. The physiological waveform stops ejecting
      // abruptly at phase 0.35 where the smooth one decelerates over half a
      // cycle, and an abrupt stop in a weakly compressible scheme launches a
      // pressure wave: sound crosses this domain in 361/0.577 = 626 steps, so
      // it reverberates several times per beat rather than leaving. Ramping
      // over two beats spreads the startup step over the same number of
      // acoustic round trips instead of one. It does not remove the per-beat
      // deceleration transient, which is physical in origin and is what the
      // run is there to show -- only the start-from-rest one.
      if (!ramp_set) ramp = 2.0 * period;
      if (beats > 0) steps = std::size_t(beats * period);

      const double wmean = wave_mean(wave, period);
      const double Aphys = M_PI * Dphys * Dphys / 4.0;             // cm^2
      const double CO = wmean * upeak * Aphys * 60.0 / 1000.0;     // L/min
      const double delta = Rlb * std::sqrt(2.0) / alpha_run;       // cells
      const double dtime = Rlb * Rlb / nulb;                       // steps

      std::printf("\n  PHYSIOLOGY -> LATTICE\n");
      std::printf("    blood          nu = %.4f cm^2/s  (mu 3.5 mPa s / rho 1060)\n", nu_blood);
      std::printf("    vessel         D  = %.3f cm at the inlet cap (true area %.1f lu^2)\n", Dphys, Ain);
      std::printf("    heart rate     %.0f bpm -> T = %.4f s\n", hr, Tphys);
      std::printf("    peak systolic  U  = %.0f cm/s -> cardiac output %.2f L/min\n", upeak, CO);
      std::printf("    physiological  Re_peak = %.0f      Womersley alpha = %.2f\n", Rephys, alpha);
      std::printf("    grid           dx = %.6f cm (file pitch)   dt = %.3e s\n", dx, dt);
      std::printf("\n    MATCHED   alpha    = %.2f      against %.2f wanted (period rounded to %.0f)\n",
                  alpha_run, alpha, period);
      std::printf("    NOT       Re_peak  = %.0f      short by %.0fx  (Mach clamp: the\n",
                  Re, Rephys / Re);
      std::printf("                                 drive physiology wants is u = %.3f,\n", u_ideal);
      std::printf("                                 capped at %.3f)\n", umax);
      std::printf("    so the simulated peak velocity is %.2f cm/s, not %.0f\n",
                  u_use * dx / dt, upeak);
      std::printf("\n    resolution     D = %.1f cells, Stokes layer delta = %.2f cells,\n", Dtrue, delta);
      std::printf("                   Re_cell = %.1f at peak\n", u_use / nulb);
      std::printf("    convergence    viscous diffusion time R^2/nu = %.0f steps\n", dtime);
      std::printf("                   = alpha^2/2pi = %.1f beats, whatever tau is\n",
                  alpha_run * alpha_run / (2.0 * M_PI));
      std::printf("    waveform       %s\n",
                  wave == Wave::Physio ? "physiological (ejection 0.35 of cycle, peak at 0.13)"
                                       : "smooth sinusoid -- NOT an aortic waveform");
    }

    std::printf("\n  inlet %zu voxels -> D = %.1f lattice units (voxel count)\n", n_in, Dlb);
    std::printf("                     D = %.1f lattice units (true area %.1f)   Re = %.0f\n",
                Dtrue, Ain, Re);
    std::printf("  U = %.4f   nu = %.6e   tau = %.6f\n",
                double(U), double(nu), 3.0 * double(nu) + 0.5);
    if (pulse) {
      // The Womersley number is what characterises a pulsatile flow: the ratio
      // of the oscillatory inertia to the viscous term. It fixes how far the
      // wall-driven shear layer penetrates in a beat, and so whether the
      // profile is quasi-steady (small alpha) or plug-like with a thin
      // oscillating boundary layer (large alpha). Re alone says nothing here.
      //
      // ON THE TRUE DIAMETER, not the voxel count. alpha goes as D, so the 14%
      // area overstatement above is a 14% overstatement here -- it read 15.11
      // against 13.24 for the same run, which is the difference between this
      // vessel and one beating half again as fast.
      const double omega_c = 2.0 * M_PI / period;
      const double alpha = 0.5 * Dtrue * std::sqrt(omega_c / double(nu));
      std::printf("  pulsatile: period %.0f   ramp %.0f   Womersley alpha = %.2f\n",
                  period, ramp, alpha);
      if (wave == Wave::Physio)
        std::printf("             physiological waveform: U is the SYSTOLIC PEAK, ejection\n"
                    "             lasts 0.35 of the cycle, diastole is at REST\n");
      else
        std::printf("             U is the SYSTOLIC PEAK; diastolic floor is 15%% of it\n");
    }
    std::printf("\n");

    // FRAGMENT CAPS. The voxeliser tags a frontier voxel as an outlet wherever
    // stepping outward leaves the fluid, and where the vessel is clipped by the
    // domain edge that produces specks: this geometry has FOUR real caps (526,
    // 461, 459, 449 voxels, D = 1.5 cm each) and TWENTY-FIVE fragments of 1 to
    // 32 voxels, 105 voxels in total, all at z = 1..8 beside the descending
    // aorta's own cap. A one-voxel outlet is not a vessel; it is a hole in the
    // wall with a pressure condition on it.
    //
    // -fragmin N demotes every cap smaller than N to Solid, which turns those
    // holes back into wall. It is off by default: the four real caps are all
    // >= 449 voxels, so -fragmin 33 removes exactly the fragments and nothing
    // else, and leaving it off reproduces every number recorded for this case.
    std::vector<std::uint8_t> demote(g.count(), 0);
    if (fragmin > 0) {
      // 26-connected flood fill over the outlet tag.
      std::vector<int> lab(g.count(), -1);
      std::vector<std::vector<Index>> comps;
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x) {
            const Index n0 = Index(x) + g.nx * (Index(y) + g.ny * Index(z));
            if (g.at(x, y, z) != VoxelGeometry::TagOutlet || lab[n0] >= 0) continue;
            const int id = int(comps.size());
            comps.push_back({});
            std::vector<Index> st{n0};
            lab[n0] = id;
            while (!st.empty()) {
              const Index c = st.back(); st.pop_back();
              comps[id].push_back(c);
              const Index cx = c % g.nx, cy = (c / g.nx) % g.ny, cz = c / (g.nx * g.ny);
              for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                  for (int dx = -1; dx <= 1; ++dx) {
                    const Index rx = cx + dx, ry = cy + dy, rz = cz + dz;
                    if (rx < 0 || rx >= g.nx || ry < 0 || ry >= g.ny ||
                        rz < 0 || rz >= g.nz) continue;
                    const Index m = rx + g.nx * (ry + g.ny * rz);
                    if (lab[m] >= 0 || g.at(rx, ry, rz) != VoxelGeometry::TagOutlet) continue;
                    lab[m] = id; st.push_back(m);
                  }
            }
          }
      std::size_t nd = 0, ncap = 0;
      for (const auto& c : comps) {
        if (c.size() >= fragmin) { ++ncap; continue; }
        for (const Index m : c) { demote[m] = 1; ++nd; }
      }
      std::printf("  [fragmin %zu] %zu outlet caps kept, %zu demoted to Solid (%zu voxels)\n",
                  fragmin, ncap, comps.size() - ncap, nd);
    }

    Domain d(g.nx, g.ny, g.nz, false, false, false);
    CM coll;
    coll.omega = CM::omega_from_viscosity(nu);
    FluidSolver<L, EsotericPull<L>, CM> s(d, coll);

    // Solid stays solid. Inlet and outlet caps become regularised walls; every
    // other fluid voxel collides normally.
    s.set_geometry([&](Index x, Index y, Index z) -> CellType {
      const std::uint8_t t = g.at(x, y, z);
      if (demote[std::size_t(x) + g.nx * (std::size_t(y) + g.ny * std::size_t(z))]) return Solid;
      if (t == VoxelGeometry::TagSolid) return Solid;
      if (t == VoxelGeometry::TagInlet || t == VoxelGeometry::TagOutlet) return RegWall;
      return Fluid;
    });

    // The inlet velocity is along the cap normal stored in the file. The
    // regularised condition needs a face normal code as well, and a voxelised
    // oblique cap has no single one -- so the cap is given NrmCorner, whose
    // unknown set is built geometrically per node rather than from an axis.
    const Real ux = Real(U * Real(g.inlet_normal[0]));
    const Real uy = Real(U * Real(g.inlet_normal[1]));
    const Real uz = Real(U * Real(g.inlet_normal[2]));
    using WS = decltype(s)::WallSpec;
    s.set_regularized_walls([&](Index x, Index y, Index z) -> WS {
      const std::uint8_t t = g.at(x, y, z);
      if (demote[std::size_t(x) + g.nx * (std::size_t(y) + g.ny * std::size_t(z))]) return WS{};
      if (t == VoxelGeometry::TagInlet)  return WS{NrmCorner, ux, uy, uz, Real(1)};
      // The outlet caps are oblique to the voxel axes, so they get the
      // arbitrary-face outflow: rho pinned to 1, velocity taken from a fluid
      // neighbour found per node. Imposing u = 0 here instead would make them
      // walls and the domain would be closed -- measured as mass rising
      // monotonically, +8.5e-3 in 2000 steps, before this was fixed.
      if (t == VoxelGeometry::TagOutlet) return WS{NrmOutFree, Real(0), Real(0), Real(0), Real(1)};
      return WS{};
    });
    // The finite-difference corner stress walks a two-node stencil off each
    // wall node. In a box those neighbours are fluid or wall; in a vessel they
    // are frequently SOLID, whose populations Esoteric Pull never updates, so
    // the stencil reads stale memory and feeds garbage stress into the inlet.
    // Use the local closure instead until that path is made geometry-aware.
    s.set_fd_corners(false);
    s.initialize(Real(1));

    // Flux through a tagged cap, as a sum over EXPOSED FACES.
    //
    // A voxelised cap is a staircase of unit axis-aligned faces. For a
    // staircase approximating a plane of true area A and unit normal n, the
    // exposed faces satisfy  sum(n_face) = A n  -- the two surfaces bound a
    // closed region, whose total vector area is zero. So
    //
    //     sum over exposed faces of  u . n_face   =   A (u . n)
    //
    // which is the true flux, exactly, for uniform u. No area or normal needs
    // to be known: the geometry supplies both.
    //
    // The exposed-face test is the same one scripts/voxelize.py uses to DEFINE
    // a cap -- a frontier voxel is one where stepping outward leaves the fluid
    // -- so measuring the cap the way it was tagged keeps the two consistent.
    //
    // Two earlier versions were wrong in opposite directions. Counting voxels
    // and multiplying by the imposed speed UNDERcounts the inlet: a one-voxel
    // frontier layer holds about A*max|n_i| voxels, so with a dominant
    // component of 0.832 it reported 17.02 for a true flux near 20.5.
    // Normalising the summed face normals to a unit vector per voxel, as the
    // outlet first did, discards the |n| that carries the area and is wrong
    // whenever a voxel exposes more than one face.
    //
    // Sign convention: n_face points OUT of the fluid, so an inlet returns a
    // negative flux and an outlet a positive one.
    //
    // THEY DO NOT BALANCE, and that is a property of the outlet condition, not
    // a measurement error. The outflow REPLACES the populations at its nodes
    // with a rescaled copy of the donor's, so those nodes are a mass source and
    // sink rather than a conserving boundary; the velocity reported there is
    // the reconstructed one, not a record of what actually crossed the face.
    // Total mass is held flat by that same rescaling, so mass drift is NOT an
    // independent check either -- both quantities are downstream of the same
    // imposed rho. Read Q in as exact (it is the imposed inlet velocity through
    // a cap whose area is known to 0.65 degrees) and Q out as indicative only.
    // PRECOMPUTED TRAVERSALS. Every probe used to walk all 7.24M voxels three
    // times -- once for |u|max and once per cap -- to touch 1.18M fluid nodes
    // and 2851 cap nodes. The lists below are built once, in exactly the z,y,x
    // order those loops used, so the arithmetic is performed on the same values
    // in the same order and the output is bit-identical: verified against the
    // 400-step reference row (Q in -1.298043e+01, Q out 1.142291e+00, |u|max
    // 3.346624e-02, mass 4.50e-03). That is the whole reason the per-face
    // entries are kept separate rather than pre-summed into one normal per
    // voxel -- summing them early is algebraically identical and changes the
    // last bits.
    //
    // Measured at 0.146 s per probe afterwards (39 extra probes cost 5.70 s on
    // the 400-step timing run), which is what makes sampling the beat twenty
    // times per cycle affordable over a 28-beat run.
    std::vector<Index> live;                       // every non-solid node
    // AND A SECOND LIST WITHOUT THE CAPS. A regularised wall reports the
    // IMPOSED velocity rather than a population moment, so a cap node is not a
    // measurement -- and there are 851 of them at exactly U against a top-0.1%
    // cut of 1184 nodes, which pinned p99.9 to the inlet drive to four figures
    // for the whole early transient. |u|max still uses `live`, because the
    // non-finite check has to cover every node the solver writes and because
    // every number recorded for this case was measured that way.
    std::vector<Index> interior;                   // fluid only, for statistics
    live.reserve(std::size_t(n_in) + n_out + g.count_of(VoxelGeometry::TagFluid));
    struct Face { Index n; int dx, dy, dz; };
    std::vector<Face> face_in, face_out;
    std::vector<double> speeds;                    // probe scratch for the percentile
    {
      const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
      for (Index z = 0; z < g.nz; ++z)
        for (Index y = 0; y < g.ny; ++y)
          for (Index x = 0; x < g.nx; ++x) {
            const std::uint8_t tv = g.at(x, y, z);
            if (tv == VoxelGeometry::TagSolid) continue;
            // A demoted fragment is Solid to the solver, so its velocity is
            // never written; leaving it in the probe lists would average zeros
            // into the statistics and add null faces to the outlet flux.
            if (demote[std::size_t(x) + g.nx * (std::size_t(y) + g.ny * std::size_t(z))]) continue;
            live.push_back(d.id(x, y, z));
            if (tv == VoxelGeometry::TagFluid) interior.push_back(d.id(x, y, z));
            if (tv != VoxelGeometry::TagInlet && tv != VoxelGeometry::TagOutlet) continue;
            auto& dst = (tv == VoxelGeometry::TagInlet) ? face_in : face_out;
            for (int k = 0; k < 6; ++k) {
              const Index qx = x + dirs[k][0], qy = y + dirs[k][1], qz = z + dirs[k][2];
              const bool exposed =
                  (qx < 0 || qx >= g.nx || qy < 0 || qy >= g.ny ||
                   qz < 0 || qz >= g.nz) ||
                  g.at(qx, qy, qz) == VoxelGeometry::TagSolid;
              if (exposed) dst.push_back(Face{d.id(x, y, z), dirs[k][0], dirs[k][1], dirs[k][2]});
            }
          }
      std::printf("  [probe] %zu live nodes (%zu interior), %zu inlet faces, %zu outlet faces\n",
                  live.size(), interior.size(), face_in.size(), face_out.size());
    }

    //
    // THE HOST MIRRORS ARE PASSED IN, not made here. Each mirror is 58 MB on
    // this geometry and the probe already holds three; making three more per
    // cap call meant nine copies per probe instead of three. Bit-identical --
    // the caller's mirrors are taken after the same compute_macroscopic() this
    // used to follow -- and it is a third of the probe cost, which on a
    // 28-beat run is the difference between 11 and 4 minutes of pure copying.
    auto cap_flux = [&](const std::vector<Face>& faces, const auto& hxf,
                        const auto& hyf, const auto& hzf) {
      double q = 0;
      for (const Face& f : faces) {
        const double u3[3] = {double(hxf(f.n)), double(hyf(f.n)), double(hzf(f.n))};
        q += u3[0] * f.dx + u3[1] * f.dy + u3[2] * f.dz;
      }
      return q;
    };

    if (pulse)
      std::printf("  %8s %6s %11s %11s %11s %13s %13s %10s %10s %9s\n", "step", "phase",
                  "|u| mean", "|u| p99.9", "|u| max", "Q in", "Q out", "imbalance",
                  "mass", "out rho");
    else
      std::printf("  %8s %11s %11s %11s %13s %13s %10s %10s %9s\n", "step",
                  "|u| mean", "|u| p99.9", "|u| max", "Q in", "Q out", "imbalance",
                  "mass", "out rho");
    std::printf("  %s\n", std::string(pulse ? 117 : 110, '-').c_str());

    const Real m0 = s.total_mass();

    // CONSERVING OUTFLOW. The outlet overwrites its nodes every step, so
    // whatever density is imposed there is injected regardless of what arrived
    // -- pinning rho = 1 makes the boundary a mass source and sink, and the
    // inlet and outlet fluxes then differ by ~60% with no way to tell which is
    // wrong. Closing a loop on TOTAL mass instead makes it conserving: raise
    // the outlet density when mass is being lost, lower it when mass is
    // accumulating. The fixed point is the density at which what leaves equals
    // what enters, and it is found rather than assumed.
    //
    // Proportional control on the RELATIVE mass error. Scaling by total mass
    // rather than by outlet-node count matters: the latter gave an error of
    // order 3 for a drift of 5e-3, which saturated the step clamp every time
    // and turned the controller into bang-bang -- density slammed between
    // limits, flow reversed through the outlet, and the run blew up in 1500
    // steps.
    //
    // The controller is also held off until the vessel has filled. Engaging it
    // from rest makes it chase the filling transient, during which mass SHOULD
    // be changing, and it winds up fighting the physics.
    const double gain = 0.6;
    const std::size_t ctrl_every = 50;
    // Under pulsation the hold-off also waits out the ramp and one whole beat,
    // so the first correction is made against a complete cycle, not part of one.
    const std::size_t ctrl_start =
        pulse ? std::max<std::size_t>(3000, std::size_t(ramp + period)) : 3000;
    const double dr_max = 5e-4;
    double out_rho = 1.0;

    // One correction: mass above target -> lower the outlet density, let more
    // out. Clamped per application and in absolute value; an unclamped loop on
    // this went bang-bang once and reversed the flow.
    auto correct = [&](double err) {
      double dr = gain * err;
      if (dr >  dr_max) dr =  dr_max;
      if (dr < -dr_max) dr = -dr_max;
      out_rho -= dr;
      if (out_rho < 0.9) out_rho = 0.9;
      if (out_rho > 1.1) out_rho = 1.1;
      s.set_outflow_density(Real(out_rho));
    };

    // PULSATILE RETIMING. Mass genuinely oscillates within a beat -- the fluid
    // is compressible at O(Ma^2) and the drive is time-varying -- so a
    // controller sampling every 50 steps corrects against the beat itself and
    // flattens the pressure swing the run exists to produce. Under pulsation
    // the error is averaged over a whole period and applied once per cycle,
    // which separates the secular drift, which does need correcting, from the
    // cyclic variation, which does not.
    double err_sum = 0.0;
    std::size_t err_n = 0;

    for (std::size_t t = 0; t <= steps; ++t) {
      // The drive for the step about to be taken, set before ANYTHING reads the
      // wall table this iteration. corner_density() reads it inside step(), and
      // so does the macroscopic kernel, which at a regularised wall reports the
      // IMPOSED velocity rather than a population moment. Setting it after the
      // diagnostics instead makes the t = 0 row -- and the t = 0 volume dump --
      // show the inlet at full U while the waveform says the drive is zero.
      if (pulse)
        s.set_wall_velocity_scale(Real(inlet_profile(double(t), ramp, period, wave)));
      if (t >= ctrl_start && t % ctrl_every == 0) {
        const double err = (double(s.total_mass()) - double(m0)) / double(m0);
        if (pulse) { err_sum += err; ++err_n; }
        else       { correct(err); }
      }
      if (pulse && err_n > 0 && t >= ctrl_start && t % std::size_t(period) == 0) {
        correct(err_sum / double(err_n));
        err_sum = 0.0; err_n = 0;
      }
      if (t % probe == 0) {
        s.compute_macroscopic();
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
        auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
        auto hz = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
        // |u|max IS A SINGLE STAIRCASE CORNER VOXEL and has been since this
        // case was written -- p100 = 0.070 against p99.9 = 0.026 on the
        // documented steady run. It is kept because a non-finite value has to
        // be caught somewhere, and it is the wrong thing to draw a conclusion
        // from or to scale a colour map by. The mean and the 99.9th percentile
        // are what a cycle-to-cycle comparison should read: a corner spike
        // moves with the geometry's worst voxel, a mean moves with the flow.
        double um = 0, usum = 0; bool finite = true;
        for (const Index i : live) {
          const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
          if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) finite = false;
          um = std::max(um, std::sqrt(a * a + b * b + c * c));
        }
        speeds.clear();
        for (const Index i : interior) {
          const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
          const double sp = std::sqrt(a * a + b * b + c * c);
          usum += sp;
          speeds.push_back(sp);
        }
        const double umean = usum / double(interior.size());
        double u999 = 0;
        if (finite && !speeds.empty()) {
          const std::size_t k =
              std::min(speeds.size() - 1, std::size_t(0.999 * double(speeds.size())));
          std::nth_element(speeds.begin(), speeds.begin() + std::ptrdiff_t(k), speeds.end());
          u999 = speeds[k];
        }
        if (!finite) { std::printf("  DIVERGED at step %zu\n", t); status = 1; break; }
        // outward flux at the outlet is -n.u with the file's inward normal
        // convention, but the outlet caps have no stored normal, so the outlet
        // flux is measured as the shortfall in total divergence instead: what
        // enters must leave, and the inlet is the only driven face.
        const double qin = cap_flux(face_in, hx, hy, hz);
        const double qout = cap_flux(face_out, hx, hy, hz);
        // qin is negative (entering), qout positive (leaving); at steady
        // state they cancel, so their sum is the conservation error.
        const double imb = (std::abs(qout) > 0) ? (qin + qout) / std::abs(qout) : 0.0;
        if (pulse)
          std::printf("  %8zu %6.3f %11.4e %11.4e %11.4e %13.6e %13.6e %10.2e %10.2e %9.5f\n",
                      t, std::fmod(double(t), period) / period, umean, u999, um,
                      qin, qout, imb,
                      double(s.total_mass() - m0) / double(m0), out_rho);
        else
          std::printf("  %8zu %11.4e %11.4e %11.4e %13.6e %13.6e %10.2e %10.2e %9.5f\n",
                      t, umean, u999, um, qin, qout, imb,
                      double(s.total_mass() - m0) / double(m0), out_rho);
        // FIGVEC dumps the three velocity COMPONENTS rather than the speed,
        // which is what streamline integration needs -- speed alone gives no
        // direction to follow. Layout is component-major (all ux, then uy, then
        // uz), each plane x-fastest, matching what the source project's
        // render_3d.py reads.
        //
        // No solid sentinel here. The scalar dump marks solid with a negative
        // speed, but a velocity component is legitimately negative, so there is
        // no value left to steal. The renderer takes the mask from geometry.bin
        // instead, which it already opens for the cap tags.
        //
        // These are 3x the size of a speed dump, so -dumpfrom restricts them to
        // a window at the end of the run: a long run needs to converge, but only
        // its last couple of beats need rendering.
        if (std::getenv("FIGVEC") && t >= dumpfrom) {
          using namespace lbm::figdump;
          const std::size_t N = std::size_t(g.nx) * g.ny * g.nz;
          std::vector<float> buf(3 * N);
          std::size_t o = 0;
          for (int c = 0; c < 3; ++c)
            for (Index zz = 0; zz < g.nz; ++zz)
              for (Index yy = 0; yy < g.ny; ++yy)
                for (Index xx = 0; xx < g.nx; ++xx) {
                  if (g.at(xx, yy, zz) == VoxelGeometry::TagSolid) { buf[o++] = 0.0f; continue; }
                  const Index i = d.id(xx, yy, zz);
                  buf[o++] = float(c == 0 ? hx(i) : c == 1 ? hy(i) : hz(i));
                }
          char vf[64];
          std::snprintf(vf, sizeof vf, "aorta_u%04zu.bin", (t - dumpfrom) / probe);
          std::ofstream out(vf, std::ios::binary);
          const std::int32_t a1 = g.nx, b1 = g.ny, c1 = g.nz;
          out.write(reinterpret_cast<const char*>(&a1), 4);
          out.write(reinterpret_cast<const char*>(&b1), 4);
          out.write(reinterpret_cast<const char*>(&c1), 4);
          out.write(reinterpret_cast<const char*>(buf.data()),
                    std::streamsize(buf.size() * sizeof(float)));
          std::printf("  wrote %s (%d x %d x %d, 3 components)\n",
                      vf, int(g.nx), int(g.ny), int(g.nz));
        }
        // -dumpfrom APPLIES HERE TOO. It did not, and FIGVEC's did, which is
        // the kind of asymmetry that is invisible until it fills a disk: a
        // 28-beat run probed twenty times a beat is 560 frames at 29 MB, or
        // 16 GB, where one cardiac cycle is 21 frames and 0.6 GB. A volume
        // animation wants the last beat of a converged run, never the whole
        // history.
        if (std::getenv("FIGVOL") && t >= dumpfrom) {
          // The whole speed field, for the volume renderer. Any single x-plane
          // cuts this vessel into disconnected islands -- the aorta curves out
          // of every plane -- so a slice misrepresents the geometry however the
          // plane is chosen. Solid is written as a negative sentinel: the
          // renderer has to tell wall from stationary fluid, and a zero cannot,
          // because stagnant fluid is also zero.
          using namespace lbm::figdump;
          char vf[64];
          std::snprintf(vf, sizeof vf, "aorta_v%04zu.bin", t / probe);
          scalar_volume(vf, g.nx, g.ny, g.nz, [&](Index xx, Index yy, Index zz) {
            if (g.at(xx, yy, zz) == VoxelGeometry::TagSolid) return -1.0;
            const Index i = d.id(xx, yy, zz);
            const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
            return std::sqrt(a * a + b * b + c * c);
          });
        }
        if (std::getenv("FIGDUMP")) {
          // Speed on the x-slice carrying the most fluid: the vessel runs along
          // z, so a slice normal to x cuts it lengthwise and shows the arch.
          using namespace lbm::figdump;
          static Index xs = -1;
          if (xs < 0) {
            Index best = 0;
            for (Index xx = 0; xx < g.nx; ++xx) {
              Index c = 0;
              for (Index zz = 0; zz < g.nz; ++zz)
                for (Index yy = 0; yy < g.ny; ++yy)
                  if (g.at(xx, yy, zz) != VoxelGeometry::TagSolid) ++c;
              if (c > best) { best = c; xs = xx; }
            }
            std::printf("  [frames] slice x = %d (%d fluid cells)\n", int(xs), int(best));
          }
          char fn[64];
          std::snprintf(fn, sizeof fn, "aorta_f%04zu.bin", t / probe);
          scalar_slice(fn, g.ny, g.nz, [&](Index yy, Index zz) {
            if (g.at(xs, yy, zz) == VoxelGeometry::TagSolid) return -1.0;   // mask
            const Index i = d.id(xs, yy, zz);
            const double a = double(hx(i)), b = double(hy(i)), c = double(hz(i));
            return std::sqrt(a * a + b * b + c * c);
          });
        }
        std::fflush(stdout);
      }
      if (t < steps) s.step();
    }
  }
  Kokkos::finalize();
  return status;
}
