//==============================================================================
//  Murat Kilic's melt-electrowriting set-up in 2-D: PCL melt fed through a
//  0.4 mm nozzle held at 6.5 kV, 5.5 mm above a grounded collector moving at
//  2100 mm/min. What the melt does at the tip -- a cone and a jet, or a growing
//  pile ("oozing") -- is the output.
//
//  For Ju, Omar, Mirihanage & Kilic, "Suppressing oozing during filament-based
//  melt electrowriting" (SSRN 7582986, preprint). The paper is experimental and
//  qualitative, so nothing here is compared against it; this is a demonstrator.
//
//  THE MODEL. FreeSurfaceSolver for the melt (D2Q9, central moments, surface
//  tension, gravity), the gas a void; the field from validation/FsElectric.hpp
//  (D2Q5 TRT potential; melt within -lc of the tip is a conductor at the nozzle
//  voltage and feels eps0 E^2 / 2, verified by validation/fs_electric.cpp). The
//  feed is a static wall with a velocity INTO the bore -- moving-wall bounce-back
//  injects rho U per cell per step, a rate-controlled feed as a filament drive
//  is -- and the collector a static wall moving along itself. Melt reaching
//  either side wall leaves through FreeSurfaceSolver::drain(). D2Q9 + D2Q5
//  rather than the 3-D pair because the free surface has only ever been
//  validated on D2Q9, and a steady Laplace field is the same five-point
//  solution on D2Q5 and D3Q7.
//
//  UNITS. Matched: the capillary numbers of the feed and of the collector, the
//  Bond number and eps0 V^2 / (gamma R). Not matched: Re and Oh, which the
//  lattice keeps creeping (Re_coll 0.32, Re_bore 0.019, Oh 43) where the melt's
//  are ~1e-5 and ~6e3. The 2-D feed keeps the 3-D bore's mean velocity. R = 8
//  cells across the bore half-width: 320 x 271, dx 25 um, dt 1.43e-5 s.
//
//  THE REAL FEED IS 0.0208 mm^3/s (-feed 0.0208). Table 1 prints 0.255 mm^3/s,
//  and Murat Kilic confirmed on 2026-11-03 that the value was never updated:
//  every experiment, the fan's included, ran at 0.0208 mm^3/s and 35 mm/s. It
//  is neither earlier reading (12.3x below the printed one, 4.9x above
//  0.255 mm^3/min). Bore velocity 0.166 mm/s, Ca_feed 2.76, Re_bore 0.0015.
//  Measured 2026-11-03/04, FP64, to 120 s; 25 um cells cost 90 s of wall clock
//  per simulated second field on and 67 field off, 50 um cells (-R 4
//  -Ucol 0.04) 6.0 and 4.6.
//
//      cells, field        what the melt does
//      25 um, field off    a drop hangs from the nozzle and grows at the feed;
//                          its lowest point falls at 40 um/s (40-100 s) and
//                          first touches the collector at 104.968 s, 22x
//                          later than at the printed feed
//      50 um, field off    the same drop, 42 um/s; first contact 110.325 s
//      25 um, 6.5 kV       FASTER, THEN SLOWER: 0.40 mm below the field-off
//                          drop at 20 s, 34 um/s over 40-100 s, and still
//                          0.0875 mm up when the run ends at 120 s (row 1 at
//                          about 122.7 s, extrapolated): about 18 s late
//      50 um, 6.5 kV       the same: 0.45 mm ahead at 20 s, 0.075 mm up at
//                          120 s (about 121.4 s), about 11 s late
//
//  Melt within 1 mm of the tip 0.690 / 0.725 mm^2 at 25 um (off / on, 30-100 s)
//  and 0.855 / 0.867 at 50 um, about what the printed feed piles there; the
//  neck is the bore plus four cells again (0.50 / 0.60 mm); peak pull 1.16 and
//  0.66 gamma/R; ledger within +0.24 % to contact. At 100 s the drop is
//  1.90 mm wide with the field off and 2.05-2.10 mm with it on. A sheet carrying
//  this feed at 35 mm/s would be 1.9 um thick in 2-D, below any cell, so
//  nothing carries melt away and the drop grows at the feed. Filling a drop of
//  the same outline from the same bore velocity takes w/2R longer in 3-D than
//  in 2-D (w its width), four to five times here, so a round nozzle would take
//  longer to bridge the gap: an estimate, not a run.
//
//  AT THE REAL FEED THE FIELD'S EFFECT IS SET BY -lc, AND BY AN EDGE -lc MAKES
//  (measured 2026-11-04, 50 um, -lc 1 and 3 run to 140 s). The melt conducts
//  only within -lc of the tip -- FsElectric's guard against shorting the gap --
//  and below that it is vacuum to the potential and feels no pull. The
//  field-on drop slows EXACTLY when its lowest point leaves the conducting
//  zone: 90 -> 34 um/s on crossing 3.5 mm at the default 2 mm (25 um:
//  107 -> 40), where the field-off drop passes the same height at 49 -> 47.
//  Moving the cut moves the bend: 93 -> 53 um/s at 4.5 mm for -lc 1 (field off
//  75 -> 57 there) and 120 -> 16 at 2.5 mm for -lc 3. The three runs are
//  IDENTICAL to every printed digit until the lowest point first passes the
//  shallower cut (to 9.072 and 20.304 s), and first contact comes at 117.712 /
//  ~121.4 / 127.296 s for -lc 1 / 2 / 3, a delay of 7.4 / 11 / 17.0 s that
//  grows with the depth. WHY: while the whole drop conducts, its strongest pull
//  is at its lowest point (3.9x the median over the conducting surface); once
//  the drop hangs below the cut, the conductor ends in a flat face inside it,
//  and the strongest pull sits at that face's edge on the drop's two sides,
//  7-56x the median. A sideways pull that widens the drop is consistent with
//  the slowdown and the wider drop; that link is not tested. So the field-on
//  approach here is shaped by a numerical edge: neither the size nor the sign
//  of its delay is a prediction. A melt whose charge relaxes in less than the
//  drop's minute of growth would be conducting all the way down, which is the
//  run to make next (-lc close to the gap); FsElectric has no leaky dielectric.
//
//  THE TWO EARLIER READINGS OF TABLE 1, measured 2026-11-03, FP64, 4 threads
//  (about 80 s of wall clock per simulated second at 25 um; run it with
//  taskpolicy -B, CLAUDE.md's machine note). Viscosity 500 Pa s and surface
//  tension 0.03 N/m are nominal.
//
//      feed, voltage            what the melt does
//      0.255 mm^3/s, 6.5 kV     a column on the axis (mirror projection, below);
//        (Table 1 as printed)   front at 1.127 mm/s, melt within 1 mm of the tip
//                               level at 0.647 mm^2, first contact 4.690 s,
//                               peak pull 1.19 gamma/R before it; after it the
//                               rope leaves the collector and STAYS UP (below)
//      the same, field OFF      front 1.130 mm/s, 0.647 mm^2, contact 4.682 s;
//                               after it the rope BOUNCES -- off the collector
//                               within 0.1 s, 0.64 mm up at 5.02 s -- sags back,
//                               touches again at 5.87 s near the box's
//                               downstream edge and is laid there: 4.83 mm^2 by
//                               7.26 s, when the drawn thread's thinnest row
//                               holds 0.80 cells of melt; it then breaks and
//                               nothing more is laid. THE BOUNCE IS THIS GRID'S:
//                               at 12.5 um the field-off rope rises like the
//                               field-on one (AFTER CONTACT, below)
//      0.255 mm^3/MIN, 6.5 kV   a meniscus that grows at 0.01354 mm^2/s, the
//                               feed's own 2 R U = 0.01353: 0.108 mm^2 at 8 s,
//                               0.32 mm below the tip face; peak pull 0.68 gamma/R
//      0.255 mm^3/min, 12 kV    the same growth to every printed digit, 0.35 mm
//                               below the tip; peak pull 2.51 gamma/R
//
//  Ledger, total + drained - injected over injected: +0.50..0.57 % before
//  contact at the printed feed (the piston injects rho U, and rho sits that far
//  above 1 under the bore's viscous pressure drop), +0.01..0.1 % at the
//  per-minute feed and +0.03..0.24 % at the real one. WHILE THE FIELD-OFF
//  THREAD IS DRAWN AT THE PRINTED FEED IT CLIMBS TO +5.4 %
//  (7.21 s): melt is created as the thread thins. Not diagnosed; the first
//  suspect is the free surface's own settle() defect, which its banner measures
//  creating mass where a filling cell sits beside an emptying one.
//
//  NOT GRID-CONVERGED, AND THE NOZZLE IS WHY. Halving the cell from 50 to 25 um
//  (R = 4 -> 8) first moved the front from 0.89 to 1.13 mm/s and first contact
//  from 5.87 to 4.69 s, and after contact that coarse grid showed no effect of
//  the field at all (both ropes rise 1.83 mm). The ladder at fixed physics
//  (Re_coll 0.32: -Ucol 0.04 / default / 0.01 at R = 4 / 8 / 16, field on,
//  windows ending at 3.5 s) gives a front of 0.913 / 1.137 / 1.342 mm/s, melt
//  within 1 mm of the tip of 0.808 / 0.648 / 0.550 mm^2 and first contact at
//  5.84 / 4.69 / 3.98 s: apparent orders 0.13, 0.71 and 0.68, so no
//  extrapolation means anything. THE NECK SETS ALL THREE. The melt in the first
//  row under the tip face spans EXACTLY the bore plus four cells on every grid
//  (12 / 20 / 36 cells, 0.60 / 0.50 / 0.45 mm), and the column below is a plug
//  carrying the feed (width 2 mm down x front speed = 0.82 / 0.80 / 0.81 mm^2/s
//  against 0.81), so a narrower neck makes a narrower, faster column and an
//  earlier contact. Where the melt let go of the nozzle was the grid's choice,
//  because the free surface read the nozzle as gas -- a non-wetting wall -- and
//  refinement converges at first order to a neck of exactly the bore.
//  -wet DEG gives the nozzle a contact angle (FreeSurfaceSolver::
//  set_contact_angle, 2026-11-03). At R = 4, 60 degrees widened the neck from 12
//  to 14 cells by 2.9 s and no further by 4.3 s, and 120 degrees changed
//  nothing measurable. The melt spreads on the face only as fast as its contact
//  line moves, which here is numerical slip at a no-slip wall (gamma/mu is
//  0.06 mm/s against a 2 mm/s bore flow), and validation/sessile_drop measures a
//  contact-line window that WIDENS with resolution -- so the law does not by
//  itself make this column converge. Before contact the field changes nothing
//  on any grid (front, tip melt and contact time agree within 1.4 %); quote no
//  absolute number here as converged, and no difference after contact either
//  (AFTER CONTACT, below).
//
//  WHAT IT SAYS. At the real feed the bore's capillary number is 2.8, and with
//  nothing to carry it away the melt hangs from the nozzle as a drop that grows
//  at the feed and bridges the gap in about two minutes in 2-D (THE REAL FEED,
//  above). What the field does to that approach belongs to -lc. The fibres do
//  not carry the feed either: a straight fibre laid at 35 mm/s carrying
//  0.0208 mm^3/s is 27.5 um, so the paper's 19-22 um fibres carry 48-64 % of it
//  and the fan's 9.5-13.5 um ones 12-24 %, with no oozing reported in either:
//  the reported feed, speed and diameters do not balance for straight fibres,
//  a question for the experiment rather than for this model. At the feed
//  Table 1 prints, the bore's capillary number is 34:
//  the melt is extruded as a column whatever the field does, which in the
//  experiment would read as oozing. At 0.255 mm^3/min -- the reading under which
//  the fan-cooled fibres carry the feed -- the field shapes a meniscus that
//  grows at exactly the feed rate, because nothing in a planar model can carry
//  melt away: a sheet carrying it at collector speed would be ~0.4 um thick,
//  against 25 um cells and the experiment's 10-20 um fibres. A planar tip has
//  no conical equilibrium and a weaker field than a round one, which is why a
//  12 kV row is here: about the pull a round 1 mm tip feels at 6.5 kV
//  (2V/(r ln(4d/r)) ~ 6.9 kV/mm), an estimate.
//
//  THE SIDEWAYS DRIFT, AND WHAT REMOVED IT. Without help both printed-feed
//  columns leave the axis from ~1.5 s -- field on to -0.42 mm, field off to
//  +0.27 mm by 4.5 s -- in a set-up mirror-symmetric to the cell. At R = 4,
//  field off, the front's offset grew x1.9..3.4 every 0.5 s from 3.4 s while
//  the root stayed on the axis to 0.002 mm, so it is the advancing front that
//  amplifies round-off; what amplifies it is NOT diagnosed.
//  FreeSurfaceSolver::normal_reconstruction does not cure it: it removed the
//  drift at R = 4 and made it worse at R = 8 (-0.51 mm by 4 s against -0.35).
//  What does: until the melt first reaches the cell beside the collector the
//  state is projected onto its mirror-symmetric part every M steps
//  (mirror_symmetrize_x), and both fronts then stay on the axis, +0.000 mm, to
//  contact; there the moving collector breaks the symmetry for real and the
//  projection stops. -nosym brings the drift back: field on, the foot leaves
//  the axis at ~1.5 s and is at -0.41 mm by 4.47 s.
//
//  AFTER CONTACT BOTH ROPES LEAVE THE COLLECTOR, within 0.1 s of touching it,
//  field on or off, AND WHETHER THEY COME BACK IS THE GRID'S CHOICE, NOT THE
//  FIELD'S. Measured on the melt still attached to the nozzle, its lowest point
//  against time ("sticks" was first read off the laid area, which starts only
//  at the SECOND touch; the frames show the bounce). At 25 um the two runs
//  parted company: field on, the rope curled into a hook 2.31 mm up by 6.0 s,
//  still 0.89 mm up at 8.95 s; field off, it bounced 0.64 mm and came back
//  1.19 s after contact, to be laid near the box's downstream edge -- where the
//  drain takes what reaches the collector, so the BOX set the lag -- and drawn
//  below a cell until it broke. That was read as the field keeping the rope up,
//  and a -lc test chased the mechanism (cut 1, 2 and 3 mm below the tip: hooks
//  2.19 / 2.31 / 2.06 mm up). THE OTHER TWO GRIDS SAY OTHERWISE (the ladder at
//  fixed Re_coll 0.32, 2026-11-03). At 12.5 um both ropes rise and sink alike:
//  2.29 and 2.38 mm at their highest, field on and off, 0.93 and 1.08 s after
//  contact, then 1.32 and 1.53 mm up 2.6 s after it, neither back. At 50 um
//  both rise too, 1.93 and 1.68 mm when the runs end 0.76 s after contact. So
//  the 25 um field-off bounce is the odd one out, and nothing after contact that
//  this model resolves is decided by the field. A DIFFERENCE BETWEEN TWO RUNS
//  AT ONE RESOLUTION IS NOT GRID-INDEPENDENT EITHER: this banner's own advice
//  was to compare field on against off at one grid, and the contrast it
//  compared belonged to the grid.
//   - The first version's field-off rope never came back at all: its drain
//     removed melt at the side walls at EVERY height and ate the dragged rope
//     from the bottom up. A 90-degree wetting mirror was tried first and changed
//     nothing; the drain now acts on the collector only.
//   - One interface cell (fill 0.54) is left on the collector at contact and
//     stays there, unmoved, to the end: classify()'s detached-speck rule spares
//     a cell touching a wall, and with no liquid neighbour it exchanges no mass.
//     At the real feed (25 um, field off) it has fill 0.985 and sits 0.44 mm
//     downstream of the axis, inside the window z_low scans, so z_low reads
//     0.0125 mm from contact to the end while the drop hangs a millimetre up.
//     After contact, read the frames, not z_low.
//   - AT THE REAL FEED NOTHING AFTER CONTACT IS USABLE. The drop that touches
//     holds 7.4-7.8 mm^2 of melt, and the collector drags it to the box's
//     downstream wall within 2 s. At 50 um the neck snaps, leaving a small drop
//     on the nozzle, and 3.3 mm^2 (41 % of what was fed) drains out by 112 s,
//     the ledger peaking at +1.3 % and settling at +0.6 %; at 25 um the drop
//     lifts off the collector and hangs from the nozzle as a hook against that
//     wall, and nothing drains. The grids disagree and the box decides both.
//   - The drawn sheet of the real process, Q/U_c = 23 um in 2-D, is below the
//     25 um grid, which is why the field-off thread there broke once drawn.
//
//  FIXED ON THE WAY, so a rerun does not repeat them: the nozzle is centred,
//  because 2 mm from one mirror wall and 4 mm from the other the image nozzles
//  bent the column toward the far side before it touched anything; the feed
//  piston spans the lips' top row, because over the bore alone its edge cells
//  drew a diagonal link from a static wall and the ledger read -3.4 % at R = 4;
//  the drain acts on the collector only (above).
//
//  WHAT IT DOES NOT MODEL. The jet; wetting and contact lines; free charge,
//  conduction or polarisation of the melt (a leaky dielectric); heat (the melt
//  is isothermal at 120 C -- the hot end is demonstrator/mew_liquefier);
//  shear-thinning and viscoelasticity; the axisymmetric geometry a real cone
//  needs. tools/plot_mew_nozzle.py and tools/anim_mew_nozzle.py draw the frames.
//==============================================================================
#include "FsElectric.hpp"
#include "NpyDump.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using PL = D2Q5;
using FS = FreeSurfaceSolver<L>;

namespace {

constexpr double kPi  = 3.14159265358979323846;
constexpr double kEps0 = 8.8541878128e-12;

//------------------------------------------------------------------------------
// The physical set-up. Table 1 gives the voltage, the gap, the nozzle
// temperature (through mu), the feed and the collector speed; everything else
// is NOMINAL and printed with every run.
//------------------------------------------------------------------------------
struct Physical {
  double R     = 0.2e-3;    // nozzle bore half-width: the 0.4 mm nozzle
  double lip   = 0.3e-3;    // nozzle wall at the tip, so the tip face is 1.0 mm (NOMINAL)
  double gap   = 5.5e-3;    // nozzle-collector distance, Table 1
  double up    = 1.25e-3;   // nozzle length inside the box above its tip
  // THE NOZZLE IS CENTRED, AND THAT IS A MEASURED CHOICE. The sides are zero-
  // field mirrors, so each puts an image nozzle at 6.5 kV beside the real one.
  // With the nozzle 2 mm from one wall and 4 mm from the other (a first version,
  // to give the deposit room) the field was stronger on the far side and bent
  // the extruded column toward it before it touched the collector.
  double W     = 8.0e-3;    // box width
  double xnoz  = 4.0e-3;    // nozzle axis from the left wall; the collector moves +x
  double mu    = 500.0;     // PCL at 120 C [Pa s] (NOMINAL; Fig. 1 is normalised)
  double gam   = 0.030;     // surface tension [N/m] (NOMINAL)
  double rho   = 1080.0;    // [kg/m^3]
  double g     = 9.81;
  double V     = 6.5e3;     // Table 1
  double Q     = 0.255e-9;  // feed [m^3/s], Table 1 AS PRINTED; see -feed
  double Uc    = 0.035;     // collector, 2100 mm/min, Table 1
  double lc    = 2.0e-3;    // liquid within this of the tip conducts (FsElectric)

  // The 2-D feed keeps the 3-D nozzle's MEAN VELOCITY, U = Q / (pi R^2): that
  // is what sets the viscous stress in the bore and so the capillary number
  // of the extrusion. The flux per unit depth, 2 R U, is then a consequence.
  double Un()  const { return Q / (kPi * R * R); }
  double Ca_n() const { return mu * Un() / gam; }
  double Ca_c() const { return mu * Uc / gam; }
  double Bo()   const { return rho * g * R * R / gam; }
  double LamE() const { return kEps0 * V * V / (gam * R); }   // eps0 V^2 / (gamma R)
  double tvc()  const { return mu * R / gam; }                 // viscous-capillary time
};

struct Opts {
  int    R      = 8;       // cells across the bore half-width
  double nu     = 0.5;     // lattice kinematic viscosity (tau = 2)
  double Ucol   = 0.02;    // collector speed in lattice units
  double tauphi = 10.0;    // potential solver rate (TRT, Lambda = 1/4)
  // The melt surface moves ~0.05 cells between updates at the as-printed feed,
  // and the host-side refresh is a third of a step's cost at 20.
  int    M      = 40;      // fluid steps per field update
  double t_end  = 10.0;    // [s]
  double t_out  = 0.05;    // series sampling [s]
  double t_frame = 1.0;    // frame interval [s]
  bool   field  = true;
  bool   nrec   = false;   // FreeSurfaceSolver::normal_reconstruction
  bool   sym    = true;    // mirror-project until first contact; -nosym to see the drift
  // -wet DEG: the melt's contact angle on the NOZZLE (bore walls and tip face),
  // FreeSurfaceSolver::set_contact_angle through wet_cos_of. Off by default: the
  // nozzle then reads as gas in the curvature, a non-wetting wall. The collector
  // and the feed piston keep reading as gas either way.
  double wet    = -1;
  std::string out;
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Physical P;
    Opts o;
    std::string feed = "s";
    for (int i = 1; i < argc; ++i) {
      auto num = [&]() {
        if (i + 1 >= argc) { std::printf("mew_nozzle: %s needs a value\n", argv[i]); std::exit(2); }
        return std::atof(argv[++i]);
      };
      const char* a = argv[i];
      if      (!std::strcmp(a, "-R"))      o.R = int(num());
      else if (!std::strcmp(a, "-nu"))     o.nu = num();
      else if (!std::strcmp(a, "-Ucol"))   o.Ucol = num();
      else if (!std::strcmp(a, "-tend"))   o.t_end = num();
      else if (!std::strcmp(a, "-tframe")) o.t_frame = num();
      else if (!std::strcmp(a, "-mu"))     P.mu = num();
      else if (!std::strcmp(a, "-V"))      P.V = num() * 1e3;          // kV
      else if (!std::strcmp(a, "-lc"))     P.lc = num() * 1e-3;        // mm
      else if (!std::strcmp(a, "-Uc"))     P.Uc = num() * 1e-3;        // mm/s
      else if (!std::strcmp(a, "-nofield")) o.field = false;
      else if (!std::strcmp(a, "-nrec"))   o.nrec = true;
      else if (!std::strcmp(a, "-nosym"))  o.sym = false;
      else if (!std::strcmp(a, "-wet"))    o.wet = num();
      else if (!std::strcmp(a, "-feed") && i + 1 < argc) feed = argv[++i];
      else if (!std::strcmp(a, "-out") && i + 1 < argc)  o.out = argv[++i];
      else {
        std::printf("mew_nozzle: unknown option '%s'\n", a);
        Kokkos::finalize();
        return 2;
      }
    }
    // -feed s: 0.255 mm^3/s as Table 1 prints it. -feed min: 0.255 mm^3/MIN,
    // the reading under which the fan-cooled fibres carry the feed. A number
    // is taken as mm^3/s.
    if (feed == "s")        P.Q = 0.255e-9;
    else if (feed == "min") P.Q = 0.255e-9 / 60.0;
    else                    P.Q = std::atof(feed.c_str()) * 1e-9;

    //---- lattice units ---------------------------------------------------------
    // The collector's capillary number is the largest velocity group, so it
    // fixes the capillary velocity gamma/mu in lattice units; everything else
    // follows from matching Ca, Bo and Lambda_E. Re and Oh are NOT matched --
    // the real flow is creeping (Re ~ 1e-5) and the lattice keeps it so.
    const int    R    = o.R;
    const double dx   = P.R / R;
    const double ucap = o.Ucol / P.Ca_c();          // gamma/mu, lattice
    const double mu_l = o.nu;                       // rho = 1
    const double gam_l = ucap * mu_l;
    const double Un_l = P.Ca_n() * ucap;
    const double g_l  = P.Bo() * gam_l / double(R * R);
    const double C_l  = 0.5 * P.LamE() * gam_l * double(R);
    const double tvc_l = double(R) / ucap;          // steps
    const double dt   = P.tvc() / tvc_l;            // s per step
    const double Re_c = o.Ucol * R / o.nu, Re_n = Un_l * R / o.nu;
    const double Oh_l = std::sqrt(mu_l * mu_l / (gam_l * R));

    const Index nx   = Index(std::lround(P.W / dx));
    const Index gapc = Index(std::lround(P.gap / dx));
    const Index ytip = 1 + gapc;                    // first lip row; face at ytip - 1/2
    const Index ny   = ytip + Index(std::lround(P.up / dx));
    const Index xc   = Index(std::lround(P.xnoz / dx));
    const Index lipc = Index(std::lround(P.lip / dx));
    const Index lcc  = Index(std::lround(P.lc / dx));
    const Index tipz = Index(std::lround(1.0e-3 / dx));   // the "tip zone": 1 mm

    std::printf("mew_nozzle: D2Q9 free surface + D2Q5 TRT potential, precision %s\n",
                sizeof(Real) == 4 ? "float" : "double");
    std::printf("  PCL (nominal): mu %.0f Pa s  gamma %.3f N/m  rho %.0f   field %s\n",
                P.mu, P.gam, P.rho, o.field ? "ON" : "OFF");
    std::printf("  set-up: bore 0.4 mm, tip face %.1f mm, gap %.1f mm, %.1f kV, collector %.0f mm/s,"
                " feed %.4g mm^3/s (-feed %s)\n", 2e3 * (P.R + P.lip), P.gap * 1e3, P.V * 1e-3,
                P.Uc * 1e3, P.Q * 1e9, feed.c_str());
    std::printf("  groups: Ca_feed %.3g  Ca_coll %.3g  Bo %.3g  Lambda_E %.3g  (U_bore %.4g mm/s,"
                " t_vc %.3g s)\n", P.Ca_n(), P.Ca_c(), P.Bo(), P.LamE(), P.Un() * 1e3, P.tvc());
    std::printf("  lattice: %lld x %lld, dx %.1f um, dt %.3e s, t_vc %.3g steps, gamma %.3e,"
                " U_bore %.3e, U_coll %.3g, g %.2e, C_E %.3e\n", (long long)nx, (long long)ny,
                dx * 1e6, dt, tvc_l, gam_l, Un_l, o.Ucol, g_l, C_l);
    std::printf("           Oh %.1f  Re_coll %.3f  Re_bore %.4f  Ma_coll %.3f   conducting %.1f mm"
                " below the tip\n", Oh_l, Re_c, Re_n, o.Ucol / std::sqrt(1.0 / 3.0), P.lc * 1e3);

    //---- geometry --------------------------------------------------------------
    Domain d(nx, ny, 1, false, false, true);
    auto is_lip   = [=](Index x, Index y) {
      return y >= ytip && ((x >= xc - R - lipc && x < xc - R) || (x >= xc + R && x < xc + R + lipc));
    };
    // The feed piston spans the bore AND the lips' top row. Over the bore alone,
    // each edge cell drew its diagonal feed link from a static lip cell and lost
    // a sixth of its injection: the ledger read -3.4 % at R = 4, 2 x (1/6)/(2R).
    auto is_piston = [=](Index x, Index y) {
      return y == ny - 1 && x >= xc - R - lipc && x < xc + R + lipc;
    };

    FS fs(d);
    fs.coll.omega = FS::omega_from_viscosity(Real(o.nu));
    fs.set_gravity(Real(0), Real(-g_l));
    fs.set_surface_tension(Real(gam_l));
    fs.set_normal_reconstruction(o.nrec);
    fs.set_geometry([&](Index x, Index y, Index) -> FsCell {
      if (y == 0 || is_lip(x, y) || is_piston(x, y)) return FsSolid;
      return FsGas;
    });
    if (o.wet > 0) {
      // cos(theta) on the lip cells; 2 (outside [-1, 1]) everywhere else, which
      // leaves the collector and the piston reading as gas, as without -wet.
      View1D<Real> wc("mew_wet_cos", d.n_padded);
      auto hwc = Kokkos::create_mirror_view(wc);
      Kokkos::deep_copy(hwc, Real(2));
      const Real cw = Real(std::cos(o.wet * M_PI / 180.0));
      for (Index y = 0; y < ny; ++y)
        for (Index x = 0; x < nx; ++x)
          if (is_lip(x, y)) hwc(d.id(x, y)) = cw;
      Kokkos::deep_copy(wc, hwc);
      fs.wet_cos_of = wc;
      fs.set_contact_angle(Real(o.wet * M_PI / 180.0));
      std::printf("  nozzle contact angle %.0f deg (FreeSurfaceSolver::set_contact_angle)\n", o.wet);
    }
    const Domain dd = d;
    const Index hx = d.hx, hy = d.hy;
    fs.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Index x = px - hx, y = py - hy;
      const bool slot = y >= ytip && x >= xc - R && x < xc + R && y < ny - 1;
      return typename FS::Seed{slot ? Real(1) : Real(0), Real(1)};
    });
    // The collector and the feed piston are static walls with a velocity: the
    // moving-wall bounce-back of the obstacle path. Tangential at the
    // collector, it drags; NORMAL at the piston, it injects rho U per cell per
    // step -- a feed controlled by rate, as a filament drive is.
    const Real ucol = Real(o.Ucol), uin = Real(Un_l);
    fs.move_obstacle(
        KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; dd.coords(n, px, py, pz);
          const Index x = px - hx, y = py - hy;
          return y == 0 || (y == ny - 1 && x >= xc - R - lipc && x < xc + R + lipc);
        },
        KOKKOS_LAMBDA(Index n, Real u[3]) {
          Index px, py, pz; dd.coords(n, px, py, pz);
          const Index y = py - hy;
          u[0] = (y == 0) ? ucol : Real(0);
          u[1] = (y == 0) ? Real(0) : -uin;
          u[2] = Real(0);
        },
        Real(0), Real(0));

    FsElectric<PL> el(d, Real(o.tauphi));
    el.coef = o.field ? Real(C_l) : Real(0);
    el.set_roles(
        [&](Index x, Index y) -> ElecRole {
          if (y == 0 || is_lip(x, y) || is_piston(x, y)) return ElecElectrode;
          if (x == 0 || x == nx - 1 || y == ny - 1) return ElecFrame;
          return ElecFree;
        },
        [&](Index, Index y) -> Real { return y == 0 ? Real(0) : Real(1); });
    el.set_conduct_region([&](Index, Index y) { return y >= ytip - lcc; });
    el.refresh(fs);
    el.initialize(Real(0.5));
    el.relax(20000);
    el.apply(fs);

    //---- diagnostics -----------------------------------------------------------
    const double a2 = dx * dx * 1e6;                 // cell area in mm^2
    const double m0 = double(fs.total_mass());
    struct Diag { double t, total, tip, gap, zlow, width, drained, ledger, pe; };
    auto measure = [&](double t) {
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      Diag D{};
      D.t = t;
      Index ylow = ny;
      Index xmin = nx, xmax = -1;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 0; x < nx; ++x) {
          const std::uint8_t f = hf(d.id(x, y, 0));
          const double v = (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(d.id(x, y, 0))) : 0.0);
          if (v <= 0) continue;
          D.total += v;
          if (y < ytip) {
            D.gap += v;
            if (y >= ytip - tipz) D.tip += v;
            if (std::abs(x - xc) <= 3 * R + lipc && y < ylow) ylow = y;
            if (y == ytip - 1 && v > 0.5) { xmin = std::min(xmin, x); xmax = std::max(xmax, x); }
          }
        }
      D.zlow  = (ylow < ny) ? (double(ylow) - 0.5) * dx * 1e3 : P.gap * 1e3;
      D.width = (xmax >= xmin) ? double(xmax - xmin + 1) * dx * 1e3 : 0.0;
      const double drained = double(fs.drained_mass());
      const double injected = double(uin) * 2.0 * R * (t / dt);
      D.drained = drained * a2;
      D.ledger = (double(fs.total_mass()) + drained - m0 - injected) / std::max(1.0, injected);
      D.total *= a2; D.tip *= a2; D.gap *= a2;
      D.pe = el.max_stress() / (gam_l / R);          // in units of gamma/R
      return D;
    };
    auto frame = [&](double t) {
      if (o.out.empty()) return;
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      std::vector<float> fl(std::size_t(nx * ny)), ph(std::size_t(nx * ny));
      for (Index y = 0; y < ny; ++y)
        for (Index x = 0; x < nx; ++x) {
          const std::uint8_t f = hf(d.id(x, y, 0));
          const std::size_t k = std::size_t(y * nx + x);
          fl[k] = (f == FsSolid) ? -1.0f : (f == FsFluid ? 1.0f : (f == FsInterface ? float(he(d.id(x, y, 0))) : 0.0f));
          ph[k] = float(el.phi(x, y));
        }
      char nm[64];
      std::snprintf(nm, sizeof nm, "_t%06.2f", t);
      write_npy(o.out + nm + "_fill.npy", fl, {std::size_t(ny), std::size_t(nx)});
      write_npy(o.out + nm + "_phi.npy", ph, {std::size_t(ny), std::size_t(nx)});
    };

    FILE* fs_out = nullptr;
    if (!o.out.empty()) {
      fs_out = std::fopen((o.out + "_series.csv").c_str(), "w");
      if (fs_out) std::fprintf(fs_out, "t_s,melt_total_mm2,tip_zone_mm2,gap_mm2,z_low_mm,"
                               "width_at_tip_mm,drained_mm2,ledger,max_pe_over_gamma_R\n");
    }
    std::printf("\n  %8s %10s %10s %10s %8s %8s %10s %9s %9s\n", "t [s]", "melt mm^2",
                "tip mm^2", "gap mm^2", "z_low", "width", "laid mm^2", "ledger", "pE/(g/R)");

    const long nsteps = long(o.t_end / dt);
    const long sout = std::max<long>(o.M, long(o.t_out / dt) / o.M * o.M);
    const long sframe = std::max<long>(sout, long(o.t_frame / dt) / sout * sout);
    frame(0.0);
    bool contacted = false;
    for (long t = 0; t < nsteps; t += o.M) {
      for (int j = 0; j < o.M; ++j) fs.step();
      // Melt the collector carries to a side wall leaves the box there -- ON THE
      // COLLECTOR ONLY. Draining at every height ate a dragged rope from the
      // bottom up, and the rope, taut between the nozzle and its drained end,
      // swung up off the collector: the "no sticking" of the first version.
      const Index ydr = Index(std::lround(0.25e-3 / dx));
      fs.drain(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; dd.coords(n, px, py, pz);
        const Index x = px - hx, y = py - hy;
        return (x < 6 || x >= nx - 6) && y <= ydr;
      });
      // THE SET-UP IS MIRROR-SYMMETRIC UNTIL THE MELT TOUCHES THE COLLECTOR, and
      // the solver turns round-off into a sideways drift of the column (see
      // FreeSurfaceSolver::mirror_symmetrize_x). Project onto the symmetric part
      // until first contact; from then on the moving collector breaks the
      // symmetry for real and the projection must stop.
      if (o.sym && !contacted) {
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
        for (Index x = 0; x < nx && !contacted; ++x) {
          const std::uint8_t f = hf(d.id(x, 1, 0));
          contacted = (f == FsFluid || f == FsInterface);
        }
        if (!contacted) fs.mirror_symmetrize_x(nx - 1);
        else std::printf("  first contact with the collector at t = %.3f s:"
                         " mirror projection off\n", (t + o.M) * dt);
      }
      if (o.field) {              // with the field off, rho_G_of stays uniform
        el.refresh(fs);
        el.relax(o.M);
        el.apply(fs);
      }
      const long ts = t + o.M;
      if (ts % sout == 0) {
        const Diag D = measure(ts * dt);
        if (!std::isfinite(D.total)) { std::printf("  NOT FINITE at t = %.3f s\n", ts * dt); rc = 1; break; }
        if (fs_out) std::fprintf(fs_out, "%.5f,%.6f,%.6f,%.6f,%.5f,%.5f,%.6f,%.3e,%.4f\n", D.t,
                                 D.total, D.tip, D.gap, D.zlow, D.width, D.drained, D.ledger, D.pe);
        if (ts % sframe == 0) {
          std::printf("  %8.3f %10.4f %10.4f %10.4f %8.3f %8.3f %10.4f %9.1e %9.3f\n", D.t, D.total,
                      D.tip, D.gap, D.zlow, D.width, D.drained, D.ledger, D.pe);
          std::fflush(stdout);
          frame(ts * dt);
        }
      }
    }
    if (fs_out) std::fclose(fs_out);
  }
  Kokkos::finalize();
  return rc;
}
