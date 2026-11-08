//==============================================================================
//  MELT ELECTROWRITING IN r-z: the round nozzle of mew_cone3d at the cost of
//  mew_nozzle -- FreeSurfaceSolver's axisymmetric mode (validation/axisym_fs),
//  an r-z potential with an ohmic melt (validation/FsElectricAxi.hpp,
//  validation/axisym_field) and, with -heat, the melt's temperature, its
//  Arrhenius viscosity and its freezing (validation/FsHeatAxi.hpp,
//  validation/axisym_heat).
//
//  THE SET-UP is mew_cone3d's, axisymmetric: a 0.4 mm bore with a 0.3 mm wall
//  at the tip (tip face 1.0 mm across), 5.5 mm above a grounded collector, the
//  nozzle at 6.5 kV, PCL at 820 Pa s (measured), 0.030 N/m (NOMINAL),
//  1080 kg/m^3, fed at 0.0208 mm^3/s by a piston over the bore. x is the radius
//  (the axis on the left face of column 0), y the height.
//
//  WHAT r-z BUYS OVER BOTH SIBLINGS. Against mew_nozzle: the azimuthal
//  curvature a cone needs (a planar wedge has none), the round tip's stronger
//  field, and a feed that matches in velocity AND volume at once. Against
//  mew_cone3d: 2-D cost, so the grid can be refined and the box widened, and an
//  exact bore -- the piston covers pi R^2 exactly in the r-weighted sense,
//  where the 3-D bore rounded to 52 cells for 50.27 and fed 3.45 % too much.
//
//  IT REPRODUCES THE 3-D RUN WHERE THE TWO OVERLAP (2026-11-06). With
//  mew_cone3d's settings -- equipotential melt cut at 0.5 mm above the
//  collector (-equip -lc 5), nu = 1/2, the default exchange, its measured feed
//  of 0.02159 mm^3/s -- in a cylinder INSCRIBED in its 4 x 4 mm box (-W 2), the
//  lowest point agrees to one cell up to 12 s (two at 15 s) and reaches the
//  cut (0.5 mm) at 31.06 s against 32.39 s; both then park on the cut. In the cylinder of the
//  box's AREA (-W 2.26) the column gets there at 27.18 s: the field at the tip
//  (1.37 / 2.18 against 1.54 gamma/R at 5 s) is what the box's shape moves.
//  The gap volume agrees to 0.09 % over 1-32 s in both boxes.
//
//  WHAT IT CANNOT DO. The collector cannot move: a translating plate is not
//  axisymmetric. Up to first contact that costs nothing (the collector touches
//  only the gas, which this free surface does not resolve); after it, the melt
//  lands on a STATIC plate, which is not the experiment. Read it to contact.
//  No jet thinner than about two cells is representable: an MEW jet carrying
//  this feed at 35 mm/s is 27.5 um across, half a cell at 50 um.
//
//  THE OHMIC MELT (on; -equip for the equipotential drop). Before contact it
//  IS the whole drop at the nozzle's potential (FsElectricAxi's banner: the
//  same field to round-off), so -lc is not needed: there is no conductor edge.
//  -lc MM restores mew_nozzle's cut for comparison.
//
//  THE BOX AND THE PRINTHEAD. The potential's side (r = W) and top are
//  zero-normal-field frames by default: the side is a mirror, and the top makes
//  the nozzle look infinitely long. -plate holds the whole top at the nozzle's
//  potential, -plateR MM a disc of that radius (a printhead's heater block, at
//  the nozzle's potential in the experiment) with a frame beyond it. The pull at
//  the tip at t = 0.05 s, in gamma/R: 0.835 in the default box (-W 4 -up 1.25),
//  1.031 for a long open needle (-W 16 -up 10), 1.124 at -W 16 -up 5, 0.559
//  under a printhead disc (-W 16 -up 5 -plateR 10) and 0.505 under a whole
//  plate. A narrow box weakens it further: 0.24 at -W 2. In the 8 mm box the
//  bracket runs use: 0.763 for the needle (-up 10), 0.497 for a disc of 8 mm
//  (-up 5 -plateR 8) and 1.351 for the default 1.25 mm stub, whose zero-field
//  top sits close above the tip. So the electrodes around the nozzle are worth
//  up to a factor of two at the tip, and the experiment's printhead -- not
//  modelled in any sibling -- sits somewhere inside that range.
//
//  THE LATTICE VISCOSITY is 1/6 (tau = 1). A capillary scaling makes dt
//  independent of it, so it is free in principle; in practice the axis bounds
//  it (FreeSurfaceSolver's member axisymmetric): at nu = 1/2 the column, two
//  cells thick, came apart 0.02 s before contact until the axis column's
//  viscous term was made implicit. With it, contact comes at 15.448 / 15.691 /
//  15.803 s at nu = 1/2 / 1/4 / 1/6 (isothermal, default box), a quadratic
//  through which gives 16.07 s as nu -> 0: a 1.7 % lattice error at the default.
//
//  HEAT (-heat). The melt carries a temperature (FsHeatAxi): the nozzle, its
//  wall and the piston's melt at 120 C, the collector at 22 C, the air at 25 C,
//  a surface loss h (90 W/m^2/K NOMINAL: natural convection and radiation off a
//  ~1 mm drop and a ~0.2 mm thread in still air are both ~90), k = 0.20 W/m/K
//  and c_p = 2000 J/kg/K (NOMINAL). The viscosity follows Arrhenius with
//  Ea = 31.3 kJ/mol (PCL over 80-120 C) up to the axis bound, nu = 1, i.e.
//  sixfold, reached at 58 C (-nu 0.0417, i.e. 1/24, lifts the cap to x24,
//  reached at 22 C: below the air, so none; the sensitivity runs use it). Below
//  a band of 30-40 C the melt is held by an
//  implicit drag, a Darcy number Da = A R^2 / mu = 100 on the bore radius
//  (A = 1.04 at 50 um): a crystallising melt, with no latent heat. -h, -Ea,
//  -Tair, -Tcol, -Tnoz, -Tfreeze, -nucap, -Dafr and -nofreeze move them.
//
//  THE FEED IS A VOLUME RATE: the piston moves at U / rho_piston (see
//  rho_piston below), which removes the +0.34 % its compressed melt injected;
//  -rawfeed keeps the siblings' piston.
//
//  THE JET (-jet Q, mm^3/s; added 2026-11-07). The jet itself is below the grid
//  (WHAT IT CANNOT DO), so -jet takes out of the drop the melt it would carry
//  instead: Q drawn through the density of the Fluid cells within -jetr (0.1 mm)
//  of the axis, in a box 2 jetr tall starting -jetdepth (four cells) above the
//  melt's lowest point there -- the root of a cone's apex -- by
//  FreeSurfaceSolver::draw(), after every step while the field is on and until
//  contact, and with -jetTmin C only while the melt at the apex is at least that
//  warm (a frozen tip does not jet). Q is the fibres' flow, which nothing here
//  predicts: a straight fibre d across laid at 35 mm/s carries pi d^2 v / 4,
//  0.01155 mm^3/s for the 20.5 um fibres without the fan (56 % of the feed) and
//  0.00364 for the fan's 11.5 um (17 %). What leaves is the series' jet_mm3
//  (what it asked for, jet_asked_mm3; less only where the box holds too little,
//  draw() taking at most a fifth of it a step), and is in drained_mm3 with the
//  rim's, so the ledger closes but for what the free surface itself creates
//  (below). The momentum it carries is not removed (negligible at MEW speeds).
//  demonstrator/mew_jet is the jet itself.
//  draw() lowers the pressure in the box, so the melt around flows to it. The
//  first version took Q out of the SURFACE cells there (sink()), which leaves the
//  pressure alone, and it tunnelled up the axis: the cells within 0.1 mm of it
//  hollow to the nozzle by 10 s, the drop hanging below them, and 78 % of Q
//  taken, then almost none.
//  THE SUCTION THAT DRAWS THE MELT IN ALSO DIMPLES THE APEX, which a real jet
//  pulls OUT. At the no-fan flow the surface on the axis rises above the ring of
//  melt round it from about 8 s, by 2.1-2.6 cells over 16-24 s isothermal and
//  2.4-3.6 over 16-28 s with heat, and the box's lower edge is then within about
//  a cell of it, at times in it (read off the frames). AND WHERE THE SURFACE
//  NEARS THE DRAWN CELLS THE FREE SURFACE CREATES MASS: nothing until the gap
//  closes (the run's own 1e-6 mm^3), then 0.4 % of the jet by 20 s and 0.6 % by
//  24 s; then the column thins round the box as well, its sides next to the box
//  from 25 s and in its outer cells from 27 s, and it is 7.2 % by contact (5.6 %
//  of the melt), isothermal; 3.1 % by 60 s with heat. The fan's weaker jet
//  dimples later (from about 20 s with heat) and creates 1.2 % by 60 s;
//  isothermal it creates 1.3 %, all in the 1.9 s before contact while its column
//  runs down round the box. The same column with no jet runs to the collector
//  creating none (2e-6 mm^3): it is the draw, not the thin melt. Neither
//  neighbour of the default does better. A box one cell across
//  (-jetr 0.05) dimples sooner and creates 0.37 % of the jet by 10 s. One eight
//  cells deep (-jetdepth 0.4) leaves a cap below it that the field draws into a
//  thread one or two cells thick, pinching at the box: it creates six times as
//  much by 20-24 s, takes 95 % of what it asks, and touches at 28.47 s against
//  32.14 s, so the box's depth is worth 11 % of that contact. Before any dimple
//  (5 s) the box against the surface created 8.2 % of the jet, two cells in
//  2.5 %, four or six cells in 1e-5 to 5e-5; neither the field (-V 0: 7.9 %), the
//  piston's correction (-rawfeed: 8.4 %) nor the geometric exchange (-nogeom:
//  8.6 %) moved that, and in a resting r-z cylinder (validation/axisym_fs -sec J)
//  a box ON the axis but deep in the liquid costs 3e-4 of what is drawn, as one
//  off it does. The mechanism is NOT diagnosed. The depth is in cells because the
//  effect is the lattice's.
//
//  WHAT IT SAYS (2026-11-06; 50 um unless said; the series and logs are in
//  results/mew_axi/, named there; tools/plot_mew_axi.py draws them):
//   - ISOTHERMAL (the melt at 120 C throughout, as in both siblings), the field
//     draws the drop into a column that touches the collector at 15.80 s on
//     50 um cells and 9.80 s on 25 um. NOT CONVERGED: the finer grid resolves a
//     sharper tip -- the pull there reads 42 against 6 gamma/R at 9 s, and the
//     drop is at 5.0 mm by 3.18 against 3.83 s -- and nothing here caps a
//     perfect conductor's tip: no charge relaxation, and no corona although the
//     field at the drop, 9-45 kV/mm, is above air's onset. At 25 um the column
//     leaves the drop as a cone narrowing into a thread; at 50 um as a rod. No
//     isothermal contact time is a prediction.
//   - WITH HEAT (-heat) the drop's lower end is below 45 C from 18.2 s and 40 C
//     from 23.2 s (a 1 s median of the lowest axis cell), and the drop STALLS about
//     3.5 mm above the collector, widening instead: 3.53 mm up at 60 s and
//     3.38 mm at 150 s, 1.64 mm across and 1.34 mm^3 within 1 mm of the tip,
//     the melt below the tip at 51 C (nu0 = 1/6; nu0 = 1/24, whose viscosity
//     is not capped above 22 C, is within a cell of it from 10 to 60 s, two
//     cells before 10 s, and in the same cell at 60 s). THAT is grid-robust: at 28 s 3.69 mm on
//     25 um against 3.78 mm on 50 um, width 0.767 against 0.763 mm, tip melt
//     6 % apart. What decides it is not measured. Without the freezing drag the
//     cold melt touches at 34.29 s. With the air at 50 / 40 / 35 C: touches at
//     28.14 / 30.53 s, stalls at 2.7 mm -- the drop stalls exactly when the air
//     can cool its end into the band. The band at 20-30 C: stalls at 2.6 mm;
//     h = 30 / 90 / 300 W/m^2/K: stalls at 1.9 / 3.5 / 4.3 mm, holding
//     0.47 / 0.75 / 1.17 mm^3 within 1 mm of the tip at 60 s -- the more the
//     drop cools, the sooner its end freezes and the more melt it holds.
//   - FIELD OFF the oozing drop sags: isothermal, 1.78 mm up at 150 s and
//     falling 0.02 mm/s, then faster: it touches at 194.56 s (the planar
//     model at 91.3 s); with
//     heat it stalls at 3.3 mm, 1.54 mm^3 at the tip by 150 s, its end at
//     30-39 C over the last 5 s (one cell, and noisy: inside the freezing band).
//     With heat the field brings the drop down sooner but not further --
//     3.53 against 3.78 mm at 60 s, and 3.43 mm both at 120 s: in the cooling
//     regime the drop is oozing, not jetting.
//   - THE PROTOCOL, 60 s of oozing with the field off and then on (-fieldon 60,
//     heat): the cooled drop does not jet. By 120 s it is 3.375 mm up against
//     3.425 mm with the field off throughout, one cell, and holds 1.202
//     against 1.319 mm^3 at the tip, under a field of ~10 kV/mm at its surface.
//   - THE ELECTRODES (isothermal, -W 8): a long needle (-up 10, initial pull
//     0.763 gamma/R) touches at 15.10 s, close to the default box (0.835,
//     15.80 s), and a printhead disc (-up 5 -plateR 8, 0.497) at 18.19 s: a
//     pull 35 % weaker delays contact by 15-20 %.
//   - SURFACE TENSION (NOMINAL) at -20 / +20 %, with heat: the same cell
//     (3.525 mm at 60 s) and 0.751 / 0.750 against 0.750 mm^3 at the tip --
//     in the stalled regime the freezing sets everything, the capillary
//     pressure nothing. Isothermal, where it would matter, it was not run.
//   - THE JET (-jet, 2026-11-07; the box four cells deep; nu0 = 1/24 with heat,
//     against R4_heat_best): the fibres' flow, taken at the apex, delays the
//     isothermal contact from 15.80 s to 17.89 s with the fan's jet (17 % of the
//     feed) and to 32.14 s with the no-fan one (56 %; 28.47 s from a box eight
//     cells deep): it is the melt left at the tip that reaches the collector.
//     With heat the drop stalls whatever the jet, and the jet sets how much it
//     holds: 0.750 / 0.629 / 0.368 mm^3 within 1 mm of the tip at 60 s with no
//     jet, the fan's and the no-fan one, its lowest point 3.525 / 3.625 /
//     3.825 mm up. So the no-fan tip holds half of what it would with no jet and
//     the fan's 84 %: at the same h, the fan's thinner fibres leave 1.7x as much
//     at the tip. A fan also cools the drop; at h = 300 (forced convection's
//     order, NOMINAL) with the fan's jet the tip holds 0.989 mm^3 at 60 s,
//     4.375 mm up (1.169 with no jet), 2.7x the no-fan tip at h = 90 -- so
//     1.7-2.7x, the fibres and the cooling pulling the same way. What the no-fan
//     jet leaves is younger and so warmer, 82 against 68 C below the tip at
//     20 s, but within 2 C of the jetless drop by 60 s. Jetting only while the
//     apex is at 40 C or more takes 0.4 % less, is within a cell of it to 45 s
//     and three cells higher at 60 s (3.975 mm) -- the freezing end is that
//     sensitive. The mass the free surface creates (THE JET above) is 1-7 % of
//     what the jet takes, against a threefold difference between the two jets.
//     (These runs' logs print the jet's line in sink()'s words, "from the
//     surface cells", which outlived the switch to draw() until after they ran;
//     what ran is the draw, four cells in and 0.1 mm across unless the name says
//     otherwise: _d8 eight cells in, _r05 one cell across.)
//  So the model does not predict a contact time. It predicts that the tip's
//  thermal state decides between a column drawn to the collector in 10-35 s
//  and a drop that cools, crystallises at its end and grows at the tip -- and,
//  given the fibres' flow, how much of the feed the tip keeps.
//
//  Usage: mew_axi [-R 4] [-Ucol 0.04] [-nu 0.1667] [-mu 820] [-gamma 0.030] [-V 6.5]
//                 [-feed 0.0208] [-W 4] [-up 1.25] [-plate | -plateR MM] [-equip] [-lc MM]
//                 [-nofield] [-nogeom] [-rawfeed] [-heat [-h 90] [-Ea 31.3] [-Tair 25]
//                 [-Tcol 22] [-Tnoz 120] [-Tfreeze 30] [-nucap 1] [-Dafr 100] [-nofreeze]]
//                 [-fieldon T] [-jet Q [-jetr 0.1] [-jetdepth MM] [-jetTmin C]] [-M 20] [-sweeps 400]
//                 [-tend 20] [-tframe 0.5] [-watch T] [-out PREFIX]
//                 (mm, kV, mm^3/s, W/m^2/K, kJ/mol, C)
//==============================================================================
#include "FsElectricAxi.hpp"
#include "FsHeatAxi.hpp"
#include "NpyDump.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace lbm;

using L  = D2Q9;
using FS = FreeSurfaceSolver<L, CentralMoments<L, AxiGuo, RawPopulations>>;

namespace {

constexpr double kPi  = 3.14159265358979323846;
constexpr double kEps0 = 8.8541878128e-12;

struct Physical {
  double R    = 0.2e-3;      // bore radius: the 0.4 mm nozzle
  double lip  = 0.3e-3;      // nozzle wall, so the tip face is 1.0 mm across (NOMINAL)
  double gap  = 5.5e-3;      // nozzle-collector distance, Table 1
  double up   = 1.25e-3;     // nozzle length inside the box above its tip
  double W    = 4.0e-3;      // box radius
  double mu   = 820.0;       // PCL at 120 C, measured (Murat Kilic, 2026-11-04)
  double gam  = 0.030;       // surface tension [N/m] (NOMINAL)
  double rho  = 1080.0;
  double g    = 9.81;
  double V    = 6.5e3;
  double Q    = 0.0208e-9;   // feed [m^3/s], Table 1 corrected
  double Uc   = 0.035;       // collector, 2100 mm/min: sets the scaling only (it cannot move here)
  double lc   = -1;          // > 0: liquid within this of the tip conducts (mew_nozzle's cut)
  double plateR = -1;        // with -plate: the plate's radius (a printhead's block); < 0 the whole top
  // heat (-heat): all NOMINAL except the nozzle's 120 C
  double Tnoz = 120.0;       // the nozzle, the piston's melt and the reference of mu
  double Tair = 25.0;        // the air the surface loses to
  double Tcol = 22.0;        // the collector
  double hconv = 90.0;       // W/m^2/K: natural convection + radiation off a ~1 mm drop and a ~0.2 mm thread in still air
  double kmelt = 0.20;       // W/m/K
  double cp   = 2000.0;      // J/kg/K
  double Ea   = 31.3e3;      // J/mol, PCL flow activation energy over 80-120 C
  double Ts   = 30.0;        // the freezing band: fully held below Ts, free above Tl
  double Tl   = 40.0;

  double Un()   const { return Q / (kPi * R * R); }
  double Ca_n() const { return mu * Un() / gam; }
  double Ca_c() const { return mu * Uc / gam; }
  double Bo()   const { return rho * g * R * R / gam; }
  double LamE() const { return kEps0 * V * V / (gam * R); }
  double tvc()  const { return mu * R / gam; }
};

struct Opts {
  int    R      = 4;         // cells across the bore radius
  double nu     = 1.0 / 6.0; // lattice kinematic viscosity (tau = 1); see THE LATTICE VISCOSITY
  double Ucol   = 0.04;      // the collector's speed in lattice units: the scaling's anchor
  int    M      = 20;        // fluid steps per field update
  double t_end  = 20.0;
  double t_out  = 0.05;
  double t_frame = 0.5;
  bool   field  = true;
  bool   ohmic  = true;
  bool   geom   = true;      // the geometric exchange, donor and follows-line
  bool   plate  = false;     // the top held at the nozzle's potential
  bool   feedfix = true;     // the piston at U / rho_piston: a volume feed (see rho_piston)
  double watch  = -1;        // > 0: from this time on, a line and a frame at every field update
  double t_on   = 0;         // the field switched on at this time (s): the drop oozes first
  double jet    = 0;         // -jet Q: a jet leaving the melt's lowest point at Q mm^3/s (FreeSurfaceSolver::draw)
  double jet_r  = 0.1;       // -jetr MM: drawn from the melt within this of the axis there
  double jet_Tmin = -1e9;    // -jetTmin C: and only while the melt at the apex is at least this warm (with -heat)
  double jet_depth = -1;     // -jetdepth MM: the drawn box starts this far inside the melt above its lowest point
                             // (default four cells: see THE JET in the banner)
  bool   heat   = false;     // temperature, Arrhenius viscosity and freezing (FsHeatAxi)
  bool   freeze = true;      // with heat: the drag below the band
  double nucap  = 1.0;       // with heat: the largest lattice viscosity (the axis bound)
  double Dafr   = 100.0;     // with heat: the freezing drag as a Darcy number on the bore, A R^2 / mu
  int    sweeps = 400;       // SOR sweeps per field update, at most
  std::string out;
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Physical P;
    Opts o;
    for (int i = 1; i < argc; ++i) {
      auto num = [&]() {
        if (i + 1 >= argc) { std::printf("mew_axi: %s needs a value\n", argv[i]); std::exit(2); }
        return std::atof(argv[++i]);
      };
      const char* a = argv[i];
      if      (!std::strcmp(a, "-R"))       o.R = int(num());
      else if (!std::strcmp(a, "-Ucol"))    o.Ucol = num();
      else if (!std::strcmp(a, "-nu"))      o.nu = num();
      else if (!std::strcmp(a, "-M"))       o.M = int(num());
      else if (!std::strcmp(a, "-sweeps"))  o.sweeps = int(num());
      else if (!std::strcmp(a, "-tend"))    o.t_end = num();
      else if (!std::strcmp(a, "-tframe"))  o.t_frame = num();
      else if (!std::strcmp(a, "-mu"))      P.mu = num();
      else if (!std::strcmp(a, "-gamma"))   P.gam = num();
      else if (!std::strcmp(a, "-V"))       P.V = num() * 1e3;
      else if (!std::strcmp(a, "-lc"))      P.lc = num() * 1e-3;
      else if (!std::strcmp(a, "-W"))       P.W = num() * 1e-3;
      else if (!std::strcmp(a, "-up"))      P.up = num() * 1e-3;
      else if (!std::strcmp(a, "-feed"))    P.Q = num() * 1e-9;
      else if (!std::strcmp(a, "-plate"))   o.plate = true;
      else if (!std::strcmp(a, "-plateR"))  { o.plate = true; P.plateR = num() * 1e-3; }
      else if (!std::strcmp(a, "-equip"))   o.ohmic = false;
      else if (!std::strcmp(a, "-nofield")) o.field = false;
      else if (!std::strcmp(a, "-nogeom"))  o.geom = false;
      else if (!std::strcmp(a, "-rawfeed")) o.feedfix = false;
      else if (!std::strcmp(a, "-watch"))   o.watch = num();
      else if (!std::strcmp(a, "-heat"))    o.heat = true;
      else if (!std::strcmp(a, "-fieldon")) o.t_on = num();
      else if (!std::strcmp(a, "-jet"))     o.jet = num();
      else if (!std::strcmp(a, "-jetr"))    o.jet_r = num();
      else if (!std::strcmp(a, "-jetTmin")) o.jet_Tmin = num();
      else if (!std::strcmp(a, "-jetdepth")) o.jet_depth = num();
      else if (!std::strcmp(a, "-nofreeze")) o.freeze = false;
      else if (!std::strcmp(a, "-nucap"))   o.nucap = num();
      else if (!std::strcmp(a, "-Dafr"))    o.Dafr = num();
      else if (!std::strcmp(a, "-h"))       P.hconv = num();
      else if (!std::strcmp(a, "-Ea"))      P.Ea = num() * 1e3;
      else if (!std::strcmp(a, "-Tair"))    P.Tair = num();
      else if (!std::strcmp(a, "-Tcol"))    P.Tcol = num();
      else if (!std::strcmp(a, "-Tnoz"))    P.Tnoz = num();
      else if (!std::strcmp(a, "-Tfreeze")) { P.Ts = num(); P.Tl = P.Ts + 10.0; }
      else if (!std::strcmp(a, "-out") && i + 1 < argc) o.out = argv[++i];
      else {
        std::printf("mew_axi: unknown option '%s'. NOTHING WAS RUN.\n", a);
        Kokkos::finalize();
        return 2;
      }
    }

    //---- lattice units: mew_nozzle's scaling --------------------------------------
    const int    R    = o.R;
    const double dx   = P.R / R;
    const double ucap = o.Ucol / P.Ca_c();
    const double mu_l = o.nu;
    const double gam_l = ucap * mu_l;
    const double Un_l = P.Ca_n() * ucap;
    const double g_l  = P.Bo() * gam_l / double(R * R);
    const double C_l  = 0.5 * P.LamE() * gam_l * double(R);
    const double tvc_l = double(R) / ucap;
    const double dt   = P.tvc() / tvc_l;
    const double Oh_l = std::sqrt(mu_l * mu_l / (gam_l * R));
    // THE FREEZING DRAG IS A DARCY NUMBER, not a lattice constant: drag
    // against viscosity at a length L is A L^2 / mu, so a fixed lattice A
    // would hold four times harder at each halving of the cell. A = Da mu / R^2
    // fixes it on the bore radius (Da = 100: A = 1.04 at 50 um).
    const double A_fr = o.Dafr * mu_l / double(R * R);

    const Index nx   = Index(std::lround(P.W / dx));
    const Index gapc = Index(std::lround(P.gap / dx));
    const Index ytip = 1 + gapc;                    // first nozzle row; face at ytip - 1/2
    const Index ny   = ytip + Index(std::lround(P.up / dx));
    const Index lipc = Index(std::lround(P.lip / dx));
    const Index Ro   = R + lipc;
    const Index tipz = Index(std::lround(1.0e-3 / dx));
    const Index lcc  = P.lc > 0 ? Index(std::lround(P.lc / dx)) : ny;

    std::printf("mew_axi: r-z free surface (D2Q9, axisymmetric) + r-z finite-volume potential, %s\n",
                sizeof(Real) == 4 ? "float" : "double");
    std::printf("  PCL: mu %.0f Pa s  gamma %.3f N/m  rho %.0f   field %s, melt %s%s\n", P.mu, P.gam, P.rho,
                o.field ? "ON" : "OFF", o.ohmic ? "OHMIC" : "equipotential",
                P.lc > 0 ? " (conducting only near the tip, -lc)" : "");
    std::printf("  set-up: round nozzle, bore %.1f mm, tip face %.1f mm, gap %.1f mm, %.1f kV, feed %.4g mm^3/s,"
                " box radius %.1f mm, nozzle %.2f mm into the box, top %s\n", 2e3 * P.R, 2e3 * (P.R + P.lip),
                P.gap * 1e3, P.V * 1e-3, P.Q * 1e9, P.W * 1e3, P.up * 1e3,
                o.plate ? (P.plateR > 0 ? "a plate of finite radius at the nozzle's potential"
                                        : "a plate at the nozzle's potential") : "a zero-field frame");
    if (o.plate && P.plateR > 0) std::printf("  plate radius %.2f mm, a zero-field frame beyond it\n", P.plateR * 1e3);
    std::printf("  groups: Ca_feed %.3g  Bo %.3g  Lambda_E %.3g  (U_bore %.4g mm/s, t_vc %.3g s)\n",
                P.Ca_n(), P.Bo(), P.LamE(), P.Un() * 1e3, P.tvc());
    std::printf("  lattice: %lld x %lld (r x z), dx %.1f um, dt %.3e s, gamma %.3e, U_bore %.3e, g %.2e,"
                " C_E %.3e, Oh %.1f, M %d, exchange %s\n", (long long)nx, (long long)ny, dx * 1e6, dt,
                gam_l, Un_l, g_l, C_l, Oh_l, o.M, o.geom ? "geometric" : "default");

    //---- geometry ----------------------------------------------------------------
    auto is_lip = [=](Index x, Index y) { return y >= ytip && x >= R && x < Ro; };
    // The piston spans the bore and the lip's top row, as in mew_nozzle.
    auto is_piston = [=](Index x, Index y) { return y == ny - 1 && x < Ro; };

    Domain d(nx, ny, 1, false, false, true);
    FS fs(d);
    fs.coll.omega = FS::omega_from_viscosity(Real(o.nu));
    fs.set_axisymmetric(true);
    fs.set_gravity(Real(0), Real(-g_l));
    fs.set_surface_tension(Real(gam_l));
    fs.set_height_function(true);
    if (o.geom) {
      fs.set_exchange_geometric(true);
      fs.exchange_donor = true;
      fs.surface_follows_line = true;
    }
    fs.set_geometry([&](Index x, Index y, Index) -> FsCell {
      if (y == 0 || is_lip(x, y) || is_piston(x, y)) return FsSolid;
      return FsGas;
    });
    const Domain dd = d;
    const Index hx = d.hx, hy = d.hy;
    fs.initialize(KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; dd.coords(n, px, py, pz);
      const Index x = px - hx, y = py - hy;
      const bool bore = y >= ytip && y < ny - 1 && x < R;
      return typename FS::Seed{bore ? Real(1) : Real(0), Real(1)};
    });
    const Real uin = Real(Un_l);
    fs.move_obstacle(
        KOKKOS_LAMBDA(Index n) {
          Index px, py, pz; dd.coords(n, px, py, pz);
          const Index x = px - hx, y = py - hy;
          return y == ny - 1 && x < Ro;
        },
        KOKKOS_LAMBDA(Index n, Real u[3]) {
          (void)n;
          u[0] = Real(0);  u[1] = -uin;  u[2] = Real(0);
        },
        Real(0), Real(0));

    FsElectricAxi el(d);
    el.coef = o.field ? Real(C_l) : Real(0);
    el.ohmic = o.ohmic;
    el.liquid_value = Real(1);
    el.set_roles(
        [&](Index x, Index y) -> ElecRole {
          if (y == 0 || is_lip(x, y) || is_piston(x, y)) return ElecElectrode;
          if (y == ny - 1 && o.plate && (P.plateR < 0 || double(x) + 0.5 < P.plateR / dx)) return ElecElectrode;
          if (x == nx - 1 || y == ny - 1) return ElecFrame;
          return ElecFree;
        },
        [&](Index, Index y) -> Real { return y == 0 ? Real(0) : Real(1); });
    el.set_conduct_region([&](Index, Index y) { return y >= ytip - lcc; });
    el.refresh(fs);
    el.initialize(0.5);
    const int s0 = el.solve(200000, 1e-10);
    if (o.t_on > 0) el.coef = Real(0);            // -fieldon: off until then
    el.apply(fs);
    std::printf("  initial field: %d SOR sweeps\n", s0);
    if (o.t_on > 0) std::printf("  the field is switched on at t = %.1f s; the melt oozes until then\n", o.t_on);
    if (o.jet > 0) {
      const long jd = o.jet_depth < 0 ? 4L : std::lround(o.jet_depth * 1e-3 / dx);
      char warm[96] = "";
      if (o.heat && o.jet_Tmin > -1e8)
        std::snprintf(warm, sizeof warm, ", and only while the melt at the apex is at least %.1f C", o.jet_Tmin);
      std::printf("  jet: %.5f mm^3/s (%.0f %% of the feed) drawn from the melt within %.3f mm of the axis, in a box\n"
                  "       starting %ld cells above its lowest point there, while the field is on and until contact%s\n",
                  o.jet, o.jet * 1e-9 / P.Q * 100.0, o.jet_r, jd, warm);
    }

    //---- heat ------------------------------------------------------------------------
    FsHeatAxi heat(d);
    HeatFields HF;
    const double alpha_phys = P.kmelt / (P.rho * P.cp);
    const double EaR = P.Ea / 8.314462618;
    if (o.heat) {
      heat.alpha = alpha_phys * dt / (dx * dx);
      heat.hloss = P.hconv * dt / (P.rho * P.cp * dx);
      heat.T_air = P.Tair;
      heat.set_roles(
          [&](Index x, Index y) -> HeatRole {
            if (y == 0 || is_lip(x, y) || is_piston(x, y)) return HeatWall;
            return HeatFree;
          },
          [&](Index, Index y) { return y == 0 ? P.Tcol : P.Tnoz; });
      fields_from(fs, d, HF);
      heat.initialize(HF, [&](Index, Index) { return P.Tnoz; });
      heat.write_omega(fs, o.nu, EaR, P.Tnoz, o.nucap);
      if (o.freeze) heat.write_drag(fs, A_fr, P.Ts, P.Tl);
      const double Tcap = 1.0 / (std::log(o.nucap / o.nu) / EaR + 1.0 / (P.Tnoz + 273.15)) - 273.15;
      std::printf("  heat: alpha %.3e m^2/s (%.3e lattice), h %.0f W/m^2/K (%.3e), Bi(R) %.3f, Ea %.1f kJ/mol,"
                  " air %.0f C, collector %.0f C\n", alpha_phys, heat.alpha, P.hconv, heat.hloss,
                  P.hconv * P.R / P.kmelt, P.Ea * 1e-3, P.Tair, P.Tcol);
      // The band printed is the band applied: a literal "30-40 C" here once labelled
      // the -Tfreeze 20 run with the default band while its physics used 20-30 C.
      char band[64];
      if (o.freeze) std::snprintf(band, sizeof band, "below %.0f-%.0f C (drag)", P.Ts, P.Tl);
      else          std::snprintf(band, sizeof band, "off");
      std::printf("        viscosity capped at nu %.3f (x %.1f, reached at %.1f C); freezing %s, Da %.0f (A %.3f)\n",
                  o.nucap, o.nucap / o.nu, Tcap, band, o.Dafr, A_fr);
      std::printf("        times: drop cooling rho c_p (0.5 mm / 3) / h = %.2f s, conduction over 0.5 mm %.2f s\n",
                  P.rho * P.cp * (0.5e-3 / 3.0) / P.hconv, 0.25e-6 / alpha_phys);
    }

    // THE FEED IS A VOLUME RATE, AND THE LATTICE IS COMPRESSIBLE. The piston's
    // moving-wall bounce-back injects rho U per cell at the density of the melt
    // beside it, and that melt is compressed by the pressure that drives it down
    // the bore: 8 mu U / R^2 per cell over the bore's length, about 1.2e-3 in
    // lattice pressure at 50 um, i.e. rho = 1.0035. Released into the drop, the
    // extra 0.35 % is extra volume -- the 3-D run's feed came out 3.81 % high where
    // the bore's rounding explains 3.45 %, and this is the rest. So the piston's
    // speed is U / rho_piston, refreshed with the field, and the mass it injects
    // is U per cell. -rawfeed keeps the uncorrected piston, as the siblings have.
    auto rho_piston = [&]() {
      auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.rho());
      double s = 0, w = 0;
      for (Index x = 0; x < R; ++x) { s += double(hr(d.id(x, ny - 2))) * (double(x) + 0.5); w += double(x) + 0.5; }
      return s / w;
    };
    auto set_piston = [&](Real u) {
      fs.move_obstacle(
          KOKKOS_LAMBDA(Index n) {
            Index px, py, pz; dd.coords(n, px, py, pz);
            const Index x = px - hx, y = py - hy;
            return y == ny - 1 && x < Ro;
          },
          KOKKOS_LAMBDA(Index n, Real uu[3]) {
            (void)n;
            uu[0] = Real(0);  uu[1] = -u;  uu[2] = Real(0);
          },
          Real(0), Real(0));
    };

    //---- diagnostics ---------------------------------------------------------------
    // r-weighted cell volume 2 pi r dx^3 in mm^3, r = x + 1/2 cells
    const double v3 = 2.0 * kPi * dx * dx * dx * 1e9;
    const double m0 = double(fs.total_mass());
    const double inj_step = double(uin) * 0.5 * double(R) * double(R);   // sum_{x<R} uin (x + 1/2)
    double jet_asked = 0;                       // -jet: the mass the jet has asked draw() for
    struct Diag { double t, total, tip, gap, zlow, neck, drained, jet, jet_asked, ledger, pe, emax, etip, phitip, rho_p,
                  Tend, Tgap, Tmin, frozen, width; };
    auto measure = [&](double t) {
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      Diag D{};
      D.t = t;
      Index zlow = ny;
      double neck = 0;
      for (Index y = 1; y < ny - 1; ++y)
        for (Index x = 0; x < nx; ++x) {
          const std::uint8_t f = hf(d.id(x, y));
          const double v = (f == FsFluid) ? 1.0 : (f == FsInterface ? double(he(d.id(x, y))) : 0.0);
          if (v <= 0) continue;
          const double rv = v * (double(x) + 0.5);
          D.total += rv;
          if (y >= ytip) continue;
          D.gap += rv;
          if (y >= ytip - tipz) D.tip += rv;
          if (x <= 3 * Ro && y < zlow) zlow = y;
          if (y == ytip - 1) neck += rv;
        }
      D.zlow = (zlow < ny) ? (double(zlow) - 0.5) * dx * 1e3 : P.gap * 1e3;
      // what a side camera sees: the melt's widest outline below the tip face
      // (the outermost liquid cell of a row, plus its fill), as a diameter
      D.width = 0;
      for (Index y = 1; y < ytip; ++y)
        for (Index x = nx - 1; x >= 0; --x) {
          const std::uint8_t f = hf(d.id(x, y));
          if (f != FsFluid && f != FsInterface) continue;
          const double re = double(x) + (f == FsFluid ? 1.0 : double(he(d.id(x, y))));
          D.width = std::max(D.width, 2.0 * re * dx * 1e3);
          break;
        }
      D.neck = 2.0 * std::sqrt(2.0 * neck) * dx * 1e3;   // pi (D/2)^2 = 2 pi sum(r v) dx^2
      const double drained = double(fs.drained_mass());
      const double injected = inj_step * (t / dt);
      D.drained = drained * v3;
      D.jet = double(fs.sunk_mass()) * v3;
      D.jet_asked = jet_asked * v3;
      D.ledger = (double(fs.total_mass()) + drained - m0 - injected) / std::max(1.0, injected);
      D.total *= v3; D.tip *= v3; D.gap *= v3;
      D.pe = el.max_stress() / (gam_l / R);
      double e2m = 0, e2t = 0;
      // the strongest field below the tip, and on the axis just below the melt's lowest point
      for (Index y = 1; y < ytip - 1; ++y)
        for (Index x = 0; x < nx - 1; ++x)
          if (el.cell(x, y) == 3) e2m = std::max(e2m, el.e2(x, y));
      if (zlow < ny && zlow > 1) e2t = el.e2(0, zlow - 1);
      D.emax = std::sqrt(e2m) * P.V / dx * 1e-6;                 // kV/mm
      D.etip = std::sqrt(e2t) * P.V / dx * 1e-6;
      if (!o.field || t < o.t_on) D.emax = D.etip = 0;          // off: not the stale t = 0 solve
      // the melt's potential at its lowest conducting cell on the axis
      D.phitip = 1.0;
      for (Index y = 1; y < ytip; ++y)
        if (el.cell(0, y) == 4) { D.phitip = el.phi(0, y); break; }
      D.rho_p = rho_piston();
      D.Tend = D.Tgap = D.Tmin = NAN;
      D.frozen = 0;
      if (o.heat) {
        // the melt's lowest axis cell, its mean below the tip, its coldest cell, and what is held
        for (Index y = 1; y < ytip; ++y)
          if (heat.liquid(0, y)) { D.Tend = heat.T(0, y); break; }
        double sT = 0, sw = 0, tmin = 1e9;
        for (Index y = 1; y < ny - 1; ++y)
          for (Index x = 0; x < nx; ++x) {
            if (!heat.liquid(x, y)) continue;
            const double T = heat.T(x, y), w = heat.phi(x, y) * (double(x) + 0.5);
            tmin = std::min(tmin, T);
            if (y < ytip) { sT += w * T; sw += w; }
            if (o.freeze && FsHeatAxi::solid_fraction(T, P.Ts, P.Tl) >= 0.5) D.frozen += w;
          }
        D.Tgap = sw > 0 ? sT / sw : NAN;
        D.Tmin = tmin < 1e8 ? tmin : NAN;
        D.frozen *= v3;
      }
      return D;
    };
    auto frame = [&](double t) {
      if (o.out.empty()) return;
      auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
      auto he = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.fill());
      const std::size_t N = std::size_t(nx * ny);
      std::vector<float> fl(N), ph(N);
      for (Index y = 0; y < ny; ++y)
        for (Index x = 0; x < nx; ++x) {
          const Index n = d.id(x, y);
          const std::uint8_t f = hf(n);
          const std::size_t k = std::size_t(y * nx + x);
          fl[k] = (f == FsSolid) ? -1.0f : (f == FsFluid ? 1.0f : (f == FsInterface ? float(he(n)) : 0.0f));
          ph[k] = float(el.phi(x, y));
        }
      char nm[64];
      if (o.watch > 0) std::snprintf(nm, sizeof nm, "_t%08.4f", t);
      else std::snprintf(nm, sizeof nm, "_t%06.2f", t);
      const std::vector<std::size_t> shape{std::size_t(ny), std::size_t(nx)};
      write_npy(o.out + nm + "_fill.npy", fl, shape);
      write_npy(o.out + nm + "_phi.npy", ph, shape);
      if (o.heat) {
        std::vector<float> tt(N);
        for (Index y = 0; y < ny; ++y)
          for (Index x = 0; x < nx; ++x)
            tt[std::size_t(y * nx + x)] = heat.liquid(x, y) ? float(heat.T(x, y)) : NAN;
        write_npy(o.out + nm + "_T.npy", tt, shape);
      }
    };

    FILE* fs_out = nullptr;
    if (!o.out.empty()) {
      fs_out = std::fopen((o.out + "_series.csv").c_str(), "w");
      if (fs_out) std::fprintf(fs_out, "t_s,melt_total_mm3,tip_zone_mm3,gap_mm3,z_low_mm,neck_mm,drained_mm3,"
                               "ledger,max_pe_over_gamma_R,max_E_kV_per_mm,E_below_tip_kV_per_mm,phi_end,rho_piston,"
                               "T_end_C,T_gap_mean_C,T_min_C,frozen_mm3,width_mm,jet_mm3,jet_asked_mm3\n");
    }
    std::printf("\n  %8s %10s %10s %10s %7s %6s %9s %8s %7s %7s %6s %8s %s\n", "t [s]", "melt mm^3", "tip mm^3",
                "gap mm^3", "z_low", "neck", "ledger", "pE/g/R", "E kV/mm", "E end", "phi_e", "rho_pst", "wall s/s");

    const long nsteps = long(o.t_end / dt);
    const long sout = std::max<long>(o.M, long(o.t_out / dt) / o.M * o.M);
    const long sframe = std::max<long>(sout, long(o.t_frame / dt) / sout * sout);
    frame(0.0);
    bool contacted = false, touched = false;
    double t_fluid = 0, t_field = 0;
    long sweeps_used = 0, solves = 0;
    auto wall0 = std::chrono::steady_clock::now();
    double last_wall = 0, last_t = 0;
    const Index ydr = Index(std::lround(0.25e-3 / dx));
    for (long t = 0; t < nsteps; t += o.M) {
      auto c0 = std::chrono::steady_clock::now();
      // -jet: the melt a jet carries away, DRAWN through the density of the Fluid cells at the
      // melt's lowest point near the axis -- the cone's apex -- so the melt around flows to it
      // (FreeSurfaceSolver::draw; sink(), which lowers surface mass, tunnelled up the axis).
      // Where it is, is found once per field update; it acts after every step, while the field
      // that pulls it is on, until contact, and with -jetTmin only while the melt there is warm.
      bool jet_now = false;
      Index ya = 0, nr = 1;
      if (o.jet > 0 && o.field && t * dt >= o.t_on && !contacted) {
        auto hfj = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
        nr = std::max<Index>(1, Index(std::lround(o.jet_r * 1e-3 / dx)));
        Index yap = ny;
        for (Index y = 1; y < ytip && yap == ny; ++y)
          for (Index x = 0; x < nr; ++x) {
            const std::uint8_t f = hfj(d.id(x, y));
            if (f == FsFluid || f == FsInterface) { yap = y; break; }
          }
        bool warm = true;
        if (o.heat && o.jet_Tmin > -1e8 && yap < ny) {
          double Ts = 0; int nT = 0;
          for (Index y = yap; y < std::min<Index>(ytip, yap + 2 * nr); ++y)
            for (Index x = 0; x < nr; ++x)
              if (heat.liquid(x, y)) { Ts += heat.T(x, y); ++nT; }
          warm = nT > 0 && Ts / nT >= o.jet_Tmin;
        }
        jet_now = yap < ny && warm;
        ya = yap;
      }
      const Real dmj = Real(inj_step * (o.jet * 1e-9 / P.Q));
      for (int j = 0; j < o.M; ++j) {
        fs.step();
        if (jet_now) {
          const Index y0 = ya + (o.jet_depth < 0 ? Index(4) : Index(std::lround(o.jet_depth * 1e-3 / dx))), r0 = nr;
          fs.draw(dmj, KOKKOS_LAMBDA(Index n) {
            Index px, py, pz; dd.coords(n, px, py, pz);
            const Index x = px - hx, y = py - hy;
            return x < r0 && y >= y0 && y < y0 + 2 * r0;
          });
          jet_asked += double(dmj);
        }
      }
      Kokkos::fence();
      auto c1 = std::chrono::steady_clock::now();
      // Melt that spreads on the collector to the box's rim leaves there.
      fs.drain(KOKKOS_LAMBDA(Index n) {
        Index px, py, pz; dd.coords(n, px, py, pz);
        const Index x = px - hx, y = py - hy;
        return x >= nx - 3 && y <= ydr;
      });
      if (!contacted) {
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
        for (Index x = 0; x < nx && !contacted; ++x) {
          const std::uint8_t f = hf(d.id(x, 1));
          contacted = (f == FsFluid || f == FsInterface);
        }
        if (contacted) std::printf("  first contact with the collector at t = %.3f s\n", (t + o.M) * dt);
      }
      if (o.feedfix) set_piston(Real(Un_l / rho_piston()));
      auto c2 = std::chrono::steady_clock::now();
      // -fieldon: until t_on the melt oozes with the field off (no stress, no solve)
      const bool field_now = o.field && (t + o.M) * dt >= o.t_on;
      if (o.field && !field_now && el.coef != Real(0)) { el.coef = Real(0); el.apply(fs); }
      bool switched_on = false;
      if (field_now && el.coef == Real(0)) { el.coef = Real(C_l); switched_on = true; }
      if (field_now) {
        el.refresh(fs);
        // A CHANGE OF CONTACT STATE IS SOLVED TO CONVERGENCE: warm-started
        // from a column at the nozzle's potential, a capped solve would leave
        // the melt by the collector near 1 with the collector at 0 half a cell
        // away for several refreshes. (The run that came apart at contact on
        // 2026-11-06 was NOT this: with this in place it still did, 0.02 s
        // before touching, and the cause was the axis column's explicit
        // viscous term in a thread two cells thick -- see FreeSurfaceSolver's
        // member axisymmetric. Kept because the field it avoids is wrong.)
        const bool touch = el.touching();
        sweeps_used += el.solve((touch != touched || switched_on) ? 400000 : o.sweeps, 1e-9);
        touched = touch;
        ++solves;
        el.apply(fs);
      }
      if (o.heat) {
        fields_from(fs, d, HF);
        heat.update(HF, double(o.M));
        heat.write_omega(fs, o.nu, EaR, P.Tnoz, o.nucap);
        if (o.freeze) heat.write_drag(fs, A_fr, P.Ts, P.Tl);
      }
      auto c3 = std::chrono::steady_clock::now();
      if (o.watch > 0 && (t + o.M) * dt >= o.watch) {
        // -watch: where the state is extreme, every field update
        auto hf = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.flags());
        auto hr = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.rho());
        auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.ux());
        auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.uy());
        auto hk = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.kappa());
        auto hg = Kokkos::create_mirror_view_and_copy(HostSpace{}, fs.rho_G_of);
        double um = 0, rmin = 9, rmax = 0, km = 0, gm = 0;
        Index ux_ = 0, uy_ = 0, kx = 0, ky = 0, gx = 0, gy = 0, rnx = 0, rny = 0, rxx = 0, rxy = 0;
        int nif = 0;
        for (Index y = 1; y < ny - 1; ++y)
          for (Index x = 0; x < nx; ++x) {
            const Index n = d.id(x, y);
            const std::uint8_t f = hf(n);
            if (f != FsFluid && f != FsInterface) continue;
            nif += (f == FsInterface);
            const double u = std::hypot(double(hu(n)), double(hv(n)));
            if (!(u <= um)) { um = u; ux_ = x; uy_ = y; }
            if (!(double(hr(n)) >= rmin)) { rmin = hr(n); rnx = x; rny = y; }
            if (!(double(hr(n)) <= rmax)) { rmax = hr(n); rxx = x; rxy = y; }
            if (f == FsInterface && !(std::fabs(double(hk(n))) <= km)) { km = std::fabs(double(hk(n))); kx = x; ky = y; }
            if (hg.extent(0) && !(std::fabs(double(hg(n)) - 1.0) <= gm)) { gm = std::fabs(double(hg(n)) - 1.0); gx = x; gy = y; }
          }
        std::printf("  watch t %.4f |u|max %.3e @(%lld,%lld) rho [%.5f @(%lld,%lld), %.5f @(%lld,%lld)]"
                    " |k|max %.3f @(%lld,%lld) |rhoG-1| %.3e @(%lld,%lld) iface %d touch %d\n",
                    (t + o.M) * dt, um, (long long)ux_, (long long)uy_, rmin, (long long)rnx, (long long)rny, rmax,
                    (long long)rxx, (long long)rxy, km, (long long)kx, (long long)ky, gm, (long long)gx,
                    (long long)gy, nif, int(touched));
        std::fflush(stdout);
        frame((t + o.M) * dt);
      }
      t_fluid += std::chrono::duration<double>(c1 - c0).count();
      t_field += std::chrono::duration<double>(c3 - c2).count();
      const long ts = t + o.M;
      if (ts % sout == 0) {
        const Diag D = measure(ts * dt);
        if (!std::isfinite(D.total) || !std::isfinite(D.ledger) || !(D.total > 0)) {
          std::printf("  NOT FINITE, or the melt is gone, at t = %.3f s\n", ts * dt);
          rc = 1;
          break;
        }
        if (fs_out) std::fprintf(fs_out, "%.5f,%.6f,%.6f,%.6f,%.5f,%.5f,%.6f,%.3e,%.4f,%.4f,%.4f,%.5f,%.6f,"
                                 "%.3f,%.3f,%.3f,%.6f,%.4f,%.6f,%.6f\n",
                                 D.t, D.total, D.tip, D.gap, D.zlow, D.neck, D.drained, D.ledger, D.pe, D.emax,
                                 D.etip, D.phitip, D.rho_p, D.Tend, D.Tgap, D.Tmin, D.frozen, D.width, D.jet,
                                 D.jet_asked);
        if (ts % sframe == 0) {
          const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
          const double rate = (wall - last_wall) / std::max(1e-9, D.t - last_t);
          last_wall = wall; last_t = D.t;
          std::printf("  %8.3f %10.5f %10.5f %10.5f %7.3f %6.3f %9.1e %8.3f %7.2f %7.2f %6.3f %8.5f %6.0f", D.t,
                      D.total, D.tip, D.gap, D.zlow, D.neck, D.ledger, D.pe, D.emax, D.etip, D.phitip, D.rho_p, rate);
          if (o.heat) std::printf("  T end %5.1f gap %5.1f min %5.1f frozen %.4f", D.Tend, D.Tgap, D.Tmin, D.frozen);
          if (o.jet > 0) std::printf("  jet %.5f of %.5f", D.jet, D.jet_asked);
          std::printf("\n");
          std::fflush(stdout);
          if (fs_out) std::fflush(fs_out);
          frame(ts * dt);
        }
      }
    }
    std::printf("  time in the fluid %.0f s, in the field %.0f s (%.1f SOR sweeps a solve)\n", t_fluid, t_field,
                solves ? double(sweeps_used) / double(solves) : 0.0);
    if (fs_out) std::fclose(fs_out);
  }
  Kokkos::finalize();
  return rc;
}
