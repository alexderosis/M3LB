//==============================================================================
//  Flow past a CIRCULAR cylinder in a transverse magnetic field -- how a
//  Lorentz force kills a von Karman street.
//
//  THIS IS A DEMONSTRATOR AND NOT A VALIDATION CASE, and the distinction is the
//  reason it lives here rather than in validation/. There is no analytic
//  solution, no published table this setup matches term for term, and no
//  convergence rate being claimed: what it can fail is to crash. Every number
//  it prints is a measurement of THIS configuration, not an agreement with
//  anything. The one thing it does check against a known answer is its own
//  force evaluation, below, because a demonstrator that reports a force it has
//  never shown to be right is worse than one that reports no force at all.
//
//  THE SETUP. A plane channel, a circular cylinder of diameter D on the
//  centreline at 8D from the inlet with 17D of wake behind it, blockage
//  D/H = 1/8 exactly (H fluid rows between two solid rows, so ny = 8D + 2),
//  parabolic inlet of peak speed u_max, and a UNIFORM APPLIED FIELD
//  B0 ALONG y -- transverse to the flow and in the plane, so u x B points along
//  z and the induced field it drives is back in the plane. That is a closed
//  two-dimensional MHD system and not a truncation of a three-dimensional one:
//  j has only a z component, so div j = 0 holds identically and there is no
//  return-path question to answer.
//
//      Re = u_max D / nu        (on the PEAK inlet speed, as in Breuer et al.)
//      Ha = B0 D / sqrt(nu eta)
//      N  = Ha^2 / Re           the interaction parameter -- Lorentz over inertia
//
//  N is the parameter that matters here. It is the ratio of the Joule braking
//  time to the eddy turnover time, and the wake is expected to go from shedding
//  to steady as it passes order one. Re is held fixed and Ha swept, so N is
//  swept with it.
//
//  WHAT THE SWEEP GIVES, at D = 30, Re = 100, Pr_m = 0.01, 45000 steps of
//  transient and 45000 averaged:
//
//      Ha     N      C_d      St      wake amp     |b|/B0    dmass
//       0    0.00   1.2937   0.1579   64.9 %       --       +1.7e-04
//       2    0.04   1.1897   0.1369   19.8 %       0.433    +3.3e-04
//       4    0.16   1.3409    --       0.10 %      0.377    +7.3e-04
//       6    0.36   1.6904    --       0.05 %      0.332    +1.1e-03
//       8    0.64   2.2629    --       0.07 %      0.299    +1.5e-03
//      10    1.00   3.0301    --       0.06 %      0.280    +2.0e-03
//      15    2.25   5.5525    --       0.05 %      0.268    +3.4e-03
//      20    4.00   8.7021    --       0.04 %      0.262    +4.8e-03
//
//  THE DRAG FALLS BEFORE IT RISES, and that is the one shape worth pointing at.
//  C_d drops 8 % from Ha = 0 to Ha = 2 and only then climbs. Two effects are
//  competing and they have opposite signs: suppressing the shedding REMOVES
//  form drag, and the Lorentz force ADDS drag. The first saturates once the
//  wake is steady -- by Ha = 4 the lift amplitude is 90x down and the probe is
//  at the noise floor -- and the second does not.
//
//  ABOVE N = 1 THE INCREMENT IS LINEAR IN N, which is the asymptotic scaling a
//  Joule-braked wake is supposed to have and is the closest this case comes to
//  a quantitative check:
//
//      N                  0.64   1.00   2.25   4.00
//      (C_d - C_d0) / N   1.51   1.74   1.89   1.85
//
//  constant to +-4 % over the top three, and visibly not yet linear below.
//
//  AND IT REPRODUCES ACROSS RESOLUTION TO ABOUT 1 %. The same Re and Pr_m at
//  D = 20 gives 1.3003 / 3.0605 / 8.7962 at Ha = 0 / 10 / 20 against 1.2937 /
//  3.0301 / 8.7021 here -- 0.5 %, 1.0 %, 1.1 % apart, over a 1.5x change in
//  resolution and with the cylinder's Hartmann layer down to 1.5 cells at the
//  top of the range. The two runs used different averaging windows, which is
//  immaterial where the wake is steady and is worth a little caution at Ha = 0.
//
//  THE OUTLET'S MASS DRIFT GROWS WITH THE FIELD, from 1.7e-04 at Ha = 0 to
//  4.8e-03 at Ha = 20, a factor of 28. NrmOutXp derives its normal velocity
//  from the inverted closure, which knows nothing about the Maxwell stress, so
//  this is expected rather than mysterious -- but a drag normalised by rho = 1
//  is high by exactly that drift, i.e. by 0.5 % at the top of the sweep. It is
//  printed every row for that reason.
//
//  THE FORCE IS BY MOMENTUM EXCHANGE, AND UNDER MHD IT CARRIES THE MAXWELL
//  STRESS AS WELL. Neither MHD operator applies the Lorentz force as a body
//  force -- that would need derivatives of B and lose an order -- both put the
//  Maxwell stress into the fluid equilibrium's second moment, MhdBGK through an
//  explicit addition to f^eq and MhdCentralMoments through the relaxation
//  targets themselves (see those files' banners). So the populations arriving
//  at a boundary link already carry the magnetic traction, and the
//  momentum-exchange sum
//
//      F = sum over boundary links of  2 c_i g(n, opp(i))
//
//  returns the TOTAL momentum flux into the solid, hydrodynamic plus magnetic.
//  That is the right quantity to call drag, but it is not separable here and
//  this file does not pretend to separate it.
//
//  -selftest IS RUN FIRST AND IS NOT OPTIONAL, for the reason
//  validation/square_cylinder.cpp gives: a force evaluation that has never
//  reproduced a force it cannot get wrong is not a measurement. The check is a
//  periodic channel driven by a uniform body force G, and in steady state the
//  walls must carry exactly G x (number of fluid nodes) -- a momentum balance,
//  with no appeal to a profile or a viscosity.
//
//  The MHD twist is what makes it worth repeating here rather than citing the
//  square cylinder's. Run it with a uniform field ALONG the flow: u x B
//  vanishes identically, so there is no Lorentz force and no induced field, and
//  the balance must come out exactly as it does at B = 0. But the uniform field
//  is still in every equilibrium, adding a constant to the second moment -- and
//  a momentum-exchange sum runs over the directions pointing INTO the wall and
//  no further, which is precisely where a constant added to every population
//  does NOT cancel. That is the same mechanism as the shifted-storage trap this
//  tree already carries a helper for, and it is the one way a uniform field
//  could silently offset every force this file reports.
//
//  BUT THE FORCE BALANCE IS BLIND TO THE FIELD, so it cannot be the whole test.
//  Steady state pins the wall force at G N_fluid by momentum conservation
//  whatever the profile is, and measured, the three cases agree to the last
//  bits -- 2.5600000e-04 at Ha = 0, at Ha = 5 along the flow, and at Ha = 5
//  across it. That is the right answer and it is also exactly what a DEAD
//  coupling would produce, which is the failure CLAUDE.md records for GPU/'s
//  ehd_cavity: a null velocity pointer left the charge un-advected and every
//  diagnostic still looked reasonable. So the self-test also measures the
//  centreline velocity, which the field does move: at Ha = 5 the exact Hartmann
//  midplane value is 39 % of the parabolic peak, and the run gives 35 %.
//
//  AND THAT LAST 10 % IS THE HALF-CELL PAIRING, MEASURED RATHER THAN ASSERTED.
//  The magnetic Dirichlet condition sits ON nodes 0 and ny-1 while halfway
//  bounce-back puts the no-slip planes at 0.5 and H + 0.5, so the two
//  boundaries bound channels of different width and the field sees one half
//  cell more than the flow does. That is an O(1/H) error and therefore a
//  falsifiable claim -- double H and it must halve. -pair runs the ladder:
//
//      H        32        64       128
//      u_c   -10.69 %   -5.77 %   -3.00 %          ratios 1.85, 1.92
//
//  which is first order, converging on 2 as the ladder refines. An error in
//  the coupling itself would not do that. It is the cost
//  validation/hartmann3d.cpp warns about, priced on the configuration this file
//  actually runs, and it is why the channel walls here are on-node regularised
//  walls and only the cylinder -- which needs momentum exchange -- is not.
//
//  WHY THE CYLINDER IS BOUNCE-BACK AND THE CHANNEL WALLS ARE NOT. Momentum
//  exchange needs halfway bounce-back, so the cylinder is Solid. The channel
//  walls are regularised on-node walls instead, which puts them on the same
//  plane as their magnetic Dirichlet condition; no force is taken there, so
//  nothing is lost. The cylinder's own magnetic condition is the one place the
//  planes could disagree, and -cyl chooses:
//
//    cond   (default) NO magnetic boundary on the cylinder at all. The field
//           diffuses through it, and since the velocity is identically zero
//           inside a Solid cell the induction equation there reduces to pure
//           diffusion -- which IS a solid conductor at rest of the same
//           conductivity as the fluid. A real configuration, not a dodge, and
//           it has no plane mismatch because it has no magnetic boundary.
//    insul  b = 0 on the fluid nodes ringing the cylinder. Physically the other
//           standard choice, and it carries the half-cell offset: the magnetic
//           plane sits half a cell inside the no-slip plane.
//
//  THE MAGNETIC PRANDTL NUMBER IS 0.01 BY DEFAULT, AND AT 1 THIS CASE DOES NOT
//  RUN AT ALL. That is both a physical choice and a measured constraint, and
//  they happen to agree. Physically, liquid metals have Pr_m of order 1e-6 and
//  every laboratory MHD wake is in the quasi-static limit, so Pr_m << 1 is the
//  regime the problem belongs to. Numerically, the induction equation here is
//  unstable at Pr_m = 1 -- and the deletion test is what showed it, because the
//  first four blow-ups all looked like a wake failing. Running with NO CYLINDER
//  AT ALL, which leaves a plain channel whose exact answer is a developing
//  Hartmann profile:
//
//      Pr_m      omega_mag     max|b|/B0 at t = 4000 / 8000 / 12000
//      1.00        1.942        3.73   7.17   17.69      runaway
//      0.10        1.538        2.07   2.92    3.76      still climbing
//      0.01        0.500        0.280  0.279   0.278     saturated
//
//  and at Pr_m = 0.01 the velocity sits at 0.05000 to five figures throughout
//  while at Pr_m = 1 it is the induced field that runs away FIRST and drags the
//  velocity after it. So the failure is in the induction equation and not in
//  the wake, which is why the progress line reports max|b| beside max|u|: with
//  only the velocity instrumented, four separate runs read as "the cylinder
//  wake went unstable" and the cylinder was not involved.
//
//  WHAT THIS DOES NOT SEPARATE. Lowering Pr_m at fixed Re lowers the magnetic
//  Reynolds number Rm = Re Pr_m AND moves omega_mag away from 2 at the same
//  time, and those are the two obvious mechanisms -- an unresolved induced
//  field, or the near-reflection this tree has documented for BGK at omega -> 2
//  three times over. They cannot be separated by this knob because they ARE
//  this knob: Rm and omega_mag are both functions of eta alone at fixed u and
//  D. It is stated as a bound, not apportioned.
//
//  THE MAGNETIC OUTLET. MagOutXp is documented in MagneticSolver.hpp as not
//  validated and measured 6 % high over its last ten nodes, so the default here
//  is a Dirichlet outlet at the applied field -- 20D downstream, where the
//  induced field has decayed. -magout grad selects the zero-gradient one for
//  comparison. Neither is transparent to an arriving vortex; 20D is the
//  defence.
//==============================================================================
#include "Campaign.hpp"
#include "collision/MhdBGK.hpp"
#include "collision/MhdCentralMoments.hpp"
#include "solver/MagneticSolver.hpp"

using namespace lbm;
using namespace campaign;

using FL = D2Q9;
using ML = D2Q5;

// BGK with Guo, for the self-test's body-forced channel only.
using CollG = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations, Guo>;
// The sweep. Central moments because at Re = 200 with D = 20 the relaxation
// time is 0.515, and validation/square_cylinder.cpp measured BGK dying at
// tau = 0.512 once the wake became energetic while central moments held.
using CollB = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations>;
using CollC = MhdCentralMoments<FL, true>;

// UNSHIFT BEFORE TAKING A ONE-SIDED MOMENT -- see square_cylinder.cpp. Shifted
// storage holds g_i = f_i - w_i, and sum_i c_i w_i = 0 only over the FULL
// direction set, which a boundary-link sum is not.
template <class C>
KOKKOS_INLINE_FUNCTION Real unshift(Real stored, int i) {
  if constexpr (C::Storage::shifted) return stored + weight<FL, Real>(i);
  else                               return stored;
}

static inline int opp_of(int i) { return i == 0 ? 0 : ((i % 2) ? i + 1 : i - 1); }

struct Link { Index n; int i; };

//------------------------------------------------------------------------------
// Legacy binary VTK, STRUCTURED_POINTS, BIG-ENDIAN because the format says so
// and carries no byte-order field to say otherwise.
//
// VORTICITY AND CURRENT ARE THE PAIR THAT SHOWS THE PHYSICS. Vorticity alone
// shows a wake dying; it does not show WHY. The current density j_z = curl b is
// where the Joule dissipation lives, and the two pictures side by side are the
// braking and the braked.
//------------------------------------------------------------------------------
static void put_be(std::FILE* f, float v) {
  unsigned char* p = reinterpret_cast<unsigned char*>(&v);
  const unsigned char b[4] = {p[3], p[2], p[1], p[0]};
  std::fwrite(b, 1, 4, f);
}

template <class Solid>
static void write_vtk(const std::string& path, Index nx, Index ny, const Domain& d,
                      const std::vector<double>& ux, const std::vector<double>& uy,
                      const std::vector<double>& bx, const std::vector<double>& by,
                      double B0, Solid solid) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { std::printf("  cannot open %s\n", path.c_str()); return; }
  std::fprintf(f, "# vtk DataFile Version 3.0\nmhd cylinder\nBINARY\n"
                  "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n"
                  "ORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA %lld\n",
               int(nx), int(ny), (long long)(std::size_t(nx) * std::size_t(ny)));
  auto at = [&](Index x, Index y) { return std::size_t(d.id(x, y, 0)); };
  auto interior = [&](Index x, Index y) {
    return !(x == 0 || x == nx - 1 || y == 0 || y == ny - 1) &&
           !solid(x, y) && !solid(x + 1, y) && !solid(x - 1, y) &&
           !solid(x, y + 1) && !solid(x, y - 1);
  };
  std::fprintf(f, "SCALARS vorticity float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      double w = 0;
      if (interior(x, y))
        w = 0.5 * (uy[at(x + 1, y)] - uy[at(x - 1, y)]) -
            0.5 * (ux[at(x, y + 1)] - ux[at(x, y - 1)]);
      put_be(f, float(w));
    }
  // j_z = d(b_y)/dx - d(b_x)/dy. The stencil runs over the magnetic field,
  // which is defined inside a `cond` cylinder as well -- but the solid mask is
  // used all the same, so the same picture is comparable between -cyl modes.
  std::fprintf(f, "\nSCALARS current float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      double j = 0;
      if (interior(x, y))
        j = 0.5 * (by[at(x + 1, y)] - by[at(x - 1, y)]) -
            0.5 * (bx[at(x, y + 1)] - bx[at(x, y - 1)]);
      put_be(f, float(j));
    }
  // The INDUCED field only: the applied B0 is a constant of the setup and
  // plotting it swamps the part that responds to the flow.
  std::fprintf(f, "\nSCALARS binduced float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const double dx = bx[at(x, y)], dy = by[at(x, y)] - B0;
      put_be(f, solid(x, y) ? 0.0f : float(std::sqrt(dx * dx + dy * dy)));
    }
  std::fprintf(f, "\nSCALARS solid float 1\nLOOKUP_TABLE default\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) put_be(f, solid(x, y) ? 1.0f : 0.0f);
  std::fprintf(f, "\nVECTORS velocity float\n");
  for (Index y = 0; y < ny; ++y)
    for (Index x = 0; x < nx; ++x) {
      const bool sd = solid(x, y);
      put_be(f, sd ? 0.0f : float(ux[at(x, y)]));
      put_be(f, sd ? 0.0f : float(uy[at(x, y)]));
      put_be(f, 0.0f);
    }
  std::fclose(f);
}

struct Opts {
  double umax = 0.05, Re = 100.0, prm = 0.01, seed = 1e-2;
  Index D = 40;
  std::size_t transient = 60000, avg = 60000, probe = 250, vtkevery = 0, vtkfrom = 0;
  bool nocyl = false;
  std::size_t trace = 0;
  std::size_t binevery = 0, binfrom = 0;
  std::string vtkdir, bindir, cyl = "cond", magout = "dirichlet", outlet = "press";
};

//------------------------------------------------------------------------------
// One Hartmann number. Templated on the collision operator for the same reason
// square_cylinder.cpp is: the operator is a compile-time policy of FluidSolver,
// so a runtime branch would need both solvers instantiated at once.
//------------------------------------------------------------------------------
template <class C>
static void run_one(const Opts& o, double Ha, int& status) {
  const Index D = o.D;
  // BOUNCE-BACK CHANNEL WALLS, and the first version got this wrong. They were
  // regularised on-node walls, to put them on the same plane as their magnetic
  // Dirichlet condition -- and CLAUDE.md records that regularised walls are not
  // mass conserving, because BC3 overwrites populations. Two columns of them at
  // an inlet and an outlet is what square_cylinder.cpp lives with; two ROWS of
  // them 750 cells long is not the same thing, and the measured drift was
  // +21 % of the total mass in ten thousand steps, which the servo cannot chase
  // and which scales every force by 1.21. Halfway bounce-back conserves mass
  // exactly. It costs the half-cell offset priced in the banner -- about 2 % in
  // the wall layer at this H -- four diameters away from the cylinder, which is
  // the right trade against 21 % on the density everywhere.
  const Index ny = 8 * D + 2, H = ny - 2;        // H fluid rows between two solid
  const Index nx = 25 * D;
  const double cx = 8.0 * double(D) + 0.5;       // half-integer: a symmetric circle
  const double cy = 0.5 * double(ny) - 0.5;      // the midplane of rows 1..H
  const double R  = 0.5 * double(D);
  const Real nu   = Real(o.umax * double(D) / o.Re);
  const Real eta  = Real(double(nu) / o.prm);
  const Real B0   = Real(Ha * std::sqrt(double(nu) * double(eta)) / double(D));
  const double Nint = (o.Re > 0) ? Ha * Ha / o.Re : 0.0;

  const bool nocyl = o.nocyl;
  auto is_cyl = [&](Index x, Index y) {
    if (nocyl) return false;
    const double dx = double(x) - cx, dy = double(y) - cy;
    return dx * dx + dy * dy < R * R;
  };
  auto solid = [&](Index x, Index y) { return is_cyl(x, y); };

  Domain d(nx, ny, 1, false, false, false);

  MagneticBGK<ML> mc; mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
  C fc;
  fc.omega = C::omega_from_viscosity(nu);
  fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, C> fl(d, fc);

  fl.set_geometry([&](Index x, Index y, Index) -> CellType {
    if (y == 0 || y == ny - 1) return Solid;           // halfway bounce-back
    if (is_cyl(x, y)) return Solid;                    // bounce-back: MEM needs it
    if (x == 0 || x == nx - 1) return RegWall;
    return Fluid;
  });
  // The parabola vanishes on the two wall PLANES at y = 0.5 and y = H + 0.5.
  const Real orho0 = Real(1);
  auto prof = [&](Index y) {
    const double yy = double(y);
    return 4.0 * o.umax * (yy - 0.5) * (double(H) + 0.5 - yy) /
           (double(H) * double(H));
  };
  using WS = typename decltype(fl)::WallSpec;
  fl.set_regularized_walls([&](Index x, Index y, Index) -> WS {
    if (is_cyl(x, y)) return WS{};
    if (x == 0)      return WS{NrmXm, Real(prof(y)), Real(0), Real(0), Real(1)};
    // THE OUTLET IS A MASS BUDGET DECISION HERE, not a stability one.
    // NrmOutEq imposes rho AND all three velocity components, which
    // over-determines the boundary and makes it a source:
    // square_cylinder.cpp measured 8.3e-2 of the total mass against 2.6e-4 for
    // NrmOutXp, whose normal velocity comes from the inverted closure instead.
    // Measured HERE at D = 10, where the boundary columns are 0.8 % of the
    // cells instead of 0.1 %, NrmOutEq drifted +20.7 % in fourteen thousand
    // steps -- far past what the servo can chase, and a 1.21x scaling on every
    // force. NrmOutXp is the default for that reason; -outlet eq is the
    // fallback if a wake ever needs it, which is the trade square_cylinder had
    // to make at Re = 300 and this case at Re = 100 does not.
    if (x == nx - 1) {
      if (o.outlet == "eq")
        return WS{NrmOutEq, Real(0), Real(0), Real(0), Real(orho0)};
      return WS{NrmOutXp, Real(0), Real(0), Real(0), Real(orho0)};
    }
    return WS{};
  });
  // The FD corner stress knows nothing about the Lorentz force -- see
  // validation/shercliff.cpp, where it leaked mass linearly at a duct edge.
  fl.set_fd_corners(false);

  const Real B0c = B0;
  const bool insul = (o.cyl == "insul");
  if (insul) mag.set_geometry([&](Index x, Index y, Index) { return is_cyl(x, y); });
  const bool magrad = (o.magout == "grad");
  using WB = typename decltype(mag)::WallB;
  mag.set_moment_walls([&](Index x, Index y, Index) -> WB {
    if (insul && is_cyl(x, y)) return WB{};
    if (x == nx - 1 && magrad) { WB w; w.is_wall = true; w.outflow = true; return w; }
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1)
      return WB{true, Real(0), B0c, Real(0)};
    if (insul) {
      // A fluid node with any neighbour inside the cylinder rings it.
      for (int i = 1; i < FL::Q; ++i)
        if (is_cyl(x + cvel<FL>(i, 0), y + cvel<FL>(i, 1)))
          return WB{true, Real(0), B0c, Real(0)};
    }
    return WB{};
  });
  mag.initialize_field(KOKKOS_LAMBDA(Index) {
    Kokkos::Array<Real, 3> b; b[0] = Real(0); b[1] = B0c; b[2] = Real(0);
    return b;
  });
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());

  // Start from the developed channel -- the inlet profile is imposed exactly,
  // so only the wake has to develop. A transverse blob behind the cylinder
  // breaks the mirror symmetry the shedding mode is orthogonal to.
  const double umax_c = o.umax, seed_c = o.seed, Hc = double(H), Dc = double(D);
  const double cxc = cx, cyc = cy;
  fl.initialize_field(KOKKOS_LAMBDA(Index n) {
    Index x, y, z; d.coords(n, x, y, z); (void)z;
    const double dx0 = double(x) - cxc, dy0 = double(y) - cyc;
    const bool sd = dx0 * dx0 + dy0 * dy0 < (0.5 * Dc) * (0.5 * Dc);
    const double yy = double(y);
    const bool wall = (y == 0 || y == Index(Hc) + 1);
    const double u = (sd || wall)
                         ? 0.0
                         : 4.0 * umax_c * (yy - 0.5) * (Hc + 0.5 - yy) / (Hc * Hc);
    const double xs = double(x) - (cxc + 1.5 * Dc);
    const double v = sd ? 0.0
                        : seed_c * umax_c *
                              std::exp(-(xs * xs + dy0 * dy0) / (0.5 * Dc * Dc));
    return FlowState{Real(1), Real(u), Real(v), Real(0)};
  });

  const bool servo = (o.outlet == "eq");
  const Real m0 = fl.total_mass();
  // MASS SERVO. NrmOutEq imposes rho AND u, which over-determines the boundary
  // and makes it a source: square_cylinder.cpp measured 8.3e-2 drift and a drag
  // exactly 8.3 % high as a result, the whole discrepancy being the density.
  double orho = 1.0;

  // Boundary links of the CYLINDER only.
  std::vector<Link> links;
  for (Index y = 1; y < ny - 1; ++y)
    for (Index x = 1; x < nx - 1; ++x) {
      if (is_cyl(x, y)) continue;
      for (int i = 0; i < FL::Q; ++i)
        if (is_cyl(x + cvel<FL>(i, 0), y + cvel<FL>(i, 1)))
          links.push_back(Link{d.id(x, y, 0), i});
    }
  auto forces = [&](double& Fx, double& Fy) {
    Fx = Fy = 0;
    auto g = fl.gather_populations();
    auto hg = Kokkos::create_mirror_view_and_copy(HostSpace{}, g);
    for (const Link& lk : links) {
      const int io = opp_of(lk.i);
      const double f = double(unshift<C>(hg(lk.n, io), io));
      Fx += 2.0 * cvel<FL>(lk.i, 0) * f;
      Fy += 2.0 * cvel<FL>(lk.i, 1) * f;
    }
  };
  std::size_t vframe = 0, bframe = 0;
  // A COMPACT DUMP FOR THE ANIMATION, in doc/fig/mkpng.py's own format: two
  // little-endian int32 then nx*ny float32. A full VTK frame of this domain is
  // 9 MB and an animation wants a hundred of them per Hartmann number; a
  // cropped single-field dump is 0.8 MB. The crop is the cylinder and thirteen
  // diameters of wake, which is the whole picture -- the inlet approach and the
  // last stretch before the outlet carry nothing to look at.
  const Index bx0 = Index(std::max(0.0, cx - 3.0 * double(D)));
  const Index bx1 = Index(std::min(double(nx), cx + 13.0 * double(D)));
  auto bindump = [&]() {
    if (o.bindir.empty()) return;
    fl.compute_macroscopic(); mag.compute_field();
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    auto hbx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    auto hby = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    const std::int32_t w = std::int32_t(bx1 - bx0), h = std::int32_t(ny);
    auto emit = [&](const char* tag, bool current) {
      char nm[512];
      std::snprintf(nm, sizeof nm, "%s/cyl_ha%03d_%s%04zu.bin", o.bindir.c_str(),
                    int(std::lround(Ha)), tag, bframe);
      std::FILE* f = std::fopen(nm, "wb");
      if (!f) return;
      std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);
      for (Index y = 0; y < ny; ++y)
        for (Index x = bx0; x < bx1; ++x) {
          float v = 0.0f;
          const bool edge = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
          // Solid cells and the cells whose stencil reaches one get a NaN, so
          // the renderer can paint the body rather than a ring of false shear.
          if (edge || is_cyl(x, y) || is_cyl(x + 1, y) || is_cyl(x - 1, y) ||
              is_cyl(x, y + 1) || is_cyl(x, y - 1))
            v = std::nanf("");
          else if (current)
            v = float(0.5 * (double(hby(d.id(x + 1, y, 0))) - double(hby(d.id(x - 1, y, 0)))) -
                      0.5 * (double(hbx(d.id(x, y + 1, 0))) - double(hbx(d.id(x, y - 1, 0)))));
          else
            v = float(0.5 * (double(hy(d.id(x + 1, y, 0))) - double(hy(d.id(x - 1, y, 0)))) -
                      0.5 * (double(hx(d.id(x, y + 1, 0))) - double(hx(d.id(x, y - 1, 0)))));
          std::fwrite(&v, 4, 1, f);
        }
      std::fclose(f);
    };
    emit("w", false);
    if (B0 > Real(0)) emit("j", true);
    ++bframe;
  };
  auto dump = [&]() {
    if (o.vtkdir.empty()) return;
    fl.compute_macroscopic(); mag.compute_field();
    auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    auto hbx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    auto hby = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    std::vector<double> vx(std::size_t(d.n_padded)), vy(std::size_t(d.n_padded)),
                        wx(std::size_t(d.n_padded)), wy(std::size_t(d.n_padded));
    for (Index n = 0; n < d.n_padded; ++n) {
      vx[n] = double(hx(n)); vy[n] = double(hy(n));
      wx[n] = double(hbx(n)); wy[n] = double(hby(n));
    }
    char nm[512];
    std::snprintf(nm, sizeof nm, "%s/mhdcyl_ha%03d_%04zu.vtk",
                  o.vtkdir.c_str(), int(std::lround(Ha)), vframe);
    write_vtk(nm, nx, ny, d, vx, vy, wx, wy, double(B0), solid);
    ++vframe;
  };

  const double cnorm = 0.5 * o.umax * o.umax * double(D);
  // A WAKE PROBE, because the lift is the wrong instrument once the field wins.
  // When shedding is suppressed the lift amplitude goes to zero and its zero
  // crossings become round-off noise, which a frequency estimator will happily
  // convert into a Strouhal number. The transverse velocity one cylinder
  // diameter off the centreline, five diameters downstream, is the same signal
  // with a floor that can be quoted.
  const Index px = Index(std::lround(cx + 5.0 * double(D)));
  const Index py = Index(std::lround(cy + 1.0 * double(D)));
  const Index pn = d.id(px, py, 0);

  std::printf("  Ha %5.1f   N %7.4f   B0 %10.3e   delta_Ha %6.2f cells\n",
              Ha, Nint, double(B0), Ha > 0 ? double(D) / Ha : 999.0);
  std::fflush(stdout);

  std::vector<double> cds, cls, vps; std::vector<std::size_t> ts;
  const double u_panic = 8.0 * o.umax;
  bool ok = true;
  std::size_t t = 0;
  for (; t < o.transient + o.avg && ok; ++t) {
    mag.compute_field(); fl.step(true); mag.step(true);
    if (servo && (t + 1) % 500 == 0 && t > 2000) {
      const double err = double(fl.total_mass()) / double(m0) - 1.0;
      double dr = 0.6 * err;
      if (dr >  5e-4) dr =  5e-4;
      if (dr < -5e-4) dr = -5e-4;
      orho -= dr;
      if (orho < 0.9) orho = 0.9;
      if (orho > 1.1) orho = 1.1;
      fl.set_wall_density(Real(orho));
    }
    if (o.vtkevery && (t + 1) >= o.vtkfrom && (t + 1) % o.vtkevery == 0) dump();
    if (o.binevery && (t + 1) >= o.binfrom && (t + 1) % o.binevery == 0) bindump();
    if ((t + 1) % 1000 == 0) {
      // WATCH THE PRECURSOR, NOT THE CORPSE. NaN spreads about a cell a step,
      // so by the first check after onset the location of the "first" bad node
      // is scan order and nothing else.
      auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      auto hy = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
      double um = 0; Index mx = 0, my = 0; bool fin = true;
      for (Index yy = 1; yy < ny - 1; ++yy)
        for (Index xx = 0; xx < nx; ++xx) {
          const Index m = d.id(xx, yy, 0);
          const double a = double(hx(m)), b = double(hy(m));
          if (!std::isfinite(a) || !std::isfinite(b)) { fin = false; continue; }
          const double sp = std::sqrt(a * a + b * b);
          if (sp > um) { um = sp; mx = xx; my = yy; }
        }
      // WHICH FIELD RUNS AWAY FIRST. The fluid and the induction equation are
      // coupled both ways, so "the velocity blew up" does not say which one
      // went. Track the induced field on the same interval and print both.
      double bm = 0; Index bmx = 0, bmy = 0;
      {
        mag.compute_field();
        auto hbx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
        auto hby = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
        for (Index yy = 0; yy < ny; ++yy)
          for (Index xx = 0; xx < nx; ++xx) {
            const Index m = d.id(xx, yy, 0);
            const double ax = double(hbx(m)), ay = double(hby(m)) - double(B0);
            if (!std::isfinite(ax) || !std::isfinite(ay)) continue;
            const double sp = std::sqrt(ax * ax + ay * ay);
            if (sp > bm) { bm = sp; bmx = xx; bmy = yy; }
          }
      }
      if (o.trace && (t + 1) % o.trace == 0) {
        std::printf("       t %7zu  max|u| %.5f at (%d,%d)   max|b|/B0 %.5f at "
                    "(%d,%d)\n", t + 1, um, int(mx), int(my),
                    B0 > Real(0) ? bm / double(B0) : 0.0, int(bmx), int(bmy));
        std::fflush(stdout);
      }
      if (!fin || um > u_panic) {
        std::printf("       BLOW-UP at step %zu: max|u| = %.4f at (%d,%d)%s   "
                    "max|b_ind|/B0 = %.4g at (%d,%d)\n",
                    t + 1, um, int(mx), int(my), fin ? "" : "  (NaN present)",
                    B0 > Real(0) ? bm / double(B0) : 0.0, int(bmx), int(bmy));
        std::printf("       cylinder centre (%.1f,%.1f) R %.1f; outlet x=%d; "
                    "walls y=0,%d\n", cx, cy, R, int(nx - 1), int(ny - 1));
        if (o.vtkevery) { dump(); std::printf("       wrote the failing field\n"); }
        std::fflush(stdout);
        ok = false; break;
      }
    }
    if ((t + 1) <= o.transient || (t + 1) % o.probe) continue;
    double Fx, Fy; forces(Fx, Fy);
    if (!std::isfinite(Fx)) { ok = false; break; }
    auto hv = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.uy());
    cds.push_back(Fx / cnorm); cls.push_back(Fy / cnorm);
    vps.push_back(double(hv(pn))); ts.push_back(t + 1);
    if ((t + 1) % 25000 == 0) {
      std::printf("       step %zu   C_d %.4f   C_l %+.4f   v_probe %+.3e\n",
                  t + 1, Fx / cnorm, Fy / cnorm, double(hv(pn)));
      std::fflush(stdout);
    }
  }

  double m = 0, lo = 1e30, hi = -1e30, lm = 0, lmin = 1e30, lmax = -1e30;
  double vm = 0, vmin = 1e30, vmax = -1e30;
  for (std::size_t k = 0; k < cds.size(); ++k) {
    m += cds[k]; lm += cls[k]; vm += vps[k];
    lo = std::min(lo, cds[k]); hi = std::max(hi, cds[k]);
    lmin = std::min(lmin, cls[k]); lmax = std::max(lmax, cls[k]);
    vmin = std::min(vmin, vps[k]); vmax = std::max(vmax, vps[k]);
  }
  if (!cds.empty()) { m /= double(cds.size()); lm /= double(cls.size()); vm /= double(vps.size()); }
  // Strouhal from UPWARD zero crossings of the probe about its own mean,
  // linearly interpolated: the signal is smooth at a crossing and flat at a
  // peak, so crossings are the quieter estimator.
  std::vector<double> cross;
  for (std::size_t k = 1; k < vps.size(); ++k)
    if (vps[k - 1] - vm < 0 && vps[k] - vm >= 0) {
      const double a = vps[k - 1] - vm, b = vps[k] - vm;
      cross.push_back(double(ts[k - 1]) + (0.0 - a) / (b - a) * double(ts[k] - ts[k - 1]));
    }
  double St = 0;
  if (cross.size() >= 2) {
    const double period = (cross.back() - cross.front()) / double(cross.size() - 1);
    St = o.umax > 0 ? double(D) / (period * o.umax) : 0.0;
  }
  // The amplitude floor. Below this the crossings are numerical noise and the
  // frequency they imply is meaningless, so it is printed as suppressed rather
  // than as a number.
  const double amp = vmax - vmin, floor_amp = 1e-3 * o.umax;
  const double mrel = double(fl.total_mass()) / double(m0) - 1.0;
  double bmax = 0;
  {
    mag.compute_field();
    auto hbx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
    auto hby = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.By());
    for (Index yy = 1; yy < ny - 1; ++yy)
      for (Index xx = 1; xx < nx - 1; ++xx) {
        if (is_cyl(xx, yy)) continue;
        const Index n = d.id(xx, yy, 0);
        const double ax = double(hbx(n)), ay = double(hby(n)) - double(B0);
        const double s = std::sqrt(ax * ax + ay * ay);
        if (std::isfinite(s) && s > bmax) bmax = s;
      }
  }
  std::printf("       C_d  mean %.4f  min %.4f  max %.4f  |  C_l mean %+.4f  amp %.4f\n",
              m, lo, hi, lm, lmax - lmin);
  if (amp > floor_amp)
    std::printf("       St %.4f over %.1f cycles   probe amp %.3e (%.2f%% of u_max)\n",
                St, cross.size() >= 2 ? double(cross.size() - 1) : 0.0,
                amp, 100.0 * amp / o.umax);
  else
    std::printf("       SHEDDING SUPPRESSED   probe amp %.3e < %.3e floor\n",
                amp, floor_amp);
  std::printf("       max |b_induced| / B0 %.4f   dmass %+.3e   %zu samples%s\n",
              B0 > Real(0) ? bmax / double(B0) : 0.0, mrel, cds.size(),
              ok ? "" : "   NON-FINITE");
  if (vframe) std::printf("       wrote %zu VTK frames to %s\n", vframe, o.vtkdir.c_str());
  if (bframe) std::printf("       wrote %zu binary frames to %s\n", bframe, o.bindir.c_str());
  std::fflush(stdout);
  if (!ok) status = 1;
}

//------------------------------------------------------------------------------
// The self-test: momentum exchange against a force it cannot get wrong, with
// the MHD operator and a uniform field in the loop. See the banner.
//------------------------------------------------------------------------------
static void selftest(double Ha_t, bool transverse, int& status, bool hard,
                     Index Hh = 32) {
  const Index nx = 8, ny = Hh + 2;
  const Real G = Real(1e-6), nu = Real(0.1), eta = nu;
  const Real B0 = Real(Ha_t * double(nu) / (0.5 * double(Hh)));
  Domain d(nx, ny, 1, true, false, false);
  MagneticBGK<ML> mc; mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
  MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
  CollG coll;
  coll.omega = CollG::omega_from_viscosity(nu);
  coll.forcing = Guo{G, Real(0), Real(0)};
  coll.Bx = mag.Bx(); coll.By = mag.By(); coll.Bz = mag.Bz();
  FluidSolver<FL, EsotericPull<FL>, CollG> fl(d, coll);
  fl.set_geometry([&](Index, Index y, Index) -> CellType {
    return (y == 0 || y == ny - 1) ? Solid : Fluid;
  });
  const Real bx0 = transverse ? Real(0) : B0, by0 = transverse ? B0 : Real(0);
  using WB = typename decltype(mag)::WallB;
  mag.set_moment_walls([&](Index, Index y, Index) -> WB {
    if (y == 0 || y == ny - 1) return WB{true, bx0, by0, Real(0)};
    return WB{};
  });
  mag.initialize_field(KOKKOS_LAMBDA(Index) {
    Kokkos::Array<Real, 3> b; b[0] = bx0; b[1] = by0; b[2] = Real(0);
    return b;
  });
  fl.initialize(Real(1));
  mag.set_velocity(fl.ux(), fl.uy(), fl.uz());
  std::vector<Link> links;
  for (Index y = 1; y < ny - 1; ++y)
    for (Index x = 0; x < nx; ++x)
      for (int i = 0; i < FL::Q; ++i) {
        const Index qy = y + cvel<FL>(i, 1);
        if (qy != 0 && qy != ny - 1) continue;
        links.push_back(Link{d.id(x, y, 0), i});
      }
  // RUN TO CONVERGENCE, NOT TO A FIXED STEP COUNT. The first version marched
  // a flat 60000 steps, which is six diffusive times at H = 32 and two fifths
  // of one at H = 128 -- so a ladder in H built on it would have been measuring
  // how far each rung got, and reporting that as a boundary error.
  std::size_t taken = 0;
  {
    const std::size_t probe = 2000, cap = 4000000;
    double prev = 0;
    for (std::size_t t = 0; t < cap; t += probe) {
      for (std::size_t k = 0; k < probe; ++k) {
        mag.compute_field(); fl.step(true); mag.step(true);
      }
      taken += probe;
      fl.compute_macroscopic();
      auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
      const double cur = double(hu(d.id(nx / 2, ny / 2, 0)));
      if (!std::isfinite(cur)) break;
      if (t > 0 && std::abs(cur - prev) < 1e-12 * (std::abs(cur) + 1e-30)) break;
      prev = cur;
    }
  }
  // THE FORCE BALANCE CANNOT SEE THE FIELD, SO IT CANNOT BE THE WHOLE TEST.
  // Steady state pins the wall force at G x N_fluid by momentum conservation
  // whatever the profile is -- and measured, the three cases below agree to the
  // last bits. That is the right answer and it is also completely blind: a dead
  // magnetic coupling would produce it too, which is precisely the failure mode
  // CLAUDE.md records for GPU/'s ehd_cavity, where a null velocity pointer left
  // the charge un-advected and every diagnostic still looked reasonable.
  //
  // So measure something the field DOES move. At Ha = 5 the Hartmann profile is
  // flattened to 39 % of the parabolic peak, and the exact centreline value is
  //     u_c = (G L / B0) sqrt(eta/nu) coth(Ha) (1 - sech Ha),
  // which is validation/hartmann.cpp's solution evaluated at the midplane. The
  // deviation is NOT round-off: the magnetic Dirichlet condition sits ON nodes
  // 0 and ny-1 while halfway bounce-back puts the no-slip planes at 0.5 and
  // H + 0.5, so the two boundaries bound channels of different width. That is
  // the half-cell pairing validation/hartmann3d.cpp warns about, and this line
  // is its price on the configuration this file actually runs.
  fl.compute_macroscopic();
  double uc = 0, uc_ref = 0;
  {
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
    // The midplane sits between rows ny/2 - 1 and ny/2 for an even channel.
    uc = 0.5 * (double(hu(d.id(nx / 2, ny / 2, 0))) +
                double(hu(d.id(nx / 2, ny / 2 - 1, 0))));
    const double Lh = 0.5 * double(Hh);
    if (transverse && Ha_t > 0)
      uc_ref = (double(G) * Lh / double(B0)) * std::sqrt(double(eta) / double(nu)) *
               (1.0 / std::tanh(Ha_t)) * (1.0 - 1.0 / std::cosh(Ha_t));
    else
      uc_ref = double(G) * Lh * Lh / (2.0 * double(nu));      // Poiseuille
  }
  double Fx = 0, Fy = 0;
  {
    auto g = fl.gather_populations();
    auto hg = Kokkos::create_mirror_view_and_copy(HostSpace{}, g);
    for (const Link& lk : links) {
      const int io = opp_of(lk.i);
      const double f = double(unshift<CollG>(hg(lk.n, io), io));
      Fx += 2.0 * cvel<FL>(lk.i, 0) * f;
      Fy += 2.0 * cvel<FL>(lk.i, 1) * f;
    }
  }
  const double want = double(G) * double(nx) * double(Hh);
  const double rel = std::abs(Fx - want) / want;
  // The transverse component must vanish by symmetry whatever the field does.
  std::printf("    Ha %5.1f  %-10s H %3d  F_x %-13.7e  want %-13.7e  rel %8.2e  "
              "F_y %+8.1e  %zu steps  %s\n",
              Ha_t, transverse ? "transverse" : "streamwise", int(Hh), Fx, want,
              rel, Fy, taken,
              hard ? (rel < 1e-6 ? "PASS" : "FAIL") : "(measurement)");
  std::printf("                            u_c %-13.7e  want %-13.7e  %+7.2f %%  "
              "(%s)\n",
              uc, uc_ref, 100.0 * (uc / uc_ref - 1.0),
              (transverse && Ha_t > 0) ? "Hartmann -- the field IS coupled"
                                       : "Poiseuille -- no Lorentz force");
  if (hard && !(rel < 1e-6)) status = 1;
  // A transverse field that leaves the profile parabolic is a dead coupling,
  // not a small Hartmann number: at Ha = 5 the exact centreline is 39 % of the
  // Poiseuille value, so a 2x discrepancy is unmistakable.
  if (transverse && Ha_t > 0) {
    const double poise = double(G) * (0.5 * double(Hh)) * (0.5 * double(Hh)) /
                         (2.0 * double(nu));
    if (std::abs(uc / poise - 1.0) < 0.1) {
      std::printf("      FAIL: the profile is still parabolic -- the Lorentz "
                  "coupling is not acting\n");
      status = 1;
    }
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Opts o;
    std::vector<double> Has = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0};
    std::string op = "cm";
    bool selftest_only = false, skip_selftest = false, pair = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-u"    && i + 1 < argc) o.umax = std::atof(argv[++i]);
      if (a == "-re"   && i + 1 < argc) o.Re = std::atof(argv[++i]);
      if (a == "-prm"  && i + 1 < argc) o.prm = std::atof(argv[++i]);
      if (a == "-d"    && i + 1 < argc) o.D = Index(std::atol(argv[++i]));
      if (a == "-seed" && i + 1 < argc) o.seed = std::atof(argv[++i]);
      if (a == "-transient" && i + 1 < argc) o.transient = std::size_t(std::atol(argv[++i]));
      if (a == "-avg"  && i + 1 < argc) o.avg = std::size_t(std::atol(argv[++i]));
      if (a == "-probe" && i + 1 < argc) o.probe = std::size_t(std::atol(argv[++i]));
      if (a == "-vtk"  && i + 1 < argc) o.vtkdir = argv[++i];
      if (a == "-vtkevery" && i + 1 < argc) o.vtkevery = std::size_t(std::atol(argv[++i]));
      if (a == "-vtkfrom"  && i + 1 < argc) o.vtkfrom = std::size_t(std::atol(argv[++i]));
      if (a == "-bin"      && i + 1 < argc) o.bindir = argv[++i];
      if (a == "-binevery" && i + 1 < argc) o.binevery = std::size_t(std::atol(argv[++i]));
      if (a == "-binfrom"  && i + 1 < argc) o.binfrom = std::size_t(std::atol(argv[++i]));
      if (a == "-cyl"  && i + 1 < argc) o.cyl = argv[++i];
      if (a == "-magout" && i + 1 < argc) o.magout = argv[++i];
      if (a == "-outlet" && i + 1 < argc) o.outlet = argv[++i];
      if (a == "-nocyl") o.nocyl = true;
      if (a == "-trace" && i + 1 < argc) o.trace = std::size_t(std::atol(argv[++i]));
      if (a == "-op"   && i + 1 < argc) op = argv[++i];
      if (a == "-selftest") selftest_only = true;
      if (a == "-noselftest") skip_selftest = true;
      if (a == "-pair") { pair = true; selftest_only = true; }
      if (a == "-has" && i + 1 < argc) {
        Has.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { if (!t.empty()) Has.push_back(std::atof(t.c_str())); t.clear(); } else t += c; }
      }
    }
    std::printf("Flow past a circular cylinder in a transverse magnetic field  "
                "-- DEMONSTRATOR\n");
    std::printf("backend %s   precision %s   %s fluid + %s magnetic\n\n",
                ExecSpace::name(), precision_name(), FL::name, ML::name);

    if (!skip_selftest) {
      std::printf("  SELF-TEST  momentum exchange against a known force, with a "
                  "uniform field in the loop\n");
      std::printf("    periodic channel H=32, G=1e-6, 256 fluid nodes; the walls "
                  "must carry exactly G x N_fluid\n");
      selftest(0.0, false, status, true);     // no field at all
      selftest(5.0, false, status, true);     // field along the flow: no Lorentz force
      // Transverse: the net Lorentz force integrates to zero in the continuum,
      // so the same balance should hold -- but the magnetic Dirichlet plane and
      // the bounce-back plane differ by half a cell, so it does not exactly.
      // Reported, not asserted. See the banner.
      selftest(5.0, true, status, false);
      std::printf("\n");
      if (pair) {
        // PRICE THE PAIRING RATHER THAN ASSERT IT. The transverse row above is
        // out by about 10 %, and the banner's claim is that this is the half
        // cell between the two wall planes and not an error in the coupling.
        // A half-cell offset in a channel of height H is an O(1/H) effect, so
        // the claim is falsifiable: double H and the deviation must halve. If
        // it does not, the explanation is wrong.
        std::printf("  PAIRING LADDER  the same transverse case at three "
                    "channel heights\n");
        for (const Index Hh : {Index(32), Index(64), Index(128)})
          selftest(5.0, true, status, false, Hh);
        std::printf("\n");
      }
      std::fflush(stdout);
    }
    if (selftest_only) { Kokkos::finalize(); return status; }

    const Index ny = 8 * o.D + 2;
    std::printf("  D %d   channel H %d (blockage %.4f)   nx %d   Re %.0f (peak)   "
                "Pr_m %.2f\n", int(o.D), int(ny - 2), double(o.D) / double(ny - 2),
                int(25 * o.D), o.Re, o.prm);
    std::printf("  u_max %.4f (Ma %.4f)   nu %.5f   tau %.5f   operator %s   "
                "cylinder %s   outlet %s/%s\n",
                o.umax, o.umax / 0.5773502692, o.umax * double(o.D) / o.Re,
                0.5 + 3.0 * o.umax * double(o.D) / o.Re,
                op == "cm" ? "central moments" : "BGK", o.cyl.c_str(),
                o.outlet.c_str(), o.magout.c_str());
    std::printf("  transient %zu steps, averaging %zu   (one convective time is "
                "%.0f steps)\n\n", o.transient, o.avg, double(25 * o.D) / o.umax);
    for (const double Ha : Has) {
      if (op == "bgk") run_one<CollB>(o, Ha, status);
      else             run_one<CollC>(o, Ha, status);
      std::printf("\n");
    }
  }
  Kokkos::finalize();
  return status;
}
