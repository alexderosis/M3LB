#pragma once
//==============================================================================
//  Free-surface flow: a sharp interface, and only the liquid.
//
//  THE THIRD ROUTE TO A MULTIPHASE FLOW IN THIS CODE, and the one that changes
//  the bargain rather than the parameters. PhaseFieldSolver.hpp and
//  ColourGradientSolver.hpp both resolve BOTH fluids, and everything difficult
//  about them follows from that: the pressure-gauge conditioning at a density
//  ratio, the alpha interpolation, the recolouring. This one does not resolve
//  the gas at all. The gas is a VOID held at a prescribed pressure -- it has no
//  populations, no velocity and no dynamics -- and the liquid meets it across a
//  boundary condition rather than across a diffuse interface.
//
//  What that buys is the thing the other two fight for. There is no density
//  ratio here because there is no second density: the ratio is infinite by
//  construction, and it costs nothing, because the expensive half of the problem
//  was deleted rather than discretised. It is the standard tool for violent
//  free-surface flow -- dam break, sloshing, filling, splashing -- and it is
//  useless for anything where the gas does work: a rising bubble, a driven
//  gas-liquid shear layer, anything with a Weber number the gas participates in.
//
//      module            interface   fluids resolved   density ratio
//      phase field       diffuse     both              ~100, with the gauge problem
//      colour gradient   diffuse     both              1000 in principle
//      free surface      sharp       liquid only       infinite, by construction
//
//  PROVENANCE, STATED PLAINLY. This follows the standard formulation of Korner,
//  Thies, Hofmann, Theobald & Rude, J. Stat. Phys. 121, 179 (2005), and the
//  refinements in Thurey's and Rude's later work. Unlike the colour-gradient
//  module there is no paper in SomeRefs/ to check equation by equation, so
//  nothing here cites an equation number and every choice below is justified
//  from the mechanics rather than by reference. Where the literature offers a
//  refinement this code does not implement, it says so.
//
//==============================================================================
//  THE STATE. Every cell is Gas, Interface or Fluid (or a wall, or halo). Fluid
//  cells are ordinary LBM cells. Gas cells hold nothing. Interface cells are the
//  entire method: they carry an explicit MASS m alongside their populations, and
//  a fill level
//
//      epsilon = m / rho,
//
//  which is 1 when the cell is full and 0 when it is empty. The interface is a
//  closed, one-cell-thick shell separating Fluid from Gas, and keeping it closed
//  through arbitrary topology change is most of the work.
//
//  MASS IS ADVECTED, NOT A MOMENT, and that is the sharpest difference from
//  everything else in this tree. Elsewhere density is sum_i f_i and conservation
//  is a property of the collision operator, checkable by algebra. Here mass moves
//  because populations cross faces, and the exchange between two cells is
//
//      dm_i(x) = f_ibar(x + c_i) - f_i(x)          both cells Fluid or Interface,
//
//  scaled by the mean fill level when both are Interface, because a half-full
//  cell can only exchange half as much. Nothing enforces conservation but the
//  antisymmetry of that expression: what x sends to its neighbour is exactly
//  what the neighbour receives from x. That is why validation checks total mass
//  to round-off first and everything else second -- it is the one property that
//  can be silently lost.
//
//  THE FREE-SURFACE CONDITION reconstructs the populations that would have
//  arrived from the gas. For every direction i whose source cell x - c_i is Gas,
//
//      f_i(x) = f_i^eq(rho_G, u) + f_ibar^eq(rho_G, u) - f_ibar(x),
//
//  with rho_G fixing the gas pressure through p = rho_G cs^2. Its two moments
//  are what the free surface is: the normal stress equals the gas pressure, and
//  the tangential stress vanishes. f_ibar(x) here is the post-collision
//  population at x from the previous step, which the two-lattice scheme still
//  holds in its source array -- no extra storage is needed to find it.
//
//  THE VELOCITY IN THAT RECONSTRUCTION IS ONE STEP OLD. u is needed to build the
//  equilibria, and u comes from a sum over the populations, some of which are
//  the ones being reconstructed. The circle is broken with the previous step's
//  velocity, which is first-order in time and is what the literature does. It
//  matters where the interface accelerates hard; it is invisible in a standing
//  wave.
//
//==============================================================================
//  WHY TWO-LATTICE STREAMING, when everything else in this code uses Esoteric
//  Pull. Three requirements, and Esoteric Pull fails two of them outright.
//
//   1. Mass exchange needs f_i(x) and f_ibar(x + c_i) as POST-COLLISION values,
//      after the whole domain has collided. Under an in-place scheme those slots
//      hold a mixture of two time levels, and which one depends on parity and on
//      the direction's index. Under two-lattice they are simply the destination
//      array, complete and unambiguous, once the collide pass has fenced.
//   2. A cell that fills up turns its gas neighbours into interface cells, which
//      then need populations. In place, a cell's populations live in its
//      neighbours' slots -- initialising a neighbour means writing storage that
//      other cells are concurrently reading through their own aliases.
//   3. Reading a neighbour's post-collision state while writing one's own is the
//      normal case here and impossible there.
//
//  Requirement 2 is dodged rather than met: every pass below is a GATHER. A cell
//  that is about to become interface initialises ITSELF by averaging its
//  neighbours, and a cell with excess mass to give away publishes it in a field
//  that its neighbours collect. Nothing ever writes another cell's memory, so
//  there is no race to reason about and no atomics. That is a deliberate shape,
//  not an accident of the streaming choice -- it would be worth keeping even if
//  scatters were free.
//
//  The cost is two population arrays instead of one. For a method that deletes
//  an entire phase, that is not the expensive part.
//
//==============================================================================
//  WHAT THIS DOES NOT MODEL.
//
//   * NO SURFACE TENSION -- STILL, BUT THE REASON HAS NARROWED. It used to be
//     that rho_G was uniform, so the gas pressure could carry no curvature term
//     at all. rho_G_of (below) makes the gas pressure a FIELD, so the normal
//     stress can now vary from node to node, and p_G = p_atm - sigma kappa is
//     expressible. WHAT IS STILL MISSING IS kappa: nothing here computes a
//     curvature from the fill level, and that is the hard half. So surface
//     tension is not present, a caller that wants it must supply the curvature
//     itself, and everything at a Bond number where surface tension competes
//     with gravity remains out of scope -- most droplet problems, and none of
//     the dam-break family. What rho_G_of DOES deliver on its own is a
//     prescribed non-uniform normal stress, which is a laser's recoil pressure:
//     validation/recoil.cpp measures that against the exact hydrostatic
//     depression.
//   * THE CONTACT ANGLE IS OPT-IN, AND IT SETS A WINDOW RATHER THAN A LINE.
//     By default the curvature reads a wall as gas, so every wall is
//     non-wetting wherever surface tension is on: a sessile drop sits at 152
//     degrees (validation/sessile_drop, R0 = 16), not the "about 180" this entry
//     used to say. set_contact_angle() (2026-11-03; the member `wetting` argues
//     the condition) imposes an angle through the curvature and nothing else,
//     and validation/sessile_drop measures what that buys. THE CONTACT LINE
//     STOPS ANYWHERE IN A WINDOW: seeded 15 degrees either side, a drop at
//     R0 = 16 ends at 57.6 / 63.4 degrees for 60, 83.9 / 93.5 for 90 and
//     115.5 / 125.2 for 120 -- the prescribed angle inside every window, the
//     midpoints within 1.3 degrees of it. AND THE WINDOW WIDENS WITH
//     RESOLUTION: half-widths 2.0 / 2.9 / 6.9 degrees at 60, 3.9 / 4.8 / 7.5 at
//     90, 2.1 / 4.9 / 10.8 at 120, at R0 = 8 / 16 / 32, so refinement pins the
//     line harder rather than freeing it. A static state needs kappa uniform
//     over the interface, and the one-pass smoothing leaves fill patterns that
//     change the true curvature without changing kappa (the R0 = 16 60-degree
//     drop is a circle of R 24.0 that the solver reads as 26.5 at every
//     interface cell), so the line can stop wherever such a pattern absorbs the
//     mismatch; a fixed absolute tolerance in kappa is a relative one growing
//     with R. That reading fits the trend and is not proven, and the cure is not
//     known. Below about 45 degrees the curvature next to the wall is
//     biased (kappa R reads 1.16 / 1.49 / 2.05 there at R = 12 / 24 / 48 on an
//     exact 30-degree cap, growing with R), and a 30-degree drop stops at 43
//     degrees from above. A 90-degree mirror of the colour function tried
//     earlier the same day against demonstrator/mew_nozzle's melt peeling off
//     its collector changed nothing there; what kept that rope from coming back
//     was the case's own drain.
//   * NO GAS DYNAMICS, and this is the method, not an omission. An enclosed
//     bubble does not compress, because its pressure is prescribed rather than
//     solved. Modelling that needs a volume-of-gas tracker with a separate
//     bubble-merging graph, which is a larger piece of work than this file.
//   * THE RECONSTRUCTION IS APPLIED TO GAS-FACING DIRECTIONS ONLY. The
//     literature also reconstructs directions with c_i . n > 0 for the interface
//     normal n, which stabilises interfaces that are nearly tangential to the
//     lattice. That refinement is not here, and if a case shows interface
//     roughening along a diagonal this is the first thing to add.
//   * EXCESS MASS IS REDISTRIBUTED UNIFORMLY among eligible neighbours rather
//     than weighted by the interface normal. Uniform redistribution conserves
//     mass exactly -- which is what matters most -- but is known to be more
//     diffusive at a sharply curved interface than the normal-weighted form.
//   * NO SUBGRID INTERFACE. The interface is resolved to one cell, so a film
//     thinner than a cell either survives as a one-cell layer or vanishes.
//     Thinning one CREATES MASS: demonstrator/mew_nozzle's field-off thread,
//     drawn by a moving collector until its thinnest row held 0.80 cells,
//     took the ledger from +0.5 % to +5.4 % of the injected melt, and then
//     broke (2026-11-03). Not diagnosed; the first suspect is settle()'s
//     defect (ii) below.
//   * A MIRROR-SYMMETRIC SET-UP DOES NOT STAY SYMMETRIC. An advancing front
//     amplifies round-off: mew_nozzle's extruded column left the axis by
//     -0.42 mm in 4.5 s. Not diagnosed; mirror_symmetrize_x() projects the
//     asymmetry away while the problem is symmetric, and its comment has the
//     numbers.
//   * A MOVING OBSTACLE IS BETTER THAN IT WAS AND STILL NOT RELIABLE. What this
//     entry used to say -- that mass_exchange()'s per-face antisymmetry is
//     broken by the mask changing between passes -- was WRONG, and measuring it
//     is what showed so: summed over a sweeping body, the interface-interface
//     term of mass_exchange() is 0.000000 to every digit. flags_ is fixed for
//     the whole of that pass and both cells of every pair do agree on it.
//
//     What was actually wrong, and is now fixed, is that
//     transfer_covered_mass() chose its recipients by reading reinit_. Nothing
//     clears reinit_ between settle() at the end of one step and
//     transfer_covered_mass() early in the next, so interface cells that
//     settle() had just built out of gas counted as cells this body had
//     uncovered. The share was divided among more cells than the body exposed,
//     the genuinely uncovered ones came back too empty and converted to gas,
//     and the body dug a void in its own wake. On the isolated test in
//     tests/test_free_surface.cpp that was not a drift but a DIVERGENCE, to a
//     total of 7.7e28 by step 1200; it is now bounded at 6.0e-2.
//     water_entry_fs.cpp reaches t* = 8.86 against 6.16 before.
//
//     TWO FURTHER DEFECTS ARE IDENTIFIED, MEASURED, AND DELIBERATELY NOT FIXED,
//     which is a worse-sounding sentence than it is. Both are real:
//
//       (i)  placing the body covers cells and uncovers none, and the early
//            return below then abandons their mass -- 49 of 1175 units, 4%, on
//            the first call of the isolated test. The same happens on any step
//            whose discretised outline covers without uncovering, which for a
//            body moving less than a cell per step is most of them.
//       (ii) settle() keeps the WHOLE mass of a cell whose conversion promote()
//            afterwards overruled, while classify() has already published that
//            cell's excess and settle() has already handed it to the
//            neighbours. It creates mass, exactly the published amount, and it
//            fires wherever a filling cell sits beside an emptying one: +25.0
//            units in a single step of the isolated test.
//
//     Each was fixed, and each fix made water_entry_fs.cpp fail SOONER -- (ii)
//     alone takes t* = 6.16 down to 3.39, (i) with it to 2.54, both with the
//     recipient fix to 2.23 -- against a baseline trajectory that is physically
//     sensible out to t* = 5.4, the body decelerating through impact and
//     settling to y_c/L = -1.12. The demonstrator's reach depends on these two
//     errors cancelling something else, and shipping either one alone trades a
//     correct ledger for a demonstrator that dies during the impact it exists
//     to show. Whatever they are cancelling has not been found. Do not "fix"
//     (i) or (ii) without re-running that case; the books getting better while
//     the physics gets worse is the signature to expect.
//
//     Bounce-back, momentum exchange and refill are all in place and are not
//     the problem.
//==============================================================================
#include "collision/MomentCollision.hpp"
#include "core/Types.hpp"
#include "equilibrium/Equilibrium.hpp"
#include "forcing/Forcing.hpp"
#include "grid/Domain.hpp"
#include "lattice/GradientLattice.hpp"
#include "lattice/Lattices.hpp"
#include "memory/TwoLattice.hpp"

namespace lbm {

//------------------------------------------------------------------------------
// Cell states. A distinct enum from CellType and from PhaseCell, with values
// that deliberately do NOT line up with either: all three are stored as uint8
// and mixing them would compile silently.
//------------------------------------------------------------------------------
enum FsCell : std::uint8_t {
  FsGas       = 10,   // no liquid; not simulated, holds nothing
  FsInterface = 11,   // partially filled; carries mass and the free surface
  FsFluid     = 12,   // full; an ordinary LBM cell
  FsSolid     = 13,   // wall: halfway bounce-back
  FsExcluded  = 14,   // halo
};

//------------------------------------------------------------------------------
// CENTRAL MOMENTS, not BGK, and gravity through Guo.
//
// The collision is a policy exactly as it is for FluidSolver, and the default is
// the central-moment operator of MomentCollision.hpp. That is the same choice
// the Rayleigh-Taylor cases make and for the same reason: BGK relaxes every mode
// at omega, and a free-surface flow worth simulating -- a collapsing column, a
// breaking wave -- reaches the same high-Reynolds regime where that stops being
// survivable. It costs a moment transform per cell against a scheme that has
// already deleted an entire phase.
//
// The force is Guo's with a CONSTANT vector, which is gravity at rho_0 = 1. That
// is exact for this method rather than an approximation: the liquid is nearly
// incompressible, the gas has no weight because it has no populations, and an
// interface cell's rho is close to one whatever its fill level -- the fill level
// is carried by the mass, not by the density.
//------------------------------------------------------------------------------
template <class L, class Collision = CentralMoments<L, Guo, RawPopulations>>
class FreeSurfaceSolver {
 public:
  using Lattice   = L;
  using Streaming = TwoLattice<L>;
  static constexpr int Q  = L::Q;
  static constexpr int D  = L::D;
  static constexpr int NF = Streaming::nb_first;
  static constexpr int NS = Streaming::nb_stride;

  static_assert(L::supports_navier_stokes,
                "the free surface is a Navier-Stokes solver and needs a lattice "
                "with isotropic fourth-order moments.");
  // The solver's own machinery is lattice-generic, but the central-moment
  // operator needs the factorised transform, so this runs on the product
  // lattices: D2Q9 and D3Q27.
  static_assert(ProductBasis<L>::enabled,
                "the free surface currently pairs with the product-basis central "
                "moment operator: D2Q9 or D3Q27.");

  explicit FreeSurfaceSolver(const Domain& dom)
      : dom_(dom), pop_(dom),
        flags_("fs_flags", dom.n_padded),
        newf_("fs_newflags", dom.n_padded),
        final_("fs_finalflags", dom.n_padded),
        reinit_("fs_reinit", dom.n_padded),
        uncov_("fs_uncovered", dom.n_padded),
        wux_("fs_wux", dom.n_padded), wuy_("fs_wuy", dom.n_padded),
        wuz_("fs_wuz", dom.n_padded), obst_("fs_obst", dom.n_padded),
        was_body_("fs_wasbody", dom.n_padded),
        mass_("fs_mass", dom.n_padded),
        eps_("fs_eps", dom.n_padded),
        excess_("fs_excess", dom.n_padded),
        rho_("fs_rho", dom.n_padded),
        ux_("fs_ux", dom.n_padded), uy_("fs_uy", dom.n_padded),
        uz_("fs_uz", dom.n_padded) {
    h_flags_ = Kokkos::create_mirror_view(flags_);
    for (Index n = 0; n < dom.n_padded; ++n) {
      Index px, py, pz; dom_.coords(n, px, py, pz);
      h_flags_(n) = dom_.is_interior(px, py, pz) ? FsGas : FsExcluded;
    }
    Kokkos::deep_copy(flags_, h_flags_);
  }

  //---- parameters -------------------------------------------------------------
  Collision coll{};              // omega, omega_bulk and the force live here
  Real rho_G  = Real(1);         // gas density: p_G = rho_G cs^2
  // OPTIONAL PER-NODE GAS DENSITY, and the door through which a NON-UNIFORM
  // normal stress reaches the free surface. Empty (the default) means the
  // uniform rho_G above, and then nothing here behaves differently.
  //
  // The free-surface condition already imposes "normal stress = gas pressure";
  // it was only ever the UNIFORMITY of rho_G that made that a single
  // atmosphere. So a laser's recoil pressure is not new machinery, it is
  // p_G(x) = p_atm + P_r(T(x)) written into this field -- and the banner's
  // own recipe for surface tension, p_G = p_atm - sigma kappa, is the same
  // field with a different right-hand side. Both want a per-node gas pressure
  // and neither wants anything else, which is why this is one View and not a
  // model.
  //
  // WHAT IT DOES NOT BUY: it does not add surface tension. A curvature still
  // has to be computed from the fill level by the caller, and nothing here
  // does that. It also does not make the gas dynamic -- the pressure is still
  // PRESCRIBED, so an enclosed bubble still does not compress.
  //
  // The rigid-body surface force below subtracts the LOCAL value as its
  // gauge. With a uniform rho_G that is the old constant and the closed-surface
  // cancellation the banner argues is exact; with a varying one the pressure
  // difference across a body is a real force and the local gauge is what keeps
  // it. That combination is untested -- no case in this tree pairs a rigid body
  // with a non-uniform gas pressure.
  View1D<Real> rho_G_of;         // optional per-node rho_G; empty means uniform

  // SURFACE TENSION. Zero (the default) means none and costs nothing -- no
  // storage is allocated and no kernel runs. set_surface_tension() turns it on.
  //
  // It reaches the flow through rho_G_of's own door: the dynamic condition sets
  // the liquid's normal stress to the imposed gas pressure, so imposing
  // p_G + sigma kappa instead of p_G puts exactly the Laplace jump across the
  // interface. In lattice terms the reconstruction sees
  //     rho_eff(n) = rho_G(n) + sigma kappa(n) / cs^2
  // and the two effects ADD, which is what a keyhole needs: recoil pushes the
  // surface down and surface tension resists it, in one field.
  //
  // SIGN: kappa is POSITIVE for a liquid drop (convex liquid), so the pressure
  // inside a drop is higher, which is Laplace's law the right way up.
  Real sigma = Real(0);

  // CONTACT ANGLE, opt-in (2026-11-03). Off (the default), the colour function
  // reads every wall as gas, which is a non-wetting wall of about 180 degrees,
  // and none of the code it adds runs. set_contact_angle(theta) turns it on;
  // theta is measured THROUGH THE LIQUID, in radians.
  //
  // IT ACTS ONLY THROUGH THE CURVATURE, so only with surface tension on. The
  // wall enters compute_curvature() at the solid cells beside the liquid, and
  // nowhere else: the mass exchange, the reconstruction and the bounce-back are
  // untouched, so the contact line moves by whatever slip the halfway wall
  // already allows. What the angle changes is the capillary pressure at the
  // interface cells next to the wall -- lower for a wetting wall, which pulls
  // liquid along it, higher for a non-wetting one.
  //
  // THE CONDITION is the one a sharp-interface (CSF) solver imposes at a wall:
  // the unit normal there, n = grad(colour)/|grad(colour)|, pointing into the
  // liquid, satisfies n . n_w = -cos(theta), with n_w the wall normal into the
  // domain. Written as a slope it is d(colour)/dn_w = -cot(theta) |d(colour)/dt|,
  // and that is what the solid cells next to the liquid are given: a ghost
  // colour extrapolated from the open neighbours with that slope, once for the
  // raw colour and once for the smoothed one, and then the unit normal itself,
  // -cos(theta) n_w + sin(theta) t, where pass 4 reads it. All three are needed
  // because each pass reads the one before it from the wall cells; a wall cell
  // corrected in one field and not the next still reads as gas through it.
  //
  // ONLY WHERE LIQUID TOUCHES THE WALL. A solid cell whose open neighbours hold
  // no liquid keeps reading as gas, so the wall does not reach across a gas gap
  // to a drop that has not landed. A cell open on two opposite sides (a sheet one
  // cell thick) has no wall normal and keeps the default too.
  //
  // wet_cos_of, if set, gives cos(theta) per SOLID cell and overrides the
  // uniform value, in the idiom of rho_G_of. A value outside (-1, 1) leaves that
  // wall reading as gas, so a case can wet one wall and not another. It still
  // needs set_contact_angle() to switch the law on; step() aborts on the field
  // alone, on `wetting` set by hand, and on a contact angle without sigma.
  bool wetting = false;
  Real wet_cos = Real(0);        // cos(theta); set through set_contact_angle()
  View1D<Real> wet_cos_of;       // optional per-node cos(theta); empty means uniform
  // Conversion hysteresis. A cell is promoted only past 1 + fill_offset and
  // demoted only below -fill_offset, so a cell sitting exactly at a threshold
  // cannot convert back and forth every step. The literature's value; without
  // it, a quiescent interface flickers and the flicker radiates noise.
  Real fill_offset = Real(1e-3);

  // RECONSTRUCT ALONG THE NORMAL TOO (Korner et al.'s refinement, which the
  // banner names as the first thing to add if an interface misbehaves). Off
  // (the default), only the populations whose source cell is Gas are rebuilt.
  // On, an interface cell also rebuilds every population travelling INTO the
  // liquid -- c_i . n > 0 with n = grad(colour), the normal compute_curvature()
  // already builds -- so the free-surface condition no longer depends on how
  // the surface happens to sit against the lattice. Needs the normals, so it
  // turns the curvature passes on; set it through set_normal_reconstruction().
  // Only populations from INTERFACE sources are added: a first version also
  // rebuilt ones from Fluid sources wherever the smoothed normal said so, and
  // the extruded column shed specks within a second.
  //
  // MEASURED 2026-11-03, AND IT IS A TRADE, NOT AN UPGRADE -- kept opt-in so the
  // trade can be re-measured (surface_tension -nrec, fs_electric -nrec):
  //   spurious current around a static drop   ~5e-6 -> ~3e-9    (better)
  //   Laplace error, R = 8/12/16/24 (%)       +3.39/+0.88/-0.64/+1.14 ->
  //                                           +3.39/+1.08/-0.40/+2.99 (same band,
  //                                           R = 24 past surface_tension's 2 % gate)
  //   film under a gentle cosine pull         9.2 % -> 21.6 % short (worse)
  // It was tried against demonstrator/mew_nozzle's sideways drift and does NOT
  // cure it: at R = 4 the drift vanished, at R = 8 it grew to -0.51 mm by 4 s
  // against -0.35 without. See mirror_symmetrize_x() for what does.
  bool normal_reconstruction = false;

  // HEIGHT-FUNCTION CURVATURE, opt-in (2026-11-04). Off (the default), kappa is
  // the once-smoothed -div(n) of compute_curvature()'s pass 4 and nothing below
  // runs. On, a fifth pass REPLACES kappa at every interface cell whose height
  // stencil is well formed: along the axis on which the normal is largest, each
  // of the three (2-D) or nine (3-D) columns around the cell gives the height
  // at which its liquid ends, and kappa = -h''/(1 + h'^2)^(3/2) from those
  // heights (the 3-D form with the cross term). A column is well formed when,
  // within seven cells either way of the cell's row, it reaches a Fluid cell
  // toward the liquid and then a Gas cell toward the gas through Interface
  // cells only, and meets no wall or halo; a cell with any column that is not
  // keeps pass 4's value, which is what keeps a contact line on the wetting law.
  //
  // WHY. pass 4 has fill patterns that change the shape without changing kappa,
  // and the dynamics find them: surface_tension -rate measured a free drop that
  // stops relaxing with its deformation invisible to kappa (see the banner of
  // compute_curvature). A height is a column SUM, so fill moved within a column
  // does not change it, and moving fill between columns is moving the surface.
  // What it does NOT do: it is not consulted where the stencil fails (thin
  // films, walls, columns that graze a small sphere), and it does not change the
  // normal, so normal_reconstruction and the wetting passes are untouched.
  //
  // MEASURED 2026-11-04 (validation/surface_tension, tests/test_free_surface).
  //  On exact fills: worst |kappa R - 1| 0.0033 on a disc of R = 16, where pass 4
  //  gives 0.52; 2e-15 on an inclined plane; a ball of R = 9 within 3.3 % (mean
  //  +0.21 %) where heights are used, with 83 of 1540 cells -- corner columns
  //  near (1,1,1) that graze a ball this small -- keeping pass 4's kappa.
  //  On the solver's own static drop: Laplace +0.85 / +0.32 / +0.34 / -0.02 % at
  //  R = 8 / 12 / 16 / 24 against pass 4's +3.39 / +0.88 / -0.64 / +1.14 %, and the
  //  spurious current at round-off (5e-16 to 4e-14) against about 5e-6.
  //  On a free drop relaxing from an ellipse (-rate): no stall, but the rate still
  //  FALLS as the drop is resolved -- 1.03 / 0.90 / 0.79 / 0.70 of the
  //  exact one at R = 8 / 16 / 32 / 64, where pass 4 gives 1.27 / 0.80 / 0.05 /
  //  0.005 -- and the ellipse turned 45 degrees is slower still, 0.74 / 0.70 /
  //  0.37 / 0.28. The fill field the dynamics leave carries cell-scale
  //  ripples -- rms 6e-4 of a cell, at most 1.1e-3, over the top of an R = 32
  //  drop -- which a second difference of heights reads as curvature noise of
  //  0.06/R, and kappa's mode-2 part falls to about half the shape's. That is
  //  NOT the conversion hysteresis (fill_offset 1e-5 moves the rates by under
  //  1 %). It is the MASS EXCHANGE (2026-11-05): see mass_exchange(). With
  //  exchange_geometric, exchange_donor and surface_follows_line the same
  //  heights relax the drop at 1.003 of the exact rate at R = 64 (1.007 turned
  //  45 degrees); those members have the numbers.
  //  With a contact angle, cells whose stencil meets the wall keep pass 4's
  //  kappa and the law, and the mixture is biased: sessile_drop -hf narrows the
  //  90- and 120-degree windows (half-widths 1.2 and 2.5 degrees against 4.8
  //  and 4.9), widens the 60-degree one (5.4 against 2.9), moves the midpoints
  //  at 60-150 degrees low by 1.2 to 11 degrees, and the 90-degree window
  //  misses 90 by 1.3. Do not use it with wetting until the wall has ghost
  //  heights.
  bool height_function = false;

  // HOW MANY COLUMNS THE HEIGHTS ARE TAKEN OVER (set_height_fit), 2-D only. 3,
  // the default, is the central difference above, unchanged. 5, 7 or 9 smooth
  // the heights with [1 2 1]/4 once, twice or three times before the central
  // differences, so the second difference answers a wavenumber k along the
  // surface with -4 sin^2(k/2) cos^(2p)(k/2) after p passes: a ripple a cell or
  // two long is damped out of kappa, a parabola keeps its h' and h'' exactly,
  // and no wavenumber gets a curvature of the wrong sign. It exists to test the
  // ripples named above. What it costs: every column of the wider stencil must
  // be well formed, so more cells keep pass 4's kappa near walls and in thin
  // films; and the zigzag, k = pi, gets no capillary restoring force at all.
  // A LEAST-SQUARES PARABOLA OVER 5 OR 7 COLUMNS WAS TRIED FIRST AND IS UNSTABLE
  // (2026-11-05). Its second-difference weights are (j^2 - 2)/7 over five
  // columns, which give a zigzag +4/7 where the central difference gives -4, so
  // surface tension pumps the zigzag instead of flattening it; over seven,
  // (j^2 - 4)/42, the same holds in a band around k = 2 pi/3. It was right on
  // exact fills (0.999 of an ellipse's mode-2 curvature at R = 32) and blew up
  // the moment anything moved: surface_tension -hf's static drop drifted its
  // mass by up to 85 % and its spurious current reached 0.045 (five columns)
  // and 0.039 (seven), against round-off with three.
  // MEASURED 2026-11-05 (validation/surface_tension -rate -hf -hfit N, Oh = 3):
  // the rate over the exact one at R = 16 / 32 / 64,
  //      columns   along x                    turned 45 degrees
  //         3      0.897 / 0.795 / 0.695      0.696 / 0.367 / 0.284
  //         5      0.982 / 0.922 / 0.890      0.958 / 0.495 / 0.440
  //         7      1.015 / 1.002 / 0.973      1.093 / 0.853 / 0.692
  //         9      1.057 / 1.011 / 0.997      1.332 / 0.969 / 0.825
  // So the ripples ARE the slowing: damping them out of kappa restores the rate,
  // and with nine columns the R = 64 drop relaxes at 0.997 of it, a clean
  // exponential (halves 0.996 and 0.995) whose curvature keeps 0.92 of the
  // shape's mode-2 part to the end. But it is NOT A CURE. At any one width the
  // rate still falls as R grows -- along x the width that reaches the exact
  // rate goes up by about one pass per doubling of R (five at 16, seven at 32,
  // nine at 64) -- and turned 45 degrees nine columns still give 0.825 at
  // R = 64. A width that grows with the drop is a fixed physical filter, whose
  // own error does not refine away. The source of the ripples is what needs
  // fixing, and it is the mass exchange (mass_exchange()); smoothed, the
  // curvature merely stops seeing them while they keep growing in the fill
  // (validation/fs_band). With the exchange fixed (exchange_geometric and its
  // two refinements) three columns converge and no smoothing is needed. Small
  // drops pay twice: at R = 8 seven
  // columns leave 20 of 44 interface cells on pass 4's kappa and nine 28 of 36,
  // the wider fits' rates there scatter from 0.97 to 1.34, and the static
  // Laplace error is +3.4 / +1.3 / +0.8 / +0.3 % at R = 8 / 12 / 16 / 24 with
  // five columns and +4.7 / +2.9 / +1.5 / +0.7 % with seven.
  int height_fit = 3;

  // THE MASS EXCHANGE BETWEEN TWO INTERFACE CELLS, WEIGHTED BY THE UPWIND FILL.
  // Off by default, and mass_exchange() is then exactly what it was. The
  // default weights the net population flux across a link by the MEAN of the
  // two cells' fills, which along the surface is a centred advection of the
  // fill, with nothing to damp a cell-scale ripple; on, it weights it by the
  // fill of the cell the net flux leaves. Both vanish at rest and both conserve
  // mass pair by pair. MEASURED 2026-11-05 (validation/fs_band -upwind) and
  // WORSE: a band sliding along itself ripples by 0.36 of a cell after 10 cells
  // against 0.19 with the mean, and swept across, its surface thickens from 320
  // to 1376 interface cells. The ripples do not come from which fill weights the
  // link; they come from the links that are not weighted (mass_exchange()).
  bool exchange_upwind = false;

  // THE MASS EXCHANGE THROUGH FACES, WEIGHTED BY HOW MUCH OF EACH FACE IS WET
  // (2-D, opt-in through set_exchange_geometric; mass_exchange() is otherwise
  // exactly what it was). See mass_exchange() for what it replaces and why.
  // Each interface cell's surface is reconstructed as a straight line -- the
  // normal from height functions where its three columns are well formed and
  // from the Youngs gradient of the fill where not, placed to hold the cell's
  // fill exactly -- and every face gets the wet fraction of that line: 1 against
  // a Fluid cell, 0 against Gas, and between two interface cells the mean of the
  // two cells' estimates, so both sides agree and mass is conserved pair by
  // pair. Every diagonal link's net flux is routed half through each of the two
  // cells beside its corner, so each face carries axis(n->j) plus half of the
  // four diagonal links that cross it, and for a uniform flow that sum is the
  // finite-volume face flux rho u.n_f exactly. A link to a Gas cell has no
  // populations to read, so its flux is the equilibrium difference of the
  // liquid cell at its other end. Then dm = -sum_faces W_f Phi_f, and for a
  // planar surface sum_faces W_f n_f = -l n, the surface's length in the cell
  // times its normal: liquid sliding along a flat surface moves no fill, and
  // liquid crossing it fills each cell at rho l u_n, its exact rate. Mass stays
  // conserved because a Fluid cell's faces all carry weight 1 and Fluid never
  // touches Gas. Checked on the exact fills of validation/fs_band before it was
  // written: along the band 1e-16 (slope 1/2) and 5e-15 (3/7), across it the
  // exact rate to 8e-15, with height-function or exact normals alike.
  // It needs one conversion rule of its own (see classify()): it fills a cell
  // at its exact rate, which falls to zero as the surface leaves, so cells can
  // stop inside the hysteresis band and be stranded.
  // MEASURED 2026-11-05, ALONE (exchange_donor and surface_follows_line below
  // are what complete it). fs_band, sliding along itself at slope 3/7: the
  // column sums scatter by 1.8-3.5e-5 of a cell over 10 cells, against 0.04-0.26
  // with the default exchange, and mass holds to 6e-15. Swept ACROSS itself at
  // slopes 1/2 and 3/7 it is no better than the default (0.02-0.07): that error
  // is in when cells switch on, not in how mass moves. surface_tension -rate
  // -hf -geom (Oh = 3): 1.083 / 1.037 / 1.016 / 0.941 of the exact rate at
  // R = 8 / 16 / 32 / 64 (default exchange 1.030 / 0.897 / 0.795 / 0.695), and
  // turned 45 degrees 0.830 / 0.537 / 0.652 at R = 8 / 16 / 32, not exponential.
  bool exchange_geometric = false;
  void set_exchange_geometric(bool on) {
    if (on && L::D != 2)
      Kokkos::abort("FreeSurfaceSolver::set_exchange_geometric: the face-routed exchange is 2-D only");
    exchange_geometric = on;
    if (on && wf_.size() == 0) wf_ = View1D<Real>("fs_wetface", 8 * dom_.n_padded);
  }

  // Two refinements of exchange_geometric, each read only with it on.
  // exchange_donor: between two interface cells, weight a face by the
  // reconstruction of the cell its net flux LEAVES rather than by the mean of
  // the two. The same cell from either side, so it conserves mass; it matters
  // where one cell has just been switched on and holds no surface of its own
  // yet, which the mean would weight at half.
  // surface_follows_line: switch a cell to Interface as soon as an interface
  // neighbour's reconstructed line reaches into it -- a Gas cell when the
  // shared face or corner is wet, a Fluid cell when it is dry (promote()) --
  // instead of waiting for that neighbour to convert. On an inclined surface the
  // line enters the next cell before this one is full, and until then the
  // next cell can hold none of the liquid that has reached it.
  // MEASURED 2026-11-05, ALL THREE ON. fs_band over 10 cells: across itself the
  // column sums scatter by 7e-6 to 9e-5 (slope 1/2) and 5e-6 to 6e-5 (3/7),
  // against 0.02-0.07 without the two, and along itself by 1.4-3.8e-6; at slope
  // 1 by round-off; the surface keeps its thickness. Either alone does much
  // less: donor 0.005-0.032 across, follow 0.001-0.058 and a thickened surface.
  // THE FREE DROP NOW CONVERGES (surface_tension -rate -hf -geom -donor -follow,
  // Oh = 3, 3-column heights): 1.060 / 1.020 / 1.010 / 1.003 of the exact rate
  // at R = 8 / 16 / 32 / 64, and turned 45 degrees 1.110 / 1.049 / 1.019 /
  // 1.007, every decay a clean exponential (rms of ln eps 3e-4 to 1.1e-2), the
  // curvature keeping all of the mode-2 part to the end (0.99-1.13) and its
  // scatter 0.0011-0.0073 x R against 0.044 at R = 32 with the default. What it
  // costs at rest: the static sections pass, but the spurious current is up to
  // 3.6e-5 (R = 12) where the default holds round-off -- a straight line drawn
  // through a curved surface's cell reaches a little into cells the circle
  // does not, and switches them on. Not measured: walls, contact angles, 3-D
  // (the setter refuses it), and any flow but these two.
  bool exchange_donor = false;
  bool surface_follows_line = false;

  bool drop_detached = true;   // isolated specks -> gas; see classify()

  // NO CELL MAY BECOME LIQUID THAT WAS GAS. Off by default, and everything in
  // this file behaves exactly as before when it is off.
  //
  // WHY IT EXISTS. Coupling this solver to a TRANSPORTED FIELD -- an enthalpy,
  // for a melt pool -- runs into one hard question: when a gas cell becomes
  // liquid, what temperature does it have? The liquid arrives from nowhere, and
  // ScalarSolver has no notion of a domain that changes shape. Forbidding the
  // transition deletes the question rather than answering it: the liquid region
  // then only ever SHRINKS, so every liquid cell has a continuous history back
  // to the initial condition and no cell is ever created from nothing. Only the
  // receding direction is left, where a cell's field has to be removed rather
  // than invented.
  //
  // IT IS A MODEL, NOT A TRICK, AND IT IS RIGHT FOR DRILLING. A hole being
  // bored only deepens. Melt displaced sideways into EXISTING liquid is
  // unaffected -- the prohibition is on filling GAS -- so what it actually
  // forbids is melt piling up above the surface into a recast rim. That melt is
  // instead accounted as EJECTED, which is what melt leaving a keyhole does.
  //
  // WHAT IT COSTS, stated so a case cannot quietly rely on it.
  //   * NO RECAST RIM and no resolidified crown. Real drilling makes one.
  //   * NO REFILL after the pulse, so only the drilling phase is modelled.
  //   * The bias is ONE-DIRECTIONAL: removing the only sink for displaced melt
  //     other than ejection can only increase removal, so a hole predicted this
  //     way is biased DEEP. How much is not measured.
  //   * MASS IS NO LONGER CONSERVED, deliberately. total_mass() alone stops
  //     being a correctness check; the budget that replaces it is
  //     total_mass() + ejected_mass() = the initial mass, and a case using this
  //     must assert THAT instead.
  bool no_refill = false;

  static Real omega_from_viscosity(Real nu) {
    return Collision::omega_from_viscosity(nu);
  }
  static Real viscosity_from_omega(Real w) {
    return Collision::viscosity_from_omega(w);
  }
  // Gravity, as the constant Guo force it is.
  void set_gravity(Real ax, Real ay, Real az = Real(0)) {
    coll.forcing.fx = ax;  coll.forcing.fy = ay;  coll.forcing.fz = az;
  }

  //----------------------------------------------------------------------------
  // fn(x, y, z) -> FsCell, over interior coordinates. Call BEFORE initialize().
  //----------------------------------------------------------------------------
  template <class Fn>
  void set_geometry(Fn fn) {
    for (Index z = 0; z < dom_.nz; ++z)
      for (Index y = 0; y < dom_.ny; ++y)
        for (Index x = 0; x < dom_.nx; ++x)
          h_flags_(dom_.id(x, y, z)) = fn(x, y, z);
    Kokkos::deep_copy(flags_, h_flags_);
    Kokkos::deep_copy(newf_, flags_);
    Kokkos::deep_copy(final_, flags_);
  }

  //----------------------------------------------------------------------------
  // fn(n) -> Seed. `fill` in [0, 1] describes the liquid: 1 seeds Fluid, 0 seeds
  // Gas, anything between seeds an Interface cell, and the classification and
  // one-cell interface shell follow from it. `rho` seeds the pressure through
  // p = rho cs^2.
  //
  // THE DENSITY IS WORTH SETTING. Left at 1 the liquid starts with no pressure
  // gradient at all, so a column under gravity has to build its own hydrostatic
  // profile, and it does that by ringing: an acoustic transient that crosses the
  // domain in a few hundred steps and takes far longer to damp. That is harmless
  // in a dam break, where the flow is violent enough not to care, and ruinous in
  // a standing-wave measurement, where it is noise at the amplitude being
  // measured. rho = rho_G + g (y_surface - y) / cs^2 starts it quiet.
  //----------------------------------------------------------------------------
  struct Seed { Real fill = 0, rho = Real(1); };

  template <class Fn>
  void initialize(Fn fn) {
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, newf = newf_;
    auto mass = mass_, eps = eps_, rho = rho_;
    auto ux = ux_, uy = uy_, uz = uz_;
    Kokkos::parallel_for("fs_init", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t was = flags(n);
      Real e = Real(0), r = Real(1);
      if (was != FsSolid && was != FsExcluded) {
        const Seed sd = fn(n);
        e = sd.fill;
        r = sd.rho;
        e = e < Real(0) ? Real(0) : (e > Real(1) ? Real(1) : e);
        flags(n) = (e >= Real(1)) ? FsFluid : (e <= Real(0) ? FsGas : FsInterface);
      }
      newf(n) = flags(n);
      rho(n) = r;
      ux(n) = Real(0); uy(n) = Real(0); uz(n) = Real(0);
      eps(n)  = e;
      mass(n) = e * r;
      // Every cell is given populations, walls and gas included. A wall holds
      // in-transit populations exactly as it does elsewhere in this code, and a
      // gas cell that later becomes interface is initialised properly then --
      // but leaving either as raw zeros makes the first step read garbage.
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      for (int i = 0; i < Q; ++i) {
        const Real v = Collision::seed_value(i, r, Real(0), Real(0), Real(0));
        acc.src(n, i) = v;
        acc.dst(n, i) = v;
      }
    });
    Kokkos::fence();
    // The seeded fill levels need not describe a closed interface -- a caller
    // may hand over a sharp step. One repair pass turns it into one.
    close_interface();
  }

  //----------------------------------------------------------------------------
  // One step. Five fenced passes; see the banner for why they cannot be fewer.
  //----------------------------------------------------------------------------
  // Allocates the five curvature fields and switches the term on. Calling it
  // with 0 switches it off again but does not free the storage.
  void set_normal_reconstruction(bool on) {
    normal_reconstruction = on;
    if (on) allocate_curvature();
  }
  void set_surface_tension(Real s) {
    sigma = s;
    if (s != Real(0)) allocate_curvature();
  }
  // Acts only where kappa is computed, i.e. with surface tension (or the normal
  // reconstruction) on; with sigma = 0 it is computed and never read.
  void set_height_function(bool on) {
    height_function = on;
    if (on) allocate_curvature();
  }
  // Columns in the height fit (the member height_fit): odd, 3 to 9, and 3 in 3-D.
  // Read only while height_function is on.
  void set_height_fit(int columns) {
    if (columns < 3 || columns > 9 || columns % 2 == 0 || (L::D != 2 && columns != 3))
      Kokkos::abort("FreeSurfaceSolver::set_height_fit: need an odd 3 to 9 columns, and 3 in 3-D");
    height_fit = columns;
  }
  // Interface cells the last curvature pass gave a height-function kappa, and
  // those it left on pass 4's because a column was not well formed. Both 0
  // unless height_function is on.
  Index height_function_cells() const { return hf_used_; }
  Index height_function_fallbacks() const { return hf_fell_; }
  // theta in radians, through the liquid, strictly inside (0, pi): the ghost
  // colour carries cot(theta), which has no finite value at either end.
  void set_contact_angle(Real theta) {
    if (!(theta > Real(0) && theta < Real(3.14159265358979323846)))
      Kokkos::abort("FreeSurfaceSolver::set_contact_angle: theta must lie in (0, pi) radians");
    wetting = true;
    wet_cos = Kokkos::cos(theta);
    allocate_curvature();
    if (wet_.size() == 0) wet_ = View1D<std::uint8_t>("fs_wet", dom_.n_padded);
  }
  void allocate_curvature() {
    if (cf_.size() == 0) {
      cf_  = View1D<Real>("fs_colour",   dom_.n_padded);
      cfs_ = View1D<Real>("fs_colour_s", dom_.n_padded);
      knx_ = View1D<Real>("fs_nx",       dom_.n_padded);
      kny_ = View1D<Real>("fs_ny",       dom_.n_padded);
      knz_ = View1D<Real>("fs_nz",       dom_.n_padded);
      kap_ = View1D<Real>("fs_kappa",    dom_.n_padded);
    }
  }

  //----------------------------------------------------------------------------
  //  CURVATURE FROM THE FILL LEVEL, in four passes.
  //
  //  kappa = -div(n) with n = grad(eps)/|grad(eps)|. The sign is what makes a
  //  drop's interior the high-pressure side: eps falls outward, so grad(eps)
  //  points INTO the liquid and div of it is -1/R for a 2-D disc; negating
  //  gives +1/R.
  //
  //  THE FILL LEVEL IS NOT A FIELD UNTIL IT IS MADE ONE. eps_ is meaningful
  //  only at interface cells -- a Fluid cell's entry is not maintained at 1 --
  //  so pass 1 builds the colour function the differencing needs: 1 in fluid,
  //  eps at the interface, 0 in gas. Reading eps_ directly would differentiate
  //  whatever was left in the bulk.
  //
  //  AND IT IS SMOOTHED BEFORE IT IS DIFFERENTIATED, because a one-cell-thick
  //  interface is very nearly a step and the curvature of a step is noise.
  //  Pass 2 is one lattice-weighted average, sum_i w_i eps(x + c_i), whose
  //  weights sum to one so it is a mollifier and not a gain. One pass, not
  //  several: each one widens the effective interface and biases kappa toward
  //  zero, and the measurement in validation/surface_tension.cpp is what
  //  settled the count rather than a preference.
  //
  //  THE STENCIL IS THE GRADIENT LATTICE'S, NOT THIS LATTICE'S -- D2Q9 for a
  //  2-D field, D3Q27 for a 3-D one -- for the reason GradientLattice.hpp
  //  gives: the isotropy of the gradient stencil is what sets the
  //  spurious-current floor around a static drop, and an axis-only stencil is
  //  markedly worse. That is the whole quantity this case measures, so it is
  //  not a detail.
  //
  //  WHERE |grad eps| IS BELOW A FLOOR the normal is left at zero rather than
  //  normalised, which is the bulk and is correct there. It does mean the
  //  divergence at the OUTER edge of the interface band reads some zero
  //  normals from the bulk, and that is a real error of this construction; it
  //  is why kappa is used only at interface cells, where the band is centred.
  //
  //  AND THIS CURVATURE PINS SLOW DEFORMATIONS, MORE SO ON A FINER GRID
  //  (measured 2026-11-04, validation/surface_tension -rate). A free 2-D drop
  //  relaxing from an ellipse loses the mode-2 part of this kappa into
  //  cell-scale fill patterns: at R = 32 it reads 0.91 of the exact 3 eps/R at
  //  the start and about 0 within 0.7 relaxation times, while the drop is still
  //  0.62 as deformed as it began, so nothing drives it and it stops. What is
  //  left has a curvature of its own, 3 eps/R, of 0.0018 / 0.0027 / 0.0020 per
  //  cell where the runs end at R = 16 / 32 / 64: a tolerance about fixed in
  //  CELLS, so a finer grid puts more of a given shape below it. Neither
  //  normal_reconstruction nor turning the ellipse 45 degrees removes it. The
  //  case's banner has the rates. A curvature built from column sums of the
  //  fill does not see fill moved within a column; that is height_function
  //  (opt-in, pass 5 below), which removes the stall and not the slowing.
  //
  //  WITH A CONTACT ANGLE (wetting, opt-in) three passes are added, 1b, 2b and
  //  3b, each rewriting the wall cells beside the liquid in the field the next
  //  pass reads; see the member `wetting` for the condition they impose.
  //
  //  WITH height_function (opt-in) a fifth pass overwrites pass 4's kappa
  //  wherever a height stencil is well formed; see that member.
  //----------------------------------------------------------------------------
  // A wall cell's view of the liquid: its gradient-lattice neighbours that are
  // open (Fluid, Interface or Gas), the unit normal into the domain they define,
  // and, for a field F, their weighted mean, their mean offset along that normal
  // and the least-squares slope of F ALONG the wall. The slope is taken about
  // the mean, so a uniform F has none however lopsided the open set is.
  struct WallView {
    Real nw[3] = {Real(0), Real(0), Real(0)};
    Real mean = Real(0), delta = Real(0);
    Real gt[3] = {Real(0), Real(0), Real(0)};
    bool ok = false;     // false: buried, or open on two opposite sides
  };
  template <class GL, class FV>
  KOKKOS_INLINE_FUNCTION static WallView wall_view(const Neighbours<GL>& nb,
                                                   const View1D<std::uint8_t>& flags,
                                                   const FV& F) {
    WallView v;
    Real ws = Real(0), wn[3] = {Real(0), Real(0), Real(0)};
    for (int i = 1; i < GL::Q; ++i) {
      const std::uint8_t f = flags(nb.j[i]);
      if (f != FsFluid && f != FsInterface && f != FsGas) continue;
      const Real w = weight<GL, Real>(i);
      ws += w;
      for (int a = 0; a < GL::D; ++a) wn[a] += w * Real(cvel<GL>(i, a));
    }
    const Real m = Kokkos::sqrt(wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2]);
    if (ws <= Real(0) || m <= Real(1e-12) * ws) return v;
    for (int a = 0; a < 3; ++a) v.nw[a] = wn[a] / m;
    Real s = Real(0), del = Real(0);
    for (int i = 1; i < GL::Q; ++i) {
      const Index j = nb.j[i];
      const std::uint8_t f = flags(j);
      if (f != FsFluid && f != FsInterface && f != FsGas) continue;
      const Real w = weight<GL, Real>(i);
      Real cn = Real(0);
      for (int a = 0; a < GL::D; ++a) cn += Real(cvel<GL>(i, a)) * v.nw[a];
      s += w * F(j);
      del += w * cn;
    }
    v.mean = s / ws;
    v.delta = del / ws;
    Real g[3] = {Real(0), Real(0), Real(0)}, gg = Real(0);
    for (int i = 1; i < GL::Q; ++i) {
      const Index j = nb.j[i];
      const std::uint8_t f = flags(j);
      if (f != FsFluid && f != FsInterface && f != FsGas) continue;
      const Real w = weight<GL, Real>(i);
      Real cn = Real(0);
      for (int a = 0; a < GL::D; ++a) cn += Real(cvel<GL>(i, a)) * v.nw[a];
      const Real dF = F(j) - v.mean;
      for (int a = 0; a < GL::D; ++a) {
        const Real pc = Real(cvel<GL>(i, a)) - cn * v.nw[a];
        g[a] += w * dF * pc;
        gg += w * pc * pc;
      }
    }
    // gg sums |P c|^2 over the D - 1 tangential directions; per direction it is
    // the normalisation of a least-squares slope.
    if (gg > Real(0))
      for (int a = 0; a < 3; ++a) v.gt[a] = g[a] * Real(GL::D - 1) / gg;
    v.ok = true;
    return v;
  }

  void compute_curvature() {
    using GL = typename GradientLatticeOf<L>::type;
    const Domain d = dom_;
    auto flags = flags_; auto eps = eps_;
    auto cf = cf_, cfs = cfs_, knx = knx_, kny = kny_, knz = knz_, kap = kap_;
    constexpr Real icsg = inv_cs2<GL, Real>();
    const bool wet = wetting && wet_.data();
    auto wetm = wet_;
    const auto wcf = wet_cos_of;
    const Real wc = wet_cos;

    // pass 1: the colour function
    Kokkos::parallel_for("fs_colour", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t f = flags(n);
      cf(n) = (f == FsFluid) ? Real(1)
            : (f == FsInterface) ? eps(n) : Real(0);
    });
    Kokkos::fence();

    // pass 1b: the ghost colour in the wall cells the liquid touches. Writes
    // only solid cells and reads only open ones, so it cannot race.
    if (wet) {
      Kokkos::parallel_for("fs_wet_colour", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        wetm(n) = 0;
        if (flags(n) != FsSolid) return;
        const Real c = wcf.data() ? wcf(n) : wc;
        if (!(c > Real(-1) && c < Real(1))) return;      // this wall reads as gas
        Neighbours<GL> nb;
        d.template fill_neighbours<GL, 1, 1>(n, nb);
        bool touch = false;
        for (int i = 1; i < GL::Q; ++i) {
          const Index j = nb.j[i];
          const std::uint8_t f = flags(j);
          if ((f == FsFluid || f == FsInterface) && cf(j) > Real(0)) touch = true;
        }
        if (!touch) return;
        const WallView v = wall_view<GL>(nb, flags, cf);
        if (!v.ok) return;
        const Real cot = c / Kokkos::sqrt(Real(1) - c * c);
        const Real gt = Kokkos::sqrt(v.gt[0] * v.gt[0] + v.gt[1] * v.gt[1] + v.gt[2] * v.gt[2]);
        cf(n) = v.mean + v.delta * cot * gt;
        wetm(n) = 1;
      });
      Kokkos::fence();
    }

    // pass 2: one mollifying pass
    Kokkos::parallel_for("fs_colour_smooth", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<GL> nb;
      d.template fill_neighbours<GL, 1, 1>(n, nb);
      Real a = weight<GL, Real>(0) * cf(n);
      for (int i = 1; i < GL::Q; ++i) a += weight<GL, Real>(i) * cf(nb.j[i]);
      cfs(n) = a;
    });
    Kokkos::fence();

    // pass 2b: the same ghost for the smoothed colour. Pass 2 averaged the wall
    // cell with the buried ones behind it, which still read as gas.
    if (wet) {
      Kokkos::parallel_for("fs_wet_colour_s", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        if (!wetm(n)) return;
        const Real c = wcf.data() ? wcf(n) : wc;
        Neighbours<GL> nb;
        d.template fill_neighbours<GL, 1, 1>(n, nb);
        const WallView v = wall_view<GL>(nb, flags, cfs);
        const Real cot = c / Kokkos::sqrt(Real(1) - c * c);
        const Real gt = Kokkos::sqrt(v.gt[0] * v.gt[0] + v.gt[1] * v.gt[1] + v.gt[2] * v.gt[2]);
        cfs(n) = v.mean + v.delta * cot * gt;
      });
      Kokkos::fence();
    }

    // pass 3: the unit normal
    Kokkos::parallel_for("fs_normal", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Neighbours<GL> nb;
      d.template fill_neighbours<GL, 1, 1>(n, nb);
      Real g[3] = {Real(0), Real(0), Real(0)};
      for (int i = 1; i < GL::Q; ++i) {
        const Real wp = weight<GL, Real>(i) * cfs(nb.j[i]);
        for (int a = 0; a < GL::D; ++a) g[a] += wp * Real(cvel<GL>(i, a));
      }
      for (int a = 0; a < 3; ++a) g[a] *= icsg;
      const Real m = Kokkos::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
      // The floor is on the GRADIENT, not on eps: it is the bulk that must be
      // excluded, and the bulk is where the colour function is flat.
      const Real inv = (m > Real(1e-6)) ? Real(1) / m : Real(0);
      knx(n) = g[0] * inv;  kny(n) = g[1] * inv;  knz(n) = g[2] * inv;
    });
    Kokkos::fence();

    // pass 3b: the wall's own normal is the contact-angle normal,
    // -cos(theta) n_w + sin(theta) t with t along the wall toward the liquid.
    // Pass 3 built it from the buried cells too; this is what pass 4 reads at
    // the contact line. Same floor as pass 3: no slope along the wall, no normal.
    if (wet) {
      Kokkos::parallel_for("fs_wet_normal", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        if (!wetm(n)) return;
        const Real c = wcf.data() ? wcf(n) : wc;
        const Real s = Kokkos::sqrt(Real(1) - c * c);
        Neighbours<GL> nb;
        d.template fill_neighbours<GL, 1, 1>(n, nb);
        const WallView v = wall_view<GL>(nb, flags, cfs);
        const Real gt = Kokkos::sqrt(v.gt[0] * v.gt[0] + v.gt[1] * v.gt[1] + v.gt[2] * v.gt[2]);
        const Real it = (gt > Real(1e-6)) ? s / gt : Real(0);
        const Real ic = (gt > Real(1e-6)) ? c : Real(0);
        knx(n) = -ic * v.nw[0] + it * v.gt[0];
        kny(n) = -ic * v.nw[1] + it * v.gt[1];
        knz(n) = -ic * v.nw[2] + it * v.gt[2];
      });
      Kokkos::fence();
    }

    // pass 4: kappa = -div(n)
    Kokkos::parallel_for("fs_kappa", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) != FsInterface) { kap(n) = Real(0); return; }
      Neighbours<GL> nb;
      d.template fill_neighbours<GL, 1, 1>(n, nb);
      Real div = Real(0);
      for (int i = 1; i < GL::Q; ++i) {
        const Real w = weight<GL, Real>(i);
        const Index j = nb.j[i];
        div += w * (Real(cvel<GL>(i, 0)) * knx(j)
                  + Real(cvel<GL>(i, 1)) * kny(j)
                  + Real(cvel<GL>(i, 2)) * knz(j));
      }
      kap(n) = -icsg * div;
    });
    Kokkos::fence();

    // pass 5 (opt-in): height functions, over pass 4's value where well formed.
    if (height_function) {
      Index used = 0, fell = 0;
      const int hw = (height_fit - 1) / 2;
      Kokkos::parallel_reduce("fs_kappa_hf", Range(0, dom_.n_padded),
        KOKKOS_LAMBDA(Index n, Index& u, Index& f) {
          if (flags(n) != FsInterface) return;
          Real k = Real(0);
          if (height_curvature(d, flags, cf, knx(n), kny(n), knz(n), n, k, hw)) {
            kap(n) = k;
            ++u;
          } else {
            ++f;
          }
        }, used, fell);
      hf_used_ = used;
      hf_fell_ = fell;
    }
  }

  // One padded coordinate moved by any offset: wrapped in a periodic direction,
  // clamped into the padded array (onto the halo, which no stencil accepts) in
  // a walled one. Domain's own helper assumes an offset of one; this stencil
  // reaches three.
  KOKKOS_INLINE_FUNCTION static Index hf_axis(Index v, Index h, Index n, Index s, bool per) {
    if (per) {
      Index r = (v - h) % n;
      if (r < 0) r += n;
      return h + r;
    }
    return v < 0 ? Index(0) : (v >= s ? s - 1 : v);
  }

  // exchange_geometric's reconstruction, read by surface_follows_line: does the
  // line of interface cell j reach into the cell at offset (dx, dy) from it?
  // For a Gas cell, is the shared face (axis) or corner (diagonal) wet; for a
  // Fluid cell, is it dry; for an Interface cell, does the line cross the shared
  // face (axis offsets only). A margin of 1e-3 keeps a line that merely touches a
  // face or corner from switching anything.
  template <class WV>
  KOKKOS_INLINE_FUNCTION static bool line_reaches(const WV& wf, Index j, int dx, int dy,
                                                  std::uint8_t want) {
    const Real tol = Real(1e-3);
    if (dx == 0 || dy == 0) {
      const int f = dx > 0 ? 0 : (dx < 0 ? 1 : (dy > 0 ? 2 : 3));
      const Real w = wf(8 * j + f);
      if (want == FsGas) return w > tol;
      if (want == FsFluid) return w < Real(1) - tol;
      return w > tol && w < Real(1) - tol;
    }
    if (want != FsGas && want != FsFluid) return false;
    const int c = dx > 0 ? (dy > 0 ? 4 : 7) : (dy > 0 ? 5 : 6);
    const Real dc = wf(8 * j + c);
    return want == FsGas ? dc < -tol : dc > tol;
  }

  // One height-function column through padded coordinates p, offset (k, l)
  // along the tangential axes t1, t2, along the axis a on which the liquid lies
  // toward +s: its height in z, or false where it is not well formed. cf is the
  // fill of interface cells (clamped to [0, 1] here).
  //
  // Each column is walked from the cell's own row, t = 0, with t counted
  // toward the GAS: toward the liquid to the first Fluid cell (t_L), then
  // toward the gas to the first Gas cell (t_G), meeting only Interface cells
  // between. Its height is where the liquid ends, measured from that shared
  // row: t_L + 1/2 plus the fill of the cells between. Where a fixed window of
  // seven would have held the column it is the same number; at 45 degrees,
  // where the neighbouring column's transition sits a cell or two further
  // out, a fixed window centred on a nearly empty cell missed it (8 of the 56
  // interface cells of an R = 8 drop, all on the diagonals). The reach is
  // seven because five was one short in 3-D: on exact fills of a ball the
  // corner columns of a cell near (1,1,1) cross four interface cells two
  // rows out, so the first Fluid cell is six away (124 of 1416 cells failed).
  template <class FlagV, class ColV>
  KOKKOS_INLINE_FUNCTION static bool hf_column(const Domain& d, const FlagV& flags,
                                               const ColV& cf, const Index p[3], int a, int s,
                                               int t1, int t2, int k, int l, Real& z) {
    constexpr int J = 7;                             // reach, each way
    auto at = [&](int t) {                           // the cell t steps toward the gas
      Index q[3] = {p[0], p[1], p[2]};
      q[a] -= Index(s * t);
      q[t1] += Index(k);
      if (L::D == 3) q[t2] += Index(l);
      return d.idp(hf_axis(q[0], d.hx, d.nx, d.sx, d.periodic[0]),
                   hf_axis(q[1], d.hy, d.ny, d.sy, d.periodic[1]),
                   hf_axis(q[2], d.hz, d.nz, d.sz, d.periodic[2]));
    };
    int tl = 1;                                      // toward the liquid: first Fluid
    for (int t = 0; t >= -J; --t) {
      const std::uint8_t f = flags(at(t));
      if (f == FsFluid) { tl = t; break; }
      if (f != FsInterface && f != FsGas) return false;
    }
    if (tl > 0) return false;
    int tg = tl;                                     // toward the gas: first Gas
    z = Real(tl) + Real(0.5);
    for (int t = tl + 1; t <= J; ++t) {
      const Index m = at(t);
      const std::uint8_t f = flags(m);
      if (f == FsGas) { tg = t; break; }
      if (f == FsFluid) { z += Real(1); continue; }  // full, past a part-full cell
      if (f != FsInterface) return false;
      Real c = cf(m);
      c = c < Real(0) ? Real(0) : (c > Real(1) ? Real(1) : c);
      z += c;
    }
    return tg != tl;                                 // false: no gas within reach
  }

  // The height-function curvature at interface cell n, or false where the
  // stencil is not well formed. See the member height_function. The heights are
  // the LIQUID in each seven-cell window, whichever side the liquid is on, so
  // kappa = -h''/(1 + h'^2)^(3/2) is positive for a drop on either side of it:
  // over the top of a disc h = c - x^2/(2R), and under it the liquid is counted
  // from the other end and h has the same form.
  template <class FlagV, class ColV>
  KOKKOS_INLINE_FUNCTION static bool height_curvature(const Domain& d, const FlagV& flags,
                                                      const ColV& cf, Real nx_, Real ny_,
                                                      Real nz_, Index n, Real& kappa,
                                                      int hw = 1) {
    const Real nv[3] = {nx_, ny_, nz_};
    int a = 0;
    for (int k = 1; k < L::D; ++k)
      if (Kokkos::fabs(nv[k]) > Kokkos::fabs(nv[a])) a = k;
    if (!(Kokkos::fabs(nv[a]) > Real(0))) return false;
    const int s = nv[a] > Real(0) ? 1 : -1;          // the liquid lies toward +s along a
    const int t1 = (L::D == 2) ? 1 - a : (a + 1) % 3;
    const int t2 = (a + 2) % 3;                      // read only in 3-D
    constexpr int lw = (L::D == 3) ? 1 : 0;
    Index p[3];
    d.coords(n, p[0], p[1], p[2]);
    // The height of column (k, l), or false where it is not well formed.
    auto column = [&](int k, int l, Real& z) -> bool {
      return hf_column(d, flags, cf, p, a, s, t1, t2, k, l, z);
    };
    // height_fit (2-D only; the setter refuses it in 3-D): the 2 hw + 1 heights
    // smoothed hw - 1 times with [1 2 1]/4, then the central differences of the
    // middle three. Each pass multiplies the second difference's response to a
    // wavenumber k by cos^2(k/2), never negative, and leaves a parabola's h' and
    // h'' exact. See the member height_fit for the least-squares fit this is not.
    if (L::D == 2 && hw > 1) {
      Real g[9] = {Real(0), Real(0), Real(0), Real(0), Real(0),
                   Real(0), Real(0), Real(0), Real(0)};          // hw <= 4
      for (int k = -hw; k <= hw; ++k)
        if (!column(k, 0, g[k + hw])) return false;
      for (int pass = 1; pass < hw; ++pass)      // in place: g[i] reads i, i+1, i+2
        for (int i = 0; i <= 2 * (hw - pass); ++i)
          g[i] = Real(0.25) * (g[i] + Real(2) * g[i + 1] + g[i + 2]);
      const Real hx = Real(0.5) * (g[2] - g[0]);  // g[0..2]: columns -1, 0, 1
      const Real hxx = g[2] - Real(2) * g[1] + g[0];
      const Real q = Real(1) + hx * hx;
      kappa = -hxx / (q * Kokkos::sqrt(q));
      return true;
    }
    Real h[3][3] = {{Real(0), Real(0), Real(0)}, {Real(0), Real(0), Real(0)},
                    {Real(0), Real(0), Real(0)}};
    for (int k = -1; k <= 1; ++k)
      for (int l = -lw; l <= lw; ++l)
        if (!column(k, l, h[k + 1][l + 1])) return false;
    const Real hx = Real(0.5) * (h[2][1] - h[0][1]);
    const Real hxx = h[2][1] - Real(2) * h[1][1] + h[0][1];
    if (L::D == 2) {
      const Real q = Real(1) + hx * hx;
      kappa = -hxx / (q * Kokkos::sqrt(q));
      return true;
    }
    const Real hy = Real(0.5) * (h[1][2] - h[1][0]);
    const Real hyy = h[1][2] - Real(2) * h[1][1] + h[1][0];
    const Real hxy = Real(0.25) * (h[2][2] - h[2][0] - h[0][2] + h[0][0]);
    const Real q = Real(1) + hx * hx + hy * hy;
    kappa = -(hxx * (Real(1) + hy * hy) + hyy * (Real(1) + hx * hx) - Real(2) * hxy * hx * hy)
            / (q * Kokkos::sqrt(q));
    return true;
  }

  // A contact angle acts only through the curvature and only once it has been
  // switched on, so each way of half-setting it would run without it and say
  // nothing. Refuse them instead.
  void check_wetting() const {
    if (wet_cos_of.data() && !wetting)
      Kokkos::abort("FreeSurfaceSolver: wet_cos_of is set but set_contact_angle() was not "
                    "called, so it would be ignored");
    if (wetting && !wet_.data())
      Kokkos::abort("FreeSurfaceSolver: wetting = true without set_contact_angle(); call that");
    if (wetting && sigma == Real(0))
      Kokkos::abort("FreeSurfaceSolver: a contact angle acts only through surface tension, "
                    "and sigma is 0");
  }

  void step() {
    if (wetting || wet_cos_of.data()) check_wetting();
    if (sigma != Real(0) || normal_reconstruction) compute_curvature();
    stream_collide();
    mass_exchange();
    classify();
    close_interface();
    pop_.end_of_step();
    ++t_;
  }

  //---- accessors --------------------------------------------------------------
  View1D<std::uint8_t> flags() const { return flags_; }
  //----------------------------------------------------------------------------
  // Non-zero where this cell's populations AND stored velocity were rebuilt from
  // its neighbours during the step just finished -- that is, where a gas cell
  // became an interface cell and had to be given a state from nothing.
  //
  // ANY MODULE THAT FORCED THE FLUID LAST STEP NEEDS THIS. A direct-forcing
  // scheme like PenalisedBody applies a force, then on the next step subtracts
  // its own contribution back out of the stored velocity. That subtraction is
  // only valid if the velocity still contains the contribution -- and in a
  // reinitialised cell it does not, because the average of the neighbours
  // replaced it. Subtracting a force whose effect was erased is the same
  // runaway, and it is not small: it is what a body touching a free surface
  // does to itself within a few hundred steps.
  //----------------------------------------------------------------------------
  View1D<std::uint8_t> reinitialised() const { return reinit_; }

  //----------------------------------------------------------------------------
  // The force and torque the fluid exerted on the obstacle during the last
  // step, by momentum exchange. Unlike PenalisedBody's reaction -- which is a
  // closed form derived from the solve -- this is a genuine measurement: it is
  // summed link by link over the boundary from the populations that actually
  // bounced.
  //----------------------------------------------------------------------------
  struct Exchange { Real fx = 0, fy = 0, torque = 0; };
  Exchange exchange() const { return Exchange{fx_, fy_, tz_}; }

  //----------------------------------------------------------------------------
  // Move the obstacle. `inside(n)` says whether cell n is now within the body;
  // `wall_vel(n, u[3])` gives the body's surface velocity there. Both are device
  // lambdas, so the solver needs no notion of a body.
  //
  // THIS RUNS BEFORE step(), and it must: a cell that is solid at the start of a
  // step can never be given a state during one -- promote() and settle() both
  // return early on FsSolid -- so refill has to happen first or not at all.
  //
  // MASS IS MOVED FROM THE FRONT OF THE BODY TO THE BACK, which is what a solid
  // sweeping through liquid physically does. The cells the body newly COVERS
  // hand their mass to the cells it newly UNCOVERS, in equal shares. Both sets
  // are the same size to within the body's motion in one step, so the transfer
  // is local, and it is exactly conservative by construction rather than by an
  // argument about fluxes. If nothing was uncovered -- a body that has stopped --
  // the covered mass goes into the ordinary excess machinery instead, and if
  // that has no takers either it is reported as lost rather than dropped
  // silently.
  //----------------------------------------------------------------------------
  template <class InsideFn, class WallFn>
  void move_obstacle(InsideFn inside, WallFn wall_vel, Real cx, Real cy) {
    bcx_ = cx;  bcy_ = cy;
    mark_obstacle(inside, wall_vel);
    transfer_covered_mass();
  }
  //----------------------------------------------------------------------------
  // REMOVE THE LIQUID IN A REGION -- the only way liquid can LEAVE this solver,
  // which otherwise has walls and nothing else. `inside(n)` is a device lambda.
  // Call it between steps; it returns the mass it removed and adds it to
  // drained_mass(), so total_mass() + drained_mass() is the ledger a case with
  // a drain must close, the way no_refill cases close theirs with
  // ejected_mass().
  //
  // IT IS A CONVERSION, NOT A NEW MECHANISM. A drained cell is turned to Gas
  // with its excess zeroed -- its mass leaves rather than being handed on --
  // and then promote() and settle() run exactly as they do at the end of a
  // step: a Fluid cell left facing the new gas becomes Interface and keeps
  // its whole density as mass, so the shell stays closed and nothing else is
  // created or destroyed. excess_ is cleared first because settle() would
  // otherwise hand out the previous step's published excess a second time.
  //
  // THE FLUID NEIGHBOURS ARE DEMOTED HERE, BEFORE promote(), AND THAT IS THE
  // FIX OF A MEASURED BUG. promote() closes the shell in BOTH directions, so
  // left to it a drained cell beside a cell that is still Fluid is pulled back
  // to Interface -- and since its old flag was Fluid, settle() gives it its
  // whole density back. The first version did exactly that: 31 units created
  // on a 126-unit drain in the unit test. Demoting the neighbours first leaves
  // every drained cell with no Fluid neighbour, so promote() has nothing to
  // reverse.
  //
  // settle() also clears reinitialised(), as it does at the end of a step, so
  // a module that reads that flag (PenalisedBody's force correction) must read
  // it before a drain, not after. No case pairs the two today.
  //
  // Opt-in: a case that never calls it is unchanged. Added 2026-11-03 for
  // demonstrator/mew_nozzle.cpp, where melt laid on a moving collector must
  // leave the box rather than pile against its far wall;
  // tests/test_free_surface.cpp block 4 pins the ledger and the closure.
  //----------------------------------------------------------------------------
  template <class InsideFn>
  Real drain(InsideFn inside) {
    auto flags = flags_, newf = newf_;
    auto mass = mass_, rho = rho_, excess = excess_;
    Real removed = 0;
    Kokkos::parallel_reduce("fs_drain", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc) {
        excess(n) = Real(0);
        const std::uint8_t f = flags(n);
        newf(n) = f;
        if ((f == FsFluid || f == FsInterface) && inside(n)) {
          acc += (f == FsFluid) ? rho(n) : mass(n);
          newf(n) = FsGas;
        }
      }, removed);
    Kokkos::fence();
    // A closed shell means no Fluid cell had a Gas neighbour before this call,
    // so any Gas in newf_ beside a Fluid cell is a cell just drained. Only
    // Fluid cells are written, and they test only neighbours that are Gas,
    // which this pass never writes.
    const Domain d = dom_;
    Kokkos::parallel_for("fs_drain_shell", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) != FsFluid || newf(n) != FsFluid) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      for (int i = 1; i < Q; ++i)
        if (newf(nb.j[i]) == FsGas) { newf(n) = FsInterface; return; }
    });
    Kokkos::fence();
    promote();
    settle();
    drained_ += removed;
    return removed;
  }
  Real drained_mass() const { return drained_; }

  //----------------------------------------------------------------------------
  // PROJECT THE STATE ONTO ITS MIRROR-SYMMETRIC PART about the plane between
  // columns x and xsum - x (xsum = nx - 1 for a box symmetric about its middle).
  // Call it between steps, and only while the problem IS symmetric: it is not a
  // model, it removes an asymmetry the exact solution does not have.
  //
  // WHY IT EXISTS. demonstrator/mew_nozzle extrudes a column into a set-up
  // mirror-symmetric to the cell, and the column drifted sideways anyway --
  // -0.42 / +0.27 mm by 4.5 s with the field on / off at R = 8. At R = 4 the
  // front's offset grew x1.9..3.4 every 0.5 s from round-off while the root
  // stayed on the axis to 0.002 mm.
  // Its conversions are discrete, so mirror cells that differ in the last bit
  // convert a step apart and the asymmetry jumps to a fill_offset's worth of
  // mass; something in the advancing front then amplifies it. That something is
  // NOT diagnosed (normal_reconstruction does not remove it; see that member).
  // What is known is that the exact solution is symmetric, so this projection
  // is the defensible cure, and the case stops calling it at first contact,
  // where the moving collector breaks the symmetry for real.
  //
  // Per mirror pair: a cell is Fluid only if both were (so no Fluid cell can
  // gain a Gas neighbour), Gas only if both were, Interface otherwise; mass,
  // density and populations are averaged with the partner's MIRRORED ones (u_x
  // changes sign); a cell that was Gas takes its partner's state. Mass is
  // conserved exactly. Both population arrays are written, as mark_obstacle's
  // refill does, since TwoLattice swaps at end_of_step.
  //
  // It also suppresses any GENUINE lateral instability. That is the price, and
  // the reason it must be switched off the moment the problem stops being
  // symmetric. Added 2026-11-03; tests/test_free_surface.cpp block 5 pins it.
  //----------------------------------------------------------------------------
  void mirror_symmetrize_x(Index xsum) {
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, newf = newf_, fin = final_;
    auto mass = mass_, eps = eps_, rho = rho_, ux = ux_, uy = uy_, uz = uz_;
    const Index hx = dom_.hx, hy = dom_.hy, hz = dom_.hz;
    Kokkos::parallel_for("fs_mirror", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      Index px, py, pz; d.coords(n, px, py, pz);
      if (!d.is_interior(px, py, pz)) return;
      const Index x = px - hx, xm = xsum - x;
      if (x >= xm || xm < 0 || xm >= d.nx) return;      // each pair once
      const Index m = d.id(xm, py - hy, pz - hz);
      const std::uint8_t fa = flags(n), fb = flags(m);
      const bool la = fa == FsFluid || fa == FsInterface;
      const bool lb = fb == FsFluid || fb == FsInterface;
      if (!la && !lb) return;                             // both gas, or walls
      if ((fa == FsSolid || fa == FsExcluded) || (fb == FsSolid || fb == FsExcluded)) return;
      const std::uint8_t g = (fa == FsFluid && fb == FsFluid) ? FsFluid : FsInterface;
      const Real ma = (fa == FsFluid) ? rho(n) : (fa == FsInterface ? mass(n) : Real(0));
      const Real mb = (fb == FsFluid) ? rho(m) : (fb == FsInterface ? mass(m) : Real(0));
      const Real mm = Real(0.5) * (ma + mb);
      // populations: average with the partner's mirror image, or copy it
      for (int i = 0; i < Q; ++i) {
        int j = i;                                        // the x-mirror of i
        for (int k = 0; k < Q; ++k)
          if (cvel<L>(k, 0) == -cvel<L>(i, 0) && cvel<L>(k, 1) == cvel<L>(i, 1) &&
              cvel<L>(k, 2) == cvel<L>(i, 2)) { j = k; break; }
        if (j < i) continue;                              // handle each pair (i, j) once
        const Real an = acc.src(n, i), am = acc.src(m, j);
        const Real bn = acc.src(n, j), bm = acc.src(m, i);
        Real vi, vj;                                      // n's new f_i and f_j
        if (la && lb) { vi = Real(0.5) * (an + am); vj = Real(0.5) * (bn + bm); }
        else if (la)  { vi = an; vj = bn; }
        else          { vi = am; vj = bm; }
        acc.src(n, i) = vi; acc.dst(n, i) = vi;
        acc.src(n, j) = vj; acc.dst(n, j) = vj;
        acc.src(m, j) = vi; acc.dst(m, j) = vi;           // m is n's mirror image
        acc.src(m, i) = vj; acc.dst(m, i) = vj;
      }
      const Real r = (la && lb) ? Real(0.5) * (rho(n) + rho(m)) : (la ? rho(n) : rho(m));
      const Real u = (la && lb) ? Real(0.5) * (ux(n) - ux(m)) : (la ? ux(n) : -ux(m));
      const Real v = (la && lb) ? Real(0.5) * (uy(n) + uy(m)) : (la ? uy(n) : uy(m));
      const Real w = (la && lb) ? Real(0.5) * (uz(n) + uz(m)) : (la ? uz(n) : uz(m));
      rho(n) = r; rho(m) = r;
      ux(n) = u;  ux(m) = -u;
      uy(n) = v;  uy(m) = v;
      uz(n) = w;  uz(m) = w;
      mass(n) = (g == FsFluid) ? r : mm;
      mass(m) = mass(n);
      eps(n) = (g == FsFluid) ? Real(1) : (r > Real(0) ? mm / r : Real(0));
      eps(m) = eps(n);
      flags(n) = g; flags(m) = g;
      newf(n) = g;  newf(m) = g;
      fin(n) = g;   fin(m) = g;
    });
    Kokkos::fence();
  }

  // Cumulative mass removed because no neighbour could accept it. Zero unless
  // no_refill is on. total_mass() + ejected_mass() is the conserved quantity
  // there, and it is what a case must assert instead of total_mass() alone.
  Real ejected_mass() const { return ejected_; }
  View1D<Real> fill() const { return eps_; }
  View1D<Real> mass() const { return mass_; }
  View1D<Real> rho()  const { return rho_; }
  View1D<Real> ux() const { return ux_; }
  View1D<Real> uy() const { return uy_; }
  View1D<Real> uz() const { return uz_; }
  // The curvature working set, as compute_curvature() last left it. Empty until
  // surface tension, the normal reconstruction or a contact angle is switched on.
  View1D<Real> kappa() const { return kap_; }
  View1D<Real> normal_x() const { return knx_; }
  View1D<Real> normal_y() const { return kny_; }
  View1D<Real> normal_z() const { return knz_; }
  const Domain& domain() const { return dom_; }
  std::size_t timestep() const { return t_; }

  // The conserved quantity, and the one that can be silently lost. Fluid cells
  // count their full density; interface cells count their explicit mass.
  Real total_mass() const {
    auto flags = flags_; auto mass = mass_; auto rho = rho_;
    Real s = 0;
    Kokkos::parallel_reduce("fs_mass", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc) {
        const std::uint8_t f = flags(n);
        if (f == FsFluid)          acc += rho(n);
        else if (f == FsInterface) acc += mass(n);
      }, s);
    return s;
  }

  // How many cells of each kind -- the cheapest way to see an interface that is
  // dissolving or a conversion storm.
  struct Census { Index gas = 0, interface_ = 0, fluid = 0; };
  Census census() const {
    auto flags = flags_;
    Index g = 0, i2 = 0, fl = 0;
    Kokkos::parallel_reduce("fs_census", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Index& a, Index& b, Index& c) {
        const std::uint8_t f = flags(n);
        if (f == FsGas) ++a; else if (f == FsInterface) ++b; else if (f == FsFluid) ++c;
      }, g, i2, fl);
    return Census{g, i2, fl};
  }

 public:
  //----------------------------------------------------------------------------
  // NVCC CONSTRAINT, NOT A DESIGN CHOICE -- the same note FluidSolver,
  // PhaseFieldSolver and ColourGradientSolver carry, for the same reason: CUDA
  // forbids an extended host-device lambda inside a private member function.
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // Edit the mask, and refill what the body uncovered.
  //
  // REFILL IS EQUILIBRIUM AT THE WALL VELOCITY PLUS THE DONOR'S NON-EQUILIBRIUM.
  // A cell that was solid holds no distribution -- its slots carry reflected
  // populations, whose moments are not a fluid state -- so all Q populations
  // have to be constructed. The cheap choice is equilibrium alone, which is what
  // settle() uses for a gas cell arriving from nothing; it is wrong here,
  // because it discards the local viscous stress and the very next momentum
  // exchange on this cell's links reports that stress as a force. Copying the
  // non-equilibrium part from ONE fluid neighbour keeps it, needs no collinear
  // stencil -- so it does not fail in a narrow gap or against a thin body, where
  // a three-point normal extrapolation has nowhere to stand -- and cannot
  // overshoot, because it adds a valid non-equilibrium to a valid equilibrium.
  //
  // The donor is the fluid neighbour most nearly along the outward normal, which
  // here is approximated by the direction away from the body centre. seed_value
  // rather than Equilibrium::eq, so the refill agrees with whatever operator is
  // plugged in and with its storage convention.
  //----------------------------------------------------------------------------
  template <class InsideFn, class WallFn>
  void mark_obstacle(InsideFn inside, WallFn wall_vel) {
    Kokkos::deep_copy(was_body_, obst_);     // the mask before this move
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, newf = newf_, fin = final_, reinit = reinit_;
    auto mass = mass_, eps = eps_, rho = rho_, excess = excess_;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto wux = wux_, wuy = wuy_, wuz = wuz_;
    auto obst = obst_, uncov = uncov_;
    const Index hx = dom_.hx, hy = dom_.hy;
    const Real cx = bcx_, cy = bcy_;

    // Pass A: the mask itself, plus the wall velocity the body's cells need.
    Kokkos::parallel_for("fs_mask", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      uncov(n) = 0;                 // this move's own mask, cleared before use
      Real uw[3] = {Real(0), Real(0), Real(0)};
      wall_vel(n, uw);
      wux(n) = uw[0]; wuy(n) = uw[1]; wuz(n) = uw[2];
      const std::uint8_t was = flags(n);
      const std::uint8_t was_body = obst(n);
      if (was == FsExcluded) { fin(n) = was; obst(n) = 0; return; }
      const bool in = inside(n);
      // Only the BODY's cells carry a wall velocity; a static wall is at rest
      // and must stay there.
      if (!in) { wux(n) = Real(0); wuy(n) = Real(0); wuz(n) = Real(0); }
      obst(n) = in ? 1 : 0;
      if (in) {
        fin(n) = FsSolid;
        if (!was_body && was != FsSolid) {
          excess(n) = mass(n); mass(n) = Real(0); eps(n) = Real(0);
        }
      } else {
        // Only a cell the BODY has left needs refilling; a static wall never
        // becomes anything.
        fin(n) = was_body ? FsInterface : was;
      }
    });
    Kokkos::fence();

    // Pass B: refill, reading the OLD flags for donors so nothing half-updated
    // is used as a source.
    auto obst_prev = was_body_;
    Kokkos::parallel_for("fs_refill", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (!(obst_prev(n) && fin(n) != FsSolid)) return;
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      const Real rx = Real(Index(n % 1) ) ;  (void)rx;
      Index px, py, pz; d.coords(n, px, py, pz);
      const Real ox = Real(px - hx) - cx, oy = Real(py - hy) - cy;
      const Real om = Kokkos::sqrt(ox * ox + oy * oy);
      const Real nx2 = (om > Real(0)) ? ox / om : Real(0);
      const Real ny2 = (om > Real(0)) ? oy / om : Real(0);

      // The fluid neighbour most nearly along the outward normal.
      Index donor = n;  Real best = Real(-2);
      for (int i = 1; i < Q; ++i) {
        const Index j = nb.j[i];
        const std::uint8_t fj = flags(j);
        if (fj != FsFluid && fj != FsInterface) continue;
        const Real cxx = Real(cvel<L>(i, 0)), cyy = Real(cvel<L>(i, 1));
        const Real cm = Kokkos::sqrt(cxx * cxx + cyy * cyy);
        const Real dot = (cm > Real(0)) ? (cxx * nx2 + cyy * ny2) / cm : Real(-2);
        if (dot > best) { best = dot; donor = j; }
      }

      const Real uw[3] = {wux(n), wuy(n), wuz(n)};
      if (donor == n) {                       // no fluid neighbour at all
        for (int i = 0; i < Q; ++i)
          acc.dst(n, i) = Collision::seed_value(i, Real(1), uw[0], uw[1], uw[2]);
        rho(n) = Real(1);
      } else {
        const Real rd = rho(donor), ud = ux(donor), vd = uy(donor), wd = uz(donor);
        for (int i = 0; i < Q; ++i) {
          const Real fneq = acc.src(donor, i)
                          - Collision::seed_value(i, rd, ud, vd, wd);
          acc.dst(n, i) = Collision::seed_value(i, rd, uw[0], uw[1], uw[2]) + fneq;
        }
        rho(n) = rd;
      }
      // TwoLattice swaps at end_of_step, so a refill written to dst would only
      // be seen after the coming step's pull. Write both.
      for (int i = 0; i < Q; ++i) acc.src(n, i) = acc.dst(n, i);
      ux(n) = uw[0]; uy(n) = uw[1]; uz(n) = uw[2];
      mass(n) = Real(0); eps(n) = Real(0);
      reinit(n) = 1;  uncov(n) = 1;
    });
    Kokkos::fence();
    Kokkos::deep_copy(flags_, final_);
    Kokkos::deep_copy(newf_, final_);
  }

  // The covered cells' mass, handed to the uncovered ones.
  //
  // THE RECIPIENT TEST READS uncov_, NOT reinit_, and that was a real defect
  // rather than a tidying. reinit_ still holds settle()'s gas-arrivals from the
  // END of the previous step -- nothing clears it in between -- so an interface
  // cell that had just been built out of gas counted as a cell this body had
  // uncovered. Two consequences, both wrong: the share was divided among more
  // cells than the body exposed, so the genuinely uncovered ones came back too
  // empty and converted straight to gas, leaving a void in the body's wake; and
  // `mass(n) = share` OVERWROTE the mass those gas-arrivals had been given by
  // settle(), which is not conservative and is not this function's business.
  //
  // Measured on demonstrator/water_entry_fs.cpp: the run reaches t* = 8.86
  // against 6.16 before.
  void transfer_covered_mass() {
    auto flags = flags_; auto excess = excess_, mass = mass_, eps = eps_, rho = rho_;
    auto uncov = uncov_;
    Real m_cov = 0; Index n_unc = 0;
    Kokkos::parallel_reduce("fs_cover_sum", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc_m, Index& acc_n) {
        if (flags(n) == FsSolid) acc_m += excess(n);
        else if (uncov(n) && flags(n) == FsInterface) ++acc_n;
      }, m_cov, n_unc);
    Kokkos::fence();
    if (n_unc == 0) return;                  // leave it to the excess machinery
    const Real share = m_cov / Real(n_unc);
    Kokkos::parallel_for("fs_cover_give", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      if (flags(n) == FsSolid) { excess(n) = Real(0); return; }
      if (uncov(n) && flags(n) == FsInterface) {
        mass(n) = share;
        const Real r = rho(n);
        eps(n) = (r > Real(0)) ? share / r : Real(0);
      }
    });
    Kokkos::fence();
  }

  // Pass 1. Pull, reconstruct the gas-facing populations, collide, store.
  void stream_collide() {
    const auto acc = pop_.template access<0>();
    const auto cl  = coll;
    const Domain d = dom_;
    auto flags = flags_;
    auto rho = rho_, ux = ux_, uy = uy_, uz = uz_, mass = mass_, eps = eps_;
    auto wux = wux_, wuy = wuy_, wuz = wuz_;
    auto obst = obst_;
    const Real rg = rho_G;
    const auto rgf = rho_G_of;          // empty unless the caller set it
    const auto kap = kap_;              // empty unless surface tension is on
    const Real sig = sigma;
    constexpr Real icsL = inv_cs2<L, Real>();
    const Index hx = dom_.hx, hy = dom_.hy;
    const Real bcx = bcx_, bcy = bcy_;
    constexpr Real ics = inv_cs2<L, Real>();
    const bool nrec = normal_reconstruction && knx_.data();
    const auto knx = knx_, kny = kny_, knz = knz_;

    Real sfx = 0, sfy = 0, stz = 0;
    Kokkos::parallel_reduce("fs_stream_collide", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& afx, Real& afy, Real& atz) {
        const std::uint8_t fl = flags(n);
        if (fl != FsFluid && fl != FsInterface) return;
        // Same test as ScalarBGK's omega_of: the field, once set, is
        // authoritative and the scalar is not consulted.
        Real rgn = rgf.data() ? rgf(n) : rg;
        // Surface tension ADDS to whatever pressure the caller imposed: a
        // recoil field and a Laplace jump are both normal stresses and they
        // superpose. p -> p + sigma kappa, i.e. rho -> rho + sigma kappa / cs^2.
        if (sig != Real(0) && kap.data()) rgn += sig * kap(n) * icsL;

        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);

        // The velocity that goes into the reconstruction is the previous step's;
        // see the banner. rho_G, not rho(n), sets the gas pressure.
        const Real u0[3] = {ux(n), uy(n), uz(n)};

        // The normal-reconstruction test: a population travelling along +n (into
        // the liquid) came from the gas side, wherever its source cell sits.
        const Real nn[3] = {nrec ? knx(n) : Real(0), nrec ? kny(n) : Real(0),
                            nrec ? knz(n) : Real(0)};
        const bool along = nrec && fl == FsInterface;

        Real f[Q];
        for (int i = 0; i < Q; ++i) {
          const Index from = (i == 0) ? n : nb.j[opp(i)];
          const std::uint8_t ff = flags(from);
          // Gas or Interface sources only: a Fluid neighbour is inside the liquid
          // whatever a smoothed normal says, and overwriting what it sent broke
          // the column into specks within a second in the first version.
          const bool inward = along && i != 0 && ff == FsInterface &&
              (Real(cvel<L>(i, 0)) * nn[0] + Real(cvel<L>(i, 1)) * nn[1] +
               Real(cvel<L>(i, 2)) * nn[2]) > Real(0);
          if (ff == FsGas || inward) {
            // Free-surface condition: impose the gas pressure through the two
            // moments of an anti-bounce-back pair, against this cell's own
            // outgoing population in the opposite direction. The equilibrium is
            // the COLLISION's own, so the boundary condition and the operator
            // agree about what equilibrium means.
            f[i] = Collision::seed_value(i, rgn, u0[0], u0[1], u0[2])
                 + Collision::seed_value(opp(i), rgn, u0[0], u0[1], u0[2])
                 - acc.src(n, opp(i));
          } else if (ff == FsSolid || ff == FsExcluded) {
            // HALFWAY BOUNCE-BACK OFF A MOVING WALL, and the momentum it
            // delivers. `i` is the direction the population comes BACK along, so
            // the wall lies along c_opp(i) and the population that went into it
            // is acc.src(n, opp(i)).
            //
            // The correction is exactly 2 w_i rho (c_i . u_w) / cs^2. It is not
            // an expansion: it is the difference between the equilibria in
            // direction i and opp(i) at the wall velocity, whose even parts
            // cancel identically, and it is what makes a fluid moving WITH the
            // wall a fixed point of the boundary. The 1/cs^2 is written as
            // inv_cs2 and never folded into a literal 6 -- that is only right
            // for the cs^2 = 1/3 lattices.
            const Real fout = acc.src(n, opp(i));
            const Real cuw = Real(cvel<L>(i, 0)) * wux(from)
                           + Real(cvel<L>(i, 1)) * wuy(from)
                           + Real(cvel<L>(i, 2)) * wuz(from);
            const bool is_body = obst(from) != 0;
            const Real corr = is_body
                ? Real(2) * weight<L, Real>(i) * rho(n) * cuw * ics : Real(0);
            f[i] = fout + corr;
            // MOMENTUM EXCHANGE. What the link hands the wall is
            // c_q (f_out + f_back) with c_q = -c_i: the two travel in opposite
            // directions, so after the reversal their momenta ADD -- the ball
            // that bounces delivers 2mv, not mv. Summing the pair rather than
            // the expanded form keeps this correct if the closure ever changes.
            //
            // Only the liquid pushes: an interface cell a tenth full has a tenth
            // of the fluid on that link, so the exchange is scaled by the fill.
            if (is_body) {
              // THE GAS PRESSURE HAS TO BE PUT BACK. Momentum exchange delivers
              // the ABSOLUTE pressure force on each link. Around a fully wetted
              // closed body that cancels -- sum_links w_q c_q = 0 -- and only
              // the gradient survives, which is buoyancy. A partly submerged
              // body has no such cancellation: its dry side faces gas, and the
              // gas is a void with no populations, so it pushes back with
              // nothing. The body then feels a whole atmosphere on its wetted
              // face and none on top.
              //
              // Measured, that is not a small bias: a square entering at Mach
              // 0.07 reversed its velocity within forty steps under a force
              // twenty times the physical slamming load.
              //
              // The fix needs no extra pass. Since the dry links would carry
              // exactly 2 w_q rho_G each and the whole closed surface sums to
              // zero, subtracting that same constant from every WET link is
              // identical to adding it on every dry one. What is left is the
              // gauge pressure, which is what a surface force is.
              const Real wgt = (fl == FsInterface) ? eps(n) : Real(1);
              const Real p = wgt * (fout + f[i]
                                  - Real(2) * weight<L, Real>(i) * rgn);
              const Real dx = -Real(cvel<L>(i, 0)) * p;
              const Real dy = -Real(cvel<L>(i, 1)) * p;
              afx += dx;  afy += dy;
              Index qx, qy, qz; d.coords(n, qx, qy, qz);
              atz += (Real(qx - hx) - bcx) * dy - (Real(qy - hy) - bcy) * dx;
            }
          } else {
            f[i] = acc.src(from, i);
          }
        }

        // The collision operator owns the macroscopics, the force and the
        // relaxation. Everything above this line is the free surface; everything
        // below it is the same operator the single-phase solver runs.
        const Macro mac = cl.macroscopic(f, n);
        const Real r = Collision::density(mac);
        cl.collide(f, mac, n);
        for (int i = 0; i < Q; ++i) acc.dst(n, i) = f[i];

        rho(n) = r;
        ux(n) = mac.ux; uy(n) = mac.uy; uz(n) = mac.uz;
        if (fl == FsFluid) mass(n) = r;    // a full cell's mass IS its density
      }, sfx, sfy, stz);
    Kokkos::fence();
    fx_ = sfx;  fy_ = sfy;  tz_ = stz;
  }

  // Pass 2. Mass across faces, from the post-collision state of both cells.
  //
  // THIS IS WHERE THE FILL RIPPLES COME FROM, and they are not a bug in it but
  // what it is. Each link of an interface cell carries its net population flux
  // weighted by 1 to a Fluid cell, by the mean fill to an Interface cell and by
  // 0 to Gas. For liquid sliding ALONG a flat surface the fills should not
  // change, and with equilibrium populations each link's net flux is
  // -2 w_i rho (c_i . u)/cs^2, so the weights sum that to zero only where the
  // cell's neighbourhood is symmetric about the flow. A surface at slope 0 or 1
  // has every cell alike and is exact (validation/fs_band: zero and 4e-15); at
  // slope 1/2 or 3/7 the steps are uneven and each cell drifts by its own
  // amount -- the formula, evaluated from the initial state, predicts the
  // solver's change cell by cell at correlation 0.9998-0.9999 and amplitude
  // 0.9994-1.0004 -- so at first the column sums scatter by about a tenth of
  // the distance slid, in cells, whatever the box. A drop's surface has every
  // slope and its relaxation slides liquid along it; given that drop's surface
  // tension and surface speed, a sliding band settles at curvature noise of the
  // same order as the drop's (fs_band; the drop's is under height_function). The
  // fluid-link weight cannot simply be changed: a Fluid cell's mass IS its
  // density, so its links must carry the full flux or mass is not conserved.
  // THE FIX is exchange_geometric with exchange_donor and surface_follows_line
  // (opt-in, 2-D): the flux routed through faces and weighted by how much of
  // each face the reconstructed surface wets, and cells switched on where that
  // surface reaches; this pass is then not called (mass_exchange_geometric()).
  void mass_exchange() {
    if (exchange_geometric) { mass_exchange_geometric(); return; }
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_; auto mass = mass_, eps = eps_;
    const bool upw = exchange_upwind;

    Kokkos::parallel_for("fs_mass_exchange", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n) {
        if (flags(n) != FsInterface) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        const Real e0 = eps(n);
        Real dm = Real(0);
        for (int i = 1; i < Q; ++i) {
          const Index j = nb.j[i];
          const std::uint8_t fj = flags(j);
          // What j sends me in the opposite direction, minus what I send j.
          const Real in  = acc.dst(j, opp(i));
          const Real out = acc.dst(n, i);
          if (fj == FsFluid) {
            dm += in - out;
          } else if (fj == FsInterface) {
            // Two partly filled cells can only exchange in proportion to how
            // much liquid there is to exchange. The mean of the two fill levels
            // is symmetric between them, which is what keeps the pair's total
            // mass unchanged. exchange_upwind takes the fill of the cell the net
            // flow comes FROM instead -- the same cell seen from either side, so
            // it is just as conservative -- and see that member for why.
            const Real w = upw ? ((in > out) ? eps(j) : e0) : Real(0.5) * (e0 + eps(j));
            dm += w * (in - out);
          }
          // Gas and wall neighbours exchange nothing.
        }
        mass(n) += dm;
      });
    Kokkos::fence();
  }

  // Pass 2, geometric (exchange_geometric, 2-D): the wet fraction of every face
  // of every interface cell, then the exchange through faces. See the member.
  void mass_exchange_geometric() {
    if constexpr (L::D != 2) {
      Kokkos::abort("FreeSurfaceSolver: exchange_geometric is 2-D only");
    } else {
      if (wf_.size() == 0) wf_ = View1D<Real>("fs_wetface", 8 * dom_.n_padded);
      struct Tab { int i[9]; };                    // direction index of (dx, dy)
      Tab tab{};
      for (int i = 0; i < Q; ++i) tab.i[(cvel<L>(i, 0) + 1) * 3 + (cvel<L>(i, 1) + 1)] = i;
      const auto acc = pop_.template access<0>();
      const Domain d = dom_;
      auto flags = flags_; auto mass = mass_, eps = eps_, wf = wf_;
      auto rho = rho_, ux = ux_, uy = uy_, uz = uz_;
      const bool donor = exchange_donor;

      Kokkos::parallel_for("fs_wet_faces", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        if (flags(n) != FsInterface) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        auto clamp01 = [](Real v) { return v < Real(0) ? Real(0) : (v > Real(1) ? Real(1) : v); };
        const Real e0 = clamp01(eps(n));
        auto colour = [&](int dx, int dy) {
          const std::uint8_t f = flags(nb.j[tab.i[(dx + 1) * 3 + (dy + 1)]]);
          return f == FsFluid ? Real(1) : (f == FsInterface ? clamp01(eps(nb.j[tab.i[(dx + 1) * 3 + (dy + 1)]]))
                                                            : (f == FsGas ? Real(0) : e0));
        };
        // Youngs: the gradient of the fill, which points into the liquid.
        const Real gx = (colour(1, 1) + Real(2) * colour(1, 0) + colour(1, -1))
                      - (colour(-1, 1) + Real(2) * colour(-1, 0) + colour(-1, -1));
        const Real gy = (colour(1, 1) + Real(2) * colour(0, 1) + colour(-1, 1))
                      - (colour(1, -1) + Real(2) * colour(0, -1) + colour(-1, -1));
        Real nx = -gx, ny = -gy;                   // the normal into the GAS
        const int a = Kokkos::fabs(gx) >= Kokkos::fabs(gy) ? 0 : 1;
        const Real ga = a == 0 ? gx : gy;
        if (ga != Real(0)) {                       // height functions where they hold
          const int sg = ga > Real(0) ? 1 : -1;    // the liquid lies toward +sg along a
          Index p[3];
          d.coords(n, p[0], p[1], p[2]);
          Real z[3];
          bool ok = true;
          for (int k = -1; k <= 1 && ok; ++k) ok = hf_column(d, flags, eps, p, a, sg, 1 - a, 2, k, 0, z[k + 1]);
          if (ok) {
            const Real zp = Real(0.5) * (z[2] - z[0]);
            const Real q = Real(1) / Kokkos::sqrt(Real(1) + zp * zp);
            const Real nt = -zp * q, na = -Real(sg) * q;
            nx = a == 0 ? na : nt;
            ny = a == 0 ? nt : na;
          }
        }
        const Real l1 = Kokkos::fabs(nx) + Kokkos::fabs(ny);
        if (!(l1 > Real(0))) {                     // no direction at all: wet in proportion
          for (int f = 0; f < 4; ++f) wf(8 * n + f) = e0;
          for (int c = 4; c < 8; ++c) wf(8 * n + c) = Real(0);   // on the line: reaches nowhere
          return;
        }
        nx /= l1;  ny /= l1;
        // The line n.x = alpha (cell centred, liquid below it) holding fill e0.
        const Real m1 = Kokkos::fmin(Kokkos::fabs(nx), Kokkos::fabs(ny));
        const Real m2 = Kokkos::fmax(Kokkos::fabs(nx), Kokkos::fabs(ny));
        Real ap;
        if (m1 < Real(1e-12))                         ap = e0;
        else if (e0 < m1 / (Real(2) * m2))            ap = Kokkos::sqrt(Real(2) * m1 * m2 * e0);
        else if (e0 <= Real(1) - m1 / (Real(2) * m2)) ap = m2 * e0 + Real(0.5) * m1;
        else ap = Real(1) - Kokkos::sqrt(Real(2) * m1 * m2 * (Real(1) - e0));
        const Real al = ap - Real(0.5);
        auto wet = [&](Real x0, Real y0, Real x1, Real y1) {
          const Real d0 = nx * x0 + ny * y0 - al, d1 = nx * x1 + ny * y1 - al;
          if (d0 <= Real(0) && d1 <= Real(0)) return Real(1);
          if (d0 >= Real(0) && d1 >= Real(0)) return Real(0);
          return d0 < Real(0) ? -d0 / (d1 - d0) : -d1 / (d0 - d1);
        };
        const Real h = Real(0.5);
        wf(8 * n + 0) = wet( h, -h,  h,  h);       // +x
        wf(8 * n + 1) = wet(-h, -h, -h,  h);       // -x
        wf(8 * n + 2) = wet(-h,  h,  h,  h);       // +y
        wf(8 * n + 3) = wet(-h, -h,  h, -h);       // -y
        // and the signed distance of each corner from the line, negative in the
        // liquid: (+,+), (-,+), (-,-), (+,-). Read by surface_follows_line.
        wf(8 * n + 4) = nx * h + ny * h - al;
        wf(8 * n + 5) = -nx * h + ny * h - al;
        wf(8 * n + 6) = -nx * h - ny * h - al;
        wf(8 * n + 7) = nx * h - ny * h - al;
      });
      Kokkos::fence();

      Kokkos::parallel_for("fs_mass_exchange_faces", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
        if (flags(n) != FsInterface) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        auto at = [&](int dx, int dy) { return nb.j[tab.i[(dx + 1) * 3 + (dy + 1)]]; };
        auto dir = [&](int dx, int dy) { return tab.i[(dx + 1) * 3 + (dy + 1)]; };
        auto solid = [](std::uint8_t f) { return f == FsSolid || f == FsExcluded; };
        auto liquid = [](std::uint8_t f) { return f == FsFluid || f == FsInterface; };
        // The net flux from cell ia to cell ib along direction i (c_i = ib - ia).
        auto delta = [&](Index ia, Index ib, int i) -> Real {
          const std::uint8_t fa = flags(ia), fb = flags(ib);
          if (solid(fa) || solid(fb)) return Real(0);
          const bool la = liquid(fa), lb = liquid(fb);
          if (la && lb) return acc.dst(ia, i) - acc.dst(ib, opp(i));
          if (!la && !lb) return Real(0);
          const Index m = la ? ia : ib;
          return Collision::seed_value(i, rho(m), ux(m), uy(m), uz(m))
               - Collision::seed_value(opp(i), rho(m), ux(m), uy(m), uz(m));
        };
        const int ex[4] = {1, -1, 0, 0}, ey[4] = {0, 0, 1, -1};
        const int back[4] = {1, 0, 3, 2};          // the same face, seen from the other cell
        Real dm = Real(0);
        for (int f = 0; f < 4; ++f) {
          const int px = -ey[f], py = ex[f];       // along the face
          const Index j = at(ex[f], ey[f]);
          const std::uint8_t fj = flags(j);
          if (fj == FsGas || solid(fj)) continue;
          const Real phi = delta(n, j, dir(ex[f], ey[f]))
              + Real(0.5) * (delta(n, at(ex[f] + px, ey[f] + py), dir(ex[f] + px, ey[f] + py))
                           + delta(n, at(ex[f] - px, ey[f] - py), dir(ex[f] - px, ey[f] - py))
                           + delta(at(px, py), j, dir(ex[f] - px, ey[f] - py))
                           + delta(at(-px, -py), j, dir(ex[f] + px, ey[f] + py)));
          // exchange_donor: the face as the cell the flux LEAVES reconstructs it,
          // the same cell from either side since phi changes sign with it.
          const Real W = fj == FsFluid ? Real(1)
                       : donor ? (phi > Real(0) ? wf(8 * n + f) : wf(8 * j + back[f]))
                               : Real(0.5) * (wf(8 * n + f) + wf(8 * j + back[f]));
          dm -= W * phi;
        }
        mass(n) += dm;
      });
      Kokkos::fence();
    }
  }

  //----------------------------------------------------------------------------
  // Pass 3. Fill levels, and which cells want to convert.
  //
  // Two of the four conversion rules are about the fill level and two are about
  // the NEIGHBOURHOOD, and the second pair is not optional.
  //
  // A DETACHED INTERFACE CELL -- one with no liquid neighbour at all -- is a
  // speck of liquid with nothing touching it. Nothing can exert pressure on it,
  // so nothing balances gravity, and it accelerates at g for as long as the run
  // lasts. That is not a slow degradation: in a dam break the splash throws such
  // specks constantly, and each one is a free-fall trajectory that never ends.
  // Measured before this rule existed, max|u| grew LINEARLY at exactly g from
  // the moment the sheet first broke up, reaching Mach 1.2 by the end of an
  // otherwise healthy run, and the mass drift went with it.
  //
  // A BURIED INTERFACE CELL -- one with no gas neighbour -- looks like the
  // symmetric case and a rule for it was written, tested and REMOVED. Promoting
  // it to Fluid means declaring it full, and a cell that was thirty per cent
  // full then owes seventy per cent of a density to its neighbours through the
  // excess mechanism. Those neighbours are interface cells with little mass to
  // give; they go negative, convert to gas, expose more fluid, and the whole
  // surface unzips. Measured on the standing wave: the liquid was gone inside
  // six hundred steps, from a rule that fires on a handful of cells a step. The
  // detached rule alone leaves that case at 1.77%.
  //
  // The moral is that the two rules are not symmetric. Deleting a speck loses a
  // little mass and stops an unbounded velocity; filling a cell CREATES mass and
  // makes its neighbours pay, and there is no reason to think they can.
  //
  // The mass a detached cell takes with it is genuinely lost: by construction it
  // has no interface neighbour to hand it to. That is the documented
  // no-subgrid-interface limitation arriving in the accounts, and it is why the
  // demonstrators print mass drift every frame.
  //----------------------------------------------------------------------------
  void classify() {
    const Domain d = dom_;
    auto flags = flags_, newf = newf_;
    auto mass = mass_, eps = eps_, rho = rho_, excess = excess_;
    const Real off = fill_offset;
    const bool do_detach = drop_detached;
    const bool nrf = no_refill;
    const bool geo = exchange_geometric;
    const bool follow = exchange_geometric && surface_follows_line;
    auto wf = wf_;

    Kokkos::parallel_for("fs_classify", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t fl = flags(n);
      excess(n) = Real(0);
      newf(n) = fl;
      if (fl == FsFluid)     { eps(n) = Real(1); return; }
      if (fl != FsInterface) { if (fl == FsGas) eps(n) = Real(0); return; }
      const Real r = rho(n);
      const Real e = (r > Real(0)) ? mass(n) / r : Real(0);
      eps(n) = e;

      if (e > Real(1) + off) {
        if (nrf) {
          // Stay an interface cell, full. The surplus is published exactly as
          // it would have been on promotion, so settle() hands it on unchanged;
          // what differs is that the cell does NOT become Fluid, so no gas
          // neighbour is promoted behind it and the liquid cannot advance.
          // mass is reset HERE because the FsFluid branch of settle() -- which
          // normally does it -- is no longer reached.
          excess(n) = mass(n) - r;
          mass(n) = r;
          eps(n) = Real(1);
          return;
        }
        newf(n) = FsFluid;
        excess(n) = mass(n) - r;        // the part that will not fit
        return;
      }
      if (e < -off) {
        newf(n) = FsGas;
        excess(n) = mass(n);            // a debt, carried as a negative
        return;
      }
      // The neighbourhood rules. flags_ is stable through this pass -- nothing
      // here writes it -- so reading a neighbour's is safe.
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      bool has_gas = false, has_liquid = false, has_fluid = false;
      for (int i = 1; i < Q; ++i) {
        const std::uint8_t fj = flags(nb.j[i]);
        if (fj == FsGas) has_gas = true;
        else if (fj == FsFluid || fj == FsInterface) {
          has_liquid = true;
          if (fj == FsFluid) has_fluid = true;
        }
        // A cell resting against a wall has something holding it, so it is not
        // the free-falling speck the rule is about. Without this an obstacle
        // sweeping past leaves a trail of cells that the detached rule deletes.
        else if (fj == FsSolid) has_liquid = true;
      }
      if (!has_liquid && do_detach) {   // a speck in free fall
        newf(n) = FsGas;
        excess(n) = mass(n);
        return;
      }
      // exchange_geometric only. It fills a cell at its exact rate, which falls
      // to zero as the surface leaves the cell, so a cell can stop INSIDE the
      // hysteresis band -- just over 1, or just under 0 -- and be left behind
      // when the surface moves on, buried among Fluid cells (or stranded among
      // Gas) with a net flux of zero and no way to convert. Measured on
      // fs_band's 45-degree band swept across itself: interface cells at
      // 1.0007-1.0009 between Fluid ones, at -0.0009 between Gas ones, and the
      // surface 2.25 times as thick. Only a cell already PAST its bound is
      // converted, so the mass handed on is less than fill_offset; the general
      // buried-cell rule the banner above rejects declared 30 %-full cells full.
      // surface_follows_line puts the surface where the lines say it is, so a
      // cell an interface neighbour's line crosses is ON the surface whatever
      // its fill, and switching it would undo promote() the next step.
      bool on_line = false;
      if (follow)
        for (int i = 1; i < Q && !on_line; ++i) {
          const Index j = nb.j[i];
          if (flags(j) == FsInterface &&
              line_reaches(wf, j, -cvel<L>(i, 0), -cvel<L>(i, 1), FsInterface))
            on_line = true;
        }
      if (geo && !on_line) {
        if (e > Real(1) && !has_gas) {
          newf(n) = FsFluid;
          excess(n) = mass(n) - r;
        } else if (e < Real(0) && !has_fluid) {
          newf(n) = FsGas;
          excess(n) = mass(n);
        }
      }
    });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // Pass 4. Keep the interface closed, and settle the books.
  //
  // Three things at once, all as gathers:
  //   * a Gas cell beside a cell becoming Fluid must become Interface, or the
  //     new fluid cell would sit against a void with no boundary condition;
  //   * a Fluid cell beside a cell becoming Gas must become Interface, for the
  //     same reason from the other side;
  //   * the mass that would not fit in a converting cell is handed to the
  //     interface cells around it.
  //
  // A newly created Interface cell has no populations worth keeping, so it takes
  // the average density and velocity of its non-gas neighbours and is seeded at
  // equilibrium. That average is a gather, and it is why this can be one pass.
  //----------------------------------------------------------------------------
  void close_interface() {
    promote();
    if (no_refill) account_ejected();
    settle();
  }

  //----------------------------------------------------------------------------
  // The mass that no neighbour can accept, measured rather than dropped.
  //
  // settle() shares a donor's excess among the neighbours that end up
  // Interface and, when there are none, `if (takers > 0)` simply does not run --
  // so the mass vanishes silently. That is tolerable in a closed free-surface
  // flow, where it is a rounding-level leak; it is NOT tolerable under
  // no_refill, where it is the whole ejection signal. This reads the same
  // predicate settle() uses, from the same array, before settle() runs, and
  // adds up what is about to be lost.
  //
  // THE PREDICATE MUST MATCH settle()'s EXACTLY or the ledger is wrong in the
  // quiet direction -- it would under-count ejection and the budget would still
  // appear to close against a total that had already lost the mass.
  //----------------------------------------------------------------------------
  void account_ejected() {
    const Domain d = dom_;
    auto fin = final_; auto excess = excess_; auto flags = flags_;
    Real lost = 0;
    Kokkos::parallel_reduce("fs_ejected", Range(0, dom_.n_padded),
      KOKKOS_LAMBDA(Index n, Real& acc) {
        const std::uint8_t was = flags(n);
        if (was == FsSolid || was == FsExcluded) return;
        const Real ex = excess(n);
        if (ex == Real(0)) return;
        Neighbours<L> nb;
        d.template fill_neighbours<L, NF, NS>(n, nb);
        int takers = 0;
        for (int k = 1; k < Q; ++k)
          if (fin(nb.j[k]) == FsInterface) ++takers;
        if (takers == 0) acc += ex;
      }, lost);
    ejected_ += lost;
  }

  //----------------------------------------------------------------------------
  // Pass 4a. Decide every cell's FINAL state, reading only the intents that
  // pass 3 published. It writes a THIRD array rather than editing the one it
  // reads, and that is not tidiness.
  //
  // The first version of this did both in one kernel: it computed a cell's new
  // flag from its neighbours' newf() and wrote its own newf() at the end. Every
  // cell was therefore reading an array other cells were concurrently writing,
  // so what a cell saw of its neighbour depended on thread scheduling. The
  // damage was not a subtly wrong interface -- it was mass creation, because the
  // redistribution below counts how many neighbours will take a share and that
  // count disagreed with how many actually did. Measured: total mass reached
  // 1e67 in three hundred steps.
  //----------------------------------------------------------------------------
  void promote() {
    const Domain d = dom_;
    auto flags = flags_, newf = newf_, fin = final_;
    const bool follow = exchange_geometric && surface_follows_line;
    auto wf = wf_;
    Kokkos::parallel_for("fs_promote", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t was = flags(n);
      if (was == FsSolid || was == FsExcluded) { fin(n) = was; return; }
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);
      std::uint8_t now = newf(n);
      // A cell next to one that is about to be Fluid cannot be Gas: the new
      // fluid cell would face a void with no boundary condition to apply.
      if (now == FsGas) {
        for (int i = 1; i < Q; ++i)
          if (newf(nb.j[i]) == FsFluid) { now = FsInterface; break; }
      }
      // And symmetrically from the other side.
      if (now == FsFluid) {
        for (int i = 1; i < Q; ++i)
          if (newf(nb.j[i]) == FsGas) { now = FsInterface; break; }
      }
      // surface_follows_line: a Gas cell that an interface neighbour's line
      // enters, or a Fluid cell it leaves, holds part of the surface and becomes
      // Interface NOW rather than when that neighbour converts. Only cells that
      // were already Gas or Fluid: a cell converting this step has published its
      // excess, and keeping it Interface would count that twice.
      if (follow && ((was == FsGas && now == FsGas) || (was == FsFluid && now == FsFluid))) {
        for (int i = 1; i < Q; ++i) {
          const Index j = nb.j[i];
          if (flags(j) != FsInterface || newf(j) != FsInterface) continue;
          if (line_reaches(wf, j, -cvel<L>(i, 0), -cvel<L>(i, 1), was)) { now = FsInterface; break; }
        }
      }
      fin(n) = now;
    });
    Kokkos::fence();
  }

  //----------------------------------------------------------------------------
  // Pass 4b. Settle the books against the final states.
  //
  // The excess a converting cell cannot hold is shared equally among the
  // neighbours that end up Interface. Each taker recomputes that neighbour's
  // taker COUNT itself rather than having the donor publish it -- which keeps
  // this a pure gather -- and the predicate it counts with must be exactly the
  // predicate under which it takes, or the shares do not sum to the excess.
  // Both now read final_, which nothing in this pass writes.
  //----------------------------------------------------------------------------
  void settle() {
    const auto acc = pop_.template access<0>();
    const Domain d = dom_;
    auto flags = flags_, fin = final_;
    auto mass = mass_, eps = eps_, rho = rho_, excess = excess_;
    auto ux = ux_, uy = uy_, uz = uz_;
    auto reinit = reinit_;

    Kokkos::parallel_for("fs_settle", Range(0, dom_.n_padded), KOKKOS_LAMBDA(Index n) {
      const std::uint8_t was = flags(n);
      reinit(n) = 0;
      if (was == FsSolid || was == FsExcluded) return;
      const std::uint8_t now = fin(n);
      Neighbours<L> nb;
      d.template fill_neighbours<L, NF, NS>(n, nb);

      if (now == FsInterface) {
        Real got = Real(0);
        for (int i = 1; i < Q; ++i) {
          const Index j = nb.j[i];
          const Real ex = excess(j);
          if (ex == Real(0)) continue;
          Neighbours<L> nj;
          d.template fill_neighbours<L, NF, NS>(j, nj);
          int takers = 0;
          for (int k = 1; k < Q; ++k)
            if (fin(nj.j[k]) == FsInterface) ++takers;
          if (takers > 0) got += ex / Real(takers);
        }

        if (was == FsGas) {
          // Arriving from nothing: take the mean state of the neighbours that
          // have one, and start at equilibrium.
          Real sr = Real(0), su[3] = {Real(0), Real(0), Real(0)};
          int cnt = 0;
          for (int i = 1; i < Q; ++i) {
            const Index j = nb.j[i];
            const std::uint8_t fj = flags(j);
            if (fj != FsFluid && fj != FsInterface) continue;
            sr += rho(j); su[0] += ux(j); su[1] += uy(j); su[2] += uz(j);
            ++cnt;
          }
          const Real inv = cnt ? Real(1) / Real(cnt) : Real(0);
          const Real r = cnt ? sr * inv : Real(1);
          const Real u[3] = {su[0] * inv, su[1] * inv, su[2] * inv};
          for (int i = 0; i < Q; ++i)
            acc.dst(n, i) = Collision::seed_value(i, r, u[0], u[1], u[2]);
          rho(n) = r;  ux(n) = u[0]; uy(n) = u[1]; uz(n) = u[2];
          mass(n) = Real(0);
          reinit(n) = 1;      // its state came from nowhere; see reinitialised()
        } else if (was == FsFluid) {
          mass(n) = rho(n);           // still full; it just has a surface now
        }
        mass(n) += got;
        const Real r = rho(n);
        eps(n) = (r > Real(0)) ? mass(n) / r : Real(0);
      } else if (now == FsFluid) {
        mass(n) = rho(n);  eps(n) = Real(1);
      } else {  // FsGas
        mass(n) = Real(0); eps(n) = Real(0);
      }
    });
    Kokkos::fence();
    Kokkos::deep_copy(flags_, final_);
    Kokkos::deep_copy(newf_, final_);
  }

 private:
  Domain dom_;
  Streaming pop_;
  View1D<std::uint8_t> flags_, newf_, final_, reinit_;
  // WHICH CELLS THIS MOVE UNCOVERED, kept apart from reinit_ on purpose.
  // reinit_ is a PUBLIC signal that also carries settle()'s gas-arrivals from
  // the previous step, and transfer_covered_mass() reading it counted those as
  // recipients of the body's displaced mass -- diluting the share the genuinely
  // uncovered cells got, and OVERWRITING the mass the gas-arrivals had just
  // been given. This mask means only what its name says.
  View1D<std::uint8_t> uncov_;
  Real ejected_ = 0;          // see ejected_mass()
  Real drained_ = 0;          // see drain()
  HostView1D<std::uint8_t> h_flags_;
  View1D<Real> mass_, eps_, excess_;
  // Curvature working set. Empty until set_surface_tension() is called.
  View1D<Real> cf_, cfs_, knx_, kny_, knz_, kap_;
  View1D<Real> wf_;                   // exchange_geometric: wet fraction of each face, 4 per cell
  Index hf_used_ = 0, hf_fell_ = 0;   // the last height-function pass's coverage
  // Wall cells the contact-angle law rewrote this step. Empty until
  // set_contact_angle() is called.
  View1D<std::uint8_t> wet_;
  View1D<Real> wux_, wuy_, wuz_;      // wall velocity, meaningful in body cells
  // WHICH SOLID CELLS ARE THE BODY'S. A tank wall and a moving obstacle are both
  // FsSolid and must not be confused: the wall does not move, does not get
  // refilled when the body passes, and above all does not contribute to the
  // body's force. Conflating them was measured -- the tank floor and lid took
  // the body's velocity, pumped the fluid, converted themselves to interface
  // cells on the first call, and added their own links to the momentum exchange.
  View1D<std::uint8_t> obst_, was_body_;
  Real bcx_ = 0, bcy_ = 0;            // body centre, for the torque
  Real fx_ = 0, fy_ = 0, tz_ = 0;     // momentum-exchange force and torque
  View1D<Real> rho_, ux_, uy_, uz_;
  std::size_t t_ = 0;
};

}  // namespace lbm
