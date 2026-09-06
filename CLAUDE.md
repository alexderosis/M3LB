# M3LB — working notes for Claude

Lattice Boltzmann solver, C++20 on Kokkos. Read this before proposing changes or
writing a case; most of it is knowledge that is otherwise spread across
`doc/m3lb.pdf` (107 pages) and the banner comments at the top of each header.

**The banners are the documentation.** Every non-obvious decision in this tree is
argued at the top of the file that implements it, usually with the measurement
that settled it. When you are about to change something, read that file's banner
first — it frequently already says why the obvious change is wrong.

---

## Two independent codebases

| | what it is | when to use |
|---|---|---|
| `src/` + `validation/` | the main solver, Kokkos, five lattices, four collision operators, thermal + MHD + multiphase + free surface | default; anything on CPU; anything needing the full physics |
| `GPU/` | a second implementation written directly in CUDA, sharing **no headers** with the first | GPU runs, or cross-checking one implementation against the other |

They deliberately duplicate the physics. That is the point: they agree where they
overlap, and disagreement is a bug in one of them. Do not "de-duplicate" them.

`GPU/` compiles as plain C++ too (`-DLBM_HOST_ONLY=ON`), so every CUDA driver can
be built and run on a laptop before it touches a device. Do that first — it is
how wrong initial conditions and wrong diagnostics get found cheaply.

---

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
cmake --build build -j4
cd build && ctest --output-on-failure
```

**Pass `-DKokkos_ENABLE_THREADS=ON`.** Without it CMake prints *"no host-parallel
backend, tests run serial"* and everything runs single-threaded — several times
slower, and easy not to notice. Tests then get `--kokkos-num-threads=4`
automatically (`LBM_TEST_THREADS` in `CMakeLists.txt`); pass it by hand when
running an executable directly:

```bash
./build/validation/poiseuille --kokkos-num-threads=4
```

FP32 build: `-DLBM_PRECISION=float`. It cannot resolve the finest convergence
tests — that is documented, not a bug.

GPU (`GPU/` is its own CMake project, nvcc only, no Kokkos):

```bash
cd GPU && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=75
cmake --build build -j4          # 75 = T4/Turing, 80 = A100, 90 = Hopper
```

---

## Choosing a solver

| physics | use | notes |
|---|---|---|
| single-phase, single-component | `FluidSolver` | **the default — see the rule below** |
| + temperature / passive scalar | `ScalarSolver` alongside | own lattice, velocity is an input; `ScalarBGK` by default, `ScalarRegularised` above ω ≈ 1.9 |
| free-slip / symmetry wall | `set_specular_walls` | exact mirror; **halfway** (ghost cell) |
| zero-flux scalar wall | `ScalarSpecular` + `ScalarSolver::set_specular_walls` | **on-node**; the only one usable where the field is differentiated or integrated — see below |
| + charge carriers in an electric field | `ScalarSolver` + `ChargeCentralMoments` | D3Q27, advects at the **drift** velocity `u + KE`, not at `u` |
| + electric potential (Poisson) | `ScalarSolver` + `ScalarBGK` + `add_source` | no new solver — see `validation/ehd_hydrostatic.cpp` |
| + magnetic field | `MagneticSolver` | Dellar vector distribution |
| two-phase, diffuse interface | `PhaseFieldSolver` | conservative Allen–Cahn, prescribed interface width, density ratio ~100 |
| two-phase, diffuse, high ratio | `ColourGradientSolver` | no interface equation; width is an *outcome*; 1000 in the source paper's own static tests |
| liquid + void, sharp interface | `FreeSurfaceSolver` | gas not resolved; infinite ratio by construction |

**Standing rule: do not use a multiphase solver for single-phase or
single-component flow.** The multiphase path changes the equation of state — the
populations carry a normalised pressure rather than a density — and that
propagates through the macroscopic definitions, forcing, boundaries and the
initial condition. Use `FluidSolver` unless the problem genuinely has two phases
or two components.

The two immiscible models do not dominate each other. On a matched static droplet
the colour gradient is more accurate on Laplace's law at every density ratio
measured; the phase field carries a spurious current 3× to 118× smaller. That is
one static case, not a flow.

---

## From a physical problem to lattice units

Most cases here are dimensionless and need no conversion at all: they are
specified by Re, Ra, Pr or Ha and a resolution, and the answer is a
dimensionless number checked against a table. Conversion only bites when the
geometry carries a real length — `urban`, `height_field`, `aorta`.

**What the code converts, and what it does not.** Every operator turns a
transport coefficient into a relaxation rate, and most turn it back:

| operator | forward | inverse |
|---|---|---|
| `BGK`, `TRT`, `MomentCollision`, `MultiphaseBGK`, `MhdBGK`, `MhdCentralMoments` | `omega_from_viscosity(nu)` | `viscosity_from_omega(w)` |
| `ScalarBGK`, `ScalarRegularised` | `omega_from_diffusivity(d)` | `diffusivity_from_omega(w)` |
| `MagneticBGK` | `omega_from_resistivity(eta)` | — |
| `PhaseFieldBGK` | `omega_from_mobility(m)` | — |
| `ColourGradient` | — | `viscosity_from_tau(tau)` |

They take the coefficient **already in lattice units**, and each reads its own
lattice's `cs2`, so they are the safe way to get ω. `ScalarBGK` also accepts a
*field* of rates, `omega_of`, for conjugate heat transfer — a solid inclusion in
a fluid. It reproduces a conductivity ratio only where ρc_p is uniform, since the
scheme transports the temperature rather than the enthalpy; where it applies, the
interface is exact rather than second order (`validation/zhou_thermal.cpp`
measures 1e-9 % at ratios from 0.1 to 100). Nothing in `src/` converts
metres and seconds — that arrow belongs to the case. The only worked example in
the tree is the local `struct Scaling` at `demonstrator/urban.cpp:123`, and it
is local on purpose: its banner argues one particular strategy (fix `dt` by
capping the fastest cell, let ω follow), which is right for an
advection-dominated problem and not in general.

### The dimensionless recipe

Three choices, and the fourth is not yours. Fixing the resolution, the Reynolds
number and the lattice velocity fixes the viscosity — you do not also get to
pick τ.

1. **Resolve the characteristic length**: `N` cells across `L`. Wall-bounded
   cases size the domain `ny = H + 2`, and the Reynolds length is `H`, not `ny`.
2. **Choose `u0` in lattice units.** It is a Mach number in disguise:
   `Ma = u0 / cs` with `cs = 1/sqrt(3) = 0.577`, and the compressibility error
   grows as `Ma^2`. Keep `u0 <= 0.05` (`Ma <= 0.087`) unless you have measured
   what a larger one costs.
3. **Viscosity follows**: `nu = u0 * N / Re`.
4. **Then** `coll.omega = Coll::omega_from_viscosity(nu)`.
5. **Read τ = 1/ω back before running.** τ → 1/2 is the stability floor. Since
   `nu = u0 * N / Re`, a larger Re is bought either with a bigger grid or a
   smaller `u0` — and a smaller `u0` costs steps (below).
6. **Pick the timescale deliberately.** One convective time is `N / u0` steps;
   the momentum-diffusion time is `N^2 / nu`. They differ by a factor of Re, and
   quoting a result after the wrong one is how a case looks converged when it is
   not: `examples/flow_past_square.cpp` records `Re_eff` falling from 49 to 26
   over 30,000 steps against a diffusive time of 576,000.

`examples/flow_past_square.cpp:92-118` is this recipe as running code, with the
τ floor check and the Mach print.

### When the geometry is physical

Set `dx = L_phys / N` metres per cell. One further choice fixes `dt`, and the
two strategies are not equivalent:

- **Cap the velocity.** Pick `u_lat_max`, then `dt = u_lat_max * dx / u_phys_max`.
  Bounds the Courant number; use it when advection dominates
  (`demonstrator/urban.cpp:256`).
- **Fix τ.** Pick τ, then `dt = (tau - 0.5) * cs2 * dx * dx / nu_phys`. Bounds
  the diffusive accuracy; use it when diffusion dominates
  (`demonstrator/urban.cpp:253`).

With `dx` and `dt` chosen, everything else follows:

```
u_lat  = u_phys  * dt / dx
nu_lat = nu_phys * dt / (dx * dx)          # and D_lat, eta_lat, the same way
a_lat  = a_phys  * dt * dt / dx            # accelerations, e.g. gravity
t_phys = steps * dt
```

### Traps

- **`cs2` is not 1/3 on every lattice.** D3Q7 is **1/4**; D2Q5 and the rest are
  1/3 (`src/lattice/Lattices.hpp`). So `tau = 3*nu + 1/2` is simply wrong on
  D3Q7. Use `omega_from_*`, which reads the lattice's own `cs2`; keep the closed
  form only for a printed diagnostic on a `cs2 = 1/3` lattice. The literal
  `0.25` in `demonstrator/urban.cpp:253` is this, in the open: it is correct
  only because that case is D3Q7, and copying the line to a D3Q19 or D3Q27
  scalar would be wrong by 4/3 without failing.
- **Halving `u0` at fixed Re and N hurts twice**: it doubles the steps to the
  same convective time *and* halves ν, pushing τ toward 1/2. Raising `N` instead
  raises ν and moves τ away from the floor; resolution is the expensive knob,
  not the dangerous one.
- Two entries under **Measurement discipline** below are units errors wearing
  another hat, and belong here too: match *kinematic* viscosity across a density
  ratio, not dynamic; and a wrong constant still gives a consistent simulation,
  so check the inputs against the source, not only against themselves.

---

## Invariants that break silently

These produce plausible, converged, wrong answers rather than crashes.

- **Direction ordering is a contract.** `opp(i)` is `i+1` for odd `i` and `i-1`
  for even (`src/lattice/Lattices.hpp`). Esoteric Pull depends on it. If you add
  or reorder a lattice's velocity set, opposite pairs must stay adjacent.
- **Storage × streaming pairings are not free.** Shifted populations centre the
  stored variable on `p̃ = 1`, which is exactly the pressure gauge that must be
  avoided at a density ratio — so the multiphase operators declare
  `RawPopulations`. `FreeSurfaceSolver` `static_assert`s *against* Esoteric Pull
  and needs `TwoLattice`, because it reads a neighbour's post-collision state
  while writing its own.
- **D3Q19 is not a product lattice.** It is D3Q27 minus its corners, so it uses a
  generated monomial basis. Neither `MultiphaseCentralMoments` nor
  `PhaseFieldCentralMoments` runs on it at all (both `static_assert` on
  `ProductBasis::enabled`, so it is a compile error rather than a wrong answer),
  and its MHD Maxwell sum leaves a ghost-mode residual. Prefer D3Q27 for
  anything above second order.
- **A published moment list belongs to a basis.** `ProductBasis` is *shifted*,
  phi_2 = C^2 - cs2; most papers tabulate *monomial* central moments, and the
  same physics occupies different slots in the two. De Rosis & Enan's Eq. (61)
  lists nine nonzero phase-field source entries; in the shifted basis six of
  them are identically zero, because the (a,a,b) slot gets
  `cs4 A_b - cs2*cs2 A_b = 0`. Transcribing such a list slot-for-slot
  double-counts; deleting terms from a monomial implementation loses them.
  Neither crashes. `tests/test_phase_field.cpp` block 8 pins both directions.
- **Wall conventions.** Wall-bounded cases size the domain as `ny = H + 2`: `H`
  fluid nodes plus one solid row each side. Getting this wrong shifts the
  Reynolds or Rayleigh number silently.
- **Coupling order belongs to the solver, not the driver.** For the phase field
  it lives in `step()`; refreshing φ late misplaces the interface rather than
  merely damping it.
- **`temperature()` is ZERO at an adiabatic scalar node**, because bounce-back
  puts the insulated plane at 0.5 and the node is a ghost outside the fluid
  (`ScalarSolver.hpp`'s `field_kernel`). Harmless when that node is `Solid` for
  the fluid — it never collides. *Not* harmless when it is a `RegWall`, which is
  a fluid node that does collide and does get forced: `BoussinesqGuo` then reads
  T = 0 against your `T0` and applies a body force along the whole wall. This is
  the concrete cost of mixing the two wall families, and it cost a benchmark run
  in `validation/zhou_thermal.cpp` (case 3.5, whose moving lid forces the fluid
  side to be on-node). Two defences: keep the temperature gauge symmetric about
  zero so that `field = 0` means *neutrally buoyant*, and use `ScalarOutflow` —
  which is on-node, zero-gradient, and reports the real temperature — where an
  on-node adiabatic wall is what you actually need.
- **THE HALF CELL IS HARMLESS UNTIL YOU DIFFERENTIATE IT.** The tree's default
  wall family puts the plate at the halfway plane, and `rb_high_ra`'s banner
  measures that as costing "1% in H, 3% in Ra" — negligible. It is **not**
  negligible when the quantity of interest is a *derivative* of the field
  carrying the boundary value. In `validation/ehd_hydrostatic.cpp` the electric
  field is `E = −∇φ`, and with halfway plates the one-sided stencil works purely
  from interior nodes and never sees the imposed φ: measured as a uniform +3.3%
  bias in the bulk charge at H = 40, halving to +1.3% at H = 80, which is
  O(1/H) and dragged the whole convergence rate to 1.0. On-node plates
  (`ScalarMoment`, Dellar's condition) fixed it — C = 10 at H = 80 went from
  11.19% to 1.49%. Ask which family you need before defaulting to halfway.
- **A ZERO-FLUX SCALAR WALL RINGS AT ω → 2, AND THE COLLISION CANNOT SAVE IT.**
  Measured 2026-09-05 in `validation/ehd_electroconvection.cpp -freeslip`.
  `ChargeCentralMoments` annihilates the ghost moments in the *bulk* — that is
  its whole relaxation schedule — but `ScalarAdiabatic` is bounce-back applied
  *outside* the collision, and at ω_q = 1.99952 bounce-back re-injects an
  odd–even mode that never damps. The cell-to-cell amplitude reaches **0.4808**
  in the first columns against 0.0167 for a boundary-free box: 29× larger, and
  10× its own mid-box level. 85.7% of the error sits within three cells of the
  wall, in columns that are 24% of the domain, so it falls at roughly *first*
  order — 3.96% at ny=41, 3.30% at 81, 1.63% at 163. `ScalarOutflow` is not the
  fix: it is an *open* boundary, and it bled charge to −0.165 q0 against −0.021.
  **THE GAP IS NOW CLOSED — by `ScalarSpecular`, and the sharper measurement is
  `validation/ehd_cavity.cpp`,** whose closed square has no way to avoid a
  lateral scalar wall the way the doubled box avoids one. At N=81, T=1000, with
  the *fluid frozen* so nothing but the charge/potential pair is in play:
  `ScalarAdiabatic` went **NON-FINITE inside 0.05 t0**; `ScalarOutflow` drove q
  to −0.279 q0 and left the hydrostatic reference **158% wrong**, and it got
  *worse* under refinement (−0.083 q0 at N=41); `ScalarSpecular` reproduced the
  boundary-free box's interior to the printed digits. The mechanism is the
  distinction between the two reflections: bounce-back reverses **every**
  component, which is what injects the odd–even mode, while a mirror reverses
  only the **normal** one. `ScalarSpecular` also acts on the *unknown*
  directions only and then collides, so the node is a real node — it reports
  the real value, where `ScalarAdiabatic` reports a structural **zero**. That
  matters whenever the field is differentiated or integrated across the wall
  column, which is exactly what `E = −∇φ` and Eq. (81)'s volume integral do.
- **A BOUNDARY FLAG THAT DOES NOT TAKE THE SOURCE LEAVES ITS PDE UNSOLVED, AND
  NOTHING CRASHES.** `ScalarSolver::source_kernel` skipped every cell that was
  not `ScalarBulk`, which is right for Dirichlet, Moment, Outflow and Adiabatic
  — all of them prescribed downstream, so the source would be thrown away — and
  wrong for `ScalarSpecular`, which is a *bulk node carrying a mirror closure*.
  The Poisson source `β q/ε` was therefore missing in the wall columns, so those
  columns solved ∇²φ = 0 instead. Measured: I0 came out **+8.87%** at N=41
  against +0.51% for the boundary-free box, and the convergence dropped from
  second order to first. It looked exactly like a mediocre boundary condition
  rather than like a bug. When adding a scalar cell type, ask whether it is
  *prescribed* or *closed* — the source list is the thing that will not tell
  you.
- **REMOVING A BOUNDARY CAN BEAT DISCRETISING IT.** Same measurement, and it
  inverted the reasoning it was meant to check. A free-slip box of width Lx is
  the mirror-symmetric half of a *periodic* box of width 2Lx, so the doubled box
  solves the same problem with **no lateral boundary at all**. It costs 2× the
  cells and is the better discretisation, not a workaround for a missing
  feature — the real free-slip wall, which is exact for the fluid, is worse here
  because its scalar companion is not. Keep both anyway: the gap between two
  lateral discretisations measures the lateral boundary error the way the
  D2Q9/D3Q27 gap measures the interior, and neither alone can show it.
- **A SEED CHOOSES A BRANCH, NOT JUST A TRANSIENT.** Measured 2026-09-05 in
  `validation/ehd_cavity.cpp`. The tree's usual defence of a seed is that it
  sets the transient and not the answer, checked by halving the amplitude and
  watching the converged value not move. In a SUBCRITICAL bifurcation that
  check passes and still misleads: at T = 250 the saturated peak velocity is
  `u_max/u0 = 2.00` at `-amp 1e-2` and `1.99` at `-amp 1e-4` — two decades of
  seed, the same answer, exactly the stability the check looks for — while
  `-amp 0` gives `0.08`, i.e. no convection at all. Ne reads 1.39, 1.24 and
  **1.0003** against the reference's 1.03. The seed was not setting the
  amplitude; it was deciding which BRANCH the flow landed on, and the reference
  starts from exactly zero (its Eqs. 17-20) so its only seed is round-off. Two
  rules follow. Seed-independence of the amplitude does not establish
  seed-independence of the state. And where a case is known to be subcritical —
  this one's own Sec. 3.2.1 plots a hysteresis loop — `-amp 0` is the protocol
  that reproduces the reference, not a degenerate case to be avoided.
- **A COARSE LADDER CANNOT BE EXTRAPOLATED INTO A REGIME IT NEVER REACHED, AND
  THIS TREE DID IT ANYWAY.** `ehd_cavity` at T = 5000 gave Ne = 2.8558, 2.8017,
  2.4903 at N = 81, 129, 201 against a digitised 2.80. The N = 129 run was
  within **0.1 %** of the reference and that meant nothing — the sequence was
  falling and that grid was on its way past. So far so good; the conclusion
  drawn from it was that refinement moves AWAY from the reference and the
  *converged* deficit is about 11 % low. **That was wrong.** All three points
  are under-resolved — Re_cell 8.5, 6.1, 3.6 — and when `GPU/` finally ran the
  reference's own 500² grid on a T4 (Re_cell = **1.5**, where the paper wants
  about 1.6) it gave **3.10**, i.e. 10.8 % HIGH. The sequence is not monotonic
  through to the resolved grid, and three under-resolved points do not
  extrapolate into a regime none of them is in.
  What is claimable at the resolved grid is `Ne = 3.10 ± 0.41` with the paper's
  2.80 inside it — consistent with, not converged to, because the r.m.s. is
  13 % of the mean over only three t0. The lattice family is not the variable:
  D3Q27/D3Q7 at N = 201 gives 2.4735 against D2Q9/D2Q5's 2.4903.
  Two rules. When the runs that agree with a reference are the only ones
  breaking your own Mach and cell-Reynolds rules, distrust the agreement — that
  part was right. And do not name a number "converged" until a run has actually
  reached the resolution its own diagnostic says it needs; say "under-resolved,
  trend unknown" instead, which is what the ladder really showed.
- **GRID INDEPENDENCE WITHIN ONE FAMILY IS NOT GRID INDEPENDENCE.** The doubled
  box's mirror planes sit on nodes at every resolution, so refining it cannot
  see an error that depends on the half-cell alignment. `ehd_electroconvection`
  reported 0.33% from ny=81 to 163 and was 3.3% from the other alignment. Ask
  what a refinement holds *fixed* before quoting it as convergence.
- **`gather` AND `scatter` ARE A STREAMING PAIR, NOT A READ-MODIFY-WRITE.** In
  `GPU/`, `scatter` writes `in[i]` into the slot `gather` took `out[i+1]` from —
  that crossing IS the stream. So a pass that gathers, modifies and scatters
  straight back does not update a node in place; it advances the field by a
  step. Found 2026-09-06 while porting the EHD stack: the Poisson source did
  exactly that, and the potential came out with a first-cell gradient 40 % short
  of the interior one, which read as a bad boundary condition rather than as a
  bug. The fix is to swap each opposite pair before storing — the same identity
  that lets an adiabatic cell be skipped entirely. `src/`'s accessor does not
  have this hazard: `acc.load(nb,i)` and `acc.scatter(nb,i,·)` address the same
  slot.
  **TWO OBVIOUS TESTS OF IT ARE BLIND, and both were written and passed before
  the third caught it.** A constant source on a flat field: a uniform field is
  invariant under streaming. A one-shot before/after comparison: the crossing
  merely SWAPS each opposite pair, and the field is their sum, so nothing moves
  until a step flips the parity. Only a DIFFERENTIAL run over several steps sees
  it — the same case with and without a 1e-30 source, which gives 0.224 with the
  bug and exactly 0 without (`GPU/test/host_physics.cpp`). When a test for an
  in-place update passes, check that it would fail if the update streamed.
- **"OUTSIDE THE FIELD" HAS TWO SPELLINGS, AND A GEOMETRY THAT USES THE OTHER
  ONE FALLS THROUGH.** `GPU/`'s outflow donor rule is "one step inward along
  every outward axis at once", so a face node takes its axis neighbour and a
  corner the diagonal. Outward was detected only by a neighbour marked
  `ScalarExcluded` — right for a channel with bounce-back walls, which is what
  it was written against, and blind to a box with ON-NODE walls, which excludes
  nothing at all. The collector corners of `ehd_cavity` then found no outward
  axis, fell through to the axial fallback, and came out **inert**: their four
  axial neighbours are two more outflow nodes, a specular column, and — through
  the periodic wrap — the injector. Two nodes out of 441, and they cost 0.6 % on
  the volume-averaged charge flux: I0 went from 1.325615e-05 to 1.333898e-05
  against the Kokkos twin's 1.334004e-05, i.e. the gap between the two
  codebases fell from 0.63 % to **0.008 %**. The fix is the same one the
  unknown-mask needed — outward also means "off a non-periodic edge" — and an
  all-periodic box is unchanged by it, which `host_physics.cpp` asserts
  alongside the corners themselves.
- **AN OPT-IN BUFFER PLUS A DEFENSIVE NULL-CHECK IS A SILENT WRONG ANSWER.**
  `GPU/`'s fluid does not allocate the velocity field coupled solvers advect
  with unless `enable_velocity_output()` is called — deliberately, since it is
  11 % more traffic an uncoupled run should not pay — and `ux_device()` returns
  **null** until then. `ehd_cavity.cu` read it through a
  `p.ux ? p.ux[n] : Real(0)`, which was written so that the hydrostatic
  reference could pass a null on purpose. The two met: the Coulomb force still
  drove the fluid, the fluid still moved, and the charge was simply never
  advected by it, so the electroconvective feedback loop was open. It did not
  look broken — the seed still grew, by drift alone, and saturated at
  `u_max/u0 = 0.578` against the Kokkos twin's `11.583` at identical
  parameters. FP64 gave the same 0.578, so it was not precision.
  What found it was deleting the coupling on purpose: setting `ep.ux = nullptr`
  unconditionally changed the answer by NOTHING, which proves the term was
  never contributing. **When a coupling term is suspected, delete it and see
  whether the answer moves** — that is one run and it is unambiguous, where
  reading the code had already failed twice. And a sentinel that legitimately
  means one thing ("no fluid, by design") must not be reachable by an accident
  that means another ("you forgot to allocate it"); the driver now aborts if a
  non-hydrostatic run sees a null velocity.
- **A NODE WHOSE POPULATIONS ARE PRESCRIBED MUST NOT HAVE ITS FIELD RECOMPUTED.**
  `GPU/`'s scalar field kernel summed an outflow node's populations like any
  other node's, arguing that an outflow cell holds a real concentration. It
  does — but the pass that set it wrote POST-COLLISION populations and the field
  kernel runs at the NEXT parity, so what it sums is the post-STREAMING state:
  everything that arrived, including whatever the periodic wrap delivered from
  the opposite face. Harmless in an open channel, where the far face is far
  away; wrong in `GPU/src/ehd_cavity.cu`'s closed box, where the collector read
  **0.33 q0** against a neighbour at 0.075 because the wrap handed it the
  injector. `src/` skips outflow nodes in its field kernel for this reason and
  says so; `GPU/` now does too.
- **A MOMENT INDEX MUST BE A COMPILE-TIME CONSTANT.** The moment operators reach
  their exponents through `Basis::p_of(n)`, which is a lookup in a 432-byte
  table. Called with a compile-time `n` it folds and the moment arrays live in
  registers; called from a *runtime loop* it cannot fold, so the table is
  materialised in memory, the `p` it returns then indexes `Qf`/`Aw`, and the
  27-moment array follows them out of the register file. Nothing fails: the
  answer is bit-identical and every test passes. On a CPU it costs almost
  nothing (a 464-byte frame is L1-resident; `cmbench` reads 2.18× BGK either
  way), which is why it survived from the first commit of `MomentCollision.hpp`.
  In DEVICE code that frame is per-thread *local* memory — off-chip DRAM, every
  subscript uncoalesced — and this tree has measured that mechanism at **47×**
  in `GPU/`'s colour gradient. So it is the leading candidate for the Kokkos
  central-moment collapse, and it is invisible to every instrument in the tree
  except the compiler's own output. `tests/frame_check.sh` is that instrument:
  run it after touching a moment operator and look at the `loops` and `regidx`
  columns, not the wall clock. Fixed in `MomentCollision` and
  `MultiphaseCentralMoments` (2026-09-04); still present in `ColourGradient`,
  which has the largest frame in the tree and needs the closed-form derivation
  rather than loop unrolling. Note that `static_assert` guards the `constexpr`
  half of this but cannot guard the loop half — only the script can.

---

## Adding a case

Two steps.

1. Write `validation/<name>.cpp` (a regression test with an exact answer) or
   `demonstrator/<name>.cpp` (something to look at). Copy the shape from a small
   existing one — `validation/tgv2d.cpp` is a compact fluid-only example,
   `validation/natural_convection.cpp` a coupled one at 56 lines.
2. Register it. `validation/CMakeLists.txt` has **two** `foreach` lists: the
   first is built *and run by `ctest`*, the second is analysis runs too slow to
   be tests. Put a regression test in the first, a sweep in the second.
   `demonstrator/CMakeLists.txt` has one list.

The skeleton — `validation/claudemd_skeleton_check.cpp` is this snippet as a
buildable file, so it stays true:

```cpp
#include "collision/BGK.hpp"       // BEFORE FluidSolver.hpp: it defines Macro,
#include "solver/FluidSolver.hpp"  // which FluidSolver.hpp uses but does not include
#include "memory/EsotericPull.hpp"

Kokkos::initialize(argc, argv);
{
  Domain d(nx, ny, nz, /*periodic x,y,z=*/true, true, true);
  Coll coll;  coll.omega = Coll::omega_from_viscosity(nu);
  FluidSolver<Lattice, EsotericPull<Lattice>, Coll> s(d, coll);
  s.initialize_field(KOKKOS_LAMBDA(Index n) { return FlowState{rho, ux, uy, uz}; });
  for (std::size_t t = 0; t < T; ++t) s.step();
}
Kokkos::finalize();
```

Headers are **not** self-contained: `FluidSolver.hpp` refers to `Macro`, which
`collision/BGK.hpp` defines, so alphabetical include order fails to compile. The
existing cases dodge this by including `validation/Campaign.hpp`, which pulls
things in the right order — use that in a validation case and you will not meet
the problem.

A validation case should compare against something with a known answer —
analytic, a published table, or a convergence rate — and print a PASS/FAIL. Cases
that merely run are demonstrators.

---

## Out of scope, and known unreliable

Do not spend time on these without saying so first; several are deliberate.

- **No MPI.** Single rank. `Domain` carries halo machinery but there is no
  exchange.
- **No contact line or wetting model** in the phase field; no open boundary for φ.
- **The two specular walls sit on DIFFERENT PLANES, and that is the remaining
  gap.** The fluid's (`set_specular_walls`, `CellType::SpecWall`,
  `src/boundary/Specular.hpp`) is a **ghost cell**, so its mirror plane is half
  a cell outside the last fluid node — exact to 5.5e-14 against the channel it
  mirrors (`validation/specular.cpp`, plane measured at 16.5000). The scalar's
  (`ScalarSpecular`, `mirror_unknowns`) is **on-node**, because the scalar
  Dirichlet family it must pair with (`ScalarMoment`) is on-node for the reason
  the half-cell entry above gives. Both are exact — `validation/specular.cpp`
  and `validation/scalar_specular.cpp` each assert an identity against the box
  they mirror, at 5.5e-14 and 5.3e-15 — but they are exact about planes half a
  cell apart. So they PAIR in `validation/ehd_cavity.cpp`, where every wall is
  on-node (`RegWall` + `ScalarMoment` + `ScalarSpecular`), and they do NOT pair
  in `ehd_electroconvection -freeslip`, whose fluid mirror is halfway. Closing
  that needs an on-node specular *fluid* wall; until then, do not read the
  free-slip electroconvection gap as a measurement of the scalar wall alone.
  Axis-aligned normals only, and no specular corners, in either.
- **The free surface has no surface tension** (uniform gas pressure, no curvature
  term) and **no gas dynamics** — an enclosed bubble does not compress.
- **The free surface's moving obstacle is not reliable.** The cause is in
  `transfer_covered_mass()` and `settle()`, is written up in the module banner
  with measurements, and is not a caller error. Do not present a run with a
  moving obstacle as a result.
- **`GPU/` HAS THE EHD STACK as of 2026-09-06** (`src/ehd_cavity.cu`,
  `include/lbm/ehd.cuh`, `include/lbm/specular.cuh`). Porting it added four
  things to the scalar module that `GPU/` did not have: Dellar's on-node
  Dirichlet (`ScalarMoment`), the on-node zero-flux wall (`ScalarSpecular`), a
  per-node source, and the central-moment charge collision. The solvers are now
  templated on their lattice, so the charge is `ScalarSolverT<D3Q27>` and not a
  second implementation; `ScalarSolver` and `ChargeSolver` are aliases.
  **The charge collision is a CLOSED FORM, not the parent's moment transform,**
  and that is deliberate: a 27-element moment array indexed at run time is the
  local-memory trap named above. On a product lattice the target is separable
  into two 1-D factors per axis, `E(+-1) = (cs2 + v^2 +- v)/2` and
  `G(+-1) = v +- 1/2`, and the product is EXACT rather than an approximation —
  it reproduces `k110 = 0`, which a naive product form does not.
  `test/host_check.cpp` pins twelve moment identities against it.
  **Cross-checked against the Kokkos twin at matched lattices** (D3Q27 charge,
  D3Q7 potential, N = 21, T = 150): I0 = 1.3256e-05 against 1.3340e-05, 0.63 %
  apart with no shared headers, the collector charge on 0.0722 q0 in both, and
  Ne = 1.0001 below onset. That comparison is what found the two bugs recorded
  above, and neither was in the new code.
- **`GPU/` is D3Q27 only**, and that is now the main thing it does not share
  with `src/`. As of 2026-09-02 it also has TRT, shifted storage, central
  moments for *both* multiphase distributions (its phase field runs on D3Q7 or
  D3Q27, and the central-moment operator needs the latter), the open scalar
  boundary, Dellar's moment-based magnetic wall, the penalised rigid body with
  both shapes, and the free surface.
  regularised (on-node) velocity walls, and `hartmann` — so a wall-bounded MHD
  benchmark now runs there too.
  What it still lacks: an open boundary for the FLUID (the parent's `NrmOutXp` /
  `NrmOutFree`), D3Q19, raw MRT, and a moving obstacle in the free surface — the
  last deliberately, see that module's banner and the entry above.
  As of 2026-09-04 the D3Q7 scalar also has a **regularised collision**
  (`collide_scalar_regularised`, `ScalarOp::Regularised`) beside BGK: it relaxes
  the flux moments at ω and annihilates the three ghost moments rather than
  relaxing them. Identical to BGK at ω = 1, and it matters only near ω = 2 —
  where BGK becomes a reflection, so the ghosts invert every step and never
  damp. At Ra = 1e14 (ω_T = 1.99999905) BGK's near-wall temperature *rings*:
  Nu_bot swung 34.9 ↔ 93.6 over sixty free-fall times against an analytic ~100,
  bounded rather than diverging, so it averages into a plausible number.
  `rb_high_ra` defaults to the regularised operator and keeps `-sop bgk` to
  reproduce the ringing on demand.
  **The cold-start undershoot belongs to the SCALAR OPERATOR, not to the initial
  condition** (measured 2026-09-04, and it corrected the opposite assumption).
  A cold start puts the whole ΔT across one half cell and a D3Q7 scalar near
  ω = 2 undershoots rather than smoothing it, but by how much depends on which
  operator: worst `T_min` against a halt threshold of −0.5 is **−0.813**
  (`ScalarBGK`, Kokkos twin, Ra = 1e14, 100 t_ff) but only **−0.112** with the
  regularised operator at the same point, and **−0.160 recovering to
  [0.005, 0.919]** at ω_g = 1.997 while convecting. So `-ic cold` runs on
  `GPU/rb_high_ra` as it stands; it is the Kokkos twin's `ScalarBGK` that cannot
  take a step IC. `-grace G` suppresses the maximum-principle *halt* (never the
  report) for G free-fall times, defaulting to 4× the diffusive smoothing time
  `16/(α t_ff)` for `cold` and 0 for `cond`, and **refusing** — grace 0, with the
  arithmetic printed — where that would exceed a quarter of the run, which is
  exactly the case where the undershoot is permanent rather than transient
  (H = 50, Ra = 1e14: the step needs 5.4e4 t_ff to smooth over four cells
  against a 1e4 t_ff run). Every run now ends with the worst excursion, when it
  happened, and whether it recovered. `tests/frame_check.sh`'s sibling
  discipline applies: read the verdict line, not the fact that it finished.
  Also `-aspect` is now a **double** — it was an `int`, which silently truncated
  both the reference's 2.02 and the single-critical-cell 2.0158 to 2.
  **Nu against a published table, with grid convergence** —
  `validation/natural_convection -conv` (added 2026-09-04) sweeps N at fixed Ra
  against de Vahl Davis (1983) for the side-heated cavity at Pr = 0.71. Ra = 1e5
  falls monotonically -1.00, -0.39, -0.18, -0.05 % at N = 32/48/64/96, order
  2.4-3.4. Ra = 1e3 and 1e4 are already inside the TABLE's own precision (it
  prints four figures, so +/-0.045, 0.022, 0.011 % of its value) and Ra = 1e4's
  deviation crosses zero, so no order can be fitted there — an order is only
  measurable where the error exceeds the reference's precision, and that is
  stated in the banner rather than papered over. This validates the buoyancy
  coupling, the isothermal walls and the Nu wall-gradient evaluation externally;
  it does NOT validate Rayleigh-Benard in the turbulent regime, for which no
  matched published table has been located.

  **Rayleigh-Benard now checks three references, not one** (`-conv`,
  `-marginal`, `-rate` in `validation/rayleigh_benard.cpp`). Ra_c converges at
  order **2.36** for the midway wall pair (+0.25% at H=48) and only **1.75** for
  the on-node pair with twice the error — the curvature slip showing up as a
  convergence signature, not just an offset. The neutral curve's minimum gives
  k_c H = 3.1107 (-0.20%) at H=16 and 3.1196 (+0.08%) at H=32, against Ra_c
  errors of +3.27% and +0.65% from the SAME runs: **k_c is the sharper test**,
  because the O(1/H^2) error shifts the curve vertically and a vertical shift
  does not move a vertex sideways. Two traps: locating that minimum with a
  parabola over the whole sampled range is biased (+2.81% against +0.08% for a
  local 3-point fit) because the curve is asymmetric by a factor 2.13 in
  curvature; and the growth rate's zero crossing (1719.6) agrees with the
  bisected Ra_c (1718.9) to 0.04%, which is a far stronger use of the same runs
  than bracketing the sign of sigma.
  **The two `rb_high_ra` drivers were ALIGNED on 2026-09-04**, because two
  independent implementations of one case are only worth something if they are
  the same setup, and four things differed. `demonstrator/rb_high_ra` now uses
  the new **`ScalarRegularised<D3Q7>`** (`src/collision/ScalarRegularised.hpp`)
  rather than `ScalarBGK` — same equilibrium, same diffusivity, identical at
  ω = 1, but it relaxes only the flux moments and annihilates the three ghosts,
  which is what the reference does and what `GPU/` does. BGK relaxes the ghosts
  at ω too, and at ω → 2 that is a *reflection*: they invert every step and
  never damp. It also now sets **`omega_bulk = 1`** (it was −1, "follow omega",
  so the trace relaxed at 1.9984 while both siblings used 1), prints
  **`Nu_ref`**, and defaults to **`-ic cond`** like its twin. `-sop reg|bgk`
  selects the operator — a template dispatch, not a runtime branch, so `Opts`
  and `run` sit at namespace scope for the nvcc restriction. The temperature
  gauge stays deliberately different (symmetric here, [0,1] there): a gauge is
  not physics, and symmetric is the defence recommended above. So `Nu_bot`,
  `Nu_top` and `Nu_vol` are comparable between the twins and `T_min`/`T_max`
  and `Nu_ref` are not.

  **Ra_max is a RESOLUTION rule, not a stability rule** (measured 2026-09-05,
  and it contradicts itself in both directions at Ra = 1e11). At H = 498,
  Ra/Ra_max = 0.62 — "comfortable" — and the run HALTED at t/t_ff = 25 with
  T = 1.19 against a physical 0.5 and Nu_bot/Nu_top = 80.4/57.4; `GPU/`
  reproduced that from a *cold* start on an A100, so H = 498 fails from both
  ICs in both codebases. At H = 98, Ra/Ra_max = 169 — "hopeless" — and it did
  NOT halt: T_min hit −0.92 at t = 20 and stayed out of bounds to t = 64, with
  Nu_bot ≈ 95 at **thirty times** Nu_top ≈ 3.1 and ⟨T⟩ at mid-depth still
  −0.5000 at t = 19, i.e. a layer filling rather than a steady state. Neither is
  a measurement. The grid 25× coarser SURVIVED where the finer one halted — a
  two-point hypothesis, stated as one: far above the H/2 ceiling the scheme
  clips and acts as an implicit LES; *near* the ceiling it attempts a layer it
  cannot represent and fails. It predicts H = 998 (Nu/ceiling = 0.43) behaves
  better than H = 498 (0.87), which is what `GPU/csf3/rb_cold.sub` tests.

  **A COLD START'S SEED HAS TO GO WHERE THE GRADIENT IS.** Both drivers seeded a
  density mode at mid-depth — right for `cond`, useless for `cold`, because a
  cold start has no gradient except in the diffusing layer at the bottom plate
  and δ = √(D t) is 2.55 cells at t/t_ff = 7 (H = 498). Measured: the mode
  decayed 9.69e-05 → 6.26e-05 over seven free-fall times with its peak pinned to
  the seed row, while `cond` at the same parameters had it *growing* at ×1.51.
  Since 2026-09-05 `cold` seeds the TEMPERATURE in the boundary layer instead —
  non-negative, so the initial field itself stays inside the maximum principle,
  and broadband, because the layer picks its wavelength from δ (≈ 5 cells here)
  not from the box. Measured at 200×100: fluctuation rms grows ×1.47/t_ff
  (σ = 0.38) from t = 2 with its peak at y = 1–2, against σ = 0.45 for `cond`.
  `cond` is untouched, so every number measured with it still stands.

  `demonstrator/rb_high_ra` (Kokkos, D3Q27 CM + D3Q7 scalar) measures where that
  configuration is usable, at 200x100, U_f = 0.05, conductive IC, 100 free-fall
  times: Ra = 1e6 gives Nu_bot/Nu_top = 7.40/7.48, agreeing to 1% and to 7% of
  0.14 Ra^0.29 = 8.0; Ra = 1e10 gives 46.57/46.79, agreeing to 0.5% but landing
  on the ceiling H/2 = 49 rather than the physical 111; Ra = 1e14 DIVERGES by
  t/t_ff ~ 30 with either IC. So Ra_max ~ (H/0.28)^(1/0.29) -- 6e8 at H = 98,
  2e12 at H = 998 -- and Ra = 1e14 wants H >= 3200. Two traps that cost real
  time: the volume Nusselt relation must use the FLUCTUATIONS,
  <vT> - <v><T>, because Guo's half shift leaves a uniform F/(2 rho) in the
  reported velocity (the textbook form read Nu = 53.77 for a fluid AT REST);
  and Nu_vol carries H/D = 1.7e8 at Ra = 1e14, so it is pure amplified noise
  unless it is far above its floor -- believe the two plate estimators when they
  agree with each other.
  **FP16 population storage was measured and rejected** (2026-09-04).
  `-DLBM_STORE16_EMULATE=ON` quantises on the way into the existing FP32 array,
  so it costs memory nothing, buys speed nothing, and reproduces the arithmetic
  exactly — the accuracy question answered with no GPU and no layout change.
  Result: host_check 0 -> 2 failures, host_physics 0 -> 30. The sharp number is
  newpaths' shear decay WITH the shift on: FP16 gives 0.0071934 against 0.005217
  at A = 1e-2 (38% off) and -0.966 at A = 1e-5, where RAW FP32 gives -0.968 —
  i.e. **FP16 storage costs exactly what removing the shift costs**. The
  arithmetic predicts it: FP16 loses 2^13 = 3.9 decimal digits, the shift buys
  1/(3u) = 0.8 digits at u = 0.05, and they cancel at u = 4.1e-05, which is
  where the measurement crosses over. This does not refute FluidX3D — it uses a
  custom 16-bit layout (~0.9 digits back, still 2.2 short) and targets
  percent-level accuracy, whereas this tree asserts Poiseuille x H^2 to three
  digits and mass drift of exactly zero. The trade is 1.7x traffic for the
  accuracy standard; the flag is kept as a standing instrument for retrying it.
  One property to know rather than a gap: **regularised walls are not mass
  conserving**, since BC3 overwrites populations. A closed box holds its mass
  exactly; a driven cavity leaks linearly and does not saturate (−1.7e-2 over
  20000 steps at 32²). Do not read an absolute pressure off a long cavity run.
- **The rigid body is of uniform density**, and there is no collision model, so
  bodies interpenetrate. It is no longer 2-D: `RigidBody3D.hpp` and
  `PenalisedBody`'s `refresh6`/`advance6` solve the full 6x6 with a rotating
  inertia tensor and a quaternion pose (`Box` shape, `demonstrator/cube_entry`).
  The two paths are **separate**, not layered: `Rect` and `Wedge` are prisms with
  `six_dof = false` and keep the validated 3x3, so calling `refresh()` on a Box
  or `refresh6()` on a Rect is a compile error rather than a wrong answer.
  `set_uniform_density6()` measures the inertia in the world frame and stores it
  in the body frame — using the 2-D `set_uniform_density()` on a 6-DOF body
  fills the mass and leaves the tensor at **zero**, which makes the angular half
  singular and produces a plausible tumble rather than a failure.

---

## Measurement discipline

This tree has produced several confident wrong conclusions. All of them came from
the same few mistakes, so they are worth naming.

- **Change one thing.** Two runs that differ in geometry *and* precision, or in
  viscosity *and* interface width, cannot attribute a difference. A comparison of
  two models at their own driver defaults is a comparison of drivers.
- **Match the right quantity.** For a density ratio, match *kinematic* viscosity,
  not dynamic. Matching μ across a ratio of 100 leaves the heavy phase at ν/100
  and drives ω to 1.994 against a limit of 2 — which reads as the model failing
  when it is the setup.
- **Check convergence before quoting.** Report a number only after the time
  series is flat, and say so if it is still creeping.
- **Agreement between a port and its host reference proves the port, not the
  physics.** Both run the same arithmetic. When a port sits several times worse
  than the code it came from, that gap is a defect until shown otherwise — do not
  write it up as a property of the model.
- **Check that a metric is reproducible before ranking with it.** The step at
  which an unstable run blows up moves by 2× with the Kokkos backend alone. It is
  not a measurement.
- **A wrong constant is still a consistent simulation.** Every test can pass, the
  device can match the host to every digit, and a quantity can converge and hold
  — while the case being solved is not the one intended. Verify inputs against
  the source paper and against the sibling implementation, not only against
  internal consistency.

---

## Conventions

- Match the surrounding code: banner comments that argue the decision, measured
  numbers rather than adjectives, and an explicit statement of what a piece does
  *not* do.
- When a limitation is found, write it into the module banner and into
  `doc/m3lb.tex`'s "Known limitations" — not only into a commit message.
- Results in `results/` are tracked reference data. Build trees and field dumps
  are ignored; `doc/m3lb.pdf` is tracked because it is the deliverable.
- Rebuild the document with `make -C doc` (needs a full TeX install).
