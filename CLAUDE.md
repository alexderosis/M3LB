# M3LB — working notes for Claude

Lattice Boltzmann solver, C++20 on Kokkos. Read this before proposing changes or
writing a case; most of it is knowledge that is otherwise spread across
`doc/m3lb.pdf` (154 pages) and the banner comments at the top of each header.

**The banners are the documentation.** Every non-obvious decision in this tree is
argued at the top of the file that implements it, usually with the measurement
that settled it. When you are about to change something, read that file's banner
first — it frequently already says why the obvious change is wrong.

---

## Two independent codebases

| | what it is | when to use |
|---|---|---|
| `src/` + `validation/` | the main solver, Kokkos, four lattices, four collision operators, thermal + MHD + multiphase + free surface | default; anything on CPU; anything needing the full physics |
| `GPU/` | a second implementation written directly in CUDA, sharing **no headers** with the first | GPU runs, or cross-checking one implementation against the other |

They deliberately duplicate the physics. That is the point: they agree where they
overlap, and disagreement is a bug in one of them. Do not "de-duplicate" them.

**One overlap is now enforced rather than checked by hand.** `tests/cross_colour`
drives BOTH colour-gradient operators through 320 identical randomised states —
two `namespace lbm`s in separate translation units joined by `extern "C"` — and
diffs all 27 populations: worst difference **0 exactly**, at both precisions. It
lives in the parent because the parent already links Kokkos and `GPU/`
deliberately does not; making `GPU/`'s CMake acquire Kokkos to be cross-checked
would spend the independence that makes the comparison worth anything. The
pattern generalises to any other operator the two trees share.

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
| free-slip where the other walls are on-node | `set_specular_nodes` | exact mirror; **on-node**, collides, takes edges and corners |
| zero-flux scalar wall | `ScalarSpecular` + `ScalarSolver::set_specular_walls` | **on-node**; the only one usable where the field is differentiated or integrated — see below |
| + melting / solidification, latent heat | `ScalarSolver` + `EnthalpyBGK` | transports the total enthalpy H, **not** T — `temperature()` returns an ENTHALPY, and `PhaseChange::invert` gives T and f_l. `EnthalpyRegularised` above ω ≈ 1.9 |
| + charge carriers in an electric field | `ScalarSolver` + `ChargeCentralMoments` | D3Q27, advects at the **drift** velocity `u + KE`, not at `u` |
| + electric potential (Poisson) | `ScalarSolver` + `ScalarBGK` + `add_source` | no new solver — see `validation/ehd_hydrostatic.cpp` |
| + magnetic field | `MagneticSolver` | Dellar vector distribution |
| two-phase, diffuse interface | `PhaseFieldSolver` + `MultiphaseCentralMoments` | conservative Allen–Cahn, prescribed interface width, density ratio ~100. **3-D is D3Q27 + D3Q27, not D3Q27 + D3Q7** — see the rule below |
| two-phase, diffuse, high ratio | `ColourGradientSolver` | no interface equation; width is an *outcome*. **Sub-percent to γ = 20, ~3% at γ = 100, comes apart at γ = 1000** — and the steps needed scale with γ; see the rules below |
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

**Standing rule: the phase field's fluid operator is `MultiphaseCentralMoments`,
and in 3-D the pair is D3Q27 + D3Q27.** There are THREE multiphase fluid
operators and they are two physics tiers, not three collisions of one model:

| operator | model | when |
|---|---|---|
| `MultiphaseCentralMoments` | pressure-based, density + viscosity ratio, central moments | **the default** |
| `MultiphasePotentialBGK` | the same model and field set, BGK | only to attribute a difference to the collision; this is what `-op bgk` selects |
| `MultiphaseBGK` | **matched density**, surface tension as a capillary *stress* | a different equation of state. Carries no `rho_L`/`rho_H`, `mu_L`/`mu_H` or `Lap`, so substituting it does not compile rather than running wrong. `validation/laplace.cpp` wants it, deliberately |

The FLUID lattice is not a choice: all three `static_assert`
`supports_navier_stokes`, false on D3Q7, so a 3-D phase-field fluid is D3Q27 and
a D3Q7 one is a compile error. The PHASE lattice is a separate, free choice, and
D3Q7 is legal there — but only under `PhaseFieldBGK`, because
`PhaseFieldCentralMoments` needs a product basis and D3Q7 is not a product
lattice. So **D3Q27 + D3Q7 is the THERMAL scalar's pairing, not this one**, and
it buys nothing here anyway: `GradientLatticeOf<D3Q7>` is D3Q27, so the
27-neighbour gradient gather is paid either way. The one case in either codebase
still running D3Q27 + D3Q7 is `GPU/src/bubble.cu`.

**Both defaults now exist in code rather than as a habit.** In `src/`,
`PhaseFieldSolver`'s collision template argument defaults to
`DefaultPhaseCollision<L>` — `PhaseFieldCentralMoments<L>` on a product lattice,
`PhaseFieldBGK<L>` on D3Q7/D2Q5, because a flat default would turn "omit the
operator" into a compile error on exactly the lattices entitled to omit it. The
fluid side is the alias `DefaultMultiphaseCollision<L>`; it is deliberately NOT a
default on `FluidSolver`, which is shared with single-phase flow where the
standing rule above forbids a multiphase operator. Both live in
`src/solver/PhaseFieldSolver.hpp`. Adding them changed no existing case: all 13
instantiations in the tree name their operator explicitly.

In `GPU/` the defaults are runtime members, `MultiOp fluid_op_` and
`PhaseOp phase_op_` (`phasefield.cuh`, `hostsim.hpp`), both moved to central
moments on 2026-09-18. `phase_op_` is lattice dependent for the same reason as
the parent's trait, and that is a correctness point rather than a nicety:
`pf_phase_node`'s central-moment branch sits inside an `if constexpr (Q == 27)`,
so on D3Q7 a CM request does not fail, it EVAPORATES — and the guard in
`set_phase_op` cannot catch it, because a member initialiser never reaches the
setter.
Flipping it reaches `bubble.cu`, the only driver that inherits it and solves a
flow. Re-measured on a T4 at the table's own settings — 64³, FP32, 40000 steps,
2026-09-19 — the four rows that already converged **did not move**, agreeing with
BGK in the fourth digit. The row that used to DIVERGE now completes: gamma = 100
at matched dynamic viscosity, −3.68 % with a spurious current of 1.194e-05, where
BGK blew up at omega = 1.994. **An earlier version of this entry claimed a
−19.62 % → −9.40 % improvement from a 48³ FP64 run at 2000 STEPS. That run was
not converged**, and what it measured was the two operators approaching the same
answer at different rates rather than reaching different answers — a converged
static droplet is a force balance and the collision has almost no say in where it
balances. The figure was quoted in two places before a converged run contradicted
it. Quote a converged number or none.
`newpaths.cu` also inherits it and does NOT move, because its slab rows are
density-matched and force-free, so the fluid populations stay identically zero
— verified, both rows still 4.44951. In `src/` there is no library default to
change: each driver names its own `using FColl = ...`, and every one that
couples a flow already names the central-moment operator.

**Standing rule: "two-dimensional" means a 3-D lattice with `nz = 1`, not D2Q9.**
Every `Domain` in this tree is three-dimensional — `cavity.cpp:108`,
`hartmann.cpp:68`, `laplace.cpp:167`, `orszag_tang.cpp:189` all size
`Domain d(N, N, 1, ...)` — so `nz = 1` is how a plane is spelled whatever the
lattice, and the only real choice is D3Q27 (+ D3Q7) against D2Q9 (+ D2Q5). New
work takes the 3-D pair, and a figure says which one it used.

**For the FLUID the reduction is exact rather than approximate.** D3Q27 is the
product lattice D2Q9 x D1Q3, so at `nz = 1` with z periodic the three
z-populations wrap onto the node itself and the dynamics ARE D2Q9's. Measured
2026-08-31 in `validation/tgv2d.cpp` over N = 8..256: the two lattices agree **to
every printed digit**, all six relative L2 errors and all six
viscosity-from-decay fits. That makes a 2-D case the cheapest available check
that the D3Q27 path is correct, and `validation/orszag_tang.cpp:349` runs it as
exactly that reduction test.

**IT IS NOT EXACT FOR THE SCALAR OR MAGNETIC PARTNER, AND THAT IS THE TRAP.**
D3Q7 has `cs2 = 1/4` where D2Q5 has `1/3`, and the two discretise the same
operator with different truncation — a weighted 27-point Laplacian against a
5-point one — so swapping them MOVES the numbers. `orszag_tang`'s `-maglat`
exists to hold the magnetic lattice fixed across fluid lattices for precisely
this reason. Where the two pairs agree anyway the answer belongs to the physics
and not to the stencil: `ehd_electroconvection` at ny = 81, T = 190 reads 3.7413
on D3Q27 + D3Q7 against 3.7472 on D2Q9 + D2Q5, **0.16 %** apart with the same
lateral mode content. So quote the digit-for-digit claim of the fluid only.

**The cost is real and it is memory traffic.** That same case moves 81
populations per node per step against 23, and D2Q9 measured **3.7x faster**
(20.5 s -> 5.5 s at ny = 41) — which is what made its refinement ladder
affordable. Where a sweep rather than the answer is the point, the 2-D pair is
the right tool; say that it was used.

**Two things forbid the 3-D lattice outright.** `PenalisedBody` `static_assert`s
`L::D == 2 || Shape::three_d`: `Rect` and `Wedge` have a chi independent of z,
so on D3Q27 they would model an infinite prism with a one-angle solve, and a 2-D
rigid-body case therefore stays on D2Q9 (`validation/enan_wedge.cpp:81` argues it
at the instantiation). And the existing D2Q9/D2Q5 cases are **not** to be
converted: `results/`, `doc/fig/` and the README tables were measured with them,
and on the scalar half a conversion would change the answer rather than confirm
it.

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
scheme transports the temperature rather than the enthalpy. **The interface is
SECOND ORDER, NOT EXACT** — `src/collision/ScalarBGK.hpp:54` says so and
`validation/zhou_thermal.cpp` measures the cost: 0.11 % at κ = 100 and N = 400,
falling as N^-2. An earlier version of this entry said "exact rather than second
order (1e-9 %)", which was the κ = 1 row — where the exact solution is linear and
the scheme is exact for that reason — misquoted as the whole table. The
distinction matters for a melting front, which *is* an interface between two
diffusivities. Where ρc_p is NOT uniform, use `EnthalpyBGK`; that header derives
why no choice of ω can substitute. Nothing in `src/` converts
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
  only because that case is D3Q7, and copying the line to a D3Q27
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

## Importing a city from OpenStreetMap

`tools/osm_city.py` turns OSM building footprints into the
`<prefix>_heights.npy` + `<prefix>_meta.json` pair `src/io/HeightField.hpp`
reads, so `demonstrator/urban` can be pointed at any city without leaving this
tree. It is **stdlib only** — no osmnx, no shapely, no pyproj, no numpy, no
venv — because the job is an HTTP GET, a map projection and a polygon fill.

```bash
python3 tools/osm_city.py --place "Manchester city centre, UK" \
    --domain-m 2000 --dx 5 --top-m 300 --out geom/mcr
./build/demonstrator/urban -geom geom/mcr -bearing 250 -diff 20 \
    -minutes 12 --kokkos-num-threads=4
```

`dz == dx` always: the solver is isotropic and `HeightField.hpp` rejects
anything else. Responses are cached under `<out>_osm_cache`, so a re-run is
offline and a figure's raster can be rebuilt from the same bytes months later.

**IT AGREES WITH THE osmnx IMPLEMENTATION ON EVERY CELL, AND THAT IS THE ONLY
REASON TWO SILENT BUGS WERE FOUND.** Against the Pollutant project's
`mcr_heights.npy`, reading the *frozen* 2026-07-30 Overpass snapshot that built
it (`--osmnx-cache`, which is what makes it two rasterisers on identical input
rather than two downloads seven weeks apart): **160,000 of 160,000 cells
identical**, all 47,787 built cells included, worst difference 0.0 m. A live
query at the same centre gives 99.91%, and those 149 cells are two-sided —
71 gained, 8 lost — i.e. real editing, not a defect. Repeat it with:

```bash
python3 tools/osm_city.py --lat 53.4729842 --lon -2.2502070 \
  --domain-m 2000 --dx 5 --top-m 300 --fallback flat \
  --osmnx-cache ~/Desktop/02_Code_Projects/Pollutant/osm_cache_mcr \
  --out /tmp/mcr --compare-to ~/Desktop/02_Code_Projects/Pollutant/mcr_heights.npy
```

- **`--fallback flat` is the DEFAULT and is the worse model**, deliberately.
  It puts a terraced house and a seven-storey office at the same 10 m, where
  `--fallback type` calibrates a storey count per building type. Flat is the
  default because it is what the existing Manchester and Manhattan rasters were
  built with, and a cross-check whose two sides differ in the model checks
  nothing. **Use `type` for new work and say which one a figure used.**
- **THE INDEX ORDER IS `i*ny + j`, EASTING SLOW.** The contiguous-looking
  `j*nx + i` is the same array transposed: it rotates the city and leaves every
  cell count, built fraction and mass budget unchanged. This file's importer
  got it wrong first, and no total saw it — the diff did.
- **A RELATION'S MEMBERS MUST BE RESOLVED BY `ref`.** In the node-reference
  form (what osmnx caches, and what `--osmnx-cache` reads) a member is a bare
  `{type, ref, role}` with no geometry; reading it in place returns nothing and
  every multipolygon **evaporates**. Worse, a multipolygon's member ways are
  untagged — the tags live on the relation — so the ordinary way loop does not
  recover them either. Cost: 39 whole buildings, 1.4% of Manchester. The
  `out geom;` form carries member geometry inline, so every synthetic test
  passed.
- **A PLACE NAME IS NOT A COORDINATE.** "Manchester city centre, UK" geocodes
  today to a point **1,459 m** from where the same string landed in July —
  Nominatim matches *City Centre* inside a postal address, and both hits are
  offices, not districts. On a 2 km domain that is 73% of a side. Use
  `--lat/--lon` for anything to be repeated; the provenance of a raster is the
  coordinate in its `meta.json` and its cached response, never the place string.
- **`ctest` carries `osm_city` (the importer's own checks) and
  `osm_city_fixture` + `height_field_osm` (the importer against the solver).**
  The last two live in `validation/CMakeLists.txt`, not `tests/`, because
  `tests/` is configured FIRST and an `if(TARGET height_field)` there is simply
  false — the test would register nowhere, in silence. The Manchester
  comparison is NOT a test: it needs a 4 MB snapshot outside the repository, so
  it is a recorded measurement. Run it by hand when the rasteriser changes.
- **The prescribed wind's stability floor bites immediately on a real city.**
  Manchester at the default `-diff 5` reports a margin of 48x against the 100x
  threshold and diverges at t = 60 s; `-diff 20` gives 192x and runs. The
  solver prints the diffusivity that would reach 100x — read it before the run,
  not after.

---

## Invariants that break silently

These produce plausible, converged, wrong answers rather than crashes.

- **Direction ordering is a contract.** `opp(i)` is `i+1` for odd `i` and `i-1`
  for even (`src/lattice/Lattices.hpp`). Esoteric Pull depends on it. If you add
  or reorder a lattice's velocity set, opposite pairs must stay adjacent.
- **Storage × streaming pairings are not free.** Shifted populations centre the
  stored variable on `p̃ = 1`, which is exactly the pressure gauge that must be
  avoided at a density ratio — so the multiphase operators declare
  `RawPopulations`. `FreeSurfaceSolver` does not take a streaming policy at all:
  `src/solver/FreeSurfaceSolver.hpp:222` hardcodes `using Streaming =
  TwoLattice<L>;`, because it reads a neighbour's post-collision state while
  writing its own. An earlier version of this entry said it `static_assert`s
  *against* Esoteric Pull; it does not — its only two assertions are
  `supports_navier_stokes` and `ProductBasis<L>::enabled`. Same practical
  effect, but it means there is no streaming clash to resolve when pairing it
  with a `ScalarSolver`, which carries its own lattice and its own storage.
- **EVERY NAVIER-STOKES LATTICE HERE IS NOW A PRODUCT LATTICE, AND THE REDUCED
  ONES NEVER WERE.** D2Q9 and D3Q27 are product lattices; **D2Q5 and D3Q7 are
  NOT**, and no removal changed that -- `ProductBasis::is_product_lattice()`
  demands all 9 or all 27 sign combinations, which a 5- or 7-velocity set simply
  does not have. An earlier version of this entry said "EVERY LATTICE", which is
  the sentence a reader would use to conclude a D3Q7 scalar or phase field could
  take a central-moment operator. It cannot: that is the `static_assert` below,
  and it is why a 3-D phase field wanting central moments must run its transport
  on D3Q27 rather than D3Q7. `doc/m3lb.tex`'s "Both three-dimensional lattices in
  this tree satisfy it" carries the same error and is not yet fixed.
  What D3Q19's removal changed is narrower: it was the one NAVIER-STOKES lattice
  that was not a product set -- D3Q27 minus its corners, reached through a
  generated monomial basis -- and it went on 2026-09-18 along with
  `MonomialBasis.hpp` and `MATLAB/D3Q19_CM.m`. `SelectBasis` therefore always
  returns `ProductBasis`,
  and the operators that need the factorised transform
  (`MultiphaseCentralMoments`, `PhaseFieldCentralMoments`, `ChargeCentralMoments`,
  `FreeSurfaceSolver`) still `static_assert` on `ProductBasis::enabled`, which is
  now a guard against a future lattice rather than against D3Q19. If you add a
  non-product lattice, that assertion is what will catch it -- a compile error
  rather than a wrong answer. Historical measurements on D3Q19 are kept in
  `results/`, `doc/fig/` and the README tables and are marked as such.
  **A STALE `-lat d3q19` IS NOW A HARD ERROR, because it briefly was not.**
  D3Q19 was `tgv3d`'s DEFAULT and a documented `orszag_tang` option, so every
  saved command line carried it. `campaign::dispatch` returned false quietly and
  the callers disagreed about what that meant: two dropped the bool entirely,
  four printed a line and returned 0, and `tgv3d` printed a header naming
  `lattice d3q19` above a run that never happened. It now names the valid sets
  on stderr, says NOTHING WAS RUN, is `[[nodiscard]]`, and every caller exits 1;
  the two cases whose banner names the lattice check `known_configuration()`
  BEFORE printing it. `orszag_tang` has its own else-chain and validates `-lat`,
  `-maglat` and `-op` up front rather than falling through to D2Q9.
- **THE COLOUR GRADIENT'S ALPHA INTERPOLATION DECIDES WHETHER IT SURVIVES A
  DENSITY RATIO, AND BOTH READINGS LOOK IDENTICAL AT gamma = 1.** The rest term
  of the equilibrium is `rho phi_i(alpha)` under either reading — `phi_i` is
  *affine* in alpha, so `sum_k rho_k phi_i(alpha_k) == rho phi_i(alpha_P)`
  identically, with `alpha_P = 1 - 19P/(9 rho)` and `P = sum_k rho_k cs_k^2`
  (pinned at 2.6e-15 in `tests/test_colour_gradient.cpp` block 2). So the ONLY
  difference between De Rosis, Huang & Coreixas Eq. (D6)/(D13) and the
  per-colour form is **the rule for interpolating alpha**: linear in the order
  parameter, or `alpha_P`. Linear interpolation makes the interface pressure
  spike by exactly **(gamma+1)^2 / (4 gamma)** — 1.00, 3.03, 25.50, 250.50 at
  gamma = 1, 10, 100, 1000 — against a bulk of 1/3, because `cs^2` is linear in
  alpha while `rho` is not correspondingly interpolated. Measured on
  `validation/static_droplet` at 48^3, R = 16, tau = 1, **converged**, Laplace
  error at gamma = 1 / 10 / 20 / 100 / 1000: `PerColour` gives
  **0.88 / 0.68 / 0.46 / 3.21 % / comes apart**, `AlphaBar`
  **0.88 / 3.34 / 1.51 % / NaN / NaN**. `ColourGradient::RestTerm` selects it,
  **`PerColour` is the default**, and `static_droplet -rest bar|percolour`
  measures the gap. At gamma = 1 the two are algebraically identical, so a
  matched-density test cannot see any of this. **The sixth-order equilibrium is
  NOT implicated**: holding it fixed and swapping only the reading reproduces
  Saito et al.'s third-order ladder to four or five significant figures at every
  ratio.
- **THE COLOUR GRADIENT'S STEP COUNT SCALES WITH THE DENSITY RATIO, SO A LADDER
  AT ONE STEP COUNT MEASURES THE TRANSIENT.** Relaxation slows as gamma grows
  while tau is held at 1. Measured: gamma <= 20 is converged by 8000 steps (0.04
  points of drift out to 32000), but gamma = 100 reads **2.12% at 8000, 3.11% at
  16000 and 3.21% at 32000**. A 1500-step ladder gave 493% at gamma = 100, and
  that number was written into three files as a property of the model before a
  longer run contradicted it — the same mistake `ehd_cavity` already records,
  made again. At gamma = 1000 it is not slow convergence at all: the Laplace
  jump crosses **zero** (+4.84e-3, +1.20e-3, −2.34e-4 at 8000/16000/32000) while
  the interface widens 4.92 → 5.47 cells, which is a droplet dissolving rather
  than a transient. Quote nothing there.
- **A SEED THAT DISAGREES WITH ITS OWN COLLISION READS AS A BROKEN MODEL.** The
  first attempt at the isolation above changed the collision's rest term but left
  `ColourGradientSolver`'s seed and the recolouring on the other reading. That
  put a 3x pressure mismatch at the interface on step 0 and returned NaN at
  gamma = 10 — and it looked exactly like the sixth-order equilibrium failing,
  which is the conclusion it produced. The operator now owns the split
  (`seed_at_rest`), so the solver cannot pick a different reading from the
  collision. When a model has a switch, ask what ELSE reads it before attributing
  anything to it.
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
- **A `PhaseChange` THAT WAS NEVER NORMALISED INVERTS THROUGH ITS DEFAULTS AND
  RETURNS A PLAUSIBLE WRONG TEMPERATURE.** `EnthalpyBGK::set_material` takes its
  argument **by value** and calls `normalise()` on its OWN copy, so the caller's
  object keeps the declared defaults `dTm_ = 0, H_l_ = 1, B_ = inv_B_ = 1`.
  `invert()` then takes the LIQUID branch at any `H > 1` and returns
  `T_l + (H - 1)/cp_l` instead of `T_l + (H - H_l_)/cp_l` — high by the latent
  heat in temperature units — and inside the band it returns `T_s` flat. Nothing
  fails; the run converges. **Take the normalised copy from
  `coll.material()`**, which is the intended door, or call `normalise()`
  yourself. Since 2026-09-21 `invert()` **aborts** on `!ready_`, so this is now
  loud rather than silent; the entry stays because the shape of the mistake
  (a library that normalises a copy and leaves the caller holding a stale one)
  generalises. Taking the argument by non-const reference was tried first and
  rejected: it forces every caller to give up const on its own material, and it
  still cannot catch an object that never reached a collision.
  It cost two cases. `validation/melt_pool.cpp` extracted the **1879 K** contour
  where it asked for 1928 K, moving the melt-pool width by 1.59 um and the depth
  by 0.75 um. `validation/keyhole.cpp`'s `-enthalpy` path ran 414 K too hot
  above the band and gave 283.090 um against the apparent-cp path's 134.219 um —
  **the 2x gap that case recorded as UNEXPLAINED through two earlier bug
  fixes**. One line took it to 123.594 um, i.e. 0.92x. `validation/stefan.cpp`
  was never exposed: it calls `normalise()` explicitly and never inverts a bare
  `pc`, which is why its eleven criteria passed throughout.
  **AND THE SELF-CHECK BOTH CASES CARRIED WAS STRUCTURALLY BLIND TO IT.** Each
  inverted the AMBIENT enthalpy and required ambient back; keyhole printed
  `H_amb = -1578.0000 -> T = 300.00 K` *identically* with and without the bug,
  because at `H < 0` the SOLID branch reads only `T_s` and `cp_s` — which the
  caller sets directly — so the branch carrying the defect is never reached. A
  self-check on a piecewise map must exercise the **band edges**: round-trip
  `enthalpy_of(T_l, 1) -> T_l` and `enthalpy_of(T_s, 0) -> T_s`. A point in the
  interior of one branch certifies that branch and nothing else.
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
  **"CLOSED" MEANT THE RINGING, NOT THE WALL ERROR, AND THE DIFFERENCE WAS
  MEASURED ON 2026-09-06.** `ehd_electroconvection -frozen` (new) drops the
  fluid and the seed, so the charge/potential solution is exactly
  ONE-DIMENSIONAL and any lateral structure IS the wall; the doubled box is
  laterally flat to 0.00 % there. Measured at T = 140, C = 10, at matched
  distance from each geometry's own mirror plane, `ScalarSpecular` is about
  TWICE the halfway bounce-back wall's error at every resolution -- peak
  16.68 / 7.06 / 2.23 % against 14.41 / 4.49 / 1.14 % at ny = 41 / 81 / 163,
  both converging near second order. What it did remove is the SAW-TOOTH: the
  halfway profile alternates into the wall (0.1200, 0.1034, 0.1012, 0.1027) and
  the on-node one is monotone (0.1220, 0.1155, 0.1057, 0.1024). The odd-even
  mode is gone and a larger SMOOTH wall layer took its place.
  **AND THAT LAYER IS ITSELF AN ω → 2 EFFECT.** At weak injection, where a
  correct zero-flux wall would be nearly exact, sweeping α at C = 0.5 gives
  9.76 / 6.60 / 1.97 / 0.04 % as ω_q goes 1.99952 / 1.99521 / 1.95312 / 1.61290,
  against 0.87 / 0.21 / 0.10 / 0.01 % for the halfway wall -- ten times worse,
  and collapsing as ω_q leaves 2. It is the CHARGE's own wall, not the
  potential's: at C = 0.05 the charge bends φ by 0.006 % and its wall error is
  still 12.60 %. **The mechanism is NOT diagnosed.** Two pairings are untested
  and are the obvious suspects: `validation/scalar_specular.cpp` exercises
  `ScalarSpecular` with `ScalarBGK` only, a SOLENOIDAL velocity, no source and
  ω = 1.99, while the charge here runs `ChargeCentralMoments` with a drift whose
  divergence is `Kq/ε > 0`. `ehd_cavity`'s specular-vs-periodic check is
  consistent and could not have seen it — it is a VOLUME INTEGRAL, measured at
  0.42 % (0.05 % interior-only), which is what a 10 % excess in two columns of
  forty-one contributes to a volume average. **A wall layer hides inside an
  integral diagnostic; use a pointwise one, on a problem whose exact solution
  you know is 1-D.**
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
- **TWO COUPLED LATTICES RUNNING AT DIFFERENT TIMESTEPS DO NOT SHARE A VELOCITY
  UNIT, AND HANDING ONE THE OTHER'S VIEW COMPILES.** A lattice velocity is
  `u_lat = u_phys dt / dx`, so if the fluid sub-cycles at `dt_f = dt/N` while a
  scalar keeps the diffusive `dt`, the scalar's velocity is **N times** the
  fluid's. `ScalarSolver::set_velocity` takes the Views by handle and cannot
  know this: at `N = 1` it is exactly right, and at `N > 1` it under-advects the
  transported field by N — an advection-diffusion problem quietly solved at 1/N
  of its Peclet number, which converges and looks like a weakly coupled flow.
  `validation/melt_pool.cpp` passes a scaled copy when `nsub > 1`. The same
  factor bites every *reported* quantity derived from a lattice velocity:
  `u_phys = u_lat dx / dt_f`, and using the scalar `dt` there makes the peak
  speed appear to fall as 1/N, i.e. makes a correct sub-cycle look like a bug.
  That was three separate sites in one file, including a "the flow is not
  moving" guard that would then fire on a healthy run. **The self-consistency
  test is that the PHYSICAL velocity does not depend on N** — measured invariant
  to 2.7 % over N = 1..8 while Ma fell 0.197 -> 0.025, exactly 1/N. A force
  carries `dt_f^2` and a viscosity `dt_f`, so those are two more places to check.
  **THAT VELOCITY CHECK IS NECESSARY AND NOT SUFFICIENT**, which the next entry
  cost a day to learn: it passed while the mushy sink was still rescaling with
  N, because the sink acts at the pool edge and the peak velocity is read at the
  surface. Check the REPORTED RESULT for N-independence, not only the field the
  conversion is most obviously about.
- **A COEFFICIENT SPELLED IN LATTICE UNITS IS A PHYSICAL QUANTITY IN DISGUISE,
  AND IT SILENTLY RESCALES WHEN THE TIMESTEP MOVES.** `melt_pool`'s Carman-Kozeny
  mushy sink was `A_lat = 0.8`, "the strength at f_l = 0, IN LATTICE UNITS", and
  `validation/mushy_sink.cpp` bounds it at 1.0 — so 0.8 looked like a
  well-argued numerical choice. It is not a free number: `A_lat = C dt_f/(rho
  eps)` for a physical drag `C` in kg/(m^3 s), so holding it FIXED while the
  fluid sub-cycled multiplied the real drag by N. What is invariant is the Darcy
  ratio `A_lat/nu_lat` — both carry `dt_f` — and that ratio is what sets how much
  the mush leaks, so the error changed the answer without failing. Measured: the
  pool drifted 3.2 % in width and 3.1 % in depth over `-fsub 1..8`, and 0.17 %
  and 0.14 % once `A_lat` was derived from `C`. `mushy_sink.cpp`'s banner had
  already said it — "the C = 1e6..1e8 the AM literature quotes are SI values;
  what has to be checked is the A they map to in lattice units" — and the case
  that used it stored the lattice end and never named the SI one.
  **THE DIAGNOSIS WAS AVAILABLE WITHOUT RUNNING ANYTHING, AND WAS MISSED:
  `-fsub` changes only `dt_f`, so a SPATIAL error at fixed `dx` cannot drift
  with it.** An earlier version of this entry's companion in `melt_pool.cpp`
  blamed the first-order Marangoni surface stencil, which is a `dx` error and
  was therefore excluded a priori. When a sweep in a parameter that should not
  touch the physics moves the answer, the fault is a constant that secretly
  carries that parameter — audit every lattice quantity for the factor before
  reaching for a discretisation story.
  **AND THE SAME SLIP HID A SECOND, LOUDER FAILURE.** With `A_lat` derived,
  SS316L at dx = 8 asks for 3.855, four times the stability bound, so its
  `-fsub 1` run was diverging from the SINK as well as from Ma — two
  `dt_f`-limited terms behind one symptom. A driver that derives its lattice
  constants can check them against their own bounds before starting; one that
  stores them cannot.
- **A DEFAULT THAT IS APPLIED AFTER THE ARGUMENT LOOP IS NOT A DEFAULT, IT IS AN
  OVERRIDE.** `melt_pool -steel` selects SS316L and, with it, a viscosity, a
  `d(gamma)/dT` and an evaporation set. Applied after parsing, it silently
  discarded an explicit `-mu` or `-dgdT` on the same command line — the run
  printed the alloy's value and the user's flag did nothing. It is now a first
  pass over `argv` that sets the defaults, with the ordinary loop second, so a
  flag wins over the material it follows or precedes. Any flag that changes
  several others needs this shape.

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
  `MultiphaseCentralMoments` (2026-09-04) by unrolling, and in `ColourGradient`
  (2026-09-19) — which needed a **different** fix, because its two live arrays
  were a whole equilibrium and perturbation moment set built as POPULATIONS and
  transformed, so there was nothing for unrolling to reach. The closed-form
  central moments that `GPU/include/lbm/colour.cuh` already carried were ported
  instead, and moving that operator to Eq. (D5) the same day shortened them
  further — the total-order table and the second separable product exist only to
  carry Saito's truncation residuals, which the complete Hermite set does not
  have. Host frame 1200 → **480** (FP64) and 624 → **400** (FP32), the surviving
  loop gone, and the six remaining `regidx` are the six genuine
  grad-phi/grad-rho field loads rather than demoted arrays. `tests/test_colour_gradient.cpp`
  block 6 keeps the population path and asserts the two agree to 4.2e-16 over 60
  states. Note that `static_assert` guards the `constexpr` half of this but
  cannot guard the loop half — only the script can.

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
- **THERE ARE NOW TWO SPECULAR FLUID WALLS, ON DIFFERENT PLANES, AND PICKING
  THE WRONG ONE IS SILENT.** `SpecWall` (`set_specular_walls`) is a **ghost
  cell**: its mirror plane is half a cell outside the last fluid node, exact to
  5.5e-14 against the channel it mirrors (`validation/specular.cpp`, plane
  measured at 16.5000), and it neither collides nor reports a state. `SpecNode`
  (`set_specular_nodes`, added 2026-09-06) puts the plane **on the node**: it
  mirrors the UNKNOWN directions and then collides, so it is a real fluid node
  reporting a real rho and u. It is the same construction as the scalar's
  `ScalarSpecular`, and it was written because the on-node scalar families
  (`ScalarMoment`, `ScalarSpecular`) had no fluid partner on their own plane.
  Use `SpecNode` where the walls of the case are on-node; use `SpecWall` where
  the no-slip family is halfway, which is the tree's default.
  `validation/specular_node.cpp` asserts the on-node one as an IDENTITY against
  the box it mirrors -- 1.9e-15 for the half box, 3.7e-15 for the quarter box --
  and prints the control that makes the choice matter: the SAME box closed by
  the halfway mirror is out by **16 % of the field**. They are exact about
  different planes, not interchangeable.
  **ONLY THE ON-NODE ONE TAKES AN EDGE.** `SpecWall` refuses two mirrors at a
  corner and can afford to, because a ghost outside the domain is never read
  diagonally; an on-node wall node in the corner of a closed box is a real fluid
  node and there is nowhere else to put it. So `SpecNode` takes a face BITMASK
  (`SpecXm | SpecZm`), not a `NormalCode` -- a NormalCode names one outward
  direction, and reusing it would collide, `NrmYp == 3` reading as
  `SpecXp|SpecXm`. Axis-aligned faces only in both; a mask carrying BOTH faces
  of one axis is rejected at setup rather than resolved. `GPU/` has the same
  wall (`specular.cuh`, `set_specular_nodes`), asserted the same way in
  `test/host_physics.cpp`.
  **MEASURED 2026-09-06, and it reduces the gap by about 3x without closing
  it.** `ehd_electroconvection -fsnode` is the pairing with every plane on a
  node. Against the boundary-free doubled box at T = 190: the halfway pairing
  is +3.96 / +3.17 / +1.55 % at ny = 41 / 81 / 163, the on-node one is
  (branch-trapped) / -0.54 / -0.56 %. Read ny = 163, where the on-node and
  doubled runs both converged and the halfway one was still creeping: 2.8x
  closer. The halfway gap falls at roughly first order; the on-node one is flat
  over the single refinement available, and two points are not an order.
  **AT ny = 41 THE ON-NODE BOX LANDS ON THE WRONG BRANCH** -- m = 2 at
  `u_max/u0 = 1.34` where the other two give m = 1 at 3.6-3.7, for seeds of
  1e-2, 1e-3 and 1e-4 alike. So it is NOT the seed kick recorded elsewhere in
  this file; it is a coarse-grid trap that ny = 81 escapes. A lateral boundary
  can select the wrong branch of a subcritical bifurcation without failing,
  without diverging, and while converging cleanly.
- **THE UPWIND BOUNDARY COLUMNS OF `demonstrator/urban` REPORT A VALUE THE
  INTERIOR CANNOT HAVE GIVEN THEM, AND THE MECHANISM IS NOT DIAGNOSED.**
  Measured 2026-09-19 on Manchester, 400x400x60 at 5 m, bearing 250 (so the
  wind blows ENE and `i=0` is UPWIND), `-diff 20`, steady at t = 300 s with
  dM/dt = 1.2e-10. At street level `i=0` and `i=1` hold 9.3904e-04, **exactly
  equal to each other**, against a plume peak of 3.4996e-02 -- 2.68 % -- while
  `i=2`, the first cell inside them, is **exactly 0.0 at every frame** and
  `i=3..50` are ~1e-37. So the value did not arrive through the interior. It
  co-evolves with the DOWNWIND face (west/east: 8.63e-06/8.66e-06 at frame 4,
  9.3903e-04/9.6784e-04 at frame 12, saturating together) but the two peak at
  DIFFERENT `j`, 265 against 277, so it is not a plain x-wrap -- and the domain
  is non-periodic in x anyway (`urban.cpp:300`). The run also prints
  `126 of 205774 outflow node(s) have no bulk neighbour and are inert`. It is
  0.77 % of the slice total in 2 columns of 400, so it changes no budget, but
  it lands in every plume figure. Suspect the same class as the two entries
  above about prescribed nodes and outflow donors; **do not assume it is the
  same bug without measuring**.
- **The free surface has no surface tension** and **no gas dynamics** — an
  enclosed bubble does not compress. **THE REASON FOR THE FIRST HAS NARROWED,
  SO DO NOT READ IT AS "THE GAS PRESSURE IS UNIFORM" ANY MORE.** `rho_G_of`
  (added 2026-09-22) is an optional per-node gas density beside the scalar
  `rho_G`, following `ScalarBGK`'s `omega_of` idiom — empty means uniform and
  nothing behaves differently. The free-surface condition ALREADY imposes
  "normal stress = gas pressure"; it was only the uniformity of `rho_G` that
  made that one atmosphere. So `p_G = p_atm − σκ` is now *expressible*.
  **What is still missing is κ**: nothing computes a curvature from the fill
  level, and that is the hard half, so surface tension is still absent and a
  caller wanting it must supply the curvature itself. What the field does
  deliver alone is a prescribed NON-UNIFORM normal stress, which is a laser's
  recoil pressure — `validation/recoil.cpp` measures it against the exact
  hydrostatic depression `dz = −[p_G(x) − ⟨p_G⟩]/(ρg)`, converging at order
  1.84 to 1.4 % at Lx = 256.
  **SURFACE TENSION NOW EXISTS TOO, so the sentence above is the history rather
  than the state.** `set_surface_tension(σ)` (2026-09-22) computes
  `κ = −∇·(∇ε/|∇ε|)` on the GRADIENT lattice and adds `σκ/cs²` to whatever gas
  density is imposed, so recoil and capillarity superpose in one field — which
  is what a keyhole needs, being a hole held open by recoil and closed by
  surface tension. `validation/surface_tension.cpp` measures Laplace's law:
  **1.2 % for R ≥ 12 and 3.4 % at R = 8**, with spurious currents of 1e-8 to
  5e-6. Zero storage and zero kernels when σ = 0, which is the default.
  **NO ORDER IS QUOTED AND THAT IS DELIBERATE.** The error is +3.39, +0.88,
  −0.64, +1.14 % at R = 8/12/16/24 — NOT monotone, because curvature on a
  volume-of-fluid field depends on how the circle happens to sit on the grid, so
  two radii can land on opposite sides of the answer. A rate fitted through
  those points would be arithmetic, not a measurement. Quote the band.
  Two guards make the case non-vacuous, and the second caught something: σ = 0
  gives *exactly* zero pressure jump, isolating the term from the rest of the
  solver; and an ellipse must relax to a circle, which tests the SIGN that no
  static measurement can see. That second test is worthless unless the seed is
  really an ellipse, so its aspect is measured at t = 0 — and that guard read
  1.6863 for a requested 1.30, which is 1.30², a naming error in the seeding
  parameter rather than the bug it was written to catch.
  **THE ERROR IS SET BY THE DEPRESSION MEASURED IN CELLS, NOT BY THE GRID.**
  36 % at one cell, 13 % at two, 5.0 % at four, 1.4 % at eight — and the control
  is that a two-cell depression reads 13.06 % at Lx = 128 and 13.16 % at
  Lx = 64, the same error on grids a factor of two apart. The free-surface
  condition imposes the gas pressure at the CELL, not at the sub-cell position
  of the surface within it, so the error is a fixed fraction of a cell rather
  than of the answer. Size a free-surface deformation in cells before believing
  it.
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
  `NrmOutFree`), raw MRT, and a moving obstacle in the free surface — the
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
- **RAISING THE VISCOSITY TO REACH A STEADY STATE FASTER IS BACKWARDS ONCE THE
  SYSTEM IS OVERDAMPED, AND THE RESULT LOOKS LIKE A BROKEN SCHEME RATHER THAN A
  SHORT RUN.** A relaxing free surface is a damped oscillator: rate `γ = 2νk²`
  while underdamped, but `ω₀²/γ` once `γ > ω₀`, which FALLS as ν rises. Writing
  `validation/recoil.cpp` I raised ν from 1/6 to 0.5 to "damp the transient
  faster"; it went overdamped, every grid stalled at 69 % of the exact answer,
  and the convergence order fell from 1.84 to **0.03** — an unconverged run
  reading exactly like a scheme with no spatial convergence. The safe rate is
  `min(γ, ω₀²/γ)`, always. And the fastest approach is not the most accurate
  one: critical damping (`2νk² = ω₀`) settles quickest and was measurably WORSE
  (order 1.59, and 7.1 % against 2.7 % on a localised bump) with more relaxation
  times elapsed, because ω = 1 is where an LBM boundary sits where it claims to.
  Spend a free parameter on accuracy, not on wall clock.
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
