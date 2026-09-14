# Demonstrator cases

A **validation** case has something to be right against and fails if it misses
it. A **demonstrator** has none of that: it shows the solver running on a
problem the rest of the suite cannot express. Nothing here is evidence of
accuracy, and nothing here is registered with `add_test`.

## urban

Passive-scalar dispersion through a voxelised city, D3Q7 advection-diffusion on
a prescribed logarithmic wind. Written up as the urban section of
`doc/m3lb.tex`. Two cities have been run, from the same OSM-derived height
fields: Manchester city centre (400x400x60 at 5 m, 1.79% solid) and Midtown
Manhattan (400x400x100 at 5 m, 6.87% solid, towers to 472 m).

| file | what it is |
|---|---|
| `urban.cpp` | the case: height-field ingest, log-law wind, physical-units scaling, source injection, div(u) and stability diagnostics, legacy-VTK output |
| `vol_urban.cpp` | volume renderer — nearest-neighbour buildings, three translucent concentration shells, front-to-back compositing onto a light ground |
| `../doc/fig/make_urban_anim.sh` | the collage pipeline: three cameras per frame, SVG overlay, ImageMagick, ffmpeg |
| `../doc/fig/plume_stats.py` | measures the plume — crosswind spread below and above the rooflines, and the shell levels by enclosed mass |
| `../doc/fig/cams/*.env` | camera presets, one per domain shape |

### Running

```
build/demonstrator/urban -geom <prefix> -out <dir> -diff 35 -bearing 208 \
  -src 100 29 1 -minutes 15 -out-every 18 --kokkos-num-threads=4
```

`<prefix>` names `<prefix>_heights.npy` and `<prefix>_meta.json`. `-bearing` is
meteorological, the direction the wind comes FROM, so 270 is a westerly blowing
toward +x. `-diffusion-only` drops the wind entirely and takes its time step
from the relaxation rate instead. `-top open` holds the top at C = 0 rather
than treating it as a lid. `-mode solved` solves the wind with D3Q27 TRT instead
of prescribing it, and is **interior-only**: the lateral boundaries jet.

**Pass `--kokkos-num-threads`.** This build uses the Threads backend, which
defaults to one thread and ignores `OMP_NUM_THREADS`. Four is the whole gain on
an M1 — D3Q7 is memory bound and the efficiency cores add nothing. 84 MLUPS
serial, 173 at four threads, on the Manhattan grid.

**Read the stability margin before letting a run go long.** The prescribed wind
is not divergence-free, which puts a spurious `-C div(u)` source in the
transport term, and below about 8x the damping it wins. The failure does not
look like a failure: mass tracks injection while the undershoot grows
geometrically. The `min C` column is the early warning, and the run now stops
itself when the undershoot passes half the peak. A larger `-diff` is the fix; a
shorter time step is not, and neither is `-u-lat`.

**`retained` is a real conservation statement now** — it sums every population
in the lattice, not the macroscopic field over fluid cells. The `in fluid`
column beside it is how much of that total the plotted field sees; the rest is
in slots owned by walls, in flight for one step. Over Manhattan that is 9%. See
`validation/scalar_mass.cpp`.

### Rendering

```
CAMS=doc/fig/cams/manhattan.env doc/fig/make_urban_anim.sh \
  <vtk_dir> <run.log> <work_dir> out.mp4 "$(python3 doc/fig/plume_stats.py \
  <vtk_dir>/conc_0049.vtk --log <run.log> --levels)"
```

The run log is an input, not a convenience: frame times, the city, the grid, the
wind and the fetch are all read out of it, so a caption cannot disagree with the
run it is captioning. Shell levels are measured from the last frame — the same
release spread over 2 km carries an order of magnitude less than it does over
200 m, and levels carried over from another run mean nothing.

## aorta

Flow through a voxelised patient-specific aorta (SimVascular case
`0074_H_AO_H`), D3Q27 central moments. Written up as §12 of `doc/m3lb.tex`.

| file | what it is |
|---|---|
| `aorta.cpp` | the case: geometry ingest, boundary conditions, conserving-outflow controller, steady and pulsatile drive |
| `vol_aorta.cpp` | volume renderer for the dumps — translucent vessel shell, speed-coloured interior, streamlines, cardiac-phase inset |
| `make_aorta_anim.py` | driver: shared colour scale across frames, then ffmpeg |

Two things it depends on that deliberately live elsewhere:

- **`src/io/VoxelGeometry.hpp`** is the voxel-geometry reader. It is a *solver
  capability* — `set_geometry` takes an arbitrary predicate, so any voxelised
  geometry can drive the solver — and the aorta is one use of it, not its owner.
  It stays in `src/` for the same reason the lattices do.
- **`validation/FieldDump.hpp`** is the diagnostic dump helper, shared with five
  validation cases. It is not part of the solver.

Generated figures land in `doc/fig/` with every other figure in the document,
since that is where `\graphicspath` points.

### Running

```
build/demonstrator/aorta -re 50 -u 0.02 -steps 13500
build/demonstrator/aorta -re 50 -u 0.02 -pulse -period 2000 -ramp 800 -steps 12800
```

`FIGVOL=1` dumps speed volumes; `FIGVEC=1` dumps the three velocity components,
which is what streamline integration needs. `-dumpfrom N` restricts dumping to a
window at the end of a run — vector dumps are 3x the size, and a long run needs
to converge but only its last beats need rendering.

```
python3 demonstrator/make_aorta_anim.py <dump_dir> out.mp4 15 --period 2000 --probe 100
```

## rayleigh_taylor

Heavy fluid over light in a box W x 4W, periodic in x, no-slip top and bottom,
with the interface given a single-mode perturbation of amplitude 0.1 W. The
multiphase module doing what it was built for: an interface that rolls up,
reconnects and keeps going. Uses the pressure-based operator, which is the only
one here that reaches a density ratio at all.

| file | what it is |
|---|---|
| `rayleigh_taylor.cpp` | the case: hydrostatic initialisation, front tracking, raw field dumps |
| `render_rt.cpp` | frame renderer — phase field and vorticity, three palettes |

### Running

```
build/demonstrator/rayleigh_taylor -w 192 -at 0.1 -re 30000 -op cm \
  -nframes 180 -dump <dir> --kokkos-num-threads=4
build/demonstrator/render_rt -in <dir> -out <frames> -n 181 -pal aurora -up 2 -crop 350,860
ffmpeg -framerate 24 -i <frames>/rt_%04d.ppm -c:v libx264 -pix_fmt yuv420p out.mp4
```

`-at` is the Atwood number, so `rho_H/rho_L = (1+At)/(1-At)`; `-re` is built on
the free-fall velocity `sqrt(gW)`; time is reported as `t* = t / sqrt(W/(g At))`
so runs at the same At compare regardless of the lattice numbers underneath.

**THE RENDERER IS SEPARATE ON PURPOSE**, for the reason `vol_aorta` and
`vol_urban` are: welded to the simulation, every change of colour map costs a
full re-run — eight and a half minutes at W = 192 to alter a hue. Fields go out
raw in `FieldDump.hpp`'s format and `render_rt` turns 181 frames into pictures in
2.6 seconds. Palettes are `aurora` (default), `paper` (light ground, for slides
on white) and `neon`.

**The vorticity scale calibrates on the LAST frame, not the first.** The first
frame of this case is a fluid at rest, so its 99.9th-percentile vorticity is
round-off — 1.4e-07 against the 6.2e-03 the developed flow reaches — and
calibrating there puts the whole film at full saturation. A percentile rather
than a maximum, so one hot cell cannot flatten everything else.

**Use `-op cm` at any serious Reynolds number.** BGK relaxes every mode at
omega, and at Re = 30000 omega is 1.99795; it diverges at `t* = 1.5` where the
central-moment operator completes `t* = 3`.

### The three clips in `anim/`

| file | what it shows |
|---|---|
| `rayleigh_taylor.mp4` | At = 0.5, Re = 256 — the reference case, sound |
| `rayleigh_taylor_re30000.mp4` | At = 0.1, Re = 30000 — the high-Re case, sound |
| `rayleigh_taylor_at998.mp4` | At = 0.998 under **BGK**, and a KNOWN ARTEFACT |

The third is kept as evidence, not as a showcase. Its phase field is smooth and
physically sensible but its velocity field is dominated by a one-cell
alternating mode, measured 70x stronger than the same render at a density ratio
of 3 — grid-scale oscillation that BGK does not damp, which is the measurement
that motivated the central-moment operator. Do not show it as a result.

## water_entry

A square dropped into a free water surface: approach, impact, cavity, splash-up
jets. The multiphase module carrying a moving rigid body, coupled by volume
penalisation rather than an immersed boundary — see `PenalisedBody.hpp` for why.

| file | what it is |
|---|---|
| `water_entry.cpp` | the case: diffuse free surface, hydrostatic seed through the interface, a falling body with its reaction fed back into Newton |

### Running

```
build/demonstrator/water_entry -l 48 -ratio 50 -rhob 2 -theta 0 \
  -tmax 6 -nframes 150 -dump <dir> --kokkos-num-threads=4
build/demonstrator/render_rt -in <dir> -out <frames> -n 151 -pal aurora
```

`-rhob` is the body's density as a multiple of the water's, `-drop` its release
height in body widths, and `-theta` its release tilt in degrees. `-l` sets the
side of the square and everything scales off it.

**IT FALLS RATHER THAN BEING PUSHED.** De Rosis & Enan run this problem with a
prescribed constant entry velocity; here the reaction closes Newton's equations,
so the deceleration on impact is a result. `-rhob` below 1 floats: the square
enters, stops, reverses and bobs — measured at `-rhob 0.6`, reversing at
`t U/L = 5.0` and rising through 0.56 L before falling back. That case is the
one the classical Uhlmann fictitious-mass correction cannot express at all,
since its denominator `m_b - m_f` changes sign there.

**`-theta` makes it roll.** An off-axis square strikes one corner first and
slaps flat, which is not available to a translation-only body. The rigid-body
solve is a coupled 3x3 in sway, heave and roll; the hydrostatics it rests on is
checked against Archimedes and metacentric theory in
`validation/floating_body.cpp`, which is where the numbers are.

**There is no exact answer here**, which is why this is a demonstrator. Wagner's
slamming theory covers a wedge; a flat-bottomed square has a singular impact
pressure and no closed form. There is also no contact-line model, so read the
splash, not the meniscus.

### The three clips in `anim/`

| file | what it shows |
|---|---|
| `water_entry.mp4` | `-rhob 2`, upright, ratio 50 — the reference entry |
| `water_entry_ratio800.mp4` | the same at a density ratio of 800 |
| `water_entry_floating.mp4` | `-rhob 0.6 -theta 25` — enters on a corner, slaps flat, sinks past its draft and comes back up upright |

The third is the one the previous formulation could not produce at all. It is
also the clearest picture of what the roll actually does: the square strikes one
corner, sheds a vortex pair of opposite signs from the two corners, and the
imbalance between them is what turns it.

## mhd_cylinder

Flow past a circular cylinder in a **transverse magnetic field** -- a von Karman
street being switched off by a Lorentz force. Plane channel, blockage 1/8,
parabolic inlet, uniform applied field along `y` so that `u x B` points along
`z` and the induced field it drives is back in the plane. That is a closed
two-dimensional MHD system rather than a truncation of a three-dimensional one:
`j` has only a `z` component, so `div j = 0` holds identically.

The governing group is the interaction parameter `N = Ha^2 / Re`, the ratio of
Joule braking to inertia. `Re` is held fixed and `Ha` swept.

| file | what it is |
|---|---|
| `mhd_cylinder.cpp` | the case: momentum-exchange drag, wake probe, Hartmann sweep, VTK and compact frame dumps |
| `../doc/fig/mhd_anim.py` | the frame renderer -- shared or per-frame colour scale, NaN-masked bodies, optional stacked pair |

### Running

```
build/demonstrator/mhd_cylinder --kokkos-num-threads=4
build/demonstrator/mhd_cylinder -selftest      # the force check alone
build/demonstrator/mhd_cylinder -pair          # price the half-cell wall pairing
```

### The self-test is not optional

A periodic channel driven by a body force `G` must have its walls carry exactly
`G x N_fluid` in steady state. That is momentum conservation and holds whatever
the profile is -- which is also why it is **blind to the magnetic field**: all
three cases agree to the last bits. So the self-test also measures the
centreline velocity, which the field does move (the Hartmann midplane value is
39 % of the parabolic peak), because a dead coupling would reproduce the force
balance perfectly. `-pair` then prices the one approximation the geometry
forces: the cylinder needs halfway bounce-back for momentum exchange while the
magnetic condition sits on the node, and the resulting deviation is `-10.69 %`,
`-5.77 %`, `-3.00 %` at `H = 32, 64, 128` -- first order in `1/H`, which is what
a half cell predicts and what an error in the coupling would not do.

### Results

`D = 30`, `Re = 100`, `Pr_m = 0.01`, 45000 steps of transient and 45000 averaged.

| Ha | N | C_d | St | wake amplitude | max&#124;b&#124;/B0 | dmass |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 1.2937 | 0.1579 | 64.9 % | -- | +1.7e-04 |
| 2 | 0.04 | 1.1897 | 0.1369 | 19.8 % | 0.433 | +3.3e-04 |
| 4 | 0.16 | 1.3409 | -- | 0.10 % | 0.377 | +7.3e-04 |
| 6 | 0.36 | 1.6904 | -- | 0.05 % | 0.332 | +1.1e-03 |
| 8 | 0.64 | 2.2629 | -- | 0.07 % | 0.299 | +1.5e-03 |
| 10 | 1.00 | 3.0301 | -- | 0.06 % | 0.280 | +2.0e-03 |
| 15 | 2.25 | 5.5525 | -- | 0.05 % | 0.268 | +3.4e-03 |
| 20 | 4.00 | 8.7021 | -- | 0.04 % | 0.262 | +4.8e-03 |

**The drag falls before it rises.** `C_d` drops 8 % from `Ha = 0` to `Ha = 2`
and only then climbs. Two effects compete with opposite signs: suppressing the
shedding *removes* form drag, and the Lorentz force *adds* drag. The first
saturates once the wake is steady -- by `Ha = 4` the lift amplitude is 90x down
-- and the second does not.

**Above `N = 1` the increment is linear in `N`**, which is the asymptotic
scaling a Joule-braked wake should have and is the closest this case comes to a
quantitative check:

| N | 0.64 | 1.00 | 2.25 | 4.00 |
|---|---|---|---|---|
| `(C_d - C_d0) / N` | 1.51 | 1.74 | 1.89 | 1.85 |

constant to +-4 % over the top three, and visibly not yet linear below.

**It reproduces across resolution to about 1 %.** The same `Re` and `Pr_m` at
`D = 20` gives 1.3003 / 3.0605 / 8.7962 at `Ha = 0 / 10 / 20` against 1.2937 /
3.0301 / 8.7021 here -- 0.5 %, 1.0 %, 1.1 % apart over a 1.5x change in
resolution. The two runs used different averaging windows, which is immaterial
where the wake is steady and deserves a little caution at `Ha = 0`.

**What is not resolved.** `delta_Ha` is the cylinder's Hartmann layer in cells
and it is printed every row: 15 cells at `Ha = 2` but only 2.0 and 1.5 at
`Ha = 15` and `20`. The fine structure near the body in those panels of
`doc/fig/mhd_cylinder_sweep.png` is therefore not resolved and should not be
read as physics. The integrated drag is the quantity that survives it, and the
resolution check above is the evidence for that.

**The outlet's mass drift grows with the field**, 1.7e-04 to 4.8e-03 over the
sweep. `NrmOutXp` derives its normal velocity from the inverted closure, which
knows nothing about the Maxwell stress, so this is expected rather than
mysterious -- but a drag normalised by `rho = 1` is high by exactly that drift,
i.e. by 0.5 % at the top of the sweep. It is printed every row for that reason.

### The clips in `anim/`

| file | what it shows |
|---|---|
| `mhd_cylinder_suppression.mp4` | `Ha = 0` above, `Ha = 6` below, **one colour scale** -- the street switching off, and the channel's background shear flattening into a Hartmann profile |
| `mhd_cylinder_current.mp4` | `Ha = 2`, vorticity above, current below -- the mechanism rather than the outcome: current sheets riding the shed vortices, and the two wall bands where the induced current closes |

`Ha = 2` is deliberate for the second clip: it still sheds, so there is
something moving to watch. `Ha = 6` is steady and would be a still image.

### Two things that are measured rather than assumed

**The magnetic Prandtl number must be small, and this is the regime the problem
belongs to anyway** (liquid metals sit near `1e-6`). At `Pr_m = 1` the induction
equation runs away before the wake does anything, and the deletion test found it
-- with the cylinder removed entirely, leaving a plain channel, `max|b|/B0` goes
`3.73 / 7.17 / 17.69` at `Pr_m = 1` and saturates at `0.278` at `Pr_m = 0.01`.
Four earlier blow-ups had read as "the wake went unstable" with the cylinder not
involved. What the knob does not separate is `Rm` from `omega_mag`: both are
functions of `eta` alone.

**The outlet is a mass budget, not a stability choice.** `NrmOutEq` imposes
density *and* velocity, which over-determines the boundary and makes it a
source: `+20.7 %` of the total mass in fourteen thousand steps here, against
`+6.2e-05` for `NrmOutXp`. A drag normalised by `rho = 1` is then high by
exactly the drift.

## mhd_decay

Freely decaying MHD turbulence in a **confined circular domain** -- Neffaa, Bos
and Schneider, Phys. Fluids 20, 075104 (2008). No forcing, no applied field, no
through-flow: `u` and `b` are seeded from random-phase potentials banded around
`k0` and everything after that is decay. The fluid takes regularised on-node
walls with per-node normals and `NrmCorner` at the staircase corners, so the
magnetic condition sits on the same plane.

This is the only MHD case in the repository in which the magnetic boundary acts
on a field the boundary itself is shaping, rather than on a small induced
correction to an imposed one.

| file | what it is |
|---|---|
| `mhd_decay.cpp` | the case: divergence-free random IC from stream and potential functions, energy/helicity/integral-scale diagnostics, VTK and compact frame dumps |

### Running

```
build/demonstrator/mhd_decay --kokkos-num-threads=4
build/demonstrator/mhd_decay -shape square -walls hunt   # the Neumann control
```

### What it shows

Two-dimensional MHD has three ideal invariants and they do not decay at the same
rate, so the run reports integral scales and not only energies. Over 29 turnovers
at `N = 321`, `Re = 1000`: total energy falls by a factor of 6000 while the
magnetic integral scale grows from 11.0 to 60.0 cells and the kinetic one barely
moves (11.4 to 16.3). `E_kin/E_mag` falls from 1 to 0.018. Mass holds to
`1.1e-07`. That is the shape of selective decay -- a length that grows while the
energy falls is the one diagnostic that cannot be mistaken for the flow simply
dying.

**But do not quote the 5.5x.** Over the same window `div b` grows from machine
zero to parity with the field's own curl -- 0.097 of it at 1.7 turnovers, 0.354
at 4.9, 1.04 at 29 -- and no divergence cleaning is implemented. A spurious
gradient field adds to `E_mag` and adds nothing to `<j^2>`, so
`L_b = sqrt(2 E_mag / <j^2>)` is inflated once that matters, and 5.5x is an
upper bound. Inside the window where the ratio stays under 10 % the growth is
`1.26x`, and that is the number this case can defend.

What survives it is anything built from `curl b`, which is blind to a gradient
field by construction. The animation's late-time picture -- the current
organising into a single domain-scale structure where it began as banded `k = 4`
noise -- is evidence that does not depend on the divergence at all. The
qualitative result rests on that; the number does not.

The diagnostic reports **two** divergences because the obvious one has a floor.
The initial `b` is built from a potential by central differences, for which
`div b` cancels identically cell by cell, so `t = 0` must read machine zero.
Excluding only cells with a solid neighbour reads `2.57e-03` there; excluding
cells with any solid within three reads `8.20e-16`. The gap is the ring one step
inside the wall, whose stencils reach nodes the Dirichlet condition overwrote --
and the naive measure runs about **twice** the bulk one at every later point too,
so it would have overstated all of the above by a factor of two.

### The conducting wall does not work here, and that narrows `MagNeumann`

`-walls cond` -- Neumann on the whole closed boundary -- goes non-finite inside
sixty steps at the grid scale. Three explanations were tested and all three are
wrong: not the staircase (a square domain with 0.9 % of its Neumann nodes
self-referential against a disc's 14.6 % fails identically), not `omega -> 2`
(1.949 through 1.586 all fail identically), not a slow drift of the unpinned
level (far too slow for a factor 263 in twenty steps). What holds is the
control: `-walls hunt`, Neumann on one face pair with Dirichlet on the other --
Hunt's own arrangement -- decays cleanly. So the stencil is sound and the
**closure** is not: the condition needs a Dirichlet boundary somewhere on the
domain. Written up in `doc/m3lb.tex`'s known limitations.

### The clips in `anim/`

| file | what it shows |
|---|---|
| `mhd_decay_disc.mp4` | vorticity above, current below, 29 turnovers of decay at `Re = 1000` |

`--pernorm` is used for that clip and is the exception to this repository's
shared-scale rule. The amplitude falls by four decades and on a shared scale the
whole second half is blank -- which is where the result is, since selective decay
is a statement about the length scale and not the amplitude. The amplitude that
has been divided out is the table's `E_kin` and `E_mag` columns.
