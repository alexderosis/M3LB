# Disruption of a single vortex by a weak magnetic field

Series behind the numbers in the banner of `validation/vortex_disruption.cpp`.
All runs: laptop, Kokkos Threads, FP64, D2Q9 + D2Q5, eps = 0.05, Pm = 1, box
20 R0, time in R0/eps. Made on 2026-11-19. Each directory holds the run's
`series.dat` and its log; field frames were not kept.

Columns of `series.dat`: `t EK B2 bmax coh surv ang wmax jmax Lb divb osc Ma
mass ens dow`, defined in the case's banner. `surv` is the projection of the
velocity on the initial vortex over its exact viscous decay `(1 + t/Re)^-2`;
`dow` is the Okubo-Weiss Delta against the exact vortex; `ens` the enstrophy
over the exact vortex's. **`ens` is not grid-converged and is not to be quoted.**
The series in `author_case`, `grid_check` and `controls` predate the last two
columns.

## map_r020

Fifty-one runs, R0 = 20, each to t = 12 Rm^(1/3). Directory `rm<Rm>_x<M^2 Rm>`.

- The ladder: Rm = 250, 500, 1000, 2000, 4000, 8000 by M^2 Rm = 0.3125, 0.625,
  1.25, 2.5, 5, 10 (36 runs).
- Its refinement at the destruction threshold: M^2 Rm = 1.4865, 1.7678, 2.1022
  at Rm = 250 .. 4000 (15 runs), so the fields between 1.25 and 2.5 are a factor
  1.09 apart in M.

With the refinement, the field giving Delta = 0.8 at any time in the run is
M_c = 0.0859, 0.0673, 0.0414, 0.0297, 0.0223 at Rm = 250 .. 4000:
M_c^2 Rm = 1.84, 2.26, 1.72, 1.77, 2.00 and M_c ~ Rm^-0.506, which the spacing
of the ladder allows to move by 0.038. The exponent depends on the level: -0.40
at Delta = 0.5, -0.45 at 0.7, -0.53 at 0.9. Three did not finish: `rm8000_x2.5`, `rm8000_x5` and
`rm8000_x10` went non-finite at t = 64.5, 68.5 and 44, and their series end on
the driver's last, non-finite row.

```bash
python3 tools/vortex_disruption_map.py results/Q_vortex_disruption/map_r020 \
    --obs dow --reading max -o map.png
```

prints every run at five reading times, the fits, and draws the figure.
`--obs` is `surv`, `dow` or `ens`; `--reading` is `t50`, `c3`, `c6`, `c12` or
`max`. The three figures here (the first was redrawn with the refined ladder):

| file | observable | reading |
|---|---|---|
| `map2_surv_c3.png` | 1 - surv | t = 3 Rm^(1/3) |
| `map2_dow_c3.png` | Delta | t = 3 Rm^(1/3) |
| `map2_dow_max.png` | Delta | largest in the run |

## pm_fixRe1000 and pm_fixRm1000

Re and Rm moved one at a time, through the map's point Re = Rm = 1000:
`pm_fixRe1000` holds Re = 1000 and varies Rm, `pm_fixRm1000` holds Rm = 1000 and
varies Re, each over 250, 500, 1000, 2000, 4000 (Pm = 0.25 to 4). Six fields at
each point, M^2 Rm = 0.625, 1.25, 1.4865, 1.7678, 2.1022, 2.5; R0 = 20; to
t = 12 Rm^(1/3). Directory `re<Re>_rm<Rm>_x<M^2 Rm>`. The six runs at
Re = Rm = 1000 are the map's own, copied into both.

```bash
python3 tools/vortex_disruption_map.py results/Q_vortex_disruption/pm_fixRe1000 --obs dow --by Rm
python3 tools/vortex_disruption_map.py results/Q_vortex_disruption/pm_fixRm1000 --obs dow --by Re
```

| | Re = 1000, Rm varied | Rm = 1000, Re varied |
|---|---|---|
| early Delta ~ X^q (0.625 row) | q = -0.318 | q = -0.090 |
| M_c at Delta = 0.8 ~ X^p | p = -0.467 | p = -0.069 |
| M_c at Delta = 0.9 ~ X^p | p = -0.519 | p = -0.013 (Re >= 1000) |

Figures: `slice_fixRe_dow_c3.png`, `slice_fixRe_dow_max.png`,
`slice_fixRm_dow_max.png`.

## grid_check

Four of the map's runs repeated at R0 = 30, to t = 3 Rm^(1/3) only:
`r030_rm2000_x1.25`, `r030_rm4000_x1.25`, `r030_rm8000_x0.3125`,
`r030_rm8000_x1.25`. 1 - surv at that time differs from the map's by -0.08,
-0.08, +0.31 and +0.75 %.

## grid_check_gradient

M^2 Rm = 1.25 at Rm = 4000 (to t = 6 Rm^(1/3)) and Rm = 8000 (to 3), each at
R0 = 20 and R0 = 30, with the gradient measures in the series. Between the two
grids Delta differs by at most 1.0 %, 1 - surv by at most 0.7 %, and 1 - ens by
15 to 130 %.

## threshold_r030

The two rows of the COARSE ladder that bracket the destruction threshold,
M^2 Rm = 1.25 and 2.5, on the finer grid R0 = 30, at Rm = 250, 500, 1000, 2000,
4000, each to t = 12 Rm^(1/3). The field giving Delta = 0.8 at any time in the
run, interpolated in log M between those two rows alone (so these differ from
the refined values above, and the spacing allows the exponent 0.15 either way):

| Rm | 250 | 500 | 1000 | 2000 | 4000 | M_c ~ Rm^p |
|---|---|---|---|---|---|---|
| R0 = 20 (map): M_c^2 Rm | 1.73 | 2.09 | 1.86 | 1.61 | 2.00 | p = -0.498 |
| R0 = 30: M_c^2 Rm | 1.74 | 2.12 | 1.98 | 1.56 | 1.94 | p = -0.506 |

Single runs differ between the grids by under 1 % in Delta at Rm = 250 and 500
and by up to 7 % at Rm = 1000 and 2000.

## author_case

The author's setup, Re = Rm = 500, M = 0.05, on four grids, plus two others.

| directory | what |
|---|---|
| `r015`, `r020`, `r030`, `r045` | R0 = 15, 20, 30, 45 cells: surv(50) = 0.71193, 0.71181, 0.71170, 0.71165 |
| `kinematic_r030` | no Lorentz force, R0 = 30: the exact-decay check (0.9990) |
| `rm2000_r030` | Re = Rm = 2000, M = 0.05, R0 = 30: surv(50) = 0.471, one grid |

These six were written by the build that still carried the flux-function
formulation, so their headers end in `form vector ...`. The code that formulation
shared with the present case is unchanged: the present build reproduces `r015`
row for row.

## controls

All at Re = Rm = 2000, R0 = 20, to t = 151.2.

| directory | what | surv at t = 151 |
|---|---|---|
| `field_kept` | M^2 Rm = 2.5, the map's run with frames | -0.05 (a tripole by t = 76-88, gone by 120) |
| `field_off_at_50` | the same, field removed at t = 50 (`-off 50`) | 0.673 (lowest 0.589 at t = 82.5) |
| `kin_ellip1e-2` | no Lorentz force, ellipticity 0.01 (`-ellip`) | 0.9968 |
| `kin_ellip1e-3` | no Lorentz force, ellipticity 0.001 | 0.9986 |
| `kin_noseed` | no Lorentz force, no seed | 0.9999 |

`frames_rm2000_x2.5.png` is the vorticity of `field_kept` at eight times.

## gpu_port

The GPU/ driver, `GPU/src/vortex_disruption.cu`. D3Q27 + D3Q7 with nz = 1. All
but `device_verify/` are its HOST builds (no CUDA on the laptop).

- `twin/`: the FP64 host build against the parent's `-lat d3q27` (Kokkos, FP64)
  at R0 = 10, box 20 R0, Re = Rm = 500, to t = 20, M = 0.05 and M = 0. Every
  column agrees to all ten printed digits at all 21 probes; Delta at t = 0 is
  round-off in both.
- `device_verify/`: the DEVICE build on a CSF3 A100 (job 24933077), the same
  two twin runs: every column within 4.6e-14 of its scale of the parent, every
  printed value but Delta at t = 0 identical; and a throughput run at
  R0 = 80, 1600^2 (1599 MLUPS FP64 without probes).
- `precision/`: R0 = 20, box 10 R0. Case `a`, Re = Rm = 4000, M^2 Rm = 1.77, to
  t = 12 Rm^(1/3): early Delta 0.19214 / 0.19213 / 0.19214 (FP64 / FP32 /
  FP32-shifted), largest Delta 0.698 / 0.580 / 0.729. Case `b`, Re = Rm = 16000,
  M^2 Rm = 1.25: non-finite in all three (R0 = 20 is too coarse there).

## gpu_campaign (INTERIM: 47 of 55 elements)

`GPU/csf3/vortex_disruption_map.sub`'s series and logs, copied back on
2026-11-23: `map/` at Rm = 2000, 4000, 8000, 16000 (elements 0-31), `grid/`
(40-49) and `scatter/` (50-54). D3Q27 + D3Q7, FP64, A100. **The Rm = 32000 rung,
elements 32-39, had not run** (resubmitted as job 25174641). Its rule is fixed in
that job's header and in `tools/vortex_disruption_campaign.py`, before any run;
over all five rungs it gives the verdict, and over four it gives only this:

```bash
python3 tools/vortex_disruption_campaign.py results/Q_vortex_disruption/gpu_campaign
```

| Rm | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|
| M_c (largest Delta = 0.8) | 0.0300 | 0.0216 | 0.0165 | 0.0103 |
| M_c^2 Rm | 1.81 | 1.87 | 2.16 | 1.71 |

p = -0.502 with U = 0.061 (the grid shift; the ladder allows 0.050, the scatter
0.004), so the rule's reading on four rungs is "M^2 Rm DECIDES DESTRUCTION". The
wind-up exponent is -0.339 / -0.335 along the rows 0.625 / 1.25; the change of
lattice moves M_c by +1.1 % at Rm = 2000 and -3.3 % at 4000 against `map_r020`,
inside the ladder's +-4.4 %. `map/re16000_rm16000_r80_x3.5355` and `_x5` went
non-finite at t = 118 and 124; both are above the threshold.

**What the rule does not see, found reading these runs.** Near the threshold
Delta is STILL RISING when the window t = 12 Rm^(1/3) closes: at Rm = 16000
every run that sets the crossing has its largest Delta in the window's last
tenth, three on its last probe (`x1.7678`: 0.716 at 0.8 t_end, 0.915 at
t_end). So the threshold is "destroyed within the window", and destruction comes
later in that window as Rm rises -- `x2.5` is destroyed by 0.55 t_end at
Rm = 8000 and reaches 0.678 by t_end at 16000. And the largest Delta is not
monotonic in M at Rm = 8000 and 16000 (0.34, 0.73, 0.58, 0.73, 1.15 along
M^2 Rm = 1.25 .. 2.5 at 8000), single runs moving by up to 0.2 under a 1e-5
perturbation: the +14 % grid shift at Rm = 16000 is the one run `x1.7678`
(0.915 here, 0.506 on R0 = 120), the next two fields shifting the other way.
A window growing faster than Rm^(1/3), or the time to destruction as the
observable, would be a new test with a rule of its own.

## What is not here

The refined ladder or the two slices on a second grid; a slice in Pm through any
point but Re = Rm = 1000; anything above Rm = 8000. The flux-function formulation's runs were not kept: the case's banner
records what they did.
