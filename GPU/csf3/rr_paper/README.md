# RR-MHD paper — CSF3 campaign with the M3LB GPU code

Three-dimensional Orszag–Tang vortex at Re = 1000, 3000 and 10000 on one A100 per run,
FP64, with M3LB's `GPU/src/orszag_tang.cu`. Goal for the paper: convergence through the
current peak against a pseudo-spectral reference (Re = 1000), robustness and energy
spectra at high Re (RR runs where BGK diverges), the effect of the bulk rate, and 3-D
views of the current sheets.

## Why the M3LB GPU code is the paper's scheme

`-op cm` is the central-moment collision in the shifted Hermite basis with all orders
above two at equilibrium. By the paper's Theorem 1 it **is** recursive regularisation.
With the options below (`-wbulk omega -fullinit`), the FP64 host build of this driver and
the paper's own D3Q27 RR code agree **to round-off over a whole run**: M = 32, Re = 100,
2935 steps, at most 2.6e-13 in the kinetic energy, 1.8e-14 in the magnetic energy and
7e-14 in J_max, for `-wbulk omega` and `-wbulk 1` alike. They are two independent implementations: different
streaming (esoteric pull vs swap) and a moment-space collision vs a closed-form reconstruction.

## The driver options (commit 0a43732)

Two options in `GPU/src/orszag_tang.cu`; the defaults are unchanged (default output
identical to before, checked for `-op cm` and `-op bgk`), so existing scripts are unaffected:

* `-wbulk W|omega` — relaxation rate of the trace of the second-order moments (default 1,
  the Solver's default). `omega` = the shear rate = single-rate RR (the paper's scheme).
* `-fullinit` — start from the complete equilibrium (product form + Maxwell stress) via
  `seed_populations_with`, as `tg_mhd.cu` does. Without it the Maxwell part is missing at
  t = 0 (an acoustic start-up transient; this was the 3.4e-4 residual seen before).
* `-divu` (follow-up 2) — an eighth probe column, max|div u| / k max|u|. Opt-in because
  `results/G_orszag_tang_3d/make_ot3d_animation.py` reads the probe lines by their seven
  columns; without it the output is byte-identical to the build before it (host FP64,
  M = 32). The column agrees with the paper's CPU code to its 4 printed digits (M = 32,
  Re = 100, single rate, 21 probes).

## On CSF3

From the repo root, on an up-to-date `mhd` checkout:

```bash
ssh csf3
cd ~/scratch/M3LB && git checkout mhd && git pull   # this folder and the driver options
bash --login GPU/csf3/rr_paper/rr_paper_build.sh     # FP64, sm_80, GPU/build64_rr, a few minutes

sbatch --array=0-2,5,6,8  --time=1:00:00 GPU/csf3/rr_paper/rr_paper.sub
sbatch --array=3,4,7,9,10 --time=6:00:00 GPU/csf3/rr_paper/rr_paper.sub
sbatch GPU/csf3/rr_paper/rr_paper_spectral.sub      # CPU, runs alongside
```

The omega_b = 1 follow-up (elements 11-13, see "Outcome" below), and its post-processing
limited to the three new runs:

```bash
sbatch --array=11,13 --time=1:00:00 GPU/csf3/rr_paper/rr_paper.sub
sbatch --array=12    --time=3:00:00 GPU/csf3/rr_paper/rr_paper.sub
ONLY="re3000_m256_cm_wb1 re10000_m512_cm_wb1 re1000_m256_cm_wb1 re3000_m512_cm_wb1" sbatch GPU/csf3/rr_paper/rr_paper_post.sub
```

`re3000_m512_cm_wb1` is in that list again because `spectra_vti` now cuts the spectrum below the
Nyquist wavenumber of the sampled grid before it samples the 512³ |j| volume every second node;
the first volumes were aliased (serrated sheet edges in the 3-D views). Its frames are still on scratch.

When everything has finished:

```bash
sbatch GPU/csf3/rr_paper/rr_paper_post.sub          # spectra + |j| volumes + tarball
```

and copy back the one file it writes (about 1 GB), with `rsync -avP` so that a dropped link
resumes instead of restarting:

```bash
rsync -avP csf3:scratch/M3LB/runs/rr_paper/rr_paper_results.tar.gz ~/Downloads/
```

## The runs (`rr_paper.sub`, t ≤ 3, 60 probes, a .vti frame every 0.5 for cm)

| idx | Re | M | operator | ω_b | purpose | outcome / A100 |
|---|---|---|---|---|---|---|
| 0 | 1000 | 128 | RR (cm) | ω | convergence through the peak | t = 3, 17 s |
| 1 | 1000 | 256 | RR (cm) | ω | convergence through the peak | t = 3, 232 s |
| 2 | 3000 | 256 | RR (cm) | ω | high Re, coarse | diverged t = 0.85 |
| 3 | 3000 | 512 | RR (cm) | ω | high Re, fine | diverged t = 0.65 |
| 4 | 3000 | 512 | RR (cm) | 1 | bulk-rate effect | t = 3, 3475 s |
| 5 | 1000 | 128 | BGK | – | BGK at the same settings | diverged t = 0.70 |
| 6 | 3000 | 256 | BGK | – | BGK | diverged t = 0.35 |
| 7 | 3000 | 512 | BGK | – | BGK | diverged t = 0.25 |
| 8 | 10000 | 256 | RR (cm) | ω | if it holds up | diverged t = 0.65 |
| 9 | 10000 | 512 | RR (cm) | ω | if it holds up | diverged t = 0.35 |
| 10 | 10000 | 512 | BGK | – | BGK | diverged t = 0.15 |
| 11 | 3000 | 256 | RR (cm) | 1 | ω_b = 1, coarse grid at Re = 3000 | est. ~4 min |
| 12 | 10000 | 512 | RR (cm) | 1 | ω_b = 1: does it hold at Re = 10000? | est. ~1 h |
| 13 | 1000 | 256 | RR (cm) | 1 | ω_b = 1: accuracy cost against the spectral reference | est. ~4 min |
| 14 | 1000 | 256 | BGK | – | last cells of the paper's table of divergence times | diverged t = 0.55 |
| 15 | 10000 | 256 | BGK | – | same | diverged t = 0.30 |
| 16 | 10000 | 256 | RR (cm) | 1 | same (no frames) | diverged t = 1.10 |

Measured on gpuA: 1.08, 1.27 and 1.36 GLUPS FP64 at M = 128, 256 and 512, probes and
frames included (the last line of each `log.txt`); at most four jobs run at once under
the free-at-point-of-use limit. M = 512 FP64 uses ~60 GB of the 80 GB.
Disk: a 512³ frame is 4.3 GB, seven per cm run that reaches t = 3 (elements 4 and 12, ~30 GB
each; the diverged runs stop writing); delete the .vti once the post-processing tarball exists.

A diverged run prints `DIVERGED at t = ...` in its log; that time is the result.
If element 8 (Re = 10000, M = 256) diverges, element 9 is still worth its slot: the
cell Reynolds number at M = 512 is half as large.

`rr_paper_spectral.sub` is the pseudo-spectral reference for elements 0, 1 and 5
(N_s = 256, t ≤ 3, ~14 GB, 32 cores, a few hours). It samples at exactly the probe
times of the M = 128 and M = 256 runs and evaluates J_max with the lattice stencil on
those nodes. Its last column (tailJ) says whether it is resolved.

## Outcome of elements 0-10 (2026-10-06)

* Re = 1000: RR on M = 128 and 256 follows the N_s = 256 reference (resolved to t = 3)
  through the current peak at t ~ 2.1; max errors over 0.25 <= t <= 3 in E_k / E_m / Jmax
  0.27 / 0.12 / 9.3 % and 0.090 / 0.063 / 4.0 %. The kinetic spectrum is within 2 % up to
  k dx ~ 0.5 on both grids (10 % up to ~0.8), the magnetic one up to k dx ~ 0.75.
* BGK diverges first everywhere it was run. Single-rate RR (`-wbulk omega`) lasts more
  than twice as long but fails before the peak at Re = 3000 and 10000, EARLIER on the
  finer grid (Re = 3000: t = 1.65 / 0.85 / 0.65 on M = 64 / 256 / 512), so it is not
  under-resolution. It is local: on M = 512 the omega and 1 runs agree to 1e-4 up to
  t = 0.5, then max|div b| of the omega run jumps 36-fold in 0.05 at unchanged Jmax and the
  run blows up at the next probe.
* `-wbulk 1` (bulk viscosity 1/9 in lattice units, ~100x the shear viscosity there) runs
  to t = 3 at Re = 3000 on M = 512, through the peak (Jmax ~ 296 at t ~ 2.3).

Elements 11-13 complete the omega_b = 1 row: its coarse grid at Re = 3000, whether it holds
at Re = 10000, and what it costs in accuracy at Re = 1000, where the reference exists.

## Outcome of elements 11-13 (2026-10-08)

* Re = 1000, M = 256, `-wbulk 1`: max errors against the N_s = 256 reference 0.049 / 0.060 /
  4.0 % in E_k / E_m / Jmax, against 0.090 / 0.063 / 4.0 % for single-rate RR; the two
  differ by at most 0.1 %. The unit bulk rate costs nothing in accuracy here.
* Re = 3000, M = 256, `-wbulk 1`: runs to t = 3. Its energies agree with M = 512 to 0.24 %
  (kinetic) and 0.07 % (magnetic); the peak current is 222 against 296 on M = 512, both at
  t ~ 2.3 -- the sheets need the finer grid.
* Re = 10000, M = 512, `-wbulk 1`: diverges at t = 0.90 (single rate 0.35, BGK 0.15), again
  before the peak and again after a jump of max|div b| (sevenfold between t = 0.75 and 0.80).

## Follow-up 2 (2026-10-13): where the failure starts, and a reference at Re = 3000

The paper's CPU code (D3Q27 RR, Re = 3000, M = 128) located the start of the single-rate
failure: max|div u| doubles between t = 1.38 and 1.45 while Jmax, the vorticity and div b
stay within 0.5 % of the `-wbulk 1` run; div b follows from t = 1.45, the current from
1.64. A bulk-rate scan there (bulk viscosity 0.67, 20, 42, 125, 375 times the shear one)
diverges for the first three, survives with a compressive burst at 125 and is clean at 375
(`-wbulk 1`). So the jump of div b seen here is the second symptom, not the first.

* `rr_paper_divu.sub` (array 0-3, ~2 h each at most): single rate and `-wbulk 1` at
  Re = 3000 on M = 512 and 256, with the driver's new `-divu` column and a probe every
  ~0.005. It tests the CPU finding where the failure comes earliest. Elements 4-5 (~10 min
  each) halve the lattice velocity on M = 256 (`-ma 0.017`): omega as at M = 128, field and
  flow halved in lattice units. The linear analysis of uniform states predicts that the
  single rate (4) still fails (threshold 0.0041 with the mean flow, peak field 0.0127), but
  the CPU code at M = 128 with v0/2 ran to t = 2 without the compressive departure, against
  the same prediction; 4 tests this on the finer grid, and `-wbulk 1` (5) should run.
* `rr_paper_spectral3000.sub`: the N_s = 512 pseudo-spectral reference at Re = 3000, at the
  probe times of elements 4 and 11. Check tailJ before trusting it; ~110 GB, about a day
  on 64 cores.

```
    sbatch --array=0-3 GPU/csf3/rr_paper/rr_paper_divu.sub
    sbatch --array=4-5 GPU/csf3/rr_paper/rr_paper_divu.sub
    sbatch GPU/csf3/rr_paper/rr_paper_spectral3000.sub
```

Rebuild first (`rr_paper_build.sh`): the job refuses a binary older than orszag_tang.cu.

## Files

* `rr_paper_build.sh`, `rr_paper.sub`, `rr_paper_spectral.sub`, `rr_paper_post.sub`
* `rr_paper_divu.sub`, `rr_paper_spectral3000.sub` (follow-up 2)
* `tools/spectral3d_ref.cpp` — the pseudo-spectral solver (probe-time sampling, spectra)
* `tools/spectra_vti.cpp`, `tools/fft3.hpp` — spectra and |j| volumes from the .vti frames


## Follow-up 3 (2026-11-11): the last cells of the table of divergence times

The paper's Table 2 (BGK, RR, RR with omega_b = 1 on M = 64-512 at Re = 1000, 3000, 10000)
lacks three cases on M = 256, which do not fit the workstation that ran M <= 128: BGK at
Re = 1000 and at Re = 10000, and `-wbulk 1` at Re = 10000. They are elements 14-16, a few
minutes of one A100 in all, with the binary of the campaign (rebuild it with
`rr_paper_build.sh` if scratch has been purged):

```bash
ssh csf3
cd ~/scratch/M3LB && git checkout mhd && git pull
sbatch --array=14-16 --time=0:30:00 GPU/csf3/rr_paper/rr_paper.sub
```

Only the three `log.txt` are needed (a diverged run prints `DIVERGED at t = ...`, a run
that reaches t = 3 ends with its GLUPS line); no post-processing. Copy each back into a
folder named after its tag, next to the results of the earlier elements:

```bash
for tag in re1000_m256_bgk re10000_m256_bgk re10000_m256_cm_wb1; do
  mkdir -p $tag && rsync -avP csf3:scratch/M3LB/runs/rr_paper/$tag/log.txt $tag/
done
```

## Follow-up 4 (2026-11-13): Re = 10000 at half the lattice velocity

Outcome of follow-ups 2 and 3, now in the paper: the N_s = 512 reference at Re = 3000 is resolved
by the tail criterion and `-wbulk 1` follows it (energies within 0.05 % and the peak current
within 1 % on M = 512); on M = 256 and 512 the single rate departs through max|div u| first, as
on M = 128; and at half the lattice velocity on M = 256 the single rate still diverges
(t = 1.02) while `-wbulk 1` runs to t = 2. At Re = 10000 and the standard velocity every scheme
fails before the peak (`-wbulk 1`: t = 1.10 on M = 256, 0.90 on M = 512).

The open question is whether `-wbulk 1` at half the lattice velocity carries Re = 10000 through
the peak. Elements 6-9 of `rr_paper_divu.sub` (same binary, `-divu`, no frames):

```bash
ssh csf3
cd ~/scratch/M3LB && git checkout mhd && git pull
sbatch --array=6,7 --time=4:00:00 GPU/csf3/rr_paper/rr_paper_divu.sub   # M = 512: -wbulk 1 and the single rate
sbatch --array=8,9 --time=0:30:00 GPU/csf3/rr_paper/rr_paper_divu.sub   # M = 256
```

Element 6 is the run that answers the question (about 2.5 h on one A100 if it reaches t = 3);
7-9 make it interpretable (the single rate under the same conditions, and the coarser grid).
Only the `log.txt` of `runs/rr_paper/divu_re10000_m{512,256}_cm_wb{1,omega}_ma0.017` is needed.

## Outcome of follow-ups 3 and 4 (2026-11-14)

* Elements 14-16 of `rr_paper.sub` (M = 256): BGK diverges at t = 0.55 at Re = 1000 and at 0.30
  at Re = 10000; `-wbulk 1` diverges at t = 1.10 at Re = 10000. The paper's table of divergence
  times has no case left that was not run.
* Elements 6-9 of `rr_paper_divu.sub`, Re = 10000 at half the lattice velocity (Ma = 0.017):
  every run diverges, later than at the standard velocity but still before the current peak.

  | M | operator | Ma = 0.034 | Ma = 0.017 |
  |---|---|---|---|
  | 256 | single rate | 0.65 | 0.76 |
  | 512 | single rate | 0.35 | 0.40 |
  | 256 | `-wbulk 1` | 1.10 | 1.38 |
  | 512 | `-wbulk 1` | 0.90 | 1.14 |

  So a lower lattice velocity does not carry Re = 10000 through the peak. The linear analysis of
  uniform states calls the compressive mode of `-wbulk 1` stable in all four of its cases
  (threshold 0.037-0.039 in lattice units, against a peak field of 0.025 at Ma = 0.034 and 0.013
  at Ma = 0.017): these failures are outside what that analysis accounts for.
