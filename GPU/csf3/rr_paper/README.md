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

| idx | Re | M | operator | ω_b | purpose | est. A100 |
|---|---|---|---|---|---|---|
| 0 | 1000 | 128 | RR (cm) | ω | convergence through the peak | ~2 min |
| 1 | 1000 | 256 | RR (cm) | ω | convergence through the peak | ~15 min |
| 2 | 3000 | 256 | RR (cm) | ω | high Re, coarse | ~15 min |
| 3 | 3000 | 512 | RR (cm) | ω | high Re, fine | ~2–3 h |
| 4 | 3000 | 512 | RR (cm) | 1 | bulk-rate effect | ~2–3 h |
| 5 | 1000 | 128 | BGK | – | BGK at the same settings | ~2 min |
| 6 | 3000 | 256 | BGK | – | expected to diverge | short |
| 7 | 3000 | 512 | BGK | – | expected to diverge | short |
| 8 | 10000 | 256 | RR (cm) | ω | if it holds up | ~15 min |
| 9 | 10000 | 512 | RR (cm) | ω | if it holds up | ~2–3 h |
| 10 | 10000 | 512 | BGK | – | expected to diverge | short |

Estimates assume ~0.7 GLUPS FP64; the last line of each `log.txt` gives the measured
rate. About 10 A100-hours in all, at most four running at once under the
free-at-point-of-use limit. M = 512 FP64 uses ~60 GB of the 80 GB.
Disk: a 512³ frame is 4.3 GB, seven per cm run, so ~90 GB of .vti on scratch for
elements 3, 4 and 9; delete the .vti once the post-processing tarball exists.

A diverged run prints `DIVERGED at t = ...` in its log; that time is the result.
If element 8 (Re = 10000, M = 256) diverges, element 9 is still worth its slot: the
cell Reynolds number at M = 512 is half as large.

`rr_paper_spectral.sub` is the pseudo-spectral reference for elements 0, 1 and 5
(N_s = 256, t ≤ 3, ~14 GB, 32 cores, a few hours). It samples at exactly the probe
times of the M = 128 and M = 256 runs and evaluates J_max with the lattice stencil on
those nodes. Its last column (tailJ) says whether it is resolved.

## Files

* `rr_paper_build.sh`, `rr_paper.sub`, `rr_paper_spectral.sub`, `rr_paper_post.sub`
* `tools/spectral3d_ref.cpp` — the pseudo-spectral solver (probe-time sampling, spectra)
* `tools/spectra_vti.cpp`, `tools/fft3.hpp` — spectra and |j| volumes from the .vti frames
