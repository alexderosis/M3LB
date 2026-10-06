# Running M3LB on CSF3

CSF3 is Manchester's Computational Shared Facility. It runs **SLURM** (it used
to be SGE — anything you find mentioning `qsub` is stale).

Host `csf3.itservices.manchester.ac.uk`, your **IT username, not your email**,
Duo 2FA (enter `1` for a push, or type a fob passcode). Off campus you need the
GlobalProtect VPN first.

---

## GPU partitions, and which one to use

| partition | GPU | per node | access | max wallclock |
|---|---|---|---|---|
| `gpuA` | A100 80 GB | 4 | **open to all** | 4 days |
| `gpuH` | H200 141 GB | 8 | **restricted — request it** | 4 days |
| `gpuH_short` | H200 141 GB | 8 | restricted | 1 day, interactive allowed |
| `gpuL` | L40S 48 GB | 4 | open to all | 4 days |
| `gpuA40GB` | A100 40 GB | 4 | very restricted | 4 days |
| `gpuV` | V100 | — | **withdrawn Oct 2025** | — |

**Use `gpuA`. Do not use `gpuL` for anything in this tree.** The L40S is Ada:
FP64 runs at 1/64 of FP32, about 1.4 TFLOPS. Every convergence test here is
FP64 and the central-moment collision is arithmetic-heavy — which is exactly
the regime that made a T4 give only 120 MLUPS FP64 at H = 1000, ALU-bound
rather than bandwidth-bound. The A100 is 1:2 FP64 at 9.7 TFLOPS, the H200
34 TFLOPS.

`gpuH` is worth requesting but do not wait for it; `gpuA` is open and enough.

---

## One-time setup

`~/.ssh/config` on your laptop, so login is one word and Duo asks once per ten
minutes rather than once per command:

```
Host csf3
    HostName csf3.itservices.manchester.ac.uk
    User <your-IT-username>
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Then, on CSF3 — build in scratch, because the field dumps are GB-scale and home
has a quota. Scratch is **not backed up** and files unused for three months can
be deleted, so copy anything that matters back to home or RDS.

**That purge is not hypothetical: it took the whole checkout on 2026-09-17.**
Nothing was lost, because the tree lives on GitHub and the results that mattered
had already been pulled down — which is the actual rule. Treat `~/scratch/M3LB`
as a build directory that happens to contain a git clone, never as the only copy
of anything. Re-setup is the block below, verbatim, and takes about four
minutes: three of them are the build.

```bash
ssh csf3
cd ~/scratch && git clone https://github.com/alexderosis/M3LB.git && cd M3LB
module load libs/cuda/12.8.1        # check `module avail cuda`

# GPU/ -- its own CMake project, nvcc only, no Kokkos
cd GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=80 -DLBM_DOUBLE=ON
cmake --build build -j4
```

`LBM_GPU_ARCH` is **80** for the A100, **90** for an H200, 89 for an L40S.

For `cmbench` you also need the *parent* Kokkos tree, which is a separate CMake
project — see the header of `cmbench.sub`.

---

## The jobs here

**Submit from the repo root.** `sbatch` copies the script, so it cannot locate
itself and the submit directory is the only handle it has. Either submit from
the root or set `M3LB_ROOT`; both scripts check and fail with the fix printed.

```bash
cd ~/scratch/M3LB
sbatch GPU/csf3/cmbench.sub                          # first: under a minute
sbatch --array=0 GPU/csf3/rb_cold.sub                # H = 498,  ~5 min
sbatch --array=1 GPU/csf3/rb_cold.sub                # H = 998,  ~45 min
sbatch --array=2 --time=12:00:00 GPU/csf3/rb_cold.sub  # H = 1998, ~6 h
sbatch GPU/csf3/ot3d_re3040.sub                      # OT 3-D, M=288, ~10 min
sbatch GPU/csf3/mhd_jet.sub                          # MHD jet, nx=512 fp32, ~12 h
sbatch GPU/csf3/tg_mhd_verify.sub                    # confined MHD: device twin check
sbatch --array=0-5,12-17 --time=0:30:00 GPU/csf3/tg_mhd_ladder.sub   # then the ladder,
sbatch --array=6-8,18-23 --time=1:30:00 GPU/csf3/tg_mhd_ladder.sub   # in three groups
sbatch --array=9-11      --time=3:00:00 GPU/csf3/tg_mhd_ladder.sub   # (see below)
sbatch --array=0-1 --time=5:00:00 GPU/csf3/tg_mhd_mach.sub           # its Mach check
sbatch --array=2-3 --time=2:30:00 GPU/csf3/tg_mhd_mach.sub
sbatch --array=0-9   --time=5:00:00 GPU/csf3/tg_mhd_round2.sub     # round 2 (see below)
sbatch --array=10-28 --time=2:00:00 GPU/csf3/tg_mhd_round2.sub
sbatch --array=0-2 GPU/csf3/tg_mhd_snap.sub                         # the paper's snapshots
sbatch GPU/csf3/tg_mhd_vol.sub                                     # and its 3-D volumes
sbatch --array=0-3  --time=4:00:00 GPU/csf3/tg_mhd_budget.sub       # Phase 1: the energy budget,
sbatch --array=4-7  --time=2:30:00 GPU/csf3/tg_mhd_budget.sub       # chained on the verify job
sbatch --array=8-15 --time=1:00:00 GPU/csf3/tg_mhd_budget.sub       # (see below)
sbatch --array=0-5   --time=5:00:00 GPU/csf3/tg_mhd_controls.sub    # Phase 2: four controls,
sbatch --array=6-12  --time=3:00:00 GPU/csf3/tg_mhd_controls.sub    # chained on the verify job
sbatch --array=13-21 --time=1:30:00 GPU/csf3/tg_mhd_controls.sub    # (see below)
sbatch --array=0-17,36-37 --time=1:00:00 GPU/csf3/tg_mhd_pilot.sub  # Phase 3: the forced boxes' pilot,
sbatch --array=18-35 --time=0:30:00 GPU/csf3/tg_mhd_pilot.sub       # chained on the verify job (see below)
sbatch --array=0-1   --time=16:00:00 GPU/csf3/tg_mhd_pilot2.sub    # its second round, likewise
sbatch --array=2-13  --time=2:30:00  GPU/csf3/tg_mhd_pilot2.sub
sbatch --array=14-25 --time=1:00:00  GPU/csf3/tg_mhd_pilot2.sub
bash --login GPU/csf3/rr_paper/rr_paper_build.sh                  # RR-MHD paper: OT 3-D campaign,
sbatch --array=0-2,5,6,8  --time=1:00:00 GPU/csf3/rr_paper/rr_paper.sub   # FP64 build first, then
sbatch --array=3,4,7,9,10 --time=6:00:00 GPU/csf3/rr_paper/rr_paper.sub   # two groups by wallclock
sbatch GPU/csf3/rr_paper/rr_paper_spectral.sub                    # its spectral reference (CPU)
sbatch GPU/csf3/rr_paper/rr_paper_post.sub                        # after both: spectra + tarball
sbatch --array=11,13 --time=1:00:00 GPU/csf3/rr_paper/rr_paper.sub         # its omega_b = 1 follow-up,
sbatch --array=12    --time=3:00:00 GPU/csf3/rr_paper/rr_paper.sub         # then ONLY="<tags>" post
```

`mhd_jet.sub` needs a **per-precision build tree**, because `Real` is a
compile-time typedef and one tree cannot hold both — a stale binary would run at
the other precision and say so nowhere but its own header line:

```bash
cd ~/scratch/M3LB/GPU
cmake -S . -B build_fp32 -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=80 \
      -DLBM_DOUBLE=OFF -DLBM_ONLY_OP=cm
cmake --build build_fp32 --target mhd_jet -j8
```

`NX` and `PREC` override the defaults: `NX=256 PREC=fp64 sbatch ...`, with a
matching `build_fp64`. The job's own header carries the cost table and the two
numbers to read in the log first.

`ot3d_re3040.sub` is the odd one out: it builds **`GPU/`**, not the Kokkos
tree, and it exists because the run does not fit a free Colab session --
26417 steps at 288^3 took ~50 min on a T4 and the GPU usage limit killed it
at 62 %, losing every frame. On a gpuA A100 it is minutes. It writes 244
slice and 244 volume frames for an animation; render them afterwards on any
machine with the repo, no GPU needed (the commands are in its header).

**`rr_paper/`** is the three-dimensional Orszag–Tang campaign of the RR-MHD paper:
`orszag_tang -op cm -wbulk omega -fullinit`, which is single-rate recursive
regularisation (to 2.6e-13 against that paper's own D3Q27 code over a whole M = 32
run), at Re = 1000, 3000 and 10000 on M = 128 to 512 in FP64, with BGK at the same
settings, one `-wbulk 1` run, a pseudo-spectral reference on 32 CPU cores and a
post-processing job that leaves one tarball to copy back. About 10 A100-hours.
Its own `README.md` has the table of elements and what each is for.

**Ask for the wallclock you need, not the maximum.** SLURM backfills short jobs
into gaps ahead of long ones, so a 1-day request can only start when a 1-day
hole opens. Submitting the 45-minute H = 998 element with `-t 1-0` got it
scheduled to start **sixteen hours later** with reason `(Priority)`; at
`--time=2:00:00` it schedules against a two-hour hole instead. The script now
defaults to 2 h, which covers elements 0 and 1; element 2 must override.

### Checking on it

An array job gets a compound id, `<jobid>_<index>` — `12345_0` for `--array=0`.

```bash
squeue --me                     # queued / running; ST is PD pending, R running
squeue --me --start             # SLURM's estimate of when a pending job starts
tail -f rbcold-12345_0.log      # live progress, one row per output interval
sacct -j 12345 --format=JobID,State,Elapsed,MaxRSS,ExitCode   # after it ends
scancel 12345_0                 # kill one array element
scancel 12345                   # kill the whole array
```

`squeue` showing nothing means the job has finished (or never started) — look at
the log. A job that died in the first seconds almost always says why in the
first ten lines: a missing module, or the `ERROR: no executable at ...` check.

The log lands in the directory you submitted from; the **field dumps go to
`runs/rb_cold_h<H>/`**, deliberately, since at `-out 1` they are hundreds of
megabytes and have no business next to the source.

**`cmbench.sub` first.** It settles the Kokkos central-moment collapse, which
has been open since the port and is the only thing in this tree that genuinely
cannot be diagnosed without a GPU. It is bounded by construction and cannot
hang. Its header says what each possible answer means.

**`rb_cold.sub`** is Rayleigh–Bénard at Ra = 10¹¹ with the reference's cold
start, at H = 498 / 998 / 1998. The H = 498 element reproduces a CPU run that
**failed** — halted on the maximum principle at t = 25 with T = 1.19 against a
physical maximum of 0.5 — so it is a cross-code check of a known failure, and
the other two are the refinement that failure calls for.

Rough cost on one A100 at ~1.5 GLUPS FP64, 100 free-fall times:

| H | grid | cells | steps | est. |
|---|---|---|---|---|
| 498 | 1000 × 500 | 5.0e5 | 1.0e6 | ~5 min |
| 998 | 2004 × 1000 | 2.0e6 | 2.0e6 | ~45 min |
| 1998 | 4012 × 2000 | 8.0e6 | 4.0e6 | ~6 h |

For scale, H = 498 took **3 hours on four CPU threads** before it died.

**`tg_mhd_verify.sub`, then `tg_mhd_ladder.sub`** are the confined-MHD box
(`GPU/src/tg_mhd.cu`: Taylor–Green MHD in a closed cube, on branch `mhd`).
The verify job builds `GPU/build64` itself, runs the free-slip and no-slip
boxes at N = 65 in FP64, and checks them against the Kokkos twin's series in
`results/P_tg_mhd/xcheck_n65_re200/` — one `PASS`/`FAIL` line per wall, to
1e-7 of each column's scale. It then times an FP32 pair at N = 257 (577 and
723 MLUPS with probes on the first device run). **Read the two PASS lines before
submitting the ladder**: the ladder has no twin to check it against.

The ladder is 24 array elements: setups A (free slip + conducting), B (no slip
+ conducting) and C (no slip, no field), at Re = 125 / 250 / 500 / 1000, each at
a primary N = 192 / 256 / 384 / 512 and a coarser check grid, FP64, ~9 GPU-hours
in all as measured on the first ladder (2026-09-28; the job's header has the
table, from 1 min at N = 128 to ~2 h at N = 512). It does **not** build — it refuses a `build64` binary older than the
source and prints the rebuild command, so after every `git pull` rebuild on the
login node (the job's header has the four lines). Submit in the three groups
above so each asks for the wallclock it needs; the job's header maps elements to
runs. Output goes to `runs/tg_mhd_ladder/<A|B|C>_re<Re>_n<N>/`: `series.dat`
and `profile.dat` (the dissipation against wall distance, which f_w is read
from) always, `anim_frames/` for the primary runs, and full `raw/` snapshots
only with `RAW=1` (3.8 GB each at N = 512 — ask for them per run, for the
figures).

Then, on the login node — pure stdlib:

```bash
python3 tools/tg_mhd_ladder.py runs/tg_mhd_ladder
```

It prints one row per run (f_w at the dissipation peak, min E_M/E_V, mass drift,
div b), the plan's 5 % resolution gate per rung, and B − A and B − C against the
resolution band. Exit status 1 means some rung failed the gate. Series from
before 2026-09-28 counted f_w in whole node layers, which made the first
ladder's gate measure where the band edge fell between nodes rather than the
flow; the tool corrects those from their three columns and marks them `*`, and
only a re-run gives numbers for a figure. The files are small — `series.dat`,
`profile.dat` and `log.txt` for all 24 runs are a few MB:

```bash
rsync -av --include='*/' --include='series.dat' --include='profile.dat' --include='log.txt' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_ladder/ results/P_tg_mhd/ladder/
```

run on the laptop from the repo root; `.gitignore` keeps the frames, raw
volumes and films out of git at that depth.

**`tg_mhd_mach.sub`** is the ladder's Mach-halving check, the plan's last
Stage 3 gate: A and B at u0 = 0.025 instead of 0.05, at Re = 1000 (N = 512, the
gate) and Re = 500 (N = 384, the fallback, since the top rung's tau - 1/2 = 0.012
sits at RegWall's floor). The ladder's peak Mach is 0.117, above this tree's
0.087 rule, so the cost of it has to be measured, not assumed. Halving u0 at
fixed N also halves tau - 1/2, so the check sees the two errors together. The
verdict is fixed in the job's header before the runs -- the shift in B - A
against the rung's resolution band (PASS) or half the Re step it could fake
(MARGINAL) -- and `tools/tg_mhd_mach.py` applies it; exit status 2 means the top
rung gave no verdict and only the fallback did.

**`tg_mhd_round2.sub`** is what the PRL still needs, 29 elements and about 22
GPU-hours: the Re = 2000 rung (A, B, C in FP32 at N = 640 against N = 512, with
an FP64 N = 512 control of the precision, kept in `runs/tg_mhd_fp64check/` so
the ladder tool never sees two runs at one grid); the insulating pair A' / B'
(pseudo-vacuum walls with the TG-I field) at Re = 500 and 1000 on two grids
each; and A and B rerun at Re = 250-1000 on both grids, because the drivers now
also write `profile_visc.dat` and `profile_ohm.dat`, the viscous and Ohmic
parts of the wall-distance profile. Its three verdicts are fixed in the header
before the runs. Chain it on the verify job, which rebuilds BOTH `build64` and
`build32` and checks the device on the split profiles too:

```
jid=$(sbatch --parsable GPU/csf3/tg_mhd_verify.sub)
sbatch --dependency=afterok:$jid --array=0-9   --time=5:00:00 GPU/csf3/tg_mhd_round2.sub
sbatch --dependency=afterok:$jid --array=10-28 --time=2:00:00 GPU/csf3/tg_mhd_round2.sub
```

and copy back `runs/tg_mhd_round2/` and `runs/tg_mhd_fp64check/` with the
same rsync line as the ladder, adding `--include='profile_*.dat'`.

**`tg_mhd_snap.sub`** is the paper's field snapshots: A, B and C at Re = 1000,
N = 512, FP64, to t = 4.7 with full float32 dumps at t = 0, 2.3 and 4.6, then
`tools/tg_mhd_slices.py` on the node cuts planes of the local viscous and Ohmic
dissipation out of them -- with the drivers' own derivative rules, checked to
reproduce `series.dat`'s eps to 1.7e-8. About 40 min a run; the 11 GB of raw
dumps per run are kept unless `KEEP_RAW=0`. It needs no rebuild: the round-2
`build64` is current. Copy back only the slices and the series:

```
rsync -av --include='*/' --include='series.dat' --include='profile*.dat' --include='log.txt' --include='slices/*' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_snap/ results/P_tg_mhd/snap/
```

**`tg_mhd_vol.sub`** turns those dumps into the paper's 3-D views: it builds
`tools/tg_mhd_vol.cpp` with the node's `g++` and writes, for A, B and C at
t = 2.3 and 4.6, the viscous and Ohmic dissipation reduced by the block maximum
to 256^3 -- 768 MB in all, so the 3.8 GB dumps stay on scratch. A few minutes;
no GPU is used. Copy the volumes back with

```
rsync -av --include='*/' --include='vol/*' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_snap/ results/P_tg_mhd/snap/
```

**`tg_mhd_budget.sub`** is Phase 1 of the paper: A and B on both grids of every
rung, at round 2's exact settings, now also writing `budget.dat` -- the kinetic
and magnetic energy budgets against wall distance. Its header FIXES, before the
runs, the test of the paper's conjecture for the near-wall Ohmic deficit (phi,
the share of the deficit that the stretching deficit accounts for, against 1/2
at every rung), and every element checks that its `series.dat` is round 2's
exactly. About 13 GPU-hours. Chain it on the verify job, which rebuilds both
trees from the new source and checks the device's budget against the host's:

```
git pull
jid=$(sbatch --parsable GPU/csf3/tg_mhd_verify.sub)
sbatch --dependency=afterok:$jid --array=0-3  --time=4:00:00 GPU/csf3/tg_mhd_budget.sub
sbatch --dependency=afterok:$jid --array=4-7  --time=2:30:00 GPU/csf3/tg_mhd_budget.sub
sbatch --dependency=afterok:$jid --array=8-15 --time=1:00:00 GPU/csf3/tg_mhd_budget.sub
```

then `python3 tools/tg_mhd_budget.py runs/tg_mhd_budget` on the login node, and
copy back with

```
rsync -av --include='*/' --include='series.dat' --include='profile*.dat' --include='budget.dat' --include='log.txt' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_budget/ results/P_tg_mhd/budget/
```

**`tg_mhd_controls.sub`** is Phase 2: the four controls a referee will ask for,
each with its rule fixed in the header before the runs -- a smooth start
(`-ramp`), the box without a field at B's initial energy (`-vamp sqrt2`), the
Mach number at Re = 2000 (u0 = 0.04, as far as RegWall's tau floor allows at
N = 640), and Pm = 0.5 and 2. 22 elements, about 30 GPU-hours, chained on the
verify job, which now also checks the device on `-ramp` and `-vamp` (the device
half of the per-node wall velocity is code no host build compiles):

```
git pull
jid=$(sbatch --parsable GPU/csf3/tg_mhd_verify.sub)
sbatch --dependency=afterok:$jid --array=0-5   --time=5:00:00 GPU/csf3/tg_mhd_controls.sub
sbatch --dependency=afterok:$jid --array=6-12  --time=3:00:00 GPU/csf3/tg_mhd_controls.sub
sbatch --dependency=afterok:$jid --array=13-21 --time=1:30:00 GPU/csf3/tg_mhd_controls.sub
```

then `python3 tools/tg_mhd_controls.py runs/tg_mhd_controls`, and copy back with

```
rsync -av --include='*/' --include='series.dat' --include='profile*.dat' --include='budget.dat' --include='log.txt' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_controls/ results/P_tg_mhd/controls/
```

**`tg_mhd_pilot.sub`** is the pilot for Phase 3, the forced, statistically
steady boxes: both drivers' `-drive F0` (the bulk Taylor-Green force) and
`-bdrive FM` (the bulk TG-C source on the field -- no dynamo keeps the field),
scanned over F0 = 0.1/0.2/0.4 x FM = 0.1/0.2/0.4 for A and B at Re = 250 and
1000, to t = 50. It tests nothing: its header fixes, before the runs, how the
production amplitudes are chosen (A at Re = 1000 closest to its decaying
turbulent peak), and `tools/tg_mhd_forced.py pilot` applies that rule reading no
f_w. 38 elements, about 10 GPU-hours, chained on the verify job, which now also
checks the device's forced runs:

```
git pull
jid=$(sbatch --parsable GPU/csf3/tg_mhd_verify.sub)
sbatch --dependency=afterok:$jid --array=0-17,36-37 --time=1:00:00 GPU/csf3/tg_mhd_pilot.sub
sbatch --dependency=afterok:$jid --array=18-35      --time=0:30:00 GPU/csf3/tg_mhd_pilot.sub
```

then `python3 tools/tg_mhd_forced.py pilot runs/tg_mhd_pilot`, and copy back with

```
rsync -av --include='*/' --include='series.dat' --include='profile*.dat' --include='budget.dat' --include='log.txt' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_pilot/ results/P_tg_mhd/pilot/
```

Its rule gave INCONCLUSIVE (2026-10-04): at Re = 1000 the field was still
growing at t = 50, the strong fields diverged, and the target lies beyond the
Mach bound. **`tg_mhd_pilot2.sub`** is the second round its header calls for:
F0 = 0.15/0.2/0.3 x FM = 0.05/0.1 to t = 150, A and B at Re = 250 and 1000, plus
the top rung -- A and B at Re = 2000 on N = 512, FP32, for (0.2, 0.1) -- with a
stationarity condition over t = 100..150. 26 elements, about 50 GPU-hours, the
two top-rung elements ~13 h each:

```
git pull
jid=$(sbatch --parsable GPU/csf3/tg_mhd_verify.sub)
sbatch --dependency=afterok:$jid --array=0-1   --time=16:00:00 GPU/csf3/tg_mhd_pilot2.sub
sbatch --dependency=afterok:$jid --array=2-13  --time=2:30:00  GPU/csf3/tg_mhd_pilot2.sub
sbatch --dependency=afterok:$jid --array=14-25 --time=1:00:00  GPU/csf3/tg_mhd_pilot2.sub
```

then `python3 tools/tg_mhd_forced.py pilot runs/tg_mhd_pilot2 --round 2`, and copy back with

```
rsync -av --include='*/' --include='series.dat' --include='profile*.dat' --include='budget.dat' --include='log.txt' --exclude='*' csf3:scratch/M3LB/runs/tg_mhd_pilot2/ results/P_tg_mhd/pilot2/
```

---

## Reading the output

In this order, and the first one is not optional:

1. **`T_min` / `T_max`.** Advection–diffusion with Dirichlet data obeys a
   maximum principle, so T outside `[0, 1]` is the scheme failing and nothing
   else. Out-of-bounds lines are marked `!`; the run halts at twice the range.
   The **last line of every run** states the worst excursion, when it happened,
   and whether it recovered — a run that *completed* is not thereby a run that
   *stayed in bounds*, and a Nusselt number measured outside them is not a
   measurement.
2. **`Nu_bot` against `Nu_top`.** They must agree with each other. Their
   disagreement is the honest error bar. The failed CPU run had 80.4 against
   57.4.
3. **`Nu_vol`** only when it is far above `Nu_floor` *and* not flipping sign.
   It carries a factor H/α — 5.3e6 at these parameters — so it is mostly
   amplified noise. In the CPU run it flipped sign every output row for the
   first twenty free-fall times while both plate estimators sat correctly at 1.
4. `Nu_ref` is the reference driver's own normalisation, printed only so the two
   codes can go in one table. It is the raw correlation divided by `nx-1`, and
   it is 3% high by construction. Do not mix it with `Nu_vol`.

## Post-processing

Field dumps are `<prefix>_T_*.bin` (temperature) and `<prefix>_u_*.bin` (speed),
`nx × ny` float32 behind two `int32` of header, one pair per output interval.

`doc/fig/mkpng.py` renders them and is **pure stdlib** — no numpy, which matters
because CSF3's default python may not have it:

```bash
cd runs/rb_cold_h498
python3 ../../doc/fig/mkpng.py seq rb_cold_h498_T_0050.bin T.png 0 1
```

**Pin the range** (`0 1` above, the physical range of T). Without it every frame
gets its own scale and the colours move when the field does not; pinned, a field
that leaves its physical range saturates visibly and the tool prints `CLIPPED` —
which is how an out-of-bounds run looks in a picture rather than in a column.

A whole sequence, then a video (`module load apps/binapps/ffmpeg` if `ffmpeg` is
not already on PATH):

```bash
mkdir -p png
for f in rb_cold_h498_T_*.bin; do
  n=${f##*_T_}; n=${n%.bin}
  python3 ../../doc/fig/mkpng.py seq "$f" "png/T_$n.png" 0 1
done
ffmpeg -y -framerate 10 -pattern_type glob -i 'png/T_*.png' \
       -vf vflip -c:v libx264 -pix_fmt yuv420p -crf 18 rb_cold_h498.mp4
```

**`-vf vflip` is not optional.** The dump writes row y = 0 first and PNG puts the
first row at the top, so the raw image has the HOT plate at the top and the
plumes falling. Without the flip the physics reads upside down.

### ParaView

`doc/fig/bin2vtk.py` converts the dumps to legacy VTK, which ParaView opens
directly. Pure stdlib, and the output is the same size as the input rather than
the ~5x an ASCII VTK costs:

```bash
cd runs/rb_cold_h998
python3 ../../doc/fig/bin2vtk.py --glob 'rb_cold_h998_T_*.bin' \
        --out h998 --pair rb_cold_h998_u
```

That writes `h998_0000.vtk .. h998_00NN.vtk`, each carrying **both** scalars --
`Temperature` and `Speed` -- so you can switch fields in ParaView without
reloading. Open the collapsed name (`h998_..vtk`) in the file dialog and
ParaView groups them as a time series with the animation controls live.

Two things the script exists to get right. Legacy VTK binary is specified
**big-endian** whatever machine wrote it; native little-endian floats produce a
file ParaView opens without complaint and renders as values around 1e-40 and
1e38, which looks like a diverged simulation rather than a byte-order mistake.
And the dumps carry a two-`int32` header that a raw reader has to be told to
skip.

If you would rather not convert, ParaView's **Raw (binary) Files** reader can
open a `.bin` directly: Data Extent `0 nx-1  0 ny-1  0 0`, Scalar Type `float`,
Byte Order `LittleEndian`, **Header Size `8`**. That last one is the two int32s;
without it the first two values are read as floats and the field is shifted by
two cells.

The driver's own `-vtk` flag also writes real VTK, but ASCII and only if you
decide before the run: ~40 MB per frame at 2004 x 1000 against 8 MB for the
binary dump.

For the linear phase the raw field is useless — the perturbation is four orders
below the mean profile. Render the departure from the horizontal average
instead, `T'(x,y) = T(x,y) − ⟨T⟩ₓ(y)`, with the diverging map:

```bash
python3 - <<'EOF'
import struct, sys
src, dst = 'rb_cold_h498_T_0010.bin', 'tp.bin'
d = open(src,'rb').read(); nx, ny = struct.unpack('<ii', d[:8])
v = list(struct.unpack('<%df' % (nx*ny), d[8:8+4*nx*ny]))
for y in range(ny):
    row = v[y*nx:(y+1)*nx]; m = sum(row)/nx
    for x in range(nx): v[y*nx+x] = row[x] - m
open(dst,'wb').write(struct.pack('<ii', nx, ny) + struct.pack('<%df' % (nx*ny), *v))
EOF
python3 ../../doc/fig/mkpng.py div tp.bin Tpert.png 1.0
```

That is what showed the mode structure on the CPU run while every Nusselt
column still read 1.0.

Copy anything worth keeping back to home or RDS — **scratch is not backed up**
and files unused for three months can be deleted.

---

## Getting results back

**Render on CSF3 and download the movies, not the frames.** The renderers in
`results/N_mhd_sphere/` are deliberately pure stdlib — no numpy, no PIL, no
matplotlib — and need no GPU, so they run on a login node. For the Re = 3040
Orszag-Tang that is 20 MB instead of 1.11 GB, a factor of 55.

What `ot3d_re3040.sub` leaves behind, at M = 288 with 244 frames:

| | each | 244 frames |
|---|---|---|
| `umag`/`bmag`/`jmag` slices, 288² float32 | 332 kB ×3 | 243 MB |
| `jvol_*.raw`, 96³ after `-volstride 3` | 3.54 MB | 864 MB |
| `log.txt` | 30 kB | — |
| **total `anim_frames/`** | | **1.11 GB** |

### The cheap route — about 20 MB

On CSF3:

```bash
cd ~/scratch/M3LB/ot3_re3040
python3 ../results/N_mhd_sphere/render_slices.py anim_frames png
python3 ../results/N_mhd_sphere/render_volume.py anim_frames
module avail ffmpeg            # then load whatever it names
ffmpeg -y -framerate 24 -i png/frame_%04d.png -c:v libx264 -pix_fmt yuv420p \
       -crf 20 -movflags +faststart slices.mp4
ffmpeg -y -framerate 24 -i anim_frames/png/jvol_%04d.png -c:v libx264 \
       -pix_fmt yuv420p -crf 18 -vf scale=iw*2:ih*2:flags=neighbor jvol.mp4
```

On the laptop:

```bash
rsync -avP csf3:scratch/M3LB/ot3_re3040/{slices.mp4,jvol.mp4,log.txt} ~/Downloads/
rsync -avP csf3:scratch/M3LB/ot3_re3040/anim_frames/meta.txt ~/Downloads/
```

### No ffmpeg on CSF3 — about 300 MB

Render there, encode here:

```bash
rsync -avP csf3:scratch/M3LB/ot3_re3040/png ~/Downloads/ot3_re3040/
rsync -avP csf3:scratch/M3LB/ot3_re3040/anim_frames/png ~/Downloads/ot3_re3040/jvolpng/
```

### The raw frames — 1.11 GB, and only if you will re-render

```bash
rsync -avP csf3:scratch/M3LB/ot3_re3040/anim_frames ~/Downloads/ot3_re3040/
rsync -avP --exclude='jvol_*' csf3:scratch/M3LB/ot3_re3040/anim_frames ~/Downloads/   # slices only, 243 MB
```

**`rsync -avP`, not `scp`.** `-P` keeps a partial transfer and resumes it. At
gigabyte scale over a VPN that is the difference between a dropped link costing
seconds and costing the whole copy — the Colab run this job replaced was lost
at 62 % for exactly that reason, nothing retrieved as it went.

**Always take `log.txt`.** It is 30 kB and carries the whole diagnostic table:
`J_max` in both lattice and dimensionless units, and the `max|divB|/k|B|` column,
which is the number to read before quoting any `J_max` from a run this close to
the tau floor.

---

## Notes

- **One GPU.** `GPU/` has no MPI and no multi-GPU, so `-G 1` always. Asking for
  a whole node idles three A100s and queues far longer. Multi-GPU would need a
  two-way, parity-dependent halo exchange because of Esoteric Pull — see the
  known-limitations section of `doc/m3lb.pdf`.
- **No restart.** A run that outlives its wallclock is lost, so size `-t`
  generously; you have 4 days on `gpuA` and the runs above need hours.
- **Check the module versions.** `libs/cuda/12.8.1` is what CSF3's own GPU
  example uses, but run `module avail cuda` and `module avail gcc` — nvcc needs
  a C++20 host compiler and the parent Kokkos tree will fail first if the
  default `gcc` is too old.
