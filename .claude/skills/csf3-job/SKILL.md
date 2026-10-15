---
name: csf3-job
description: Plan and write a CSF3 (SLURM, GPU) campaign for M3LB's GPU/ code - the job script, the element table, the cost estimate, the rule fixed before the runs, and the commands to submit and copy results back. Use for any run too big for a laptop, a Reynolds or resolution ladder, or a sweep.
argument-hint: "[what the campaign tests]"
---

# A CSF3 campaign

Read [GPU/csf3/README.md](../../../GPU/csf3/README.md) first; it is the authority on
partitions, setup and copying results back. Use `GPU/csf3/tg_mhd_mach.sub` as the
template: it is short and has every part below.

**You write the scripts; the student submits them.** `ssh`, `scp`, `rsync` and
`sbatch` are on the ask list in `.claude/settings.json`, CSF3 needs Duo 2FA, and
nothing here should be submitted without the student reading it. Give the exact
commands to paste, one block each.

## Before writing a job

1. **Run it on the host first.** `GPU/` compiles as plain C++ with
   `-DLBM_HOST_ONLY=ON`: run the case at a laptop-sized N, check the initial
   condition and the diagnostics it prints, and, where `src/` has a twin, compare. Wrong
   initial conditions and wrong diagnostics are found for free there.
2. **Estimate the cost** as cells x steps / MLUPS and put the table in the header.
   Measured rates are in the README and in the existing `.sub` headers; a rate from a
   different operator or precision is not a rate for this one.
3. **Fix the rule before the runs** (see below), and commit it before the first
   submission so that git's dates mean something.

## The job script

- `#!/bin/bash --login`, a long header comment, then
  `#SBATCH -p gpuA -G 1 -c 4 -t ... -J <name> -o <name>-%A_%a.log`, then
  `set -euo pipefail`, `module purge`, `module load libs/cuda/12.8.1`.
- **`gpuA` only.** Never `gpuL`: the L40S runs FP64 at 1/64 of FP32 and every
  convergence test here is FP64. **One GPU** (`-G 1`): `GPU/` has no MPI.
- `ROOT=${M3LB_ROOT:-$SLURM_SUBMIT_DIR}`: the job is copied by `sbatch`, so the
  submit directory is its only handle. Submit from the repo root.
- **Refuse a stale binary**: `find ... -newer "$BIN"` against the sources, and print
  the rebuild command. A per-precision build tree (`build64`, `build32`) because `Real`
  is a compile-time typedef; a stale binary runs at the other precision and says so
  only in its own header line.
- **An element table** `case $i in ... esac` mapping `SLURM_ARRAY_TASK_ID` to one run
  each, with a `*)` branch that errors. Submit elements in groups by wallclock:
  `sbatch --array=0-5 --time=0:30:00 ...`. Ask for the time you need, not the
  maximum; short jobs backfill and long requests wait.
- **Options in a bash array, never a string variable.** A flag string in a zsh
  variable does not word-split and the run silently falls back to the defaults.
- Output to `runs/<job>/<tag>/`, `tee log.txt`. `runs/` is git-ignored.
- **No restart**: a run that outlives its wallclock is lost, so size `-t` generously.
- Chain on a verify job: `jid=$(sbatch --parsable ...verify.sub)` then
  `sbatch --dependency=afterok:$jid ...`. Read the PASS lines before the ladder.

## The rule, fixed in the header before the runs

The header says, in this order: the **observable**; **when** it is read (and why that
instant, since a peak can move with the parameter); the **verdict thresholds**
(PASS, MARGINAL, FAIL, each with a number); what has already been seen when the
rule was written; and the analysis script that applies it. Write that script too
(`tools/<name>.py`, stdlib only, exit status 1 on failure, header docstring stating the
rule). Where amplitudes are being chosen, the choosing rule must not read the
observable. `GPU/csf3/tg_mhd_ladder.sub` and `tools/tg_mhd_ladder.py` are the model.

## Getting results back

Copy small files only, with `rsync -av --include='*/' --include='series.dat'
--include='profile*.dat' --include='log.txt' --exclude='*' csf3:scratch/M3LB/runs/<job>/
results/<name>/`. Always take `log.txt`. Scratch is **not backed up** and is purged:
treat `~/scratch/M3LB` as a build directory, never the only copy. Then run the
analysis script and `/analyse`.

## Add to the README

Append the new job's `sbatch` lines and its copy-back command to
`GPU/csf3/README.md` in the style of the existing entries.
