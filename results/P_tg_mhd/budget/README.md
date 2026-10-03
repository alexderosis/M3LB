# Phase 1: the energy budget runs

`GPU/csf3/tg_mhd_budget.sub`, run on CSF3 A100s on 2026-10-02: boxes A (free slip)
and B (no slip), conducting walls, on both grids of every rung (Re = 250, 500,
1000, 2000), at round 2's exact settings. The job's header fixed the test before
the runs; `tools/tg_mhd_budget.py results/P_tg_mhd/budget` applies it.

Each directory keeps `budget.dat` (the twelve terms of the kinetic and magnetic
energy budgets per wall-distance layer, out to 8 delta), `series.dat` and
`log.txt`, whose CSF3 paths are masked as `~/scratch`.

- **These are round 2's runs.** Every `series.dat` is byte-identical to
  `../round2/<run>/series.dat` (the job checked, and the log ends saying so),
  and so are all 48 `profile*.dat` files, compared after copying. Git keeps
  round 2's profiles only (`.gitignore`).
- **Cost: 15.1 GPU-hours**, against the job header's estimate of 13. The steps
  took the same time as round 2's (ratio 1.00-1.01); the probes took 36-77 % of
  each run, against 14-52 % in round 2 -- the budget made a probe 1.8-4.8x as
  dear on CSF3's host CPUs, where the laptop had measured 1.6x.
