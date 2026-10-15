---
name: analyse
description: Analyse and plot M3LB results - read series and profile files, build a figure or table, and quote a number correctly. Use after a run or campaign finishes, when asked for a plot, a convergence table, or a comparison against a reference.
argument-hint: "[the results directory or the question]"
---

# Analysis and plots

## Look at the data before computing anything

1. **Read the run's own log first**: the diagnostics it printed, the worst excursion,
   whether it finished. A run that completed is not thereby a run that stayed in
   bounds.
2. **Plot every series against time at every parameter value** before reading a
   number off it. A quantity read "at the peak" can switch peaks as the parameter
   moves, and its trend then measures the switch (the `tg_mhd` ladder's lesson). Try
   more than one definition of "when", and say which one a figure uses.
3. **Check resolution and convergence** with `/check-claim` before a figure is made,
   not after.

## Where things live

- Tracked reference data: `results/<case>/` summary tables. Not tracked:
  per-node field dumps, frames, videos, `runs/`. See `.gitignore`; it lists the
  patterns and says why.
- Analysis tools: `tools/*.py`. The convention is a docstring that states the
  observable and the rule, a table printed to stdout, and an exit status that
  carries the verdict. Match it. `tools/tg_mhd_ladder.py` is the model.
- **Any script that must run on CSF3's login node is stdlib-only**; its default
  Python may have no numpy. `doc/fig/*.py` (`mkpng.py`, `bin2vtk.py`) are the
  stdlib examples.
- **For plotting on a laptop use a venv outside the repo**:
  `python3 -m venv ~/.venvs/m3lb && ~/.venvs/m3lb/bin/pip install numpy matplotlib`.
  System Python often has neither.

## Rules for a figure

- **The script is tracked; its output is not**, unless a document needs it to build.
  Do not commit PNGs, videos or dumps (the ignore rules say which).
- **Pin the colour range** to the physical range of the quantity, so that a field
  leaving its range saturates visibly instead of silently rescaling.
- Field dumps: `<prefix>_T_*.bin` are `nx x ny` float32 behind two `int32` of header,
  row `y = 0` first, so a PNG needs `-vf vflip` or the physics reads upside down.
  VTK binary is big-endian whatever machine wrote it. `doc/fig/bin2vtk.py` gets both
  right; `GPU/csf3/README.md` ("Post-processing") has the commands.
- **Mind memory.** A D3Q27 field at 512^3 is gigabytes. Do not preload every frame;
  stream or reduce first (`tools/tg_mhd_vol.cpp` reduces to 256^3 by block maximum).
- **A paper's numbers come from its `numbers.py`**, and a figure from a script that
  reads the same files. To change a number, change the script and regenerate; never
  type one in. Do not run Python with `paper/tg_mhd` as the working directory:
  `numbers.py` shadows the standard library's `numbers`.

## Reporting

For every number say: what it is, which file and column, which window, which
definition, and its band. Use the vocabulary the data supports: "converged",
"still creeping", "under-resolved, trend unknown", "consistent with" (not "converged
to") when the scatter overlaps the reference. Say which lattice, operator,
precision and solver (`src/` or `GPU/`) produced it.

When the figure or number will go into a paper, run `/check-claim` and offer a read by
the `claim-auditor` agent.
