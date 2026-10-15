---
name: check-claim
description: Audit a number or a sentence before it goes into a result, figure, README or paper. Use whenever a quantity is about to be quoted, a convergence or trend is claimed, or two runs are compared.
argument-hint: "[the claim, or the file the number comes from]"
---

# Checking a claim

This tree's confident wrong conclusions all came from the same few mistakes (see
"Measurement discipline" in [CLAUDE.md](../../../CLAUDE.md)). For each number or
sentence, go through the list below and report one line per item: **pass, fail, or
not checked**, with the file and column it was read from. "Not checked" is an
honest answer; guessing is not.

1. **Provenance.** Which file, which column, which run, which commit? A number that
   no script can regenerate is a number to distrust. Paper numbers come from
   `paper/tg_mhd/numbers.py`; never type one in.
2. **Converged?** Plot the series. Is it flat, over what window, and is it still
   creeping? Say "still creeping" if so. A per-step residual is not a convergence
   test; use the whole-field, interval-based one.
3. **One thing changed?** Two runs that differ in geometry *and* precision, or
   viscosity *and* interface width, cannot attribute a difference. Two models at their
   own driver defaults compare drivers.
4. **Matched quantity?** Kinematic viscosity across a density ratio, not dynamic.
   The right length, `H` and not `ny`.
5. **Resolved?** Has the run reached the resolution its own diagnostic asks for
   (cell Reynolds number, delta k_max, the tau floor)? If not, the honest verdict is
   "under-resolved, trend unknown". A coarse ladder cannot be extrapolated into a
   regime it never reached.
6. **At least three points before an order**, and an apparent order well above the
   scheme's is a symptom, not a bonus.
7. **A reading "at the peak" or at one instant**: does the peak change as the
   parameter moves? Plot the observable against time at every rung and try more than
   one definition of "when". Sweep N across one cell before trusting a band, shell or
   threshold observable.
8. **Reproducible metric?** Blow-up step counts move by 2x with the backend alone;
   that is not a measurement.
9. **Port against host?** Agreement between a port and its reference proves the port,
   not the physics. A port several times worse than its parent is a defect until
   shown otherwise.
10. **Inputs against the source.** The paper's equation, the sibling code, not only
    internal consistency.
11. **The wording.** `every`, `mostly`, `about`, `monotonically`, `converged`,
    `exact` are claims too. Check each against the data, and give the range, not a
    rounded adjective (41-45 % is not "about 40 %").
12. **Say what was used**: lattice, collision operator, precision, solver (`src/` or
    `GPU/`), grid, and whether the run was a regression test or a measurement.

## Output

A short table: claim, verdict, evidence (path and column). Then one sentence: what
may be quoted as it stands, what needs a run or a re-read first, and what to drop.

For a claim that will appear in a paper or a figure, offer a second read by the
`claim-auditor` agent. Give it the data paths and the claim, **not your conclusion**.
