---
name: new-feature
description: Implement a new feature in M3LB (an operator, boundary, solver option, coupling, or a GPU/ port) without introducing a silent error. Use when adding or changing anything under src/ or GPU/.
argument-hint: "[the feature]"
---

# Implementing a feature

The risk in this tree is a plausible, converged, wrong answer, not a crash. Work
in this order.

1. **Which tree?** `src/` (Kokkos, full physics, CPU) or `GPU/` (CUDA, shares no headers
   with it). They deliberately duplicate the physics; do not de-duplicate them. Where a
   feature touches both, write both and compare, as `tests/cross_colour` does for the
   colour gradient (320 states, all populations, difference 0).
2. **Read the banner of the file you will change**, and of the file that implements
   the thing you are copying. It frequently already says why the obvious change is
   wrong. Read the matching "Invariants that break silently" entries in
   [CLAUDE.md](../../../CLAUDE.md).
3. **Make it opt-in, and free when off.** The default path must reproduce every
   existing reference exactly (precedents: `set_wall_velocity_field`, `rho_G_of`,
   `omega_of`). Show it: run `ctest` and compare the output of one affected validation
   case before and after.
4. **Add the test with the feature**: a unit test in `tests/` (first `foreach` in
   `tests/CMakeLists.txt`) and a validation case with a known answer (`/new-case`).
   Make the test able to fail; delete the new term and confirm the answer moves.
5. **GPU/ work: host first.** Build with `-DLBM_HOST_ONLY=ON` and run
   `ctest --test-dir build-host --output-on-failure`; only then the device, through
   `/csf3-job`. Precision defaults are opposite in the two trees (`Real` is float
   unless `LBM_DOUBLE` in `GPU/`, double unless `LBM_SINGLE_PRECISION` in `src/`); a
   mismatch compiles and reads as a 1e-7 physics bug.
6. **After touching a moment operator**, run `tests/frame_check.sh <kokkos build dir>`
   and read the `loops` and `regidx` columns, not the wall clock. A moment index
   must be a compile-time constant.
7. **Document it**: banner comment (the decision, the measurement that settled it,
   what it does *not* do) and, for a limitation, `doc/m3lb.tex`'s "Known
   limitations". CLAUDE.md edits are on the ask list; propose the wording and let
   the student or supervisor decide.
8. **Review it**: ask the `trap-reviewer` agent to read the diff before saying done.

## The short list of silent failures to check against the diff

- Direction ordering: `opp(i)` is `i+1` for odd `i`; opposite pairs stay adjacent.
- Storage x streaming pairing; shifted populations versus a density ratio.
- `cs2` is 1/4 on D3Q7; take omega from `omega_from_*`.
- D2Q5 and D3Q7 are not product lattices: no central-moment operator on them.
- A new scalar cell type: prescribed or closed? A closed one must take the source.
- A node whose populations are prescribed must not have its field recomputed.
- A lattice-unit coefficient is a physical quantity; derive it from SI, because it
  silently rescales with dt.
- A seed must agree with its own collision; a switch has other readers.
- A flag that changes several others must be applied before the argument loop.
- `GPU/` gather and scatter are a streaming pair, not a read-modify-write.
- A coupling term behind an opt-in buffer plus a null check: delete the term and see
  whether the answer moves.
- A normalised copy that the caller never sees (`PhaseChange`).

Do not commit unless asked.
