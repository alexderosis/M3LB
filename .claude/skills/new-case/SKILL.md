---
name: new-case
description: Write a new M3LB validation case (an exact-answer regression test) or a demonstrator, following this repo's recipe. Use when asked to add a case, test, benchmark or validation, or to reproduce a published result.
argument-hint: "[what it should be validated against]"
---

# Writing a case

A validation case **has something to be right against** (an analytic answer, a
published table, a convergence rate) and prints PASS/FAIL. A case that only runs is a
demonstrator and goes in `demonstrator/`. If the student cannot name the answer, that
is the first thing to settle; ask before writing code.

## Steps

1. **State the answer first**, in one line: the quantity, the reference, the
   tolerance or order. Check the reference's formulas against the source and against
   a sibling. A wrong constant still gives a consistent simulation, and a published
   equation can carry a typesetting error (`validation/tgv2d.cpp`'s banner records a
   lost minus sign). **Derive the tolerance from the expected order or the physics,
   not from a round number.** If a first run forces a gate to change, say so in the
   banner: the gate is then a regression guard on what the code does, not a
   derivation of it.
2. **Choose the solver** from "Choosing a solver" in [CLAUDE.md](../../../CLAUDE.md).
   Standing rules: no multiphase solver for single-phase flow; "2-D" means a 3-D
   lattice with `nz = 1`; a 3-D phase-field fluid is D3Q27.
3. **Read the banner of every header the case touches** (the solver, the collision
   operator, `grid/Domain.hpp`, the streaming policy). The banners are the
   documentation, and they often say why the obvious choice is wrong.
4. **Copy the shape of a small existing case**: `validation/tgv2d.cpp` (fluid only,
   convergence ladder), `validation/natural_convection.cpp` (coupled, 126 lines),
   `validation/claudemd_skeleton_check.cpp` (the bare skeleton). Include
   `Campaign.hpp` first; headers are not self-contained and alphabetical order fails
   to compile. `open_out` writes into `results/<dir>/`, which is tracked reference
   data: use it only for output meant to be kept.
5. **Size it with `/units`.** Resolution, u0 <= 0.05, nu from Re, omega from
   `Coll::omega_from_viscosity`, read tau back and print Ma.
6. **Make the check able to fail.** Add a control that the check must catch (a
   collision built from a viscosity 5 % off, say), or delete the term under test and
   confirm the answer moves. Use at least three resolutions before quoting an order,
   and treat an order far above the scheme's as a symptom. A test of an in-place
   update needs a differential run over several steps; two obvious tests are blind to
   it. **Do not write what the scheme will do into the banner before the first run.**
   Mark a prediction as a prediction and delete the ones the run contradicts: the
   dry run of this skill guessed a plateau where the run showed second-order
   convergence.
7. **Print PASS/FAIL and return non-zero on failure.** Call
   `known_configuration(lat, op)` (see `validation/tgv3d.cpp`) right after parsing
   and before printing any header that names the lattice, so a stale `-lat` prints an
   error and not a header for a run that never happens; `dispatch` then gives the hard
   error, never a quiet fall-through. Make an unknown option fatal (exit 2) too.
   Kokkos strips its own `--kokkos-*` arguments first, so `ctest`'s thread flag does
   not trip it.
8. **Register it.** `validation/CMakeLists.txt` has two `foreach` lists: the first is
   built and run by `ctest`, the second is analysis sweeps too slow to be tests.
   `demonstrator/CMakeLists.txt` has one list and no `add_test`. Add the name to
   the right list; the first is one very long line, so check that `git diff` shows
   one added token and touch nothing else. A registered case is a commitment: the
   file must be committed in the same change, or every other checkout fails at
   `cmake`.
9. **Build and run** (note `-DKokkos_ENABLE_THREADS=ON`; without it everything runs
   serial and nothing says so):

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
         -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
   cmake --build build -j4 --target <name>
   ./build/validation/<name> --kokkos-num-threads=4
   cd build && ctest -R <name> --output-on-failure
   ```

   Run it in the foreground. On macOS a `nohup ... &` job can be throttled to a
   fraction of its speed with nothing in its own output saying so. **Pass options
   directly, never through a shell variable**: zsh does not word-split
   `$args`, so `-lat d3q27` arrives as one unknown option and every misuse test
   reads the same.
10. **Run it on the FP32 build as well** (`-DLBM_PRECISION=float`), because `ctest`
    runs there too and FP32 hits a floor in the finest rows: the dry run's order read
    2.00, 1.62, 0.07 and its round-off gate failed by a factor of 20. Scale tolerances
    by `sizeof(Real)` as `validation/decaying_flows.cpp:154-155` does, or restrict the
    ladder, or print a skip under float; scaling alone may not be enough. Say in the
    banner which precision the gates were measured for.
11. **Write the banner**: the argument, the measured numbers, and what the case does
    *not* test. **Copy the numbers from the output of the final build, run again
    after the last edit; do not round or derive them by hand** (the dry run's banner
    had two orders wrong in the last digit and a run time from the first version).
    A limitation found along the way also goes into `doc/m3lb.tex`'s "Known
    limitations", not only into a commit message.

## Before reporting done

- Say which lattice, operator and precision the figure used.
- Report the PASS/FAIL line verbatim; if a tolerance was loosened, say so and why.
- Run `/check-claim` on any number that will be quoted, and offer a read by the
  `trap-reviewer` agent.
- Do not commit unless asked.
