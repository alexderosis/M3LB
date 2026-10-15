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
   lost minus sign).
2. **Choose the solver** from "Choosing a solver" in [CLAUDE.md](../../../CLAUDE.md).
   Standing rules: no multiphase solver for single-phase flow; "2-D" means a 3-D
   lattice with `nz = 1`; a 3-D phase-field fluid is D3Q27.
3. **Read the banner of every header the case touches.** The banners are the
   documentation, and they often say why the obvious choice is wrong.
4. **Copy the shape of a small existing case**: `validation/tgv2d.cpp` (fluid only,
   convergence ladder), `validation/natural_convection.cpp` (coupled, 126 lines),
   `validation/claudemd_skeleton_check.cpp` (the bare skeleton). Include
   `Campaign.hpp` first; headers are not self-contained and alphabetical order fails
   to compile.
5. **Size it with `/units`.** Resolution, u0 <= 0.05, nu from Re, omega from
   `Coll::omega_from_viscosity`, read tau back and print Ma.
6. **Make the check able to fail.** Delete the term under test and confirm the
   answer moves. Use at least three resolutions before quoting an order, and treat
   an order far above the scheme's as a symptom. A test of an in-place update needs a
   differential run over several steps; two obvious tests are blind to it.
7. **Print PASS/FAIL and return non-zero on failure.** An unrecognised `-lat` or
   `-op` is a hard error (see `Campaign.hpp`), never a quiet fall-through.
8. **Register it.** `validation/CMakeLists.txt` has two `foreach` lists: the first is
   built and run by `ctest`, the second is analysis sweeps too slow to be tests.
   `demonstrator/CMakeLists.txt` has one list and no `add_test`. Add the name to
   the right list and touch nothing else in that file.
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
   fraction of its speed with nothing in its own output saying so.
10. **Write the banner**: the argument, the measured numbers, and what the case does
    *not* test. A limitation found along the way also goes into `doc/m3lb.tex`'s
    "Known limitations", not only into a commit message.

## Before reporting done

- Say which lattice, operator and precision the figure used.
- Report the PASS/FAIL line verbatim; if a tolerance was loosened, say so and why.
- Run `/check-claim` on any number that will be quoted.
- Do not commit unless asked.
