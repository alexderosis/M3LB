---
name: units
description: Convert a physical or dimensionless problem (Re, Ra, Pe, a real length and speed) into M3LB lattice units and check omega, tau and Mach before a run. Use when sizing a case, choosing u0, N or dt, or when a run diverges and tau might be too close to 1/2.
argument-hint: "[the problem, e.g. 'Re 1000 cavity at N 128' or 'Manchester 2 km, 6 m/s']"
---

# Lattice units

The recipe is in [CLAUDE.md](../../../CLAUDE.md) under "From a physical problem to
lattice units". This skill runs it as arithmetic with `units.py` and reads the
result back to the student. Run the helper rather than doing the sums in your head:

```bash
python3 -I .claude/skills/units/units.py dimensionless --N 128 --re 1000          # fluid
python3 -I .claude/skills/units/units.py dimensionless --N 128 --pe 500 --lattice d3q7   # a scalar
python3 -I .claude/skills/units/units.py physical --L 2000 --N 400 --u-phys 6 \
        --nu-phys 1.5e-5 --u-lat-max 0.05 --t-end 720
```

## Procedure

1. **Which kind of problem?** Specified by Re, Ra, Pr, Ha and a resolution:
   `dimensionless`. Geometry carries a real length (`urban`, `height_field`,
   `aorta`): `physical`. If it is neither, ask.
2. **Fix N first.** Cells across the Reynolds length. Wall-bounded cases size the
   domain `ny = H + 2`, and the length is `H`, not `ny`.
3. **Choose u0 <= 0.05** (Ma <= 0.087) unless the student has measured what a larger
   one costs. Viscosity then follows: `nu = u0 N / Re`. The student does not also
   get to pick tau.
4. **Run the helper and read it back**: tau, tau - 1/2, omega, Ma, and both
   timescales. Say them in a sentence. The helper warns on the cases that fail
   silently.
5. **Pick the physical-case strategy deliberately.** Advection-dominated: cap the
   velocity (`--u-lat-max`). Diffusion-dominated: fix tau (`--tau`). They give
   different dt.
6. **In the driver, take omega from the operator**, never from a closed form:
   `coll.omega = Coll::omega_from_viscosity(nu)` (or `omega_from_diffusivity`),
   which reads its own lattice's `cs2`. Keep the closed form for a printed
   diagnostic only.

## Traps to say out loud

- **`cs2` is 1/4 on D3Q7**, 1/3 elsewhere, so `tau = 3 nu + 1/2` is wrong for a D3Q7
  scalar and still compiles. Copying `demonstrator/urban.cpp`'s literal `0.25` onto
  a D3Q27 scalar would be wrong by 4/3.
- **Halving u0 at fixed Re and N hurts twice**: twice the steps, and tau moves toward
  1/2. Raise N instead.
- **A coefficient fixed in lattice units while dt moves silently rescales the
  physics.** Derive each one from its SI value; check the reported result is
  independent of the sub-cycling.
- **Match kinematic viscosity across a density ratio, not dynamic.**
- **A one-convective-time run is not a converged run** for a diffusion-limited
  quantity; the helper prints both times. Quote a result after the right one.

`examples/flow_past_square.cpp:92-118` is this recipe as running code, including the
tau floor check and the Mach print.
