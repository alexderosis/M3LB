# Nianzhi's mixer: topology optimisation with a discrete adjoint

*Design proposal, 2026-10-15, branch `optimisation`. P1 (the forward
evaluator) is implemented; section 10 says where it stands. A number marked
"estimate" is arithmetic, not a run.*

## 1. The problem, as far as it is pinned down

Nianzhi's figure (sent 2026-10-11; the image is no longer on disk, so ask for it
again and keep it in the repository) shows a square duct of side L: an inlet
section of length L, a design domain 2.5L long, full width and height H, sitting
on the floor, and an outlet section of length L. Two streams of the same liquid
enter with c = 0 and c = 1 either side of y = L/2. The mixer is passive and c is
a label, so this is `FluidSolver` + `ScalarSolver`, not a multiphase solver.

Decided 2026-10-15: **the gradients come from a discrete adjoint**, and the
mixing measure must be trustworthy for **Pe up to about 10^3**. Still open: H,
Re, the pressure-drop budget, the exact mixing measure, the inlet profile, and
who runs the optimisation loop (section 9).

## 2. Formulation

- **Design variables.** One gamma in [0, 1] per lattice cell of the design
  domain (1 = fluid). It is filtered (density filter, radius about 2-3 cells)
  and projected (smoothed Heaviside, beta continuation) before use. The chain
  rule through the filter and the projection belongs to the optimiser, not to
  the solver.
- **Brinkman penalisation through `DarcyGuo`**, the implicit drag that is stable
  at any A: A(gamma) = A_max q (1 - gamma) / (q + gamma) (RAMP). A_max comes from
  a Darcy number, A_max = nu / (Da L^2) in lattice units, and is RE-DERIVED
  whenever nu or L changes. Holding A fixed while nu moves is the `melt_pool`
  mistake (CLAUDE.md, "a coefficient spelled in lattice units"). Estimate: Da =
  1e-5 at L = 40, Re = 10 gives A_max = 6.25.
- **Flow.** D3Q27 with `CentralMoments<D3Q27, DarcyGuo>`, the standing default,
  on raw populations. The duct walls use halfway bounce-back. The inlet is
  `RegWall`/`NrmXm` carrying the fully developed square-duct profile (the series
  solution, which the empty-duct check needs anyway), and the outlet is
  `NrmOutXp` at rho = 1. The design domain stops a full L short of both faces,
  so the drag never reaches a `RegWall` node, where `FluidSolver`'s banner
  records a body-force slip.
- **Concentration.** D3Q7 with `ScalarTRT` at Lambda = 1/4 (changed in P1, see
  section 10; the plan first named `ScalarRegularised`, which diverges here);
  omega_c = 1.984 at Pe = 10^3, U = 0.025, L = 40. The inlet is Dirichlet, with the step smoothed over
  about 2 cells, the outlet is `ScalarOutflow`, and the duct walls are zero flux.
  D is uniform, so c also diffuses into the Brinkman "solid". That is a known
  artefact of density methods. The per-node rate (`omega_at`) is the fix if it
  matters, and P6 measures whether it does.
- **Objective.** The flux-weighted variance on a plane x_m a few cells upstream
  of the outlet:
  Phi = sum u_x (c - cbar)^2 / sum u_x, with cbar = sum u_x c / sum u_x,
  reported as the mixing index M = 1 - sqrt(Phi / (cbar (1 - cbar))).
- **Constraint.** Dp = cs2 (<rho>_1 - <rho>_2) between two interior planes, with
  Dp <= beta Dp_0, where Dp_0 is the empty duct's.

## 3. Why a discrete, steady-state adjoint

**Discrete** because it is the exact derivative of the code that actually runs.
Its gradient can therefore be checked against finite differences down to the
forward solve's own convergence tolerance. Luo, Chen, Yaji & Tao (2025) also
report that the continuous adjoint LBM is inconsistent for OPEN flows, and they
derive consistent boundary conditions from the discrete adjoint instead. An
inlet plus an outlet is exactly that case.

**Steady-state (fixed-point)** because the answer is a steady state. With
f* = K(f*; A) the forward fixed point, the adjoint solves

    lambda = (dK/df)^T lambda + dJ/df

by the same kind of iteration as the forward. dK/df has the forward's spectrum,
so the adjoint converges at the forward's asymptotic rate, and only f* is kept.
No time history is stored; an unsteady adjoint has to keep or recompute one, and
Chen et al. (2017) exist to reduce exactly that storage.

Per design iteration:
1. Forward flow.
2. Forward scalar, with u frozen.
3. Scalar adjoint, with source dPhi/dc.
4. Flow adjoint for Phi. Its sources are dPhi/du_x on the plane and the scalar
   adjoint, which reaches the flow through u.
5. Flow adjoint for Dp.

The two flow adjoints are linear in their sources, so one sweep can advance
both. The design gradient is then dJ/dA_n = lambda^T dK/dA_n + eta^T dG/dA_n,
followed by the RAMP derivative.

## 4. What has to be transposed

| piece | in the forward | transpose | difficulty |
|---|---|---|---|
| pull streaming + halfway bounce-back | `run_step`'s loads | a permutation, so push the other way | trivial |
| `DarcyGuo` velocity | u = (m + F_ext/2)/(rho + A/2) | du_a/df_i = (c_ia - u_a)/(rho + A/2), and d/dA for the design | easy |
| CM collision | f+ = T(u)^-1 [D T(u) f + E F(u)] | T^T D T^-T lambda plus a rank-3 term (below) | moderate |
| `RegWall` velocity inlet | `BC::density` + `BC::apply` | LINEAR in f at fixed u_w, so an exact 27x27, probed with unit vectors once | easy |
| `NrmOutXp` outlet | rho imposed, u_n from the closure, u_t from the upstream node | nonlocal and nonlinear; analytic Jacobian, scattered in its own pass | **the hard one** |
| `ScalarTRT` (was `ScalarRegularised`) | linear in h at fixed u, bilinear in (c, u); flux at omega, ghosts at omega_plus | 7x7 transpose; its d/du feeds the flow adjoint | easy |
| scalar inlet, walls, `ScalarOutflow` | affine / permutation / donor equilibrium | its own second pass, as in the forward | easy |
| objective, constraint | Phi(c, u_x) on a plane; Dp linear in f | sources | easy |

**The central-moment transpose.** In this basis phi_p(c - u) has
dphi_p/du = -p phi_(p-1), so dT/du_a = -N_a T, where N_a raises the order along
axis a. With mu = T^-T lambda and k = T f:

    J^T lambda = T^T D mu + sum_a (du_a/df) s_a,
    s_a = mu . ( [N_a, D] k + N_a E F + E dF/du_a ).

Each of `fwd1d` and `inv1d` transposes into its own 3x3 pass, applied in reverse
order, so the whole thing costs about two forward collisions (an estimate). The
moment indices must stay compile-time constants; `tests/frame_check.sh` checks
that after the change.

**The outlet is the only piece with a real risk.** If its analytic Jacobian costs
too much, the fallback is a local outlet for this case only (u_t = 0). It would
be adopted only after measuring its effect on Phi against `NrmOutXp`. Luo et al.
derive discrete adjoint boundary conditions for open flows, so read them first.

## 5. Architecture: opt-in, and free when off

- **New `src/adjoint/`.** `AdjointFluid.hpp` and `AdjointScalar.hpp` hold the
  adjoint fields in two plain arrays, the push kernel, the sources and the
  sensitivity. `AdjointOf<Coll>` is specialised for `CentralMoments` and
  `ScalarTRT` (P1 replaced `ScalarRegularised`, which diverges here). Any other operator, or shifted storage, is a compile
  error rather than a wrong gradient.
- **`FluidSolver` and `ScalarSolver`** gain const accessors to their flags and
  wall tables. Their kernels are not touched. The adjoint reads f* only through
  `gather()`, never through the Esoteric Pull storage.
- **Driver `demonstrator/mixer_topopt.cpp`.** gamma comes in as float32 `.npy`,
  which the existing `<f4` reader already handles. Phi, Dp, dPhi/dgamma and
  dDp/dgamma go out the same way.
- **The gate for "free when off"** is `ctest` plus one validation case's output
  diffed before and after. Each phase ends with the `trap-reviewer`.

## 6. Phases, and gates fixed before their runs

**P1, the forward evaluator.**
- The empty duct's Dp against the exact series solution, on two grids,
  converging at second order.
- The plug-flow mixing layer (free-slip duct, uniform inflow) against the erf
  solution, point by point.
- A solid slab at A_max, whose leak stays under a fraction fixed beforehand.
- One fixed grey design on two grids at fixed Da, with Phi inside a band fixed
  beforehand.

**P2, the operator transposes** (`tests/test_adjoint_ops.cpp`). For every piece,
check <J v, w> = <v, J^T w> on random states, with J v from central differences
of the FORWARD code. The tolerance is 1e-8 relative in FP64; linear pieces must
agree to 1e-13.

**P3, the adjoint solvers.** The scalar first, with u fixed, then the flow.

**P4, the end-to-end gradient** (`validation/adjoint_mixer.cpp`, run by ctest).
L = 12, a random grey gamma, the forward converged to round-off. The adjoint
gradient must match central finite differences on 12 design cells, for Phi and
for Dp, to 1e-5 relative. Then delete the scalar-to-flow coupling and confirm
that the gate FAILS.

**P5, the optimiser.** Filter, projection and MMA (Svanberg 1987), with warm
starts between iterations.

**P6, production at L = 40.** The thresholded design is then re-run with Solid
voxels and bounce-back at 1.5L, and M must hold inside a band fixed beforehand.
That is the guard against an optimiser that has exploited Brinkman leakage or an
under-resolved scalar.

## 7. Cost (estimates; P1 measures them)

The basis is the 19.4 MLUPS measured 2026-10-11 (D3Q27 CM, FP64, 4 threads).
At L = 40:
- **Domain:** about 3.2e5 nodes, with nu = 0.1 (tau = 0.8). The diffusive time
  L^2/nu is 16 000 steps.
- **Forward, from cold:** about 13 min to three diffusive times.
- **Each flow adjoint step:** about 1.5-2x a forward step.
- **One design iteration:** about 1 h from cold, and several times less once
  warm-started.

That makes 100-200 iterations take one to two days per optimisation on the laptop
at L = 40. L = 64 costs about 6.6x more (L^4) and belongs on CSF3 CPU nodes.
`GPU/` cannot run it: it has no fluid outlet and no adjoint.

## 8. Traps specific to this problem

- **The Brinkman solid's no-slip plane depends on tau:** +0.81 cells on CM at
  tau = 0.8, per `DarcyGuo`'s banner. Hold Da fixed across grids, not A. P6
  re-runs the design with bounce-back.
- **`DarcyGuo`'s per-step factor (2 - A)/(2 + A) tends to -1 as A grows.** The
  raw momentum inside the solid then alternates and decays slowly, both in the
  forward residual and in the adjoint, which has the same spectrum. The factor
  is -0.52 at A = 6.25 and -0.94 at A = 62.5.
- **The optimiser will find numerical diffusion.** At Pe = 10^3 and L = 40, one
  pass leaves a mixing layer about L sqrt(4.5/Pe) = 2.7 cells wide, at a cell
  Peclet number of 25.
- **The finite-difference check needs the forward converged to round-off,** on
  the whole field and over an interval. A per-step residual is not enough.
- **Stay in FP64 until P4 passes.** An FP32 forward under an FP64 adjoint is a
  different function.

## 9. Decisions

From Nianzhi. None of these blocks P1-P4, where each one is a parameter:
1. H, Re and beta.
2. Whether Phi above is the measure they want, and which inlet profile.
3. The figure, and the paper it comes from.

From us, by P5:
4. Who owns the loop. One option is a C++ MMA inside the driver, which keeps
   Python off the cluster. The other is Nianzhi's own code calling the driver for
   (Phi, Dp, gradients). P1-P4 build the driver either way.

## References

Checked 2026-10-15 against publisher or index records. Full texts were NOT read,
so each one is cited only for what its abstract or record says.

- Luo, Chen, Yaji & Tao (2025), *Improved adjoint lattice Boltzmann method for
  topology optimization of laminar convective heat transfer*, Int. J. Heat Mass
  Transf. 251, 127315, doi:10.1016/j.ijheatmasstransfer.2025.127315; preprint
  arXiv:2501.02161, *Unveiling the gap between continuous and discrete adjoint
  lattice Boltzmann methods*.
- Pingen, Evgrafov & Maute (2007), *Topology optimization of flow domains using
  the lattice Boltzmann method*, Struct. Multidisc. Optim. 34(6), 507-524.
- Pingen, Evgrafov & Maute (2009), *Adjoint parameter sensitivity analysis for
  the hydrodynamic lattice Boltzmann method with applications to design
  optimization*, Comput. Fluids 38(4), 910-923.
- Laniewski-Wollk & Rokicki, *Adjoint Lattice Boltzmann for Topology Optimization
  on multi-GPU architecture*, arXiv:1501.04741. A discrete adjoint for a wide
  class of LBMs, used to design a mixer in channel flow.
- Chen, Yaji, Yamada, Izui & Nishiwaki (2017), Mechanical Engineering Journal. A
  local-in-time discrete adjoint for unsteady LBM.
- Liu, Geier, Liu, Krafczyk & Chen (2014), *Discrete adjoint sensitivity analysis
  for fluid flow topology optimization based on the generalized lattice Boltzmann
  method*. Venue not checked.
- Andreasen, Gersborg & Sigmund (2009), *Topology optimization of microfluidic
  mixers*, Int. J. Numer. Meth. Fluids 61(5), 498-513, doi:10.1002/fld.1964.
  Later work summarises its optimised designs as grooved; that summary has not
  been checked against the paper.
- Svanberg (1987), *The method of moving asymptotes -- a new method for
  structural optimization*, Int. J. Numer. Meth. Eng. 24(2), 359-373,
  doi:10.1002/nme.1620240207.

## 10. P1 status (2026-10-16)

The thresholds were registered at 12:25 UTC before any run;
`results/mixer_forward/registration/REGISTERED.txt` holds them, the three gates
added later (each before its own first run), every code change made after
registration, and every run's verdict. In brief:

- **Check 1, empty duct: PASS.** G against the exact series at L = 8..64:
  +1.24e-2, +2.42e-3, +5.65e-4, +1.38e-4; order 2.36, 2.10, 2.03.
- **Check 2, plug-flow mixing layer: FAIL on one threshold**, the spreading rate,
  under either operator: ScalarRegularised 1.0327, ScalarTRT 0.9846 at L = 40,
  against 1 +/- 0.01. Over the range where the statistic's window spans 4 sigma
  (ungated) they read 1.0061 and 0.9945. Under TRT the pointwise error halves
  (2.99e-3), converges at order 1.96-1.98, and c stays in [0, 1].
- **Check 3, solid slab: FAIL at L = 16, PASS at L = 40.** The leak is grid-
  independent (9.75e-4, 9.81e-4). Darcy's law misses by 14% at the centre of the
  8-cell slab: the drag region carries an odd-even velocity layer next to every
  solid surface, decaying x(-0.42) per cell at tau = 0.62 and x(-0.19) at 0.8.
- **Check 4, two grids: PASS on the third attempt.** M 0.164867 / 0.164240,
  Phi 0.1743615 / 0.1746236, Dp/Dp0 3.92972 / 3.94161 at L = 40 / 60 -- 0.38%,
  0.15% and 0.30% apart -- with the converged field in [-0.0027, 1.0927] and
  [-0.00002, 1.0023]. The first attempt (ScalarRegularised) DIVERGED: that
  operator is linearly unstable above |u| = 0.052 at Pe = 10^3, L = 40, below the
  empty duct's own centreline speed; hence ScalarTRT at Lambda = 1/4 (stable to
  |u| = 0.5 at any D). The second (TRT, uncorrected) HALTED, the field settling
  at 1.42; hence the correction below.
- **The fluid's velocity is not divergence-free for the scalar's stencil where
  the drag varies over a cell** (0.145 U per cell at the most on the L = 16
  grooves, and more than |u| itself at near-stagnant drag nodes), so the
  conservative scalar piles c up where the flow converges. A UNIFORM inlet came
  out between 0.70 and 1.44 at L = 40, with a false outlet Phi of 6.7e-5. The fix
  chosen 2026-10-15 is an exact uniform-preserving correction
  (`ScalarSolver::set_uniform_correction`, calibrated per flow by
  `MixerEvaluator::calibrate_uniform`): **check 6 (added) PASSES** -- a uniform
  inlet stays uniform to 2.3e-10 at L = 16 (the calibration's accuracy, not
  round-off), outlet Phi 1.9e-22, against 1.24 and 2.8e-3 without it. It holds
  for uniform fields only.
- **Check 5 (added), L = 16 grooves: FAIL.** With the correction the TRT
  converges, but to c in [-0.084, 1.727]: the maximum inside a near-stagnant
  solid ridge (divergence 310% of |u| per cell), the minimum in open fluid at a
  local cell Peclet number near 190. L = 16 is under-resolved at Pe = 10^3; the
  check-4 rerun says whether L = 40 is.

**What the correction changes, converged (grooves, L = 16, 24, 32, 40).** With
it Phi = 0.17443, 0.17416, 0.17423, 0.17436, grid-independent to 0.12% over
L = 24..40. Without it 0.15209, 0.14934, 0.14872, 0.14868 -- also grid-converged,
to a limit 15% lower. Most of that is COMPRESSIBILITY, not the drag: at
U = 0.025 the density falls ~8.5% from the inlet to the mixing plane, the fluid
conserves rho u rather than u, and the scalar is diluted by the density ratio
(cbar 0.461 instead of 0.5) and Phi by its square -- (cbar_on/cbar_off)^2 matches
Phi_on/Phi_off to 0.03% at L >= 32. The drag part is 1.2%, 0.17%, 0.03%, 0.03%,
a discretisation error that refines away. Two consequences for P2: a lower U
would shrink the dilution as Ma^2, and advecting the scalar with rho u / rho_0
would remove it without a calibration -- worth weighing against differentiating
through the calibration.

**What the correction means for the adjoint (P3).** The factor depends on the
velocity, through its calibration -- a linear fixed point in the scalar
populations. The gradient must therefore include dJ/dfix . dfix/du, which is one
more adjoint solve of the scalar kind per design. Nothing else in the inventory of
section 4 changes.

**Where P1 leaves the plan.** The evaluator is usable at the production grid
L = 40: grid-converged against L = 60 to well inside its bands, with a residual
overshoot of 9% (its location at L = 40 is not yet recorded; at L = 16 the
maximum sits in a near-stagnant drag region). Three registered thresholds stay failed
(check 2's spreading rate, check 3 at L = 16, check 5 at L = 16), each for a
measured reason above; none was moved. Before P2: the flow's density falls 8%
along the duct at U = 0.025 (Ma_max 0.175), which a fixed-U grid study cannot
see, and the plan's estimate of the scalar cost was low -- with the solid's
diffusive time the corrected L = 40 solve took 385k steps (~20 min).

