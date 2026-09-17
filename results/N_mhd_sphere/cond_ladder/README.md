# The penalised conducting wall's resolution ceiling

Raw stdout of every run behind the "penalised perfectly conducting wall has a
resolution ceiling" entry in `doc/m3lb.tex`'s known limitations. Each `.dat` is
the driver's own output: a parameter header, then the diagnostic table whose
columns are named in the header line.

Two ladders: the default one, and a scaling control where the penalisation is
scaled with N (see the last section -- it is the one that answers "did you scale
anything?"). All runs are `GPU/src/mhd_sphere.cu` on a Tesla T4, FP32, D3Q27 fluid + D3Q7
magnetic, operator `cm`, built `-DLBM_ONLY_OP=cm -DLBM_ONLY_FORCE=field`:

    mhd_sphere -n N -re 500 -kmax 6 -k0 4 -steps 32N -probe 32N/256 -mwall cond

`32N` steps is exactly two turnovers, since `R = 0.4N` and `T_e = 2R/u0` with
`u0 = 0.05`. Everything but the swept parameter is identical, seed included.

## The ladder — `cond_<N>.dat`

| N | nu | omega | outcome | blow-up |
|---|---|---|---|---|
| 96 | 7.68e-3 | 1.912 | **survives 2 T_e** | — |
| 128 | 1.024e-2 | 1.884 | non-finite | step 448, t/Te 0.219 |
| 160 | 1.280e-2 | 1.857 | non-finite | step 460, t/Te 0.180 |
| 192 | 1.536e-2 | 1.831 | non-finite | step 432, t/Te 0.141 |
| 224 | 1.792e-2 | 1.806 | non-finite | step 420, t/Te 0.117 |

N = 224 is `sphere_n224_re500_insul_d3q27_cms.dat`'s conducting twin; that
insulating run survives every N, so the wall flag is the one variable.

THE BLOW-UP STEP IS FIXED, THE BLOW-UP TIME IS NOT. 440 +- 20 steps at every N
while t/Te falls monotonically 0.219 -> 0.117. The timescale belongs to the
lattice, not to the flow.

IT IS NOT omega -> 2, AND THE LADDER RUNS THE WRONG WAY TO READ IT THAT WAY.
`nu = u0 2R/Re` grows with N, so omega *falls* from 1.912 to 1.806 across the
ladder. A relaxation-rate instability would have taken the coarse grids first.

IT IS NOT A PORT DEFECT. At N = 96 this CUDA run agrees with its Kokkos parent
(`../sphere_n96_re500_cond_d3q27_cms.dat`, no shared headers) to -0.008 % on
E_u and -1.6 % on E_b at t/Te = 2, with Bn/B equal to four decimals at every
sampled point. Both codebases solve the same problem; both would clear N = 96.

## The sweep at N = 128 — `sw_*.dat`

The cheapest failing point, one parameter at a time.

| eps_m (smooth=1) | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|
| | fail @208 | fail @288 | fail @448 | survives | survives |

| smooth (eps_m=2) | 1.0 | 1.333 | 2.0 | 2.667 |
|---|---|---|---|---|
| | fail @448 | fail @480 | survives | fail @672 |

`eps_m` is monotonic and is the explicit penalisation's relaxation time in
lattice steps -- the `epsilon >~ dt` bound of the neighbouring limitation entry,
with the threshold no longer O(1). `smooth` is NOT monotonic and must not be
read as a control; 2.667 failing after 2.0 survives is unexplained.

MATCHING THE LAYER AS A FRACTION OF R DOES NOT HELP. `smooth`=1.333 at N=128 is
2.6 % of R, the same fraction as `smooth`=1 at N=96, and it still dies (step
480). What matters is an absolute number of cells -- a grid-scale signature, not
an under-resolved wall layer.

Accuracy cost of the two survivors, at t/Te = 2:

| run | E_u | E_b | Bn/B | B_sh |
|---|---|---|---|---|
| N=96, eps_m=2, smooth=1 | 5.556e-06 | 2.501e-05 | 0.0005 | 1.161 |
| N=128, eps_m=2, smooth=2 | 4.733e-06 | 2.528e-05 | 0.0001 | 1.195 |
| N=128, eps_m=4, smooth=1 | 5.639e-06 | 2.535e-05 | 0.0019 | 1.151 |
| N=128, eps_m=8, smooth=1 | 5.554e-06 | 2.569e-05 | 0.0046 | 1.149 |

Softening with eps_m degrades `B.n = 0` about linearly. `smooth` = 2 does not.

## The transfer test at N = 224 — `scr_*.dat`

1024 steps, i.e. 2.4x past where the default dies. NONE of the N = 128 fixes
survive:

| config | blow-up |
|---|---|
| smooth=2.0 | step 448 |
| smooth=2.5 | step 544 |
| eps_m=4.0 | step 672, and Bn/B has degraded 0.0005 -> 0.0141 |

They buy delay proportional to the softening and nothing else. This is why the
limitation is written as a ceiling rather than as a parameter to tune: by the
time eps_m is large enough to matter it is no longer the boundary condition
asked for. THE MECHANISM IS NOT DIAGNOSED.

## The scaling control — `phys_<N>.dat`

THE LADDER ABOVE HOLDS THE PENALISATION FIXED IN LATTICE UNITS, WHICH IS NOT
THE SAME AS HOLDING THE BOUNDARY CONDITION FIXED. `eps`, `epsm` and `smooth`
are a time in steps and a length in cells. Since `T_e = 16N` steps,

    eps_m / T_e  =  2 / 16N  =  1/(8N)

so refining at `epsm = 2` makes the wall PHYSICALLY STIFFER as N grows -- 2.33x
stiffer at N = 224 than at N = 96 -- and stiffness is the one knob the sweep
above says is fatal. That is a second variable, and it moves the wrong way. The
first version of this note did not test it.

So: repeat the ladder with all three scaled by N/96, which is the same physical
problem refined rather than a stiffer one.

| N | factor | eps = epsm | smooth | outcome | blow-up | (unscaled) |
|---|---|---|---|---|---|---|
| 128 | 1.333 | 2.667 | 1.333 | non-finite | step 432 | 448 |
| 160 | 1.667 | 3.333 | 1.667 | non-finite | step 440 | 460 |
| 192 | 2.000 | 4.000 | 2.000 | non-finite | step 408 | 432 |
| 224 | 2.333 | 4.667 | 2.333 | non-finite | step 420 | 420 |

ALL FOUR STILL FAIL, and the blow-up step moves by at most 4 % -- inside the
tree's own rule that a blow-up step is not a reproducible metric. The
instability is INDIFFERENT to this whole family of penalisation settings, so
the ceiling is not an artefact of freezing them in lattice units.

AND IT SHARPENS THE MECHANISM. At N = 128 the physical match epsm = 2.667 FAILS
while epsm = 4 survives; at N = 224 the physical match 4.667 fails and so does
4. The epsm needed for stability therefore grows FASTER than linearly in N, so
no fixed physical wall stiffness survives refinement: stability can only be
bought by making the wall progressively softer THAN PHYSICAL, which is giving up
the boundary condition rather than imposing it. That is the strongest statement
the data supports, and it is why this is written as a ceiling.

What is still NOT established: the ladder is ACOUSTIC (u0 = 0.05 fixed), so
Ma = 0.0866 at every N and the compressibility error does not vanish under
refinement. This is a stability ladder, not a convergence study. A diffusive
ladder (u0 ~ 1/N, nu fixed, steps ~ N^2) would separate the two and has not
been run.
