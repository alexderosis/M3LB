# Neffaa, Bos & Schneider (2008) — the four Table I regimes, reproduced and compared

*Phys. Plasmas* **15**, 092304 (2008), *The decay of magnetohydrodynamic turbulence
in a confined domain*. Source PDF in `SomeRefs/Neffaa_POF_2008.pdf`.

```
mhd_decay -regime 1..4 -seed 99 -n 321 -steps 304950 -probe 6099 \
          -bin <dir> -binevery 3050 --kokkos-num-threads=3
```

N = 321, r = (19/20)π in a 2π box, Re = 1000, Pm = 1, penalised walls (B·n = 0),
50 domain crossings = their t = 450 on the advective clock, 100 frame pairs each.
Films in `doc/video/neffaa_regime[1-4].mp4` (untracked, as every video here is);
stills at t/T_e = 0.5, 4.5, 50 beside this file. Run logs are `regime[1-4].dat`;
`compare.py` regenerates `comparison.txt` from them against
`neffaa_reference.json`.

## The clock, and a correction that changed the result

**E_u is the total kinetic energy**, `½∫|u|²d²x` over V_f — Eq. (7) says so. The
velocity scale is therefore `u_rms = √(2E_u/A)` with A = πr² = 27.983, *not*
`√(2E_u)`. One eddy turnover is `2r/u_rms` in the paper's units and `2R/u0` in
ours; those are the same dimensionless turnover, so crossings map one-for-one and
only the scale differs:

| regime | u_rms (paper) | T_e (paper units) | our 50 crossings reach | their t = 450 needs |
|---|---|---|---|---|
| I | 0.1035 | 57.65 | t = 2882 | 7.8 crossings |
| II | 26.06 | 0.229 | t = 11 | 1964 crossings |
| III | 0.2155 | 27.69 | t = 1385 | 16.2 crossings |
| IV | 0.1890 | 31.58 | t = 1579 | 14.3 crossings |

The first version of this comparison used 9.0 paper units per crossing for all
four regimes. That came from reading Table I's Re column as the velocity scale —
but that column is the printed formula `2r√(2E_u)/ν` taken literally with E_u a
total, which carries units of **length** and equals the physical Reynolds number
times √A = 5.29. The consequence was that the runs were read **6.4×, 3.1× and
3.5× too late** for regimes I, III and IV, and 39× too early for II. Every
time-series number below moved when that was fixed, and mostly toward the
reference.

**The Reynolds gap also does not exist.** On the same correction the reference's
*physical* Reynolds numbers are

| regime | their physical Re | ours | |
|---|---|---|---|
| I | 618 | 1000 | we are 1.6× **above** |
| III | 1287 | 1000 | 1.3× below |
| IV | 1128 | 1000 | 1.1× below |

so the campaign ran at essentially the reference's Reynolds number throughout.
The earlier claim of "Re = 1000 against their 3868–7920" was comparing our
physical Re against their length-dimensioned column. Regime II is irreconcilable
on any reading (155,536 physical, 822,773 literal, against 7920 printed).

## Where the reference numbers come from

Table I and the paper's textual claims are exact. Fig. 2 and Fig. 4 were
**pixel-digitised** from the PDF, and the digitisation carries its own check: the
t = 0 column of Fig. 2's centre panel reads 16.1 / 3.43e5 / 30.4 / 16.3 against
Table I's 16 / 3.4e5 / 31 / 16. Reproducing four values across five decades to
1–3 % pins the axis maps and the colour assignment at once. Fig. 2's bottom panel
independently lands regime III on 2.04 against the paper's stated minimum of
exactly 2. **Fig. 2's time axis is logarithmic**, which is easy to get wrong.

Fig. 4's numbers have only qualitative text to check against, so they are the
least certain of the digitised set and are marked as such below. **Fig. 6 was not
digitised**; its comparison uses the paper's text values, which are exact.

## 1. The initial condition — a check on the construction, not a result

| regime | quantity | ours | paper | ratio |
|---|---|---|---|---|
| I | E_u/E_B | 0.300 | 0.3 | 1.000 |
| I | cos θ | 0.04382 | 0.04382 | 1.000 |
| I | **E/A** | **5.49** | **16** | **0.34** |
| II | E_u/E_B | 1.9e4 | 1.9e4 | 1.000 |
| II | cos θ | 1e-6 | 5.08e-7 | — |
| II | **E/A** | **7.96e4** | **3.4e5** | **0.23** |
| III | E_u/E_B | 1.300 | 1.3 | 1.000 |
| III | cos θ | 0.4736 | 0.4736 | 1.000 |
| III | **E/A** | **10.6** | **31** | **0.34** |
| IV | E_u/E_B | 1.000 | 1.0 | 1.000 |
| IV | cos θ | 0.0900 | 0.090 | 1.000 |
| IV | **E/A** | **8.51** | **16** | **0.53** |

Both columns of Table I are now hit to four figures, because Eq. (11) is
implemented and the cross-helicity is *imposed* rather than drawn — see the
commit `Both columns of Table I...`. That is a statement about the setup. **E/A
is the one initial quantity that can disagree**, being an outcome of the
spectrum, and it does: low by a factor 1.9 to 4.3.

**That shortfall is not ours. Eq. (6) itself does not produce Table I's E/A.**

E/A follows from the spectrum analytically, with no run involved. Since
**b** = ∇×a, the shell spectrum of a is E_b(k)/k², so

    E/A = (1 + alf) · Σ E_b(k) / Σ E_b(k)/k²

— a weighted mean square wavenumber. The paper's box is 2π, so the mode index is
the wavenumber itself and no unit conversion enters. `spectrum_check.py`
evaluates it; the result is that Eq. (6) predicts **E/A = 4.76** for regime I,
against Table I's 16, and the prediction is *converged* in the spectrum cut
(4.58 at kmax = 24, 4.76 at 170, 4.76 at 400).

Against that prediction our initial condition is faithful:

| regime | ours | Eq. (6) | ours/Eq. (6) | Table I | Eq. (6)/Table I |
|---|---|---|---|---|---|
| I | 5.49 | 4.58 | 1.20 | 16 | 0.29 |
| II | 7.96e4 | 6.69e4 | 1.19 | 3.4e5 | 0.20 |
| III | 10.6 | 8.10 | 1.31 | 31 | 0.26 |
| IV | 8.51 | 7.04 | 1.21 | 16 | 0.44 |

We sit **1.2–1.3× above Eq. (6)** — finite-mode realisation scatter, consistent
across all four regimes. Eq. (6) sits at 0.20–0.44× of Table I. **So the
disagreement is between Table I and Eq. (6), not between this tree and the
paper.**

Three further readings of Eq. (6) were tested and none reaches Table I. Taking it
as the *enstrophy and current* spectrum makes it **worse** (0.07–0.16×), which is
obvious once E/A is recognised as a mean square wavenumber: moving the spectrum
onto ω and j removes two powers of k from **b** and smooths the field, where
raising E/A needs more small-scale magnetic power. Taking it as the spectrum of
the *potentials* overshoots by 8–18×. The gauge cannot close it either, and for
the same directional reason — a non-zero mean in a only ever *increases* A, and
zero mean already minimises it, so no gauge choice raises E/A at all.

Solving instead for the spectrum Table I would need gives E_b(k) = k^0.86 ×
Eq. (6), i.e. a large-k slope of about **k^−2.1** — against the **k^−3** the
paper states in the sentence immediately following Eq. (6). The four regimes
agree on that exponent independently (p = 0.86, 1.06, 0.91, 0.60).

**Conclusion: this is an inconsistency in the reference, not a defect here.** Our
initial condition implements Eq. (6) as written, and every reading that would
raise E/A contradicts the stated k^−3 tail. Changing the spectrum to chase
Table I would mean abandoning the equation the paper gives.

## 2. E_u/E_B — Fig. 2, top

| regime | t=40 | | t=100 | | t=200 | | t=450 | |
|---|---|---|---|---|---|---|---|---|
| | ours | paper | ours | paper | ours | paper | ours | paper |
| I | 0.0479 | 0.0618 | 0.0096 | 0.0323 | 0.0081 | 0.0214 | 0.0956 | 0.0202 |
| II | — | 3.72e6 | — | 3.10e7 | — | 6.41e7 | — | 1.01e8 |
| III | 0.341 | 0.811 | 0.148 | 0.768 | 0.127 | 0.769 | 0.0352 | 0.717 |
| IV | 0.179 | 0.0757 | 0.0198 | 0.0560 | 0.0140 | 0.0437 | 0.0233 | 0.0347 |

**Regime II cannot be compared at all.** Its velocity scale puts t = 450 at 1964
crossings where the campaign ran 50, so the run reaches only t = 11 — every
reference time is out of range. That is a limitation of the campaign's duration,
not a disagreement, and the earlier write-up reported a "qualitative
disagreement" here that was an artefact of the wrong clock.

Regime I agrees within 1.3× at t = 40 and then falls below; the late rise to
0.0956 is ours alone. Regime IV brackets the reference — 2.4× high at t = 40,
then 2.8× low by t = 100. Regime III runs 2.4–6× low throughout and does not hold
the equipartition its regime is defined by.

## 3. E/A — Fig. 2, centre

| regime | t=40 | | t=100 | | t=200 | | t=450 | |
|---|---|---|---|---|---|---|---|---|
| | ours | paper | ours | paper | ours | paper | ours | paper |
| I | 1.93 | 1.61 | 1.59 | 1.36 | 1.41 | 1.29 | 0.678 | 1.29 |
| II | — | 9.40e7 | — | 3.64e8 | — | 3.55e8 | — | 3.34e8 |
| III | 3.19 | 5.43 | 1.61 | 3.31 | 1.13 | 2.84 | 0.489 | 2.55 |
| IV | 2.40 | 2.69 | 1.48 | 2.61 | 1.06 | 2.35 | 0.536 | 1.78 |

**Regime I now agrees to 17–20% over t = 40 to 200** — 1.93/1.61, 1.59/1.36,
1.41/1.29 — where the old clock made it look 3× low. Regime IV agrees to 11% at
t = 40 and drifts to 2.2× low. The absolute offset at t = 0 discussed in §1 is
therefore largely worked off during the run rather than persisting.

## 4. cos θ — Fig. 4 *(reference values digitised, unverified)*

| regime | t=100 | | t=200 | | t=450 | |
|---|---|---|---|---|---|
| | ours | paper | ours | paper | ours | paper |
| I | −0.439 | 0.092 | −0.311 | 0.657 | −0.152 | 0.906 |
| II | — | 0.193 | — | 0.178 | — | 0.316 |
| III | **0.866** | 0.733 | **0.691** | 0.740 | 0.360 | 0.847 |
| IV | −0.064 | −0.095 | −0.196 | −0.162 | −0.142 | −0.201 |

**Regime III's dynamic alignment is largely reproduced**, which the wrong clock
had hidden. At t = 100 we read 0.866 against 0.733 — 18% *high*; at t = 200,
0.691 against 0.740 — 6.6% low. The alignment is sustained through the window the
paper plots, and only at t = 450, some 28× further into the decay than our
t = 0, does ours fall to 0.360 against 0.847.

**Regime IV's negative alignment is reproduced to within 33% at every time** —
(−0.064, −0.095), (−0.196, −0.162), (−0.142, −0.201) — including the sign, which
is the paper's cancelling-sub-regions mechanism.

**Regime I is the genuine disagreement.** The reference climbs to 0.906; ours goes
negative and stays there. Being 1.6× *above* the reference's Reynolds number, that
cannot now be blamed on insufficient Re.

## 5. Energy decay — Fig. 6 *(paper values from the text, exact)*

| regime | exponent, ours | paper | late α, ours | paper |
|---|---|---|---|---|
| I | n/a | −0.6 | 0.81 | 1.5 |
| II | n/a | — | 1.76 | 2 |
| III | −0.80 | — | 0.78 | 2 |
| IV | **−0.70** | **−0.4** | 0.77 | 2 |

E ~ exp(−2ανt); the circle's largest Stokes eigenvalue is α = 1.64, stated exactly
in `validation/stokes_disc.cpp` from the Bessel zero j(1,1) = 3.8317.

On the corrected clock the power-law window t ∈ [20, 140] falls where the paper
draws its reference lines. **Regime IV's exponent is −0.70 against −0.4**, a
factor 1.7 — where the old clock gave −1.33 and a factor 3.3. Regime I's window
spans only ~2 crossings at its scale, too few rows to fit, so no exponent is
quoted rather than one fitted to noise.

The late-time α is unchanged at 0.77–0.81 for I, III and IV against 1.5–2.
Regime II's 1.76 remains the closest, but it is now read at t ≈ 11 rather than
450, so it is not the same comparison the paper makes.

## 6. The figures — Figs. 7 and 8

Each film is vorticity above, current density below, per-frame normalised.

- **Regime I** reproduces the progression: filaments → a few structures → one
  domain-scale structure, with no circular vortices.
- **Regime II reproduces the exception the paper singles out** — *"the only case
  in which the formation of circular vortices is well pronounced, leading to a
  roll up of the current sheets"*. Smooth circular vorticity blobs above,
  spiralled current sheets below. The Lorentz force is too weak to suppress them.
- **Regime III reproduces its own stated signature** — *"almost identical
  magnetic and velocity fields"*. The two panels are indistinguishable at
  t/T_e = 0.5 and 4.5, separating only late, which tracks the cos θ decay.
- *"Vorticity and current are near-copies at every instant"*, asserted in this
  tree's own `demonstrator/README.md`, is **not supported**: true for regimes I
  and III early, false for regime I late, and false for regime II throughout.
  The paper exempts case II explicitly.

## What is reproduced, and what is not

**Reproduced.** Both columns of Table I, exactly and seed-independently. The
Figs. 7–8 structural progression for all four regimes, including two
regime-specific claims (II's rolled-up current sheets, III's identical fields).
Regime III's dynamic alignment through the window the paper plots: cos θ 18% high
at t = 100 and 6.6% low at t = 200. Regime IV's negative alignment to within 33%
at every time, sign included. Regime I's E/A to 17–20% over t = 40–200. Regime
IV's decay exponent, −0.70 against −0.4.

**Not reproduced.** Regime I's alignment: the reference climbs to 0.906 while ours
goes negative and stays there — and since the campaign ran 1.6× *above* that
regime's Reynolds number, insufficient Re cannot be the explanation. Regime III's
equipartition: E_u/E_B falls to 0.035 where the reference holds 0.72. And the
absolute E/A at t = 0, low by 1.9–4.3× — traced in §1 to Table I disagreeing with
Eq. (6) by the same factor, not to this tree.

**Out of range rather than disagreeing.** Every regime II time-series comparison.
Its velocity scale puts t = 450 at 1964 crossings where the campaign ran 50, so
the run reaches t = 11. Extending it would need a 39× longer run.

**Not attempted.** Fig. 9's final-state scatter plots (ω–ψ, a–ψ, a–j) need ψ,
which would be a Poisson solve; Fig. 5's PDF of local cos θ; Fig. 3's periodic
control.

## Three places Table I does not close against the paper's own equations

1. **E/A.** Eq. (6) predicts 4.76 for regime I against the printed 16, converged
   in the spectrum cut. Reproducing Table I would need a k^−2.1 spectrum against
   the k^−3 stated one sentence after Eq. (6). See §1.
2. **Re against the ratio column.** Eq. (10) fixes E_B = ½, so E_u/E_B alone
   determines E_u and hence Re; the implied E_B instead comes out 0.70, 4.6e−5,
   0.29, 0.46, with regime II off by four orders.
3. **Re against its own formula.** `Re = 2r√(2E_u)/ν` with E_u a total carries
   units of length. The column is that formula taken literally, and equals the
   physical Reynolds number times √A = 5.29 — which reproduces regime IV to 4%
   and regime I to 15%, confirming the reading.

None of this touches the campaign, which uses only the ratio columns — E_u/E_B
and H_c — and Fig. 2/4, which are mutually consistent.

## The largest remaining caveat

Not Reynolds number: at 618, 1287 and 1128 physical for regimes I, III and IV
against our 1000, the campaign is at the reference's Re. It is **duration and
realisation**. Regime II is 39× short. And every regime is one draw: the paper
does not state how many realisations its curves average, and a single 2-D decay
at these Reynolds numbers is not self-averaging, which is the most likely reason
regime I's alignment goes the other way.
