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

Two candidate explanations were tested and **both ruled out**:

- **Not the spectrum cut.** `-kmax` at 24 / 40 / 60 / 80 / 120 gives E/A =
  7.6, 8.9, 7.3, 4.8, 9.8 — non-monotonic, because the central-difference curl
  under-resolves **b** as kmax approaches the Nyquist.
- **Not the gauge.** A is gauge-dependent and the paper's *note added in proof*
  says so explicitly. Fixing the zero-mean gauge over the fluid disc — the gauge
  a spectral inversion gives, and the one that *minimises* A — moves the number
  by 0.1 %.

**This is open.** It is the sharpest single discrepancy in the comparison and it
is present at t = 0, before any dynamics.

## 2. E_u/E_B — Fig. 2, top

| regime | t=40 | | t=100 | | t=200 | | t=450 | |
|---|---|---|---|---|---|---|---|---|
| | ours | paper | ours | paper | ours | paper | ours | paper |
| I | 0.0239 | 0.0618 | 0.0148 | 0.0323 | 0.0112 | 0.0214 | 0.0911 | 0.0202 |
| II | 1.76e5 | 3.72e6 | 1.35e5 | 3.10e7 | 3.09e4 | 6.41e7 | 1.67e3 | 1.01e8 |
| III | 0.148 | 0.811 | 0.0491 | 0.768 | 0.0076 | 0.769 | 0.104 | 0.717 |
| IV | 0.0194 | 0.0757 | 0.0230 | 0.0560 | 0.0043 | 0.0437 | 0.127 | 0.0347 |

**No regime tracks this panel well.** Regimes I and IV both run a factor 2–4
below the paper for most of the run and then turn *up* near the end — I to 0.0911
and IV to 0.127, against 0.0202 and 0.0347 — so the late rise is ours alone. Both
energies are still falling there; the kinetic one simply stops falling as fast.

**Regime II disagrees qualitatively and neither side should be trusted here.**
The paper's ratio *rises* four decades to 1e8; ours *falls* to 1.7e3. But our
E_mag there is 2.1e-11 with max|b| ≈ 1e-5, four decades below the seeded field,
which is plausibly a numerical floor rather than physics. Reported as
uninterpretable rather than as a discrepancy.

**Regime III loses its equipartition.** The paper holds 0.72–0.81 throughout —
"approximately equipartitioned", which is the defining property of that regime.
Ours falls to 0.008 by t = 200 before recovering to 0.104, an order of magnitude
short and non-monotonic where the reference is flat.

## 3. E/A — Fig. 2, centre

| regime | t=40 | | t=100 | | t=200 | | t=450 | |
|---|---|---|---|---|---|---|---|---|
| | ours | paper | ours | paper | ours | paper | ours | paper |
| I | 1.28 | 1.61 | 0.490 | 1.36 | 0.251 | 1.29 | 0.472 | 1.29 |
| II | 1.16e6 | 9.40e7 | 5.70e5 | 3.64e8 | 2.18e5 | 3.55e8 | 3.89e4 | 3.34e8 |
| III | 1.61 | 5.43 | 0.725 | 3.31 | 0.334 | 2.84 | 0.150 | 2.55 |
| IV | 1.32 | 2.69 | 0.680 | 2.61 | 0.337 | 2.35 | 0.200 | 1.78 |

**The selective-decay signature is reproduced in sign and roughly in rate.**
Regime I's E/A falls 5.49 → 0.472, a factor 11.6; the paper's falls 16 → 1.29, a
factor 12.4. Those agree to **6 %**, while the absolute values differ by a factor
of 3. A is decaying more slowly than E in both, which is what selective decay
means, and the *rate* of that is the part this tree gets right.

That decomposition matters: the disagreement is in the initial length scale, not
in the decay.

## 4. cos θ — Fig. 4 *(reference values digitised, unverified)*

| regime | t=100 | | t=200 | | t=450 | |
|---|---|---|---|---|---|
| | ours | paper | ours | paper | ours | paper |
| I | −0.246 | 0.092 | 0.043 | 0.657 | −0.027 | 0.906 |
| II | −0.122 | 0.193 | −0.336 | 0.178 | −0.717 | 0.316 |
| III | 0.479 | 0.733 | 0.237 | 0.740 | 0.029 | 0.847 |
| IV | −0.236 | −0.095 | −0.088 | −0.162 | −0.024 | −0.201 |

**This is the headline non-reproduction, and Eq. (11) sharpened it rather than
fixing it.** Regime III reaches **cos θ = 0.866 at t = 40 against the paper's
0.864 at t = 30** — essentially exact — and then falls away to 0.029 where the
paper climbs to 0.847. So the alignment mechanism *is* captured and then not
sustained. That is a much more specific statement than "alignment not
reproduced", and it was unavailable before, because regime III could not be run.

The corroborating measurement is E/|H_c| (Fig. 2, bottom), whose minimum of 2 is
the paper's own signature of dynamic alignment: regime III goes 3.45 → 9.89 →
48.7 → 119 where the paper holds **2.04** at every time. Alignment decays here
instead of locking in.

## 5. Energy decay — Fig. 6 *(paper values from the text, exact)*

| regime | exponent, ours | paper | late α, ours | paper |
|---|---|---|---|---|
| I | −1.63 | −0.6 | 0.81 | 1.5 |
| II | −2.50 | — | 1.76 | 2 |
| III | −1.45 | — | 0.78 | 2 |
| IV | −1.33 | −0.4 | 0.77 | 2 |

E ~ exp(−2ανt); the circle's largest Stokes mode is α = 1.64, stated exactly in
`validation/stokes_disc.cpp`.

**Regime II's α = 1.76 is the best quantitative agreement in the report** — 12 %
below the paper's 2, and 7 % above the geometric eigenvalue 1.64.

The power-law exponents are 2–3× too steep, and the reason is structural rather
than a defect: **both clocks cannot be matched at once.** Fifty crossings is the
paper's t = 450 on the *advective* clock, but at Re = 1000 against their
3868–7920 that is νt/R² ≈ 0.20 against their 0.0505 — about 4× further into
diffusive decay. A steeper measured exponent is what that predicts.

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
Selective decay in regime I: the E/A decay factor agrees to 6 %. Regime II's
late-time decay rate α to 12 %.

**Not reproduced.** Dynamic alignment is not sustained: regime III matches to
0.2 % at t ≈ 30–40 and then decays away, with E/|H_c| leaving 2 rather than
settling on it. Absolute E/A is low by 1.4–4.3× *at t = 0*, cause unknown.
Intermediate-time decay exponents are 2–3× too steep, explained by the clock
mismatch at Re = 1000.

**Not attempted.** Fig. 9's final-state scatter plots (ω–ψ, a–ψ, a–j) need ψ,
which would be a Poisson solve; Fig. 5's PDF of local cos θ; Fig. 3's periodic
control.

**The single largest caveat** is Re = 1000 against the paper's 3868–7920. The
paper's own sentence — *"nontrivial final states are only observed if the initial
Reynolds number is sufficiently high"* — is about Fig. 9's scatter plots in
preliminary low-resolution runs, not about Fig. 4, so it should not be quoted as
covering the alignment result. Matching Re = 3868 at fixed u0 and τ needs
N ≈ 1242.
