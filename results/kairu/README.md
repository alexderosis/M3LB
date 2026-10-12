# Kairu's single tracks, and a blind 316L prediction for the follow-up

Kairu, MSc dissertation, Manchester 2026 (supervisor O. Allegre): single tracks
at **2 mm/s** with Gaussian and annular (LG3, LG5) beams, on 316L at
102 / 123 / 150 W and on molybdenum at 91.8 / 151 / 202 W, sectioned unetched.
Everything here was run on 2026-10-12 on the laptop, four threads, FP64. The
verdicts of sections 1 and 2 are argued in `validation/keyhole.cpp`'s banner
and summarised here; the logs are the evidence.

## 1. `keyhole/` -- the reduced-order model at her conditions

`run.sh` is the script as run (bash; absolute paths). 10 um cells; a track is
approximated by a STATIONARY exposure of the moving beam's dwell (47 ms for
316L), which `keyhole` has to do because it has no scan speed.

| file | what |
|---|---|
| `st_g{102,123,150}` | 316L, as calibrated (eta_min 0.12 on the P/2 beam) |
| `v2_st_g*.log` | the same three, rerun after the width readout was added -- these print the widths |
| `fr102.log` | 102 W with frames; its readout printed a melt width of 0 um, which is how the apparent-cp centre dip was found |
| `stp_g150` | 316L with physical inputs (A = 0.35, full 150 W) |
| `mo_*` | Mo as calibrated, the five conditions first reported |
| `mop_*` | Mo with physical inputs (A = 1 - R = 0.38, full power), all nine conditions |

`.csv` files are each run's depth trace. **Result:** the model fails in
opposite directions on the two materials. 316L as calibrated gives melt widths
of 30 / 60 / 110 um against her 863 / 941 / 1001 um (9-29x too narrow, and no
removal, which is right); with a physical absorptivity it drills through the box
where she removed nothing. Mo removes nothing at any condition, while she cut
11 / 52 / 91 um Gaussian grooves -- but every condition that grooved is hotter
in the model than every one that did not, ~1300 K below Mo's melting point,
consistent with oxide (MoO3) removal, which an evaporation model cannot see.

## 2. `mo_test/` -- can one temperature law carry Mo's removal across beams?

`mo_test_PREREG.txt` was written before any fit was computed; `mo_test.py`
applies it (standard library only) and `mo_test.out` is its output.

```
python3 results/kairu/mo_test/mo_test.py
```

The law `v = V0 exp(-E/RT)` on the steady Gaussian surface field, fitted exactly
on two Gaussian grooves, predicting the third and the six LG cases, three
leave-one-out fits. **0 of 3 pass, at A/k x 0.8, 1 and 1.25: "does not extend".**
The held-out Gaussian is fine (+5 / -4 / +17 %); every fit puts an 8-12 um groove
under LG3 at 151 W, where she saw a flat surface. The Gaussian ladder wants
E = 51-57 kJ/mol and silencing LG3 at 151 W needs 87.

## 3. `prediction_316L/` -- a BLIND prediction for an experiment not yet run

**This is not a comparison with Kairu's tracks.** At 2 mm/s `melt_pool` would
need an ~8 x 4 x 2 mm box and seconds of physical time against microseconds of
melt flow: months on this laptop with the Marangoni flow on. The prediction is
for the experiment her conclusion asks for -- scanning fast enough that the beam
controls the pool -- at a condition the same lab could run, and it is committed
**before** any such experiment so that a comparison would be blind.

Conditions (`batch.sh`, `validation/melt_pool`, tier (d) flow + tier (e)
evaporation): 316L **under argon**, Gaussian 1064 nm beam, 1/e^2 diameter
**150 um**, **500 mm/s** (Pe = v w0 / alpha ~ 7 on the 1/e^2 radius; the log's
`Pe = v sigma / (2 alpha)` reads 1.75), absorptivity **0.35**,
clean-316L surface tension (**d(gamma)/dT = -4.3e-4 N/(m K)**, so the melt flows
outward), box 1200 x 480 x 240 um, 1.92 ms of scan.

| P | width 2w | depth d | length L | d / 2w | Eagar-Tsai conduction 2w / d (d / 2w) | peak surface T | log |
|---|---|---|---|---|---|---|---|
| 100 W | 144.4 um | 22.9 um | 215.6 um | **0.16** | 136.6 / 38.0 um (0.28) | 2692 K | `m100` |
| 150 W | 190.7 um | 29.8 um | 300.2 um | **0.156** | 160.6 / 53.6 um (0.33) | 3128 K | `m150` |
| 150 W, dx 4 | 197.0 um | 29.1 um | 300.6 um | 0.148 | | 3304 K | `g150` |
| 200 W | 231.5 um | 33.4 um | 369.1 um | 0.14 | 177.2 / 65.9 um (0.37) | 3486 K | `m200` |

dx = 6 um with 12 fluid sub-steps unless stated; peak Mach 0.042 / 0.063 / 0.079
at 100 / 150 / 200 W. The Eagar-Tsai column is the analytic conduction pool
without latent heat, flow or evaporation, which each log prints as its
reference. **The measurement
that discriminates is the aspect ratio from one etched section per power**:
outward Marangoni flow predicts a pool about six times wider than deep,
conduction alone about three; width alone barely separates them.

What these numbers are and are not:

- **Mach:** `p150.log` is the same 150 W case at 8 sub-steps (Ma 0.095):
  2w / d / L = 190.4 / 29.8 / 301.9 um, i.e. 0.2 / 0.01 / 0.5 % from `m150`.
- **Grid:** 6 -> 4 um (8 sub-steps) moves the width +3.3 % and the depth
  -2.4 %, but the peak speed +31 % (1.31 -> 1.72 m/s, hence Ma 0.083 there), so
  the pool SHAPE is resolved and the velocity is NOT.
- **200 W is beyond the model:** a peak of 3486 K is past boiling (3100 K) and
  the recoil would dent the surface, which a flat-surface run cannot follow.
- **Sulfur-rich 316L (inward flow) could not be predicted at this grid.**
  `m150_rev` (12 sub-steps, tau_f 0.511) and `r150_f8` (8, tau_f 0.517) both
  diverged. Those two logs predate the fix that makes `melt_pool` say so and
  print "THE FLOW IS NOT MOVING"; `r150_new` is `r150_f8` rerun with the fixed
  binary, which reports 1,280,000 non-finite values. It needs finer cells.
- **Every log here except `p150_new` and `r150_new` predates a print fix:** the
  KEYHOLE CRITERION line names a boiling point of 3315 K (Ti-6Al-4V's), where
  the evaporation model already used steel's 3100 K. `p150_new` is `p150` rerun
  after the fix and differs on that line only.
- **The inputs to pin down before the experiment** are the absorptivity and the
  plate's sulfur content, which sets the sign of d(gamma)/dT and so the flow
  direction.

`anim.sh` reruns the 150 W row with `-frames` for `tools/render_melt_pool.py`.
The frames and the film are not tracked (no video in this tree is).
