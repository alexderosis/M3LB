# Reduced keyhole model — reference data for `validation/keyhole.cpp`

## `ref_95um_104W.csv`

Depth-vs-time from the **Python** twin at
`~/Desktop/02_Code_Projects/Laser/keyhole_validation`, 95 µm / 104 W,
production grid (61×61×300, dx = 5 µm), shipped seven-case constants,
2 ms, 1332 steps. Columns are `t_s, depth_m` — **SI, not microns**.

Regenerated 2026-09-21. It had to be regenerated because **every simulation
`.npz` in that directory has been deleted**; only the experimental curves under
`exp_data/` survive. The model was not modified: `solver.py` and
`case_runner.py` were copied byte-identically (md5 `81824856d70f…`,
`c14a50c5795c…`) and `run_case()` inlined verbatim to add a diagnostic record.

It reproduces that project's own published numbers for this case — ignition
0.515 ms, final depth 134.5 µm — which is the strongest available check that
this is the trace that was lost.

**It is not experimental data and not a physical reference.** The Python's
agreement with Cunningham et al. (2019) is a fit with thirteen calibrated
numbers. Agreement against this file proves the port, not the model.

## `python_clamp_census.log`

The Python instrumented to count how often each clamp binds. The headline: of
the columns actually receding, **47.5 % are truncated by
`max_recession_per_step`**, the 300 K floor binds on 67.8 % of ignited steps
catching raw excursions to −31334 K, and `Vm_max = 15 m/s` never binds at all.
The surface energy balance is forward Euler on a stiff evaporative feedback and
the two-sided clamp is what makes it look stable.
