# The hot end, as measured (2026-11-17)

What Z. Murat Kilic sent on 17 November 2026 for the melt-electrowriting models
(`demonstrator/mew_axi`, `demonstrator/mew_jet`), and what it comes to.

| file | what it is |
|---|---|
| `viscosity_pcl.csv` | his spreadsheet, cell for cell: PCL in steady shear, 1–100 1/s at 90, 100, 110 and 120 °C. **Not tracked**: it is his unpublished data and this repository is public (`.gitignore`) |
| `hotend_numbers.py` | every number below, from that table, the heater's cycle and E3D's drawings. It needs the table |
| `hotend_numbers.txt` | its output, tracked so that the numbers can be read without the table |
| `mew_hotend.png` | the figure for Murat: the table, its temperature dependence, the cycle, the outflow, and the model with the jet (`tools/plot_mew_hotend.py`). Not tracked either: its first panel plots every point of the table |

Also received, and typed into the script with their sources: the heater's cycle
(110 → 128 °C in 13 s with the heater on, back in 53 s with it off), its power
(30 W), and E3D-Online's drawings of the v6 hot end (nozzle series edition 8,
1.75 mm heat break 6, heater block for sensor cartridges 8.1, heat sink 6,
assembly 6; GPL, not copied here).

The nozzle fitted is the 0.4 mm one (confirmed 2026-11-17); the drawing lists
nine orifices, and 0.4 mm is what the models have used throughout.

## What it says

- **Viscosity.** Newtonian to within 4 % up to 10 1/s at all four temperatures:
  2220, 1690, 1087 and 815 Pa s. The 820 Pa s the models use at 120 °C stands.
  Above 16–25 1/s the *stress* falls as the rate rises, 8- to 22-fold by
  100 1/s, which a melt in steady uniform shear cannot do; those readings are
  not a viscosity and are not used.
- **Activation energy.** 40.9 kJ/mol, standard error 2.9. Four temperatures make
  the 95 % interval 29–53, so the 31.3 kJ/mol the models took from the
  literature is not excluded — but the measured value is the one to use, and a
  melt cooled to 40 °C is then 24 times its 120 °C viscosity where 31.3 gave 12
  (an extrapolation 50 °C below the data).
- **The heater's cycle.** Period 66 s, mean 119 °C, mean power 5.9 W. Across it
  the viscosity changes by a factor 1.78. As one lump the hot parts hold
  17.4 J/K; what the drawings show accounts for 56 % of that.
- **The melt breathes.** The hot zone holds 50.6 mm³ of melt between a solid
  filament and the orifice. With a melt expansion of 6.4e-4 /K (nominal, **not
  verified for PCL**; 5–7e-4 is the range of polymer melts) the cycle pushes
  0.53 mm³ out and takes it back, 39 % of the 1.37 mm³ fed in a cycle: about
  0.062 mm³/s leaves the orifice while the block heats and 0.011 while it
  cools, against a feed of 0.0208. Not checked: that the filament and its drive
  do not give.
- **The drawings.** The models' 0.4 mm bore and 1.0 mm tip face are the
  drawing's. Outside, the tip is the end of a 2 mm cone under a 7 mm hexagon,
  with the 23 × 16 mm block 5 mm above it — the electrode the r-z model
  brackets with `-up 5 -plateR 8`.

`python3 results/mew_hotend/hotend_numbers.py > results/mew_hotend/hotend_numbers.txt`
regenerates the numbers. The runs that use them are `R4_heat_Ea41*` in
`results/mew_axi` and `measured_inputs.txt` in `results/mew_jet`.

## What the models make of it

The r-z model (`mew_axi -Ea 40.9 -cycle 110 128 13 53 -breathe 0.0293`, 50 µm
cells, heat, to 150 s; table in `results/mew_axi/cycle_numbers.txt`):

- **Without a jet the cycle sets when the melt arrives, not where the drop
  stops.** All four runs stall 3.1–3.2 mm above the collector at 150 s. The
  measured activation energy alone moves the drop by one cell and 4 % at 60 s;
  the cycle on the temperature alone by a cell and 3 % at most.
- **With the no-fan fibres' flow taken at the apex, the melt hanging below the
  tip grows only while the heater is on**: by 0.65 mm³ in each 13 s of heating,
  and by −0.004 to +0.053 over each 53 s of cooling (flat in the model; it would
  fall by 0.04 without the jet sink's mass-creation artefact, which is 5–6 % of
  what the jet takes). With a steady feed it grows evenly.
- **The fibres' own flow, 0.0116 mm³/s, is what the orifice delivers while the
  heater is off**, 0.0108 (0.0097–0.0132 over the expansion coefficient's
  range): a 19.9 µm fibre at 35 mm/s against the 20.5 µm measured. That is what
  a jet carrying the cooling leg's flow steadily, with the surplus oozing in the
  13 s of heating, would give — one agreement between two numbers, one of them
  nominal: a thing to test, not a result.

The jet model (`mew_jet -Ea 40.9`): the current that lands the jet at 35 mm/s is
1.29 nA (0.86 with 31.3 kJ/mol), and 1.09–1.58 nA for a jet starting at 128 and
110 °C; at a fixed 1.29 nA the jet lands on its own at 29.6–40.1 mm/s across the
cycle.

Animations (git-ignored, `tools/anim_mew_axi.py cycle | cyclejet`):
`results/mew_axi/mew_axi_cycle.mp4` and `mew_axi_cycle_jet.mp4`.

## Murat's answers (2026-11-17, his second message)

1. **The drop at the tip grows in steps, about every 50 s.** His reading: after
   the Taylor cone grows it needs time to settle again.
2. **The controller can run PID, and they do not.** Holding 120 °C keeps the
   heater working, and each time it works the PCL expands and the cone loses its
   stability; on/off control at least gives a stable process.
3. **110–128 °C is what the sensor read**, and it differs from run to run: a
   range, not a fixed cycle.
4. **The melt's density is not available.**

What they mean for the model:

- **Steps are what it predicts.** With the no-fan jet the melt below the tip
  grows only while the heater is on, one step a cycle: 0.65 mm³ in each 13 s of
  heating, then flat for 53 s. His spacing, about 50 s, is close to that quiet
  leg (53 s) and shorter than the period (66 s), which he says varies from run
  to run. The open question is whether his steps keep
  time with the heater. If they do, they are the hot end's breathing; if not,
  they are the cone's own settling, which this model cannot make, since it does
  not resolve the cone.
- **A step's volume measures the expansion**, which is what the density was
  wanted for. If the jet carries a steady flow through the step, a step is
  `(Q - Q_jet) t_on + B dT`: 0.120 + 0.527 = 0.647 mm³ with the no-fan jet,
  t_on = 13 s, dT = 18 K and B = 0.0293 mm³/K. A measured step V gives
  `B = (V - 0.120) / dT`, and the melt's expansion `B / 50.6 + 5.5e-5` per K
  (the second term is the brass bore's own growth). A step of 0.65 mm³ is a
  sphere about 1.07 mm across, so it should be visible in a frame before and
  after a step.
- **Tightening the heater's band is not the test** that the earlier reply
  proposed: PID control, which keeps the heater working, upset the cone, so it
  changes the thing the test was meant to hold fixed. Whether the steps keep
  time with the heater, and how large they are, is the test instead.
- **A swing that changes from run to run scales the breathing in proportion**:
  0.44 mm³ a cycle at 15 K and 0.59 at 20 K, against 0.53 at the 18 K used.
  The conclusions above do not hang on the exact values.
- **No published PCL melt expansion was found** (2026-11-17: four web searches
  in all, and one fetched paper). It stays nominal at 6.4e-4 /K.
