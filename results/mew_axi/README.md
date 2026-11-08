# The round MEW drop with a jet sink: the scenarios

This file covers the jet-sink runs (`demonstrator/mew_axi -jet`, 2026-11-07)
and the three videos made from them. The other runs in this folder are named
and described in `demonstrator/mew_axi.cpp`'s banner, under WHAT IT SAYS; the
jet's own paragraph there (THE JET) records how the sink was chosen and tested.

## The shared set-up

- **Geometry.** Axisymmetric (r-z) on 50 µm cells. A round nozzle with a 0.4 mm
  bore and a 0.3 mm wall (tip face 1.0 mm across), 5.5 mm above a grounded
  collector, at 6.5 kV.
- **Melt.** PCL at 820 Pa s (measured, at 120 °C), 1080 kg/m³, surface tension
  0.030 N/m (nominal). The melt conducts: the whole drop is at the nozzle's
  potential.
- **Feed.** 0.0208 mm³/s, Murat's, pushed by a piston in the bore. The feed and
  the field start together at t = 0, the bore full to its tip face.
- **Collector.** It cannot move (a translating plate is not axisymmetric), so a
  run is read up to its first contact and no further.

## The jet

The jet narrows from about 0.1 mm at the apex to the fibre's 20 µm, and is
thinner than a 50 µm cell over almost all its length, so the model does not
resolve it. It takes the melt the jet would carry out of the drop instead: the
jet's flow is drawn out of the melt in a box 0.1 mm in radius and 0.2 mm tall,
starting four cells above the drop's lowest point near the axis
(`FreeSurfaceSolver::draw`), for as long as the field is on and the drop has not
touched the collector.

That flow is an input, taken from the measured fibres laid at 35 mm/s
(π d² v / 4):

| | fibre | jet flow | share of the feed |
|---|---|---|---|
| no fan | 20.5 µm | 0.011552 mm³/s | 56 % |
| fan | 11.5 µm | 0.0036354 mm³/s | 17 % |

The fan enters the model through that flow only, except in scenario 7, which
gives the drop the fan's cooling as well.

## The seven scenarios

Each run is

```
./build/demonstrator/mew_axi --kokkos-num-threads=2 -tframe 1 <options> \
    -out results/mew_axi/<run> > results/mew_axi/<run>.log
```

from the repository root. Scenarios 1 and 4 are earlier runs of the same driver
(1 with frames every 0.5 s), which the reruns for the videos reproduce row for
row. The series and logs are in this folder; the frames are not (git ignores
them).

| # | run | options | jet | what happens |
|---|---|---|---|---|
| 1 | `R4_iso_nu6` | `-tend 20` | none | drawn into a column that touches at 15.80 s |
| 2 | `R4_iso_jet036` | `-jet 0.0036354 -tend 30` | fan's | touches at 17.89 s |
| 3 | `R4_iso_jet116` | `-jet 0.011552 -tend 45` | no-fan | grows at only 0.0092 mm³/s, touches at 32.14 s |
| 4 | `R4_heat_best` | `-heat -nu 0.0416666667 -tend 60` | none | stalls 3.525 mm up, 0.750 mm³ at the tip at 60 s |
| 5 | `R4_heat_jet036` | as 4, `-jet 0.0036354` | fan's | stalls 3.625 mm up, 0.629 mm³ |
| 6 | `R4_heat_jet116` | as 4, `-jet 0.011552` | no-fan | stalls 3.825 mm up, 0.368 mm³ |
| 7 | `R4_heat_jet036_h300` | as 4, `-h 300 -jet 0.0036354` | fan's, and the fan's cooling | stalls 4.375 mm up, 0.989 mm³ |

Scenarios 1–3 keep the melt at 120 °C throughout; 4–7 let it cool.

**With heat** the melt arrives at 120 °C, the air is at 25 °C and the collector
at 22 °C. The surface loses heat at h = 90 W/m²K, nominal for still air; 300
stands in for a fan blowing on the drop (nominal too). The viscosity rises as
the melt cools (Arrhenius, 31.3 kJ/mol), and below a nominal 30–40 °C band the
melt is held by a drag, standing in for crystallisation. `-nu 0.0416666667` is
the lattice viscosity at 120 °C, low enough that the viscosity can rise 24-fold
before the axis caps it.

**How the two families read.** Isothermal, the question is when the column
reaches the collector, and the jet delays it, because the melt left at the tip
is what gets there. With heat the drop stalls in every case, and the jet decides
how much melt piles up at the tip. Without the fan the tip keeps half of what it
keeps with no jet. With the fan it keeps 1.7 times the no-fan amount, or 2.7
times if the fan also cools the drop.

## Sensitivity runs

| run | change | result |
|---|---|---|
| `R4_iso_jet116_d8` | the box eight cells deep (`-jetdepth 0.4`) | touches at 28.47 s, not 32.14; the field pulls the undrained melt below the box into a thread that pinches at it; takes 95 % of the jet's flow |
| `R4_iso_jet116_r05` | the box one cell across (`-jetr 0.05`) | dimples sooner; 2.875 mm up at 30 s, against 2.725 |
| `R4_heat_jet116_T40` | the jet only while the apex is at 40 °C or more (`-jetTmin 40`) | takes 0.4 % less; within a cell of scenario 6 to 45 s, then 3.975 mm up and 0.416 mm³ at 60 s |
| `R4_heat_h300` | scenario 4 at h = 300, no jet | stalls 4.325 mm up, 1.169 mm³ at the tip |

## The artefact

The box's suction pulls the surface on the axis 2–3.6 cells up into a dimple,
the opposite of what a real jet does to the apex. Once the surface is within
about a cell of the box, the free surface creates melt that was never fed.
In scenario 3 the dimple alone brings this to 0.6 % of the jet's take by 24 s,
and it reaches 7.2 % (5.6 % of the melt) by contact once the thinning column's
sides close in on the box as well. With heat it is 1–3 % by 60 s. With no jet
the same column creates none, and the cause is not diagnosed.

## The videos

`mew_axi_jet_iso.mp4`, `mew_axi_jet_heat.mp4` and `mew_axi_jet_apex.mp4`, made by
`tools/anim_mew_axi.py`. Git ignores them, as it does every video here.

- **iso**: scenarios 1–3 side by side, mirrored about the axis, with the
  equipotentials in the air, beside their lowest point and the melt held at the
  tip. Each drop stops at its first contact.
- **heat**: scenarios 4–7 coloured by temperature over 60 s.
- **apex**: scenario 3's apex cell by cell, with the box the jet is drawn from,
  the dimple, the gap from the box to the surface, and the melt created.

The runs above kept one frame a second, too few to animate, so the videos come
from reruns with finer frames, whose series are identical to the ones here.
To make them again:

```
D=<an empty scratch folder>
B=./build/demonstrator/mew_axi
$B -tend 16.2 -tframe 0.1 -out $D/iso_none
$B -jet 0.0036354 -tend 18.2 -tframe 0.1 -out $D/iso_fan
$B -jet 0.011552  -tend 32.5 -tframe 0.1 -out $D/iso_nofan
$B -heat -nu 0.0416666667 -tend 60 -tframe 0.25 -out $D/heat_none
$B -heat -nu 0.0416666667 -jet 0.0036354 -tend 60 -tframe 0.25 -out $D/heat_fan
$B -heat -nu 0.0416666667 -jet 0.011552  -tend 60 -tframe 0.25 -out $D/heat_nofan
$B -heat -nu 0.0416666667 -h 300 -jet 0.0036354 -tend 60 -tframe 0.25 -out $D/heat_fan_h300
python3 tools/anim_mew_axi.py iso  $D results/mew_axi/mew_axi_jet_iso.mp4
python3 tools/anim_mew_axi.py heat $D results/mew_axi/mew_axi_jet_heat.mp4
python3 tools/anim_mew_axi.py apex $D results/mew_axi/mew_axi_jet_apex.mp4
```

Each `mew_axi` line also takes `--kokkos-num-threads=2`, and each run needs its
log beside its frames: redirect its output to `$D/<name>.log`.

## Caveats

- The isothermal contact times are not grid-converged: with no jet, 25 µm cells
  touch at 9.80 s against 15.80 s on 50 µm cells, because the tip sharpens on
  finer cells.
- The heated stall held on the finer grid without a jet; the jet runs exist on
  50 µm cells only.
- The jet flows are inputs taken from the fibres, not predictions, and h, the
  surface tension and the freezing band are nominal.
