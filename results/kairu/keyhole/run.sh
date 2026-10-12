#!/bin/bash
# Kairu's conditions through keyhole. bash, not zsh, so $MO and $ST split.
# Outputs live in build/ (git-ignored, survives a reboot -- /tmp did not).
K=/Users/alessandroderosis/Desktop/M3LB/build/validation/keyhole
D=/Users/alessandroderosis/Desktop/M3LB/build/kairu_keyhole
T=--kokkos-num-threads=4
ST="-mat steel -dx 10 -n 161 -nz 120 -t 47 -every 1000"
MO="-mat mo -dx 10 -n 121 -nz 80 -t 2 -every 200"
# --- NEW: 316L, keyhole as calibrated, the full 47 ms dwell
$K $ST -P 150 -spot 150 -out $D/st_g150.csv $T > $D/st_g150.log 2>&1
$K $ST -P 123 -spot 150 -out $D/st_g123.csv $T > $D/st_g123.log 2>&1
$K $ST -P 102 -spot 150 -out $D/st_g102.csv $T > $D/st_g102.log 2>&1
# --- NEW: Mo "LG" beams (Table 4 D86, run as Gaussians), physical inputs
$K $MO -etamin 0.38 -P 183.6 -spot 300 -out $D/mop_l3_92.csv  $T > $D/mop_l3_92.log  2>&1
$K $MO -etamin 0.38 -P 302   -spot 305 -out $D/mop_l3_151.csv $T > $D/mop_l3_151.log 2>&1
$K $MO -etamin 0.38 -P 404   -spot 277 -out $D/mop_l3_202.csv $T > $D/mop_l3_202.log 2>&1
$K $MO -etamin 0.38 -P 183.6 -spot 424 -out $D/mop_l5_92.csv  $T > $D/mop_l5_92.log  2>&1
$K $MO -etamin 0.38 -P 302   -spot 434 -out $D/mop_l5_151.csv $T > $D/mop_l5_151.log 2>&1
$K $MO -etamin 0.38 -P 404   -spot 425 -out $D/mop_l5_202.csv $T > $D/mop_l5_202.log 2>&1
# --- NEW: 316L physical (full power, A = 0.35), the top power only
$K $ST -etamin 0.35 -P 300 -spot 150 -out $D/stp_g150.csv $T > $D/stp_g150.log 2>&1
# --- REPRODUCTIONS of the eight Mo runs already reported (lost with /tmp)
$K $MO -P 91.8 -spot 142 -out $D/mo_g92.csv    $T > $D/mo_g92.log    2>&1
$K $MO -P 151  -spot 143 -out $D/mo_g151.csv   $T > $D/mo_g151.log   2>&1
$K $MO -P 202  -spot 155 -out $D/mo_g202.csv   $T > $D/mo_g202.log   2>&1
$K $MO -P 202  -spot 277 -out $D/mo_l3_202.csv $T > $D/mo_l3_202.log 2>&1
$K $MO -P 202  -spot 425 -out $D/mo_l5_202.csv $T > $D/mo_l5_202.log 2>&1
$K $MO -etamin 0.38 -P 183.6 -spot 142 -out $D/mop_g92.csv  $T > $D/mop_g92.log  2>&1
$K $MO -etamin 0.38 -P 302   -spot 143 -out $D/mop_g151.csv $T > $D/mop_g151.log 2>&1
$K $MO -etamin 0.38 -P 404   -spot 155 -out $D/mop_g202.csv $T > $D/mop_g202.log 2>&1
echo ALLDONE
