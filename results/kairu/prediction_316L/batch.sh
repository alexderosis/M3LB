#!/bin/bash
# 316L Gaussian prediction ladder. bash, explicit arguments on every line.
M=/Users/alessandroderosis/Desktop/M3LB/build/validation/melt_pool
B="-steel -v 500 -spot 150 -A 0.35 -Lx 1200 -Ly 480 -Lz 240 -flow -evap --kokkos-num-threads=4"
# Mach check: same grid as the pilot, Ma 0.095 -> 0.063
$M $B -P 150 -dx 6 -fsub 12 > m150.log 2>&1
# the ladder and the reversed flow sign, at the corrected Mach
$M $B -P 100 -dx 6 -fsub 12 > m100.log 2>&1
$M $B -P 200 -dx 6 -fsub 12 > m200.log 2>&1
$M $B -P 150 -dx 6 -fsub 12 -dgdt 4.3e-4 > m150_rev.log 2>&1
# grid check: same Mach (0.063), cells 6 -> 4 um, cell Reynolds ~10 -> ~6.5
$M $B -P 150 -dx 4 -fsub 8 > g150.log 2>&1
echo BATCHDONE
