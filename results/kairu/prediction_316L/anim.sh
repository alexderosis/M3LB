#!/bin/bash
# The 150 W row of the 316L prediction (m150.log), rerun with -frames for the
# animation. Same arguments as batch.sh's Mach-check line plus the dump.
M=/Users/alessandroderosis/Desktop/M3LB/build/validation/melt_pool
B="-steel -v 500 -spot 150 -A 0.35 -Lx 1200 -Ly 480 -Lz 240 -flow -evap --kokkos-num-threads=4"
mkdir -p anim150
$M $B -P 150 -dx 6 -fsub 12 -frames anim150 > anim150.log 2>&1
echo ANIMDONE
