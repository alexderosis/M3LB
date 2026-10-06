#!/bin/bash --login
#==============================================================================
#  Build the FP64 A100 binary of GPU/src/orszag_tang.cu for the RR-MHD paper's
#  runs. Login node, from the repo root of an up-to-date mhd checkout (0a43732+):
#
#      cd ~/scratch/M3LB
#      bash --login GPU/csf3/rr_paper/rr_paper_build.sh
#
#  A tree of its own (GPU/build64_rr), all operators (the campaign uses -op cm
#  and -op bgk from the same binary), FP64 (Real is a compile-time typedef, so a
#  stale FP32 binary would run at the wrong precision and say so only in its
#  header line). Only the orszag_tang target is built.
#==============================================================================
set -euo pipefail
ROOT=${M3LB_ROOT:-$PWD}
cd "$ROOT"
if [ ! -f GPU/src/orszag_tang.cu ]; then
  echo "ERROR: run from the M3LB repo root (or set M3LB_ROOT)" >&2; exit 1
fi
if ! grep -q -- '-fullinit' GPU/src/orszag_tang.cu; then
  echo "ERROR: GPU/src/orszag_tang.cu has no -fullinit option: update the checkout first" >&2
  echo "    git checkout mhd && git pull     (needs 0a43732 or later)" >&2
  exit 1
fi
module purge
module load libs/cuda/12.8.1
cmake -S GPU -B GPU/build64_rr -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH=80 -DLBM_DOUBLE=ON
cmake --build GPU/build64_rr --target orszag_tang -j8
echo
echo "built: $ROOT/GPU/build64_rr/orszag_tang"
