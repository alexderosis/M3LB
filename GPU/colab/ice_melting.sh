#!/bin/bash
#==============================================================================
#  GPU/src/ice_melting.cu on a free Colab GPU, from a notebook cell:
#
#     !bash /content/GPU/colab/ice_melting.sh 5.6 [more ice_melting flags]
#     !H=256 bash /content/GPU/colab/ice_melting.sh 5.6     (another resolution)
#
#  WHY A SCRIPT THAT BLOCKS. Free Colab recycles a runtime whose notebook has
#  no RUNNING cell, even while a nohup'd job is computing -- a 5.6 C run was
#  lost that way at 1.4M steps on 2026-09-24, with every snapshot in /content.
#  So this keeps the cell running for the whole job, and copies every probe
#  line and a text digest of every snapshot (scallops.py -trace) into the
#  cell's output, which the browser keeps after the runtime and its files go.
#  THAT COPY IS THE PART THAT MATTERS: a running cell stops the idle recycle
#  but not the usage cap -- the same day a runtime was cut off mid-run, with
#  this script's cell running, after ~3 h of T4 time, and every file went
#  again. Only the text in the output survived. Long runs belong on CSF3.
#
#  Builds only what the run needs the first time (cm + darcy: ~3 min on a T4,
#  where the full operator set takes over 20). Output in /content/run_T<T>_H<H>/.
#==============================================================================
T=${1:-5.6}; shift
H=${H:-512}
cd /content/GPU || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
if [ ! -x bcm/ice_melting ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  cmake -S . -B bcm -DCMAKE_BUILD_TYPE=Release -DLBM_GPU_ARCH="$ARCH" \
        -DLBM_ONLY_OP=cm -DLBM_ONLY_FORCE=darcy > bcm.cfg.log 2>&1 || { tail -20 bcm.cfg.log; exit 1; }
  cmake --build bcm -j2 --target ice_melting 2>&1 | grep -E -i 'error|built target'
  [ -x bcm/ice_melting ] || exit 1
fi
OUT=/content/run_T${T}_H$H; mkdir -p "$OUT"
echo "=== T_inf = $T C: ice_melting -T $T -H $H -probe 20000 -dump 10 -melt 0.6 -drift 0.4 $* ==="
./bcm/ice_melting -T "$T" -H "$H" -probe 20000 -dump 10 -melt 0.6 -drift 0.4 -out "$OUT/ice" "$@" \
  > "$OUT/log.txt" 2>&1 &
PID=$!
n=0
while :; do
  alive=1; kill -0 $PID 2>/dev/null || alive=0
  m=$(wc -l < "$OUT/log.txt")
  if [ "$m" -gt "$n" ]; then
    # a snapshot is complete once the driver has printed "wrote" for it (after fclose)
    sed -n "$((n + 1)),${m}p" "$OUT/log.txt" | while IFS= read -r line; do
      case "$line" in
        *"wrote "*thickness*) python3 /content/GPU/csf3/scallops.py "${line##* }" -trace ;;
        *) echo "$line" ;;
      esac
    done
    n=$m
  fi
  [ $alive = 1 ] || break
  sleep 60
done
wait $PID; echo "=== exit $? ==="
