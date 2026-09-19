#!/bin/sh
#==============================================================================
#  DO THE MOMENT ARRAYS STAY IN REGISTERS?  A stack-frame measurement on the
#  collision operators, from compiler output rather than from a stopwatch.
#
#  WHY THIS IS A SHELL SCRIPT AND NOT A ctest CASE. What it measures -- frame
#  bytes, whether a loop survived, whether an array is subscripted by a register
#  -- is a property of one compiler on one architecture. Asserting a NUMBER
#  would be brittle enough to be deleted within a month. So this prints a table
#  and leaves the judgement to a reader, and the header of MomentCollision.hpp
#  records the numbers that were current when the finding was made.
#
#  WHAT TO LOOK FOR, in order of how much it matters:
#
#    regidx > 0   an array is indexed by a REGISTER. If it is a local array
#                 this is the bad case: the compiler could not prove the index
#                 constant, so the array cannot live in registers AT ANY
#                 register budget. Check what the index is before calling it a
#                 defect -- a genuine per-node field load looks the same and is
#                 fine (`[x9, w3, sxtw #2]` with w3 the Index argument).
#    loops > 0    a loop over the moments survived. Every such loop keeps its
#                 induction variable at runtime, which is what stops the
#                 basis's 432-byte Ord table from folding.
#    frame        after the two above are clean, this is ordinary register
#                 pressure: 72 doubles do not fit arm64's 32 FP registers and
#                 the compiler spills some at constant offsets. A GPU thread has
#                 up to 255 registers, so a frame that is only pressure may
#                 vanish on a device where a runtime-indexed one cannot.
#
#  Reference table, clang -O3, arm64, 2026-09-04 (ColourGradient 2026-09-19).
#  frame bytes / loops / regidx:
#
#      operator                   FP64 before   FP64 after  FP32 before  FP32 after
#      BGK                           0 / 2 /10   (untouched)   0 / 1 / 6  (untouched)
#      MomentCollision             464 / 1 / 4   368 / 0 / 0  320 / 1 / 4  224 / 0 / 0
#      MultiphaseCentralMoments    400 / 1 / 5   400 / 0 / 1  336 / 1 / 4  256 / 0 / 1
#      PhaseFieldCentralMoments    656 / 1 / 1   (untouched)  432 / 0 / 0  (untouched)
#      ColourGradient             1200 / 1 /10   480 / 0 / 6  624 / 0 / 6  416 / 0 / 6
#
#  The three fixed operators lost their surviving loop and all of their
#  register-indexed LOCAL accesses at both precisions. The one regidx left in
#  MultiphaseCentralMoments is a genuine per-node field load, not a demoted
#  array -- it reads `[x9, w3, sxtw #2]` with w3 the Index argument. The six left
#  in ColourGradient are the same thing six times over: grad phi and grad rho,
#  read as `[x8, w3, sxtw #3]` with w3 the Index argument. Read the column, then
#  read the operand -- the count alone does not distinguish them.
#
#  BGK's own loops and regidx are its walk over f[i], a caller-provided pointer.
#  Its frame is zero at both precisions, which is the contrast that matters.
#
#  COLOURGRADIENT IS DONE, AND IT NEEDED A DIFFERENT FIX (2026-09-19). Loop
#  unrolling alone could not have helped it: its `ke` and `kp` arrays were whole
#  equilibrium and perturbation moment sets, built as POPULATIONS and
#  transformed, so there was nothing to unroll them into. What removed them is
#  the closed-form derivation GPU/'s sibling already carried -- the operator
#  measured there at 47x BGK (20.2 against 950 MLUPS) and recovered 12.5x by it.
#  Ported to src/, the host frame goes 1200 -> 480 (FP64) and 624 -> 416 (FP32),
#  the surviving loop goes, and what is left is register pressure plus the six
#  genuine field loads named above. tests/test_colour_gradient.cpp block 6 keeps
#  the population path and asserts the two agree to 4.2e-16 over 60 states, which
#  is what makes this a rewrite rather than a different operator.
#
#  ONE STILL OPEN, recorded rather than half-fixed:
#
#  PhaseFieldCentralMoments keeps a loop at FP64 but not at FP32, on the same
#  source -- so it is the register budget deciding whether to unroll, not an
#  indexing failure. Its regidx of 1 is a field load, as above. Left alone: the
#  moment loops there are `k[n] = 0` and `k[n] = r[n]`, which carry no table
#  lookup, so there is nothing for the fix in MomentCollision.hpp to remove.
#
#  USAGE
#      tests/frame_check.sh [<kokkos build dir>]     # default: build_th
#==============================================================================
set -e

BUILD=${1:-build_th}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

K=$BUILD/_deps
if [ ! -d "$K/kokkos-src" ]; then
  echo "no Kokkos sources under $K -- pass a configured Kokkos build dir" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/probe.cpp" <<'CPP'
#include "collision/BGK.hpp"
#include "collision/MomentCollision.hpp"
#include "collision/MultiphaseCentralMoments.hpp"
#include "collision/PhaseFieldCentralMoments.hpp"
#include "collision/ColourGradient.hpp"
#include "collision/MhdCentralMoments.hpp"
#include "collision/MhdCentralMomentsShifted.hpp"
using namespace lbm;
using Bgk  = BGK<D3Q27, SecondOrderEquilibrium<D3Q27>, NoForcing, RawPopulations>;
using CM   = MomentCollision<D3Q27, NoForcing, RawPopulations, true>;
using MpCM = MultiphaseCentralMoments<D3Q27>;
using PfCM = PhaseFieldCentralMoments<D3Q27>;
using CG   = ColourGradient<D3Q27>;
using MhdS = MhdCentralMomentsShifted<D3Q27>;
using MhdM = MhdCentralMoments<D2Q9, true>;
extern "C" void probe_bgk (Real* f, const Macro* m, const Bgk*  c) { c->collide(f, *m, 0); }
extern "C" void probe_cm  (Real* f, const Macro* m, const CM*   c) { c->collide(f, *m, 0); }
extern "C" void probe_mpcm(Real* f, const Macro* m, const MpCM* c) { c->collide(f, *m, 0); }
extern "C" void probe_pfcm(Real* h, Real phi, const Real* u, const Real* A, const PfCM* c) {
  c->collide(h, phi, u, A);
}
extern "C" void probe_cg(Real* f, Real rho, const Real* u, Real p, const CG* c) {
  c->collide(f, rho, u, p, 0);
}
extern "C" void probe_mhds(Real* f, const Macro* m, const MhdS* c) { c->collide(f, *m, 0); }
extern "C" void probe_mhdm(Real* f, const Macro* m, const MhdM* c) { c->collide(f, *m, 0); }
CPP

for PREC in double float; do
  DEF=""
  [ "$PREC" = float ] && DEF="-DLBM_SINGLE_PRECISION"

  ${CXX:-clang++} -std=c++20 -O3 -S $DEF -o "$TMP/probe.s" "$TMP/probe.cpp" \
    -Isrc \
    -I"$K/kokkos-build" -I"$K/kokkos-build/core/src" -I"$K/kokkos-src/core/src" \
    -I"$K/kokkos-build/containers/src" -I"$K/kokkos-src/containers/src" \
    -I"$K/kokkos-build/algorithms/src" -I"$K/kokkos-src/algorithms/src" \
    -I"$K/kokkos-build/simd/src" -I"$K/kokkos-src/simd/src" \
    -isystem "$K/kokkos-src/tpls/desul/include" \
    -isystem "$K/kokkos-src/tpls/mdspan/include"

  echo
  echo "  precision $PREC"
  printf "  %-28s %7s %8s %7s %8s\n" operator frame instrs loops regidx
  for FN in probe_bgk probe_cm probe_mpcm probe_pfcm probe_cg probe_mhds probe_mhdm; do
    LN=$(awk -v f="_$FN:" '$0 ~ "^"f {print NR; exit}' "$TMP/probe.s")
    awk -v s="$LN" 'NR>=s{print} NR>s && /\.cfi_endproc/{exit}' "$TMP/probe.s" > "$TMP/b.s"
    # A tail call means the body was not inlined into the probe; follow it,
    # otherwise the frame reported is the trampoline's and always zero.
    T=$(grep -oE "^	b	_*ZN[A-Za-z0-9_]+" "$TMP/b.s" | head -1 | sed 's/^	b	//')
    if [ -n "$T" ]; then
      LN=$(awk -v f="$T:" '$0 ~ "^"f {print NR; exit}' "$TMP/probe.s")
      awk -v s="$LN" 'NR>=s{print} NR>s && /\.cfi_endproc/{exit}' "$TMP/probe.s" > "$TMP/b.s"
    fi
    FR=$(grep -oE 'sub[ 	]+sp, sp, #[0-9]+' "$TMP/b.s" | head -1 | grep -oE '[0-9]+$' || true)
    [ -z "$FR" ] && FR=0
    printf "  %-28s %7s %8s %7s %8s\n" "$FN" "$FR" \
      "$(grep -cE '^	[a-z]' "$TMP/b.s" || true)" \
      "$(grep -c 'Inner Loop Header' "$TMP/b.s" || true)" \
      "$(grep -cE '\[x[0-9]+, (x|w)[0-9]+' "$TMP/b.s" || true)"
  done
done
echo
echo "  see the banner in this script for what the columns mean"
