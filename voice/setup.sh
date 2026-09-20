#!/usr/bin/env bash
# setup.sh -- bring a machine to the state jarvis.py refuses to start without,
# and PROVE it, rather than asserting it.
#
# Run it as often as you like: every step detects what is already there and does
# nothing.  A fully satisfied re-run measured 8.60, 8.39 and 8.39 s here -- of
# which 5.0 s is the microphone probe and 2.4 s the whisper round trip, i.e.
# almost all of it is the two things that CANNOT be cached because they are the
# two most likely to have broken since last time.  --checksum adds 5.4 s.  A live
# `claude -p` probe adds several seconds more; it could not be timed here, because
# claude is not installed on this machine yet.
#
#     ./voice/setup.sh            # install what is missing, prompting first
#     ./voice/setup.sh --check    # touch nothing; report only
#     ./voice/setup.sh --yes      # take the prompts as answered yes
#
# EVERY NUMBER BELOW WAS MEASURED ON THIS MACHINE (macOS 27.0 Darwin 27.0.0, M1,
# 8 GB, bash 3.2.57, Python 3.14.7) ON 2026-09-20.  Some were taken while the
# machine was loaded and some while it was idle, and they differ by 5x; where that
# matters it is said.  CLAUDE.md's Measurement discipline applies here too.
#
# ---------------------------------------------------------------------------
# IT CHECKS BINARIES, NOT BREW FORMULAE, AND THAT IS NOT PEDANTRY
#
# `brew list` on this machine shows ffmpeg AND ffmpeg-full, both 9.0.2, and both
# provide `ffmpeg`.  A check for the formula "ffmpeg" would have reinstalled a
# binary that was already there via the other one.  In the other direction the
# formula that provides whisper-server is `whisper.cpp` -- with a dot --
# and `whisper-cpp`, which is what every write-up calls it, is only an ALIAS:
#
#     $ brew list --versions whisper-cpp   ->  whisper.cpp 1.9.4
#     $ ls -la /opt/homebrew/bin/whisper-server
#       -> ../Cellar/whisper.cpp/1.9.4/bin/whisper-server
#
# So this file tests `command -v` for the thing it actually needs and only
# consults brew when something is missing.  A machine that built whisper.cpp from
# source, or installed ffmpeg some other way, passes -- which is right, because
# the daemon calls binaries, not formulae.
#
# ---------------------------------------------------------------------------
# THERE IS NO `timeout` ON THIS MACHINE AND EVERY PROBE HERE CAN HANG
#
#     $ command -v timeout gtimeout   ->  (nothing)
#
# Both are absent, so every `timeout 10 whisper-...` line copied from a README
# fails with command-not-found -- and each probe below is one that genuinely hangs
# when it goes wrong: ffmpeg blocks on a microphone that never delivers a byte,
# whisper-server blocks for ~115 s on a cold Metal cache, and `claude -p` has NO
# wall-clock bound at all (--max-turns and --max-budget-usd cap agentic turns and
# spend, not time, so one long Bash tool call blocks the turn indefinitely).
#
# `cap SECONDS argv...` below is the replacement.  It delegates to python3 rather
# than to a bash `sleep ... ; kill` watchdog for one reason: killing the direct
# child is not enough.  `claude` spawns node and bash GRANDCHILDREN that survive
# it and keep the pipe open.  python3's start_new_session=True plus os.killpg
# takes the whole process group, which was verified rather than assumed:
#
#     cap 2 sleep 30                      -> rc=124 after 2 s
#     cap 2 sh -c 'sleep 60 & sleep 60'   -> rc=124, and pgrep found NO orphans
#     cap 5 definitely-not-a-binary       -> rc=127, message on stderr
#     out=$(cap 5 /bin/echo captured)     -> out="captured"   (stdout passes through)
#
# 124 is GNU timeout's convention and is reused here so the call sites read the
# same way.  python3 is a safe dependency: jarvis.py is written in it.
#
# ---------------------------------------------------------------------------
# THE GLOBAL npm INSTALL IS ALWAYS ASKED FOR, NEVER PERFORMED SILENTLY
#
# `npm i -g @anthropic-ai/claude-code` writes into a directory shared by every
# node project on the machine ($(npm prefix -g)/lib/node_modules, here
# /opt/homebrew, which is WRITABLE so no sudo is involved and nothing will stop
# it).  That is the user's machine to change, not this script's, so the exact
# command and its destination are printed and a y/N answer is taken first.
#
# `--yes` covers the brew installs and the model downloads (565 MB), because those
# are recoverable and local.  IT DELIBERATELY DOES NOT COVER THE GLOBAL npm
# INSTALL, which needs either a typed `y` or the dedicated `--install-claude`.
#
# That separation is not theoretical caution; it is a correction.  While testing
# the model-download path here, this script was run with `--yes` for the models
# and it installed @anthropic-ai/claude-code@2.1.278 into
# /opt/homebrew/lib/node_modules at 07:55:08 as a side effect.  Nobody had
# decided that.  A blanket yes aimed at a 74 MB file reached a package shared by
# every node project on the machine, which is precisely what "never install it
# silently" was meant to prevent -- so the two consents are now separate flags,
# and the general one is not enough for the global one.
#
# A non-interactive run with neither flag DECLINES rather than proceeding: a
# setup script invoked from something else must not be how a global package
# arrives.
#
# ---------------------------------------------------------------------------
# MODELS: SIZE EVERY RUN, CHECKSUM ON DEMAND, AND NEVER A PARTIAL FILE IN PLACE
#
# The two checksums below were verified from BOTH ends and agree exactly:
#
#     $ shasum -a 256 ~/.cache/whisper-models/*.bin
#       921e4cf8...920b1f  ggml-tiny.en.bin
#       c6138d6d...c41e5d  ggml-small.en.bin
#     $ curl -s https://huggingface.co/api/models/ggerganov/whisper.cpp/tree/main
#       ggml-tiny.en.bin  77704715   921e4cf8...920b1f     <- LFS oid, upstream
#       ggml-small.en.bin 487614201  c6138d6d...c41e5d
#
# so the local files are byte-identical to what the repository publishes, and the
# constants below are a real pin rather than a note of what happened to be on disk.
# The HTTP headers agree a third time: x-linked-size is 77704715 and 487614201.
#
# Hashing both costs 5.02 s of CPU and 5.79 s wall for 565 MB, which is over a
# third of a whole clean run, so it is NOT done every time.  The size check is
# free and catches the failure that actually happens -- a truncated download --
# so it runs always; --checksum forces the full hash; and a file this script has
# just downloaded is ALWAYS hashed before it is accepted, because that is the one
# moment the bytes are new.
#
# A download lands on `<name>.part` and is renamed only after both checks pass.
# That is load-bearing rather than tidy: an interrupted curl writing straight to
# the final name leaves a short file that the NEXT run sees as "present".  The
# .part file is also what `curl -C -` resumes from, so an interrupted 488 MB
# download continues instead of restarting.
#
# THE RESUME WAS EXERCISED FOR REAL, NOT REASONED ABOUT.  A tiny.en download into
# a scratch directory died at 65,262,951 of 77,704,715 bytes (84 %).  The run
# reported "download failed; the partial file is kept" and FAILED the check rather
# than installing 84 % of a model; the next run printed "resuming 62 MB already
# fetched", completed it, hashed it to 921e4cf8...920b1f, and moved it into place.
# The self-test then transcribed against the file it had just fetched.  That is
# the whole .part design, end to end, and it was luck that the network broke
# during the one run that tested it.
#
# AND THE SIZE CHECK REALLY IS NOT ENOUGH, which was demonstrated rather than
# argued.  A file of EXACTLY 77704715 zero bytes was planted as ggml-tiny.en.bin:
#
#     without --checksum   "tiny.en (wake tier)   74 MB, size ok      PASS"
#     with    --checksum   "tiny.en (wake tier)   sha256 mismatch     FAIL"
#
# That is the whole argument for the flag.  Feeding the corrupt model to a server
# then failed in 3.3 s rather than hanging, and whisper-server DID say why --
# "whisper_model_load: invalid model data (bad magic)" -- but it said it in its
# own log, which nothing reads.  The daemon would have seen only "the server did
# not come up".  So the loud message exists and is invisible where it matters,
# which is why section 4 tails that log on failure and points at --checksum.
#
# ---------------------------------------------------------------------------
# THE SELF-TEST AND THE METAL PRE-WARM ARE THE SAME OPERATION
#
# The first ggml run ever on a machine compiles Metal shaders for ~112-116 s
# (measured twice: 112.411 s from `whisper-cli --help`, 115.74 s from a cold
# transcribe; the very next run took 0.18 s and 3.23 s).  It looks EXACTLY like a
# hung process, and the first person to meet it will be the user starting the
# daemon.  So it is spent here, once, behind a message that says what is happening.
#
# Rather than burn that on a throwaway `--help`, it is spent on the real thing:
# `say` renders a sentence straight into whisper's native format (16 kHz mono
# s16le, no ffmpeg step), a whisper-server is started, and the clip is POSTed.
# One operation proves the shader cache is warm AND that the model loads AND that
# the HTTP surface the daemon depends on works AND that the prose prompt biases
# the way it is supposed to -- WITH NO MICROPHONE, which matters because the
# microphone is the part most likely to be broken.  Measured here, warm:
#
#     say -o clip.wav --data-format=LEI16@16000 ...   1.06 s   (idle machine)
#     whisper-server tiny.en, launch to ready          0.32 s
#     POST /inference, three reps                      0.14 / 0.11 / 0.11 s
#     transcript  " Jarvis, run the colour gradient case at gamma 100."
#
# and the prompt earned its place in the same run, one variable changed:
#
#     with the prose prompt     "...the colour gradient case..."   (British)
#     without it                "...the color gradient case..."
#
# GET / answering is NOT readiness.  It returned 200 after 0.32 s here, but the
# port can bind before the model has finished loading, so the POST is the proof
# and the GET is only how we avoid hammering.  jarvis.py keeps a warm-up POST for
# the same reason and this file mirrors it deliberately.
#
# Port 18099 is used, not the daemon's 18080/18081, so that running this while
# jarvis is up neither collides nor quietly tests jarvis's server instead of a
# fresh one.
#
# ---------------------------------------------------------------------------
# THE MICROPHONE REMEDY IS NOT WRITTEN HERE, ON PURPOSE
#
# "Digital silence" has THREE causes and permissions is only the third most likely
# on this machine.  On the same ffmpeg invocation, from the same process, in the
# same second, the built-in "MacBook Air Microphone" returned 48000 consecutive
# samples of EXACT zero while the Continuity iPhone microphone returned peak 811 --
# and a TCC denial cannot be device-selective, so that machine had the grant and
# one device was dead.  Two hours later the same built-in device read peak 31,
# zero_frac 0.053: LIVE, merely a quiet room.
#
# Sending someone into System Settings to fix a permission that was never revoked
# is worse than saying nothing, so the ordered remedy (wrong device, dead device,
# TCC, input volume) lives in miccheck.py's REMEDY next to the measurement that
# justifies its order, and this file CALLS miccheck rather than restating it.  Two
# copies of that text would drift and the wrong one would be read.  What is added
# here, because miccheck has no reason to know it, is the click-path and the URL
# that opens the pane directly.
#
# The discriminator is the fraction of EXACTLY-ZERO samples, not amplitude: live
# captures measured 3.4 % to 42.5 % zeros, a dead one 100.0000 %.  An amplitude
# threshold cannot tell a blocked microphone from a quiet room -- both read "low".
#
# ---------------------------------------------------------------------------
# THE claude CHECKS RUN IN A STRIPPED ENVIRONMENT, OR THEY MEASURE THE WRONG THING
#
# This script will usually be run from inside a Claude Code session, because that
# is how the repository is worked on.  That session exports 24 variables here:
#
#     CLAUDECODE, CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE_SESSION_ID,
#     CLAUDE_CODE_MESSAGING_SOCKET, CLAUDE_CODE_MESSAGING_TOKEN,
#     CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH, ANTHROPIC_BASE_URL, ... (24 in total)
#
# A child that inherits those behaves differently from the one a plain terminal
# will start, so a check run under them proves nothing about the daemon.  Every
# claude invocation below is made with CLAUDE_*, CLAUDECODE, ANTHROPIC_BASE_URL and
# ANTHROPIC_API_KEY removed -- which is exactly what jarvis.py does to build its
# child environment, for the same reason.
#
# ANTHROPIC_API_KEY is the dangerous one and it is checked for separately, in the
# environment AND in the five shell profiles, because in -p mode "the key is always
# used when present" with no approval step: a stray export moves every request off
# the subscription silently.  It is absent from both here today.  ANTHROPIC_BASE_URL
# is merely pointless -- it is an endpoint override with no role in credential
# selection, absent from the documented precedence list -- so it is reported as a
# note, not a failure, and stripped anyway.
#
# AUTH IS NOT INHERITED FROM THE DESKTOP APP AND THE KEYCHAIN DOES NOT HELP.  The
# "Claude Code-credentials" item exists and its 3067-byte secret reads back with
# rc=0 and no prompt, and `claude auth status` still answers {"loggedIn": false,
# "authMethod": "none"}: the desktop app authenticates its child over a host
# socket, not through anything a standalone CLI reads.  A one-time interactive
# `claude auth login` is required and this script cannot do it for you -- it is a
# browser flow -- so it stops and says so.
#
# THE PROBE BRANCHES ON is_error, NEVER ON subtype AND NEVER ON THE EXIT CODE.
# A measured auth-failure envelope carried "subtype": "success" WITH
# "is_error": true, terminal_reason "api_error", and the sentence "Not logged in -
# Please run /login" in `result` on STDOUT.  A check that trusted subtype would
# report success; one that used --output-format text could not tell an answer from
# an error at all.
#
# JSON is parsed with python3 and not with jq.  jq is present here (/usr/bin/jq)
# but is not part of a stock macOS, and python3 is already required.
#
# ---------------------------------------------------------------------------
# WHAT THIS DOES NOT DO
#
#   * It does not install Homebrew.  If brew is missing it says so and stops;
#     installing a package manager is not a thing a setup script should do behind
#     a y/N.
#   * It does not run `claude auth login`.  That is an interactive browser flow.
#   * It does not download the silero VAD model.  jarvis.py does not use it (its
#     VAD is energy-based, in Python), and fetching an unused 885 kB model would
#     imply otherwise.
#   * It does not install a LaunchAgent.  The microphone grant belongs to the APP
#     that owns the process tree and a LaunchAgent has no app identity, so a daemon
#     started that way meets the digital-zero stream jarvis.py refuses to run
#     against.  Start it from a terminal you have granted.
#   * It does not configure the microphone, set the input volume, or change the
#     output device.  It only reports.
#   * It does not prune ~/.claude/projects, which already holds a 260 MB and a
#     121 MB transcript on this 8 GB machine and which a chatty daemon will grow.
#   * It does not verify the daemon end to end.  It proves every component and the
#     speech path; the first real utterance is still the first real utterance.
#   * It is macOS-only and says so rather than degrading: avfoundation, `say`,
#     `afplay` and the TCC model have no counterpart elsewhere.
#
# ---------------------------------------------------------------------------
# shellcheck was NOT available on this machine (`command -v shellcheck` -> nothing),
# so the usual claim of a clean run cannot honestly be made.  What was done
# instead: `bash -n` and `sh -n` parse clean, and the file is written to bash 3.2
# because /bin/bash here IS 3.2.57 and `/usr/bin/env bash` resolves to it -- so no
# associative arrays, no ${var,,}, no mapfile, and every array expansion that can
# be empty is guarded, since "${arr[@]}" on an empty array is an unbound-variable
# error under `set -u` in 3.2.  Run shellcheck if you install it.
#
# One `set -e` trap was checked rather than assumed, because it bites exactly the
# clean machine this script is for: `A && B` failing does NOT exit under set -e
# (verified), but a function whose LAST statement is such a list returns nonzero
# and the caller does (verified, and it exits silently).  Every function here ends
# in an explicit `return` or a printf for that reason.
#
# Exercised for real, on this machine: absent models, a truncated model, a
# right-size wrong-content model, an interrupted download and its resume, a
# corrupt model fed to the server, argument errors, --check, --checksum,
# --model-dir, and the global-install refusal under --yes.  NOT exercised: the
# brew install branches (nothing was missing) and the `--install-claude` proceed
# branch (the package was already installed by then -- see the npm note above).

set -euo pipefail

# --------------------------------------------------------------------------
# Constants.  The model names, the port base and the model directory MUST agree
# with jarvis.py; the "agrees with jarvis.py" check below asserts that rather
# than trusting this comment.
MODEL_DIR="${HOME}/.cache/whisper-models"
MODEL_BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

TINY_NAME="ggml-tiny.en.bin"
TINY_SIZE=77704715
TINY_SHA="921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f"

SMALL_NAME="ggml-small.en.bin"
SMALL_SIZE=487614201
SMALL_SHA="c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"

SELFTEST_PORT=18099          # deliberately NOT the daemon's 18080/18081
SERVER_READY_S=200           # a cold Metal cache costs ~115 s of that
MIC_CAP_S=90                 # miccheck probes every device; each has its own 20 s
CLAUDE_PROBE_CAP_S=120

NPM_PKG="@anthropic-ai/claude-code"

# The prose prompt, trimmed to the clause that demonstrates the bias.  The full
# 142-word version lives in jarvis.py as WHISPER_PROMPT; this is a self-test, not
# a second copy of the daemon's configuration, and it is short on purpose so that
# nobody edits this one believing they have changed the daemon's.
SELFTEST_PROMPT="We are working on M3LB, a lattice Boltzmann solver written in C++ on Kokkos. The lattices are D2Q9, D2Q5, D3Q7 and D3Q27, and the collision operators are BGK, TRT, MRT, central moments, the colour gradient and the conservative Allen-Cahn phase field."
SELFTEST_TEXT="Jarvis, run the colour gradient case at gamma one hundred"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

# --------------------------------------------------------------------------
# Options
DO_INSTALL=1
ASSUME_YES=0
DO_CHECKSUM=0
DO_MIC=1
DO_CLAUDE_PROBE=1
MIC_DEVICE=""
MODEL_DIR_SET=0
INSTALL_CLAUDE=0

usage() {
    cat <<USAGE
usage: setup.sh [options]

  --check           report only: install nothing, download nothing.
                    Probes still run -- they open the microphone and start a
                    whisper-server, neither of which changes the machine.
  --yes             take the brew and model-download prompts as answered yes.
                    It does NOT cover the global npm install of Claude Code --
                    that one is shared by every node project on the machine and
                    needs a typed 'y' or --install-claude.
  --install-claude  permission to run 'npm i -g @anthropic-ai/claude-code'
                    without a prompt. Nothing else implies it.
  --checksum        sha256 the whisper models even when their size is right
                    (5.79 s for 565 MB here).  A freshly downloaded model is
                    always hashed regardless of this flag.
  --no-mic          skip the microphone probe.
  --device NAME     probe only this microphone (index, exact name or substring);
                    passed through to miccheck.py.
  --model-dir DIR   look for the whisper models here instead of
                    ~/.cache/whisper-models. jarvis.py takes the same flag and
                    you must pass it to BOTH -- this script cannot make the
                    daemon look somewhere else, it can only check somewhere else.
  --no-claude-probe skip the live 'claude -p' round trip (it costs a few cents
                    and needs the network).  Installation and login are still
                    checked.
  -h, --help        this text.

exit status: 0 if every check passed, 1 if any FAILED.  WARN does not fail.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check|-n)        DO_INSTALL=0 ;;
        --yes|-y)          ASSUME_YES=1 ;;
        --install-claude)  INSTALL_CLAUDE=1 ;;
        --checksum)        DO_CHECKSUM=1 ;;
        --no-mic)          DO_MIC=0 ;;
        --no-claude-probe) DO_CLAUDE_PROBE=0 ;;
        --device)          shift; [ $# -gt 0 ] || { echo "--device needs a value" >&2; exit 2; }; MIC_DEVICE="$1" ;;
        --model-dir)       shift; [ $# -gt 0 ] || { echo "--model-dir needs a value" >&2; exit 2; }; MODEL_DIR="$1"; MODEL_DIR_SET=1 ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# --------------------------------------------------------------------------
# Result table.  A temp file rather than an array: bash 3.2 has no associative
# arrays and an empty "${arr[@]}" is an error under set -u.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/m3lb-voice-setup.XXXXXX")"
RESULTS="${WORK}/results.tsv"
: > "$RESULTS"
SRV_PID=""

cleanup() {
    if [ -n "$SRV_PID" ]; then
        kill "$SRV_PID" 2>/dev/null || true
        sleep 0.4
        kill -9 "$SRV_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

hdr()  { printf '\n%s\n' "$1"; printf '%s\n' "------------------------------------------------------------------------"; }
info() { printf '  %s\n' "$1"; }
detail() { printf '      %s\n' "$1"; }

record() {   # record VERDICT NAME VALUE
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$RESULTS"
}
pass() { record PASS "$1" "$2"; printf '  %-46s %-24s PASS\n' "$1" "$2"; }
warn() { record WARN "$1" "$2"; printf '  %-46s %-24s WARN\n' "$1" "$2"; }
fail() { record FAIL "$1" "$2"; printf '  %-46s %-24s FAIL\n' "$1" "$2"; }

# cap SECONDS argv... -- see the banner.  124 on timeout, 127 if unrunnable.
cap() {
    local secs="$1"; shift
    python3 - "$secs" "$@" <<'PY'
import os, signal, subprocess, sys
cap = float(sys.argv[1]); argv = sys.argv[2:]
try:
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
except (OSError, ValueError) as exc:
    print("cap: cannot run %r: %s" % (argv[:1], exc), file=sys.stderr)
    sys.exit(127)
try:
    sys.exit(p.wait(timeout=cap))
except subprocess.TimeoutExpired:
    for sig, grace in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            p.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            continue
    sys.exit(124)
PY
}

# ask QUESTION -- 0 for yes.  Never yes by default, never yes without a tty.
ask() {
    if [ "$DO_INSTALL" -eq 0 ]; then
        detail "--check: not doing it."
        return 1
    fi
    if [ "$ASSUME_YES" -eq 1 ]; then
        detail "--yes: proceeding."
        return 0
    fi
    if [ ! -t 0 ]; then
        detail "not a terminal and --yes was not given: declining."
        return 1
    fi
    local reply=""
    printf '      %s [y/N] ' "$1"
    read -r reply || reply=""
    case "$reply" in
        y|Y|yes|YES) return 0 ;;
        *)           detail "declined."; return 1 ;;
    esac
}

# The global npm install gets its OWN consent. --yes does not reach it: see the
# banner for the run in which a blanket --yes, meant for a model download,
# installed a machine-wide package nobody had decided on.
ask_npm_global() {
    if [ "$DO_INSTALL" -eq 0 ]; then
        detail "--check: not doing it."
        return 1
    fi
    if [ "$INSTALL_CLAUDE" -eq 1 ]; then
        detail "--install-claude: proceeding."
        return 0
    fi
    if [ ! -t 0 ]; then
        detail "Declining: this is a machine-wide install, there is no terminal to"
        detail "ask at, and --yes deliberately does not cover it. Re-run"
        detail "interactively, pass --install-claude, or run the command yourself."
        return 1
    fi
    if [ "$ASSUME_YES" -eq 1 ]; then
        detail "(--yes does not cover a global npm install. Asking anyway.)"
    fi
    local reply=""
    printf '      %s [y/N] ' "$1"
    read -r reply || reply=""
    case "$reply" in
        y|Y|yes|YES) return 0 ;;
        *)           detail "declined."; return 1 ;;
    esac
}

# The names of every inherited variable that must not reach a claude child.
claude_env_unsets() {
    env | sed -n 's/^\(CLAUDECODE\|CLAUDE_[A-Za-z0-9_]*\|ANTHROPIC_BASE_URL\|ANTHROPIC_API_KEY\)=.*/\1/p' | sort -u
}

# run_claude CAP argv... -- claude, in a stripped environment, under a time cap.
run_claude() {
    local secs="$1"; shift
    local unsets="" k=""
    while IFS= read -r k; do
        [ -n "$k" ] && unsets="${unsets} -u ${k}"
    done <<< "$(claude_env_unsets)"
    # shellcheck disable=SC2086
    cap "$secs" env $unsets "$@"
}

bytes_of() { # portable-ish stat
    if [ -f "$1" ]; then stat -f%z "$1" 2>/dev/null || echo 0; else echo 0; fi
}

human_mb() { python3 -c "import sys;print('%.0f MB'%(int(sys.argv[1])/1048576.0))" "$1"; }

printf '\n'
printf 'M3LB voice assistant -- setup\n'
printf '%s\n' "========================================================================"
printf '  repo      %s\n' "$REPO"
printf '  models    %s\n' "$MODEL_DIR"
if [ "$DO_INSTALL" -eq 0 ]; then
    printf '  mode      --check: nothing will be installed or downloaded\n'
else
    printf '  mode      install what is missing, asking first\n'
fi

# ==========================================================================
hdr "0. platform"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Darwin" ]; then
    fail "macOS" "$OS_NAME"
    detail "This assistant is avfoundation + say + afplay + TCC. There is no"
    detail "degraded mode to fall back to; it does not run here."
    printf '\n'
    exit 1
fi
pass "macOS" "Darwin $(uname -r) $(uname -m)"

if command -v python3 >/dev/null 2>&1; then
    PY_VER="$(python3 -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || echo "?")"
    PY_MAJOR_OK="$(python3 -c 'import sys;print(1 if sys.version_info[:2] >= (3,9) else 0)' 2>/dev/null || echo 0)"
    if [ "$PY_MAJOR_OK" = "1" ]; then
        pass "python3" "$PY_VER"
    else
        fail "python3 >= 3.9" "$PY_VER"
    fi
    # Not a failure -- it is the REASON the Python here is stdlib-only and shells
    # out to ffmpeg.  Worth printing so nobody "fixes" jarvis.py with audioop.
    if python3 -c 'import audioop' >/dev/null 2>&1; then
        warn "audioop" "present"
        detail "This Python still has audioop. jarvis.py does not use it anyway:"
        detail "it was removed in 3.13 (PEP 594) and the daemon must run on 3.14."
    else
        pass "audioop absent (PEP 594)" "array('h') is used"
    fi
else
    fail "python3" "not on PATH"
fi

pass "bash" "${BASH_VERSION%%(*}"

# ==========================================================================
hdr "1. the daemon's own files"

AGREE_OK=1
for f in jarvis.py miccheck.py; do
    if [ -f "${HERE}/${f}" ]; then
        if python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "${HERE}/${f}" >/dev/null 2>&1; then
            pass "$f" "parses"
        else
            fail "$f" "syntax error"
            AGREE_OK=0
        fi
    else
        fail "$f" "missing"
        AGREE_OK=0
    fi
done

# Two files that must agree about the same three facts is exactly the shape this
# tree keeps finding bugs in, so it is asserted rather than commented.  If
# jarvis.py's model names, model directory or port base are edited, this fails
# here instead of at 3 a.m. against a model that is not where it was looked for.
if [ -f "${HERE}/jarvis.py" ]; then
    MISMATCH=""
    grep -q "$TINY_NAME"  "${HERE}/jarvis.py" || MISMATCH="${MISMATCH} tiny-name"
    grep -q "$SMALL_NAME" "${HERE}/jarvis.py" || MISMATCH="${MISMATCH} small-name"
    if [ "$MODEL_DIR_SET" -eq 0 ]; then
        grep -q '"whisper-models"' "${HERE}/jarvis.py" || MISMATCH="${MISMATCH} model-dir"
    fi
    grep -q 'default=18080' "${HERE}/jarvis.py" || MISMATCH="${MISMATCH} port-base"
    if [ -n "$MISMATCH" ]; then
        : # reported below
    elif [ "$MODEL_DIR_SET" -eq 1 ]; then
        # The agreement this check exists to enforce has been opted out of: the
        # models are being checked somewhere the daemon will not look unless it
        # is told to. Saying PASS here would be a lie.
        warn "setup.sh agrees with jarvis.py" "--model-dir overrides"
        detail "Checking ${MODEL_DIR}, which is NOT jarvis.py's default. Pass the"
        detail "same --model-dir to the daemon or it will look in"
        detail "~/.cache/whisper-models and find nothing."
    else
        pass "setup.sh agrees with jarvis.py" "names, dir, port"
    fi
    if [ -n "$MISMATCH" ]; then
        fail "setup.sh agrees with jarvis.py" "differs:${MISMATCH}"
        detail "This script and the daemon disagree about where the models live or"
        detail "what they are called. Fix one of them before trusting either."
    fi
fi

# ==========================================================================
hdr "2. binaries"

# Checked by BINARY, not by formula -- see the banner.  Formula names are only
# used to offer an install when the binary is genuinely absent.
need_binary() {   # need_binary BINARY FORMULA DESCRIPTION
    local bin="$1" formula="$2" what="$3" path=""
    path="$(command -v "$bin" 2>/dev/null || true)"
    if [ -n "$path" ]; then
        pass "$bin" "$what"
        return 0
    fi
    fail "$bin" "not on PATH"
    if [ -z "$formula" ]; then
        detail "$bin ships with macOS; a machine without it is broken in a way"
        detail "this script cannot repair."
        return 1
    fi
    if ! command -v brew >/dev/null 2>&1; then
        detail "brew is not installed either, so this cannot be offered."
        detail "Install Homebrew from https://brew.sh, then re-run."
        return 1
    fi
    detail "provided by the Homebrew formula '${formula}'"
    if ask "run: brew install ${formula} ?"; then
        if brew install "$formula"; then
            path="$(command -v "$bin" 2>/dev/null || true)"
            [ -n "$path" ] && detail "installed: $path"
        else
            detail "brew install failed."
        fi
    fi
    return 1
}

FFMPEG_VER="?"
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_VER="$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
fi
need_binary ffmpeg ffmpeg "$FFMPEG_VER" || true

WHISPER_VER="?"
if command -v whisper-cli >/dev/null 2>&1; then
    # --version goes through the ggml backend load, so it is capped.
    WHISPER_VER="$(cap 30 whisper-cli --version 2>/dev/null | sed -n 's/.*version: *//p' | head -1 || true)"
    [ -z "$WHISPER_VER" ] && WHISPER_VER="installed"
fi
# The formula is whisper.cpp -- with a dot.  whisper-cpp is only an alias.
need_binary whisper-server "whisper.cpp" "whisper.cpp ${WHISPER_VER}" || true
need_binary whisper-cli    "whisper.cpp" "whisper.cpp ${WHISPER_VER}" || true

need_binary say    "" "macOS TTS"      || true
need_binary afplay "" "macOS playback" || true
need_binary curl   "" "$(curl --version 2>/dev/null | head -1 | awk '{print $2}')" || true
need_binary shasum "" "sha256"         || true

# Stated rather than checked: every `timeout 10 ...` line in any README fails here.
if command -v timeout >/dev/null 2>&1 || command -v gtimeout >/dev/null 2>&1; then
    warn "GNU timeout" "present"
    detail "Unusual on macOS. Nothing here or in jarvis.py uses it; both use a"
    detail "python3 process-group wrapper instead. No action needed."
else
    pass "GNU timeout absent (expected)" "python3 cap() used"
fi

# ==========================================================================
hdr "3. whisper models"

mkdir -p "$MODEL_DIR"

# The explanation is written to $WHY rather than printed, so that check_model can
# print it UNDER its verdict line. A reason that appears above the verdict it
# explains reads as if it belongs to the previous check.
WHY="${WORK}/why.txt"

verify_model() {   # verify_model NAME EXPECTED_SIZE EXPECTED_SHA FORCE_HASH
    local name="$1" want_size="$2" want_sha="$3" force_hash="$4"
    local path="${MODEL_DIR}/${name}" got_size="" got_sha=""
    : > "$WHY"
    got_size="$(bytes_of "$path")"
    if [ "$got_size" = "0" ]; then
        return 2                      # absent
    fi
    if [ "$got_size" != "$want_size" ]; then
        {
            echo "size is ${got_size} bytes, expected ${want_size} -- truncated,"
            echo "or a different model under the same name."
        } > "$WHY"
        return 3
    fi
    if [ "$force_hash" -eq 1 ]; then
        got_sha="$(shasum -a 256 "$path" | awk '{print $1}')"
        if [ "$got_sha" != "$want_sha" ]; then
            {
                echo "the size is right and the CONTENT is not."
                echo "sha256   ${got_sha}"
                echo "expected ${want_sha}"
            } > "$WHY"
            return 4
        fi
    fi
    return 0
}

fetch_model() {   # fetch_model NAME EXPECTED_SIZE EXPECTED_SHA
    local name="$1" want_size="$2" want_sha="$3"
    local url="${MODEL_BASE_URL}/${name}"
    local final="${MODEL_DIR}/${name}"
    local part="${final}.part"
    local got_size="" got_sha=""

    detail "source  ${url}"
    detail "size    $(human_mb "$want_size")"
    if ! ask "download ${name} ?"; then
        return 1
    fi
    # -f so an HTTP error page is never written into a model file; -C - to resume
    # the .part from an interrupted run; --retry for a flaky link.
    if [ -f "$part" ]; then
        detail "resuming $(human_mb "$(bytes_of "$part")") already fetched"
    fi
    if ! curl -fL --retry 3 --retry-delay 2 --progress-bar -C - -o "$part" "$url"; then
        detail "download failed; the partial file is kept at ${part} and the next"
        detail "run will resume it."
        return 1
    fi
    got_size="$(bytes_of "$part")"
    if [ "$got_size" != "$want_size" ]; then
        detail "downloaded ${got_size} bytes, expected ${want_size}. Not installing it."
        return 1
    fi
    # Always hashed: these bytes are new, and this is the one moment it is cheap
    # relative to what has just been spent.
    detail "verifying sha256..."
    got_sha="$(shasum -a 256 "$part" | awk '{print $1}')"
    if [ "$got_sha" != "$want_sha" ]; then
        detail "sha256 MISMATCH -- got ${got_sha}"
        detail "kept at ${part} for inspection; not installing it."
        return 1
    fi
    mv "$part" "$final"
    detail "installed ${final}"
    return 0
}

check_model() {   # check_model LABEL NAME SIZE SHA
    local label="$1" name="$2" size="$3" sha="$4" rc=0 fetched=0 how=""

    verify_model "$name" "$size" "$sha" "$DO_CHECKSUM" || rc=$?

    # Remediate BEFORE recording a verdict. Recording the failure first and the
    # download second would leave a FAIL in the summary for a model that is now
    # correctly installed, and the run would end "SETUP INCOMPLETE" having just
    # completed the setup.
    #
    # Only an ABSENT model is offered as a download. A wrong-size or wrong-hash
    # file is left exactly where it is: overwriting the evidence of a bad
    # download with another download hides why it happened.
    if [ "$rc" = "2" ]; then
        info "${label}: not in ${MODEL_DIR}"
        if fetch_model "$name" "$size" "$sha"; then
            fetched=1
            rc=0
            # fetch_model already hashed the new bytes, so this only re-confirms
            # that the file landed where it was meant to.
            verify_model "$name" "$size" "$sha" 0 || rc=$?
        fi
    fi

    if [ "$rc" = "0" ]; then
        if [ "$fetched" -eq 1 ]; then
            # Not "$(human_mb) + this": the combined string overflows the value
            # column, and the size was already printed by fetch_model.
            pass "$label" "downloaded, sha256 ok"
            return 0
        elif [ "$DO_CHECKSUM" -eq 1 ]; then
            how="sha256 ok"
        else
            how="size ok"
        fi
        pass "$label" "$(human_mb "$size"), ${how}"
        return 0
    fi

    case "$rc" in
        2) fail "$label" "absent" ;;
        3) fail "$label" "wrong size" ;;
        4) fail "$label" "sha256 mismatch" ;;
        *) fail "$label" "unverifiable (rc=${rc})" ;;
    esac
    if [ -s "$WHY" ]; then
        while IFS= read -r line; do detail "$line"; done < "$WHY"
    fi
    if [ "$rc" != "2" ]; then
        detail "Move or delete ${MODEL_DIR}/${name} and re-run to fetch it again."
    fi
    return 1
}

check_model "tiny.en (wake tier)"    "$TINY_NAME"  "$TINY_SIZE"  "$TINY_SHA"  || true
check_model "small.en (command tier)" "$SMALL_NAME" "$SMALL_SIZE" "$SMALL_SHA" || true

# ==========================================================================
hdr "4. speech recognition, end to end, with no microphone"

MODELS_READY=0
if [ -f "${MODEL_DIR}/${TINY_NAME}" ] && \
   [ "$(bytes_of "${MODEL_DIR}/${TINY_NAME}")" = "$TINY_SIZE" ]; then
    MODELS_READY=1
fi

if [ "$MODELS_READY" -eq 0 ]; then
    fail "whisper round trip" "no tiny.en to test with"
elif ! command -v whisper-server >/dev/null 2>&1 || ! command -v say >/dev/null 2>&1; then
    fail "whisper round trip" "whisper-server or say missing"
else
    # A cold Metal cache is ~115 s and looks exactly like a hang. Say so first.
    METAL_CACHE="$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null || echo /tmp/)com.apple.metal"
    METAL_KB=0
    if [ -d "$METAL_CACHE" ]; then
        METAL_KB="$(du -sk "$METAL_CACHE" 2>/dev/null | awk '{print $1}')"
        [ -z "$METAL_KB" ] && METAL_KB=0
    fi
    if [ "$METAL_KB" -lt 1024 ]; then
        info "THE METAL SHADER CACHE IS COLD (${METAL_KB} kB)."
        info "The first ggml run on a machine compiles shaders for about TWO"
        info "MINUTES and looks exactly like a hang. It happens once, and it is"
        info "being spent here rather than on your first spoken command."
    else
        info "Metal shader cache warm ($(python3 -c "print('%.0f MB'%(int('$METAL_KB')/1024.0))")), so this is quick."
    fi

    CLIP="${WORK}/selftest.wav"
    T0="$(python3 -c 'import time;print(time.time())')"
    if cap 60 say -o "$CLIP" --data-format=LEI16@16000 "$SELFTEST_TEXT" && [ -f "$CLIP" ]; then
        SAY_S="$(python3 -c "import time,sys;print('%.2f'%(time.time()-float(sys.argv[1])))" "$T0")"
        pass "say renders whisper's native format" "${SAY_S} s, $(bytes_of "$CLIP") B"
    else
        fail "say renders whisper's native format" "failed"
        CLIP=""
    fi

    if [ -n "$CLIP" ] && [ -f "$CLIP" ]; then
        if command -v curl >/dev/null 2>&1 && \
           curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${SELFTEST_PORT}/" 2>/dev/null; then
            warn "self-test port ${SELFTEST_PORT}" "already in use"
            detail "Something is listening there. Skipping rather than testing it"
            detail "by accident; stop it and re-run for a clean measurement."
        else
            whisper-server -m "${MODEL_DIR}/${TINY_NAME}" \
                --host 127.0.0.1 --port "$SELFTEST_PORT" -t 2 \
                > "${WORK}/server.log" 2>&1 &
            SRV_PID=$!
            T0="$(python3 -c 'import time;print(time.time())')"
            READY=0
            WAITED=0
            DIED=0
            while [ "$WAITED" -lt "$SERVER_READY_S" ]; do
                if ! kill -0 "$SRV_PID" 2>/dev/null; then
                    DIED=1                      # it exited; the log is read below
                    break
                fi
                if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${SELFTEST_PORT}/" 2>/dev/null; then
                    READY=1
                    break
                fi
                sleep 1
                WAITED=$((WAITED + 1))
            done
            READY_S="$(python3 -c "import time,sys;print('%.2f'%(time.time()-float(sys.argv[1])))" "$T0")"

            if [ "$READY" -eq 0 ]; then
                # "Timed out" and "exited immediately" are different faults and
                # reporting the second as the first sends you looking for a slow
                # machine instead of reading the reason it printed.
                if [ "$DIED" -eq 1 ]; then
                    fail "whisper-server starts" "exited after ${READY_S} s"
                else
                    fail "whisper-server starts" "no answer in ${SERVER_READY_S} s"
                fi
                detail "last lines of its log:"
                tail -5 "${WORK}/server.log" 2>/dev/null | sed 's/^/      | /' || true
                if [ "$DIED" -eq 1 ]; then
                    detail "The server said why, above. A model that is the right SIZE but"
                    detail "the wrong CONTENT reports 'invalid model data (bad magic)'"
                    detail "here and nowhere the daemon can see it -- re-run with"
                    detail "--checksum to have that diagnosed instead of guessed."
                fi
            else
                pass "whisper-server starts" "${READY_S} s on ${SELFTEST_PORT}"

                # GET / answering is not readiness: the port can bind before the
                # model has loaded. The POST is the proof.
                T0="$(python3 -c 'import time;print(time.time())')"
                BODY="${WORK}/reply.json"
                RC=0
                curl -s --max-time 120 -X POST \
                     "http://127.0.0.1:${SELFTEST_PORT}/inference" \
                     -F "file=@${CLIP}" \
                     -F "response_format=json" \
                     -F "prompt=${SELFTEST_PROMPT}" \
                     -o "$BODY" || RC=$?
                POST_S="$(python3 -c "import time,sys;print('%.2f'%(time.time()-float(sys.argv[1])))" "$T0")"

                TEXT=""
                if [ "$RC" -eq 0 ] && [ -s "$BODY" ]; then
                    TEXT="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
t = d.get("text") if isinstance(d, dict) else None
print(" ".join(str(t).split()) if t else "")
' "$BODY")"
                fi

                if [ -z "$TEXT" ]; then
                    fail "POST /inference transcribes" "no text returned"
                    detail "raw reply: $(head -c 200 "$BODY" 2>/dev/null || echo '(empty)')"
                    detail "Raw PCM is rejected with HTTP 400; this posted a WAV, so"
                    detail "a 400 here means something else. Read the log:"
                    detail "${WORK}/server.log (removed when this script exits)"
                else
                    pass "POST /inference transcribes" "${POST_S} s"
                    info "heard: \"${TEXT}\""
                    # The transcript is the only thing that proves the STT path is
                    # actually working rather than merely answering.
                    if printf '%s' "$TEXT" | grep -qi 'gradient'; then
                        pass "transcript is intelligible" "matched 'gradient'"
                    else
                        warn "transcript is intelligible" "did not match"
                        detail "Expected something close to: ${SELFTEST_TEXT}"
                    fi
                    # One variable, measured here: the prose prompt is what turns
                    # whisper's American 'color' into this tree's 'colour'.
                    if printf '%s' "$TEXT" | grep -q 'colour'; then
                        pass "prose prompt biases the vocabulary" "'colour', not 'color'"
                    else
                        warn "prose prompt biases the vocabulary" "no 'colour'"
                        detail "Not fatal: jarvis.py normalises both spellings after"
                        detail "transcription. It does mean the prompt did less than"
                        detail "it did when this was measured."
                    fi
                fi
            fi
            kill "$SRV_PID" 2>/dev/null || true
            wait "$SRV_PID" 2>/dev/null || true
            SRV_PID=""
        fi
    fi
fi

# ==========================================================================
hdr "5. microphone"

if [ "$DO_MIC" -eq 0 ]; then
    warn "microphone" "skipped (--no-mic)"
elif [ ! -f "${HERE}/miccheck.py" ]; then
    fail "microphone" "miccheck.py missing"
elif ! command -v ffmpeg >/dev/null 2>&1; then
    fail "microphone" "ffmpeg missing"
else
    info "Opening the microphone. ffmpeg's avfoundation device open costs"
    info "1.55-3.71 s per device and every device is probed, so this is the"
    info "slowest check here."
    MIC_RC=0
    # Tee'd rather than just printed, so that the start line at the bottom of
    # this script can carry a REAL device name.  jarvis.py requires --device
    # now -- there is no sensible default, since an avfoundation index is
    # assignment order and "Microsoft Teams Audio" is an enumerated input --
    # so a copy-pasteable line without one is a line that exits 2.
    MIC_OUT="$(mktemp -t jarvis-mic)"
    if [ -n "$MIC_DEVICE" ]; then
        cap "$MIC_CAP_S" python3 "${HERE}/miccheck.py" --device "$MIC_DEVICE" 2>&1 | tee "$MIC_OUT" || MIC_RC=$?
    else
        cap "$MIC_CAP_S" python3 "${HERE}/miccheck.py" --quiet 2>&1 | tee "$MIC_OUT" || MIC_RC=$?
    fi
    if [ -z "$MIC_DEVICE" ]; then
        MIC_DEVICE="$(sed -n "s/^USE: --device '\(.*\)'.*/\1/p" "$MIC_OUT" | head -1)"
    fi
    rm -f "$MIC_OUT"
    case "$MIC_RC" in
        0)   pass "a live microphone exists" "see the line above" ;;
        1)   fail "a live microphone exists" "digital zero"
             # miccheck has already printed the ordered remedy -- wrong device,
             # dead device, TCC, input volume -- next to the measurements that
             # justify that order. Do not restate it. Add only what it has no
             # reason to know: the click path and the URL that opens it.
             printf '\n'
             info "macOS 27 click path, if the remedy above points at permissions:"
             detail "  Apple menu > System Settings"
             detail "  > Privacy & Security  (left sidebar)"
             detail "  > Microphone"
             detail "  > turn ON the row for the app that launches this daemon"
             detail "    (Terminal, iTerm, or Claude -- whichever owns the"
             detail "     process tree; it is NOT python3 and NOT ffmpeg)"
             detail "  > quit and relaunch that app: the grant only takes effect"
             detail "    on a restart of the granted process."
             printf '\n'
             info "That pane opens directly with:"
             detail "  open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone'"
             printf '\n'
             info "Read the remedy ABOVE first. On this machine the built-in mic"
             info "read digital zero while the iPhone mic worked in the same second,"
             info "which is not a permission problem and System Settings cannot fix it."
             ;;
        2)   fail "a live microphone exists" "no device / bad --device" ;;
        124) fail "a live microphone exists" "probe timed out (${MIC_CAP_S} s)"
             detail "ffmpeg never returned. A device that has gone away fails fast"
             detail "(rc=251, 'Invalid audio device index'), so a hang here is"
             detail "something else -- try --device with one name." ;;
        *)   fail "a live microphone exists" "miccheck rc=${MIC_RC}" ;;
    esac
fi

# ==========================================================================
hdr "6. Claude Code (the execution engine)"

if command -v node >/dev/null 2>&1; then
    NODE_V="$(node -v 2>/dev/null)"
    NODE_OK="$(node -e 'process.stdout.write(process.versions.node.split(".")[0] >= 22 ? "1" : "0")' 2>/dev/null || echo 0)"
    if [ "$NODE_OK" = "1" ]; then
        pass "node >= 22" "$NODE_V"
    else
        fail "node >= 22" "$NODE_V"
        detail "${NPM_PKG} declares engines >= 22.0.0."
    fi
else
    fail "node" "not on PATH"
fi

NPM_PREFIX=""
if command -v npm >/dev/null 2>&1; then
    NPM_PREFIX="$(npm prefix -g 2>/dev/null || true)"
    if [ -n "$NPM_PREFIX" ] && [ -w "${NPM_PREFIX}/lib/node_modules" ]; then
        pass "npm global prefix writable" "$NPM_PREFIX"
    elif [ -n "$NPM_PREFIX" ]; then
        warn "npm global prefix writable" "needs sudo"
        detail "${NPM_PREFIX}/lib/node_modules is not writable by you, so a"
        detail "global install would need sudo. This script will not run sudo."
    else
        warn "npm global prefix" "unknown"
    fi
else
    fail "npm" "not on PATH"
fi

CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
if [ -z "$CLAUDE_BIN" ]; then
    # Offer the install BEFORE recording a verdict, for the same reason as the
    # models: a FAIL recorded here would survive a successful install and the run
    # would end "SETUP INCOMPLETE" having just done the setup.
    info "claude is not on PATH."
    # The desktop app ships a copy. It is worth naming, because it proves node and
    # npm are not the blocker -- but it is NOT a substitute: it has no login of its
    # own (see below), and pointing the daemon at a path inside a versioned app
    # bundle breaks on the next app update.
    BUNDLED="$(ls -d "${HOME}/Library/Application Support/Claude/claude-code/"*/claude.app/Contents/MacOS/claude 2>/dev/null | tail -1 || true)"
    if [ -n "$BUNDLED" ]; then
        detail "The desktop app ships one at:"
        detail "  ${BUNDLED}"
        detail "Do not point the daemon at it: it lives under a version number"
        detail "that changes, and it has no login of its own either."
    fi
    printf '\n'
    info "The daemon hands every spoken command to Claude Code. Without it,"
    info "jarvis.py runs its speech pipeline and then has nothing to dispatch to."
    detail "install command:  npm i -g ${NPM_PKG}"
    if [ -n "$NPM_PREFIX" ]; then
        detail "destination:      ${NPM_PREFIX}/lib/node_modules/${NPM_PKG}"
    fi
    detail "this is a GLOBAL install shared by every node project on this machine."
    detail "remove it later with:  npm rm -g ${NPM_PKG}"
    if ask_npm_global "install it now?"; then
        if npm i -g "$NPM_PKG"; then
            CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
            if [ -z "$CLAUDE_BIN" ]; then
                detail "npm reported success but claude is still not on PATH."
                detail "Check that ${NPM_PREFIX}/bin is in your PATH."
            fi
        else
            detail "npm install failed."
        fi
    fi
fi

if [ -n "$CLAUDE_BIN" ]; then
    CLAUDE_V="$(cap 30 "$CLAUDE_BIN" --version 2>/dev/null | head -1 || true)"
    pass "claude on PATH" "${CLAUDE_V:-installed}"
else
    fail "claude on PATH" "not installed"
fi

# Environment hygiene. Checked whether or not claude is installed, because it is
# about this machine rather than about the CLI.
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    fail "ANTHROPIC_API_KEY unset" "IT IS SET"
    detail "In non-interactive (-p) mode this key is ALWAYS used when present, with"
    detail "no approval step, so it silently replaces the Pro/Max subscription for"
    detail "every spoken command -- and it fails outright if the key's org is"
    detail "disabled. Unset it in the shell that starts the daemon. (jarvis.py"
    detail "strips it from its child environment, so this is about you, not it.)"
else
    pass "ANTHROPIC_API_KEY unset" "not in this env"
fi

PROFILE_HITS=""
for p in "${HOME}/.zshrc" "${HOME}/.zprofile" "${HOME}/.zshenv" \
         "${HOME}/.bash_profile" "${HOME}/.profile"; do
    if [ -f "$p" ] && grep -qE '^[^#]*ANTHROPIC_API_KEY' "$p" 2>/dev/null; then
        PROFILE_HITS="${PROFILE_HITS} $(basename "$p")"
    fi
done
if [ -n "$PROFILE_HITS" ]; then
    fail "no ANTHROPIC_API_KEY in profiles" "found in:${PROFILE_HITS}"
    detail "It is not set right now, but a new terminal would set it -- and the"
    detail "daemon is started from a new terminal."
else
    pass "no ANTHROPIC_API_KEY in profiles" "5 profiles clean"
fi

if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
    if [ "${ANTHROPIC_BASE_URL}" = "https://api.anthropic.com" ]; then
        warn "ANTHROPIC_BASE_URL" "set to the default"
        detail "Harmless: it is an endpoint override with no role in credential"
        detail "selection, and api.anthropic.com is the first-party host. It is"
        detail "also pointless, and it is stripped from the child environment"
        detail "here and by jarvis.py. Nothing to do."
    else
        fail "ANTHROPIC_BASE_URL" "non-default host"
        detail "Set to: ${ANTHROPIC_BASE_URL}"
        detail "Pointed at a non-first-party host this disables MCP tool search and"
        detail "Remote Control, and routes every spoken command through it."
    fi
else
    pass "ANTHROPIC_BASE_URL" "unset"
fi

STRIPPED_COUNT="$(claude_env_unsets | grep -c . || true)"
if [ "${STRIPPED_COUNT:-0}" -gt 0 ]; then
    info "${STRIPPED_COUNT} CLAUDE_*/ANTHROPIC_* variables are set in this shell and"
    info "are stripped from the claude checks below. A child that inherits them"
    info "behaves differently from one a plain terminal starts, so a check run"
    info "under them would prove nothing about the daemon."
fi

# --- login -----------------------------------------------------------------
LOGGED_IN=0
if [ -n "$CLAUDE_BIN" ]; then
    AUTH_JSON="${WORK}/auth.json"
    AUTH_RC=0
    run_claude 45 "$CLAUDE_BIN" auth status > "$AUTH_JSON" 2>/dev/null || AUTH_RC=$?
    AUTH_STATE="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("unreadable"); raise SystemExit
print(("yes:" if d.get("loggedIn") else "no:") + str(d.get("authMethod")))
' "$AUTH_JSON" 2>/dev/null || echo unreadable)"

    case "$AUTH_STATE" in
        yes:*)
            LOGGED_IN=1
            pass "claude is logged in" "auth ${AUTH_STATE#yes:}"
            ;;
        *)
            fail "claude is logged in" "loggedIn=false"
            detail "auth status rc=${AUTH_RC}, reported: ${AUTH_STATE#no:}"
            printf '\n'
            info "THE DESKTOP APP'S LOGIN IS NOT INHERITED. The keychain item"
            info "\"Claude Code-credentials\" exists and reads back fine, and the CLI"
            info "still reports loggedIn=false: the desktop app authenticates its"
            info "child over a host socket, not through anything a standalone CLI"
            info "can read. There is no way around this but to log in once."
            printf '\n'
            detail "Run this yourself, in a terminal (it opens a browser):"
            detail "  claude auth login"
            detail "Then re-run this script."
            printf '\n'
            detail "If you ever run the daemon somewhere with no browser:"
            detail "  claude setup-token          # prints a 1-year token, saved nowhere"
            detail "  export CLAUDE_CODE_OAUTH_TOKEN=<token>"
            ;;
    esac
fi

# --- does it actually answer ------------------------------------------------
if [ "$DO_CLAUDE_PROBE" -eq 0 ]; then
    warn "claude -p answers" "skipped"
elif [ -z "$CLAUDE_BIN" ]; then
    fail "claude -p answers" "no claude to ask"
elif [ "$LOGGED_IN" -eq 0 ]; then
    fail "claude -p answers" "not logged in"
    detail "Skipped the round trip: it would only repeat the login failure."
else
    info "Asking Claude Code a trivial question. Expect ~3.5 s of fixed overhead"
    info "before the model is even reached."
    OUT="${WORK}/probe.json"
    PROBE_RC=0
    T0="$(python3 -c 'import time;print(time.time())')"
    run_claude "$CLAUDE_PROBE_CAP_S" "$CLAUDE_BIN" \
        -p "Reply with exactly the word PONG and nothing else." \
        --output-format json \
        --permission-mode dontAsk \
        --allowedTools "" \
        --permission-prompts none \
        --max-turns 1 \
        > "$OUT" 2>/dev/null || PROBE_RC=$?
    PROBE_S="$(python3 -c "import time,sys;print('%.1f'%(time.time()-float(sys.argv[1])))" "$T0")"

    # is_error, never subtype and never the exit code: a measured auth-failure
    # envelope carried subtype="success" WITH is_error=true and the failure
    # sentence in `result` on stdout.
    VERDICT="$(python3 -c '
import json, sys
try:
    raw = open(sys.argv[1]).read()
except Exception:
    print("noout\t\t"); raise SystemExit
raw = raw.strip()
d = None
if raw:
    try:
        d = json.loads(raw)
    except Exception:
        i = raw.find("{")
        while i != -1 and d is None:
            try:
                d = json.loads(raw[i:])
            except Exception:
                i = raw.find("{", i + 1)
if not isinstance(d, dict):
    print("unparsed\t\t" + raw[:160].replace("\t", " ").replace("\n", " "))
    raise SystemExit
res = d.get("result")
res = res if isinstance(res, str) else json.dumps(res)
res = " ".join(res.split())[:160]
state = "error" if d.get("is_error") else "ok"
cost = d.get("total_cost_usd") or 0.0
try:
    cost = "%.4f" % float(cost)
except Exception:
    cost = "?"
print("%s\t%s\t%s" % (state, cost, res))
' "$OUT" 2>/dev/null || printf 'unparsed\t\t')"

    P_STATE="$(printf '%s' "$VERDICT" | cut -f1)"
    P_COST="$(printf '%s' "$VERDICT" | cut -f2)"
    P_RESULT="$(printf '%s' "$VERDICT" | cut -f3)"

    case "$P_STATE" in
        ok)
            pass "claude -p answers" "${PROBE_S} s, \$${P_COST}"
            info "replied: \"${P_RESULT}\""
            ;;
        error)
            fail "claude -p answers" "is_error=true"
            detail "result: ${P_RESULT}"
            detail "Note that this arrived on STDOUT with exit code ${PROBE_RC}, and the"
            detail "envelope may still say subtype=success. is_error is the field"
            detail "that means anything; jarvis.py branches on it too."
            ;;
        noout|unparsed)
            if [ "$PROBE_RC" = "124" ]; then
                fail "claude -p answers" "timed out (${CLAUDE_PROBE_CAP_S} s)"
                detail "Killed by process group. There is no wall-clock cap in the CLI"
                detail "itself -- --max-turns and --max-budget-usd bound turns and"
                detail "spend, not time -- so this cap is the only bound there is."
            else
                fail "claude -p answers" "unreadable reply (rc=${PROBE_RC})"
                [ -n "$P_RESULT" ] && detail "stdout began: ${P_RESULT}"
            fi
            ;;
    esac
fi

# ==========================================================================
# Summary, in the shape the validation cases print.
N_PASS="$(awk -F'\t' '$1=="PASS"' "$RESULTS" | wc -l | tr -d ' ')"
N_WARN="$(awk -F'\t' '$1=="WARN"' "$RESULTS" | wc -l | tr -d ' ')"
N_FAIL="$(awk -F'\t' '$1=="FAIL"' "$RESULTS" | wc -l | tr -d ' ')"
N_ALL="$(wc -l < "$RESULTS" | tr -d ' ')"

printf '\n\nacceptance:\n'
while IFS="$(printf '\t')" read -r verdict name value; do
    [ -z "$verdict" ] && continue
    printf '  %-46s %-24s %s\n' "$name" "$value" "$verdict"
done < "$RESULTS"

printf '\n  %d checks: %d PASS, %d WARN, %d FAIL\n' "$N_ALL" "$N_PASS" "$N_WARN" "$N_FAIL"

if [ "$N_FAIL" -gt 0 ]; then
    printf '\n  SETUP INCOMPLETE. Fix the FAIL lines above and run this again.\n'
    if [ "$DO_INSTALL" -eq 0 ]; then
        printf '  (This was --check, so nothing was installed. Re-run without it.)\n'
    fi
    printf '\n'
    exit 1
fi

printf '\n  Everything the daemon needs is present and was exercised, not assumed.\n'
printf '\n  Start it with:\n'
printf '      cd %s\n' "$REPO"
if [ -n "$MIC_DEVICE" ]; then
    printf "      python3 voice/jarvis.py --device '%s'\\n" "$MIC_DEVICE"
else
    printf '      python3 voice/miccheck.py --quiet        # pick a LIVE device\n'
    printf "      python3 voice/jarvis.py --device '<that name>'\\n"
fi
printf '\n'
printf '  --device is REQUIRED. An avfoundation index is assignment order, not\n'
printf '  identity; it meant three different microphones in seventy minutes on\n'
printf '  this machine, and "Microsoft Teams Audio" is an enumerated INPUT, so a\n'
printf '  default of index 0 can put a meeting on the wake word.\n'
printf '\n'
printf '  Then say:  "jarvis, run the fast tests"\n'
printf '\n'
printf '  Three things this cannot check for you:\n'
printf '    - the daemon is DEAF while it speaks and while claude works, so "stop"\n'
printf '      is only heard between utterances; Ctrl-C is the real stop.\n'
printf '    - it reads the room, not you. Anyone audible can drive it, and\n'
printf '      voice/logs/*.jsonl keeps a record of what it heard.\n'
printf '    - a destructive command is released ONLY by "jarvis, confirm" -- the\n'
printf '      whole utterance, wake word included, and nothing else.\n'
printf '\n'
exit 0
