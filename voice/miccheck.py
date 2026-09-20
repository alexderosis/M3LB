#!/usr/bin/env python3
# miccheck.py -- prove the microphone is LIVE before the daemon enters its loop.
#
# WHY THIS EXISTS, AND WHY IT IS NOT AN AMPLITUDE TEST.
# On this machine (macOS 27.0 / 26A428, M1, 2026-09-20) the built-in "MacBook Air
# Microphone" delivers a stream of EXACT ZEROES: 48000 consecutive s16 samples,
# peak 0, three independent 1.5 s captures and one continuous 3.0 s pipe read, all
# 0/0.  The Continuity "iPhone ... Microphone" on the SAME ffmpeg invocation, from
# the SAME parent process, in the SAME second, gives peak 811 (0.0248 FS) and a
# noise floor of 7-15 counts.  So the dead device is NOT a TCC denial -- a denial
# would kill both -- and any check that only asks "is it loud enough" cannot tell
# a blocked mic from a quiet room.
#
# The discriminator that does work is the fraction of EXACTLY-ZERO samples.  A live
# converter dithers: the quiet iPhone mic produced zero_frac 0.02-0.06 while sitting
# in a silent room.  A dead/muted/TCC-denied path produces zero_frac 1.000000.  This
# file therefore classifies on zero_frac first and amplitude second, and reports
# DEAD / SILENT / LIVE as three different outcomes rather than two.
#
# COST.  The arithmetic is free: 2.79 ms of pure Python for peak + zero-fraction
# over 1.0 s of 16 kHz audio (6.16 ms if r.m.s. is included; 7.49 ms for the
# explicit per-sample loop the first draft used).  What is NOT free is opening the
# device -- one named-device probe costs 7.1-9.7 s wall against 1.1 s of CPU,
# because it spends two ffmpeg AVFoundation opens (one to enumerate, one to
# capture) at 2.5-3.4 s each.  Probing all three devices costs 17.2 s.
# SO THE DAEMON SHOULD NOT RUN THIS AT STARTUP.  It should run the SAME analysis
# on the first 1.3 s of its own long-running capture stream, which costs one
# extra 2.79 ms call and no extra device open.  This program is the setup and
# diagnosis tool: run it once, by hand, to find out which device to name.
#
# audioop and aifc were removed in Python 3.13 (PEP 594) and this machine runs
# 3.14.7, so there is no audioop.rms() and no audioop.max().  Everything here is
# array('h') over the raw s16le bytes ffmpeg writes to the pipe; no numpy, no
# sounddevice, no venv -- the same stdlib-only rule tools/osm_city.py follows.
#
# WARM-UP.  ffmpeg's avfoundation open is slow and variable: measured time-to-first-
# byte 2.43 s (built-in) and 3.71 s (iPhone) at load average 26.  Once open, the pipe
# runs continuously and does not stall.  The first 100 ms window also carries a
# device-open transient (peak 1200 against a 7-15 floor two windows later), so
# SKIP_S of audio is discarded AFTER the first byte, never before.
#
# WHAT IT DOES NOT DO.  It does not ask TCC anything -- there is no read API;
# `tccutil` only implements `reset`, and the unified log shows the request but marks
# the verdict <private>.  It does not distinguish a TCC denial from a muted device
# from broken hardware: all three read as DEAD, and the remedy text says so.  It does
# not detect speech; it detects that a converter is running.

import argparse
import array
import re
import selectors
import signal
import subprocess
import sys
import time

FFMPEG = "ffmpeg"
RATE = 16000          # what whisper.cpp wants; also what we measure at
WIDTH = 2             # s16le
SKIP_S = 0.30         # discarded after the first byte: device-open transient
TEST_S = 1.00         # analysed window
# A live-but-quiet converter still dithers.  Measured zero_frac: 1.000000 on the
# dead built-in, 0.02-0.06 on the quiet iPhone mic.  0.98 leaves an order of
# magnitude of margin on both sides.
DEAD_ZERO_FRAC = 0.98
# Peak below this is a live mic in a silent room, not a fault.  0.0015 FS = 49
# counts, about 4x the measured quiet-room floor of 7-15 counts.
QUIET_PEAK_FS = 0.0015


def list_audio_devices():
    """Parse ffmpeg's avfoundation device list.  Returns [(index, name), ...].

    THE INDEX IS NOT STABLE.  Between two measurements 40 minutes apart on this
    machine the list went  [0] MacBook Air / [1] AirPods Pro / [2] iPhone / [3] Teams
    to  [0] iPhone / [1] MacBook Air / [2] Teams -- unplugging the AirPods renumbered
    everything, so a saved `-i ":0"` silently changed which microphone it meant.
    Always resolve a NAME to an index at startup; never persist an index.

    THE timeout= IS NOT DECORATION.  jarvis.start_audio() calls this on its
    startup path, before it has printed anything, so an ffmpeg that opens the
    AVFoundation layer and never returns hangs the daemon in a place where
    there is nothing on screen to explain it.  AVFoundation enumeration was
    measured at 2.5-3.4 s on this machine at load average 20-27, so 30 s is
    about ten times the worst observed case.
    """
    try:
        p = subprocess.run(
            [FFMPEG, "-nostdin", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=30.0,
        )
    except subprocess.TimeoutExpired:
        print("ffmpeg did not return a device list within 30 s", file=sys.stderr)
        return []
    except OSError as exc:
        print(f"could not run ffmpeg: {exc}", file=sys.stderr)
        return []
    # ffmpeg exits non-zero here by design ("Error opening input") -- the list is on
    # stderr and the exit status means nothing.
    out, devs, in_audio = p.stderr, [], False
    for line in out.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if not in_audio:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if m:
            devs.append((int(m.group(1)), m.group(2)))
    return devs


def resolve(name_or_index, devs):
    """Accept an index, a full name, or a case-insensitive substring."""
    s = str(name_or_index)
    if s.isdigit():
        return int(s)
    low = s.lower()
    for i, n in devs:
        if n.lower() == low:
            return i
    for i, n in devs:
        if low in n.lower():
            return i
    return None


def _drain(p, grace):
    """Collect a killed child's stderr WITHOUT the possibility of blocking.

    A bare `p.stderr.read()` after a kill returns only when every writer has
    closed the pipe, which is not the same thing as "the direct child is gone":
    a grandchild inherits the descriptor and holds it open.  That was found
    here rather than reasoned about -- a test stub that spawned `sleep` hung
    this function for the whole of the sleep, in the FINALLY clause of the code
    whose timeout had just worked correctly.  communicate() carries a timeout;
    read() does not.
    """
    try:
        _, errb = p.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            _, errb = p.communicate(timeout=2)
        except Exception:
            return ""
    except Exception:
        return ""
    return (errb or b"").decode("utf-8", "replace").strip()


def capture(index, skip_s=SKIP_S, test_s=TEST_S, timeout_s=20.0):
    """Read (skip_s + test_s) of s16le mono from one device, return the tail.

    Reads by BYTE COUNT and then kills ffmpeg, rather than using -t.  With -t the
    device-open latency is charged against the requested duration: `-t 1.5` returned
    only 19440 of 24000 samples (81%), and `-t 2` returned 25754 of 32000 (80%).
    Counting bytes gets exactly the samples asked for.

    -nostdin is kept as insurance, not because a failure was reproduced: a
    background ffmpeg without it did NOT steal a line from the parent's stdin
    here (both spellings left "LINE_ONE" for the parent to read) and did NOT
    take SIGTTIN (STAT R, rc=0).  ffmpeg's interactive key reader only arms
    itself when stdin is a tty, which a daemon's is not.  Costs nothing; keep it.

    ALL TIMINGS IN THIS FILE WERE MEASURED AT LOAD AVERAGE 20-27 on an 8 GB M1.
    They are upper bounds, not the idle cost.

    THE DEADLINE MUST NOT DEPEND ON DATA ARRIVING, AND IT USED TO.  The
    `if time.perf_counter() - t0 > timeout_s: break` sat INSIDE the body that
    runs after a chunk has been read, so an ffmpeg that opened the device and
    then emitted nothing blocked forever in p.stdout.read(3200) and the declared
    20 s cap never fired.  Reproduced by substituting a stub that produces no
    stdout and never exits: capture(0, timeout_s=2.0) was still running on its
    thread after 8 s.  selectors waits on the PIPE with the REMAINING time, so
    the cap is real whether or not anything is ever written.
    """
    need = int(RATE * (skip_s + test_s)) * WIDTH
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-f", "avfoundation", "-i", f":{index}",
           "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         bufsize=0)
    buf, t_first = bytearray(), None
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    try:
        while len(buf) < need:
            remaining = timeout_s - (time.perf_counter() - t0)
            if remaining <= 0:
                break
            if not sel.select(min(remaining, 0.5)):
                continue                         # nothing yet; re-check the clock
            chunk = p.stdout.read(3200)          # 100 ms, and it is ready
            if not chunk:
                break
            if t_first is None:
                t_first = time.perf_counter() - t0
            buf += chunk
    finally:
        sel.close()
        p.send_signal(signal.SIGTERM)
        err = _drain(p, 3.0)
    skip_b = int(RATE * skip_s) * WIDTH
    tail = bytes(buf[skip_b:])
    return tail, t_first, time.perf_counter() - t0, err


def analyse(raw):
    """peak (0..32767), rms, fraction of exactly-zero samples.  No audioop."""
    a = array.array("h")
    a.frombytes(raw[: len(raw) // 2 * 2])
    n = len(a)
    if n == 0:
        return {"n": 0, "peak": 0, "rms": 0.0, "zero_frac": 1.0}
    # max/min/count run in C over the array; the explicit per-sample loop this
    # replaced cost 7.49 ms against 6.16 ms for 1.0 s of 16 kHz audio.  Note
    # -min(a) can overflow to 32768 for a full-scale negative sample; clamp it.
    peak = min(max(max(a), -min(a)), 32767)
    zeros = a.count(0)
    acc = sum(map(lambda v: v * v, a))
    return {"n": n, "peak": peak, "rms": (acc / n) ** 0.5, "zero_frac": zeros / n}


def classify(st):
    if st["n"] == 0:
        return "DEAD", "no samples arrived at all (ffmpeg could not open the device)"
    if st["zero_frac"] >= DEAD_ZERO_FRAC:
        return "DEAD", (f"{st['zero_frac']*100:.4f}% of samples are exactly zero -- "
                        "this is a digital-zero stream, not a quiet room")
    if st["peak"] / 32768.0 < QUIET_PEAK_FS:
        return "SILENT", "converter is running but nothing is above the noise floor"
    return "LIVE", "signal present"


REMEDY = """\
The stream is digital zero.  In order of how often it is the cause here:

  1. WRONG DEVICE.  avfoundation indices renumber whenever a device appears or
     disappears.  Run with --list and pick by NAME.
  2. THAT PARTICULAR DEVICE IS DEAD.  On this machine the built-in "MacBook Air
     Microphone" returns exact zeroes while the Continuity iPhone microphone,
     queried one second later by the same process, returns real audio.  Try
     another device before blaming permissions.
  3. TCC.  macOS attaches a microphone grant to the APP that owns the process
     tree, not to python3 or ffmpeg.  Grant it to whichever of these launches
     the daemon:
         System Settings > Privacy & Security > Microphone
     and turn on the row for Terminal (com.apple.Terminal), or Claude
     (com.anthropic.claudefordesktop), or -- if you run it from a LaunchAgent,
     which has no app identity -- the row that appears under the script's own
     name the first time it asks.  A change there RESTARTS the granted process:
     quit and relaunch the terminal afterwards.
     There is no CLI that reads this back.  `tccutil` implements only `reset`,
     and `log show --predicate 'subsystem == "com.apple.TCC"'` shows the request
     but prints the verdict as <private>.  This program IS the query.
  4. Input volume.  `osascript -e 'input volume of (get volume settings)'`;
     0 there produces zeroes with the permission fully granted.
"""


def main():
    ap = argparse.ArgumentParser(description="prove the microphone is live")
    ap.add_argument("--device", default=None,
                    help="index, exact name, or substring (default: probe all)")
    ap.add_argument("--list", action="store_true", help="list devices and exit")
    ap.add_argument("--seconds", type=float, default=TEST_S)
    ap.add_argument("--quiet", action="store_true", help="one line per device")
    args = ap.parse_args()

    t_start = time.perf_counter()
    devs = list_audio_devices()
    if not devs:
        print("no avfoundation audio devices at all -- is ffmpeg installed?",
              file=sys.stderr)
        return 2
    if args.list:
        for i, n in devs:
            print(f"  [{i}] {n}")
        return 0

    if args.device is not None:
        idx = resolve(args.device, devs)
        if idx is None:
            print(f"no device matching {args.device!r}; have:", file=sys.stderr)
            for i, n in devs:
                print(f"  [{i}] {n}", file=sys.stderr)
            return 2
        targets = [(i, n) for i, n in devs if i == idx]
    else:
        targets = devs

    results = []
    for i, name in targets:
        raw, t_first, t_tot, err = capture(i, test_s=args.seconds)
        st = analyse(raw)
        verdict, why = classify(st)
        results.append((i, name, verdict, st))
        if args.quiet:
            print(f"[{i}] {name:38s} {verdict:6s} "
                  f"peak={st['peak']:5d} zero={st['zero_frac']*100:7.3f}%")
        else:
            print(f"[{i}] {name}")
            print(f"     verdict      : {verdict}  ({why})")
            print(f"     samples      : {st['n']} ({st['n']/RATE:.3f} s "
                  f"after a {SKIP_S:.2f} s warm-up discard)")
            print(f"     peak         : {st['peak']} "
                  f"({st['peak']/32768.0:.6f} FS)")
            print(f"     rms          : {st['rms']:.1f} "
                  f"({st['rms']/32768.0:.6f} FS)")
            print(f"     exactly zero : {st['zero_frac']*100:.4f}% of samples")
            print(f"     ffmpeg       : first byte {('%.3f s' % t_first) if t_first else 'NEVER'}"
                  f", total {t_tot:.3f} s")
            if err:
                print(f"     stderr       : {err.splitlines()[0][:100]}")
            print()

    live = [r for r in results if r[2] == "LIVE"]
    quiet = [r for r in results if r[2] == "SILENT"]
    print(f"-- probed {len(results)} device(s) in {time.perf_counter()-t_start:.3f} s")
    if live:
        i, name, _, st = live[0]
        print(f"USE: --device {name!r}  (index {i} TODAY; resolve by name, not index)")
        return 0
    if quiet:
        i, name, _, st = quiet[0]
        print(f"USE: --device {name!r} -- converter is live but the room is silent.")
        return 0
    print("NO LIVE MICROPHONE.\n")
    print(REMEDY)
    return 1


if __name__ == "__main__":
    sys.exit(main())
