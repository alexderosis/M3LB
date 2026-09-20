#!/usr/bin/env python3
# captureprobe.py -- measure whether the microphone stream is actually arriving in
# real time.  Run it before believing a transcription, and after any change to what
# else the machine is doing.
#
# WHY.  On 2026-09-20, while ./build/demonstrator/urban held 82.3 % CPU and 909 MB
# of an 8 GB M1 (load average 19-27), a long-running
#     ffmpeg -f avfoundation -i ":1" -ac 1 -ar 16000 -f s16le -
# delivered 0.75x REAL TIME, sustained.  Measured from the first byte, so no
# device-open latency is folded in:
#     t =  5 s  ->  3.562 s delivered, deficit -1.441 s, 0.712x
#     t = 10 s  ->  7.498 s delivered, deficit -2.510 s, 0.749x
#     t = 15 s  -> 11.274 s delivered, deficit -3.732 s, 0.751x
#     t = 20 s  -> 15.082 s delivered, deficit -4.927 s, 0.754x
#     t = 25 s  -> 18.890 s delivered, deficit -6.121 s, 0.755x
# The deficit GROWS LINEARLY at about 0.24 s per second.  That is the distinction
# that matters: a fixed start-up offset would have held at -1.4 s and been
# harmless, and the first two points alone could not tell the two apart.  A quarter
# of every second of speech is simply missing.
#
# IT IS NOT THE PIPE AND IT IS NOT THE READER.  Writing straight to a .wav with no
# Python in the loop gives the same answer: 22.24 s of wall produced 10.955 s of
# audio.  It is not the Continuity link (built-in 0.686, iPhone 0.651 in the same
# minute), not the resampler (48000 native 0.651, 16000 resampled 0.686), and not
# scheduling priority (plain 0.733, `taskpolicy -t 0 -l 0` 0.762, `nice` 0.757 --
# all within the run-to-run scatter, which is itself large: two back-to-back 10 s
# runs gave 0.467 and 0.693).
#
# AND FFMPEG SAYS NOTHING.  stderr is empty.  There is no "drop" warning, no
# non-zero exit, no gap in the byte stream to notice.  The samples that arrive are
# valid audio; they are just fewer than a second's worth per second, so speech
# comes out time-compressed and gapped.  Whisper will transcribe that into
# confident nonsense.  THIS IS THE FAILURE THIS FILE EXISTS TO CATCH.
#
# WHAT IS NOT ESTABLISHED.  Every number above was taken on a machine running an
# LBM demonstrator at 82 % CPU.  Nobody has measured the idle ratio.  Do not quote
# 0.75x as a property of this Mac until an idle run has been done -- but do not
# assume it goes away either, because the case the assistant is FOR is being told
# that a simulation has finished, i.e. while a simulation is running.
#
# stdlib only, Python 3.14: audioop and aifc were removed in 3.13 (PEP 594).

import argparse
import selectors
import signal
import subprocess
import sys
import time

RATE, WIDTH = 16000, 2
# Below this the audio is gapped enough that a transcript is not trustworthy.
# 0.95 leaves room for scheduler jitter and is far above the 0.75 measured here.
GOOD = 0.95


def main():
    ap = argparse.ArgumentParser(description="is the mic stream arriving in real time?")
    ap.add_argument("--device", default=":1", help='avfoundation spec, e.g. ":1"')
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="hard wall-clock cap (default: --seconds plus 30 s of "
                         "slack for the device open)")
    args = ap.parse_args()
    # --seconds IS A LIST OF REPORT MARKS, NOT A DEADLINE, and it never was
    # one: the loop ran until the last mark was printed, so an ffmpeg that
    # opens the device and then emits nothing blocked forever in
    # p.stdout.read(65536) with no cap of any kind.  The device open itself
    # costs 1.55-3.71 s here (eight measurements), so the slack is generous.
    deadline_s = args.timeout if args.timeout > 0 else args.seconds + 30.0

    dev = args.device if args.device.startswith(":") else ":" + args.device
    p = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "avfoundation", "-i", dev,
         "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, bufsize=0)

    t0 = time.perf_counter()
    tot, first = 0, None
    marks = [m for m in (5, 10, 15, 20, 25, 30, 45, 60) if m <= args.seconds]
    k, last = 0, (0.0, 0.0)
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    timed_out = False
    try:
        while k < len(marks):
            remaining = deadline_s - (time.perf_counter() - t0)
            if remaining <= 0:
                timed_out = True
                break
            if not sel.select(min(remaining, 0.5)):
                continue
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            now = time.perf_counter() - t0
            if first is None:
                first = now
                print(f"device {dev}: time to first byte {first:.3f} s")
            tot += len(chunk)
            el = now - first
            while k < len(marks) and el >= marks[k]:
                aud = tot / WIDTH / RATE
                print(f"  t={marks[k]:3d} s after first byte : delivered {aud:7.3f} s"
                      f"   deficit {aud-el:+7.3f} s   rate {aud/el:.3f}x")
                last = (aud, el)
                k += 1
    finally:
        sel.close()
        p.send_signal(signal.SIGTERM)
        # communicate(), not stderr.read(): read() returns only when EVERY
        # writer has closed the pipe, and a grandchild that inherited the
        # descriptor is a writer even after the direct child is killed.  See
        # miccheck._drain, where this was measured.
        err = ""
        try:
            _, errb = p.communicate(timeout=3)
            err = (errb or b"").decode("utf-8", "replace").strip()
        except subprocess.TimeoutExpired:
            p.kill()
        except Exception:
            pass

    if timed_out:
        print(f"\nGAVE UP after {deadline_s:.0f} s without reaching the last "
              f"report mark.")
    if first is None:
        print("NO BYTES AT ALL -- wrong device index, or ffmpeg could not open it.")
        if err:
            print("ffmpeg:", err.splitlines()[0])
        return 2

    aud, el = last
    if el <= 0:
        print("ran too briefly to judge")
        return 2
    ratio = aud / el
    print(f"\nsustained rate {ratio:.3f}x real time"
          f"   (ffmpeg stderr: {err.splitlines()[0][:80] if err else 'silent'})")
    if ratio >= GOOD:
        print("OK -- the stream keeps up; a transcript from it is worth trusting.")
        return 0
    print(f"BAD -- {100*(1-ratio):.0f}% of the audio is missing and nothing reports it.")
    print("Speech will be time-compressed and gapped, and whisper will invent text")
    print("to cover the gaps.  Free the machine (check `ps -Ao %cpu,rss,comm -r |")
    print("head`) and re-run before trusting any transcription.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
