# voice/ — a wake-word front end for this repository

Say "jarvis, run the fast tests" and it runs. `jarvis.py` listens on one
long-lived microphone stream, detects the wake word with a local `tiny.en`
whisper model, re-transcribes the command with `small.en`, rewrites spoken forms
into the ones this tree actually accepts ("dee three queue twenty seven" ->
`D3Q27`), and hands the sentence to `claude -p` running headless in this repo —
which reads `CLAUDE.md` and decides what to do. Speech recognition is entirely
local and free; the dispatch is a network call.

Everything is stdlib-only Python plus binaries that are already installed
(`ffmpeg`, `whisper-server`, `say`, `afplay`, `claude`). No numpy, no
sounddevice, no pip, no venv — the same rule `tools/osm_city.py` follows, and for
the same reason: this must run on a fresh machine.

    voice/jarvis.py         1936 lines   the daemon
    voice/miccheck.py        290 lines   is the microphone live? (run this first)
    voice/captureprobe.py    122 lines   is the stream arriving in real time?
    voice/logs/*.jsonl                   one line per event, per day

---

## What it does NOT do

Stated up front, per this tree's convention, because several of these will
otherwise read as bugs.

* **No barge-in.** While `claude` is working, and while the daemon is speaking,
  it is deaf. The self-trigger guard is time-based: the VAD is closed while the
  speaking process is alive plus 450 ms of speaker decay. The failure mode is
  exactly the one that matters — it also deafens the assistant to a "stop"
  spoken over its own reply, which is the one word you most want to interrupt
  with. Cancelling a running build means Ctrl-C in the terminal.
* **No deterministic shell routing.** A spoken "build" is *not* pattern-matched
  to a `cmake` argv. It is handed to the agent. This was left out on purpose:
  two places that know what "build" means drift apart, and the agent is the one
  that can read the banner it is about to contradict. It costs latency and
  tokens, and it buys correctness.
* **No speaker identification.** Anyone audible can drive it, including a
  podcast.
* **No offline LLM.** Only the speech recognition is local. Every dispatch is a
  network call through the `claude` CLI.
* **The agent cannot use the Edit or Write tools by default — but it can run
  `cmake` and `ctest`, and those can delete and execute.** `--permission-mode
  dontAsk` plus an explicit `--allowedTools` list covering reading,
  `cmake --build`, `cmake -S`, the three `ctest` forms the registry uses,
  `./build/*` and git's *read* verbs, **plus a `--disallowedTools` backstop**
  that keeps `cmake -E` (recursive delete, and `cmake -E env -- <any binary>`),
  `cmake -P`, `ctest -S` and `ctest --build-and-test` out of reach even under
  `--allow-edits`. This line used to say "the agent cannot write files by
  default", and that was **false**: the allowlist carried `Bash(cmake *)`, and
  `cmake -E rm -rRf <anything>` begins with `cmake `. Edits are opt-in per
  daemon launch with `--allow-edits`, never per utterance.
* **The spoken destructive word list is a courtesy, not the boundary.** It
  matches words in the *transcript*; you do not have to *say* a destructive word
  for the agent to *choose* a destructive action. "jarvis, the build directory
  is stale, sort it out" contains no word any list would carry. The boundary is
  the two lines above.
* **It keeps a log of the room.** A wake-word daemon hears everything audible,
  and `voice/logs/*.jsonl` records it. Transcripts are written **only for
  segments where the wake word actually hit**; everything else is logged as
  metadata (duration, voiced milliseconds, best wake ratio, near misses).
  `--log-all-audio` records the rest and warns at startup. Files are pruned
  after 14 days and rolled at 32 MB.
* **No launchd / LaunchAgent install.** The microphone grant belongs to the app
  that owns the process tree, and a LaunchAgent has no app identity. Start it
  from a terminal you have granted.
* **No transcript pruning.** Every `-p` turn writes to
  `~/.claude/projects/-Users-alessandroderosis-Desktop-M3LB/`, which already
  holds **798 MB** (largest single file 248 MB) on this 8 GB machine. Nothing
  here cleans it.
* **No silero VAD, no GBNF grammar, no ElevenLabs/Fish Audio.** The VAD is
  energy-based in Python; `ggml-silero-v5.1.2.bin` was never downloaded.
  `whisper-cli` has `--grammar` but `whisper-server` does not expose it, and
  per-utterance `whisper-cli` would cost the 1.6–4.0 s cold start the resident
  server exists to avoid. `say` is worse than ElevenLabs and it is already
  installed and needs no key.
* **No multi-directory.** The working directory is pinned to this repo at
  startup. `claude -p` skips the workspace-trust dialog and silently ignores
  malformed settings files, so pointing this at an arbitrary directory would run
  that directory's hooks and MCP servers with no prompt.
* **`voice/setup.sh` does not exist yet.** Setup is the four manual steps below.

---

## 1. The microphone — read this before anything else

**Nothing works against a digital-zero stream, and macOS reports one as success.**
`ffmpeg` exits 0, writes a valid WAV, and every sample is exactly zero.
Whisper will transcribe that as empty or invent text over it.

### Check it with one command

```bash
python3 /Users/alessandroderosis/Desktop/M3LB/voice/miccheck.py --quiet
```

Three outcomes, and the difference between them matters:

| verdict | what it means | what to do |
|---|---|---|
| `LIVE` | converter running, signal present | use it: `--device 'MacBook Air Microphone'` |
| `SILENT` | converter running, quiet room | fine — this is a working mic, not a fault |
| `DEAD` | 100.000 % of samples exactly zero | wrong device, dead device, or TCC — in that order |

The discriminator is the **fraction of exactly-zero samples**, not amplitude. A
live converter dithers even in silence; a blocked or dead one reads
`zero_frac = 1.000000`. An amplitude threshold cannot tell a blocked microphone
from a quiet room — both read "low".

### A single DEAD reading is not proof of a permission problem

Measured on this machine today, same device, same process tree, minutes apart:

```
07:33  [0] MacBook Air Microphone   DEAD   peak=  0   zero=100.000%
07:35  [0] MacBook Air Microphone   LIVE   peak=591   zero=  0.650%
07:36  [0] MacBook Air Microphone   LIVE   peak=740   zero=  0.531%
07:36  [0] MacBook Air Microphone   LIVE   peak=575   zero=  1.218%
07:37  [0] MacBook Air Microphone   LIVE   peak=153   zero=  1.893%
```

No System Settings change happened in between; input volume was 27 throughout.
The first reading was a cold CoreAudio input stream that had not started
delivering. **Re-run `miccheck.py` before touching System Settings.**

And in every one of those sweeps:

```
       [1] Microsoft Teams Audio     DEAD   peak=  0   zero=100.000%
```

A virtual device with no input, dead in the *same second* that the built-in
microphone was live, from the *same process*. A TCC denial cannot be
device-selective, so this is the proof that DEAD means "this device", not "this
permission". Try another device before blaming macOS.

### If it really is the permission

The grant attaches to the **app that owns the process tree**, not to `python3`
and not to `ffmpeg`. So it is the app you launch the daemon *from*:

* launched from Terminal -> grant **Terminal** (`com.apple.Terminal`)
* launched from iTerm -> grant **iTerm**
* launched from a Claude Code session -> grant **Claude**
  (`com.anthropic.claudefordesktop`)

Click path on macOS 27:

```
 System Settings  >  Privacy & Security  >  Microphone  >  [toggle your terminal on]
```

or jump straight there:

```bash
open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone'
```

**Quit and relaunch the terminal afterwards** — the toggle restarts the granted
process, and a running shell keeps the old denial.

Two things that do *not* work, so do not waste time on them: there is no CLI
that reads the grant back (`tccutil` implements only `reset`, and
`log show --predicate 'subsystem == "com.apple.TCC"'` prints the verdict as
`<private>`), and the unified log will show you the *request* but never the
answer. `miccheck.py` **is** the query.

Last, the cheap one:

```bash
osascript -e 'input volume of (get volume settings)'   # 0 here gives zeros with the grant fully on
```

### The daemon refuses to start against a zero stream

`jarvis.py` runs the same liveness test on the first 1.3 s of its **own**
capture stream — 2.79 ms of Python, against 7–10 s and two extra device opens
for spawning `miccheck` — and exits rather than entering its main loop.
Transcribing silence in silence is the failure nobody can debug afterwards.

---

## 2. Setup

Three of the four steps are already done on this machine. There is no
`setup.sh`; these are the commands it would contain.

```bash
# 1. whisper.cpp and ffmpeg      [DONE HERE: whisper.cpp 1.9.4, ffmpeg 9.0.2]
brew install whisper-cpp ffmpeg

# 2. models — brew ships none    [DONE HERE: both present in ~/.cache/whisper-models/]
#    Both URLs verified live today; each content-length matches the installed
#    file byte for byte (77,704,715 and 487,614,201).
mkdir -p ~/.cache/whisper-models
curl -L -o ~/.cache/whisper-models/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin   #  78 MB
curl -L -o ~/.cache/whisper-models/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin  # 488 MB

# 3. pre-warm the Metal shader cache — ONCE PER MACHINE, TAKES ~2 MINUTES
#    [DONE HERE: the cache is warm, whisper-server now starts in 0.22 s]
whisper-cli --help     # the first ever ggml run compiles 20 Metal libraries

# 4. the claude CLI               [NOT DONE HERE — this is the blocker]
npm i -g @anthropic-ai/claude-code    # no sudo: the npm prefix is writable
claude auth login                     # interactive, once, opens a browser
```

**Step 4 is required and is not optional.** The desktop app's
`Claude Code-credentials` keychain item is **not** inherited: it reads back with
`rc=0` and no prompt, and `claude auth status` still returns
`{"loggedIn": false, "authMethod": "none"}`. The desktop app authenticates its
child over a host socket, which a standalone CLI cannot read. Verify with:

```bash
claude auth status | jq -e '.loggedIn == true' && echo OK
```

`jarvis.py` makes this a hard preflight and refuses to start without it.

**Never set `ANTHROPIC_API_KEY`.** In `-p` mode "the key is always used when
present", with no approval step, so a stray key silently moves every request off
your subscription. It is absent from this machine's environment and from every
shell profile; `jarvis.py` strips it (and every `CLAUDE_*` var, and
`ANTHROPIC_BASE_URL`) from the child environment anyway.

---

## 3. Quickstart

### Start with no microphone at all

`--text` reads typed lines through the **identical** dispatch path — the same
normaliser, the same destructive gate, the same argv. `--dry-run` prints the
`claude` argv instead of running it. Together they need no microphone, no
whisper server and no login:

```bash
cd /Users/alessandroderosis/Desktop/M3LB

# the whole pipeline, typed, nothing dispatched
python3 voice/jarvis.py --text --dry-run --no-speak

# or non-interactively
printf 'jarvis, run the fast tests\n' | python3 voice/jarvis.py --text --dry-run --no-speak
```

That prints the exact 17-element argv it would have run, including the
`--allowedTools` list, and logs it as a `dry_run_dispatch` event. This is the
regression harness: it is how you check the normaliser and the safety gate
without spending a token or opening a device.

Drop `--dry-run` to actually dispatch typed commands (needs step 4 above):

```bash
python3 voice/jarvis.py --text
```

### Then with the microphone

```bash
python3 voice/miccheck.py --quiet                          # find a LIVE device
python3 voice/jarvis.py --device 'MacBook Air Microphone'  # resolve by NAME
```

**`--device` is required.** There is no default, because the only candidate for
one was avfoundation index 0 and this file spends a page explaining that index 0
means nothing. On this machine `[0]` has been the built-in microphone, the
Continuity iPhone and — on a machine with a call open — it can be
`Microsoft Teams Audio`, in which case the daemon captures the **meeting** and
every remote participant becomes a wake-word speaker. A device whose name looks
virtual or conference-like (Teams, Zoom, BlackHole, Loopback, Aggregate, …) is
refused unless you name it **exactly**.

**Pass a name, never an index.** avfoundation indices are assignment order, not
identity. This machine's list changed three times in seventy minutes:

```
 ~06:00   [0] MacBook Air  [1] AirPods Pro  [2] iPhone  [3] Teams
 ~06:50   [0] iPhone       [1] MacBook Air  [2] Teams
  07:12   [0] MacBook Air  [1] Teams
```

A saved `-i ":0"` therefore meant three different microphones. `jarvis.py`
enumerates at every start, matches a name, and warns loudly if you pass a digit.

Useful flags:

```bash
--verbose            per-segment VAD and transcription detail on stderr
--no-speak           print replies instead of speaking them
--allow-edits        let the agent write files (acceptEdits + Edit,Write)
--wake kestrel       change the wake word
--threads 2          whisper threads per tier (default 2, not 4 — see below)
--command-model ~/.cache/whisper-models/ggml-tiny.en.bin   # one server, less RAM
```

---

## 4. How it works

```
  ONE ffmpeg for the life of the daemon  (never one per utterance)
  ffmpeg -f avfoundation -i ":N" -ac 1 -ar 16000 -f s16le -
        |
        |  reader thread, 100 ms chunks, drops OLDEST when the queue is full
        v
  32 ms frames --> EnergyVAD          adaptive floor, +9.5 dB speech factor
        |                             512 ms pre-roll, 704 ms hangover
        v
  one segment (>= 320 ms)
        |
   state IDLE                              state ARMED / CONFIRM
        |                                          |
        v                                          v
  tiny.en  :18080   0.19 s               small.en  :18081   1.11 s
  "jarvis"? difflib ratio >= 0.75         + the PROSE initial prompt
        |                                          |
        | no -> discard, log the ratio             |
        v yes                                      |
  chime (Tink.aiff, 0.564 s)                       |
        +------------------------------------------+
        v
  normalise()      "dee three queue twenty seven" -> D3Q27,  "ptest" -> ctest
        v
  safety gate      destructive? -> CONFIRM, 20 s from the END of the question,
        |                          released only by "jarvis confirm", then
        |                          DISCARD (never queue)
        |                          D3Q19? -> refuse out loud, nothing runs
        v
  claude -p  --output-format json  --permission-mode dontAsk
             --allowedTools <read/build/test only>
             --disallowedTools <Write,Edit,cmake -E,cmake -P,ctest -S,...>
             --resume <session_id>
             cwd = the repo,  300 s wall-clock cap (os.killpg, no GNU timeout)
        v
  is_error?  -- yes --> speak the failure AS a failure
        v no
  say -v Daniel -r 190
```

### Why two tiers

The wake word is checked on every segment of room noise; the command is
transcribed once. So the cheap model runs constantly and the accurate one runs
only after a hit. Measured on a 3.000 s clip, `-t 4`, serial runs:

| | warm resident server | cold `whisper-cli` |
|---|---|---|
| `tiny.en` | **0.19 s** (0.171 0.238 0.156 0.162 0.199) | 1.60 s |
| `small.en` | **1.11 s** (1.113 1.264 0.767 1.310 1.078) | 3.96 s |

The resident server saves 1.4 s per wake check and 2.85 s per command — not the
~1 s a process spawn suggests, because a cold ggml process spends ~0.55 s
loading the BLAS/Metal/CPU backends before it even opens a model.

`tiny.en` is not good enough for commands: it cannot say **ctest** (it produces
"ptest") or **tau** ("that he are you"). Both are load-bearing words in this
repo. So a one-shot "jarvis, run the fast tests" is *re-transcribed* on
`small.en` rather than trusting `tiny.en`'s version — 1.1 s against making you
repeat yourself.

RSS with `tiny.en` resident is 128 MB. If 488 MB of `small.en` is too much on
this 8 GB machine, set `--command-model` equal to the wake model and one server
serves both tiers.

### The whisper prompt is prose, and deliberately contains no spoken forms

Whisper's initial prompt is conditioning *transcript text*, not a vocabulary
list, and it imitates whatever style it is given. Measured on identical audio at
temperature 0, so this reproduces and will never look like flakiness:

```
comma-separated keyword list:  "Run the D3Q27, K8, gamma, 100 with tau1."
                                ^^ it DELETED "colour gradient case"
the same vocabulary as prose:  "Run the D3Q27 colour gradient case at gamma one
                                hundred with tau one."          (exact)
```

And putting the spoken forms *in* the prompt backfires the same way: with "dee
two queue nine" in the prompt whisper emitted the literal `dee two queue nine`;
with that one clause removed and nothing else changed, `D2Q9`. So the prompt
holds only canonical forms and the spoken-to-canonical table lives in
`normalise()`, applied *after* transcription.

What the prompt actually buys is narrower than expected. `tiny.en` already gets
`D3Q27` right unprompted. The prompt fixes Orszag-Tang (from "Orces Actang"),
Rayleigh-Benard ("Rayleigh Bernard"), Poiseuille ("Poisell"), Esoteric Pull
("esoteric cool") and the British "colour". It is **not** sufficient — the same
sentence, same model, same prompt, differing only in the room noise spliced
around it, transcribed once as `ctest` and once as `ptest`. One run would have
"proved" the prompt fixed it. The alias table is what closes the gap.

### Latency budget

Every number measured on this machine (M1, 8 GB), most of them while a solver
held 70–82 % of the CPU. **They are upper bounds under load, not the idle cost
of the Mac** — which is the condition this assistant exists for, since you ask
it whether a solve has finished while the solve is running.

| stage | cost | note |
|---|---|---|
| VAD hangover after you stop talking | **0.70 s** | 22 frames; this is the floor, by design |
| `tiny.en` wake check | **0.19 s** | warm server, 3 s clip |
| `small.en` command transcribe | **1.11 s** | warm server, 3 s clip |
| chime (`afplay` Tink.aiff) | 0.56 s file + **0.9–1.9 s** CoreAudio open | overlaps the next stage |
| `claude -p` fixed overhead | **3.62 s** | before the model is reached at all |
| the actual agent turn | seconds to minutes | capped at 300 s wall clock |
| speaking a dynamic reply (`say`) | **1.3–2.1 s** at load 10.7, **3.5–6.5 s** at load 19–27 | synthesis only |
| speaking a *cached* fixed phrase | **1.4–2.0 s** | pre-rendered at startup, replayed with `afplay` |
| **floor before the agent is reached** | **≈ 5.6 s** | 0.70 + 0.19 + 1.11 + 3.62 |

Things that are *not* on the critical path: the VAD costs 34.0 ms of Python for
a 15.83 s stream (0.21 % of one core), and `normalise()` plus the wake match are
microseconds. Do not optimise them.

Two costs worth knowing separately:

* **The first ever ggml run on a machine compiles Metal shaders for
  112–116 seconds** and looks exactly like a hang. Measured twice (112.411 s,
  115.74 s); the very next run took 0.18 s. `SERVER_READY_S` is therefore 180 s,
  not 10, and the daemon prints why it is waiting. The cache is the *system* one
  under `/private/var/folders/.../C/com.apple.metal`, not anything in `~/.cache`,
  so it survives reinstalling whisper.cpp but not a cache wipe. On this machine
  it is warm: the daemon's own log records a 0.22 s warm-up POST.
* **`--threads 2`, not 4.** You run solvers on this machine. `whisper-cli` at
  `-t 4` fights them, and a foreground `validation/galilean` took 51.8 s wall at
  67 % CPU against a recorded ctest cost of 0.69 s.

### The stream can silently run slow, and nothing reports it

Measured sustained **0.75x real time** with the machine loaded, deficit growing
**linearly** at 0.24 s per second:

```
  t =  5 s -> 3.562 s delivered, deficit -1.441 s, 0.712x
  t = 10 s -> 7.498 s delivered, deficit -2.510 s, 0.749x
  t = 25 s -> 18.890 s delivered, deficit -6.121 s, 0.755x
```

`ffmpeg` stderr is empty, the exit code is zero, there is no gap in the byte
stream. A fixed start-up offset would have been harmless; a *growing* one means
a quarter of every second of speech is missing, and whisper covers gaps with
confident invention. It is not the pipe (writing straight to a `.wav` gives the
same), not the Continuity link, not the resampler, not scheduling priority.

So the daemon keeps a delivered-samples against wall-clock counter and splits
the response by tier:

* below **0.95x** — log `stream_slow` and warn on stderr
* below **0.80x** — **refuse the command tier out loud**, while still running
  the wake tier

That split is deliberate. A six-letter fuzzy match survives a gapped stream; a
sentence dispatched to a coding agent must not be invented. A blanket refusal at
0.95 would make the assistant useless on exactly the loaded machine it is for.

**It reproduced on the first real run: 0.682x, logged within seconds of the loop
starting, at load average 12.** This is the normal state of this Mac, not a
hypothetical, and it means the command tier really will refuse while a long
solve is running. `captureprobe.py` is the instrument for finding out what is
eating the audio thread:

```bash
python3 voice/captureprobe.py --device ':0' --seconds 25
ps -Ao %cpu,rss,comm -r | head
```

---

## 5. What you can say

**There is no fixed command list.** The daemon does not pattern-match your
sentence to a shell command — it normalises it and hands it to an agent that has
read `CLAUDE.md`. So the vocabulary below is two different things: what the
normaliser *rewrites*, and what the agent reliably *understands*.

### What the normaliser rewrites

Applied longest-match-first, case-insensitively, on word boundaries, tolerant of
whisper's free punctuation (`[\s,\-]+` between words). Every entry is a form this
machine's whisper models actually produced.

| you say | it becomes | why the alias exists |
|---|---|---|
| "dee three queue twenty seven" | `D3Q27` | safety net; whisper usually gets this unprompted |
| "dee two queue nine" / "d2 q9" | `D2Q9` | |
| "dee three queue seven" | `D3Q7` | |
| "dee three queue nineteen" | `D3Q19` | **so it can be refused** — see below |
| "bee gee kay" / "see em" | `BGK` / `central moments` | |
| "color gradient" | `colour gradient` | the tree is British; the `.en` models are not |
| "per color" / "percolour" | `percolour` | `static_droplet -rest percolour` |
| "rayleigh bernard" / "riley bernard" | `Rayleigh-Benard` | measured `tiny.en` output |
| "orces actang" / "orzac tang" | `Orszag-Tang` | measured `tiny.en` output |
| "poisell" / "pwazoy" / "poise oil" | `Poiseuille` | measured |
| "esoteric cool" / "esoteric pool" | `Esoteric Pull` | measured |
| "ptest" / "see test" / "c test" | `ctest` | **`tiny.en` cannot say ctest** |
| "that he are you" / "tao" | `tau` | **`tiny.en` cannot say tau** |
| "caucus" / "cocus" / "k8" | `Kokkos` | measured |
| "em three ell bee" | `M3LB` | |
| "ten to the six" / "one e six" | `1e6` | also `1e10`, `1e14` |
| "eff pee thirty two" | `FP32` | also `FP64` |
| "ee aitch dee" / "em aitch dee" | `EHD` / `MHD` | |

Bare **"test"** is deliberately *not* an alias for `ctest`: it would rewrite
"run the fast tests" into "run the fast ctests", and this repo genuinely has
both meanings.

### Utterances that work

These are not parsed — they work because the agent reads `CLAUDE.md`.

| say | what happens |
|---|---|
| "jarvis, build" | `cmake --build build -j4` |
| "jarvis, run the fast tests" | the sub-0.1 s ctest subset — **21 of the 38** finish that fast |
| "jarvis, run the tests" | all 38, **165 s** (`layered_poiseuille` alone is 74.7 s) — see the cost note below |
| "jarvis, did it pass" / "what failed" | `ctest --rerun-failed --output-on-failure` |
| "jarvis, run poiseuille" | `./build/validation/poiseuille --kokkos-num-threads=4` |
| "jarvis, run the static droplet at gamma one hundred" | the agent supplies `-n 48 -r 16 -tau 1 -steps 32000` |
| "jarvis, run the colour gradient cross check" | `ctest -R cross_colour` |
| "jarvis, git status" / "what changed on this branch" | read-only git |
| "jarvis, what's running" | `uptime`, `ps` |
| "jarvis, why is per colour the default" | it reads the banner and answers |
| "jarvis, frame check" | `sh tests/frame_check.sh` |

Two forms both work: **one-shot** ("jarvis, run the fast tests" — one breath,
the remainder after the wake word is the command) and **two-step** ("jarvis"
-> chime -> "run the fast tests", 8 s window).

`--kokkos-num-threads=4` is *not* automatic outside ctest. The agent knows this
from `CLAUDE.md`; if a reply reports a suspiciously slow run, that is the first
thing to check.

### What is gated or refused

| trigger | behaviour |
|---|---|
| `D3Q19` anywhere in the sentence | **refused out loud**, nothing dispatched. It was deleted 2026-09-18 and is now a hard error that prints NOTHING WAS RUN, so a stale spoken command line must not return "done" |
| delete, remove, wipe, `rm -rf` | "That would delete something. Say confirm if you mean it." |
| push, force-push | "...push to a remote" |
| commit, amend, stash | "...change the git history" |
| reset, revert, checkout, rebase, merge, discard | "...throw away uncommitted work" |
| kill, pkill, killall | "...kill a running process" |
| sudo, chmod, chown, launchctl | "...change the system" |
| curl, wget, pip/npm/brew install | "...download or install software" |
| "stop", "cancel", "never mind", "abort" | stops speech, returns to IDLE |

An unconfirmed destructive command is **discarded after 20 s, never queued** — a
destructive command that fires later, out of context, is worse than one that
never ran. The clock starts when the question has **finished being spoken**: it
used to start before, and the stop_the_run question alone is 5.1 s of audio, so
the window could close before you heard the word "confirm". When it closes, the
daemon says so.

**Say exactly "jarvis, confirm".** The whole utterance must be the wake word
followed by "confirm" (or "ok confirm", "confirmed") and nothing else. Not
"yes", not "go ahead", and — the reason for the rule — not "can you confirm the
numbers for Friday", which the earlier substring test approved, as it did "no,
don't confirm that". There is no speaker identification in this daemon, so
without the wake word anyone audible, including a meeting and the daemon's own
sentence coming back off the speakers, could release a pending `pkill`.
"no", "don't", "negative", "stop" and "cancel" all drop it.

Four defences stand behind that gate, and none is optional: **no shell, ever**
(every subprocess takes a list argv — a mis-heard "and rm dash rf" can only
become a literal argument, never shell syntax); the denylist above; **the agent
cannot edit by default**; and a **300 s wall-clock cap** implemented with
`os.killpg` on a new session, because there is no GNU `timeout` here and neither
`--max-turns` nor `--max-budget-usd` bounds *time* — a single Bash call running
a 40,000-step solve blocks the turn indefinitely under both.

---

## 6. Tuning

### Wake-word sensitivity

`WAKE_CUTOFF = 0.75` in `jarvis.py`, a `difflib.SequenceMatcher` ratio against
the wake word. The cutoff sits in the middle of a measured gap, not at a guess:

```
accept: jarvis 1.000  arvis 0.909  javis 0.909  darvis 0.833  jervis 0.833
        ---------------------------- 0.75 ----------------------------
reject: jar 0.667  harvest 0.615  java 0.600  harvey 0.500  service 0.462
        charles 0.462  carl 0.400  chavez 0.333
```

Lowest accept 0.833, highest reject 0.667. Raise the cutoff if it fires on
speech that is not aimed at it; lower it if it ignores you. Every *rejected*
near-miss above `WAKE_LOG_FLOOR = 0.55` is logged with its ratio, so retune from
the log rather than from the memory of being ignored.

If you change the wake word with `--wake`, pick something with the same
property: two syllables, no common English near-neighbours. Check candidates
against your own speech by reading the logged ratios.

### VAD thresholds

All at the top of `jarvis.py`, each with its measurement:

| constant | value | what moves it |
|---|---|---|
| `SPEECH_FACTOR` | 3.0 (+9.5 dB over the tracked floor) | room floor is 7–15 counts, speech is rms 4047–4656 — two and a half orders apart, so this is not a close call |
| `ABS_MIN_RMS` | 60 counts (0.0018 FS) | stops a pathologically quiet floor making the relative threshold meaningless |
| `ONSET_FRAMES` | 3 (96 ms) | raise it if key clicks start segments |
| `HANGOVER_FRAMES` | 22 (704 ms) | raise it if you pause mid-sentence and get cut off; it is also the latency floor |
| `PREROLL_FRAMES` | 16 (512 ms) | lower it and "jarvis" arrives as "arvis" (still matches at 0.909) but a two-word command loses its verb (does not) |
| `MIN_SEG_MS` | 320 | shorter than this is a cough |
| `SPEECH_TAIL_S` | 0.45 | speaker decay before the VAD reopens; raise it if the daemon hears itself |
| `ARM_TIMEOUT_S` | 8.0 | how long "jarvis" leaves the door open |

The floor rises slowly (tau 3.0 s) and falls quickly (tau 0.3 s), so speech
cannot drag it up and deafen the daemon, while a fan switching off is tracked at
once.

### Reading the log

One JSON object per line, per day, in `voice/logs/jarvis-YYYYMMDD.jsonl`.

**Diagnosing a missed wake** — this is the query that matters:

```bash
cd /Users/alessandroderosis/Desktop/M3LB/voice
jq -r 'select(.event=="wake_check") |
       "\(.ts)  hit=\(.hit)  best=\(.best)  rt=\(.realtime)  \(.heard)"' \
   logs/jarvis-*.jsonl
```

A real line from this machine:

```
2026-09-20T07:24:16.295+02:00  hit=false  best=0.308  rt=0.698  [MUSIC PLAYING]
```

Read it in three parts. `heard` is what whisper returned — here a non-speech
annotation, from a quiet room, which sailed through as if it were speech and was
wake-checked at ratio 0.308. (The current build strips annotations *by form*,
`\[...\]` and `*...*`, rather than by an enumerated list, because the list is not
knowable in advance — that fix was written after this line was logged.) `best`
is the wake ratio: if your "jarvis" is landing at 0.6–0.7, the cutoff is the
problem; if `heard` is empty or nonsense, the audio is. And `rt=0.698` says the
stream was delivering 70 % of real time at that moment, which is its own answer.

Other events worth knowing:

| event | tells you |
|---|---|
| `liveness` | `verdict`, `peak`, `zero_frac` from the daemon's own first 1.3 s |
| `device_selected` | which index a name resolved to, and the full list that day |
| `stream_slow` | `realtime` ratio and `dropped_chunks`, at most once a minute |
| `refused_gapped_stream` | a command the daemon declined to guess at |
| `utterance` | `raw` vs `normalised` — this is how you check the alias table |
| `confirm_required` / `confirmed` / `confirmation_refused` | the destructive gate, with what it heard |
| `refused_removed_lattice` | a D3Q19 command that was stopped |
| `dry_run_dispatch` | the full argv, under `--dry-run` |
| `claude_result` | `is_error`, `subtype`, `num_turns`, `cost_usd`, `session_id`, `permission_denials` |
| `shutdown` | `spend_usd` for the session, `dropped_chunks` |

To see what the normaliser is doing to you:

```bash
jq -r 'select(.event=="utterance") | "\(.raw)\n  -> \(.normalised)\n"' logs/jarvis-*.jsonl
```

To total what a day cost:

```bash
jq -s 'map(select(.event=="claude_result") | .cost_usd // 0) | add' logs/jarvis-*.jsonl
```

---

## 7. Troubleshooting

**The microphone is silent / everything transcribes as nothing.**
`python3 voice/miccheck.py --quiet`. If it says DEAD, run it *again* — a cold
CoreAudio stream reads DEAD once (measured above). If it stays DEAD on one
device but another is LIVE in the same sweep, that device is dead, not your
permissions. Only if *every* device is DEAD across repeated runs is it TCC, and
then grant the terminal app in section 1 and **relaunch the terminal**.

**The wake word never fires.**
Check `best` in the `wake_check` log lines. If `heard` is empty, it is audio,
not the cutoff — check `liveness` and `rt`. If `best` sits at 0.6–0.7, lower
`WAKE_CUTOFF` or pick a wake word that survives your accent. If there are no
`wake_check` lines at all, the VAD never emitted a segment: run with `--verbose`
to see per-segment `floor=` and `thr=` and raise `SPEECH_FACTOR` down, or check
that `ABS_MIN_RMS = 60` is not above your speech level (it should be an order of
magnitude below it).

**The wake word fires constantly.**
Raise `WAKE_CUTOFF` toward 0.85 — you have room, since the lowest genuine accept
measured was 0.833. Check the logged `near_misses` to see what is colliding. If
segments are being emitted from pure room noise, the floor tracker is being
dragged down: raise `ABS_MIN_RMS`.

**It triggers on its own voice.**
The guard is time-based: VAD closed while `say` is alive, plus `SPEECH_TAIL_S`
(0.45 s) of decay, with the pre-roll buffer cleared on ungating. Raise
`SPEECH_TAIL_S` first. Acoustic self-trigger was never measured here — system
output was muted during the recon and the user's setting was not changed — so no
claim is made that this guard is sufficient with loud speakers. If it proves
insufficient, the fix is **headphones**, not a cleverer timer.

**`claude` is not authenticated.**
The symptom is a spoken "Claude Code is not logged in." The daemon detects this
before the loop starts and refuses, and it detects it *again* per dispatch,
because an auth failure returns `"subtype": "success"` with `"is_error": true`
and the failure sentence in `result` on **stdout**. A daemon that branched on
`subtype`, or used `--output-format text`, would speak "Not logged in" in the
voice of an answer. Fix: `claude auth login`, once, interactively. The desktop
app's keychain item does not help.

**`whisper-server` will not start.**
If this is the first ever ggml run on the machine, it is compiling Metal shaders
and will take **112–116 seconds**. It looks exactly like a hang. Wait — the
daemon waits up to 180 s and prints why. Otherwise:

```bash
lsof -nP -iTCP:18080 -sTCP:LISTEN      # something else on the port?
whisper-server -m ~/.cache/whisper-models/ggml-tiny.en.bin --host 127.0.0.1 --port 18080 -t 2
```

An already-listening server is **not adopted** unless you pass
`--adopt-whisper`: the port is unprivileged and fixed, the liveness test is a
bare HTTP GET, and whatever bound 18080 first would then return the `{"text":
...}` that becomes a spoken command — i.e. it would choose every word the daemon
believes it heard, and could emit free text straight into the agent. By default
a busy port is worked around: the daemon picks a free ephemeral one and starts
its own server, which costs the memory that adoption was avoiding (a second copy
of `small.en` is 488 MB) and is the only option that does not rest on a
stranger. With `--adopt-whisper`, adoption is checked rather than assumed —
`POST /load` must succeed and the server must answer a real `/inference` with a
whisper-shaped envelope, or startup fails. That proves the API, not the
server's honesty. An adopted server is not reaped on exit — it was not ours to
kill. If the models are missing,
`ls -la ~/.cache/whisper-models/` should show 77,704,715 and 487,614,201 bytes.

**AirPods connect mid-session.**
Two different failures, and they look nothing alike. If the *current* device
disappears, `ffmpeg` fails **loudly** — EOF on the pipe plus a non-zero exit
(rc=251, "Invalid audio device index") — and the daemon prints
`the capture stream ended` and exits 1. That is correct: it must never read a
vanished device as silence. **Restart it**; it will re-enumerate. If the AirPods
merely *appear*, every index shifts under you, but the daemon resolved a name at
startup and keeps its open stream, so nothing moves mid-session — the surprise
comes at the *next* start, when `:0` means something else. This is the whole
argument for `--device 'name'`.

**Replies are fine but take forever.**
Check `rt` in the log and run `captureprobe.py`. Then check what is eating the
machine (`ps -Ao %cpu,rss,comm -r | head`). Expect ~5.6 s before the agent is
even reached; that floor is structural.

---

## 8. Cost

### Free — everything local

Speech recognition (`whisper-server`, both tiers), capture (`ffmpeg`), the VAD,
the wake match, `normalise()`, the safety gate, `say`, `afplay`, and all
logging. These cost CPU and nothing else. The wake tier runs on every segment of
room noise all day and costs zero tokens, which is the entire reason for the
two-tier split.

### Paid — one `claude -p` turn per dispatched command

**Not measured on this machine: the `claude` CLI is not installed yet**, so
there is no recorded `total_cost_usd`. What follows is structural — the knobs,
not a measurement. Measure it yourself after step 4 with the `jq` one-liner in
section 6.

What each dispatched utterance pays for:

* **The `CLAUDE.md` system prompt, every turn.** This repo's `CLAUDE.md` is
  large, and `--bare` is deliberately *not* used — it would skip `CLAUDE.md`
  auto-discovery, and that file is exactly what makes "run the static droplet at
  gamma one hundred" mean anything. It would also break subscription auth
  outright. This is the single biggest per-turn input cost and it is the price
  of the design.
* **Tool output.** A "run the tests" that streams 165 s of ctest output back is
  far more expensive than "git status".
* **Agentic turns**, capped at `--max-turns 12`.

The caps, per turn:

```
--max-budget-usd 0.50     # --budget
--max-turns 12            # --max-turns
300 s wall clock          # --claude-timeout, enforced here, not by the CLI
```

`--max-budget-usd` stops a turn when the cap is hit (`subtype`
`error_max_budget_usd`); it does **not** bound time. Spend from subagents counts
toward it.

**Session continuity cuts the cost of a follow-up.** The daemon captures
`session_id` from each result envelope and passes `--resume` on the next
utterance within `SESSION_IDLE_S = 1800`, so the second question in a
conversation reads the prompt from cache rather than re-sending it. `--continue`
is deliberately not used: it resolves to "most recent session in this
directory", which is a race the moment you also have Claude Code open in this
repo.

The running total for a session is summed from every `total_cost_usd` and
written to the `shutdown` log line as `spend_usd`. Note that all cost figures are
client-side estimates, that `usage` excludes subagent tokens while
`total_cost_usd` includes them, and that `total_cost_usd` is per-call under
`--output-format json` (what this uses) but *cumulative* under `stream-json` —
never mix the two.

**The non-token cost is disk.** Every `-p` turn appends to
`~/.claude/projects/-Users-alessandroderosis-Desktop-M3LB/<session-id>.jsonl`.
That directory already holds **798 MB** on an 8 GB machine.
`--no-session-persistence` exists but is incompatible with the `--resume`
continuity above, so the answer is periodic pruning, which nothing here does for
you.
