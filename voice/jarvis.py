#!/usr/bin/env python3
# jarvis.py -- a wake-word voice front end for this repository.
#
#   mic -> ffmpeg (raw s16le 16 kHz mono) -> ring buffer -> energy VAD -> segment
#       -> tiny.en on a resident whisper-server   -> wake word "jarvis"?
#       -> chime, then small.en with a prose prompt -> transcript
#       -> normalise -> safety gate -> claude -p in this repo -> `say`
#
# EVERY NUMBER IN THIS BANNER WAS MEASURED ON THIS MACHINE (macOS 27.0/26A428,
# M1, 8 GB, Python 3.14.7) ON 2026-09-19/20, MOSTLY WHILE ./build/demonstrator/urban
# HELD 82 % OF THE CPU.  They are therefore upper bounds under load, which is the
# condition this assistant exists to work in -- you ask it whether a solve has
# finished while the solve is running -- but they are NOT the idle cost of the Mac
# and must not be quoted as one.  CLAUDE.md's Measurement discipline applies to
# this file as much as to the solver.
#
# ---------------------------------------------------------------------------
# WHY A RESIDENT SERVER AND NOT whisper-cli
#
# Warm whisper-server against cold whisper-cli, 3.000 s clip, -t 4, serial runs:
#
#     tiny.en    warm server 0.19 s (5 reps: 0.171 0.238 0.156 0.162 0.199)
#                cold CLI    1.60 s (3 reps: 2.02 1.28 1.51)
#     small.en   warm server 1.11 s (5 reps: 1.113 1.264 0.767 1.310 1.078)
#                cold CLI    3.96 s (3 reps: 3.50 4.04 4.35)
#
# So the server saves 1.4 s per wake check and 2.85 s per command -- not the ~1 s
# a process spawn would suggest, because a cold ggml process spends ~0.55 s loading
# the BLAS/Metal/CPU backends before it even opens a model.  RSS with tiny.en
# resident is 128 MB.
#
# THE 4 s FIRST-POST PENALTY DID NOT REPRODUCE, AND THE WARM-UP POST IS KEPT
# ANYWAY.  An earlier measurement recorded the first POST after server launch at
# 4.04 s (GPU pipeline warm-up) against 0.34-0.48 s for every later one, and this
# file was written to spend that 4 s on a throwaway health check rather than on
# the user's first command.  Re-measured here on a freshly started server,
# tiny.en, 4.42 s clip, five POSTs: 0.40, 0.45, 0.20, 0.48, 0.36 s -- the first is
# indistinguishable from the rest.  The variable is the system Metal shader cache,
# which was cold for that first measurement and is warm now; the 4 s was the GPU
# pipeline being built, not a per-server cost.  So the warm-up POST no longer buys
# 4 s.  It is kept because it costs 0.4 s, because it is the only thing that
# proves the tier can actually transcribe before the daemon claims to be ready,
# and because a wiped Metal cache would bring the penalty straight back.
#
# AND THE FIRST ggml RUN EVER ON A MACHINE COMPILES METAL SHADERS FOR ~112-116 s.
# Measured twice (112.411 s from `whisper-cli --help`, 115.74 s from a cold
# transcribe); the very next run took 0.18 s and 3.23 s.  The cache is the system
# one at /private/var/folders/.../C/com.apple.metal, not anything under ~/.cache,
# so it survives reinstalls of whisper.cpp but not a cache wipe.  It looks EXACTLY
# like a hung daemon.  SERVER_READY_S is therefore 180 s, not 10, and the wait
# prints why.
#
# whisper-command and whisper-stream were both rejected, despite whisper-command
# having real wake-word gating (-p activation prompt, -cmd command list, --grammar):
# they link libSDL2 and open the microphone THEMSELVES, so they cannot use the
# resolve-by-name this machine needs; neither accepts an audio file as input (in
# both, -f is the TEXT OUTPUT file), so neither can be tested without a live mic;
# each holds its own model copy instead of sharing the resident server; and
# whisper-command endpoints on fixed windows (-pms 5000 / -cms 8000) rather than on
# speech.  The state machine below is that borrowed idea, rewritten against
# whisper-server.
#
# ---------------------------------------------------------------------------
# WHY THE AUDIO IS WRAPPED BY `wave` AND NOT PIPED RAW
#
# whisper-server REJECTS raw PCM: 352000 header-less bytes POSTed to /inference
# returned `Invalid request`, HTTP 400.  The container is mandatory.  numpy and
# sounddevice are banned here (the stdlib-only rule tools/osm_city.py already
# follows, and there are no reliable 3.14 wheels on this machine anyway), so the
# ffmpeg s16le bytes are wrapped in-memory by the stdlib `wave` module.  That was
# checked, not assumed: the stdlib header produced a BYTE-IDENTICAL transcript to
# ffmpeg's own WAV, and is in fact the cleaner file -- ffmpeg inserts a LIST/INFO
# chunk where `wave` goes straight to `data`.  Full path proven end to end at
# 0.47 s: ffmpeg -> wave -> urllib multipart -> {"text": ...}.
#
# audioop and aifc were removed in Python 3.13 (PEP 594) and this box runs 3.14.7,
# so every RMS/VAD snippet on the internet crashes at import here.  All the signal
# arithmetic below is array('h') plus integer maths.  Note that -min(a) overflows
# to 32768 on a full-scale negative sample; it is clamped.
#
# ---------------------------------------------------------------------------
# WHY THE MICROPHONE IS RESOLVED BY NAME, EVERY START, AND NEVER CACHED
#
# The avfoundation device list changed THREE TIMES in one morning:
#
#     brief, ~06:00   [0] MacBook Air  [1] AirPods Pro  [2] iPhone  [3] Teams
#     recon, ~06:50   [0] iPhone       [1] MacBook Air  [2] Teams
#     this run, 07:12 [0] MacBook Air  [1] Teams
#
# A saved `-i ":0"` therefore meant three different microphones in seventy minutes.
# Indices are assignment order, not identity.  This file enumerates at every start
# and matches a NAME; it refuses to write an index anywhere persistent.
#
# A DEAD DEVICE IS NOT A PERMISSION PROBLEM, AND SAYING SO WOULD BE A LIE.  On the
# same ffmpeg invocation, from the same process, in the same second, the built-in
# "MacBook Air Microphone" returned 48000 consecutive samples of EXACT ZERO
# (mean_volume = max_volume = -91.0 dB, histogram_91db covering every sample) while
# the Continuity iPhone microphone returned peak 811, 0.0248 FS.  A TCC denial
# cannot be device-selective, so that machine had the grant and one device was
# dead.  The remedy text in miccheck.py lists wrong-device and dead-device BEFORE
# permissions for that reason.  What this file does is refuse to enter the main
# loop against a zero stream -- transcribing silence in silence is the failure
# nobody can debug afterwards.
#
# The discriminator is the fraction of EXACTLY-ZERO samples, not amplitude: a live
# converter dithers (measured zero_frac 0.034 to 0.425 across live captures, quiet
# room included) and a dead one reads 1.000000.  An amplitude threshold cannot tell
# a blocked microphone from a quiet room; both read "low".
#
# AND THAT DISTINCTION IMMEDIATELY EARNED ITSELF.  On this daemon's first run
# against real hardware the built-in "MacBook Air Microphone" -- the device
# recorded as digital-zero two hours earlier -- reported peak 31, rms 8.0,
# zero_frac 0.052782: LIVE, and merely in a quiet room.  An amplitude test would
# have called peak 31 blocked and sent the user into System Settings to fix a
# permission that was never revoked.  The device that is dead is not a property of
# the machine; it changes between runs, which is the whole argument for testing it
# at every start instead of remembering it.  That test lives in
# miccheck.py and is IMPORTED rather than copied, because the argument for it is
# written there and two copies would drift apart.  The test runs on the first 1.3 s
# of this daemon's OWN long-running stream: 2.79 ms of Python, against 7-10 s and
# two extra device opens for spawning miccheck as a subprocess.
#
# ONE FFMPEG FOR THE LIFE OF THE DAEMON, NEVER ONE PER UTTERANCE.  The avfoundation
# device open costs 1.55-3.71 s (eight measurements), and `-t N` charges that
# against the window you asked for: `-t 1.5` delivered 81 % of the samples and
# `-t 2` delivered 80 %, so an utterance bounded by -t loses its first syllable.
# The stream is read by byte count and segmented in Python instead.  The first
# 100 ms after a cold open also carries a device-open transient at ~100x the
# settled floor (peak 1200 against 7-15 two windows later), so WARMUP_S of audio is
# discarded AFTER the first byte, never from process start -- otherwise the daemon
# wakes itself on the sound of its own microphone opening.
#
# THE STREAM CAN SILENTLY RUN SLOW AND NOTHING REPORTS IT.  Measured sustained
# 0.75x real time with the machine loaded, deficit growing LINEARLY at 0.24 s per
# second (-1.44 s at t=5 s, -6.12 s at t=25 s), ffmpeg stderr empty, exit code zero,
# no gap in the byte stream.  A fixed start-up offset would have been harmless; a
# growing one means a quarter of every second of speech is missing, and whisper
# covers gaps with confident invention.  So this daemon keeps a delivered-samples
# against wall-clock counter (captureprobe.py is the standalone instrument) and
# splits the response by tier: below RATE_WARN=0.95 it logs and warns, and below
# RATE_REFUSE=0.80 it refuses the COMMAND tier out loud while still running the
# wake tier.  That split is deliberate -- a six-letter fuzzy match survives a gapped
# stream, but a sentence dispatched to a coding agent must not be invented.  A
# blanket refusal at 0.95 would make the assistant useless on exactly the loaded
# machine it is for.
#
# IT REPRODUCED ON THE FIRST REAL RUN: 0.682x, logged as `stream_slow` within
# seconds of the loop starting, on a machine at load average 12.  So this is not a
# defence against a hypothetical -- it is the normal state of this Mac, and the
# command tier really will refuse until the machine is freed.  That is the
# intended behaviour and it is better than a confident wrong transcript, but it
# does mean the assistant is degraded exactly when a long solve is running, which
# is when it is most wanted.  The fix is not in this file; it is whatever is
# eating the audio thread, and captureprobe.py is the instrument for finding it.
#
# ---------------------------------------------------------------------------
# THE WHISPER PROMPT IS PROSE, AND THE SPOKEN FORMS ARE DELIBERATELY NOT IN IT
#
# whisper's initial prompt is conditioning TRANSCRIPT TEXT, not a vocabulary list,
# and it imitates whatever style it is given.  Both halves of that were measured on
# identical audio at temperature 0 (so it reproduces, and will never look like
# flakiness):
#
#   comma-separated keyword list: "Run the D3Q27, K8, gamma, 100 with tau1."
#                                 -- it DELETED "colour gradient case"
#   the same vocabulary as prose: "Run the D3Q27 colour gradient case at gamma one
#                                  hundred with tau one."   (exact)
#
# And putting the spoken forms in the prompt BACKFIRES for the same reason.  With
# "dee two queue nine" in the prompt whisper emitted the literal `dee two queue
# nine`; with that one clause removed, and nothing else changed, it emitted `D2Q9`.
# So the prompt contains ONLY canonical forms, and the spoken-to-canonical table
# is applied AFTER transcription.  Both now live in vocabulary.json rather than
# here; that file's banner carries the measurements.
#
# What the prompt actually buys is narrower than expected: tiny.en already gets
# D3Q27 right with no prompt at all.  The prompt fixes Orszag-Tang (from "Orces
# Actang"), Rayleigh-Benard (from "Rayleigh Bernard"), Poiseuille (from "Poisell"),
# Esoteric Pull (from "esoteric cool") and the British "colour".  Measured through
# this file's own transcribe(), tiny.en, `say`-rendered clips:
#
#     no prompt  "Jarvis, build M3LB with caucus threads on and run, test."
#     prose      "Jarvis, build M3LB with Kokkos spreads on and run ctest."
#     no prompt  "...at Gamma 100 with TOU1."
#     prose      "...at gamma 100 with tau1."
#
# AND THE PROMPT IS NOT SUFFICIENT, WHICH IS WHY THE ALIAS TABLE EXISTS.  The same
# sentence, the same model, the same prompt, differing only in the room noise
# spliced around it, transcribed once as `ctest` and once as `ptest`.  One run
# would have "proved" the prompt fixed it.  Two runs show the prompt shifts the
# odds and normalise() is what closes them, so the aliases for ptest and for tau
# ("that he are you", at tiny.en) are load-bearing rather than belt-and-braces,
# and the command tier still defaults to small.en, where both are correct.
#
# ---------------------------------------------------------------------------
# THE WAKE CUTOFF IS A MEASURED GAP, NOT A GUESS
#
# difflib.SequenceMatcher ratio against "jarvis", over the near-misses whisper
# actually produces plus the words most likely to collide:
#
#     accept: jarvis 1.000  arvis 0.909  javis 0.909  darvis 0.833  jervis 0.833
#     reject: jar 0.667  harvest 0.615  java 0.600  harvey 0.500  service 0.462
#             charles 0.462  carl 0.400  chavez 0.333
#
# The gap between the lowest accept (0.833) and the highest reject (0.667) is wide,
# and WAKE_CUTOFF = 0.75 sits in the middle of it.  Trailing punctuation is stripped
# first because whisper reliably emits "jarvis," with a comma.  Every REJECTED
# near-miss above 0.55 is logged with its ratio, so the cutoff can be retuned from
# evidence rather than from memory of being ignored.
#
# Checked through wake_match() on whole sentences rather than on bare tokens, which
# is how it is actually used:
#
#     "jarvis run the fast tests"          hit, 1.000, rest "run the fast tests"
#     "hey jarvis what changed ..."        hit, 1.000, rest "what changed ..."
#     "darvis, git status"                 hit, 0.833, rest "git status"
#     "the jar is full"                    MISS, 0.667
#     "is there a service running"         MISS, 0.500
#
# The remainder is what makes a one-shot "jarvis, run the fast tests" work without
# a second capture window.
#
# THE VAD COSTS 0.21 % OF ONE CORE, so none of this is worth optimising.  Measured
# over a synthetic 15.83 s stream (3.0 s room, 4.42 s speech, 2.5 s room, 4.42 s
# speech, 1.5 s room): 34.0 ms of Python for the whole stream, two segments emitted
# for two utterances with nothing split or merged, each carrying +0.96 s of pre-roll
# and hangover around the speech, and the adapted floor settling at 5.38 against a
# synthetic room of +-9 counts -- below ABS_MIN_RMS, so the absolute floor is what
# set the threshold, which is exactly the case it is there for.  Both segments
# transcribed with the wake word intact, i.e. the pre-roll did its job.
#
# That run predates PREROLL_FRAMES going from 16 to 19 frames, so the +0.96 s
# figure is 96 ms light against the code as it stands; the cost and the segment
# count are unaffected.  --selftest now carries the VAD's own regression tests
# (vad_selftest), which measure the lead-in at 512 ms rather than asserting it,
# and which STEP the room level rather than ramping it -- see EnergyVAD's own
# banner for why that distinction is the whole thing.
#
# ---------------------------------------------------------------------------
# THE DATA LIVES IN TWO JSON FILES, AND ONE OF THEM REVERSES AN EARLIER DECISION
#
# vocabulary.json turns sound into canonical text -- the whisper prompt and the
# alias table -- and decides nothing.  commands.json decides: the spoken phrases,
# the destructive table, the refusals, and a registry that routes an utterance
# either to a literal argv or to a prompt template for the agent.  Neither table
# is in this file any more; this file loads, validates and applies them.
#
# THE REGISTRY CONTRADICTS WHAT THIS BANNER USED TO SAY, AND ONLY HALF OF THE
# OLD ARGUMENT SURVIVES.  It used to read: "No deterministic shell routing.  A
# spoken 'build' is NOT pattern-matched to a cmake argv; it is handed to the
# agent, which reads CLAUDE.md and decides.  A routing table is faster and
# cheaper, and it was left out on purpose: two places that know what 'build'
# means drift apart, and the agent is the one that can read the banner it is
# about to contradict."
#
# The drift half is right and is why the registry is ONE file, declarative, with
# nothing about any command in this module.  The conclusion was wrong.  Sending
# "git status" to the agent charges ~3.5 s of fixed CLI overhead before the model
# is reached -- measured: --version 0.12 s, auth status 0.97 s, -p to first
# verdict 3.62 s -- plus a network round trip, to run a command that takes
# milliseconds.  And it is not only slower: a closed set of literal argvs cannot
# mis-read the request.  It either matched a pattern in a file that can be
# reviewed in a diff, or it did not.
#
# WHAT THE REGISTRY IS NOT ALLOWED TO BE IS A GATE.  An utterance matching
# nothing still goes to the agent as free text, so adding an entry can only make
# an answer cheaper and more predictable -- it can never make a question
# unanswerable.  That is what stops the table from having to be complete.
#
# NO TRANSCRIPT TEXT EVER REACHES AN argv.  An argv element may carry {repo} and
# {threads}, expanded from this daemon's own configuration, and nothing else; an
# unknown {token} is a fatal load error rather than a literal brace on a command
# line.  {text} is substituted only into a PROMPT, which travels as one list
# element to `claude -p`.  So a mis-heard word can pick the wrong entry; it
# cannot invent one and it cannot add a flag.
#
# ORDER IS LOAD-BEARING AND IS CHECKED RATHER THAN TRUSTED.  Patterns are tried
# in file order and the first match wins, so "run the fast tests" has to precede
# "run the tests".  `--selftest` asserts that every entry's own examples reach
# that entry and not an earlier one; it caught three shadowing bugs the first
# time it ran, including "import the city for Leeds" being swallowed by the
# urban plume entry.
#
# AND THE SELF-TEST'S FIRST VERSION HAD THE BUG IT EXISTS TO CATCH.  It
# re-implemented the gate ordering instead of calling it, and therefore could not
# see that "stop the run" never reached its own entry: the cancel rule matched a
# cancel word within three words and answered "Cancelled." The gate is now one
# function, gate(), and the self-test calls it.  An ordering described in two
# places will differ in two places.
#
# ---------------------------------------------------------------------------
# THE TRANSCRIPT IS UNTRUSTED INPUT THAT REACHES A CODING AGENT
#
# Four defences, none of them optional:
#
#   1. NO SHELL, EVER.  Every subprocess in this file takes a LIST argv.  There is
#      no shell=True and no string interpolation into a command anywhere, so a
#      mis-heard "and rm dash rf" cannot become shell syntax -- it can only become
#      a literal argument to something.
#   2. A DENYLIST OF DESTRUCTIVE INTENTS requires a spoken "jarvis, confirm"
#      before dispatch -- the WHOLE utterance, wake word included, nothing else
#      (see CONFIRM_RE, which records the sentences the old substring test
#      approved).  Unconfirmed requests are DISCARDED, not queued: a queued
#      destructive command that fires later, out of context, is worse than one
#      that never ran.
#
#      AND THIS DENYLIST IS A COURTESY FOR OBVIOUS PHRASING, NOT THE SECURITY
#      BOUNDARY.  It matches words in the TRANSCRIPT, and a user does not have
#      to SAY a destructive word for the agent to CHOOSE a destructive action:
#      "jarvis, the build directory is stale, sort it out" carries no word any
#      list would have, and the agent's reasonable response to it is to delete
#      the build tree.  Of twenty-two utterances run through the real gate(),
#      thirteen reached the agent as free text with no confirmation.  The
#      entries in commands.json close the ones that can be closed; defence 3 is
#      the one that holds.
#   3. WHAT THE CHILD IS PERMITTED TO DO -- AND THIS IS THE ACTUAL BOUNDARY.
#      `claude -p` starts in Manual mode, which is READS ONLY, so it would
#      politely refuse everything -- that is the silent "the assistant is
#      useless" failure.  This file passes --permission-mode dontAsk (so the
#      session never waits for input that a daemon cannot supply) together with
#      an explicit --allowedTools list AND a --disallowedTools backstop.  git is
#      allowed only in its read verbs (status/log/diff/show), and edits are
#      opt-in per daemon launch with --allow-edits, not per utterance, because a
#      single mis-heard word must not be able to rewrite a source file.
#
#      THE ALLOWLIST USED TO SAY Bash(cmake *), WHICH IS ARBITRARY EXECUTION AND
#      ARBITRARY RECURSIVE DELETION.  `cmake -E rm -rRf <anything>`,
#      `cmake -E env -- <any binary> <any args>` and `cmake -P <script>.cmake`
#      all begin with `cmake `; `ctest -S` and `ctest --build-and-test
#      --test-command` are the same shape.  A prefix glob over a multi-tool
#      binary is not an allowlist.  It is now the exact forms the registry
#      needs, plus the deny list -- see READ_TOOLS and DENY_TOOLS.
#
#      ONE THING HERE IS UNVERIFIED: whether --permission-mode dontAsk DENIES a
#      tool call that is not on the allowlist or PROCEEDS with it.  It could not
#      be checked on this machine, where the claude CLI is installed but not
#      logged in.  If it proceeds, the allowlist is advisory and only
#      --disallowedTools is load-bearing, which is why both are passed.
#   4. A WALL-CLOCK CAP, IMPLEMENTED HERE.  There is no GNU `timeout` on this
#      machine, and neither --max-turns nor --max-budget-usd bounds time: a single
#      Bash tool call running a 40,000-step solve blocks the turn indefinitely
#      under both.  Every child is spawned with start_new_session=True and killed
#      with os.killpg -- the load-bearing part, since claude spawns node and bash
#      GRANDCHILDREN that survive killing the direct child.  SIGTERM first (Claude
#      Code exits 143, tears down its Bash process tree and runs SessionEnd hooks,
#      and the unfinished turn stays resumable by session id), SIGKILL after a
#      grace period.
#
# ---------------------------------------------------------------------------
# WHAT THE claude CLI ACTUALLY RETURNS, AS OPPOSED TO WHAT IT LOOKS LIKE
#
# AN AUTH FAILURE RETURNS `"subtype": "success"` WITH `"is_error": true`, and the
# failure sentence arrives in `result` on STDOUT, not stderr.  Measured envelope:
# subtype=success, is_error=true, terminal_reason=api_error, result="Not logged in
# - Please run /login".  So a daemon that branches on subtype, or that uses
# --output-format text, SPEAKS "Not logged in" in the voice of an answer.  This file
# branches on is_error and only ever uses --output-format json.
#
# Auth is NOT inherited from the desktop app, and the keychain does not help.  The
# "Claude Code-credentials" item exists and its 3067-byte secret reads back with
# rc=0 and no prompt, and `claude auth status` still reports {"loggedIn": false,
# "authMethod": "none"} -- the desktop app authenticates its child over a host
# socket, not through anything a standalone CLI reads.  `claude auth status` exits 1
# when logged out, so it is a hard preflight here.
#
# --bare is NOT used even though the documentation recommends it for scripts: its
# own help says OAuth and keychain are never read, which would break subscription
# auth outright, and it also skips CLAUDE.md auto-discovery -- and CLAUDE.md is
# exactly the knowledge that makes "run the static droplet at gamma one hundred"
# mean anything.  EVERY ANTHROPIC_* VARIABLE is stripped from the child
# environment, as a family: in -p mode "the key is always used when present",
# with no approval step, which would silently move every request off the
# subscription.  Naming the members instead of the family was itself a hole --
# the list said ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL and missed
# ANTHROPIC_AUTH_TOKEN, which Claude Code reads with the same precedence over
# OAuth, so a user with it exported in a shell profile had every spoken command
# billed and routed through that credential with nothing in the log to show it.
# ANTHROPIC_BASE_URL is harmless when it points at api.anthropic.com, but it is
# an endpoint override with no role in credential selection, so the child is
# better off without it.
#
# THE CHILD ENVIRONMENT IS BUILT EXPLICITLY, because developing this from inside a
# Claude Code session is the normal case and that session exports CLAUDECODE=1,
# CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE_SESSION_ID, CLAUDE_CODE_MESSAGING_SOCKET,
# CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH and more.  A child that inherits those
# behaves differently from one launched by a plain Terminal, so every CLAUDE_* key
# is dropped.
#
# Continuity uses --resume with a captured session_id, never --continue: --continue
# resolves to "the most recent session in this directory", which is a race the
# moment the user also has Claude Code open in the same repo.
#
# Latency to expect: ~3.5 s of fixed overhead before the model is reached
# (--version 0.12 s, auth status 0.97 s, -p to first verdict 3.62 s).  stdin is
# DEVNULL because an open, never-closed stdin does not deadlock but costs 2.1 s
# (5.70 s against 3.62 s).
#
# ---------------------------------------------------------------------------
# SPEAKING IS SLOW AND THERE IS NO TRICK THAT FIXES IT
#
# `say` costs 3.5-13 s wall for "Ready." with user+sys only 0.6-1.1 s, i.e. it is
# waiting on the speech stack, not computing.  There is no persistent-say: `say -f
# /dev/stdin` and `-f /dev/fd/0` both give "Bad file descriptor" and `-f -`
# silently does nothing.  Pre-rendering with `say -o` and replaying with afplay is
# 4-5x faster (1.4-2.0 s against 6.9-7.1 s), but even afplay pays 0.9-1.9 s of
# CoreAudio output open on a 50 ms file, against a 0.015 s process-spawn floor.
# So the fixed phrases here are rendered once into a cache directory at startup and
# replayed, and anything dynamic pays for `say`.  A signal stops speech within
# 60-134 ms (SIGTERM 1.259 s, SIGINT 1.273 s, SIGKILL 1.334 s against a 1.200 s
# start), and poll() is a reliable "still speaking" gate, returning None throughout
# a 12 s utterance.
#
# THE SELF-TRIGGER GUARD IS TIME-BASED AND THAT IS A REAL LIMITATION.  The VAD is
# gated while the speaking process is alive plus SPEECH_TAIL_S afterwards for
# speaker decay, and the pre-roll buffer is cleared on ungating so the tail of the
# daemon's own sentence cannot become the head of the next utterance.  The failure
# mode is exactly the one that matters: it also deafens the assistant to a genuine
# "stop" spoken over its own speech, which is the one word a user most wants to
# interrupt with.  Acoustic self-trigger was never measured -- system output was
# muted during the recon (output volume 6, muted true) and the user's setting was
# not changed -- so no claim is made that this guard is sufficient with loud
# speakers.  If it proves insufficient, the fix is a separate output device
# (headphones), not a cleverer timer.
#
# THE ONE CASE WHERE THAT WOULD HAVE BEEN DANGEROUS IS NOW CLOSED BY SOMETHING
# ELSE.  The daemon speaks the release token out loud -- the confirmation
# question necessarily contains the word "confirm" -- so a room with enough
# reverberation past 450 ms, or a microphone that is not on the same body as the
# speakers (the Continuity iPhone, or AirPods worn while the MacBook speaker
# plays), could have fed the tail of that sentence back in during CONFIRM state
# and released a pending pkill with nobody in the room having said anything.
# The guard against that is NOT the timer: it is that a confirmation must be the
# WHOLE utterance and must carry the wake word, which the question does not
# satisfy ("That would kill ... Say jarvis confirm if you mean it." fullmatches
# nothing).  That case is pinned in confirm_selftest().  The general acoustic
# self-trigger limitation above still stands, unmeasured.
#
# ---------------------------------------------------------------------------
# WHAT THIS DOES NOT DO
#
#   * No speaker identification.  Anyone audible can drive it, including a podcast.
#   * No barge-in.  While claude is working, and while the daemon is speaking, it
#     is deaf; "stop" is only heard between utterances.  Cancelling a running build
#     means Ctrl-C in the terminal.
#   * No offline LLM.  Every dispatch is a network call through the claude CLI.
#     Only the speech recognition is local.
#   * No parameters parsed out of speech.  The registry has an entry for the
#     static droplet at gamma 100 and one at gamma 20; it has none that turns "at
#     gamma fifty" into `-gamma 50`, because that would put transcript text into
#     an argv.  The grid flag is not even the same letter across drivers (-n for
#     static_droplet and ehd_cavity, -ny for rb_high_ra, -h for rayleigh_benard,
#     -d for tgv3d), so a general parser would be a second grammar to keep in
#     step with every driver's parser.  Ask for an unlisted parameter and the
#     agent takes it -- it can read the parser.
#   * Nothing is backgrounded.  A registry argv runs to completion under a
#     wall-clock cap while the loop waits, so the daemon is deaf for the whole of
#     `run the tests`.  Backgrounding it would be worse, not better: a detached
#     solver on this Mac is QoS-throttled to ~14 % CPU -- validation/poiseuille
#     ran fifteen minutes in the background for work ctest records at 14.3 s.
#   * No GBNF grammar constraint.  whisper-cli has --grammar/--grammar-rule/
#     --grammar-penalty and whisper-server does NOT expose them, and per-utterance
#     whisper-cli would cost the 1.6-4.0 s cold start the resident server exists to
#     avoid.  Constraining happens after the fact, in normalise().
#   * No silero VAD.  ggml-silero-v5.1.2.bin (885,098 bytes) was never downloaded,
#     whisper-vad-speech-segments hard-fails without it, and its timestamps are in
#     CENTISECONDS, which is a 10x/100x bug waiting to happen.  The VAD here is
#     energy-based, in Python, and its constants are stated below.
#   * No dedicated wake-word model.  No Porcupine, no openWakeWord: both are pip
#     packages, and the stdlib-only rule is the whole reason this tree can be run
#     on a fresh machine.  The wake word is tiny.en plus difflib.
#   * No ElevenLabs, no Fish Audio.  Both need a network round trip, an API key and
#     a dependency, to replace a binary that is already installed.  `say` is worse
#     and it is here.
#   * No launchd/LaunchAgent install.  The microphone grant belongs to the APP that
#     owns the process tree, and a LaunchAgent has no app identity, so a daemon
#     started that way would hit the very digital-zero stream this file refuses to
#     run against.  Start it from a terminal you have granted.
#   * No transcript pruning.  Every -p run writes to ~/.claude/projects/<encoded
#     cwd>/<session-id>.jsonl, which on this 8 GB machine already holds a 260 MB and
#     a 121 MB file.  A chatty daemon will add to that and nothing here cleans it.
#     This daemon's OWN log is bounded (LOG_RETENTION_DAYS, LOG_MAX_BYTES) and
#     records a transcript only for segments the wake word actually hit --
#     everything else is metadata.  It still hears the whole room, and
#     --log-all-audio will write down what it heard.
#   * No microphone by default.  --device is required outside --text: an
#     avfoundation index is assignment order, and binding to whatever is at 0
#     today can mean binding to a conference device and putting a meeting on the
#     wake word.  A name matching a virtual/conference pattern must be given
#     exactly.
#   * No adopting a stranger's whisper-server.  --adopt-whisper is opt-in; by
#     default a busy port is worked around with an ephemeral one.
#   * No multi-user, no multi-directory.  The working directory is pinned to the
#     repo at startup; `-p` skips the workspace-trust dialog and silently ignores
#     malformed settings files, so pointing this at an arbitrary directory would run
#     that directory's hooks and MCP servers with no prompt.
#   * No cost accounting across formats.  total_cost_usd is per-call under
#     --output-format json and CUMULATIVE under stream-json; this file only ever
#     reads the json form, and sums it, so the two are never mixed.

from __future__ import annotations

import argparse
import array
import datetime
import difflib
import io
import json
import os
import pathlib
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque

# miccheck.py holds the liveness test and the device enumeration, with the
# argument for both.  Imported, not copied.  A missing miccheck must not break
# --text mode, which is the path that works with no microphone at all, so the
# failure is deferred to the audio self-check.
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import miccheck as _mic
except Exception:                                    # pragma: no cover
    _mic = None

# --------------------------------------------------------------------------
# Audio constants.  16 kHz mono s16le throughout: it is what whisper.cpp wants,
# what `say -o --data-format=LEI16@16000` emits, and what ffmpeg is asked for, so
# no resampling happens anywhere in this file.
RATE = 16000
WIDTH = 2
FRAME_MS = 32                                        # 512 samples
FRAME_BYTES = RATE * FRAME_MS // 1000 * WIDTH        # 1024
CHUNK_BYTES = 3200                                   # 100 ms per pipe read

# 512 ms of pre-roll, mandated by the first syllable: without it "jarvis" arrives
# as "arvis" (ratio 0.909 -- it still matches, but a two-word command loses its
# verb, which does not).
#
# THE DEQUE IS DEEPER THAN THE LEAD-IN IT DELIVERS, AND FOR A MEASURED REASON.
# Frames are appended to the deque BEFORE they are tested, so when the onset
# counter reaches ONSET_FRAMES the deque already ends with those ONSET_FRAMES
# speech frames: a 16-frame deque delivered 13 frames = 416 ms of true lead-in,
# not the 512 ms this comment used to claim.  Measured by tagging frames and
# locating them in the emitted segment.  An 18.75 % shortfall against a figure
# the banner calls load-bearing is itself the defect, so the depth is now
# LEAD_IN + ONSET: 19 frames = 608 ms buffered, 512 ms of it ahead of the first
# speech frame.
PREROLL_LEAD_FRAMES = 16                             # 512 ms of TRUE lead-in
ONSET_FRAMES = 3                                     # 96 ms; rejects key clicks
PREROLL_FRAMES = PREROLL_LEAD_FRAMES + ONSET_FRAMES  # 19 frames = 608 ms held

# VAD.  A fixed RMS threshold fails in a different room, so the floor is tracked
# and the threshold is relative to it.
#   SPEECH_FACTOR 3.0 is +9.5 dB over the floor.  Measured room floor is 7-15
#   counts peak; speech in the reference clips is rms 4047-4656, i.e. two and a
#   half orders of magnitude above, so the factor is not a close call.
#   ABS_MIN_RMS 60 counts (0.0018 FS) stops a pathologically quiet floor from
#   making the relative threshold meaningless; it sits an order of magnitude below
#   speech and an order above the room.
SPEECH_FACTOR = 3.0
ABS_MIN_RMS = 60.0
MIN_FLOOR_RMS = 4.0
# The floor rises slowly and falls quickly, so that speech cannot drag it up
# (which would deafen the daemon) while a fan switching off is tracked at once.
# Per 32 ms frame: tau 3.0 s up, tau 0.3 s down.
FLOOR_A_UP = FRAME_MS / 3000.0
FLOOR_A_DOWN = FRAME_MS / 300.0

# ONSET_FRAMES is defined above, with the pre-roll it sizes.
HANGOVER_FRAMES = 22                                 # 704 ms of silence ends it
# MIN_SEG_MS IS A COUNT OF VOICED FRAMES, NOT A SEGMENT LENGTH, AND MEASURING IT
# THE OTHER WAY MADE IT UNREACHABLE.  Every emitted segment carries the
# ONSET_FRAMES that opened it plus the HANGOVER_FRAMES that closed it, so the
# SHORTEST possible hangover-terminated segment is (3+22) x 32 ms = 800 ms
# against a 320 ms threshold: the test `dur_s(seg)*1000 >= MIN_SEG_MS` was
# always true and had never rejected anything.  Measured on a warmed VAD (floor
# 5.4, threshold 60.0), a 96 ms burst -- exactly the key click this constant
# exists to reject -- emitted a 1216 ms segment and was POSTed to whisper.  The
# VAD now counts the frames that were actually above the threshold.
MIN_SEG_MS = 320                                     # of VOICED audio; less is a cough
MAX_SEG_WAKE_S = 8.0
MAX_SEG_CMD_S = 20.0

WARMUP_S = 0.30          # discarded AFTER the first byte: device-open transient
LIVENESS_S = 1.30        # analysed for digital silence, on our own stream
RATE_WARN = 0.95         # below this the stream is gapped; warn
RATE_REFUSE = 0.80       # below this, refuse the command tier out loud
RATE_WINDOW_S = 10.0

# A STREAM THAT GOES DIGITAL-ZERO MID-SESSION IS THE STARTUP FAILURE ARRIVING
# LATE, AND UNTIL NOW ONLY THE STARTUP ONE WAS CAUGHT.  analyse()/classify() ran
# exactly once, on the first LIVENESS_S of the stream; after that the daemon
# would happily transcribe zeroes for the rest of its life.  That is not
# hypothetical on this machine: the built-in "MacBook Air Microphone" read DEAD
# once and LIVE four times inside seventy minutes, and conferencing processes
# that can seize the device were running throughout.  It is also exactly the
# failure this tree keeps naming -- plausible, quiet, and undebuggable after the
# fact, because a dead mic and an idle user produce the SAME log.
#
# The detector is a byte compare, not a re-probe: re-opening the device would
# cost 1.55-3.71 s and fight the capture ffmpeg for it.  chunk.strip(b"\x00")
# is a C-level scan of audio already in hand, it consumes nothing the VAD needs,
# and a PARTIALLY zero chunk RESETS the run -- so a quiet room (measured
# zero_frac 0.0528, i.e. zeroes constantly interrupted) can never trip it, and
# only a true all-zero stream accumulates.  The threshold is sustained rather
# than instantaneous because an AirPods handover blanks the stream briefly and
# that is a recovery, not a fault.
MIC_WARN_S = 12.0        # all-zero for this long: say so once, keep running
MIC_DEAD_S = 45.0        # all-zero for this long: the device is gone, stop

SPEECH_TAIL_S = 0.45     # speaker decay after `say` exits, before the VAD reopens
ARM_TIMEOUT_S = 8.0      # how long "jarvis" leaves the door open
# THE CONFIRMATION CLOCK USED TO START BEFORE THE QUESTION WAS SPOKEN, AND THE
# QUESTION IS LONGER THAN THE CLOCK WAS.  Measured: the rendered audio for the
# stop_the_run question at `say -v Daniel -r 190` is 5.119 s, and this file's own
# banner measures live `say` wall cost at 3.5-13 s for the single word "Ready."
# Add SPEECH_TAIL_S of VAD gating, HANGOVER_FRAMES = 704 ms to close the user's
# segment and ~1.11 s to transcribe it, and a 10 s window could expire before the
# user heard the word "confirm" -- silently, because the expiry only flipped the
# state back to IDLE.  The clock now starts when the question has FINISHED being
# spoken, the window is 20 s, and the expiry is said out loud.  A safety control
# that never works is one the user switches off, and the available switch here is
# --allow-edits.
CONFIRM_TIMEOUT_S = 20.0
SESSION_IDLE_S = 1800.0  # a gap this long starts a fresh claude session

# Log hygiene.  A wake-word daemon hears the whole room, so the log is a
# transcript of the room unless something stops it being one: see
# handle_segment(), which logs a non-wake segment as METADATA ONLY.  These two
# bound the file itself.  14 days and 32 MB are a stated retention, not a
# measurement -- nothing here needs a longer history than the last fortnight of
# wake-cutoff tuning, and 32 MB is about a month of chatty use.
LOG_RETENTION_DAYS = 14
LOG_MAX_BYTES = 32 * 1024 * 1024

WAKE_CUTOFF = 0.75       # see the banner: the measured gap is 0.667 to 0.833
WAKE_LOG_FLOOR = 0.55    # rejected near-misses above this are logged

SERVER_READY_S = 180.0   # first-ever ggml run compiles Metal shaders for ~115 s
CLAUDE_TIMEOUT_S = 300.0

# Names that mean "this input is a conference call or another application's
# output", not "this input is the room".  Enumerated on this machine:
# [3] Microsoft Teams Audio.  See start_audio().
VIRTUAL_DEVICE_RE = re.compile(
    r"teams|zoom|webex|skype|blackhole|loopback|soundflower|aggregate|"
    r"multi-?output|virtual|vb-?audio|obs", re.I)

MODEL_DIR = pathlib.Path.home() / ".cache" / "whisper-models"
CHIME = "/System/Library/Sounds/Tink.aiff"           # 0.564 s, shortest system chime

# The two data files.  Nothing below this line hard-codes a spoken form, a
# canonical form, a phrase, a destructive pattern or a command: they are loaded
# from vocabulary.json and commands.json, which carry their own arguments.  See
# THE DATA LIVES IN TWO JSON FILES in the banner.
VOCAB_PATH = _HERE / "vocabulary.json"
COMMANDS_PATH = _HERE / "commands.json"

# Filled by load_vocabulary() and load_commands(), both called from main()
# BEFORE the daemon is constructed.  They are module globals rather than daemon
# attributes so that normalise() and destructive_label() keep working as plain
# functions -- they are called from the tests and from --selftest, where there is
# no daemon.
WHISPER_PROMPT = ""
SPOKEN_STYLE = ""
HALLUCINATIONS: list[str] = []
PHRASES: dict[str, str] = {}
_ALIAS_RULES: list[tuple[re.Pattern, str]] = []
REQUIRED_PHRASES = ("ready", "working", "cancelled", "gapped", "mic_dead",
                    "confirm_expired", "not_confirmed", "confirm_gapped")
DESTRUCTIVE: list[tuple[re.Pattern, str]] = []
REFUSALS: list[tuple[re.Pattern, str]] = []
REGISTRY: list["Command"] = []

# Reading, building, testing and running.  Nothing that writes; git in its read
# verbs only.  --allow-edits adds EDIT_TOOLS.  See defence 3 in the banner.
#
# A PREFIX GLOB ON A MULTI-TOOL BINARY IS NOT AN ALLOWLIST, AND `Bash(cmake *)`
# WAS A FULL SHELL ESCAPE.  `cmake -E help` on this machine (cmake 4.4.3,
# checked) lists, among others:
#     cmake -E rm [-rRf] [--] <file/dir>...      recursive delete
#     cmake -E env [--] <command> [<arg>...]     run ANY binary with ANY argv
#     cmake -P <script>.cmake                    CMake script, execute_process()
# Every one of those command lines begins with `cmake `, so every one of them sat
# inside the glob.  `Bash(ctest *)` was the same shape: `ctest -S script.cmake`
# and `ctest --build-and-test ... --test-command <any command>`.  The README's
# claim that "the agent cannot write files by default" was therefore false: it
# could not use the Edit tool, and it could delete the tree.  The blast radius on
# this repository is total, because the only uncommitted work in it is voice/
# itself and `git restore` is NOT allowlisted -- tracked files were better
# protected than this assistant's own source.
#
# So the allowlist now names the exact FORMS the registry needs, and there is a
# --disallowedTools backstop underneath it (below), which is kept even under
# --allow-edits.  Belt and braces on purpose: an allowlist is a claim about what
# the child WILL be asked for, and the denylist is a claim about what it must not
# do whatever it is asked.
READ_TOOLS = (
    "Read,Glob,Grep,"
    "Bash(cmake --build *),Bash(cmake -S *),"
    "Bash(ctest --output-on-failure*),Bash(ctest -R *),"
    "Bash(ctest --rerun-failed*),Bash(ctest -N*),"
    "Bash(./build/*),"
    "Bash(git status*),Bash(git log*),Bash(git diff*),Bash(git show*),"
    "Bash(ls *),Bash(uptime),Bash(sh tests/frame_check.sh)"
)
EDIT_TOOLS = "Edit,Write"

# The backstop.  Write/Edit are dropped from it (and only they) when
# --allow-edits is passed; the cmake/ctest escapes are denied either way,
# because "let it edit a source file" is not "let it run an arbitrary binary".
DENY_ALWAYS = (
    "NotebookEdit,WebFetch,WebSearch,"
    "Bash(cmake -E *),Bash(cmake -P *),"
    "Bash(ctest -S *),Bash(ctest --build-and-test *)"
)
DENY_TOOLS = "Write,Edit," + DENY_ALWAYS


# ==========================================================================
# Small utilities
# ==========================================================================

def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


# EVERY LIVE CHILD, SO THAT A SIGNAL CAN REACH THEM.  start_new_session puts each
# child in its own process group, which is what makes a terminal Ctrl-C NOT reach
# them -- deliberate, so that they are torn down in a defined order rather than
# racing the shell.  The cost is that a handler which only sets an Event does not
# shorten anything: measured, a flag-only SIGINT handler fired at t = 1.01 s and
# `run_capped(["sleep","6"], cap=20.0)` still did not return until t = 6.02 s,
# because Popen.communicate(timeout=...) retries across EINTR (PEP 475).  The
# same holds for urlopen(timeout=...) against whisper and for wait_ready's sleep.
# So Ctrl-C during a claude turn was ignored for up to --claude-timeout (300 s)
# WHILE FFMPEG STILL HELD THE MICROPHONE.  The handler now reaps this registry,
# and every blocking call returns at once because its child is gone.
_CHILDREN: "set[subprocess.Popen]" = set()
_CHILDREN_LOCK = threading.Lock()


def spawn(argv: list[str], **kw) -> subprocess.Popen:
    """Popen with a LIST argv, in its own process group.

    start_new_session is the load-bearing part: killing only the direct child
    leaves node and bash GRANDCHILDREN alive holding the pipe.  It also stops a
    terminal Ctrl-C reaching the children directly, so that this file reaps them
    in a defined order instead of racing the shell.
    """
    kw.setdefault("stdin", subprocess.DEVNULL)
    kw.setdefault("start_new_session", True)
    p = subprocess.Popen(argv, **kw)
    with _CHILDREN_LOCK:
        for q in [c for c in _CHILDREN if c.poll() is not None]:
            _CHILDREN.discard(q)
        _CHILDREN.add(p)
    return p


def reap_all(grace: float = 1.0) -> int:
    """SIGTERM/SIGKILL every child this process still has.  Signal-handler safe.

    Called from on_signal, where the point is not tidiness but LATENCY: killing
    the child is what makes the main thread's blocking communicate() or urlopen()
    return, and what takes ffmpeg off the microphone within a few hundred
    milliseconds instead of at the end of a 300 s agent turn.
    """
    with _CHILDREN_LOCK:
        live = [p for p in _CHILDREN if p.poll() is None]
        _CHILDREN.clear()
    for p in live:
        reap(p, grace=grace)
    return len(live)


def reap(p: subprocess.Popen | None, grace: float = 3.0) -> None:
    """SIGTERM the whole group, then SIGKILL what is left.

    SIGTERM first is not politeness: Claude Code exits 143 on it, tears down the
    process tree of a running Bash command and runs SessionEnd hooks, and the
    unfinished turn stays resumable by session id.  A measured `say` stops within
    60-134 ms of any signal.
    """
    if p is None or p.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_capped(argv: list[str], cap: float, cwd: str | None = None,
               env: dict | None = None) -> tuple[int | None, str, str, bool]:
    """Run to completion or kill the process GROUP.  Replaces GNU `timeout`.

    There is no `timeout` or `gtimeout` on this machine, so every documented
    `timeout 10 whisper-...` line fails with command-not-found.  This is the
    twelve-line stdlib replacement, verified to kill grandchildren.

    A FAILED SPAWN IS A RESULT, NOT AN EXCEPTION.  Popen raises OSError when
    argv[0] does not exist or when cwd does not, and that killed the whole daemon
    with a traceback in the middle of a conversation.  It is not a corner case
    here: `build/` is gitignored, so a fresh clone raises FileNotFoundError on
    `<repo>/build` for the very first spoken "run the fast tests", and every
    registry entry whose argv[0] is an unbuilt ./build/validation/... binary does
    the same.  --selftest only WARNS about a missing executable, so nothing
    upstream stops it.  rc=None with the message in stderr is the shape
    verdict() and dispatch() already know how to speak.
    """
    try:
        p = spawn(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                  text=True, cwd=cwd, env=env)
    except OSError as exc:
        return None, "", str(exc), False
    try:
        out, err = p.communicate(timeout=cap)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        reap(p)
        try:
            out, err = p.communicate(timeout=2)
        except Exception:
            out, err = "", ""
        return p.returncode, out, err, True


def wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw s16le in a RIFF container, in memory, with the stdlib.

    Mandatory: raw PCM POSTed to whisper-server returns HTTP 400 "Invalid
    request".  The stdlib header was checked against ffmpeg's and gives a
    byte-identical transcript.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(WIDTH)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def frame_rms(frame: bytes) -> float:
    """RMS of one s16le frame.  No audioop -- it was removed in Python 3.13."""
    a = array.array("h")
    a.frombytes(frame[: len(frame) // 2 * 2])
    n = len(a)
    if n == 0:
        return 0.0
    return (sum(v * v for v in a) / n) ** 0.5


def dur_s(pcm: bytes) -> float:
    return len(pcm) / WIDTH / RATE


# ==========================================================================
# The data files
#
# vocabulary.json turns sound into canonical text and decides nothing;
# commands.json decides.  Both carry their own banners, which is where the
# argument for each entry lives -- this module loads and validates them and
# holds no table of its own.
#
# Every failure here is fatal at STARTUP and names the offending entry.  A voice
# daemon that silently skipped a malformed command would be indistinguishable
# from one that mis-heard the user, which is the failure mode the whole design is
# built to avoid.
# ==========================================================================

class DataError(Exception):
    """A data file is missing, malformed, or says something impossible."""


def _phrase_re(phrase: str) -> re.Pattern:
    """Word-boundary regex for a spoken phrase, tolerant of commas and hyphens.

    whisper punctuates freely ("D3Q27, K8, gamma"), so the separator between
    words is [\\s,\\-]+ rather than a literal space.
    """
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"(?<!\w)" + r"[\s,\-]+".join(words) + r"(?!\w)", re.I)


def _read_json(path: pathlib.Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            v = json.load(fh)
    except FileNotFoundError:
        raise DataError(f"{path} does not exist") from None
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(v, dict):
        raise DataError(f"{path} must hold a JSON object, not {type(v).__name__}")
    return v


def _compile(pattern: str, where: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.I)
    except re.error as exc:
        raise DataError(f"{where}: bad regex {pattern!r}: {exc}") from None


# The ONLY tokens an argv element may carry.  {text} is additionally allowed in a
# prompt, because a prompt is a question being asked; it is NOT allowed in an
# argv, which is the whole safety argument -- see commands.json's banner.
_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(s: str, values: dict[str, str], where: str) -> str:
    """Substitute {repo}/{threads}/{text}, and REFUSE any other token.

    Refusing rather than passing it through is deliberate: a typo like {repos}
    would otherwise reach a command line as a literal brace and fail somewhere
    far away from the file that caused it.
    """
    bad = [m.group(1) for m in _TOKEN_RE.finditer(s) if m.group(1) not in values]
    if bad:
        raise DataError(f"{where}: unknown token(s) {bad} in {s!r}; "
                        f"only {sorted(values)} are substituted")
    return _TOKEN_RE.sub(lambda m: values[m.group(1)], s)


class Command:
    """One registry entry: a matcher plus EITHER an argv or a prompt template.

    argv entries run a fixed, closed command with no transcript text in it.
    prompt entries hand a template to the agent with {text} filled in.
    """

    __slots__ = ("id", "patterns", "argv", "prompt", "cwd", "timeout_s",
                 "slow", "say", "confirm", "confirm_label", "empty_say",
                 "examples", "counterexamples", "rc_say")

    def __init__(self, raw: dict, values: dict[str, str], index: int):
        where = f"commands[{index}]"
        self.id = raw.get("id")
        if not isinstance(self.id, str) or not self.id:
            raise DataError(f"{where}: missing a string 'id'")
        where = f"command {self.id!r}"

        pats = raw.get("patterns")
        if not isinstance(pats, list) or not pats:
            raise DataError(f"{where}: 'patterns' must be a non-empty list")
        self.patterns = [_compile(p, where) for p in pats]

        has_argv, has_prompt = "argv" in raw, "prompt" in raw
        if has_argv == has_prompt:
            raise DataError(f"{where}: give exactly one of 'argv' or 'prompt', "
                            f"not {'both' if has_argv else 'neither'}")

        if has_argv:
            av = raw["argv"]
            if not isinstance(av, list) or not av or \
                    not all(isinstance(a, str) for a in av):
                raise DataError(f"{where}: 'argv' must be a non-empty list of "
                                f"strings")
            # {text} is NOT in the substitution set for an argv.  A transcript
            # must never become a command-line argument.
            argv_values = {k: v for k, v in values.items() if k != "text"}
            self.argv = [_expand(a, argv_values, where) for a in av]
            self.prompt = None
        else:
            if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
                raise DataError(f"{where}: 'prompt' must be a non-empty string")
            self.prompt = raw["prompt"]
            self.argv = None
            # Validate the tokens now rather than at the moment of dispatch.
            _expand(self.prompt, values, where)

        cwd = raw.get("cwd")
        self.cwd = _expand(cwd, values, where) if isinstance(cwd, str) else None
        self.timeout_s = float(raw.get("timeout_s", 300.0))
        self.slow = bool(raw.get("slow", False))
        self.confirm = bool(raw.get("confirm", False))
        # Read back to the user as the question they are answering, so it must
        # describe the CONSEQUENCE.  The fallback names the entry, which is
        # honest but clumsy; every confirming entry should supply its own.
        self.confirm_label = raw.get("confirm_label") or \
            f"run {self.id}, which is destructive"
        if not self.confirm and raw.get("confirm_label"):
            raise DataError(f"{where}: has a 'confirm_label' but confirm is "
                            f"false, so it would never be spoken")
        self.say = raw.get("say")
        self.empty_say = raw.get("empty_say")
        self.examples = list(raw.get("examples") or [])

        # NEGATIVE EXAMPLES, because --selftest could only see an entry failing
        # to catch its OWN phrasing and never an entry catching somebody
        # else's.  Two live mis-routes were found by probing gate() by hand:
        # "blow away the build tree" matched the build entry's noun pattern and
        # silently ran cmake --build, and "scrap the uncommitted work" matched
        # git_status's \buncommitted\b.  Both were harmless; the mechanism is
        # not, and it is the same class as the "stop the run" shadowing the
        # banner already records.  Each string here must NOT reach this entry.
        self.counterexamples = list(raw.get("counterexamples") or [])
        for x in self.counterexamples:
            if not isinstance(x, str) or not x.strip():
                raise DataError(f"{where}: a counterexample is empty")

        # Exit code -> the sentence to speak, for the entries where a non-zero
        # code is a real answer rather than a failure.  Keys are strings
        # because JSON object keys are.
        rs = raw.get("rc_say") or {}
        if not isinstance(rs, dict) or not all(
                isinstance(k, str) and isinstance(v, str) and v.strip()
                for k, v in rs.items()):
            raise DataError(f"{where}: 'rc_say' must map an exit code string "
                            f"to a non-empty sentence")
        for k in rs:
            if not re.fullmatch(r"-?\d+", k):
                raise DataError(f"{where}: rc_say key {k!r} is not an exit code")
        self.rc_say = dict(rs)

    @property
    def kind(self) -> str:
        return "argv" if self.argv is not None else "prompt"


def load_vocabulary(path: pathlib.Path) -> None:
    """Populate WHISPER_PROMPT and the alias table.

    Aliases are sorted LONGEST FIRST so that "dee three queue twenty seven"
    cannot be eaten by a shorter rule.  File order is therefore irrelevant, which
    is deliberate: an alias table that depended on its own ordering would be a
    trap for whoever adds the next spoken form.
    """
    global WHISPER_PROMPT, _ALIAS_RULES, HALLUCINATIONS
    doc = _read_json(path)

    # OBSERVED SILENCE HALLUCINATIONS.  Optional, and belt-and-braces to the
    # acoustic gate in handle_segment() -- see that function.  Kept in the data
    # file rather than here because the list grows from voice/logs/*.jsonl.
    hall = doc.get("silence_hallucinations") or []
    if not isinstance(hall, list) or not all(isinstance(h, str) and h.strip()
                                             for h in hall):
        raise DataError(f"{path}: 'silence_hallucinations' must be a list of "
                        f"non-empty strings")
    HALLUCINATIONS = [" ".join(h.lower().split()) for h in hall]

    prompt = doc.get("whisper_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise DataError(f"{path}: 'whisper_prompt' must be a non-empty string")
    WHISPER_PROMPT = prompt.strip()

    aliases = doc.get("aliases")
    if not isinstance(aliases, list) or not aliases:
        raise DataError(f"{path}: 'aliases' must be a non-empty list")
    rules: list[tuple[re.Pattern, str]] = []
    for i, entry in enumerate(aliases):
        where = f"{path} aliases[{i}]"
        if not isinstance(entry, dict):
            raise DataError(f"{where}: must be an object")
        canon = entry.get("canonical")
        forms = entry.get("spoken")
        if not isinstance(canon, str) or not canon:
            raise DataError(f"{where}: missing a string 'canonical'")
        if not isinstance(forms, list) or not forms:
            raise DataError(f"{where} ({canon}): 'spoken' must be a non-empty "
                            f"list")
        for f in forms:
            if not isinstance(f, str) or not f.strip():
                raise DataError(f"{where} ({canon}): a spoken form is empty")
            rules.append((_phrase_re(f), canon))
    _ALIAS_RULES = sorted(rules, key=lambda r: -len(r[0].pattern))


def load_commands(path: pathlib.Path, repo: pathlib.Path, threads: int) -> None:
    """Populate SPOKEN_STYLE, PHRASES, DESTRUCTIVE, REFUSALS and REGISTRY."""
    global SPOKEN_STYLE, PHRASES, DESTRUCTIVE, REFUSALS, REGISTRY
    doc = _read_json(path)

    style = doc.get("spoken_style")
    if not isinstance(style, str) or not style.strip():
        raise DataError(f"{path}: 'spoken_style' must be a non-empty string")

    phrases = doc.get("phrases")
    if not isinstance(phrases, dict) or not phrases:
        raise DataError(f"{path}: 'phrases' must be a non-empty object")
    for k, v in phrases.items():
        if not isinstance(v, str) or not v.strip():
            raise DataError(f"{path}: phrase {k!r} is not a non-empty string")
    # Every key this module looks up by name.  Checked at STARTUP rather than
    # discovered as a KeyError in the middle of a conversation, which is where
    # a missing phrase used to surface.
    missing = [k for k in REQUIRED_PHRASES if k not in phrases]
    if missing:
        raise DataError(f"{path}: 'phrases' is missing {missing}, which "
                        f"jarvis.py looks up by name")

    dest: list[tuple[re.Pattern, str]] = []
    for i, e in enumerate(doc.get("destructive") or []):
        where = f"{path} destructive[{i}]"
        if not isinstance(e, dict) or not isinstance(e.get("pattern"), str) \
                or not isinstance(e.get("label"), str):
            raise DataError(f"{where}: needs string 'pattern' and 'label'")
        dest.append((_compile(e["pattern"], where), e["label"]))

    refuse: list[tuple[re.Pattern, str]] = []
    for i, e in enumerate(doc.get("refuse") or []):
        where = f"{path} refuse[{i}]"
        if not isinstance(e, dict) or not isinstance(e.get("pattern"), str):
            raise DataError(f"{where}: needs a string 'pattern'")
        key = e.get("phrase")
        if key not in phrases:
            raise DataError(f"{where}: phrase {key!r} is not in 'phrases'")
        refuse.append((_compile(e["pattern"], where), key))

    # {uid} exists for pkill -U: a pattern match over process command lines has
    # to be bounded to this user, and commands.json cannot run `id -u`.
    values = {"repo": str(repo), "threads": str(threads),
              "uid": str(os.getuid()), "text": "{text}"}
    cmds: list[Command] = []
    seen: set[str] = set()
    for i, raw in enumerate(doc.get("commands") or []):
        if not isinstance(raw, dict):
            raise DataError(f"{path} commands[{i}]: must be an object")
        c = Command(raw, values, i)
        if c.id in seen:
            raise DataError(f"{path}: duplicate command id {c.id!r}")
        seen.add(c.id)
        cmds.append(c)

    SPOKEN_STYLE = style.strip()
    PHRASES = dict(phrases)
    DESTRUCTIVE = dest
    REFUSALS = refuse
    REGISTRY = cmds


def load_data(vocab: pathlib.Path, commands: pathlib.Path,
              repo: pathlib.Path, threads: int) -> None:
    load_vocabulary(vocab)
    load_commands(commands, repo, threads)


def match_command(text: str) -> "Command | None":
    """First match in FILE ORDER wins.

    Order is load-bearing -- "run the fast tests" must be tried before "run the
    tests" -- and is checked rather than trusted: --selftest asserts that every
    entry's own examples route to that entry and not to an earlier one.
    """
    for c in REGISTRY:
        for rx in c.patterns:
            if rx.search(text):
                return c
    return None


def gate(text: str) -> tuple[str, object]:
    """Decide what an utterance DOES, before anything acts on it.

    Returns ("cancel", None), ("refuse", phrase_key), ("confirm", (label, cmd))
    or ("route", cmd_or_None).

    This exists as one function because the ordering inside it is subtle and was
    wrong once.  THE CANCEL RULE IS TRIED ONLY WHEN NOTHING IN THE REGISTRY
    MATCHES.  "stop" on its own must cancel, and it matches no entry; "stop the
    run" is a real command and is three words, so the original rule -- a cancel
    word within three words -- swallowed it and the pkill entry was unreachable.
    The self-test did not see that, because it re-implemented this ordering
    instead of calling it.  Both call this now, which is the whole point:
    an ordering that is only described in two places will differ in two places.
    """
    cmd = match_command(text)
    if cmd is None and CANCEL_RE.search(text) and len(text.split()) <= 3:
        return "cancel", None

    key = refusal_phrase(text)
    if key is not None:
        return "refuse", key

    # Two independent reasons to require confirmation, and neither subsumes the
    # other.  An entry is marked `confirm` because of what it DOES; any
    # utterance can trip the destructive table because of what it SAYS.  The
    # entry's own label wins so the user hears the consequence of the command
    # that actually matched, not of the word that happened to match.
    label = (cmd.confirm_label if cmd is not None and cmd.confirm
             else destructive_label(text))
    if label is not None:
        return "confirm", (label, cmd)
    return "route", cmd


# ==========================================================================
# Transcript normalisation, wake matching, safety gate
# ==========================================================================

def normalise(text: str) -> str:
    """Spoken form -> the form the repo actually accepts.

    Applied AFTER transcription and never inside the whisper prompt: putting the
    spoken forms in the prompt made whisper emit them verbatim ("dee two queue
    nine" instead of "D2Q9"), measured on identical audio with nothing else
    changed.
    """
    out = " ".join(text.split())
    for rx, canon in _ALIAS_RULES:
        out = rx.sub(canon, out)
    return out.strip()


_WORD_RE = re.compile(r"[a-z0-9']+")

# whisper's non-speech annotations: [BLANK_AUDIO], [MUSIC PLAYING], [SILENCE],
# *laughs*.  Matched by form rather than by an enumerated list, because the list
# is not knowable in advance -- see the note in WhisperTier.transcribe().
_ANNOTATION_RE = re.compile(r"\[[^\]]{0,60}\]|\*[^*]{0,60}\*")


def wake_match(text: str, wake: str, cutoff: float = WAKE_CUTOFF
               ) -> tuple[bool, str, float, list[tuple[str, float]]]:
    """Fuzzy wake detection.  Returns (hit, remainder, best_ratio, near_misses).

    Trailing punctuation is stripped first because whisper reliably emits
    "jarvis," with a comma.  "hey jarvis" and "ok jarvis" are accepted by
    matching each token independently rather than by pattern, so any filler in
    front of the wake word works.  The remainder is everything after the matched
    token, which is what makes a one-shot "jarvis, run the fast tests" possible.
    """
    toks = _WORD_RE.findall(text.lower())
    best, best_i, misses = 0.0, -1, []
    for i, t in enumerate(toks):
        r = difflib.SequenceMatcher(None, wake, t).ratio()
        if r > best:
            best, best_i = r, i
        if cutoff > r >= WAKE_LOG_FLOOR:
            misses.append((t, round(r, 3)))
    if best >= cutoff and best_i >= 0:
        rest = " ".join(toks[best_i + 1:])
        return True, rest, best, misses
    return False, "", best, misses


CANCEL_RE = re.compile(
    r"(?<!\w)(stop|cancel|never ?mind|forget it|abort|belay that"
    r"|no|nope|negative|don'?t|do not)(?!\w)", re.I)

# THE RELEASE WORD MUST BE THE WHOLE UTTERANCE, NOT A SUBSTRING OF ONE.  The old
# test was `CONFIRM_RE.search(text) and not CANCEL_RE.search(text)` over
# `(confirm|confirmed)`, and run against real sentences it approves the opposite
# of what was said: "no, don't confirm that" APPROVES, "I cannot confirm that"
# APPROVES, "can you confirm the numbers for Friday" APPROVES, "confirm the
# meeting time with Sarah" APPROVES.  Negation was not handled at all and "no"
# was not a cancel word.  Meanwhile "yes", "go ahead" and "do it" all DROPPED --
# so the words a user reaches for failed and the words a Teams call produces
# succeeded.  Combined with a CONFIRM branch that required no wake word and a
# file that ships no speaker identification, anyone audible could release a
# pending pkill or agent dispatch inside the window.
#
# fullmatch, plus the wake word on the same segment (see resolve_confirmation),
# is also what makes the daemon's own question safe to speak out loud: it
# necessarily contains the token "confirm", and this file's banner is honest
# that the 450 ms self-trigger guard was never measured with the speakers on.
# "That would kill ... Say jarvis confirm if you mean it." cannot fullmatch.
CONFIRM_RE = re.compile(r"(ok(ay)?[\s,]+)?(confirm|confirmed)[.!]?", re.I)


def is_confirmation(text: str) -> bool:
    return bool(CONFIRM_RE.fullmatch(" ".join(text.split()))) \
        and not CANCEL_RE.search(text)


def is_hallucination(text: str) -> tuple[bool, float, str]:
    """Is this one of the sentences whisper invents over silence?

    tiny.en does not return nothing for room noise; it returns a fixed English
    sentence, reproducibly.  Measured on 8 s of genuine room silence (mean
    -46.8 dB, max -31.0 dB) POSTed to tiny.en: 3/3 runs with the bias prompt and
    1/1 without it returned " This is a deliberately long-sentence so that the
    rendering takes a measurable amount of time."  In the first live daemon run
    a variant of it arrived as "Jarvis has a letter along the sentence so that
    the rendering takes a measurable amount of time..." -- wake_match scored
    that 1.0, the command tier confirmed a similar hallucination, and the
    daemon built a full claude argv out of pure noise.

    _ANNOTATION_RE cannot see this: it strips [BLANK_AUDIO] and *laughs* by
    FORM, and free prose has no form to match.  The primary defence is acoustic
    (handle_segment requires voiced frames); this is the cheap second one, and
    it is a data file so that the list grows from voice/logs/*.jsonl.
    """
    t = " ".join(text.lower().split())
    if not t:
        return False, 0.0, ""
    for h in HALLUCINATIONS:
        r = difflib.SequenceMatcher(None, h, t).ratio()
        if r >= 0.9:
            return True, round(r, 3), h[:60]
    return False, 0.0, ""


def destructive_label(text: str) -> str | None:
    """The spoken CONSEQUENCE of a destructive intent, or None.

    The table is commands.json's `destructive`.  Each label describes what would
    happen, not what matched, because it is read back to the user as the question
    they are answering.
    """
    for rx, label in DESTRUCTIVE:
        if rx.search(text):
            return label
    return None


def refusal_phrase(text: str) -> str | None:
    """The phrase KEY for a stale request that must not be run at all.

    Today this is only D3Q19, removed from the tree on 2026-09-18, which is now a
    hard error printing NOTHING WAS RUN.  Every saved command line from before
    that date carries it, so a spoken one will too, and the difference between
    the user hearing "done" and hearing why nothing ran is this check.
    """
    for rx, key in REFUSALS:
        if rx.search(text):
            return key
    return None


# ==========================================================================
# Output: chime and speech
# ==========================================================================

class Mouth:
    """Everything that makes noise, and the gate that stops it being heard.

    One process at a time, tracked so the VAD can be closed while it runs.  Fixed
    phrases are pre-rendered with `say -o` at startup and replayed with afplay
    (1.4-2.0 s against 6.9-7.1 s for live say); anything dynamic pays for say.
    """

    def __init__(self, voice: str | None, rate: int, enabled: bool,
                 cache: pathlib.Path, log):
        self.voice, self.rate, self.enabled = voice, rate, enabled
        self.cache, self.log = cache, log
        self.proc: subprocess.Popen | None = None
        self.quiet_until = 0.0
        self._lock = threading.Lock()

    # -- gating -----------------------------------------------------------
    def busy(self) -> bool:
        """True while sound is coming out, plus a decay tail afterwards.

        The tail exists because the speakers keep ringing after the process
        exits.  It is also the limitation named in the banner: it deafens the
        daemon to a "stop" spoken over its own reply.
        """
        with self._lock:
            p = self.proc
            if p is not None:
                if p.poll() is None:
                    return True
                self.proc = None
                self.quiet_until = time.monotonic() + SPEECH_TAIL_S
            return time.monotonic() < self.quiet_until

    def shutup(self) -> None:
        with self._lock:
            p, self.proc = self.proc, None
        if p is not None:
            reap(p, grace=1.0)
            self.quiet_until = time.monotonic() + SPEECH_TAIL_S

    # -- making noise -----------------------------------------------------
    def _play(self, argv: list[str]) -> None:
        self.shutup()
        try:
            p = spawn(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            self.log("audio_out_failed", argv=argv[0], error=str(exc))
            return
        with self._lock:
            self.proc = p

    def chime(self) -> None:
        if self.enabled and os.path.exists(CHIME):
            self._play(["afplay", CHIME])

    def prerender(self, key: str, text: str) -> pathlib.Path | None:
        """Render a fixed phrase once.  Costs 3.5-6.5 s each, at startup only.

        RENDERED TO A TEMPORARY NAME AND os.replace()d INTO PLACE, BECAUSE
        `say -o` GROWS ITS OUTPUT FILE IN PLACE AND say() GATES ON exists().
        Measured on `say -v Daniel -o t.aiff <sentence>`: the file appears at
        t = 0.568 s holding 0 bytes, sits at 4096 bytes for about 0.8 s, and
        reaches its final 352152 bytes only when the process exits at
        t = 1.404 s.  run() starts this thread and loop_audio then says
        PHRASES["ready"] with key="ready" a couple of seconds later, squarely
        inside that window: afplay on a 4096-byte truncation exits 0 having
        played 0.06 s, and on a 0-byte file exits 1 with AudioFileOpen failed --
        both with stderr=DEVNULL, so the phrase was silently dropped and the
        user heard a click.  Worse, a `say` killed by the cap left the fragment
        on disk and `if path.exists(): return path` then handed that fragment
        back on every future run, so the phrase was broken forever.
        os.replace() is atomic within one filesystem, so the cache path now
        either does not exist or is a complete file.

        A file below MIN_AIFF_BYTES is treated as absent and re-rendered: that
        is the fragment left by an earlier version of this function, and a real
        phrase here is 47 kB at the short end.
        """
        if not self.enabled:
            return None
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"{key}.aiff"
        if self._usable(path):
            return path
        path.unlink(missing_ok=True)
        tmp = path.with_name(path.name + f".part{os.getpid()}")
        argv = ["say"]
        if self.voice:
            argv += ["-v", self.voice]
        argv += ["-o", str(tmp), text]
        rc, _, err, to = run_capped(argv, cap=30.0)
        if rc != 0 or to or not self._usable(tmp):
            self.log("prerender_failed", key=key, rc=rc, timed_out=to,
                     bytes=(tmp.stat().st_size if tmp.exists() else 0),
                     error=err.strip()[:200])
            tmp.unlink(missing_ok=True)
            return None
        os.replace(tmp, path)
        return path

    # A complete `say -o` phrase is 47 kB at the short end ("Ready."); the
    # truncations measured above were 0 and 4096 bytes.  8 kB separates them by
    # an order of magnitude on both sides.
    MIN_AIFF_BYTES = 8192

    def _usable(self, path: pathlib.Path) -> bool:
        try:
            return path.stat().st_size >= self.MIN_AIFF_BYTES
        except OSError:
            return False

    def say(self, text: str, key: str | None = None, wait: bool = False) -> None:
        """Speak, blocking only if asked.  Never passes text to a shell."""
        text = re.sub(r"^[\s\-]+", "", " ".join(text.split()))[:600]
        if not text:
            return
        if not self.enabled:
            print(f"[jarvis] {text}", flush=True)
            return
        cached = self.cache / f"{key}.aiff" if key else None
        if cached is not None and self._usable(cached):
            self._play(["afplay", str(cached)])
        else:
            argv = ["say"]
            if self.voice:
                argv += ["-v", self.voice]
            argv += ["-r", str(self.rate), text]
            self._play(argv)
        print(f"[jarvis] {text}", flush=True)
        if wait:
            with self._lock:
                p = self.proc
            if p is not None:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    reap(p)


# ==========================================================================
# Input: one long-lived ffmpeg, a reader thread, and an energy VAD
# ==========================================================================

class Ear:
    """The capture spine: ONE ffmpeg for the life of the daemon.

    Per-utterance spawning was rejected: the avfoundation open costs 1.55-3.71 s
    and `-t N` charges it against the requested window (81 % and 80 % of the
    samples asked for, measured).  The reader thread must keep draining the pipe
    even while the main thread is blocked on whisper or on claude, or ffmpeg
    stalls and the stream desynchronises from the wall clock; the queue therefore
    drops its OLDEST audio when full and counts the drops rather than blocking.
    """

    def __init__(self, device_index: int, log, verbose: bool = False):
        self.index, self.log, self.verbose = device_index, log, verbose
        self.q: queue.Queue[bytes] = queue.Queue(maxsize=200)     # ~20 s
        self.zero_run = 0          # consecutive all-zero BYTES seen by _reader
        self.zero_warned = False
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.err_thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.first_byte: float | None = None
        self.total_bytes = 0
        self.delivered_bytes = 0
        self.drops = 0
        self._overflow_logged = False
        self.marks: deque[tuple[float, int]] = deque()
        self.eof = threading.Event()
        self._err: deque[str] = deque(maxlen=40)

    def start(self) -> None:
        argv = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "avfoundation", "-i", f":{self.index}",
                "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
        self.proc = spawn(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          bufsize=0)
        self.thread = threading.Thread(target=self._reader, daemon=True,
                                       name="ear")
        self.thread.start()
        # STDERR IS DRAINED FOR THE LIFE OF THE DAEMON, NOT READ AT THE END.
        # With -loglevel error a healthy run writes nothing, so the old code
        # (a PIPE nobody read until died() read it) was correct for every run
        # that was ever observed -- and would wedge ffmpeg's writer on the
        # first device that emits a per-frame warning, once ~64 KiB of pipe
        # buffer filled.  The daemon would then see the stream stop delivering
        # rather than an error, which is the exact failure realtime_ratio()
        # exists to notice and the hardest one to attribute.  Worse, died()
        # is reached whenever `eof` is set even while poll() is still None,
        # and its blocking stderr.read() would then hang loop_audio itself.
        self.err_thread = threading.Thread(target=self._drain_err, daemon=True,
                                           name="ear-err")
        self.err_thread.start()

    def _drain_err(self) -> None:
        f = self.proc.stderr if self.proc is not None else None
        if f is None:
            return
        try:
            for line in f:
                s = line.decode("utf-8", "replace").rstrip()
                if s:
                    self._err.append(s)
        except Exception:
            pass

    def stderr_text(self) -> str:
        return "\n".join(self._err)

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while not self.stop.is_set():
            chunk = self.proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                self.eof.set()
                return
            t = time.monotonic()
            if self.first_byte is None:
                self.first_byte = t
            self.total_bytes += len(chunk)
            # Cheap because it is a C-level scan that stops at the first
            # non-zero byte, so the common case is the fast one: measured
            # 0.130 us on a live chunk against 6.100 us on an all-zero one,
            # i.e. 0.00013% of one core at 10 chunks/s.  The asymmetry IS the
            # design -- only a stream that is already failing pays the scan.
            if chunk.strip(b"\x00"):
                self.zero_run = 0
                self.zero_warned = False
            else:
                self.zero_run += len(chunk)
            self.marks.append((t, self.total_bytes))
            while self.marks and t - self.marks[0][0] > RATE_WINDOW_S:
                self.marks.popleft()
            try:
                self.q.put_nowait(chunk)
                self.delivered_bytes += len(chunk)
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.drops += 1
                    # TWO DIFFERENT LOSSES, REPORTED SEPARATELY, BECAUSE
                    # CONFUSING THEM IS WHAT THE RATE_* SPLIT EXISTS TO STOP.
                    # realtime_ratio() counts what ffmpeg DELIVERED, which is
                    # the number the WARN/REFUSE tiering is about; this branch
                    # is audio the daemon threw away on purpose because the
                    # main thread was inside a claude turn.  Measured with a
                    # counter-stamped producer at exactly 32000 B/s and the
                    # main thread blocked for 30 s: q filled to 200/200, 131
                    # chunks = 13.1 s were discarded, and realtime_ratio()
                    # read a healthy 1.000 throughout -- correctly, but with
                    # no way to tell it from a gapped stream.  It is logged
                    # ONCE per overflow episode, not per chunk.
                    if not self._overflow_logged:
                        self._overflow_logged = True
                        self.log("queue_overflow", dropped_chunks=self.drops,
                                 note="the daemon was busy; capture itself is "
                                      "not gapped")
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(chunk)
                    self.delivered_bytes += len(chunk)
                except queue.Full:
                    pass
            else:
                self._overflow_logged = False

    def zero_run_seconds(self) -> float:
        """Wall seconds the stream has been UNBROKEN digital zero.

        Read without a lock deliberately: one writer (_reader), one
        reader (loop_audio), and an int rebind is atomic under the GIL.
        A torn read here would cost one polling interval, not a wrong
        verdict, because the threshold is sustained over many chunks.
        """
        return self.zero_run / float(RATE * WIDTH)

    def realtime_ratio(self) -> float:
        """Audio seconds ARRIVING FROM FFMPEG per wall second, over RATE_WINDOW_S.

        Measured at 0.75x sustained under load with ffmpeg stderr empty and exit
        code zero -- there is no other way to notice it.  Returns 1.0 until there
        is enough of a window to judge, so a fresh start never trips the refusal.

        This counts PRODUCTION, not consumption, and that is the intended
        reading: it answers "is the microphone stream gapped", which is the
        question a transcript's trustworthiness depends on.  Audio the daemon
        itself discarded because it was busy is a different loss and is reported
        as `queue_overflow` in _reader(); do not fold the two together.
        """
        if len(self.marks) < 2:
            return 1.0
        (t0, b0), (t1, b1) = self.marks[0], self.marks[-1]
        dt = t1 - t0
        if dt < 2.0:
            return 1.0
        return ((b1 - b0) / WIDTH / RATE) / dt

    def died(self) -> tuple[bool, str]:
        """EOF plus a non-zero exit means the device went away, not silence.

        A vanished device fails LOUDLY (rc=251, "Invalid audio device index"),
        which is the opposite of the under-delivery failure.  Reading it as
        silence would leave the daemon permanently deaf while looking healthy.
        """
        if self.proc is None:
            return True, "never started"
        rc = self.proc.poll()
        if rc is None and not self.eof.is_set():
            return False, ""
        # Whatever the drain thread has collected so far.  This used to be a
        # blocking stderr.read(), which returns only at EOF -- and this branch
        # is reached whenever `eof` is set even while ffmpeg is still alive, so
        # it could hang the audio loop instead of reporting on it.
        return True, self.stderr_text().strip() or f"ffmpeg exited rc={rc}"

    def read_seconds(self, seconds: float, timeout: float = 30.0) -> bytes:
        """Block until `seconds` of audio have arrived.  Used by the self-check."""
        need = int(RATE * seconds) * WIDTH
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while len(buf) < need and time.monotonic() < deadline:
            try:
                buf += self.q.get(timeout=0.5)
            except queue.Empty:
                dead, _ = self.died()
                if dead:
                    break
        return bytes(buf)

    def close(self) -> None:
        self.stop.set()
        reap(self.proc)


class EnergyVAD:
    """Adaptive-threshold voice activity detection over 32 ms frames.

    No silero: ggml-silero-v5.1.2.bin was never downloaded, and its timestamps
    are in centiseconds, which is a 10x bug waiting to be written.  The constants
    are at the top of this file, each with the measurement behind it.

    A STEP CHANGE IN ROOM NOISE USED TO LOCK THIS PERMANENTLY ACTIVE, AND A
    GRADUAL ONE NEVER DID -- SO EVERY TEST THAT RAMPED THE LEVEL PASSED.  The
    floor was updated only on a frame that was BOTH inactive and non-speech, so
    once every frame exceeded the threshold there was no path back to the update
    at all.  Driven straight through push(): 19 s of quiet room (+-9 counts)
    settles floor = 5.42, threshold = 60.0; a STEP to steady noise at rms 61.1
    (peak +-105, -54.6 dBFS -- a fan, an open window, someone talking in the
    room) then held floor = 5.42 and threshold = 60.0 for the next 96 s, active
    the whole time, emitting a segment of exactly max_seg_s every max_seg_s --
    15 of them in 96 s, each POSTed to whisper, and loop_audio flushing the
    capture queue after each one.  The hangover endpointer never fires in that
    state, so a spoken command is cut at an arbitrary 8 s boundary and can be
    split across two segments.  Measured crossover: rms 59.5 adapts correctly
    (floor -> 56.87, threshold 170.6) and rms 61.1 locks; a gradual RAMP to the
    same rms 115 ratchets up perfectly well (floor 111.95, threshold 335.8).
    Recovery came only when the noise stopped.

    The signal that distinguishes the two cases is already there and was being
    thrown away: a segment terminated by max_seg_s rather than by hangover is
    one whose "speech" never ended, which is what noise looks like and what an
    utterance does not.  So that exit re-learns the floor from the segment's own
    tail before resetting.  A genuinely long utterance re-learns a floor that is
    much too high for one to two seconds -- FLOOR_A_DOWN is tau 0.3 s, measured
    at about 1.35 s to fall from a speech-level floor back to a 15-count room --
    and that is the right trade against being deaf until the noise stops.
    vad_selftest() steps the level rather than ramping it, because a ramping
    test cannot see any of this.
    """

    def __init__(self, max_seg_s: float = MAX_SEG_WAKE_S):
        self.max_seg_s = max_seg_s
        self.floor: float | None = None
        self.pre: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self.seg = bytearray()
        self.active = False
        self.voiced = 0
        self.silent = 0
        self.last_rms = 0.0
        # Frames actually above the threshold in the segment being built, and
        # in the one last emitted.  MIN_SEG_MS is measured against the first;
        # handle_segment() uses the second as the acoustic evidence that there
        # was anything to transcribe.
        self.voiced_total = 0
        self.last_voiced_frames = 0
        self.last_total_frames = 0
        self.last_exit = ""              # "hangover" | "max_seg"

    def reset(self) -> None:
        """Drop everything, including the pre-roll.

        Called on ungating, so the tail of the daemon's own sentence cannot
        become the head of the user's next one.
        """
        self.pre.clear()
        self.seg = bytearray()
        self.active = False
        self.voiced = self.silent = 0
        self.voiced_total = 0

    def threshold(self) -> float:
        f = self.floor if self.floor is not None else MIN_FLOOR_RMS
        return max(f * SPEECH_FACTOR, ABS_MIN_RMS)

    def _emit(self, exit_reason: str) -> bytes:
        seg = bytes(self.seg)
        self.last_voiced_frames = self.voiced_total
        self.last_total_frames = len(seg) // FRAME_BYTES
        self.last_exit = exit_reason
        return seg

    def push(self, frame: bytes) -> bytes | None:
        """Feed one frame; returns a completed segment, or None."""
        rms = frame_rms(frame)
        self.last_rms = rms
        if self.floor is None:
            self.floor = max(rms, MIN_FLOOR_RMS)
        speech = rms > self.threshold()

        if not self.active:
            self.pre.append(frame)
            if speech:
                self.voiced += 1
                if self.voiced >= ONSET_FRAMES:
                    # The deque is PREROLL_LEAD_FRAMES + ONSET_FRAMES deep and
                    # already ends with the onset frames, so this carries 512 ms
                    # of lead-in ahead of the first speech frame and no clipped
                    # first syllable.
                    self.seg = bytearray(b"".join(self.pre))
                    self.pre.clear()
                    self.active = True
                    self.silent = 0
                    self.voiced_total = self.voiced
            else:
                self.voiced = 0
                a = FLOOR_A_UP if rms > self.floor else FLOOR_A_DOWN
                self.floor = (1.0 - a) * self.floor + a * max(rms, 0.0)
            return None

        self.seg += frame
        if speech:
            self.silent = 0
            self.voiced_total += 1
        else:
            self.silent += 1
            if self.silent >= HANGOVER_FRAMES:
                seg = self._emit("hangover")
                voiced_ms = self.voiced_total * FRAME_MS
                self.reset()
                # The cough filter, counted in VOICED frames.  Measured against
                # the segment length it could never reject anything: the
                # hangover and onset frames alone are 800 ms.
                return seg if voiced_ms >= MIN_SEG_MS else None
        if dur_s(self.seg) >= self.max_seg_s:
            seg = self._emit("max_seg")
            # RE-LEARN THE FLOOR HERE AND NOWHERE ELSE.  Reaching max_seg_s
            # means the level never dropped below the threshold for 704 ms, so
            # either the room changed or this is not speech.  The tail is used
            # rather than the whole segment so that a long utterance followed by
            # a step in noise still learns the noise.
            tail = seg[-FRAME_BYTES * 31:] or seg          # ~1 s
            self.floor = max(self.floor, frame_rms(tail) / SPEECH_FACTOR,
                             MIN_FLOOR_RMS)
            self.reset()
            return seg
        return None


# ==========================================================================
# Speech recognition: one resident whisper-server per tier
# ==========================================================================

class WhisperTier:
    """A resident whisper-server, started and reaped by this daemon.

    ADOPTING WHATEVER IS ALREADY ON THE PORT IS NOW OPT-IN, BECAUSE IT HANDED
    EVERY WORD THIS DAEMON BELIEVES IT HEARD TO AN UNKNOWN LOCAL PROCESS.  The
    old rule was: if something answers on 127.0.0.1:18080, adopt it.  The
    liveness test is a bare HTTP GET that counts even an HTTPError as success,
    so "something" was anything that binds a TCP port -- and 18080 is
    unprivileged, fixed by default, and reachable by any process running as this
    user (a dev server, an npm postinstall, a browser extension's helper).
    Whatever bound it first would then return arbitrary JSON {"text": ...} for
    every POST, and that text flows through normalise() -> gate() -> route()
    into `claude -p` with the tool permissions listed at the top of this file.
    The daemon logged `whisper_adopted` and looked entirely healthy.  load_model
    also POSTed the local model path to that unknown listener, and its return
    value was discarded.

    So: by default a busy port is WORKED AROUND, not adopted -- this daemon
    picks a free ephemeral port and starts its own server, which costs the
    memory the old behaviour was avoiding and is the only choice that does not
    rest on a stranger.  --adopt-whisper restores the old behaviour for the case
    it was written for (a server the user started by hand), and then adoption is
    CHECKED rather than assumed: POST /load must succeed, and the server must
    answer a real /inference with a whisper-shaped envelope, or startup fails.
    That check proves it is a whisper-server API; it does not prove the server
    is honest, and nothing short of a known-plaintext audio fixture would.  An
    adopted server is still not reaped on exit: it was not ours to kill.
    """

    def __init__(self, name: str, model: pathlib.Path, port: int, threads: int,
                 log, verbose: bool = False, adopt: bool = False,
                 stop: threading.Event | None = None):
        self.name, self.model, self.port = name, model, port
        self.threads, self.log, self.verbose = threads, log, verbose
        self.allow_adopt = adopt
        self.stop = stop if stop is not None else threading.Event()
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.adopted = False

    # -- lifecycle --------------------------------------------------------
    def _listening(self, timeout: float = 1.0) -> bool:
        try:
            urllib.request.urlopen(self.url + "/", timeout=timeout).read(1)
            return True
        except urllib.error.HTTPError:
            return True                      # answered; that is all we need
        except Exception:
            return False

    @staticmethod
    def _free_port() -> int:
        """An ephemeral port the kernel says is free right now.

        There is a race between closing this socket and whisper-server binding
        it, and it is the smaller risk: the alternative is trusting whatever
        already owns a fixed port.
        """
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def start(self) -> bool:
        if self._listening():
            if not self.allow_adopt:
                busy = self.port
                self.port = self._free_port()
                self.url = f"http://127.0.0.1:{self.port}"
                print(f"  ! port {busy} is already in use by SOMETHING and "
                      f"--adopt-whisper was not\n    given, so the {self.name} "
                      f"tier starts its own server on {self.port} instead.\n"
                      f"    Adopting an unknown listener would let it choose "
                      f"every word I hear.", file=sys.stderr)
                self.log("whisper_port_busy", tier=self.name, busy_port=busy,
                         using_port=self.port)
            else:
                self.adopted = True
                self.log("whisper_adopted", tier=self.name, port=self.port)
                if not self.load_model():
                    print(f"  the server on port {self.port} refused POST /load, "
                          f"so it is not a whisper-server this daemon can use.",
                          file=sys.stderr)
                    return False
                return True
        argv = ["whisper-server", "-m", str(self.model),
                "--host", "127.0.0.1", "--port", str(self.port),
                "-t", str(self.threads)]
        try:
            self.proc = spawn(argv, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        except OSError as exc:
            self.log("whisper_spawn_failed", tier=self.name, error=str(exc))
            print(f"  could not start whisper-server: {exc}", file=sys.stderr)
            return False
        self.log("whisper_spawned", tier=self.name, port=self.port,
                 model=self.model.name, threads=self.threads)
        return True

    def wait_ready(self, cap: float = SERVER_READY_S) -> bool:
        """Poll until it answers, giving up at `cap` -- or at once on Ctrl-C.

        stop.wait() rather than time.sleep(): the first ggml run on a machine
        compiles Metal shaders for ~115 s, so SERVER_READY_S is 180 s, and a
        sleep loop that never consults the stop Event ignores Ctrl-C for the
        whole of it with whisper-server children already spawned.
        """
        deadline = time.monotonic() + cap
        while time.monotonic() < deadline and not self.stop.is_set():
            if self.proc is not None and self.proc.poll() is not None:
                return False
            if self._listening():
                return True
            self.stop.wait(0.5)
        return False

    def verify(self) -> bool:
        """Prove the tier can transcribe before the daemon claims to be ready.

        Doubles as the warm-up POST, and for an ADOPTED server it is the only
        evidence that the thing on the port speaks whisper's API at all: a
        successful /inference returning a JSON object with a `text` key.
        """
        body, ctype = _multipart({"response_format": "json", "temperature": "0"},
                                 "probe.wav", wav_bytes(b"\x00\x00" * int(RATE * 0.5)))
        req = urllib.request.Request(self.url + "/inference", data=body,
                                     headers={"Content-Type": ctype})
        try:
            raw = urllib.request.urlopen(req, timeout=SERVER_READY_S).read()
            doc = json.loads(raw.decode("utf-8", "replace"))
        except Exception as exc:
            self.log("whisper_verify_failed", tier=self.name, error=str(exc)[:200])
            return False
        ok = isinstance(doc, dict) and "text" in doc
        if not ok:
            self.log("whisper_verify_failed", tier=self.name,
                     error=f"no 'text' in the reply: {str(doc)[:160]}")
        return ok

    def load_model(self) -> bool:
        """POST /load -- hot-swap without a restart.  3.2 s for small.en."""
        body, ctype = _multipart({"model": str(self.model)}, None, None)
        try:
            req = urllib.request.Request(self.url + "/load", data=body,
                                         headers={"Content-Type": ctype})
            urllib.request.urlopen(req, timeout=120).read()
            return True
        except Exception as exc:
            self.log("whisper_load_failed", tier=self.name, error=str(exc)[:200])
            return False

    def warm(self) -> tuple[bool, float]:
        """One real POST, whose ANSWER IS CHECKED.  Returns (ok, seconds).

        The first inference after launch costs 4.04 s of GPU pipeline warm-up and
        every later one 0.34-0.48 s.  Spending it here means the user's first real
        command does not.  It used to throw the reply away; it is now the tier's
        verification, which is what makes an adopted server checkable at all.
        """
        t0 = time.monotonic()
        ok = self.verify()
        return ok, time.monotonic() - t0

    def close(self) -> None:
        if self.proc is not None and not self.adopted:
            reap(self.proc)

    # -- use --------------------------------------------------------------
    def transcribe(self, pcm: bytes, prompt: str | None,
                   cap: float = 60.0) -> str:
        fields = {"response_format": "json", "temperature": "0"}
        if prompt:
            fields["prompt"] = prompt
        body, ctype = _multipart(fields, "seg.wav", wav_bytes(pcm))
        req = urllib.request.Request(self.url + "/inference", data=body,
                                     headers={"Content-Type": ctype})
        t0 = time.monotonic()
        try:
            raw = urllib.request.urlopen(req, timeout=cap).read()
        except Exception as exc:
            self.log("whisper_error", tier=self.name, error=str(exc)[:200])
            return ""
        dt = time.monotonic() - t0
        try:
            text = json.loads(raw.decode("utf-8", "replace")).get("text", "")
        except json.JSONDecodeError:
            text = raw.decode("utf-8", "replace")
        # whisper annotates non-speech rather than returning nothing, and it does
        # not confine itself to a fixed token list.  A first version of this
        # stripped exactly [BLANK_AUDIO] and [SOUND]; the first quiet room this
        # daemon ever listened to produced "[MUSIC PLAYING]", which sailed through
        # and was wake-checked as if it were speech (rejected at ratio 0.308, so
        # it was noise in the log rather than a wrong action -- but the next such
        # string is not guaranteed to be).  Strip the FORM, not the vocabulary.
        text = _ANNOTATION_RE.sub(" ", text)
        text = " ".join(text.split())
        if self.verbose:
            print(f"    [{self.name}] {dur_s(pcm):.2f} s audio -> {dt:.2f} s "
                  f"-> {text!r}", flush=True)
        return text


def _multipart(fields: dict[str, str], filename: str | None,
               filedata: bytes | None) -> tuple[bytes, str]:
    """Hand-rolled multipart/form-data.  The audio field must be named `file`."""
    boundary = "----M3LBjarvis" + os.urandom(12).hex()
    body = bytearray()
    for k, v in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                 f"{v}\r\n").encode()
    if filename is not None and filedata is not None:
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="file"; '
                 f'filename="{filename}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode()
        body += filedata
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


# ==========================================================================
# Dispatch: the claude CLI, headless, in this repo
# ==========================================================================

# Dropped from every child environment: see Agent.env().
_CHILD_ENV_DROP = re.compile(r"^(CLAUDE_|CLAUDECODE$|ANTHROPIC_)")


class Agent:
    """One spoken command -> one `claude -p` turn -> one short spoken answer."""

    def __init__(self, repo: pathlib.Path, binary: str, allow_edits: bool,
                 cap: float, max_turns: int, budget: float, log,
                 dry_run: bool = False):
        self.repo, self.binary, self.cap = repo, binary, cap
        self.allow_edits, self.max_turns, self.budget = allow_edits, max_turns, budget
        self.log, self.dry_run = log, dry_run
        self.session_id: str | None = None
        self.last_used = 0.0
        self.spend = 0.0

    def env(self) -> dict[str, str]:
        """Explicit child environment.

        Every CLAUDE_* key and CLAUDECODE are dropped, because developing this
        from inside a Claude Code session is the normal case and a child that
        inherits CLAUDE_CODE_MESSAGING_SOCKET behaves differently from one a
        terminal launches.  ANTHROPIC_API_KEY is dropped because in -p mode it is
        always used when present, with no approval step, silently moving every
        request off the subscription.  ANTHROPIC_BASE_URL is dropped as
        unnecessary: it is an endpoint override with no role in auth.

        THE FILTER IS ON THE FAMILY, NOT ON A LIST OF TWO MEMBERS, BECAUSE THE
        LIST WAS ALREADY WRONG.  It named ANTHROPIC_API_KEY and
        ANTHROPIC_BASE_URL and missed ANTHROPIC_AUTH_TOKEN, which Claude Code
        reads with the same precedence over OAuth that the paragraph above
        argues about -- so a user with it exported in a shell profile had every
        spoken command billed and routed through that credential instead of the
        subscription, with no approval step and nothing in the log to show it.
        setup.sh's own scan already counted "CLAUDE_*/ANTHROPIC_* variables",
        i.e. the family was the intent on the other side of the fence.  Anything
        ANTHROPIC_* that the child genuinely needs must be re-added here
        deliberately, by name.
        """
        env = {k: v for k, v in os.environ.items()
               if not _CHILD_ENV_DROP.match(k)}
        # RE-ADDED BY NAME, which is exactly the escape hatch the paragraph above
        # reserves -- and it is needed because the advice and the code disagreed.
        # setup.sh:1194 tells the user to `claude setup-token` and export this
        # when the daemon runs somewhere with no browser to log in from, and the
        # CLAUDE_ family filter then dropped it silently, so the one documented
        # headless auth path did not work and failed as "Not logged in".  It is
        # safe to pass where ANTHROPIC_API_KEY is not: it IS the subscription
        # credential, so it cannot quietly move spend onto metered billing.
        tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if tok:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
        return env

    def argv(self, prompt: str) -> list[str]:
        """The child's argv, allowlist AND denylist.

        --disallowedTools is the backstop the allowlist cannot be: an allowlist
        entry is a glob over a command line, and a glob over a multi-tool binary
        is not a description of what that binary will do (see READ_TOOLS).  The
        deny list keeps `cmake -E` and `cmake -P`, which are a recursive delete
        and an arbitrary-binary launcher, out of reach even under --allow-edits;
        --allow-edits removes Write and Edit from it and nothing else.
        """
        tools = READ_TOOLS + ("," + EDIT_TOOLS if self.allow_edits else "")
        deny = DENY_ALWAYS if self.allow_edits else DENY_TOOLS
        a = [self.binary, "-p", prompt,
             "--output-format", "json",
             "--permission-mode", "acceptEdits" if self.allow_edits else "dontAsk",
             "--allowedTools", tools,
             "--disallowedTools", deny,
             "--permission-prompts", "none",
             "--max-turns", str(self.max_turns),
             "--max-budget-usd", f"{self.budget:.2f}",
             "--append-system-prompt", SPOKEN_STYLE]
        if self.session_id and time.monotonic() - self.last_used < SESSION_IDLE_S:
            a += ["--resume", self.session_id]
        return a

    def dispatch(self, prompt: str) -> tuple[bool, str]:
        """Returns (ok, spoken_text).  Branches on is_error, never on subtype."""
        argv = self.argv(prompt)
        if self.dry_run:
            print("[dry-run] would run:", flush=True)
            for i, a in enumerate(argv):
                print(f"    argv[{i}] = {a!r}", flush=True)
            print(f"    cwd  = {self.repo}", flush=True)
            self.log("dry_run_dispatch", prompt=prompt, argv=argv,
                     cwd=str(self.repo))
            return True, "Dry run. Nothing was dispatched."

        t0 = time.monotonic()
        rc, out, err, timed_out = run_capped(argv, cap=self.cap,
                                             cwd=str(self.repo), env=self.env())
        dt = time.monotonic() - t0
        if timed_out:
            self.log("claude_timeout", prompt=prompt, seconds=round(dt, 1))
            return False, (f"That took longer than {int(self.cap)} seconds, so I "
                           "stopped it. The turn is still resumable.")
        if rc is None and not out:
            # run_capped could not even spawn it: a missing binary or a cwd
            # that does not exist.  Distinguished from a bad reply because
            # saying "I could not read the reply" would send the user looking
            # in the wrong place.
            self.log("claude_spawn_failed", error=err[:200])
            return False, f"I could not start Claude Code. {' '.join(err.split())[:160]}"

        env = parse_claude_json(out)
        if env is None:
            self.log("claude_unparseable", rc=rc, stdout=out[:800],
                     stderr=err[:400], seconds=round(dt, 1))
            snippet = (out or err or "").strip().splitlines()
            return False, ("I could not read the reply from Claude Code. "
                           + (snippet[0][:160] if snippet else ""))

        sid = env.get("session_id")
        if isinstance(sid, str) and sid:
            self.session_id = sid
            self.last_used = time.monotonic()
        cost = env.get("total_cost_usd") or 0.0
        try:
            self.spend += float(cost)
        except (TypeError, ValueError):
            pass
        result = env.get("result")
        text = result if isinstance(result, str) else json.dumps(result)[:400]

        self.log("claude_result", prompt=prompt, is_error=bool(env.get("is_error")),
                 subtype=env.get("subtype"), num_turns=env.get("num_turns"),
                 session_id=self.session_id, cost_usd=cost,
                 seconds=round(dt, 1), result=text[:2000],
                 permission_denials=env.get("permission_denials"))

        # is_error is the load-bearing field: an auth failure arrives as
        # subtype="success" with is_error=true and the failure sentence in
        # `result`, which a daemon would otherwise speak as an answer.
        if env.get("is_error"):
            if "not logged in" in text.lower():
                return False, ("Claude Code is not logged in. Run claude auth "
                               "login in a terminal once, then restart me.")
            return False, f"That failed. {text[:240]}"
        return True, text


def parse_claude_json(stdout: str) -> dict | None:
    """Pull the result envelope out of stdout, tolerating leading warnings."""
    s = stdout.strip()
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    while start != -1:
        try:
            v = json.loads(s[start:])
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            start = s.find("{", start + 1)
    for line in reversed(s.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                v = json.loads(line)
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                continue
    return None


# ==========================================================================
# The daemon
# ==========================================================================

def verdict(cmd: "Command", rc: int | None, out: str, err: str,
            timed_out: bool) -> tuple[bool, str]:
    """Turn a local command's output into one spoken sentence.

    The three shapes this tree actually prints, in the order they are looked for:
    ctest's own summary line, the PASS/FAIL acceptance block every validation
    case ends with, and then plain output.  Reading the whole of a driver's
    stdout aloud would be unusable -- static_droplet alone prints a table -- so
    the rule is one verdict plus at most the first line of context.
    """
    if timed_out:
        return False, (f"{cmd.id} was still going after "
                       f"{int(cmd.timeout_s)} seconds, so I stopped it.")

    # rc=None without a timeout is run_capped reporting that the child never
    # started: argv[0] or cwd does not exist.  `build/` is gitignored, so this
    # is the ordinary state of a fresh clone and not an exotic failure.
    if rc is None:
        return False, (f"I could not start {cmd.id}. "
                       + " ".join((err or "").split())[:200])

    # An exit code the entry gives a spoken meaning to.  It exists for
    # stop_the_run, where pkill's rc=1 means NOTHING MATCHED -- the success
    # case -- and the generic rule read it aloud as "exited with code 1 and
    # said nothing", i.e. reported the good outcome as a failure.
    say = cmd.rc_say.get(str(rc))
    if say is not None:
        return True, say

    text = (out or "") + ("\n" + err if err else "")

    # ctest OMITS the "N tests failed" clause entirely when nothing failed, so
    # the clause is optional.  Both forms captured from real runs:
    #   "100% tests passed out of 2"
    #   "33% tests passed, 2 tests failed out of 3"
    # Requiring the clause meant the all-pass case fell through to the raw log
    # and the daemon read several hundred characters of ctest output aloud.
    m = re.search(r"(\d+)% tests passed(?:,\s*(\d+) tests? failed)? "
                  r"out of (\d+)", text)
    if m:
        pct, failed, total = m.group(1), int(m.group(2) or 0), int(m.group(3))
        if failed == 0:
            return True, f"All {total} tests passed."
        names = re.findall(r"^\s*\d+\s*-\s*(\S+)\s*\(Failed\)", text, re.M)
        who = (", ".join(names[:4]) + (" and others" if len(names) > 4 else "")
               ) if names else ""
        return False, (f"{failed} of {total} tests failed"
                       + (f": {who}." if who else f", {pct} percent passed."))

    passes = len(re.findall(r"(?<!\w)PASS(?!\w)", text))
    fails = len(re.findall(r"(?<!\w)FAIL(?!\w)", text))
    if passes or fails:
        if fails == 0:
            return True, f"Passed, all {passes} checks."
        line = next((l.strip() for l in text.splitlines()
                     if re.search(r"(?<!\w)FAIL(?!\w)", l)), "")
        return False, (f"{fails} of {passes + fails} checks failed. "
                       + " ".join(line.split())[:160])

    body = " ".join((out or "").split())
    if not body:
        if rc == 0:
            return True, (cmd.empty_say or "Done. It printed nothing.")
        return False, (f"{cmd.id} exited with code {rc} and said nothing."
                       + (" " + " ".join(err.split())[:160] if err else ""))
    ok = rc == 0
    head = body[:240]
    return ok, head if ok else f"It exited with code {rc}. {head}"


class VoiceDaemon:
    """Wake word, capture, transcribe, confirm, dispatch, speak.

    States: IDLE waits for the wake word on the cheap tier; ARMED has heard it and
    is listening for a command on the accurate tier; CONFIRM is waiting for the
    word "confirm" before a destructive dispatch.  ARMED and CONFIRM both time
    out, and a timeout DISCARDS rather than queues.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = pathlib.Path(args.repo).resolve()
        self.logdir = _HERE / "logs"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logpath = self.logdir / f"jarvis-{datetime.date.today():%Y%m%d}.jsonl"
        self._logfh = None
        self._loglock = threading.Lock()
        self._prune_logs()

        self.mouth = Mouth(args.voice, args.speech_rate, not args.no_speak,
                           _HERE / "cache", self.log)
        self.agent = Agent(self.repo, args.claude_bin, args.allow_edits,
                           args.claude_timeout, args.max_turns, args.budget,
                           self.log, dry_run=args.dry_run)
        self.ear: Ear | None = None
        self.wake_tier: WhisperTier | None = None
        self.cmd_tier: WhisperTier | None = None
        self.state = "IDLE"
        self.armed_until = 0.0
        self.pending: str | None = None
        self.pending_label: str | None = None
        self.stop = threading.Event()
        self._rate_warned = 0.0
        # Set by preflight_tools().  False means the registry still works and
        # free-text questions do not -- see the degradation argument there.
        self.agent_ok = True

    # -- logging ----------------------------------------------------------
    #
    # A WAKE-WORD DAEMON HEARS THE WHOLE ROOM, SO ITS LOG IS A TRANSCRIPT OF THE
    # ROOM UNLESS SOMETHING STOPS IT BEING ONE.  handle_segment() logs `heard`
    # only when the wake word actually hit, or under --log-all-audio, which
    # prints a warning at startup; every other segment is recorded as METADATA
    # (duration, rms, best ratio, near misses), which is what the cutoff-tuning
    # argument in the banner actually needs.  These two methods bound the file
    # itself: nothing used to prune, rotate or size-cap it, and it sits on an
    # 8 GB laptop whose ~/.claude/projects already holds a 260 MB transcript.
    def _prune_logs(self) -> None:
        cutoff = time.time() - LOG_RETENTION_DAYS * 86400
        for p in self.logdir.glob("jarvis-*.jsonl*"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass

    def _roll_if_large(self) -> None:
        """Called with the lock held, from log()."""
        try:
            if self._logfh is None or self._logfh.tell() < LOG_MAX_BYTES:
                return
        except OSError:
            return
        self._logfh.close()
        self._logfh = None
        for n in range(8, 0, -1):
            src = self.logpath.with_name(self.logpath.name + (f".{n}" if n else ""))
            dst = self.logpath.with_name(self.logpath.name + f".{n + 1}")
            if src.exists():
                try:
                    os.replace(src, dst)
                except OSError:
                    pass
        try:
            os.replace(self.logpath, self.logpath.with_name(
                self.logpath.name + ".1"))
        except OSError:
            pass

    def log(self, event: str, **kw) -> None:
        rec = {"ts": now_iso(), "event": event, **kw}
        line = json.dumps(rec, default=str, ensure_ascii=False)
        with self._loglock:
            if self._logfh is None:
                self._logfh = open(self.logpath, "a", encoding="utf-8")
            self._logfh.write(line + "\n")
            self._logfh.flush()
            self._roll_if_large()
        if self.args.verbose:
            print(f"    . {event} "
                  f"{ {k: v for k, v in kw.items() if k != 'argv'} }", flush=True)

    # -- startup ----------------------------------------------------------
    def preflight_tools(self) -> list[str]:
        """Everything that can be checked without opening the microphone.

        A MISSING OR LOGGED-OUT claude IS A DEGRADATION, NOT A FATAL ERROR, AND
        THAT CHANGED WHEN THE REGISTRY LANDED.  It used to be fatal, correctly:
        before commands.json every single utterance went to the agent, so a
        daemon without one could do nothing at all and refusing to start was
        honest.  Now 20 of the 28 registry entries are literal argvs -- build,
        ctest, git, the validation cases -- and none of them needs an agent or a
        network.  Aborting startup would refuse to run `git status` because a
        node package is not installed.  So this warns loudly, sets agent_ok, and
        lets the local half work; dispatch() then says what is missing at the
        moment a question actually needs it, which is the point where the user
        can act on it.
        """
        problems = []
        # Checked HERE rather than in start_audio, which runs after whisper --
        # and whisper's first-ever start on a machine compiles Metal shaders
        # for two minutes.  Being told the command line was wrong after that is
        # not being told it early.
        if not self.args.text and not self.args.device:
            problems.append(
                "--device is required (there is no default: an avfoundation "
                "index is\n        assignment order, not identity, and index 0 "
                "has been a conference\n        device on this machine).  List "
                "them with:\n"
                "            python3 voice/miccheck.py --quiet")
        need = ["ffmpeg", "whisper-server"] if not self.args.text else []
        if not self.args.no_speak:
            need += ["say", "afplay"]
        for b in need:
            if shutil.which(b) is None:
                problems.append(f"{b} is not on PATH")

        if not self.args.text:
            for label, path in (("wake", self.args.wake_model),
                                ("command", self.args.command_model)):
                if not pathlib.Path(path).exists():
                    problems.append(
                        f"{label}-tier model missing: {path}\n"
                        f"        fetch it with:  whisper-cli --help  (brew ships "
                        f"none)\n"
                        f"        or download ggml-{label}.en.bin into "
                        f"{self.args.model_dir}")

        if not self.args.dry_run:
            warn = None
            cb = shutil.which(self.args.claude_bin) or (
                self.args.claude_bin if os.path.exists(self.args.claude_bin) else None)
            if cb is None:
                warn = ("the claude CLI is not installed.\n"
                        "    install:  npm i -g @anthropic-ai/claude-code\n"
                        "    then:     claude auth login\n"
                        "    (the npm prefix is writable, so no sudo; node "
                        "v26.9.0 satisfies engines >=22)")
            else:
                self.args.claude_bin = cb
                self.agent.binary = cb
                rc, out, _, to = run_capped([cb, "auth", "status"], cap=30.0,
                                            env=self.agent.env())
                logged_in = False
                try:
                    logged_in = bool(json.loads(out or "{}").get("loggedIn"))
                except json.JSONDecodeError:
                    pass
                if to or not logged_in:
                    warn = ("claude is installed but NOT LOGGED IN "
                            f"(auth status rc={rc}).\n"
                            "    The desktop app's keychain item is NOT "
                            "inherited: it authenticates its\n"
                            "    child over a host socket, which a standalone "
                            "CLI cannot read.\n"
                            "    Run once, interactively:  claude auth login")
            if warn is not None:
                self.agent_ok = False
                local = sum(1 for c in REGISTRY if c.kind == "argv")
                print(f"\n  ! {warn}\n"
                      f"    Continuing WITHOUT the agent: the {local} literal "
                      f"commands in\n"
                      f"    {self.args.commands} still work, and anything else "
                      f"will say so.\n", file=sys.stderr)
                self.log("agent_unavailable", reason=warn.splitlines()[0])
        return problems

    def start_whisper(self) -> bool:
        wake_model = pathlib.Path(self.args.wake_model)
        cmd_model = pathlib.Path(self.args.command_model)
        self.wake_tier = WhisperTier("tiny", wake_model, self.args.port_base,
                                     self.args.threads, self.log,
                                     self.args.verbose,
                                     adopt=self.args.adopt_whisper,
                                     stop=self.stop)
        # One model means one server: a second copy of small.en is 488 MB on an
        # 8 GB machine that is also running the solver.
        if cmd_model == wake_model:
            self.cmd_tier = self.wake_tier
        else:
            self.cmd_tier = WhisperTier("small", cmd_model,
                                        self.args.port_base + 1,
                                        self.args.threads, self.log,
                                        self.args.verbose,
                                        adopt=self.args.adopt_whisper,
                                        stop=self.stop)
        tiers = [self.wake_tier] + ([self.cmd_tier]
                                    if self.cmd_tier is not self.wake_tier else [])
        if self.args.adopt_whisper:
            print("  ! --adopt-whisper: a server already listening on the port "
                  "will be USED.\n    Whatever owns that port then chooses every "
                  "word I believe I heard.", file=sys.stderr)
        for t in tiers:
            if not t.start():
                return False
        print("  waiting for whisper-server. THE FIRST ggml RUN ON A MACHINE",
              "compiles Metal\n  shaders for about two minutes and looks exactly",
              "like a hang; later starts\n  take under a second.", flush=True)
        for t in tiers:
            if not t.wait_ready():
                if self.stop.is_set():
                    return False
                print(f"  {t.name} tier did not come up on port {t.port} within "
                      f"{SERVER_READY_S:.0f} s", file=sys.stderr)
                return False
            # The warm-up POST is now the tier's VERIFICATION: its answer is
            # checked rather than thrown away, so the daemon cannot announce
            # itself ready against a server that does not transcribe.
            ok, dt = t.warm()
            if not ok:
                print(f"  {t.name} tier on port {t.port} did not return a "
                      f"transcript for a known probe, so it is not usable.",
                      file=sys.stderr)
                return False
            print(f"  {t.name} tier ready on {t.port} "
                  f"({t.model.name}, verified warm-up POST {dt:.2f} s)",
                  flush=True)
            self.log("whisper_ready", tier=t.name, port=t.port,
                     adopted=t.adopted, warmup_s=round(dt, 2))
        return True

    def start_audio(self) -> bool:
        """Open the microphone and REFUSE to proceed against a zero stream."""
        if _mic is None:
            print("  miccheck.py could not be imported; it holds the liveness "
                  "test.\n  Keep jarvis.py and miccheck.py in the same directory, "
                  "or use --text.", file=sys.stderr)
            return False
        devs = _mic.list_audio_devices()
        if not devs:
            print("  no avfoundation audio devices at all", file=sys.stderr)
            return False
        print("  audio devices (indices are TODAY's; they renumber on every "
              "plug/unplug):", flush=True)
        for i, n in devs:
            print(f"      [{i}] {n}", flush=True)
        want = self.args.device
        # --device IS REQUIRED, BECAUSE THE DEFAULT WAS INDEX 0 AND THIS FILE
        # SPENDS A PAGE ARGUING THAT INDEX 0 MEANS NOTHING.  The list above
        # changed three times in seventy minutes on this machine, once with the
        # Continuity iPhone at [0]; "Microsoft Teams Audio" is an enumerated
        # input device on it.  Binding to whatever is at index 0 today can
        # therefore capture a MEETING -- and then every remote participant is a
        # wake-word speaker, the 8 s ARMED window needs no second wake word, and
        # whatever the call says next is transcribed on small.en and dispatched
        # into this repository.  The liveness test passes happily on a
        # conference device, because it carries real audio.
        if not want:
            print("\n  --device is REQUIRED.  There is no sensible default: an "
                  "avfoundation index\n  is assignment order, not identity, and "
                  "the list above is TODAY's.  Pick a\n  name from it, or find a "
                  "live one with:\n      python3 voice/miccheck.py --quiet\n"
                  "  then:\n      python3 voice/jarvis.py --device "
                  f"{devs[0][1]!r}\n", file=sys.stderr)
            return False
        idx = _mic.resolve(want, devs)
        if idx is None:
            print(f"  no device matching {want!r}", file=sys.stderr)
            return False
        name = dict(devs).get(idx, "?")
        # A VIRTUAL OR CONFERENCE DEVICE IS THE ROOM'S OTHER END, NOT THE ROOM.
        # Refused unless the name was given EXACTLY, so that "--device Teams"
        # (a substring match, easy to reach by accident) cannot put a meeting on
        # the wake word while "--device 'Microsoft Teams Audio'" still can.
        if VIRTUAL_DEVICE_RE.search(name) and name.lower() != str(want).lower():
            print(f"\n  [{idx}] {name} is a virtual or conference audio device: "
                  f"capturing it means\n  capturing the other end of a call, "
                  f"and everyone on that call can then drive\n  this daemon. "
                  f"If you mean it, name it exactly:\n"
                  f"      python3 voice/jarvis.py --device {name!r}\n",
                  file=sys.stderr)
            self.log("device_refused", index=idx, name=name, requested=want)
            return False
        if str(want).isdigit():
            print(f"  WARNING: you passed an INDEX. It meant three different "
                  f"microphones\n           in seventy minutes on this machine. "
                  f"Pass a name.", flush=True)
        print(f"  using [{idx}] {name}", flush=True)
        self.log("device_selected", index=idx, name=name, requested=want,
                 devices=[f"[{i}] {n}" for i, n in devs])

        self.ear = Ear(idx, self.log, self.args.verbose)
        self.ear.start()
        # The liveness test runs on the first LIVENESS_S of our own stream:
        # 2.79 ms of Python, against 7-10 s and two more device opens for
        # spawning miccheck.  WARMUP_S is discarded AFTER the first byte, never
        # from process start -- the first 100 ms carries a device-open transient
        # at ~100x the settled floor.
        raw = self.ear.read_seconds(WARMUP_S + LIVENESS_S, timeout=30.0)
        dead, err = self.ear.died()
        if len(raw) == 0:
            print(f"  NO AUDIO ARRIVED. ffmpeg said: {err or '(nothing)'}",
                  file=sys.stderr)
            return False
        tail = raw[int(RATE * WARMUP_S) * WIDTH:]
        st = _mic.analyse(tail)
        verdict, why = _mic.classify(st)
        print(f"  liveness: {verdict}  peak={st['peak']} "
              f"rms={st['rms']:.1f} zero={st['zero_frac']*100:.4f}%  ({why})",
              flush=True)
        self.log("liveness", verdict=verdict, peak=st["peak"],
                 rms=round(st["rms"], 1), zero_frac=round(st["zero_frac"], 6))
        if verdict == "DEAD":
            print("\n" + _mic.REMEDY, file=sys.stderr)
            print("This daemon will NOT enter its main loop against a digital-zero "
                  "stream:\ntranscribing silence in silence is the failure nobody "
                  "can debug afterwards.\nRun  python3 voice/miccheck.py --quiet  "
                  "to find a device that works,\nor  python3 voice/jarvis.py --text "
                  "  to drive the identical pipeline by typing.",
                  file=sys.stderr)
            return False
        return True

    # -- shutdown ---------------------------------------------------------
    def shutdown(self) -> None:
        """SIGINT must leave nothing holding the microphone."""
        self.stop.set()
        self.mouth.shutup()
        if self.ear is not None:
            self.ear.close()
        for t in (self.wake_tier, self.cmd_tier):
            if t is not None:
                t.close()
        self.log("shutdown", spend_usd=round(self.agent.spend, 4),
                 dropped_chunks=(self.ear.drops if self.ear else 0),
                 delivered_bytes=(self.ear.delivered_bytes if self.ear else 0),
                 read_bytes=(self.ear.total_bytes if self.ear else 0))
        with self._loglock:
            if self._logfh is not None:
                self._logfh.close()
                self._logfh = None

    # -- the shared path --------------------------------------------------
    def handle_text(self, raw: str, source: str) -> None:
        """Normalise, gate, route, speak.  --text and audio share this."""
        text = normalise(raw)
        self.log("utterance", source=source, raw=raw, normalised=text)
        if not text:
            return

        kind, payload = gate(text)

        if kind == "cancel":
            self.mouth.shutup()
            self.state, self.pending = "IDLE", None
            self.mouth.say(PHRASES["cancelled"], key="cancelled")
            self.log("cancelled", source=source)
            return

        if kind == "refuse":
            self.mouth.say(PHRASES[payload], key=payload)
            self.log("refused", text=text, phrase=payload)
            return

        if kind == "confirm":
            label, cmd = payload
            self.pending, self.pending_label = text, label
            self.state = "CONFIRM"
            self.log("confirm_required", text=text, label=label,
                     command=cmd.id if cmd else None)
            # THE CLOCK STARTS WHEN THE QUESTION HAS BEEN SPOKEN, NOT BEFORE IT.
            # wait=True, then armed_until: the question is dynamic text, so it
            # pays for a live `say`, and the rendered audio for the stop_the_run
            # question alone is 5.119 s against what used to be a 10 s window
            # that had already started running.  See CONFIRM_TIMEOUT_S.
            self.mouth.say(f"That would {label}. Say jarvis confirm if you "
                           f"mean it.", wait=True)
            self.armed_until = time.monotonic() + CONFIRM_TIMEOUT_S
            return

        self.route(text, source, payload)

    def resolve_confirmation(self, text: str, source: str,
                             require_wake: bool = False) -> None:
        """Release or drop the pending destructive command.

        require_wake is True for anything heard through the microphone.  The
        CONFIRM branch of handle_segment used to act on ANY segment the VAD
        emitted, with no wake word, while the test for approval was a substring
        search for "confirm" -- so a bystander's "can you confirm the numbers
        for Friday", or the daemon's own question coming back off the speakers,
        released a pending pkill or agent dispatch.  There is no speaker
        identification in this file and there is not going to be, so the
        defence is that the WHOLE utterance must be the wake word followed by
        the word confirm and nothing else.
        """
        pending, label = self.pending, self.pending_label
        self.state, self.pending, self.pending_label = "IDLE", None, None
        if pending is None:
            return
        heard = text
        if require_wake:
            hit, rest, _, _ = wake_match(text, self.args.wake)
            if not hit:
                self.log("confirmation_no_wake", text=pending, label=label,
                         heard=text)
                self.mouth.say(PHRASES["not_confirmed"], key="not_confirmed")
                return
            text = rest
        if is_confirmation(text):
            self.log("confirmed", text=pending, label=label, heard=heard)
            self.route(pending, source + "+confirmed")
        else:
            # Discarded, never queued: a destructive command that fires later,
            # out of context, is worse than one that never ran.
            self.log("confirmation_refused", text=pending, label=label,
                     heard=heard)
            self.mouth.say(PHRASES["not_confirmed"], key="not_confirmed")

    def route(self, text: str, source: str,
              cmd: "Command | None" = None) -> None:
        """Registry first, agent otherwise.

        The registry is a FAST PATH, never a gate: an utterance that matches
        nothing here is still answered, by the agent, as free text.  That is what
        keeps adding an entry a pure optimisation -- it can make an answer
        cheaper and more predictable, and it cannot make a question unanswerable.
        """
        if cmd is None:
            cmd = match_command(text)

        if cmd is None:
            self.log("route", source=source, text=text, command=None,
                     kind="agent-freetext")
            self.dispatch(text, source)
            return

        if cmd.kind == "prompt":
            prompt = cmd.prompt.replace("{text}", text)
            self.log("route", source=source, text=text, command=cmd.id,
                     kind="agent-template")
            self.dispatch(prompt, source + f"+{cmd.id}")
            return

        self.run_local(cmd, text, source)

    def run_local(self, cmd: "Command", text: str, source: str) -> None:
        """Run a registry argv to completion under a wall-clock cap.

        Nothing is backgrounded.  A detached solver on this Mac is QoS-throttled
        to about 14 % CPU -- validation/poiseuille, moved to the background, ran
        fifteen minutes for work ctest records at 14.3 s -- so waiting is both
        simpler and faster.  The cost is that the daemon is deaf while it waits,
        which is why `slow` entries say so before they start.
        """
        self.log("run_local", source=source, text=text, command=cmd.id,
                 argv=cmd.argv, cwd=cmd.cwd, timeout_s=cmd.timeout_s)

        if self.args.dry_run:
            print("[dry-run] would run locally:", flush=True)
            for i, a in enumerate(cmd.argv):
                print(f"    argv[{i}] = {a!r}", flush=True)
            print(f"    cwd  = {cmd.cwd or os.getcwd()}", flush=True)
            print(f"    cap  = {cmd.timeout_s:.0f} s", flush=True)
            self.mouth.say(f"Dry run. {cmd.id} was not started.")
            return

        if cmd.say:
            self.mouth.say(cmd.say)
        elif cmd.slow:
            self.mouth.say(f"Running {cmd.id}. This takes a while.")

        t0 = time.monotonic()
        rc, out, err, timed_out = run_capped(
            cmd.argv, cap=cmd.timeout_s, cwd=cmd.cwd, env=self.agent.env())
        dt = time.monotonic() - t0

        ok, reply = verdict(cmd, rc, out, err, timed_out)
        self.log("local_result", command=cmd.id, rc=rc, timed_out=timed_out,
                 seconds=round(dt, 1), ok=ok, stdout=(out or "")[:4000],
                 stderr=(err or "")[:2000])
        self.mouth.say(reply)
        self.log("spoken", ok=ok, text=reply[:2000])

    def dispatch(self, text: str, source: str) -> None:
        if not self.agent_ok:
            # Said at the moment a question needs the agent, not at startup:
            # this is the point at which the user can act on it.
            self.log("dispatch_unavailable", source=source, text=text)
            self.mouth.say("I can only run the commands I already know. Claude "
                           "Code is not available, so I cannot answer that one.")
            return
        self.log("dispatch", source=source, text=text,
                 allow_edits=self.args.allow_edits,
                 session_id=self.agent.session_id)
        if not self.args.dry_run:
            self.mouth.say(PHRASES["working"], key="working")
        ok, reply = self.agent.dispatch(text)
        self.mouth.say(reply)
        self.log("spoken", ok=ok, text=reply[:2000])

    # -- audio loop -------------------------------------------------------
    def handle_segment(self, pcm: bytes, voiced_frames: int = 0,
                       last_frame_rms: float = 0.0) -> None:
        assert self.wake_tier is not None and self.cmd_tier is not None
        ratio = self.ear.realtime_ratio() if self.ear else 1.0
        secs = dur_s(pcm)
        voiced_ms = voiced_frames * FRAME_MS

        # THE GAPPED-STREAM REFUSAL COMES FIRST, FOR EVERY STATE, BECAUSE THE
        # ONE WORD WHISPER MUST NOT INVENT WAS THE ONE WORD EXEMPT FROM IT.
        # The CONFIRM branch used to return above this check, so RATE_REFUSE --
        # whose whole justification in this file's banner is "a sentence
        # dispatched to a coding agent must not be invented" -- never ran on a
        # confirmation.  The banner also records that a gapped stream is the
        # NORMAL state of this Mac under load (0.682x on the first real run,
        # a third of the audio missing), i.e. exactly the condition in which a
        # solve is running and a "stop the run" is being confirmed.  A pending
        # destructive command is DROPPED here rather than left armed, so a
        # second mis-transcription cannot land on it inside the window.
        if ratio < RATE_REFUSE:
            if self.state == "CONFIRM":
                self.pending = self.pending_label = None
                self.state = "IDLE"
                self.log("confirm_dropped_gapped", realtime=round(ratio, 3))
                self.mouth.say(PHRASES["confirm_gapped"], key="confirm_gapped")
                return
            self._say_gapped(ratio)
            self.state = "IDLE"
            return

        # ACOUSTIC EVIDENCE BEFORE ANY TRANSCRIPT IS TRUSTED.  whisper does not
        # return nothing for a segment with no speech in it; tiny.en returns a
        # fixed English sentence, reproducibly (see is_hallucination), and when
        # that sentence begins with something like "Jarvis" it is a FALSE WAKE
        # that builds a claude argv out of room noise.  The VAD already knows
        # how many frames were above the adapted floor by SPEECH_FACTOR; a
        # segment that cannot muster MIN_SEG_MS of them never reaches whisper.
        if voiced_frames and voiced_ms < MIN_SEG_MS:
            self.log("segment_dropped", reason="not enough voiced audio",
                     voiced_ms=voiced_ms, seconds=round(secs, 2))
            return

        if self.state == "IDLE":
            heard = self.wake_tier.transcribe(pcm, prompt=None)
            if not heard:
                return
            halluc, sim, which = is_hallucination(heard)
            if halluc:
                self.log("segment_dropped", reason="silence hallucination",
                         similarity=sim, matched=which, seconds=round(secs, 2))
                return
            hit, rest, best, misses = wake_match(heard, self.args.wake)
            # WHAT IS LOGGED WHEN THE WAKE WORD DID NOT HIT IS METADATA, NOT
            # THE TRANSCRIPT.  Every segment that clears the energy VAD reaches
            # this line -- a phone call, a meeting, a colleague, a podcast --
            # and `heard` used to be written to voice/logs/*.jsonl for all of
            # them, unrotated and uncapped.  near_misses and the ratio are what
            # the cutoff-tuning argument in the banner actually needs; the
            # sentence itself is not.  --log-all-audio restores it and says so
            # at startup.
            if hit or self.args.log_all_audio:
                self.log("wake_check", heard=heard, hit=hit, best=round(best, 3),
                         near_misses=misses, seconds=round(secs, 2),
                         voiced_ms=voiced_ms, realtime=round(ratio, 3),
                         last_frame_rms=round(last_frame_rms, 1))
            else:
                self.log("wake_check", hit=False, best=round(best, 3),
                         near_misses=misses, seconds=round(secs, 2),
                         voiced_ms=voiced_ms, realtime=round(ratio, 3),
                         last_frame_rms=round(last_frame_rms, 1),
                         words=len(heard.split()))
            if not hit:
                return
            # A one-shot "jarvis, run the fast tests" is re-transcribed on the
            # ACCURATE tier rather than trusting tiny.en's version of the
            # command: tiny.en cannot say ctest or tau, and this costs 1.1 s
            # against making the user repeat themselves.
            if len(rest.split()) >= 2:
                better = self.cmd_tier.transcribe(pcm, prompt=self.prompt())
                _, rest2, _, _ = wake_match(better, self.args.wake)
                cmd = rest2 if len(rest2.split()) >= 2 else rest
                self.mouth.chime()
                self.handle_text(cmd, "voice-oneshot")
                return
            self.state = "ARMED"
            self.armed_until = time.monotonic() + ARM_TIMEOUT_S
            self.mouth.chime()
            self.log("armed", best=round(best, 3))
            return

        text = self.cmd_tier.transcribe(pcm, prompt=self.prompt())
        if not text:
            return
        halluc, sim, which = is_hallucination(text)
        if halluc:
            self.log("segment_dropped", reason="silence hallucination",
                     similarity=sim, matched=which, state=self.state)
            return
        if self.state == "CONFIRM":
            self.resolve_confirmation(normalise(text), "voice",
                                      require_wake=True)
            return
        self.state = "IDLE"
        self.handle_text(text, "voice")

    def prompt(self) -> str:
        return self.args.prompt_text

    def _say_gapped(self, ratio: float) -> None:
        self.log("refused_gapped_stream", realtime=round(ratio, 3))
        self.mouth.say(PHRASES["gapped"], key="gapped")
        # The spoken phrase is pre-rendered, so it cannot carry the number.
        # Print it, with the command that measures it independently: on this
        # machine the sustained rate was 0.72-0.74x from raw ffmpeg with no
        # Python in the loop, i.e. the refusal is correct and the fault is
        # outside this repository.
        dev = self.ear.index if self.ear else 0
        print(f"  ! refused: capture is at {ratio:.2f}x real time "
              f"({100 * (1 - ratio):.0f}% of the audio is missing).\n"
              f"    Measure it on its own:  python3 voice/captureprobe.py "
              f"--device ':{dev}'", file=sys.stderr)

    def loop_audio(self) -> int:
        assert self.ear is not None
        vad = EnergyVAD(MAX_SEG_WAKE_S)
        partial = bytearray()
        gated_before = False
        self.mouth.say(PHRASES["ready"], key="ready")

        while not self.stop.is_set():
            dead, err = self.ear.died()
            if dead:
                self.log("capture_died", error=err)
                print(f"\nthe capture stream ended: {err}\n"
                      "A device that disappears fails loudly like this; it is NOT "
                      "silence.\nRe-run to re-enumerate.", file=sys.stderr)
                return 1

            # The startup liveness test, continued for the life of the run.
            zr = self.ear.zero_run_seconds()
            if zr >= MIC_DEAD_S:
                self.log("mic_went_dead", zero_run_s=round(zr, 1))
                self.mouth.say(PHRASES["mic_dead"], key="mic_dead")
                print(f"\nTHE MICROPHONE WENT SILENT {zr:.0f} s AGO and has not "
                      f"recovered.\nThis is a digital-zero stream, not a quiet "
                      f"room: every sample since has been\nexactly zero, which a "
                      f"room cannot produce.\n", file=sys.stderr)
                print(_mic.REMEDY if _mic is not None else
                      "Re-run  python3 voice/miccheck.py  to find a live device.",
                      file=sys.stderr)
                self.mouth.drain()
                return 3
            if zr >= MIC_WARN_S and not self.ear.zero_warned:
                self.ear.zero_warned = True
                self.log("mic_silent", zero_run_s=round(zr, 1))
                print(f"  [warning] the stream has been digital zero for "
                      f"{zr:.0f} s; stopping at {MIC_DEAD_S:.0f} s",
                      file=sys.stderr, flush=True)

            if self.state in ("ARMED", "CONFIRM") and \
                    time.monotonic() > self.armed_until:
                self.log("arm_timeout", state=self.state,
                         dropped=self.pending)
                was = self.state
                self.state, self.pending = "IDLE", None
                self.pending_label = None
                # A CONFIRMATION WINDOW THAT CLOSES SAYS SO.  It used to expire
                # in silence: the state flipped back to IDLE, the next segment
                # was wake-checked, no wake word was found, and nothing was
                # said -- so the user was left waiting on an answer to a
                # question the daemon had already withdrawn.
                if was == "CONFIRM":
                    self.mouth.say(PHRASES["confirm_expired"],
                                   key="confirm_expired")
                # reset(), not a new VAD: the adapted floor is the whole point of
                # tracking one, and rebuilding it would re-learn the room over the
                # next three seconds for no reason.
                vad.reset()

            try:
                chunk = self.ear.q.get(timeout=0.2)
            except queue.Empty:
                continue

            # Self-trigger guard: while the daemon is making noise, and for a
            # tail afterwards, the VAD is closed and its buffers are cleared, so
            # the end of its own sentence cannot become the start of the next
            # utterance.
            if self.mouth.busy():
                gated_before = True
                partial.clear()
                continue
            if gated_before:
                gated_before = False
                vad.reset()

            vad.max_seg_s = (MAX_SEG_CMD_S if self.state in ("ARMED", "CONFIRM")
                             else MAX_SEG_WAKE_S)
            partial += chunk
            while len(partial) >= FRAME_BYTES:
                frame = bytes(partial[:FRAME_BYTES])
                del partial[:FRAME_BYTES]
                seg = vad.push(frame)
                if seg is None:
                    continue
                ratio = self.ear.realtime_ratio()
                if ratio < RATE_WARN and time.monotonic() - self._rate_warned > 60:
                    self._rate_warned = time.monotonic()
                    self.log("stream_slow", realtime=round(ratio, 3),
                             dropped_chunks=self.ear.drops,
                             delivered_bytes=self.ear.delivered_bytes)
                    print(f"  ! capture running at {ratio:.2f}x real time; "
                          f"{100*(1-ratio):.0f}% of the audio is missing",
                          file=sys.stderr)
                if self.args.verbose:
                    print(f"  segment {dur_s(seg):.2f} s  voiced="
                          f"{vad.last_voiced_frames * FRAME_MS} ms  exit="
                          f"{vad.last_exit}  floor="
                          f"{(vad.floor or 0):.1f}  thr={vad.threshold():.1f}  "
                          f"state={self.state}  rt={ratio:.2f}x", flush=True)
                self.handle_segment(seg, vad.last_voiced_frames, vad.last_rms)
                # Anything that arrived while whisper or claude was working is
                # stale; drop it rather than transcribing the past.
                if self.state == "IDLE":
                    vad.reset()
                    partial.clear()
                    while True:
                        try:
                            self.ear.q.get_nowait()
                        except queue.Empty:
                            break
        return 0

    def loop_text(self) -> int:
        """The identical path, driven by typing.

        This is not a toy mode: the built-in microphone on this machine has
        returned digital zero all morning, so this is the only path that can be
        exercised end to end today, and it is the one the regression harness
        uses.

        STDIN IS READ ON A THREAD, FOR THE SAME REASON loop_audio READS A QUEUE:
        `for line in sys.stdin` CANNOT BE INTERRUPTED BY A SIGNAL.  Under PEP
        475 the read is restarted after the handler returns, so with a handler
        that only sets an Event -- which is what this file has, because the
        children are in their own process groups -- neither SIGINT nor SIGTERM
        ended the process.  Measured: started `--text --dry-run --no-speak` with
        stdin on a FIFO that stays open (exactly what a terminal is), sent
        SIGINT and waited 3 s: alive, STAT SN.  A second SIGINT and then a
        SIGTERM: still alive 2 s later.  Only SIGKILL ended it, and the printed
        "reaping children..." was false three times over -- run()'s
        `finally: self.shutdown()` never ran, so the jsonl handle was never
        closed and no `shutdown` record was ever written.  Ctrl-C in the mode
        the banner calls "the one the regression harness uses" left a stuck
        terminal.

        The stdin thread is a daemon thread, so an EOF-less stdin does not keep
        the process alive once this loop returns.
        """
        print("text mode: one command per line, Ctrl-D to quit. "
              "The wake word is optional here.", flush=True)
        lines: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            try:
                for ln in sys.stdin:
                    lines.put(ln)
            except Exception:
                pass
            lines.put(None)

        threading.Thread(target=reader, daemon=True, name="stdin").start()

        while not self.stop.is_set():
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            if self.state == "CONFIRM":
                # No wake word is required when the confirmation was TYPED:
                # the keyboard establishes who is speaking in a way the room
                # cannot.  The whole line must still BE the confirmation.
                self.resolve_confirmation(normalise(line), "text")
                continue
            hit, rest, _, _ = wake_match(line, self.args.wake)
            text = rest if (hit and rest) else line
            self.handle_text(text, "text")
            if self.state == "CONFIRM":
                print("confirm? (type: confirm)", flush=True)
        return 0

    # -- entry ------------------------------------------------------------
    def run(self) -> int:
        self.log("start", argv=sys.argv, repo=str(self.repo),
                 text_mode=self.args.text, dry_run=self.args.dry_run,
                 allow_edits=self.args.allow_edits)
        print(f"M3LB voice assistant -- repo {self.repo}", flush=True)
        print(f"  log {self.logpath}  (kept {LOG_RETENTION_DAYS} days, "
              f"rolled at {LOG_MAX_BYTES // (1024 * 1024)} MB)", flush=True)
        if self.args.log_all_audio:
            print("  ! --log-all-audio: EVERY transcript is written to that "
                  "file, including\n    segments with no wake word in them. "
                  "This daemon hears the whole room.",
                  file=sys.stderr)

        problems = self.preflight_tools()
        if problems:
            print("\nSTARTUP SELF-CHECK FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  * {p}", file=sys.stderr)
            self.log("preflight_failed", problems=problems)
            return 2

        if not self.args.no_speak:
            # Off the startup path: `say -o` costs 3.5-6.5 s per phrase measured,
            # so rendering all of PHRASES serially here would be half a minute of
            # dead air before the daemon listens.  It renders in the background and
            # anything not yet cached falls back to live `say`, which is slower per
            # utterance but available immediately.  `say -o` writes a file and never
            # opens the output device, so it cannot collide with playback.
            threading.Thread(
                target=lambda: [self.mouth.prerender(k, t)
                                for k, t in PHRASES.items()],
                daemon=True, name="prerender").start()

        try:
            if self.args.text:
                if not self.args.dry_run:
                    print("  (no audio is opened in --text mode)", flush=True)
                return self.loop_text()
            if not self.start_whisper():
                return 2
            if not self.start_audio():
                return 2
            return self.loop_audio()
        finally:
            self.shutdown()


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jarvis.py",
        description="Wake-word voice front end for the M3LB repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python3 voice/jarvis.py --text --dry-run\n"
            "      the whole pipeline with no microphone and no agent call\n"
            "  echo 'jarvis, run the fast tests' | python3 voice/jarvis.py "
            "--text --dry-run\n"
            "  python3 voice/jarvis.py --device 'MacBook Air Microphone'\n"
            "      resolve by NAME: indices renumber on every plug/unplug\n"
            "  python3 voice/miccheck.py --quiet\n"
            "      find a microphone that is not returning digital zero\n"))
    ap.add_argument("--text", action="store_true",
                    help="read typed lines from stdin instead of audio, through "
                         "the identical dispatch path")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the claude argv instead of running it")
    ap.add_argument("--device", default=None,
                    help="microphone NAME or substring; REQUIRED unless --text. "
                         "An index is accepted and warned about, and a virtual "
                         "or conference device must be named exactly")
    ap.add_argument("--wake", default="jarvis", help="wake word (default: jarvis)")
    ap.add_argument("--verbose", action="store_true",
                    help="per-segment VAD and transcription detail")
    ap.add_argument("--model-dir", default=str(MODEL_DIR),
                    help=f"whisper ggml models (default: {MODEL_DIR})")
    ap.add_argument("--no-speak", action="store_true",
                    help="print replies instead of speaking them")

    ap.add_argument("--wake-model", default=None,
                    help="wake-tier model (default: <model-dir>/ggml-tiny.en.bin)")
    ap.add_argument("--command-model", default=None,
                    help="command-tier model (default: <model-dir>/"
                         "ggml-small.en.bin; set it equal to the wake model to "
                         "run a single server)")
    ap.add_argument("--port-base", type=int, default=18080,
                    help="wake tier on this port, command tier on the next; if "
                         "either is busy a free port is chosen instead")
    ap.add_argument("--adopt-whisper", action="store_true",
                    help="USE a whisper-server that is already listening on the "
                         "port instead of starting one. Off by default: whatever "
                         "owns that port would choose every word I hear")
    ap.add_argument("--log-all-audio", action="store_true",
                    help="write the TRANSCRIPT of every segment to the log, "
                         "including ones with no wake word in them. Off by "
                         "default: this daemon hears the whole room")
    ap.add_argument("--threads", type=int, default=2,
                    help="whisper threads per tier (default 2: the user runs "
                         "solvers on this machine and -t 4 fights them)")
    ap.add_argument("--prompt-file", default=None,
                    help="override the initial prompt; it must be PROSE, not a "
                         "keyword list -- a comma list measurably destroys the "
                         "transcript")

    ap.add_argument("--vocabulary", default=str(VOCAB_PATH),
                    help=f"whisper prompt + alias table (default: {VOCAB_PATH})")
    ap.add_argument("--commands", default=str(COMMANDS_PATH),
                    help=f"spoken command registry (default: {COMMANDS_PATH})")
    ap.add_argument("--selftest", action="store_true",
                    help="check both data files and the routing table, then "
                         "exit; opens no microphone and runs nothing")

    ap.add_argument("--repo", default=str(_HERE.parent),
                    help="working directory for the agent (pinned; -p skips the "
                         "workspace-trust dialog)")
    ap.add_argument("--claude-bin", default="claude", help="the claude CLI")
    ap.add_argument("--allow-edits", action="store_true",
                    help="let the agent write files (acceptEdits + Edit,Write). "
                         "Off by default so one mis-heard word cannot rewrite a "
                         "source file")
    ap.add_argument("--claude-timeout", type=float, default=CLAUDE_TIMEOUT_S,
                    help="wall-clock cap per turn in seconds (there is no GNU "
                         "timeout here and --max-turns does not bound time)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--budget", type=float, default=0.50,
                    help="--max-budget-usd per turn")
    ap.add_argument("--voice", default="Daniel",
                    help="say voice; only Daniel, Samantha, Alice, Alex, Fred and "
                         "Albert are installed here")
    ap.add_argument("--speech-rate", type=int, default=190)
    return ap


def selftest() -> int:
    """Check the data files and the routing table.  Runs nothing.

    Three things are checked, and the second is the one that will actually catch
    a mistake.  Loading proves every regex compiles and every token is known.
    ROUTING proves the file's ORDER is right: each entry's own examples must
    reach that entry and not an earlier one, which is the failure that would
    otherwise appear months later as "it ran the whole suite when I asked for the
    fast ones".  The last checks argv[0] exists, as a warning only -- the build
    tree is not required to be present to edit the registry.
    """
    ok = True
    print(f"vocabulary  {len(_ALIAS_RULES)} alias rules, "
          f"prompt {len(WHISPER_PROMPT.split())} words")
    print(f"commands    {len(REGISTRY)} entries, {len(PHRASES)} phrases, "
          f"{len(DESTRUCTIVE)} destructive patterns, "
          f"{len(REFUSALS)} refusals")

    for c in REGISTRY:
        if not c.examples:
            print(f"  ! {c.id}: no examples, so its ordering is unchecked")
            ok = False
    print()

    for c in REGISTRY:
        for ex in c.examples:
            text = normalise(ex)
            got = match_command(text)
            gid = got.id if got else None
            mark = "ok  " if gid == c.id else "FAIL"
            if gid != c.id:
                ok = False
            print(f"  {mark} {c.id:<26} {ex!r}"
                  + ("" if gid == c.id else f"   -> {gid}"))

    # THE NEGATIVE HALF.  The loop above can only see an entry failing to catch
    # its own phrasing; it is structurally blind to an entry catching somebody
    # else's, which is how "blow away the build tree" came to run `cmake
    # --build` and "scrap the uncommitted work" came to run `git status`.
    print()
    for c in REGISTRY:
        for ex in c.counterexamples:
            got = match_command(normalise(ex))
            gid = got.id if got else None
            good = gid != c.id
            ok = ok and good
            print(f"  {'ok  ' if good else 'FAIL'} {c.id:<26} NOT {ex!r}"
                  + ("" if good else "   -> it matched anyway"))

    print()
    for c in REGISTRY:
        if c.kind != "argv":
            continue
        exe = c.argv[0]
        found = shutil.which(exe) or (os.path.exists(exe) and exe)
        if not found:
            print(f"  ! {c.id}: {exe} not found "
                  f"(build it, or it is not on PATH)")

    # THE GATE, THROUGH THE REAL gate() AND NOT A COPY OF ITS RULES.  An earlier
    # version of this block re-implemented the ordering and therefore could not
    # see that "stop the run" never reached its entry at all: the cancel rule
    # matched it three lines earlier and the daemon said "Cancelled." Every
    # probe below is a full utterance and the expected verdict is written out.
    print()
    probes: list[tuple[str, str, str | None]] = [
        ("stop",                       "cancel",  None),
        ("never mind",                 "cancel",  None),
        ("stop the run",               "confirm", "stop_the_run"),
        ("kill the solver",            "confirm", "stop_the_run"),
        ("delete the build directory", "confirm", None),
        ("push to origin",             "confirm", None),
        ("write this up",              "confirm", "write_up"),
        ("run tgv3d on D3Q19",         "refuse",  None),
        ("run the fast tests",         "route",   "fast_tests"),
        ("what is the Reynolds number here", "route", None),
        # The destructive synonyms the word list did not have.  They are a
        # COURTESY FOR OBVIOUS PHRASING and not the security boundary -- see the
        # banner and commands.json -- but the two at the top were live
        # mis-routes into a real command, which is worse than a miss.
        ("blow away the build tree",   "confirm", None),
        ("scrap the uncommitted work", "confirm", None),
        ("get rid of the build directory", "confirm", None),
        ("throw away my changes",      "confirm", None),
        ("undo my changes",            "confirm", None),
        ("empty the build directory",  "confirm", None),
        ("purge the geom cache",       "confirm", None),
        ("clear the results directory", "confirm", None),
        ("roll back that commit",      "confirm", None),
        ("rename the voice directory", "confirm", None),
        # And the shape no word list can catch, kept here as a standing
        # reminder that it reaches the agent with no confirmation at all.
        ("the build directory is stale, sort it out", "route", None),
        ("what is running",            "route",   "machine_load"),
    ]
    for probe, want_kind, want_id in probes:
        kind, payload = gate(normalise(probe))
        if kind == "confirm":
            label, cmd = payload
            got_id, detail = (cmd.id if cmd else None), label
        elif kind == "route":
            got_id = payload.id if payload else None
            detail = f"{payload.kind} {got_id}" if payload else "agent free text"
        else:
            got_id, detail = None, (payload or "")
        good = kind == want_kind and got_id == want_id
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {probe!r:<38} -> {kind}"
              f"  {detail}"
              + ("" if good else f"   WANTED {want_kind} {want_id}"))

    ok = confirm_selftest() and ok
    ok = vad_selftest() and ok

    print("\nSELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def confirm_selftest() -> bool:
    """The release word, against the sentences that used to release it.

    Every APPROVE line below was measured approving under the old substring
    test, and each one is either an explicit refusal or a sentence from an
    unrelated conversation.  The DROP lines are the words a user actually
    reaches for, which the old test also got wrong -- in the safe direction,
    but wrong -- and they stay dropped deliberately: one word, spoken, is the
    whole protocol, and widening it is how a bystander gets back in.
    """
    print()
    cases = [
        ("confirm",                        True),
        ("Confirm.",                       True),
        ("ok confirm",                     True),
        ("okay, confirmed",                True),
        ("no, don't confirm that",         False),
        ("no do not confirm that",         False),
        ("I cannot confirm that",          False),
        ("can you confirm the numbers for Friday", False),
        ("confirm the meeting time with Sarah", False),
        ("that would kill every solver process running out of this build "
         "tree say jarvis confirm if you mean it", False),
        ("yes",                            False),
        ("go ahead",                       False),
        ("stop",                           False),
    ]
    ok = True
    for text, want in cases:
        got = is_confirmation(normalise(text))
        good = got == want
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} confirm {text!r:<58} -> "
              f"{'APPROVE' if got else 'drop'}"
              + ("" if good else "   WANTED the other one"))
    return ok


def _tone(rms: float, frames: int, seed: int = 1) -> list[bytes]:
    """`frames` frames of pseudo-random noise at approximately this r.m.s.

    Deterministic, and stdlib only: there is no numpy on this machine and there
    is not going to be.  Uniform noise on [-a, a] has r.m.s. a/sqrt(3).
    """
    import random
    rng = random.Random(seed)
    a = int(rms * (3 ** 0.5))
    out = []
    for _ in range(frames):
        arr = array.array("h", [max(-32768, min(32767, rng.randint(-a, a)))
                                for _ in range(FRAME_BYTES // WIDTH)])
        out.append(arr.tobytes())
    return out


def vad_selftest() -> bool:
    """The VAD against a STEP in room noise, which is what used to lock it.

    A ramp cannot see this failure -- the floor ratchets up perfectly well when
    the level rises gradually -- so the test steps.  Three checks: the floor
    adapts to the new room within a few seconds of a step; a 96 ms burst (the
    cough this MIN_SEG_MS exists to reject) emits nothing; and an utterance in
    a quiet room still emits exactly one segment.
    """
    print()
    ok = True

    # 1. STEP.  Quiet room, then a jump to rms 61.1 -- the measured crossover
    #    was 59.5 adapting and 61.1 locking, so this is the failing case and
    #    not a comfortable margin.
    vad = EnergyVAD(MAX_SEG_WAKE_S)
    for f in _tone(5.0, 600, seed=2):          # 19.2 s of quiet room
        vad.push(f)
    settled = vad.floor or 0.0
    segs = 0
    for f in _tone(61.1, 3000, seed=3):        # 96 s of steady noise
        if vad.push(f) is not None:
            segs += 1
    adapted = vad.threshold() > 61.1 and not vad.active
    ok = ok and adapted
    print(f"  {'ok  ' if adapted else 'FAIL'} vad step 5 -> 61.1 rms: floor "
          f"{settled:.2f} -> {vad.floor:.2f}, threshold "
          f"{vad.threshold():.1f}, {segs} segment(s) emitted, "
          f"active={vad.active}"
          + ("" if adapted else "   WANTED the threshold above the new room"))

    # 2. COUGH.  Three voiced frames is the onset count, so this is the
    #    shortest thing that can open a segment at all.
    vad = EnergyVAD(MAX_SEG_WAKE_S)
    for f in _tone(5.0, 300, seed=4):
        vad.push(f)
    emitted = None
    for f in _tone(4000.0, 3, seed=5):         # 96 ms burst
        emitted = vad.push(f) or emitted
    for f in _tone(5.0, 40, seed=6):           # let the hangover close it
        emitted = vad.push(f) or emitted
    good = emitted is None
    ok = ok and good
    print(f"  {'ok  ' if good else 'FAIL'} vad 96 ms burst rejected as a cough"
          + ("" if good else
             f"   -> emitted {dur_s(emitted) * 1000:.0f} ms"))

    # 3. UTTERANCE.  1.5 s of speech in a quiet room, one segment, and the
    #    pre-roll ahead of it.
    vad = EnergyVAD(MAX_SEG_WAKE_S)
    for f in _tone(5.0, 300, seed=7):
        vad.push(f)
    got = []
    for f in _tone(4000.0, 47, seed=8):        # 1.5 s
        s = vad.push(f)
        if s:
            got.append(s)
    for f in _tone(5.0, 40, seed=9):
        s = vad.push(f)
        if s:
            got.append(s)
    good = len(got) == 1 and vad.last_voiced_frames >= 47
    ok = ok and good
    lead = ((len(got[0]) // FRAME_BYTES) - 47 - HANGOVER_FRAMES) * FRAME_MS \
        if got else 0
    print(f"  {'ok  ' if good else 'FAIL'} vad 1.5 s utterance -> {len(got)} "
          f"segment(s), {vad.last_voiced_frames * FRAME_MS} ms voiced, "
          f"{lead} ms of lead-in"
          + ("" if good else "   WANTED exactly one"))
    return ok


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    md = pathlib.Path(args.model_dir).expanduser()
    args.model_dir = str(md)
    args.wake_model = args.wake_model or str(md / "ggml-tiny.en.bin")
    args.command_model = args.command_model or str(md / "ggml-small.en.bin")

    # The data files are loaded BEFORE anything else, because everything else --
    # the prompt, the alias table, the phrases the daemon pre-renders, the
    # routing -- comes out of them.  A failure here is fatal and names the entry:
    # a daemon that started with half a registry would be indistinguishable from
    # one that mis-heard the user.
    repo = pathlib.Path(args.repo).resolve()
    try:
        load_data(pathlib.Path(args.vocabulary), pathlib.Path(args.commands),
                  repo, args.threads)
    except DataError as exc:
        print(f"voice data: {exc}", file=sys.stderr)
        return 2

    if args.selftest:
        return selftest()

    if args.prompt_file:
        args.prompt_text = pathlib.Path(args.prompt_file).read_text(
            encoding="utf-8").strip()
    else:
        args.prompt_text = WHISPER_PROMPT

    daemon = VoiceDaemon(args)

    def on_signal(signum, _frame):
        # Children are in their own process groups, so a terminal Ctrl-C does not
        # reach them; reaping is this handler's job and nothing else's.
        #
        # AND IT HAS TO REAP THEM HERE, NOT LATER.  Setting the Event alone
        # shortened nothing: measured, `run_capped(["sleep","6"], cap=20.0)`
        # with a flag-only handler fired at t = 1.01 s returned at t = 6.02 s,
        # because communicate(timeout=...) retries across EINTR (PEP 475).  The
        # same holds for urlopen against whisper (60 s, 180 s for the warm-up)
        # and for wait_ready's poll loop.  So Ctrl-C during a claude turn was
        # ignored for up to --claude-timeout -- 300 s by default -- while
        # ffmpeg still held the microphone, and the message below was a lie.
        # Killing the children is what makes every blocking call return.
        daemon.stop.set()
        n = reap_all(grace=1.0)
        print(f"\nsignal {signum}: reaped {n} child process(es).",
              file=sys.stderr)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        return daemon.run()
    except KeyboardInterrupt:
        daemon.shutdown()
        return 130


if __name__ == "__main__":
    sys.exit(main())
