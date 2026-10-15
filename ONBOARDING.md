# Working on M3LB with Claude Code

For a new group member. This is about *how to work here with Claude Code*;
[GETTING_STARTED.md](GETTING_STARTED.md) is about running the physics, and
[CLAUDE.md](CLAUDE.md) is the long reference that Claude reads at the start of every
session (read it yourself too, at least the section headings).

M3LB is a Lattice Boltzmann solver in two independent codebases, `src/` (C++20 on
Kokkos, the full physics) and `GPU/` (plain CUDA, shares no headers with `src/`). They
duplicate the physics on purpose: where they overlap they must agree, and a
disagreement is a bug in one of them.

## Set up

```bash
git clone git@github.com:alexderosis/M3LB.git && cd M3LB
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=ON
cmake --build build -j4
cd build && ctest --output-on-failure
```

Keep `-DKokkos_ENABLE_THREADS=ON`: without it everything runs single-threaded, several
times slower, and nothing tells you. Run executables by hand with
`--kokkos-num-threads=4`.

Then start Claude Code **from the repository root** (the terminal command `claude`, or
open this folder in the Claude desktop app). It picks up `CLAUDE.md` and everything in
`.claude/` automatically. You need your own Claude access; never put a key in the repo.

## What is set up for you

Type `/` to see the commands. Describing the task in plain words also works; Claude
will use the right one.

| command | use it to |
|---|---|
| `/new-case` | write a validation case (an exact-answer test) or a demonstrator |
| `/units` | turn Re, Ra, Pe or a real length and speed into lattice units, and check tau and Mach |
| `/new-feature` | add or change anything under `src/` or `GPU/` without a silent error |
| `/csf3-job` | plan and write a CSF3 GPU campaign, including the rule fixed before the runs |
| `/analyse` | read results, plot, build a table, quote a number properly |
| `/check-claim` | audit a number or sentence before it goes into a result or a paper |

Two reviewer agents give an independent read. Ask for them by name, and give them the
diff or the data, not your conclusion:

- `trap-reviewer`: reads a diff against the failure modes this tree has actually had.
- `claim-auditor`: recomputes a number from the raw files and tests its definition.

They are specialists you call one at a time when you need a second opinion, not a swarm.

## What Claude will stop and ask about

`ssh`, `scp`, `rsync` and `sbatch` (CSF3 uses Duo 2FA, and you should read every job
before it runs), edits under `results/` (tracked reference data), and edits to
`CLAUDE.md` or `.claude/`. It will never edit `paper/tg_mhd/provenance/` (the
pre-registration record) and will not force-push, hard-reset or `rm -rf`. It does not
commit unless you ask.

## The habits this repo asks of you

This tree has produced several confident wrong results, all from the same few
mistakes. The first two sentences of each are the rule:

1. **Read the banner at the top of a file before changing it.** The banners are the
   documentation and usually say why the obvious change is wrong.
2. **Know the answer before you run.** A validation case has an exact or published
   reference and prints PASS/FAIL. One that only runs is a demonstrator.
3. **Change one thing at a time**, and compare like with like (kinematic viscosity
   across a density ratio, `H` not `ny`).
4. **Check convergence before quoting.** Plot the series; say "still creeping" if it is.
   Three resolutions before an order. "Under-resolved, trend unknown" is a real result.
5. **Fix the rule before the runs.** Observable, the instant it is read, the thresholds,
   all in the job header and committed first. Do not choose a window after seeing it.
6. **A number that no script regenerates is a number to distrust.**
7. **Do not trust an AI-written number, or a human-written one.** Run `/check-claim`.

## Gotchas that have each cost hours

- Include `validation/Campaign.hpp` first in a case: the headers are not self-contained.
- `cs2` is 1/4 on D3Q7, so `tau = 3 nu + 1/2` is wrong for a D3Q7 scalar.
- "Two-dimensional" means a 3-D lattice with `nz = 1`, not D2Q9.
- In zsh a command held in a variable does not word-split; flags silently fall back to
  defaults. Use a bash array.
- On macOS a backgrounded run (`nohup ... &`) can be throttled to a fraction of its
  speed without any message. Run in the foreground, or check `%CPU` against the thread
  count in the first minutes.
- System Python often has no numpy. Use a venv outside the repo for plotting, and keep
  anything that must run on CSF3's login node stdlib-only.
- Do not run Python with `paper/tg_mhd` as the working directory: `numbers.py` shadows
  the standard library.

## CSF3

Read [GPU/csf3/README.md](GPU/csf3/README.md): SLURM, `gpuA` only (never `gpuL`), your
IT username, Duo, and the VPN off campus. Scratch is not backed up and is purged, so
treat it as a build directory, never the only copy of anything. Copy small result files
back with the `rsync` lines in that README.

## Privacy

Claude Code sends the files it reads and your prompts to Anthropic's API. The repository
is public; anything unpublished (a draft, someone else's data, a grant) is not, so check
your institution's policy before pointing a session at it. Claude's per-project memory
is personal to you and is not shared through git.

## Improving this setup

The skills and agents are plain Markdown in `.claude/skills/` and `.claude/agents/`.
If one gave you bad advice, fix the file and open a pull request with the case that
exposed it. `CLAUDE.md` is the shared reference; a change there is a deliberate edit
that should say what was measured.
