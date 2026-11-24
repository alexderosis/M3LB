# Archiving the letter's code, data and protocol record

What a reader needs to check the vortex-disruption letter, where it is now, and
the steps that need the author's own accounts and decisions. Nothing here has
been done yet; it is written down so that it can be done once, in order. The
tg_mhd paper's plan (`../tg_mhd/ARCHIVING.md`) is the model, and three of its
steps (the licence, Zenodo's GitHub switch, `sacct`) are shared with it.

**Do it after the Rm = 32000 runs.** The letter's map stops at Rm = 16000 while
elements 32-39 of the registered campaign (job 25227362) wait on CSF3; when they
land the letter's Rm range, p, its uncertainty and Fig. 3 change. A deposit made
before then would not be the letter that is published. A Zenodo record can take
a new version later, but the letter should cite the one it was written from.

## 1. A DOI for the repository at the letter's commit (Zenodo)

The repository is public on GitHub, but a moving branch is not a citable record.
Zenodo archives a GitHub release and mints a DOI for it. Everything the letter's
numbers and line plots are made from is in git, 28 MB of it:

| path | what it is | what it rebuilds |
|---|---|---|
| `results/Q_vortex_disruption/` | series.dat and log.txt of every run, and a README that says where each group came from | every number and every line plot |
| `paper/vortex_disruption/vd_numbers.py`, `vd_numbers.txt` | the script that reads them, and its output | every number the letter quotes |
| `paper/vortex_disruption/figures.py` | the figure script (the snapshots also need section 2) | Figs. 1-6 |
| `GPU/src/vortex_disruption.cu`, `validation/vortex_disruption.cpp` | the CUDA driver and its Kokkos twin | the runs |
| `GPU/csf3/vortex_disruption_{verify,map,d3q27}.sub` | the CSF3 jobs; the map job's header holds the registered rule | the campaign |
| `tools/vortex_disruption_campaign.py` | applies that rule | the verdict on p |

Steps:

1. **Choose a licence.** The repository has no LICENSE file, so its code can be
   read but not reused, and Zenodo asks for one. Usual choices are MIT or
   BSD-3-Clause for code; CC-BY-4.0 is the usual one for data (section 2). The
   choice is the author's, and it is the same one the tg_mhd plan is waiting
   on, so make it once for both.
2. **Give the release its own metadata.** Zenodo reads `.zenodo.json` at the
   repository root at release time, and today that file describes the tg_mhd
   paper. At the commit to be tagged, rewrite its title and description for
   this letter (and add the `"license"` key, an SPDX id). A later tg_mhd
   release then needs its own text back at its own tag: each release carries
   whatever `.zenodo.json` says at that commit.
3. Sign in at <https://zenodo.org> with GitHub, open **Settings -> GitHub**, and
   switch on `alexderosis/M3LB` (once for both papers). Zenodo only sees
   releases published after this.
4. Publish a GitHub release from the commit that holds the submitted letter,
   with a tag such as `vortex-disruption-letter-v1`. Zenodo archives the tarball
   and mints two DOIs: one for this version, one for every version.
5. Put the version DOI in the letter's data availability statement (section 4).

Optional, independent of Zenodo: <https://archive.softwareheritage.org/save/>
takes a public repository URL and archives it with the date of the visit.

## 2. The frames behind the snapshot panels (a Zenodo dataset)

Fig. 1(a) and Figs. 4-6 are drawn from frames of the vorticity and the current
density that git does not keep. A run with `-frames 24` writes 25 (k = 0..24, at
t = k/2 Rm^(1/3)), and the figures read five of them, k = 3, 6, 12, 18 and 24,
i.e. t = 1.5, 3, 6, 9 and 12 Rm^(1/3), and each run's `log.txt`, from which
`frame_at()` reads dt. All of that is on this laptop for the 17 runs the figures
use, all from `GPU/csf3/vortex_disruption_d3q27.sub` (job 25203000), in
`runs/vortex_disruption_d3q27/`:

| group | runs | used in |
|---|---|---|
| `pair/` | Re = 1000, Rm = 250 and 4000, R0 = 40 | Fig. 1(a) |
| `gallery/` | the map's five Rm = 4000 runs, M^2 Rm = 0.625-5, rerun with frames | Figs. 4-6 |
| `map/` | Rm = 250 and 1000 at the same five fields | Figs. 4-6 |

187 files (170 frames and 17 logs), 142 MB as a tarball. The directory holds
504 MB because some runs carry more of their 25 frames than the figures read,
two (`pair/re1000_rm4000_r40_x3.15`, `map/re1000_rm1000_r20_x0.625`) a subset
broken off by an interrupted copy, and `map/re1000_rm1000_r20_x0.3125` frames
without its log; so deposit the defined set, not the directory. The commands
below pick exactly the runs `frames_ok()` in `figures.py` accepts. Every frame the
job wrote is on CSF3 scratch, `~/scratch/M3LB/runs/vortex_disruption_d3q27/`,
which is not backed up and deletes files unused for three months: copy it to
backed-up storage too if the full set is to be kept, and deposit it as a second
file of the same record if it is wanted (its size is not measured here).

Each frame is `w_NNNN.bin` or `j_NNNN.bin`: int32 nx, int32 ny, then nx*ny
float32, x fastest, as `vd_write_field` in `GPU/src/vortex_disruption.cu` writes
them (`vd_measure` defines the two fields) and `raw()` in `figures.py` reads
them. Frame k of a run is at step k * floor(n_max / 24), with dt and n_max in
the run's `log.txt`.

From the repository root:

```bash
cd runs
: > vd_frames.list
for d in vortex_disruption_d3q27/*/*; do
  set -- log.txt w_0003.bin w_0006.bin w_0012.bin w_0018.bin w_0024.bin \
         j_0003.bin j_0006.bin j_0012.bin j_0018.bin j_0024.bin
  ok=1; for f; do [ -f "$d/$f" ] || ok=0; done
  [ $ok = 1 ] && for f; do echo "$d/$f"; done >> vd_frames.list
done
wc -l < vd_frames.list                     # 187
tar -cf vortex_disruption_frames.tar -T vd_frames.list
shasum -a 256 vortex_disruption_frames.tar > vortex_disruption_frames.tar.sha256
```

(Tested on 2026-11-24 in bash and zsh: 17 runs, 187 files, a 142 MB tarball.)

Upload the tarball and its checksum by hand as a Zenodo **dataset** record (a
GitHub release cannot carry files git does not hold), under the data licence,
with the section 1 record as a related identifier ("is supplement to"), and keep
the `.sha256` file in `results/Q_vortex_disruption/` so the repository records
what was deposited.

## 3. When the rule was fixed: the record outside this repository

The letter says that for the runs at Rm = 2000-16000 the rule was fixed before
they ran. The rule is the header of `GPU/csf3/vortex_disruption_map.sub` and the
docstring of `tools/vortex_disruption_campaign.py`, both added in commit
**aada22f**, committed 2026-11-19 16:51:31 UTC by the author's clock. Neither
rule has changed since: the only later commit to either file, 86acdb9
(2026-11-20), adds a note on resubmitting with a longer time limit and touches
no rule line (`git diff aada22f HEAD -- tools/vortex_disruption_campaign.py
GPU/csf3/vortex_disruption_map.sub`).

A commit's date is written by its author, so two records kept by someone else
are what show the order:

1. **GitHub's push log.** GitHub records server side when each push reached it,
   and its public API keeps roughly the last quarter, so aada22f's push can be
   read until about mid-February 2027 and not after. Capture it before then,
   into a new `provenance/` directory beside this file:

   ```bash
   mkdir -p paper/vortex_disruption/provenance && cd paper/vortex_disruption/provenance
   d=$(date -u +%Y-%m-%dT%H%MZ)
   curl -s -D github_activity_$d.headers -o github_activity_$d.json \
     "https://api.github.com/repos/alexderosis/M3LB/activity?per_page=100&time_period=quarter"
   ```

   and check that aada22f is in it. The API returns the 100 most recent pushes,
   and this branch has had many since 19 November: if aada22f is not on the
   first page, follow the `Link: rel="next"` header in the `.headers` file and
   save each page the same way.
2. **The cluster's accounting.** SLURM records when each job was submitted and
   started, which closes the other end: a rule pushed before the jobs it
   governs were submitted. On CSF3:

   ```bash
   sacct -X -u $USER -S 2026-11-19 -E now -P -n \
         --format=JobID,JobName%30,Submit,Start,End,State,Elapsed \
         | grep vortex > vortex_disruption_sacct.txt
   ```

   then copy `vortex_disruption_sacct.txt` into `provenance/`. It should show the
   verify job (24933077) and every map array, including the cancelled 25174641
   and 25227362 for Rm = 32000.

Until both are captured, the letter can claim only what `git log` shows, which
is the author's clock. The rest of the letter's runs (Rm = 250-1000, the two
slices, the controls, the frames) were not pre-registered, and the letter does
not say they were.

## 4. After all three

- **The data availability statement.** The letter has none at the moment: the
  acknowledgments, which held the only link to the code and data, are commented
  out in `main.tex`. APS asks for one. It should give the version DOI of
  section 1 and the DOI of the frames dataset of section 2.
- Record both DOIs, and where the frames and the scratch copies are kept, in
  `results/Q_vortex_disruption/README.md` and in CLAUDE.md's section on papers.
