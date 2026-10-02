# Archiving the paper's code, data and protocol record

Three things in this list need the author's own accounts and decisions, so they
are written down here rather than done.

## 1. A DOI for the repository at the paper's commit (Zenodo)

The data availability statement cites the GitHub repository, which is public, and
a moving branch is not a citable record. Zenodo archives a GitHub release and
mints a DOI for it. `.zenodo.json` at the repository root carries the metadata;
Zenodo reads it at release time.

1. **Choose a license.** The repository has no LICENSE file. Until it has one,
   others may read the code but not reuse it, and Zenodo will ask for one. That
   choice is the author's; add the file (and a `"license"` key in `.zenodo.json`,
   an SPDX id such as `"MIT"` or `"BSD-3-Clause"`) before the release.
2. Sign in at <https://zenodo.org> with GitHub, open **Settings → GitHub**, and
   switch on `alexderosis/M3LB`. Zenodo only sees releases published after this.
3. Publish a GitHub release from the commit that holds the submitted paper, with
   a tag such as `tg-mhd-paper-v1`. Zenodo archives the tarball and mints two
   DOIs: one for this version, one for every version.
4. Put the version DOI in `main.tex`'s data availability statement (its TODO),
   and in CLAUDE.md's section on papers.

Optional, independent of Zenodo: <https://archive.softwareheritage.org/save/>
takes a public repository URL and archives it with the date of the visit.

**What a deposit made now can and cannot show.** A Zenodo record from October
2026 does not date anything in September. What dates the protocol is GitHub's
server-side push log, captured in `provenance/` and readable from GitHub's API
by anyone until about the end of December 2026. The deposit's job is to keep
that capture, and the repository as it was, after the API has forgotten it.
`provenance/README.md` also says how to add the other end of each interval, the
cluster's record of when the jobs were submitted (`sacct`); do that before the
release, so that it is in the archive too.

## 2. The field dumps behind the snapshots (CSF3 scratch)

Figures 4 and 5 and Figs. S1 and S2 come from full dumps that only exist on CSF3
scratch: `~/scratch/M3LB/runs/tg_mhd_snap/{A,B,C}_re1000_n512/raw/fields_*.raw`,
three dumps (t = 0, 2.3, 4.6) per box, 3.8 GB each, 34 GB in all. Scratch is not
backed up and deletes files unused for three months; it took the whole checkout
once already (2026-09-17, `GPU/csf3/README.md`). Git keeps only the plane means.
The planes and the 256^3 volumes the figures are drawn from (about 290 MB and
770 MB, under `results/P_tg_mhd/snap/*/{slices,vol}/`) are gitignored, so their
only copies are this laptop and scratch: back them up with the dumps.

Each dump is `"TGMHDRAW"`, int32 nx ny nz nv, double t, double h, then nv float32
blocks u_x u_y u_z b_x b_y b_z rho, x fastest, in the paper's units
(`tools/tg_mhd_slices.py` reads them).

On CSF3, with `$DEST` a directory on Research Data Storage (or any store that is
backed up and big enough):

```bash
cd ~/scratch/M3LB/runs/tg_mhd_snap
du -sh */raw
find . -path '*/raw/*' -type f -name 'fields_*.raw' -print0 | sort -z \
    | xargs -0 sha256sum > raw_sha256.txt
mkdir -p "$DEST/M3LB_tg_mhd_snap"
rsync -a --relative ./*/raw ./raw_sha256.txt "$DEST/M3LB_tg_mhd_snap/"
cd "$DEST/M3LB_tg_mhd_snap" && sha256sum -c --quiet raw_sha256.txt && echo VERIFIED
```

Then, from the laptop's repository root, keep the checksums with the data:

```bash
rsync -av csf3:scratch/M3LB/runs/tg_mhd_snap/raw_sha256.txt results/P_tg_mhd/snap/
```

so that anyone who receives the dumps can check them. A second copy can go to a
Zenodo dataset record of its own (34 GB is under Zenodo's default 50 GB per
record), cited from the paper in place of "available from the author".

## 3. After both

Replace the two TODOs in `main.tex`'s data availability statement, and record
the DOI(s) and where the dumps live in CLAUDE.md and `GPU/csf3/README.md`.
