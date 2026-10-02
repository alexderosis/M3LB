# When the protocol was fixed: the record outside this repository

A commit's date is written by its author, so `git log` alone cannot show that a
rule was fixed before the runs it governs. Two records are kept by someone else:

1. **GitHub's push log.** GitHub records, server side, when each push reached
   it. `github_activity_2026-10-02.json` is that log for this repository as the
   public API returned it at 10:12:27 GMT on 2026-10-02 (the response headers,
   with the request id, are in `github_activity_2026-10-02.headers`), fetched
   without authentication with

       curl -s "https://api.github.com/repos/alexderosis/M3LB/activity?per_page=100&time_period=quarter"

   It holds the 100 most recent pushes, 2026-09-17 to 2026-10-02, which include
   every protocol commit. The API keeps roughly a quarter, so anyone can repeat
   the request and compare until about the end of December 2026; after that, a
   copy deposited with a dated archive (Zenodo, `../ARCHIVING.md`) is the
   record. `numbers.py` reads this file and writes `tab_protocol.tex`, the
   supplementary material's Table S1.

   | rule | commit | committed (author's clock) | pushed (GitHub, UTC) |
   |---|---|---|---|
   | observable, peak rule and gate | 663929c | 2026-09-27 11:57:38 UTC | 2026-09-27 11:58:24 |
   | f_w estimator corrected | 9f65706 | 2026-09-28 11:55:32 UTC | 2026-09-28 11:56:52 |
   | Mach control | cd07f45 | 2026-09-29 06:22:12 UTC | 2026-09-29 06:22:15 |
   | decision rule and FP64 control (round 2) | d061ee8 | 2026-09-30 20:03:29 UTC | 2026-09-30 20:03:34 |

2. **The cluster's accounting.** SLURM records when each job was submitted
   and started, which closes the other end: a rule pushed before the jobs it
   governs were submitted. Not captured yet. On CSF3:

       sacct -X -u $USER -S 2026-09-26 -E 2026-10-02 -P -n \
             --format=JobID,JobName%30,Submit,Start,End,State,Elapsed \
             | grep tg_mhd > tg_mhd_sacct.txt

   then copy `tg_mhd_sacct.txt` into this directory. Until it is here, the
   paper claims only what the push log shows.

The two qualitative outcomes (25 September 2026) were stated in a plan that
is not in this repository, so no record here dates them.
