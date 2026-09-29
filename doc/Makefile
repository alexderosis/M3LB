# Build the scheme/validation document.
#   make          -> m3lb.pdf
#   make figures  -> regenerate the figure data from scratch (needs the solver built)
#   make clean    -> remove auxiliary files
#
# The figure inputs are committed, so `make` alone works without running anything.

# pdflatex until LaTeX stops asking for a rerun, at most five passes. Two were
# not enough for this document from a clean state: the table of contents fills
# in on the second pass and moves the pages -- 156 to 160 on 2026-09-29 -- so
# the page references settle only on the third. The log holds non-UTF-8 bytes,
# which macOS grep treats as binary, hence LC_ALL=C grep -a; and the pattern is
# the requests themselves, 'Rerun to get ...' and longtable's 'Rerun LaTeX',
# because rerunfilecheck's banner ('Rerun checks for auxiliary files') is in
# every log. A log with undefined references is kept, and said so.
m3lb.pdf: m3lb.tex $(wildcard fig/*.dat) $(wildcard fig/*.png)
	@for pass in 1 2 3 4 5; do \
	  pdflatex -interaction=nonstopmode $< >/dev/null || { echo "pdflatex failed on pass $$pass: see m3lb.log"; exit 1; }; \
	  if [ $$pass -ge 2 ] && ! LC_ALL=C grep -a -q 'Rerun to get\|Rerun LaTeX' m3lb.log; then \
	    echo "cross-references settled after $$pass passes"; break; fi; \
	  if [ $$pass -eq 5 ]; then echo "WARNING: cross-references still unsettled after 5 passes"; fi; \
	done
	@if LC_ALL=C grep -a -q 'There were undefined references' m3lb.log; then \
	  echo "WARNING: undefined references -- m3lb.log kept"; rm -f m3lb.aux m3lb.out m3lb.toc; \
	else rm -f m3lb.aux m3lb.log m3lb.out m3lb.toc; fi
	@echo "wrote $@"

# Convergence data: ../build/validation/grid_convergence > fig/grid.csv  (~45 s)
# Orszag-Tang snapshots: run in fig/ so the .bin files land there  (~2 min)
figures:
	cd fig && ../../build/validation/grid_convergence > grid.csv
	cd fig && ../../build/validation/orszag_tang -n 512 -op cm -dump
	cd fig && python3 mkpng.py && rm -f *.bin
	@echo "figure inputs regenerated; now run make"

clean:
	rm -f m3lb.aux m3lb.log m3lb.out m3lb.toc
.PHONY: clean figures
