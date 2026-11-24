# The vortex-disruption letter.
#
#   make            vd_numbers.txt from the committed results, then main.pdf
#   make figs       every figure, PDF and PNG, into this directory; needs a python
#                   with numpy and matplotlib: make figs PY=/path/to/python. The
#                   snapshots also need the frames in runs/vortex_disruption_d3q27/,
#                   which git does not keep; GPU/csf3/vortex_disruption_d3q27.sub
#                   makes them
#   make clean      the LaTeX by-products
#
# vd_numbers.py reads results/Q_vortex_disruption/, which make does not track:
# after new runs, make -B. (Not numbers.py: that name shadows the standard
# library's `numbers`, which matplotlib imports.) latexmk can wrongly report the
# PDF up to date even with -g; run pdflatex and bibtex by hand if it does.
#
# There is no Supplemental Material: what it held (Figs. 4-6 and their text) is in
# the letter since 2026-11-24.

PY   ?= python3
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

FIGS := fig_pair.pdf fig_collapse.pdf fig_threshold.pdf \
        fig_gallery_t6.pdf fig_gallery_t12.pdf fig_fixed.pdf

all: main.pdf

main.pdf: main.tex refs.bib vd_numbers.txt $(FIGS)
	$(LATEXMK) main.tex

vd_numbers.txt: vd_numbers.py
	python3 vd_numbers.py > /dev/null

figs:
	$(PY) figures.py

clean:
	latexmk -C main.tex
	rm -f mainNotes.bib

.PHONY: all figs clean
