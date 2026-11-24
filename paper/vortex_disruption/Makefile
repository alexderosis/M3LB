# The vortex-disruption letter.
#
#   make            vd_numbers.txt from the committed results, then main.pdf and
#                   supplementary.pdf (the Supplemental Material)
#   make figs       every figure, PDF and PNG, into this directory; needs a python
#                   with numpy and matplotlib: make figs PY=/path/to/python. The
#                   snapshots also need the frames in runs/vortex_disruption_d3q27/,
#                   which git does not keep; GPU/csf3/vortex_disruption_d3q27.sub
#                   makes them
#   make clean      the LaTeX by-products
#
# vd_numbers.py reads results/Q_vortex_disruption/, which make does not track:
# after new runs, make -B. (Not numbers.py: that name shadows the standard
# library's `numbers`, which matplotlib imports.)

PY   ?= python3
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

all: main.pdf supplementary.pdf

main.pdf: main.tex refs.bib vd_numbers.txt fig_pair.pdf fig_collapse.pdf fig_threshold.pdf
	$(LATEXMK) main.tex

supplementary.pdf: supplementary.tex vd_numbers.txt fig_gallery_t6.pdf fig_gallery_t12.pdf fig_fixed.pdf
	$(LATEXMK) supplementary.tex

vd_numbers.txt: vd_numbers.py
	python3 vd_numbers.py > /dev/null

figs:
	$(PY) figures.py

clean:
	latexmk -C main.tex supplementary.tex
	rm -f mainNotes.bib supplementaryNotes.bib

.PHONY: all figs clean
