# The confined-MHD paper and its supplementary material.
#
#   make            tables from the committed results (numbers.py), then main.pdf and
#                   supplementary.pdf
#   make figs       the figures, PNG and PDF, into results/P_tg_mhd/fig/;
#                   needs a python with numpy and matplotlib: make figs PY=/path/to/venv/bin/python.
#                   The snapshots and the 3-D views also need the .f32 planes and volumes
#                   under results/P_tg_mhd/snap/*/{slices,vol}/, which git does not keep
#                   (GPU/csf3/tg_mhd_snap.sub and tg_mhd_vol.sub make them; their headers say
#                   how to copy them back).
#   make clean      the LaTeX by-products
#
# numbers.py writes numbers.txt and every tab_*.tex together, so numbers.txt
# stands for all of them. It reads results/P_tg_mhd/ and provenance/, which make
# does not track: after new runs, make -B.
#
# The two documents cite each other's labels through xr-hyper (S-<label> in
# main.tex, M-<label> in supplementary.tex), so each reads the other's .aux. One
# recipe builds both, supplement first, paper, then supplement again; latexmk
# reruns a document only when one of its inputs, the other's .aux included, has
# changed, so the third call is cheap once the labels have settled.

PY   ?= python3
ROOT := ../..
# The 3-D views' colour range: what plot_tg_mhd_3d.py chose from the data at
# t = 2.3 (and printed), fixed so that t = 4.6 shares it and both rebuild exactly.
RANGE3D := 0.172 51.7

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

all: main.pdf supplementary.pdf

main.pdf supplementary.pdf: main.tex supplementary.tex refs.bib numbers.txt
	$(LATEXMK) supplementary.tex
	$(LATEXMK) main.tex
	$(LATEXMK) supplementary.tex

numbers.txt: numbers.py
	python3 numbers.py > /dev/null

figs:
	cd $(ROOT) && $(PY) tools/plot_tg_mhd_ladder.py results/P_tg_mhd/round2 results/P_tg_mhd/ladder \
	    -o results/P_tg_mhd/fig/tg_mhd --fmt png,pdf
	cd $(ROOT) && for t in 2.3 4.6; do \
	    $(PY) tools/plot_tg_mhd_snapshots.py results/P_tg_mhd/snap/A_re1000_n512 \
	        results/P_tg_mhd/snap/B_re1000_n512 results/P_tg_mhd/snap/C_re1000_n512 \
	        --t $$t -o results/P_tg_mhd/fig/tg_mhd_snap_t$$t --fmt png,pdf || exit 1; done
	cd $(ROOT) && $(PY) tools/plot_tg_mhd_budget.py results/P_tg_mhd/budget \
	    -o results/P_tg_mhd/fig/tg_mhd_budget --fmt png,pdf
	cd $(ROOT) && for t in 2.3 4.6; do \
	    $(PY) tools/plot_tg_mhd_3d.py results/P_tg_mhd/snap/A_re1000_n512 \
	        results/P_tg_mhd/snap/B_re1000_n512 results/P_tg_mhd/snap/C_re1000_n512 \
	        --t $$t --range $(RANGE3D) -o results/P_tg_mhd/fig/tg_mhd_3d_t$$t --fmt png,pdf || exit 1; done

clean:
	latexmk -C main.tex supplementary.tex
	rm -f mainNotes.bib supplementaryNotes.bib

.PHONY: all figs clean
