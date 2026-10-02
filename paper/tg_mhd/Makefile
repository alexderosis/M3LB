# The confined-MHD paper.
#
#   make            tables from the committed results (numbers.py), then main.pdf
#   make figs       the five figures, PNG and PDF, into results/P_tg_mhd/fig/;
#                   needs a python with numpy and matplotlib: make figs PY=/path/to/venv/bin/python.
#                   The two snapshots also need the .f32 planes under
#                   results/P_tg_mhd/snap/*/slices/, which git does not keep
#                   (GPU/csf3/tg_mhd_snap.sub makes them; its header says how to copy them back).
#   make clean      the LaTeX by-products
#
# numbers.py writes numbers.txt and the four tab_*.tex together, so numbers.txt
# stands for all of them. It reads results/P_tg_mhd/, which make does not track:
# after new runs, make -B.

PY   ?= python3
ROOT := ../..

main.pdf: main.tex refs.bib numbers.txt
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

numbers.txt: numbers.py
	python3 numbers.py > /dev/null

figs:
	cd $(ROOT) && $(PY) tools/plot_tg_mhd_ladder.py results/P_tg_mhd/round2 results/P_tg_mhd/ladder \
	    -o results/P_tg_mhd/fig/tg_mhd --fmt png,pdf
	cd $(ROOT) && for t in 2.3 4.6; do \
	    $(PY) tools/plot_tg_mhd_snapshots.py results/P_tg_mhd/snap/A_re1000_n512 \
	        results/P_tg_mhd/snap/B_re1000_n512 results/P_tg_mhd/snap/C_re1000_n512 \
	        --t $$t -o results/P_tg_mhd/fig/tg_mhd_snap_t$$t --fmt png,pdf || exit 1; done

clean:
	latexmk -C main.tex
	rm -f mainNotes.bib

.PHONY: figs clean
