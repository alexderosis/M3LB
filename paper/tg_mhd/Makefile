# The confined-MHD paper.
#
#   make            tables from the committed results (numbers.py), then main.pdf
#   make figs       the three figures, PNG and PDF, into results/P_tg_mhd/fig/;
#                   needs a python with numpy and matplotlib: make figs PY=/path/to/venv/bin/python
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

clean:
	latexmk -C main.tex
	rm -f mainNotes.bib

.PHONY: figs clean
