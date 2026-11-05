main.pdf: main.tex fig/fig1_real.pdf fig/fig2_cut.pdf fig/fig3_cells.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

figs:
	python3 make_figs.py

clean:
	rm -f main.aux main.log main.out

.PHONY: figs clean
