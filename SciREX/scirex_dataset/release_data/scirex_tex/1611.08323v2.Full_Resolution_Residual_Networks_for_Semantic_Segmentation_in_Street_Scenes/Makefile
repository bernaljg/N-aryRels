# yet another simple Makefile

all: frrn_paper.pdf supplemantary_material.pdf

frrn_paper.pdf: *.tex $(shell find images -iname '*.png'| sed 's/ /\\ /g') $(shell find data -iname '*.csv' | sed 's/ /\\ /g') egbib.bib abbrev_short.bib
	pdflatex frrn_paper.tex
	bibtex frrn_paper
	pdflatex frrn_paper.tex
	pdflatex frrn_paper.tex

supplemantary_material.pdf: *.tex $(shell find images -iname '*.png'| sed 's/ /\\ /g') $(shell find data -iname '*.csv' | sed 's/ /\\ /g') egbib.bib abbrev_short.bib
	pdflatex supplemantary_material.tex
	bibtex supplemantary_material
	pdflatex supplemantary_material.tex
	pdflatex supplemantary_material.tex


clean:
	rm -rf *.aux *.bbl *.blg *.log
.PHONY: clean
