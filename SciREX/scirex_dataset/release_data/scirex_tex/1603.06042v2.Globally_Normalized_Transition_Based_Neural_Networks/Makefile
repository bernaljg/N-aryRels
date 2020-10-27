all: paper.pdf

paper.aux: paper.tex
	pdflatex paper

paper.bib: paper.tex
	pdflatex paper

paper.bbl: paper.bib paper.aux
	bibtex paper

paper.pdf: paper.tex paper.bbl tex/intro.tex tex/model.tex tex/experiments.tex tex/experiments-compression.tex tex/experiments-wsj.tex tex/experiments-conll.tex tex/experiments-tagging.tex tex/conclusions.tex tex/experiments-parsing.tex tex/labelbias.tex tex/experiments-tagging-results.tex
	pdflatex paper
	cp paper.pdf /google/data/rw/users/da/danielandor/www/structured_training.pdf
