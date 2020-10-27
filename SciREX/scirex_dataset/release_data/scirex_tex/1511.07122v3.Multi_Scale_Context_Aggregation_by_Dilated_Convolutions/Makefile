R = dilation
STY = # *.cls
FIG = figs/*
BIB = *.bib

pdf: $(R).pdf 

$(R).bbl: $(BIB)
	pdflatex $(R)
	bibtex $(R)
	pdflatex $(R)

$(R).pdf: *.tex $(STY) $(FIG) $(R).bbl
	pdflatex $(R)

clean:
	rm -f $(R).log $(R).aux $(R).bbl $(R).blg $(R).out $(R).brf
