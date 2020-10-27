paper:	deep_flux.pdf

%.pdf:	%.tex
	latexmk --pdf $<
clean:
	latexmk -C deep_flux.tex
	rm -f deep_flux.bbl
