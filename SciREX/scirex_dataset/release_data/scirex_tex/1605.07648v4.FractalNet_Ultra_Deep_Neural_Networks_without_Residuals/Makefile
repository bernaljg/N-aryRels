# programs
PDFLATEX = pdflatex
CP       = cp
RM       = rm
GREP     = grep
DIFF     = diff -q
TOUCH    = touch

# input and output files
TEXFILE = fractalnet.tex
PDFFILE = fractalnet.pdf
TEXINCLUDES = $(wildcard include/*)

# intermediate files
TEXLOG     = $(basename $(TEXFILE)).log
AUXFILE    = $(basename $(TEXFILE)).aux
OLDAUXFILE = $(AUXFILE).old

# arxiv submission files
ARXIV_FILES := $(if $(wildcard files.txt),$(shell cat files.txt),)
ARXIV_OUT   := arxiv.tar.gz

# If this message appears in TEXLOG, we must re-run LaTeX.
RERUNLATEX = "Rerun to get cross-references right"

# The primary target (created by the command "make") is the PDF file.
.PHONY: pdf
pdf: $(PDFFILE)

# This rule uses PDFLaTeX to create a PDF file from a LaTeX file.
# Then PDFLaTeX is re-run to resolve references, as necessary.
$(PDFFILE): $(TEXFILE) $(TEXINCLUDES)
	$(TOUCH) $(AUXFILE)
	$(CP) $(AUXFILE) $(OLDAUXFILE)
	$(PDFLATEX) $<
	while $(GREP) -q $(RERUNLATEX) $(TEXLOG); do $(PDFLATEX) $<; done

# arxiv submission
.PHONY: submit
submit: $(ARXIV_OUT)
$(ARXIV_OUT): files.txt $(ARXIV_FILES)
	@COPYFILE_DISABLE=1 tar cvzfh $(ARXIV_OUT) $(ARXIV_FILES)

# This rule cleans out the files made by the make process.
clean:
	$(RM) -f $(AUXFILE)\
                 $(OLDAUXFILE)\
                 $(TEXLOG)\
                 $(PDFFILE)\
                 $(ARXIV_OUT)
