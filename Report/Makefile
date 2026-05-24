MAIN = main
TEX = pdflatex
BIB = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,toc,pdf,fls,fdb_latexmk,synctex.gz}

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &
