# Keep latexmk consistent with the Makefile: pdflatex + bibtex (natbib).
$pdf_mode  = 1;
$pdflatex  = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;          # always run bibtex when a .bib is present
$clean_ext  = 'bbl run.xml synctex.gz';
