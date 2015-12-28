# IMG=azerty1.pdf_tex azerty2.pdf_tex azerty3.pdf_tex
IMG=

prez.pdf: prez.tex $(IMG)
	pdflatex prez.tex

%.pdf_tex: %.svg
	inkscape -z -D --file=$< --export-pdf=$(basename $@).pdf --export-latex
