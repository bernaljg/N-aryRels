CURPATH=`pwd`
echo $CURPATH
while true; do
  cd $CURPATH
  pdflatex -output-directory=/tmp/texbuild main
  cp $CURPATH/*.bib $CURPATH/*.bst $CURPATH/*.sty /tmp/texbuild
  cd /tmp/texbuild
  bibtex main
  cd $CURPATH
  pdflatex -output-directory=/tmp/texbuild main
  # cp /tmp/texbuild/main.pdf $CURPATH/
  sleep 1s
done
