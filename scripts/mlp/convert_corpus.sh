#! /bin/sh
#
# convert_corpus.sh
# Copyright (C) 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
#


languages=( "basque" "farsi" "filipino" "finnish" "japanese" "kazakh" "marathi" "uyghur" "vietnamese" )

BASEDIR=$1
OUTDIR=$2
MORPHSEG_REPO=/home/judit/repo/morph-segmentation
STANDOFF_CONVERTER=$MORPHSEG_REPO/scripts/mlp/convert_file.py

tmp=`mktemp`

for lang in "${languages[@]}"; do
    echo $lang
    cat $BASEDIR/${lang}17training.segd.c.txt | python $STANDOFF_CONVERTER > $tmp
    paste $BASEDIR/${lang}17training.orig.c.txt $tmp > $OUTDIR/$lang.train
    cat $BASEDIR/${lang}17development.segd.c.txt | python $STANDOFF_CONVERTER > $tmp
    paste $BASEDIR/${lang}17development.orig.c.txt $tmp > $OUTDIR/$lang.dev
done

rm $tmp
