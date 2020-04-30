#!/usr/bin/env bash
WORD_SIM_SRC_DIR=eval-word-vectors
WORD_SIM_DATASET_DIR=similarity_datasets
OUT_DIR=Results

if [ $# -ne 1 ]; then
    echo "Wrong vector arguments"
    exit 1
fi

VECTORS_FILE=$1
OUT_FILE=$OUT_DIR/word_similarity_results.txt

mkdir -p $OUT_DIR
python2 $WORD_SIM_SRC_DIR/filterVocab.py $WORD_SIM_SRC_DIR/fullVocab.txt <$VECTORS_FILE |
python2 $WORD_SIM_SRC_DIR/all_wordsim.py /dev/stdin $WORD_SIM_DATASET_DIR |
tee $OUT_FILE	
