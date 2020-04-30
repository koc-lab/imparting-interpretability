#!/usr/bin/env bash
#set -a
DATETIME=$(date +"%Y_%m_%d--%H_%M_%S")

if [ $# -ne 4 ]; then
    echo "Missing or too many arguments"
    exit 1
fi

echo "Corpus: $1"
echo "Memory: $2"
echo "Vector size: $3"
echo "Number of threads: $4"

# Build target (release/debug)
TARGET=release

# Directories
IMBUE_SRC_DIR=Source
IMBUE_INIT_DIR=Initialization
IMBUE_PARAMS_DIR=Params
BUILD_DIR=$IMBUE_SRC_DIR/$TARGET
OUTPUTS_DIR=$IMBUE_SRC_DIR/out

# Input corpus
CORPUS=$1

# Temporary file names
OVERFLOW_FILE=$OUTPUTS_DIR/overflow
COOCCURRENCE_FILE=$OUTPUTS_DIR/cooccurrence.bin
TEMPSHUFFLE=$OUTPUTS_DIR/temp_shuffle

# Imbue initialization files (automatically generated)
COOCCURRENCE_SHUF_FILE=$IMBUE_INIT_DIR/cooccurrence_$DATETIM.shuf.bin
INIT_FILE=$IMBUE_INIT_DIR/init_$DATETIME.bin

# Imbue parameters (must be specified)
DIMS_FILE=$IMBUE_PARAMS_DIR/forced_up_to_300
POLS_FILE=$IMBUE_PARAMS_DIR/positive_all
FORCEDIDS_FILE=roget_groups_out/forced_words_roget
KVALS_FILE=$IMBUE_PARAMS_DIR/k_0.1_all

# Output file names
VOCAB_FILE=$OUTPUTS_DIR/vocab.txt
SAVE_FILE=$OUTPUTS_DIR/vectors_imbued.txt

# Glove parameters
VERBOSE=2
MEMORY=$2
VOCAB_MIN_COUNT=65
VECTOR_SIZE=$3
MAX_ITER=20
WINDOW_SIZE=15
BINARY=0
NUM_THREADS=$4
X_MAX=75

# Error if the target is invalid
if [[$TARGET != release]]; then
    echo 'Error: Invalid Makefile target.'
    exit 1; fi


# Build the target
set -x
make $TARGET -C $IMBUE_SRC_DIR; { set +x; echo; } 2>/dev/null

# Run

$BUILD_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE <$CORPUS >$VOCAB_FILE
python roget_word_groups_final.py --vocab_file $VOCAB_FILE --dim_num $VECTOR_SIZE
$BUILD_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -overflow-file $OVERFLOW_FILE <$CORPUS >$COOCCURRENCE_FILE
$BUILD_DIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $TEMPSHUFFLE <$COOCCURRENCE_FILE >$COOCCURRENCE_SHUF_FILE
$BUILD_DIR/generate_init_file -vector-size $VECTOR_SIZE -vocab-file $VOCAB_FILE -verbose $VERBOSE -INIT_FILE $INIT_FILE
$BUILD_DIR/glove_imbue -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -INIT_FILE $INIT_FILE -DIMS_FILE $DIMS_FILE -POLS_FILE $POLS_FILE -FORCEDIDS_FILE $FORCEDIDS_FILE -KVALS_FILE $KVALS_FILE
