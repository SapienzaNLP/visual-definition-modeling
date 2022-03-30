#!/bin/bash

if [ $# -lt 3 ]
then
    echo "bert-score.sh referece_path predictions_path lang"
    exit 1
fi
RUN_DIR=/
REF_PATH=/tmp/references
CAND_PATH=/tmp/predictions
IN_1=$1 #"$RUN_DIR/$1"
IN_2=$2 #"$RUN_DIR/$2"
echo 'START'
echo '======================================================================================='
echo "Running directory: $RUN_DIR"
echo "Reference file: $IN_1"
echo "Rrediction file: $IN_2"

echo '======================================================================================='
sort -k1 $IN_1 > $REF_PATH
sort -k1 $IN_2 > $CAND_PATH

bert-score -r $REF_PATH -c $CAND_PATH --lang $3 --rescale_with_baseline
#cd `dirname $0`
#perl multi-bleu.perl $REF_PATH < $CAND_PATH 
#echo 'DONE'
