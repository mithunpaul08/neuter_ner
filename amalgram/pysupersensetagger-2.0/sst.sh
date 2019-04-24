#!/bin/bash

# Predict with an existing MWE model.
# Usage: ./mwe_identify.sh model input

set -eu
set -o pipefail

input=$1 # word and POS tag on each line (tab-separated)
echo "value of input file is $input"
./predict_sst.sh $input > ./outputs/$input.pred.tags

src/tags2sst.py -l $input.pred.tags > $input.pred.sst

