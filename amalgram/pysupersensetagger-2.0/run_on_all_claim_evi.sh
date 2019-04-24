#! /bin/bash
for f in $(ls ./inputs/); do
    $(./sst.sh ./inputs/$f)
done
