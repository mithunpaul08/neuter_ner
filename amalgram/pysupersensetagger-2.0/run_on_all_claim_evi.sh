!# /bin/bash

for f in $(ls /Users/mordor/research/neuter/outputs); do

$(./sst.sh $f)
done
