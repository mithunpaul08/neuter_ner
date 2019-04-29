set -eu

IN=$1
model=sst.model

# prediction only
	python2.7 src/main.py --cutoff 5 --YY tagsets/bio2gNV --defaultY O --predict $IN --debug --load $model --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --input_folder /net/kate/storage/work/mithunpaul/neuter_ner_fever_training/outputs --output_folder outputs_sstagged --use_xargs 
