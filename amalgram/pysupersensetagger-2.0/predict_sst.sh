set -eu

IN=$1
model=sst.model
# prediction only
	python2.7 src/main.py --cutoff 5 --YY tagsets/bio2gNV --defaultY O --predict $IN --debug --load $model --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json --input_folder /net/kate/storage/work/mithunpaul/neuter_ner_fnc_dev/pos_tagged_files_fnc_test --output_folder fnc_test_outputs_sstagged
