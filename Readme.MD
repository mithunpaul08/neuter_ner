# smart ner convertor

This code takes a claim and evidence pairs, finds where all NER tags exist and replace them smartly. refer examples below. 
This is being done to show the NN model that there is overlap between claim and evidence.

```
conda create --name smartner python=3
source activate conda
pip install tqdm
pip install git+https://github.com/myedibleenso/py-processors.git
```

**note: we are using `pyprocessors` to do annotation/NER/POS tagging etc. The documentation for the same can be found [here](https://py-processors.readthedocs.io/en/latest/)
We are using the jar option mentioned in the file to run pyprocessors server. However if you are adventurous enough to go the docker route, below are the commands you must use.
**

commands to run
```
source activate meanteacher
python main.py --pyproc_port 8887 --use_docker false --inputFile fn_dev_split_fourlabels.jsonl 
```
#### optional command line arguments"

`--pyproc_port 8886` By default pyprocessors , the java version runs off port 8888. If you intend to change it/want to run it over another port, you can pass it as
a command line argument like this.

`--use_docker true` if you are using docker for pyprocessors


#### Some sample conversions

```
hypothesis_before_annotation: Isis claims to behead US journalist
hypothesis_ann: ORGANIZATION-c1 claims to behead LOCATION-c1 journalist
premise_before_annotation: BREAKING : Islamic State , in video , beheads American journalist James Wright Foley who was kidnapped in 2012 - @BNONews
premise_ann: BREAKING : ORGANIZATION-e1 , in video , beheads MISC-e1 journalist PERSON-e1 who was kidnapped in DATE-e1 - @BNONews

['The', 'Boston', 'Celtics', 'play', 'their', 'home', 'games', 'at', 'TD', 'Garden', '.']

['The', 'Celtics', 'play', 'their', 'home', 'games', 'at', 'the', 'TD', 'Garden', ',', 'which', 'they', 'share', 'with', 'the', 'National', 'Hockey', 'League', '-LRB-', 'NHL', '-RRB-', "'s", 'Boston', 'Bruins', '.']

****['The', 'ORGANIZATION-c1', 'play', 'their', 'home', 'games', 'at', 'the', 'LOCATION-c1', ',', 'which', 'they', 'share', 'with', 'the', 'ORGANIZATION-e2', '-LRB-', 'ORGANIZATION-e3', '-RRB-', "'s", sed , '.']
```


### to run super sense tagging use my [fork](https://github.com/mithunpaul08/pysupersensetagger)
`python superSenseTag.py --pyproc_port 8887 --use_docker false --inputFile data/fever_train_split_fourlabels.jsonl  
`