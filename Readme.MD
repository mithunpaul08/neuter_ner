# smart ner convertor

This code takes a claim and evidence pairs, finds where all NER tags exist and replace them smartly. refer examples below. 
This is being done to show the NN model that there is overlap between claim and evidence.

commands to run
```
source activate meanteacher
python colllapser.py --inputFile fn_train_split_fourlabels.jsonl --pyproc_port 8889
```

#### Some sample conversions

```
hypothesis_before_annotation: Isis claims to behead US journalist
hypothesis_ann: ORGANIZATION-c1 claims to behead LOCATION-c1 journalist
premise_before_annotation: BREAKING : Islamic State , in video , beheads American journalist James Wright Foley who was kidnapped in 2012 - @BNONews
premise_ann: BREAKING : ORGANIZATION-e1 , in video , beheads MISC-e1 journalist PERSON-e1 who was kidnapped in DATE-e1 - @BNONews
```

```
['The', 'Boston', 'Celtics', 'play', 'their', 'home', 'games', 'at', 'TD', 'Garden', '.']

['The', 'Celtics', 'play', 'their', 'home', 'games', 'at', 'the', 'TD', 'Garden', ',', 'which', 'they', 'share', 'with', 'the', 'National', 'Hockey', 'League', '-LRB-', 'NHL', '-RRB-', "'s", 'Boston', 'Bruins', '.']

****['The', 'ORGANIZATION-c1', 'play', 'their', 'home', 'games', 'at', 'the', 'LOCATION-c1', ',', 'which', 'they', 'share', 'with', 'the', 'ORGANIZATION-e2', '-LRB-', 'ORGANIZATION-e3', '-RRB-', "'s", sed , '.']```