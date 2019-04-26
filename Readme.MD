# This code base has two pre-processing codes. Mostly to do with POS tagging, NER, Super sense tagging etc

## smart ner convertor

This code takes a claim and evidence pairs, finds where all NER tags exist and replace them smartly. refer examples below. 
This is being done to show the NN model that there is overlap between claim and evidence.

```
conda create --name meanteacher python=3
source activate conda
pip install tqdm
pip install git+https://github.com/myedibleenso/py-processors.git
```

**note: we are using `pyprocessors` to do annotation/NER/POS tagging etc. The documentation for the same can be found [here](https://py-processors.readthedocs.io/en/latest/)
We are using the jar option mentioned in the file to run pyprocessors server. However if you are adventurous enough to go the docker route, below are the commands you must use.
**

commands to run:

`source activate meanteacher`

`docker pull myedibleenso/processors-server:latest`
` docker run myedibleenso/processors-server`

`docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server`
or


```docker start procserv``` (if you are using docker)

`
python main.py --pyproc_port 8887 --use_docker False --convert_prepositions False --convert_NERs True --inputFile data/dev_fourlabels_new.jsonl
`
#### optional command line arguments"

`--pyproc_port 8886` By default pyprocessors , the java version runs off port 8888. If you intend to change it/want to run it over another port, you can pass it as
a command line argument like this.

`--use_docker true` if you are using docker for pyprocessors (usually in laptops its easier to use a docker, where as in machines where you don't have
root/sudo access use java processors server)


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

# Super Sense Tagger

Super sense tagging is when you can take a sentence and assign the abstract super sense to it. like NER but more abstract.
Eg:

Before tagging:

`I do n't think he 's afraid to take a strong stand on gun control , what with his upbringing in El Paso .`

After Tagging:
```
I do|`a n't think|cognition he 's|stative afraid to take_a_ strong _stand|cognition on gun_control|ARTIFACT , what_with his upbringing|ATTRIBUTE in El_Paso|LOCATION .
```

For more details on SS taggging refer Noah Schnieder's github [page](https://github.com/nschneid/pysupersensetagger)

I have a folder `amalgram/` in this repo where the code and trained models are replicated

#### Step 1:
 
The SStagger needs as input the POS tag and the tokens of a given sentence, in a particular one line format.

Eg:
```Sounds	VBZ
haunting	VBG
,	,
and	CC
a	DT
```
Refer to Noah's code base above for more details.

This code base of mine, which you are looking at, I am using to generate these tokens/tags in the required format for the claim evidence pairs from [FEVER1.0](http://fever.ai/2018/task.html) data set. To do that run the command below.:

`python superSenseTag.py --pyproc_port 8887 --use_docker false --inputFile data/fever_train_split_fourlabels.jsonl  
`


Notes
- classic fever data has only 3 classes/labels,. viz.,SUPPORTS, REFUTES, NOT ENOUGH INFO. Here we have already converted into 4 classes after that of [fnc](http://www.fakenewschallenge.org/), viz., AGREE, DISAGREE, DISCUSS, UNRELATED 
- this command will create a huge number of files, one per each claim-evidence pair.
- if you can get the docker to run for pyprocessors and then use `--use_docker true` that will be fastest way to run this code.``
    - `turn on docker`
    - `docker start procserv`
    - open `localhost:8886` and confirm that the `pyprocessor` server is running

steps to create a conda environment and run the sst tagger for my input file

```
 conda create -name py2_decompattn_nonallennlp python=2.7
 source activate py2_decompattn_nonallennlp
 pip install cython
 pip install nltk
 python
 nltk.download('wordnet')
 exit
 ./sst.sh ../../outputs/evidence_words_pos_datapointid_10120
``` 

Notes:
 - output files (the .tags) file will be created at the same location as the input file

# notes to self/delete later

 - status as of 8pm 25th april.
    - pos tagging: still ongoing for fever-train in tmux 0. status 25841/119197. This had crashed after 50k files yday. check on this before sleeping tonight. 
    - have started sstagger runon 3 tmuxes (tmux 2, 5, 6). refer white board for details
    - i don't expect any of it to even finish by tomorrow morning . maybe the fn-dev might. since its only 18k files
    - next: need to write cython code to make it not load lexicon for every file. so its slowing down big time. leaving it overnight, just in case the conversion finishes.
 