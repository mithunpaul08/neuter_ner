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

#### Step 2: running sstagger

the above came these are the commands i used to combine multiple claim files to one

steps to create a conda environment and run the sst tagger for my input file

- move the output of pos tagging to input folder of ss tagging.
    - note: dont do plain `mv * ../amalgram/pysupersensetagger-2.0/input_to_sstagger_output_from_pos_tagger/`. Linux will tell you argument list too long. Instead create a shell script like this:
    ```for each in ./*;
    do
    mv $eachfile ../amalgram/pysupersensetagger-2.0/input_to_sstagger_output_from_pos_tagger/
    done
    ```
- commands for create a conda run environment for running the actual sstagger    
```
 conda create -name py2_decompattn_nonallennlp python=2.7
 source activate py2_decompattn_nonallennlp
 pip install cython
 pip install nltk
 python
 nltk.download('wordnet')
 exit
```
- remember to create a folder inside outputs dir if using xargs, and the output (the sstagged files)
 will be written there.
 
 Example:
 
 ``mkdir outputs_sstagged/input_to_sstagger_output_from_pos_tagger``

- Check two things. *.tags doesn't exist in input folder or output folder. Might be vestigial/left over from older runs, but yeah, that will be detrimental if a *.tags is provided as input. THe code will crash
- now run the below command from the place where the file `sst.sh` exists.

`find input_to_sstagger_output_from_pos_tagger -print0 | xargs -0 -n 1 -P 72 -I{} ./sst.sh {}`

 
Notes:
- the xargs command will run the sst.sh on each input file from the folder: input_to_sstagger_output_from_pos_tagger
- i have modified the python code to create output file with $inputfilename.pred.tags in the output folder mentioned above
- P is the number of cores you can spare in your machine. 

#### Step 3: merging NER tags and ss tags

In this phase, we need to do the smartner tagging plus ss tagging. Eg:

`input: Daniel Craig was the longest serving James Bond`

`output: PERSONc1 was the longest COGNITIONc1 PERSONc2`

This can be run using

`python main.py --use_docker true  --inputFile data/fever_train_split_fourlabels.jsonl --convert_prepositions False --create_smart_NERs False --merge_ner_ss True`

# notes to self/delete later

 
        
- status as of 29th april
  - back to square one. tried writing cython code to load lexicon within, but it also gets stuck after like 184 files. going very slow. 
  - marco said, run xargs on one side, i.e let xargs load files and write pred tags for everyone
  - while read up about parallel processing in python
  - in folder: `neuter_ner_fever_training`, will start xargs based code
  - in folder: `neuter_ner_fever_dev` will continue experimenting with  
  - update: stared a run on tmux 13 for fever_training. I think i'll just let it keep going and
  not even work on python parallelization. Marco said if you want to do parallelization you cant beat xargs. No point trying to hack the system. 
  it'll get there when you get there.e
  - i don't want to also run the other way wher i load one single file concatenated with evidences...that way i wont even know which file is where.
  - only question now though is , should i start other 3 runs also..but that'll mean i'll have to reduce the 72 multi core processes...rehne de..
  
 - status of sstagging for each dataset
   - **fever-train**: 
        - started running sstagging again in **tmux 11** at 3pm on april 29th in **clara**
        - you can check status using
        `ls ~/neuter_ner_fever_training/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger | wc -l`
       
        - latest status:count  is 33,588/238,394 files done
    - **fever-dev**: 
        - have started ss tagging run of all claims and evidences of fever dev from **tmux 13** in clara. 
        -  you can check outputs using
         `ls ~/neuter_ner_fever_dev/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger | wc -l`
         - latest status: count is 29,774/52504 files
        
    - **fnc-train**
        - have started a run on **jenny** from **tmux 0** 
        - status can be checked using
        `ls /net/kate/storage/work/mithunpaul/neuter_ner_fnc_train/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger/ | wc -l`
        - latest status:count is 110/81808
    - **fnc-dev**
        - have started a run on **river** from tmux 0
        - status can be checked using 
        ` ls /net/kate/storage/work/mithunpaul/neuter_ner_fnc_dev/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger/ | wc -l`
       - am using tmux1 as a watcher.
       - latest status: 211/18136 files are done.
        
    - merging code
        was able to run till combining both files. next have to create a sample output
        - able to create personc1 based on the input ner tags. However, we might have to use the tags given from SStagging, which is using collapsing. Becuase right now
        michael and schumacher is being tagged as personc1 pesronc2, which is wrong.
        qn) what if we never add person dash from SS tag and let our old code do collapsing? 
 