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

`python superSenseTag.py `
`
--write_pos_tags True --pyproc_port 8887 --use_docker True --inputFile data/fever_train_split_fourlabels.jsonl
`


Notes to self
- on laptop and clara use conda environment `meanteacher` 
steps
    
- status of ss tagging as of 12 midnight may 9th 2019
    - had to do it again after becky suggested adding new line after every sentence in evidence.
    - steps for self verification
        - go to a new folder check
         -remove outputs if exists check
         -create outputs folder check
        - git pull check
        - verify the log -1
        - start meanteacher conda env
        - change port
        - change input file
        - change docker false
        - verify claim written
        - verify new line written after evidences    
    - fnc train
        - server:clara
        - tmux: 3
        - folder: neuter_ner_fnc_train
        - port:8887
        - status: 57% done
    - fnc dev
        - server:clara
        - tmux: 4
        - folder: neuter_ner_fnc_dev
        - port:8888
        - status:**all done**. 9068 data points=18136 files
    - fever-train
        - server:clara
        - tmux: 5
        - folder: neuter_ner_fever_training
        - port:    8889
        - status: **all done**238394 files
    - fever-dev
        - server:clara
        - tmux: 6
        - folder: neuter_ner_fever_dev
        - port:    8886
        - status: **all done**52504 files
         
    
    
Notes
- classic fever data has only 3 classes/labels,. viz.,SUPPORTS, REFUTES, NOT ENOUGH INFO. Here we have already converted into 4 classes after that of [fnc](http://www.fakenewschallenge.org/), viz., AGREE, DISAGREE, DISCUSS, UNRELATED 
- this command will create a huge number of files, one per each claim-evidence pair.
- if you can get the docker to run for pyprocessors and then use `--use_docker true` that will be fastest way to run this code.``
    - `turn on docker`
    - `docker start procserv`
    - open `localhost:8886` and confirm that the `pyprocessor` server is running

#### Step 2: running sstagger

the above came these are the commands i used to combine multiple claim files to one

steps 
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

also here is a version of the same command that i desinged which will ask the user permission
`find ./input_to_sstagger_output_from_pos_tagger/ -print0 | xargs -0 -n 1 -I{} -t -p ./sst.sh {}`
 
Notes:
- the xargs command will run the sst.sh on each input file from the folder: input_to_sstagger_output_from_pos_tagger
- i have modified the python code to create output file with $inputfilename.pred.tags in the output folder mentioned above
- P is the number of cores you can spare in your machine. 
- this command will create the output (*.pred.tags) at exactly the same location as the input file. Which means, you must do `rm *.tags` before every run else it will
start creating files like `*.pred.tags.pred.tags`

#### Step 3: merging NER tags and ss tags

In this phase, we need to do the smartner tagging plus ss tagging. Eg:

`input: Daniel Craig was the longest serving James Bond`

`output: PERSONc1 was the longest COGNITIONc1 PERSONc2`


other examples
```
claim: 		'A 	seven 		time 		Formula 	       One 		     World 	Champion 	is 		Michael Shumacher 	.'
combined: '	A 	NUMBERc1 	TIMEc1 		Foodc1 		Numberc2 	MISCc1 			stativec1 	PERSONc1 		O.'



evidence: 'He 	is 		a 	seven-time 	Formula 	One 		World Champion 	and 	is 	widely 	regarded 	as one of the greatest Formula One drivers of all time .'
combined:	'He 	stativee1 	a 	seven-time 	MISCc1 	Numberc2  	MISCc1 			and 	is 	widely      sociale1 	as NUMBERe2 of the greatest Foodc1 Numberc2 PERSONe1 of all time .'
```
This can be run using

`python main.py --use_docker true  --inputFile data/fever_train_split_fourlabels.jsonl --convert_prepositions False --create_smart_NERs False --merge_ner_ss True`



# notes to self/delete later

  
 - status of sstagging for each dataset:status as of may 9th 2pm: restarted whole pos tagging and sstagging after becky found the /n solution in data files
- per folder status is here `https://docs.google.com/spreadsheets/d/1nfewuq33Hxkwp9WaiLldN4LjRJo3vbVH2vxGHRHKKbI/edit?usp=sharing`
 
    - although one thing i can do is move out the input files corresponding to the ones where outputs were generated
    - when you want to learn something, check if there is a course in stanford or berkeley that covers that topic
    or google: ```xargs site:*.edu```
    
   - have started on tmux0 on clara and watching from tmux5 on amy
   - ideas from becky
        - add an empty line after every sentence in evidence.
        - try time out, i.e if the djikstra doesn't time out after say 10 mins, move onto next file.

   - **fever-train**:
   yu can check status using
        `ls ~/neuter_ner_fever_training/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger | wc -l`
        - latest status:count  is 178670/238,394 files done
        -update: its written 10636 files more of which 10407 have size more than zero bytes. So hopefully there is some stuff there.
        - for fever-train: moved all the already written .pred.tags files to :`net/kate/storage/work/mithunpaul/neuter_ner_fever_training/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger$`
        - also deleted the corresponding files from `/net/kate/storage/work/mithunpaul/neuter_ner_fever_training/amalgram/pysupersensetagger-2.0/input_to_sstagger_output_from_pos_tagger`
       - will try to run the xargs based command again to make it convert the rest of the files.
       - update: the plain xargs is getting stuck at huge evidence files. So i am going to run pairs of claims and evidences
       `./input_to_sstagger_output_from_pos_tagger/ -iname "*1*" -print0 | xargs -0 -n 1 -I{} -t -P 30 ./sst.sh {}`
       
    - **fever-dev**: 
    status as of 2pm may 9th: restarted sstagging after postagging with fresh data points. run on tmux 4.
    wrote 1 file in 1 minute. Might have to do parallel processing on this. its printing stuff on screen though 
    checklist
        - move the output of pos tagging to input folder of ss tagging.
        - source deactivate current
        - source activate  py2
        
        -  you can check outputs using
         `ls ~/neuter_ner_fever_dev/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger | wc -l`
         - latest status: count is 51,676/52504 files
        -update: have moved already tagged files to :
        `:~/neuter_ner_fever_dev/amalgram/pysupersensetagger-2.0/outputs_sstagged/already_tagged`
        -will try the python based parallelization thing now- update: didn't work still
        - remove all input files corresponding to the 51k files that were already tagged.
        - so now the input file directory must be left with 52504-51676=828 input files. a good number to check if the cython parallel processing code is workig
        - the actual set of input files are kept here: `~/neuter_ner_fever_dev/outputs`
        - update: as of 2pm on wednesday may 8th. its done 171 files of 828
        - had started using xargs, but even after 12+ hours it did only 173 files. 
        - killed tmux10
        - might as well try out becky's separate line idea 
        - keep only this file in input folder `evidence_words_pos_datapointid_26246`
        - run without new lines: started at 339pm- 3.58 pm no luck
        - with newlines as spaces:started at 359pm- good lord. finished in less than 1 minute.
        - yep, that was the issue...will have to rewrite all data with spaces after 1 line
        
         
    - **fnc-train** (this is the only thing running.)
        - have started a run on **started amy** from **tmux 0** 
        - status can be checked using
        `ls /net/kate/storage/work/mithunpaul/neuter_ner_fnc_train/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger/ | wc -l`
        - latest status:count is 1456/81808
        update: mihai, wants me to stop using jenny so that robert can use it. i'll move to amy if its easy.
        update: started again on amy. tmux 0. also uncommented code that will ignore already written files, hopefully. at 7pm wednesday
    - **fnc-dev** all runs are dead   
        - status can be checked using 
        ` ls /net/kate/storage/work/mithunpaul/neuter_ner_fnc_dev/amalgram/pysupersensetagger-2.0/outputs_sstagged/input_to_sstagger_output_from_pos_tagger/ | wc -l`
  
       - latest status: 845/18136 files are done.
        - update: moved tagged files to
        `/net/kate/storage/work/mithunpaul/neuter_ner_fnc_dev/amalgram/pysupersensetagger-2.0/outputs_sstagged/already_tagged`
- merging sstags+nertags with collapsing: status as of may 11th 2019.9pm
  
        - write logic for merging based on file id
        - start runs
                
            
- Commonly encountered errors
    - if you hit compile time error with text as
    `ImportError: Building module discriminativeTagger failed: ["CompileError: command 'gcc' failed with exit status 1\n"]` it means you have a compilation error. Kill the command and scroll up to find the compilation error.
    - if your code is not responding/hangs for more than a minute after you hit `./sst.sh logs` 
    or hit a error which says cannot find compiled *.so files, do this:`rm ~/.pyxbld/lib.linux-x86_64-2.7/*.so`
    - if it says package x not found, that means you haven't turned on conda. do `source activate py2`
    - if nonoe of this works, i.e tmux window is not responding, get out of the tmux window and do `tmux kill-session -t 4`
    
    

 