#for all files which were already written, remove the corresponding input files. This way we can check if the code is even running at all or not
#run this in folder where the sst.sh is
import os,sys,subprocess
from os import listdir
currentFolder=os.getcwd()
folderWithPredTags=os.path.join(currentFolder,"outputs_sstagged/already_tagged")
folderWithInputFilesToRemove=os.path.join(currentFolder,"input_to_sstagger_output_from_pos_tagger/")
files=listdir(folderWithPredTags)
for x in files:
    if x.endswith(".pred.tags"):
        x_split=x.split(".pred.tags")
        print(x)
        fileToRemoveFullpath=os.path.join(folderWithInputFilesToRemove,x_split[0])
        print(fileToRemoveFullpath)
        if os.path.isfile(fileToRemoveFullpath):
            os.unlink(fileToRemoveFullpath)
        else:
            print("file not found")
            print(fileToRemoveFullpath)
