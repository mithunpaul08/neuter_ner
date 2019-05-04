from tqdm import tqdm
import json,mmap,os,argparse,string
import processors
from processors import *

def get_new_name( prev, unique_new_ners, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim, full_name,
                 unique_new_tokens, dict_newner_token):
    separator = ""
    #curr_ner = prev[0]
    new_nertag_i = ""
    full_name_c = " ".join(full_name)

    if (full_name_c in unique_new_tokens.keys()):

        new_nertag_i = unique_new_tokens[full_name_c]

    else:

        if (curr_ner in unique_new_ners.keys()):
            old_index = unique_new_ners[curr_ner]
            new_index = old_index + 1
            unique_new_ners[curr_ner] = new_index
            # to try PERSON SPACE C1 instead of PERSON-C1
            new_nertag_i = curr_ner + separator + ev_claim + str(new_index)
            # new_nertag_i = curr_ner + separator + ev_claim + str(new_index)
            unique_new_tokens[full_name_c] = new_nertag_i

        else:
            unique_new_ners[curr_ner] = 1
            new_nertag_i = curr_ner + separator + ev_claim + "1"
            unique_new_tokens[full_name_c] = new_nertag_i

    if not ((full_name_c, prev[0]) in dict_tokenner_newner):
        dict_tokenner_newner[full_name_c, prev[0]] = new_nertag_i
    else:
        dict_tokenner_newner[full_name_c, prev[0]] = new_nertag_i

    dict_newner_token[new_nertag_i] = full_name_c

    new_sent.append(new_nertag_i)

    full_name = []
    prev = []
    if (curr_ner != "O"):
        prev.append(curr_ner)

    return prev, dict_tokenner_newner, new_sent, full_name, unique_new_ners, unique_new_tokens, dict_newner_token

def attach_freq_to_nertag(ner_tag, ner_dictionary,separator,ev_claim):
    new_index = get_frequency_of_tag(ner_tag, ner_dictionary)
    new_nertag_i = ner_tag + separator + ev_claim + str(new_index)
    return new_nertag_i

def collapse_continuous_names(claims_words_list, claims_ner_list, ev_claim):

    #dict_newNerBasedName_lemma:a mapping from newNerBasedName to its old lemma value(called henceforth as token) Eg:{PERSONc1:Michael Schumacher}.
    dict_newNerBasedName_lemma = {}
    # dict_token_ner_newner:a mapping from a tuple (lemma, original NER tag of the word) to its newNerBasedName  Eg:{(Michael Schumacher,PERSON):PERSONc1}
    dict_token_ner_newner = {}
    #dict_lemmas_newNerBasedName. A mapping from LEMMA/token of the word to its newNerBasedName Eg:{Michael Schumacher:PERSONc1}
    dict_lemmas_newNerBasedName = {}
    #dict_newNerBasedName_freq: A mapping from newNerBasedName to the number of times it occurs in a given sentences
    dict_newNerBasedName_freq = {}
    #a stack to hold all the ner tags before current- this is useful for checking if a name is spread across multiple NER tags. Eg: JRR Tolkein= PERSON,PERSON,PERSON
    prev = []
    #the final result of combining all tags and lemmas is stored here
    new_sent = []
    #this full_name is used to store multiple parts of same name.Eg:Michael Schumacher
    full_name = []


    #in this code, triggers happen only when a continuous bunch of nER tags end. Eg: PERSON , PERSON, O.
    for index, (curr_ner, curr_word) in enumerate(zip(claims_ner_list, claims_words_list)):
        if (curr_ner == "O"):
            if (len(prev) == 0):
                # if there were no tags just before this O, it means, its the end of a combined name. just add that O and move on
                new_sent.append(curr_word)
            else:
                # instead if there was something pushed into a stack, this O means, we are just done with those couple of continuous tags. Collapse names, add new name to dictionaries and empty stack
                prev, dict_token_ner_newner, new_sent, full_name, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma \
                    = get_new_name(prev, dict_newNerBasedName_freq, curr_ner, dict_token_ner_newner, curr_word,
                                                               new_sent, ev_claim, full_name, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)
                new_sent.append(curr_word)
        else:
            #if length of the list called previous is zero, it means, no tag was collapsed until now.
            if (len(prev) == 0):
                prev.append(curr_ner)
                full_name.append(curr_word)
            else:
                #if the previous ner tag and current ner tag is the same, it means its most probably part of same name. Eg: JRR Tolkein. Collapse it into one nER entity
                if (prev[(len(prev) - 1)] == curr_ner):
                    prev.append(curr_ner)
                    full_name.append(curr_word)
                else:


                    # if the previous ner tag and current ner tag are not the same, this O means, we are just done with those couple of continuous tags. No collapsing, but add both names to dictionaries and empty stack
                    prev, dict_token_ner_newner, new_sent, full_name, dict_newNerBasedName_freq, \
                    dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma = \
                        append_count_to_two_consecutive_ner_tags(prev, dict_newNerBasedName_freq, curr_ner, dict_token_ner_newner, curr_word, new_sent,
                                                                 ev_claim, full_name, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)

    return new_sent, dict_token_ner_newner, dict_newNerBasedName_lemma


def collapse_continuous_names_with_dashes(claims_words_list, claims_ner_list, ev_claim):

    #dict_newNerBasedName_lemma:a mapping from newNerBasedName to its old lemma value(called henceforth as token) Eg:{PERSONc1:Michael Schumacher}.
    dict_newNerBasedName_lemma = {}
    # dict_token_ner_newner:a mapping from a tuple (lemma, original NER tag of the word) to its newNerBasedName  Eg:{(Michael Schumacher,PERSON):PERSONc1}
    dict_token_ner_newner = {}
    #dict_lemmas_newNerBasedName. A mapping from LEMMA/token of the word to its newNerBasedName Eg:{Michael Schumacher:PERSONc1}
    dict_lemmas_newNerBasedName = {}
    #dict_newNerBasedName_freq: A mapping from newNerBasedName to the number of times it occurs in a given sentences
    dict_newNerBasedName_freq = {}
    #a stack to hold all the ner tags before current- this is useful for checking if a name is spread across multiple NER tags. Eg: JRR Tolkein= PERSON,PERSON,PERSON
    prev_ner = ""
    #the final result of combining all tags and lemmas is stored here
    new_sent = []
    #this full_name is used to store multiple parts of same name.Eg:Michael Schumacher
    full_name = []


    #in this code, triggers happen only when a continuous bunch of nER tags end. Eg: PERSON , PERSON, O.
    for index, (curr_ner, curr_word) in enumerate(zip(claims_ner_list, claims_words_list)):
        if (curr_ner == "O"):
            if ((prev_ner) == ""):
                # if there were no tags just before this O, it means, its the end of a combined name. just add that O and move on
                new_sent.append(curr_word)
        else:
            if (curr_ner=="_"):
                print("ner tag is _")
                list_of_indices_to_collapse = find_how_many_indices_to_collapse(index, claims_ner_list)
                assert (len(claim_ann.tags) is not 0)
                new_lemma_name=join_indices_to_new_name(claims_words_list,list_of_indices_to_collapse)
                str_new_lemma_name=" ".join(new_lemma_name)

                #add it to all dictionaries where the curr_ner=prev_ner Eg: Michael Schumacher:Person and curr_word=str_new_lemma_name
                dict_token_ner_newner, new_sent, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma \
                    = append_count_to_ner_tags(dict_newNerBasedName_freq, prev_ner, dict_token_ner_newner, str_new_lemma_name,
                                               new_sent, ev_claim,
                                               dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)
            else:
                prev_ner=curr_ner
                #look ahead, if the next NER value is a dash, don't add to dictionary. else add.
                if not (claims_ner_list[index+1]=="_"):


                    dict_token_ner_newner, new_sent, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma\
                        = append_count_to_ner_tags(dict_newNerBasedName_freq, curr_ner, dict_token_ner_newner, curr_word,
                                                 new_sent, ev_claim,
                                                 dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)





    return new_sent, dict_token_ner_newner, dict_newNerBasedName_lemma


def append_count_to_ner_tags( dict_newNerBasedName_freq, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim,
                                             dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma):
    # if the ner value is stative, don't find new ner based name. eg:stativec1. just add it to all dictionaries
    if(curr_ner=="stative"):
        new_nertag_i=curr_ner
    else:
        new_nertag_i = attach_freq_to_nertag(curr_ner, dict_newNerBasedName_freq, "", ev_claim)
    lemma_ner_tuple = (curr_word, curr_ner)
    new_sent.append(new_nertag_i)
    add_to_dict_if_not_exists(curr_word, new_nertag_i, dict_lemmas_newNerBasedName)
    add_to_dict_if_not_exists(new_nertag_i,curr_word, dict_newNerBasedName_lemma)
    add_to_dict_if_not_exists(lemma_ner_tuple, new_nertag_i, dict_tokenner_newner)

    return dict_tokenner_newner, new_sent, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma


def append_count_to_two_consecutive_ner_tags(prev, dict_newNerBasedName_freq, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim, full_name,
                                             dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma):

    #do same thing twice for both current and previous tags/words
    prev_tag=prev[(len(prev)-1)]
    prev_word="".join(full_name)
    new_nertag_i=attach_freq_to_nertag(prev_tag,dict_newNerBasedName_freq , "", ev_claim)
    add_to_dict_if_not_exists(prev_word, new_nertag_i, dict_lemmas_newNerBasedName)
    add_to_dict_if_not_exists(new_nertag_i,prev_word, dict_newNerBasedName_lemma)
    new_sent.append(new_nertag_i)

    new_nertag_i=attach_freq_to_nertag(curr_ner, dict_newNerBasedName_freq, "", ev_claim)
    add_to_dict_if_not_exists(curr_word, new_nertag_i, dict_lemmas_newNerBasedName)
    new_sent.append(new_nertag_i)

    full_name = []
    prev = []
    if (curr_ner != "O"):
        prev.append(curr_ner)

    return prev, dict_tokenner_newner, new_sent, full_name, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma

def add_to_dict_if_not_exists(key, value, dict):
    if not (key in dict.keys()):
        dict[key]=value

def replace_value(key, value, dict):
        dict[key]=value


def get_frequency_of_tag(curr_ner,dict_newner_token):
    freq=1
    if curr_ner in dict_newner_token.keys():
        old_count= dict_newner_token[curr_ner]
        freq=old_count+1
        dict_newner_token[curr_ner]=freq
    else:
        dict_newner_token[curr_ner] = 1
    return freq

#if there is one NER followed by more than one dashes, collect them all together so that it can be assigned to one name/new_ner_tag etc
def find_how_many_indices_to_collapse(curr_index, list_ner_tags):
    #very first time add the NER tag before _ Eg: Formula in Formula one
    list_indices_to_collapse=[]
    list_indices_to_collapse.append(curr_index-1)
    #then keep adding indices unti you hit a word that is not _

    while (curr_index<len(list_ner_tags)):
        if  (list_ner_tags[curr_index] == "_"):
            list_indices_to_collapse.append(curr_index)
            curr_index = curr_index + 1
        else:
            return list_indices_to_collapse
    return list_indices_to_collapse

def join_indices_to_new_name(all_words,list_indices):
    new_name=[]
    for i in list_indices:
        new_name.append(all_words[i])
    return new_name

def append_tags_with_count(claims_words_list, claims_ner_list, ev_claim):
    dict_ner_freq = {}
    new_sent = []

    for index, (curr_ner, curr_word) in enumerate(zip(claims_ner_list, claims_words_list)):
        if (curr_ner == "O"):
            new_sent.append(curr_word)
        else:
                freq = get_frequency_of_tag( curr_ner,dict_ner_freq)
                new_sent.append(curr_ner+ev_claim+str(freq))
    return new_sent



def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

def read_rte_data(filename,args):
        tr_len=1000
        all_labels = []
        all_claims = []
        all_evidences = []

        with open(filename) as f:
            for index,line in enumerate(tqdm(f, total=get_num_lines(filename))):
                multiple_ev = False
                x = json.loads(line)
                claim = x["claim"]
                evidences = x["evidence"]
                label = x["label"]

                if (args.remove_punctuations == True):
                    claim = claim.translate(str.maketrans('', '', string.punctuation))
                    evidences = evidences.translate(str.maketrans('', '', string.punctuation))

                all_claims.append(claim)
                all_evidences.append(evidences)
                all_labels.append(label)

        return all_claims, all_evidences, all_labels


def write_json_to_disk(claim, evidence,label,outfile):
    total = {'claim': claim,
             'evidence':evidence,
             "label":label}
    json.dump(total, outfile)
    outfile.write('\n')

def annotate(headline, body, API):
    claim_ann = API.fastnlp.annotate(headline)
    ev_ann = API.fastnlp.annotate(body)
    return claim_ann, ev_ann


def check_exists_in_claim(new_ev_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev, dict_tokenner_newner_claims):


        combined_sent=[]


        found_intersection = False



        #for every token (irrespective of NER or not) in evidence
        for ev_new_ner_value in new_ev_sent_after_collapse:

            found_intersection=False

            #check if its an ner
            if ev_new_ner_value in dict_newner_token_ev.keys():

                #if thats true find its corresponding string/lemma value Eg: "tolkein" from dict_newner_token_ev which maps PERSON-E1 ->tolkein
                token=dict_newner_token_ev[ev_new_ner_value]
                token_split=set(token.split(" "))


                #now go to through the keys in the dictionary that maps tokens in claim to its new ner Eg: tolkein:PERSON
                for tup in dict_tokenner_newner_claims.keys():
                    name_cl = tup[0]
                    ner_cl=tup[1]
                    name_cl_split = set(name_cl.split(" "))


                    #check if any of the names/tokens in claim have an intersection with what you just got from evidence ev_new_ner_value. Eg: tolkein
                    if (token_split.issubset(name_cl_split) or name_cl_split.issubset(token_split)):
                        found_intersection = True


                        # also confirm that NER value of the thing you found just now in evidence also matches the corresponding NER value in claim. This is to avoid john amsterdam PER overlapping with AMSTERDAM LOC
                        actual_ner_tag=""
                        for k, v in dict_tokenner_newner_evidence.items():

                            if (ev_new_ner_value == v):
                                actual_ner_tag=k[1]

                                break

                        #now check if this NER tag in evidence also matches with that in claims
                        if(actual_ner_tag==ner_cl):
                            val_claim = dict_tokenner_newner_claims[tup]
                            combined_sent.append(val_claim)


                        #now that you found that there is an overlap between your evidence token and the claim token, no need to go through the claims dictionary which maps tokens to ner
                        break;


                if not (found_intersection):
                    combined_sent.append(ev_new_ner_value)
                    new_ner=""


                    #get the evidence's PER-E1 like value
                    for k,v in dict_tokenner_newner_evidence.items():
                        #print(k,v)
                        if(ev_new_ner_value==v):
                            new_ner=k[1]

                    dict_tokenner_newner_claims[token, new_ner] = ev_new_ner_value



            else:
                combined_sent.append(ev_new_ner_value)



        return combined_sent,found_intersection

def parse_commandline_args():
    return create_parser().parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_parser():
    parser = argparse.ArgumentParser(description='Pg')
    parser.add_argument('--inputFile', type=str, default='fever_train_split_fourlabels.jsonl',
                        help='name of the input file to convert to smart ner format')
    parser.add_argument('--pyproc_port', type=int, default=8888,
                        help='port at which pyprocessors server should run. If you are running'
                             'multiple servers on the same machine, will need different port for each')
    parser.add_argument('--use_docker', default=False, type=str2bool,
                        help='use docker for loading pyproc. useful in machines where you have root access.', metavar='BOOL')
    parser.add_argument('--convert_prepositions', default=False, type=str2bool,
                        help='.',
                        metavar='BOOL')
    parser.add_argument('--create_smart_NERs', default=False, type=str2bool,
                        help='mutually ',
                        metavar='BOOL')
    parser.add_argument('--merge_ner_ss', default=False, type=str2bool,
                        help='once you have output from sstagger, merge them both.',
                        metavar='BOOL')
    parser.add_argument('--run_on_dummy_data', default=False, type=str2bool,
                        help='once you have output from sstagger, merge them both.',
                        metavar='BOOL')
    parser.add_argument('--remove_punctuations', default=False, type=str2bool,
                        help='once you have output from sstagger, merge them both.',
                        metavar='BOOL')
    print(parser.parse_args())
    return parser




def collapseAndCreateSmartTagsSSNer(claim_words, claim_ner_tags, evidence_words, evidence_ner_tags):
    ev_claim = "c"
    neutered_claim, dict_tokenner_newner_claims, dict_newner_token = collapse_continuous_names_with_dashes(claim_words,
                                                                                               claim_ner_tags,
                                                                                               ev_claim)
    ev_claim = "e"
    new_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = collapse_continuous_names_with_dashes(
        evidence_words, evidence_ner_tags, ev_claim)

    neutered_evidence, found_intersection = check_exists_in_claim(new_sent_after_collapse,
                                                                  dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                  dict_tokenner_newner_claims)

    claimn = " ".join(neutered_claim)
    evidencen = " ".join(neutered_evidence)

    return claimn, evidencen



def collapseAndReplaceWithNerSmartly(claim_words,claim_ner_tags, evidence_words, evidence_ner_tags):
        ev_claim="c"
        neutered_claim, dict_token_ner_newner_claims, dict_newNerBasedName_lemma = collapse_continuous_names(claim_words,
                                                                                                      claim_ner_tags,
                                                                                                      ev_claim)

        ev_claim = "e"
        new_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = collapse_continuous_names(evidence_words, evidence_ner_tags, ev_claim)

        neutered_evidence, found_intersection = check_exists_in_claim(new_sent_after_collapse,
                                                                       dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                      dict_token_ner_newner_claims)


        claimn = " ".join(neutered_claim)
        evidencen = " ".join(neutered_evidence)

        return claimn,evidencen

#whenever you see a preposition in this sentence, replace the NER tags of this sentence with PREP. This is
#being done so that when we do neutering, the PREP also gets added in along with the NER tags. Just another
# experiment to check if prepositions have an effect on linguistic domain transfer
def replacePrepositionsWithPOSTags(claim_pos_tags, ev_pos_tags,claim_ner_tags,ev_ner_tags):
    for index,pos in enumerate(claim_pos_tags):
        if (pos=="IN"):
            claim_ner_tags[index]="PREP"
    for index,pos in enumerate(ev_pos_tags):
        if (pos=="IN"):
            ev_ner_tags[index]="PREP"



    return claim_ner_tags, ev_ner_tags



#for every word, if a NER tag exists, give that priority. if not, check if it has a SS tag, if yes, pick that.
# if a sstag exists and the word has no NER tag, pick SStag
def mergeSSandNERTags(ss_tags, ner_tags ):
    # give priority to NER tags when there is a collision,. Except when NER tag is MISC. In that case pick SSTag
    for index,sst in enumerate(ss_tags):
        if not (sst==""):
            #if the sstag is _, we need it as is for the collapsing process
            if(sst=="_"):
                ner_tags[index] = sst
            else:
                # if the ss TAG IS NOT empty  #get the corresponding ner tag
                nert=ner_tags[index]
                if not (nert=="O"):
                    # if the NER tag is not O,  there is a collision between NER and SSTag. Check if the NER tag is MISC
                    if(nert=="MISC"):
                        #if its MISC, pick the corresponding SSTag #if not, pick the NER tag itself -i.e dont, do anything.
                        ner_tags[index]=sst
                else:
                    #if the NER tag is 0 and SSTag exists, replace NER tag with SSTag
                    ner_tags[index] = sst
    return ner_tags


def read_sstagged_data(filename,args):
    sstags = []
    puncts = set(string.punctuation)
    with open(filename,"r") as f:
            line=f.readline()
            while(line):
                split_line=line.split("\t")
                #pick only the words/lemmas that are not punctutations. -this had to be done in a existential check way because we had already written a million sstag files by the time we coded up merging
                # and didn't want to rewrite them all without punctuations in strings.
                if not(split_line[1] in puncts) and (args.remove_punctuations == True):
                    #if the 6th column has a dash, add it. A dash in sstagger means, this word, with the word just before it was collapsed into one entity. i.e it was I(inside) in BIO notation.
                    sstag6=split_line[6]
                    sstag7 = split_line[7]
                    if(sstag6=="_"):
                        sstag=sstag6
                    else:
                        sstag=sstag7
                    sstags.append(sstag)
                line = f.readline()

    return sstags

if __name__ == '__main__':

    args = parse_commandline_args()
    if(args.use_docker==True):
        API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
    else:
        API = ProcessorsAPI(port=args.pyproc_port)

    filename=args.inputFile
    #if not (args.run_on_dummy_data):
    all_claims, all_evidences, all_labels=read_rte_data(filename,args)
    all_claims_neutered=[]
    all_evidences_neutered = []
    with open('output.jsonl', 'w') as outfile:
        outfile.write('')


    ssfilename_claims = "sstagged_sample_files/claim_words_pos_datapointid_91034.pred.tags"
    ssfilename_ev = "sstagged_sample_files/evidence_words_pos_datapointid_91034.pred.tags"
    if (args.merge_ner_ss):
        claims_sstags = read_sstagged_data(ssfilename_claims,args)
        ev_sstags = read_sstagged_data(ssfilename_ev,args)

        # hardcoding claim, evidence and label for debugging purposes of merging NER and SStagging
        c = all_claims[91034]
        e = all_evidences[91034]
        l = all_labels[91034]
        claim_ann, ev_ann = annotate(c, e, API)
        assert (claim_ann is not None)
        assert (ev_ann is not None)
        assert (len(claim_ann.tags) is len(claims_sstags))
        assert (len(ev_ann.tags) is len(ev_sstags))
        claim_ner_tags = claim_ann._entities
        ev_ner_tags= ev_ann._entities

        claim_ner_ss_tags_merged = mergeSSandNERTags(claims_sstags, claim_ner_tags)
        ev_ner_ss_tags_merged = mergeSSandNERTags(ev_sstags, ev_ner_tags)
    #uncomment below portion for running over all claims and evidences. Commented out for debugging on just one data point
    # for (index, (c, e ,l)) in enumerate(zip(all_claims, all_evidences,all_labels)):
    #
    #         claim_ann, ev_ann = annotate(c, e, API)
    #         assert (claim_ann is not None)
    #         assert (ev_ann is not None)

    claim_pos_tags = claim_ann.tags
    ev_pos_tags = ev_ann.tags


    if(args.convert_prepositions==True):
        claim_ner_ss_tags_merged, ev_ner_ss_tags_merged=replacePrepositionsWithPOSTags(claim_pos_tags, ev_pos_tags, claim_ner_ss_tags_merged, ev_ner_ss_tags_merged)
    if (args.create_smart_NERs == True):
        claim_neutered, ev_neutered =collapseAndReplaceWithNerSmartly(claim_ann.words, claim_ner_ss_tags_merged, ev_ann.words, ev_ner_ss_tags_merged)
    if (args.merge_ner_ss == True):
        claim_neutered, ev_neutered =collapseAndCreateSmartTagsSSNer(claim_ann.words, claim_ner_ss_tags_merged, ev_ann.words, ev_ner_ss_tags_merged)


        with open('output.jsonl', 'a+') as outfile:
            write_json_to_disk(claim_neutered, ev_neutered,l.upper(),outfile)


#            print(index)




