from tqdm import tqdm
import json,mmap,os,argparse
import processors
from processors import *

def get_new_name( prev, unique_new_ners, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim, full_name,
                 unique_new_tokens, dict_newner_token):
    separator = ""
    prev_ner_tag = prev[0]
    new_nertag_i = ""
    full_name_c = " ".join(full_name)

    if (full_name_c in unique_new_tokens.keys()):

        new_nertag_i = unique_new_tokens[full_name_c]

    else:

        if (prev_ner_tag in unique_new_ners.keys()):
            old_index = unique_new_ners[prev_ner_tag]
            new_index = old_index + 1
            unique_new_ners[prev_ner_tag] = new_index
            # to try PERSON SPACE C1 instead of PERSON-C1
            new_nertag_i = prev_ner_tag + separator + ev_claim + str(new_index)
            # new_nertag_i = prev_ner_tag + separator + ev_claim + str(new_index)
            unique_new_tokens[full_name_c] = new_nertag_i

        else:
            unique_new_ners[prev_ner_tag] = 1
            new_nertag_i = prev_ner_tag + separator + ev_claim + "1"
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

def get_frequency_of_tag(curr_ner,dict_newner_token):
    freq=1
    if curr_ner in dict_newner_token.keys():
        old_count= dict_newner_token[curr_ner]
        freq=old_count+1
        dict_newner_token[curr_ner]=freq
    else:
        dict_newner_token[curr_ner] = 1
    return freq



def collapse_continuous_names(claims_words_list, claims_ner_list, ev_claim):
    dict_newner_token = {}
    dict_tokenner_newner = {}
    unique_new_tokens = {}
    unique_new_ners = {}
    prev = []
    new_sent = []

    full_name = []

    for index, (curr_ner, curr_word) in enumerate(zip(claims_ner_list, claims_words_list)):
        if (curr_ner == "O"):
            if (len(prev) == 0):
                new_sent.append(curr_word)
            else:
                prev, dict_tokenner_newner, new_sent, full_name, unique_new_ners, unique_new_tokens, dict_newner_token \
                    = get_new_name(prev, unique_new_ners, curr_ner,dict_tokenner_newner, curr_word,
                                        new_sent, ev_claim, full_name, unique_new_tokens, dict_newner_token)
                new_sent.append(curr_word)
        else:
            if (len(prev) == 0):
                prev.append(curr_ner)
                full_name.append(curr_word)
            else:
                if (prev[(len(prev) - 1)] == curr_ner):
                    prev.append(curr_ner)
                    full_name.append(curr_word)
                else:
                    prev, dict_tokenner_newner, new_sent, full_name, unique_new_ners, \
                    unique_new_tokens, dict_newner_token = \
                        get_new_name(prev, unique_new_ners, curr_ner, dict_tokenner_newner, curr_word, new_sent,
                                          ev_claim, full_name, unique_new_tokens, dict_newner_token)

    return new_sent, dict_tokenner_newner, dict_newner_token


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

def read_rte_data(filename):
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
    print(parser.parse_args())
    return parser




def mergeSmartNerAndSSTags(claim_words, claim_ner_tags, evidence_words, evidence_ner_tags):
    ev_claim = "c"
    neutered_claim, dict_tokenner_newner_claims, dict_newner_token = append_tags_with_count(claim_words,
                                                                                               claim_ner_tags,
                                                                                               ev_claim)

    ev_claim = "e"
    new_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = append_tags_with_count(
        evidence_words, evidence_ner_tags, ev_claim)

    neutered_evidence, found_intersection = check_exists_in_claim(new_sent_after_collapse,
                                                                  dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                  dict_tokenner_newner_claims)

    claimn = " ".join(neutered_claim)
    evidencen = " ".join(neutered_evidence)

    return claimn, evidencen



def collapseAndReplaceWithNerSmartly(claim_words,claim_ner_tags, evidence_words, evidence_ner_tags):
        ev_claim="c"
        neutered_claim, dict_tokenner_newner_claims, dict_newner_token = collapse_continuous_names(claim_words,
                                                                                                      claim_ner_tags,
                                                                                                      ev_claim)

        ev_claim = "e"
        new_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = collapse_continuous_names(evidence_words, evidence_ner_tags, ev_claim)

        neutered_evidence, found_intersection = check_exists_in_claim(new_sent_after_collapse,
                                                                       dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                       dict_tokenner_newner_claims)


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
def mergeSSandNERTags(claim_ss_tags, ev_ss_tags, claim_ner_tags, ev_ner_tags):
    for index,sst in enumerate(claim_ss_tags):
        if not (sst==""):
            nert=claim_ner_tags[index]
            if (nert=="O"):
                claim_ner_tags[index]=sst

    for index,sst in enumerate(ev_ss_tags):
        if not (sst==""):
            nert=ev_ner_tags[index]
            if (nert=="O"):
                ev_ner_tags[index]=sst



    return claim_ner_tags, ev_ner_tags


def read_sstagged_data(filename):

    sstags = []


    with open(filename,"r") as f:
        #for index, line in enumerate(tqdm(f, total=get_num_lines(filename))):
            line=f.readline()
            while(line):
                split_line=line.split("\t")
                sstag=split_line[7]
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
    all_claims, all_evidences, all_labels=read_rte_data(filename)
    all_claims_neutered=[]
    all_evidences_neutered = []
    with open('output.jsonl', 'w') as outfile:
        outfile.write('')


    ssfilename_claims = "sstagged_sample_files/claim_words_pos_datapointid_91034.pred.tags"
    ssfilename_ev = "sstagged_sample_files/evidence_words_pos_datapointid_91034.pred.tags"
    if (args.merge_ner_ss):
        claims_sstags = read_sstagged_data(ssfilename_claims)
        ev_sstags = read_sstagged_data(ssfilename_ev)

        # hardcoding claim, evidence and label for debugging purposes of merging NER and SStagging
        c = all_claims[91034]
        e = all_evidences[91034]
        l = all_labels[91034]
        claim_ann, ev_ann = annotate(c, e, API)
        assert (claim_ann is not None)
        assert (ev_ann is not None)
        assert (len(claim_ann.tags) is len(claims_sstags))
        claim_ner_tags = claim_ann._entities
        ev_ner_tags= ev_ann._entities

        claim_ner_ss_tags_merged, ev_ner_ss_tags_merged=mergeSSandNERTags(claims_sstags, ev_sstags, claim_ner_tags, ev_ner_tags)

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
        #def collapseAndReplaceWithNerSmartly(claim_words, claim_pos_tags, evidence_words, evidence_ner_tags):
        claim_neutered, ev_neutered =collapseAndReplaceWithNerSmartly(claim_ann.words, claim_ner_ss_tags_merged, ev_ann.words, ev_ner_ss_tags_merged)
    if (args.merge_ner_ss == True):
        claim_neutered, ev_neutered =mergeSmartNerAndSSTags(claim_ann.words, claim_ner_ss_tags_merged, ev_ann.words, ev_ner_ss_tags_merged)


        with open('output.jsonl', 'a+') as outfile:
            write_json_to_disk(claim_neutered, ev_neutered,l.upper(),outfile)


#            print(index)




