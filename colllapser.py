from tqdm import tqdm
import json,mmap,os
import processors
from processors import *

def get_new_name( prev, unique_new_ners, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim, full_name,
                 unique_new_tokens, dict_newner_token):
    separator = "-"
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




def collapse_both(claims_words_list, claims_ner_list, ev_claim):
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

def annotate_and_save_doc_with_label_as_id(headline, body, API):
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



def neuter(claim_ann,evidence_ann):
        ev_claim="c"
        neutered_headline, dict_tokenner_newner_claims, dict_newner_token = collapse_both(claim_ann.words,
                                                                                          claim_ann._entities,
                                                                                               ev_claim)



        ev_claim = "e"
        new_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = collapse_both(
            evidence_ann.words, evidence_ann._entities, ev_claim)



        neutered_body, found_intersection = check_exists_in_claim(new_sent_after_collapse,
                                                                       dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                       dict_tokenner_newner_claims)


        claimn = " ".join(neutered_headline)
        evidencen = " ".join(neutered_body)

        return claimn,evidencen

if __name__ == '__main__':
    API = ProcessorsAPI(port=8886)
    #API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
    claims_words_list=["frank sinatra","is","a", "good","person","working","with","USA"]
    claims_ner_list = ["PERSON", "O", "ORGANIZATION","O","ORGANIZATION","LOCATION","ORGANIZATION","O"]

    filename="data/"+"fever_train_split_fourlabels.jsonl"
    all_claims, all_evidences, all_labels=read_rte_data(filename)
    all_claims_neutered=[]
    all_evidences_neutered = []
    with open('output.jsonl', 'w') as outfile:
        outfile.write('')

    with open('output.jsonl', 'a+') as outfile:
        for (index, (c, e ,l)) in enumerate(zip(all_claims, all_evidences,all_labels)):
            claim_ann, ev_ann = annotate_and_save_doc_with_label_as_id(c, e, API)
            claim_neutered,ev_neutered= neuter(claim_ann, ev_ann)
            write_json_to_disk(claim_neutered, ev_neutered,l,outfile)
            print(index)




