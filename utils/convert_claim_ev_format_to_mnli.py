import pandas as pd
import json
from collections import OrderedDict

delex_file = "delex_file.jsonl"
lex_file = "lex_file.jsonl"
out_file="output_mnli_format.jsonl"
delex_file_df = pd.read_json(delex_file, lines=True)
lex_file_df = pd.read_json(lex_file, lines=True)

for ((i,row1), (j,row2)) in zip(delex_file_df.iterrows(), lex_file_df.iterrows()):
    with open(out_file, 'a+') as outfile:
        json.dump(OrderedDict(
            [("annotator_labels", row2.annotator_labels), ("genre", row2.genre), ("gold_label", row2.gold_label),
             ("pairID", row2.pairID), ("promtID", row2.promptID), ("sentence1", row1.claim),
             ("sentence1_binary_parse", row2.sentence1_binary_parse), ("sentence1_parse", row2.sentence1_parse),
             ("sentence2", row1.evidence), ("sentence2_binary_parse", row2.sentence2_binary_parse),
             ("sentence2_parse", row2.sentence2_parse)]), outfile)
        outfile.write("\n")