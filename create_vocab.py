import pandas as pd
import os
import nltk
import re
import time
from heuristic_tokenize import sent_tokenize_rules
from tqdm import tqdm
import spacy
import operator
import collections
import sys

MIMIC_NOTES_PATH = '../../../data/MIMIC3/NOTEEVENTS.csv'
category = "Discharge summary"
discharge = False # flag to decide whether to process discharge summary or other categories

#setting sentence boundaries

def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

nlp = spacy.load('en_core_sci_lg', disable=['tagger','ner'])
nlp.add_pipe(sbd_component, before='parser')  


# create vocabulary
start = time.time()
vocab = {}

notes = pd.read_csv(MIMIC_NOTES_PATH, index_col = 0, usecols = ['ROW_ID', 'ISERROR','SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY','TEXT'])
notes = notes[notes.ISERROR != 1]

notes = notes[notes.CATEGORY == category]

# for other notes
if len(sys.argv) < 2:
        print('Please specify the batch number.')
        sys.exit()

batch = sys.argv[1]
other_notes = pd.read_csv('data/notes_batch_{}.csv'.format(batch))

print("start processing: batch {}".format(batch))
to_process = notes if discharge else other_notes

for text in tqdm(to_process.TEXT):
    sents = sent_tokenize_rules(text)
    for sent in sents:
        sent = re.sub("\[\*\*.{0,15}.*?\*\*\]", "unk", sent)
        if not sent or sent.strip() == '\n':
            continue

        sent = sent.replace('\n', ' ')
        sent = sent.replace('/', ' ')
        
        tokens = nlp(sent)

        for token in tokens:
            word = token.string.strip().lower()
            
            if not word:
                continue

            if word not in vocab:
                vocab[word] = 0
            
            vocab[word] += 1
 
end = time.time()
print("total time: {} seconds".format(end-start))

# sort vocab based on word occurence in descending order
sorted_vocab = collections.OrderedDict(sorted(vocab.items(), key = operator.itemgetter(1), reverse = True))
import json

filename = "discharge_summary" if discharge else "all"
with open('data/{}_vocab_{}.json'.format(filename, batch), 'w') as fp:
    json.dump(sorted_vocab, fp)

print("finished")


