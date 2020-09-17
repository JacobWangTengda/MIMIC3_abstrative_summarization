import os
import nltk
import re
import numpy as np
import pandas as pd
from heuristic_tokenize import sent_tokenize_rules
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#setting sentence boundaries
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

#convert de-identification text into one token
def fix_deid_tokens(text, processed_text):
    deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex,text,flags=re.IGNORECASE)]
    else:
        indexes = []
    for start,end in indexes:
        processed_text.merge(start_idx=start,end_idx=end)
    return processed_text

print("Loading spacy")
nlp = spacy.load('en_core_sci_lg', disable=['tagger','ner'])
nlp.add_pipe(sbd_component, before='parser')  
print("spacy loaded")

# check current directory
# print('current directory:', os.getcwd())

# Read original notes csv file
path = '../../../data/MIMIC3/NOTEEVENTS.csv'
notes = pd.read_csv(path, usecols = ['ROW_ID', 'ISERROR','SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY','TEXT'])

# Drop notes that are identified as erros
notes = notes[notes.ISERROR != 1]
notes.head()

# Pre-processings on notes
notes.dropna(subset = ['HADM_ID'], inplace = True)
notes['HADM_ID'] = notes['HADM_ID'].astype(np.int64)

summary_path = '../../../data/liu/mimic3/CLAMP_NER/input/Rule0/HOSPITAL_COURSE/'
summary_names = os.listdir(summary_path)
# print("sample summary: [SUBJECT_ID]_[EPISODE_ID]_[orderNO].txt :", summary_names[0])

# count the number of episodes with single dischagre summary
summaries = notes[notes.CATEGORY == 'Discharge summary']
# table = summaries.groupby(['SUBJECT_ID', "HADM_ID"]).count()[['ROW_ID']]
# print("total number of episodes: ", table.shape[0])
# print("total number of episodes with one discharge summary ", table[table.ROW_ID == 1].shape[0])
# print("percentage: ", table[table.ROW_ID == 1].shape[0]/table.shape[0])
# conclusion: almost 90% of the episodes has one dischage summary only, so we'll focus on these summaries first

admissions_path = '../../../data/MIMIC3/ADMISSIONS.csv'
admission = pd.read_csv(admissions_path, usecols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME'])
# admission = admission.set_index(['SUBJECT_ID', 'HADM_ID']).copy()[['DISCHTIME']]
admission.DISCHTIME = admission.DISCHTIME.apply(lambda x: str(x).split(' ')[0])


def same_date(subject_id, episode_id):
    """
    Function to check discharge summary is written on the same date when the patient is discharged from the hospital
    """
    discharge_date = admission[(admission.SUBJECT_ID == subject_id) & (admission.HADM_ID == episode_id)].DISCHTIME.iloc[0]
    summary_date = summaries[(summaries.SUBJECT_ID == subject_id) & (summaries.HADM_ID == episode_id)].CHARTDATE.iloc[0]
    
    return discharge_date == summary_date
    

def has_one_summary(subject_id, episode_id):
    """
    Function to check if the given hospital episode has only one discharge summary
    ---------
    Args:
        subject_id
        episode_id
    Return:
        True if the hospital episode has only one discharge summary else False 
    """
    return notes[(notes.SUBJECT_ID == subject_id) & (notes.HADM_ID == episode_id)
                & (notes.CATEGORY == 'Discharge summary')].shape[0] == 1


def extract_description(subject_id, episode_id):
    """
    Function to extract the input to the summariser for an hospital course summary
    * currently focus on those epidoes from one and only discharge summary
    """
    
    date = summaries[(summaries.SUBJECT_ID == subject_id) & (summaries.HADM_ID == episode_id)].CHARTDATE.iloc[0]
    
    # extract 
    relevent_rows = notes[(notes.SUBJECT_ID == subject_id) & (notes.HADM_ID == episode_id) 
                          & (notes.CHARTDATE <= date) & (notes.CATEGORY != 'Discharge summary')]
    
    text = relevent_rows.TEXT.str.cat(sep=' ')
    
    # tokenisation
    sents = sent_tokenize_rules(text)

    output = ""

    for sent in sents:

        # convert to lower case
        sent = sent.lower()
        
        # replace confidential tokens
        sent = re.sub("\[\*+.+\*+\]", "unk", sent)

        # replace patterns like "**** CPT codes *****""
        sent = re.sub('^\*+.+\*+$', "", sent)

        # replace new line character
        sent = sent.replace('\n', ' ')
        sent = sent.replace('/', ' ')

        doc = nlp(sent)
        output += " ".join([token.text for token in doc if token.text.strip()]) + " "

    return '<sec> ' + output.strip() + '\n' if output else None

def extract_summary(file_name):
    """
    Generate hospital course summary in the required format for LeafNATS
    ----------------
    Args:
        file_name: name of the file for the raw summary
    Returns:
        summary: processed hospital course summary
    """

    f = open(summary_path + file_name, 'r')
    summary = f.read()
    
    sections = sent_tokenize_rules(summary)
    
    output = ""

    for sec in sections:
        # convert to lower case
        sec = sec.lower()
        
        # replace confidential tokens
        sec = re.sub("\[\*+.+\*+\]", "unk", sec)

        # replace patterns like "**** CPT codes *****"" 
        sent = re.sub('^\*+.+\*+$', "", sent)

        # replace new line character
        sec = sec.replace('\n', ' ')
        sec = sec.replace('/', ' ')

        for sent in nltk.sent_tokenize(sec):
            output += '<s> ' + ' '.join([token for token in nltk.word_tokenize(sent) if token.strip()]) + ' </s> '

    return output.strip()


def train_test_split(data_path):
    """
    Function to create training, validation and testing data as required by LeafNATS
    """
    # read files
    subject_ids = os.listdir(data_path)
    string = ""
    data = []
    for subject_id in subject_ids[:100]:
        note_path = '{}/{}'.format(data_path, subject_id)
        episode_ids = os.listdir(note_path)
        for episode_id in episode_ids:
            try: 
                input_file = open('{}/{}/input.txt'.format(note_path, episode_id), 'r')
                output_file = open('{}/{}/output.txt'.format(note_path, episode_id), 'r')

                data.append(output_file.read() + " " + input_file.read())
            except:
                print("file error", "subject: ", subject_id, "episode: ", episode_id)


    train_validate, test = train_test_split(data, test_size = 0.1, random_state = 1234, shuffle = True)
    train, validate = train_test_split(train_validate, test_size = 0.1, random_state = 1234, shuffle = True)
    
    # train data
    f = open("data/train.txt", "w")
    f.write(' '.join(train))
    f.close()

    # validation data
    f = open("data/validate.txt", "w")
    f.write(' '.join(validate))
    f.close()

    # test data
    f = open("data/test.txt", "w")
    f.write(' '.join(test))
    f.close()

    return

if __name__ == "__main__":

    data_path = "../../../data/tengda/summary"

    for summary_name in tqdm(summary_names): 
        SUBJECT_ID, EPISODE_ID, _ = list(map(lambda x: int(x), summary_name.split('.')[0].split('_')))
        
        
        # only focus on patient episode with  summary written on the same date as hospital discharge
        # if not same_date(SUBJECT_ID, SUBJECT_ID):
        #     continue

        # only focus on patient episode with exactlty one discharge summary
        if not has_one_summary(SUBJECT_ID, EPISODE_ID):
            continue 
        
        summarizer_input = extract_description(SUBJECT_ID, EPISODE_ID)
        summarizer_output = extract_summary(summary_name)
        
        # check if folder exists
        subject_folder = '{}/{}'.format(data_path, SUBJECT_ID)
        episode_folder = '{}/{}/'.format(subject_folder, EPISODE_ID)
        
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)
        
        if not os.path.exists(episode_folder):
            os.mkdir(episode_folder)
        
        # write file

        if not summarizer_input or not summarizer_output:
            continue
    
        input_file = open(episode_folder + "input.txt", "wt")
        input_file.write(summarizer_input)
        input_file.close()
        
        output_file = open(episode_folder + "output.txt", "wt")
        output_file.write(summarizer_output)
        output_file.close()
    
    train_test_split(data_path)
