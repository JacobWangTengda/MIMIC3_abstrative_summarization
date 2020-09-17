import pandas as pd
import numpy as np

MIMIC_NOTES_PATH = '../../../data/MIMIC3/NOTEEVENTS.csv'
notes = pd.read_csv(MIMIC_NOTES_PATH, index_col = 0, usecols = ['ROW_ID', 'ISERROR','SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY','TEXT'])
notes = notes[notes.CATEGORY != "Discharge summary"]


batch = notes.shape[0]//15

for g, df in notes.groupby(np.arange(notes.shape[0])//batch):
    df.to_csv("data/notes_batch_{}.csv".format(g))

