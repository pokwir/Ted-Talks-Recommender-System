#---------------------------------------------Imports------------------------------------------------------#
import pandas as pd
import numpy as np
import os
import sys

import time
from tqdm import tqdm


#---------------------------------------------Data------------------------------------------------------#
# relative path to the data file

data = '/Users/patrickokwir/Desktop/Git_Projects/Ted-Talks-Recommender-System/Data_output/ted_talk_clean_merged_bert.csv'
df = pd.read_csv(data, index_col=0)


#---------------------------------------------BERTSUM------------------------------------------------------#
#------Imports------------#
from summarizer import Summarizer,TransformerSummarizer


#------Functions----------#
pbar = tqdm(total=len(df), colour='green', dynamic_ncols=True)

for i in range(len(df)):
    # loc transcript text of each tedtalk
    body = df.loc[i]['transcript']
    bert_model = Summarizer(hidden=-2, reduce_option= 'mean')
    bert_summary = (bert_model(body, min_length=20, max_length=100))
    # add summary to each tedtalk in df
    df.loc[i,'summary'] = bert_summary
    pbar.update(1)
    pbar.set_description(f'Processed {i+1} of {len(df)}')


df.to_csv('/Users/patrickokwir/Desktop/Git_Projects/Ted-Talks-Recommender-System/Data_output/ted_talk_clean_merged_bert_summary.csv')
    

