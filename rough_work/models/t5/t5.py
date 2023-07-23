# import libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained(
    't5-small', return_dict=True)


def text_summarizer(transcript):
    """
    input: text
    output: 10-80-word summary of text
    """
    transcript = transcript.strip().replace("\n", " ")
    inputs = tokenizer.encode("summarize: " + transcript,
                              return_tensors='pt',
                              max_length=len(transcript.split()),
                              truncation=True)
    summarization_ids = model.generate(inputs, max_length=80, min_length=10,
                                       no_repeat_ngram_size=3,
                                       length_penalty=5., num_beams=1,
                                       early_stopping=True)
    summarization = tokenizer.decode(
        summarization_ids[0], skip_special_tokens=True)
    return summarization


def similarity(stsb_sentence_1, stsb_sentence_2):
    """
    input: 2 texts
    output: similarity score 0-5, with 5 as maximum similarity and 0 as no similarity
    """
    input_ids = tokenizer("stsb sentence 1: "+stsb_sentence_1 +
                          " sentence 2: "+stsb_sentence_2, return_tensors="pt").input_ids
    stsb_ids = model.generate(input_ids)
    stsb = tokenizer.decode(stsb_ids[0], skip_special_tokens=True)
    return stsb


def similarity_matrix(df, feature):
    """
    input: dataframe with 'summary' feature
    returns similarity matrix by cross-comparing the rows and column values
    """
    # values = df['summary'].values # values will be the sentence summaries
    values = df.feature.values
    # make a df with rows and column names from summary columns
    df_summary = pd.DataFrame(index=values, columns=values)

    # loop through rows & columns to do similarity (row,col)
    df = df_summary.copy()
    for row_idx, row in tqdm(enumerate(df.index)):
        for col in df.columns[row_idx:]:
            # similarity function
            df.loc[row, col] = similarity(row, col)
    # mirror the df across the diagonal
    for row_idx, row in enumerate(df.index):
        for col in df.columns[row_idx:]:
            df.loc[col, row] = df.loc[row, col]
    return df

# sim_matrix = similarity_matrix(df_summary)
