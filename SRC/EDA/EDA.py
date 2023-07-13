# ----------------------------------------Imports--------------------------------------------# 

import pandas as pd
import numpy as np

from lets_plot import *
LetsPlot.setup_html()

import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from collections import defaultdict

import collections 
Counter = collections.Counter

from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('../Data_output/talks_2.csv').drop(columns='Unnamed: 0')

df = df.dropna()
df = df.drop_duplicates()
# checking the duplicated transcript
# display(df[df['transcript'].duplicated(keep=False)])
# delete that record
df = df.drop(index=df[df['transcript'].duplicated(keep='last')].index)
# removing wrong values in likes and views
df = df[pd.to_numeric(df['likes'], errors='coerce').notna()]
df = df[pd.to_numeric(df['views'], errors='coerce').notna()]
# replacing special characters (e.g. $quot;) with proper syntax
df['transcript'] = df['transcript'].replace(['&quot;','&apos;', '&amp;', '&lt;', '&gt;'], 
                                            ["'","'"," and ", " less than ", " greater than "], regex=True)
# updating datatype                                            
df['likes'] = df['likes'].astype('int')
df['views'] = df['views'].astype('int')   
# optional save to file
# df.to_csv('../Data_output/ted_talk_clean.csv')   
# 
# ----------------------------------------EDA--------------------------------------------# 
# 
# 
# ----------------------------------------Word Frequencies--------------------------------------------#
# Data preprocessing
#  function to count the frequency of each word in a trsnacript
def count_words(df):
   # get sentence length for each transcript 
    df['sentence_length'] = df['transcript'].apply(lambda x: len(x.split('.')))

# get number of words for each transcript 
    df['word_count'] = df['transcript'].apply(lambda x: len(x.split()))

# get average word length for each transcript 
    df['avg_word_length'] = df['transcript'].apply(lambda x: np.mean([len(w) for w in x.split()]))
    return df

df = count_words(df)


# ----------------------------------------Visualizations--------------------------------------------#
# plot of sentence length
p = ggplot(df, aes(x='sentence_length')) + \
    geom_histogram(bins=50, color= 'white', fill = 'black') + \
    labs(x='Sentence Length', y=' ') + \
    theme_classic() + \
    theme(axis_text_x = element_text(angle = 0, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'))+ \
    flavor_high_contrast_dark() + \
    ggtitle('Histogram of Sentence Length') + font_family_info('Avenir', mono=False)
ggsave(plot = p, filename='sentence_length_histogram.png', path='../Images')

# Histogram of word count
p = ggplot(df, aes(x='word_count')) + \
    geom_histogram(bins=30, color= 'white', fill = 'black') + \
    labs(x='Word Count', y=' ') + \
    theme_classic() + \
    theme(axis_text_x = element_text(angle = 0, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'))+ \
    flavor_high_contrast_dark() + \
    ggtitle('Histogram of Word Counts') + font_family_info('Avenir', mono=False)
ggsave(plot = p, filename='Histogram of Word Counts.png', path='../Images')

#Histogram of average word length
p = ggplot(df, aes(x='avg_word_length')) + \
    geom_histogram(bins=50, color= 'white', fill = 'black') + \
    labs(x='Average Word Length', y=' ') + \
    theme_classic() + \
    theme(axis_text_x = element_text(angle = 0, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'))+ \
    flavor_high_contrast_dark() + \
    ggtitle('Histogram of Average Word Length') + font_family_info('Avenir', mono=False)

ggsave(plot = p, filename='Histogram of Average Word Length', path='../Images')

# ------------stop words analysis -------------------#

data = df['transcript']
data.reset_index(drop=True, inplace=True)

corpus=[]
new= data.str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

# create a stop words df
stop_df = pd.DataFrame.from_dict(dic, orient='index')
stop_df = stop_df.reset_index()
stop_df = stop_df.rename(columns={'index':'word', 0:'count'})


stop_df = stop_df.sort_values(by='count', ascending=False)
# select top 10% of stopwords
stop_df = stop_df.iloc[:round(len(stop_df)*0.1), :]

# plot stopwords
p = ggplot(stop_df, aes(x='word', y='count')) + \
    geom_bar(stat='identity', color= 'white', fill = 'black') + \
    theme_classic() + \
    theme(axis_text_x = element_text(angle = 0, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'), axis_line_x= element_blank())+ \
    flavor_high_contrast_dark() + scale_y_continuous(name='', breaks=[50000, 250000, 400000])+ ggsize(800, 500) +\
    ggtitle('Count of Stop Words')
ggsave(plot = p, filename='Count of Stop Words.png', path='../Images')


# -----------------------most common words-----------------------------#
counter=Counter(corpus)
most=counter.most_common()

x, y= [], []
for word,count in most[:70]:
    if (word not in stop):
        x.append(word)
        y.append(count)

# x, y to a dataframe
df_w_c = pd.DataFrame({'word':x, 'count':y})


# plot most common words
p = ggplot(df_w_c, aes(x='word', y='count')) + \
    geom_bar(stat='identity', color= 'white', fill = 'black') + \
    theme_classic() + \
    theme(axis_text_x = element_text(angle = 0, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'), axis_line_x= element_blank())+ \
    flavor_high_contrast_dark() + scale_y_continuous(name='', breaks=[0, 75000, 129000])+ ggsize(800, 500) +\
    ggtitle('Most Common Words')
ggsave(plot = p, filename='Most Common Words.png', path='../Images')


# -----------------------Ngeam exploration -----------------------------#
# function to get ngrams
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:15]

# get top 10 ngrams
top_bigrams = get_top_ngram(corpus, 2)[:15]
top_trigrams = get_top_ngram(corpus, 3)[:15]

# create dataframe 
top_bigrams = pd.DataFrame(top_bigrams, columns=['bigram', 'count'])
top_bigrams = top_bigrams.sort_values(by='count', ascending=False)  # sort by count

top_trigrams = pd.DataFrame(top_trigrams, columns=['trigram', 'count'])
top_trigrams = top_trigrams.sort_values(by='count', ascending=False)  # sort by count

# plot ngrams - Bigrams
# select top 10 bigrams
top_bigrams = top_bigrams.head(10)
p = ggplot(top_bigrams, aes(x='bigram', y='count')) +\
    geom_lollipop(stat='identity',fatten=5, linewidth=0.5, stroke = 4, color = 'red') +\
    theme_classic() +\
    theme(axis_text_x = element_text(angle = 0, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'), axis_line_x= element_blank())+ \
    flavor_high_contrast_dark() + scale_y_continuous(name='', breaks=[0, 6000, 13000])+ ggsize(800, 500)+\
    ggtitle('Bigram word count distribution')
ggsave(plot = p, filename='Bigram word count distribution.png', path='../Images')

# plot ngrams - Trigrams
top_trigrams = top_trigrams.head(10)

p = ggplot(top_trigrams, aes(x='trigram', y='count')) +\
    geom_lollipop(stat='identity',fatten=5, linewidth=0.5, stroke = 4, color = 'red') +\
    theme_classic() +\
    theme(axis_text_x = element_text(angle = 45, hjust = 1, color='orange'), axis_line_y= element_line(color= 'black'), axis_line_x= element_blank())+ \
    flavor_high_contrast_dark() + scale_y_continuous(name='', breaks=[0, 60, 112])+ ggsize(800, 500)+\
    ggtitle('Trigram word count distribution')
ggsave(plot = p, filename='Trigram word count distribution.png', path='../Images')


# -----------------------Word Cloud-----------------------------#