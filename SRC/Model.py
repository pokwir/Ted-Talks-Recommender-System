#---------------------------------------Imports----------------------------------------# 
import numpy as np
# import pandas
import pandas as pd

# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
import string

# ---------------------------------------Functions-------------------------------------------# 


# ---------------------------------------Read The Data----------------------------------------#
data = '/Users/patrickokwir/Desktop/Git_Projects/Ted-Talks-Recommender-System/Data_output/talks.csv'
df = pd.read_csv(data, index_col=0)

# ---------------------------------------Clean The Data-------------------------------------------#
stop_words = set(stopwords.words('english'))
df['description'] = df['description'].str.strip()
df['description'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df['description'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in string.punctuation]))
df['description'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in string.digits]))

# ---------------------------------------Vectorize The Data----------------------------------------#
tfidf = TfidfVectorizer(stop_words='english')
# fit and transform 'description' column to get the tfidf matrix
tfidf_matrix = tfidf.fit_transform(df['description'])

# ---------------------------------------Cosine Similarity----------------------------------------#
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------------------------------Results-------------------------------------------------#
results = {}
for i in range(len(cosine_sim)):
    results[i] = cosine_sim[i].argsort()[:-11:-1].tolist()

# split the 'description' so it returns only the item
df['item'] = df['description'].str.split('-').str[0]
# create a reverse map of indices and descriptions
indices = pd.Series(df.index, index=df['item']).drop_duplicates()

# ---------------------------------------Recommendations-------------------------------------------------#
# now we create a function recomender that will recommend simmilar products, function must take item_id and count as parameters
cosine_sim=cosine_sim

talk_to_search = input('whats the title of the talk?')
talk_to_search = talk_to_search.lower()
top_n_results = input('how many results would you like to return?')
top_n_results = int(top_n_results)

results = []
def recomender(talk_to_search, top_n_results):
   
    count = top_n_results

    id = df.loc[df['talk'] == talk_to_search].index.values[0]
       # Get the index of the item that matches the title
    idx = indices[id]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    item_indicies = [i[0] for i in sim_scores]

    # Return the top 10 most similar talks
    top_talks_idx = df['item'].iloc[item_indicies].index[:count]
    # get author, talk using top_talks_idx
    top_talks_author = df['author'].iloc[item_indicies].values[:count]
    top_talks_talk = df['talk'].iloc[item_indicies].values[:count]

    # create a result df 
    result_df = pd.DataFrame({'author': top_talks_author, 'talk': top_talks_talk})

    # rename columns
    result_df.columns = ['author', 'talk']

    print(result_df)

recomender(talk_to_search = talk_to_search, top_n_results = top_n_results)
# ---------------------------------------Recommendations-------------------------------------------------#