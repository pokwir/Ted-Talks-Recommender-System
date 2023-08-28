import pandas as pd
import numpy as np
from datasets import load_from_disk
import faiss
try:
    from google.colab import drive
    SOURCE_DIR = "/content/drive/MyDrive/MLOPs_Projects/TED_Project/Data_output/T5/"
except:
    SOURCE_DIR = "./Datasets/"
# load the t5_dataset_with_sentence_embeddings   
t5_dataset = load_from_disk(SOURCE_DIR+'t5_embedded_dataset')
t5_dataset.add_faiss_index(column="embeddings")
# create Pandas dataframe
df = pd.DataFrame(t5_dataset[:])

def get_embeddings_with_topic(topic):
    """
    input: topic string
    output: either the topic embeddings or an error 
    """

    # filter by topic
    title = df[df['title'].str.contains(topic)].index
    num_results = len(title)

    # count the number of topics and return if it's not 1
    if num_results != 1:
        print(f"Multiple search results were found for topic '{topic}'.") if num_results else print(f"No search results for topic '{topic}'.")
        print("Performing query search instead.")
        return None, topic

    result = df.loc[title[0]].to_dict()
    topic = f"{result['title']} by {result['author']}"

    embedding = result['embeddings']
    return np.array(embedding).reshape((1,768)), topic


def get_recommendations(topic = None, query=None, num=3):
    """
    input: a query asking for a topic recommendation
    OR
    input: one of the recommender topics
    output: a list of the top 3 most relevant topics
    """
    query_embedding = None
    if topic:
        # call a function to get the embeddings
        query_embedding, topic = get_embeddings_with_topic(topic)
        if query_embedding is None:
            query = topic[:]
            topic = None
    if query:
        print(f"Searching the Ted Talk Database for recommendations based on the query '{query}'.")
        # embed the query with get_embeddings & conver to numpy
        query_embedding = get_embeddings([query]).to('cpu').numpy()

    # use get_nearest_example to get similar embeddings
    scores, samples = t5_dataset.get_nearest_examples(
        'embeddings',
        query_embedding,
        k=10)
    
    # create a df with the results
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df['scores'] = scores
    samples_df = samples_df.sort_values('scores', ascending=False)[:num]

    # print results
    print("\n\nRecommendations based on",
          f"the query '{query}'\n" if query else f"the topic: '{topic}'\n"
          )
    for _, row in samples_df.iterrows():
        print(f"TITLE: {row.title}")
        print(f"AUTHOR: {row.author}")
        print(f"SCORE: {row.scores}")
        print(f"DESCRIPTION: {row.description}")
        print(f"TAGS: {row.tags}")
        print(f'TRANSCRIPT: {" ".join(row.transcript.split(" ")[:20])}')
        print("============")
        print("")