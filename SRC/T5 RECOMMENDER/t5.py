import argparse
parser = argparse.ArgumentParser(description="T5-based Recommender")
parser.add_argument("--topic", help="Topic for recommendations")
parser.add_argument("--query", help="Query for recommendations")
parser.add_argument("--num", type=int, default=3, help="Number of recommendations to show")
args = parser.parse_args()
  
import os   

import pandas as pd
import numpy as np
from datasets import load_from_disk
import faiss

import subprocess
import sys

def install(name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', name])

for package in ['datasets', 'transformers', 'sentence_transformers', 'faiss-cpu']:
    install (package)
    os.system('clear')
    
    
    
    
try:
    from google.colab import drive
    install ('faiss-gpu')
    SOURCE_DIR = "/content/drive/MyDrive/MLOPs_Projects/TED_Project/Data_output/T5/"
except:
    install ('faiss-cpu')
    SOURCE_DIR = "./Datasets/"
# load the t5_dataset_with_sentence_embeddings   
t5_dataset = load_from_disk(SOURCE_DIR+'t5_embedded_dataset')
t5_dataset.add_faiss_index(column="embeddings")
# create Pandas dataframe
df = pd.DataFrame(t5_dataset[:])


# import the t5-base model and tokenizer
from transformers import AutoTokenizer, AutoModel
model_ckpt = 'sentence-transformers/sentence-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# word_embedding_model = models.Transformer(model_ckpt)
# model = SentenceTransformer(modules=[word_embedding_model])

import torch
# set device diagnostics
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# create a get_embeddings function

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text):
    # tokenizes the text column
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # change to the gpu device
    encoded_input = {k:v.to(device) for k, v in encoded_input.items()}

    # Forward pass through the model to obtain embeddings
    with torch.no_grad():
        model_output = model(**encoded_input,
                             # required for t5 transformer
                             decoder_input_ids=torch.tensor([[0]]).to(device),
                             return_dict=True)
    # feed model output to cls pooling
    return cls_pooling(model_output)
    

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
    clear_output(wait=True)
    display(HTML('<script>Jupyter.notebook.clear_all_output()</script>'))
    os.system('clear')
    if num > 10:
        return ("Can only return top 10 or fewer recommendations for now.")
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
        # if the topic cannot be found in the dataset, then it becomes a query
        if query_embedding is None:
            query = topic[:]
            topic = None
    if query:
        print(f"Searching the Ted Talk Database for recommendations based on the query '{query}'.")
        # embed the query with get_embeddings & convert to numpy
        query_embedding = get_embeddings([query]).to('cpu').numpy()
        
        
    if query_embedding is not None:
        # Ensure query_embedding is of type float32 to avoid a type error
        query_embedding = query_embedding.astype('float32')
        # Use get_nearest_example to get similar embeddings
        scores, samples = t5_dataset.get_nearest_examples(
                'embeddings',
                query_embedding,
                k=10)
                
                
                

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
        
        
        
        
def main():
    # Your code to call get_recommendations or other functions based on CLI arguments
    if args.topic or args.query:
        get_recommendations(topic=args.topic, query=args.query, num=args.num)
    else:
        print("You must provide either a --topic or a --query to get recommendations.")

if __name__ == "__main__":
    main()