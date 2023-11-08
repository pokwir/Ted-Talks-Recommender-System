# T5 Recommender

TED Talk Recommender using [T5 Text-to-Text transformer](https://arxiv.org/abs/1910.10683), Hugging Face Datasets and Semantic Search with FAISS. Compatible with Linux and UNIX.

# Building the Recommender

1. Dataset:

   - Import `ted_talk_clean_merged_bert`

   - Extract only required columns

   - Create a huggingface dataset

   - Save the dataset to `t5_dataset`

2. Semantic search with FAISS

   - create new text column that concatenates title, description and transcript.

   - create `get_embeddings` function that:

     - tokenizes the `text` column
     - forward pass token tensors through model to get `output`
     - feed model `output` to CLS pooling

   - create embeddings with `get_embeddings` function

   - add FAISS index

   - save the dataset as `t5_with_sentence_embedding_dataset`.

3. Testing

   - embed sample query with `get_embeddings`

   - use `.get_nearest_examples()` method to get similar embeddings

   - create `get_recommendations` to give results in pandas dataframe

# Execution

### 1. Download the sub-directory

In Terminal, download this sub-directory with:

```bash
#!/bin/bash

git clone https://github.com/pokwir/Ted-Talks-Recommender-System.git
cd Ted-Talks-Recommender-System
git config core.sparseCheckout true
echo "SRC/T5 RECOMMENDER" >> .git/info/sparse-checkout
git read-tree -mu HEAD
```

### 2. Run the script

Run from terminal with either:

`!python t5.py --topic "example topic" `

or

`!python t5.py --query "example query" `

**Required**: datasets.zip

### 3. Troubleshooting

If you get errors, confirm that you're running from within the src directory.

# Next Steps:

Develop the Chatbot with [Streamlit.io Platform](https://streamlit.io/).
