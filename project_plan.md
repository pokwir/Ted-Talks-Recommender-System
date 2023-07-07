# Project Plan

### Step 1: Data Scraping

**1.1 Create a list of page URLs:** 
- Generate a list of page URLs to scrape data from.
**1.2 Scrape talk links from pages:** 
- Iterate over each page URL and extract the talk links using BeautifulSoup.
**1.3 Save talk links to a database:** 
- Create a database connection and store the scraped talk links in the database.


## Step 2: Exploratory Data Analysis
**2.1 Load the TED Talks dataset:**
- Read the dataset of TED Talks data into a dataframe.
**2.2 Explore the dataset:**
- Analyze the structure of the dataset.
- Check for missing values and handle them appropriately.
- Examine the distribution of variables such as transcripts, titles, authors, topics, views, and dates.
- Gain insights into the content of TED Talks by analyzing the transcripts and topics.
**2.3 Perform text preprocessing:**
- Clean the text data by removing special characters, punctuation, and stopwords.
- Convert text to lowercase and handle any inconsistencies.
- Apply tokenization, stemming, or lemmatization to standardize the text data.
**2.4 Analyze text statistics and patterns:**
- Calculate word frequency and distribution in the transcripts.
- Identify the most common words or phrases used in TED Talks.
- Explore the relationship between topics, authors, and viewer engagement.

## Step 3: Data Cleaning

**3.1 Handle missing values:**
- Identify missing values in the TED Talks dataset.
- Decide on an appropriate strategy to handle missing values, such as imputation or removal.

**3.2 Clean and preprocess text data:**   CONSIDER WHAT WE ACTUALLY WANT TO REMOVE (CONSIDER HOW SEMANTICS AND WORDING CAN CHANGE MODEL OUTPUT)
- Perform text cleaning techniques such as removing special characters, punctuation, and stopwords.
- Convert text to lowercase and handle any inconsistencies.
- Apply tokenization, stemming, or lemmatization to standardize the text data.
- Remove any irrelevant or noisy text elements.

**3.3 Handle outliers or anomalies:**
- Identify outliers or anomalies in numerical variables such as views or dates.
- Decide on an appropriate approach to handle outliers, such as removing them or applying statistical transformations.

**3.4 Remove duplicate records:**
- Identify and remove any duplicate rows in the dataset.

## Step 4: Model Building and Recommendation

**4.1 Perform feature engineering:**
- Identify relevant features from the TED Talks dataset that could contribute to movie recommendation, such as topics, titles, or authors.
- Extract additional features or create embeddings to capture semantic meaning for better recommendation performance.
- Utilize transformer models like BERT, DialoGPT, GPT, or T5 to process the text data (transcripts, titles) and generate contextualized embeddings.
- Fine-tune the transformer models on the TED Talks dataset to capture specific patterns and relationships.
- Explore different approaches, such as masked language modeling or next sentence prediction, depending on the transformer model used.

## Step 4.2: Generate movie recommendations

4.2.1 Perform feature extraction:
- Utilize the trained transformer models (e.g., BERT, DialoGPT, GPT, T5) to generate embeddings or representations for the TED Talks and movies.
- Extract relevant features from the TED Talks dataset, such as topics, titles, authors, or generated embeddings from the transformer models.

4.2.2 Cluster TED Talks and movies:
- Apply clustering algorithms, such as K-means, DBSCAN, or hierarchical clustering, on the extracted features or embeddings.
- Group similar TED Talks and movies into clusters based on their feature similarities or embeddings.

4.2.3 Generate movie recommendations within clusters:
- For a given TED Talk or user preference, identify the cluster to which it belongs.
- Within the cluster, compute similarity scores or distances between TED Talks and movies based on their features or embeddings.
- Utilize recommendation algorithms, such as K-nearest neighbors (KNN), to select the top-k most similar movies to the TED Talk or user preference.

**4.3 Evaluate and refine the recommendations:**
- Evaluate the generated movie recommendations within each cluster using appropriate metrics, such as precision, recall, or NDCG (Normalized Discounted Cumulative Gain).
- Collect user feedback and ratings to further refine and personalize the movie recommendations.
- Fine-tune the recommendation algorithm based on user feedback and iterate on the feature engineering and clustering processes to improve the recommendations.

## Step 5: Data Validation

**5.1 Validate the processed data format:**
- Ensure that the TED Talks dataset, after data cleaning and feature engineering, is properly preprocessed and formatted for analysis.
- Validate that numerical variables (views, dates) are in the appropriate format.

**5.2 Check for inconsistencies or errors:**
- Perform data quality checks to identify any inconsistencies or errors in the processed dataset. (i.e. coherence scores, similarity checks across semantic scores etc.)
- Look for missing values, duplicate records, or outliers that may affect the recommendation or analysis.
- Validate that the features, embeddings, and topic labels created in the feature engineering step are accurate and appropriate.

**5.3 Rectify inconsistencies or errors:**
- Address any identified inconsistencies or errors in the dataset.
- Decide on an appropriate approach to handle missing values, duplicate records, or outliers.
- Apply data cleaning techniques or imputation methods to rectify errors or inconsistencies.


## Chatbot Project Plan

### Step 1: Data Collection

1. Collect conversational data that includes user queries, responses, and feedback related to movie recommendations.
   - Techniques like web scraping or API integration can be used to gather relevant data.

### Step 2: Data Preprocessing

1. Clean and preprocess the collected conversational data by removing noise, formatting the text, and handling special characters or stopwords.
2. Perform text normalization techniques like lemmatization or stemming to standardize the text.

### Step 3: Intent Recognition

1. Implement an intent recognition system to understand the user's requests and extract relevant information from their input.
   - Use natural language processing (NLP) techniques, such as named entity recognition (NER), to identify entities like movie names, genres, or preferences.

### Step 4: Recommendation System

1. Utilize the recommendation system developed in Step 4.2 to generate movie recommendations based on user preferences and the cluster-based similarity approach.
2. Implement recommendation algorithms like content-based filtering or collaborative filtering to provide personalized recommendations.

### Step 5: Dialogue Management

1. Design a dialogue management system to handle the flow of conversation between the chatbot and the user.
2. Implement a rule-based or machine learning-based approach to generate appropriate responses based on user input and the chatbot's understanding.

### Step 6: User Interface

1. Develop a user interface that allows users to interact with the chatbot and receive movie recommendations.
   - This can be a web-based interface, a command-line interface, or an integration with messaging platforms like Slack or Facebook Messenger.

### Step 7: User Feedback Handling

1. Design a mechanism for users to provide feedback on movie recommendations, including reasons for liking or disliking a recommendation.
2. Capture and store user feedback in a database or file for further analysis and model improvement.

### Step 8: Model Training and Fine-tuning

1. Train and fine-tune the chatbot's language model, including the transformer models like BERT, DialoGPT, GPT, or T5, using the collected conversational data.
2. Use machine learning techniques, such as transfer learning, to improve the chatbot's performance and language understanding.

### Step 9: Continuous Improvement

1. Continuously gather user feedback and evaluate the chatbot's performance.
2. Analyze user feedback to identify areas of improvement and iterate on the chatbot's design, dialogue system, or recommendation mechanism.

### Step 10: Deployment

1. Deploy the chatbot to a server or hosting platform, making it accessible to users.
2. Ensure scalability and reliability by handling multiple user requests concurrently.



Different Transformer Model Options:

**1. BERT (Bidirectional Encoder Representations from Transformers):**
- Benefit for Movie Recommender: BERT captures bidirectional contextual information and is effective in understanding the semantics and context of the text data. It can generate contextualized embeddings for movie recommendations, capturing nuanced relationships between movies and TED Talks.
- Benefit for Chatbot: BERT can be fine-tuned for chatbot tasks, enabling it to understand and generate human-like responses. It can handle conversation context and generate more contextually relevant and meaningful replies.
- Recommended Cleaning: For BERT, perform standard text cleaning techniques such as removing special characters, punctuation, and stopwords. Lowercasing the text, handling inconsistencies, and applying tokenization.
 - Cons: May not be the most suitable for generating interactive or dialogue-based responses.

**2. DialoGPT (Dialogue-based Generative Pre-trained Transformer):**
- Benefit for Movie Recommender: DialoGPT is specifically designed for dialogue generation tasks. It can be leveraged to generate interactive movie recommendations by engaging in a conversation with users and understanding their preferences.
- Benefit for Chatbot: DialoGPT excels in chatbot scenarios as it is trained on conversational data. It can produce coherent and context-aware responses, making it suitable for engaging in meaningful conversations with users.
- Recommended Cleaning: Removing special characters, punctuation, and stopwords, as well as handling lowercase and tokenization, are recommended for DialoGPT.
- Cons: May not be the best choice for tasks that don't involve dialogue or conversation.

**3. GPT (Generative Pre-trained Transformer):**
- Benefit for Movie Recommender: GPT, similar to DialoGPT, can be used to generate movie recommendations by considering user preferences and generating relevant suggestions. It excels at capturing context and generating coherent and informative recommendations.
- Benefit for Chatbot: GPT is known for its ability to generate natural language text and can be employed for chatbot tasks. It can provide detailed responses, understand conversation context, and engage in longer conversations.
- Recommended Cleaning: Cleaning steps such as removing special characters, punctuation, and stopwords, as well as handling lowercase and tokenization, are recommended for GPT models.

**4. T5 (Text-to-Text Transfer Transformer):**
- Benefit for Movie Recommender: T5 is a versatile transformer model that can be fine-tuned for various NLP tasks, including movie recommendation. It can process the TED Talks dataset and generate movie recommendations based on user preferences and talk content.
- Benefit for Chatbot: T5 can be utilized for chatbot tasks by fine-tuning it on conversational data. It can understand user queries and generate appropriate responses, providing a conversational experience.
- Recommended Cleaning: Similar to other transformer models, recommended cleaning steps include removing special characters, punctuation, and stopwords, handling lowercase, and applying tokenization for T5.