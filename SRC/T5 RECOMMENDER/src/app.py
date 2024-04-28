# dummy app
import streamlit as st

with st.sidebar:
    st.header("About")
    st.markdown("""
        TED Talk Recommender using T5 Text-to-Text transformer, Hugging Face Datasets and Semantic Search with FAISS. 
        """)

st.title("T5 Transformer TedTalk Recommender")


from t5 import get_recommendations

# def get_recommendations(topic=None, query=None, num=3):
#     """
#     input: a query asking for a topic recommendation
#     OR
#     input: one of the recommender topics
#     output: a list of the top 3 most relevant topics
#     """

#     if num > 10:
#         raise Exception ("Can only return top 10 or fewer recommendations for now.")


#     if topic:
#         return (f"Your Topic is: {topic}")
#     if query:
#         return (f"Your Query is: {query}")
    
#     else:
#         raise Exception ("An error occured.")
    
    
def main(prompt, p_or_q=False, num=5):
    # prompt
    try:
        if p_or_q: # topic
            results_df = get_recommendations(topic=prompt, num=num) 
        else: # query
            results_df = get_recommendations(query=prompt, num=num)

        # print results
        st.write("\n\nRecommendations based on",
              f"the query '{prompt}'\n" if not p_or_q else f"the topic: '{prompt}'\n"
              )
        for _, row in results_df.iterrows():
            st.write(f"TITLE: {row.title}")
            st.write(f"AUTHOR: {row.author}")
            st.write(f"DESCRIPTION: {row.description}")
            st.write(f"TAGS: {row.tags}")
#             st.write(f'TRANSCRIPT: {" ".join(row.transcript.split(" ")[:20])}')
            st.write("---")
            st.write("")   
    
    except Exception as e:
        st.error(e)
        
        
        
with st.form("my_form"):
    
    prompt = st.text_input('Prompt or query.')
    st.write("---")
    
    col1, col2 = st.columns(2) 
    with col1:
        p_or_q = st.toggle('Defaults to query. Toggle for prompt')  
        if p_or_q:
            st.write("Prompt")
        else:
            st.write("Query")
    with col2:
        num = st.slider('Number of recommendations', 1, 5, 3, step=1)
    
    submit = st.form_submit_button('Get recommendation')
    


    
if submit and not prompt:
    st.error("You must provide either a prompt or query.")
else:
    print("main")
    main(prompt, p_or_q, num)
    