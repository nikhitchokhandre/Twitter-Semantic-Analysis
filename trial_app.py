import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch


pipe_lr=joblib.load(open('trial_tweet.pkl', 'rb'))

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#function to read the emotion
def predict_emotions(docx):
    results=pipe_lr.predict([docx] )
    return results

def get_prediction_proba(docx):
    results=pipe_lr.predict_proba([docx] )
    return results

def getPolarity(docx):
    token=tokenizer.encode(docx, return_tensors='pt')
    result=model(token)
    result.logits
    return int(torch.argmax(result.logits))+1


def main():
    st.title('Twitter Semantic Analysis')
    menu=["Home", "Monitor", "About"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Text Box")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Please enter your text")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            
            prediction=predict_emotions(raw_text)
            probability=get_prediction_proba(raw_text)
            
            sentiment_score=getPolarity(raw_text)
            if sentiment_score == 5:
                sentiment_label = "Very_Positive"
            elif sentiment_score == 1:
                sentiment_label = "Very_Negative"
            elif sentiment_score == 3:
                sentiment_label = "Neutral"
            elif sentiment_score == 2:
                sentiment_label = "Negitive"
            elif sentiment_score == 4:
                sentiment_label = "Positive"


            
            st.success('Sentiment')
            st.write(f"Sentiment: {sentiment_label}")
            st.write(f"Sentiment Score: {sentiment_score:.2f}")

            col1,col2 = st.columns(2)
            with col1:
                st.success('Original text')
                st.write(raw_text)
                st.success("Prediction")
                st.write("{}".format(prediction[0]))
                st.write("Confidence: {}".format(np.max(probability)))
                

            with col2:
                st.success('Prediction Probability')
                st.write(probability)
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                st.write(proba_df.transpose())
                proba_df_clean=proba_df.transpose().reset_index()
                proba_df_clean.columns=["class","probability"]

            fig=alt.Chart(proba_df_clean).mark_bar().encode(x='class', y='probability',color='class')
            st.altair_chart(fig,use_container_width=True)
                

    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")
        st.write("This is an NLP powered webapp that can predict sentiment from text with 60 percent accuracy, Many python libraries like Numpy, Pandas, Seaborn, Scikit-learn, Scipy, Joblib, ,textblob, altair, streamlit was used. Streamlit was mainly used for the front-end development, MultinomialNB model from the scikit-learn library was used to train a dataset containing tweets and their respective classes. Joblib was used for storing and using the trained model in the website")
        

if __name__ == "__main__":
    main()