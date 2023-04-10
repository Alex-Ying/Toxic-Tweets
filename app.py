import streamlit as st
import time
from transformers import pipeline
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"



st.title("Sentiment Analysis App")

form = st.form(key='Sentiment Analysis')
box = form.selectbox('Select Pre-trained Model:', ['bertweet-base-sentiment-analysis',
                                                   'distilbert-base-uncased-finetuned-sst-2-english',
                                                   'twitter-roberta-base-sentiment'
                                                   ], key=1)
tweet = form.text_input(label='Enter text to analyze:', value="\"We've seen in the last few months, unprecedented amounts of Voter Fraud.\" @SenTedCruz True!")
submit = form.form_submit_button(label='Submit')

if submit and tweet:
    with st.spinner('Analyzing...'):
        time.sleep(1)
    # st.header(tweet)

    if tweet is not None:
        col1, col2, col3 = st.columns(3)
        if box == 'bertweet-base-sentiment-analysis':
            pipeline = pipeline(task="sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        elif box == 'twitter-xlm-roberta-base-sentiment':
            pipeline = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        else:
            pipeline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        predictions = pipeline(tweet)
        print(predictions)
        col1.header("Tweet")
        col1.subheader(tweet)
        col2.header("Judgement")
        col3.header("Probability")
        for p in predictions:
            if box == 'bertweet-base-sentiment-analysis':
                if p['label'] == "POS":
                    col2.success(f"{ p['label'] }")
                    col3.success(f"{ round(p['score'] * 100, 1)}%")
                elif p['label'] == "NEU":
                    col2.warning(f"{ p['label'] }")
                    col3.warning(f"{round(p['score'] * 100, 1)}%")
                else:
                    col2.error(f"{p['label']}")
                    col3.error(f"{round(p['score'] * 100, 1)}%")
            elif box == 'distilbert-base-uncased-finetuned-sst-2-english':
                if p['label'] == "POSITIVE":
                    col2.success(f"{p['label']}")
                    col3.success(f"{round(p['score'] * 100, 1)}%")
                else:
                    col2.error(f"{p['label']}")
                    col3.error(f"{round(p['score'] * 100, 1)}%")
            else:
                if p['label'] == "POSITIVE":
                    col2.success(f"{p['label']}")
                    col3.success(f"{round(p['score'] * 100, 1)}%")
                else:
                    col2.error(f"{p['label']}")
                    col3.error(f"{round(p['score'] * 100, 1)}%")