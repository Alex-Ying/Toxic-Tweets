import streamlit as st
import time
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import numpy as np
import pandas as pd


os.environ['KMP_DUPLICATE_LIB_OK'] = "True"


st.title("Sentiment Analysis App")
if 'logs' not in st.session_state:
    st.session_state.logs = dict()
if 'labels' not in st.session_state:
    st.session_state.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
if 'id2label' not in st.session_state:
    st.session_state.id2label = {idx: label for idx, label in enumerate(st.session_state.labels)}
if 'filled' not in st.session_state:
    st.session_state.filled = False
if 'model' not in st.session_state:
    st.session_state.model = AutoModelForSequenceClassification.from_pretrained("Ptato/Modified-Bert-Toxicity-Classification")
    st.session_state.model.eval()
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

form = st.form(key='Sentiment Analysis')
st.session_state.options = [
    'bertweet-base-sentiment-analysis',
           'distilbert-base-uncased-finetuned-sst-2-english',
           'twitter-roberta-base-sentiment',
           'Modified Bert Toxicity Classification'
           ]
box = form.selectbox('Select Pre-trained Model:', st.session_state.options, key=1)
tweet = form.text_input(label='Enter text to analyze:', value="\"We've seen in the last few months, unprecedented amounts of Voter Fraud.\" @SenTedCruz True!")
submit = form.form_submit_button(label='Submit')
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv("test.csv")

if not st.session_state.filled:
    for s in st.session_state.options:
        st.session_state.logs[s] = []
if not st.session_state.filled:
    st.session_state.filled = True
    for x in range(10):
        print(x)
        text = st.session_state.df["comment_text"].iloc[x][:128]
        for s in st.session_state.options:
            pline = None
            predictions = None
            encoding = None
            logits = None
            probs = None
            if s == 'bertweet-base-sentiment-analysis':
                pline = pipeline(task="sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
            elif s == 'twitter-roberta-base-sentiment':
                pline = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            elif s == 'distilbert-base-uncased-finetuned-sst-2-english':
                pline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            else:
                encoding = st.session_state.tokenizer(text, return_tensors="pt")
                encoding = {k: v.to(st.session_state.model.device) for k, v in encoding.items()}
                predictions = st.session_state.model(**encoding)
                logits = predictions.logits
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze().cpu())
                predictions = np.zeros(probs.shape)
                predictions[np.where(probs >= 0.5)] = 1
                predicted_labels = [st.session_state.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
            log = []
            if pline:
                predictions = pline(text)
                log = [0] * 4
                log[1] = text
                for p in predictions:
                    if s == 'bertweet-base-sentiment-analysis':
                        if p['label'] == "POS":
                            log[0] = 0
                            log[2] = "POSITIVE"
                            log[3] = f"{ round(p['score'] * 100, 1)}%"
                        elif p['label'] == "NEU":
                            log[0] = 2
                            log[2] = f"{ p['label'] }"
                            log[3] = f"{round(p['score'] * 100, 1)}%"
                        else:
                            log[2] = "NEG"
                            log[0] = 1
                            log[3] = f"{round(p['score'] * 100, 1)}%"
                    elif s == 'distilbert-base-uncased-finetuned-sst-2-english':
                        if p['label'] == "POSITIVE":
                            log[0] = 0
                            log[2] = "POSITIVE"
                            log[3] = (f"{round(p['score'] * 100, 1)}%")
                        else:
                            log[2] = ("NEGATIVE")
                            log[0] = 1
                            log[3] = (f"{round(p['score'] * 100, 1)}%")
                    elif s == 'twitter-roberta-base-sentiment':
                        if p['label'] == "LABEL_2":
                            log[0] = 0
                            log[2] = ("POSITIVE")
                            log[3] = (f"{round(p['score'] * 100, 1)}%")
                        elif p['label'] == "LABEL_0":
                            log[0] = 1
                            log[2] = ("NEGATIVE")
                            log[3] = f"{round(p['score'] * 100, 1)}%"
                        else:
                            log[0] = 2
                            log[2] = "NEUTRAL"
                            log[3] = f"{round(p['score'] * 100, 1)}%"
            else:
                log = [0] * 6
                log[1] = text
                if max(predictions) == 0:
                    log[0] = 0
                    log[2] = ("NO TOXICITY")
                    log[3] = (f"{100 - round(probs[0].item() * 100, 1)}%")
                    log[4] = ("N/A")
                    log[5] = ("N/A")
                else:
                    log[0] = 1
                    _max = 0
                    _max2 = 2
                    for i in range(1, len(predictions)):
                        if probs[i].item() > probs[_max].item():
                            _max = i
                        if i > 2 and probs[i].item() > probs[_max2].item():
                            _max2 = i
                    log[2] = (st.session_state.labels[_max])
                    log[3] = (f"{round(probs[_max].item() * 100, 1)}%")
                    log[4] = (st.session_state.labels[_max2])
                    log[5] = (f"{round(probs[_max2].item() * 100, 1)}%")
            st.session_state.logs[s].append(log)

if submit and tweet:
    with st.spinner('Analyzing...'):
        time.sleep(1)

    if tweet is not None:
        pline = None
        if box != 'Modified Bert Toxicity Classification':
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
        if box == 'bertweet-base-sentiment-analysis':
            pline = pipeline(task="sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        elif box == 'twitter-roberta-base-sentiment':
            pline = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        elif box == 'distilbert-base-uncased-finetuned-sst-2-english':
            pline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            encoding = st.session_state.tokenizer(tweet, return_tensors="pt")
            encoding = {k: v.to(st.session_state.model.device) for k,v in encoding.items()}
            predictions = st.session_state.model(**encoding)
            logits = predictions.logits
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze().cpu())
            print(probs[0].item())
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= 0.5)] = 1
            predicted_labels = [st.session_state.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
        if pline:
            predictions = pline(tweet)
            col2.header("Judgement")
        else:
            col2.header("Category")
            col4.header("Type")
            col5.header("Score")

        col1.header("Tweet")
        col3.header("Score")

        if pline:
            log = [0] * 4
            log[1] = tweet
            for p in predictions:
                if box == 'bertweet-base-sentiment-analysis':
                    if p['label'] == "POS":
                        col1.success(tweet.split("\n")[0][:20])
                        log[0] = 0
                        col2.success("POS")
                        col3.success(f"{ round(p['score'] * 100, 1)}%")
                        log[2] = ("POS")
                        log[3] = (f"{ round(p['score'] * 100, 1)}%")
                    elif p['label'] == "NEU":
                        col1.warning(tweet.split("\n")[0][:20])
                        log[0] = 2
                        col2.warning(f"{ p['label'] }")
                        col3.warning(f"{round(p['score'] * 100, 1)}%")
                        log[2] = ("NEU")
                        log[3] = (f"{round(p['score'] * 100, 1)}%")
                    else:
                        log[0] = 1
                        col1.error(tweet.split("\n")[0][:20])
                        col2.error("NEG")
                        col3.error(f"{round(p['score'] * 100, 1)}%")
                        log[2] = ("NEG")
                        log[3] = (f"{round(p['score'] * 100, 1)}%")
                elif box == 'distilbert-base-uncased-finetuned-sst-2-english':
                    if p['label'] == "POSITIVE":
                        col1.success(tweet.split("\n")[0][:20])
                        log[0] = 0
                        col2.success("POSITIVE")
                        log[2] = "POSITIVE"
                        col3.success(f"{round(p['score'] * 100, 1)}%")
                        log[3] = f"{round(p['score'] * 100, 1)}%"
                    else:
                        col2.error("NEGATIVE")
                        col1.error(tweet.split("\n")[0][:20])
                        log[2] = ("NEGATIVE")
                        log[0] = 1
                        col3.error(f"{round(p['score'] * 100, 1)}%")
                        log[3] = f"{round(p['score'] * 100, 1)}%"
                elif box == 'twitter-roberta-base-sentiment':
                    if p['label'] == "LABEL_2":
                        log[0] = 0
                        col1.success(tweet.split("\n")[0][:20])
                        col2.success("POSITIVE")
                        col3.success(f"{round(p['score'] * 100, 1)}%")
                        log[3] = f"{round(p['score'] * 100, 1)}%"
                        log[2] = "POSITIVE"
                    elif p['label'] == "LABEL_0":
                        log[0] = 1
                        col1.error(tweet.split("\n")[0][:20])
                        col2.error("NEGATIVE")
                        col3.error(f"{round(p['score'] * 100, 1)}%")
                        log[3] = f"{round(p['score'] * 100, 1)}%"
                        log[2] = "NEGATIVE"
                    else:
                        log[0] = 2
                        col1.warning(tweet.split("\n")[0][:20])
                        col2.warning("NEUTRAL")
                        col3.warning(f"{round(p['score'] * 100, 1)}%")
                        log[3] = f"{round(p['score'] * 100, 1)}%"
                        log[2] = "NEUTRAL"
                for a in st.session_state.logs[box][::-1]:
                    if a[0] == 0:
                        col1.success(a[1].split("\n")[0][:20])
                        col2.success(a[2])
                        col3.success(a[3])
                    elif a[0] == 1:
                        col1.error(a[1].split("\n")[0][:20])
                        col2.error(a[2])
                        col3.error(a[3])
                    else:
                        col1.warning(a[1].split("\n")[0][:20])
                        col2.warning(a[2])
                        col3.warning(a[3])
                st.session_state.logs[box].append(log)
        else:
            log = [0] * 6
            log[1] = tweet
            if max(predictions) == 0:
                col1.success(tweet.split("\n")[0][:10])
                col2.success("NO TOXICITY")
                col3.success(f"{100 - round(probs[0].item() * 100, 1)}%")
                col4.success("N/A")
                col5.success("N/A")
                log[0] = 0
                log[2] = "NO TOXICITY"
                log[3] = (f"{100 - round(probs[0].item() * 100, 1)}%")
                log[4] = ("N/A")
                log[5] = ("N/A")
            else:
                _max = 0
                _max2 = 2
                for i in range(1, len(predictions)):
                    if probs[i].item() > probs[_max].item():
                        _max = i
                    if i > 2 and probs[i].item() > probs[_max2].item():
                        _max2 = i
                col1.error(tweet.split("\n")[0][:10])
                col2.error(st.session_state.labels[_max])
                col3.error(f"{round(probs[_max].item() * 100, 1)}%")
                col4.error(st.session_state.labels[_max2])
                col5.error(f"{round(probs[_max2].item() * 100, 1)}%")
                log[0] = 1
                log[2] = (st.session_state.labels[_max])
                log[3] = (f"{round(probs[_max].item() * 100, 1)}%")
                log[4] = (st.session_state.labels[_max2])
                log[5] = (f"{round(probs[_max2].item() * 100, 1)}%")
            for a in st.session_state.logs[box][::-1]:
                if a[0] == 0:
                    col1.success(a[1].split("\n")[0][:10])
                    col2.success(a[2])
                    col3.success(a[3])
                    col4.success(a[4])
                    col5.success(a[5])
                elif a[0] == 1:
                    col1.error(a[1].split("\n")[0][:10])
                    col2.error(a[2])
                    col3.error(a[3])
                    col4.error(a[4])
                    col5.error(a[5])
                else:
                    col1.warning(a[1].split("\n")[0][:10])
                    col2.warning(a[2])
                    col3.warning(a[3])
                    col4.warning(a[4])
                    col5.warning(a[5])
            st.session_state.logs[box].append(log)