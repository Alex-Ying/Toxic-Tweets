# Import stuff
import streamlit as st
import time
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import numpy as np
import pandas as pd

# Mitigates an error on Macs
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Set the titel
st.title("Sentiment Analysis App")

# Set the variables that should not be changed between refreshes of the app.


# logs is a map that records the results of past sentiment analysis queries.
#     Type: dict() {"key" --> value[]}
#         key: model_name (string)    - The name of the model being used
#         value: log[] (list)         - The list of values that represent the model's results
#             --> For the pretrained labels, len(log) = 4
#                 --> log[0] (int) - The prediction of the model on its input
#                     --> 0 = Positive
#                     --> 1 = Negative
#                     --> 2 = Neutral (if applicable)
#                 --> log[1] (string) - The tweet/inputted string
#                 --> log[2] (string) - The judgement of the tweet/input (Positive/Neutral/Negative)
#                 --> log[3] (string) - The score of the prediction (includes '%' sign)
#             --> For the finetuned model, len(log) = 6
#                 --> log[0] (int) - The prediction of the model on the toxicity of the input
#                     --> 0 = Nontoxic
#                     --> 1 = Toxic
#                 --> log[1] (string) - The tweet/inputted string
#                 --> log[2] (string) - The highest scoring overall category of toxicity out of:
#                     'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', and 'identity_hate'
#                 --> log[3] (string) - The score of log[2] (includes '%' sign)
#                 --> log[4] (string) - The predicted type of toxicity, the highest scoring category of toxicity out of:
#                     'obscene', 'threat', 'insult', and 'identity_hate'
#                 --> log[5] (string) - The score of log[4] (includes '%' sign)

if 'logs' not in st.session_state:
    st.session_state.logs = dict()
    
# labels is a list of toxicity categories for the finetuned model
if 'labels' not in st.session_state:
    st.session_state.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# filled is a boolean that checks whether logs is prepopulated with data.
if 'filled' not in st.session_state:
    st.session_state.filled = False

# model is the finetuned model that I created. It wasn't working well locally on HuggingFace so I uploaded it to HuggingFace as
#   a pretrained model. I also set it to evaluation mode.
if 'model' not in st.session_state:
    st.session_state.model = AutoModelForSequenceClassification.from_pretrained("Ptato/Modified-Bert-Toxicity-Classification")
    st.session_state.model.eval()


# tokenizer is the same tokenizer that is used by the "bert-base-uncased" model, which my finetuned model is built off of.
#   tokenizer is used to input the tweets into my model for prediction.

if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# This form allows users to select their preferred model for training
form = st.form(key='Sentiment Analysis')

# st.session_state.options pre-sets the available model choices.
st.session_state.options = [
    'bertweet-base-sentiment-analysis',
           'distilbert-base-uncased-finetuned-sst-2-english',
           'twitter-roberta-base-sentiment',
           'Modified Bert Toxicity Classification'
           ]

# box is the dropdown box that users use to select their choice of model
box = form.selectbox('Select Pre-trained Model:', st.session_state.options, key=1)


# tweet refers to the text box for users to input their tweets.
# Has a default value of "\"We've seen in the last few months, unprecedented amounts of Voter Fraud.\" @SenTedCruz True!"
#     (Tweeted by former president Donald Trump)

tweet = form.text_input(label='Enter text to analyze:', value="\"We've seen in the last few months, unprecedented amounts of Voter Fraud.\" @SenTedCruz True!")

# Submit button
submit = form.form_submit_button(label='Submit')

# Read in some test data for prepopulation
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv("test.csv")

# Initializes logs if not already initialized
if not st.session_state.filled:
    # Iterates through all the options, initializing the logs for each.
    for s in st.session_state.options:
        st.session_state.logs[s] = []

# Pre-populates logs if not already pre-populated
if not st.session_state.filled:
    
    # Esnure pre-population happen again
    st.session_state.filled = True
    
    # Initialize 10 entries
    for x in range(10):
        
        # Helps me see which entry is being evaluated on the backend
        print(x)
        
        # Shorten tweets, as some models may not handle longer ones
        text = st.session_state.df["comment_text"].iloc[x][:128]
        
        # Iterate thru the models
        for s in st.session_state.options:
            
            # Reset everything
            
            # pline is the pipeline, which is used to load in the proper HuggingFace model for analysis
            pline = None
            
            # predictions refer to the predictions made by each model
            predictions = None
            
            # encoding is used by the finetuned model as input
            encoding = None
            
            # logits and probs are used to transform the results from predictions into usable/outputable data
            logits = None
            probs = None
            
            # Perform different actions based on the model selected by the user
            if s == 'bertweet-base-sentiment-analysis':
                pline = pipeline(task="sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
            elif s == 'twitter-roberta-base-sentiment':
                pline = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            elif s == 'distilbert-base-uncased-finetuned-sst-2-english':
                pline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            else:
                # encode data
                encoding = st.session_state.tokenizer(text, return_tensors="pt")
                encoding = {k: v.to(st.session_state.model.device) for k, v in encoding.items()}
                
                # feed data into model and store the predictions
                predictions = st.session_state.model(**encoding)
                
                # modify the data to get probabilities for each toxicity (scale of 0 - 1)
                logits = predictions.logits
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze().cpu())
                
                # Reform the predictions to note where probabilities are actually high
                predictions = np.zeros(probs.shape)
                predictions[np.where(probs >= 0.5)] = 1
            
            # Prepare the log entry
            log = []
            
            # If there was a pipeline, then we used a pretrained model.
            if pline:
                # Get the prediction
                predictions = pline(text)
                
                # Initialize the log to the proper shape
                log = [0] * 4
                
                # Record the text
                log[1] = text
                
                # predictions ends up being length 1, so this only happens for the prediction with the highest probability (the returned value)
                for p in predictions:
                    
                    # Different models have different outputs, so we standardize them in the logs
                    # Note, some unecessary repetions may occur here
                    if s == 'bertweet-base-sentiment-analysis':
                        if p['label'] == "POS":
                            log[0] = 0
                            log[2] = "POS"
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
            
            # Otherwise, we are using the finetuned model
            else:
                
                #Initialize log to the proper shape and store the text
                log = [0] * 6
                log[1] = text
                
                # Determine whether or not there was toxicity
                if max(predictions) == 0:
                    # No toxicity, input log values as such
                    log[0] = 0
                    log[2] = ("NO TOXICITY")
                    log[3] = (f"{100 - round(probs[0].item() * 100, 1)}%")
                    log[4] = ("N/A")
                    log[5] = ("N/A")
                
                # There was toxicity
                else:
                    # Record the toxicity
                    log[0] = 1
                    
                    # Find the maximum overall toxic category and the maximum toxic category of each type
                    _max = 0
                    _max2 = 2
                    for i in range(1, len(predictions)):
                        if probs[i].item() > probs[_max].item():
                            _max = i
                        if i > 2 and probs[i].item() > probs[_max2].item():
                            _max2 = i
                            
                    # Input data into log
                    log[2] = (st.session_state.labels[_max])
                    log[3] = (f"{round(probs[_max].item() * 100, 1)}%")
                    log[4] = (st.session_state.labels[_max2])
                    log[5] = (f"{round(probs[_max2].item() * 100, 1)}%")
            # Add the log to the proper model's logs
            st.session_state.logs[s].append(log)

# Check if there was a submitted input
if submit and tweet:
    
    # Small loading message :)
    with st.spinner('Analyzing...'):
        time.sleep(1)

    # Double check that there was an input
    if tweet is not None:
        
        # Reset variable
        pline = None
        
        # Set up shape for output
        # Pretrained models should have 3 columns, while the finetuned model should have 5
        if box != 'Modified Bert Toxicity Classification':
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
        
        # Perform different actions based on the model selected by the user
        if box == 'bertweet-base-sentiment-analysis':
            pline = pipeline(task="sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        elif box == 'twitter-roberta-base-sentiment':
            pline = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        elif box == 'distilbert-base-uncased-finetuned-sst-2-english':
            pline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            
            # encode data
            encoding = st.session_state.tokenizer(tweet, return_tensors="pt")
            encoding = {k: v.to(st.session_state.model.device) for k,v in encoding.items()}
            
            # feed data into model and store the predictions
            predictions = st.session_state.model(**encoding)
            
            # modify the data to get probabilities for each toxicity (scale of 0 - 1)
            logits = predictions.logits
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze().cpu())
            
            # Reform the predictions to note where probabilities are actually high
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= 0.5)] = 1

        # Title columns differently for different models
        # The existence of pline implies that a pretrained model was used
        if pline:
            
            # Predict the tweet here
            predictions = pline(tweet)
            
            # Title the column
            col2.header("Judgement")
        else:
            # Titling columns
            col2.header("Category")
            col4.header("Type")
            col5.header("Score")

        # Title more columns
        col1.header("Tweet")
        col3.header("Score")

        # If we used a pretrained model, process the prediction below
        if pline:
            # Set log to correct shape
            log = [0] * 4
            
            # Store the tweet
            log[1] = tweet
            
            # predictions ends up being length 1, so this only happens for the prediction with the highest probability (the returned value)
            for p in predictions:
                
                # Different models have different outputs, so we standardize them in the logs
                # Note, some unecessary repetions may occur here
                if box == 'bertweet-base-sentiment-analysis':
                    if p['label'] == "POS":
                        
                        # Only print the first 20 characters of the first line, so that the table lines up
                        # Also store the proper values into log while printing the outcome of this tweet
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
                
                # Print out the past inputs in reverse order
                for a in st.session_state.logs[box][::-1]:
                    if a[0] == 0:
                        # Again, only limit the tweet printed to 20 characters to have everything line up
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
                # Add the log to the logs
                st.session_state.logs[box].append(log)
        
        # We used the finetuned model, so proceed below
        else:
            
            # Initialize log to the proper shape and store the tweet
            log = [0] * 6
            log[1] = tweet
            
            # Check if nontoxic
            if max(predictions) == 0:
                
                # Only display the first 10 characters, as more columns means less characters can fit (make everything line up)
                # Display and input the data as we go
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
                
                # Look for the maximum toxicity category and the highest toxicity type
                _max = 0
                _max2 = 2
                for i in range(1, len(predictions)):
                    if probs[i].item() > probs[_max].item():
                        _max = i
                    if i > 2 and probs[i].item() > probs[_max2].item():
                        _max2 = i
                        
                # Display and input the data as we go
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
                
            # Print out the past logs in reverse order
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
            
            # Add result to logs
            st.session_state.logs[box].append(log)