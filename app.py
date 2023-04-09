import streamlit as st
import time


st.title("Sentiment Analysis App")

form = st.form(key='Sentiment Analysis')
tweet = form.text_input(label='Enter text to analyze:', value="\"We've seen in the last few months, unprecedented amounts of Voter Fraud.\" @SenTedCruz True!")
submit = form.form_submit_button(label='Submit')

if submit:
    with st.spinner('Analyzing ur mom...'):
        time.sleep(5)
    st.header(tweet)