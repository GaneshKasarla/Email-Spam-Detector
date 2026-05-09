import pickle
import streamlit as st
import pandas as pd
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


# =========================
# CLEAN TEXT
# =========================

def clean_text(text):

    text = text.lower()

    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text
# =========================
# STREAMLIT UI
# =========================

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩"
)
# side bar
 
st.sidebar.title("About")
st.sidebar.write(
    "This project detects whether a message is Spam or Ham using Machine Learning and NLP."
)

# Simple title
st.title("📩 Spam Message Detector")

# Subtitle
st.write("Check whether a message is Spam or Ham using Machine Learning.")

# Background color
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    textarea {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input box
input_sms = st.text_area(
    "Enter Your Message"
)

# Predict button
if st.button("Predict"):
  with st.spinner("Analyzing message..."):

    transformed_sms = clean_text(input_sms)

    vector_input = vectorizer.transform([transformed_sms])

    result = model.predict(vector_input)[0]
    

    if result == 1:
        st.error("🚨 Spam Message Detected")
    else:
        st.success("✅ Not a Spam Message")