import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page setup
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

# Load and train (cached so it doesn't reload every time you type)


@st.cache_resource
def load_model():
    df = pd.read_csv("data/IMDB_Dataset_cleaned.csv")
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model


vectorizer, model = load_model()

# Function to clean new text the same way


def clean_new_review(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


# Title
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a movie review below and see if it's predicted as positive or negative.")

# Text input box
user_review = st.text_area("Enter your review:", height=150)

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_new_review(user_review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)

        if prediction[0] == "positive":
            st.success(f"Predicted Sentiment: **Positive** 😊")
        else:
            st.error(f"Predicted Sentiment: **Negative** 😞")

        confidence = max(probability[0]) * 100
        st.write(f"Model confidence: {confidence:.1f}%")
