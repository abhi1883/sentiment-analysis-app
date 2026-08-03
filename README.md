# 🎬 Movie Review Sentiment Analyzer

A machine learning web app that predicts whether a movie review is positive or negative, built using Natural Language Processing (NLP) techniques.

## 🌐 Live Demo
Try the app here: **[abhinanda-sentiment-analyzer.streamlit.app](https://abhinanda-sentiment-analyzer.streamlit.app)**

## 📊 Project Overview
This project analyzes movie reviews from the IMDB dataset (50,000 reviews) and predicts whether a review is positive or negative using Natural Language Processing (NLP) and a Naive Bayes machine learning model.

## 🔍 Key Steps
1. **Data Cleaning:** Removed HTML tags, punctuation, converted to lowercase, removed stopwords
2. **Text Vectorization:** Converted cleaned text into numeric features using CountVectorizer (top 5,000 words)
3. **Model Training:** Trained a Multinomial Naive Bayes classifier
4. **Result:** Achieved 84.8% accuracy on unseen test data
5. **Deployment:** Built an interactive Streamlit web app for live predictions

## 🛠️ Tech Stack
- **Python** — core language
- **Pandas** — data handling
- **NLTK** — stopword removal
- **Scikit-learn** — CountVectorizer, Naive Bayes model
- **Streamlit** — interactive web app

## 🚀 How to Run Locally
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python -m streamlit run app.py`

## 👤 Author
Abhinanda Udupa — 3rd Semester Engineering Student