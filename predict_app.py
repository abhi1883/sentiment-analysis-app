import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the cleaned dataset
df = pd.read_csv("data/IMDB_Dataset_cleaned.csv")

# Convert text to numbers
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Train the model on ALL the data this time (not just 80%, since we're done testing)
model = MultinomialNB()
model.fit(X, y)

# Function to clean a NEW review the same way we cleaned the training data


def clean_new_review(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


# Let the user type a review and get a live prediction
print("Sentiment Analysis - Type a movie review (or 'quit' to stop)")
while True:
    user_review = input("\nEnter a review: ")
    if user_review.lower() == "quit":
        break
    cleaned = clean_new_review(user_review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    print(f"Predicted sentiment: {prediction[0]}")
