import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the cleaned dataset
df = pd.read_csv("data/IMDB_Dataset_cleaned.csv")

# Convert text into numbers (same as before)
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["review"])

# Our target: the sentiment label (positive/negative)
y = df["sentiment"]

# Split into training (80%) and testing (20%) - just like your salary project
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model (a common, fast model for text classification)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test it
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}  ({accuracy*100:.1f}%)")

# Try it on a few sample reviews
print("\nSample predictions vs actual:")
comparison = pd.DataFrame(
    {"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
print(comparison)
