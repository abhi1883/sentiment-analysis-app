import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the CLEANED dataset (from our previous step)
df = pd.read_csv("data/IMDB_Dataset_cleaned.csv")

# Create the vectorizer - only keep the top 5000 most common words
vectorizer = CountVectorizer(max_features=5000)

# Build the word-count table from all reviews
X = vectorizer.fit_transform(df["review"])

# Let's peek at what happened
print("Shape of our table (rows, columns):", X.shape)
print("\nFirst 20 words it picked as columns:")
print(vectorizer.get_feature_names_out()[:20])
