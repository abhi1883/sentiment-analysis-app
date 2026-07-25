import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download the stopwords list (only needs to happen once)
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# ---- STEP 1: Remove HTML tags ----


def remove_html_tags(text):
    clean_text = re.sub(r"<.*?>", "", text)
    return clean_text


df["review"] = df["review"].apply(remove_html_tags)

# ---- STEP 2: Lowercase + remove punctuation/numbers ----


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


df["review"] = df["review"].apply(clean_text)

# ---- STEP 3: Remove stopwords ----
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


print("BEFORE stopword removal:")
print(df["review"][1][:200])

df["review"] = df["review"].apply(remove_stopwords)

print("\nAFTER stopword removal:")
print(df["review"][1][:200])

# ---- Save the fully cleaned dataset ----
df.to_csv("data/IMDB_Dataset_cleaned.csv", index=False)
print("\nCleaning complete! Saved as data/IMDB_Dataset_cleaned.csv")
