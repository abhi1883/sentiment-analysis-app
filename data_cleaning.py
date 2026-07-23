import pandas as pd
import re

# Load the dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Function to remove HTML tags like <br /><br />


def remove_html_tags(text):
    clean_text = re.sub(r"<.*?>", "", text)
    return clean_text


# Test it on the first review to see if it worked
print("BEFORE cleaning:")
print(df["review"][1][:200])  # show first 200 characters of review #1

df["review"] = df["review"].apply(remove_html_tags)

print("\nAFTER cleaning:")
# Function to clean text: lowercase + remove punctuation


def clean_text(text):
    text = text.lower()  # convert to lowercase
    # remove anything that's not a letter or space
    text = re.sub(r"[^a-z\s]", "", text)
    return text


print("\nBEFORE lowercase/punctuation cleaning:")
print(df["review"][1][:200])

df["review"] = df["review"].apply(clean_text)

print("\nAFTER lowercase/punctuation cleaning:")
print(df["review"][1][:200])
