import pandas as pd

# Load the dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Show the first 5 rows
print(df.head())

# Show basic info: column names, data types, missing values
print(df.info())

# Show how many positive vs negative reviews exist
print(df["sentiment"].value_counts())
