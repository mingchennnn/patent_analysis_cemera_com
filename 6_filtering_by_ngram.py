import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import time
import ast
import spacy

# Choose the short or the full CSV file
# short = '_short'
short = ''

# Load NLP component
nlp = spacy.load("en_core_web_sm")

# Load the filtered CSV file
df = pd.read_csv(f'./output/first_filtered_patent_documents{short}.csv')
# Convert the 'tokens' column back to a list
df['tokens'] = df['tokens'].apply(ast.literal_eval)
print("CSV loaded and strings converted back to lists of tokens")


# Function to aggregate n-grams for a single document
def aggregate_document_ngrams(tokens):
    bigrams = Counter(ngrams(tokens, 2))
    return bigrams

# Aggregate n-grams for all documents
aggregate_bigrams = Counter()

for tokens in df['tokens']:
    doc_bigrams = aggregate_document_ngrams(tokens)
    aggregate_bigrams.update(doc_bigrams)
print("Ngrams aggregated")

# Apply frequency threshold (300 for big one, 10 for small one)
if short == '_short':
    truncated_bigrams = {k: v for k, v in aggregate_bigrams.items() if v >= 5}
else:
    truncated_bigrams = {k: v for k, v in aggregate_bigrams.items() if v >= 100}

# Convert filtered results to DataFrame
bigram_list = list(truncated_bigrams.items())

bigram_df = pd.DataFrame(bigram_list, columns=['bigram', 'frequency'])
print("Ngrams truncated and dataframed")

# Filter bigrams using noun chunck
def extract_noun_chunks(df, bigram_col, frequency_col):
    """
    Extract and filter noun chunks from a DataFrame containing bigrams.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the bigrams and their frequencies.
    bigram_col (str): The name of the column containing the bigrams.
    frequency_col (str): The name of the column containing the frequencies.

    Returns:
    pd.DataFrame: A new DataFrame containing only the noun chunks and their frequencies.
    """


    # Initialize an empty list to store valid noun chunk bigrams
    valid_noun_chunks = []
    # Iterate over the DataFrame
    for _, row in df.iterrows():
        bigram = row[bigram_col]
        bigram = ' '.join(bigram)    # convert the turple to a string
        frequency = row[frequency_col]
        # Create a SpaCy Doc object
        doc = nlp(bigram)

        # Check if the bigram matches any noun chunk in the Doc
        for chunk in doc.noun_chunks:
            if chunk.text == bigram:
                bigram = tuple(bigram.split(' '))    # convert it back
                valid_noun_chunks.append((bigram, frequency))
                break

    # Create a new DataFrame with the valid noun chunks
    noun_chunk_df = pd.DataFrame(valid_noun_chunks, columns=[bigram_col, frequency_col])

    return noun_chunk_df

filtered_bigram_df = extract_noun_chunks(bigram_df, 'bigram', 'frequency')

# Save to CSV
filtered_bigram_df.to_csv(f'./output/bigram_frequencies{short}.csv', index=False)
print("Filtered Ngram saved to CSV.")


# Convert bigram list to a set of tuples for easy matching
filtered_bigram_set = set([bigram for bigram in filtered_bigram_df['bigram']])
print("Bigrams converted to sets")

# Function to concatenate matching bigrams in a list of tokens
def concatenate_matching_bigrams(tokens):
    # Initialize an empty list to store the modified tokens
    modified_tokens = []
    skip_next = False

    # Iterate over tokens
    for i in range(len(tokens)):
        if skip_next:
            # Skip this iteration if the previous token was part of a concatenated bigram
            skip_next = False
            continue

        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in filtered_bigram_set:
            # Concatenate the current and next token if they form a bigram in the list
            modified_tokens.append(tokens[i] + '_' + tokens[i+1])
            skip_next = True
        else:
            # Otherwise, keep the token as is
            modified_tokens.append(tokens[i])

    return modified_tokens

# Apply the concatenation function to each document's tokens
df['tokens'] = df['tokens'].apply(concatenate_matching_bigrams)
print("Tokens concatenated as bigrams")


# Function to check if a token is a noun
def is_noun(pos):
    return pos in ['NN', 'NNS', 'NNP', 'NNPS']

# Function to filter tokens that are not bigrams or nouns
def filter_tokens(tokens):
    # Apply POS tagging to tokens
    pos_tags = nltk.pos_tag(tokens)

    # Initialize an empty list for the filtered tokens
    filtered_tokens = []

    # Iterate over each token and its corresponding POS tag
    for token, tag in zip(tokens, pos_tags):
        # Check if the token is a noun or a bigram (contains '_')
        if is_noun(tag[1]) or '_' in token:
            # If the condition is met, add the token to the filtered list
            filtered_tokens.append(token)

    return filtered_tokens

# Initialize a counter and determine the total number of items
counter = 0
total = len(df)

# Apply the filtering function to each document's tokens
for index, row in df.iterrows():
    df.at[index, 'tokens'] = filter_tokens(row['tokens'])
    counter += 1

    # Print progress every 1000 documents or adjust the number as needed
    if counter % 10000 == 0:
        print(f"Processed {counter} out of {total} documents.")

print('Tokens other than nouns and bigrams filtered out')


# Save the filtered tokens to CSV
df.to_csv(f'./output/second_filtered_patent_documents{short}.csv', index=False)
print("Secondly processed patent tokens saved to CSV.")
