import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from collections import Counter
import numpy as np
import ast


# Define whether to use smoothing
smoothing = False

# Choose the short or the full CSV file
# short = '_short'
short = ''
# short = '_test'

# Load the filtered CSV file
df = pd.read_csv(f'./output/second_filtered_patent_documents{short}.csv')
# Convert the 'tokens' column back to a list
df['tokens'] = df['tokens'].apply(ast.literal_eval)
print("CSV loaded and strings converted back to lists of tokens")

# Function to calculate TF
def calculate_tf(document):
    tf_dict = {}
    document = list(document)
    total_tokens = len(document)
    token_counts = Counter(document)
    for token, count in token_counts.items():
        tf_dict[token] = count / total_tokens
    # sort the dict in the descending order of values
    tf_dict = dict(sorted(tf_dict.items(), key=lambda item: item[1], reverse=True))
    return tf_dict

# Calculate TF for each document
df['tf'] = df['tokens'].apply(calculate_tf)
print("Term Frequency (TF) calculated for each document.")

# Calculate document frequency for each token
all_tokens = set([token for tokens in df['tokens'] for token in tokens])

# Convert each document's tokens to a set for efficient processing
token_sets = df['tokens'].apply(set)

# Initialize an empty dictionary for document frequencies
doc_freq = {token: 0 for token in all_tokens}

# Efficient loop to calculate document frequency
for i, doc_set in enumerate(token_sets):
    # Update document frequency for each token in the document
    for token in doc_set:
        doc_freq[token] += 1

    # Print progress every 100 documents (or adjust this number as needed)
    if (i+1) % 10000 == 0:
        print(f"Processed {i+1} out of {len(token_sets)} documents.")

print("Document Frequency calculation completed.")

# Number of documents
N_doc = len(df)

# Function to calculate IDF
def calculate_idf(N_doc, document_frequency, smoothing):
    if smoothing == False:
        return np.log(N_doc/document_frequency)
    else:
        return np.log(N_doc/(document_frequency+1))+1

# Get IDF for each document from the dictionary idf
def get_idf(document, idf):
    return {token: idf[token] for token in document}

# Calculate IDF for each token
idf = {token: calculate_idf(N_doc, document_frequency, smoothing) for token, document_frequency in doc_freq.items()}
df['idf'] = df['tf'].apply(lambda x: get_idf(x, idf))
print("Inverse Document Frequency (IDF) calculated for each token.")

# Function to calculate TF-IDF
def calculate_tfidf(tf_dict):
    tfidf_dict = {token: tf * idf[token] for token, tf in tf_dict.items()}
    # sort the dict in the descending order of values
    return dict(sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True))

# Calculate TF-IDF for each document
df['tfidf'] = df['tf'].apply(calculate_tfidf)
print("TF-IDF calculated for each document.")

# Save the tfidf data dictionary to CSV
if smoothing:
    df.to_csv(f'./output/tfidf_smooth_patent_documents{short}.csv', index=False)
else:
    df.to_csv(f'./output/tfidf_patent_documents{short}.csv', index=False)

print("TF-IDF of tokens saved to CSV.")

# Manually remove any further unwanted words at this stage
unwanted_words = ['image_image', 'average', 'proc_de', 'said_image', 'fig_shown', 'use_image', 
                  'sp_sp', 'fig', 'figs', 'figure', 'figures', 'inc', 'co', 'et', 'use', 'al']
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in unwanted_words])
print('Unwanted words filtered out')

# Function to get top N unique tokens based on TF-IDF values
def get_top_n_unique_tokens(tokens, tfidf_values, n=20):
    # Create a list of tuples (token, tfidf_value) and sort by tfidf_value in descending order
    sorted_tokens = sorted([(token, tfidf_values.get(token, 0)) for token in tokens], key=lambda x: x[1], reverse=True)
    
    # Collect top N unique tokens
    unique_tokens = []
    for token, _ in sorted_tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
        if len(unique_tokens) == n:
            break

    return unique_tokens

# Apply the filtering function to each document's tokens
df['tokens'] = df.apply(lambda row: get_top_n_unique_tokens(row['tokens'], row['tfidf']), axis=1)
print("Tokens with TF-IDF value lower than the threshold filtered out.")


# Remove the specified columns
df.drop(['tf', 'idf', 'tfidf'], axis=1, inplace=True)
# Save the modified DataFrame to CSV
df.to_csv(f'./output/third_filtered_patent_documents{short}.csv', index=False)
print("Thirdly processed patent tokens saved to CSV")