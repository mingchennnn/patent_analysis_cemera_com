import pandas as pd
import numpy as np
from collections import Counter
import ast

# Define whether to use smoothing
smoothing = False

# Load DataFrame efficiently from pkl form
df_moving_accumulated = pd.read_pickle("./output/moving_accumulate_company_tokens.pkl")

# Function to calculate TF for a document
def calculate_tf(document):
    tf_dict = {}
    total_tokens = len(document)
    token_counts = Counter(document)
    for token, count in token_counts.items():
        tf_dict[token] = count / total_tokens
    return tf_dict

# Function to calculate IDF for a token
def calculate_idf(token, year_docs, num_docs_with_token, smoothing):
    if smoothing:
        return np.log(len(year_docs) / (1 + num_docs_with_token[token])) + 1
    else:
        return np.log(len(year_docs) / (num_docs_with_token[token]))

# Initialize a column for TF-IDF in df_moving_accumulated
df_moving_accumulated['tf_idf'] = None

# Main calculation loop
for year in range(1970, 2023):
    year_df = df_moving_accumulated[df_moving_accumulated['year'] == year]
    # Flatten all tokens for the year and count the number of documents containing each token
    all_tokens = [token for doc in year_df['moving_acc_tokens'] for token in doc]
    num_docs_with_token = Counter({token: sum(token in doc for doc in year_df['moving_acc_tokens']) for token in set(all_tokens)})
    
    # Calculate TF-IDF for each document
    for index, row in year_df.iterrows():
        tf = calculate_tf(row['moving_acc_tokens'])
        idf = {token: calculate_idf(token, year_df['moving_acc_tokens'], num_docs_with_token, smoothing) for token in tf}
        tf_idf = {token: tf_val * idf[token] for token, tf_val in tf.items()}
        df_moving_accumulated.at[index, 'tf_idf'] = tf_idf
    print(year)


# Remove the 'moving_acc_tokens' column
df_moving_accumulated.drop(columns=['moving_acc_tokens'], inplace=True)

# Save the modified DataFrame
df_moving_accumulated.to_pickle("./output/moving_accumulate_company_tfidf_19702022.pkl")
print("Data of tf-idf saved")
