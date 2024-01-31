from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np

def vectorize_tfidf(tfidf_dicts):
    vectorizer = DictVectorizer(sparse=False)
    vectors = vectorizer.fit_transform(tfidf_dicts)
    feature_names = vectorizer.get_feature_names_out()
    return vectors, feature_names

end_year = 2022
df_moving_accumulated = pd.read_pickle(f"./output/moving_accumulate_company_tfidf_1970{end_year}.pkl")

# Store the cosine similarity matrices for each year
cosine_similarity_matrices = {}

for year in range(1970, end_year+1):
    year_df = df_moving_accumulated[df_moving_accumulated['year'] == year]
    # Extract the list of TF-IDF dictionaries for this year
    tfidf_dicts = year_df['tf_idf'].tolist()

    # Vectorize the TF-IDF dictionaries
    vectors, feature_names = vectorize_tfidf(tfidf_dicts)
    print(len(vectors[0]))

    # Calculate cosine similarity matrix
    cosine_sim_matrix = np.zeros((len(vectors), len(vectors)))
    
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            # If both vectors have length
            if sum(vectors[i]) != 0 and sum(vectors[j]) != 0:
                cosine_sim_matrix[i][j] = 1 - cosine(vectors[i], vectors[j])
            # If both vectors have zero length, two vectors are identical
            elif sum(vectors[i]) == 0 and sum(vectors[j]) == 0:
                cosine_sim_matrix[i][j] = 1
            # If one vector has zero length, two vectors are orthogonal
            else:
                cosine_sim_matrix[i][j] = 0
    
    # Convert the cosine similarity matrix to a DataFrame
    cos_sim_df = pd.DataFrame(cosine_sim_matrix, index=year_df['company'], columns=year_df['company'])

    # Save the cosine similarity matrix for this year as a CSV file
    cos_sim_df.to_csv(f"./output/cosine_similarity/cosine_similarity_matrix_{year}.csv")
    print(year)