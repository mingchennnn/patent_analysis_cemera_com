import pandas as pd
import os
import ast

# Choose the short or the full CSV file
# short = '_short'
short = ''

# Base directory
base_dir = './company_names'

# Dynamically generate list of company names based on the folder names under base_dir
company_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Generate file paths
file_paths = [os.path.join(base_dir, company_name, f"{company_name}_patent_with_abstracts_filtered.csv") for company_name in company_names]

# Load the filtered CSV file
df_token = pd.read_csv(f'./output/third_filtered_patent_documents{short}.csv')
# Convert the 'tokens' column back to a list
df_token['tokens'] = df_token['tokens'].apply(ast.literal_eval)

# Initialize an empty DataFrame for concatenating all df_merged
df_com_token = pd.DataFrame()

for file, company_name in zip(file_paths, company_names):
    # Read each company's dataframe
    df_com_id = pd.read_csv(file)

    # Merge the dataframes on the 'id' column
    df_merged = pd.merge(df_com_id, df_token[['id', 'tokens']], on='id', how='left')

    # Add 'company' column
    df_merged['company'] = company_name

    # Create 'year' column from 'publication date'
    df_merged['year'] = pd.to_datetime(df_merged['publication date']).dt.year

    # Remove columns other than 'company', 'id' and 'tokens'
    df_merged = df_merged[['company', 'id', 'year', 'tokens']]

    # Concatenate to the total dataframe
    df_com_token = pd.concat([df_com_token, df_merged], ignore_index=True)
    print(company_name)

# Save the concatenated DataFrame to a CSV file
df_com_token.to_csv("./output/mapped_company_id_year_tokens.csv", index=False)
print("mapped_company_id_year_tokens.csv created")

# Group by 'company' and 'year', and merge tokens
df_com_token_grouped = df_com_token.groupby(['company', 'year'])['tokens'].sum().reset_index()

# Save the grouped DataFrame to a CSV file
df_com_token_grouped.to_csv("./output/mapped_company_year_tokens.csv", index=False)
print("mapped_company_year_tokens.csv created")