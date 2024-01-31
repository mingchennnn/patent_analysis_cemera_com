import pandas as pd
import os

# Base directory
base_dir = './company_names'

# Dynamically generate list of company names based on the folder names under base_dir
company_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Generate file paths
file_paths = [os.path.join(base_dir, company_name, f"{company_name}_patent_with_abstracts_filtered.csv") for company_name in company_names]

# Concatenate all CSV files into a single DataFrame
all_data = pd.concat((pd.read_csv(file) for file in file_paths if os.path.exists(file)))

# Combine "title" and "abstract" columns with a space in between and create a new column "combined"
all_data['combined'] = all_data['title'] + ' ' + all_data['abstract']

# Keep only "id" and "combined" columns
all_data = all_data[['id', 'combined']]

# Remove duplicates based on the "id" column
all_data.drop_duplicates(subset='id', inplace=True)

# Save the final DataFrame to a CSV file
all_data.to_csv('./output/concatenated_patent_documents.csv', index=False)
