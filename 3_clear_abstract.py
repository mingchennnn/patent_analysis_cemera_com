import pandas as pd
import re
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the "company_names" folder within the same directory
company_names_dir = os.path.join(script_dir, "company_names")

# List everything in the directory
all_items = os.listdir(company_names_dir)

# Filter out only the folders
companies = [item for item in all_items if os.path.isdir(os.path.join(company_names_dir, item))]

for company in companies:
    # Load the CSV file
    df = pd.read_csv(f'./company_names/{company}/{company}_patent_with_abstracts.csv')

    # Print the number of rows before filtering
    print(f"Number of rows before filtering for {company}: {df.shape[0]}")

    # Define the keywords to search for
    keywords_include = ["camera", "photo ", "photograph", "film", "optic", "aperture", "viewfinder",
                        "image sens", "imaging device", "lens", "ccd", "cmos sensor", "focusing ring",
                        "shutter", "exposure control", "autofocus", "automatic focus", "focus detect",
                        "imaging sens", "focus adjust", "imaging apparatus",
                        "image apparatus", "focus control", "automatic exposure"]
    
    if company != "intel":
        # Define keywords to exclude if found in 'abstract' or 'title'
        keywords_exclude = ["fibre", "fiber", "optical circuit", "electronic package",
                            "lithograph", "thermo-optic", "broadband"]
    else:
        # Define keywords to exclude if found in 'abstract' or 'title'
        keywords_exclude = ["fibre", "fiber", "optical circuit", "electronic package",
                            "lithograph", "thermo-optic", "broadband",
                            "semiconductor", "circuit", "waveguide"]  # these exlusion words are for intel only

    # Regex pattern to match "image (one word) apparatus"
    pattern = r"image?(ing)?-? ?\b\w+\b (apparatus|device)"


    # Function to check if any inclusion keyword is in the text
    def contains_keyword_include(text):
        text = text.lower()
        return any(keyword in text for keyword in keywords_include) or re.search(pattern, text) is not None

    # Function to check if any exclusion keyword is in 'abstract'
    def contains_keyword_exclude(text):
        text = text.lower()
        return any(keyword in text for keyword in keywords_exclude)

    # Filter the DataFrame
    # 1. Drop rows where 'title' or 'abstract' is NaN
    df = df.dropna(subset=['title', 'abstract'])

    # 2. Keep rows where either 'title' or 'abstract' contains any of the inclusion keywords
    # 3. Exclude rows where 'title' or 'abstract' contains any of the exclusion keywords
    filtered_df = df[df['title'].apply(contains_keyword_include) | df['abstract'].apply(contains_keyword_include)]
    filtered_df = filtered_df[~filtered_df['abstract'].apply(contains_keyword_exclude)]
    filtered_df = filtered_df[~filtered_df['title'].apply(contains_keyword_exclude)]

    # Print the number of rows after filtering
    print(f"Number of rows after filtering for {company}: {filtered_df.shape[0]}")

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(f'./company_names/{company}/{company}_patent_with_abstracts_filtered.csv', index=False)

    # Get the filtered out rows (rows that do not meet the keyword criteria or contain exclusion keywords)
    filtered_out_df = df[~(df['title'].apply(contains_keyword_include) | df['abstract'].apply(contains_keyword_include)) | 
                          df['title'].apply(contains_keyword_exclude) | df['abstract'].apply(contains_keyword_exclude)]

    # Save the filtered out data to a new CSV file
    filtered_out_df.to_csv(f'./company_names/{company}/{company}_patent_with_abstracts_not_filtered.csv', index=False)
    print(company)
