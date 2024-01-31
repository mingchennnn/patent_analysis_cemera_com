import pandas as pd

# Company name
company = "asus"
# Load the Excel file (skip the first url line)
df = pd.read_excel(f'./company_names/{company}/{company}_keywords_google_patent.xlsx', skiprows=1)

# Define the keywords to search for
keywords_include = ["asus", "華碩", "华硕"]

# Define keywords to exclude if found in 'assignee'
keywords_exclude = []

# Filter rows where the 'assignee' column contains any of the inclusion keywords
# and does not contain any of the exclusion keywords
filtered_df = df[df['assignee'].apply(lambda x: 
                                      any(keyword.lower() in str(x).lower() for keyword in keywords_include) and
                                      all(exclude_keyword.lower() not in str(x).lower() for exclude_keyword in keywords_exclude))]

# Save the filtered data to a new Excel file
filtered_df.to_excel(f'./company_names/{company}/{company}_keywords_google_patent_selected_assignee.xlsx', index=False)

# Output the keywords to a text file with UTF-8 encoding
with open(f'./company_names/{company}/keyword_assignee.txt', 'w', encoding='utf-8') as file:
    for keyword in keywords_include:
        file.write(keyword + '\n')
