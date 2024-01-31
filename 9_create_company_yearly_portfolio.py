import pandas as pd
import os
import ast

# Base directory for company names
base_dir = './company_names'
company_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Load the grouped company-year tokens data and convert strings back to tokens
df_com_yearly_token = pd.read_csv("./output/mapped_company_year_tokens.csv")
df_com_yearly_token['tokens'] = df_com_yearly_token['tokens'].apply(ast.literal_eval)
print('Data loaded and converted to tokens')


### Yearly tokens
# Initialize a list to store the new rows
new_rows = []

# Iterate over each company
for company in df_com_yearly_token['company'].unique():
    # Get existing years for the current company
    existing_years = df_com_yearly_token[df_com_yearly_token['company'] == company]['year'].unique()

    # Check for missing years and add new rows
    for year in range(1970, 2024):
        if year not in existing_years:
            new_rows.append({'company': company, 'year': year, 'tokens': []})
    print(company)

# Create a DataFrame from the new rows
df_missing_years = pd.DataFrame(new_rows)

# Concatenate with the original DataFrame
df_com_yearly_token = pd.concat([df_com_yearly_token, df_missing_years], ignore_index=True)

# Sort the DataFrame by company and year
df_com_yearly_token = df_com_yearly_token.sort_values(by=['company', 'year'])

df_com_yearly_token.to_csv("./output/yearly_incremental_company_tokens.csv", index=False)
df_com_yearly_token.to_pickle("./output/yearly_incremental_company_tokens.pkl")    # Save to pickle for easier loading


### Moving accumulated tokens
# Initialize the DataFrame for moving accumulated tokens
df_moving_accumulated = pd.DataFrame(columns=['company', 'year', 'moving_acc_tokens'])

# Initialize a list to store rows for the new DataFrame
accumulated_rows = []

# Iterate over each company and year range
for company in company_names:
    for year in range(1970, 2023):
        # Initialize an empty list for accumulated tokens
        moving_acc_tokens = []

        # Get tokens from the past 5 years, including the current year
        for prev_year in range(year - 4, year + 1):
            # Filter tokens for the specific company and year
            tokens = df_com_yearly_token[(df_com_yearly_token['company'] == company) & (df_com_yearly_token['year'] == prev_year)]['tokens'].values
            if tokens.size > 0:
                moving_acc_tokens.extend(tokens[0])

        # Append to the list
        accumulated_rows.append({'company': company, 'year': year, 'moving_acc_tokens': moving_acc_tokens})

    print(company)

# Create the DataFrame from the list
df_moving_accumulated = pd.DataFrame(accumulated_rows)

# Save the DataFrame to a CSV file
df_moving_accumulated.to_csv("./output/moving_accumulate_company_tokens.csv", index=False)
df_moving_accumulated.to_pickle("./output/moving_accumulate_company_tokens.pkl")
print("Moving accumulated company tokens saved.")


### Yearly accumulative tokens
# Initialize an empty DataFrame for accumulated tokens
df_accumulated = pd.DataFrame(columns=['company', 'year', 'accumulated_tokens'])

# Initialize an empty list to store DataFrame chunks
df_chunks = []

# Iterate over each company and year
for company in company_names:
    accumulated_tokens = []
    for year in range(1970, 2023):
        # Check if there are tokens for the current year
        current_year_tokens = df_com_yearly_token[(df_com_yearly_token['company'] == company) & (df_com_yearly_token['year'] == year)]['tokens'].values
        if current_year_tokens.size > 0:
            # Append current year tokens to accumulated tokens
            accumulated_tokens.extend(current_year_tokens[0])

        # Create a temporary DataFrame for the current row
        temp_df = pd.DataFrame({'company': [company], 'year': [year], 'accumulated_tokens': [accumulated_tokens.copy()]})

        # Add the temporary DataFrame to the list
        df_chunks.append(temp_df)

    print(company)

# Concatenate all DataFrame chunks into one DataFrame
df_accumulated = pd.concat(df_chunks, ignore_index=True)

# Save the DataFrame to a CSV file
df_accumulated.to_csv("./output/yearly_accumulate_company_portfolio.csv", index=False)
df_accumulated.to_pickle("./output/yearly_accumulate_company_portfolio.pkl")
print("yearly_accumulate_company_portfolio created")
