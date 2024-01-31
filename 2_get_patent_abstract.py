import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_abstract(url):
    # Ensure the URL ends with 'en'
    if not url.endswith('en'):
        url = url[:-2] + 'en'

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        abstract_tag = soup.find('meta', {'name': 'DC.description'})
        return abstract_tag['content'] if abstract_tag else "Abstract not found"
    except requests.RequestException:
        return "Failed to retrieve abstract"

def process_patents(file_path):
    patents = []
    df = pd.read_excel(file_path)

    for count, row in df.iterrows():
        abstract = scrape_abstract(row['result link'])
        patents.append({**row.to_dict(), 'abstract': abstract})

        # Print progress every 10 patents
        if (count + 1) % 10 == 0:
            print(f"Processed {count + 1} patents")

    return patents

def write_to_csv(patents, output_file):
    df = pd.DataFrame(patents)
    df.to_csv(output_file, index=False, encoding='utf-8')

companies = ['dell', 'microsoft', 'fujitsu', 'ibm', 'google', 'lenovo', 'acer', 'asus', 'hasselblad']

for company in companies:
    input_file = f'./company_names/{company}/{company}_keywords_google_patent_selected_assignee.xlsx'
    output_file = f'./company_names/{company}/{company}_patent_with_abstracts.csv'

    patents = process_patents(input_file)
    write_to_csv(patents, output_file)
    print(company)
