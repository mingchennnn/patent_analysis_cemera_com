import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import time

# Choose the short or the full CSV file
# short = '_short'
short = ''

# Load the CSV file
df = pd.read_csv(f'./output/concatenated_patent_documents{short}.csv')
print("CSV file loaded.")

# Lowercase the 'combined' column
df['combined'] = df['combined'].str.lower()
print("Text converted to lowercase.")

# Define a tokenizer function using the specified regex
def tokenize(text):
    pattern = re.compile(r'[a-z0-9][a-z0-9-]*[a-z0-9]+|[a-z0-9]')
    return pattern.findall(text)

# Apply the tokenizer
df['tokens'] = df['combined'].apply(tokenize)
print("Tokenization completed.")

# Remove stopwords, one-character words, and words that contain any numerical digits
stop_words = set(stopwords.words('english'))
# Add some words manually into stop words
stop_words.update(['first', 'second', 'third', 'sb', 'thereof', 'include', 
                   'method', 'including', 'includes', 'solve', 'solved',
                   'jpo', 'inpit', 'ncip', 'copyright', 'problem', 'invention',
                   'innovation', 'provide', 'provides', 'provided', 'following',
                   'result', 'describe', 'wherein', 'left', 'right'])
stop_words.update(['january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'may', 'june',
                   'july', 'august', 'aug', 'september', 'sep', 'october', 'oct', 'november', 'nov',
                   'december', 'dec', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
stop_words.update(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 
                'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 
                'seventy', 'eighty', 'ninety', 'hundred'])

def contains_digit(word):
    return any(char.isdigit() for char in word)

df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words and len(word) > 1 and not contains_digit(word)])
print("Stopwords, one-character words, and words with numerical digits removed.")

# Save the first filtered patent tokens to CSV
df.to_csv(f'./output/first_filtered_patent_documents{short}.csv', index=False)
print("Processed patent tokens saved to CSV.")


# Download required NLTK data
# nltk.download('stopwords')

# Download required NLTK data
# nltk.download('averaged_perceptron_tagger')  # This is used for POS tagging

