import nltk
from rake_nltk import Rake
# from gensim.summarization import keywords
import pandas as pd 
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

# Load the DataFrame
df = pd.read_csv("Book1.csv")
df.info()

# Set display options
pd.set_option('display.max_colwidth', 500)

# Initialize RAKE
r = Rake()

# Preprocessing function
def preprocess_text(text):
    # Remove links (non-regex approach)
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    # Convert text to lowercase
    text = text.lower()
    # # Remove punctuation (non-regex approach)
    # text = ''.join(char for char in text if char not in string.punctuation)
    # Remove stopwords
    stop_list = stopwords.words('english')
    # stop_list.extend(["-", "extra", "certainly", "also", "always"]) # Extend stopwords
    words = text.split()
    clean_words = [word for word in words if word not in stop_list]
    clean_text = ' '.join(clean_words)
    return clean_text

# Function to preprocess each column
def preprocess_column(column):
    return column.apply(preprocess_text)

# List of columns to preprocess
columns_to_preprocess = ['abstract'] # , 'summary_5_sentences', 'summary_ChatGPT_5_sentences', 'summary_15_sentences', 'summary_ChatGPT_15_sentences']  # Add more columns as needed

# Apply preprocessing to each column
for column_name in columns_to_preprocess:
    df[column_name + '_clean'] = preprocess_column(df[column_name])

# # Display the DataFrame
# for column_name in columns_to_preprocess:
#     print(f"Preprocessed content of column '{column_name}':")
#     print(df[column_name + '_clean'])
#     print()

# Extract key phrases using RAKE for multiple columns
key_phrases_dict = {}  # Dictionary to store key phrases for each column

for column_name in columns_to_preprocess:
    key_phrases = []  # List to store key phrases for the current column
    for text in df[column_name + '_clean']:
        r.extract_keywords_from_text(text)
        phrases = r.get_ranked_phrases_with_scores()
        key_phrases.append(phrases[:10]) 
    key_phrases_dict[column_name] = key_phrases  # Store key phrases for the current column

# Add key phrases with scores to DataFrame for each column
for column_name, key_phrases in key_phrases_dict.items():
    df[column_name + "_key_phrases_with_scores"] = key_phrases

# Print out key phrases for each column
for column_name, key_phrases in key_phrases_dict.items():
    print(f"Key Phrases with Scores for Column '{column_name}':")
    for index, phrases in enumerate(key_phrases):
        print(f"Row {index}:")
        # Print top 10 key phrases with scores for the current row
        for rank, (score, phrase) in enumerate(phrases[:10], start=1):
            print(f"Phrase: {phrase}, Score: {score}, Rank: {rank}")
    print()




