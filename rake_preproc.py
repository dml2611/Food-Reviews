import nltk
from rake_nltk import Rake
import pandas as pd 
from nltk.tokenize import sent_tokenize
import re
import string
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the DataFrame or your text corpus
df = pd.read_csv("Book1.csv")

# Preprocessing function
def preprocess_text(text):
    # Remove links (non-regex approach)
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    stop_list = stopwords.words('english')
    # stop_list.extend(["-", "extra", "certainly", "also", "always"]) # Extend stopwords
    words = text.split()
    clean_words = [word for word in words if word not in stop_list]
    clean_text = ' '.join(clean_words)
    return clean_text

# Initialize RAKE
r = Rake() # include_repeated_phrases=False)

# use this to store keyword-score pair
keyword_score = {}

# create a list to store summary of each row in column
summary = []

# Preprocess the abstract column
df['clean_abstract'] = df['abstract'].apply(preprocess_text)

# Iterate through each row in the 'abstract' column and extract keywords
for index, row in df.iterrows():
    abstract = row['clean_abstract']
    
    # Extract keywords using RAKE
    r.extract_keywords_from_text(abstract)
    
    # Get the keywords and print them
    keywords = r.get_ranked_phrases_with_scores()[:30]
    
    # Store keyword-score pairs in the dictionary
    for score, keyword in keywords:
        # If the keyword already exists in the dictionary, update its score if the new score is higher
        if keyword in keyword_score:
            keyword_score[keyword] = max(keyword_score[keyword], score)
        else:
            keyword_score[keyword] = score

      # Extract summary for each row
    row_summary = []
    num_sentences = 0
    for sentence in sent_tokenize(abstract):
        for keyword, score in keyword_score.items():
            if keyword in sentence:
                row_summary.append(sentence)
                num_sentences += 1
                break  # Stop searching for this keyword in other sentences
        if num_sentences == 10:  # If 10 sentences have been extracted, break
            break
    
    # # error msg for checking if there are a tota of 10 sentences generated
    # if num_sentences < 10:
    #     raise ValueError(f"Summary for row {index + 1} has only {num_sentences} sentences. It should have 10 sentences.")

    summary.append(row_summary)

# Print the summary for each row
for index, row in enumerate(summary):
    print(f"Summary for row {index + 1}:")
    for i, sentence in enumerate(row):
        print(f"{i + 1}: {sentence}")
    print()