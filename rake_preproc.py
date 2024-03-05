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
r = Rake(include_repeated_phrases=False)

# use this to store keyword-score pair
keyword_score = {}

# create a list to store summary of each row in column
summary = []

# Create a list to store the scores of sentences
sentence_scores = []

# Preprocess the abstract column
df['clean_abstract'] = df['abstract'].apply(preprocess_text)

# Iterate through each row in the 'abstract' column and extract keywords
for index, row in df.iterrows():
    abstract = row['clean_abstract']
    
    # Extract keywords using RAKE
    r.extract_keywords_from_text(abstract)
    
    # Get the keywords and print them
    keywords = r.get_ranked_phrases_with_scores()
    
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
    total_score = 0
    for sentence in sent_tokenize(abstract):
        sentence_score = 0
        for keyword, score in keyword_score.items():
            if keyword in sentence:
                sentence_score += score  # Accumulate the score for keywords present in the sentence
        sentence_scores.append(sentence_score)
        row_summary.append(sentence)
        num_sentences += 1
        total_score += sentence_score
        if num_sentences == 10:  # If 10 sentences have been extracted, break
            break
    
    # error msg for checking if there are a tota of 10 sentences generated
    if num_sentences < 10:
        raise ValueError(f"Summary for row {index + 1} has only {num_sentences} sentences. It should have 10 sentences.")

    # error msg for checking if there are a tota of 10 sentences generated
    if num_sentences > 10:
        raise ValueError(f"Summary for row {index + 1} has {num_sentences} sentences. It should have 10 sentences.")

    # Append the total score of the summary to the sentence scores list
    sentence_scores.append(total_score)
    summary.append(row_summary)

# Preprocess the text column
df['clean_chatgpt'] = df['summary_ChatGPT_10_sentences'].apply(preprocess_text)  # Change 'your_other_column' to the name of your column

# Iterate through each row in the column and extract keywords
for index, row in df.iterrows():
    chatgpt = row['clean_chatgpt']  # Change 'clean_text' to the name you assigned during preprocessing
    
    # Extract keywords using RAKE
    r.extract_keywords_from_text(chatgpt)
    
    # Get the keywords and print them
    keywords = r.get_ranked_phrases_with_scores()
    
    # Store keyword-score pairs in the dictionary
    keyword_score = {}
    for score, keyword in keywords:
        keyword_score[keyword] = score

    # Extract summary for each row
    row_summary = []
    total_score = 0
    for sentence in sent_tokenize(chatgpt):
        sentence_score = 0
        for keyword, score in keyword_score.items():
            if keyword in sentence:
                sentence_score += score  # Accumulate the score for keywords present in the sentence
        sentence_scores.append(sentence_score)
        row_summary.append(sentence)
        total_score += sentence_score
    
    # Append the total score of the summary to the sentence scores list
    sentence_scores.append(total_score)
    summary.append(row_summary)

    import sys

    rake3_sent = sys.stdout
    with open('outputRake3Preproc.txt', 'w') as f:
        sys.stdout = f

        # Print out sentences with their individual scores
        for index, sentences in enumerate(summary):
            print(f"Summary for row {index + 1}:")
            for i, sentence in enumerate(sentences):
                print(f"Sentence {i + 1}: {sentence} Sentence Score: {sentence_scores[index * 10 + i]}")
            print()
        
        sys.stdout = rake3_sent
