import nltk
from rake_nltk import Rake
import pandas as pd 
from nltk.tokenize import sent_tokenize
import torch
from rouge import Rouge

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the DataFrame or your text corpus
df = pd.read_csv("Book1.csv")

# Initialize RAKE
r = Rake(include_repeated_phrases=False)

# use this to store keyword-score pair for abstract and ChatGPT
keyword_score_text = {}
keyword_score_gpt = {}

# create a list to store summary of each row in column
summary = []

# Initialize RAKE
r = Rake(include_repeated_phrases=False)

# use this to store keyword-score pair
keyword_score = {}

# create a list to store summary of each row in column
summary = []

# Create a list to store the scores of sentences
sentence_scores = []
# 
# # initialise ROUGE object
rouge = Rouge()

# Preprocessing function
def preprocess_text(text):
    words = text.split()
    # clean_words = [word for word in words if word not in stop_list]
    clean_word = ' '.join(words)
    return clean_word

# Preprocess the text & chatGPT summary column
df['clean_text'] = df['text'].apply(preprocess_text)
df['clean_gpt'] = df['summary_ChatGPT_15_sentences'].apply(preprocess_text)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    text = row['clean_text']
    text_gpt = row['clean_gpt']
    
    # Extract keywords using RAKE for the text
    r.extract_keywords_from_text(text)
    keywords_text = r.get_ranked_phrases_with_scores()
    
    # Extract keywords using RAKE for the GPT summary
    r.extract_keywords_from_text(text_gpt)
    keywords_gpt = r.get_ranked_phrases_with_scores()
    
    # Store keyword-score pairs in the dictionary for text
    for score, keyword in keywords_text:
        if keyword in keyword_score_text:
            keyword_score_text[keyword] = max(keyword_score_text[keyword], score)
        else:
            keyword_score_text[keyword] = score
    
    # Store keyword-score pairs in the dictionary for GPT summary
    for score, keyword in keywords_gpt:
        if keyword in keyword_score_gpt:
            keyword_score_gpt[keyword] = max(keyword_score_gpt[keyword], score)
        else:
            keyword_score_gpt[keyword] = score

    # Append the summary for each row to the summary list
    summary.append((text, text_gpt))
  


    # ============================================================================ 

    # Iterate through the summary list and print the output
    for index, (row_text, row_gpt) in enumerate(summary):
        print(f"Top 10 Sentences with the Highest Scores for Row {index + 1}:")
        
        # Print the text summary
        print("Text Summary:")
        # Sort sentences based on their scores
        sorted_sentences_txt = sorted(sent_tokenize(row_text)[:10], key=lambda x: sum(keyword_score_text.get(word, 0) for word in x.split()), reverse=True)
        print(sorted_sentences_txt)
        print("\n")

        # # Print the ChatGPT summary
        print("ChatGPT Summary:")
        # # Sort sentences based on their scores
        sorted_sentences_gpt = sorted(sent_tokenize(row_gpt)[:10], key=lambda x: sum(keyword_score_gpt.get(word, 0) for word in x.split()), reverse=True)
        print(sorted_sentences_gpt)
        print("\n")


# =============================================================================================
 
    #     # scores_text = rouge.get_scores(sorted_sentences_txt, [text])
    #     # print(scores_text)
    # # scores_gpt = rouge.get_scores(sorted_sentences_gpt, [text])
    # # print(scores_gpt)

    #     print(len(sorted_sentences_txt), len(text))
    #     # print(text)

import sys

rake_wo_preproc = sys.stdout
with open('top10.txt', 'w') as f:
    sys.stdout = f

   # Iterate through the summary list and print the output
    for index, (row_text, row_gpt) in enumerate(summary):
        print(f"Top 10 Sentences with the Highest Scores for Row {index + 1}:")
        
        # Print the text summary
        print("Text Summary:")
        # Sort sentences based on their scores
        sorted_sentences_txt = sorted(sent_tokenize(row_text)[:10], key=lambda x: sum(keyword_score_text.get(word, 0) for word in x.split()), reverse=True)
        print(sorted_sentences_txt)
        print("\n")

        # # Print the ChatGPT summary
        print("ChatGPT Summary:")
        # # Sort sentences based on their scores
        sorted_sentences_gpt = sorted(sent_tokenize(row_gpt)[:10], key=lambda x: sum(keyword_score_gpt.get(word, 0) for word in x.split()), reverse=True)
        print(sorted_sentences_gpt)
        print("\n")

    sys.stdout = rake_wo_preproc