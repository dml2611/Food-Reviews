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
keyword_score_abs = {}
keyword_score_gpt = {}

# create a list to store summary of each row in column
summary_abs = []
summary_gpt = []

# create a list to store summary of the current row
row_summary_abs = []
row_summary_gpt = []

# Create a list to store the scores of sentences
sentence_scores_abs = []
sentence_scores_gpt = []

# Preprocess the abstract column
df['clean_abstract'] = df['abstract'].apply(preprocess_text)

# Preprocess the other column (e.g., 'summary_ChatGPT_10_sentences')
df['clean_gpt'] = df['summary_ChatGPT_10_sentences'].apply(preprocess_text)

# Iterate through each row in the 'abstract' column and extract keywords
for index, row in df.iterrows():
    abstract = row['clean_abstract']
    gpt = row['clean_gpt']

    # Extract keywords using RAKE
    r.extract_keywords_from_text(abstract)
    # Get the keywords and print them
    keywords_abs = r.get_ranked_phrases_with_scores()
    
    # Extract keywords using RAKE
    r.extract_keywords_from_text(gpt)
    # Get the keywords and print them
    keywords_gpt = r.get_ranked_phrases_with_scores()
    
     # Store keyword-score pairs in dictionaries for the abstract and ChatGPT summary
    keyword_score_abs = {keyword: score for score, keyword in keywords_abs}
    keyword_score_gpt = {keyword: score for score, keyword in keywords_gpt}

        # Iterate through sentences in the abstract and calculate sentence scores
    for sentence in sent_tokenize(abstract):
        sentence_score = sum(keyword_score_abs.get(word, 0) for word in sentence.split())
        sentence_scores_abs.append(sentence_score)
        row_summary_abs.append(sentence)

    # Iterate through sentences in the ChatGPT summary and calculate sentence scores
    for sentence in sent_tokenize(gpt):
        sentence_score = sum(keyword_score_gpt.get(word, 0) for word in sentence.split())
        sentence_scores_gpt.append(sentence_score)
        row_summary_gpt.append(sentence)

    # Append the total score of the summary to the sentence scores list
    sentence_scores_abs.append(sum(sentence_scores_abs))
    sentence_scores_gpt.append(sum(sentence_scores_gpt))

    # Append the row summary and sentence scores for the abstract and ChatGPT summary to the summary list
    summary.append((row_summary_abs, row_summary_gpt))
    sentence_scores.append((sentence_scores_abs, sentence_scores_gpt))

    import sys

    rake3_sent = sys.stdout
    with open('outputRake3Sent.txt', 'w') as f:
        sys.stdout = f

        # Iterate through each row in the DataFrame
        for index, (row_summary_abs, row_summary_gpt) in enumerate(summary):
            print(f"Top 10 Sentences with the Highest Scores for Row {index + 1}:")
            
            # Sort the abstract summary sentences based on their scores
            sorted_sentences_abs = sorted(zip(row_summary_abs[:-1], sentence_scores[index][0]), key=lambda x: x[1], reverse=True)
            
            # Print the top 10 sentences with the highest scores for the abstract summary
            print("Abstract Summary:")
            for i, (sentence_abs, score_abs) in enumerate(sorted_sentences_abs[:10]):
                print(f"Sentence {i + 1}: {sentence_abs} - Score: {score_abs}")
            print()

            # Sort the ChatGPT summary sentences based on their scores
            sorted_sentences_gpt = sorted(zip(row_summary_gpt[:-1], sentence_scores[index][1]), key=lambda x: x[1], reverse=True)
            
            # Print the top 10 sentences with the highest scores for the ChatGPT summary
            print("ChatGPT Summary:")
            for i, (sentence_gpt, score_gpt) in enumerate(sorted_sentences_gpt[:10]):
                print(f"Sentence {i + 1}: {sentence_gpt} - Score: {score_gpt}")
            print()
        
        sys.stdout = rake3_sent
