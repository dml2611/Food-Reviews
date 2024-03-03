import nltk
from rake_nltk import Rake
import pandas as pd 

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the DataFrame or your text corpus
df = pd.read_csv("Book1.csv")

# Initialize RAKE
r = Rake() # include_repeated_phrases=False)

# use this to store keyword-score pair
keyword_score = {}

# Iterate through each row in the 'abstract' column and extract keywords
for index, row in df.iterrows():
    abstract_text = row['abstract']
    
    # Extract keywords using RAKE
    r.extract_keywords_from_text(abstract_text)
    
    # Get the keywords and print them
    keywords = r.get_ranked_phrases_with_scores()[:10]

    # # Print keywords with scores for each row
    # print(f"Keywords with scores for row {index + 1}:")
    # for score, keyword in keywords:
    #     print(f"Keywords: {keyword}, Score: {score}")
    # print()

    # Store keyword-score pairs in the dictionary
    for score, keyword in keywords:
        # If the keyword already exists in the dictionary, update its score if the new score is higher
        if keyword in keyword_score:
            keyword_score[keyword] = max(keyword_score[keyword], score)
        else:
            keyword_score[keyword] = score

# Print the keyword-score dictionary
print(keyword_score)

    
        