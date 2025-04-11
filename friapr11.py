#Adjectives


import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Load English tokenizer, POS tagger, parser, NER
nlp = spacy.load("en_core_web_sm")
#
# text = "The quick brown fox jumps over the lazy dog."
#
# # Process the text
# doc = nlp(text)
#
# # Extract adjectives
# adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
#
# print(adjectives)

def extract_adjectives(text):
    doc = nlp(text)

    # Extract adjectives
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return adjectives

#Cosine similarity / classifier


def vectorize(text1, text2):
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine distance
    cosine_distance = cosine_distances(tfidf_matrix[0], tfidf_matrix[1])[0][0]


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('fashion_results.csv')
    df['adjectives'] = ""
    for idx, row in df.iterrows():
        text = row['Fashion Sentence']
        df['adjectives'] = extract_adjectives(text)


