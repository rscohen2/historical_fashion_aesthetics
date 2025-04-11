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


def extract_nouns(text):
    doc = nlp(text)

    # Extract adjectives
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return nouns



#Cosine similarity / classifier


def vectorize(text1, text2):
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine distance
    cosine_distance = cosine_distances(tfidf_matrix[0], tfidf_matrix[1])[0][0]

def noun_adj_map(text):
    # Process the text
    doc = nlp(text)

    # Find nouns and their adjectives
    noun_adj_map = {}

    for token in doc:
        # If the token is a noun
        if token.pos_ == "NOUN":
            adjectives = [child.text for child in token.lefts if child.dep_ == "amod"]
            if adjectives:
                noun_adj_map[token.text] = adjectives
                return noun_adj_map



if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('fashion_results.csv')
    df['adjectives'] = ""
    df['nouns'] = ""
    df['noun_adj_map'] = ""
    for idx, row in df.iterrows():
        text = row['Fashion Sentence']
        df.at[idx, 'adjectives'] = extract_adjectives(text)
        df.at[idx, 'nouns'] = extract_nouns(text)
        df.at[idx, 'noun_adj_map'] = noun_adj_map(text)

    df.to_csv('df_spacy.csv')

