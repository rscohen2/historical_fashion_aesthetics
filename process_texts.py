import spacy
import os
import csv
import re
nlp = spacy.load('en_core_web_sm')

def strip_punctuation(text):
    # Use regex to remove all punctuation
    return re.sub(r'[^\w\s]', '', text)

#hi
def preprocess(text):
    text = text.lower()
    # text = strip_punctuation(text)
    doc = nlp(text)
    sentences = list(doc.sents)  # Split into sentences
    return sentences

def extract_fashion_terms(sentences):
    # for sent in sentences:
    fashion_keywords = ["dress", "style", "couture", "outfit", "designer", "seamstress", "silk", "tailor", "dressed",
                        "shirt", "dress", "skirt", "sleeve", "collar"]

    fashion_sentences = []
    for sentence in sentences:
        for word in sentence:
            if word in fashion_keywords:
                fashion_sentences.append(sentences)
                return fashion_sentences

# fashion_keywords = ["dress", "style", "couture", "outfit", "designer", "seamstress", "silk", "tailor","dressed","shirt","dress","skirt","sleeve","collar"]
fashion_adjectives = ["elegant", "vintage", "floral", "sleek", "chic"]

# Function to read a text file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text



def save_results_csv(fashion_results, output_filename):
    with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Fashion Sentence'])  # Header row

        for filename, sentences in fashion_results.items():
            for sentence in sentences:
                writer.writerow([filename, sentence])


# Main function to process all files in a directory
def process_all_files(directory):
    fashion_results = {}

    # Get a list of all .txt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Read the file
            text = read_file(file_path)
            text = preprocess(text)

            # Process the text for fashion-related sentences
            fashion_sentences = extract_fashion_terms(text)
            # Store results
            fashion_results[filename] = fashion_sentences

            # # Optionally, save the results to a file (one per novel)
            # output_file = f"fashion_output_{filename}"
            # save_results(output_file, fashion_sentences)

        return fashion_results



if __name__ == "__main__":
    # Directory where the .txt files are stored
    # input_directory = "ChicagoCorpus/CHICAGO_NOVEL_CORPUS"
    #
    # # Process all files in the directory
    # results = process_all_files(input_directory)
    # output_filename = 'ChicagoCorpus/CHICAGO_NOVEL_CORPUS/output_file'
    # results = save_results_csv(results, output_filename)

    text = 'She was a lovely woman who dressed in loads of colored silk. She loved to go out on Saturdays.'
    text = preprocess(text)
    for sentence in text:
        fashion_terms = extract_fashion_terms(text)
        print(fashion_terms)