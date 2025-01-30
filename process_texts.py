import spacy
import os
nlp = spacy.load('en_core_web_sm')

#hi
def preprocess(text):
    text = text.lower()
    doc = nlp(text)
    sentences = list(doc.sents)  # Split into sentences
    return sentences

def extract_fashion_terms(sentences):
    for sent in sentences:
        fashion_sentences = []
        if any(keyword in sent.text for keyword in fashion_keywords):
            fashion_sentences.append(sent.text)
            return fashion_sentences

fashion_keywords = ["dress", "style", "couture", "outfit", "designer", "seamstress", "silk", "tailor"]
fashion_adjectives = ["elegant", "vintage", "floral", "sleek", "chic"]

# Function to read a text file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Function to save results to a file
def save_results(file_name, fashion_sentences):
    with open(file_name, 'w', encoding='utf-8') as f:
        for sentence in fashion_sentences:
            f.write(sentence + '\n')


# Main function to process all files in a directory
def process_all_files(directory):
    fashion_results = {}

    # Get a list of all .txt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Read the file
            text = read_file(file_path)

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
    input_directory = "ChicagoCorpus/CHICAGO_NOVEL_CORPUS"

    # Process all files in the directory
    results = process_all_files(input_directory)

