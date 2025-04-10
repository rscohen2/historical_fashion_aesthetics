from tqdm import tqdm
import spacy
import os
import csv
import re

nlp = spacy.load("en_core_web_sm")


def strip_punctuation(text):
    # Use regex to remove all punctuation
    return re.sub(r"[^\w\s]", "", text)


# hi
def preprocess(text):
    text = text.lower()
    # text = strip_punctuation(text)
    doc = nlp(text)
    sentences = list(doc.sents)  # Split into sentences
    return sentences


def load_keywords(path: str):
    keywords = [line.strip() for line in open(path, "r")]
    return set(keywords)


def extract_fashion_terms(sentences):
    # for sent in sentences:
    fashion_keywords = load_keywords("data/clothes.txt")
    fashion_sentences = []
    # for sentence in sentences:
    # Loop through each sentence
    for sentence in sentences:
        # Check if any fashion keyword is in the sentence
        if any(word.text in fashion_keywords for word in sentence):
            fashion_sentences.append(sentence.text)
    return fashion_sentences


# fashion_keywords = ["dress", "style", "couture", "outfit", "designer", "seamstress", "silk", "tailor","dressed","shirt","dress","skirt","sleeve","collar"]
# fashion_adjectives = ["elegant", "vintage", "floral", "sleek", "chic"]


# Function to read a text file
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


# Function to save the results to a CSV file
def save_results_csv(fashion_results, output_filename):
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Fashion Sentence"])  # Header row

        for filename, sentences in fashion_results.items():
            for sentence in sentences:
                writer.writerow([filename, sentence])


# Main function to process all files in a directory (limiting to first 3 files)
def process_all_files(directory, max_files=3):
    fashion_results = {}
    files_processed = 0

    # Get a list of all .txt files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".txt"):
            if files_processed >= max_files:
                break  # Stop after processing the first 'max_files' files

            file_path = os.path.join(directory, filename)

            # Read the file
            text = read_file(file_path)
            sentences = preprocess(text)

            # Process the text for fashion-related sentences
            fashion_sentences = extract_fashion_terms(sentences)
            # Store results
            fashion_results[filename] = fashion_sentences
            files_processed += 1

    # Save results to CSV
    output_filename = os.path.join(directory, "fashion_results.csv")
    save_results_csv(fashion_results, output_filename)

    return fashion_results


if __name__ == "__main__":
    # Directory where the .txt files are stored
    # input_directory = "ChicagoCorpus/CHICAGO_CORPUS/CHICAGO_NOVEL_CORPUS"
    input_directory = "ChicagoCorpus/CLEAN_TEXTS"

    # Process the first 3 files in the directory
    results = process_all_files(input_directory, max_files=10)

    # Save the results to the CSV file
    output_filename = "fashion_results.csv"
    save_results_csv(results, output_filename)

    # output_filename = os.path.join("..", "fashion_results.csv")
    # Get the absolute path of the input_directory
    # absolute_input_directory = os.path.abspath(input_directory)
    #
    # # Get the parent directory of the absolute path
    # parent_directory = os.path.dirname(absolute_input_directory)
    #
    # output_filename = os.path.join(parent_directory, 'fashion_results.csv')
    #
    # print(f"Results have been saved to {output_filename}")
