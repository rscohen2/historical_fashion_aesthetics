import argparse
import csv
from html import parser
import os
import re
from collections import defaultdict

import spacy
from tqdm import tqdm

from fashion.utils import CHICAGO_PATH, DATA_DIR


def strip_punctuation(text):
    # Use regex to remove all punctuation
    return re.sub(r"[^\w\s]", "", text)


def preprocess(doc):
    # text = strip_punctuation(text)
    sentences = list(doc.sents)  # Split into sentences
    return sentences


def load_keywords(path: str):
    keywords = [line.strip() for line in open(path, "r")]
    return set(keywords)


# Function to read a text file
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


# Function to save the results to a CSV file
def save_results_csv(fashion_results, output_filename):
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "sentence_count"])  # Header row

        for filename, sentence_count in fashion_results.items():
            writer.writerow([filename, sentence_count])


def load_texts(directory, max_files):
    """
    Breaks up text into newline-delimited chunks and yields filename and text content.
    """
    filepaths = os.listdir(directory)
    if max_files is not None:
        filepaths = filepaths[:max_files]

    for filename in tqdm(filepaths, desc="Loading files"):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            text = read_file(file_path)

            current_chunk = ""
            for chunk in text.split("\n\n"):
                if chunk.strip():
                    current_chunk += chunk.strip() + "\n\n"
                if len(current_chunk) > 10000:
                    yield filename, current_chunk.strip()
                    current_chunk = ""
            if current_chunk.strip():
                yield filename, current_chunk


# Main function to process all files in a directory (limiting to first 3 files)
def process_all_files(directory, max_files=None):
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "parser"])
    filenames, texts = zip(*load_texts(directory, max_files=max_files))
    docs = nlp.pipe(texts, batch_size=64, n_process=32)

    sentence_counts = defaultdict(int)

    for filename, doc in tqdm(
        zip(filenames, docs), total=len(filenames), desc="Extracting fashion sentences"
    ):
        sentences = preprocess(doc)
        sentence_counts[filename] += len(sentences)

    return sentence_counts


def main(input_directory, max_files, output_filename):
    # Load the text files from the specified directory
    fashion_results = process_all_files(input_directory, max_files)

    # Save the results to a CSV file
    save_results_csv(fashion_results, output_filename)

    print(f"Results have been saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory", type=str, default=CHICAGO_PATH / "CLEAN_TEXTS"
    )
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--output_filename", type=str, default=DATA_DIR / "sentence_counts.csv"
    )

    args = parser.parse_args()
    main(**vars(args))
