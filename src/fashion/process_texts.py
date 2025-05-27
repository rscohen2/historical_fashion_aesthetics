import argparse
import csv
from html import parser
import os
import re
from collections import defaultdict

import spacy
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

from fashion.utils import CHICAGO_PATH


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


class FashionExtractor:
    def is_fashion(self, sentence) -> list[tuple[int, int]]:
        raise NotImplementedError

    def extract(
        self, sentences, chunk_start_idx=0
    ) -> list[tuple[str, list[tuple[int, int]]]]:
        fashion_sentences = []
        for sentence in sentences:
            if word_inds := self.is_fashion(sentence):
                fashion_sentences.append(
                    (
                        sentence.text,
                        word_inds,
                        chunk_start_idx + sentence[0].idx,
                        chunk_start_idx + sentence[-1].idx + len(sentence[-1].text),
                    )
                )
        return fashion_sentences


class NaiveKeywordExtractor(FashionExtractor):
    def __init__(self, keywords_path="data/clothes.txt"):
        self.keywords = load_keywords(keywords_path)

    def is_fashion(self, sentence):
        return [
            (word.idx - sentence[0].idx, word.idx - sentence[0].idx + len(word.text))
            for word in sentence
            if word.text in self.keywords and word.pos == 92
        ]


# Function to read a text file
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


# Function to save the results to a CSV file
def save_results_csv(fashion_results, output_filename):
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "filename",
                "sentence",
                "term",
                "start_idx",
                "end_idx",
                "sentence_start_idx",
                "sentence_end_idx",
            ]
        )  # Header row

        for filename, (sentences) in fashion_results.items():
            for sentence, indices, sentence_start, sentence_end in sentences:
                for start, end in indices:
                    # Extract the fashion term from the sentence
                    fashion_term = sentence[start:end]
                    # Write the filename and the fashion term to the CSV file
                    writer.writerow(
                        [
                            filename,
                            sentence,
                            fashion_term,
                            start,
                            end,
                            sentence_start,
                            sentence_end,
                        ]
                    )


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
            chunk_start_idx = 0
            for chunk in text.split("\n\n"):
                current_chunk += chunk + "\n\n"
                if len(current_chunk) > 10000:
                    yield filename, current_chunk, chunk_start_idx
                    chunk_start_idx += len(current_chunk)
                    current_chunk = ""
            if current_chunk.strip():
                yield filename, current_chunk, chunk_start_idx


# Main function to process all files in a directory (limiting to first 3 files)
def process_all_files(directory, max_files=None):
    fashion_results = defaultdict(list)

    nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe("sentencizer")
    filenames, texts, chunk_start_idxs = zip(
        *load_texts(directory, max_files=max_files)
    )
    docs = nlp.pipe(texts, batch_size=64, n_process=32)

    extractor = NaiveKeywordExtractor()
    # extractor = BertFashionExtractor()

    for filename, doc, chunk_start_idx in tqdm(
        zip(filenames, docs, chunk_start_idxs),
        total=len(filenames),
        desc="Extracting fashion sentences",
    ):
        sentences = preprocess(doc)
        fashion_sentences = extractor.extract(
            sentences, chunk_start_idx=chunk_start_idx
        )
        fashion_results[filename].extend(fashion_sentences)

    # sentence_futures = [
    #     process_file.remote(doc, filename) for filename, doc in zip(filenames, docs)
    # ]

    # progress_bar = tqdm(total=len(sentence_futures), desc="Processing files")
    # while sentence_futures:
    #     ready_sentences, sentence_futures = ray.wait(
    #         sentence_futures, num_returns=min(len(sentence_futures), 128)
    #     )
    #     for filename, sentences in ray.get(ready_sentences):
    #         if sentences:
    #             fashion_results[filename].extend(sentences)
    #         progress_bar.update(1)
    # progress_bar.close()

    # # Save results to CSV
    # output_filename = os.path.join(directory, "fashion_results.csv")
    # save_results_csv(fashion_results, output_filename)

    return fashion_results


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
    parser.add_argument("--output_filename", type=str, default="fashion_results.csv")

    args = parser.parse_args()
    main(**vars(args))

    # # Directory where the .txt files are stored
    # # input_directory = "ChicagoCorpus/CHICAGO_CORPUS/CHICAGO_NOVEL_CORPUS"
    # input_directory = "./data/ChicagoCorpus/CLEAN_TEXTS"

    # # Process the first 3 files in the directory
    # results = process_all_files(input_directory)

    # # Save the results to the CSV file
    # output_filename = "fashion_results_pos.csv"
    # save_results_csv(results, output_filename)

    # # output_filename = os.path.join("..", "fashion_results.csv")
    # # Get the absolute path of the input_directory
    # # absolute_input_directory = os.path.abspath(input_directory)
    # #
    # # # Get the parent directory of the absolute path
    # # parent_directory = os.path.dirname(absolute_input_directory)
    # #
    # # output_filename = os.path.join(parent_directory, 'fashion_results.csv')
    # #
    # # print(f"Results have been saved to {output_filename}")
