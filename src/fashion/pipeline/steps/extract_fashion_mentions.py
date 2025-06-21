"""
Extract fashion mentions from text files using keyword matching.

python -m fashion.process_texts --data_source hathitrust  --output_filename data/hathitrust_fashion_texts.csv
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import spacy
from tqdm import tqdm

from fashion.sources import Source, add_source_argument


def strip_punctuation(text):
    # Use regex to remove all punctuation
    return re.sub(r"[^\w\s]", "", text)


def preprocess(doc):
    # TODO: do faster sentence segmentation than spacy
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
def save_results_csv(fashion_results, output_filename: Path):
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "mention_id",
                "filename",
                "sentence",
                "term",
                "start_idx",
                "end_idx",
                "sentence_start_idx",
                "sentence_end_idx",
            ]
        )  # Header row

        mention_id = 0
        for filename, (sentences) in fashion_results.items():
            for sentence, indices, sentence_start, sentence_end in sentences:
                for start, end in indices:
                    # Extract the fashion term from the sentence
                    fashion_term = sentence[start:end]
                    # Write the filename and the fashion term to the CSV file
                    writer.writerow(
                        [
                            mention_id,
                            filename,
                            sentence,
                            fashion_term,
                            start,
                            end,
                            sentence_start,
                            sentence_end,
                        ]
                    )
                    mention_id += 1


def load_texts(datasource: Source, max_files: int | None = None):
    """
    Breaks up text into newline-delimited chunks and yields filename and text content.
    """
    for i, text in enumerate(
        tqdm(
            datasource.iter_texts(),
            desc="Loading files",
            total=len(datasource) if max_files is None else max_files,
        )
    ):
        if max_files is not None and i >= max_files:
            break

        content = text.text

        def yield_chunks_with_divider(content, divider):
            current_chunk = ""
            chunk_start_idx = 0
            for chunk in content.split(divider):
                current_chunk += chunk + divider
                if len(current_chunk) > 10000:
                    yield current_chunk, chunk_start_idx
                    chunk_start_idx += len(current_chunk)
                    current_chunk = ""
            if current_chunk.strip():
                yield current_chunk, chunk_start_idx

        for chunk, chunk_start_idx in yield_chunks_with_divider(content, "\n\n"):
            if len(chunk) - 10_000 > 10_000:
                for smaller_chunk, smaller_chunk_start_idx in yield_chunks_with_divider(
                    chunk, "\n"
                ):
                    yield (
                        text.filename,
                        smaller_chunk,
                        chunk_start_idx + smaller_chunk_start_idx,
                    )
            else:
                yield text.filename, chunk, chunk_start_idx


def process_all_files(directory, max_files=None):
    fashion_results = defaultdict(list)

    nlp = spacy.load("en_core_web_sm")
    filenames, texts, chunk_start_idxs = zip(
        *load_texts(directory, max_files=max_files)
    )
    docs = nlp.pipe(texts, batch_size=64, n_process=32)

    extractor = NaiveKeywordExtractor()

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

    return fashion_results


def main(source: Source, max_files, output_filename):
    # Load the text files from the specified directory
    fashion_results = process_all_files(source, max_files)

    # Save the results to a CSV file
    save_results_csv(fashion_results, output_filename)

    print(f"Results have been saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_argument(parser)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--output_filename", type=Path, default="fashion_results.csv")

    args = parser.parse_args()
    main(**vars(args))
