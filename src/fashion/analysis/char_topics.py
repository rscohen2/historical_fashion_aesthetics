"""
Construct character archetypes with topic models over the char adjectives.
"""

import argparse
import pickle

import lda
import numpy as np
import pandas as pd
from lda.utils import lists_to_matrix
from tqdm import tqdm

from fashion.paths import DATA_DIR


def preprocess_docs(docs: list[list[str]]):
    # each doc is a list of character adjectives
    for i, doc in enumerate(docs):
        for w in sorted(set(w for w in doc if w is not None)):
            yield i, w.lower()


def docterm_matrix(docs: list[list[str]]):
    vocab = {}
    word_doc_pair = []
    for i, w in tqdm(
        preprocess_docs(docs),
        desc="Building vocab and word-doc pairs...",
        total=sum(len(doc) for doc in docs),
    ):
        if w not in vocab:
            vocab[w] = len(vocab)
        word_doc_pair.append((vocab[w], i))

    ws, ds = zip(*word_doc_pair)
    return vocab, lists_to_matrix(ws, ds)


def main(args: argparse.Namespace) -> None:
    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df: pd.DataFrame = pd.read_parquet(input_path)

    vocab, dtm = docterm_matrix(df.groupby(level=1).adjectives_char.first().tolist())
    model = lda.LDA(n_topics=args.num_topics)
    model.fit(dtm)

    vocab = np.array(list(vocab.keys()))

    for i, topic_dist in enumerate(model.topic_word_):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-10:-1]
        print(f"Topic {i}: {', '.join(topic_words)}")

    output_dir = DATA_DIR / f"analysis/char_topics{'_debug' if args.debug else ''}"
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "vocab.npy", vocab)
    with (output_dir / f"model_{args.num_topics}.pkl").open("wb") as f:
        pickle.dump(model, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze adjective categories.")
    _ = parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a smaller sample for faster analysis.",
    )
    _ = parser.add_argument(
        "-k",
        "--num-topics",
        type=int,
        default=10,
        help="Number of topics to extract.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
