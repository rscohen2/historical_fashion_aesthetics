"""
Write out diagnostic files for character archetype topics, including the top words for each topic and the characters most associated with each topic.


In data/analysis/char_topics/diagnostic_{num_topics}.txt, we write out:
1. The top words for each topic.
2. The characters most associated with each topic, along with their adjectives and book title.
"""

import argparse
import pickle
import re

import numpy as np
import pandas as pd

from fashion.paths import DATA_DIR

N_TOP_WORDS = 10
N_TOP_CHARS = 10


def load_book_titles() -> dict[str, str]:
    meta = pd.read_csv(DATA_DIR / "hathimeta/titlemeta.tsv", sep="\t", low_memory=False)
    meta = meta.set_index("docid")
    # Strip the pipe-delimited suffix that appears in raw titles (e.g. "Title | $c: ...")
    meta["title_clean"] = meta["title"].str.split(r"\|").str[0].str.strip()
    return meta["title_clean"].to_dict()


def main(args: argparse.Namespace) -> None:
    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    topics_dir = DATA_DIR / f"analysis/char_topics{'_debug' if args.debug else ''}"
    vocab = np.load(topics_dir / "vocab.npy")
    with (topics_dir / f"model_{args.num_topics}.pkl").open("rb") as f:
        model = pickle.load(f)

    df: pd.DataFrame = pd.read_parquet(input_path)
    char_df = df.groupby(level=1).first()

    book_titles = load_book_titles()

    # book_ids stored as index level 0; groupby(level=1) loses it, so recover via reset
    book_id_per_char = (
        df.index.to_frame(index=False).groupby("character_id")["book_id"].first()
    )

    output_path = topics_dir / f"diagnostic_{args.num_topics}.txt"
    with output_path.open("w") as out:
        for topic_idx in range(args.num_topics):
            topic_dist = model.topic_word_[topic_idx]
            top_word_indices = np.argsort(topic_dist)[::-1][:N_TOP_WORDS]
            top_words = vocab[top_word_indices]

            out.write(f"=== Topic {topic_idx} ===\n")
            out.write(f"Top words: {', '.join(top_words)}\n\n")

            char_topic_weights = model.doc_topic_[:, topic_idx]
            top_char_indices = np.argsort(char_topic_weights)[::-1][:N_TOP_CHARS]

            out.write("Top characters:\n")
            for rank, char_idx in enumerate(top_char_indices, 1):
                character_id = char_df.index[char_idx]
                row = char_df.iloc[char_idx]
                adjectives = row["adjectives_char"]
                gender = row["gender_booknlp"]
                weight = char_topic_weights[char_idx]

                book_id = book_id_per_char.get(character_id, "")
                doc_id = re.sub(r"\.clean$", "", str(book_id))
                title = book_titles.get(doc_id, doc_id)

                out.write(
                    f"  {rank}. [{weight:.3f}] {character_id} ({gender}) — {title}\n"
                )
                out.write(f"     Adjectives: {', '.join(str(x) for x in adjectives)}\n")

            out.write("\n")

    print(f"Wrote diagnostics to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write diagnostic files for character topic models."
    )
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
