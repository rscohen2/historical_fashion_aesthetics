"""
Port the adjective category code to a script.

Output HTML file that visualizes the categories and the adjectives that belong to them.
"""

import argparse
import json
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.wsd import lesk
from tqdm import tqdm

from fashion.paths import DATA_DIR

EXPERIMENT = "adjective_categories"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT


def get_attribute_categories(
    term: str | None = None, synset: Synset | None = None, context: str | None = None
) -> list[Synset]:
    if synset is None:
        if context is None:
            synsets: list[Synset | None] = wn.synsets(term, pos="a")
        else:
            # we add some fashion-related terms to the context to help disambiguate the sense of the adjective
            words = nltk.word_tokenize(
                context + " clothes clothing garment apparel fashion"
            )
            synset = lesk(words, term, pos="a")
            if synset is None:
                synset = lesk(words, term, pos="s")
            synsets = [synset]
    else:
        synsets = [synset]
    attributes = []
    for synset in synsets:
        if synset is None:
            continue
        if synset.attributes():
            attributes.extend(synset.attributes())
        elif synset.similar_tos():
            for similar_to in synset.similar_tos():
                if similar_to.attributes():
                    attributes.extend(similar_to.attributes())
    return attributes


def get_category_members(synset: Synset) -> list[Synset]:
    members = []
    for member in synset.member_holonyms():
        members.append(member)
    for member in synset.substance_holonyms():
        members.append(member)
    for member in synset.part_holonyms():
        members.append(member)
    return members


def get_adjective_category_distribution(
    adj_series: pd.Series, sentences: pd.Series | None = None
) -> tuple[Counter, dict[str, set[str]], dict[tuple[str, str], list[str]]]:
    if sentences is None:
        sentences = pd.Series([None] * len(adj_series))

    df = pd.DataFrame({"adjective": adj_series, "sentences": sentences})
    df = df.explode("adjective").dropna()

    term_adjs = df.adjective.str.lower().copy()

    adj_categories = [
        get_attribute_categories(term=adj, context=sentence)
        for adj, sentence in tqdm(zip(term_adjs, df.sentences), total=len(df))
    ]
    term_adjs_categories = pd.DataFrame(
        {
            "adjective": term_adjs,
            "categories": [
                [category_synset.name() for category_synset in category_synsets]
                for category_synsets in adj_categories
            ],
            "sentences": df.sentences,
            # "categories": term_adjs.apply(lambda adj: get_attribute_categories(adj)),
        }
    )

    exploded_categories = term_adjs_categories.explode("categories")
    category_to_term_map = exploded_categories.groupby("categories").adjective.agg(set)
    category_counts = Counter(exploded_categories.categories.dropna())

    category_adjective_sentence_map = (
        exploded_categories.groupby(["categories", "adjective"])
        .sentences.agg(
            lambda x: x.sample(min(10, len(x))).tolist() if len(x) > 10 else x.tolist()
        )
        .to_dict()
    )

    return (
        category_counts,
        category_to_term_map.to_dict(),
        category_adjective_sentence_map,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze adjective categories.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a smaller sample for faster analysis.",
    )
    return parser.parse_args()


def main(args) -> None:
    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df = pd.read_parquet(input_path)
    category_counts, category_to_term_map, category_adj_sentences = (
        get_adjective_category_distribution(df.adjectives_fashion, df.sentence)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    category_counts_df = pd.DataFrame(
        category_counts.items(),
        columns=["category", "count"],  # pyright: ignore[reportArgumentType]
    )
    category_counts_df.sort_values(by="count").to_csv(
        OUTPUT_DIR / "category_counts.csv", index=False
    )

    with open(OUTPUT_DIR / "category_to_term_map.jsonl", "w") as f:
        for category in category_counts_df.sort_values(
            by="count", ascending=False
        ).category:
            synset = wn.synset(category)
            terms = category_to_term_map[category]

            if synset is None:
                print(f"Warning: No synset found for category {category}")
                continue
            output_object = {
                "category": category,
                "definition": synset.definition(),
                "terms": list(terms),
                "example_sentences": {
                    adj: category_adj_sentences.get((category, adj), None)
                    for adj in terms
                },
            }
            f.write(f"{json.dumps(output_object)}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
