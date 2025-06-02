import argparse
from collections import namedtuple
import itertools
from pathlib import Path

import pandas as pd
import stanza
from google.protobuf import json_format
from stanza.server.semgrex import Semgrex
from tqdm import tqdm

from fashion.span_utils import get_overlap_rows_single

Word = namedtuple("Word", ["start_idx", "end_idx", "text"])
Pair = namedtuple("Pair", ["adjective", "noun", "negated"])


def process_batch(nlp: stanza.Pipeline, lines: list[str]) -> list[set[Pair]]:
    def get_word(doc, word) -> Word:
        i, j = word
        word = doc.sentences[i].words[j]
        return Word(word.start_char, word.end_char, word.text)
        # return doc.sentences[i].words[j].text

    def get_compound(doc, compounds, word):
        # build compound string
        output = [word]
        while word in compounds:
            word = compounds[word]
            output.append(word)
        start_tok = output[-1]
        end_tok = output[0]
        start_idx = get_word(doc, start_tok).start_idx
        end_idx = get_word(doc, end_tok).end_idx
        return Word(
            start_idx,
            end_idx,
            " ".join([get_word(doc, word).text for word in output[::-1]]),
        )

    in_docs = [stanza.Document([], text=d.strip()) for d in lines]
    docs = nlp(in_docs)

    def process_doc(doc):
        pairs = set()
        description = dict()
        compound = dict()
        conj = dict()
        matches = sem.process(
            doc,
            "{cpos:ADJ}=adjective <amod=adj {pos:/NN.*|PRP/}=noun",
            "{cpos:ADJ}=adjective >nsubj=adj {pos:/NN.*|PRP/}=noun",
            "{cpos:ADJ}=adjective </acl.*/=adj {pos:/NN.*|PRP/}=noun",
            "{cpos:VERB}=link >xcomp=xcomp {cpos:ADJ}=adjective >obj {pos:/NN.*|PRP/}=noun",
            "{pos:/NN.*|PRP/}=first_noun <compound=compound {pos:/NN.*|PRP/}=second_noun",
            "{cpos:ADJ}=adjective <conj=conj {cpos:ADJ}=second_adjective",
            "{lemma:not}=not <advmod=not {cpos:ADJ}=adjective",
        )
        matches = json_format.MessageToDict(matches)
        # print(matches)
        negations = set()
        for i, result in enumerate(matches["result"]):
            for match in result["result"]:
                if "match" not in match:
                    continue
                for m in match["match"]:
                    match_dict = {}
                    relname = m["reln"][0]["name"]
                    for node in m["node"]:
                        match_dict[node["name"]] = (
                            i,
                            node["matchIndex"] - 1,
                        )  # (sentence_idx, word_idx)
                    if relname == "adj":
                        description[match_dict["adjective"]] = match_dict["noun"]
                    elif relname == "compound":
                        compound[match_dict["second_noun"]] = match_dict["first_noun"]
                    elif relname == "conj":
                        conj[match_dict["adjective"]] = match_dict["second_adjective"]
                    elif relname == "xcomp":
                        description[match_dict["adjective"]] = match_dict["noun"]
                    elif relname == "not":
                        negations.add(match_dict["adjective"])

        for adj in description:
            negation = adj in negations
            pairs.add(
                Pair(
                    get_word(doc, adj),
                    get_compound(doc, compound, description[adj]),
                    negation,
                )
            )
        for adj in conj:
            if conj[adj] not in description:
                continue
            negation = adj in negations or conj[adj] in negations
            pairs.add(
                Pair(
                    get_word(doc, adj),
                    get_compound(doc, compound, description[conj[adj]]),
                    negation,
                )
            )
        return pairs

    with Semgrex() as sem:
        outputs = list(map(process_doc, docs))

    return outputs


def test():
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")
    test_batch = [
        # "The red dress is beautiful. The blue shoes are ugly.",
        # "The red dress is beautiful.",
        # "I love the blue shoes and the green hat.",
        # "She wore a stylish black jacket with a white shirt.",
        # "Lily found herself confused and sad.",
        # "This is the wonderful, talented Michelle.",
        # "Sam, my sad brother, arrived.",
        # "Sam, who was sad, arrived.",
        # "Lauren was wonderful.",
        # "Lauren was a wonderful student.",
        "The horse rider was lovely.",
        # "Lauren and Bob are happy.",
        # "I feel happy.",
        # "I'm so pretty and witty and bright.",
        # "I am not happy.",
    ]

    outputs = process_batch(nlp, test_batch)
    for i, output in enumerate(outputs):
        print(test_batch[i])
        for pair in output:
            negation = " (negated)" if pair.negated else ""
            print(
                f"  Adjective: {pair.adjective.text} ({pair.adjective.start_idx}-{pair.adjective.end_idx}), "
                f"Noun: {pair.noun.text} ({pair.noun.start_idx}-{pair.noun.end_idx}){negation}"
            )
        print()


def main(fashion_mention_file: Path, output_file: Path):
    mentions = pd.read_csv(fashion_mention_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")
    for batch in tqdm(
        itertools.batched(mentions.itertuples(), 128), total=(len(mentions) // 128) + 1
    ):
        batch_texts = [str(mention.sentence) for mention in batch]
        outputs = process_batch(nlp, batch_texts)

        for mention, pairs in zip(batch, outputs):
            if not pairs:
                continue
            pairs = list(pairs)
            span_starts, span_ends = zip(
                *[(pair.noun.start_idx, pair.noun.end_idx) for pair in pairs]
            )
            overlap = get_overlap_rows_single(
                (mention.start_idx, mention.end_idx),
                (span_starts, span_ends),
                partial=True,
            )
            if not overlap:
                continue
            for idx in overlap:
                pair = pairs[idx]
                results.append(
                    {
                        "adjective": pair.adjective.text,
                        "adjective_start_idx": pair.adjective.start_idx,
                        "adjective_end_idx": pair.adjective.end_idx,
                        "noun": pair.noun.text,
                        "noun_start_idx": pair.noun.start_idx,
                        "noun_end_idx": pair.noun.end_idx,
                        "negated": pair.negated,
                        "sentence": mention.sentence,
                        "sentence_start_idx": mention.sentence_start_idx,
                        "sentence_end_idx": mention.sentence_end_idx,
                        "mention_start_idx": mention.start_idx,
                        "mention_end_idx": mention.end_idx,
                        "filename": mention.filename,
                    }
                )

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fashion_mention_file",
        type=Path,
        help="Path to the file containing fashion mentions.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to the output file where results will be saved.",
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Run in test mode with sample data."
    )

    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args.fashion_mention_file, args.output_file)
