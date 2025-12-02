"""
Extract pronouns for the fashion contexts using stanza coref.

Input is the wearing/coocs ndjson files.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import stanza
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed
from fashion.sources import Source, add_source_argument


gender_pronouns = {
    "male": ["he", "him", "his", "himself", "he's", "he'll", "he'd"],
    "female": ["she", "her", "hers", "herself", "she's", "she'll", "she'd"],
    "nonbinary": [
        "they",
        "them",
        "their",
        "themselves",
        "they're",
        "they'll",
        "they'd",
        "they've",
    ],
    "other": [
        "xe",
        "xem",
        "xyr",
        "xir",
        "ze",
        "zem",
        "zir",
        "hir",
        "xe's",
        "xe'll",
        "xe'd",
    ],
}

all_pronouns = set(sum(gender_pronouns.values(), []))
pronoun_map = {
    pronoun: gender
    for gender, pronouns in gender_pronouns.items()
    for pronoun in pronouns
}


def get_possessive_pronoun(doc, fashion_start_idx: int, fashion_end_idx: int) -> str | None:
    """
    Find the possessive pronoun for a fashion mention using dependency parsing.

    Args:
        doc: Stanza Document
        fashion_start_idx: Character start index of the fashion mention in the excerpt
        fashion_end_idx: Character end index of the fashion mention in the excerpt

    Returns:
        The possessive pronoun text if found, otherwise None
    """
    # Find the word(s) that correspond to the fashion mention
    fashion_words = []
    for sent in doc.sentences:
        for word in sent.words:
            # Check if this word overlaps with the fashion mention
            if (word.start_char < fashion_end_idx and word.end_char > fashion_start_idx):
                fashion_words.append((sent, word))

    if not fashion_words:
        return None

    # Look for possessive relationships
    # Common patterns: nmod:poss (possessive modifier)
    for sent, fashion_word in fashion_words:
        for word in sent.words:
            # Check if this word has a possessive relationship with the fashion word
            if word.head == fashion_word.id and word.deprel == "nmod:poss":
                # This word is the possessive modifier of the fashion word
                if word.upos == "PRON":
                    return word.text
            # Also check if the fashion word is a compound and has possessive
            if word.deprel == "nmod:poss":
                head_word = sent.words[word.head - 1] if word.head > 0 else None
                if head_word:
                    # Check if the head is part of the fashion mention
                    if (head_word.start_char < fashion_end_idx and
                        head_word.end_char > fashion_start_idx):
                        if word.upos == "PRON":
                            return word.text

    return None


def process_cooc_file_possessive(
    cooc_file: Path,
    output_dir: Path,
    source: Source,
    nlp: stanza.Pipeline,
):
    """Process cooc file to extract possessive pronouns."""
    with open(cooc_file, "r") as f:
        results = []
        for line in tqdm(f, desc=f"Processing {cooc_file.name}"):
            cooc = json.loads(line)
            excerpt_text = cooc["excerpt_text"]

            # Parse the excerpt with stanza
            doc = nlp(excerpt_text)

            # Calculate the fashion mention position relative to excerpt
            fashion_start_in_excerpt = cooc["fashion_start_idx"] - cooc["excerpt_start"]
            fashion_end_in_excerpt = cooc["fashion_end_idx"] - cooc["excerpt_start"]

            # Get the possessive pronoun
            possessive = get_possessive_pronoun(
                doc, fashion_start_in_excerpt, fashion_end_in_excerpt
            )

            results.append(
                {
                    "book_id": cooc["book_id"],
                    "mention_id": cooc["mention_id"],
                    "possessive_pronoun": possessive,
                }
            )

        return results


def process_cooc_file(
    cooc_file: Path,
    output_dir: Path,
    source: Source,
):
    with open(cooc_file, "r") as f:
        results = []
        for line in tqdm(f, desc=f"Processing {cooc_file.name}"):
            cooc = json.loads(line)
            wearer_genders = defaultdict(list)
            for character in cooc["characters"]:
                if (
                    character.get("wearing", False)
                    and character["text"].lower() in all_pronouns
                ):
                    wearer_genders[pronoun_map[character["text"].lower()]].append(
                        character["text"].lower()
                    )
            results.append(
                {
                    "book_id": cooc["book_id"],
                    "mention_id": cooc["mention_id"],
                    "gender": wearer_genders,
                }
            )

        return results


def main(
    source: Source,
    wearing_dir: Path,
    output_dir: Path,
    num_processes: int,
    concurrent_processes: int,
    debug: bool,
    get_possessive: bool = False,
):
    def process(subset: list[Path]):
        # Initialize stanza pipeline if using possessive mode
        nlp = None
        if get_possessive:
            nlp = stanza.Pipeline(
                "en",
                processors="tokenize,pos,lemma,depparse",
            )

        for cooc_file in subset:
            if get_possessive:
                if nlp is None:
                    raise ValueError("Stanza pipeline is not initialized")
                results = process_cooc_file_possessive(cooc_file, output_dir, source, nlp)
            else:
                results = process_cooc_file(cooc_file, output_dir, source)
            output_file = output_dir / f"{cooc_file.stem}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                for result in results:
                    f.write(f"{json.dumps(result)}\n")

    files = sorted(list(wearing_dir.glob("*.ndjson")))
    if debug:
        files = files[:10]

    extra_args = [
        "--source",
        str(source.name.lower()),
        "--wearing_dir",
        str(wearing_dir),
        "--output_dir",
        str(output_dir),
    ]
    if get_possessive:
        extra_args.append("--get-possessive")

    run_distributed(
        process,
        files,
        script_path=__file__,
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=extra_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_argument(parser)
    add_distributed_args(parser)
    parser.add_argument("--wearing_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--get-possessive",
        action="store_true",
        help="Extract possessive pronouns for fashion mentions using dependency parsing instead of counting wearer genders",
    )
    args = parser.parse_args()

    main(
        args.source,
        args.wearing_dir,
        args.output_dir,
        args.num_processes,
        args.concurrent_processes,
        args.debug,
        args.get_possessive,
    )
