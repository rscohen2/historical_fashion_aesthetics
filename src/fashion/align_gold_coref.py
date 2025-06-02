"""
Convert gold coreference annotations to booknlp format to use in character_fashion_cooc.py


python -m fashion.align_gold_coref --gold_coref=/mnt/data1/corpora/litbank/coref/tsv/ --processed_text_dir=/mnt/data1/corpora/litbank/coref/tsv/ --original_text_dir=/mnt/data1/corpora/litbank/original/ --output_dir=data/gold_coref/
"""

# book.entities file:
# COREF	start_token	end_token	prop	cat	text

# book.tokens file:
# paragraph_ID	sentence_ID	token_ID_within_sentence	token_ID_within_document	word	lemma	byte_onset	byte_offset	POS_tag	fine_POS_tag	dependency_relation	syntactic_head_ID	event

import argparse
from collections import defaultdict
from pathlib import Path


def main(
    gold_coref_files: list[Path],
    processed_text_dir: Path,
    original_text_dir: Path,
    output_dir: Path,
):
    for gold_coref_file in gold_coref_files:
        book_id = "_".join(gold_coref_file.stem.split("_")[:-1])

        booknlp_entities_file = f"{book_id}.entities"
        booknlp_tokens_file = f"{book_id}.tokens"
        book_output_dir = output_dir / book_id
        book_output_dir.mkdir(parents=True, exist_ok=True)

        with open(gold_coref_file, "r") as f:
            lines = f.readlines()

        token_ids = []
        # create tokens file
        with open(book_output_dir / booknlp_tokens_file, "w") as f:
            f.write(
                "paragraph_ID\tsentence_ID\ttoken_ID_within_sentence\ttoken_ID_within_document\tword\tlemma\tbyte_onset\tbyte_offset\tPOS_tag\tfine_POS_tag\tdependency_relation\tsyntactic_head_ID\tevent\n"
            )
            processed_file = processed_text_dir / f"{book_id}_brat.txt"
            original_file = original_text_dir / f"{book_id}.txt"
            with open(processed_file, "r") as p:
                processed_lines = p.readlines()

            with open(original_file, "r") as o:
                original_text = o.read()

            cursor = 0
            token_id_within_document = 0
            for sentence_id, line in enumerate(processed_lines):
                tokens = line.strip().split()
                line_token_ids = []
                for token_id_within_sentence, token in enumerate(tokens):
                    # advance cursor to the start of this token in the original text
                    while (
                        cursor < len(original_text)
                        and original_text[cursor] != token[0]
                    ):
                        cursor += 1
                    if cursor >= len(original_text):
                        raise ValueError(
                            f"Cursor exceeded original text length for book {book_id} at line {sentence_id}, token {token_id_within_sentence}."
                        )
                    start_idx = cursor
                    cursor += len(token)
                    end_idx = cursor
                    if original_text[cursor - 1] != token[-1]:
                        raise ValueError(
                            f"Token mismatch at book {book_id}, line {sentence_id}, token {token_id_within_sentence}. Expected '{token[-1]}', found '{original_text[cursor - 1]}'."
                        )
                    # write to tokens file
                    f.write(
                        f"{0}\t{sentence_id}\t{token_id_within_sentence}\t{token_id_within_document}\t{token}\t{token.lower()}\t{start_idx}\t{end_idx}\t_\t_\t_\t_\t_\n"
                    )
                    line_token_ids.append(token_id_within_document)
                    token_id_within_document += 1
                token_ids.append(line_token_ids)

        corefs = defaultdict(list)
        mentions = {}
        for line in lines:
            if line.startswith("MENTION"):
                parts = line.strip().split("\t")
                mention_id = parts[1]
                start_sentence = int(parts[2])
                start_sentence_idx = int(parts[3])
                end_sentence = int(parts[4])
                end_sentence_idx = int(parts[5])
                text = parts[6]
                mentions[mention_id] = {
                    "mention_id": mention_id,
                    "start_sentence": start_sentence,
                    "start_sentence_idx": start_sentence_idx,
                    "end_sentence": end_sentence,
                    "end_sentence_idx": end_sentence_idx,
                    "text": text,
                }
            elif line.startswith("COREF"):
                parts = line.strip().split("\t")
                mention_id = parts[1]
                coref_head = parts[2]
                corefs[coref_head].append(mention_id)

        with open(book_output_dir / booknlp_entities_file, "w") as f:
            f.write("COREF\tstart_token\tend_token\tprop\tcat\ttext\n")
            for coref_head, mention_ids in corefs.items():
                for mention_id in mention_ids:
                    mention = mentions[mention_id]
                    start_token = token_ids[mention["start_sentence"]][
                        mention["start_sentence_idx"]
                    ]
                    end_token = token_ids[mention["end_sentence"]][
                        mention["end_sentence_idx"]
                    ]
                    f.write(
                        f"{coref_head}\t{start_token}\t{end_token}\t_\t_\t{mention['text']}\n"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_coref",
        type=Path,
        required=True,
        help="Path to directory containing the gold coreference annotations files.",
    )
    parser.add_argument(
        "--processed_text_dir",
        type=Path,
        required=True,
        help="Path to directory containing the processed text files for the coref annotations.",
    )
    parser.add_argument(
        "--original_text_dir",
        type=Path,
        required=True,
        help="Path to directory containing the text files for the books.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Directory to write the booknlp formatted files.",
    )
    args = parser.parse_args()

    print(f"Using gold coref directory: {args.gold_coref}")
    files = list(args.gold_coref.glob("*_brat.ann"))
    print(f"Found {len(files)} gold coref files.")

    main(files, args.processed_text_dir, args.original_text_dir, args.output_dir)
