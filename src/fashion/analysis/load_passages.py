"""Extract the passage around each character mention and write them to parquet.

This is the *loading* half of ``llm_showtell``: it does the slow, CPU/disk-bound
work of opening each book, Punkt-tokenizing it once, and slicing out the passage
of sentences enclosing every mention. The result is a compact parquet of passage
text plus the byte offsets needed to build a prompt, which the inference script
(``llm_showtell.py``) streams through a vLLM server.

Splitting the two stages means tokenization runs once and its output is cached on
disk, so inference can be re-run (different model, prompt, sampling) without
re-tokenizing every book.
"""

import os
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer
from tqdm import tqdm

from fashion.paths import DATA_DIR
from fashion.sources import HathiAll

# Punkt is unsupervised and rule-based, so it is much faster than a neural
# sentence segmenter (spacy/stanza) while being accurate enough for trimming
# context. Instantiate once and reuse across calls.
_SENTENCE_TOKENIZER = PunktSentenceTokenizer()

# Cached per worker process: HathiAll() stats every book listed in all.txt on
# construction, so we build it lazily once and reuse it for all books a worker
# handles instead of paying that cost per book.
_SOURCE = None


def _get_source():
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = HathiAll()
    return _SOURCE


def passages_path(debug):
    """Path of the passages parquet for the given ``--debug`` flag.

    Kept as a function so the inference script can import it and read exactly the
    file this script wrote.
    """
    name = "passages_debug.parquet" if debug else "passages.parquet"
    return DATA_DIR / "analysis" / "llm_showtell" / name


@dataclass
class Passage:
    text: str
    start_idx: int
    end_idx: int


def find_enclosing_sentences(
    text, spans, start_idx, end_idx, max_sentences=1
) -> Passage | None:  #
    """
    Return the passage of ``text`` covering the sentence(s) that overlap
    ``[start_idx, end_idx)`` plus up to ``max_sentences`` sentences on each side.

    The result is a byte-accurate slice of ``text`` (from the start offset of
    the first kept sentence to the end offset of the last), so all original
    whitespace and character offsets are preserved. Returns ``None`` if no
    sentence overlaps the span.
    """
    # Punkt spans are sorted and non-overlapping, so the sentences overlapping
    # [start_idx, end_idx) form a contiguous range. Both endpoints of that range
    # are found by binary search instead of scanning every span:
    #   first = first sentence whose end is past start_idx (ends are ascending)
    #   last  = last sentence whose start is before end_idx (starts are ascending)
    first = bisect_right(spans, start_idx, key=lambda span: span[1])
    last = bisect_left(spans, end_idx, key=lambda span: span[0]) - 1
    if first > last:  # empty text, or the span falls between/outside sentences
        return None

    lo = max(0, first - max_sentences)
    hi = min(len(spans) - 1, last + max_sentences)
    passage_start_idx = spans[lo][0]
    passage_end_idx = spans[hi][1]
    return Passage(
        text=text[passage_start_idx:passage_end_idx],
        start_idx=passage_start_idx,
        end_idx=passage_end_idx,
    )


def load_book_passages(book_id, book_df):
    """Load one book, tokenize it once, and slice out each mention's passage.

    Runs in a worker process (see ``main``): loading the text (disk I/O) and Punkt
    tokenization (CPU-bound, holds the GIL) are the slow parts, and doing them per
    book lets the process pool parallelize across books. Returns a list of records
    holding the passage text and the book-level byte offsets needed to build a
    prompt; rows whose span has no enclosing sentence are dropped.
    """
    text = _get_source().load_text(book_id).text
    spans = list(_SENTENCE_TOKENIZER.span_tokenize(text))
    records = []
    for row_id, row in book_df.iterrows():
        passage = find_enclosing_sentences(
            text, spans, row.sentence_start_idx, row.sentence_end_idx, max_sentences=3
        )
        if passage is None:
            print(f"Row ID: {row_id} - No passage found.")
            continue
        # ``start_idx``/``end_idx`` are offsets into the sentence string, so lift
        # them to book-level offsets (matching character_*_idx) by adding the
        # sentence's start. Everything stored here is a book-level byte offset.
        records.append(
            {
                "row_id": row_id,
                "book_id": book_id,
                "passage": passage.text,
                "passage_start_idx": int(passage.start_idx),
                "passage_end_idx": int(passage.end_idx),
                "character_start_idx": int(row.character_start_idx),
                "character_end_idx": int(row.character_end_idx),
                "fashion_start_idx": int(row.sentence_start_idx + row.start_idx),
                "fashion_end_idx": int(row.sentence_start_idx + row.end_idx),
            }
        )
    return records


def main(args):
    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df = pd.read_parquet(input_path).reset_index()
    groups = [(book, rows) for book, rows in df.groupby("book_id")]

    output_path = passages_path(args.debug)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting passages for {len(df)} rows ({args.workers} workers)...")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(load_book_passages, book, book_df) for book, book_df in groups
        ]
        for future in tqdm(futures, desc="loading books"):
            try:
                records.extend(future.result())
            except Exception as error:  # noqa: BLE001 - skip a book we can't load
                print(f"Failed to load passages for a book: {error}")

    passages = pd.DataFrame(records)
    passages.to_parquet(output_path, index=False)
    print(f"Wrote {len(passages)} passages to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fashion Passage Loader")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes that extract passages (tokenize books) in parallel.",
    )
    args = parser.parse_args()

    main(args)
