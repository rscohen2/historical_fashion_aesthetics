"""Book metadata shared by every page in ``visualize/``.

Example sentences in the visualization data carry only their ``book_id``
(``bk``). The pages look up its title, author, and year in this one file, so
none of the per-page exports repeat them.

Titles and authors come from the Hathi metadata (``hathimeta/titlemeta.tsv``,
keyed by ``docid``; a ``book_id`` is the docid plus ``.clean``). The year is the
book's ``inferreddate`` in the analysis parquet. Every book in that parquet is
listed, whether or not Hathi has a title for it.

Output: data/analysis/books/books.json
    {book_id: [title, author, year]}, title and author null when unknown
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from fashion.paths import DATA_DIR

EXPERIMENT = "books"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
TITLEMETA_PATH = DATA_DIR / "hathimeta" / "titlemeta.tsv"


def clean_text(text) -> str | None:
    """Collapse whitespace and fix ligatures; None for missing or empty text."""
    if not isinstance(text, str):
        return None
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text).strip() or None


def load_titles() -> dict[str, tuple[str | None, str | None]]:
    """Hathi (title, author) keyed by ``book_id``; empty if unavailable."""
    if not TITLEMETA_PATH.exists():
        print(f"No title metadata at {TITLEMETA_PATH}; books will have no titles")
        return {}
    meta = pd.read_csv(TITLEMETA_PATH, sep="\t", usecols=["docid", "title", "author"])
    return {
        f"{docid}.clean": (clean_text(title), clean_text(author))
        for docid, title, author in meta.itertuples(index=False)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input parquet (overrides --debug).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )
    print(f"Loading {input_path}")
    df = pd.read_parquet(input_path, columns=["inferreddate"]).reset_index()
    years = df.dropna(subset=["inferreddate"]).groupby("book_id").inferreddate.first()

    titles = load_titles()
    books = {
        book_id: [*titles.get(book_id, (None, None)), int(year)]
        for book_id, year in years.items()
    }
    n_titled = sum(book[0] is not None for book in books.values())
    print(f"{len(books):,} books, {n_titled:,} with Hathi titles")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "books.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
