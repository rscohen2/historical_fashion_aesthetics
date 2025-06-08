"""
An interface to load corpora of text.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fashion.utils import CHICAGO_PATH, CONTEMP_LITBANK_PATH, LITBANK_PATH


@dataclass
class Text:
    filename: str
    text_id: str
    text: str
    ref: str | None  # reference to original filepath --- for debugging


class Source:
    name: str | None = None

    def __init__(self):
        pass

    def iter_texts(self) -> Iterator[Text]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TextDir(Source):
    def __init__(self, path: Path):
        self.path = path
        self.files = list(self.path.glob("*.txt"))

    def iter_texts(self):
        for file in self.files:
            with open(file, "r") as f:
                yield Text(file.name, file.stem, f.read(), str(file))

    def load_text(self, book_id: str) -> Text:
        book_path = self.path / f"{book_id}.txt"
        with open(book_path, "r") as f:
            return Text(book_id, book_id, f.read(), str(book_path))

    def __len__(self):
        return len(self.files)


class Chicago(TextDir):
    name = "Chicago"

    def __init__(self, path: Path = CHICAGO_PATH / "CLEAN_TEXTS"):
        super().__init__(path)


class LitBank(TextDir):
    name = "LitBank"

    def __init__(self, path: Path = LITBANK_PATH):
        super().__init__(path)


class ContempLitBank(TextDir):
    name = "ContempLitBank"

    def __init__(self, path: Path = CONTEMP_LITBANK_PATH):
        super().__init__(path)


def add_source_argument(parser: argparse.ArgumentParser):
    source_map = {
        "chicago": Chicago,
        "litbank": LitBank,
        "contemp": ContempLitBank,
    }

    def load_source(source_name: str) -> Source:
        if source_name not in source_map:
            raise ValueError(f"Invalid source name: {source_name}")
        return source_map[source_name]()

    # Add a post-processing action to convert the string to a Source object
    class SourceAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not isinstance(values, str):
                raise ValueError(f"Expected string, got {type(values)}")
            setattr(namespace, self.dest, load_source(values))

    parser.add_argument(
        "--data_source",
        action=SourceAction,
        default="litbank",
        choices=source_map.keys(),
        help="Source of the data to process",
    )
