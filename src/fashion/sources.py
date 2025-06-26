"""
An interface to load corpora of text.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fashion.paths import CHICAGO_PATH, CONTEMP_LITBANK_PATH, DATA_DIR, LITBANK_PATH


@dataclass
class Text:
    filename: str
    text_id: str
    text: str
    ref: str  # path to the file text


class Source:
    name: str

    def __init__(self):
        pass

    def iter_texts(self) -> Iterator[Text]:
        raise NotImplementedError

    def load_text(self, book_id: str) -> Text:
        raise NotImplementedError

    def iter_book_ids(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TextDir(Source):
    def __init__(self, path: Path):
        self.path = path
        self.files = list(self.path.glob("*.txt"))

    def iter_texts(self):
        for file in self.files:
            with open(file, "r", encoding="utf-8") as f:
                yield Text(file.name, file.stem, f.read(), str(file))

    def load_text(self, book_id: str) -> Text:
        book_path = self.path / f"{book_id}.txt"
        with open(book_path, "r", encoding="utf-8") as f:
            return Text(book_id, book_id, f.read(), str(book_path))

    def iter_book_ids(self):
        for file in self.files:
            yield file.stem

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


class Debug(TextDir):
    name = "Debug"

    def __init__(self, path: Path = LITBANK_PATH):
        super().__init__(path)
        self.files = self.files[:5]


class ContempLitBank(TextDir):
    name = "Contemp"

    def __init__(self, path: Path = CONTEMP_LITBANK_PATH):
        super().__init__(path)


class HathiSource(Source):
    filenames: Path
    hathi_path: Path

    def __init__(self):
        self.load_hathi_ids(self.filenames, self.hathi_path)

    def load_hathi_ids(self, path: Path, hathi_path: Path):
        self.filepaths = [
            hathi_path / hathi_file.strip()
            for hathi_file in path.read_text().splitlines()
        ]
        self.hathi_ids = [filepath.stem for filepath in self.filepaths]
        self.id_lookup = {
            hathi_id: filepath
            for hathi_id, filepath in zip(self.hathi_ids, self.filepaths)
        }
        # check that all files exist
        missing = []
        for filepath in self.filepaths:
            if not filepath.exists():
                missing.append(filepath)
        if missing:
            print(f"{len(missing)}/{len(self.filepaths)} files not found")
            # discard any missing files
            self.filepaths = [
                filepath for filepath in self.filepaths if filepath not in missing
            ]
            self.hathi_ids = [
                hathi_id
                for hathi_id, filepath in self.id_lookup.items()
                if filepath not in missing
            ]
            self.id_lookup = {
                hathi_id: filepath
                for hathi_id, filepath in self.id_lookup.items()
                if filepath not in missing
            }

    def iter_texts(self):
        for filepath in self.filepaths:
            filepath = Path(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                yield Text(filepath.name, filepath.stem, f.read(), str(filepath))

    def load_text(self, book_id: str) -> Text:
        filepath = self.id_lookup[book_id]
        with open(filepath, "r", encoding="utf-8") as f:
            return Text(filepath.name, book_id, f.read(), str(filepath))

    def iter_book_ids(self):
        yield from list(self.id_lookup.keys())

    def __len__(self):
        return len(self.filepaths)


class HathiManual(HathiSource):
    name = "hathi_manual"
    filenames = DATA_DIR / "hathitrust/manual.txt"
    hathi_path = DATA_DIR / "hathitrust/stripped/"


def add_source_argument(parser: argparse.ArgumentParser):
    """
    --source: Source of the data to process (chicago, litbank, contemp, hathitrust, debug)
    """
    sources = [Chicago, LitBank, ContempLitBank, HathiManual, Debug]
    source_map = {source.name.lower(): source for source in sources}

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
        "--source",
        action=SourceAction,
        default="litbank",
        choices=source_map.keys(),
        help="Source of the data to process",
    )
