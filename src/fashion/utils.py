from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

CHICAGO_PATH = DATA_DIR / "ChicagoCorpus"
LITBANK_PATH = Path("/mnt/data1/corpora/litbank/original/")
CONTEMP_LITBANK_PATH = Path(
    "/mnt/data1/corpora/contemporary_litbank/english/stripped_paratext/"
)
