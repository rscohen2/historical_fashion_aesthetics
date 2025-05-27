from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHICAGO_PATH = DATA_DIR / "ChicagoCorpus"
