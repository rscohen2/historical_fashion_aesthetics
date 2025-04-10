# historical_fashion_aesthetics
Trying to extract and study historical fashion trends from a corpus of novels


Working on this branch:
1. Install `pixi`, a package and environment manager: [instructions](https://pixi.sh/latest/).
2. Clone this repository
3. Place `ChicagoCorpus` inside `data/` (such that you have `data/ChicagoCorpus/CHICAGO_CORPUS` and `data/ChicagoCorpus/CLEAN_TEXTS`)
3. Finally:
```
cd historical_fashion_aesthetics
pixi install
pixi shell
```

Python code belongs in `src/fashion`. You can run e.g. `process_texts.py` by calling `python -m fashion.process_texts`.