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

### vLLM environment

The `vllm` pixi environment is isolated from `default` (it sets `no-default-feature`)
because current vLLM pins `torch==2.11` + CUDA 13, which conflicts with the default
env's torch 2.5.1 / CUDA 12.4. Torch and the CUDA runtime come from the vLLM wheel,
so nothing in `default` is affected. Verified working with vLLM 0.24.0 on an L40S.

The editable `fashion` package is installed into this env too (so `import fashion`
works), but the default feature is intentionally *not* included: its conda pins
(`numpy==2.0.0`, `cuda` 12.4, `transformers`) would cap the resolver and drag vLLM
back to 0.8.5, which has no Qwen3.5 support. For the same reason `vllm` is pinned
`>=0.24.0` in `pyproject.toml`. Conda-only research tools (spacy, nltk, llm, jq)
are not available in this env — use the `default` env for those.

The env also installs a **CUDA 13 toolkit** (`cuda-toolkit`, matching torch's
cu130) and sets `CUDA_HOME`. This is required for runtime CUDA JIT compilation
(FlashInfer, torch `cpp_extension`, custom kernels). Without it, `nvcc` resolves
to the `default` env's CUDA 12.4 (leaked onto PATH via `.envrc`), which links
against `libcudart.so.12` and produces `ninja` build errors. The vllm feature
uses its own channel list (`nvidia`, `conda-forge`, `pytorch` — note the
`nvidia/label/cuda-12.4.0` workspace channel is omitted) so the CUDA 13 packages
resolve without being pinned back to 12.4.

```
# smoke test (loads Qwen/Qwen3.5-9B and generates)
pixi run -e vllm python scratch/vllm_qwen35_smoketest.py

# OpenAI-compatible server
pixi run -e vllm vllm serve Qwen/Qwen3.5-9B --max-model-len 8192 --reasoning-parser qwen3
```

> If numpy in the vllm env ever fails to import with "you should not try to
> import numpy from its source directory", pixi's uv cache
> (`~/.cache/rattler/cache/uv-cache`) has a corrupted numpy archive. Fix:
> `find ~/.cache/rattler/cache/uv-cache -path '*numpy*/__config__.py'`, remove
> any archive whose `__config__.py` size doesn't match the installed wheel's
> RECORD, delete the env's `site-packages/numpy*`, and re-run `pixi install -e vllm`.


## Pipeline:

This repository contains a series of scripts which:

1. Identify fashion mentions using a list of keywords (`process_texts.py`)
2. Apply a deberta model to filter out some false positives (`filter_texts.py`)
3. Run booknlp on the original texts (`characters.py`)
4. Identify the entities present in passages with fashion mentions (`character_fashion_cooc.py`)
5. Apply a deberta model to identify which entity in a passage is in possession of the mentioned article of clothing (`wearing/inference.py`)
6. Extract sentences which mention entities that have been linked to clothing (`extract_entity_mentions.py`)
7. Use dependency parsing to find adjectives which describe a particular noun (`extract_adjectives.py`)

## Visualization

Interactive D3/React visualizations live in `visualize/`. To run:

```
cd visualize
npm install
npm run dev
```

Pages:
- **Home** — overview and navigation
- **Gender Classifier** — per-decade logistic regression coefficients showing which fashion terms index male vs. female characters; supports both top-terms-by-decade bar chart and per-term trajectory over time
- **Adjective Categories** — WordNet semantic category frequency bar chart with count filter

Data files are loaded from `public/data/` (symlinked to `data/analysis/`). To add a new visualization, create a page in `src/pages/`, add a route in `src/App.jsx`, and add a nav entry in `src/components/Nav.jsx`.
