## Analysis for fashion project

For the analysis scripts in this folder, we always take as input one of the following files:
- `data/final_for_analysis.parquet` (for true results)
- `data/sample_for_analysis.parquet` (for development and debugging dataflow)
Select this with a `--debug, -d` flag.

We always output results and visualizations to `data/analysis/<experiment>`,
where `<experiment>` can be the filename by default.

Unless specified otherwise, we always use `gender_dep_pron` for the gender values.

Utility functions that are shared across files are in the `analysis/utils.py` file.

## Scripts

### `logodds.py`

Computes corpus-level (no temporal split) Monroe et al. (2008) Dirichlet-prior log-odds
of each character adjective being associated with each fashion term. Character adjectives
are the "words" and fashion terms are the "documents". Counts unique (book_id, character_id)
pairs per (term, adjective) combination to avoid inflating counts from repeated mentions.
Also collects example sentences for each top (term, adjective) pair.

```
python -m fashion.analysis.logodds [--debug] [--top-terms N] [--top-adj N]
                                    [--min-count N] [--max-examples N]
                                    [--col-doc COLUMN] [--col-word COLUMN]
```

Output: `data/analysis/logodds/logodds.json`

JSON fields: `terms` (list of fashion terms), `logodds` (list of records with
`term`, `adjective`, `logodds`, `sigma`, `score`, `count`), `examples`
(dict `{term: {adjective: [sentences]}}`).

### `llm_showtell.py`

**Runtime**: To run on the debug set of 100,000 rows on 2 L40s for Qwen3-8B, it
took 5:55:49.

There are 4,039,248 rows in the full `final_for_analysis.parquet` dataset, so
that would take roughly 10 days all in (40 x 6 = 240 hours). But I think this
does make it seem like running on one of the UIUC servers might actually be
tractable; if we have an H100 node, I suspect we could finish running quite
quickly...

Uses an LLM to generate adjectival descriptions of each character mention, given
the surrounding passage (with the character span wrapped in `**asterisks**`).
Inference runs against a **separately launched** vLLM OpenAI-compatible server;
the script fires requests concurrently and relies on vLLM's continuous batching
for throughput.

The server is constrained (via `response_format` guided decoding) to emit
`{"adjectives": [{"word", "reasoning"}, ...]}`, which the client parses before
storing.

Prompt-building and inference are pipelined: a process pool builds each book's
prompts in parallel (loading the text and Punkt-tokenizing it once per book),
and prompts flow onto a bounded queue that `--concurrency` async consumers drain
as soon as each book is ready — so inference on the first ready book overlaps
with tokenizing the rest instead of waiting for every prompt up front.

First launch the server (in the `vllm` pixi env), e.g.:

```
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --max-model-len 8192 --port 8000
```

Then run:

```
python -m fashion.analysis.llm_showtell [--debug] [--base-url URL] [--model NAME]
                                        [--concurrency N] [--loader-workers N]
                                        [--max-tokens N] [--temperature T]
                                        [--no-thinking] [--thinking-budget N]
                                        [--max-retries N]
```

`--thinking-budget` caps reasoning at N tokens before the model is forced to
answer (default 512, `-1` for unlimited); it requires the server to be launched
with `--reasoning-parser qwen3` and is ignored under `--no-thinking`.

Output: `data/analysis/llm_showtell/descriptions.parquet` (written at the end),
plus `descriptions.jsonl` which is appended to as each request completes (a
progress bar tracks completion) so partial results survive an interrupted run.

Columns: `row_id`, `prompt`, `reasoning` (Qwen3 thinking trace, `None` if
disabled), `adjectives` (parsed list of `{word, reasoning}` objects, `None` if
the reply could not be parsed or the request failed), `response` (raw model text,
kept for debugging parse failures; `None` if the request failed after retries).

### `black_fashion.py`

Extracts all passages where a fashion item is described as "black" into a TSV,
plus a stratified sample of 50 passages per decade.

Output: `data/analysis/black_fashion/`

### `gender_classifier.py`

Trains a per-decade logistic regression to predict character gender (male/female)
from fashion terms. The coefficient for each term reflects how strongly it
indexes male (positive) or female (negative) gender. Must be run before
`colors.py`.

```
python -m fashion.analysis.gender_classifier [--debug] [--min-df N]
```

Output: `data/analysis/gender_classifier/coefficients.parquet`
        `data/analysis/gender_classifier/coefficients.tsv`

Columns: `decade`, `term`, `coefficient` (positive = male-indexing).

### `colors.py`

Plots color adjective prevalence over time for gendered fashion mentions,
disaggregated by fashion item. For each gender, uses the classifier coefficients
from `gender_classifier.py` to select the top N gender-indexing fashion terms
(those with the highest coefficient magnitude in each decade), then produces a
faceted Altair chart (one panel per item) showing the relative prevalence of each
color adjective across decades, with binomial 95% confidence intervals.

Requires `gender_classifier.py` to have been run first.

```
python -m fashion.analysis.colors [--debug] [--gender {male,female,both}]
                                   [--top-terms N] [--top-colors N]
                                   [--group-bw]
                                   [--classifier-output PATH]
```

Output: `data/analysis/colors/color_prevalence_{gender}.html`

With `--group-bw`, colors are collapsed into "black", "white", and "other" instead of
top-N individual colors. The color scale is shared across facets. Output files are named
`color_prevalence_{gender}_bw.html`.

### `gp_logodds.py`

For the top N fashion terms (by frequency), computes per-decade log odds (Monroe et al. 2008
Dirichlet-prior) of each character adjective being associated with that term, then fits a
Gaussian process regression (RBF + WhiteKernel) to the temporal log-odds signal. Produces an
interactive Altair chart stacking one panel per term; each panel shows the GP posterior mean,
95% credible band, and the raw per-decade observations. Click a legend entry to highlight
one adjective.

```
python -m fashion.analysis.gp_logodds [--debug] [--top-terms N] [--top-adj N] [--min-df N]
```

Output: `data/analysis/gp_logodds/gp_logodds.html`

The top adjectives shown per term are selected by maximum absolute log-odds across all decades.
