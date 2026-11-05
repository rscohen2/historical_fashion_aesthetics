## Analysis for fashion project

For the analysis scripts in this folder, we always take as input one of the following files:
- `data/final_for_analysis.parquet` (for true results)
- `data/sample_for_analysis.parquet` (for development and debugging dataflow)
Select this with a `--debug, -d` flag.

We always output results to `data/analysis/<experiment>`, where `<experiment>` can be
the filename by default. Figures (PDF/PNG and HTML charts) go to
`data/analysis/figures/<experiment>` instead, so they're all in one place.

Unless specified otherwise, we always use `gender_dep_pron` for the gender values.

Utility functions that are shared across files are in the `analysis/utils.py` file
(e.g. `monroe_logodds`, the Dirichlet-prior log-odds used by `logodds.py`,
`gp_logodds.py` and `sae_logodds.py`).

## Scripts

### `books.py`

Book metadata shared by every page in `visualize/`: example sentences in the other
exports carry only their `book_id` (`bk`), and the pages look up its title, author, and
year here, so no export repeats them. Titles and authors come from the Hathi metadata
(`data/hathimeta/titlemeta.tsv`, whitespace collapsed); the year is the book's
`inferreddate`. Every book in the input parquet is listed, titled or not. Rerun it when
the input or the Hathi metadata changes.

```
python -m fashion.analysis.books [--debug] [--input PATH]
```

Output: `data/analysis/books/books.json`, `{book_id: [title, author, year]}` (title
and author null when unknown; about 1.3 MB for the full data before titles), read by
the page through the `visualize/public/data/books.json` symlink.

### `logodds.py`

Computes corpus-level (no temporal split) Monroe et al. (2008) Dirichlet-prior log-odds
of each character adjective being associated with each fashion term, one term vs. all
other terms. Character adjectives are the "words" and fashion terms are the "documents".
Counting follows `notebooks/fashion_gender_classifier.ipynb`:

- **Default** (the notebook's corpus-wide cell): one count per mention (duplicate
  mentions dropped) per character adjective; documents are lowercased `term`s.
- **`--gender {male,female}`** (the notebook's *Gender-controlled log-odds* cell): only
  that gender's characters; one count per (book, character, term), using the first
  mention's adjectives; documents are lowercased `term_stem`s.

Both drop terms and adjectives with fewer than `--min-count` (default 10) pairs
*before* building the matrix, so they are excluded from the prior too. No decade filter
is applied. Example sentences are collected for each kept (term, adjective) pair.

Two deliberate departures from the notebook's code, which change its printed numbers:
its `logodds()` turns `group2=None` into `{None}`, so "the rest" matches no documents
and each term is compared against the prior alone, not against the other terms (the
docstring's intent, and what this script does); and its corpus-wide cell calls
`dropna()` on the whole frame after exploding, which also drops every row with *any*
null column (e.g. a `gender_dep_pron` of `None`). Here only rows with no adjective are
dropped.

Per term it keeps the top `--top-adj` adjectives by |z| (`score`, δ/σ).

```
python -m fashion.analysis.logodds [--debug] [--gender {male,female}]
                                    [--top-terms N] [--top-adj N]
                                    [--min-count N] [--max-examples N]
                                    [--col-word COLUMN]
```

Output: `data/analysis/logodds/logodds.json` (default) or
        `data/analysis/logodds/logodds_{gender}.json` (`--gender`)

JSON fields: `terms` (list of fashion terms), `logodds` (list of records with
`term`, `adjective`, `logodds`, `sigma`, `score`, `count`), `examples`
(dict `{term: {adjective: [{s: sentence, bk: book_id}]}}`), `setup` (`gender`, `unit` — `mention` or
`character`, which is what `count` counts — and `min_count`). Older files lack `setup`;
they counted distinct (book, character) pairs across all characters, restricted to the
1800s–1910s.

### `adjective_categories.py`

Data for the *Adjective Categories* page: groups fashion adjectives into WordNet
attribute categories (each use disambiguated with Lesk against its sentence).

```
python -m fashion.analysis.adjective_categories [--debug]
```

Output, under `data/analysis/adjective_categories/`: `category_counts.csv` (`category`,
`count`) and `category_to_term_map.jsonl`, one line per category, most used first:
`{category, definition, terms, example_sentences}`, where `example_sentences` is
`{adjective: [{s: sentence, bk: book_id}]}` (up to 10, sampled).

### `unique_items.py`

Builds per-character "unique items" fashion profiles for the interactive
*Unique Items* page in `visualize/`. Mirrors the notebook's *Unique items*
section: for every character (`book_id`, `character_id`) it measures `num_mentions`
(total fashion mentions), `num_distinct` (distinct items worn), and `ratio`
(mentions ÷ distinct), and produces three ranking lists from these. For a bounded
pool of the most fashion-y characters (the union of each ranking's top cards and
the top `--scatter-n` by mention count) it also collects the top clothing items,
fashion adjectives, character adjectives, and the context sentences for each item,
so the front-end can expand a card down to the passage level and render a
scatter of mentions vs. distinct items.

Narrator / non-character narration (BookNLP `character_id == 0`) is *kept* and
flagged with `is_narrator` so the interface can toggle it on or off (the notebook's
`include_narr` default was off). Because narrators dominate the raw counts, each
ranking/scatter pool is the union of the top-N over all characters and the top-N
over non-narrators, so the interface still has a full top-N whichever way the
narrator toggle is set.

```
python -m fashion.analysis.unique_items [--debug] [--top-cards N] [--scatter-n N]
                                         [--min-mentions N] [--top-clothes N]
                                         [--top-adjs N] [--max-sentences N]
```

Output: `data/analysis/unique_items/unique_items.json`

JSON fields: `config` (the knobs above), `characters` (dict keyed by
`"{book_id}::{character_id}"` with `book_id`, `character_id`, `gender`, `is_narrator`,
`num_mentions`, `num_distinct`, `ratio`, `top_clothes`, `fashion_adjs`,
`char_adjs`, and `sentences` = `{item: [context sentences]}`), `rankings`
(`most_mentions`,
`most_distinct`, `most_repeated`, each a list of character ids), and `scatter`
(list of character ids plotted on the scatter). Every ranking/scatter id has a
full entry in `characters`. Titles, authors, and years come from `books.py`'s output,
by `book_id`.

### `adjective_trends.py`

Data for the interactive *Adjective Trends* page in `visualize/`. Builds on
`notebooks/cooccurrences.ipynb` (fashion adjectives × `term_stem`, adjectives NFKC-normalized
and lowercased) and splits every count by decade and by gender (`gender_dep_pron`; only
`male`/`female` are split out, everything counts toward `all`), so the page can compute
its measures on the fly.

The unit is the fashion **mention**: an adjective counts once per mention it describes
(repeats within a mention collapse). Next to every mention count is the number of distinct
books it comes from, for the page's *% of books* measure. Adjectives that aren't words
(OCR debris like `■mart`, elisions like `an'`) are dropped. Adjectives, term stems, and
(adjective, stem) pairs below `--min-adj-count` / `--min-term-count` / `--min-pair-count`
total mentions (default 20 / 20 / 3) are left out. The pair cutoff is low because the
page's adjective groups add up their members' pair counts: a cutoff of 3 keeps 95% of
pair uses (10 would keep 87%). All decades in the input are kept (1800s–1920s), and the
page picks the range.

**Preset adjective groups** are WordNet attribute categories from
`adjective_categories.py`'s `get_attribute_categories`, called *without* context: an
adjective joins every category that any of its senses maps to (e.g. *hue*: red, green,
grey, …; but also *sarcasm*: black, grim). Categories with fewer than `--min-group-size`
(default 2) kept adjectives are dropped. The page shows each preset's members and makes it
an editable group. That script's per-mention Lesk disambiguation isn't used: on a sample
it left half of all adjective uses with no category (including most uses of *white*,
*black*, and *long*), put most uses of *grey* under *confederacy* rather than *hue*, and
*blue* under *union* / *north*. Presets need the NLTK WordNet data
(`nltk.download('wordnet')`). Without it the script warns and writes no presets.

Example sentences are sampled per (adjective, decade), (term, decade), and (adjective,
term, decade): `--examples` / `--pair-examples` (default 5 / 3) over all characters, plus
`--gender-examples` (default 2) each for male and female characters so gender-filtered
views have examples too. Unsampled books come first, so the examples spread across books.
Pair examples are only sampled for pairs with at least `--min-pair-example-count` (default
10) mentions, to keep the files small. Each sentence keeps up to 220 characters of context
on each side of the term.

```
python -m fashion.analysis.adjective_trends [--debug] [--input PATH] [--output-dir PATH]
                                            [--min-adj-count N] [--min-term-count N]
                                            [--min-pair-count N]
                                            [--min-pair-example-count N]
                                            [--min-group-size N] [--examples N]
                                            [--pair-examples N] [--gender-examples N]
                                            [--seed N]
```

Output, under `data/analysis/adjective_trends/` (about 10 MB for `trends.json`, about 1 MB
gzipped as served, and 135 MB of example files on the full data; the run takes under a
minute):
- `trends.json`: `setup` (the knobs above), `decades` (e.g. `1880`), `genders`
  (`all`, `male`, `female`), `totals` (`{gender: {mentions, books, described, adj_tokens}}`,
  per-decade arrays, where `described` counts mentions with at least one adjective), `terms`
  (`{stem: {id, label, n, b}}`, where `label` is the stem's most common surface form), `adjs`
  (`{adj: {id, n, b}}`), and `pairs` (`{adj: {stem: {n, b}}}`). `n` is mentions and `b` is distinct
  books, each `{gender: [count per decade]}`. Also `groups`: the presets, as
  `[{id, name, definition, members}]` (`id` is the WordNet synset, e.g. `hue.n.01`; members
  are ordered by use, presets by their members' total use).
- `examples/adjs/{id}.json`: `{adj, examples: {decade: [ex]}, pairs: {stem: {decade: [ex]}}}`
- `examples/terms/{id}.json`: `{term, examples: {decade: [ex]}}`

An example `ex` is `{s: sentence, h: [start, end] of the term in s, a: the mention's
adjectives, bk: book_id, g: "m" | "f" | null}`. The page cites `bk` from `books.py`'s
output. Example files are named by the `id` in `trends.json`,
and ids change between runs, so the script clears `examples/` before writing.

The page reads these through the `visualize/public/data/adjective_trends` symlink. Its
measures are:
- adjectives: per 1,000 fashion mentions, % of described mentions, % of books, raw count
- terms: per 1,000 fashion mentions, % of books, raw count
- pairs: % of the term's mentions using the adjective, % of the adjective's uses that are
  on the term, per 10,000 fashion mentions, % of books, raw count

Intervals are Wilson 95% intervals. *Discover* ranks series by the log ratio of their rate in
the later half of the selected range to the earlier half.

Anywhere an adjective is used, the page can use a **group** of adjectives instead (from a
preset, or by grouping the selected adjectives), and groups can be edited. A group's counts
are the sum of its members' counts, so they count adjective uses rather than mentions. A
mention described by two members counts twice, which adds about 3.5% for the *hue* preset.
Book counts can't be summed, so *% of books* isn't available for groups.

*Top pairs* lists the terms an adjective (or group) describes, or the adjectives used on a
term, over the selected decades and characters. It can rank by count (shown as the share of
the focus's uses/mentions) or by *lift*, P(partner | focus) / P(partner per fashion mention).
Lift equals count × all mentions / (focus count × partner count), so it reads the same
either way. Lift is only computed for partners above a minimum count, since it is noisy for
rare pairs. For a term, it can also rank **adjective groups** instead of single adjectives:
your groups plus the presets, filterable to either. A preset with the same name as one of
your groups is replaced by yours, since yours may be an edited copy. Group counts are summed as
above, so large, loose presets (*age*, *light*) lead on count; *Most distinctive* is usually
more telling. Hover a group's row to see its members.

### `load_passages.py`

The *loading* half of the show-tell pipeline. Opens each book once, Punkt-
tokenizes it, and slices out the passage of sentences enclosing every character
mention (the target sentence ± 3 sentences on each side). Books are processed in
a parallel process pool, since text loading (disk I/O) and tokenization (CPU) are
the slow parts. Rows whose span has no enclosing sentence are dropped.

Output is a compact parquet of passage text plus the **book-level byte offsets**
needed to build a prompt downstream, so tokenization runs once and inference can
be re-run (different model, prompt, sampling) without redoing it.

```
python -m fashion.analysis.load_passages [--debug] [--workers N]
```

Output: `data/analysis/llm_showtell/passages.parquet` (or `passages_debug.parquet`
under `--debug`).

Columns: `row_id` (position in the input parquet), `book_id`, `passage` (text),
`passage_start_idx`/`passage_end_idx`, `character_start_idx`/`character_end_idx`,
`fashion_start_idx`/`fashion_end_idx` — all offsets are book-level byte offsets
into the original text.

### `llm_showtell.py`

**Runtime**: To run on the debug set of 100,000 rows on 2 L40s for Qwen3-8B, it
took 5:55:49.

There are 4,039,248 rows in the full `final_for_analysis.parquet` dataset, so
that would take roughly 10 days all in (40 x 6 = 240 hours). But I think this
does make it seem like running on one of the UIUC servers might actually be
tractable; if we have an H100 node, I suspect we could finish running quite
quickly...

Running `load_passages.py` generates parquet files in the correct locations.
This requires Hathitrust files to exist, so Naitian runs this on David's lab
servers and provides the files --- Becca does not need to run this script (and
should instead use `{passages,passages_debug}.parquet` from the Google drive
(in `data/`). The scripts expects the `data/analysis/llm_showtell` directory to
exist. The parquet files should be placed in this directory.


Uses an LLM to generate adjectival descriptions of each character mention,
given the surrounding passage (with the character span wrapped in
`**asterisks**`). Inference runs against a **separately launched** vLLM
OpenAI-compatible server; the script fires requests concurrently and relies on
vLLM's continuous batching for throughput.

The server is constrained (via `response_format` guided decoding) to emit
`{"adjectives": [{"word", "reasoning"}, ...]}`, which the client parses before
storing.

Passages are **streamed** off the parquet in batches (so the whole corpus is
never held in memory); each row's prompt is built on the fly (subtracting the
passage start offset to place the `**` around the character span) and pushed onto
a bounded queue that `--concurrency` async consumers drain, calling the server.

First launch the server (in the `vllm` pixi env), e.g.:

```
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --max-model-len 8192 --port 8000
```

Then run:

```
python -m fashion.analysis.llm_showtell [--debug] [--base-url URL] [--model NAME]
                                        [--concurrency N] [--batch-size N]
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
`colors.py` and `term_prevalence.py`.

```
python -m fashion.analysis.gender_classifier [--debug] [--min-df N]
```

Output: `data/analysis/gender_classifier/coefficients.parquet`
        `data/analysis/gender_classifier/coefficients.tsv`

Columns: `decade`, `term`, `coefficient` (positive = male-indexing).

### `term_prevalence.py`

Plots how prevalent the strongly gendered fashion terms are over time, as
**publication-ready vector PDFs** with PNGs alongside (matplotlib, not Altair). Two panels —
male-indexing and female-indexing terms, selected from the classifier
coefficients — show each term's share of all fashion mentions per decade with
binomial 95% confidence intervals.

To keep a handful of verbose books from dominating a decade, at most
`--sample-n` mentions are sampled per book before counting (`--sample-by
book-term` caps each term per book instead, as the notebook did). The
confidence intervals are therefore intervals on the *sampled* proportion.

There is no legend: the top `--top-k` terms per panel (by mean prevalence) are
colored and labeled inline at the end of their line, with leader lines where
labels would collide; the remaining terms are drawn as unlabeled gray context
lines (`--no-context-lines` drops them). Both panels share a y-scale by default,
since they share a denominator — `--free-y` breaks that when detail matters more
than comparability.

The categorical palette is capped at six slots, which is also the cap on
`--top-k`; past that, series stop being reliably distinguishable.

Two versions are written:
- **full** (paper, `--width` × `--height`, default 7 × 3.1 in, 8pt): everything
  above, plus the method footnote.
- **simple** (slides, 16:9, 16pt): the top `--simple-top-k` (default 3) terms per
  panel, no gray context lines, no footnote, and a decade label every 40 years.
  The terms are the leading ones from the full figure, so each keeps its color.
  Line widths and markers scale with the type.

PNGs have a transparent background; PDFs keep the white one.

Requires `gender_classifier.py` to have been run first.

```
python -m fashion.analysis.term_prevalence [--debug] [--input PATH]
                                           [--classifier-output PATH]
                                           [--gender {all,male,female}]
                                           [--top-terms N] [--top-k N]
                                           [--exclude-terms TERM ...]
                                           [--sample-n N] [--sample-by {book,book-term}]
                                           [--ci-method {wald,wilson}] [--conf C]
                                           [--no-context-lines] [--free-y]
                                           [--no-footnote] [--width IN] [--height IN]
                                           [--font-size PT] [--seed N]
                                           [--version {both,full,simple}]
                                           [--simple-top-k N] [--simple-width IN]
                                           [--simple-height IN] [--simple-font-size PT]
                                           [--output-dir PATH]
```

Output: `data/analysis/figures/term_prevalence/term_prevalence_{gender}.pdf` (+ `.png` at 300 dpi)
        `data/analysis/figures/term_prevalence/term_prevalence_{gender}_simple.pdf` (+ `.png`)

`--gender` restricts *which mentions are counted*, which also changes the
denominator: the default `all` plots each term's share of every mention in the
decade, while `male`/`female` plots its share of that gender's mentions.
`--exclude-terms` drops terms the classifier picks up that aren't clothing (e.g.
`hands`). `--input` overrides the input parquet path directly, for when the data
lives outside `data/`.

### `logodds_plot.py`

Print-ready figures of the `logodds.py` results for one or more fashion terms,
in the same matplotlib style as `term_prevalence.py`. Each figure is a
diverging bar chart of z-scores (δ/σ): adjectives over-represented among
characters linked to the term point right (blue), under-represented ones point
left (red), and each adjective's name sits across the zero line from its bar.

`--gender {male,female}` plots the gender-controlled results
(`logodds_{gender}.json`; terms are then term stems) and adds `_{gender}` to the
output names. The count column and footnote follow the JSON's `setup`: *Mentions*
for the default run, *Characters* for `--gender`.

Two versions are written per term:
- **full** (paper/poster, default one 3.5in column, 8pt): z-score axis, shaded
  ±z* significance band (non-significant bars in gray), a count column, and a
  method footnote. Height fits the rows.
- **simple** (slides, 10×5.625in = 16:9, 16pt): fewer adjectives, no axis, no
  footnote. Each bar carries its z-score at the tip (inside when it fits, else
  just past it; the first reads "z = 8.3"), and each name has its count beside it
  as a small gray aside on the side away from the bar (the first reads "47.8k
  mentions"). Any non-significant bars are explained by a one-line gray key at
  the bottom.

Both versions select the top adjectives on each side by |z|; the simple one just
shows fewer (`--simple-top-n`, default 6, vs. `--top-n`, default 12).

Only terms present in `logodds.json` can be plotted (by default the top 20 by
frequency; an unknown term errors with the available list). Rerun
`logodds.py --top-terms N` to add more. `--exclude-adjectives` drops parser
noise such as `sure` or `about`. `--title` is a template with `{term}` and `{who}` (*Character adjectives*, or
*Male character adjectives* / *Female character adjectives*);
`--no-title` leaves the title to the paper caption.

Requires `logodds.py` to have been run first.

```
python -m fashion.analysis.logodds_plot TERM [TERM ...] [--gender {male,female}]
                                        [--input PATH]
                                        [--version {both,full,simple}]
                                        [--top-n N] [--simple-top-n N]
                                        [--exclude-adjectives ADJ ...]
                                        [--conf C] [--title TEMPLATE] [--no-title]
                                        [--no-footnote] [--width IN] [--height IN]
                                        [--font-size PT] [--simple-width IN]
                                        [--simple-height IN] [--simple-font-size PT]
                                        [--output-dir PATH]
```

PNGs have a transparent background (for dropping onto slides and posters); PDFs keep
the white one. The ink is dark, so use a light slide background.

Output: `data/analysis/figures/logodds_plot/logodds_{term}[_{gender}].pdf` (+ `.png` at 300 dpi)
        `data/analysis/figures/logodds_plot/logodds_{term}[_{gender}]_simple.pdf` (+ `.png`)

### `pca_axes.py`

Script version of `notebooks/pca.ipynb`. Builds a character × fashion-term-stem
matrix (named characters only, i.e. `character_id != 0`; characters and stems
with at least `--min-count` mentions; cells are mention counts), runs a 20-component
truncated SVD, and tests every component against:

- **gender** (one row per character): Kendall's τ between PC score and gender
  (male = 0, female = 1), male/female characters only.
- **genre** (one row per book): each book's score is the mean of its characters'
  scores. For each fiction genre (Hathi `genres` after `data/hathi_genre_merges.csv`),
  the mean pairwise distance between its books is compared with 10,000 label
  permutations. A ratio below 1 means that genre's books cluster more tightly than
  chance; the omnibus p tests the mean dispersion across genres.

`--normalize` controls what each character's row holds:
- `counts` (the notebook): raw mention counts. Heavily described characters
  dominate, so a component can track how much a character is described rather
  than how.
- `l2`: each row scaled to unit length first, so characters are compared on
  their mix of terms rather than their number of mentions.

`--centering` controls what is subtracted before the SVD (after normalizing):
- `term` (standard PCA): each term's mean over characters.
- `row` (the notebook): each character's mean over terms. Under this, the first
  component mostly separates common terms (*hat, dress, coat*) from rare ones
  (*toga, kirtle, periwig*).

The default `both` runs both, so the results can be compared. Component signs are
fixed so each PC's largest loading is positive. Differences from the notebook: a
(book, genre) pair is counted once even when several raw Hathi genres merge into
it (the notebook listed two *Adventure fiction* books twice).

Requires `data/hathimeta/titlemeta.tsv` and `data/hathi_genre_merges.csv`.

```
python -m fashion.analysis.pca_axes [--debug] [--normalize {counts,l2}]
                                    [--centering {term,row,both}]
                                    [--n-components N] [--min-count N]
                                    [--min-genre-books N] [--n-perm N]
                                    [--seed N] [--perm-seed N] [--output-dir PATH]
```

Output, under `data/analysis/pca_axes/{run}/`, where `{run}` is the centering,
suffixed with `_{normalize}` unless it is `counts` (e.g. `term`, `term_l2`):
- `pc_summary.csv`: per PC, variance explained, gender τ and p, Spearman ρ with a
  character's total mentions (`mentions_rho`; near ±1 means the PC mostly tracks how
  much a character is described), genre omnibus p, and mean genre dispersion ratio
- `genre_tests.csv`: per PC × genre, n, observed and null dispersion, ratio, p
- `loadings.csv`: per term stem, its mention count and loading on each PC
- `characters.parquet`: per character, book, gender, and PC scores
- `books.parquet`: per book, genres, number of characters, and mean PC scores

### `pca_plot.py`

Scatter of characters on two components from `pca_axes.py` (PC1 × PC2 by default),
colored by gender, as print-ready figures in the same style as `logodds_plot.py`.
Each male or female character is a dot (characters without a male/female label
are left out), and the margins show each gender's kernel density along both
axes: the gender component's curves pull apart, the other's overlap. Each axis is
labeled with the most strongly loading term stems at its two ends (among stems
with more than `--min-term-mentions` mentions), so it reads without PCA
vocabulary. The plotted range drops the outer `--trim` quantile on each end of
each axis, since a few heavily described characters sit far from the rest; the
footnote says how many are left out.

- **full** (paper, 6 × 6 in, 8pt): axis labels with each PC's share of variance,
  plus a method footnote with the gender counts and Kendall's τ for both PCs.
- **simple** (slides, 16:9, 14pt): no footnote, and axis labels reduced to the
  end terms.

Dots are rasterized in the PDFs to keep files small; PNGs are transparent.

Requires `pca_axes.py` to have been run first.

```
python -m fashion.analysis.pca_plot [--centering {term,row}] [--normalize {counts,l2}]
                                    [--x PC] [--y PC]
                                    [--input-dir PATH] [--version {both,full,simple}]
                                    [--trim Q] [--n-terms N] [--min-term-mentions N]
                                    [--title TEXT] [--no-title] [--no-footnote]
                                    [--width IN] [--height IN] [--font-size PT]
                                    [--point-size PT2] [--simple-width IN]
                                    [--simple-height IN] [--simple-font-size PT]
                                    [--seed N] [--output-dir PATH]
```

`--centering` and `--normalize` pick which `pca_axes.py` run to plot. Each axis
names up to `--n-terms` terms per end, dropping terms until the label fits along
the axis.

Output: `data/analysis/figures/pca_plot/pca_{x}_{y}_{run}.pdf` (+ `.png` at 300 dpi)
        `data/analysis/figures/pca_plot/pca_{x}_{y}_{run}_simple.pdf` (+ `.png`)

### `embeddings.py`

Embeds every mention with Qwen3-Embedding-0.6B (sentence-transformers, SDPA attention,
left padding), following `notebooks/sentence_embeddings.ipynb`. By default it embeds
the mention's `sentence`. Each distinct text is embedded once (~2.7M unique sentences
out of ~4.0M rows).

Variants, each best written to its own `--output-dir` (e.g.
`data/analysis/embeddings_excerpt_marked/`):

- **`--context excerpt`** embeds the excerpt the wearer model saw instead of the
  sentence. `extract_char_fashion_mentions.py` builds it from the BookNLP sentences
  holding the 10 character mentions nearest the item, clipped to 500 characters on
  either side of the item. It is read from the wearer model's output (`excerpt_start`,
  `excerpt_text` in `--wearing-dir`, default `data/pipeline/hathi_all/wearing/`),
  joined on `(book_id, mention_id)`. Rows without an excerpt fall back to their
  sentence, and the count is printed. This matters for the wearer: in 37% of rows the
  wearer's mention lies outside the sentence.
- **`--mark-spans item wearer`** (either or both) wraps the fashion item as
  `[bonnet]_item` and the wearer (the character mention the wearer model picked, i.e.
  `character_start_idx`/`character_end_idx`) as `[her]_wearer`. BookNLP mentions can be
  noun phrases that contain the item; those nest (`[a man in full [dress]_item]_wearer`).
  A span outside the context, or a partial overlap, is left unmarked, and the counts
  are printed. With `--context sentence`, about a third of wearers can't be marked.
- **`--prompt TEMPLATE`** adds a Qwen3 instruction: the model sees
  `Instruct: {instruction}\nQuery:{text}`. The template may use `{term}` and
  `{wearer}` (the wearer's mention as written, read from the excerpt). A constant
  prompt keeps the deduplication. A per-mention one (or any marker) makes most
  texts unique, so up to ~4.0M texts get embedded.

Inference is data-parallel through `fashion.distributed.run_distributed`. Rank 0 builds
the texts and writes `sentences.parquet` (one row group per shard), `sentence_idx.parquet`
and `setup.json`. It then spawns one worker per GPU (`--num_processes`, default 4). The
unique texts are split into `--num-shards` contiguous shards (default 64), and each
finished shard is saved to `shards/NNNN.npy`. Workers skip shards that already exist, so
a crashed or rerun job picks up where it stopped. A rerun reuses the texts when
`setup.json` matches its settings, and refuses to run when it doesn't. Once all workers
finish, rank 0 merges the shards and deletes them.

```
python src/fashion/analysis/embeddings.py [--debug] [--input PATH] [--output-dir DIR]
                                          [--context {sentence,excerpt}]
                                          [--mark-spans {item,wearer} ...]
                                          [--prompt TEMPLATE] [--wearing-dir DIR]
                                          [--num-shards N] [--batch-size N]
                                          [--dtype {bfloat16,float16,float32}]
                                          [--num_processes N]
```

For example:

```
python src/fashion/analysis/embeddings.py --context excerpt --mark-spans item wearer \
    --prompt "How is the wearer of the {term} (marked wearer) described or presented?" \
    --output-dir data/analysis/embeddings_excerpt_wearer
```

Run it as a script, not with `-m`: workers are started by re-running the script's
own path. The model runs in bfloat16 by default. Embeddings are stored as float16
either way; they have cosine similarity ≥ 0.9998 with the notebook's float32 output.
If `sentence_embeddings.npy` already exists, the script exits without doing anything.

Output: `data/analysis/embeddings/` (or `--output-dir`). The file names are the same for
every variant, so `sae.py --activations_path` can point at any of them:
- `sentence_embeddings.npy`: `(n_texts, 1024)` float16. Row i is the embedding
  of row i of `sentences.parquet`.
- `sentences.parquet`: the unique texts, in the order they first appear. `sentence` is
  the context with any markers. With `--prompt`, `instruction` is the filled-in
  instruction.
- `sentence_idx.parquet`: one row per input row, with the input's
  `(book_id, character_id)` index. Its `sentence_idx` column indexes into the two
  files above.
- `setup.json`: the settings the run was made with.

`sae_logodds.py` and `sae_labels.py` still read `data/analysis/embeddings/`, the default
(plain sentence) run.

### `sae.py`

Trains a TopK sparse autoencoder (vendored `dictionary_learning`, `TopKTrainer`) on
the sentence embeddings from `embeddings.py`, as a first step toward mechanistic
topic models (Zheng et al., TACL 2026) over a corpus-trained SAE: SAE codes become
the "vocabulary," counted per document. The loader keeps all embeddings on the GPU
(~5.5 GB in float16), reshuffles every pass, drops the short last batch, and
loops until `--steps`, so training runs for multiple passes over the data (10k steps at
batch 8192 is ~30 passes over ~2.7M sentences). Activations are normalized to unit
mean squared norm (`normalize_activations=True`); the Qwen embeddings are already
unit-norm, so the factor should print as ~1.

```
python src/fashion/analysis/sae.py [--activations_path PATH] [--output_dir DIR]
                                   [--batch_size N] [--num_codes N] [--k N]
                                   [--steps N] [--lr LR] [--warmup_steps N]
                                   [--num_checkpoints N] [--seed N] [--device DEV]
```

Defaults: 2048 codes, k=32, lr 5e-4, 1k warmup steps (also the step at which the
inference threshold starts updating). `--activations_path` takes a `.npy` array or a
`torch.save`d tensor. The vendored `TopKTrainer` has its dead-feature threshold
lowered from 10M to 500k activations (~61 steps at batch 8192), so the aux-k loss
revives dead codes sooner; this edit lives in the `vendor/dictionary_learning`
submodule. Progress prints L0 and fraction of variance explained every 10 steps;
the dead-feature count goes only to wandb, which is off.

Requires `embeddings.py` to have been run first.

Output: `data/analysis/sae/trainer_0/`
- `ae.pt`: final SAE state dict, with biases rescaled by the norm factor.
- `checkpoints/ae_<step>.pt`: `--num_checkpoints - 1` evenly spaced intermediate
  checkpoints.
- `config.json`: trainer config.

### `sae_logodds.py`

Which SAE codes go with which fashion terms: the `logodds.py` Monroe et al. (2008)
log-odds (one term vs. all other terms), with the SAE codes active on each
mention's sentence (from `sae_inference.py`'s `codes.pt`) as the "words" instead
of character adjectives. Documents are lowercased `term`s; each mention (duplicates
across characters dropped) counts once per active code. Terms and codes with fewer
than `--min-count` (default 10) (mention, code) pairs are dropped before the matrix
is built.

For each term given (default: the `--top-terms` most frequent), it prints the
`--top-codes` codes with the highest z-score (δ/σ), each with its LLM label from
`sae_labels.py` and the `--examples` (default 5) sentences of that term's mentions on
which the code fires most strongly. Labels are optional: if `--labels` (default
`data/analysis/sae_labels/labels.parquet`) doesn't exist, codes go unlabeled.

It also writes the data for the *SAE Log-Odds* page in `visualize/`: every term in
the matrix, with its `--json-codes` (default 30) highest-z codes (z > 0), and the
`--examples` example sentences for each of those (term, code) pairs.

Requires `embeddings.py`, `sae.py` and `sae_inference.py` to have been run first,
on the same input (`--debug` must match `embeddings.py`'s).

```
python -m fashion.analysis.sae_logodds [TERM ...] [--debug] [--trainer-dir DIR]
                                       [--labels PATH] [--top-terms N]
                                       [--top-codes N] [--json-codes N]
                                       [--examples N] [--min-count N]
                                       [--trend-terms N] [--min-decade-mentions N]
```

Output: `data/analysis/sae_logodds/`
- `sae_logodds.parquet`: one row per (term, code), with `term`, `code`, `logodds`,
  `sigma`, `score`, `count`, and the code's `label` and `description` (null if
  unlabeled).
- `sae_logodds.json`: `terms` (list, most-mentioned first, of `{term, mentions,
  codes}`, where `codes` is a list of `{code, score, logodds, count}`, highest z
  first), `labels` (`{code: {label, description}}` for the codes that appear),
  and `setup` (`min_count`, `json_codes`, `examples`).
- `examples/{i}.json`: `{code: [{s: sentence, bk: book_id}]}` for term `i` of
  `terms` (a sentence found in several books is credited to the first). The page
  loads these lazily.
- `trends.json`, for the *SAE Code Trends* page: per decade, the activations of
  every code in `sae_logodds.json`, split by its `--trend-terms` (default 20)
  highest-z terms (z > 0). Over dated mentions only, and decades with fewer than
  `--min-decade-mentions` (default 1000) of them are dropped. `decades` (e.g.
  `1880`), `mentions` (fashion mentions per decade), `terms` (`{term: [mentions
  per decade]}` for every listed term), `codes` (`{code: {total, items}}`, where
  `total` is the code's activations per decade over all terms and `items` is a
  list, highest z first, of `{term, score, count, n}`, with `n` the term's
  activations per decade and `score`/`count` as in the parquet), and `setup`.
  The page folds the activations not covered by the shown items into "other".
- `term_trends/{i}.json`, for that page's by-item view: `{mentions, codes}` for
  term `i` of `sae_logodds.json`'s `terms`, where `mentions` is the term's dated
  mentions per decade and `codes` is `{code: [activations per decade]}` for the
  codes `sae_logodds.json` lists for it. Same decades as `trends.json`. The page
  loads these lazily; the script clears the folder before writing.

### `sae_labels.py`

Labels each SAE code with an LLM, using `sae_inference.py`'s `codes.pt`. Rerun
`sae_logodds.py` afterwards to pick the labels up. For each code
the prompt shows 10 sentences it fires on (the 5 strongest plus 5 random others,
shuffled together) and 10 random sentences it does not fire on, without activation
values, and asks for a 2–5 word `label` and a one-sentence `description` of what the
firing sentences share. Codes firing on fewer than `--min-count` (default 10)
sentences are skipped. Rows of `codes.pt` index `embeddings.py`'s `sentences.parquet`
directly.

Like `llm_showtell.py`, it talks to a vLLM OpenAI-compatible server you launch
yourself (e.g. `vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --port 8000`),
with guided JSON decoding and `--concurrency` requests in flight. Results are
appended to `labels.jsonl` as they finish, and a rerun skips codes that already
have a label, so it resumes after a crash and retries only the failures.

Requires `embeddings.py`, `sae.py` and `sae_inference.py` to have been run first.

```
python -m fashion.analysis.sae_labels [--trainer-dir DIR] [--output-dir DIR]
                                      [--base-url URL] [--model NAME]
                                      [--concurrency N] [--max-tokens N]
                                      [--temperature T] [--no-thinking]
                                      [--thinking-budget N] [--max-retries N]
                                      [--min-count N] [--seed N]
```

Output: `data/analysis/sae_labels/`
- `labels.jsonl`: raw results, one line per request, in completion order.
- `labels.parquet`: one row per code (the latest result for each), with `code`,
  `label`, `description`, `prompt`, `reasoning`, `response`. `label` is null where
  the reply could not be parsed or the request failed.

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

Output: `data/analysis/figures/colors/color_prevalence_{gender}.html`

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

Output: `data/analysis/figures/gp_logodds/gp_logodds.html`

The top adjectives shown per term are selected by maximum absolute log-odds across all decades.
