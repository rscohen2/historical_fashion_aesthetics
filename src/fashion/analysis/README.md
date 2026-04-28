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
                                   [--classifier-output PATH]
```

Output: `data/analysis/colors/color_prevalence_{gender}.html`