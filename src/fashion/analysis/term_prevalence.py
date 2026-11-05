"""Plot prevalence of gender-indexing fashion terms over time as print-ready figures.

Uses classifier coefficients (from gender_classifier.py) to pick the fashion
terms that most strongly index male and female characters, then plots each
term's share of all fashion mentions per decade, with binomial 95% confidence
intervals.

To keep a handful of verbose books from dominating a decade, at most
`--sample-n` mentions are sampled per book before counting.

Series are identified by direct labels at the right edge of each line rather
than a legend; only the top `--top-k` terms per panel (by mean prevalence) are
colored and labeled, and the remaining terms are drawn as gray context lines.

Two versions are written:
  - full:   paper/poster figure (7 × 3.1 in) with context lines and a method
            footnote.
  - simple: slide figure (16:9) with large type, the top --simple-top-k terms
            per panel, no context lines, and no footnote. Each term keeps the
            color it has in the full figure.

Requires: data/analysis/gender_classifier/coefficients.parquet
Output:   data/analysis/figures/term_prevalence/term_prevalence_{gender}.pdf (+ .png)
          data/analysis/figures/term_prevalence/term_prevalence_{gender}_simple.pdf (+ .png)
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import norm

from fashion.paths import DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, PercentFormatter

EXPERIMENT = "term_prevalence"
OUTPUT_DIR = DATA_DIR / "analysis" / "figures" / EXPERIMENT
CLASSIFIER_OUTPUT = DATA_DIR / "analysis" / "gender_classifier" / "coefficients.parquet"
DECADES = list(range(180, 192))

# Categorical palette, in fixed slot order. Validated for the adjacent-pair
# gates on the light surface below (CVD ΔE 9.1, normal-vision ΔE 19.6); three
# slots sit under 3:1 contrast, which the direct labels relieve.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
]
SURFACE = "#fff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT_LINE = "#c3c2b7"

PANEL_TITLES = {
    "male": "Masculine terms",
    "female": "Feminine terms",
}


def set_style(base_size: float) -> None:
    """Publication defaults: embedded Type 42 fonts, hairline chrome, no clutter."""
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def load_coefficients(path: Path) -> pd.DataFrame:
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_parquet(path)


def get_top_terms(coef_df: pd.DataFrame, gender: str, top_n: int) -> list[str]:
    """Return terms in the top N for the given gender in at least one decade.

    Positive coefficient = male-indexing; negative = female-indexing.
    """
    ascending = gender == "female"
    top_per_decade = (
        coef_df.sort_values("coefficient", ascending=ascending)
        .groupby("decade")
        .head(top_n)
    )
    return top_per_decade["term"].unique().tolist()


def sample_mentions(
    df: pd.DataFrame, sample_n: int, seed: int, by: list[str]
) -> pd.DataFrame:
    """Keep at most `sample_n` mentions per group (a shuffle-then-take-head sample).

    Equivalent to `groupby(by).sample(n)` without replacement, but it never
    raises on groups smaller than `sample_n` and stays fast on millions of rows.
    """
    if sample_n <= 0:
        return df
    shuffled = df.sample(frac=1.0, random_state=seed)
    keep = shuffled.groupby(by, sort=False).cumcount() < sample_n
    return shuffled[keep]


def binom_ci(
    k: np.ndarray, n: np.ndarray, conf: float = 0.95, method: str = "wald"
) -> tuple[np.ndarray, np.ndarray]:
    """Binomial confidence interval for a proportion, vectorized over k and n.

    `wald` is the normal approximation around the MLE p̂ = k/n. `wilson` scores
    the interval instead, which behaves better for small n or p̂ near 0 or 1.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = float(norm.ppf(1 - (1 - conf) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        if method == "wilson":
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            halfwidth = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
            low, high = center - halfwidth, center + halfwidth
        else:
            stderr = np.sqrt(p * (1 - p) / n)
            low, high = p - z * stderr, p + z * stderr
    low = np.where(n > 0, low, 0.0)
    high = np.where(n > 0, high, 0.0)
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def compute_prevalence(
    df: pd.DataFrame, conf: float = 0.95, ci_method: str = "wald"
) -> pd.DataFrame:
    """Per-decade share of all fashion mentions held by each term, with CIs.

    The denominator is every mention in the decade (after sampling), so a term's
    prevalence is its share of the decade's fashion vocabulary.

    Every (decade, term) cell is materialized, so a term that goes unused in a
    decade reads as a genuine zero instead of a gap the line would jump across.
    """
    totals = df.groupby("decade").size()
    grid = pd.MultiIndex.from_product(
        [sorted(totals.index), sorted(df["term"].unique())], names=["decade", "term"]
    )
    counts = (
        df.groupby(["decade", "term"])
        .size()
        .reindex(grid, fill_value=0)
        .reset_index(name="count")
    )
    counts["total"] = counts["decade"].map(totals)
    counts["proportion"] = counts["count"] / counts["total"]
    counts["ci_low"], counts["ci_high"] = binom_ci(
        counts["count"].to_numpy(),
        counts["total"].to_numpy(),
        conf=conf,
        method=ci_method,
    )
    return counts


def rank_terms(prevalence: pd.DataFrame, terms: list[str], top_k: int) -> list[str]:
    """Terms ordered by mean prevalence across decades, highest first."""
    subset = prevalence[prevalence.term.isin(terms)]
    ordered = (
        subset.groupby("term")["proportion"].mean().sort_values(ascending=False).index
    )
    return list(ordered)[:top_k]


def _spread_labels(
    positions: list[float], min_gap: float, lo: float, hi: float
) -> list[float]:
    """Nudge label positions apart, preserving their order, inside [lo, hi].

    Overlapping labels are merged into a cluster and re-spaced evenly around the
    cluster's own centre of mass, so a crowded group opens symmetrically instead
    of shunting every label in one direction.
    """
    order = np.argsort(positions)
    ys = np.asarray(positions, dtype=float)[order]

    # Each cluster is [members' original positions]; centre it on their mean.
    clusters: list[list[float]] = []
    for y in ys:
        clusters.append([y])
        while len(clusters) > 1:
            top_of_prev = _cluster_span(clusters[-2], min_gap)[1]
            if _cluster_span(clusters[-1], min_gap)[0] >= top_of_prev + min_gap:
                break
            clusters[-2:] = [clusters[-2] + clusters[-1]]

    placed: list[float] = []
    for cluster in clusters:
        bottom = _cluster_span(cluster, min_gap)[0]
        placed.extend(bottom + i * min_gap for i in range(len(cluster)))

    # Clamp the whole stack into the axes without reintroducing overlaps.
    span = np.asarray(placed, dtype=float)
    span += max(0.0, lo - span[0])
    span -= max(0.0, span[-1] - hi)

    out = np.empty_like(span)
    out[order] = span
    return list(out)


def _cluster_span(members: list[float], min_gap: float) -> tuple[float, float]:
    """Bottom and top of a cluster of `len(members)` labels centred on their mean."""
    height = (len(members) - 1) * min_gap
    centre = float(np.mean(members))
    return centre - height / 2, centre + height / 2


def draw_panel(
    ax: plt.Axes,
    prevalence: pd.DataFrame,
    labeled_terms: list[str],
    context_terms: list[str],
    title: str,
    base_size: float,
    label_every: int = 2,
) -> list[tuple[str, float, str]]:
    """One panel: colored + labeled lines for the top terms, gray lines behind.

    Returns each labeled series' (term, final value, color) so the direct labels
    can be placed once the y-scale is final.
    """
    x_of = {decade: decade * 10 for decade in DECADES}
    x_min, x_max = x_of[DECADES[0]], x_of[DECADES[-1]]
    # Marks were tuned at 8pt; scale them with the type so slides don't look thin.
    scale = base_size / 8.0

    def series(term: str) -> pd.DataFrame:
        rows = prevalence[prevalence.term.eq(term)].sort_values("decade")
        return rows[rows.decade.isin(DECADES)]

    for term in context_terms:
        rows = series(term)
        ax.plot(
            rows.decade * 10,
            rows.proportion,
            color=CONTEXT_LINE,
            linewidth=0.7 * scale,
            alpha=0.75,
            zorder=1.5,
        )

    ends: list[tuple[str, float, str]] = []
    for i, term in enumerate(labeled_terms):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        rows = series(term)
        x = rows.decade * 10
        ax.fill_between(
            x,
            rows.ci_low,
            rows.ci_high,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        ax.plot(x, rows.proportion, color=color, linewidth=1.5 * scale, zorder=3)
        ax.plot(
            x,
            rows.proportion,
            color=color,
            linestyle="none",
            marker="o",
            markersize=2.6 * scale,
            markeredgewidth=0,
            zorder=3.5,
        )
        # End dot carries a surface ring so it stays legible where lines cross.
        ax.plot(
            x.iloc[-1],
            rows.proportion.iloc[-1],
            color=color,
            linestyle="none",
            marker="o",
            markersize=4.2 * scale,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0 * scale,
            zorder=4,
        )
        ends.append((term, float(rows.proportion.iloc[-1]), color))

    ax.set_title(title, fontsize=base_size + 1, color=INK_PRIMARY, pad=6, loc="left")
    ax.set_xlim(x_min - (x_max - x_min) * 0.02, x_max + (x_max - x_min) * 0.30)
    # Label every `label_every`-th decade; the rest get minor ticks.
    labeled_decades = DECADES[::label_every]
    ax.xaxis.set_major_locator(FixedLocator([x_of[d] for d in labeled_decades]))
    ax.xaxis.set_minor_locator(
        FixedLocator([x_of[d] for d in DECADES if d not in labeled_decades])
    )
    ax.set_xticklabels([f"{x_of[d]}s" for d in labeled_decades])
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return ends


def format_percent_axis(ax: plt.Axes) -> None:
    """Percent ticks with just enough decimals that no two labels read alike."""
    ticks = ax.get_yticks()
    step = float(ticks[1] - ticks[0]) if len(ticks) > 1 else 0.01
    # Below a 1-point step, whole-percent labels would repeat (0%, 0%, 1%…).
    decimals = 0 if step >= 0.00999 else 1
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))


def _draw_end_labels(
    ax: plt.Axes,
    ends: list[tuple[str, float, str]],
    x_max: float,
    label_pad: float,
    base_size: float,
) -> None:
    """Direct labels at the right edge, de-collided with leader lines when moved."""
    if not ends:
        return

    y_lo, y_hi = ax.get_ylim()
    # Minimum vertical gap between two labels, in data units.
    fig_height_in = ax.figure.get_size_inches()[1]
    axes_height_in = fig_height_in * ax.get_position().height
    min_gap = (base_size * 1.3 / 72) / axes_height_in * (y_hi - y_lo)

    raw = [y for _, y, _ in ends]
    adjusted = _spread_labels(raw, min_gap, y_lo, y_hi)

    for (term, y_end, color), y_label in zip(ends, adjusted):
        if abs(y_label - y_end) > min_gap * 0.25:
            ax.plot(
                [x_max, x_max + label_pad * 2.2],
                [y_end, y_label],
                color=color,
                linewidth=0.6,
                alpha=0.55,
                zorder=2.5,
            )
            x_text = x_max + label_pad * 2.8
        else:
            x_text = x_max + label_pad * 1.6
        ax.text(
            x_text,
            y_label,
            term,
            color=INK_PRIMARY,
            fontsize=base_size - 0.5,
            va="center",
            ha="left",
            zorder=5,
        )


def _wrap_to_width(
    fig: plt.Figure, text: str, font_size: float, max_width_in: float
) -> str:
    """Wrap `text` to the widest character count that renders within the width."""
    probe = fig.text(0, 0, "", fontsize=font_size)
    renderer = fig.canvas.get_renderer()

    def fits(wrapped: str) -> bool:
        probe.set_text(wrapped)
        bbox = probe.get_window_extent(renderer=renderer)
        return bbox.width / fig.dpi <= max_width_in

    wrapped = text
    n_chars = len(text)
    while not fits(wrapped) and n_chars > 20:
        n_chars = int(n_chars * 0.9)
        wrapped = textwrap.fill(text, n_chars)
    probe.remove()
    return wrapped


def make_figure(
    prevalence: pd.DataFrame,
    panels: list[tuple[str, list[str], list[str]]],
    width: float,
    height: float,
    base_size: float,
    footnote: str | None,
    share_y: bool = True,
    label_every: int = 2,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, height), sharey=share_y, squeeze=False
    )
    ends_per_panel = [
        draw_panel(ax, prevalence, labeled, context, title, base_size, label_every)
        for ax, (title, labeled, context) in zip(axes[0], panels)
    ]
    axes[0][0].set_ylabel("Share of fashion mentions")

    # Anchor at zero only once every panel has contributed its data — under
    # sharey, pinning the bottom mid-loop would freeze the top at whatever the
    # panels drawn so far reached, clipping the rest.
    for ax in axes[0]:
        ax.autoscale_view()
        ax.set_ylim(bottom=0)

    # Settle the layout before the direct labels go down: their spacing is
    # derived from the axes' final height, and under sharey the last panel can
    # still have widened the y-scale.
    wrapped = None
    if footnote:
        # Wrap by measuring the rendered text rather than guessing at character
        # widths, then reserve exactly the height the block needs.
        note_size = base_size - 1.5
        wrapped = _wrap_to_width(fig, footnote, note_size, width * 0.99)
        note_height = (wrapped.count("\n") + 1) * note_size * 1.4 / 72 / height
        fig.tight_layout(rect=(0, note_height + 0.01, 1, 1))
    else:
        fig.tight_layout()

    x_max = DECADES[-1] * 10
    label_pad = (DECADES[-1] - DECADES[0]) * 10 * 0.02
    for ax, ends in zip(axes[0], ends_per_panel):
        format_percent_axis(ax)
        _draw_end_labels(ax, ends, x_max, label_pad, base_size)

    if wrapped:
        fig.text(
            0.005,
            0.005,
            wrapped,
            fontsize=base_size - 1.5,
            color=INK_MUTED,
            ha="left",
            va="bottom",
            linespacing=1.4,
        )
    return fig


def save(fig: plt.Figure, path_stem: Path) -> None:
    """Write a vector PDF and a 300-dpi PNG, then close the figure.

    PNGs go on slides and posters, so they drop the white surface.
    """
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300, "transparent": True})):
        path = path_stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        print(f"Saved {path}")
    plt.close(fig)


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
        help="Override the input parquet path (default: from --debug).",
    )
    parser.add_argument(
        "--classifier-output",
        type=Path,
        default=CLASSIFIER_OUTPUT,
        help="coefficients.parquet (or .tsv) from gender_classifier.py.",
    )
    parser.add_argument(
        "--gender",
        choices=["all", "male", "female"],
        default="all",
        help=(
            "Restrict the counted mentions to one gender. Default 'all' keeps the "
            "denominator as every mention in the decade."
        ),
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=5,
        help="Top N gender-indexing terms per decade from the classifier (default: 5).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of terms per panel to color and label (default: 5, max 6).",
    )
    parser.add_argument(
        "--exclude-terms",
        nargs="*",
        default=[],
        metavar="TERM",
        help="Terms to drop from the classifier's selection (e.g. body parts).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=10,
        help="Max mentions sampled per book (default: 10; 0 disables sampling).",
    )
    parser.add_argument(
        "--sample-by",
        choices=["book", "book-term"],
        default="book",
        help=(
            "Sampling unit: 'book' caps mentions per book (preserves a book's term "
            "mix); 'book-term' caps each term per book (default: book)."
        ),
    )
    parser.add_argument(
        "--ci-method",
        choices=["wald", "wilson"],
        default="wald",
        help="Binomial interval: 'wald' is the MLE normal approximation (default).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.95, help="Confidence level (default: 0.95)."
    )
    parser.add_argument(
        "--no-context-lines",
        action="store_true",
        help="Drop the gray unlabeled lines and plot only the top-k terms.",
    )
    parser.add_argument(
        "--free-y",
        action="store_true",
        help=(
            "Give each panel its own y-scale. Off by default: both panels share a "
            "denominator, so a shared scale keeps them directly comparable."
        ),
    )
    parser.add_argument(
        "--no-footnote",
        action="store_true",
        help="Omit the method footnote under the panels.",
    )
    parser.add_argument(
        "--width", type=float, default=7.0, help="Figure width in inches (default: 7)."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=3.1,
        help="Figure height in inches (default: 3.1).",
    )
    parser.add_argument(
        "--font-size", type=float, default=8.0, help="Base font size (default: 8)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Sampling seed (default: 0)."
    )
    parser.add_argument(
        "--version",
        choices=["both", "full", "simple"],
        default="both",
        help="Which figure(s) to write (default: both).",
    )
    parser.add_argument(
        "--simple-top-k",
        type=int,
        default=3,
        help="Terms per panel on the slide figure (default: 3).",
    )
    parser.add_argument(
        "--simple-width",
        type=float,
        default=10.0,
        help="Slide figure width in inches (default: 10, i.e. 16:9 at 5.625).",
    )
    parser.add_argument(
        "--simple-height",
        type=float,
        default=5.625,
        help="Slide figure height in inches (default: 5.625).",
    )
    parser.add_argument(
        "--simple-font-size",
        type=float,
        default=16.0,
        help="Slide figure base font size (default: 16).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the PDF and PNG preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.classifier_output.exists():
        raise FileNotFoundError(
            f"Classifier output not found: {args.classifier_output}\n"
            "Run `python -m fashion.analysis.gender_classifier` first."
        )
    if args.top_k > len(SERIES_COLORS):
        raise ValueError(
            f"--top-k is capped at {len(SERIES_COLORS)} (the categorical palette's "
            "validated length); use small multiples for more series."
        )

    input_path = args.input or (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df = pd.read_parquet(
        input_path,
        columns=["mention_id", "term", "decade", "gender_dep_pron"],
    ).reset_index()
    df = df.drop_duplicates(subset=["book_id", "mention_id"])
    df = df[df.decade.isin(DECADES)]
    if args.gender != "all":
        df = df[df.gender_dep_pron.eq(args.gender)]
    df["term"] = df["term"].str.lower()

    by = ["book_id"] if args.sample_by == "book" else ["book_id", "term"]
    sampled = sample_mentions(df, args.sample_n, args.seed, by)
    print(f"{len(df):,} mentions → {len(sampled):,} after sampling ({args.sample_by})")

    prevalence = compute_prevalence(sampled, conf=args.conf, ci_method=args.ci_method)
    coef_df = load_coefficients(args.classifier_output)

    panels = []
    excluded = {t.lower() for t in args.exclude_terms}
    available = set(prevalence.term)
    for gender in ("male", "female"):
        terms = get_top_terms(coef_df, gender, args.top_terms)
        present = [t for t in terms if t in available and t not in excluded]
        labeled = rank_terms(prevalence, present, args.top_k)
        context = (
            [] if args.no_context_lines else [t for t in present if t not in labeled]
        )
        panels.append((PANEL_TITLES[gender], labeled, context))
        print(f"{gender}: labeling {labeled}, {len(context)} context lines")

    footnote = None
    if not args.no_footnote:
        conf_pct = round(args.conf * 100)
        interval = "Wilson" if args.ci_method == "wilson" else "normal-approximation"
        clauses = [
            f"Bands are {conf_pct}% binomial {interval} intervals",
            "terms selected by per-decade gender classifier coefficients",
        ]
        if args.sample_n > 0:
            unit = "book" if args.sample_by == "book" else "book and term"
            clauses.append(f"at most {args.sample_n} mentions sampled per {unit}")
        if not args.no_context_lines:
            clauses.append("gray lines are unlabeled terms from the same set")
        footnote = "; ".join(clauses) + "."

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"term_prevalence_{args.gender}"

    if args.version in ("both", "full"):
        set_style(args.font_size)
        fig = make_figure(
            prevalence,
            panels,
            args.width,
            args.height,
            args.font_size,
            footnote,
            share_y=not args.free_y,
        )
        save(fig, stem)

    if args.version in ("both", "simple"):
        # Slides keep each panel's leading terms (a prefix, so colors match the
        # full figure) and drop the gray context.
        simple_panels = [
            (title, labeled[: args.simple_top_k], [])
            for title, labeled, _context in panels
        ]
        set_style(args.simple_font_size)
        fig = make_figure(
            prevalence,
            simple_panels,
            args.simple_width,
            args.simple_height,
            args.simple_font_size,
            None,
            share_y=not args.free_y,
            # Large type fits a decade label only every 40 years.
            label_every=4,
        )
        save(fig, stem.with_name(stem.name + "_simple"))


if __name__ == "__main__":
    main()
