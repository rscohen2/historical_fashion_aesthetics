"""Scatter characters on two principal components, colored by gender.

Reads the per-character scores from pca_axes.py and plots each male or female
character as a dot on two components (PC1 × PC2 by default), with each gender's
density along both axes in the margins. The margins carry the argument: along
the gender component the two curves pull apart, along the other they overlap.

Each axis is labeled with the fashion terms at its two ends (by loading, among
stems with > --min-term-mentions mentions), so it reads without PCA vocabulary.
The plotted range is trimmed to central quantiles of the scores, since a few
heavily described characters sit far from the rest; how many fall outside is
stated in the footnote.

Two versions are written:
  - full:   paper/poster figure with a method footnote.
  - simple: slide figure (16:9) with large type and no footnote.

Requires: data/analysis/pca_axes/{run}/characters.parquet, loadings.csv,
          pc_summary.csv, where {run} is the centering plus _{normalize} unless
          the run used raw counts (e.g. term, term_l2)
Output:   data/analysis/figures/pca_plot/pca_{x}_{y}_{run}.pdf (+ .png)
          data/analysis/figures/pca_plot/pca_{x}_{y}_{run}_simple.pdf (+ .png)
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from fashion.analysis.pca_axes import run_name
from fashion.analysis.term_prevalence import (
    AXIS,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    SERIES_COLORS,
    _wrap_to_width,
    save,
    set_style,
)
from fashion.paths import DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT = "pca_plot"
OUTPUT_DIR = DATA_DIR / "analysis" / "figures" / EXPERIMENT
PCA_DIR = DATA_DIR / "analysis" / "pca_axes"

# Categorical slots 1 and 2, in fixed order.
GENDER_COLORS = {"male": SERIES_COLORS[0], "female": SERIES_COLORS[1]}


def load(pca_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chars = pd.read_parquet(pca_dir / "characters.parquet")
    loadings = pd.read_csv(pca_dir / "loadings.csv", index_col="term")
    summary = pd.read_csv(pca_dir / "pc_summary.csv", index_col="pc")
    return chars[chars.gender.isin(GENDER_COLORS)], loadings, summary


def end_terms(
    loadings: pd.DataFrame, pc: str, n: int, min_mentions: int
) -> tuple[list[str], list[str]]:
    """The `n` most negative and most positive stems on `pc`, among common ones."""
    common = loadings[loadings.mentions > min_mentions][pc].sort_values()
    return list(common.index[:n]), list(common.index[::-1][:n])


def axis_label(
    pc: str, variance: float, low: list[str], high: list[str], compact: bool
) -> str:
    """'PC2 (10.1% of variance)' over '← coat, trousers · dress, silk →'.

    `compact` (slides) keeps only the terms line, which carries the meaning.
    """
    ends = f"← {', '.join(low)}   ·   {', '.join(high)} →"
    return ends if compact else f"{pc} ({variance:.1%} of variance)\n{ends}"


def trimmed_limits(values: np.ndarray, trim: float) -> tuple[float, float]:
    lo, hi = np.quantile(values, [trim, 1 - trim])
    pad = (hi - lo) * 0.04
    return lo - pad, hi + pad


def draw_density(
    ax: plt.Axes,
    chars: pd.DataFrame,
    pc: str,
    limits: tuple[float, float],
    vertical: bool,
    label_peaks: bool,
    base_size: float,
) -> None:
    """Each gender's kernel density along `pc`, as a line over a light wash."""
    grid = np.linspace(*limits, 400)
    for gender, color in GENDER_COLORS.items():
        values = chars.loc[chars.gender.eq(gender), pc].to_numpy()
        inside = values[(values >= limits[0]) & (values <= limits[1])]
        density = gaussian_kde(inside)(grid)
        if vertical:
            ax.fill_betweenx(grid, 0, density, color=color, alpha=0.12, linewidth=0)
            ax.plot(density, grid, color=color, linewidth=1.5)
        else:
            ax.fill_between(grid, 0, density, color=color, alpha=0.12, linewidth=0)
            ax.plot(grid, density, color=color, linewidth=1.5)
        if label_peaks:
            peak = int(density.argmax())
            xy = (
                (density[peak], grid[peak]) if vertical else (grid[peak], density[peak])
            )
            ax.annotate(
                gender,
                xy=xy,
                xytext=(base_size * 0.4, 0) if vertical else (0, base_size * 0.4),
                textcoords="offset points",
                ha="left" if vertical else "center",
                va="center" if vertical else "bottom",
                fontsize=base_size,
                color=INK_PRIMARY,
            )
    if vertical:
        # Headroom past the peaks for the gender labels.
        ax.set_xlim(0, ax.get_xlim()[1] * (1.8 if label_peaks else 1.0))
    else:
        ax.set_ylim(bottom=0)
    ax.set_axis_off()


def make_figure(
    chars: pd.DataFrame,
    loadings: pd.DataFrame,
    summary: pd.DataFrame,
    x_pc: str,
    y_pc: str,
    width: float,
    height: float,
    base_size: float,
    trim: float,
    point_size: float,
    n_terms: int,
    min_term_mentions: int,
    title: str | None,
    footnote: str | None,
    seed: int,
    compact: bool = False,
) -> plt.Figure:
    fig = plt.figure(figsize=(width, height))
    note_in = 0.0
    wrapped = None
    if footnote:
        note_size = base_size - 1.5
        wrapped = _wrap_to_width(fig, footnote, note_size, width * 0.97)
        note_in = (wrapped.count("\n") + 1) * note_size * 1.45 / 72 + 0.1
    title_in = base_size * 2.4 / 72 if title else base_size * 0.6 / 72
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(5, 1.3),
        height_ratios=(1, 5),
        wspace=0.03,
        hspace=0.03,
        left=base_size * (3.5 if compact else 5.5) / 72 / width,
        right=0.98,
        top=1 - title_in / height,
        bottom=(base_size * (3.5 if compact else 5.0) / 72 + note_in) / height,
    )
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    side = fig.add_subplot(grid[1, 1], sharey=ax)

    x_lim = trimmed_limits(chars[x_pc].to_numpy(), trim)
    y_lim = trimmed_limits(chars[y_pc].to_numpy(), trim)

    # Shuffle so neither gender is drawn wholesale over the other.
    shuffled = chars.sample(frac=1.0, random_state=seed)
    ax.scatter(
        shuffled[x_pc],
        shuffled[y_pc],
        c=shuffled.gender.map(GENDER_COLORS),
        s=point_size,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.axhline(0, color=AXIS, linewidth=0.6, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=0.6, zorder=1)
    ax.grid(color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    renderer = fig.canvas.get_renderer()
    for pc, axis in ((x_pc, ax.xaxis), (y_pc, ax.yaxis)):
        # Name as many end terms as fit along the axis, up to n_terms.
        variance = summary.loc[pc, "variance_explained"]
        room = ax.get_window_extent(renderer)
        room = room.width if axis is ax.xaxis else room.height
        for n in range(n_terms, 0, -1):
            low, high = end_terms(loadings, pc, n, min_term_mentions)
            axis.set_label_text(axis_label(pc, variance, low, high, compact))
            axis.label.set_linespacing(1.6)
            extent = axis.label.get_window_extent(renderer)
            if (extent.width if axis is ax.xaxis else extent.height) <= room:
                break

    # Tick labels vary in width (−4 vs −0.40), so widen the left margin until
    # the y label clears the figure edge.
    fig.canvas.draw()
    overflow_px = (
        base_size * 0.6 / 72 * fig.dpi - ax.yaxis.label.get_window_extent(renderer).x0
    )
    if overflow_px > 0:
        grid.update(left=grid.left + overflow_px / fig.bbox.width)

    draw_density(top, chars, x_pc, x_lim, False, False, base_size)
    draw_density(side, chars, y_pc, y_lim, True, True, base_size)

    if title:
        fig.text(
            base_size * 0.6 / 72 / width,
            1 - base_size * 0.8 / 72 / height,
            title,
            ha="left",
            va="top",
            fontsize=base_size * 1.3,
            fontweight="bold",
            color=INK_PRIMARY,
        )
    if wrapped:
        fig.text(
            base_size * 0.6 / 72 / width,
            0.08 / height,
            wrapped,
            fontsize=base_size - 1.5,
            color=INK_MUTED,
            ha="left",
            va="bottom",
            linespacing=1.4,
        )
    return fig


def build_footnote(
    chars: pd.DataFrame,
    summary: pd.DataFrame,
    x_pc: str,
    y_pc: str,
    centering: str,
    normalize: str,
    trim: float,
) -> str:
    counts = chars.gender.value_counts()
    x_lim = trimmed_limits(chars[x_pc].to_numpy(), trim)
    y_lim = trimmed_limits(chars[y_pc].to_numpy(), trim)
    outside = int((~chars[x_pc].between(*x_lim) | ~chars[y_pc].between(*y_lim)).sum())
    centered = "term-centered" if centering == "term" else "character-centered"
    rows = (
        "unit-length character vectors of term-stem mention counts"
        if normalize == "l2"
        else "character × term-stem mention counts"
    )
    tau = {
        pc: f"{summary.loc[pc, 'gender_tau']:.2f}".replace("-", "\u2212")
        for pc in (x_pc, y_pc)
    }
    return (
        "Each dot is a named character with at least 10 fashion mentions "
        f"({counts.get('male', 0):,} male, {counts.get('female', 0):,} female; "
        "characters without a male/female label are omitted). Components from a "
        f"{centered} PCA of {rows}; axis ends list "
        "the most strongly loading common terms. Kendall's τ with gender: "
        f"{x_pc} {tau[x_pc]}, {y_pc} {tau[y_pc]}. Margins are kernel "
        f"density estimates. {outside:,} character{'' if outside == 1 else 's'} "
        f"beyond the {trim * 100:g}th–"
        f"{(1 - trim) * 100:g}th percentiles are not shown."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centering",
        choices=["term", "row"],
        default="term",
        help="Which pca_axes.py run to plot (default: term).",
    )
    parser.add_argument(
        "--normalize",
        choices=["counts", "l2"],
        default="counts",
        help="Which pca_axes.py normalization to plot (default: counts).",
    )
    parser.add_argument("--x", default="PC1", help="Component on x (default: PC1).")
    parser.add_argument("--y", default="PC2", help="Component on y (default: PC2).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="pca_axes.py output for one centering (default: from --centering).",
    )
    parser.add_argument(
        "--version",
        choices=["both", "full", "simple"],
        default="both",
        help="Which figure(s) to write (default: both).",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.005,
        help="Quantile trimmed from each end of each axis (default: 0.005).",
    )
    parser.add_argument(
        "--n-terms",
        type=int,
        default=3,
        help="Terms named at each end of each axis (default: 3).",
    )
    parser.add_argument(
        "--min-term-mentions",
        type=int,
        default=1000,
        help="Only stems with more mentions are named on the axes (default: 1000).",
    )
    parser.add_argument(
        "--title",
        default="Fashion vocabulary separates characters by gender",
        help="Figure title.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the title (e.g. when the paper caption carries it).",
    )
    parser.add_argument(
        "--no-footnote", action="store_true", help="Omit the method footnote."
    )
    parser.add_argument(
        "--width", type=float, default=6.0, help="Full figure width (default: 6 in)."
    )
    parser.add_argument(
        "--height", type=float, default=6.0, help="Full figure height (default: 6 in)."
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=8.0,
        help="Full figure font size (default: 8).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Dot area in pt² on the full figure (default: 2).",
    )
    parser.add_argument(
        "--simple-width", type=float, default=10.0, help="Slide width (default: 10 in)."
    )
    parser.add_argument(
        "--simple-height",
        type=float,
        default=5.625,
        help="Slide height (default: 5.625 in).",
    )
    parser.add_argument(
        "--simple-font-size",
        type=float,
        default=14.0,
        help="Slide font size (default: 14).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Draw-order seed.")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = run_name(args.centering, args.normalize)
    input_dir = args.input_dir or PCA_DIR / name
    if not (input_dir / "characters.parquet").exists():
        raise FileNotFoundError(
            f"No PCA results in {input_dir}.\n"
            "Run `python -m fashion.analysis.pca_axes "
            f"--centering {args.centering} --normalize {args.normalize}` first."
        )
    chars, loadings, summary = load(input_dir)
    for pc in (args.x, args.y):
        if pc not in chars.columns:
            raise ValueError(
                f"{pc} not in {input_dir}; available: {list(summary.index)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"pca_{args.x.lower()}_{args.y.lower()}_{name}"
    title = None if args.no_title else args.title
    common = {
        "chars": chars,
        "loadings": loadings,
        "summary": summary,
        "x_pc": args.x,
        "y_pc": args.y,
        "trim": args.trim,
        "n_terms": args.n_terms,
        "min_term_mentions": args.min_term_mentions,
        "title": title,
        "seed": args.seed,
    }

    if args.version in ("both", "full"):
        footnote = None
        if not args.no_footnote:
            footnote = build_footnote(
                chars,
                summary,
                args.x,
                args.y,
                args.centering,
                args.normalize,
                args.trim,
            )
        set_style(args.font_size)
        fig = make_figure(
            width=args.width,
            height=args.height,
            base_size=args.font_size,
            point_size=args.point_size,
            footnote=footnote,
            **common,
        )
        save(fig, stem)

    if args.version in ("both", "simple"):
        set_style(args.simple_font_size)
        fig = make_figure(
            width=args.simple_width,
            height=args.simple_height,
            base_size=args.simple_font_size,
            point_size=args.point_size * 2,
            footnote=None,
            compact=True,
            **common,
        )
        save(fig, stem.with_name(stem.name + "_simple"))


if __name__ == "__main__":
    main()
