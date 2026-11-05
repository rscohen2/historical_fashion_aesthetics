"""Plot the character adjectives most associated with a fashion term as print-ready figures.

Reads the Monroe et al. (2008) log-odds results from logodds.py and, for each
requested fashion term, draws a diverging bar chart of z-scores (δ / σ):
adjectives over-represented among characters linked to the term point right
(blue), under-represented ones point left (red). Each adjective's name sits on
the opposite side of the zero line from its bar.

Two versions are written per term:
  - full:   paper/poster figure with a z-score axis, the ±z* significance band,
            a per-adjective character count column, and a method footnote.
  - simple: slide figure (16:9) with large type, fewer adjectives, no axis;
            z-scores inside the bars, counts as gray asides beside the names,
            and a one-line key when any bar is not significant.
Both select the top adjectives on each side by |z|.

With --gender, reads the gender-controlled results (logodds.py --gender) instead;
terms are then term stems.

Requires: data/analysis/logodds/logodds.json (or logodds_{gender}.json)
Output:   data/analysis/figures/logodds_plot/logodds_{term}[_{gender}].pdf (+ .png)
          data/analysis/figures/logodds_plot/logodds_{term}[_{gender}]_simple.pdf (+ .png)
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from scipy.stats import norm

from fashion.analysis.term_prevalence import (
    AXIS,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    _wrap_to_width,
    save,
    set_style,
)
from fashion.paths import DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EXPERIMENT = "logodds_plot"
OUTPUT_DIR = DATA_DIR / "analysis" / "figures" / EXPERIMENT
LOGODDS_DIR = DATA_DIR / "analysis" / "logodds"

# Diverging pair (blue <-> red) from the reference palette; non-significant
# adjectives fall back to the neutral midpoint gray.
COLOR_OVER = "#2a78d6"
COLOR_UNDER = "#e34948"
COLOR_NEUTRAL = "#c3c2b7"
BAND = "#f0efec"

GROUP_LABELS = {
    "over": "Over-represented",
    "under": "Under-represented",
}


def load_term(path: Path, term: str) -> tuple[pd.DataFrame, dict]:
    """The term's rows plus the setup that produced them.

    Files from before logodds.py recorded its setup counted one unit per
    (book, character, term) across all characters.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    setup = data.get("setup", {"gender": None, "unit": "character"})
    df = pd.DataFrame(data["logodds"])
    rows = df[df.term.eq(term)]
    if rows.empty:
        available = ", ".join(sorted(df.term.unique()))
        raise ValueError(
            f"No log-odds results for {term!r} in {path}.\n"
            f"Available terms: {available}\n"
            "Re-run `python -m fashion.analysis.logodds --top-terms N` with a "
            "larger N to include more terms."
        )
    return rows, setup


def select_rows(
    rows: pd.DataFrame, top_n: int, exclude: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top `top_n` adjectives on each side by |z|, each sorted for display."""
    rows = rows[~rows.adjective.isin(exclude)]
    over = rows[rows.score > 0].nlargest(top_n, "score")
    under = rows[rows.score < 0].nsmallest(top_n, "score")
    # Display order top-to-bottom: strongest over-represented first, then the
    # under-represented group ending in its strongest at the bottom.
    return over, under.sort_values("score", ascending=False)


def _y_positions(n_over: int, n_under: int, group_gap: float) -> tuple[list, list]:
    """Row centers counting down from 0, with `group_gap` rows between groups."""
    over_y = [-i for i in range(n_over)]
    start = -(n_over + group_gap) if n_over else 0
    under_y = [start - i for i in range(n_under)]
    return over_y, under_y


def _text_width_data(ax: plt.Axes, texts: list[plt.Text]) -> float:
    """Widest of `texts` converted to x data units at the current limits."""
    renderer = ax.figure.canvas.get_renderer()
    widest_px = max((t.get_window_extent(renderer).width for t in texts), default=0)
    x0, x1 = ax.get_xlim()
    return widest_px / ax.bbox.width * (x1 - x0)


def _compact(n: int) -> str:
    """47,772 -> 47.8k; counts under 1,000 stay exact."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _draw_inset_scores(
    ax: plt.Axes,
    bars: list[tuple[float, float, int, str]],
    base_size: float,
    pad_pt: float,
) -> None:
    """Each bar's z at its tip: inside when it fits, else just past the tip.

    Only the first bar spells out "z = "; the rest give the number alone.
    """
    renderer = ax.figure.canvas.get_renderer()
    x0, x1 = ax.get_xlim()
    px_per_unit = ax.bbox.width / (x1 - x0)
    pad_px = pad_pt / 72 * ax.figure.dpi
    for i, (z, y, _count, color) in enumerate(bars):
        sign = 1 if z > 0 else -1
        value = f"{z:.1f}".replace("-", "\u2212")
        text = ax.annotate(
            f"z = {value}" if i == 0 else value,
            xy=(z, y),
            xytext=(-sign * pad_pt, 0),
            textcoords="offset points",
            ha="right" if z > 0 else "left",
            va="center",
            fontsize=base_size * 0.8,
            # White clears contrast on the blue and red fills; the light gray
            # neutral fill needs ink instead.
            color=INK_PRIMARY if color == COLOR_NEUTRAL else SURFACE,
            zorder=6,
        )
        width_px = text.get_window_extent(renderer).width
        if width_px + 2 * pad_px > abs(z) * px_per_unit:
            text.xyann = (sign * pad_pt, 0)
            text.set_ha("left" if z > 0 else "right")
            text.set_color(INK_SECONDARY)


def draw_chart(
    ax: plt.Axes,
    over: pd.DataFrame,
    under: pd.DataFrame,
    z_star: float,
    base_size: float,
    bar_height: float,
    show_axis: bool,
    show_band: bool,
    show_counts: bool,
    min_rows: float = 0.0,
    count_label: str = "Characters",
    inset_scores: bool = False,
) -> None:
    group_gap = 1.4
    over_y, under_y = _y_positions(len(over), len(under), group_gap)

    labels: list[plt.Text] = []
    counts: list[plt.Text] = []
    bars: list[tuple[float, float, int, str]] = []
    for frame, ys in ((over, over_y), (under, under_y)):
        for (_, row), y in zip(frame.iterrows(), ys):
            z = row.score
            if abs(z) < z_star:
                color = COLOR_NEUTRAL
            else:
                color = COLOR_OVER if z > 0 else COLOR_UNDER
            ax.barh(y, z, height=bar_height, color=color, linewidth=0, zorder=3)
            bars.append((z, y, int(row["count"]), color))
            if inset_scores:
                # Slides carry the count as a muted aside beside the name, on
                # the side away from the bar, so it isn't read as the bar's value.
                unit = f" {count_label.lower()}" if not counts else ""
                counts.append(
                    ax.text(
                        0,
                        y,
                        _compact(int(row["count"])) + unit,
                        ha="right" if z > 0 else "left",
                        va="center",
                        color=INK_MUTED,
                        fontsize=base_size * 0.72,
                        zorder=5,
                    )
                )
            labels.append(
                ax.text(
                    0,
                    y,
                    row.adjective,
                    ha="right" if z > 0 else "left",
                    va="center",
                    color=INK_PRIMARY,
                    fontsize=base_size,
                    zorder=5,
                )
            )

    # Group headings double as the legend: a swatch in the group's color beside
    # ink-colored text, set just above each group's first row.
    headings = []
    for key, ys, color in (
        ("over", over_y, COLOR_OVER),
        ("under", under_y, COLOR_UNDER),
    ):
        if ys:
            headings.append((GROUP_LABELS[key], ys[0] + 0.5 + group_gap * 0.35, color))

    # Keep the significance band in view even when bars are short.
    x_max = max([abs(z) for z in pd.concat([over.score, under.score])] + [z_star * 1.2])
    # Reserving `min_rows` keeps bar thickness fixed when a group comes up short.
    y_top = headings[0][1] + 0.6
    ax.set_ylim(min(min(over_y + under_y) - 0.7, y_top - min_rows), y_top)

    # Names sit across the zero line from their bars, so each side needs room
    # for its longest bar *and* the longest label on the opposite side. The
    # data-per-inch ratio depends on the limits, so settle them iteratively.
    pad_pt = base_size * 0.45
    left_bar = max([abs(z) for z in under.score], default=0)
    right_bar = max([z for z in over.score], default=0)
    gap_pt = base_size * 0.5
    over_labels = labels[: len(over)]
    under_labels = labels[len(over) :]
    renderer = ax.figure.canvas.get_renderer()

    def side_width(texts: list[plt.Text], asides: list[plt.Text]) -> float:
        """Widest name (plus its count aside, if any) in x data units."""
        x0, x1 = ax.get_xlim()
        gap_px = gap_pt / 72 * ax.figure.dpi
        widest = 0.0
        for i, text in enumerate(texts):
            width = text.get_window_extent(renderer).width
            if asides:
                width += gap_px + asides[i].get_window_extent(renderer).width
            widest = max(widest, width)
        return widest / ax.bbox.width * (x1 - x0)

    over_counts = counts[: len(over)]
    under_counts = counts[len(over) :]
    left, right = x_max, x_max
    for _ in range(4):
        ax.set_xlim(-left, right)
        ax.figure.canvas.draw()
        pad = pad_pt / 72 * ax.figure.dpi / ax.bbox.width * (left + right)
        left = max(left_bar, side_width(over_labels, over_counts) + pad) * 1.04
        right = max(right_bar, side_width(under_labels, under_counts) + pad) * 1.04
    ax.set_xlim(-left, right)
    pad = pad_pt / 72 * ax.figure.dpi / ax.bbox.width * (left + right)
    for text in over_labels:
        text.set_x(-pad)
    for text in under_labels:
        text.set_x(pad)

    if inset_scores:
        # Each count sits just beyond its name, measured in points so it holds
        # regardless of the x-scale.
        ax.figure.canvas.draw()
        for text, aside in zip(labels, counts):
            name_pt = text.get_window_extent(renderer).width * 72 / ax.figure.dpi
            sign = -1 if text.get_ha() == "right" else 1
            aside.set_x(0)
            aside.set_transform(
                ax.transData
                + mpl.transforms.ScaledTranslation(
                    sign * (pad_pt + name_pt + gap_pt) / 72,
                    0,
                    ax.figure.dpi_scale_trans,
                )
            )
        _draw_inset_scores(ax, bars, base_size, pad_pt)

    for text, y, color in headings:
        ax.annotate(
            text,
            xy=(0, y),
            xycoords=("axes fraction", "data"),
            xytext=(base_size * 1.25, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=base_size,
            color=INK_SECONDARY,
            fontweight="bold",
        )
        ax.plot(
            [0.0],
            [y],
            transform=mpl.transforms.blended_transform_factory(
                ax.transAxes, ax.transData
            )
            + mpl.transforms.ScaledTranslation(
                base_size * 0.55 / 72, 0, ax.figure.dpi_scale_trans
            ),
            marker="s",
            markersize=base_size * 0.7,
            color=color,
            markeredgewidth=0,
            clip_on=False,
        )

    # The zero line and band run through each group's rows only, leaving the
    # heading rows clear.
    for ys in (over_y, under_y):
        if not ys:
            continue
        top, bottom = ys[0] + 0.5, ys[-1] - 0.5
        ax.plot([0, 0], [bottom, top], color=AXIS, linewidth=0.8, zorder=4)
        if show_band:
            ax.fill_betweenx(
                [bottom, top], -z_star, z_star, color=BAND, linewidth=0, zorder=1
            )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    if show_axis:
        ax.spines["bottom"].set_color(AXIS)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=True, integer=True))
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])

    if show_counts:
        count_x = 1.0
        trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        offset = mpl.transforms.ScaledTranslation(
            base_size * 4.6 / 72, 0, ax.figure.dpi_scale_trans
        )
        for frame, ys in ((over, over_y), (under, under_y)):
            for count, y in zip(frame["count"], ys):
                ax.text(
                    count_x,
                    y,
                    f"{count:,}",
                    transform=trans + offset,
                    ha="right",
                    va="center",
                    fontsize=base_size - 1,
                    color=INK_MUTED,
                    clip_on=False,
                )
        ax.text(
            count_x,
            headings[0][1],
            count_label,
            transform=trans + offset,
            ha="right",
            va="center",
            fontsize=base_size - 1,
            color=INK_SECONDARY,
            clip_on=False,
        )


def make_figure(
    term: str,
    over: pd.DataFrame,
    under: pd.DataFrame,
    z_star: float,
    width: float,
    height: float | None,
    base_size: float,
    simple: bool,
    title: str | None,
    footnote: str | None,
    top_n: int = 0,
    count_label: str = "Characters",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(width, height or 1.0))
    right = 1 - base_size * 5.0 / 72 / width if not simple else 0.98
    title_in = base_size * 2.6 / 72 if title else base_size * 0.6 / 72
    axis_in = base_size * (1.0 if simple else 4.0) / 72
    # Slides have no footnote, so a one-line key explains any gray bars.
    has_gray = bool(pd.concat([over.score, under.score]).abs().lt(z_star).any())
    gray_key = simple and has_gray
    if gray_key:
        axis_in += base_size * 2.0 / 72
    note_in = 0.0
    wrapped = None
    if footnote:
        note_size = base_size - 1.5
        wrapped = _wrap_to_width(fig, footnote, note_size, width * 0.97)
        note_in = (wrapped.count("\n") + 1) * note_size * 1.45 / 72 + 0.08
    if height is None:
        # Fit the axes to its rows: bars, plus each group's heading row and the
        # gap between groups (see draw_chart). The pitch scales with the font.
        n_groups = int(not over.empty) + int(not under.empty)
        n_rows = len(over) + len(under) + 1.4 * n_groups + 0.6
        height = n_rows * base_size * 2.0 / 72 + title_in + axis_in + note_in
        fig.set_size_inches(width, height)

    fig.subplots_adjust(
        left=0.02,
        right=right,
        top=1 - title_in / height,
        bottom=(axis_in + note_in) / height,
    )

    draw_chart(
        ax,
        over,
        under,
        z_star,
        base_size,
        bar_height=0.66 if simple else 0.62,
        show_axis=not simple,
        show_band=not simple,
        show_counts=not simple,
        min_rows=2 * top_n + 3.0 if simple else 0.0,
        count_label=count_label,
        inset_scores=simple,
    )
    if gray_key:
        key_size = base_size * 0.8
        key_y = base_size * 0.9 / 72 / height
        fig.text(
            0.02,
            key_y,
            "\u25a0",
            color=COLOR_NEUTRAL,
            fontsize=key_size,
            ha="left",
            va="center",
        )
        fig.text(
            0.02 + key_size * 1.3 / 72 / width,
            key_y,
            f"Not significant (|z| < {z_star:.2f})",
            color=INK_MUTED,
            fontsize=key_size,
            ha="left",
            va="center",
        )

    if not simple:
        ax.set_xlabel(
            f"Association with “{term}” (log-odds z-score)",
            labelpad=base_size * 0.4,
        )
    if title:
        fig.text(
            0.02,
            1 - base_size * 0.9 / 72 / height,
            title,
            ha="left",
            va="top",
            fontsize=base_size * (1.35 if simple else 1.15),
            fontweight="bold",
            color=INK_PRIMARY,
        )
    if wrapped:
        fig.text(
            0.02,
            0.08 / height,
            wrapped,
            fontsize=base_size - 1.5,
            color=INK_MUTED,
            ha="left",
            va="bottom",
            linespacing=1.4,
        )
    return fig


def slugify(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "terms",
        nargs="+",
        metavar="TERM",
        help=(
            "Fashion term(s) to plot, as they appear in the log-odds JSON (e.g. hat; "
            "term stems with --gender)."
        ),
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default=None,
        help="Plot the gender-controlled results from `logodds.py --gender`.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Log-odds JSON (default: logodds.json, or logodds_{gender}.json).",
    )
    parser.add_argument(
        "--version",
        choices=["both", "full", "simple"],
        default="both",
        help="Which figure(s) to write (default: both).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Adjectives per side in the full figure (default: 12).",
    )
    parser.add_argument(
        "--simple-top-n",
        type=int,
        default=6,
        help="Adjectives per side in the simple figure (default: 6).",
    )
    parser.add_argument(
        "--exclude-adjectives",
        nargs="*",
        default=[],
        metavar="ADJ",
        help="Adjectives to drop (e.g. parser noise like 'sure' or 'able').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.95,
        help="Two-sided confidence for the significance band (default: 0.95).",
    )
    parser.add_argument(
        "--title",
        default="{who} linked to \u201c{term}\u201d",
        help=(
            "Title template; {term} and {who} ('Character adjectives', or e.g. "
            "'Male character adjectives' with --gender) are filled in."
        ),
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure title (e.g. when the paper caption carries it).",
    )
    parser.add_argument(
        "--no-footnote",
        action="store_true",
        help="Omit the method footnote on the full figure.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=3.5,
        help="Full figure width in inches (default: 3.5, one paper column).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Full figure height in inches (default: fit to the rows).",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=8.0,
        help="Full figure base font size (default: 8).",
    )
    parser.add_argument(
        "--simple-width",
        type=float,
        default=10.0,
        help="Simple figure width in inches (default: 10, i.e. 16:9 at 5.625).",
    )
    parser.add_argument(
        "--simple-height",
        type=float,
        default=5.625,
        help="Simple figure height in inches (default: 5.625).",
    )
    parser.add_argument(
        "--simple-font-size",
        type=float,
        default=16.0,
        help="Simple figure base font size (default: 16).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the PDFs and PNGs.",
    )
    return parser.parse_args()


def describe_setup(setup: dict, term: str) -> tuple[str, str, str]:
    """Count-column header, footnote comparison clause, and count definition."""
    gender = setup.get("gender")
    if setup.get("unit") == "mention":
        comparison = (
            f"character adjectives in mentions of \u201c{term}\u201d vs. mentions "
            "of other fashion terms"
        )
        note = "Mentions = mentions of the term whose character carries the adjective"
        return "Mentions", comparison, note
    if gender:
        comparison = (
            f"adjectives describing {gender} characters linked to the term stem "
            f"\u201c{term}\u201d vs. {gender} characters linked to other stems"
        )
        note = (
            "Characters = (book, character) pairs linked to the term whose first "
            "mention carries the adjective"
        )
        return "Characters", comparison, note
    comparison = (
        f"adjectives describing characters linked to \u201c{term}\u201d vs. "
        "characters linked to other fashion terms"
    )
    note = (
        "Characters = distinct (book, character) pairs with both the term and "
        "the adjective"
    )
    return "Characters", comparison, note


def main() -> None:
    args = parse_args()
    if args.input is None:
        name = "logodds.json" if args.gender is None else f"logodds_{args.gender}.json"
        args.input = LOGODDS_DIR / name
    if not args.input.exists():
        raise FileNotFoundError(
            f"Log-odds output not found: {args.input}\n"
            "Run `python -m fashion.analysis.logodds"
            + ("" if args.gender is None else f" --gender {args.gender}")
            + "` first."
        )

    z_star = float(norm.ppf(1 - (1 - args.conf) / 2))
    exclude = {a.lower() for a in args.exclude_adjectives}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for term in args.terms:
        rows, setup = load_term(args.input, term)
        if setup.get("gender") != args.gender:
            raise ValueError(
                f"{args.input} holds gender={setup.get('gender')!r} results, but "
                f"--gender is {args.gender!r}."
            )
        count_label, comparison, count_note = describe_setup(setup, term)
        who = (
            "Character adjectives"
            if args.gender is None
            else f"{args.gender.title()} character adjectives"
        )
        suffix = "" if args.gender is None else f"_{args.gender}"
        stem = args.output_dir / f"logodds_{slugify(term)}{suffix}"

        if args.version in ("both", "full"):
            over, under = select_rows(rows, args.top_n, exclude)
            n_gray = int(pd.concat([over.score, under.score]).abs().lt(z_star).sum())
            footnote = None
            if not args.no_footnote:
                conf_pct = round(args.conf * 100)
                footnote = (
                    "Monroe et al. (2008) log-odds with an informative Dirichlet "
                    f"prior: {comparison}. Bars show z = δ/σ; the shaded band is "
                    f"|z| < {z_star:.2f} ({conf_pct}%). {count_note}."
                )
                if n_gray:
                    footnote += " Gray bars fall inside the band (not significant)."
            title = None if args.no_title else args.title.format(term=term, who=who)
            set_style(args.font_size)
            fig = make_figure(
                term,
                over,
                under,
                z_star,
                args.width,
                args.height,
                args.font_size,
                simple=False,
                title=title,
                footnote=footnote,
                count_label=count_label,
            )
            save(fig, stem)

        if args.version in ("both", "simple"):
            over, under = select_rows(rows, args.simple_top_n, exclude)
            title = None if args.no_title else args.title.format(term=term, who=who)
            set_style(args.simple_font_size)
            fig = make_figure(
                term,
                over,
                under,
                z_star,
                args.simple_width,
                args.simple_height,
                args.simple_font_size,
                simple=True,
                title=title,
                footnote=None,
                top_n=args.simple_top_n,
                count_label=count_label,
            )
            save(fig, stem.with_name(stem.name + "_simple"))


if __name__ == "__main__":
    main()
