"""
entropy_temporal.py
-------------------
Computes entropy over SAE feature activation data stored as cumulative
decade counts — the format produced by the historical fashion aesthetics
pipeline.

Input JSON format:
  {
    "mentions": [cum_0, cum_1, ..., cum_11, undated_total],
    "codes": {
      "<feature_id>": [cum_0, cum_1, ..., cum_11, undated_total],
      ...
    }
  }

The first 12 values are cumulative counts across decade bins (default
1800s–1910s). The 13th value is a separate aggregate (undated books)
and is kept as its own column but excluded from temporal entropy.

Three entropy measures are computed:

  1. Entropy per FEATURE over decades
     — how evenly spread is each feature across time?
     Low  = feature is concentrated in one or two decades (a fad)
     High = feature fires uniformly across the century (a convention)

  2. Entropy per DECADE over features
     — how diverse is the feature mix in each decade?
     Low  = decade is dominated by a few features
     High = decade activates features broadly

  3. Activation RATE per feature per decade
     = feature_count[decade] / mentions[decade]
     Used to normalise for corpus size before computing entropy,
     and also saved as its own table for plotting.

Outputs (all in --output_dir):
  - decades.csv               : decade labels + mention counts + per-decade entropy
  - features.csv              : feature_id + temporal entropy + peak decade + trend
  - rates.csv                 : feature × decade activation rate matrix
  - counts.csv                : feature × decade raw count matrix
  - plots/feature_entropy_hist.png   : distribution of per-feature temporal entropy
  - plots/top_features_over_time.png : rate curves for highest/lowest entropy features
  - plots/decade_entropy.png         : per-decade diversity over time
  - plots/activation_heatmap.png     : feature × decade rate heatmap

Usage:
  python entropy_temporal.py --input activations.json \\
      [--term hat] \\
      [--start_year 1800] \\
      [--output_dir ./results] \\
      [--top_n 8] \\
      [--no_plots]
"""

import json
import argparse
import os

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to plain Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import pandas as pd
from scipy.stats import entropy as scipy_entropy, pearsonr


# ---------------------------------------------------------------------------
# 1. Load and decode
# ---------------------------------------------------------------------------

def load_activations(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def decode(data: dict, start_year: int = 1800) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert cumulative arrays → per-decade count matrices.

    Returns
    -------
    counts_df : DataFrame (features × decades), raw co-occurrence counts
    mentions  : Series (decades), total garment-term mentions per decade
    undated   : dict {"mentions": int, "codes": {fid: int}}
    """
    n_bins   = 12
    decades  = [f"{start_year + i * 10}s" for i in range(n_bins)]

    raw_mentions = np.array(data["mentions"])
    mentions_cum = raw_mentions[:n_bins]
    undated_mentions = int(raw_mentions[n_bins]) if len(raw_mentions) > n_bins else 0
    mentions_diff = np.diff(np.concatenate([[0], mentions_cum])).astype(int)

    rows = {}
    undated_codes = {}
    for fid, cum_arr in data["codes"].items():
        arr = np.array(cum_arr)
        feat_cum  = arr[:n_bins]
        undated_codes[fid] = int(arr[n_bins]) if len(arr) > n_bins else 0
        feat_diff = np.diff(np.concatenate([[0], feat_cum])).astype(int)
        rows[fid] = feat_diff

    counts_df = pd.DataFrame(rows, index=decades).T   # features × decades
    counts_df.index.name   = "feature_id"
    counts_df.columns.name = "decade"

    mentions_s = pd.Series(mentions_diff, index=decades, name="mentions")

    undated = {"mentions": undated_mentions, "codes": undated_codes}

    print(f"  Decoded: {len(counts_df)} features × {n_bins} decades")
    print(f"  Total dated mentions:   {mentions_diff.sum():,}")
    print(f"  Undated mentions:       {undated_mentions:,}  "
          f"({undated_mentions / (mentions_diff.sum() + undated_mentions) * 100:.1f}%)")
    return counts_df, mentions_s, undated


# ---------------------------------------------------------------------------
# 2. Rates
# ---------------------------------------------------------------------------

def compute_rates(counts_df: pd.DataFrame, mentions: pd.Series) -> pd.DataFrame:
    """
    Activation rate = feature_count / mention_count per decade.
    Decades with zero mentions → NaN (excluded from entropy).
    """
    # Avoid division by zero
    safe_mentions = mentions.replace(0, np.nan)
    rates_df = counts_df.div(safe_mentions, axis=1)
    return rates_df


# ---------------------------------------------------------------------------
# 3. Entropy helpers
# ---------------------------------------------------------------------------

def shannon_entropy(vals: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy of a non-negative vector (zeros ignored)."""
    v = vals[~np.isnan(vals)]
    v = v[v > 0].astype(float)
    if v.sum() == 0:
        return 0.0
    p = v / v.sum()
    return float(scipy_entropy(p, base=base))


def norm_entropy(h: float, n: int, base: float = 2.0) -> float:
    if n <= 1:
        return 0.0
    return h / np.emath.logn(base, n)


# ---------------------------------------------------------------------------
# 4. Per-feature temporal entropy
# ---------------------------------------------------------------------------

def entropy_per_feature(rates_df: pd.DataFrame,
                        counts_df: pd.DataFrame,
                        base: float = 2.0) -> pd.DataFrame:
    """
    For each feature: treat its rate vector across decades as a distribution.
    Also compute:
      - peak_decade  : decade with highest activation rate
      - trend_r      : Pearson r of rate vs decade index (+ = rising, - = falling)
      - trend_p      : p-value of that correlation
    """
    decade_idx = np.arange(rates_df.shape[1])
    n_decades  = rates_df.shape[1]
    records    = []

    for fid in rates_df.index:
        rates  = rates_df.loc[fid].values.astype(float)
        counts = counts_df.loc[fid].values.astype(int)

        h      = shannon_entropy(rates, base=base)
        norm_h = norm_entropy(h, n_decades, base=base)

        # Peak decade
        valid = ~np.isnan(rates)
        if valid.any() and rates[valid].max() > 0:
            peak_i = int(np.nanargmax(rates))
            peak_decade = rates_df.columns[peak_i]
        else:
            peak_decade = None

        # Linear trend over time (using rate values, ignoring NaN decades)
        valid_mask = valid & (rates > 0)
        if valid_mask.sum() >= 3:
            r, p = pearsonr(decade_idx[valid_mask], rates[valid_mask])
        else:
            r, p = np.nan, np.nan

        records.append({
            "feature_id":    fid,
            "total_counts":  int(counts.sum()),
            "n_decades":     int(valid.sum()),
            "entropy":       round(float(h), 6),
            "norm_entropy":  round(float(norm_h), 6),
            "peak_decade":   peak_decade,
            "trend_r":       round(float(r), 4) if not np.isnan(r) else None,
            "trend_p":       round(float(p), 4) if not np.isnan(p) else None,
        })

    df = pd.DataFrame(records).sort_values("norm_entropy", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Per-decade entropy
# ---------------------------------------------------------------------------

def entropy_per_decade(rates_df: pd.DataFrame,
                       mentions: pd.Series,
                       base: float = 2.0) -> pd.DataFrame:
    """
    For each decade: treat the column of feature rates as a distribution.
    Low  = one or two features dominate hat-mentions that decade
    High = feature activations are spread broadly
    """
    n_features = rates_df.shape[0]
    records    = []

    for decade in rates_df.columns:
        rates = rates_df[decade].values.astype(float)
        h     = shannon_entropy(rates, base=base)
        records.append({
            "decade":        decade,
            "mentions":      int(mentions[decade]),
            "entropy":       round(float(h), 6),
            "norm_entropy":  round(float(norm_entropy(h, n_features, base)), 6),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

def print_summary(feat_df: pd.DataFrame,
                  decade_df: pd.DataFrame,
                  term: str) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"TEMPORAL ENTROPY SUMMARY  —  term: '{term}'")
    lines.append("=" * 65)

    lines.append(f"\n  Features: {len(feat_df)}  |  "
                 f"Total dated mentions: {feat_df['total_counts'].sum():,}")

    # Feature entropy
    lines.append("\n--- Entropy per FEATURE over decades ---")
    lines.append(f"  (high = stable convention across the whole century;")
    lines.append(f"   low  = era-specific fad, concentrated in one or two decades)")
    lines.append(f"  Mean norm entropy: {feat_df['norm_entropy'].mean():.4f}")
    lines.append(f"  Std:               {feat_df['norm_entropy'].std():.4f}")

    lines.append("\n  Most STABLE features (high entropy — persist across all decades):")
    for _, r in feat_df.head(8).iterrows():
        trend = _trend_str(r["trend_r"], r["trend_p"])
        lines.append(f"    feat {r['feature_id']:<6}  norm_H={r['norm_entropy']:.3f}"
                     f"  peak={r['peak_decade']}  {trend}")

    lines.append("\n  Most ERA-SPECIFIC features (low entropy — concentrated in one period):")
    for _, r in feat_df.tail(8).iterrows():
        trend = _trend_str(r["trend_r"], r["trend_p"])
        lines.append(f"    feat {r['feature_id']:<6}  norm_H={r['norm_entropy']:.3f}"
                     f"  peak={r['peak_decade']}  {trend}")

    # Rising / falling features
    rising  = feat_df[(feat_df["trend_p"].notna()) &
                      (feat_df["trend_p"] < 0.05) &
                      (feat_df["trend_r"] > 0)].sort_values("trend_r", ascending=False)
    falling = feat_df[(feat_df["trend_p"].notna()) &
                      (feat_df["trend_p"] < 0.05) &
                      (feat_df["trend_r"] < 0)].sort_values("trend_r")

    if len(rising):
        lines.append(f"\n  Significantly RISING features (p<0.05, n={len(rising)}):")
        for _, r in rising.head(5).iterrows():
            lines.append(f"    feat {r['feature_id']:<6}  r={r['trend_r']:+.3f}")
    if len(falling):
        lines.append(f"\n  Significantly FALLING features (p<0.05, n={len(falling)}):")
        for _, r in falling.head(5).iterrows():
            lines.append(f"    feat {r['feature_id']:<6}  r={r['trend_r']:+.3f}")

    # Decade entropy
    lines.append("\n--- Entropy per DECADE over features ---")
    lines.append(f"  (high = diverse feature mix; low = one feature dominates)")
    for _, r in decade_df.iterrows():
        bar = "█" * int(r["norm_entropy"] * 30)
        lines.append(f"    {r['decade']}  norm_H={r['norm_entropy']:.3f}  "
                     f"mentions={r['mentions']:>6,}  {bar}")

    lines.append("\n" + "=" * 65)
    out = "\n".join(lines)
    print(out)
    return out


def _trend_str(r, p) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return ""
    sig = "*" if (p is not None and p < 0.05) else ""
    arrow = "↑" if r > 0 else "↓"
    return f"trend={arrow}{abs(r):.2f}{sig}"


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

def make_plots(feat_df: pd.DataFrame,
               decade_df: pd.DataFrame,
               rates_df: pd.DataFrame,
               counts_df: pd.DataFrame,
               output_dir: str,
               term: str,
               top_n: int = 8) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    decades = list(rates_df.columns)
    x = np.arange(len(decades))

    # ── 1. Distribution of per-feature temporal entropy ──────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(feat_df["norm_entropy"].dropna(), bins=20,
            color="#4A7FBF", edgecolor="white", linewidth=0.5)
    ax.axvline(feat_df["norm_entropy"].mean(), color="firebrick",
               linewidth=1.4, linestyle="--", label="mean")
    ax.set_xlabel("Normalised entropy (bits)", fontsize=10)
    ax.set_ylabel("Number of features", fontsize=10)
    ax.set_title(f"Temporal entropy of SAE features — '{term}'", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, plot_dir, "feature_entropy_hist.png")

    # ── 2. Rate curves: top-N most uniform vs most concentrated ──────────
    top_uniform = feat_df.head(top_n)["feature_id"].tolist()
    top_conc    = feat_df.tail(top_n)["feature_id"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for ax, fids, title, cmap_name in [
        (axes[0], top_uniform, f"Most stable — high entropy (whole century)", "Blues_r"),
        (axes[1], top_conc,    f"Most era-specific — low entropy (one period)", "Oranges_r"),
    ]:
        cmap   = plt.colormaps[cmap_name].resampled(len(fids) + 2)
        for i, fid in enumerate(fids):
            if fid not in rates_df.index:
                continue
            rates = rates_df.loc[fid].values.astype(float)
            ax.plot(x, rates, marker="o", markersize=3,
                    linewidth=1.4, color=cmap(i + 1), label=f"feat {fid}")
        ax.set_xticks(x)
        ax.set_xticklabels(decades, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Activation rate", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2, framealpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"SAE feature activation rates over time — '{term}'", fontsize=11)
    plt.tight_layout()
    _save(fig, plot_dir, "top_features_over_time.png")

    # ── 3. Per-decade entropy over time ───────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color_h = "#4A7FBF"
    color_m = "#999"
    ax2 = ax1.twinx()
    ax2.bar(x, decade_df["mentions"].values, color=color_m,
            alpha=0.25, width=0.6, label="mentions")
    ax1.plot(x, decade_df["norm_entropy"].values, color=color_h,
             marker="o", linewidth=2, markersize=5, label="norm entropy", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(decades, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Normalised entropy (bits)", color=color_h, fontsize=10)
    ax2.set_ylabel("Mention count", color=color_m, fontsize=10)
    ax1.set_title(f"Feature diversity per decade — '{term}'", fontsize=11)
    ax1.tick_params(axis="y", colors=color_h)
    ax2.tick_params(axis="y", colors=color_m)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.spines[["top"]].set_visible(False)
    plt.tight_layout()
    _save(fig, plot_dir, "decade_entropy.png")

    # ── 4. Heatmap: feature × decade activation rates ────────────────────
    # Show only features with enough counts to be meaningful
    min_counts = max(5, counts_df.values.max() * 0.005)
    active = feat_df[feat_df["total_counts"] >= min_counts]["feature_id"].tolist()
    sub_rates = rates_df.loc[rates_df.index.isin(active)]

    # Sort rows by peak decade index for visual clarity
    def peak_idx(fid):
        r = sub_rates.loc[fid].values.astype(float)
        return int(np.nanargmax(r)) if not np.all(np.isnan(r)) else -1

    sorted_fids = sorted(sub_rates.index.tolist(), key=peak_idx)
    sub_rates   = sub_rates.loc[sorted_fids]

    fig_h = max(4, len(sub_rates) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(sub_rates.values.astype(float),
                   aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels(decades, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sub_rates)))
    ax.set_yticklabels([f"feat {f}" for f in sub_rates.index], fontsize=7)
    plt.colorbar(im, ax=ax, label="Activation rate", shrink=0.6)
    ax.set_title(f"SAE feature activation rate heatmap — '{term}'", fontsize=11)
    plt.tight_layout()
    _save(fig, plot_dir, "activation_heatmap.png")


def _save(fig, plot_dir: str, filename: str) -> None:
    import matplotlib.pyplot as plt
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Temporal SAE entropy analysis")
    parser.add_argument("--input",       required=True,
                        help="Path to activations JSON (mentions + codes)")
    parser.add_argument("--term",        default="garment",
                        help="Name of the garment/term for plot titles")
    parser.add_argument("--start_year",  type=int, default=1800,
                        help="First decade's start year (default: 1800)")
    parser.add_argument("--output_dir",  default="./results_temporal",
                        help="Directory for outputs")
    parser.add_argument("--top_n",       type=int, default=8,
                        help="Number of features to highlight in rate-curve plot")
    parser.add_argument("--base",        type=float, default=2.0,
                        help="Logarithm base for entropy (default: 2 = bits)")
    parser.add_argument("--no_plots",    action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print(f"Loading {args.input} ...")
    data = load_activations(args.input)

    # Decode cumulative → per-decade
    print("Decoding cumulative counts ...")
    counts_df, mentions, undated = decode(data, start_year=args.start_year)

    # Rates
    print("Computing activation rates ...")
    rates_df = compute_rates(counts_df, mentions)

    # Entropy
    print("Computing entropy per feature ...")
    feat_df = entropy_per_feature(rates_df, counts_df, base=args.base)

    print("Computing entropy per decade ...")
    decade_df = entropy_per_decade(rates_df, mentions, base=args.base)

    # Save CSVs
    for df, name in [
        (feat_df,   "features.csv"),
        (decade_df, "decades.csv"),
        (rates_df.reset_index(), "rates.csv"),
        (counts_df.reset_index(), "counts.csv"),
    ]:
        path = os.path.join(args.output_dir, name)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    # Undated summary
    undated_path = os.path.join(args.output_dir, "undated.json")
    with open(undated_path, "w") as f:
        json.dump(undated, f, indent=2, cls=_NumpyEncoder)
    print(f"Saved: {undated_path}")

    # Summary
    summary = print_summary(feat_df, decade_df, term=args.term)
    summary_path = os.path.join(args.output_dir, "entropy_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved: {summary_path}")

    # Plots
    if not args.no_plots:
        print("\nGenerating plots ...")
        make_plots(feat_df, decade_df, rates_df, counts_df,
                   args.output_dir, term=args.term, top_n=args.top_n)


if __name__ == "__main__":
    main()