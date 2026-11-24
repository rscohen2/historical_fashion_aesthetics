"""
compare_terms.py
----------------
Compare normalised entropy per decade across a set of terms,
reading directly from the single trends.json file produced by the
historical fashion aesthetics pipeline.

Format of trends.json:
  {
    "decades":  [1800, 1810, ..., 1920],
    "mentions": [total_corpus_mentions_per_decade, ...],
    "terms": {
      "hat":    [hat_mentions_per_decade, ...],
      "dress":  [...],
      ...
    }
  }

All arrays are raw per-decade counts (not cumulative).
The last value may be undated — detected automatically if the
"decades" array is shorter than the counts arrays.

Usage
-----
  python compare_terms.py \\
      --input   data/sae_logodds/trends.json \\
      --terms   hat veil blazer \\
      [--metric norm_entropy]   # norm_entropy | entropy | rate | mentions
      [--output_dir ./plots]
      [--list_terms]            # just print all available terms and exit
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Load and decode
# ---------------------------------------------------------------------------

def load_trends(path: str) -> tuple[list[int], np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
      decades  : list of decade start years, length N
      mentions : np.ndarray of total corpus mention counts, length N
      terms    : {term_name: np.ndarray of counts, length N}
    """
    with open(path) as f:
        data = json.load(f)

    decades  = data["decades"]          # e.g. [1800, 1810, ..., 1920]
    n        = len(decades)

    # Total corpus mentions — truncate to n dated bins
    mentions = np.array(data["mentions"])[:n].astype(float)

    terms = {}
    for term, counts in data["terms"].items():
        terms[term] = np.array(counts)[:n].astype(float)

    return decades, mentions, terms


# ---------------------------------------------------------------------------
# 2. Compute entropy
# ---------------------------------------------------------------------------

def shannon_entropy(counts: np.ndarray, base: float = 2.0) -> float:
    from scipy.stats import entropy as scipy_entropy
    v = counts[counts > 0].astype(float)
    if v.sum() == 0:
        return 0.0
    p = v / v.sum()
    return float(scipy_entropy(p, base=base))


def norm_entropy(h: float, n: int, base: float = 2.0) -> float:
    if n <= 1:
        return 0.0
    return h / np.emath.logn(base, n)


def compute_metrics(decades: list[int],
                    mentions: np.ndarray,
                    term_counts: np.ndarray,
                    base: float = 2.0) -> pd.DataFrame:
    """
    For a single term, compute per-decade metrics.
    Returns a DataFrame with columns: decade, mentions, rate,
                                      entropy, norm_entropy
    """
    n = len(decades)
    # activation rate = term mentions / total corpus mentions that decade
    safe_mentions = np.where(mentions > 0, mentions, np.nan)
    rate = term_counts / safe_mentions

    # overall temporal entropy (scalar) — distribution over decades
    h      = shannon_entropy(term_counts, base=base)
    norm_h = norm_entropy(h, n, base=base)

    records = []
    for i, decade in enumerate(decades):
        records.append({
            "decade":       f"{decade}s",
            "mentions":     int(term_counts[i]),
            "rate":         float(rate[i]) if not np.isnan(rate[i]) else None,
            "entropy":      round(h, 6),        # same for all rows (whole-term scalar)
            "norm_entropy": round(norm_h, 6),   # same for all rows
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------

COLORS  = ["#2A78D6","#D4537E","#2BAB6F","#E8821A",
           "#7F4FBF","#C4A020","#555555","#E03030"]
MARKERS = ["o","s","^","D","v","P","X","*"]

METRIC_LABELS = {
    "norm_entropy": "Normalised entropy (bits)",
    "entropy":      "Shannon entropy (bits)",
    "rate":         "Activation rate (mentions / corpus size)",
    "mentions":     "Raw mention count",
}


def plot_comparison(combined: pd.DataFrame,
                    terms: list[str],
                    metric: str,
                    output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    decades = combined["decade"].unique().tolist()
    decades = sorted(decades, key=lambda d: int(d.replace("s","")))
    x       = np.arange(len(decades))
    ylabel  = METRIC_LABELS.get(metric, metric)

    # ── line chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, term in enumerate(terms):
        sub  = combined[combined["term"] == term].set_index("decade").reindex(decades)
        vals = sub[metric].values.astype(float)

        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(x, vals, color=color, marker=marker,
                linewidth=2.2, markersize=6, label=term, zorder=3)

        # light shading around the line
        if np.sum(~np.isnan(vals)) >= 4:
            rolled = pd.Series(vals).rolling(3, center=True, min_periods=1).std()
            ax.fill_between(x,
                            vals - rolled.values,
                            vals + rolled.values,
                            color=color, alpha=0.10)

    ax.set_xticks(x)
    ax.set_xticklabels(decades, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_title(f"{ylabel} — {', '.join(terms)}", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    slug = "_".join(terms)
    out  = os.path.join(output_dir, f"compare_{slug}_{metric}_line.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── grouped bar chart ─────────────────────────────────────────────────
    n_terms = len(terms)
    width   = 0.8 / n_terms
    offsets = np.linspace(-(n_terms-1)/2, (n_terms-1)/2, n_terms) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, term in enumerate(terms):
        sub  = combined[combined["term"] == term].set_index("decade").reindex(decades)
        vals = sub[metric].values.astype(float)
        ax.bar(x + offsets[i], vals, width=width*0.9,
               color=COLORS[i % len(COLORS)], alpha=0.82, label=term, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(decades, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_title(f"{ylabel} per decade — {', '.join(terms)}", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    out = os.path.join(output_dir, f"compare_{slug}_{metric}_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── print table ───────────────────────────────────────────────────────
    print(f"\n{metric} per decade:")
    print(f"  {'Decade':<10}" + "".join(f"  {t:>12}" for t in terms))
    print("  " + "-" * (10 + 14 * n_terms))
    for decade in decades:
        row = f"  {decade:<10}"
        for term in terms:
            sub = combined[(combined["term"]==term) & (combined["decade"]==decade)]
            val = float(sub[metric].values[0]) if len(sub) else float("nan")
            row += f"  {val:>12.4f}"
        print(row)


# ---------------------------------------------------------------------------
# 4. Entropy comparison bar chart
# ---------------------------------------------------------------------------

def plot_entropy_bars(terms: list[str],
                      all_terms: dict[str, np.ndarray],
                      decades: list[int],
                      output_dir: str,
                      base: float = 2.0) -> None:
    """
    One bar per term showing its overall normalised temporal entropy.
    Sorted from most era-specific (low) to most stable (high).
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(decades)

    rows = []
    for term in terms:
        counts = all_terms[term]
        h      = shannon_entropy(counts, base=base)
        norm_h = norm_entropy(h, n, base=base)
        rows.append({"term": term, "norm_entropy": norm_h, "entropy": h,
                     "total": int(counts.sum())})

    df = pd.DataFrame(rows).sort_values("norm_entropy")

    fig, ax = plt.subplots(figsize=(max(5, len(terms) * 0.9), 5))

    bars = ax.bar(df["term"], df["norm_entropy"],
                  color=[COLORS[i % len(COLORS)] for i in range(len(df))],
                  edgecolor="white", linewidth=0.5, zorder=2)

    # Annotate each bar with the value
    for bar, val in zip(bars, df["norm_entropy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Normalised temporal entropy (bits)", fontsize=11)
    ax.set_xlabel("Term", fontsize=11)
    ax.set_title("Temporal entropy by term\n"
                 "(low = era-specific, high = stable across century)",
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(df["term"], rotation=30, ha="right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()

    slug = "_".join(terms)
    out  = os.path.join(output_dir, f"entropy_bars_{slug}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print ranking
    print("\nTemporal entropy ranking (low = era-specific, high = stable):")
    print(f"  {'Term':<20} {'norm_entropy':>14} {'total_mentions':>16}")
    print("  " + "-" * 52)
    for _, r in df.iterrows():
        print(f"  {r['term']:<20} {r['norm_entropy']:>14.4f} {r['total']:>16,}")


# ---------------------------------------------------------------------------
# 5. Per-decade feature entropy
# ---------------------------------------------------------------------------

def load_term_trends(term_trends_dir: str, term_idx: int) -> dict:
    """Load and decode a term_trends/<idx>.json file to raw per-decade counts."""
    path = os.path.join(term_trends_dir, f"{term_idx}.json")
    with open(path) as f:
        data = json.load(f)

    # Detect cumulative vs raw
    mentions = data["mentions"][:12]
    diffs    = np.diff(mentions)
    is_cum   = bool(np.all(np.array(diffs) >= 0) and np.any(np.array(diffs) > 0)
                    and mentions[-1] > mentions[0])

    def to_raw(arr):
        dated = np.array(arr[:12])
        if is_cum:
            return np.diff(np.concatenate([[0], dated])).astype(float)
        return dated.astype(float)

    return {
        "mentions": to_raw(data["mentions"]),
        "codes":    {fid: to_raw(v) for fid, v in data["codes"].items()},
    }


def feature_entropy_per_decade(decoded: dict,
                                decades: list[int],
                                base: float = 2.0) -> list[float]:
    """
    For each decade, compute Shannon entropy over the SAE feature
    activation distribution (feature counts in that decade).
    Returns a list of norm_entropy values, one per decade.
    """
    n_features = len(decoded["codes"])
    entropies  = []

    n_decades = min(len(decades), len(next(iter(decoded["codes"].values()))))
    for i in range(n_decades):
        # vector of feature counts for this decade
        counts = np.array([v[i] for v in decoded["codes"].values()])
        h      = shannon_entropy(counts, base=base)
        norm_h = norm_entropy(h, n_features, base=base)
        entropies.append(norm_h)

    return entropies


def plot_feature_entropy_per_decade(terms: list[str],
                                    all_terms: dict[str, np.ndarray],
                                    decades: list[int],
                                    term_trends_dir: str,
                                    output_dir: str,
                                    base: float = 2.0) -> None:
    """
    For each term, compute per-decade feature entropy and plot as a
    line chart comparing all requested terms.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get term index from position in all_terms
    term_list = list(all_terms.keys())

    decade_labels = [f"{d}s" for d in decades]
    x = np.arange(len(decades))

    fig, ax = plt.subplots(figsize=(11, 5))

    all_rows = []
    for i, term in enumerate(terms):
        if term not in term_list:
            print(f"  WARNING: {term} not in trends.json, skipping")
            continue

        term_idx = term_list.index(term)
        try:
            decoded = load_term_trends(term_trends_dir, term_idx)
        except FileNotFoundError:
            print(f"  WARNING: no file for {term} (index {term_idx}), skipping")
            continue

        entropies = feature_entropy_per_decade(decoded, decades, base=base)
        n_dec = len(entropies)

        # Mask decades with zero mentions for this term
        term_counts = all_terms[term][:n_dec]
        entropies_masked = [e if term_counts[j] > 0 else np.nan
                            for j, e in enumerate(entropies)]

        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        vals = np.array(entropies_masked, dtype=float)
        xi   = x[:n_dec]
        ax.plot(xi, vals, color=color, marker=marker,
                linewidth=2.2, markersize=6, label=term, zorder=3)

        # Light shading
        valid = ~np.isnan(vals)
        if valid.sum() >= 4:
            rolled = pd.Series(vals).rolling(3, center=True, min_periods=1).std()
            ax.fill_between(xi,
                            vals - rolled.values,
                            vals + rolled.values,
                            color=color, alpha=0.10)

        for j, (d, e) in enumerate(zip(decade_labels, entropies_masked)):
            all_rows.append({"term": term, "decade": d,
                             "feature_norm_entropy": round(e, 4)
                             if not np.isnan(e) else None})

    ax.set_xticks(x)
    ax.set_xticklabels(decade_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalised feature entropy (bits)", fontsize=11)
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_title(f"SAE feature activation entropy per decade\n"
                 f"(high = diverse features, low = one feature dominates)\n"
                 f"{', '.join(terms)}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    slug = "_".join(terms)
    out  = os.path.join(output_dir, f"feature_entropy_per_decade_{slug}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print table
    df = pd.DataFrame(all_rows)
    print("\nFeature entropy per decade:")
    print(f"  {'Decade':<10}" + "".join(f"  {t:>22}" for t in terms))
    print("  " + "-" * (10 + 24 * len(terms)))
    for d in decade_labels:
        row = f"  {d:<10}"
        for term in terms:
            sub = df[(df["term"] == term) & (df["decade"] == d)]
            val = sub["feature_norm_entropy"].values[0] if len(sub) else None
            row += f"  {str(val):>22}" if val is None else f"  {float(val):>22.4f}"
        print(row)

    # Save CSV
    csv_path = os.path.join(output_dir, f"feature_entropy_per_decade_{slug}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# 6. Single-feature dominance over time
# ---------------------------------------------------------------------------

def plot_feature_dominance(term: str,
                           term_idx: int,
                           feature_id: str,
                           feature_label: str,
                           all_terms: dict[str, np.ndarray],
                           decades: list[int],
                           term_trends_dir: str,
                           output_dir: str,
                           base: float = 2.0) -> None:
    """
    For one term, plot over time:
      - feature share  : feature activations / total term mentions per decade
      - feature entropy: overall SAE feature entropy per decade
    Side by side, so you can see if share rises as entropy falls.
    """
    os.makedirs(output_dir, exist_ok=True)

    decoded     = load_term_trends(term_trends_dir, term_idx)
    n_dec       = min(len(decades), len(next(iter(decoded["codes"].values()))))
    decade_lbls = [f"{d}s" for d in decades[:n_dec]]
    x           = np.arange(n_dec)

    # Term mention counts per decade (for normalisation)
    term_counts = all_terms[term][:n_dec]

    # Feature counts per decade
    if feature_id not in decoded["codes"]:
        print(f"  Feature {feature_id} not found in {term} activation file.")
        print(f"  Available features: {list(decoded['codes'].keys())[:10]} ...")
        return

    feat_counts = decoded["codes"][feature_id]

    # Share: feature activations / term mentions (not / total feature acts,
    # so it reflects how often this feature fires per hat-mention)
    safe_term = np.where(term_counts > 0, term_counts, np.nan)
    share = feat_counts / safe_term

    # Overall feature entropy per decade
    entropies = feature_entropy_per_decade(decoded, decades[:n_dec], base=base)
    entropies_masked = np.array([
        e if term_counts[i] > 0 else np.nan
        for i, e in enumerate(entropies)
    ])

    # ── plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: feature share over time
    ax1.plot(x, share, color="#D4537E", marker="o",
             linewidth=2.2, markersize=6, zorder=3)
    ax1.fill_between(x, 0, share, color="#D4537E", alpha=0.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(decade_lbls, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Activations per mention", fontsize=11)
    ax1.set_title(f"Feature {feature_id} share over time\n'{feature_label}'",
                  fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: feature entropy over time
    ax2.plot(x, entropies_masked, color="#2A78D6", marker="s",
             linewidth=2.2, markersize=6, zorder=3)
    ax2.fill_between(x, 0, entropies_masked, color="#2A78D6", alpha=0.10)

    # Overlay mention count as grey bars for context
    ax2b = ax2.twinx()
    ax2b.bar(x, term_counts, color="#aaa", alpha=0.18, width=0.6, zorder=1)
    ax2b.set_ylabel("Mention count", color="#aaa", fontsize=9)
    ax2b.tick_params(axis="y", colors="#aaa")

    ax2.set_xticks(x)
    ax2.set_xticklabels(decade_lbls, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Normalised feature entropy", fontsize=11)
    ax2.set_title(f"SAE feature entropy per decade\n'{term}'", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"'{term}': narrative conventionalisation via feature {feature_id}\n"
        f"Hypothesis: share ↑ as entropy ↓ if '{feature_label}' becomes a stock gesture",
        fontsize=11
    )
    plt.tight_layout()

    out = os.path.join(output_dir,
                       f"dominance_{term}_feat{feature_id}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print table
    print(f"\nFeature {feature_id} ('{feature_label}') dominance for '{term}':")
    print(f"  {'Decade':<10}  {'feat_share':>12}  {'feat_entropy':>14}  {'mentions':>10}")
    print("  " + "-" * 52)
    for i, d in enumerate(decade_lbls):
        s = share[i] if not np.isnan(share[i]) else None
        e = entropies_masked[i] if not np.isnan(entropies_masked[i]) else None
        m = int(term_counts[i])
        print(f"  {d:<10}  "
              f"{str(round(s,4)) if s is not None else 'N/A':>12}  "
              f"{str(round(e,4)) if e is not None else 'N/A':>14}  "
              f"{m:>10,}")


# ---------------------------------------------------------------------------
# 7. Overall feature comparison from parquet
# ---------------------------------------------------------------------------

def plot_feature_comparison(term_features: dict[str, list[dict]],
                            output_dir: str,
                            parquet_path: str = None) -> None:
    """
    For each term, plot a grouped bar chart showing logodds and count
    for a set of named features.

    term_features: {
        "veil":  [{"code": 984,  "label": "Bridal Imagery"},
                  {"code": 1735, "label": "Face Obscured"},
                  {"code": 1545, "label": "Face Focus"}],
        "hat":   [{"code": 1311, "label": "Lifted hat and departed"}],
    }
    """
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    # Find parquet
    if not parquet_path:
        parquet_candidates = [
            "data/sae_logodds/sae_logodds.parquet",
            "sae_logodds.parquet",
        ]
        parquet_path = next((p for p in parquet_candidates if os.path.exists(p)), None)
    if not parquet_path:
        print("  ERROR: cannot find sae_logodds.parquet — tried:")
        for p in parquet_candidates:
            print(f"    {p}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded parquet: {df.shape}")

    # Collect rows
    rows = []
    for term, features in term_features.items():
        term_df = df[df["term"] == term]
        for feat in features:
            code = feat["code"]
            label = feat.get("label", f"feature {code}")
            row = term_df[term_df["code"] == code]
            if len(row):
                rows.append({
                    "term":     term,
                    "code":     code,
                    "label":    label,
                    "logodds":  float(row["logodds"].values[0]),
                    "count":    int(row["count"].values[0]),
                    "score":    float(row["score"].values[0]),
                })
                print(f"  {term} / feature {code} ({label}): "
                      f"logodds={rows[-1]['logodds']:.3f} count={rows[-1]['count']:,}")
            else:
                print(f"  WARNING: feature {code} not found for term '{term}'")

    if not rows:
        print("  No data found.")
        return

    plot_df = pd.DataFrame(rows)
    plot_df["feat_label"] = plot_df.apply(
        lambda r: f"{r['label']}\n(feat {r['code']})", axis=1)

    terms     = plot_df["term"].unique().tolist()
    all_feats = plot_df["feat_label"].unique().tolist()

    # ── fig: two subplots, logodds and count ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, ylabel, title in [
        ("logodds", ax1, "Log-odds (distinctiveness)", "Feature distinctiveness for term"),
        ("count",   ax2, "Total co-occurrence count",  "Feature frequency for term"),
    ]:
        n_terms  = len(terms)
        x        = np.arange(len(all_feats))
        width    = 0.8 / n_terms
        offsets  = np.linspace(-(n_terms-1)/2, (n_terms-1)/2, n_terms) * width

        for i, term in enumerate(terms):
            sub  = plot_df[plot_df["term"] == term]
            vals = [sub.loc[sub["feat_label"]==f, metric].values[0]
                    if f in sub["feat_label"].values else 0
                    for f in all_feats]
            bars = ax.bar(x + offsets[i], vals, width=width*0.9,
                          color=COLORS[i % len(COLORS)], alpha=0.82,
                          label=term, zorder=2)
            # value labels on bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + ax.get_ylim()[1]*0.01,
                            f"{val:.3f}" if metric=="logodds" else f"{val:,}",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(all_feats, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle("SAE feature association with garment terms\n"
                 "(log-odds = how distinctive; count = how frequent)",
                 fontsize=11)
    plt.tight_layout()

    slug = "_".join(term_features.keys())
    out  = os.path.join(output_dir, f"feature_comparison_{slug}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare term entropy across decades")
    parser.add_argument("--input",      required=True,
                        help="Path to trends.json (e.g. data/sae_logodds/trends.json)")
    parser.add_argument("--terms",      nargs="+", default=None,
                        help="Terms to compare, e.g. --terms hat veil blazer")
    parser.add_argument("--metric",     default="rate",
                        choices=["norm_entropy","entropy","rate","mentions"],
                        help="Metric to plot (default: rate)")
    parser.add_argument("--output_dir", default="./comparisons",
                        help="Where to save plots (default: ./comparisons)")
    parser.add_argument("--term_trends_dir", default=None,
                        help="Path to term_trends/ directory for per-decade "
                             "feature entropy (e.g. data/sae_logodds/term_trends)")
    parser.add_argument("--feature_id",    default=None,
                        help="SAE feature ID to track dominance for (e.g. 1311)")
    parser.add_argument("--feature_label", default=None,
                        help="Human-readable label for the feature "
                             "(e.g. 'lifted his hat and departed')")
    parser.add_argument("--parquet", default=None,
                        help="Path to sae_logodds.parquet (default: auto-detect)")
    parser.add_argument("--compare_features", action="store_true",
                        help="Plot overall logodds+count comparison for named features "
                             "using --feature_id. Use with multiple --terms.")
    parser.add_argument("--feature_ids",   nargs="+", default=None,
                        help="Multiple feature IDs for --compare_features, "
                             "e.g. --feature_ids 984 1735 1545")
    parser.add_argument("--feature_labels", nargs="+", default=None,
                        help="Labels for --feature_ids in same order")
    parser.add_argument("--list_terms", action="store_true",
                        help="Print all available terms and exit")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    decades, mentions, all_terms = load_trends(args.input)
    print(f"  {len(all_terms)} terms, {len(decades)} decades ({decades[0]}–{decades[-1]})")

    if args.list_terms:
        print("\nAvailable terms:")
        for t in sorted(all_terms.keys()):
            total = int(all_terms[t].sum())
            print(f"  {t:<30} total={total:,}")
        return

    if not args.terms:
        parser.error("Provide --terms or use --list_terms to see options.")

    # Check requested terms exist
    missing = [t for t in args.terms if t not in all_terms]
    if missing:
        print(f"Terms not found: {missing}")
        close = [k for k in all_terms if any(m.lower() in k.lower() for m in missing)]
        if close:
            print(f"Similar terms available: {close[:10]}")
        return

    # Compute metrics for each term
    frames = []
    for term in args.terms:
        df = compute_metrics(decades, mentions, all_terms[term])
        df["term"] = term
        frames.append(df)
        h = df["norm_entropy"].iloc[0]
        print(f"  {term:<20} norm_entropy={h:.3f}  "
              f"total_mentions={int(all_terms[term].sum()):,}")

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nPlotting {args.metric} ...")
    plot_comparison(combined, terms=args.terms, metric=args.metric,
                    output_dir=args.output_dir)

    # Always also plot the entropy comparison bar chart
    print("\nPlotting entropy comparison across terms ...")
    plot_entropy_bars(args.terms, all_terms, decades, args.output_dir)

    # Per-decade feature entropy if term_trends_dir is provided
    if args.term_trends_dir:
        print("\nComputing per-decade feature entropy ...")
        plot_feature_entropy_per_decade(
            args.terms, all_terms, decades,
            args.term_trends_dir, args.output_dir
        )

    # Overall feature comparison from parquet
    if args.compare_features:
        fids    = [int(x) for x in (args.feature_ids or [])]
        flabels = args.feature_labels or [f"feature {x}" for x in fids]
        feats   = [{"code": c, "label": l} for c, l in zip(fids, flabels)]
        term_features = {t: feats for t in args.terms}
        print("\nPlotting overall feature comparison ...")
        plot_feature_comparison(term_features, args.output_dir,
                                parquet_path=args.parquet)

    # Single-feature dominance plot
    if args.term_trends_dir and args.feature_id:
        if len(args.terms) != 1:
            print("\nNote: --feature_id plots one term at a time; "
                  f"using first term: {args.terms[0]}")
        term      = args.terms[0]
        term_list = list(all_terms.keys())
        term_idx  = term_list.index(term)
        label     = args.feature_label or f"feature {args.feature_id}"
        print(f"\nPlotting feature {args.feature_id} dominance for '{term}' ...")
        plot_feature_dominance(
            term, term_idx, args.feature_id, label,
            all_terms, decades, args.term_trends_dir, args.output_dir
        )


if __name__ == "__main__":
    main()