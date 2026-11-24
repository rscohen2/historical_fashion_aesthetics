"""
syntactic_homogeneity.py
------------------------
Measures syntactic homogeneity of example sentences per SAE feature per term.
Homogeneity = mean pairwise 3-gram Jaccard similarity across the 5 example
sentences for each feature. High homogeneity = sentences share a frozen
syntactic template (e.g. "He lifted his hat and departed.").

Outputs:
  - homogeneity_per_feature.csv  : term, feature, label, homogeneity, logodds
  - homogeneity_per_term.csv     : term, mean_homogeneity, max_homogeneity, n_features
  - plots/term_homogeneity_bar.png
  - plots/homogeneity_vs_logodds_<term>.png  (scatter per term)
  - plots/top_features_homogeneity.png       (top frozen templates across all terms)

Usage:
  python syntactic_homogeneity.py \\
      --examples_dir data/sae_logodds/examples \\
      --trends      data/sae_logodds/trends.json \\
      --parquet     data/sae_logodds/sae_logodds.parquet \\
      --terms       hat veil dress mask uniform \\
      [--output_dir ./results/homogeneity] \\
      [--ngram 3] \\
      [--all_terms]   # compute for all 1218 terms (slow)
"""

import argparse
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Core metric
# ---------------------------------------------------------------------------

def ngrams(text: str, n: int = 3) -> set:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()


def homogeneity(sentences: list[str], n: int = 3) -> float:
    """Mean pairwise n-gram Jaccard similarity."""
    if len(sentences) < 2:
        return 0.0
    scores = []
    for a, b in combinations(sentences, 2):
        na, nb = ngrams(a, n), ngrams(b, n)
        if not na or not nb:
            continue
        scores.append(len(na & nb) / len(na | nb))
    return float(sum(scores) / len(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# 2. Compute for a set of terms
# ---------------------------------------------------------------------------

def compute_homogeneity(terms: list[str],
                        term_indices: dict[str, int],
                        examples_dir: str,
                        logodds_df: pd.DataFrame,
                        ngram: int = 3) -> pd.DataFrame:
    rows = []
    for term in terms:
        idx  = term_indices[term]
        path = os.path.join(examples_dir, f"{idx}.json")
        if not os.path.exists(path):
            print(f"  WARNING: no examples file for {term} (index {idx})")
            continue
        with open(path) as f:
            d = json.load(f)

        for fid, sentences in d.items():
            # Deduplicate by normalized sentence text — same passage across
            # multiple editions of the same work inflates homogeneity
            import re as _re
            seen = {}
            for s in sentences:
                # normalize: lowercase, collapse whitespace, strip punctuation
                key = _re.sub(r'[^\w\s]', '', s["s"].lower().strip())
                key = ' '.join(key.split())
                if key not in seen:
                    seen[key] = s
            sents = [s["s"] for s in seen.values()]
            h     = homogeneity(sents, n=ngram)

            # get label and logodds from parquet
            row = logodds_df[(logodds_df["term"] == term) &
                             (logodds_df["code"] == int(fid))]
            label   = row["label"].values[0]   if len(row) else f"feature {fid}"
            lo      = row["logodds"].values[0]  if len(row) else 0.0
            score   = row["score"].values[0]    if len(row) else 0.0
            count   = int(row["count"].values[0]) if len(row) else 0

            rows.append({
                "term":        term,
                "feature":     fid,
                "label":       label,
                "homogeneity": round(h, 6),
                "logodds":     round(float(lo), 6),
                "score":       round(float(score), 4),
                "count":       count,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Per-term summary
# ---------------------------------------------------------------------------

def summarise_per_term(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("term").agg(
        mean_homogeneity=("homogeneity", "mean"),
        max_homogeneity=("homogeneity", "max"),
        n_features=("feature", "count"),
        top_feature=("feature", lambda x: x.iloc[df.loc[x.index, "homogeneity"].argmax()]),
        top_label=("label", lambda x: x.iloc[df.loc[x.index, "homogeneity"].argmax()]),
    ).sort_values("mean_homogeneity", ascending=False).reset_index()
    # gap: large gap = one outlier template; small gap = broadly homogeneous
    summary["gap"] = summary["max_homogeneity"] - summary["mean_homogeneity"]
    # breadth: mean / max — 1.0 = perfectly even, 0.0 = one outlier dominates
    summary["breadth"] = summary["mean_homogeneity"] / summary["max_homogeneity"].replace(0, np.nan)
    return summary


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

COLORS  = ["#2A78D6","#D4537E","#2BAB6F","#E8821A",
           "#7F4FBF","#C4A020","#555555","#E03030"]
MARKERS = ["o","s","^","D","v","P","X","*"]


def plot_term_bars(summary: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.7), 5))

    colors = [COLORS[i % len(COLORS)] for i in range(len(summary))]
    bars = ax.bar(summary["term"], summary["mean_homogeneity"],
                  color=colors, edgecolor="white", linewidth=0.5, zorder=2,
                  label="mean homogeneity")
    ax.bar(summary["term"], summary["max_homogeneity"],
           color=colors, alpha=0.25, edgecolor="none", zorder=1,
           label="max homogeneity")

    for bar, val in zip(bars, summary["mean_homogeneity"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean pairwise 3-gram Jaccard similarity", fontsize=10)
    ax.set_xlabel("Term", fontsize=10)
    ax.set_title("Syntactic homogeneity per garment term\n"
                 "(high = sentences share a frozen syntactic template)",
                 fontsize=11)
    ax.set_xticklabels(summary["term"], rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()

    out = os.path.join(output_dir, "term_homogeneity_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_gap_breadth(summary: pd.DataFrame, output_dir: str) -> None:
    """
    Two-panel plot:
    Left:  mean vs max homogeneity scatter, gap shown as vertical line
    Right: breadth score (mean/max) — how spread the homogeneity is
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    terms  = summary["term"].tolist()
    colors = [COLORS[i % len(COLORS)] for i in range(len(terms))]

    # ── left: mean vs max with gap lines ─────────────────────────────────
    for i, (_, r) in enumerate(summary.iterrows()):
        ax1.plot([r["mean_homogeneity"], r["max_homogeneity"]], [i, i],
                 color=colors[i], linewidth=2, alpha=0.5)
        ax1.scatter(r["mean_homogeneity"], i, color=colors[i], s=60,
                    zorder=3, label=r["term"])
        ax1.scatter(r["max_homogeneity"], i, color=colors[i], s=60,
                    zorder=3, marker="D", alpha=0.5)
        ax1.text(r["max_homogeneity"] + 0.005, i,
                 f"gap={r['gap']:.3f}", va="center", fontsize=8, color=colors[i])

    ax1.set_yticks(range(len(terms)))
    ax1.set_yticklabels(terms, fontsize=10)
    ax1.set_xlabel("Homogeneity (● mean, ◆ max)", fontsize=10)
    ax1.set_title("Mean vs max homogeneity per term\n"
                  "gap = max − mean (large gap = one outlier template)", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.25)

    # ── right: breadth bar chart ──────────────────────────────────────────
    summary_sorted = summary.sort_values("breadth", ascending=False)
    ax2.barh(summary_sorted["term"], summary_sorted["breadth"],
             color=[COLORS[terms.index(t) % len(COLORS)]
                    for t in summary_sorted["term"]],
             edgecolor="white", linewidth=0.5)
    for i, (_, r) in enumerate(summary_sorted.iterrows()):
        ax2.text(r["breadth"] + 0.005, i,
                 f"{r['breadth']:.3f}", va="center", fontsize=8)

    ax2.set_xlabel("Breadth = mean / max (1.0 = evenly homogeneous,\n"
                   "0.0 = one template dominates)", fontsize=9)
    ax2.set_title("Homogeneity breadth per term\n"
                  "(high = multiple frozen templates, not just one)", fontsize=10)
    ax2.set_xlim(0, 1.15)
    ax2.invert_yaxis()
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("Syntactic conventionalisation: breadth vs depth",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "homogeneity_gap_breadth.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_homogeneity_vs_logodds(df: pd.DataFrame,
                                 terms: list[str],
                                 output_dir: str) -> None:
    """
    For each term: scatter of homogeneity vs logodds per feature.
    Tests whether the most distinctive features are also the most
    syntactically frozen.
    """
    n = len(terms)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for i, term in enumerate(terms):
        ax  = axes[i // ncols][i % ncols]
        sub = df[df["term"] == term]

        ax.scatter(sub["logodds"], sub["homogeneity"],
                   alpha=0.5, s=25, color=COLORS[i % len(COLORS)], zorder=2)

        # label top-5 by homogeneity
        for _, row in sub.nlargest(5, "homogeneity").iterrows():
            ax.annotate(f"{row['label'][:20]}\n(feat {row['feature']})",
                        (row["logodds"], row["homogeneity"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5, color="#333")

        # trend line
        if len(sub) > 2 and sub["logodds"].std() > 0:
            m = np.polyfit(sub["logodds"], sub["homogeneity"], 1)
            xs = np.linspace(sub["logodds"].min(), sub["logodds"].max(), 100)
            ax.plot(xs, np.polyval(m, xs), color="firebrick",
                    linewidth=1, linestyle="--", alpha=0.6)
            r = sub["logodds"].corr(sub["homogeneity"])
            ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes,
                    fontsize=8, color="firebrick")

        ax.axvline(0, color="#ddd", linewidth=0.7)
        ax.set_xlabel("Log-odds (distinctiveness)", fontsize=9)
        ax.set_ylabel("Homogeneity", fontsize=9)
        ax.set_title(f"'{term}'", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)

    # hide unused axes
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Syntactic homogeneity vs feature distinctiveness\n"
                 "(do the most distinctive features also have frozen syntax?)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, "homogeneity_vs_logodds.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_top_frozen_templates(df: pd.DataFrame,
                               output_dir: str,
                               top_n: int = 20) -> None:
    """
    Bar chart of the top-N most syntactically frozen feature×term pairs
    across all analysed terms.
    """
    top = df.nlargest(top_n, "homogeneity").copy()
    top["label_short"] = top.apply(
        lambda r: f"{r['term']}: {r['label'][:25]}\n(feat {r['feature']})", axis=1)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = [COLORS[list(df["term"].unique()).index(t) % len(COLORS)]
              for t in top["term"]]
    ax.barh(top["label_short"], top["homogeneity"],
            color=colors, edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Syntactic homogeneity (3-gram Jaccard)", fontsize=10)
    ax.set_title(f"Top {top_n} most syntactically frozen feature templates",
                 fontsize=11)
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    # legend for terms
    seen = []
    for term, color in zip(top["term"], colors):
        if term not in seen:
            ax.barh([], [], color=color, label=term)
            seen.append(term)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(output_dir, "top_frozen_templates.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Syntactic homogeneity analysis")
    parser.add_argument("--examples_dir", required=True,
                        help="data/sae_logodds/examples/")
    parser.add_argument("--trends",       required=True,
                        help="data/sae_logodds/trends.json")
    parser.add_argument("--parquet",      required=True,
                        help="data/sae_logodds/sae_logodds.parquet")
    parser.add_argument("--terms",        nargs="+", default=None,
                        help="Terms to analyse, e.g. --terms hat veil dress")
    parser.add_argument("--output_dir",   default="./results/homogeneity")
    parser.add_argument("--ngram",        type=int, default=3)
    parser.add_argument("--all_terms",    action="store_true",
                        help="Compute for all 1218 terms (slow)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data ...")
    with open(args.trends) as f:
        trends = json.load(f)
    term_list    = list(trends["terms"].keys())
    term_indices = {t: i for i, t in enumerate(term_list)}

    logodds_df = pd.read_parquet(args.parquet)
    print(f"  {len(term_list)} terms, parquet shape: {logodds_df.shape}")

    # Choose terms to analyse
    if args.all_terms:
        terms = term_list
    elif args.terms:
        missing = [t for t in args.terms if t not in term_indices]
        if missing:
            print(f"  WARNING: terms not found: {missing}")
        terms = [t for t in args.terms if t in term_indices]
    else:
        parser.error("Provide --terms or --all_terms")

    print(f"Analysing {len(terms)} terms ...")
    df = compute_homogeneity(terms, term_indices, args.examples_dir,
                             logodds_df, ngram=args.ngram)

    summary = summarise_per_term(df)

    # Save CSVs
    df.to_csv(os.path.join(args.output_dir, "homogeneity_per_feature.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "homogeneity_per_term.csv"), index=False)
    print(f"Saved CSVs to {args.output_dir}")

    # Print summary
    print("\nSyntactic homogeneity per term:")
    print(f"  {'Term':<15} {'mean_H':>8} {'max_H':>8} {'gap':>8} {'breadth':>8}  top feature")
    print("  " + "-" * 75)
    for _, r in summary.iterrows():
        print(f"  {r['term']:<15} {r['mean_homogeneity']:>8.4f} "
              f"{r['max_homogeneity']:>8.4f} {r['gap']:>8.4f} {r['breadth']:>8.4f}  "
              f"{r['top_label'][:25]} (feat {r['top_feature']})")

    # Plots
    print("\nGenerating plots ...")
    plot_term_bars(summary, args.output_dir)
    plot_gap_breadth(summary, args.output_dir)
    plot_homogeneity_vs_logodds(df, terms, args.output_dir)
    plot_top_frozen_templates(df, args.output_dir)


if __name__ == "__main__":
    main()