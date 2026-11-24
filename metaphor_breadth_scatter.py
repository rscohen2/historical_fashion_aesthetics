import json, re, os, pandas as pd, numpy as np
from itertools import combinations
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ngrams(text, n=3):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()

def homogeneity(sentences, n=3):
    if len(sentences) < 2: return 0.0
    scores = []
    for a, b in combinations(sentences, 2):
        na, nb = ngrams(a,n), ngrams(b,n)
        if not na or not nb: continue
        scores.append(len(na&nb)/len(na|nb))
    return sum(scores)/len(scores) if scores else 0.0

# Load data
with open("data/sae_logodds/trends.json") as f:
    trends = json.load(f)
term_list = list(trends["terms"].keys())
df_lo = pd.read_parquet("data/sae_logodds/sae_logodds.parquet")

metaphor_features = [991, 1288, 389, 528, 17, 1921, 1873]
textile_features  = [207, 621, 487]

# Compute metaphor score for all terms
metaphor = df_lo[df_lo["code"].isin(metaphor_features)].groupby("term")["logodds"].sum().rename("metaphor_score")

# Compute breadth for all terms with examples
examples_dir = "data/sae_logodds/examples"
rows = []
for idx, term in enumerate(term_list):
    path = os.path.join(examples_dir, f"{idx}.json")
    if not os.path.exists(path): continue
    with open(path) as f:
        d = json.load(f)
    def dedup(sents):
        import re as _re
        seen = {}
        for s in sents:
            key = _re.sub(r'[^\w\s]', '', s["s"].lower().strip())
            key = ' '.join(key.split())
            if key not in seen:
                seen[key] = s
        return [s["s"] for s in seen.values()]
    h_vals = [homogeneity(dedup(sents)) for sents in d.values()]
    if not h_vals: continue
    mean_h = np.mean(h_vals)
    max_h  = max(h_vals)
    breadth = mean_h / max_h if max_h > 0 else 0.0
    rows.append({"term": term, "mean_h": mean_h, "max_h": max_h, "breadth": breadth})

breadth_df = pd.DataFrame(rows).set_index("term")
combined = breadth_df.join(metaphor).dropna()

# Filter to terms with enough signal (mean_h > 0.01)
combined = combined[combined["mean_h"] > 0.01]
print(f"Terms with mean_h > 0.01: {len(combined)}")

r, p = pearsonr(combined["metaphor_score"], combined["breadth"])
print(f"Pearson r (metaphor vs breadth): {r:.3f}, p={p:.4f}")

# Highlight terms of interest
highlight = ["hat","veil","dress","mask","uniform","lace","chemise",
             "blazer","cloak","gloves","boots","coat","gown","silk","robe"]

fig, ax = plt.subplots(figsize=(10, 8))

# all points
ax.scatter(combined["metaphor_score"], combined["breadth"],
           alpha=0.2, s=15, color="#888", zorder=2)

# highlighted terms
colors = ["#2A78D6","#555","#2BAB6F","#7F4FBF","#D4537E",
          "#C4A020","#E03030","#E8821A","#4FB3BF","#A0522D",
          "#708090","#2F4F4F","#8B008B","#DAA520","#CD5C5C"]
for i, term in enumerate(highlight):
    if term not in combined.index: continue
    x = combined.loc[term, "metaphor_score"]
    y = combined.loc[term, "breadth"]
    c = colors[i % len(colors)]
    ax.scatter(x, y, color=c, s=90, zorder=4)
    ax.annotate(term, (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color=c, fontweight="bold")

# trend line
m = np.polyfit(combined["metaphor_score"], combined["breadth"], 1)
xs = np.linspace(combined["metaphor_score"].min(),
                 combined["metaphor_score"].max(), 200)
ax.plot(xs, np.polyval(m, xs), color="firebrick",
        linewidth=1.2, linestyle="--", alpha=0.6)

ax.axhline(combined["breadth"].median(), color="#ddd", linewidth=0.8)
ax.axvline(0, color="#ddd", linewidth=0.8)

ax.set_xlabel("Metaphor score (sum of metaphor feature logodds)", fontsize=11)
ax.set_ylabel("Syntactic breadth (mean / max homogeneity)", fontsize=11)
ax.set_title(f"Metaphorical use vs syntactic conventionalisation breadth\n"
             f"Pearson r = {r:.3f} (p={p:.4f}), n={len(combined)} terms",
             fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig("comparisons/metaphor_vs_breadth_scatter.png", dpi=150, bbox_inches="tight")
print("Saved: comparisons/metaphor_vs_breadth_scatter.png")