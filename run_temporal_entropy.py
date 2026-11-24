"""
run_temporal_entropy.py
-----------------------
Batch-runs entropy_temporal.py over a directory of per-term JSON files.

Each file is named <term_id>.json (e.g. 0.json, 42.json) and contains:
  {
    "mentions": [count_bin0, count_bin1, ..., count_bin11, undated],
    "codes":    {"<feature_id>": [count_bin0, ...], ...}
  }

Counts may be either:
  - Raw per-decade counts     (each value independent)
  - Cumulative decade counts  (each value >= previous)

This script auto-detects which format each file uses and converts to
raw counts before passing to entropy_temporal.py.

Outputs
-------
For each term, results go to:
  <output_dir>/<term_id>/features.csv
  <output_dir>/<term_id>/decades.csv
  <output_dir>/<term_id>/rates.csv
  <output_dir>/<term_id>/counts.csv
  <output_dir>/<term_id>/plots/*.png

A combined summary across all terms is written to:
  <output_dir>/all_terms_entropy.csv

Usage
-----
  python run_temporal_entropy.py \\
      --input_dir ./data/term_jsons \\
      --output_dir ./results \\
      [--start_year 1800] \\
      [--term_names ./term_names.csv]  # optional: maps 0 -> "hat", 1 -> "dress" etc.
      [--no_plots] \\
      [--workers 4]
"""

import json
import argparse
import os
import sys
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Load the term_names lookup
# ---------------------------------------------------------------------------

def load_term_names_csv(path: str) -> dict:
    """
    CSV with columns: term_id, term_name
    e.g.   0,hat
           1,dress
    Returns {term_id_str: term_name}
    """
    df = pd.read_csv(path)
    id_col   = next((c for c in df.columns if "id"   in c.lower()), df.columns[0])
    name_col = next((c for c in df.columns if "name" in c.lower() or "term" in c.lower()), df.columns[1])
    return {str(row[id_col]): str(row[name_col]) for _, row in df.iterrows()}


def load_term_names_dir(term_dir: str) -> dict:
    """
    Read term names from a directory of JSON files (e.g. data/sae_logodds/term/).
    Each file is named <term_id>.json and contains a "term" field at the top level:
      {"term": "chemis", "examples": {...}}
    Returns {term_id_str: term_name}
    """
    names = {}
    for p in Path(term_dir).glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            if "term" in data:
                names[p.stem] = data["term"]
        except Exception:
            pass
    print(f"  Loaded {len(names)} term names from {term_dir}")
    return names


# ---------------------------------------------------------------------------
# 2. Detect and normalise count format
# ---------------------------------------------------------------------------

def is_cumulative(arr: list) -> bool:
    """
    Returns True if the array looks cumulative (non-decreasing, last
    dated bin >= all earlier bins).
    Uses the first 12 values (the decade bins); ignores the 13th.
    """
    dated = arr[:12]
    diffs = np.diff(dated)
    # allow a small number of zero-diffs (sparse term) but no decreases
    return bool(np.all(diffs >= 0) and np.any(diffs > 0) and dated[-1] > dated[0])


def to_raw(data: dict) -> dict:
    """
    If counts are cumulative, convert to per-decade raw counts in-place
    (returns a new dict, does not mutate input).
    Preserves the 13th value (undated) as-is.
    """
    def diff_arr(arr):
        dated_cum = arr[:12]
        undated   = arr[12] if len(arr) > 12 else 0
        raw       = list(np.diff(np.concatenate([[0], dated_cum])).astype(int))
        return raw + [undated]

    mentions_raw = data["mentions"]
    if is_cumulative(mentions_raw):
        return {
            "mentions": diff_arr(mentions_raw),
            "codes":    {fid: diff_arr(v) for fid, v in data["codes"].items()},
            "_was_cumulative": True,
        }
    else:
        return dict(data)


# ---------------------------------------------------------------------------
# 3. Import entropy_temporal functions directly (no subprocess)
# ---------------------------------------------------------------------------

def load_entropy_module(script_path: str):
    """Dynamically import entropy_temporal.py so we can call its functions."""
    spec   = importlib.util.spec_from_file_location("entropy_temporal", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 4. Process one term
# ---------------------------------------------------------------------------

def process_term(json_path: str,
                 output_dir: str,
                 term_name: str,
                 start_year: int,
                 base: float,
                 no_plots: bool,
                 script_path: str) -> dict:
    """
    Load one JSON file, run temporal entropy analysis, save outputs.
    Returns a summary dict for the combined CSV.
    """
    et = load_entropy_module(script_path)

    with open(json_path) as f:
        raw_data = json.load(f)

    # Skip empty files (term never appears in corpus)
    if sum(raw_data.get("mentions", [])) == 0:
        return {"term_id": Path(json_path).stem,
                "term_name": term_name,
                "total_mentions": 0,
                "skipped": True}

    data = to_raw(raw_data)
    was_cumulative = data.pop("_was_cumulative", False)

    term_id = Path(json_path).stem
    # Name the output folder "35_blazer" so it's human-readable
    folder_name = f"{term_id}_{term_name}" if term_name != term_id else term_id
    term_out = os.path.join(output_dir, folder_name)
    os.makedirs(term_out, exist_ok=True)

    # Always write a term.txt so the name is unambiguous
    with open(os.path.join(term_out, "term.txt"), "w") as f:
        f.write(f"{term_name}\n")

    # Write normalised (raw) JSON so entropy_temporal can read it
    norm_path = os.path.join(term_out, "_normalised.json")
    with open(norm_path, "w") as f:
        json.dump(data, f, default=lambda x: int(x) if hasattr(x, 'item') else x)

    # Decode + compute
    counts_df, mentions, undated = et.decode(data, start_year=start_year)
    rates_df  = et.compute_rates(counts_df, mentions)
    feat_df   = et.entropy_per_feature(rates_df, counts_df, base=base)
    decade_df = et.entropy_per_decade(rates_df, mentions, base=base)

    # Save CSVs
    for df, name in [
        (feat_df,                  "features.csv"),
        (decade_df,                "decades.csv"),
        (rates_df.reset_index(),   "rates.csv"),
        (counts_df.reset_index(),  "counts.csv"),
    ]:
        df.to_csv(os.path.join(term_out, name), index=False)

    with open(os.path.join(term_out, "undated.json"), "w") as f:
        json.dump({
            "mentions": int(undated["mentions"]),
            "codes": {k: int(v) for k, v in undated["codes"].items()}
        }, f, indent=2)

    if not no_plots:
        et.make_plots(feat_df, decade_df, rates_df, counts_df,
                      term_out, term=term_name)

    # Summary row
    total_mentions = int(mentions.sum())
    return {
        "term_id":           term_id,
        "term_name":         term_name,
        "folder":            folder_name,
        "total_mentions":    total_mentions,
        "n_features":        len(feat_df),
        "mean_norm_entropy": round(feat_df["norm_entropy"].mean(), 4),
        "std_norm_entropy":  round(feat_df["norm_entropy"].std(),  4),
        "n_rising_sig":      int(((feat_df["trend_p"] < 0.05) & (feat_df["trend_r"] > 0)).sum()),
        "n_falling_sig":     int(((feat_df["trend_p"] < 0.05) & (feat_df["trend_r"] < 0)).sum()),
        "peak_decade":       decade_df.loc[decade_df["mentions"].idxmax(), "decade"],
        "was_cumulative":    was_cumulative,
        "skipped":           False,
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch temporal entropy over term JSONs")
    parser.add_argument("--input_dir",   required=True,
                        help="Directory containing <term_id>.json files")
    parser.add_argument("--output_dir",  default="./results_temporal_batch",
                        help="Root output directory")
    parser.add_argument("--script",      default=None,
                        help="Path to entropy_temporal.py (default: same dir as this script)")
    parser.add_argument("--term_names",  default=None,
                        help="Optional CSV mapping term_id -> term_name")
    parser.add_argument("--term_dir",    default=None,
                        help="Directory of term JSONs (e.g. data/sae_logodds/term/) "
                             "each containing a 'term' field — used instead of --term_names")
    parser.add_argument("--start_year",  type=int, default=1800,
                        help="First decade start year (default: 1800)")
    parser.add_argument("--base",        type=float, default=2.0,
                        help="Entropy log base (default: 2 = bits)")
    parser.add_argument("--no_plots",    action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Parallel workers (default: 1; increase for large dirs)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Only process the first N files (useful for testing)")
    args = parser.parse_args()

    # Resolve entropy_temporal.py path
    if args.script:
        script_path = args.script
    else:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "entropy_temporal.py")
    if not os.path.exists(script_path):
        sys.exit(f"ERROR: cannot find entropy_temporal.py at {script_path}\n"
                 f"Pass --script /path/to/entropy_temporal.py explicitly.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect JSON files
    json_files = sorted(Path(args.input_dir).glob("*.json"),
                        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if args.limit:
        json_files = json_files[:args.limit]
    print(f"Found {len(json_files)} JSON files in {args.input_dir}")

    # Term name lookup — term_dir takes priority over term_names CSV
    term_names = {}
    if args.term_dir:
        print(f"Loading term names from {args.term_dir} ...")
        term_names = load_term_names_dir(args.term_dir)
    elif args.term_names:
        print(f"Loading term names from {args.term_names} ...")
        term_names = load_term_names_csv(args.term_names)
        print(f"  Loaded {len(term_names)} term names")

    # Process
    kwargs = dict(
        output_dir  = args.output_dir,
        start_year  = args.start_year,
        base        = args.base,
        no_plots    = args.no_plots,
        script_path = script_path,
    )

    results = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_term,
                          str(p),
                          term_name=term_names.get(p.stem, p.stem),
                          **kwargs): p
                for p in json_files
            }
            for i, fut in enumerate(as_completed(futures), 1):
                p = futures[fut]
                try:
                    row = fut.result()
                    results.append(row)
                    status = "SKIP" if row.get("skipped") else "OK  "
                    print(f"  [{i:>4}/{len(json_files)}] {status}  {p.name}  "
                          f"({row.get('total_mentions', 0):,} mentions)")
                except Exception as e:
                    print(f"  [{i:>4}/{len(json_files)}] ERROR  {p.name}: {e}")
    else:
        for i, p in enumerate(json_files, 1):
            term_name = term_names.get(p.stem, p.stem)
            try:
                row = process_term(str(p), term_name=term_name, **kwargs)
                results.append(row)
                status = "SKIP" if row.get("skipped") else "OK  "
                print(f"  [{i:>4}/{len(json_files)}] {status}  {p.name}  "
                      f"({row.get('total_mentions', 0):,} mentions)")
            except Exception as e:
                print(f"  [{i:>4}/{len(json_files)}] ERROR  {p.name}: {e}")

    # Combined summary CSV
    summary_df = pd.DataFrame(results).sort_values("total_mentions", ascending=False)
    summary_path = os.path.join(args.output_dir, "all_terms_entropy.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved combined summary: {summary_path}")

    # Print top/bottom by entropy
    done = summary_df[~summary_df["skipped"].fillna(False)]
    if len(done):
        print(f"\n{'Term':<20} {'mentions':>10} {'mean_H':>8} {'rising*':>8} {'falling*':>9}")
        print("-" * 60)
        for _, r in done.head(20).iterrows():
            print(f"  {str(r['term_name']):<18} {r['total_mentions']:>10,} "
                  f"{r['mean_norm_entropy']:>8.3f} "
                  f"{int(r['n_rising_sig']):>8} "
                  f"{int(r['n_falling_sig']):>9}")

    skipped = summary_df[summary_df["skipped"].fillna(False)]
    if len(skipped):
        print(f"\nSkipped {len(skipped)} empty terms: "
              f"{', '.join(skipped['term_id'].tolist()[:10])}"
              f"{'...' if len(skipped) > 10 else ''}")


if __name__ == "__main__":
    main()