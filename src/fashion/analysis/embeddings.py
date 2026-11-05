"""Sentence embeddings for every fashion mention, data-parallel across GPUs.

Follows notebooks/sentence_embeddings.ipynb (Qwen3-Embedding-0.6B via
sentence-transformers, SDPA attention, left padding), but embeds each distinct
text once and splits the work across GPUs with fashion.distributed.

What gets embedded per mention:
  --context sentence  the mention's sentence (default)
  --context excerpt   the excerpt the wearer model saw: BookNLP sentences spanning
                      the 10 character mentions nearest the item, clipped to 500
                      characters on either side of it (extract_char_fashion_mentions),
                      read from the wearer model's output in --wearing-dir
  --mark-spans item wearer
                      wrap the fashion item and/or its wearer (the character
                      mention the wearer model picked) as [..]_item / [..]_wearer
  --prompt TEMPLATE   a Qwen3 instruction, "Instruct: {prompt}\\nQuery:{text}";
                      may use {term} and {wearer} (the wearer mention as written)

Rank 0 builds the texts and writes sentences.parquet (one row group per shard) and
sentence_idx.parquet, then spawns one worker per GPU (--num_processes, default 4).
Each worker reads its shards' row groups and writes shards/NNNN.npy. Shards already
on disk are skipped, so a crashed or retried run resumes where it left off;
setup.json records the settings, and a rerun with different ones is refused. Once
all workers finish, rank 0 stitches the shards into one array.

Output: data/analysis/embeddings/ (pick another --output-dir for each variant)
          sentence_embeddings.npy  (n_texts, dim) float16, row i embeds
                                   sentences.parquet row i
          sentences.parquet        the unique texts, in first-seen order: `sentence`
                                   (the context, with any markers) and, with
                                   --prompt, `instruction`
          sentence_idx.parquet     per input row (same index as the input):
                                   sentence_idx into the two files above
          setup.json               the settings the run was made with
"""

import argparse
import itertools
import json
import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from fashion.distributed import add_distributed_args, run_distributed
from fashion.paths import DATA_DIR

EXPERIMENT = "embeddings"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
WEARING_DIR = DATA_DIR / "pipeline" / "hathi_all" / "wearing"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MARKERS = {"item": ("[", "]_item"), "wearer": ("[", "]_wearer")}
PROMPT_FIELDS = {"term", "wearer"}


def model_input(sentence: str, instruction: str | None) -> str:
    """The string the model sees, in Qwen3-Embedding's instruction format."""
    if instruction is None:
        return sentence
    return f"Instruct: {instruction}\nQuery:{sentence}"


def prompt_fields(prompt: str | None) -> set[str]:
    if prompt is None:
        return set()
    return {field for _, field, _, _ in string.Formatter().parse(prompt) if field}


def load_excerpts(wearing_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    """excerpt_start and excerpt_text, the window the wearer model saw, for each
    (book_id, mention_id) in keys."""
    from pyarrow import json as pa_json

    files = sorted(wearing_dir.glob("*.ndjson"))
    if not files:
        raise FileNotFoundError(f"No wearer model outputs (*.ndjson) in {wearing_dir}")
    schema = pa.schema(
        [
            ("book_id", pa.string()),
            ("mention_id", pa.int64()),
            ("excerpt_start", pa.int64()),
            ("excerpt_text", pa.string()),
        ]
    )
    parse_options = pa_json.ParseOptions(
        explicit_schema=schema, unexpected_field_behavior="ignore"
    )
    # each line also carries the candidate characters, so lines can be long
    read_options = pa_json.ReadOptions(block_size=1 << 24)
    keys = keys.drop_duplicates()
    parts = []
    for path in tqdm(files, desc="Loading excerpts"):
        table = pa_json.read_json(
            path, read_options=read_options, parse_options=parse_options
        )
        parts.append(table.to_pandas().merge(keys, on=["book_id", "mention_id"]))
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["book_id", "mention_id"]
    )


def relative_span(
    length: int, base: int, start: float, end: float
) -> tuple[int, int] | None:
    """Book-level [start, end) as offsets into a text of `length` starting at
    `base`, or None if it doesn't lie inside that text."""
    if pd.isna(start) or pd.isna(end):
        return None
    start, end = int(start) - base, int(end) - base
    if 0 <= start <= end <= length:
        return start, end
    return None


def insert_markers(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Wrap each (start, end, kind) span of text in MARKERS[kind]. Spans may nest
    but not partly overlap."""
    # at one position, closes come before opens, outer spans open first and
    # inner spans close first
    events = []
    for start, end, kind in spans:
        open_mark, close_mark = MARKERS[kind]
        events.append((start, 1, start - end, open_mark))
        events.append((end, 0, end - start, close_mark))
    pieces = []
    last = 0
    for position, _, _, mark in sorted(events):
        pieces += [text[last:position], mark]
        last = position
    pieces.append(text[last:])
    return "".join(pieces)


def build_texts(
    input_path: Path,
    context: str,
    mark_spans: list[str],
    prompt: str | None,
    wearing_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row codes and the unique (sentence, instruction) texts they index."""
    df = pd.read_parquet(
        input_path,
        columns=[
            "mention_id",
            "term",
            "sentence",
            "start_idx",
            "end_idx",
            "sentence_start_idx",
            "character_start_idx",
            "character_end_idx",
        ],
    )
    index = df.index
    df = df.reset_index()
    texts = df.sentence.tolist()
    bases = df.sentence_start_idx.tolist()
    # the wearer's mention as written; it may lie outside the sentence, so with
    # {wearer} the excerpt is loaded to find it even when embedding sentences
    wearer_texts, wearer_bases = texts, bases
    if context == "excerpt" or "wearer" in prompt_fields(prompt):
        excerpts = df[["book_id", "mention_id"]].merge(
            load_excerpts(wearing_dir, df[["book_id", "mention_id"]]),
            how="left",
            on=["book_id", "mention_id"],
        )
        has_excerpt = excerpts.excerpt_text.notna().to_numpy()
        print(f"{(~has_excerpt).sum():,} rows have no excerpt; using their sentence")
        excerpt_texts = excerpts.excerpt_text.where(has_excerpt, df.sentence).tolist()
        excerpt_bases = (
            excerpts.excerpt_start.where(has_excerpt, df.sentence_start_idx)
            .astype(int)
            .tolist()
        )
        wearer_texts, wearer_bases = excerpt_texts, excerpt_bases
        if context == "excerpt":
            texts, bases = excerpt_texts, excerpt_bases

    item_starts = (df.sentence_start_idx + df.start_idx).tolist()
    item_ends = (df.sentence_start_idx + df.end_idx).tolist()
    wearer_starts = df.character_start_idx.tolist()
    wearer_ends = df.character_end_idx.tolist()
    terms = df.term.tolist()
    unmarked = {kind: 0 for kind in mark_spans}
    no_wearer_text = 0
    sentences, instructions = [], []
    for i, (text, base) in enumerate(zip(texts, bases)):
        item = relative_span(len(text), base, item_starts[i], item_ends[i])
        wearer = relative_span(len(text), base, wearer_starts[i], wearer_ends[i])
        # BookNLP mentions can be noun phrases containing the item ("a man in
        # full dress"), which nest; a partial overlap can't be marked
        if wearer and item and wearer[0] < item[1] and item[0] < wearer[1]:
            nested = (wearer[0] <= item[0] and item[1] <= wearer[1]) or (
                item[0] <= wearer[0] and wearer[1] <= item[1]
            )
            if not nested:
                wearer = None
        spans = []
        for kind, span in (("item", item), ("wearer", wearer)):
            if kind not in mark_spans:
                continue
            if span is None:
                unmarked[kind] += 1
            else:
                spans.append((*span, kind))
        sentences.append(insert_markers(text, spans))

        if prompt is None:
            instructions.append(None)
            continue
        wearer_span = relative_span(
            len(wearer_texts[i]), wearer_bases[i], wearer_starts[i], wearer_ends[i]
        )
        if wearer_span is None:
            no_wearer_text += 1
            wearer_text = ""
        else:
            wearer_text = wearer_texts[i][wearer_span[0] : wearer_span[1]]
        instructions.append(prompt.format(term=terms[i], wearer=wearer_text))

    for kind, count in unmarked.items():
        print(f"{count:,} rows have no {kind} span inside their context to mark")
    if "wearer" in prompt_fields(prompt):
        print(f"{no_wearer_text:,} rows have no wearer mention; {{wearer}} is empty")

    keys = [model_input(s, p) for s, p in zip(sentences, instructions)]
    codes, _ = pd.factorize(pd.Series(keys))
    # codes count up in first-seen order, so each code's first row is ascending
    _, first = np.unique(codes, return_index=True)
    units = pd.DataFrame({"sentence": np.asarray(sentences, dtype=object)[first]})
    if prompt is not None:
        units["instruction"] = np.asarray(instructions, dtype=object)[first]
    return pd.DataFrame({"sentence_idx": codes}, index=index), units


def shard_bounds(n: int, num_shards: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, num_shards + 1).astype(int)
    return list(itertools.pairwise(edges.tolist()))


def write_units(units: pd.DataFrame, path: Path, num_shards: int) -> None:
    """Write units with one row group per shard, so a worker reads only its own."""
    table = pa.Table.from_pandas(units, preserve_index=False)
    tmp_path = path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp_path, table.schema) as writer:
        for start, end in shard_bounds(len(units), num_shards):
            writer.write_table(
                table.slice(start, end - start), row_group_size=end - start
            )
    os.replace(tmp_path, path)


def embed_shards(
    shard_ids: list[int],
    units_path: Path,
    shard_dir: Path,
    num_shards: int,
    batch_size: int,
    dtype: str,
) -> None:
    pending = [i for i in shard_ids if not (shard_dir / f"{i:04d}.npy").exists()]
    if not pending:
        return

    import torch
    from sentence_transformers import SentenceTransformer

    units_file = pq.ParquetFile(units_path)
    if units_file.num_row_groups != num_shards:
        raise RuntimeError(
            f"{units_path} has {units_file.num_row_groups} row groups, "
            f"expected one per shard ({num_shards})"
        )
    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
        model_kwargs={
            "attn_implementation": "sdpa",
            "torch_dtype": getattr(torch, dtype),
        },
        processor_kwargs={"padding_side": "left"},
    )
    for i in pending:
        shard = units_file.read_row_group(i).to_pandas()
        instructions = (
            shard.instruction if "instruction" in shard else [None] * len(shard)
        )
        print(f"Shard {i}: {len(shard):,} texts")
        embeddings = model.encode(
            [model_input(s, p) for s, p in zip(shard.sentence, instructions)],
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float16)
        # write then rename, so a worker killed mid-write leaves no partial shard
        tmp_path = shard_dir / f"{i:04d}.npy.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, embeddings)
        os.replace(tmp_path, shard_dir / f"{i:04d}.npy")


def merge_shards(shard_dir: Path, num_shards: int, n: int, output_path: Path) -> None:
    bounds = shard_bounds(n, num_shards)
    missing = [
        i for i in range(num_shards) if not (shard_dir / f"{i:04d}.npy").exists()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} shards missing after embedding: {missing}")

    dim = np.load(shard_dir / "0000.npy", mmap_mode="r").shape[1]
    merged = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(n, dim)
    )
    for i, (start, end) in enumerate(bounds):
        shard = np.load(shard_dir / f"{i:04d}.npy", mmap_mode="r")
        if shard.shape != (end - start, dim):
            raise RuntimeError(
                f"Shard {i} has shape {shard.shape}, expected {(end - start, dim)}"
            )
        merged[start:end] = shard
    merged.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Override the input parquet."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where to write outputs."
    )
    parser.add_argument(
        "--context",
        choices=["sentence", "excerpt"],
        default="sentence",
        help="Embed each mention's sentence, or the excerpt the wearer model saw.",
    )
    parser.add_argument(
        "--mark-spans",
        nargs="+",
        choices=sorted(MARKERS),
        default=[],
        help="Spans to wrap in markers: [..]_item for the fashion item, [..]_wearer "
        "for the wearer's mention (left unmarked when outside the context).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Instruction for the embedding model; may use {term} and {wearer}.",
    )
    parser.add_argument(
        "--wearing-dir",
        type=Path,
        default=WEARING_DIR,
        help="The wearer model's output (*.ndjson), for --context excerpt and "
        "{wearer}.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=64,
        help="Number of shards the unique texts are split into (the resume unit).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model weight dtype. Embeddings are stored as float16 regardless.",
    )
    add_distributed_args(parser)
    args = parser.parse_args()
    unknown = prompt_fields(args.prompt) - PROMPT_FIELDS
    if unknown:
        parser.error(f"--prompt may only use {sorted(PROMPT_FIELDS)}, not {unknown}")
    return args


def main() -> None:
    args = parse_args()
    input_path = args.input or (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )
    output_dir: Path = args.output_dir
    shard_dir = output_dir / "shards"
    units_path = output_dir / "sentences.parquet"
    setup_path = output_dir / "setup.json"
    embeddings_path = output_dir / "sentence_embeddings.npy"
    rank = int(os.environ.get("DISTRIBUTED_RANK", "0"))

    num_shards = args.num_shards
    if rank == 0:
        if embeddings_path.exists():
            print(f"{embeddings_path} already exists; delete it to recompute.")
            return
        uses_excerpts = args.context == "excerpt" or "wearer" in prompt_fields(
            args.prompt
        )
        setup = {
            "input": str(input_path),
            "context": args.context,
            "mark_spans": sorted(set(args.mark_spans)),
            "prompt": args.prompt,
            "wearing_dir": str(args.wearing_dir) if uses_excerpts else None,
            "num_shards": args.num_shards,
            "model": MODEL_NAME,
            "dtype": args.dtype,
        }
        # setup.json is written last, so its presence means the texts are complete
        if setup_path.exists():
            previous = json.loads(setup_path.read_text())
            if previous != setup:
                raise SystemExit(
                    f"{output_dir} holds a run with other settings ({previous}); "
                    "delete it or pick another --output-dir."
                )
            print(f"Reusing the texts in {units_path}")
        else:
            sentence_idx, units = build_texts(
                input_path,
                args.context,
                setup["mark_spans"],
                args.prompt,
                args.wearing_dir,
            )
            num_shards = min(args.num_shards, len(units))
            print(
                f"{len(sentence_idx):,} rows, {len(units):,} unique texts, "
                f"{num_shards} shards"
            )
            example = units.iloc[0]
            example = model_input(example.sentence, example.get("instruction"))
            print(f"Example input:\n{example}")
            output_dir.mkdir(parents=True, exist_ok=True)
            if shard_dir.exists():
                # left by an earlier run whose settings aren't known
                for shard in shard_dir.glob("*.npy"):
                    shard.unlink()
            write_units(units, units_path, num_shards)
            sentence_idx.to_parquet(output_dir / "sentence_idx.parquet")
            setup_path.write_text(json.dumps(setup, indent=2) + "\n")
        num_shards = pq.ParquetFile(units_path).num_row_groups
        shard_dir.mkdir(exist_ok=True)

    extra_args = [
        "--output-dir",
        str(output_dir),
        "--num-shards",
        str(num_shards),
        "--batch-size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
    ]
    run_distributed(
        lambda shard_ids: embed_shards(
            shard_ids, units_path, shard_dir, num_shards, args.batch_size, args.dtype
        ),
        list(range(num_shards)),
        script_path=__file__,
        total_processes=args.num_processes,
        concurrent_processes=args.concurrent_processes,
        extra_args=extra_args,
    )
    if rank != 0:
        return

    n = pq.ParquetFile(units_path).metadata.num_rows
    merge_shards(shard_dir, num_shards, n, embeddings_path)
    for shard in shard_dir.glob("*.npy"):
        shard.unlink()
    shard_dir.rmdir()
    shape = np.load(embeddings_path, mmap_mode="r").shape
    print(f"Wrote {embeddings_path} with shape {shape}")


if __name__ == "__main__":
    main()
