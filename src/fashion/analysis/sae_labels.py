"""Label SAE codes with an LLM served by a *separate* vLLM server.

For each code, the model sees the N_TOP sentences the code fires on most strongly,
N_RANDOM more drawn at random from the rest of the sentences it fires on, and
N_NEGATIVE random sentences it does not fire on, and is asked for a short label and
a one-sentence description of what the firing sentences share. Activation values
are not shown. As in llm_showtell.py, launch the server yourself, e.g.::

    vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --max-model-len 8192 --port 8000

Rows of codes.pt line up with rows of sentences.parquet: both are over the unique
sentences embedded by embeddings.py.

Requires embeddings.py, sae.py and sae_inference.py to have been run first.

Output: data/analysis/sae_labels/labels.parquet
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from openai import AsyncOpenAI
from tqdm import tqdm

from fashion.analysis.embeddings import OUTPUT_DIR as EMBEDDINGS_PATH
from fashion.analysis.sae import OUTPUT_DIR as SAE_PATH
from fashion.paths import DATA_DIR

EXPERIMENT = "sae_labels"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT

N_TOP = 5
N_RANDOM = 5
N_NEGATIVE = 10
# Truncate long sentences so a prompt of 20 stays well under --max-model-len.
MAX_CHARS = 500

PROMPT = (
    "You are interpreting a feature learned by a sparse autoencoder over sentence "
    "embeddings from works of fiction. The feature fires on every sentence under "
    "ACTIVATING and on none of the sentences under NOT ACTIVATING.\n"
    "What do the activating sentences have in common that the others lack? It may "
    "be a topic, a setting, a kind of action or event, a tone, or a grammatical "
    "pattern.\n"
    "Respond with a JSON object of the form "
    '{{"label": <a 2-5 word label>, "description": <one sentence>}}.\n\n'
    "ACTIVATING:\n{positive}\n\n"
    "NOT ACTIVATING:\n{negative}"
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["label", "description"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "code_label", "schema": LABEL_SCHEMA},
}


def sample_examples(indices, values, min_count, rng):
    """Map each code to its (positive, negative) sentence rows.

    Codes firing on fewer than ``min_count`` rows are skipped. A code fires on a
    row when it is among the row's top-k indices with a positive value.
    """
    n_rows = len(indices)
    active = values > 0
    rows = np.nonzero(active)[0]
    codes = indices[active]
    # Sort by code, then by descending activation within each code.
    order = np.lexsort((-values[active], codes))
    rows, codes = rows[order], codes[order]
    unique_codes, starts, counts = np.unique(
        codes, return_index=True, return_counts=True
    )

    examples = {}
    for code, start, count in zip(unique_codes, starts, counts):
        if count < min_count:
            continue
        code_rows = rows[start : start + count]
        rest = code_rows[N_TOP:]
        positive = np.concatenate(
            [
                code_rows[:N_TOP],
                rng.choice(rest, size=min(N_RANDOM, len(rest)), replace=False),
            ]
        )
        # Rejection-sample negatives; most codes fire on a small fraction of rows.
        negative = []
        while len(negative) < N_NEGATIVE:
            candidates = rng.integers(n_rows, size=N_NEGATIVE)
            fires = ((indices[candidates] == code) & active[candidates]).any(axis=1)
            negative.extend(candidates[~fires])
        examples[int(code)] = (positive, np.array(negative[:N_NEGATIVE]))
    return examples


def format_sentences(sentences):
    return "\n".join(f"{i}. {s[:MAX_CHARS]}" for i, s in enumerate(sentences, 1))


def build_message(sentences, positive, negative, rng):
    """Build the chat message for one code, shuffling so top examples aren't first."""
    content = PROMPT.format(
        positive=format_sentences(sentences[rng.permutation(positive)]),
        negative=format_sentences(sentences[negative]),
    )
    return {"role": "user", "content": content}


async def label_one(client, args, code, message):
    """Send a single chat request, retrying a few times on transient errors.

    ``label`` and ``description`` are ``None`` if the reply could not be parsed or
    on persistent failure, so one bad code never aborts the whole run.
    """
    extra_body = {"chat_template_kwargs": {"enable_thinking": args.thinking}}
    if args.thinking and args.thinking_budget >= 0:
        extra_body["thinking_token_budget"] = args.thinking_budget
    result = {
        "code": code,
        "label": None,
        "description": None,
        "prompt": message["content"],
        "reasoning": None,
        "response": None,
    }
    last_error = None
    for attempt in range(args.max_retries):
        try:
            completion = await client.chat.completions.create(
                model=args.model,
                messages=[message],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                response_format=RESPONSE_FORMAT,
                extra_body=extra_body,
            )
        except Exception as error:  # noqa: BLE001 - report and retry any API error
            last_error = error
            await asyncio.sleep(2**attempt)
            continue
        choice = completion.choices[0].message
        result["reasoning"] = getattr(choice, "reasoning", None)
        result["response"] = choice.content
        # A max_tokens cutoff can truncate the JSON despite guided decoding.
        try:
            parsed = json.loads(choice.content)
            result["label"] = parsed["label"]
            result["description"] = parsed["description"]
        except (TypeError, KeyError, json.JSONDecodeError):
            print(f"Code {code} - could not parse JSON response.")
        return result
    print(f"Code {code} - failed after {args.max_retries} retries: {last_error}")
    return result


async def run(args, messages, jsonl_file):
    """Label codes concurrently, appending each result to ``jsonl_file`` as it lands."""
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(code, message):
        async with semaphore:
            return await label_one(client, args, code, message)

    tasks = [bounded(code, message) for code, message in messages.items()]
    try:
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="labeling"
        ):
            jsonl_file.write(json.dumps(await task) + "\n")
            jsonl_file.flush()
    finally:
        await client.close()


def read_jsonl(path):
    if not path.exists():
        return pd.DataFrame(columns=["code", "label"])
    return pd.read_json(path, lines=True)


def main(args):
    codes = t.load(args.trainer_dir / "codes.pt")
    indices = codes["indices"].numpy()
    values = codes["values"].float().numpy()
    sentences = pd.read_parquet(EMBEDDINGS_PATH / "sentences.parquet").sentence
    sentences = sentences.to_numpy()
    assert len(sentences) == len(indices), "codes.pt and sentences.parquet differ"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "labels.jsonl"
    output_path = args.output_dir / "labels.parquet"

    # Resume: skip codes already labeled by an earlier run; failures are retried.
    previous = read_jsonl(jsonl_path)
    done = set(previous.code[previous.label.notna()])

    rng = np.random.default_rng(args.seed)
    print("Sampling examples per code...")
    examples = sample_examples(indices, values, args.min_count, rng)
    messages = {
        code: build_message(sentences, positive, negative, rng)
        for code, (positive, negative) in examples.items()
        if code not in done
    }
    print(f"Labeling {len(messages)} codes ({len(done)} already done).")

    with open(jsonl_path, "a") as jsonl_file:
        asyncio.run(run(args, messages, jsonl_file))

    labels = (
        read_jsonl(jsonl_path)
        .drop_duplicates(subset="code", keep="last")
        .sort_values("code")
    )
    labels.to_parquet(output_path, index=False)
    print(f"Wrote {labels.label.notna().sum()} labels to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label SAE codes with an LLM")
    parser.add_argument(
        "--trainer-dir",
        type=Path,
        default=SAE_PATH / "trainer_0",
        help="SAE directory holding codes.pt (default: data/analysis/sae/trainer_0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write labels (default: data/analysis/sae_labels).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Base URL of the running vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the server (vLLM ignores it unless configured).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model name as served by the vLLM server.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max in-flight requests; vLLM batches these server-side.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per request (includes reasoning).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode (enabled by default).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=512,
        help="Max reasoning tokens before the model is forced to answer "
        "(-1 for unlimited). Ignored when --no-thinking is set.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per request on transient server errors.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=N_TOP + N_RANDOM,
        help="Skip codes firing on fewer than N sentences (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling examples.",
    )
    main(parser.parse_args())
