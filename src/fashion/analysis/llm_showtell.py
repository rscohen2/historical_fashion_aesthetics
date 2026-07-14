"""Describe characters with an LLM served by a *separate* vLLM server.

Instead of loading the model in-process (offline ``vllm.LLM``), this talks to a
vLLM OpenAI-compatible server that you launch yourself, e.g.::

    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-8B \
        --reasoning-parser qwen3 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}' \
        --max-model-len 8192 \
        --port 8000

Input is the passages parquet produced by ``load_passages.py`` (run that first).
Rows are streamed off disk in batches so we never hold the whole corpus in
memory, each row's prompt is built on the fly, and the client fires many requests
concurrently (bounded by ``--concurrency``). vLLM's continuous batching keeps the
GPU saturated, so we still get high-throughput inference without hosting the model
here.
"""

import asyncio
import json

import pyarrow.parquet as pq
from openai import AsyncOpenAI
from tqdm import tqdm

from fashion.analysis.load_passages import passages_path
from fashion.paths import DATA_DIR

PROMPT = (
    "You are a close reader of literature. "
    "Given this passage of text, use adjectives to describe the character enclosed in the asterisks.\n"
    "DO NOT make up any information about the character that is not in the passage.\n"
    "Respond with a JSON object of the form "
    '{{"adjectives": [{{"word": <adjective>, "reasoning": <why it applies, '
    "grounded in the passage>}}]}}.\n"
    "{text}"
)

# PROMPT = (
#     "You are a close reader of literature. "
#     "Paraphrase this passage, with extra care towards describing the character in asterisks.\n"
#     "DO NOT make up any information about the character that is not in the passage.\n"
#     "{text}"
# )

# JSON schema the server is constrained to emit via guided decoding
# (``response_format``), so every reply parses into
# ``{"adjectives": [{"word", "reasoning"}, ...]}``.
ADJECTIVES_SCHEMA = {
    "type": "object",
    "properties": {
        "adjectives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["word", "reasoning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["adjectives"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "character_adjectives", "schema": ADJECTIVES_SCHEMA},
}


def build_prompt(row):
    """Build the chat message for one passage row from ``load_passages.py``.

    The stored offsets are book-level, so subtract the passage start to get the
    character span's position within the passage text, then wrap it in ``**`` so
    the model knows which character to describe.
    """
    passage = row["passage"]
    c_start = int(row["character_start_idx"] - row["passage_start_idx"])
    c_end = int(row["character_end_idx"] - row["passage_start_idx"])
    bracketed = (
        passage[:c_start] + "**" + passage[c_start:c_end] + "**" + passage[c_end:]
    )
    return {"role": "user", "content": PROMPT.format(text=bracketed)}


def stream_rows(parquet_path, batch_size):
    """Yield passage rows as dicts, reading the parquet in batches off disk.

    Streaming keeps memory bounded regardless of corpus size, so a fast reader
    can't materialize millions of rows ahead of slower inference.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


async def describe_one(client, args, row_id, message):
    """Send a single chat request, retrying a few times on transient errors.

    Returns a result dict with the parsed ``adjectives`` list (``None`` if the
    reply could not be JSON-parsed, or on persistent failure, so one bad row
    never aborts the whole run). Concurrency is bounded by the number of consumer
    tasks that call this; vLLM batches them server-side.
    """
    extra_body = {"chat_template_kwargs": {"enable_thinking": args.thinking}}
    # vLLM caps the reasoning span at this many tokens (a non-standard request
    # field, so it rides in extra_body). Only meaningful while thinking is on;
    # -1 means unlimited, matching vLLM's own convention.
    if args.thinking and args.thinking_budget >= 0:
        extra_body["thinking_token_budget"] = args.thinking_budget
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
            choice = completion.choices[0].message
            content = choice.content
            # Guided decoding constrains content to ADJECTIVES_SCHEMA, but a
            # max_tokens cutoff can still truncate the JSON mid-object, so parse
            # defensively and keep the raw text when it doesn't round-trip.
            try:
                adjectives = json.loads(content)["adjectives"]
            except (TypeError, KeyError, json.JSONDecodeError):
                print(f"Row ID: {row_id} - could not parse JSON response.")
                adjectives = None
            return {
                "row_id": row_id,
                "prompt": message["content"],
                # Qwen3 reasoning is split out by the server's reasoning
                # parser; it is absent when thinking is disabled.
                "reasoning": getattr(choice, "reasoning", None),
                "adjectives": adjectives,
                "response": content,
            }
        except Exception as error:  # noqa: BLE001 - report and retry any API error
            last_error = error
            await asyncio.sleep(2**attempt)
    print(f"Row ID: {row_id} - failed after {args.max_retries} retries: {last_error}")
    return {
        "row_id": row_id,
        "prompt": message["content"],
        "reasoning": None,
        "adjectives": None,
        "response": None,
    }


async def run(args, parquet_path, total, jsonl_path):
    """Pipeline passage streaming and inference so requests start ASAP.

    A producer streams passage rows off disk, builds each prompt, and pushes it
    onto a bounded queue that a pool of ``--concurrency`` consumer tasks drains,
    calling the vLLM server. Results are appended to ``jsonl_path`` the moment
    each request finishes.
    """
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    # Bound the queue so a fast reader can't build an unbounded prompt backlog
    # ahead of slower inference (backpressure), while keeping consumers fed.
    queue = asyncio.Queue(maxsize=args.concurrency * 4)
    results = []

    async def produce():
        for row in stream_rows(parquet_path, args.batch_size):
            await queue.put((row["row_id"], build_prompt(row)))
        # One sentinel per consumer so each one exits after the queue drains.
        for _ in range(args.concurrency):
            await queue.put(None)

    async def consume(jsonl_file, progress):
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                row_id, message = item
                result = await describe_one(client, args, row_id, message)
                jsonl_file.write(json.dumps(result) + "\n")
                jsonl_file.flush()
                results.append(result)
                progress.update(1)
            finally:
                queue.task_done()

    print(
        f"Streaming passages and inferring (concurrency={args.concurrency}) "
        "in parallel..."
    )
    try:
        with (
            open(jsonl_path, "w") as jsonl_file,
            tqdm(desc="describing", total=total) as progress,
        ):
            consumers = [
                asyncio.create_task(consume(jsonl_file, progress))
                for _ in range(args.concurrency)
            ]
            await produce()
            await asyncio.gather(*consumers)
    finally:
        await client.close()
    return results


def main(args):
    input_path = passages_path(args.debug)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run `python -m fashion.analysis.load_passages"
            f"{' --debug' if args.debug else ''}` first."
        )

    # num_rows comes from the parquet footer, so the progress bar gets an exact
    # total without materializing the file.
    total = pq.ParquetFile(input_path).metadata.num_rows

    output_dir = DATA_DIR / "analysis" / "llm_showtell"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "descriptions.jsonl"
    output_path = output_dir / "descriptions.parquet"

    results = asyncio.run(run(args, input_path, total, jsonl_path))

    import pandas as pd

    pd.DataFrame(results).to_parquet(output_path, index=False)
    print(f"Wrote {len(results)} descriptions to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fashion LLM Show-Tell")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Read passages_debug.parquet instead of passages.parquet.",
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
        "--batch-size",
        type=int,
        default=1024,
        help="Rows read per parquet batch while streaming passages.",
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
        default=0.7,
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
    args = parser.parse_args()

    main(args)
