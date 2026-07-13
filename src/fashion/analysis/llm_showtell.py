"""Describe characters with an LLM served by a *separate* vLLM server.

Instead of loading the model in-process (offline ``vllm.LLM``), this talks to a
vLLM OpenAI-compatible server that you launch yourself, e.g.::

    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-8B \
        --reasoning-parser qwen3 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}' \
        --max-model-len 8192 \
        --port 8000

The client fires many requests concurrently (bounded by ``--concurrency``) and
lets vLLM's continuous batching keep the GPU saturated, so we still get
high-throughput inference without hosting the model here.
"""

import asyncio
import json
import os
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer
from openai import AsyncOpenAI
from tqdm import tqdm

from fashion.paths import DATA_DIR
from fashion.sources import HathiAll

# Punkt is unsupervised and rule-based, so it is much faster than a neural
# sentence segmenter (spacy/stanza) while being accurate enough for trimming
# context. Instantiate once and reuse across calls.
_SENTENCE_TOKENIZER = PunktSentenceTokenizer()

# Cached per worker process: HathiAll() stats every book listed in all.txt on
# construction, so we build it lazily once and reuse it for all books a worker
# handles instead of paying that cost per book.
_SOURCE = None


def _get_source():
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = HathiAll()
    return _SOURCE


@dataclass
class Passage:
    text: str
    start_idx: int
    end_idx: int


def find_enclosing_sentences(
    text, spans, start_idx, end_idx, max_sentences=1
) -> Passage | None:  #
    """
    Return the passage of ``text`` covering the sentence(s) that overlap
    ``[start_idx, end_idx)`` plus up to ``max_sentences`` sentences on each side.

    The result is a byte-accurate slice of ``text`` (from the start offset of
    the first kept sentence to the end offset of the last), so all original
    whitespace and character offsets are preserved. Returns ``None`` if no
    sentence overlaps the span.
    """
    # Punkt spans are sorted and non-overlapping, so the sentences overlapping
    # [start_idx, end_idx) form a contiguous range. Both endpoints of that range
    # are found by binary search instead of scanning every span:
    #   first = first sentence whose end is past start_idx (ends are ascending)
    #   last  = last sentence whose start is before end_idx (starts are ascending)
    first = bisect_right(spans, start_idx, key=lambda span: span[1])
    last = bisect_left(spans, end_idx, key=lambda span: span[0]) - 1
    if first > last:  # empty text, or the span falls between/outside sentences
        return None

    lo = max(0, first - max_sentences)
    hi = min(len(spans) - 1, last + max_sentences)
    passage_start_idx = spans[lo][0]
    passage_end_idx = spans[hi][1]
    return Passage(
        text=text[passage_start_idx:passage_end_idx],
        start_idx=passage_start_idx,
        end_idx=passage_end_idx,
    )


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


def build_book_prompts(book_id, book_df):
    """Load one book, tokenize it once, and build the prompt for each of its rows.

    Runs in a worker process (see ``run``): loading the text (disk I/O) and Punkt
    tokenization (CPU-bound, holds the GIL) are the slow parts, and doing them
    per book lets the process pool parallelize across books. Returns a list of
    ``(row_id, message)`` where ``message`` is ``None`` when no sentence encloses
    the character span, so the caller can report the skip.
    """
    text = _get_source().load_text(book_id).text
    spans = list(_SENTENCE_TOKENIZER.span_tokenize(text))
    prompts = []
    for row_id, row in book_df.iterrows():
        passage = find_enclosing_sentences(
            text, spans, row.sentence_start_idx, row.sentence_end_idx, max_sentences=3
        )
        if passage is None:
            prompts.append((row_id, None))
            continue
        c_start = int(row.character_start_idx - passage.start_idx)
        c_end = int(row.character_end_idx - passage.start_idx)
        bracketed = (
            passage.text[:c_start]
            + "**"
            + passage.text[c_start:c_end]
            + "**"
            + passage.text[c_end:]
        )
        prompts.append(
            (row_id, {"role": "user", "content": PROMPT.format(text=bracketed)})
        )
    return prompts


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


async def run(args, df, jsonl_path):
    """Pipeline prompt-building and inference so requests start ASAP.

    A process pool builds each book's prompts in parallel (``build_book_prompts``);
    as each book finishes, its prompts are pushed onto a bounded queue that a pool
    of ``--concurrency`` consumer tasks drains, calling the vLLM server. So
    inference on the first ready book overlaps with tokenizing the rest instead of
    waiting for every prompt to be built up front. Results are appended to
    ``jsonl_path`` the moment each request finishes.
    """
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    # Bound the queue so a fast loader can't build an unbounded prompt backlog
    # ahead of slower inference (backpressure), while keeping consumers fed.
    queue = asyncio.Queue(maxsize=args.concurrency * 4)
    groups = [(book, rows) for book, rows in df.groupby("book_id")]
    results = []

    async def produce():
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=args.loader_workers) as pool:
            futures = [
                loop.run_in_executor(pool, build_book_prompts, book, book_df)
                for book, book_df in groups
            ]
            for future in asyncio.as_completed(futures):
                try:
                    book_prompts = await future
                except Exception as error:  # noqa: BLE001 - skip a book we can't load
                    print(f"Failed to build prompts for a book: {error}")
                    continue
                for row_id, message in book_prompts:
                    if message is None:
                        print(f"Row ID: {row_id} - No passage found.")
                        continue
                    await queue.put((row_id, message))
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
        f"Building prompts ({args.loader_workers} workers) and inferring "
        f"(concurrency={args.concurrency}) in parallel..."
    )
    try:
        with (
            open(jsonl_path, "w") as jsonl_file,
            tqdm(desc="describing", total=len(df)) as progress,
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
    input_path = (
        DATA_DIR / "sample_for_analysis.parquet"
        if args.debug
        else DATA_DIR / "final_for_analysis.parquet"
    )

    df = pd.read_parquet(input_path).reset_index()

    output_dir = DATA_DIR / "analysis" / "llm_showtell"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "descriptions.jsonl"
    output_path = output_dir / "descriptions.parquet"

    results = asyncio.run(run(args, df, jsonl_path))

    pd.DataFrame(results).to_parquet(output_path, index=False)
    print(f"Wrote {len(results)} descriptions to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fashion Passage Loader")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use sample_for_analysis.parquet instead of final_for_analysis.parquet.",
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
        "--loader-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes that build prompts (tokenize books) in parallel.",
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
