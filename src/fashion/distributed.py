import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import inspect
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

import torch
from tqdm import tqdm


def add_distributed_args(parser: argparse.ArgumentParser):
    """
    --num_processes: Total number of processes for distributed processing
    --concurrent_processes: Number of processes to run concurrently
    """
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Total number of processes for distributed processing",
    )
    parser.add_argument(
        "--concurrent_processes",
        type=int,
        default=0,
        help="Number of processes to run concurrently",
    )


def run_distributed(
    process_fn: Callable,
    items: list[Any],
    script_path: str | None = None,
    pixi_env: str = "default",
    total_processes: int = 8,
    concurrent_processes: int = 0,
    extra_args: list[str] = [],
):
    if script_path is None:
        current_frame = inspect.currentframe()
        if current_frame is None or current_frame.f_back is None:
            raise ValueError(
                "No current frame found. Make sure to call this function from a script."
            )
        script_path = inspect.getfile(current_frame.f_back)

    total_processes = min(total_processes, len(items))
    if concurrent_processes <= 0:
        concurrent_processes = total_processes

    rank = int(os.environ.get("DISTRIBUTED_RANK", "0"))

    def execute_work(rank: int):
        """Execute the work for a given rank."""
        subset = items[rank::total_processes]
        output = process_fn(subset)
        # free up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        return output

    def spawn_process(rank: int):
        """Spawn a subprocess for the given rank."""
        proc = subprocess.Popen(
            [
                "pixi",
                "run",
                "--environment",
                pixi_env,
                "python",
                script_path,
                "--num_processes",
                str(total_processes),
                "--concurrent_processes",
                str(concurrent_processes),
            ]
            + extra_args,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(rank % torch.cuda.device_count()),
                "DISTRIBUTED_RANK": str(rank),
                "DISTRIBUTED_WORLD_SIZE": str(total_processes),
            },
        )
        proc.wait()
        return rank

    if rank == 0:
        # Run all processes including rank 0 in parallel
        with ThreadPoolExecutor(max_workers=concurrent_processes) as executor:
            # Only processes with rank > 0 execute the script
            # Rank 0 only serves to call and track the subprocesses
            # This ensures that processes get cleaned up completely when they
            # finish.
            all_futures = [
                executor.submit(spawn_process, i) for i in range(1, total_processes + 1)
            ]

            # Wait for all processes including rank 0
            with tqdm(
                total=total_processes,
                desc=f"{Path(script_path).name} ({pixi_env})",
            ) as pbar:
                for _ in as_completed(all_futures):
                    pbar.update(1)
    else:
        # Non-rank 0 processes just run their work
        execute_work(rank - 1)
