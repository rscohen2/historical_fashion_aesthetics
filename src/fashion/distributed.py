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


def run_single(
    process_fn: Callable,
    items: list[Any],
    script_path: str | None = None,
    pixi_env: str = "default",
    extra_args: list[str] = [],
):
    if script_path is None:
        current_frame = inspect.currentframe()
        if current_frame is None or current_frame.f_back is None:
            raise ValueError(
                "No current frame found. Make sure to call this function from a script."
            )
        script_path = inspect.getfile(current_frame.f_back)

    def spawn_process():
        proc = subprocess.Popen(
            [
                "pixi",
                "run",
                "--environment",
                pixi_env,
                "python",
                script_path,
            ]
            + extra_args,
            env={
                **os.environ,
                "DISTRIBUTED_RANK": str(1),
                "DISTRIBUTED_WORLD_SIZE": str(1),
            },
        )
        proc.wait()
        return proc

    rank = int(os.environ.get("DISTRIBUTED_RANK", "0"))
    if rank == 0:
        spawn_process()
    else:
        process_fn(items)


def run_distributed(
    process_fn: Callable,
    items: list[Any],
    script_path: str | None = None,
    pixi_env: str = "default",
    total_processes: int = 8,
    concurrent_processes: int = 0,
    extra_args: list[str] = [],
    restrict_cuda: bool = True,
    max_retries: int = 3,
):
    # if restrict_cuda is True, then we will restrict the CUDA_VISIBLE_DEVICES
    # environment variable to one GPU per process.
    # you may want to set this to False if you are running a script that handles
    # multiple GPUs already.
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
        cuda_visible_devices = (
            str(rank % torch.cuda.device_count()) if restrict_cuda else ""
        )
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
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "DISTRIBUTED_RANK": str(rank),
                "DISTRIBUTED_WORLD_SIZE": str(total_processes),
            },
        )
        returncode = proc.wait()
        return rank, returncode

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
            rank_retries = {i: 0 for i in range(1, total_processes + 1)}

            # Wait for all processes including rank 0
            with tqdm(
                total=total_processes,
                desc=f"{Path(script_path).name} ({pixi_env})",
            ) as pbar:
                # this while loop is necessary because we might add new futures
                # to the list as we go.
                while all_futures:
                    done_futures = []
                    for future in as_completed(all_futures):
                        # if the script did not exit cleanly, re-run it with the same rank
                        rank, return_code = future.result()
                        print(f"Rank {rank} returned {return_code}")
                        if return_code != 0 and rank_retries[rank] < max_retries:
                            # re-run the script with the same rank
                            all_futures.append(executor.submit(spawn_process, rank))
                            rank_retries[rank] += 1
                            print(
                                f"Rank {rank} failed, retrying {rank_retries[rank]} / {max_retries}"
                            )
                        else:
                            pbar.update(1)
                            done_futures.append(future)
                    # remove the futures that successfully completed
                    for future in done_futures:
                        all_futures.remove(future)
    else:
        # Non-rank 0 processes just run their work
        execute_work(rank - 1)
