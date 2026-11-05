from pathlib import Path

import numpy as np
import torch as t
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE

from fashion.paths import DATA_DIR

EXPERIMENT = "sae"
OUTPUT_DIR = DATA_DIR / "analysis" / EXPERIMENT
EMBEDDINGS_PATH = DATA_DIR / "analysis" / "embeddings" / "sentence_embeddings.npy"


class ActivationsLoader:
    """Endless iterator over shuffled, full-size batches of activations.

    Reshuffles at the start of every pass and drops the short final batch;
    trainSAE stops once it reaches its step count.
    """

    def __init__(
        self,
        filepath,
        batch_size: int = 8192,
        device: str = "cuda",
        seed: int = 42,
        shuffle: bool = True,
        loop: bool = True,
        drop_last: bool = True,  # whether to drop the last batch if it's smaller than batch_size
    ):
        self.filepath = Path(filepath)
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        if self.filepath.suffix == ".npy":
            self.data = t.from_numpy(np.load(self.filepath)).to(self.device)
        else:
            self.data = t.load(self.filepath, map_location=self.device)
        if len(self.data) < self.batch_size:
            raise ValueError(
                f"Only {len(self.data):,} activations, fewer than one batch "
                f"({self.batch_size:,})"
            )
        self.activations_shape = self.data.shape[1]
        self.shuffle = shuffle
        self.loop = loop
        self.drop_last = drop_last

    def __iter__(self):
        gen = t.Generator(device=self.data.device).manual_seed(self.seed)
        while True:
            if self.shuffle:
                perm = t.randperm(
                    len(self.data), device=self.data.device, generator=gen
                )
            else:
                perm = t.arange(len(self.data), device=self.data.device)
            for i in range(0, len(perm) - self.batch_size + 1, self.batch_size):
                yield self.data[perm[i : i + self.batch_size]].float()
            if not self.drop_last and len(perm) % self.batch_size != 0:
                yield self.data[perm[len(perm) - len(perm) % self.batch_size :]].float()
            if not self.loop:
                break


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder (SAE)")
    parser.add_argument(
        "--activations_path",
        type=str,
        default=str(EMBEDDINGS_PATH),
        help="Path to the activations file (.npy array or PyTorch tensor)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save the trained SAE model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size for loading activations",
    )
    parser.add_argument(
        "--num_codes",
        type=int,
        default=2048,
        help="Number of codes in the SAE",
    )
    parser.add_argument(
        "--k", type=int, default=32, help="Number of active codes per activation"
    )
    parser.add_argument(
        "--steps", type=int, default=10_000, help="Number of training steps"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1_000, help="Learning rate warmup steps"
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=5,
        help="Save this many evenly spaced checkpoints (including the final model)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for SAE init and data shuffling"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    return parser.parse_args()


def main(args):
    activations_loader = ActivationsLoader(
        filepath=args.activations_path,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    steps_per_pass = len(activations_loader.data) // args.batch_size
    print(
        f"{len(activations_loader.data):,} activations, {steps_per_pass:,} steps per "
        f"pass, {args.steps / steps_per_pass:.1f} passes"
    )

    # the final model is always saved as ae.pt, so only intermediate steps here
    interval = args.steps // args.num_checkpoints if args.num_checkpoints else 0
    save_steps = list(range(interval, args.steps, interval)) if interval else None

    trainer_configs = [
        {
            "trainer": TopKTrainer,
            "dict_class": AutoEncoderTopK,
            "lr": args.lr,
            "steps": args.steps,
            "auxk_alpha": 1 / 32,
            "warmup_steps": args.warmup_steps,
            "decay_start": None,
            "threshold_beta": 0.999,
            "threshold_start_step": args.warmup_steps,
            "seed": args.seed,
            "activation_dim": activations_loader.activations_shape,
            "dict_size": args.num_codes,
            "k": args.k,
            "device": args.device,
            "layer": -1,
            "lm_name": "qwen-encoder-0.6b",
            "wandb_name": f"topk-k{args.k}-w{args.num_codes}",
        }
    ]
    trainSAE(
        data=activations_loader,
        trainer_configs=trainer_configs,
        steps=args.steps,
        use_wandb=True,
        wandb_project="fashion-sae",
        save_steps=save_steps,
        save_dir=args.output_dir,
        log_steps=10,
        device=args.device,
        verbose=True,
        normalize_activations=True,
    )


if __name__ == "__main__":
    main(parse_args())
