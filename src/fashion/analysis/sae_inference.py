from pathlib import Path

import torch as t
from dictionary_learning import AutoEncoder, utils

from fashion.analysis.sae import EMBEDDINGS_PATH, ActivationsLoader
from fashion.analysis.sae import OUTPUT_DIR as SAE_OUTPUT_DIR


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference using a trained Sparse Autoencoder (SAE)"
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default=str(EMBEDDINGS_PATH),
        help="Path to the activations file (.npy array or PyTorch tensor)",
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        default=SAE_OUTPUT_DIR / "trainer_0",
        help="Directory containing the trained SAE weights (e.g., 'trainer_0')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the inference on (e.g., 'cuda' or 'cpu')",
    )
    return parser.parse_args()


def main(args):
    ae, _ = utils.load_dictionary(str(args.weights_dir), device=args.device)

    activations = ActivationsLoader(
        filepath=args.activations_path,
        batch_size=8192,
        shuffle=False,
        loop=False,
        drop_last=False,
        device=args.device,
    )

    all_values, all_indices = [], []
    for batch in activations:
        with t.no_grad():
            _, values, indices, _ = ae.encode(batch, return_topk=True)
            all_values.append(values.cpu())
            all_indices.append(indices.cpu())
    all_values = t.cat(all_values, dim=0)
    all_indices = t.cat(all_indices, dim=0)

    # Save the codes to a file
    codes_path = args.weights_dir / "codes.pt"
    t.save({"values": all_values, "indices": all_indices}, codes_path)


if __name__ == "__main__":
    main(parse_args())
