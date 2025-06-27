import argparse
import os
from pathlib import Path
import subprocess

import torch
from tqdm import tqdm

from fashion.paths import DATA_DIR
from fashion.sources import Source, add_source_argument
from fashion.distributed import add_distributed_args, run_distributed


def remove_position_ids_and_save(model_file, device, save_path):
    state_dict = torch.load(model_file, map_location=device)

    if "bert.embeddings.position_ids" in state_dict:
        print(f'Removing "position_ids" from the state dictionary of {model_file}')
        del state_dict["bert.embeddings.position_ids"]

    torch.save(state_dict, save_path)
    print(f"Modified state dict saved to {save_path}")


def process_model_files(model_params, device):
    updated_params = {}
    for key, path in model_params.items():
        if isinstance(path, str) and os.path.isfile(path) and path.endswith(".model"):
            save_path = path.replace(".model", "_modified.model")
            remove_position_ids_and_save(path, device, save_path)
            updated_params[key] = save_path
        else:
            updated_params[key] = path
    return updated_params


def process_texts(book_ids, data_source: Source, output_dir: Path):
    try:
        from booknlp.booknlp import BookNLP  # type: ignore
    except ImportError:
        raise ImportError(
            "BookNLP is not installed. Please install it with `pip install booknlp`."
        )

    user_dir = str(Path.home())
    model_params = {
        "pipeline": "entity,quote,coref",
        "model": "custom",
        "entity_model_path": f"{user_dir}/booknlp_models/entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model",
        "coref_model_path": f"{user_dir}/booknlp_models/coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model",
        "quote_attribution_model_path": f"{user_dir}/booknlp_models/speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model",
        "bert_model_path": f"{user_dir}/.cache/huggingface/hub/",
    }

    model_params = process_model_files(model_params, device=torch.device("cpu"))
    booknlp = BookNLP("en", model_params)
    for book_id in tqdm(book_ids):
        try:
            print(f"Processing file: {book_id}")
            input_file = data_source.load_text(book_id).ref
            print(f"Using book_id: {book_id}")
            book_dir = output_dir / book_id
            print(f"Output directory: {book_dir}")
            if (Path(book_dir) / f"{book_id}.entities").exists():
                print(
                    f"Output directory {book_dir} already exists. Skipping processing for {book_id}."
                )
                continue
            booknlp.process(input_file, book_dir, book_id)
        except Exception as e:
            print(f"Error processing file {book_id}: {e}")
            continue


def run_in_env(
    source: Source,
    num_processes: int,
    concurrent_processes: int,
    output_dir: Path,
):
    # kind of a hack to run inside the booknlp environment
    subprocess.run(
        [
            "pixi",
            "run",
            "-e",
            "booknlp",
            "python",
            __file__,
            "--source",
            str(source.name.lower()),
            "--num_processes",
            str(num_processes),
            "--concurrent_processes",
            str(concurrent_processes),
            "--output_dir",
            str(output_dir),
        ]
    )


def main(
    source: Source,
    num_processes: int,
    concurrent_processes: int,
    output_dir: Path,
):
    def process(subset: list[str]):
        process_texts(subset, source, output_dir)

    run_distributed(
        process,
        sorted(list(source.iter_book_ids())),
        script_path=__file__,
        pixi_env="booknlp",
        total_processes=num_processes,
        concurrent_processes=concurrent_processes,
        extra_args=[
            "--source",
            str(source.name.lower()),
            "--output_dir",
            str(output_dir),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process texts with BookNLP.")
    add_source_argument(parser)
    add_distributed_args(parser)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "booknlp",
        help="Directory to save the output",
    )
    args = parser.parse_args()

    main(**vars(args))
