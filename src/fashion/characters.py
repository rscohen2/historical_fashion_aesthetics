import argparse
import os
from pathlib import Path
import subprocess

import torch
from booknlp.booknlp import BookNLP
from tqdm import tqdm


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


def process_texts(rank, num_processes):
    texts = sorted(list(Path("data/ChicagoCorpus/CLEAN_TEXTS").glob("*.txt")))

    for text_file in tqdm(texts[rank::num_processes]):
        print(f"Processing file: {text_file}")
        input_file = str(text_file)
        book_id = text_file.stem  # Use the filename without extension as book_id
        print(f"Using book_id: {book_id}")
        output_directory = f"data/booknlp/{book_id}/"
        print(f"Output directory: {output_directory}")
        if (Path(output_directory) / f"{book_id}.entities").exists():
            print(
                f"Output directory {output_directory} already exists. Skipping processing for {book_id}."
            )
            continue
        booknlp.process(input_file, output_directory, book_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process texts with BookNLP.")
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the process (for distributed processing)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Total number of processes for distributed processing",
    )
    args = parser.parse_args()

    rank = args.rank
    num_processes = args.num_processes
    # start processes with other ranks
    subprocesses = []

    if rank == 0:
        for i in range(1, num_processes):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i % torch.cuda.device_count())
            proc = subprocess.Popen(
                [
                    "python",
                    __file__,
                    "--rank",
                    str(i),
                    "--num_processes",
                    str(num_processes),
                ],
                env=env
            )
            subprocesses.append(proc)
    # process texts for the current rank
    process_texts(rank, num_processes)

    # wait for all subprocesses to finish
    if rank == 0:
        for proc in subprocesses:
            proc.wait()
        print("All subprocesses finished.")
    else:
        print(f"Process {rank} finished processing texts.")

