import argparse

import pandas as pd
from tqdm import tqdm

from fashion.utils import CHICAGO_PATH, DATA_DIR


def extract_paragraphs(text):
    """
    Extract paragraphs and their start and end indices from the given text.
    A paragraph is defined as a sequence of characters ending with two newline characters.
    """
    paragraphs = []
    start_idx = 0
    for i, char in enumerate(text):
        if char == "\n" and (i + 1 < len(text) and text[i + 1] == "\n"):
            paragraphs.append((text[start_idx:i], start_idx, i, len(paragraphs)))
            start_idx = i + 2  # Move past the two newlines
    if start_idx < len(text):
        paragraphs.append((text[start_idx:], start_idx, len(text), len(paragraphs)))

    paragraphs = [
        (para, start, end, para_id)
        for para, start, end, para_id in paragraphs
        if para.strip()  # Exclude empty paragraphs
    ]

    return pd.DataFrame(
        paragraphs, columns=["paragraph", "start_idx", "end_idx", "paragraph_id"]
    )


def main(input_directory, output_file, max_files=None, progress_bar=True):
    """
    Main function to read input files, extract paragraphs, and save the results.
    """
    dfs = []
    input_files = sorted(input_directory.glob("*.txt"))
    if max_files is not None:
        input_files = input_files[:max_files]
    if not input_files:
        print("No input files found in the specified directory.")
        return
    for file in tqdm(input_files, desc="Processing files", disable=not progress_bar):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            df = extract_paragraphs(text)
            df["filename"] = file.name  # Add filename to the DataFrame
            dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv(output_file, index=False)
    print(f"Extracted paragraphs saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract paragraphs from text files in a directory."
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        help="Path to the input directory containing text files.",
        default=CHICAGO_PATH / "CLEAN_TEXTS",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the extracted paragraphs as a CSV file.",
        default=DATA_DIR / "extracted_paragraphs.csv",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: None, process all).",
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="Show a progress bar while processing files.",
        default=True,
    )
    args = parser.parse_args()

    main(**vars(args))
