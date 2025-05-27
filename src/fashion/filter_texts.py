"""
Use deberta fashion model to filter fashion-related texts.
"""

import argparse

import pandas as pd
from datasets import Dataset
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from scipy.special import softmax

from fashion.train_classifier import (
    DataCollator,
    DataPreparer,
    DebertaV2ForSpanClassification,
    DebertaV2TokenizerFast,
)
from fashion.utils import DATA_DIR


def filter_fashion_texts(df) -> pd.DataFrame:
    df = df[
        df.start_idx.lt(512) & df.end_idx.lt(512)
    ]  # Filter out texts with spans longer than 512 tokens
    df.loc[:, "label"] = 0  # Initialize label column with 0 (non-fashion)

    # Load the tokenizer and model
    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        "microsoft/deberta-v3-base", use_fast=True
    )
    model = DebertaV2ForSpanClassification.from_pretrained(
        DATA_DIR / "deberta-fashion-span" / "checkpoint-150"
    )

    # Prepare the data
    data_preparer = DataPreparer(tokenizer)
    eval_dataset = Dataset.from_pandas(df)
    eval_dataset = eval_dataset.map(
        data_preparer.prepare_data,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label", "span_mask"]
    )

    # Perform inference
    test_args = TrainingArguments(
        output_dir=str(DATA_DIR / "fashion-filter"),
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=64,
        dataloader_drop_last=False,
        dataloader_num_workers=8,
    )

    # init trainer
    trainer = Trainer(model=model, args=test_args, data_collator=DataCollator())
    results = trainer.predict(eval_dataset)
    labels = results.predictions.argmax(axis=1)
    confidence_scores = softmax(results.predictions, axis=1).max(axis=1)
    df["confidence"] = confidence_scores
    df = df[labels == 1]  # Filter only fashion-related texts

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter fashion-related texts using DeBERTa model."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input CSV file containing texts.",
        default="fashion_results.csv",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the filtered fashion-related texts.",
        default=DATA_DIR / "filtered_fashion_texts.csv",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Maximum number of rows to process."
    )
    args = parser.parse_args()

    # Load the input CSV file
    df = pd.read_csv(args.input_file)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    df_filtered = filter_fashion_texts(df)

    # Save the filtered results to the output CSV file
    df_filtered.to_csv(args.output_file, index=False)
    print(
        f"{df_filtered.shape[0] / df.shape[0]:.2%} of texts ({df_filtered.shape[0]} / {df.shape[0]}) are fashion-related."
    )
    print(f"Filtered results saved to {args.output_file}")
