"""
Train huggingface deberta classifier on fashion dataset.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import (
    DebertaV2TokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from fashion.paths import DATA_DIR


class SpanContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = nn.Dropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states, span_mask: torch.Tensor):
        # We pool the model by averaging the hidden states of the span.
        # The span_mask is a binary mask indicating the positions of the span.
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # span_mask shape: (batch_size, seq_length)

        context_token = (hidden_states * span_mask.unsqueeze(-1)).mean(dim=1)
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class DebertaV2ForSpanClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = SpanContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.deberta.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        span_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer, span_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits,
                            0,
                            label_index.expand(label_index.size(0), logits.size(1)),
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            labeled_logits.view(-1, self.num_labels).float(),
                            labels.view(-1),
                        )
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,  # type: ignore
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DataCollator:
    """
    Data collator that handles batching with label IDs
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        padded_seqs = {
            k: pad_sequence(  # noqa: F821
                [feature[k] for feature in features],
                batch_first=True,
            )
            for k in features[0].keys()
            if k not in ["label", "label_ids"]
        }
        batch = {
            **padded_seqs,
            "labels": torch.tensor(
                [feature["label"] for feature in features], dtype=torch.long
            ),
        }

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return dict(batch)


@dataclass
class DataPreparer:
    """
    Prepares the data for training and evaluation.
    """

    max_length = 512  # Maximum length for the input sequences
    tokenizer: DebertaV2TokenizerFast

    def prepare_data(self, examples):
        if any(
            [
                examples["start_idx"][i] >= self.max_length
                or examples["end_idx"][i] >= self.max_length
                for i in range(len(examples["start_idx"]))
            ]
        ):
            raise ValueError(
                "Start or end indices exceed the maximum length of the model."
            )

        tokens = self.tokenizer(
            examples["sentence"],
            truncation=True,
            return_offsets_mapping=True,
            max_length=self.max_length,
        )

        span_masks = [
            self.get_span_mask(
                torch.tensor(tokens["offset_mapping"][i]),  # type: ignore
                [start],
                [end],
            )
            for i, (start, end) in enumerate(
                zip(examples["start_idx"], examples["end_idx"])
            )
        ]

        labels = examples["label"]

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens["token_type_ids"],
            "span_mask": span_masks,
            "label": labels,
        }

    def get_span_mask(self, offset_mapping, starts, stops):
        # Find the token indices corresponding to the start and stop offsets
        offset_mapping[-1, :] = (
            offset_mapping[-2, :] + 1
        )  # Fix the last token's offset mapping
        token_inds = torch.searchsorted(
            offset_mapping.T, torch.tensor([stops, starts])
        ).T.squeeze()

        mask = torch.zeros(offset_mapping.size(0), dtype=torch.bool)
        # Set the mask to True for the tokens that cover the span
        mask[token_inds[1] : token_inds[0]] = True
        assert mask.sum() > 0, "Span mask is empty, check the offsets and spans."
        return mask


def compute_metrics(eval_predictions):
    labels = eval_predictions.label_ids
    preds = eval_predictions.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    # Load the dataset
    data_path = DATA_DIR / "20250420-1837-annotations.ndjson"
    print(data_path)
    df = pd.read_json(data_path, lines=True)

    # create cleaned df with columns: sentence, start_idx, end_idx, label (yesno)
    df = pd.DataFrame(
        [
            {
                "sentence": datum["sentence"],
                "start_idx": datum["start_idx"],
                "end_idx": datum["end_idx"],
                "label": annotation["yesno"],
            }
            for datum, annotation in zip(df["datum"], df["annotation"])
        ]
    )

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Load the tokenizer and model
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")  # noqa: F821

    data_preparer = DataPreparer(tokenizer)
    train_dataset = train_dataset.map(
        data_preparer.prepare_data,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        data_preparer.prepare_data,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Set format for PyTorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label", "span_mask"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label", "span_mask"]
    )

    model = DebertaV2ForSpanClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=2
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(DATA_DIR / "deberta-fashion-span"),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=str(DATA_DIR / "logs"),
        logging_steps=10,
        save_strategy="epoch",
        # save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
