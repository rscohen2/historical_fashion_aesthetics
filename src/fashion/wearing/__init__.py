from textwrap import dedent
from typing import Iterator
from datasets import Dataset

import llm
import numpy as np
import pydantic


class WearingResponse(pydantic.BaseModel):
    reasoning: str
    entity: str
    fashion_item: str
    answer: bool


class CorefResponse(pydantic.BaseModel):
    reasoning: str
    coref: list[str]


class Wearing:
    def __init__(self):
        pass

    def is_wearing(
        self,
        sentences: list[str],
        fashion_spans: list[tuple[int, int]],
        entity_spans: list[list[tuple[int, int]]],
        coref_labels: list[list[str]],
    ) -> list[list[bool]]:
        """
        Check if the entity spans are "wearing" the target spans in the given sentences.

        :param sentences: List of sentences to check.
        :param target_spans: Tuple of lists containing start and end indices for target spans.
        :param entity_spans: List of tuples containing start and end indices for entity spans.
        :return: List of booleans indicating if each target span is wearing any entity span.
        """
        raise NotImplementedError


class WearingLLM(Wearing):
    def __init__(self, model_name: str = "gpt-4.1-nano"):
        super().__init__()
        self.model_name = model_name
        self.model = llm.get_model(model_name)

    def is_wearing(
        self,
        sentences: list[str],
        fashion_spans: list[tuple[int, int]],
        entity_spans: list[list[tuple[int, int]]],
        coref_labels: list[list[str]],
    ) -> list[list[bool]]:
        system_prompt = dedent("""
        You are a literary scholar analyzing the use of clothing in literature.
        """).strip()

        coref_user_prompt = dedent("""
        You are given a sentence. A fashion term is wrapped in a []_fashion tag. All
        mentions to specific entities are wrapped in []_<coref> tags.

        Identify which entity is wearing or has possession of the fashion item.
        Be careful to disambiguate different characters, including the narrator.
        Return the coref IDs of the entity that is wearing or has possession of
        the fashion item. If there are multiple entities, return all of them.

        If the fashion label is incorrectly applied, return an empty list.

        Try to restrict your reasoning to the given sentence without relying on
        external knowledge, which may be ahistorical.

        Provide your output in the following JSON format:
        {{"reasoning": "...", "coref": ["<coref1>", "<coref2>"]}}

        Input:
        {sentence}
        """).strip()

        results = []
        for sentence, fashion_span, datum_entity_spans, datum_coref_labels in zip(
            sentences, fashion_spans, entity_spans, coref_labels
        ):
            formatted_str = self.format_coref(
                sentence, fashion_span, datum_entity_spans, datum_coref_labels
            )
            response = self.model.prompt(
                coref_user_prompt.format(sentence=formatted_str),
                system=system_prompt,
                schema=CorefResponse,
            )

            corefs = CorefResponse.model_validate_json(response.text())
            results.append([i in corefs.coref for i in datum_coref_labels])

        return results

    # def is_wearing(
    #     self,
    #     sentences: list[str],
    #     target_spans: tuple[list[int], list[int]],
    #     entity_spans: list[tuple[list[int], list[int]]],
    # ) -> list[bool]:
    #     system_prompt = dedent("""
    #     You are a literary scholar analyzing the use of clothing in literature.
    #     """).strip()

    #     user_prompt = dedent("""
    #     You are given a sentence. A fashion term is wrapped in a []_fashion tag. All
    #     mentions to a specific entity are wrapped in []_entity tags.

    #     Identify which entity is being specified. Then identify who is wearing
    #     or has possession of the fashion item, and evaluate whether this person
    #     is the specified entity. Be careful to disambiguate different
    #     characters, including the narrator.

    #     Provide your output in the following JSON format:
    #     {{"entity": "...", "fashion_item": "...", reasoning: "...", "answer": true/false}}

    #     Input:
    #     {sentence}

    #     """).strip()

    #     results = []
    #     for sentence, (target_start, target_end), entity_span in zip(
    #         sentences, zip(*target_spans), entity_spans
    #     ):
    #         formatted_str = self.format_str(
    #             sentence, (target_start, target_end), entity_span
    #         )
    #         print(formatted_str)
    #         response = self.model.prompt(
    #             user_prompt.format(sentence=formatted_str),
    #             system=system_prompt,
    #             schema=WearingResponse,
    #         )
    #         results.append(json.loads(response.text()))

    #     print(results)
    #     return results

    def format_coref(
        self,
        text: str,
        fashion_span: tuple[int, int],
        entity_spans: list[tuple[int, int]],
        coref_labels: list[str],
    ) -> str:
        """
        Format a datum by wrapping fashion terms and entities in tagged brackets.
        Fashion terms are wrapped in []_fashion, entities in []_<coref>.

        Example input:
        {
            "book_id": "1260_jane_eyre_an_autobiography",
            "fashion_term": "bonnet",
            "fashion_start_idx": 51401,
            "fashion_end_idx": 51407,
            "excerpt_start": 51289,
            "excerpt_end": 51593,
            "excerpt_text": "Bessie was gone down to breakfast; my cousins had not yet been summoned...",
            "characters": [
                {
                    "character_start_idx": 51397,
                    "character_end_idx": 51400,
                    "coref": "146",
                    "text": "her",
                    "distance": 4,
                },
                ...
            ],
        }

        Example output:
        "Bessie was gone down to breakfast; [my cousins]_705 had not yet been summoned
        to [their mama]_474; [Eliza]_146 was putting on [her]_146 [bonnet]_fashion and warm garden-coat..."

        This should also handle nested entities. For example:
        "The [young man]_123 was wearing a [red]_fashion hat. [He]_123 was [[the man]_456's son]_123."
        or
        "[She]_3 was [the mother of the [[hairdresser]_1's wife]_2]_3."

        """
        # text = datum["excerpt_text"]
        fashion_start, fashion_end = fashion_span
        assert len(coref_labels) == len(entity_spans), (
            f"Expected {len(entity_spans)} coref labels, got {len(coref_labels)}"
        )

        # Create a list of spans to insert, including fashion term and all character mentions
        spans = []

        # Add fashion term span
        spans.append((fashion_end, fashion_start, "fashion"))

        # Add character spans
        for coref, (start, end) in zip(coref_labels, entity_spans):
            if start < end:  # Skip empty spans
                spans.append((end, start, coref))

        # Sort spans by end position in descending but start position in
        # ascending order to ensure proper nesting (where the outer span should
        # be inserted first.)
        spans.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        # Insert tags working backwards through the text
        result = list(text)
        to_apply = []
        for end, start, tag in spans:
            to_apply.append((end, f"]_{tag}"))
            to_apply.append((start, "["))
        for end, tag in sorted(to_apply, key=lambda x: x[0], reverse=True):
            result.insert(end, tag)

        return "".join(result)


class WearingBert(Wearing):
    def __init__(self, model_path: str = "data/deberta-wearing/checkpoint-900"):
        super().__init__()
        from .bert import DebertaV2ForSpanClassification
        from .bert import DebertaV2TokenizerFast
        from .bert import DataCollator
        from .bert import DataPreparer
        from transformers import TrainingArguments
        from transformers import Trainer

        self.model = DebertaV2ForSpanClassification.from_pretrained(model_path)
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
            "microsoft/deberta-v3-base"
        )
        self.data_collator = DataCollator()
        self.data_preparer = DataPreparer(self.tokenizer)

        test_args = TrainingArguments(
            # output_dir=str(DATA_DIR / "fashion-filter"),
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=64,
            dataloader_drop_last=False,
            dataloader_num_workers=8,
        )

        # init trainer
        self.trainer = Trainer(
            model=self.model, args=test_args, data_collator=DataCollator()
        )

    def format_data(
        self,
        sentences: list[str],
        fashion_spans: list[tuple[int, int]],
        entity_spans: list[list[tuple[int, int]]],
        coref_labels: list[list[str]],
    ):
        def inner():
            for sentence, fashion_span, datum_entity_spans, datum_coref_labels in zip(
                sentences, fashion_spans, entity_spans, coref_labels
            ):
                for entity_span, coref_label in zip(
                    datum_entity_spans, datum_coref_labels
                ):
                    yield {
                        "sentence": sentence,
                        "target_start_idx": fashion_span[0],
                        "target_end_idx": fashion_span[1],
                        "query_start_idx": entity_span[0],
                        "query_end_idx": entity_span[1],
                        "label": 0,  # initialize to 0 for non-wearing
                    }

        return inner

    def is_wearing(
        self,
        sentences: list[str],
        fashion_spans: list[tuple[int, int]],
        entity_spans: list[list[tuple[int, int]]],
        coref_labels: list[list[str]],
    ) -> list[list[bool]]:
        eval_dataset: Dataset = Dataset.from_generator(
            self.format_data(sentences, fashion_spans, entity_spans, coref_labels)
        )
        eval_dataset = eval_dataset.map(
            self.data_preparer.prepare_data,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
        eval_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label",
                "query_span_mask",
                "target_span_mask",
            ],
        )
        results = self.trainer.predict(eval_dataset)
        labels = results.predictions.argmax(axis=1).astype(bool).tolist()
        # renest to match the input
        entity_counts = np.cumsum(
            [0] + [len(datum_coref_labels) for datum_coref_labels in coref_labels]
        )
        nested_labels = [
            labels[entity_counts[i] : entity_counts[i + 1]]
            for i in range(len(entity_counts) - 1)
        ]

        return nested_labels
