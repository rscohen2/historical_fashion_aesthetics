import json
from textwrap import dedent

import llm
import pydantic


class WearingResponse(pydantic.BaseModel):
    reasoning: str
    entity: str
    fashion_item: str
    answer: bool


class CorefResponse(pydantic.BaseModel):
    reasoning: str
    coref: list[str]
    answer: bool


class Wearing:
    def __init__(self):
        pass

    def is_wearing(
        self,
        sentences: list[str],
        target_spans: tuple[list[int], list[int]],
        entity_spans: list[tuple[list[int], list[int]]],
    ) -> list[bool]:
        """
        Check if the entity spans are "wearing" the target spans in the given sentences.

        :param sentences: List of sentences to check.
        :param target_spans: Tuple of lists containing start and end indices for target spans.
        :param entity_spans: List of tuple of lists containing start and end indices for entity spans.
        :return: List of booleans indicating if each target span is wearing any entity span.
        """
        raise NotImplementedError


class WearingLLM(Wearing):
    def __init__(self, model_name: str = "gpt-4.1-nano"):
        super().__init__()
        self.model_name = model_name
        self.model = llm.get_model(model_name)

    def format_str(
        self,
        input_string,
        fashion_span: tuple[int, int],
        entity_spans: tuple[list[int], list[int]],
    ) -> str:
        """
        Format the input to wrap the fashion term and a given entity in tagged brackets.

        Example:
        Input: "Arthur was wearing a grey coat. It fit him well."
        Fashion span: (26, 30)
        Entity spans: ([0,, 39], [6, 42])
        Output: "[Arthur]_entity was wearing a [grey]_fashion coat. It fit [him]_entity well."

        Edge cases handled:
        - Empty input string
        - Invalid spans (out of bounds, start > end)
        - Empty spans (start == end)
        - Overlapping spans (processed in order of appearance)
        """
        if not input_string:
            return input_string

        # Validate spans
        if not (0 <= fashion_span[0] <= fashion_span[1] <= len(input_string)):
            raise ValueError(
                f"Invalid fashion span {fashion_span} for string of length {len(input_string)}"
            )

        # Validate entity spans
        for start, end in zip(entity_spans[0], entity_spans[1]):
            if not (0 <= start <= end <= len(input_string)):
                raise ValueError(
                    f"Invalid entity span ({start}, {end}) for string of length {len(input_string)}"
                )

        # Skip empty spans
        spans = []
        if fashion_span[0] < fashion_span[1]:
            spans.append((fashion_span[1], fashion_span[0], "fashion"))

        for start, end in zip(entity_spans[0], entity_spans[1]):
            if start < end:  # Skip empty spans
                spans.append((end, start, "entity"))

        if not spans:
            return input_string

        # Sort by end position in descending order
        spans.sort(reverse=True)

        # Check for overlapping spans
        for i in range(len(spans) - 1):
            curr_end, curr_start, _ = spans[i]
            next_end, next_start, _ = spans[i + 1]
            if curr_start <= next_end:
                # If spans overlap, keep the one that appears first in the text
                if curr_start <= next_start:
                    spans[i + 1] = (curr_start - 1, next_start, "entity")
                else:
                    spans[i] = (next_end - 1, curr_start, "entity")

        result = list(input_string)
        # Insert brackets working backwards
        for end, start, span_type in spans:
            result.insert(end, f"]_{span_type}")
            result.insert(start, "[")

        return "".join(result)

    def is_wearing_coref(
        self,
        datum: dict,
    ) -> list[bool]:
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

        Try to restrict your reasoning to the given sentence without relying on
        external knowledge, which may be ahistorical.

        Provide your output in the following JSON format:
        {{"coref": ["<coref1>", "<coref2>"], "reasoning": "...", "answer": true/false}}

        Input:
        {sentence}
        """).strip()

        formatted_str = self.format_datum(datum)
        print(formatted_str)
        response = self.model.prompt(
            coref_user_prompt.format(sentence=formatted_str),
            system=system_prompt,
            schema=CorefResponse,
        )
        result = json.loads(response.text())

        print(result)
        return result

    def is_wearing(
        self,
        sentences: list[str],
        target_spans: tuple[list[int], list[int]],
        entity_spans: list[tuple[list[int], list[int]]],
    ) -> list[bool]:
        system_prompt = dedent("""
        You are a literary scholar analyzing the use of clothing in literature.
        """).strip()

        user_prompt = dedent("""
        You are given a sentence. A fashion term is wrapped in a []_fashion tag. All
        mentions to a specific entity are wrapped in []_entity tags.

        Identify which entity is being specified. Then identify who is wearing
        or has possession of the fashion item, and evaluate whether this person
        is the specified entity. Be careful to disambiguate different
        characters, including the narrator.

        Provide your output in the following JSON format:
        {{"entity": "...", "fashion_item": "...", reasoning: "...", "answer": true/false}}

        Input:
        {sentence}

        """).strip()

        coref_user_prompt = dedent("""
        You are given a sentence. A fashion term is wrapped in a []_fashion tag. All
        mentions to specific entities are wrapped in []_<coref> tags.

        Identify which entity is wearing or has possession of the fashion item.
        Be careful to disambiguate different characters, including the narrator.
        Return the coref of the entity that is wearing or has possession of the fashion item.

        Provide your output in the following JSON format:
        {{"coref": "<coref>", "reasoning": "...", "answer": true/false}}

        Input:
        {sentence}
        """).strip()

        results = []
        for sentence, (target_start, target_end), entity_span in zip(
            sentences, zip(*target_spans), entity_spans
        ):
            formatted_str = self.format_str(
                sentence, (target_start, target_end), entity_span
            )
            print(formatted_str)
            response = self.model.prompt(
                user_prompt.format(sentence=formatted_str),
                system=system_prompt,
                schema=WearingResponse,
            )
            results.append(json.loads(response.text()))

        print(results)
        return results

    def format_datum(self, datum: dict) -> str:
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
        """
        text = datum["excerpt_text"]
        result = list(text)

        # Collect all spans to process
        spans = []

        # Add fashion span
        fashion_start = datum["fashion_start_idx"] - datum["excerpt_start"]
        fashion_end = datum["fashion_end_idx"] - datum["excerpt_start"]
        if 0 <= fashion_start <= fashion_end <= len(text):
            spans.append((fashion_end, fashion_start, "fashion"))

        # Add character spans
        for char in datum["characters"]:
            start = char["character_start_idx"] - datum["excerpt_start"]
            end = char["character_end_idx"] - datum["excerpt_start"]
            if 0 <= start <= end <= len(text):
                spans.append((end, start, char["coref"]))

        if not spans:
            return text

        # Sort by end position in descending order
        spans.sort(reverse=True)

        # Check for overlapping spans
        for i in range(len(spans) - 1):
            curr_end, curr_start, _ = spans[i]
            next_end, next_start, _ = spans[i + 1]
            if curr_start <= next_end:
                # If spans overlap, keep the one that appears first in the text
                if curr_start <= next_start:
                    spans[i + 1] = (curr_start - 1, next_start, spans[i + 1][2])
                else:
                    spans[i] = (next_end - 1, curr_start, spans[i][2])

        # Insert brackets working backwards
        for end, start, span_type in spans:
            result.insert(end, f"]_{span_type}")
            result.insert(start, "[")

        return "".join(result)
