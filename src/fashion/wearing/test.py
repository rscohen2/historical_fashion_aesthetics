import json

from fashion.utils import DATA_DIR
from .llm import WearingLLM


litbank_data = [
    json.loads(line)
    for line in open(DATA_DIR / "fashion_character_cooc" / "litbank" / "cooc.0.ndjson")
][:10]


if __name__ == "__main__":
    # print(litbank_data)
    wearing = WearingLLM()
    for datum in litbank_data:
        wearing.is_wearing_coref(datum)
    # wearing.is_wearing(
    #     [data["excerpt_text"]] * len(set(c["coref"] for c in data["characters"])),
    #     (
    #         [data["fashion_start_idx"] - data["excerpt_start"]]
    #         * len(set(c["coref"] for c in data["characters"])),
    #         [data["fashion_end_idx"] - data["excerpt_start"]]
    #         * len(set(c["coref"] for c in data["characters"])),
    #     ),
    #     [
    #         tuple(
    #             zip(
    #                 *(
    #                     (
    #                         c["character_start_idx"] - data["excerpt_start"],
    #                         c["character_end_idx"] - data["excerpt_start"],
    #                     )
    #                     for c in data["characters"]
    #                     if c["coref"] == coref
    #                 )
    #             )
    #         )
    #         for coref in set(c["coref"] for c in data["characters"])
    #     ],
    # )
