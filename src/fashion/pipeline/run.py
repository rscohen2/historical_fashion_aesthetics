"""
Wrapper to run the pipeline.
"""

import argparse

# from fashion.distributed import add_distributed_args
from fashion.paths import DATA_DIR
from fashion.pipeline.steps.extract_char_fashion_mentions import (
    main as extract_char_fashion_mentions,
)
from fashion.pipeline.steps.extract_entity_mentions import (
    main as extract_entity_mentions,
)
from fashion.pipeline.steps.extract_fashion_mentions import (
    main as extract_fashion_mentions,
)
from fashion.pipeline.steps.filter_fashion_mentions import (
    main as filter_fashion_mentions,
)
from fashion.pipeline.steps.run_booknlp import run_in_env as run_booknlp
from fashion.pipeline.steps.run_wearing_model import main as run_wearing_model
from fashion.pipeline.steps.extract_adjectives import main as extract_adjectives
from fashion.sources import Source, add_source_argument


def main(source: Source):
    output_dir = DATA_DIR / "pipeline" / source.name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # # 1. Extract fashion mentions
    # print("Extracting fashion mentions...")
    # extract_fashion_mentions(
    #     source,
    #     max_files=None,
    #     output_filename=output_dir / "fashion_mentions.csv",
    # )

    # # 2. Filter fashion mentions
    # print("Filtering fashion mentions...")
    # filter_fashion_mentions(
    #     input_file=output_dir / "fashion_mentions.csv",
    #     output_file=output_dir / "fashion_mentions" / "filtered.csv",
    #     max_rows=None,
    # )

    # # 3. Run BookNLP on the original texts
    # print("Running BookNLP...")
    # run_booknlp(
    #     source,
    #     num_processes=1,
    #     concurrent_processes=1,
    #     output_dir=DATA_DIR / "booknlp",
    # )

    # # 4. Identify the entities present in passages with fashion mentions
    # print("Identifying entities present in passages with fashion mentions...")
    # extract_char_fashion_mentions(
    #     source=source,
    #     fashion_mention_file=output_dir / "fashion_mentions" / "filtered.csv",
    #     booknlp_dir=DATA_DIR / "booknlp",
    #     output_dir=output_dir / "character_fashion_cooc",
    #     num_processes=1,
    #     concurrent_processes=1,
    # )

    # # 5. Apply a deberta model to identify which entity in a passage is in
    # # possession of the mentioned article of clothing
    # print("Identifying wearing entities...")
    # run_wearing_model(
    #     fashion_cooc_dir=output_dir / "character_fashion_cooc",
    #     output_dir=output_dir / "wearing",
    #     num_processes=1,
    #     concurrent_processes=1,
    # )

    # 6. Extract sentences which mention entities that have been linked to clothing
    # print(
    #     "Extracting sentences which mention entities that have been linked to clothing..."
    # )
    # extract_entity_mentions(
    #     source=source,
    #     wearing_dir=output_dir / "wearing",
    #     booknlp_dir=DATA_DIR / "booknlp",
    #     output_dir=output_dir / "entity_mentions",
    #     num_processes=1,
    #     concurrent_processes=1,
    # )

    # 7. Use dependency parsing to find adjectives which describe a particular noun
    print("Extracting adjectives which describe fashion mentions...")
    extract_adjectives(
        noun_mention_dir=output_dir / "fashion_mentions",
        output_dir=output_dir / "adjectives/fashion/",
        num_processes=4,
        concurrent_processes=1,
        do_coref=True,
    )

    print("Extracting adjectives which describe entity mentions...")
    extract_adjectives(
        noun_mention_dir=output_dir / "entity_mentions",
        output_dir=output_dir / "adjectives/entities/",
        num_processes=16,
        concurrent_processes=1,
        do_coref=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline.")
    add_source_argument(parser)
    # add_distributed_args(parser)
    args = parser.parse_args()
    main(**vars(args))
