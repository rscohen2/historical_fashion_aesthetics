"""
Wrapper to run the pipeline.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

# from fashion.distributed import add_distributed_args
from fashion.paths import DATA_DIR
from fashion.pipeline.steps.extract_adjectives import main as extract_adjectives
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
from fashion.pipeline.steps.finalize_outputs import finalize_outputs
from fashion.pipeline.steps.run_booknlp import run_in_env as run_booknlp
from fashion.pipeline.steps.run_wearing_model import main as run_wearing_model
from fashion.sources import Source, add_source_argument


@dataclass
class StepConfig:
    """Configuration for a pipeline step."""

    enabled: bool = True
    num_processes: int = 1
    concurrent_processes: int = 1


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""

    steps: dict[str, StepConfig]

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default configuration with all steps enabled."""
        return cls(
            {
                "extract_fashion_mentions": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
                "filter_fashion_mentions": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
                "run_booknlp": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
                "extract_char_fashion_mentions": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
                "run_wearing_model": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
                "extract_entity_mentions": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
                "extract_fashion_adjectives": StepConfig(
                    enabled=True, num_processes=4, concurrent_processes=1
                ),
                "extract_entity_adjectives": StepConfig(
                    enabled=True, num_processes=16, concurrent_processes=1
                ),
                "finalize_outputs": StepConfig(
                    enabled=True, num_processes=1, concurrent_processes=1
                ),
            }
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        steps = {}
        for step_name, step_data in data.get("steps", {}).items():
            steps[step_name] = StepConfig(
                enabled=step_data.get("enabled", True),
                num_processes=step_data.get("num_processes", 1),
                concurrent_processes=step_data.get("concurrent_processes", 1),
            )

        return cls(steps=steps)

    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        data = {
            "steps": {
                step_name: {
                    "enabled": step_config.enabled,
                    "num_processes": step_config.num_processes,
                    "concurrent_processes": step_config.concurrent_processes,
                }
                for step_name, step_config in self.steps.items()
            }
        }

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=False)


def create_default_yaml_config(output_path: Path):
    """Create a default YAML configuration file."""
    config = PipelineConfig.default()
    config.to_yaml(output_path)
    print(f"Created default configuration file: {output_path}")


def main(source: Source, config: PipelineConfig | None = None):
    if config is None:
        config = PipelineConfig.default()

    output_dir = DATA_DIR / "pipeline" / source.name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract fashion mentions
    if config.steps["extract_fashion_mentions"].enabled:
        print("Extracting fashion mentions...")
        extract_fashion_mentions(
            source,
            max_files=None,
            output_filename=output_dir / "fashion_mentions.csv",
        )

    # 2. Filter fashion mentions
    if config.steps["filter_fashion_mentions"].enabled:
        print("Filtering fashion mentions...")
        filter_fashion_mentions(
            input_file=output_dir / "fashion_mentions.csv",
            output_file=output_dir / "fashion_mentions" / "filtered.csv",
            max_rows=None,
        )

    # 3. Run BookNLP on the original texts
    if config.steps["run_booknlp"].enabled:
        print("Running BookNLP...")
        step_config = config.steps["run_booknlp"]
        run_booknlp(
            source,
            num_processes=step_config.num_processes,
            concurrent_processes=step_config.concurrent_processes,
            output_dir=DATA_DIR / "booknlp",
        )

    # 4. Identify the entities present in passages with fashion mentions
    if config.steps["extract_char_fashion_mentions"].enabled:
        print("Identifying entities present in passages with fashion mentions...")
        step_config = config.steps["extract_char_fashion_mentions"]
        extract_char_fashion_mentions(
            source=source,
            fashion_mention_file=output_dir / "fashion_mentions" / "filtered.csv",
            booknlp_dir=DATA_DIR / "booknlp",
            output_dir=output_dir / "character_fashion_cooc",
            num_processes=step_config.num_processes,
            concurrent_processes=step_config.concurrent_processes,
        )

    # 5. Apply a deberta model to identify which entity in a passage is in
    # possession of the mentioned article of clothing
    if config.steps["run_wearing_model"].enabled:
        print("Identifying wearing entities...")
        step_config = config.steps["run_wearing_model"]
        run_wearing_model(
            fashion_cooc_dir=output_dir / "character_fashion_cooc",
            output_dir=output_dir / "wearing",
            num_processes=step_config.num_processes,
            concurrent_processes=step_config.concurrent_processes,
        )

    # 6. Extract sentences which mention entities that have been linked to clothing
    if config.steps["extract_entity_mentions"].enabled:
        print(
            "Extracting sentences which mention entities that have been linked to clothing..."
        )
        step_config = config.steps["extract_entity_mentions"]
        extract_entity_mentions(
            source=source,
            wearing_dir=output_dir / "wearing",
            booknlp_dir=DATA_DIR / "booknlp",
            output_dir=output_dir / "entity_mentions",
            num_processes=step_config.num_processes,
            concurrent_processes=step_config.concurrent_processes,
        )

    # 7. Use dependency parsing to find adjectives which describe a particular noun
    if config.steps["extract_fashion_adjectives"].enabled:
        print("Extracting adjectives which describe fashion mentions...")
        step_config = config.steps["extract_fashion_adjectives"]
        extract_adjectives(
            noun_mention_dir=output_dir / "fashion_mentions",
            output_dir=output_dir / "adjectives/fashion/",
            num_processes=step_config.num_processes,
            concurrent_processes=step_config.concurrent_processes,
            do_coref=True,
        )

    if config.steps["extract_entity_adjectives"].enabled:
        print("Extracting adjectives which describe entity mentions...")
        step_config = config.steps["extract_entity_adjectives"]
        extract_adjectives(
            noun_mention_dir=output_dir / "entity_mentions",
            output_dir=output_dir / "adjectives/entities/",
            num_processes=step_config.num_processes,
            concurrent_processes=step_config.concurrent_processes,
            do_coref=True,
        )

    # 8. Finalize outputs
    if config.steps["finalize_outputs"].enabled:
        print("Finalizing outputs...")
        step_config = config.steps["finalize_outputs"]
        finalize_outputs(
            fashion_mention_file=output_dir / "fashion_mentions" / "filtered.csv",
            adjective_dir=output_dir / "adjectives",
            wearing_dir=output_dir / "wearing",
            booknlp_dir=DATA_DIR / "booknlp",
            output_dir=output_dir,
        )


def add_pipeline_args(parser: argparse.ArgumentParser):
    """Add pipeline configuration arguments to the parser."""
    parser.add_argument("--config", type=Path, help="Path to YAML configuration file")

    parser.add_argument(
        "--create-config",
        type=Path,
        help="Create a default YAML configuration file at the specified path",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "extract_fashion_mentions",
            "filter_fashion_mentions",
            "run_booknlp",
            "extract_char_fashion_mentions",
            "run_wearing_model",
            "extract_entity_mentions",
            "extract_fashion_adjectives",
            "extract_entity_adjectives",
            "finalize_outputs",
        ],
        help="Specific steps to run (default: all steps)",
    )

    # Add process configuration for each step
    for step in [
        "extract_fashion_mentions",
        "filter_fashion_mentions",
        "run_booknlp",
        "extract_char_fashion_mentions",
        "run_wearing_model",
        "extract_entity_mentions",
        "extract_fashion_adjectives",
        "extract_entity_adjectives",
        "finalize_outputs",
    ]:
        parser.add_argument(
            f"--{step}-processes",
            type=int,
            default=None,
            help=f"Number of processes for {step} step",
        )
        parser.add_argument(
            f"--{step}-concurrent",
            type=int,
            default=None,
            help=f"Number of concurrent processes for {step} step",
        )


def create_config_from_args(args) -> PipelineConfig:
    """Create pipeline configuration from command line arguments."""
    # If config file is specified, load from it
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig.default()

    # If specific steps are specified, disable all others
    if args.steps:
        for step_name in config.steps:
            config.steps[step_name].enabled = step_name in args.steps

    # Apply process configurations (command line args override YAML config)
    for step_name in config.steps:
        processes_attr = f"{step_name}_processes"
        concurrent_attr = f"{step_name}_concurrent"

        if hasattr(args, processes_attr) and getattr(args, processes_attr) is not None:
            config.steps[step_name].num_processes = getattr(args, processes_attr)

        if (
            hasattr(args, concurrent_attr)
            and getattr(args, concurrent_attr) is not None
        ):
            config.steps[step_name].concurrent_processes = getattr(
                args, concurrent_attr
            )

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline.")
    add_source_argument(parser)
    add_pipeline_args(parser)
    # add_distributed_args(parser)
    args = parser.parse_args()

    # Handle config file creation
    if args.create_config:
        create_default_yaml_config(args.create_config)
        exit(0)

    config = create_config_from_args(args)
    main(
        source=args.source,
        config=config,
    )
