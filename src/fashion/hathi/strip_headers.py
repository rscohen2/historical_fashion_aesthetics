import argparse
from pathlib import Path

from ht_text_prep.htrc.models import HtrcPage
from ht_text_prep.htrc.runningheaders import parse_page_structure

from fashion.distributed import add_distributed_args, run_distributed


def main(input_dir: Path, output_dir: Path):
    def process_pages(page_files: list[Path]):
        pages = []
        for page_file in page_files:
            text = page_file.read_text()
            pages = [
                HtrcPage(lines=page.strip().split("\n")) for page in text.split("<pb>")
            ]
            pages = parse_page_structure(pages)
            # conservatively filter out page numbers by excluding lines that are
            # less than 4 characters and all digits
            body_lines = [
                line
                for page in pages
                for line in page.body_lines
                if not (len(line.strip()) < 4 and line.strip().isdigit())
            ]
            full_text = "\n".join(body_lines)
            output_file = output_dir / page_file.name
            output_file.write_text(full_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    page_files = sorted(input_dir.glob("*.txt"))
    run_distributed(
        process_pages,
        page_files,
        extra_args=[
            str(input_dir),
            str(output_dir),
        ],
        total_processes=240,
        concurrent_processes=24,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_distributed_args(parser)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
