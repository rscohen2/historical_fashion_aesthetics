import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

INPUT_DIR = Path("corpus")
OUTPUT_DIR = Path("extracted")
MIN_SIZE = 200 * 1024  # 200 KB


def process_pdf(pdf_path):
    base = pdf_path.stem
    outdir = OUTPUT_DIR / base
    outdir.mkdir(parents=True, exist_ok=True)




    # Run pdfimages
    subprocess.run(
        ["pdfimages", "-all", str(pdf_path), str(outdir / "img")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Filter small images
    for img in outdir.glob("*"):
        if img.is_file() and img.stat().st_size < MIN_SIZE:
            img.unlink()

    return f"Finished {base}"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # pdfs = list(INPUT_DIR.glob("*.pdf"))
    pdfs = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() == ".pdf"]

    with Pool(cpu_count()) as pool:
        for result in pool.imap_unordered(process_pdf, pdfs):
            print(result)
