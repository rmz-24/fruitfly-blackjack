"""Download the connectome and annotation data needed by fruitfly-blackjack.

Files are written to ``data/raw/`` (gitignored):

* ``2025_Completeness_783.csv``       - neuron list / tensor index order (fly-brain repo)
* ``2025_Connectivity_783.parquet``   - signed synapse counts (fly-brain repo)
* ``neuron_annotations.tsv``          - FlyWire cell types (flyconnectome/flywire_annotations)
"""

import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FLY_BRAIN = "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data"
ANNOTATIONS = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

FILES = {
    "2025_Completeness_783.csv": f"{FLY_BRAIN}/2025_Completeness_783.csv",
    "2025_Connectivity_783.parquet": f"{FLY_BRAIN}/2025_Connectivity_783.parquet",
    "neuron_annotations.tsv": ANNOTATIONS,
}


def _progress(name):
    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100.0, 100.0 * done / total)
            sys.stdout.write(f"\r  {name}: {pct:5.1f}% of {total / 1e6:.1f} MB")
            sys.stdout.flush()
    return hook


def main(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"  {name}: already present")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp, reporthook=_progress(name))
        tmp.rename(dest)
        print()
    print(f"Data ready in {RAW_DIR}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
