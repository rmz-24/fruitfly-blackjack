"""Neuron identities: map FlyWire cell annotations onto simulator indices.

The simulator orders neurons as in ``2025_Completeness_783.csv`` (same as the
upstream fly-brain repository). Cell types come from the FlyWire annotation
release (Schlegel et al. 2024, ``flyconnectome/flywire_annotations``).
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PATH_COMP = RAW / "2025_Completeness_783.csv"
PATH_CONN = RAW / "2025_Connectivity_783.parquet"
PATH_ANNOT = RAW / "neuron_annotations.tsv"


def _require(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/download_data.py` first")
    return path


@lru_cache(maxsize=1)
def neuron_ids():
    """FlyWire root IDs in simulator index order."""
    return pd.read_csv(_require(PATH_COMP), index_col=0).index.to_numpy()


@lru_cache(maxsize=1)
def annotations():
    """Annotation table restricted to simulated neurons, with an ``index`` column."""
    ann = pd.read_csv(_require(PATH_ANNOT), sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_sub_class",
                               "cell_type", "top_nt", "side"])
    idx = pd.Series(np.arange(len(neuron_ids())), index=neuron_ids())
    ann = ann[ann.root_id.isin(idx.index)].copy()
    ann["index"] = idx.loc[ann.root_id].to_numpy()
    return ann.reset_index(drop=True)


def _indices(mask):
    return np.sort(annotations().loc[mask, "index"].to_numpy())


@dataclass
class Anatomy:
    """Index sets of the neuron populations used by the Blackjack task."""

    pn: np.ndarray               # uniglomerular antennal-lobe projection neurons
    pn_glomerulus: np.ndarray    # glomerulus/cell type of each PN (same order as ``pn``)
    kc: np.ndarray               # Kenyon cells
    mbon: np.ndarray             # mushroom body output neurons
    mbon_type: np.ndarray
    dan: np.ndarray              # dopaminergic neurons (PAM / PPL)
    apl: np.ndarray              # anterior paired lateral (KC feedback inhibition)
    dn: np.ndarray               # descending neurons (brain -> nerve cord)
    extra: dict = field(default_factory=dict)

    @property
    def glomeruli(self):
        return sorted(set(self.pn_glomerulus))

    def pns_of(self, glomerulus):
        return self.pn[self.pn_glomerulus == glomerulus]

    def recorded(self):
        """Populations whose spike counts are cached, keyed by name."""
        return {"pn": self.pn, "kc": self.kc, "mbon": self.mbon,
                "dan": self.dan, "apl": self.apl, "dn": self.dn}


@lru_cache(maxsize=1)
def load_anatomy():
    ann = annotations()
    pn_mask = (ann.cell_class == "ALPN") & (ann.cell_sub_class == "uniglomerular")
    pn = ann[pn_mask].sort_values("index")
    mbon = ann[ann.cell_class == "MBON"].sort_values("index")
    return Anatomy(
        pn=pn["index"].to_numpy(),
        pn_glomerulus=pn["cell_type"].to_numpy(dtype=str),
        kc=_indices(ann.cell_class == "Kenyon_Cell"),
        mbon=mbon["index"].to_numpy(),
        mbon_type=mbon["cell_type"].to_numpy(dtype=str),
        dan=_indices(ann.cell_class == "DAN"),
        apl=_indices(ann.cell_type == "APL"),
        dn=_indices(ann.super_class == "descending"),
    )


if __name__ == "__main__":
    a = load_anatomy()
    print(f"neurons simulated : {len(neuron_ids())}")
    for name, ix in a.recorded().items():
        print(f"{name:5s}: {len(ix)}")
    print(f"glomeruli (PN types): {len(a.glomeruli)}")
