"""
utils.py
========
Shared helpers: dataset discovery (with physical-sample grouping),
image loading, reproducibility and figure saving.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _group_of(img_path: Path, class_dir: Path) -> str:
    """
    Determine the physical-sample group for an image so the split can be
    sample-disjoint. KTH-TIPS2-b layout is <class>/<sample_x>/*.png, so the
    group is the immediate sub-folder under the class. If images sit directly
    under the class folder we fall back to a filename prefix, then to 'all'.
    """
    rel = img_path.relative_to(class_dir)
    if len(rel.parts) >= 2:
        return rel.parts[0]                # e.g. 'sample_a'
    stem = img_path.stem
    # try 'sampleA_...' or 'a_...' style prefixes
    for sep in ("_", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return "all"


def list_dataset(data_dir: Path = config.DATA_DIR,
                 classes: List[str] = None,
                 max_per_class: int = config.MAX_PER_CLASS,
                 seed: int = config.RANDOM_STATE
                 ) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """
    Return (paths, labels, groups, class_names).
    `groups[i]` is a globally-unique physical-sample id ("<class>/<sample>")
    used by the grouped split so no sample appears in more than one split.
    """
    classes = classes or config.CLASSES
    rng = random.Random(seed)
    paths, labels, groups = [], [], []

    for idx, cname in enumerate(classes):
        cdir = data_dir / cname
        if not cdir.is_dir():
            continue
        files = [p for p in cdir.rglob("*")
                 if p.suffix.lower() in config.IMG_EXTENSIONS]
        rng.shuffle(files)
        files = files[:max_per_class]
        for f in files:
            paths.append(f)
            labels.append(idx)
            groups.append(f"{cname}/{_group_of(f, cdir)}")

    if not paths:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Expected data/<class>/<sample>/*.png "
            f"for classes {classes}. See data/dataset_link.txt."
        )
    return paths, labels, groups, classes


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def savefig(fig, name: str) -> Path:
    out = config.RESULTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
