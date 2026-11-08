"""
download_data.py
================
Fetch/arrange the KTH-TIPS2-b texture dataset into the layout the pipeline
expects, preserving physical-sample sub-folders so the grouped split works:

    data/
        wood/     sample_a/*.png  sample_b/*.png  sample_c/*.png  sample_d/*.png
        cork/     sample_a/ ...
        corduroy/ ...
        cotton/   ...
        wool/     ...
        linen/    ...

KTH-TIPS2-b has 11 materials x 4 samples x 108 images. We keep the six texture
classes in config.CLASSES.

Acquisition options
-------------------
A) Official KTH site (free): download the KTH-TIPS2-b archive from
     https://www.csc.kth.se/cvap/databases/kth-tips/index.html
   Unzip it (you will get a 'KTH-TIPS2-b/' folder with one sub-folder per
   material, each containing sample_a..sample_d). Then:
     python src/download_data.py --raw "/path/to/KTH-TIPS2-b"

B) Kaggle mirror (if you have a token): search "KTH-TIPS2" on kaggle.com,
   download, unzip, and point --raw at the extracted folder the same way.

The arranger copies only the six configured classes and keeps sample folders.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import config


def arrange(raw: Path):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for cname in config.CLASSES:
        # find the material folder (case-insensitive, name may contain the class)
        matches = [d for d in raw.rglob("*") if d.is_dir()
                   and d.name.lower() == cname.lower()]
        if not matches:
            matches = [d for d in raw.rglob("*") if d.is_dir()
                       and cname.lower() in d.name.lower()]
        if not matches:
            print(f"  ! material folder for '{cname}' not found under {raw}")
            continue
        src = matches[0]
        # sample sub-folders (KTH: 'sample_a'..). If none, put all under sample_a.
        sample_dirs = [d for d in src.iterdir() if d.is_dir()]
        if not sample_dirs:
            sample_dirs = [src]
        n = 0
        for sd in sample_dirs:
            sample_name = sd.name if sd is not src else "sample_a"
            dst = config.DATA_DIR / cname / sample_name
            dst.mkdir(parents=True, exist_ok=True)
            imgs = [p for p in sd.rglob("*")
                    if p.suffix.lower() in config.IMG_EXTENSIONS]
            for p in imgs:
                shutil.copy2(p, dst / p.name)
                n += 1
        total += n
        print(f"  {cname:9s}: {n} images across {len(sample_dirs)} samples")
    print(f"Arranged {total} images into {config.DATA_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="path to the unzipped KTH-TIPS2-b folder")
    args = ap.parse_args()
    arrange(Path(args.raw))
    print("\nNext: python src/run_pipeline.py")


if __name__ == "__main__":
    main()
