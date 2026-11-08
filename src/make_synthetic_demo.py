"""
make_synthetic_demo.py
======================
Generates a TINY synthetic texture dataset (procedural patterns) so the full
pipeline -- including the sample-disjoint grouped split -- can be smoke-tested
without the real KTH-TIPS2-b images. This is NOT valid for the assignment; it
only proves the code runs. Replace data/ with the real dataset before producing
report results.

Layout matches KTH-TIPS2-b: data/<class>/sample_<a..d>/*.png so the grouped
split has multiple physical samples per class.

    python src/make_synthetic_demo.py --per-sample 20
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

import config

SAMPLES = ["sample_a", "sample_b", "sample_c", "sample_d"]


def _texture(kind: int, rng: np.random.RandomState) -> np.ndarray:
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == 0:      # wood: warped low-freq stripes (grain), brownish
        g = np.sin((xx + 6*np.sin(yy/20.0)) / 5.0)
        base = (0.5 + 0.5*g); col = np.dstack([base*0.4, base*0.55, base*0.75])
    elif kind == 1:    # cork: blobby speckle, tan
        n = rng.rand(H, W); n = cv2.GaussianBlur(n, (0,0), 3)
        base = (n - n.min())/(np.ptp(n)+1e-8); col = np.dstack([base*0.5, base*0.6, base*0.7])
    elif kind == 2:    # corduroy: strong vertical ridges
        g = 0.5 + 0.5*np.sign(np.sin(xx/3.0)); col = np.dstack([g*0.4, g*0.4, g*0.45])
    elif kind == 3:    # cotton: fine isotropic weave (checker + noise), light
        g = 0.5 + 0.5*np.sin(xx/2.0)*np.sin(yy/2.0)
        col = np.dstack([g*0.85, g*0.85, g*0.85])
    elif kind == 4:    # wool: soft cloudy noise, grey
        n = cv2.GaussianBlur(rng.rand(H, W), (0,0), 5)
        base=(n-n.min())/(np.ptp(n)+1e-8); col=np.dstack([base*0.6,base*0.6,base*0.62])
    else:              # linen: crosshatch weave
        g = 0.5 + 0.25*np.sin(xx/2.5) + 0.25*np.sin(yy/2.5)
        col = np.dstack([g*0.75, g*0.72, g*0.6])
    img = np.clip(col*255, 0, 255).astype(np.uint8)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.RandomState(config.RANDOM_STATE)
    for kind, cname in enumerate(config.CLASSES):
        for s in SAMPLES:
            d = config.DATA_DIR / cname / s
            d.mkdir(parents=True, exist_ok=True)
            for i in range(args.per_sample):
                cv2.imwrite(str(d / f"{cname}_{s}_{i:03d}.png"), _texture(kind, rng))
        print(f"  {cname}: {len(SAMPLES)*args.per_sample} synthetic images")
    print(f"Synthetic texture demo written to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
