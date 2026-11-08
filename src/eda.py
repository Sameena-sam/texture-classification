"""
eda.py
======
Exploratory Data Analysis figures for the report (Section 4):
  * class_distribution.png   - bar chart of images per class
  * sample_grid.png          - example images, one row per class
  * colour_spaces.png        - RGB/GRAY/HSV/LAB views of one cell
  * pixel_histograms.png     - mean per-channel intensity histograms per class
  * preprocess_demo.png      - raw vs pre-processed comparison

Run:  python eda.py
"""
from __future__ import annotations
import numpy as np
import cv2
import matplotlib.pyplot as plt

import config
import utils
import preprocess as pp


def run_eda():
    utils.set_seed()
    paths, labels, groups, class_names = utils.list_dataset()
    labels = np.asarray(labels)

    # 1. Class distribution -----------------------------------------------
    counts = [(labels == i).sum() for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, counts, color="steelblue")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    utils.savefig(fig, "class_distribution.png")

    # 2. Sample grid (one row per class) ----------------------------------
    n_cols = 5
    fig, axes = plt.subplots(len(class_names), n_cols,
                             figsize=(2 * n_cols, 2 * len(class_names)))
    axes = np.atleast_2d(axes)
    for r, cname in enumerate(class_names):
        cls_paths = [p for p, l in zip(paths, labels) if l == r][:n_cols]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(cls_paths):
                img = cv2.cvtColor(utils.load_image(cls_paths[c]), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            if c == 0:
                ax.set_title(cname, loc="left", fontsize=10)
    fig.suptitle("Sample images per class")
    utils.savefig(fig, "sample_grid.png")

    # 3. Colour-space views of one image ----------------------------------
    views = pp.colour_space_views(utils.load_image(paths[0]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (name, im) in zip(axes, views.items()):
        ax.imshow(im if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Colour-space representations")
    utils.savefig(fig, "colour_spaces.png")

    # 4. Mean pixel intensity histograms per class ------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for r, cname in enumerate(class_names):
        cls_paths = [p for p, l in zip(paths, labels) if l == r][:60]
        acc = np.zeros(256)
        for p in cls_paths:
            g = cv2.cvtColor(utils.load_image(p), cv2.COLOR_BGR2GRAY)
            acc += cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        acc /= max(len(cls_paths), 1)
        ax.plot(acc, label=cname)
    ax.set_xlabel("Grey level"); ax.set_ylabel("Mean count")
    ax.set_title("Mean grayscale intensity histogram by class")
    ax.legend()
    utils.savefig(fig, "pixel_histograms.png")

    # 5. Pre-processing demo ---------------------------------------------
    raw = utils.load_image(paths[0])
    proc = pp.preprocess(raw)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)); axes[0].set_title("Raw"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)); axes[1].set_title("Pre-processed"); axes[1].axis("off")
    utils.savefig(fig, "preprocess_demo.png")

    print(f"[eda] figures written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    run_eda()
