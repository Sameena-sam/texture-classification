"""
prepare.py
==========
Builds cached, ready-to-train feature matrices from the image folder.

Steps:
  1. Discover dataset with physical-sample groups (utils.list_dataset).
  2. Split into train/val/test. In "grouped" mode (default) the split is
     SAMPLE-DISJOINT within each class: whole physical samples go to exactly one
     split, so no near-duplicate patch from the same sample leaks across splits.
     This is the rigorous KTH-TIPS2 protocol and yields honest generalisation
     estimates. "stratified" mode does a plain random split for comparison.
  3. Pre-process all images; augment the TRAINING set only.
  4. For each feature family build (N, D) matrices and cache to
     cache/features_<set>.npz with labels, groups and split arrays.

Run:  python prepare.py
Other modules call load_split(feature_set).
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split

import config
import utils
import preprocess as pp
import features as ft


def _grouped_split(labels, groups):
    """Per-class, split unique physical samples into train/val/test groups."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = np.arange(len(labels))
    tr, va, te = [], [], []
    for c in np.unique(labels):
        cls_idx = idx[labels == c]
        g = np.array(groups)[cls_idx]
        uniq = np.array(sorted(set(g)))
        rng.shuffle(uniq)
        n = len(uniq)
        n_test = max(1, int(round(n * config.TEST_SIZE)))
        n_val = max(1, int(round(n * config.VAL_SIZE))) if n - n_test > 1 else 0
        test_g = set(uniq[:n_test])
        val_g = set(uniq[n_test:n_test + n_val])
        for i, gi in zip(cls_idx, g):
            if gi in test_g:   te.append(i)
            elif gi in val_g:  va.append(i)
            else:              tr.append(i)
    return np.array(tr), np.array(va), np.array(te)


def _stratified_split(labels):
    idx = np.arange(len(labels))
    tr, tmp = train_test_split(idx, test_size=config.TEST_SIZE + config.VAL_SIZE,
                               stratify=labels, random_state=config.RANDOM_STATE)
    rel = config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    va, te = train_test_split(tmp, test_size=rel, stratify=labels[tmp],
                              random_state=config.RANDOM_STATE)
    return tr, va, te


def prepare(verbose: bool = True):
    utils.set_seed()
    paths, labels, groups, class_names = utils.list_dataset()
    labels = np.asarray(labels)
    if verbose:
        print(f"[prepare] {len(paths)} images, {len(set(labels))} classes, "
              f"{len(set(groups))} physical samples")
        for i, c in enumerate(class_names):
            print(f"          {c:10s}: {(labels == i).sum()}")

    if config.SPLIT_MODE == "grouped":
        idx_train, idx_val, idx_test = _grouped_split(labels, groups)
        print(f"[prepare] split mode = grouped (sample-disjoint)")
    else:
        idx_train, idx_val, idx_test = _stratified_split(labels)
        print(f"[prepare] split mode = stratified (random)")
    if verbose:
        print(f"[prepare] train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    proc = [pp.preprocess(utils.load_image(p)) for p in paths]

    rng = np.random.RandomState(config.RANDOM_STATE)
    aug_imgs, aug_lbls = [], []
    if config.AUGMENT:
        for i in idx_train:
            for _ in range(config.AUG_PER_IMAGE):
                aug_imgs.append(pp.augment(proc[i], rng))
                aug_lbls.append(labels[i])
        if verbose:
            print(f"[prepare] added {len(aug_imgs)} augmented training images")

    out = {}
    for fs in config.FEATURE_SETS:
        if verbose:
            print(f"[prepare] extracting '{fs}' ...")
        X_train = ft.build_feature_matrix([proc[i] for i in idx_train], fs)
        y_train = labels[idx_train].copy()
        if aug_imgs:
            X_aug = ft.build_feature_matrix(aug_imgs, fs)
            X_train = np.vstack([X_train, X_aug])
            y_train = np.concatenate([y_train, np.asarray(aug_lbls)])
        X_val = ft.build_feature_matrix([proc[i] for i in idx_val], fs) if len(idx_val) else np.empty((0, X_train.shape[1]))
        X_test = ft.build_feature_matrix([proc[i] for i in idx_test], fs)

        f = config.CACHE_DIR / f"features_{fs}.npz"
        np.savez_compressed(
            f,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=labels[idx_val] if len(idx_val) else np.array([]),
            X_test=X_test, y_test=labels[idx_test],
            test_paths=np.array([str(paths[i]) for i in idx_test]),
            class_names=np.array(class_names),
            feature_names=np.array(ft.feature_names(fs)),
        )
        out[fs] = f
        if verbose:
            print(f"          saved {f.name}  X_train={X_train.shape}")
    return out


def load_split(feature_set: str = config.DEFAULT_FEATURE_SET) -> dict:
    cache = config.CACHE_DIR / f"features_{feature_set}.npz"
    if not cache.exists():
        prepare()
    d = np.load(cache, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    prepare()
