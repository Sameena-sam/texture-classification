"""
feature_analysis.py
===================
Compares the discriminative power of each feature family (brief Section 3.4):
  * PCA 2-D scatter per feature set
  * t-SNE 2-D embedding per feature set
  * Quantitative separability:
        - Fisher discriminant ratio (between-class var / within-class var)
        - Mean intra-class distance vs mean inter-class centroid distance
  * Summary table -> results/feature_separability.csv
  * Decision: which feature set gives best separability (printed + saved)

Run:  python feature_analysis.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config
import utils
from prepare import load_split


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Fisher ratio: trace(between-class) / trace(within-class)."""
    classes = np.unique(y)
    mean_all = X.mean(axis=0)
    s_w = np.zeros(X.shape[1])
    s_b = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        s_w += ((Xc - mc) ** 2).sum(axis=0)
        s_b += len(Xc) * (mc - mean_all) ** 2
    return float(s_b.sum() / (s_w.sum() + 1e-8))


def separability(X: np.ndarray, y: np.ndarray) -> dict:
    """Compactness (intra) vs separation (inter centroid) using class centroids."""
    classes = np.unique(y)
    centroids = np.vstack([X[y == c].mean(axis=0) for c in classes])
    intra = np.mean([np.linalg.norm(X[y == c] - centroids[i], axis=1).mean()
                     for i, c in enumerate(classes)])
    inter = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            inter.append(np.linalg.norm(centroids[i] - centroids[j]))
    inter = np.mean(inter)
    return {"intra_class_dist": intra, "inter_class_dist": inter,
            "separation_ratio": inter / (intra + 1e-8)}


def run_feature_analysis():
    utils.set_seed()
    rows = []
    for fs in config.FEATURE_SETS:
        d = load_split(fs)
        X, y = d["X_train"], d["y_train"]
        class_names = d["class_names"]

        Xs = StandardScaler().fit_transform(X)
        metrics = {"feature_set": fs, "dim": X.shape[1],
                   "fisher_ratio": fisher_ratio(Xs, y)}
        metrics.update(separability(Xs, y))
        rows.append(metrics)

        # PCA scatter
        p2 = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, cn in enumerate(class_names):
            m = y == i
            ax.scatter(p2[m, 0], p2[m, 1], s=8, alpha=0.5, label=cn)
        ax.set_title(f"PCA - {fs}"); ax.legend(fontsize=7)
        utils.savefig(fig, f"pca_{fs}.png")

        # t-SNE (subsampled for speed)
        n = min(config.TSNE_SAMPLE, len(Xs))
        sel = np.random.RandomState(config.RANDOM_STATE).choice(len(Xs), n, replace=False)
        ts = TSNE(n_components=2, init="pca", perplexity=30,
                  random_state=config.RANDOM_STATE).fit_transform(Xs[sel])
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, cn in enumerate(class_names):
            m = y[sel] == i
            ax.scatter(ts[m, 0], ts[m, 1], s=8, alpha=0.5, label=cn)
        ax.set_title(f"t-SNE - {fs}"); ax.legend(fontsize=7)
        utils.savefig(fig, f"tsne_{fs}.png")

    df = pd.DataFrame(rows).sort_values("separation_ratio", ascending=False)
    df.to_csv(config.RESULTS_DIR / "feature_separability.csv", index=False)
    best = df.iloc[0]["feature_set"]
    print(df.to_string(index=False))
    print(f"\n[feature_analysis] Most separable feature set: '{best}'")
    return df


if __name__ == "__main__":
    run_feature_analysis()
