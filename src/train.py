"""
train.py
========
Trains and tunes the models required by the brief (Section 3.5/3.6):

  Supervised, >=3 models from >=2 families:
    * SVM (RBF)          - margin-based            [classical ML]
    * Random Forest      - bagging ensemble        [ensemble]
    * k-NN               - instance based          [classical ML]
    * MLP                - shallow neural net       [neural, on features only]
  Unsupervised (>=1 required):
    * k-Means            - clustering, evaluated by purity / NMI vs ground truth

Every supervised model is a Pipeline(StandardScaler -> classifier) tuned with
GridSearchCV (stratified 5-fold). The best estimator, the CV table, and the
chosen hyperparameters are saved to models/ and results/.

Run:  python train.py
"""
from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import normalized_mutual_info_score

import config
import utils
from prepare import load_split


def _estimator(name: str):
    if name == "svm":
        return SVC(probability=True, random_state=config.RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(random_state=config.RANDOM_STATE,
                                      n_jobs=config.N_JOBS)
    if name == "knn":
        return KNeighborsClassifier()
    if name == "mlp":
        return MLPClassifier(max_iter=500, early_stopping=True,
                             random_state=config.RANDOM_STATE)
    raise ValueError(name)


def train_supervised():
    utils.set_seed()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    summary = []

    for name in ["svm", "random_forest", "knn", "mlp"]:
        fs = config.MODEL_FEATURE_SET[name]
        d = load_split(fs)
        X_tr, y_tr = d["X_train"], d["y_train"]

        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", _estimator(name))])
        grid = GridSearchCV(pipe, config.PARAM_GRIDS[name], cv=cv,
                            scoring="f1_weighted", n_jobs=config.N_JOBS,
                            return_train_score=True)
        t0 = time.time()
        grid.fit(X_tr, y_tr)
        train_time = time.time() - t0

        joblib.dump(grid.best_estimator_, config.MODELS_DIR / f"{name}.joblib")
        pd.DataFrame(grid.cv_results_).to_csv(
            config.RESULTS_DIR / f"cv_results_{name}.csv", index=False)

        print(f"[train] {name:14s} feat={fs:13s} "
              f"best_f1={grid.best_score_:.4f}  time={train_time:.1f}s")
        print(f"        best params: {grid.best_params_}")
        summary.append({
            "model": name, "feature_set": fs,
            "cv_f1_weighted": grid.best_score_,
            "train_time_s": round(train_time, 2),
            "best_params": json.dumps(grid.best_params_),
        })

    pd.DataFrame(summary).to_csv(
        config.RESULTS_DIR / "training_summary.csv", index=False)
    return summary


def train_unsupervised():
    """k-Means clustering; map clusters to classes by majority vote, report
    purity and NMI against the ground-truth labels."""
    fs = config.MODEL_FEATURE_SET["kmeans"]
    d = load_split(fs)
    X_tr, y_tr = d["X_train"], d["y_train"]
    k = len(np.unique(y_tr))

    scaler = StandardScaler().fit(X_tr)
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)
    clusters = km.fit_predict(scaler.transform(X_tr))

    # cluster -> majority class
    mapping = {}
    for c in range(k):
        labels_in_c = y_tr[clusters == c]
        mapping[c] = np.bincount(labels_in_c).argmax() if len(labels_in_c) else 0
    mapped = np.array([mapping[c] for c in clusters])
    purity = (mapped == y_tr).mean()
    nmi = normalized_mutual_info_score(y_tr, clusters)

    joblib.dump({"kmeans": km, "scaler": scaler, "mapping": mapping},
                config.MODELS_DIR / "kmeans.joblib")
    print(f"[train] kmeans          feat={fs:13s} purity={purity:.4f}  NMI={nmi:.4f}")
    pd.DataFrame([{"model": "kmeans", "feature_set": fs,
                   "purity": purity, "nmi": nmi}]).to_csv(
        config.RESULTS_DIR / "kmeans_summary.csv", index=False)
    return {"purity": purity, "nmi": nmi}


if __name__ == "__main__":
    train_supervised()
    train_unsupervised()
