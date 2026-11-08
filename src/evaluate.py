"""
evaluate.py
===========
Full held-out test-set evaluation required by brief Section 3.7. For every
trained model it produces:

  * Per-class + macro/weighted Precision, Recall, F1, Accuracy
  * Confusion matrix figure
  * ROC curves + AUC (one-vs-rest) for probabilistic models
  * Stratified k-fold CV: mean +/- std F1 (statistical robustness)
  * Learning curve (bias-variance diagnosis)
  * Feature importance (RF) / |weight| (SVM linear proxy via permutation)
  * Computational cost: train size, inference time per image, model size
  * 3-5 misclassified example images with predicted vs true labels
  * Consolidated comparison table -> results/model_comparison.csv

Run:  python evaluate.py
"""
from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.inspection import permutation_importance

import config
import utils
from prepare import load_split

SUP_MODELS = ["svm", "random_forest", "knn", "mlp"]


def _load(name):
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")


def _confusion(name, y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion matrix - {name}")
    utils.savefig(fig, f"confusion_{name}.png")


def _roc(name, model, X_test, y_test, class_names):
    if not hasattr(model, "predict_proba"):
        return
    y_bin = label_binarize(y_test, classes=range(len(class_names)))
    proba = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    aucs = []
    for i, cn in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        a = auc(fpr, tpr); aucs.append(a)
        ax.plot(fpr, tpr, label=f"{cn} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC (one-vs-rest) - {name}"); ax.legend(fontsize=7)
    utils.savefig(fig, f"roc_{name}.png")
    return float(np.mean(aucs))


def _learning_curve(name, model, X, y):
    sizes, train_sc, val_sc = learning_curve(
        model, X, y, cv=3, scoring="f1_weighted",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sizes, train_sc.mean(1), "o-", label="train")
    ax.plot(sizes, val_sc.mean(1), "o-", label="cv")
    ax.set_xlabel("Training examples"); ax.set_ylabel("F1 (weighted)")
    ax.set_title(f"Learning curve - {name}"); ax.legend()
    utils.savefig(fig, f"learning_curve_{name}.png")


def _feature_importance(name, model, X_test, y_test, feat_names):
    try:
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        else:
            r = permutation_importance(model, X_test, y_test, n_repeats=5,
                                       random_state=config.RANDOM_STATE,
                                       n_jobs=config.N_JOBS)
            imp = r.importances_mean
    except Exception:
        return
    order = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([feat_names[i] for i in order][::-1], imp[order][::-1], color="darkorange")
    ax.set_title(f"Top-15 feature importance - {name}")
    utils.savefig(fig, f"importance_{name}.png")


def _misclassified(name, y_true, y_pred, test_paths, class_names, k=5):
    wrong = np.where(y_true != y_pred)[0][:k]
    if len(wrong) == 0:
        return
    fig, axes = plt.subplots(1, len(wrong), figsize=(3 * len(wrong), 3))
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, wrong):
        img = cv2.cvtColor(utils.load_image(test_paths[idx]), cv2.COLOR_BGR2RGB)
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}",
                     fontsize=8)
    fig.suptitle(f"Misclassified examples - {name}")
    utils.savefig(fig, f"misclassified_{name}.png")


def evaluate():
    utils.set_seed()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    rows = []

    for name in SUP_MODELS:
        fs = config.MODEL_FEATURE_SET[name]
        d = load_split(fs)
        X_te, y_te = d["X_test"], d["y_test"]
        X_tr, y_tr = d["X_train"], d["y_train"]
        class_names = list(d["class_names"])
        feat_names = list(d["feature_names"])
        test_paths = list(d["test_paths"])

        model = _load(name)

        # timed inference
        t0 = time.time(); y_pred = model.predict(X_te); infer = time.time() - t0
        infer_per_img = infer / len(X_te) * 1000  # ms

        rep = classification_report(y_te, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).transpose().to_csv(
            config.RESULTS_DIR / f"classification_report_{name}.csv")

        # cross-validation F1 mean/std on training data
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                                    scoring="f1_weighted", n_jobs=config.N_JOBS)

        # figures
        _confusion(name, y_te, y_pred, class_names)
        mean_auc = _roc(name, model, X_te, y_te, class_names)
        _learning_curve(name, model, X_tr, y_tr)
        _feature_importance(name, model, X_te, y_te, feat_names)
        _misclassified(name, y_te, y_pred, test_paths, class_names)

        size_mb = (config.MODELS_DIR / f"{name}.joblib").stat().st_size / 1e6
        rows.append({
            "model": name, "feature_set": fs,
            "accuracy": accuracy_score(y_te, y_pred),
            "f1_macro": f1_score(y_te, y_pred, average="macro"),
            "f1_weighted": f1_score(y_te, y_pred, average="weighted"),
            "precision_w": rep["weighted avg"]["precision"],
            "recall_w": rep["weighted avg"]["recall"],
            "mean_auc": mean_auc if mean_auc else float("nan"),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "infer_ms_per_img": round(infer_per_img, 3),
            "model_size_mb": round(size_mb, 3),
        })
        print(f"[eval] {name:14s} acc={rows[-1]['accuracy']:.4f} "
              f"f1w={rows[-1]['f1_weighted']:.4f} "
              f"cvF1={cv_scores.mean():.3f}+/-{cv_scores.std():.3f}")

    # k-means row (unsupervised) for completeness
    try:
        km = pd.read_csv(config.RESULTS_DIR / "kmeans_summary.csv").iloc[0]
        rows.append({"model": "kmeans", "feature_set": km["feature_set"],
                     "accuracy": km["purity"], "f1_macro": float("nan"),
                     "f1_weighted": float("nan"), "mean_auc": float("nan"),
                     "cv_f1_mean": float("nan"), "cv_f1_std": float("nan"),
                     "infer_ms_per_img": float("nan"),
                     "model_size_mb": float("nan")})
    except Exception:
        pass

    df = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    df.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n===== MODEL COMPARISON =====")
    print(df.to_string(index=False))

    best = df.iloc[0]["model"]
    (config.MODELS_DIR / "best_model.txt").write_text(str(best))
    print(f"\n[eval] Best model by weighted F1: {best}")
    return df


if __name__ == "__main__":
    evaluate()
