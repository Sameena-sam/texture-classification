"""
predict.py
==========
Deployment / inference entry point (brief Section 3.8). Loads the best trained
model and classifies a single surface-texture image or a folder of images,
printing the predicted material class and a confidence score (posterior
probability).

Usage:
    python predict.py --image path/to/cell.jpg
    python predict.py --folder path/to/folder/
    python predict.py --image cell.jpg --model svm   # force a specific model

The same pre-processing + feature extraction used in training is applied here,
guaranteeing train/serve consistency.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np

import config
import utils
import preprocess as pp
import features as ft


def _best_model_name() -> str:
    f = config.MODELS_DIR / "best_model.txt"
    return f.read_text().strip() if f.exists() else "svm"


def load_model(name: str = None):
    name = name or _best_model_name()
    model = joblib.load(config.MODELS_DIR / f"{name}.joblib")
    feature_set = config.MODEL_FEATURE_SET[name]
    return name, model, feature_set


def predict_image(path: Path, model, feature_set: str, class_names):
    img = utils.load_image(path)
    proc = pp.preprocess(img)
    x = ft.extract(proc, feature_set).reshape(1, -1)
    pred = int(model.predict(x)[0])
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(x)[0][pred])
    return class_names[pred], conf


def main():
    ap = argparse.ArgumentParser(description="Texture classifier inference")
    ap.add_argument("--image", type=str, help="path to a single image")
    ap.add_argument("--folder", type=str, help="path to a folder of images")
    ap.add_argument("--model", type=str, default=None,
                    help="model name (svm/random_forest/knn/mlp); default=best")
    args = ap.parse_args()

    name, model, feature_set = load_model(args.model)
    class_names = config.CLASSES
    print(f"Loaded model: {name}  (feature set: {feature_set})\n")

    targets = []
    if args.image:
        targets = [Path(args.image)]
    elif args.folder:
        targets = [p for p in Path(args.folder).rglob("*")
                   if p.suffix.lower() in config.IMG_EXTENSIONS]
    else:
        ap.error("provide --image or --folder")

    for p in targets:
        try:
            label, conf = predict_image(p, model, feature_set, class_names)
            c = f"{conf*100:5.1f}%" if conf is not None else "  n/a"
            print(f"{p.name:40s} -> {label:12s}  (confidence {c})")
        except Exception as e:
            print(f"{p.name:40s} -> ERROR: {e}")


if __name__ == "__main__":
    main()
