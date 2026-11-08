# Surface-Material Texture Classification (Classical Pattern Recognition)

A complete **extract → represent → classify** pipeline that classifies close-up
photographs of material surfaces into six texture classes — **wood, cork,
corduroy, cotton, wool, linen** — from the KTH-TIPS2-b benchmark, using *classical*
feature engineering and machine learning only (no deep learning / no CNNs).

## Why this is a strong pattern-recognition problem
Texture is exactly what classical descriptors were designed for. The pipeline
implements and compares five handcrafted feature families and uses a
**sample-aware (grouped) split** so no physical sample appears in more than one
split — the rigorous protocol that gives an honest generalisation estimate.

## What it does
1. **Pre-processing** — resize, CLAHE illumination normalisation, grayscale
   z-score normalisation, training-set augmentation (rotation/flip/brightness).
2. **Feature extraction (5 families, classical)**
   - *Texture & frequency*: LBP histogram, GLCM Haralick stats, **Gabor filter-bank
     energies**, **Discrete Wavelet Transform sub-band energies**
   - *Colour*: HSV colour histogram
3. **Feature analysis** — Fisher ratio & separability, PCA & t-SNE plots to pick
   the best representation per classifier.
4. **Models (≥3, ≥2 families, supervised + unsupervised)** — SVM (RBF),
   Random Forest, k-NN, MLP, and k-Means (purity/NMI).
5. **Tuning** — GridSearchCV, stratified 5-fold.
6. **Evaluation** — per-class P/R/F1, accuracy, confusion matrices, ROC/AUC,
   CV mean±sd, learning curves, feature importance, compute cost, misclassified
   examples, consolidated comparison table.
7. **Deployment** — CLI (`predict.py`) and a Streamlit web app (`app.py`).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
See `data/dataset_link.txt`. Quickest path:
```bash
# 1. Download + unzip KTH-TIPS2-b from the KTH CVAP site
# 2. Arrange it:
python src/download_data.py --raw "/path/to/KTH-TIPS2-b"
```
Or to just verify the code runs (procedural textures, for a quick smoke test only):
```bash
python src/make_synthetic_demo.py --per-sample 20
```
Expected layout:
```
data/
  wood/sample_a/*.png  wood/sample_b/*.png  ...
  cork/  corduroy/  cotton/  wool/  linen/
```

## Run the whole experiment
```bash
python src/run_pipeline.py
```
Outputs land in `results/` (figures + CSV tables) and `models/` (.joblib).

Individual stages:
```bash
python src/eda.py
python src/prepare.py
python src/feature_analysis.py
python src/train.py
python src/evaluate.py
```

## Use the deployed classifier
```bash
python src/predict.py --image path/to/texture.png
python src/predict.py --folder path/to/folder/
streamlit run src/app.py
```

## Configuration
All settings (classes, split mode, feature params, model grids) live in
`src/config.py`. Set `SPLIT_MODE = "stratified"` to compare against the
optimistic random split.

## Repo structure
```
texture-classification/
  src/  data/  models/  results/  cache/
  requirements.txt  README.md  AI_DISCLOSURE.md
```

## Notes
- Deep-learning frameworks are intentionally **not** used (the coursework brief excluded them).
- Reproducible: fixed `RANDOM_STATE`.
- See `AI_DISCLOSURE.md` for the required AI-use declaration.

## Results

On a sample-disjoint held-out split, the MLP on the combined feature set reached a weighted F1 of
0.791 (accuracy 0.799), with Random Forest close behind (weighted F1 0.790, accuracy 0.809).

## Authors

Sameena Siddiqua and Loula Sahal Ali. Pattern Recognition coursework (CT104-3-M), MSc Artificial
Intelligence, Asia Pacific University (2026).
