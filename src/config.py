"""
config.py
=========
Central configuration for the CT104-3-M **surface-material texture
classification** pipeline (KTH-TIPS2-b). Editing values here changes the
behaviour of every stage (preprocessing, feature extraction, training,
evaluation, deployment).

Domain: classify close-up photographs of material surfaces into texture
classes drawn from the KTH-TIPS2-b benchmark. We use a wood/natural + textile
subset. Because KTH-TIPS2-b contains four distinct *physical samples* per
material, we use a SAMPLE-AWARE (grouped) split so that no image patch from a
sample seen in training can leak into validation/test -- the rigorous protocol
used in the texture-classification literature.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_ROOT / "data"     # data/<class>/<sample>/*.png  (or data/<class>/*.png)
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR   = PROJECT_ROOT / "cache"
for _d in (MODELS_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------------
# KTH-TIPS2-b material classes used (natural/woody + textile). All exist in the
# benchmark under these exact folder names.
CLASSES = ["wood", "cork", "corduroy", "cotton", "wool", "linen"]

# KTH-TIPS2-b has 108 images x 4 samples = 432 per material. 6 x 432 ~= 2,592
# images (within the brief's 2,000-5,000 range). Cap per class if desired.
MAX_PER_CLASS = 500

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ----------------------------------------------------------------------------
# 3. Pre-processing
# ----------------------------------------------------------------------------
IMG_SIZE = (128, 128)          # resize target (H, W)
NORMALISE = "zscore"           # grayscale normalisation for texture features
APPLY_CLAHE = True             # local contrast enhancement (helps LBP/GLCM)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
DENOISE = False                # texture IS high-freq signal; denoise off by default

# --- Split -------------------------------------------------------------------
# "grouped"    : disjoint physical samples across train/val/test (recommended).
# "stratified" : plain stratified random split (optimistic; for comparison only).
SPLIT_MODE = "grouped"
TEST_SIZE = 0.25               # in GROUP space when grouped (4 samples -> 1 test)
VAL_SIZE  = 0.25               # -> 1 val, 2 train for KTH-TIPS2-b
RANDOM_STATE = 42

# Augmentation (training only): texture is orientation/illumination sensitive,
# so we add rotation and brightness jitter to build invariance. Justified in
# the report.
AUGMENT = True
AUG_PER_IMAGE = 1
AUG_ROTATE_DEG = 30
AUG_FLIP = True
AUG_BRIGHTNESS = 0.15

# ----------------------------------------------------------------------------
# 4. Feature extraction (all classical, texture-oriented)
# ----------------------------------------------------------------------------
# Five families from THREE brief categories:
#   Texture & frequency : lbp, glcm, gabor, wavelet
#   Colour descriptor   : color (HSV histogram)
# Combined "all" concatenates everything.
FEATURE_SETS = ["lbp", "glcm", "gabor", "wavelet", "color", "all"]
DEFAULT_FEATURE_SET = "all"

# LBP (uniform, rotation-invariant-ish)
LBP_RADIUS = 3
LBP_POINTS = 24
LBP_METHOD = "uniform"

# GLCM
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES_DEG = [0, 45, 90, 135]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# Gabor filter bank
GABOR_FREQS = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS_DEG = [0, 45, 90, 135]

# Discrete Wavelet Transform (sub-band energies)
WAVELET = "db4"
WAVELET_LEVEL = 3

# Colour histogram (HSV)
COLOR_BINS = 16                # per channel -> 48 features

# Dimensionality reduction (visualisation / analysis)
PCA_COMPONENTS = 0.95
TSNE_SAMPLE = 1500

# ----------------------------------------------------------------------------
# 5. Models & hyperparameter grids
# ----------------------------------------------------------------------------
CV_FOLDS = 5
N_JOBS = -1

PARAM_GRIDS = {
    "svm": {
        "clf__C": [1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
        "clf__kernel": ["rbf"],
    },
    "random_forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_split": [2, 5],
    },
    "knn": {
        "clf__n_neighbors": [1, 3, 5, 7],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(128,), (128, 64)],
        "clf__alpha": [1e-4, 1e-3],
    },
}

# Feature set fed to each model (revisit after feature_analysis.py).
MODEL_FEATURE_SET = {
    "svm": "all",
    "random_forest": "all",
    "knn": "lbp",
    "mlp": "all",
    "kmeans": "all",
}
