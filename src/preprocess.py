"""
preprocess.py
=============
Pre-processing for surface-texture images. Steps chosen for the texture domain
and justified in the report (Section 5):

* Resize to 128x128 so every descriptor has a fixed length.
* CLAHE on the LAB L-channel: normalises local illumination differences between
  KTH-TIPS2-b lighting conditions without destroying the high-frequency texture
  signal (global histogram equalisation would).
* Denoising is DISABLED by default: for texture, fine high-frequency detail is
  the signal, so we must not blur it away (configurable via config.DENOISE).
* Grayscale z-score normalisation is applied inside the feature extractor so LBP/
  GLCM/Gabor/wavelet responses are illumination-comparable across images.

Exposes:
  preprocess(img)          -> uint8 BGR image for feature extraction
  normalise_gray(gray)     -> normalised float grayscale
  augment(img, rng)        -> randomly transformed training copy
  colour_space_views(img)  -> {RGB, GRAY, HSV, LAB} for EDA
"""
from __future__ import annotations
import numpy as np
import cv2

import config


def _to_size(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (config.IMG_SIZE[1], config.IMG_SIZE[0]),
                      interpolation=cv2.INTER_AREA)


def _clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = _to_size(img_bgr)
    if config.DENOISE:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if config.APPLY_CLAHE:
        img = _clahe_lab(img)
    return img


def normalise_gray(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    if config.NORMALISE == "minmax":
        rng = g.max() - g.min()
        return (g - g.min()) / (rng + 1e-8)
    return (g - g.mean()) / (g.std() + 1e-8)


def augment(img_bgr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Rotation + flip + brightness jitter to build texture invariance."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    angle = rng.uniform(-config.AUG_ROTATE_DEG, config.AUG_ROTATE_DEG)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    if config.AUG_FLIP and rng.rand() < 0.5:
        img = cv2.flip(img, rng.randint(-1, 2))

    factor = 1.0 + rng.uniform(-config.AUG_BRIGHTNESS, config.AUG_BRIGHTNESS)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img


def colour_space_views(img_bgr: np.ndarray) -> dict:
    img = _to_size(img_bgr)
    return {
        "RGB":  cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "GRAY": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "HSV":  cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
        "LAB":  cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
    }
