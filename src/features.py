"""
features.py
===========
Classical, texture-oriented feature extraction. Five families across the
brief's feature categories:

  Texture & frequency
    lbp     : Local Binary Pattern uniform histogram          (micro-texture)
    glcm    : Grey-Level Co-occurrence Matrix Haralick stats  (spatial texture)
    gabor   : Gabor filter-bank energies (multi-orient/freq)  (oriented texture)
    wavelet : Discrete Wavelet Transform sub-band energies    (multi-scale freq)
  Colour descriptor
    color   : HSV colour histogram                            (material colour)

`extract(img, which)` returns a 1-D float vector; `feature_names(which)` returns
matching names for importance ranking and plots. No deep-learning features.
"""
from __future__ import annotations
from typing import List
import numpy as np
import cv2

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import pywt

import config
from preprocess import normalise_gray


# ----------------------------------------------------------------------------
# Texture & frequency
# ----------------------------------------------------------------------------
def _lbp(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, config.LBP_POINTS, config.LBP_RADIUS,
                               method=config.LBP_METHOD)
    n_bins = config.LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def _glcm(gray: np.ndarray) -> np.ndarray:
    g = (gray / 4).astype(np.uint8)              # 64 grey levels
    angles = [np.deg2rad(a) for a in config.GLCM_ANGLES_DEG]
    glcm = graycomatrix(g, distances=config.GLCM_DISTANCES, angles=angles,
                        levels=64, symmetric=True, normed=True)
    feats = []
    for prop in config.GLCM_PROPS:
        vals = graycoprops(glcm, prop)           # (n_dist, n_angle)
        feats.append(vals.mean())
        feats.append(vals.std())                 # anisotropy / directionality
    return np.asarray(feats, dtype=np.float32)


def _gabor(gray: np.ndarray) -> np.ndarray:
    """Energy (mean + std of magnitude) of the response to a bank of Gabor
    filters over several orientations and frequencies."""
    g = normalise_gray(gray)
    feats = []
    for freq in config.GABOR_FREQS:
        for th in config.GABOR_THETAS_DEG:
            theta = np.deg2rad(th)
            # wavelength lambda = 1/freq; kernel size scales with it
            lam = 1.0 / freq
            ksize = int(2 * np.ceil(lam) + 1)
            kern = cv2.getGaborKernel((ksize, ksize), sigma=lam, theta=theta,
                                      lambd=lam, gamma=0.5, psi=0)
            resp = cv2.filter2D(g, cv2.CV_32F, kern)
            feats.append(float(np.mean(np.abs(resp))))
            feats.append(float(np.std(resp)))
    return np.asarray(feats, dtype=np.float32)


def _wavelet(gray: np.ndarray) -> np.ndarray:
    """Energy of each DWT sub-band over several decomposition levels."""
    g = normalise_gray(gray)
    coeffs = pywt.wavedec2(g, wavelet=config.WAVELET, level=config.WAVELET_LEVEL)
    feats = [float(np.mean(coeffs[0] ** 2))]     # approximation energy
    for level in coeffs[1:]:
        for sub in level:                        # (cH, cV, cD)
            feats.append(float(np.mean(sub ** 2)))
            feats.append(float(np.mean(np.abs(sub))))
    return np.asarray(feats, dtype=np.float32)


# ----------------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------------
def _color(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [config.COLOR_BINS], [0, 256])
        feats.append(cv2.normalize(h, h).flatten())
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def extract(img_bgr: np.ndarray, which: str = "all") -> np.ndarray:
    gray = _gray(img_bgr)
    if which == "lbp":     return _lbp(gray)
    if which == "glcm":    return _glcm(gray)
    if which == "gabor":   return _gabor(gray)
    if which == "wavelet": return _wavelet(gray)
    if which == "color":   return _color(img_bgr)
    if which == "all":
        return np.concatenate([_lbp(gray), _glcm(gray), _gabor(gray),
                               _wavelet(gray), _color(img_bgr)])
    raise ValueError(f"Unknown feature set: {which}")


def feature_names(which: str = "all") -> List[str]:
    lbp_n = [f"lbp_{i}" for i in range(config.LBP_POINTS + 2)]
    glcm_n = []
    for prop in config.GLCM_PROPS:
        glcm_n += [f"glcm_{prop}_mean", f"glcm_{prop}_std"]
    gabor_n = []
    for f in config.GABOR_FREQS:
        for t in config.GABOR_THETAS_DEG:
            gabor_n += [f"gabor_f{f}_t{t}_mean", f"gabor_f{f}_t{t}_std"]
    # wavelet length: 1 (approx) + level*3 subbands * 2 stats
    wav_n = ["wav_approx_energy"]
    for lvl in range(1, config.WAVELET_LEVEL + 1):
        for sub in ("H", "V", "D"):
            wav_n += [f"wav_L{lvl}_{sub}_energy", f"wav_L{lvl}_{sub}_absmean"]
    color_n = ([f"hist_H_{i}" for i in range(config.COLOR_BINS)] +
               [f"hist_S_{i}" for i in range(config.COLOR_BINS)] +
               [f"hist_V_{i}" for i in range(config.COLOR_BINS)])
    mapping = {
        "lbp": lbp_n, "glcm": glcm_n, "gabor": gabor_n,
        "wavelet": wav_n, "color": color_n,
        "all": lbp_n + glcm_n + gabor_n + wav_n + color_n,
    }
    return mapping[which]


def build_feature_matrix(images: List[np.ndarray], which: str) -> np.ndarray:
    return np.vstack([extract(im, which) for im in images]).astype(np.float32)
