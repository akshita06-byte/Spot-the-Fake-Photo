#!/usr/bin/env python3
"""Image I/O and patch-based feature extraction for real-vs-screen detection.

A photo of a screen leaves physical fingerprints that classic computer vision
can measure directly. We turn each image into a set of native-resolution
patches and describe every patch with ~44 features built around the cues the
recapture literature flags as most reliable:

  - multi-scale Local Binary Patterns (LBP)   -> screen micro-texture / grid
  - HSV + chromaticity colour statistics       -> sub-pixel colour artifacts
  - blur / sharpness                           -> recaptures are slightly soft
  - windowed-FFT high-frequency energy + peaks -> display moire
  - noise residual + colour banding            -> display noise / limited gamut

Patch scores are averaged into one image score at predict time, so a single
noisy region cannot flip the verdict.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
from skimage.feature import local_binary_pattern

register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
PATCH_SIZE = 256
MAX_PATCHES = 9


def read_image(image_path):
    """Load any supported image (incl. iPhone HEIC) as a BGR uint8 array."""
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_patches(image, size=PATCH_SIZE, max_patches=MAX_PATCHES):
    """Up to 9 native-resolution patches on a 3x3 grid (preserves the screen grid)."""
    h, w = image.shape[:2]
    if h < size or w < size:
        return [cv2.resize(image, (size, size))]
    ys = np.linspace(0, h - size, num=3, dtype=int)
    xs = np.linspace(0, w - size, num=3, dtype=int)
    patches = [image[y:y + size, x:x + size] for y in ys for x in xs]
    return patches[:max_patches]


def _lbp_hist(gray, P, R):
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(P + 3), density=True)
    return hist  # length P + 2


def patch_features(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    f = []

    # texture: multi-scale LBP (strongest recapture cue)
    f += list(_lbp_hist(gray, 8, 1))    # 10
    f += list(_lbp_hist(gray, 16, 2))   # 18

    # colour: HSV stats + chromaticity (sub-pixel colour artifacts)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    for c in range(3):
        f += [hsv[:, :, c].mean(), hsv[:, :, c].std()]   # 6
    b, gg, r = (patch[:, :, i].astype(np.float32) for i in range(3))
    s = b + gg + r + 1e-6
    rc, gc = r / s, gg / s
    f += [rc.mean(), rc.std(), gc.mean(), gc.std()]      # 4

    # blur / sharpness
    f += [cv2.Laplacian(gray, cv2.CV_64F).var()]         # 1
    sx, sy = cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1)
    f += [float(np.sqrt(sx ** 2 + sy ** 2).mean())]      # 1

    # moire: windowed FFT high-freq energy ratio + peakiness
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    mag = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win)))
    hh, ww = mag.shape
    cy, cx = hh // 2, ww // 2
    yy, xx = np.ogrid[:hh, :ww]
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = min(cy, cx)
    low = mag[rad <= 0.15 * rmax].sum()
    high = mag[(rad >= 0.35 * rmax) & (rad <= 0.95 * rmax)]
    f += [float(high.sum() / (low + 1e-6)), float(high.max() / (np.median(high) + 1e-6))]  # 2

    # noise residual + colour banding
    res = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    f += [float(res.std() / (g.std() + 1e-6))]                          # 1
    f += [len(np.unique(hsv[:, :, 2].astype(np.uint8))) / 256.0]        # 1

    return np.array(f, dtype=np.float32)


def image_features(image_path):
    """Feature matrix (one row per patch) for a single image."""
    image = read_image(image_path)
    return np.array([patch_features(p) for p in extract_patches(image)], dtype=np.float32)
