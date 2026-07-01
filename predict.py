#!/usr/bin/env python3
"""Predict whether an image is a real photo (0) or a photo of a screen (1).

Usage: python predict.py <image_path> [model_path]
Prints a single probability in [0, 1] to stdout (0 = real, 1 = screen).
"""

import os
import pickle
import sys
import time

import numpy as np

from features import read_image, extract_patches, patch_features

_CACHE = {}


def load_model(model_path="screen_detector.pkl"):
    if model_path not in _CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run: python train.py")
        with open(model_path, "rb") as f:
            _CACHE[model_path] = pickle.load(f)
    data = _CACHE[model_path]
    return data["model"], data.get("threshold", 0.5)


def predict_with_threshold(image_path, model_path="screen_detector.pkl"):
    model, threshold = load_model(model_path)
    patches = extract_patches(read_image(image_path))
    feats = np.array([patch_features(p) for p in patches], dtype=np.float32)
    score = float(model.predict_proba(feats)[:, 1].mean())
    return score, int(score >= threshold)


def predict(image_path, model_path="screen_detector.pkl"):
    score, _ = predict_with_threshold(image_path, model_path)
    return score


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "screen_detector.pkl"
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        start = time.time()
        score, prediction = predict_with_threshold(image_path, model_path)
        elapsed_ms = (time.time() - start) * 1000
        print(f"{score:.4f}")
        print(f"# Latency: {elapsed_ms:.1f}ms | Prediction: {'SCREEN' if prediction else 'REAL'} "
              f"| Confidence: {score:.2%}", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
