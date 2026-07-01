#!/usr/bin/env python3
"""Train the real-vs-screen detector and save it to screen_detector.pkl.

Usage: python train.py [real_folder] [screen_folder]   (defaults: real/ screen/)

Each image is split into patches; a HistGradientBoosting classifier is trained
on patch features. Accuracy is estimated honestly with StratifiedGroupKFold so
that all patches of an image stay in the same fold (no leakage), and the
decision threshold is tuned on those out-of-fold image scores.
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from features import (SUPPORTED_EXTENSIONS, PATCH_SIZE, MAX_PATCHES,
                      read_image, extract_patches, patch_features)

warnings.filterwarnings("ignore")


def load_patches(real_folder, screen_folder):
    X, y, groups, image_labels = [], [], [], []
    gid = 0
    for folder, label in [(real_folder, 0), (screen_folder, 1)]:
        name = "real" if label == 0 else "screen"
        print(f"Loading {name} photos from {folder}/ ...")
        for fp in sorted(Path(folder).glob("*.*")):
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            for p in extract_patches(read_image(fp)):
                X.append(patch_features(p))
                y.append(label)
                groups.append(gid)
            image_labels.append(label)
            gid += 1
    return (np.array(X, dtype=np.float32), np.array(y),
            np.array(groups), np.array(image_labels))


def oof_image_scores(estimator, X, y, groups, image_labels, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    agg = {}
    for tr, te in sgkf.split(X, y, groups):
        est = clone(estimator).fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        for gidx, pr in zip(groups[te], proba):
            agg.setdefault(gidx, []).append(pr)
    ids = sorted(agg)
    scores = np.array([np.mean(agg[i]) for i in ids])
    labels = np.array([image_labels[i] for i in ids])
    return scores, labels


def tune_threshold(labels, scores):
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        acc = accuracy_score(labels, (scores >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


if __name__ == "__main__":
    real_folder = sys.argv[1] if len(sys.argv) > 1 else "real"
    screen_folder = sys.argv[2] if len(sys.argv) > 2 else "screen"

    X, y, groups, image_labels = load_patches(real_folder, screen_folder)
    n_images = len(image_labels)
    if n_images < 10 or len(np.unique(image_labels)) < 2:
        raise SystemExit("Need >=10 images across both folders to train.")
    print(f"{n_images} images -> {len(X)} patches, {X.shape[1]} features each")

    model = HistGradientBoostingClassifier(max_iter=400, random_state=42)
    scores, labels = oof_image_scores(model, X, y, groups, image_labels)
    threshold, acc = tune_threshold(labels, scores)
    cm = confusion_matrix(labels, (scores >= threshold).astype(int))
    print(f"Cross-validated image accuracy: {acc:.2%} at threshold {threshold:.2f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")

    model.fit(X, y)
    with open("screen_detector.pkl", "wb") as fh:
        pickle.dump({"model": model, "threshold": float(threshold),
                     "patch_size": PATCH_SIZE, "max_patches": MAX_PATCHES}, fh)
    print("Saved screen_detector.pkl")
