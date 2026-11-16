"""A separately trained digit/non-digit verifier with calibrated acceptance.

The negative examples are synthetic text, not a comprehensive handwriting corpus.
Its measured limits are saved alongside the artifact; it cannot guarantee rejection
of every symbol that resembles a digit.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

ARTIFACT_NAME = "digit_verifier.pkl"


def verifier_features(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28)
    if not np.isfinite(images).all() or (images < 0).any() or (images > 1).any():
        raise ValueError("Verifier inputs must be finite, normalized 28 x 28 images.")
    return images.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)


def calibrated_threshold(digit_probabilities, target_acceptance=0.99):
    """Use only calibration digits; never select a threshold on the test split."""
    scores = np.asarray(digit_probabilities, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration requires finite digit probabilities.")
    if not 0 < target_acceptance <= 1:
        raise ValueError("Target acceptance must be between 0 and 1.")
    index = int(np.floor((1 - target_acceptance) * len(scores)))
    return float(np.sort(scores)[index])


@lru_cache(maxsize=2)
def _load_verifier(path: str, modified: int):
    artifact = joblib.load(path)
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported digit verifier. Retrain it with scripts/train_rejection.py.")
    return artifact


def load_verifier():
    from .model_utils import MODELS_DIR
    path = MODELS_DIR / ARTIFACT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            "Digit validation is not trained. Run python scripts/train_rejection.py first."
        )
    return _load_verifier(str(path), path.stat().st_mtime_ns)


def digit_acceptance_mask(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32).reshape(-1, 784)
    artifact = load_verifier()
    probabilities = artifact["model"].predict_proba(verifier_features(images))[:, 1]
    contrast = np.ptp(images, axis=1) >= 0.05
    return contrast & np.isfinite(probabilities) & (probabilities >= artifact["threshold"])
