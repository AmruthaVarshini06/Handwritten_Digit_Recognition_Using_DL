"""Shared dataset validation for training and evaluation."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    columns = [f"pixel{i}" for i in range(784)]
    if set(data.columns) != {"label", *columns}:
        raise ValueError("Training data must contain label and pixel0 through pixel783.")
    features = data[columns].apply(pd.to_numeric, errors="raise")
    labels = pd.to_numeric(data["label"], errors="raise")
    pixels = features.to_numpy(dtype=np.float32)
    if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 255).any():
        raise ValueError("Training pixels must be finite numbers between 0 and 255.")
    if labels.isna().any() or not labels.isin(range(10)).all():
        raise ValueError("Training labels must be integers from 0 to 9.")
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("A stratified split requires at least two classes and two rows per class.")
    return features.astype(np.float32) / 255.0, labels.astype(int)
