from __future__ import annotations

from pathlib import Path
import sys
import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from digit_recognition.evaluation import metric_row, write_results
from digit_recognition.data import load_training_data
from digit_recognition.rejection import ARTIFACT_NAME, digit_acceptance_mask
from scripts.train_rejection import train as train_verifier, default_font_dir
from digit_recognition.model_utils import DATA_DIR, MODELS_DIR, RESULTS_DIR
DATA_PATH = DATA_DIR / "train.csv"
MODELS_PATH = MODELS_DIR


def load_data() -> tuple[np.ndarray, np.ndarray]:
    features, labels = load_training_data(DATA_PATH)
    return features.to_numpy(), labels.to_numpy()


def save_model(model, name: str) -> None:
    MODELS_PATH.mkdir(exist_ok=True)
    output_path = MODELS_PATH / f"{name}_digit_model.pkl"
    joblib.dump(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train digit classifiers on a stratified 80/20 split.")
    parser.add_argument("--models", nargs="+", choices=["knn", "random_forest", "logistic_regression", "svm"],
                        default=["knn", "random_forest", "logistic_regression", "svm"])
    parser.add_argument("--font-dir", type=Path, default=default_font_dir())
    args = parser.parse_args()
    if not (MODELS_PATH / ARTIFACT_NAME).is_file():
        train_verifier(args.font_dir)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    accepted = digit_acceptance_mask(X_test)

    models = {
        "knn": KNeighborsClassifier(n_neighbors=3),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=500),
        "svm": SVC(kernel="rbf", C=10, gamma="scale"),
    }

    rows = []
    for name, model in models.items():
        if name not in args.models:
            continue
        print(f"Training {name} on {len(X_train):,} rows...", flush=True)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name}: {accuracy:.4f}")
        save_model(model, name)
        rows.append(metric_row(name, y_test, predictions, accepted=accepted))

    write_results(rows, RESULTS_DIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
