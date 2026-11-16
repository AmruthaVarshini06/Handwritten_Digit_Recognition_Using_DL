from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(exist_ok=True)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_PATH)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    return X, y


def save_model(model, name: str) -> None:
    output_path = MODELS_PATH / f"{name}_digit_model.pkl"
    joblib.dump(model, output_path)


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "knn": KNeighborsClassifier(n_neighbors=3),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=500),
        "svm": SVC(kernel="rbf", C=10, gamma="scale"),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name}: {accuracy:.4f}")
        save_model(model, name)


if __name__ == "__main__":
    main()
