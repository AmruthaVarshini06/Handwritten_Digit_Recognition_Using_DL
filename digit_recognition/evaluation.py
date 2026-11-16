"""Compute leaderboard percentages from a common holdout split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from .data import load_training_data

MODEL_LABELS = {
    "cnn": "CNN", "mlp": "MLP", "svm": "SVM", "knn": "KNN",
    "random_forest": "Random Forest", "logistic_regression": "Logistic Regression",
}
RESULT_COLUMNS = [
    "Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "Error Rate",
    "Acceptance Rate", "Rejection Rate", "Accepted Accuracy",
]


def metric_row(name, actual, predicted, accepted=None):
    accuracy = accuracy_score(actual, predicted)
    row = dict(zip(RESULT_COLUMNS, [
        MODEL_LABELS[name], accuracy * 100,
        precision_score(actual, predicted, average="weighted", zero_division=0) * 100,
        recall_score(actual, predicted, average="weighted", zero_division=0) * 100,
        f1_score(actual, predicted, average="weighted", zero_division=0) * 100,
        (1 - accuracy) * 100,
    ]))
    if accepted is not None:
        accepted = np.asarray(accepted, dtype=bool)
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        if accepted.shape != actual.shape:
            raise ValueError("Acceptance mask must match the evaluation rows.")
        row["Acceptance Rate"] = float(accepted.mean() * 100)
        row["Rejection Rate"] = float((~accepted).mean() * 100)
        row["Accepted Accuracy"] = (
            float(accuracy_score(actual[accepted], predicted[accepted]) * 100)
            if accepted.any() else None
        )
    return row


def write_results(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS).round(2)
    results.to_csv(output_path, index=False)
    return results


def evaluate_saved_models():
    # Artifacts must use the repository's random_state=42 stratified 80/20 split.
    from .model_utils import DATA_DIR, RESULTS_DIR, available_models, load_model

    names = available_models()
    if not names:
        raise FileNotFoundError("Train models before generating a leaderboard.")
    features, labels = load_training_data(DATA_DIR / "train.csv")
    _, X_test, _, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    rows = []
    from .rejection import digit_acceptance_mask
    accepted = digit_acceptance_mask(X_test.to_numpy())
    for name in names:
        print(f"Evaluating {MODEL_LABELS[name]}...", flush=True)
        model = load_model(name)
        if name == "cnn":
            sample = X_test.to_numpy().reshape(-1, 28, 28, 1)
            predicted = np.argmax(model.predict(sample, verbose=0), axis=1)
        else:
            sample = X_test if hasattr(model, "feature_names_in_") else X_test.to_numpy()
            predicted = model.predict(sample)
        rows.append(metric_row(name, y_test, predicted, accepted=accepted))
    return write_results(rows, RESULTS_DIR / "model_comparison.csv")


if __name__ == "__main__":
    print(evaluate_saved_models().to_string(index=False))
