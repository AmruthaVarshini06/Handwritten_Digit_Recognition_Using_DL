from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from tensorflow import keras
except ImportError:  # pragma: no cover
    keras = None

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = Path(os.environ.get("DIGIT_RECOGNITION_ASSETS_DIR", PROJECT_ROOT))
MODELS_DIR = ASSETS_ROOT / "models"
DATA_DIR = ASSETS_ROOT / "data"
RESULTS_DIR = ASSETS_ROOT / "results"

MODEL_ORDER = [
    "cnn",
    "mlp",
    "svm",
    "knn",
    "random_forest",
    "logistic_regression",
]


def _model_path(name: str) -> Path:
    if name == "cnn":
        return MODELS_DIR / "cnn_digit_model.keras"
    if name == "mlp":
        return MODELS_DIR / "mlp_digit_model.pkl"
    if name == "svm":
        return MODELS_DIR / "svm_digit_model.pkl"
    if name == "knn":
        return MODELS_DIR / "knn_digit_model.pkl"
    if name == "random_forest":
        return MODELS_DIR / "random_forest_digit_model.pkl"
    if name == "logistic_regression":
        return MODELS_DIR / "logistic_regression_digit_model.pkl"
    raise ValueError(f"Unsupported model name: {name}")


def available_models() -> tuple[str, ...]:
    return tuple(name for name in MODEL_ORDER if _model_path(name).is_file())


def load_best_model() -> Tuple[str, Any]:
    available = available_models()
    if available:
        name = available[0]
        return name, load_model(name)
    raise FileNotFoundError("No trained model files were found inside the models/ directory.")


def load_model(model_name: str) -> Any:
    model_name = model_name.lower().replace("-", "_")
    if model_name == "cnn":
        if keras is None:
            raise ImportError("TensorFlow is required to load the CNN model.")
        return keras.models.load_model(_model_path("cnn"))

    return joblib.load(_model_path(model_name))


def _as_2d_array(values: Iterable[float]) -> np.ndarray:
    try:
        arr = np.asarray(list(values), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pixel values must be numeric.") from exc
    if arr.size == 28 * 28:
        return arr.reshape(28, 28)
    raise ValueError("Input must contain exactly 784 pixel values or a 28x28 image.")


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    image = arr.astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("Pixel values must be finite numbers.")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Pixel values must be between 0 and 255.")
    if image.max() > 1.0:
        image = image / 255.0
    return image


def _preprocess_image_array(image_array: np.ndarray) -> np.ndarray:
    image = np.asarray(image_array)
    if image.ndim == 1:
        image = _as_2d_array(image)
    elif image.ndim == 2:
        if image.size != 28 * 28:
            raise ValueError("Input must contain exactly 784 pixel values or a 28x28 image.")
        image = image.reshape(28, 28)
    elif image.ndim == 3 and image.shape[:2] == (28, 28) and image.shape[-1] in (1, 3, 4):
        image = image[:, :, 0] if image.shape[-1] == 1 else np.mean(image[:, :, :3], axis=2)
    else:
        raise ValueError("Unsupported image shape. Expected (28, 28), (28, 28, 1/3/4), or (784,).")

    image = _normalize_image(image)
    return image


def _load_image_file(path: str | Path) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to load image files. Install it from requirements.txt.")

    image = Image.open(path).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    return _preprocess_image_array(image_array)


def _load_csv_file(path: str | Path) -> np.ndarray:
    try:
        df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV file is empty.") from exc

    if df.empty:
        raise ValueError("CSV file contains no data rows.")

    has_header = False
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError):
        has_header = True
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("CSV file is empty.") from exc
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("CSV values must be numeric.") from exc

    if len(df) != 1:
        raise ValueError("CSV upload must contain exactly one digit row.")

    values = df.iloc[0].to_numpy()
    if not has_header and values.size == 785:
        first_value = float(values[0])
        if first_value.is_integer() and 0 <= first_value <= 9:
            values = values[1:]
    return _preprocess_image_array(values)


def predict_from_array(values: Iterable[float] | np.ndarray, model_name: str | None = None) -> tuple[int, float | None]:
    if model_name is None:
        model_name, model = load_best_model()
    else:
        model_name = model_name.lower().replace("-", "_")
        model = load_model(model_name)

    processed = _preprocess_image_array(values)
    if model_name == "cnn":
        sample = processed.reshape(1, 28, 28, 1)
        probabilities = model.predict(sample, verbose=0)
        prediction = int(np.argmax(probabilities[0]))
        confidence = float(np.max(probabilities[0]))
        return prediction, confidence

    flat = processed.reshape(1, -1)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        if len(feature_names) != flat.shape[1]:
            raise ValueError("The selected model was trained with an incompatible number of features.")
        flat = pd.DataFrame(flat, columns=feature_names)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(flat)
        predicted_index = int(np.argmax(probabilities[0]))
        classes = getattr(model, "classes_", None)
        prediction = int(classes[predicted_index]) if classes is not None else predicted_index
        confidence = float(probabilities[0][predicted_index])
        return prediction, confidence

    prediction = int(model.predict(flat)[0])
    return prediction, None


def predict_from_file(file_path: str | Path, model_name: str | None = None) -> tuple[int, float | None]:
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        processed = _load_csv_file(file_path)
    else:
        processed = _load_image_file(file_path)
    return predict_from_array(processed, model_name=model_name)
