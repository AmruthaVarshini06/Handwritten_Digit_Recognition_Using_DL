from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None

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

MIN_DIGIT_CONFIDENCE = 0.0
UNRECOGNIZED_INPUT = (
    "Input not recognized as a clear single digit (0-9). "
    "Please upload one clearly written number, without letters or symbols."
)


class UnrecognizedDigitError(ValueError):
    """The input cannot be accepted as a confident digit prediction."""


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
    path = _model_path(model_name)
    return _load_model_cached(model_name, str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=6)
def _load_model_cached(model_name: str, path: str, modified: int) -> Any:
    """Reuse artifacts until their modification time changes."""
    if model_name == "cnn":
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("TensorFlow is required to load the CNN model.") from exc
        return keras.models.load_model(path)

    return joblib.load(path)


def _as_2d_array(values: Iterable[float]) -> np.ndarray:
    try:
        arr = np.asarray(list(values))
    except (TypeError, ValueError) as exc:
        raise ValueError("Pixel values must be numeric.") from exc
    if arr.size == 28 * 28:
        return arr.reshape(28, 28)
    raise ValueError("Input must contain exactly 784 pixel values or a 28x28 image.")


def _normalize_image(arr: np.ndarray, *, byte_scale: bool = False) -> np.ndarray:
    image = arr.astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("Pixel values must be finite numbers.")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Pixel values must be between 0 and 255.")
    if byte_scale or image.max() > 1.0:
        image = image / 255.0
    return image


def _preprocess_image_array(image_array: np.ndarray) -> np.ndarray:
    if not isinstance(image_array, (np.ndarray, list, tuple)):
        image_array = list(image_array)
    image = np.asarray(image_array)
    byte_scale = np.issubdtype(image.dtype, np.integer)
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

    image = _normalize_image(image, byte_scale=byte_scale)
    return image


def _load_image_file(path: str | Path) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to load image files. Install it from requirements.txt.")

    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert("RGBA")
        rgba = np.asarray(source)
        visible = rgba[:, :, 3] > 0
        # Contrast the canvas with the visible ink, rather than losing white ink
        # against an assumed white background. Opaque images are unchanged.
        ink = np.mean(rgba[:, :, :3], axis=2)[visible]
        canvas_color = "black" if ink.size and np.median(ink) > 127 else "white"
        background = Image.new("RGBA", source.size, canvas_color)
        image = Image.alpha_composite(background, source).convert("L")
    pixels = np.asarray(image)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if np.median(border) > 127:
        image = ImageOps.invert(image)
    minimum, maximum = image.getextrema()
    if minimum == maximum:
        raise ValueError("No visible digit found. Upload a clear image of one digit.")
    # Dataset images already have the expected margin. Larger uploads need one.
    if image.size != (28, 28):
        image = _fit_digit_image(image)
    image_array = np.asarray(image)
    return _preprocess_image_array(image_array)


def _fit_digit_image(image):
    """Fit foreground strokes into a centered 20 x 20 area, including upscaling."""
    pixels = np.asarray(image)
    rows, columns = np.where(pixels > max(20, float(pixels.max()) * 0.2))
    if not len(rows):
        raise ValueError("No visible digit found. Upload a clear image of one digit.")
    image = image.crop((columns.min(), rows.min(), columns.max() + 1, rows.max() + 1))
    scale = 20 / max(image.size)
    size = tuple(max(1, round(dimension * scale)) for dimension in image.size)
    image = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return canvas


def _load_csv_file(path: str | Path) -> np.ndarray:
    try:
        # A header plus two rows is enough to reject a dataset upload.
        df = pd.read_csv(path, header=None, nrows=3)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV file is empty.") from exc

    if df.empty:
        raise ValueError("CSV file contains no data rows.")

    header = df.iloc[0].astype(str).tolist()
    pixel_columns = [f"pixel{i}" for i in range(784)]
    has_header = header in (pixel_columns, ["label", *pixel_columns])
    if has_header:
        df = df.iloc[1:]
        if header[0] == "label":
            df = df.iloc[:, 1:]
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV values must be numeric; expected pixel0 through pixel783 headers.") from exc

    if len(df) != 1:
        raise ValueError("CSV upload must contain exactly one digit row.")

    values = df.iloc[0].to_numpy()
    if not has_header and values.size == 785:
        first_value = float(values[0])
        if first_value.is_integer() and 0 <= first_value <= 9:
            values = values[1:]
    return _preprocess_image_array(values)


def predict_from_array(
    values: Iterable[float] | np.ndarray,
    model_name: str | None = None,
    *, min_confidence: float = MIN_DIGIT_CONFIDENCE,
    reject_non_digits: bool = True,
) -> tuple[int, float | None]:
    """Check digit validity separately from the selected classifier's confidence."""
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("Minimum confidence must be between 0 and 1.")
    processed = _preprocess_image_array(values)
    if reject_non_digits and float(np.ptp(processed)) < 0.05:
        raise UnrecognizedDigitError(UNRECOGNIZED_INPUT)
    if reject_non_digits:
        from .rejection import digit_acceptance_mask
        if not digit_acceptance_mask(processed.reshape(1, -1))[0]:
            raise UnrecognizedDigitError(UNRECOGNIZED_INPUT)
    if model_name is None:
        model_name, model = load_best_model()
    else:
        model_name = model_name.lower().replace("-", "_")
        model = load_model(model_name)

    if model_name == "cnn":
        sample = processed.reshape(1, 28, 28, 1)
        probabilities = model.predict(sample, verbose=0)
        prediction = int(np.argmax(probabilities[0]))
        confidence = float(np.max(probabilities[0]))
        _require_confident_digit(prediction, confidence, min_confidence)
        return prediction, confidence

    flat = processed.reshape(1, -1)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        if len(feature_names) != flat.shape[1]:
            raise ValueError("The selected model was trained with an incompatible number of features.")
        flat = pd.DataFrame(flat, columns=feature_names)

    prediction = int(model.predict(flat)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(flat)
        classes = getattr(model, "classes_", None)
        if classes is None:
            predicted_index = int(np.argmax(probabilities[0]))
            prediction = predicted_index
        else:
            predicted_index = int(np.flatnonzero(classes == prediction)[0])
        confidence = float(probabilities[0][predicted_index])
        _require_confident_digit(prediction, confidence, min_confidence)
        return prediction, confidence

    if min_confidence > 0:
        raise ValueError("The selected model has no probability estimates for a confidence threshold.")
    _require_confident_digit(prediction, None, 0)
    return prediction, None


def _require_confident_digit(prediction: int, confidence: float | None, threshold: float) -> None:
    if prediction not in range(10):
        raise UnrecognizedDigitError(UNRECOGNIZED_INPUT)
    if confidence is not None and (
        not np.isfinite(confidence) or not 0 <= confidence <= 1 or confidence < threshold
    ):
        raise UnrecognizedDigitError(UNRECOGNIZED_INPUT)


def predict_from_file(
    file_path: str | Path, model_name: str | None = None,
    *, min_confidence: float = MIN_DIGIT_CONFIDENCE,
    reject_non_digits: bool = True,
) -> tuple[int, float | None]:
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        processed = _load_csv_file(file_path)
    else:
        processed = _load_image_file(file_path)
    return predict_from_array(
        processed, model_name=model_name, min_confidence=min_confidence,
        reject_non_digits=reject_non_digits,
    )
