"""Train and audit the digit verifier using disjoint digit and font splits."""
import argparse
import hashlib
import json
from pathlib import Path
import string
import sys

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from digit_recognition.data import load_training_data
from digit_recognition.model_utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, _fit_digit_image
from digit_recognition.rejection import ARTIFACT_NAME, calibrated_threshold, verifier_features


def default_font_dir():
    candidates = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/System/Library/Fonts")]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def font_samples(fonts, seed):
    """Generate text negatives and printed digit positives with identical rendering."""
    random = np.random.default_rng(seed)
    texts = list(string.ascii_letters + "@#$%&*+=?!<>/") + ["12", "23", "345", "2026"]
    positives, negatives = [], []
    for font_path in fonts:
        for text in [*string.digits, *texts]:
            for _ in range(8):
                font = ImageFont.truetype(str(font_path), int(random.integers(40, 65)))
                image = Image.new("L", (220, 110), 0)
                ImageDraw.Draw(image).text((12, 8), text, font=font,
                                          fill=int(random.integers(180, 256)), stroke_width=int(random.integers(0, 2)))
                image = image.rotate(float(random.uniform(-14, 14)), resample=Image.Resampling.BICUBIC)
                pixels = np.asarray(_fit_digit_image(image), dtype=np.float32).reshape(784) / 255
                (positives if len(text) == 1 and text in string.digits else negatives).append(pixels)
    return np.asarray(positives), np.asarray(negatives)


def train(font_dir: Path):
    preferred = [font_dir / name for name in (
        "calibri.ttf", "comic.ttf", "segoeui.ttf", "tahoma.ttf", "times.ttf", "verdana.ttf",
        "consola.ttf", "arial.ttf",
    )]
    fonts = [path for path in preferred if path.is_file()]
    if len(fonts) < 4:
        fonts = sorted(font_dir.rglob("*.ttf"))
    if len(fonts) < 4:
        raise ValueError("Provide a font directory containing at least four TTF fonts.")
    training_fonts, validation_fonts, test_fonts = fonts[:-2], fonts[-2:-1], fonts[-1:]
    print("Generating text examples with disjoint training/validation/test fonts...", flush=True)
    printed_train, negative_train = font_samples(training_fonts, 42)
    printed_validation, negative_validation = font_samples(validation_fonts, 43)
    printed_test, negative_test = font_samples(test_fonts, 44)
    assert len(printed_train) == len(training_fonts) * 10 * 8
    X, y = load_training_data(DATA_DIR / "train.csv")
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_fit, X_calibration = train_test_split(X_train, test_size=0.2, random_state=43, stratify=y_train)
    positive_fit = np.concatenate([X_fit.to_numpy(), printed_train])
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features=0.5,
                                 class_weight="balanced", random_state=42, n_jobs=2)
    train_images = np.concatenate([positive_fit, negative_train])
    targets = np.concatenate([np.ones(len(positive_fit), dtype=int), np.zeros(len(negative_train), dtype=int)])
    print(f"Training verifier on {len(positive_fit):,} digits and {len(negative_train):,} text negatives...", flush=True)
    model.fit(verifier_features(train_images), targets)
    calibration_scores = model.predict_proba(verifier_features(X_calibration.to_numpy()))[:, 1]
    printed_calibration_scores = model.predict_proba(verifier_features(printed_validation))[:, 1]
    threshold = min(
        calibrated_threshold(calibration_scores, 0.99),
        calibrated_threshold(printed_calibration_scores, 0.95),
    )

    def acceptance(images):
        probabilities = model.predict_proba(verifier_features(images))[:, 1]
        return float(np.mean(probabilities >= threshold) * 100)

    report = {
        "threshold": threshold, "target_calibration_digit_acceptance_percent": 99,
        "target_validation_printed_digit_acceptance_percent": 95,
        "calibration_digit_acceptance_percent": acceptance(X_calibration.to_numpy()),
        "test_digit_acceptance_percent": acceptance(X_test.to_numpy()),
        "validation_non_digit_rejection_percent": 100 - acceptance(negative_validation),
        "validation_printed_digit_acceptance_percent": acceptance(printed_validation),
        "test_non_digit_rejection_percent": 100 - acceptance(negative_test),
        "test_printed_digit_acceptance_percent": acceptance(printed_test),
        "train_fonts": [path.name for path in training_fonts],
        "validation_fonts": [path.name for path in validation_fonts],
        "test_fonts": [path.name for path in test_fonts],
        "test_digit_count": len(X_test), "test_non_digit_count": len(negative_test),
        "dataset_sha256": hashlib.sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest(),
        "limitation": "Text negatives are synthetic fonts, not representative handwritten letters, photos, or all symbols.",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "model": model, "threshold": threshold, "report": report}, MODELS_DIR / ARTIFACT_NAME)
    (RESULTS_DIR / "rejection_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, default=default_font_dir())
    arguments = parser.parse_args()
    train(arguments.font_dir)
