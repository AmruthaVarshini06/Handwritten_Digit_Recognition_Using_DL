from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from digit_recognition import model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_preprocesses_flat_pixels_and_normalizes_them(self) -> None:
        pixels = np.arange(784, dtype=np.float32) % 256

        result = model_utils._preprocess_image_array(pixels)

        np.testing.assert_allclose(result, pixels.reshape(28, 28) / 255.0)

    def test_preprocesses_rgb_images_and_rejects_wrong_dimensions(self) -> None:
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]

        result = model_utils._preprocess_image_array(image)
        rgba_result = model_utils._preprocess_image_array(
            np.concatenate([image, np.full((28, 28, 1), 255, dtype=np.uint8)], axis=2)
        )

        self.assertAlmostEqual(float(result[0, 0]), 1.0 / 3.0)
        np.testing.assert_allclose(rgba_result, result)
        with self.assertRaisesRegex(ValueError, "Unsupported image shape"):
            model_utils._preprocess_image_array(np.zeros((27, 28, 3), dtype=np.uint8))

    def test_rejects_non_finite_and_out_of_range_pixels(self) -> None:
        invalid_images = (
            np.full((28, 28), np.nan),
            np.full((28, 28), -1),
            np.full((28, 28), 256),
        )

        for image in invalid_images:
            with self.subTest(value=image.flat[0]):
                with self.assertRaises(ValueError):
                    model_utils._preprocess_image_array(image)

    def test_reads_headerless_csv(self) -> None:
        pixels = np.arange(784, dtype=np.int32) % 256
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "digit.csv"
            csv_path.write_text(",".join(map(str, pixels)), encoding="utf-8")

            result = model_utils._load_csv_file(csv_path)

        np.testing.assert_allclose(result, pixels.reshape(28, 28) / 255.0)

    def test_reads_headerless_csv_with_label(self) -> None:
        pixels = np.arange(784, dtype=np.int32) % 256
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "digit.csv"
            csv_path.write_text(f"7,{','.join(map(str, pixels))}", encoding="utf-8")

            result = model_utils._load_csv_file(csv_path)

        np.testing.assert_allclose(result, pixels.reshape(28, 28) / 255.0)

    def test_reads_headered_csv_with_label(self) -> None:
        pixels = np.arange(784, dtype=np.int32) % 256
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "digit.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["label", *(f"pixel{i}" for i in range(784))])
                writer.writerow([7, *pixels])

            result = model_utils._load_csv_file(csv_path)

        np.testing.assert_allclose(result, pixels.reshape(28, 28) / 255.0)

    def test_rejects_csv_with_more_than_one_row(self) -> None:
        pixels = ",".join(["0"] * 784)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "digits.csv"
            csv_path.write_text(f"{pixels}\n{pixels}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly one digit row"):
                model_utils._load_csv_file(csv_path)

    def test_available_models_lists_only_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            (models_dir / "svm_digit_model.pkl").touch()
            (models_dir / "logistic_regression_digit_model.pkl").touch()

            with patch.object(model_utils, "MODELS_DIR", models_dir):
                result = model_utils.available_models()

        self.assertEqual(result, ("svm", "logistic_regression"))

    def test_prediction_uses_classifier_class_label(self) -> None:
        class FakeClassifier:
            classes_ = np.array([3, 5, 8])

            def predict_proba(self, values: np.ndarray) -> np.ndarray:
                return np.array([[0.1, 0.7, 0.2]])

        with patch.object(model_utils, "load_model", return_value=FakeClassifier()):
            prediction, confidence = model_utils.predict_from_array(
                np.zeros(784),
                model_name="svm",
            )

        self.assertEqual(prediction, 5)
        self.assertAlmostEqual(confidence, 0.7)

    def test_prediction_preserves_trained_feature_names(self) -> None:
        feature_names = np.array([f"pixel{i}" for i in range(784)])

        class FakeClassifier:
            classes_ = np.array([0, 1])
            feature_names_in_ = feature_names

            def predict_proba(self, values: object) -> np.ndarray:
                self.assert_column_names(values)
                return np.array([[0.8, 0.2]])

            @staticmethod
            def assert_column_names(values: object) -> None:
                if not hasattr(values, "columns") or list(values.columns) != feature_names.tolist():
                    raise AssertionError("Trained feature names were not preserved.")

        with patch.object(model_utils, "load_model", return_value=FakeClassifier()):
            prediction, _ = model_utils.predict_from_array(
                np.zeros(784),
                model_name="logistic_regression",
            )

        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()
