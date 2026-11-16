from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from digit_recognition import model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_white_and_black_ink_on_transparency_prepare_equally(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = []
            for color in (0, 255):
                pixels = np.zeros((60, 60, 4), dtype=np.uint8)
                pixels[10:50, 27:33] = [color, color, color, 255]
                path = Path(directory) / f"ink_{color}.png"
                model_utils.Image.fromarray(pixels).save(path)
                prepared.append(model_utils._load_image_file(path))
            np.testing.assert_array_equal(*prepared)
            self.assertGreater(float(prepared[0].max()), 0.9)

    def test_small_digits_are_enlarged_to_the_training_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.zeros((40, 40), dtype=np.uint8)
            pixels[15:25, 19:21] = 255
            path = Path(directory) / "small.png"
            model_utils.Image.fromarray(pixels).save(path)
            rows, columns = np.where(model_utils._load_image_file(path) > 0.2)
            self.assertEqual(int(rows.max() - rows.min() + 1), 20)
            self.assertEqual(int(columns.max() - columns.min() + 1), 4)

    def test_light_and_dark_backgrounds_prepare_the_same_digit(self):
        pixels = np.zeros((80, 60), dtype=np.uint8)
        pixels[10:70, 27:33] = 255
        with tempfile.TemporaryDirectory() as directory:
            dark = Path(directory) / "dark.png"
            light = Path(directory) / "light.png"
            model_utils.Image.fromarray(pixels).save(dark)
            model_utils.Image.fromarray(255 - pixels).save(light)
            prepared = model_utils._load_image_file(dark)
            np.testing.assert_array_equal(prepared, model_utils._load_image_file(light))
            self.assertEqual(prepared.shape, (28, 28))
            self.assertEqual(float(prepared[0].max()), 0)
            self.assertGreater(float(prepared.max()), 0.9)

    def test_transparent_black_strokes_remain_visible(self):
        pixels = np.zeros((60, 60, 4), dtype=np.uint8)
        pixels[10:50, 27:33, 3] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transparent.png"
            model_utils.Image.fromarray(pixels).save(path)
            self.assertGreater(float(model_utils._load_image_file(path).max()), 0.9)

    def test_blank_uploads_have_a_useful_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            for value in (0, 128, 255):
                model_utils.Image.new("L", (28, 28), value).save(path)
                with self.assertRaisesRegex(ValueError, "No visible digit"):
                    model_utils._load_image_file(path)

    def test_probability_does_not_override_classifier_prediction(self):
        class Classifier:
            classes_ = np.array([2, 7])
            def predict(self, values):
                return [2]
            def predict_proba(self, values):
                return np.array([[0.4, 0.6]])
        with patch.object(model_utils, "load_model", return_value=Classifier()):
            self.assertEqual(model_utils.predict_from_array(np.zeros(784), "svm", min_confidence=0, reject_non_digits=False), (2, 0.4))

    def test_cached_model_reloads_when_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knn_digit_model.pkl"
            model_utils.joblib.dump({"version": 1}, path)
            with patch.object(model_utils, "MODELS_DIR", Path(directory)):
                first = model_utils.load_model("knn")
                self.assertIs(first, model_utils.load_model("knn"))
                model_utils.joblib.dump({"version": 222}, path)
                self.assertEqual(model_utils.load_model("knn"), {"version": 222})

    def test_dark_byte_images_keep_their_intensity(self) -> None:
        for shape in ((784,), (28, 28), (28, 28, 3), (28, 28, 4)):
            with self.subTest(shape=shape):
                result = model_utils._preprocess_image_array(np.ones(shape, dtype=np.uint8))
                np.testing.assert_allclose(result, np.full((28, 28), 1 / 255))
        normalized = np.ones((28, 28), dtype=np.float32)
        np.testing.assert_array_equal(model_utils._preprocess_image_array(normalized), normalized)

    def test_dark_png_is_not_normalized_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dark.png"
            pixels = np.zeros((28, 28), dtype=np.uint8)
            pixels[10:20, 13] = 1
            model_utils.Image.fromarray(pixels).save(path)
            class Classifier:
                def predict(self, values):
                    np.testing.assert_allclose(values, pixels.reshape(1, 784) / 255)
                    return [1]
            with patch.object(model_utils, "load_model", return_value=Classifier()):
                self.assertEqual(model_utils.predict_from_file(path, "knn", min_confidence=0, reject_non_digits=False), (1, None))

    def test_rejects_malformed_row_instead_of_treating_it_as_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.csv"
            path.write_text(','.join(['oops'] + ['0'] * 783) + '\n' + ','.join(['0'] * 784))
            with self.assertRaisesRegex(ValueError, "must be numeric"):
                model_utils._load_csv_file(path)

    def test_reads_pixel_headers_without_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "digit.csv"
            path.write_text(','.join(f'pixel{i}' for i in range(784)) + '\n' + ','.join(['1'] * 784))
            np.testing.assert_allclose(model_utils._load_csv_file(path), np.full((28, 28), 1 / 255))

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

            def predict(self, values):
                return [5]

            def predict_proba(self, values: np.ndarray) -> np.ndarray:
                return np.array([[0.1, 0.7, 0.2]])

        with patch.object(model_utils, "load_model", return_value=FakeClassifier()):
            prediction, confidence = model_utils.predict_from_array(
                np.zeros(784),
                model_name="svm", min_confidence=0, reject_non_digits=False,
            )

        self.assertEqual(prediction, 5)
        self.assertAlmostEqual(confidence, 0.7)

    def test_prediction_preserves_trained_feature_names(self) -> None:
        feature_names = np.array([f"pixel{i}" for i in range(784)])

        class FakeClassifier:
            classes_ = np.array([0, 1])
            feature_names_in_ = feature_names

            def predict(self, values):
                self.assert_column_names(values)
                return [0]

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
                model_name="logistic_regression", min_confidence=0, reject_non_digits=False,
            )

        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()
