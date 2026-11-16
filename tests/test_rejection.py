import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from digit_recognition import model_utils as utils
from digit_recognition.rejection import calibrated_threshold, digit_acceptance_mask, verifier_features


@unittest.skipUnless(
    (utils.MODELS_DIR / "digit_verifier.pkl").is_file() and Path("C:/Windows/Fonts/arial.ttf").is_file(),
    "Integration check requires a trained verifier and the Arial font.",
)
class RejectionIntegrationTests(unittest.TestCase):
    def test_previously_accepted_letters_symbols_and_multiple_digits_are_rejected(self):
        import tempfile
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 64)
        with tempfile.TemporaryDirectory() as directory:
            for text in ("A", "B", "H", "M", "+", "12", "345"):
                with self.subTest(text=text):
                    image = Image.new("RGB", (180, 100), "white")
                    ImageDraw.Draw(image).text((15, 5), text, font=font, fill="black")
                    path = Path(directory) / "input.png"
                    image.save(path)
                    pixels = utils._load_image_file(path)
                    self.assertFalse(bool(digit_acceptance_mask(pixels.reshape(1, -1))[0]))


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.zeros((28, 28), dtype=np.float32)
        self.pixels[5:23, 13:15] = 1
        guard = patch("digit_recognition.rejection.digit_acceptance_mask", return_value=np.array([True]))
        self.guard = guard.start()
        self.addCleanup(guard.stop)

    def classifier(self, probability, digit=1):
        class Classifier:
            classes_ = np.array([0, 1])
            def predict(self, values):
                return [digit]
            def predict_proba(self, values):
                return np.array([[1 - probability, probability]])
        return Classifier()

    def test_blank_array_is_rejected_before_loading_a_model(self):
        with patch.object(utils, "load_model") as load:
            with self.assertRaises(utils.UnrecognizedDigitError):
                utils.predict_from_array(np.zeros(784), "knn")
            load.assert_not_called()

    def test_non_digit_is_rejected_even_with_confident_digit_classifier(self):
        self.guard.return_value = np.array([False])
        with patch.object(utils, "load_model") as load:
            with self.assertRaises(utils.UnrecognizedDigitError):
                utils.predict_from_array(self.pixels, "knn")
            load.assert_not_called()

    def test_valid_digit_is_not_rejected_by_old_arbitrary_threshold(self):
        with patch.object(utils, "load_model", return_value=self.classifier(0.6)):
            self.assertEqual(utils.predict_from_array(self.pixels, "knn"), (1, 0.6))

    def test_optional_confidence_threshold_is_still_respected(self):
        with patch.object(utils, "load_model", return_value=self.classifier(0.79)):
            with self.assertRaises(utils.UnrecognizedDigitError):
                utils.predict_from_array(self.pixels, "knn", min_confidence=0.8)

    def test_invalid_model_probabilities_are_rejected(self):
        for probability in (np.nan, np.inf, 1.1):
            with self.subTest(probability=probability):
                with patch.object(utils, "load_model", return_value=self.classifier(probability)):
                    with self.assertRaises(utils.UnrecognizedDigitError):
                        utils.predict_from_array(self.pixels, "knn")

    def test_svm_does_not_depend_on_another_digit_classifier(self):
        class SVM:
            def predict(self, values):
                return [1]
        with patch.object(utils, "load_model", return_value=SVM()) as load:
            self.assertEqual(utils.predict_from_array(self.pixels, "svm"), (1, None))
            load.assert_called_once_with("svm")

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-1, 2, np.nan):
            with self.assertRaises(ValueError):
                utils.predict_from_array(self.pixels, "knn", min_confidence=threshold)

    def test_raw_diagnostics_can_explicitly_bypass_the_verifier(self):
        with patch.object(utils, "load_model", return_value=self.classifier(0.6)):
            utils.predict_from_array(self.pixels, "knn", reject_non_digits=False)
            self.guard.assert_not_called()


class VerifierTests(unittest.TestCase):
    def test_calibration_meets_target_on_calibration_digits(self):
        scores = np.linspace(0.1, 0.99, 100)
        threshold = calibrated_threshold(scores, 0.99)
        self.assertGreaterEqual(float(np.mean(scores >= threshold)), 0.99)

    def test_features_preserve_normalized_intensity(self):
        values = verifier_features(np.ones((2, 784), dtype=np.float32))
        self.assertEqual(values.shape, (2, 196))
        np.testing.assert_array_equal(values, 1)
        with self.assertRaises(ValueError):
            verifier_features(np.full((1, 784), 255))

    def test_batch_mask_uses_the_calibrated_threshold_and_blank_check(self):
        class Verifier:
            def predict_proba(self, values):
                return np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        images = np.zeros((3, 784), dtype=np.float32)
        images[:2, 10:50] = 1
        with patch("digit_recognition.rejection.load_verifier", return_value={"model": Verifier(), "threshold": 0.7}):
            np.testing.assert_array_equal(digit_acceptance_mask(images), [True, False, False])

    def test_missing_verifier_has_a_setup_error(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(utils, "MODELS_DIR", Path(directory)):
                with self.assertRaisesRegex(FileNotFoundError, "train_rejection.py"):
                    digit_acceptance_mask(np.zeros((1, 784)))
