import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from digit_recognition.data import load_training_data


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "train.csv"
        self.data = pd.DataFrame(np.full((20, 784), 255), columns=[f"pixel{i}" for i in range(784)])
        self.data["label"] = list(range(10)) * 2

    def test_label_position_does_not_change_features(self):
        self.data.to_csv(self.path, index=False)
        features, labels = load_training_data(self.path)
        self.assertEqual(features.shape, (20, 784))
        self.assertEqual(labels.tolist(), list(range(10)) * 2)
        np.testing.assert_array_equal(features, np.ones((20, 784)))

    def test_rejects_invalid_schema_pixels_and_labels(self):
        for field, value in (("pixel0", np.nan), ("pixel0", 256), ("label", 1.5), ("label", 10)):
            with self.subTest(field=field, value=value):
                data = self.data.astype(float)
                data.loc[0, field] = value
                data.to_csv(self.path, index=False)
                with self.assertRaises(ValueError):
                    load_training_data(self.path)
        self.data.drop(columns="pixel0").to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "must contain"):
            load_training_data(self.path)
