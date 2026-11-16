import tempfile
import unittest
from pathlib import Path

from digit_recognition.evaluation import metric_row, write_results


class EvaluationTests(unittest.TestCase):
    def test_acceptance_metrics_distinguish_rejection_from_wrong_predictions(self):
        row = metric_row("knn", [0, 1, 1, 0], [0, 1, 0, 0], [True, True, False, False])
        self.assertEqual(row["Accuracy"], 75)
        self.assertEqual(row["Acceptance Rate"], 50)
        self.assertEqual(row["Rejection Rate"], 50)
        self.assertEqual(row["Accepted Accuracy"], 100)

    def test_no_accepted_predictions_has_no_fabricated_accuracy(self):
        row = metric_row("knn", [0, 1], [0, 1], [False, False])
        self.assertEqual(row["Acceptance Rate"], 0)
        self.assertIsNone(row["Accepted Accuracy"])

    def test_metrics_come_from_predictions_and_replace_old_results(self):
        row = metric_row("knn", [0, 1, 1, 0], [0, 1, 0, 0])
        self.assertEqual(row["Accuracy"], 75)
        self.assertEqual(row["Error Rate"], 25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_results([metric_row("svm", [0], [0])], path)
            results = write_results([row], path)
            self.assertEqual(results["Algorithm"].tolist(), ["KNN"])
            self.assertNotIn("SVM", path.read_text())
