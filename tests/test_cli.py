import unittest
from unittest.mock import patch

from digit_recognition.predict import _parse_args, main


class CommandLineTests(unittest.TestCase):
    def test_requires_exactly_one_input(self):
        for arguments in ([], ["--image", "digit.png", "--csv", "digit.csv"]):
            with self.subTest(arguments=arguments), patch("sys.argv", ["predict", *arguments]):
                with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                    _parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_missing_file_returns_readable_error(self):
        with patch("sys.argv", ["predict", "--image", "missing.png"]):
            with self.assertRaisesRegex(SystemExit, "Prediction failed"):
                main()
