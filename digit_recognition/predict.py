from __future__ import annotations

import argparse

from .model_utils import MODEL_ORDER, predict_from_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from a CSV or image file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=str, help="Path to a PNG/JPG digit image.")
    source.add_argument("--csv", type=str, help="Path to a CSV file containing 784 pixel values.")
    parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        prediction, confidence = predict_from_file(args.image or args.csv, model_name=args.model)
    except (ValueError, OSError, ImportError) as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc

    if confidence is None:
        print(f"Predicted digit: {prediction}")
    else:
        print(f"Predicted digit: {prediction} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
