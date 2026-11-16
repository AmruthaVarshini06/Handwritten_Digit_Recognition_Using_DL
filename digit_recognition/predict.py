from __future__ import annotations

import argparse

from .model_utils import predict_from_array, predict_from_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from a CSV or image file.")
    parser.add_argument("--image", type=str, help="Path to a PNG/JPG digit image.")
    parser.add_argument("--csv", type=str, help="Path to a CSV file containing 784 pixel values.")
    parser.add_argument("--model", type=str, default=None, help="Optional model name: cnn, mlp, svm, knn, random_forest, logistic_regression")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.image and not args.csv:
        raise SystemExit("Provide either --image or --csv.")

    if args.image:
        prediction, confidence = predict_from_file(args.image, model_name=args.model)
    else:
        prediction, confidence = predict_from_file(args.csv, model_name=args.model)

    if confidence is None:
        print(f"Predicted digit: {prediction}")
    else:
        print(f"Predicted digit: {prediction} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
