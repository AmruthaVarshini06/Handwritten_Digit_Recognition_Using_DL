# Handwritten Digit Recognition

This project builds and compares several machine learning models for handwritten digit classification on the MNIST-style dataset stored in `data/train.csv` and `data/test.csv`.

## Project structure

- `data/` — training and test datasets
- `models/` — trained model files
- `notebooks/` — exploratory training notebooks
- `results/` — model comparison charts and CSV metrics
- `digit_recognition/` — reusable Python utilities for loading and predicting with the trained models
- `app/` — Streamlit app for live prediction from uploaded images or CSV data

## Features

- Trains and evaluates several classifiers
- Saves trained models in `models/`
- Compares model accuracy and error rate
- Exposes a Streamlit web app for image and CSV prediction
- Shows a leaderboard of model performance from `results/model_comparison.csv`
- Includes a lightweight command-line prediction workflow

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   . .venv/bin/activate  # Linux/macOS
   # or .venv\Scripts\activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the app

```bash
streamlit run app/app.py
```

## Predict from a file using Python

```bash
python -m digit_recognition.predict --image path/to/sample.png
```

You can also pass a CSV file containing 784 pixel values:

```bash
python -m digit_recognition.predict --csv path/to/sample.csv
```

## Model files

Available model artifacts include:

- `models/cnn_digit_model.keras`
- `models/mlp_digit_model.pkl`
- `models/svm_digit_model.pkl`
- `models/knn_digit_model.pkl`
- `models/random_forest_digit_model.pkl`
- `models/logistic_regression_digit_model.pkl`

## Notes

The notebooks in `notebooks/` are the original training pipeline. This project now adds a clean Python entry point and a lightweight application interface so the models can be reused outside notebook execution.
