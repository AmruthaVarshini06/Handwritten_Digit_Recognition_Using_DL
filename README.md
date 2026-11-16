# Handwritten Digit Recognition Using Deep Learning

This project compares machine-learning and deep-learning models that classify handwritten digits from 28 x 28 grayscale images. It includes a Streamlit app, a command-line prediction tool, model utilities, and Jupyter notebooks.

## Dataset

The `train.csv` and `test.csv` files are available as assets in the GitHub release **[Dataset Uploaded (v1.0)](https://github.com/AmruthaVarshini06/Handwritten_Digit_Recognition_Using_DL/releases/tag/v1.0)**. Download both files from the release and place them in the project's `data/` folder.

You can download the files directly with PowerShell:

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://github.com/AmruthaVarshini06/Handwritten_Digit_Recognition_Using_DL/releases/download/v1.0/train.csv" -OutFile "data/train.csv"
Invoke-WebRequest -Uri "https://github.com/AmruthaVarshini06/Handwritten_Digit_Recognition_Using_DL/releases/download/v1.0/test.csv" -OutFile "data/test.csv"
```

The training file contains a `label` column and 784 pixel features. The test file contains pixel features. For app or command-line prediction, provide a CSV with exactly one row of 784 pixel values; a multi-row dataset is not a single-digit prediction input.

The dataset is based on [Handwritten Digit Recognition on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition).

## Project structure

- `app/` — Streamlit interface for image and CSV predictions
- `data/` — locally downloaded training and test datasets
- `digit_recognition/` — model loading, image preprocessing, and prediction utilities
- `models/` — locally generated or supplied model artifacts
- `notebooks/` — data preparation, model training, and comparison notebooks
- `results/` — model comparison metrics and charts
- `scripts/train_models.py` — trains the classical scikit-learn models
- `tests/` — focused tests for prediction and input handling

## Setup

Use Python 3.10 or newer. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train models

After downloading `train.csv`, train the classical models with:

```powershell
python scripts/train_models.py
```

This script trains KNN, Random Forest, Logistic Regression, and SVM models and saves their artifacts under `models/`. The notebooks contain the MLP and CNN training workflows.

## Run the Streamlit app

```powershell
streamlit run app/app.py
```

The app supports PNG and JPG/JPEG images, which it converts to grayscale and resizes to 28 x 28, or a single-row CSV containing 784 pixel values. Choose an available model in the sidebar, or use `auto` to select the first available model. The app also shows the model leaderboard and sample training rows when their files are available.

The app can start without local assets, but predictions are disabled until at least one trained model is present. Model artifacts are excluded from Git, so they must be generated locally or supplied separately.

By default, the app looks for `data/`, `models/`, and `results/` under the project folder. To store them elsewhere, set `DIGIT_RECOGNITION_ASSETS_DIR` to a directory containing those folders:

```powershell
$env:DIGIT_RECOGNITION_ASSETS_DIR = "C:\path\to\digit-recognition-assets"
streamlit run app/app.py
```

## Make a prediction from the command line

For an image:

```powershell
python -m digit_recognition.predict --image path\to\digit.png --model logistic_regression
```

For a single-row CSV:

```powershell
python -m digit_recognition.predict --csv path\to\digit.csv --model logistic_regression
```

The `--model` option is optional. Supported names are `cnn`, `mlp`, `svm`, `knn`, `random_forest`, and `logistic_regression`.

## Model and results files

Supported model filenames:

- `models/cnn_digit_model.keras`
- `models/mlp_digit_model.pkl`
- `models/svm_digit_model.pkl`
- `models/knn_digit_model.pkl`
- `models/random_forest_digit_model.pkl`
- `models/logistic_regression_digit_model.pkl`

`results/model_comparison.csv` supplies the leaderboard shown in the app. Dataset and model files are not committed to the repository; the datasets are distributed through the GitHub release linked above.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
