# Handwritten Digit Recognition

A machine-learning project for classifying handwritten digits (0-9) from 28 x 28 grayscale images. It includes a Streamlit app, a command-line prediction tool, model-training notebooks, and a script for training classical machine-learning models.

## Features

- Predict a digit from a PNG/JPG image or a single-row CSV.
- Choose an available model in the Streamlit app or let it select one automatically.
- Train KNN, Random Forest, Logistic Regression, and SVM models with the training script.
- Explore preprocessing, MLP, CNN, classical models, and model comparison in Jupyter notebooks.
- Run focused tests for prediction and input validation.

## Dataset

Download `train.csv` and `test.csv` from the GitHub release: **[Dataset Uploaded (v1.0)](https://github.com/AmruthaVarshini06/Handwritten_Digit_Recognition_Using_DL/releases/tag/v1.0)**. Place both files in the project's `data/` directory. The training file has a `label` column and 784 pixel features; the test file contains pixel features.

On Windows, run these commands from the project directory:

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://github.com/AmruthaVarshini06/Handwritten_Digit_Recognition_Using_DL/releases/download/v1.0/train.csv" -OutFile "data/train.csv"
Invoke-WebRequest -Uri "https://github.com/AmruthaVarshini06/Handwritten_Digit_Recognition_Using_DL/releases/download/v1.0/test.csv" -OutFile "data/test.csv"
```

The dataset is based on [Handwritten Digit Recognition on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition).

## Requirements and installation

- Python 3.10 or newer
- Packages listed in `requirements.txt`

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To run the notebooks, open them in VS Code with the Jupyter extension, or install Jupyter separately:

```powershell
python -m pip install notebook
jupyter notebook
```

## Train models

Download `train.csv` first. The training script fits KNN, Random Forest, Logistic Regression, and SVM models, prints holdout accuracy, and saves the model files under `models/`:

```powershell
python scripts/train_models.py
```

The `notebooks/` directory contains the preprocessing, MLP, CNN, individual classical-model, and model-comparison workflows. Run the desired notebook cells from top to bottom. Notebook-generated model files may need to be copied into `models/` using the filenames in the table below before the app or CLI can load them.

| Model | Expected file |
| --- | --- |
| CNN | `models/cnn_digit_model.keras` |
| MLP | `models/mlp_digit_model.pkl` |
| SVM | `models/svm_digit_model.pkl` |
| KNN | `models/knn_digit_model.pkl` |
| Random Forest | `models/random_forest_digit_model.pkl` |
| Logistic Regression | `models/logistic_regression_digit_model.pkl` |

Model artifacts are ignored by Git and must be generated or supplied locally.

## Run the Streamlit app

From the project directory:

```powershell
streamlit run app/app.py
```

Predictions are enabled when at least one supported model file is available. The app accepts PNG/JPG/JPEG images (converted to grayscale and resized to 28 x 28) or a CSV containing exactly one digit row with 784 pixel values. A multi-row dataset such as `test.csv` is not a single-digit prediction input.

The app also displays sample training rows when `data/train.csv` is available. The optional leaderboard is loaded from `results/model_comparison.csv`; it must contain `Algorithm` and `Accuracy` columns.

By default, the app looks for `data/`, `models/`, and `results/` in the project directory. To use a different asset directory, set `DIGIT_RECOGNITION_ASSETS_DIR` to a directory containing those three folders:

```powershell
$env:DIGIT_RECOGNITION_ASSETS_DIR = "C:\path\to\digit-recognition-assets"
streamlit run app/app.py
```

The training script uses the project's `data/train.csv` and writes models to the project's `models/` directory; the environment variable above changes the app's asset location.

## Make a prediction from the command line

Run these commands from the project directory:

```powershell
python -m digit_recognition.predict --image path\to\digit.png --model logistic_regression
python -m digit_recognition.predict --csv path\to\digit.csv --model logistic_regression
```

Use either `--image` or `--csv`. The `--model` option is optional; supported model names are `cnn`, `mlp`, `svm`, `knn`, `random_forest`, and `logistic_regression`. Without a model name, the first available model is selected.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Project structure

```text
app/                    Streamlit application
data/                   Downloaded training and test datasets
digit_recognition/      Model loading, preprocessing, and prediction code
models/                 Locally generated or supplied model artifacts
notebooks/              Preprocessing, training, and comparison notebooks
results/                Optional model-comparison metrics and charts
scripts/train_models.py Classical model training script
tests/                  Unit tests
requirements.txt        Python package dependencies
```
