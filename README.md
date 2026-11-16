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

- Python 3.10 or newer (verified locally with Python 3.12)
- Packages listed in `requirements.txt`

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The exact top-level package versions used for verification are in `requirements-tested.txt`. Install that file instead of `requirements.txt` when reproducing this environment. It is a version snapshot, not a full transitive dependency lock.

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

For a quicker first run, train one model:

```powershell
python scripts/train_models.py --models logistic_regression
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

Model artifacts are ignored by Git and must be generated or supplied locally. The training script creates the digit verifier if it is missing. To retrain just the verifier:

```powershell
python scripts/train_rejection.py
```

On other platforms, fonts are discovered under `/usr/share/fonts` or `/System/Library/Fonts`. You can provide a directory with at least four TTF fonts using `--font-dir path/to/fonts` on either training script. No fonts or external datasets are downloaded. Keep `models/digit_verifier.pkl` alongside the recognizer artifacts; predictions remain disabled until it exists. After retraining the verifier, run `python -m digit_recognition.evaluation` to refresh acceptance metrics for the installed recognizers.

The training script also replaces `results/model_comparison.csv` with metrics for the models trained in that run. To compare all installed models, run `python -m digit_recognition.evaluation`, or run `notebooks/08_Model_Comparison.ipynb` to refresh the charts too, after copying the artifacts into `models/`. It evaluates actual predictions; all artifacts must have been trained with the repository's stratified 80/20 split and `random_state=42`, leaving the holdout rows out of training. Previous fixed leaderboard scores have been removed; regenerate the results before using the leaderboard or comparison charts.

CSV headers, when present, must be `pixel0` through `pixel783`, optionally preceded by `label`. Integer pixel arrays and integer CSV values use the 0–255 scale. Floating-point arrays and CSV values within 0–1 are treated as normalized; larger values use the 0–255 scale. Image files always use byte intensities before normalization.

## Run the Streamlit app

From the project directory:

```powershell
streamlit run app/app.py
```

Predictions are enabled when at least one supported recognizer and the digit verifier are available. The app accepts PNG/JPG/JPEG images (converted to grayscale, with automatic background inversion and a 28 x 28 preview) or a CSV containing exactly one digit row with 784 pixel values. A multi-row dataset such as `test.csv` is not a single-digit prediction input.

Use **Try this digit** beneath a sample image to check a locally trained model without uploading a file. A sample PNG is also available at `results/sample_digit.png`.

The app also displays sample digit images and training rows when `data/train.csv` is available. The optional leaderboard is loaded from `results/model_comparison.csv`; it must contain `Algorithm` and `Accuracy` columns.

By default, the app looks for `data/`, `models/`, and `results/` in the project directory. To use a different asset directory, set `DIGIT_RECOGNITION_ASSETS_DIR` to a directory containing those three folders:

```powershell
$env:DIGIT_RECOGNITION_ASSETS_DIR = "C:\path\to\digit-recognition-assets"
streamlit run app/app.py
```

Training, evaluation, the CLI, and the app all use this asset directory when the environment variable is set. Without it, they use the project's data, models, and results folders.

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

## Evaluation and limitations

Training and evaluation use a stratified 80/20 split with `random_state=42`. The CNN notebook reserves a further 10% of its training split for validation; its test split stays separate. Accuracy, weighted precision, weighted recall, weighted F1, and error rate are written as percentages. The comparison notebook evaluates only model artifacts installed locally. Results from a different dataset or split are not directly comparable.

The verified comparison of the six installed artifacts puts CNN at **98.31% holdout accuracy**. The freshly trained Logistic Regression model achieved **91.32%**. The other artifacts were evaluated, not retrained in this cleanup. This measures performance on the dataset, not on arbitrary phone photos. Larger image uploads are cropped around visible strokes and fitted into a 20 x 20 area inside a 28 x 28 frame. Existing 28 x 28 images keep their margins. This helps with simple scans, but does not correct perspective, shadows, or cluttered backgrounds.

Use one digit per image. Blank and nearly blank inputs are rejected. A separate Extra Trees classifier screens digit versus non-digit shapes before any of the six recognizers runs. It trains on real training digits, printed digit positives, and generated letter, symbol, and multiple-digit negatives. The old fixed 80% recognizer-confidence rule and the SVM dependency on Logistic Regression have been removed. SVM uses the same verifier as every other model.

The verifier's threshold is chosen from a calibration subset of the digit training data, targeting at least 99% acceptance of calibration handwriting. Printed validation digits are audited separately and do not set the handwriting threshold. This is a calibration target, not a test guarantee. The final 20% digit holdout is excluded from verifier fitting and threshold selection. Generated text uses disjoint training, validation, and test font files. The measured results, font split, threshold, and dataset fingerprint are saved in `results/rejection_validation.json`.

The final verifier run accepted **99.14% of 8,400 held-out handwritten digits** and rejected **99.82% of 552 held-out generated non-digit examples**. On 80 printed digit examples from the held-out font, it accepted only **50%**: typed numerals are a weak area, and the handwriting-calibrated threshold does not promise good acceptance of arbitrary typefaces.

Synthetic text is not a representative dataset of handwritten letters, photographs, or every possible symbol. Characters such as O/0 and I/1 can be visually indistinguishable. The verifier improves rejection without guaranteeing that every non-number is detected. Its test report records the remaining false acceptances and rejections. For production use, add representative labeled non-digit handwriting and external validation.

The leaderboard reports raw classifier metrics plus **Acceptance Rate**, **Rejection Rate**, and **Accepted Accuracy** using the same verifier and preprocessing scale as prediction. Its rejection rate measures refusal of genuine holdout digits; non-digit rejection is measured separately in the verifier report. An empty accepted set has no accuracy, rather than an invented zero or 100%. Library callers can explicitly set `reject_non_digits=False` for raw diagnostic predictions; the app and CLI always enable the verifier. `min_confidence` is an optional additional recognizer threshold and is disabled by default.


Model files use Python serialization; load only artifacts you trust. Scikit-learn artifacts should be used with the same dependency versions as their training environment. TensorFlow is imported only when a CNN is selected, so classical predictions do not need to initialize it.
