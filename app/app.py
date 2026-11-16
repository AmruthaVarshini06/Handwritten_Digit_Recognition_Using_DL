from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from digit_recognition.model_utils import predict_from_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_RESULTS_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"


@st.cache_data
def load_model_results() -> pd.DataFrame:
    return pd.read_csv(MODEL_RESULTS_PATH)


@st.cache_data
def load_training_sample() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DATA_PATH).head(5)


st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("Handwritten Digit Recognition")
st.caption("Upload a handwritten digit image or a 784-pixel CSV row to classify it with a trained model.")

with st.sidebar:
    st.header("Model selection")
    model_name = st.selectbox(
        "Choose a model",
        [
            "auto",
            "cnn",
            "mlp",
            "svm",
            "knn",
            "random_forest",
            "logistic_regression",
        ],
    )

    st.markdown("---")
    st.subheader("Model leaderboard")
    results = load_model_results()
    best_model = results.loc[results["Accuracy"].idxmax()]
    st.metric("Best performer", f"{best_model['Algorithm']}")
    st.metric("Best accuracy", f"{best_model['Accuracy']:.2f}%")

upload = st.file_uploader("Upload an image or CSV", type=["png", "jpg", "jpeg", "csv"])

if upload is not None:
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / upload.name
    temp_path.write_bytes(upload.getvalue())

    try:
        prediction, confidence = predict_from_file(
            temp_path,
            model_name=None if model_name == "auto" else model_name,
        )

        prediction_col, confidence_col = st.columns(2)
        with prediction_col:
            st.success(f"Predicted digit: {prediction}")
        with confidence_col:
            if confidence is not None:
                st.metric("Confidence", f"{confidence:.2%}")

        if upload.name.lower().endswith((".png", ".jpg", ".jpeg")):
            st.image(temp_path, caption="Uploaded input", width=220)
    except Exception as exc:  # pragma: no cover
        st.error(f"Prediction failed: {exc}")
else:
    st.info("No file selected. Upload a digit image or a CSV row to begin, or review the sample training rows below.")

    sample_rows = load_training_sample()
    st.subheader("Sample rows from the dataset")
    st.dataframe(sample_rows, use_container_width=True)

    st.subheader("Model comparison")
    st.dataframe(load_model_results(), use_container_width=True)
