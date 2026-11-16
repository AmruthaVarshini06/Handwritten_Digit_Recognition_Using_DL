from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from digit_recognition.model_utils import ASSETS_ROOT, available_models, predict_from_file

MODEL_RESULTS_PATH = ASSETS_ROOT / "results" / "model_comparison.csv"
TRAIN_DATA_PATH = ASSETS_ROOT / "data" / "train.csv"

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")


@st.cache_data
def load_model_results() -> pd.DataFrame | None:
    if not MODEL_RESULTS_PATH.is_file():
        return None
    return pd.read_csv(MODEL_RESULTS_PATH)


@st.cache_data
def load_training_sample() -> pd.DataFrame | None:
    if not TRAIN_DATA_PATH.is_file():
        return None
    return pd.read_csv(TRAIN_DATA_PATH).head(5)


st.title("Handwritten Digit Recognition")
st.caption("Upload a handwritten digit image or a single 784-pixel CSV row to classify it.")

model_names = available_models()
with st.sidebar:
    st.header("Model selection")
    model_name = st.selectbox(
        "Choose a model",
        ["auto", *model_names],
        disabled=not model_names,
        help="Only models found in the local models/ folder are listed.",
    )

    if model_names:
        st.caption(f"Available models: {', '.join(model_names)}")
    else:
        st.warning("No trained models are installed. Add model artifacts to models/ to enable predictions.")

    st.markdown("---")
    st.subheader("Model leaderboard")
    results = load_model_results()
    if results is None or results.empty:
        st.info("Leaderboard data is unavailable. Add results/model_comparison.csv to show it.")
    elif not {"Algorithm", "Accuracy"}.issubset(results.columns):
        st.error("The leaderboard CSV must contain Algorithm and Accuracy columns.")
    else:
        accuracy = pd.to_numeric(results["Accuracy"], errors="coerce")
        if accuracy.notna().any():
            best_model = results.loc[accuracy.idxmax()]
            st.metric("Best performer", f"{best_model['Algorithm']}")
            st.metric("Best accuracy", f"{accuracy.max():.2f}%")
        else:
            st.error("The leaderboard Accuracy column must contain numeric values.")

upload = st.file_uploader(
    "Upload an image or CSV",
    type=["png", "jpg", "jpeg", "csv"],
    disabled=not model_names,
    help="Predictions are enabled when at least one trained model is available.",
)

if upload is not None:
    try:
        with tempfile.TemporaryDirectory() as upload_dir:
            suffix = Path(upload.name).suffix.lower()
            temp_path = Path(upload_dir) / f"upload{suffix}"
            temp_path.write_bytes(upload.getvalue())

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

            if suffix in {".png", ".jpg", ".jpeg"}:
                st.image(temp_path, caption="Uploaded input", width=220)
    except (AttributeError, EOFError, FileNotFoundError, ImportError, KeyError, OSError, ValueError) as exc:
        st.error(f"Prediction failed: {exc}")
else:
    if model_names:
        st.info("Upload a digit image or a CSV row to begin, or review the sample training rows below.")
    else:
        st.info("Upload is disabled until a trained model is available.")

    sample_rows = load_training_sample()
    if sample_rows is None:
        st.info("Training samples are unavailable. Add data/train.csv to show sample rows.")
    else:
        st.subheader("Sample rows from the dataset")
        st.dataframe(sample_rows, width="stretch")

    st.subheader("Model comparison")
    if results is not None:
        st.dataframe(results, width="stretch")
