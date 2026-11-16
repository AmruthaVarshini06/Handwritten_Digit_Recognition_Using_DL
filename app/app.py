from __future__ import annotations

import tempfile
import sys
import numpy as np
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from digit_recognition.evaluation import MODEL_LABELS
from digit_recognition.model_utils import (
    ASSETS_ROOT, MODELS_DIR, available_models, predict_from_file, predict_from_array, _load_image_file,
    UnrecognizedDigitError,
)
from digit_recognition.rejection import ARTIFACT_NAME

MODEL_RESULTS_PATH = ASSETS_ROOT / "results" / "model_comparison.csv"
TRAIN_DATA_PATH = ASSETS_ROOT / "data" / "train.csv"

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2.5rem;}
h1 {letter-spacing: -0.04em;}
[data-testid="stMetric"] {border: 1px solid #dce2dc; border-radius: 12px; padding: 1rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_model_results() -> pd.DataFrame | None:
    if not MODEL_RESULTS_PATH.is_file():
        return None
    try:
        return pd.read_csv(MODEL_RESULTS_PATH)
    except (OSError, ValueError) as exc:
        st.warning(f"Could not read model results: {exc}")
        return None


@st.cache_data
def load_training_sample() -> pd.DataFrame | None:
    if not TRAIN_DATA_PATH.is_file():
        return None
    try:
        sample = pd.read_csv(TRAIN_DATA_PATH, nrows=10)
        columns = [f"pixel{i}" for i in range(784)]
        if set(sample.columns) != {"label", *columns}:
            raise ValueError("Expected label and pixel0 through pixel783 columns.")
        values = sample[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 255).any():
            raise ValueError("Sample pixels must be between 0 and 255.")
        if not sample["label"].isin(range(10)).all():
            raise ValueError("Sample labels must be digits from 0 to 9.")
        return sample[["label", *columns]]
    except (OSError, ValueError) as exc:
        st.warning(f"Could not read sample digits: {exc}")
        return None


st.caption("MACHINE LEARNING / DIGIT CLASSIFICATION")
st.title("Read a handwritten digit")
st.write("Choose a model, upload one handwritten digit, and see how it reads your writing.")

model_names = available_models()
verifier_ready = (MODELS_DIR / ARTIFACT_NAME).is_file()
with st.sidebar:
    st.header("Model selection")
    model_name = st.selectbox(
        "Choose a model",
        ["auto", *model_names],
        format_func=lambda name: "Automatic · first available" if name == "auto" else MODEL_LABELS[name],
        disabled=not model_names,
        help="Only models found in the local models/ folder are listed.",
    )

    if model_names:
        st.caption(f"Available models: {', '.join(MODEL_LABELS[name] for name in model_names)}")
    else:
        st.warning("No trained models are installed. Add model artifacts to models/ to enable predictions.")
    if not verifier_ready:
        st.warning("Digit validation needs setup. Run python scripts/train_rejection.py to enable predictions.")

    st.markdown("---")
    st.subheader("Holdout results")
    results = load_model_results()
    if results is None or results.empty:
        st.info("Leaderboard data is unavailable. Add results/model_comparison.csv to show it.")
    elif not {"Algorithm", "Accuracy"}.issubset(results.columns):
        st.error("The leaderboard CSV must contain Algorithm and Accuracy columns.")
    else:
        accuracy = pd.to_numeric(results["Accuracy"], errors="coerce")
        accuracy = accuracy.where(accuracy.between(0, 100))
        if accuracy.notna().any():
            best_model = results.loc[accuracy.idxmax()]
            st.metric("Best performer", f"{best_model['Algorithm']}")
            st.metric("Best raw accuracy", f"{accuracy.max():.2f}%")
            st.caption("Raw classifier scores exclude rejected inputs.")
            selected = model_names[0] if model_name == "auto" and model_names else model_name
            selection = results[results["Algorithm"] == MODEL_LABELS.get(selected)]
            if not selection.empty and {"Acceptance Rate", "Accepted Accuracy"}.issubset(results.columns):
                row = selection.iloc[0]
                if pd.notna(row["Acceptance Rate"]):
                    st.metric("Digit acceptance", f"{row['Acceptance Rate']:.2f}%")
                if pd.notna(row["Accepted Accuracy"]):
                    st.metric("Accuracy on accepted digits", f"{row['Accepted Accuracy']:.2f}%")
        else:
            st.error("The leaderboard Accuracy column must contain numeric values.")

upload = st.file_uploader(
    "Upload an image or CSV",
    type=["png", "jpg", "jpeg", "csv"],
    disabled=not model_names or not verifier_ready,
    help="Predictions are enabled when at least one trained model is available.",
)

if upload is not None:
    try:
        with tempfile.TemporaryDirectory() as upload_dir:
            suffix = Path(upload.name).suffix.lower()
            temp_path = Path(upload_dir) / f"upload{suffix}"
            temp_path.write_bytes(upload.getvalue())

            selected_model = model_names[0] if model_name == "auto" else model_name
            with st.spinner("Reading your digit…"):
                prediction, confidence = predict_from_file(temp_path, model_name=selected_model)
            st.caption(f"Prediction by {MODEL_LABELS[selected_model]}")

            prediction_col, confidence_col = st.columns(2)
            with prediction_col:
                st.success(f"Predicted digit: {prediction}")
            with confidence_col:
                if confidence is not None:
                    st.metric("Model probability", f"{confidence:.2%}")

            if suffix in {".png", ".jpg", ".jpeg"}:
                original, prepared = st.columns(2)
                with original:
                    st.image(str(temp_path), caption="Your image", width=220)
                with prepared:
                    st.image(_load_image_file(temp_path), caption="28 × 28 input seen by the model", width=220, clamp=True)
            st.caption("Model probabilities are not a guarantee of correctness. Try another model when the digit is ambiguous.")
    except UnrecognizedDigitError as exc:
        st.error(str(exc))
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
        st.subheader("A few digits from the training set")
        if "label" in sample_rows and sample_rows.shape[1] == 785:
            columns = st.columns(5)
            for index, (_, row) in enumerate(sample_rows.head(5).iterrows()):
                with columns[index]:
                    pixels = row.drop(labels="label").to_numpy(dtype=np.uint8).reshape(28, 28)
                    st.image(pixels, caption=f"Digit {int(row['label'])}", width=120)
                    if st.button("Try this digit", key=f"sample_{index}", disabled=not model_names or not verifier_ready):
                        try:
                            selected = model_names[0] if model_name == "auto" else model_name
                            prediction, _ = predict_from_array(pixels, selected)
                            st.success(f"Prediction: {prediction}")
                            st.caption(MODEL_LABELS[selected])
                        except (OSError, ValueError, ImportError) as exc:
                            st.error(f"Prediction failed: {exc}")
        with st.expander("View source rows"):
            st.dataframe(sample_rows)

    st.subheader("Model comparison")
    if results is not None:
        st.dataframe(results, hide_index=True)

with st.expander("Tips for a useful prediction"):
    st.write("Use one clear digit with an uncluttered background. Both light and dark backgrounds are supported. Photos are cropped and fitted into a 28 × 28 frame; existing 28 × 28 inputs keep their margins.")
    st.write("A separate verifier screens for single-digit inputs before classification. It is tested with handwritten digits and generated text; similar-looking letters and unfamiliar symbols can still pass. A model probability is not a guarantee that an input is a number.")
    st.write("This project compares six classifiers on digit images. It is not designed for full words, multiple digits, or handwriting recognition in documents.")
