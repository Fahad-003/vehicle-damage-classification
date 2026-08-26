from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# This styles native Streamlit widgets only. No UI content is rendered as HTML.
st.markdown(
    """
    <style>
        .stApp { background: #f6f8fc; }

        .block-container {
            max-width: 1280px;
            padding-top: 2.75rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #121a2a;
            border-right: 1px solid #243048;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 3.5rem;
        }

        section[data-testid="stSidebar"] * { color: #f8fafc; }

        section[data-testid="stSidebar"] h1 {
            font-size: 3.2rem !important;
            text-align: center;
            margin-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] h2 {
            font-size: 1.65rem !important;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }

        section[data-testid="stSidebar"] [data-testid="stDivider"] {
            border-color: #263247;
            margin: 3.25rem 0 3.5rem;
        }

        h1 {
            color: #111827;
            font-size: clamp(2.5rem, 4vw, 4rem) !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em;
            margin-bottom: 0.7rem;
        }

        h2, h3 { color: #182235; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #dde3ec;
            border-radius: 18px;
            box-shadow: none;
        }

        [data-testid="stMetric"] {
            min-height: 112px;
            display: flex;
            flex-direction: column-reverse;
            align-items: center;
            justify-content: center;
            text-align: center;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            justify-content: center;
            text-align: center;
        }

        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 750;
        }

        [data-testid="stMetricLabel"] {
            color: #64748b;
            font-size: 1rem;
            padding-top: 0.55rem;
        }

        [data-testid="stFileUploader"] {
            background: #ffffff;
            border-radius: 18px;
            padding: 0.25rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #f0f3f8;
            border: 0;
            border-radius: 12px;
        }

        .stButton > button {
            min-height: 48px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 650;
        }

        @media (min-width: 1100px) {
            section[data-testid="stSidebar"] { min-width: 380px; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "vehicle_damage_model.keras"
CLASS_NAMES_PATH = APP_DIR / "models" / "class_names.txt"

# These values are taken from vehicle_damage_detection.ipynb.
IMG_SIZE = (224, 224)
UNCERTAINTY_THRESHOLD = 40.0
CLASS_NAMES = [
    "bumper_dent",
    "bumper_scratch",
    "door_dent",
    "door_scratch",
    "glass_shatter",
    "head_lamp",
    "tail_lamp",
]


# ============================================================
# HELPERS
# ============================================================

def format_class_name(name: str) -> str:
    """Convert a stored class label into a display label."""
    return name.replace("_", " ").title()


def load_class_names() -> list[str]:
    """Load the exported label order and fail clearly if it is inconsistent."""
    if not CLASS_NAMES_PATH.is_file():
        return CLASS_NAMES

    labels = [
        line.strip()
        for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if labels != CLASS_NAMES:
        raise ValueError(
            "models/class_names.txt does not match the class order exported by "
            "the training notebook. Replace it with the notebook-generated file."
        )

    return labels


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(model_path: str) -> tf.keras.Model:
    """Load the saved Keras model once per Streamlit process."""
    return tf.keras.models.load_model(model_path, compile=False)


def validate_model(model: tf.keras.Model, class_names: list[str]) -> None:
    """Catch an incorrect model artifact before accepting an image upload."""
    input_shape = tuple(model.input_shape[1:])
    output_width = int(model.output_shape[-1])

    if input_shape != (*IMG_SIZE, 3):
        raise ValueError(
            "The model input shape is "
            f"{input_shape}, but this app is configured for {(224, 224, 3)}."
        )
    if output_width != len(class_names):
        raise ValueError(
            f"The model returns {output_width} scores, but {len(class_names)} "
            "class labels were loaded."
        )


def preprocess_for_model(image: Image.Image) -> tf.Tensor:
    """Apply the exact image preparation used by the notebook at inference."""
    image_array = np.asarray(image.convert("RGB"))
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, IMG_SIZE)
    return tf.expand_dims(image_tensor, axis=0)


def predict(model: tf.keras.Model, image: Image.Image) -> np.ndarray:
    """Return the model's softmax probabilities for one uploaded RGB image."""
    probabilities = np.asarray(
        model.predict(preprocess_for_model(image), verbose=0)[0], dtype=np.float32
    )

    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}.")
    if not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid prediction values.")

    return probabilities


def reset_result_if_image_changed(file_id: str) -> None:
    """Do not show a previous image's prediction after a new upload."""
    if st.session_state.get("uploaded_file_id") != file_id:
        st.session_state.uploaded_file_id = file_id
        st.session_state.predictions = None


# ============================================================
# LOAD AND VALIDATE ARTIFACTS
# ============================================================

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    class_names = load_class_names()
    model = load_model(str(MODEL_PATH))
    validate_model(model, class_names)
except Exception as error:
    st.error("The model files could not be loaded safely.")
    st.code(str(error))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# 🚗")
    st.header("Vehicle Damage AI")
    st.write("Deep learning based vehicle damage classification system.")

    st.divider()
    st.subheader("Model Information")
    st.write("**Architecture:** EfficientNetB0")
    st.write("**Learning:** Transfer Learning + Fine-Tuning")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:** 7")
    st.write("**Task:** Multiclass Classification")

    st.divider()
    st.subheader("Supported Damage Types")
    for name in class_names:
        st.write(f"• {format_class_name(name)}")

    st.divider()
    st.caption("Model trained using transfer learning and fine-tuning.")


# ============================================================
# MAIN PAGE
# ============================================================

st.title("Vehicle Damage Detection")
st.write(
    "Upload a vehicle image and let the deep learning model identify the "
    "most likely type of damage."
)

card1, card2, card3 = st.columns(3, gap="medium")
with card1:
    with st.container(border=True):
        st.metric("Damage Categories", "7")
with card2:
    with st.container(border=True):
        st.metric("Model Input Size", "224 × 224")
with card3:
    with st.container(border=True):
        st.metric("Deep Learning Architecture", "EfficientNetB0")

st.header("Upload Vehicle Image")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    with st.container(border=True):
        st.info("📷 Upload an image above to begin the damage analysis.")
    st.divider()
    st.caption("Vehicle Damage Detection • EfficientNetB0 • Transfer Learning + Fine-Tuning")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("Unable to read this image. Please upload a valid JPG, JPEG, or PNG file.")
    st.stop()

reset_result_if_image_changed(uploaded_file.file_id)

left_col, right_col = st.columns([1, 1], gap="large")
with left_col:
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)
    st.caption(f"Image size: {image.width} × {image.height} pixels")

with right_col:
    st.subheader("Damage Analysis")
    st.write(
        "The model will analyze the uploaded image and predict the most likely "
        "damage category."
    )
    analyze = st.button(
        "🔍 Analyze Vehicle Damage", type="primary", use_container_width=True
    )

    if analyze:
        with st.spinner("Analyzing vehicle image..."):
            try:
                # Deliberately no /255 and no EfficientNet preprocess_input here.
                # The training notebook passed 0–255 float32 RGB tensors directly
                # to EfficientNetB0, whose saved model contains its preprocessing.
                st.session_state.predictions = predict(model, image)
            except Exception as error:
                st.error("An error occurred during prediction.")
                st.exception(error)


# ============================================================
# RESULTS
# ============================================================

predictions = st.session_state.get("predictions")
if predictions is not None:
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index] * 100)
    is_uncertain = confidence < UNCERTAINTY_THRESHOLD

    st.divider()
    st.subheader("Damage Analysis Result")
    result_col1, result_col2 = st.columns([2, 1], gap="medium")
    with result_col1:
        with st.container(border=True):
            st.caption("Predicted Damage")
            if is_uncertain:
                st.subheader("Uncertain / Unsupported Damage Type")
                st.caption(
                    f"Closest known category: {format_class_name(predicted_class)}"
                )
            else:
                st.subheader(format_class_name(predicted_class))
    with result_col2:
        with st.container(border=True):
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

    if is_uncertain:
        st.warning(
            "The model is not confident enough to identify a supported damage "
            "type. This image may show an unsupported damage type or need a "
            "clearer close-up."
        )
    elif confidence >= 70:
        st.success("High confidence prediction.")
    elif confidence >= 50:
        st.info("Moderate confidence prediction. Some damage categories may look similar.")
    else:
        st.warning("Low confidence prediction. Consider uploading a clearer vehicle image.")

    st.divider()
    st.header("Prediction Breakdown")
    for index in np.argsort(predictions)[-3:][::-1]:
        probability = float(predictions[index])
        st.write(f"**{format_class_name(class_names[index])}** — {probability * 100:.2f}%")
        st.progress(probability)

    with st.expander("View all class probabilities"):
        probability_table = pd.DataFrame(
            {
                "Damage Type": [format_class_name(name) for name in class_names],
                "Probability (%)": np.round(predictions * 100, 2),
            }
        ).sort_values("Probability (%)", ascending=False, ignore_index=True)
        st.dataframe(probability_table, use_container_width=True, hide_index=True)

st.divider()
st.caption("Vehicle Damage Detection • EfficientNetB0 • Transfer Learning + Fine-Tuning")
