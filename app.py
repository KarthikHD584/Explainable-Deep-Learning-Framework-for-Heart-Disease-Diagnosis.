import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Heart Disease Image Classifier",
    page_icon="❤️",
    layout="centered",
)

@st.cache_resource(show_spinner="Loading the deep-learning model...")
def load_model_and_classes():
    model_candidates = [
        APP_DIR / "models" / "final_model.keras",
        APP_DIR / "outputs" / "final_model.keras",
        APP_DIR / "final_model.keras",
        APP_DIR / "models" / "best_model.keras",
    ]
    class_candidates = [
        APP_DIR / "models" / "class_names.json",
        APP_DIR / "outputs" / "class_names.json",
        APP_DIR / "class_names.json",
    ]

    model_path = next((p for p in model_candidates if p.exists()), None)
    class_path = next((p for p in class_candidates if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError(
            "Model file not found. Copy the trained final_model.keras file into "
            "the models/ folder (or outputs/) in this repository."
        )
    if class_path is not None:
        with open(class_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = ["Class 0", "Class 1"]

    model = tf.keras.models.load_model(model_path, compile=False)
    return model, class_names, str(model_path)

st.title("❤️ Heart Disease Image Classification")
st.write(
    "Upload an image to run the trained EfficientNetB0 binary image classifier. "
    "This is an educational research demonstration, not a medical diagnosis."
)

with st.expander("About this model"):
    st.markdown(
        """
        - **Architecture:** EfficientNetB0 transfer learning with a sigmoid output.
        - **Input size:** 224 × 224 pixels.
        - **Task:** Binary image classification using the two folders/classes in the training dataset.
        - **Important:** The predicted class depends on the labels used when training. This tool must not be used to make clinical decisions.
        """
    )

try:
    model, class_names, model_path = load_model_and_classes()
    st.caption(f"Loaded model: `{Path(model_path).name}`")
except Exception as exc:
    st.error(f"Cannot load the trained model: {exc}")
    st.info(
        "The notebook must be trained first. Then add `final_model.keras` and "
        "`class_names.json` to the repository as described in README.md."
    )
    st.stop()

if not isinstance(class_names, list) or len(class_names) != 2:
    st.warning("Expected exactly two class names. Check class_names.json.")

uploaded_file = st.file_uploader(
    "Choose an ECG/cardiac image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        resized = image.resize(IMG_SIZE)
        array = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)

        if st.button("Classify image", type="primary", use_container_width=True):
            with st.spinner("Running prediction..."):
                output = model.predict(batch, verbose=0)
                probability_class_1 = float(np.asarray(output).reshape(-1)[0])
                predicted_index = 1 if probability_class_1 >= 0.5 else 0
                confidence = (
                    probability_class_1
                    if predicted_index == 1
                    else 1.0 - probability_class_1
                )

            predicted_label = (
                class_names[predicted_index]
                if predicted_index < len(class_names)
                else f"Class {predicted_index}"
            )
            st.subheader("Prediction")
            st.success(f"Predicted class: **{predicted_label}**")
            st.metric("Model confidence (threshold-based)", f"{confidence * 100:.2f}%")
            st.caption(
                f"Raw sigmoid score for class 1: {probability_class_1:.4f}. "
                "This score is not necessarily a calibrated clinical probability."
            )
            st.warning(
                "Research use only. Do not use this prediction for diagnosis, treatment, "
                "or emergency decisions. Consult a qualified healthcare professional."
            )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
