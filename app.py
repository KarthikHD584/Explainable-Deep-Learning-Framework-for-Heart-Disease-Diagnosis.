"""Explainable heart disease classification from cardiac MRI — Streamlit app.

Self-contained: everything the app needs is in this one file plus the models/ folder.

Repository layout expected on Streamlit Cloud:
    app.py
    requirements.txt
    .streamlit/config.toml
    models/
        final_model.keras          (or model_run0.keras ... from export_for_deployment.py)
        model_config.json          (optional; gives class names, threshold, model list)
"""
import glob
import io
import json
import os

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Heart Disease XAI", page_icon="🫀", layout="wide")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
DISCLAIMER = ("Research prototype. Not a medical device and not for clinical use. Trained on 30 patients from one "
              "hospital and one scanner (CAD Cardiac MRI Dataset); results on other data are unknown.")


# ----------------------------------------------------------------------------- model loading
def read_config(model_dir: str) -> dict:
    cfg = {"class_names": ["Normal", "Sick"], "img_size": [224, 224], "threshold": 0.5}
    path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    if not cfg.get("models"):
        cfg["models"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(model_dir, "*.keras")))
    return cfg


@st.cache_resource(show_spinner="Loading the model (first visit only)…")
def load_models(model_dir: str):
    import tensorflow as tf
    cfg = read_config(model_dir)
    if not cfg["models"]:
        return cfg, []
    models = [tf.keras.models.load_model(os.path.join(model_dir, m), compile=False) for m in cfg["models"]]
    return cfg, models


def model_parts(model):
    import tensorflow as tf
    base = next(l for l in model.layers if l.name.startswith("efficientnet"))
    pred = model.get_layer("prediction")
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense) and l.name != "prediction"][-1]
    return base, dense, pred


# ----------------------------------------------------------------------------- inference
def load_image(data: bytes, size) -> np.ndarray:
    """Decode and resize exactly as in training: RGB, bilinear, float32 0-255 (the model normalises itself)."""
    import tensorflow as tf
    t = tf.io.decode_image(data, channels=3, expand_animations=False)
    return tf.image.resize(tf.cast(t, tf.float32), size, method="bilinear").numpy()


def gradcam(model, img: np.ndarray, target=None):
    """Grad-CAM on EfficientNet-B0's 7x7x1280 top activation, using the pre-sigmoid score of the class.
    The head (GAP -> Dense -> Dense) is applied with its own weights; dropout is the identity at inference."""
    import tensorflow as tf
    base, dense, pred = model_parts(model)
    x = tf.keras.applications.efficientnet.preprocess_input(tf.convert_to_tensor(img[None], tf.float32))
    conv = tf.convert_to_tensor(base(x, training=False))
    with tf.GradientTape() as tape:
        tape.watch(conv)
        h = tf.nn.relu(tf.matmul(tf.reduce_mean(conv, axis=(1, 2)), dense.kernel) + dense.bias)
        logit = tf.matmul(h, pred.kernel) + pred.bias
        prob = float(tf.sigmoid(logit)[0, 0])
        cls = int(prob >= 0.5) if target is None else target
        score = logit[:, 0] if cls == 1 else -logit[:, 0]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8), prob


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    import matplotlib
    import tensorflow as tf
    up = tf.image.resize(cam[..., None].astype("float32"), img.shape[:2], method="bilinear").numpy()[..., 0]
    heat = matplotlib.colormaps["jet"](np.clip(up, 0, 1))[..., :3] * 255.0
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 255).astype("uint8")


def predict(models, cfg, data: bytes):
    img = load_image(data, tuple(cfg["img_size"]))
    results = [gradcam(m, img) for m in models]
    probs = [p for _, p in results]
    prob = float(np.mean(probs))
    target = int(prob >= 0.5)
    if any(int(p >= 0.5) != target for p in probs):          # ensemble disagreement: explain the same class
        results = [gradcam(m, img, target=target) for m in models]
    cam = np.mean([c for c, _ in results], axis=0)
    cam = cam / (cam.max() + 1e-8)
    return img, prob, probs, overlay(img, cam)


# ----------------------------------------------------------------------------- page
st.title("🫀 Explainable heart disease classification from cardiac MRI")
st.caption("EfficientNet-B0 classifier (Normal vs Sick / coronary artery disease) for a single CMR slice, "
           "with a Grad-CAM heat map showing which regions drove the prediction.")
st.warning(DISCLAIMER, icon="⚠️")

try:
    cfg, models = load_models(MODEL_DIR)
except Exception as e:  # show a readable message instead of a redacted traceback
    st.error(f"The model could not be loaded from `{MODEL_DIR}/`: {type(e).__name__}: {e}")
    st.info("Check that the .keras file is in the models/ folder of the repository (not tracked with Git LFS) and "
            "that requirements.txt pins the same TensorFlow version that was used for training.")
    st.stop()

if not models:
    st.error(f"No model found. Put `final_model.keras` (or the exported bundle) into the `{MODEL_DIR}/` folder of "
             "the repository and reboot the app.")
    st.stop()

threshold = float(cfg["threshold"])
names = cfg["class_names"]
with st.sidebar:
    st.header("Model")
    st.write(f"**Models loaded:** {len(models)}")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    if cfg.get("evaluation"):
        st.write("**Evaluation:**")
        st.json(cfg["evaluation"])
    st.divider()
    st.caption("Input: one cardiac MRI slice (JPEG/PNG), like the images in the CAD Cardiac MRI Dataset. "
               "Other image types (echo, X-ray, photos) give meaningless results.")

uploaded = st.file_uploader("Upload a cardiac MRI slice", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is None:
    st.info("Upload an image to get a prediction and its Grad-CAM heat map.")
    st.stop()

data = uploaded.getvalue()
try:
    with st.spinner("Classifying…"):
        img, prob, probs, heat = predict(models, cfg, data)
except Exception as e:
    st.error(f"Could not process this file: {type(e).__name__}: {e}")
    st.stop()

label = names[int(prob >= threshold)]
c1, c2 = st.columns(2)
c1.image(img.astype("uint8"), caption="Input (resized to 224×224)", width="stretch")
c2.image(heat, caption="Grad-CAM overlay (warmer = more influence on the prediction)", width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Prediction", label)
m2.metric("Probability of Sick", f"{prob:.3f}")
m3.metric("Threshold", f"{threshold:.3f}")
st.progress(min(max(prob, 0.0), 1.0), text=f"P(Sick) = {prob:.3f}")
if len(probs) > 1:
    st.caption("Per-model probabilities: " + ", ".join(f"{p:.3f}" for p in probs))
if abs(prob - threshold) < 0.1:
    st.info("This prediction is close to the decision threshold, so the model is uncertain about this image.")

buf = io.BytesIO()
Image.fromarray(heat).save(buf, format="PNG")
st.download_button("Download heat map (PNG)", buf.getvalue(), file_name=f"gradcam_{uploaded.name}.png",
                   mime="image/png")
st.caption("The heat map shows where the evidence is, not what it is (Grad-CAM, 7×7 resolution).")
