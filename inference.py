"""Inference and Grad-CAM for the Heart Disease XAI model (EfficientNet-B0, Normal vs Sick CMR slices).

The model bundle is a folder with one or more `.keras` models and a `model_config.json`:

    {
      "class_names": ["Normal", "Sick"],
      "img_size": [224, 224],
      "threshold": 0.5,
      "models": ["model_run0.keras", ...],
      ...
    }

With several models (the five patient-level cross-validation runs) the probability and the Grad-CAM map
are averaged over the models (an ensemble).
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image

ImageInput = Union[str, os.PathLike, bytes, Image.Image, np.ndarray]


@dataclass
class Prediction:
    prob_sick: float                 # mean probability of the Sick class over the ensemble
    label: str                       # class at the bundle's decision threshold
    threshold: float
    model_probs: list[float]         # probability from each model in the bundle
    cam: np.ndarray | None = None    # Grad-CAM map, 7x7 in [0, 1], explaining the predicted class
    overlay: np.ndarray | None = None  # uint8 RGB image with the heat map blended in
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"prob_sick": round(self.prob_sick, 6), "label": self.label, "threshold": self.threshold,
                "model_probs": [round(p, 6) for p in self.model_probs], **self.extra}


class HeartXAI:
    def __init__(self, model_dir: str):
        cfg_path = os.path.join(model_dir, "model_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            self.config = json.load(f)
        self.class_names = list(self.config["class_names"])
        self.img_size = tuple(self.config["img_size"])
        self.threshold = float(self.config["threshold"])
        self.models = [tf.keras.models.load_model(os.path.join(model_dir, m), compile=False)
                       for m in self.config["models"]]
        self._parts = [self._find_parts(m) for m in self.models]

    # ------------------------------------------------------------------ preprocessing
    def load_image(self, image: ImageInput) -> np.ndarray:
        """Return a (H, W, 3) float32 array in 0-255, resized exactly as in training
        (TensorFlow decoding, bilinear resize, no normalisation: the model does its own)."""
        if isinstance(image, (str, os.PathLike)):
            with open(image, "rb") as f:
                image = f.read()
        if isinstance(image, bytes):
            try:
                t = tf.io.decode_image(image, channels=3, expand_animations=False)
            except (tf.errors.InvalidArgumentError, ValueError) as e:
                raise ValueError("Could not read the file as an image (JPEG, PNG, BMP or GIF).") from e
        elif isinstance(image, Image.Image):
            t = tf.convert_to_tensor(np.asarray(image.convert("RGB")))
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
                raise ValueError(f"Expected an HxW or HxWx3 image, got shape {arr.shape}")
            t = tf.convert_to_tensor(arr[..., :3])
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")
        t = tf.image.resize(tf.cast(t, tf.float32), self.img_size, method="bilinear")
        return t.numpy()

    # ------------------------------------------------------------------ prediction
    def predict(self, image: ImageInput, explain: bool = True) -> Prediction:
        img = self.load_image(image)
        probs, cams = [], []
        for model, parts in zip(self.models, self._parts):
            if explain:
                cam, p = _gradcam(parts, img, target=None)
                cams.append(cam)
            else:
                p = float(np.asarray(model(img[None], training=False))[0, 0])
            probs.append(p)
        prob = float(np.mean(probs))
        label = self.class_names[int(prob >= self.threshold)]
        pred = Prediction(prob_sick=prob, label=label, threshold=self.threshold, model_probs=probs)
        if explain:
            # For an ensemble, each model's map explains that model's own prediction. If the models disagree,
            # recompute the maps for the ensemble's class so that all of them explain the same decision.
            target = int(prob >= 0.5)
            if any(int(p >= 0.5) != target for p in probs):
                cams = [_gradcam(parts, img, target=target)[0] for parts in self._parts]
            cam = np.mean(cams, axis=0)
            cam = cam / (cam.max() + 1e-8)
            pred.cam = cam
            pred.overlay = overlay(img, cam)
        return pred

    def predict_batch(self, images: list[ImageInput], explain: bool = False) -> list[Prediction]:
        return [self.predict(im, explain=explain) for im in images]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _find_parts(model):
        base = next(l for l in model.layers if l.name.startswith("efficientnet"))
        pred = model.get_layer("prediction")
        dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense) and l.name != "prediction"][-1]
        return base, dense, pred

    def info(self) -> dict:
        keep = ("model_version", "threshold", "threshold_rule", "evaluation", "tensorflow", "keras", "trained_on")
        return {"class_names": self.class_names, "img_size": list(self.img_size), "n_models": len(self.models),
                **{k: self.config[k] for k in keep if k in self.config}}


def _gradcam(parts, img: np.ndarray, target: int | None = None):
    """Grad-CAM (Selvaraju et al.) on EfficientNet-B0's 7x7x1280 top_activation output.

    The classification head (GAP -> Dense 128 ReLU -> Dense 1) is applied with its own weights inside the
    gradient tape; dropout is the identity at inference, so this equals the model's forward pass.
    The score is the pre-sigmoid logit (for Normal, its negative), which avoids sigmoid saturation.
    Returns (cam in [0, 1] with shape 7x7, probability of Sick)."""
    base, dense, pred = parts
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
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8), prob


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a jet-coloured, bilinearly upsampled heat map onto the image."""
    import matplotlib
    cam_up = tf.image.resize(cam[..., None].astype("float32"), img.shape[:2], method="bilinear").numpy()[..., 0]
    heat = matplotlib.colormaps["jet"](np.clip(cam_up, 0, 1))[..., :3] * 255.0
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 255).astype("uint8")


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
