"""Test fixtures: a tiny stand-in model with the same layer names and head as the real one, trained in a few
seconds on synthetic 'lesion' images, so the full pipeline (loading, preprocessing, prediction, Grad-CAM,
API) can be tested without the 16 MB EfficientNet weights or any patient data."""
import io
import json
import os

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

SIZE = 224
LESION = (70, 140)  # pixel range of the bright square in 'Sick' images


def synthetic_image(sick: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 80, (SIZE, SIZE)).astype("float32")
    if sick:
        img[LESION[0]:LESION[1], LESION[0]:LESION[1]] += 120
    return np.repeat(img[..., None], 3, axis=-1)


def build_stub():
    aug = tf.keras.Sequential([layers.RandomFlip("horizontal")], name="data_augmentation")
    base = tf.keras.Sequential(
        [tf.keras.Input((SIZE, SIZE, 3)), layers.Rescaling(1 / 255.0)]
        + [layers.Conv2D(c, 3, strides=2, padding="same", activation="relu") for c in (8, 16, 16, 32, 32)],
        name="efficientnetb0")
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = aug(inp)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    return models.Model(inp, layers.Dense(1, activation="sigmoid", name="prediction")(x))


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    tf.keras.utils.set_random_seed(0)
    X = np.stack([synthetic_image(i % 2 == 1, i) for i in range(64)])
    y = np.array([i % 2 for i in range(64)], dtype="float32")
    d = tmp_path_factory.mktemp("models")
    names = []
    for k in range(2):  # two models -> exercises the ensemble path
        m = build_stub()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
        m.fit(X, y, epochs=12, batch_size=16, verbose=0)
        names.append(f"model_run{k}.keras")
        m.save(d / names[-1])
    (d / "model_config.json").write_text(json.dumps(
        {"class_names": ["Normal", "Sick"], "img_size": [SIZE, SIZE], "threshold": 0.5, "models": names,
         "model_version": "test"}))
    return str(d)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()
