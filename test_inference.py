import numpy as np
import pytest
from PIL import Image

from heart_xai import HeartXAI
from tests.conftest import LESION, png_bytes, synthetic_image


@pytest.fixture(scope="module")
def model(bundle):
    return HeartXAI(bundle)


def test_loads_ensemble(model):
    assert len(model.models) == 2
    assert model.class_names == ["Normal", "Sick"]


@pytest.mark.parametrize("kind", ["ndarray", "gray", "pil", "bytes", "path"])
def test_input_types_agree(model, tmp_path, kind):
    arr = synthetic_image(True, 1).astype("uint8")
    ref = model.predict(arr, explain=False).prob_sick
    if kind == "ndarray":
        x = arr
    elif kind == "gray":
        x = arr[..., 0]
    elif kind == "pil":
        x = Image.fromarray(arr)
    elif kind == "bytes":
        x = png_bytes(arr)
    else:
        p = tmp_path / "x.png"; p.write_bytes(png_bytes(arr)); x = str(p)
    assert model.predict(x, explain=False).prob_sick == pytest.approx(ref, abs=1e-4)


def test_resizes_any_size(model):
    big = np.kron(synthetic_image(True, 2)[..., 0], np.ones((2, 2)))  # 448x448
    assert model.load_image(big.astype("uint8")).shape == (224, 224, 3)


def test_predictions_separate_classes(model):
    sick = [model.predict(synthetic_image(True, 100 + i), explain=False).prob_sick for i in range(5)]
    normal = [model.predict(synthetic_image(False, 200 + i), explain=False).prob_sick for i in range(5)]
    assert min(sick) > max(normal)
    assert model.predict(synthetic_image(True, 300), explain=False).label == "Sick"


def test_gradcam_matches_prediction_and_localises(model):
    img = synthetic_image(True, 7)
    with_cam = model.predict(img, explain=True)
    without = model.predict(img, explain=False)
    assert with_cam.prob_sick == pytest.approx(without.prob_sick, abs=1e-4)
    assert with_cam.cam.shape == (7, 7) and with_cam.cam.max() == pytest.approx(1.0, abs=1e-6)
    assert with_cam.overlay.shape == (224, 224, 3) and with_cam.overlay.dtype == np.uint8
    # most of the map's mass should fall on the lesion (which covers ~10% of the image)
    import tensorflow as tf
    up = tf.image.resize(with_cam.cam[..., None], (224, 224)).numpy()[..., 0]
    share = up[LESION[0]:LESION[1], LESION[0]:LESION[1]].sum() / up.sum()
    assert share > 0.25


def test_rejects_garbage(model):
    with pytest.raises(ValueError):
        model.predict(b"not an image")
