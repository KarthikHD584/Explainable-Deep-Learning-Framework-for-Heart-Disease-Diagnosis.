import base64
import importlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import png_bytes, synthetic_image


@pytest.fixture(scope="module")
def client(bundle, monkeypatch_module):
    monkeypatch_module.setenv("MODEL_DIR", bundle)
    monkeypatch_module.setenv("MAX_UPLOAD_MB", "1")
    import api
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["n_models"] == 2 and info["class_names"] == ["Normal", "Sick"]


def test_predict_with_gradcam(client):
    r = client.post("/predict", files={"file": ("s.png", png_bytes(synthetic_image(True, 5)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "Sick" and 0 <= body["prob_sick"] <= 1 and len(body["model_probs"]) == 2
    img = Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"])))
    assert img.size == (224, 224)


def test_predict_without_gradcam(client):
    r = client.post("/predict?explain=false",
                    files={"file": ("n.png", png_bytes(synthetic_image(False, 6)), "image/png")})
    assert r.status_code == 200 and "gradcam_png_base64" not in r.json()


def test_bad_inputs(client):
    assert client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("x.png", b"garbage", "image/png")}).status_code == 422
    big = b"\x00" * (1024 * 1024 + 10)
    assert client.post("/predict", files={"file": ("big.png", big, "image/png")}).status_code == 413
