"""REST API.  Run:  uvicorn api:app --host 0.0.0.0 --port 8000
Docs at http://localhost:8000/docs

Environment variables
  MODEL_DIR        folder with model_config.json and the .keras files (default: ./models)
  MAX_UPLOAD_MB    maximum upload size in MB (default: 10)
"""
import base64
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from heart_xai import DISCLAIMER, HeartXAI, __version__, to_png_bytes

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
ALLOWED = {"image/jpeg", "image/png", "image/bmp", "image/gif", "application/octet-stream"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart_xai.api")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = HeartXAI(MODEL_DIR)          # load once at start-up
    log.info("Loaded %d model(s) from %s", len(state["model"].models), MODEL_DIR)
    yield
    state.clear()


app = FastAPI(title="Heart Disease XAI API", version=__version__, lifespan=lifespan,
              description="Normal vs Sick (CAD) classification of single cardiac MRI slices with Grad-CAM. "
                          + DISCLAIMER)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading", "version": __version__}


@app.get("/model")
def model_info():
    return {**state["model"].info(), "disclaimer": DISCLAIMER}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="One CMR slice (JPEG/PNG)"),
                  explain: bool = Query(True, description="Return the Grad-CAM overlay as base64 PNG")):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported content type {file.content_type}")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File larger than {MAX_BYTES // (1024 * 1024)} MB")
    t0 = time.perf_counter()
    try:
        pred = await run_in_threadpool(state["model"].predict, data, explain)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out = {**pred.to_dict(), "filename": file.filename, "disclaimer": DISCLAIMER,
           "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}
    if explain:
        out["gradcam_png_base64"] = base64.b64encode(to_png_bytes(pred.overlay)).decode()
    log.info("predict file=%s label=%s prob=%.4f ms=%s", file.filename, pred.label, pred.prob_sick,
             out["latency_ms"])
    return out
