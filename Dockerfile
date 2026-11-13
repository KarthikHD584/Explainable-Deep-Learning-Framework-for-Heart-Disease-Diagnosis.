# Heart Disease XAI API (CPU).  Build:  docker build -t heart-xai .
# Run API:   docker run -p 8000:8000 heart-xai
# Run demo:  docker run -p 7860:7860 heart-xai python app.py
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 MODEL_DIR=/app/models MPLCONFIGDIR=/tmp/mpl
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY models/ models/
# Pin TensorFlow/Keras to the training versions when the export wrote a lock file
RUN if [ -f models/requirements.lock.txt ]; then pip install -r models/requirements.lock.txt; fi

COPY heart_xai/ heart_xai/
COPY api.py app.py predict.py ./

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
