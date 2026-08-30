from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src.catsdogs.features import extract_features, preprocess_image
from src.catsdogs.model import load_bundle, predict_features

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
LOGGER = logging.getLogger("cats-dogs-api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/cats_dogs_sgd.joblib"))
REQUESTS = Counter("prediction_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["bundle"] = load_bundle(MODEL_PATH)
    LOGGER.info(json.dumps({"event": "model_loaded", "path": str(MODEL_PATH)}))
    yield
    STATE.clear()


app = FastAPI(title="Cats vs Dogs API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    LOGGER.info(json.dumps({
        "event": "request", "request_id": request_id, "method": request.method,
        "path": request.url.path, "status": response.status_code, "latency_ms": elapsed_ms,
    }))
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
def health():
    bundle = STATE.get("bundle")
    return {"status": "ok", "model_loaded": bundle is not None, "version": "1.0.0"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/x-portable-pixmap"}:
        REQUESTS.labels(status="invalid").inc()
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, BMP, or PPM image")
    try:
        raw = file.file.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MB")
        bundle = STATE["bundle"]
        array = preprocess_image(io.BytesIO(raw), bundle["image_size"])
        result = predict_features(bundle, extract_features(array, bundle["feature_size"]))
        REQUESTS.labels(status="success").inc()
        return result
    except HTTPException:
        raise
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Invalid or unreadable image") from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

