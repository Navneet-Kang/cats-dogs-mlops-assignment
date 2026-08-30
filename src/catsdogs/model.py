from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def save_bundle(model: Any, path: str | Path, *, image_size: int, feature_size: int, metadata: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": ["cat", "dog"],
            "image_size": image_size,
            "feature_size": feature_size,
            "metadata": metadata,
        },
        output,
    )
    return output


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    required = {"model", "classes", "image_size", "feature_size", "metadata"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing {sorted(missing)}")
    return bundle


def predict_features(bundle: dict, features: np.ndarray) -> dict:
    row = np.asarray(features, dtype=np.float32).reshape(1, -1)
    probabilities = bundle["model"].predict_proba(row)[0]
    classes = [str(value) for value in bundle["model"].classes_]
    prob_map = {label: float(prob) for label, prob in zip(classes, probabilities)}
    label = max(prob_map, key=prob_map.get)
    return {"label": label, "probabilities": prob_map}

