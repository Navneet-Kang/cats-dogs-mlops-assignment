from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def preprocess_image(source: str | Path | BinaryIO, image_size: int = 224) -> np.ndarray:
    """Convert any supported input to a normalized 224x224 RGB float array."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (image_size, image_size, 3):
        raise ValueError(f"Unexpected processed shape: {array.shape}")
    return array


def augment_image(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply deterministic-under-seed horizontal flip and mild brightness jitter."""
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255), mode="RGB")
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    return np.asarray(image, dtype=np.float32) / 255.0


def extract_features(array: np.ndarray, feature_size: int = 32) -> np.ndarray:
    """Create a compact baseline vector from the standardized RGB image."""
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB array")
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255), mode="RGB")
    small = image.resize((feature_size, feature_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
    channel_mean = array.mean(axis=(0, 1))
    channel_std = array.std(axis=(0, 1))
    return np.concatenate([pixels, channel_mean, channel_std]).astype(np.float32)

