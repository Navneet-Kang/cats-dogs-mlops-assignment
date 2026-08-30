"""Generate clearly synthetic images for pipeline rehearsal only, never final evaluation."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("data/raw")
rng = np.random.default_rng(42)
for label in ("cat", "dog"):
    folder = ROOT / label
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        base = np.zeros((240, 280, 3), dtype=np.uint8)
        if label == "cat":
            base[..., 0] = rng.integers(120, 220)
            base[..., 1] = rng.integers(50, 120)
        else:
            base[..., 1] = rng.integers(120, 220)
            base[..., 2] = rng.integers(50, 120)
        base = np.clip(base + rng.integers(0, 30, base.shape, dtype=np.uint8), 0, 255)
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image)
        if label == "cat":
            draw.polygon([(70, 90), (100, 30), (125, 90)], fill="white")
            draw.polygon([(155, 90), (180, 30), (210, 90)], fill="white")
            draw.ellipse((80, 70, 200, 190), outline="white", width=8)
        else:
            draw.ellipse((60, 50, 220, 200), outline="white", width=9)
            draw.ellipse((30, 70, 90, 180), fill="white")
            draw.ellipse((190, 70, 250, 180), fill="white")
        image.save(folder / f"{label}.{i}.png")

monitor = Path("data/monitoring")
monitor.mkdir(parents=True, exist_ok=True)
(monitor / "demo_cat.png").write_bytes((ROOT / "cat/cat.0.png").read_bytes())
(monitor / "demo_dog.png").write_bytes((ROOT / "dog/dog.0.png").read_bytes())
(monitor / "labels.csv").write_text(
    "image_path,true_label\n" +
    "data/monitoring/demo_cat.png,cat\n" +
    "data/monitoring/demo_dog.png,dog\n",
    encoding="utf-8",
)
print("Generated 120 synthetic rehearsal images. Replace data/raw with Kaggle data before submission.")

