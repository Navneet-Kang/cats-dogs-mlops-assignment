from __future__ import annotations

import csv
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("cat", "dog")


def infer_label(path: Path) -> str | None:
    """Infer cat/dog from the parent directory or Kaggle-style filename."""
    parent = path.parent.name.lower()
    name = path.name.lower()
    for label in CLASSES:
        if parent in {label, f"{label}s"} or name.startswith(f"{label}.") or name.startswith(f"{label}_"):
            return label
    return None


def discover_images(raw_dir: str | Path, max_per_class: int | None = None) -> list[tuple[Path, str]]:
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"Dataset not found: {raw}. See README 'Get the data'.")
    grouped: dict[str, list[Path]] = {label: [] for label in CLASSES}
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            label = infer_label(path)
            if label:
                grouped[label].append(path)
    if not all(grouped.values()):
        counts = {k: len(v) for k, v in grouped.items()}
        raise ValueError(f"Both classes are required. Found {counts} in {raw}")
    records: list[tuple[Path, str]] = []
    for label, paths in grouped.items():
        records.extend((p, label) for p in paths[:max_per_class])
    return records


def stratified_split(
    records: list[tuple[Path, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, str]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must leave a non-empty test proportion")
    rng = random.Random(seed)
    output: list[dict[str, str]] = []
    for label in CLASSES:
        paths = [p for p, y in records if y == label]
        rng.shuffle(paths)
        n = len(paths)
        train_end = max(1, int(n * train_ratio))
        val_end = min(n, train_end + max(1, int(n * val_ratio)))
        for i, path in enumerate(paths):
            split = "train" if i < train_end else "val" if i < val_end else "test"
            output.append({"path": str(path), "label": label, "split": split})
    rng.shuffle(output)
    return output


def write_manifest(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

