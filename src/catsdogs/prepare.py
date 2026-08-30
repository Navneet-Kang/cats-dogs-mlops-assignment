from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import discover_images, stratified_split, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))
    cfg = params["data"]
    records = discover_images(cfg["raw_dir"], cfg.get("max_samples_per_class"))
    rows = stratified_split(records, cfg["train_ratio"], cfg["val_ratio"], cfg["seed"])
    output = Path(cfg["processed_dir"]) / "manifest.csv"
    write_manifest(rows, output)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print(f"Wrote {output}: {counts}")


if __name__ == "__main__":
    main()

