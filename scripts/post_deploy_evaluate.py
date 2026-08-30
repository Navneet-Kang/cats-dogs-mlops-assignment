from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from http_helpers import post_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    with Path(args.labels).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    truth, predicted, records = [], [], []
    for row in rows:
        result = post_image(f"{args.url.rstrip('/')}/predict", row["image_path"])
        truth.append(row["true_label"]); predicted.append(result["label"])
        records.append({**row, "predicted_label": result["label"], "probabilities": result["probabilities"]})
    report = {
        "sample_count": len(rows),
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1_dog": float(f1_score(truth, predicted, pos_label="dog")),
        "records": records,
    }
    output = Path("artifacts/post_deploy_metrics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

