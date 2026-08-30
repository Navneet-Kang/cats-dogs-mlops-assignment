from __future__ import annotations

import argparse
import sys
from pathlib import Path

from http_helpers import get_json, post_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    health = get_json(f"{args.url.rstrip('/')}/health")
    if health.get("status") != "ok" or not health.get("model_loaded"):
        sys.exit(f"Health check failed: {health}")
    prediction = post_image(f"{args.url.rstrip('/')}/predict", Path(args.image))
    if prediction.get("label") not in {"cat", "dog"}:
        sys.exit(f"Prediction check failed: {prediction}")
    probabilities = prediction.get("probabilities", {})
    if abs(sum(probabilities.values()) - 1.0) > 1e-5:
        sys.exit(f"Probabilities invalid: {probabilities}")
    print({"health": health, "prediction": prediction, "smoke_test": "PASS"})


if __name__ == "__main__":
    main()

