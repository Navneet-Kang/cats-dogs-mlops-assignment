from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss

from .data import discover_images, read_manifest, stratified_split, write_manifest
from .features import augment_image, extract_features, preprocess_image
from .model import save_bundle


def rows_to_xy(rows: list[dict[str, str]], image_size: int, feature_size: int, augment: bool, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for row in rows:
        array = preprocess_image(row["path"], image_size)
        if augment:
            array = augment_image(array, rng)
        xs.append(extract_features(array, feature_size))
        ys.append(row["label"])
    return np.stack(xs), np.asarray(ys)


def maybe_mlflow(params: dict):
    try:
        import mlflow
        mlflow.set_tracking_uri(params["tracking"]["uri"])
        mlflow.set_experiment(params["tracking"]["experiment"])
        return mlflow, mlflow.start_run(run_name="sgd-logistic-baseline")
    except ImportError:
        print("MLflow not installed locally; artifacts are still written to artifacts/. Install requirements.txt to enable it.")
        return None, nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))
    data_cfg, model_cfg, artifact_cfg = params["data"], params["model"], params["artifacts"]
    manifest = Path(args.manifest or Path(data_cfg["processed_dir"]) / "manifest.csv")
    if not manifest.exists():
        records = discover_images(data_cfg["raw_dir"], data_cfg.get("max_samples_per_class"))
        rows = stratified_split(records, data_cfg["train_ratio"], data_cfg["val_ratio"], data_cfg["seed"])
        write_manifest(rows, manifest)
    rows = read_manifest(manifest)
    splits = {name: [r for r in rows if r["split"] == name] for name in ("train", "val", "test")}
    if any(not value for value in splits.values()):
        raise ValueError(f"All splits must be non-empty: { {k: len(v) for k, v in splits.items()} }")

    image_size, feature_size = data_cfg["image_size"], data_cfg["feature_size"]
    x_val, y_val = rows_to_xy(splits["val"], image_size, feature_size, False, data_cfg["seed"])
    x_test, y_test = rows_to_xy(splits["test"], image_size, feature_size, False, data_cfg["seed"])
    model = SGDClassifier(loss="log_loss", alpha=model_cfg["alpha"], random_state=data_cfg["seed"])
    history = []
    mlflow, run = maybe_mlflow(params)
    output_dir = Path(artifact_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with run:
        if mlflow:
            mlflow.log_params({**data_cfg, **model_cfg, "model": "SGDClassifier-log_loss"})
        for epoch in range(1, model_cfg["epochs"] + 1):
            x_train, y_train = rows_to_xy(splits["train"], image_size, feature_size, True, data_cfg["seed"] + epoch)
            indices = np.random.default_rng(data_cfg["seed"] + epoch).permutation(len(y_train))
            for start in range(0, len(indices), model_cfg["batch_size"]):
                batch = indices[start : start + model_cfg["batch_size"]]
                model.partial_fit(x_train[batch], y_train[batch], classes=np.asarray(["cat", "dog"]))
            val_prob = model.predict_proba(x_val)
            val_pred = model.predict(x_val)
            record = {
                "epoch": epoch,
                "val_log_loss": float(log_loss(y_val, val_prob, labels=["cat", "dog"])),
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
            }
            history.append(record)
            if mlflow:
                mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=epoch)
            print(record)

        test_prob = model.predict_proba(x_test)
        test_pred = model.predict(x_test)
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_f1": float(f1_score(y_test, test_pred, pos_label="dog")),
            "test_log_loss": float(log_loss(y_test, test_prob, labels=["cat", "dog"])),
            "train_samples": len(splits["train"]),
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        ConfusionMatrixDisplay.from_predictions(y_test, test_pred, labels=["cat", "dog"], cmap="Blues")
        plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close()
        plt.plot([r["epoch"] for r in history], [r["val_log_loss"] for r in history], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Validation log loss"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=160); plt.close()
        metadata = {
            "data_source": data_cfg.get("source", "unspecified"),
            "metrics": metrics,
            "algorithm": "SGD logistic baseline",
            "dependencies": {
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
        }
        model_path = save_bundle(model, artifact_cfg["model_path"], image_size=image_size, feature_size=feature_size, metadata=metadata)
        if mlflow:
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
            mlflow.log_artifact(str(model_path), artifact_path="model")
        print(json.dumps(metrics, indent=2))
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
