# Assignment 2 Technical Report - End-to-End MLOps

## Architecture and design choices

The solution implements binary cat-versus-dog classification as a versioned, testable, containerized service. Source is versioned in Git. Raw and processed datasets are versioned with DVC. All inputs become 224x224 RGB arrays in [0,1]. The CPU-friendly baseline then downsamples the standardized image to 32x32, appends channel statistics, and trains an online SGD logistic classifier with seeded augmentation. This is a reproducible baseline; a CNN can replace it without changing the serving contract.

Training records parameters, per-epoch validation loss/accuracy, test accuracy/F1/log-loss, loss curve, confusion matrix, and the serialized joblib model in MLflow. The model bundle contains preprocessing configuration and metadata so training-serving skew is minimized.

FastAPI exposes `/health`, `/predict`, and Prometheus `/metrics`. The prediction endpoint validates type and size, returns probabilities plus the winning label, and logs request metadata without logging image bytes or filenames. Counters and latency histograms support operational monitoring.

The Docker image is non-root, pinned, health-checked, and deployable through Docker Compose. GitHub Actions tests the source on every push/PR, builds the image, publishes main-branch images to GHCR, then deploys an immutable SHA tag on a labeled self-hosted runner. Health and prediction smoke tests fail the deployment job on error.

## Rubric evidence map

| Module | Implementation | Evidence to capture |
|---|---|---|
| M1 | Git, `dvc.yaml`, `params.yaml`, prepare/train modules, MLflow, model + plots | `git log`, `dvc dag`, MLflow run page |
| M2 | FastAPI, pinned requirements, Dockerfile | `/docs`, `docker build`, curl prediction |
| M3 | pytest tests, GitHub Actions, GHCR push | Green Actions run and Packages page |
| M4 | Compose, immutable SHA deploy, smoke test | `docker compose ps`, deploy job, PASS output |
| M5 | JSON logs, Prometheus metrics, post-deploy evaluation | `/metrics`, logs, metrics JSON |

## Reproducibility and limitations

All random generators use seed 42. Dependency versions and model preprocessing settings are pinned. The included demo data/model are synthetic rehearsal assets and must not be reported as Kaggle performance. Before submission, run the same pipeline on the assigned Kaggle dataset, commit DVC metadata (not raw images), and use the resulting metrics/model artifacts.

