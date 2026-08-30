# Assignment 2 Technical Report - End-to-End MLOps

## Architecture and design choices

The solution implements binary cat-versus-dog classification as a versioned, testable, containerized service. Source is versioned in Git. Raw and processed datasets are versioned with DVC. All inputs become 224x224 RGB arrays in [0,1]. The CPU-friendly baseline then downsamples each standardized image to 32x32, appends channel statistics, and trains an online SGD logistic classifier with seeded augmentation. This is a reproducible baseline; a CNN can replace it without changing the serving contract.

Training records parameters, per-epoch validation loss/accuracy, test accuracy/F1/log-loss, a loss curve, confusion matrix, and the serialized joblib model in MLflow. The model bundle contains preprocessing configuration and metadata to minimize training-serving skew.

FastAPI exposes `/health`, `/predict`, and Prometheus `/metrics`. The prediction endpoint validates type and size, returns probabilities plus the winning label, and logs request metadata without logging image bytes or filenames. Counters and latency histograms support operational monitoring.

The Docker image is non-root, pinned, health-checked, and deployable through Docker Compose. GitHub Actions tests the source on every push and pull request, builds the image, publishes main-branch images to GHCR, and deploys an immutable SHA tag on a labeled self-hosted Windows runner. Health and prediction smoke tests fail the deployment job on error.

## Dataset and experiment

The assigned Kaggle Cats and Dogs data was arranged under `data/raw/cat` and `data/raw/dog`. The deterministic preparation stage generated a manifest containing 8,000 training images, 1,000 validation images, and 1,000 test images. Random generators use seed 42, and parameters are stored in `params.yaml`.

## Offline model results

The MLflow run recorded the following held-out test results:

| Metric | Result |
|---|---:|
| Test samples | 1,000 |
| Test accuracy | 0.50 |
| Test F1 | 0.04 |
| Test log loss | 16.53 |

Validation accuracy remained close to chance across epochs. These results are reported honestly as baseline performance and are not presented as production-quality classification accuracy.

## Step 8 - Post-deployment evaluation

The deployed FastAPI model was evaluated through the production endpoint at `http://localhost:8001` using 20 labeled Cats and Dogs images.

| Metric | Result |
|---|---:|
| Sample count | 20 |
| Accuracy | 0.50 |
| Dog-class F1 score | 0.00 |

The end-to-end deployment and evaluation pipeline completed successfully. However, the deployed baseline strongly favored the cat class and failed to identify the dog samples reliably. This demonstrates that the MLOps pipeline is functional while the baseline model requires improved features, class balancing, hyperparameter tuning, or a convolutional neural network to improve predictive performance.

The complete per-image predictions and probabilities were generated in `artifacts/post_deploy_metrics.json`.

## Rubric evidence map

| Module | Implementation | Evidence captured |
|---|---|---|
| M1 | Git, `dvc.yaml`, `params.yaml`, preparation/training modules, MLflow, model and plots | Git history, DVC pipeline, MLflow parameters, metrics, plots and model artifact |
| M2 | FastAPI, pinned requirements and Dockerfile | API health/prediction responses and running container |
| M3 | pytest tests, GitHub Actions and GHCR publishing | Passing tests, green build workflow and GHCR image |
| M4 | Compose, immutable SHA deployment and smoke test | Healthy Compose services and green self-hosted deployment job |
| M5 | Structured JSON logs, Prometheus metrics and post-deployment evaluation | `/metrics`, application logs and post-deployment metrics output |

## Reproducibility and limitations

Dependency versions, model preprocessing settings and random seed 42 are recorded. DVC tracks data and pipeline metadata without committing raw image files to Git. MLflow records experiment parameters, metrics and artifacts. Docker and GitHub Actions provide repeatable build and deployment steps.

The primary limitation is model quality rather than infrastructure: the compact linear baseline exhibits class collapse and near-chance accuracy. Future work should use a CNN or transfer-learning model, stronger stratified evaluation, class weighting, hyperparameter optimization, and a larger labeled post-deployment evaluation set. The current report distinguishes successful MLOps engineering from limited predictive performance.
