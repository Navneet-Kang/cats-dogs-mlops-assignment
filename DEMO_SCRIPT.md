# Screen Recording Script (target: 4 minutes 30 seconds)

## 0:00-0:30 - Repository and versions

Show `git log --oneline -5`, the project tree, `params.yaml`, `dvc.yaml`, and `dvc dag`. Say: “Git versions source; DVC versions raw data and the processed manifest.”

## 0:30-1:20 - Model and experiment tracking

Run `dvc repro` (or show the latest completed output). Open MLflow with `mlflow ui --port 5000`. Show parameters, validation metrics, test metrics, confusion matrix, loss curve, and serialized model artifact. Mention 224x224 RGB preprocessing, the 80/10/10 seeded split, and augmentation.

## 1:20-2:05 - Tests and API

Run `pytest -q --cov=src`. Open `http://localhost:8000/docs`, call `/health`, then `/predict` with one cat/dog image. Point to probabilities and label.

## 2:05-2:50 - Container

Run `docker compose up -d --build`, `docker compose ps`, and the smoke test. Briefly show pinned requirements, non-root Dockerfile, and health check.

## 2:50-3:40 - CI/CD

Open the green GitHub Actions run. Show test/build, GHCR image publishing, and deploy jobs. Explain that main deploys an immutable commit-SHA image to the self-hosted Compose target, followed by health and prediction smoke tests.

## 3:40-4:20 - Monitoring and post-deployment performance

Open `/metrics`, run `docker compose logs --tail=10 api`, then run `make evaluate`. Show request count, latency, sanitized structured logs, and `artifacts/post_deploy_metrics.json` based on labeled requests.

## 4:20-4:30 - Close

Show the rubric evidence table in `REPORT.md` and state that all requested source, configuration, model, and evidence artifacts are packaged.

