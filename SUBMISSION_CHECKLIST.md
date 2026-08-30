# Final Submission Checklist

- [ ] Replace synthetic rehearsal images with the assigned Kaggle Cats and Dogs data.
- [ ] Run `dvc add data/raw`, commit `data/raw.dvc` and `.gitignore`, and configure a DVC remote if required.
- [ ] Run `dvc repro`; confirm `models/cats_dogs_sgd.joblib` and all artifacts are fresh.
- [ ] Record real test accuracy, F1, loss, confusion matrix, and loss curve; never claim demo metrics.
- [ ] Run `pytest -q --cov=src` and keep a screenshot of passing tests.
- [ ] Run MLflow and capture the run overview, parameters, metrics, and artifacts.
- [ ] Build/run Docker Compose and capture health plus prediction output.
- [ ] Push to a private/public GitHub repository and confirm CI, GHCR publishing, and CD jobs are green.
- [ ] Configure a self-hosted runner with label `cats-dogs-deploy`, or explain/show the equivalent target used.
- [ ] Capture `/metrics`, sanitized service logs, and labeled post-deployment evaluation output.
- [ ] Record a screen demo under five minutes using `DEMO_SCRIPT.md`.
- [ ] Remove secrets, `.env`, raw datasets, caches, and unrelated large files before zipping.
- [ ] Include the trained real-data model artifact in the final ZIP even though it is Git-ignored.

