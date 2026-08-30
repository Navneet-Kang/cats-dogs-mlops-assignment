.PHONY: demo-data train test api smoke evaluate docker-build docker-up docker-down clean

demo-data:
	python scripts/generate_demo_data.py

train:
	python -m src.catsdogs.train --params params.yaml

test:
	pytest -q --cov=src --cov-report=term-missing

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

smoke:
	python scripts/smoke_test.py --url http://localhost:8000 --image data/monitoring/demo_cat.png

evaluate:
	python scripts/post_deploy_evaluate.py --url http://localhost:8000 --labels data/monitoring/labels.csv

docker-build:
	docker build -t cats-dogs-api:local .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache mlruns data/processed artifacts/*.png artifacts/*.json

