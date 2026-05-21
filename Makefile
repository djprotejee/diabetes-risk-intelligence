.PHONY: setup prepare train train-no-explain fast full all finalize api ui up down logs test clean train-docker train-fast-docker train-full-docker

setup:
	pip install -r requirements.txt

prepare:
	python -m src.pipeline.prepare

train:
	python -m src.models.train $(if $(MODEL),--model $(MODEL),)

train-no-explain:
	python -m src.models.train $(if $(MODEL),--model $(MODEL),) --skip-explain --skip-ensembles

fast:
	python -m src.pipeline.run --mode fast

full:
	python -m src.pipeline.run --mode full

all: full

finalize:
	python -m src.pipeline.finalize

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

api-prod:
	gunicorn src.api.app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1 --timeout 120 --access-logfile - --error-logfile -

ui:
	streamlit run src/ui/app.py

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

train-docker:
	docker compose --profile train run --rm trainer

train-fast-docker:
	docker compose --profile train run --rm trainer python -m src.pipeline.run --mode fast

train-full-docker:
	docker compose --profile train run --rm trainer python -m src.pipeline.run --mode full

test:
	pytest -q

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('artifacts/models'), pathlib.Path('artifacts/reports')]]"
