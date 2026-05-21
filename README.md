# Diabetes Risk Intelligence

Production-style PoC for diabetes risk prediction using the CDC Diabetes Health Indicators dataset.

The system implements an end-to-end ML product flow:

- data loading from local CSV or UCI;
- domain feature engineering;
- baseline, bagging, boosting, soft-voting, and stacking models;
- validation/test metrics, threshold tuning, reliability curves, error slices, and SHAP plots;
- FastAPI service for inference and model diagnostics;
- Streamlit dashboard for prediction, model comparison, EDA, and prediction history;
- PostgreSQL logging through SQLAlchemy;
- Alembic database migrations;
- CORS middleware, `/health` and `/ready` endpoints;
- Gunicorn with Uvicorn workers for the API container;
- PowerShell helper scripts for Windows workflow;
- Docker Compose orchestration.

## Dataset

Expected file:

```text
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
```

The downloaded archive usually contains three files. This project uses the full binary file:

```text
data/archive/diabetes_binary_health_indicators_BRFSS2015.csv
```

The balanced `5050split` file can be used for a separate sensitivity experiment, and `diabetes_012` is the multiclass variant.

If the file is missing, `trainer` can fetch UCI dataset ID `891` when `ucimlrepo` and network access are available.

Source page:

```text
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
```

## Local Run

```bash
make setup
make full
make api
make ui
```

API: `http://localhost:8000`  
UI: `http://localhost:8501`

## Docker Run

```bash
cp .env.example .env
make train-full-docker
make up
```

If ports `8000` or `8501` are already used by another project, run with custom host ports:

```bash
API_PORT=8010 UI_PORT=8510 docker compose up --build
```

PowerShell:

```powershell
$env:API_PORT='8010'; $env:UI_PORT='8510'; docker compose up --build
```

Windows helper scripts:

```powershell
.\scripts\up.ps1
.\scripts\up.ps1 -Detached
.\scripts\logs.ps1
.\scripts\down.ps1
.\scripts\status.ps1
```

Services:

- `db` - PostgreSQL 16;
- `api` - FastAPI prediction service served by Gunicorn and UvicornWorker;
- `ui` - Streamlit dashboard;
- `trainer` - optional one-shot model training container.

## API Readiness and Migrations

The project includes Alembic migrations for a production-style schema workflow. For this PoC, the API creates the required table on startup with SQLAlchemy so existing local databases do not block the demo. Manual Alembic migrations can still be run separately when needed.

Endpoints:

- `GET /health` - process liveness;
- `GET /ready` - database and model registry readiness;
- `POST /predict` - prediction and PostgreSQL logging;
- `GET /history` - latest prediction logs.

Environment variables:

```text
DATABASE_URL=postgresql+psycopg2://diabetes:diabetes@db:5432/diabetes_risk
CORS_ORIGINS=http://localhost:8501,http://localhost:8510
RUN_MIGRATIONS=false
LOG_LEVEL=INFO
```

## Training Modes

The project keeps one simple full command, but also supports cheaper partial reruns:

```bash
make fast                         # quick smoke training: fewer models, no SHAP, no ensembles
make full                         # complete report-grade training
make train MODEL=lightgbm          # rerun one model and refresh manifest/comparison
make train-no-explain MODEL=xgboost # rerun one model without SHAP and ensembles
make finalize                     # rebuild manifest.json and model_comparison.csv from existing metrics
```

Docker equivalents:

```bash
make train-fast-docker
make train-full-docker
docker compose --profile train run --rm trainer python -m src.models.train --model lightgbm --skip-explain --skip-ensembles
```

PowerShell equivalents:

```powershell
.\scripts\train-fast.ps1
.\scripts\train-full.ps1
.\scripts\finalize.ps1
```

## Models

- Logistic Regression;
- Random Forest;
- Extra Trees;
- HistGradientBoosting;
- LightGBM;
- XGBoost;
- Soft Voting Ensemble;
- Stacking Ensemble.

## Main API Endpoints

- `GET /health`
- `GET /ready`
- `POST /predict`
- `GET /history`
- `GET /model/comparison`
- `GET /model/metrics/{model_name}`
- `GET /dataset/preview`
- `GET /artifacts/manifest`

## Artifacts

Training writes artifacts to:

```text
artifacts/models/
artifacts/reports/
```

Important reports:

- `model_comparison.csv`;
- `metrics_<model>.json`;
- `roc_curve_<model>_test.png`;
- `pr_curve_<model>_test.png`;
- `reliability_<model>_test.png`;
- `error_slices_<model>.csv`;
- `shap_summary_<model>.png`.
