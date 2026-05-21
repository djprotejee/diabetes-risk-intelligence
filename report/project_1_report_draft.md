# Індивідуальний проєкт 1

## Тема

Diabetes Risk Intelligence: ансамблева система прогнозування ризику діабету на основі CDC Diabetes Health Indicators.

## Мета

Розробити production-style PoC для задачі медичної класифікації: від підготовки даних і навчання моделей до REST API, Streamlit dashboard, PostgreSQL-логування та Docker Compose deployment.

## Джерело даних

Використано CDC Diabetes Health Indicators, UCI dataset ID 891. Сторінка UCI вказує 253680 instances, 21 feature, subject area Health and Medicine, task Classification. Джерело: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.

## Архітектура

Проєкт складається з таких компонентів:

- `src/models` - навчання моделей, метрики, пороги, графіки, SHAP;
- `src/api` - FastAPI сервіс прогнозування;
- `src/ui` - Streamlit dashboard;
- `src/db` - SQLAlchemy-модель для PostgreSQL-логування;
- `alembic` - версіоновані міграції схеми бази даних;
- `docker-compose.yml` - `db`, `api`, `ui`, `trainer`;
- `artifacts` - моделі, метрики, графіки, manifest.

API-контейнер запускається через Gunicorn з UvicornWorker. У застосунку додано CORS middleware, structured logging, `/health` для liveness-перевірки та `/ready` для перевірки готовності бази даних і model registry.

## Моделі

У проєкті реалізовано:

- Logistic Regression як baseline;
- Random Forest;
- Extra Trees;
- HistGradientBoostingClassifier;
- LightGBM;
- XGBoost;
- Soft Voting Ensemble;
- Stacking Ensemble.

За результатами тестового порівняння найкращою основною моделлю для інференсу обрано XGBoost. Stacking Ensemble залишено як дослідницьку ансамблеву модель, однак за PR-AUC та якістю ймовірностей він не перевершив XGBoost.

Для XGBoost реалізовано два режими прийняття рішення: `balanced_f1` з порогом 0.235 та `recall_oriented` з порогом 0.135. Другий режим підвищує recall на валідаційній вибірці приблизно до 0.80 і призначений для сценарію скринінгу, де важливо зменшити кількість пропущених ризикових пацієнтів.

## Оцінювання

Для моделей обчислюються:

- ROC-AUC;
- PR-AUC;
- F1;
- Precision;
- Recall;
- Balanced Accuracy;
- Brier Score;
- Log Loss;
- Confusion Matrix.

Додатково генеруються ROC/PR curves, reliability curve, error slices за групами ознак і SHAP summary для підтримуваних моделей.

## Streamlit

Інтерфейс містить вкладки:

- Predict - введення профілю пацієнта і прогноз ризику;
- Model Lab - порівняння моделей і графіки артефактів;
- Dataset - превʼю даних, статистика, розподіли;
- History - історія прогнозів із PostgreSQL.

## Deployment

Docker Compose піднімає PostgreSQL, FastAPI та Streamlit. Навчання можна виконати локально через `make full` або в контейнері через `make train-full-docker`. Для Windows додано PowerShell-скрипти `scripts/up.ps1`, `scripts/down.ps1`, `scripts/logs.ps1`, `scripts/train-fast.ps1`, `scripts/train-full.ps1`.

Схема бази даних керується через Alembic. Перша міграція створює таблицю `prediction_logs`, де зберігаються модель, ймовірність, поріг, клас ризику, JSON-запит і JSON-відповідь.

## Висновок

Результатом є повний PoC, який виконує вимоги методичних матеріалів щодо Python, Docker, Streamlit, ML pipeline і демонстраційної системи. Проєкт розширено дослідницьким блоком: ансамблеві моделі, калібрування ймовірностей, аналіз похибок, інтерпретація через SHAP та збереження історії прогнозів у PostgreSQL.
