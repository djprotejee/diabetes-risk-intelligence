from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_joblib(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    joblib.dump(payload, target)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))
