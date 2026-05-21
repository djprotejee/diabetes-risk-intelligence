from __future__ import annotations

import argparse

from src.data.load import load_diabetes_data
from src.data.split import stratified_train_valid_test
from src.features.engineering import prepare_features
from src.utils.config import load_yaml, resolve_path
from src.utils.io import ensure_dir, write_json


def prepare(config_path: str = "configs/config.yaml") -> None:
    cfg = load_yaml(config_path)
    raw_df = load_diabetes_data(cfg)
    target = cfg["data"]["target"]
    prepared = prepare_features(raw_df, add_features=cfg["features"]["add_domain_features"])
    X_train, X_valid, X_test, y_train, y_valid, y_test = stratified_train_valid_test(
        prepared,
        target=target,
        test_size=cfg["data"]["split"]["test_size"],
        validation_size=cfg["data"]["split"]["validation_size"],
        random_state=cfg["project"]["random_state"],
    )
    output_dir = ensure_dir(resolve_path("artifacts/data"))
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_valid.to_csv(output_dir / "X_valid.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(target).to_csv(output_dir / "y_train.csv", index=False)
    y_valid.to_frame(target).to_csv(output_dir / "y_valid.csv", index=False)
    y_test.to_frame(target).to_csv(output_dir / "y_test.csv", index=False)
    write_json(
        output_dir / "dataset_metadata.json",
        {
            "rows": int(len(prepared)),
            "features": int(X_train.shape[1]),
            "target": target,
            "positive_rate": float(prepared[target].mean()),
            "source": cfg["data"]["source"],
        },
    )
    print(f"Prepared data written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
