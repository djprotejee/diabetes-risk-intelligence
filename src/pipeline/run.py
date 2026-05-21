from __future__ import annotations

import argparse

from src.models.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config, mode=args.mode)


if __name__ == "__main__":
    main()

