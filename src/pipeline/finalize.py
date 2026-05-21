from __future__ import annotations

from src.models.train import finalize_artifacts


def main() -> None:
    manifest = finalize_artifacts()
    print(f"Finalized {len(manifest.get('models', {}))} model entries.")


if __name__ == "__main__":
    main()

