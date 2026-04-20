from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scorer.io import APP_PARQUET_PATH, APP_SUMMARY_PATH, RAW_MAIN_CSV_PATH, build_app_dataset


def main() -> None:
    summary = build_app_dataset()
    print(f"source: {RAW_MAIN_CSV_PATH}")
    print(f"parquet: {APP_PARQUET_PATH}")
    print(f"summary: {APP_SUMMARY_PATH}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
