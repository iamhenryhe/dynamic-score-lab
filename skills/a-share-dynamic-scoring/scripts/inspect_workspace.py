from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_date_token(path: Path) -> tuple[int, str]:
    match = re.search(r"(20\d{2}[-_]?\d{2}[-_]?\d{2})", path.name)
    if not match:
        return (0, "")
    return (1, match.group(1).replace("_", "-"))


def choose_best(candidates: list[Path], preferred_name: str) -> Path | None:
    if not candidates:
        return None
    exact = [path for path in candidates if path.name == preferred_name]
    if exact:
        exact.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return exact[0]
    ranked = sorted(
        candidates,
        key=lambda path: (
            extract_date_token(path)[0],
            extract_date_token(path)[1],
            path.stat().st_mtime_ns,
        ),
        reverse=True,
    )
    return ranked[0]


def rel(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a dynamic-score-lab workspace.")
    parser.add_argument("--workspace", default=".", help="Workspace root to inspect")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    repo_checks = {
        "app/streamlit_app.py": (workspace / "app" / "streamlit_app.py").exists(),
        "scorer/engine.py": (workspace / "scorer" / "engine.py").exists(),
        "scorer/io.py": (workspace / "scorer" / "io.py").exists(),
        "scripts/build_app_dataset.py": (workspace / "scripts" / "build_app_dataset.py").exists(),
    }

    main_candidates = sorted(
        {
            *workspace.glob("A股主表.csv"),
            *workspace.glob("A股主表_*.csv"),
            *workspace.glob("*A股主表*.csv"),
        }
    )
    mapping_candidates = sorted(
        {
            *workspace.glob("板块映射表.csv"),
            *workspace.glob("*板块映射表*.csv"),
            *workspace.glob("data/raw/板块映射表.csv"),
        }
    )
    propagation_files = sorted((workspace / "cbd").glob("t-*.csv")) if (workspace / "cbd").exists() else []

    result = {
        "workspace": str(workspace),
        "workspace_ok": all(repo_checks.values()),
        "repo_checks": repo_checks,
        "recommended_main_csv": rel(choose_best(main_candidates, "A股主表.csv"), workspace) if main_candidates else None,
        "main_csv_candidates": [rel(path, workspace) for path in main_candidates],
        "recommended_mapping_csv": rel(choose_best(mapping_candidates, "板块映射表.csv"), workspace) if mapping_candidates else None,
        "mapping_csv_candidates": [rel(path, workspace) for path in mapping_candidates],
        "propagation_dir_exists": (workspace / "cbd").exists(),
        "propagation_file_count": len(propagation_files),
        "latest_propagation_file": rel(propagation_files[-1], workspace) if propagation_files else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
