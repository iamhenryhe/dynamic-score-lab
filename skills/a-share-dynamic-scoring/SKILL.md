---
name: a-share-dynamic-scoring
description: Use when working in the dynamic-score-lab workflow for A股动态打分. This skill covers rebuilding the app dataset from A股主表.csv or timestamped A股主表 files, syncing 板块映射表.csv, reading cbd/t-YYYY-MM-DD.csv propagation files, running the Streamlit scoring app, and updating 包含度、容量、传播度、总分 formulas and defaults inside this repo.
---

# A股动态打分

This skill assumes the current workspace is a checkout of the dynamic scoring repo and contains:

- `app/streamlit_app.py`
- `scorer/engine.py`
- `scorer/io.py`
- `scripts/build_app_dataset.py`

If those files are missing, treat the workspace as incomplete and ask the user for the correct repo checkout before making changes.

## First pass

Start by inspecting the workspace:

```bash
python3 scripts/inspect_workspace.py --workspace "$PWD"
```

Use the result to confirm:

- whether the repo structure is present
- which `A股主表` file should be used
- which `板块映射表` file should be used
- whether `cbd/t-*.csv` propagation files exist

## Expected data layout

Prefer this layout inside the workspace:

- `A股主表.csv`
- `板块映射表.csv`
- `cbd/t-YYYY-MM-DD.csv`

The app will sync the mapping file into `data/raw/板块映射表.csv` during rebuild.

If the user only provides timestamped files such as `A股主表_20260422.csv`, copy the chosen file into the workspace root as `A股主表.csv` before rebuilding. Do the same for the chosen mapping file if needed.

Do not depend on files outside the workspace when a workspace-local copy can be used.

## Standard workflow

1. Inspect the workspace and identify the intended input files.
2. Normalize filenames into the expected workspace layout when needed.
3. Rebuild the app dataset:

```bash
python3 scripts/build_app_dataset.py
```

4. Validate Python modules after code or formula changes:

```bash
python3 -m py_compile app/streamlit_app.py scorer/engine.py scorer/io.py scorer/__init__.py
```

5. When the user wants to test locally, start the app:

```bash
streamlit run app/streamlit_app.py
```

## Formula ownership

Keep formula logic in `scorer/engine.py`.

Use `app/streamlit_app.py` for:

- sidebar controls
- default UI values
- comparison tables
- chart rendering
- compatibility fallbacks needed for Streamlit Cloud

When changing default scoring parameters, update both:

- dataclass defaults in `scorer/engine.py`
- matching input defaults in `app/streamlit_app.py`

Otherwise the formula reference and the UI defaults will drift apart.

## Repo-specific behavior

- Propagation history comes from `cbd/t-YYYY-MM-DD.csv`.
- Propagation dates are matched to the nearest previous trading date from `A股主表.csv`.
- The app dataset is rebuilt into `data/derived/a_share_main_app.parquet`.
- The total-score history chart uses propagation dates as the x-axis and shows the matched trade date in hover details.

## Common requests this skill should handle

- “用最新 A股主表 更新动态打分”
- “同步新的 板块映射表 和传播度文件”
- “把传播度放大倍数改成 1.8，默认权重改成 60/20/20”
- “重建 parquet 并启动网站”
- “给总分页加 TopN 历史趋势图”

## Guardrails

- Keep edits inside the current workspace.
- Do not modify unrelated directories such as deployment folders outside this repo.
- Prefer repo-local data copies over hardcoded external paths.
- If data files are ambiguous, surface the exact candidates and the selection rule you used.
