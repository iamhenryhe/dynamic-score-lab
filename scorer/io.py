from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RAW_MAIN_CSV_PATH = WORKSPACE_ROOT / "A股主表.csv"
EXTERNAL_MAPPING_SOURCE = Path("/Users/zijiehe/Desktop/a/板块映射表.csv")
WORKSPACE_MAPPING_PATH = WORKSPACE_ROOT / "板块映射表.csv"
APP_DATA_DIR = WORKSPACE_ROOT / "data" / "derived"
RAW_DATA_DIR = WORKSPACE_ROOT / "data" / "raw"
LOCAL_MAPPING_PATH = RAW_DATA_DIR / "板块映射表.csv"
APP_PARQUET_PATH = APP_DATA_DIR / "a_share_main_app.parquet"
APP_SUMMARY_PATH = APP_DATA_DIR / "dataset_summary.json"

APP_COLUMNS = [
    "交易日期",
    "股票代码",
    "股票简称",
    "收盘价",
    "成交额",
    "涨跌幅(%)",
    "换手率(%)",
    "总市值(亿)",
    "是否涨停",
    "连板数",
]


def _normalize_stock_code(series: pd.Series) -> pd.Series:
    normalized = (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.replace("<NA>", "", regex=False)
        .str.zfill(6)
    )
    return normalized.fillna("")


def _normalize_code_value(value: object) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip().upper().replace(" ", "")
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]
    match = re.fullmatch(r"(\d{6})\.(SZ|SH|BJ)", raw)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{1,6}", raw):
        return raw.zfill(6)
    return ""


def resolve_mapping_source() -> Path:
    candidates = [
        WORKSPACE_MAPPING_PATH,
        LOCAL_MAPPING_PATH,
        EXTERNAL_MAPPING_SOURCE,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"找不到板块映射表，已检查: {[str(path) for path in candidates]}"
    )


def ensure_mapping_file() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = resolve_mapping_source()
    if source == LOCAL_MAPPING_PATH:
        return LOCAL_MAPPING_PATH
    source_bytes = source.read_bytes()
    if not LOCAL_MAPPING_PATH.exists() or LOCAL_MAPPING_PATH.read_bytes() != source_bytes:
        LOCAL_MAPPING_PATH.write_bytes(source_bytes)
    return LOCAL_MAPPING_PATH


def build_app_dataset(
    source_csv: Path = RAW_MAIN_CSV_PATH,
    output_parquet: Path = APP_PARQUET_PATH,
    summary_path: Path = APP_SUMMARY_PATH,
) -> dict[str, object]:
    if not source_csv.exists():
        raise FileNotFoundError(f"找不到主表文件: {source_csv}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    ensure_mapping_file()

    df = pd.read_csv(source_csv, usecols=APP_COLUMNS, encoding="utf-8-sig", low_memory=False)
    df["交易日期"] = pd.to_datetime(df["交易日期"], errors="coerce")
    df = df.dropna(subset=["交易日期"]).copy()
    df["股票代码"] = _normalize_stock_code(df["股票代码"])
    df["股票简称"] = df["股票简称"].astype("string").fillna("").str.strip()

    numeric_cols = ["收盘价", "成交额", "涨跌幅(%)", "换手率(%)", "总市值(亿)", "是否涨停", "连板数"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["股票代码", "交易日期"]).reset_index(drop=True)
    df.to_parquet(output_parquet, index=False)

    summary = {
        "rows": int(len(df)),
        "columns": APP_COLUMNS,
        "unique_trade_dates": int(df["交易日期"].nunique()),
        "unique_stock_codes": int(df["股票代码"].nunique()),
        "min_trade_date": df["交易日期"].min().strftime("%Y-%m-%d"),
        "max_trade_date": df["交易日期"].max().strftime("%Y-%m-%d"),
        "parquet_path": str(output_parquet),
        "mapping_path": str(LOCAL_MAPPING_PATH),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_app_dataset(parquet_path: Path = APP_PARQUET_PATH) -> pd.DataFrame:
    if not parquet_path.exists():
        build_app_dataset(output_parquet=parquet_path)
    return pd.read_parquet(parquet_path)


def load_mapping_table(mapping_path: Path | None = None) -> pd.DataFrame:
    path = mapping_path or ensure_mapping_file()
    df = pd.read_csv(path, dtype="string", encoding="utf-8-sig", low_memory=False)
    rename_map = {}
    if "简称" not in df.columns and "股票简称" in df.columns:
        rename_map["股票简称"] = "简称"
    if "代码" not in df.columns and "股票代码" in df.columns:
        rename_map["股票代码"] = "代码"
    df = df.rename(columns=rename_map)

    required = {"板块", "简称", "代码"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"板块映射表缺少列: {sorted(missing)}")

    df = df[["板块", "简称", "代码"]].copy()
    df["板块"] = df["板块"].astype("string").fillna("").str.strip()
    df["简称"] = df["简称"].astype("string").fillna("").str.strip()
    df["股票代码"] = df["代码"].map(_normalize_code_value)
    df = df[df["股票代码"] != ""].drop_duplicates(["板块", "股票代码"]).reset_index(drop=True)
    return df[["板块", "简称", "股票代码"]]


def load_dataset_summary(summary_path: Path = APP_SUMMARY_PATH) -> dict[str, object]:
    if not summary_path.exists():
        return build_app_dataset(summary_path=summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))
