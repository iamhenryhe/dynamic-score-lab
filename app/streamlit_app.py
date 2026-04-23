from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scorer.engine import (  # noqa: E402
    CapacityConfig,
    CoverageConfig,
    TotalConfig,
    calculate_capacity_scores,
    calculate_coverage_scores,
    calculate_total_scores,
    capacity_formula_markdown,
    coverage_formula_markdown,
    default_capacity_config,
    default_coverage_config,
    default_total_config,
    ensure_feature_columns,
    total_formula_markdown,
)
from scorer.io import (  # noqa: E402
    APP_PARQUET_PATH,
    build_app_dataset,
    ensure_mapping_file,
    load_app_dataset,
    load_dataset_summary,
    load_mapping_table,
)


st.set_page_config(page_title="因子参数调整", page_icon="📊", layout="wide")


def format_weight_percent(weight: float) -> str:
    return f"{float(weight) * 100:.0f}%"


def ensure_capacity_config_compat(config: CapacityConfig) -> CapacityConfig:
    defaults = {
        "turnover_share_full_score_ratio": 0.10,
        "turnover_share_weight": 0.5,
        "limit_up_count_weight": 0.3,
        "limit_up_market_cap_weight": 0.2,
        "limit_up_turnover_weight": 0.2,
        "final_dynamic_weight": 0.8,
        "final_static_weight": 0.2,
        "score_floor": 0.0,
        "score_ceiling": 100.0,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            object.__setattr__(config, key, value)
    return config


def clip_series_local(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=lower, upper=upper)


def weighted_sum_series(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    valid_parts = [
        (pd.to_numeric(series, errors="coerce").fillna(0), float(weight))
        for series, weight in parts
        if float(weight) > 0
    ]
    if not valid_parts:
        reference = parts[0][0] if parts else pd.Series(dtype="float64")
        return pd.Series(0.0, index=reference.index, dtype="float64")
    return sum(series * weight for series, weight in valid_parts)


def capacity_formula_markdown_compat(config: CapacityConfig) -> str:
    config = ensure_capacity_config_compat(config)
    return (
        f"- 静态容量分：`板块总市值_亿 / {config.static_market_cap_divisor} * 100`\n"
        f"- 成交占比得分：`成交占比 / {format_weight_percent(config.turnover_share_full_score_ratio)} * 100`\n"
        f"- 涨停家数得分：`涨停家数 * {config.limit_up_count_multiplier}`\n"
        f"- 涨停股总市值得分：`涨停股总市值_亿 / {config.limit_up_market_cap_divisor} * 100`\n"
        f"- 涨停股成交额得分：`涨停股成交额_亿 / {config.limit_up_turnover_divisor} * 100`\n"
        f"- 动态容量分：`成交占比得分 * {format_weight_percent(config.turnover_share_weight)} + 涨停家数得分 * {format_weight_percent(config.limit_up_count_weight)} + 涨停股总市值得分 * {format_weight_percent(config.limit_up_market_cap_weight)} + 涨停股成交额得分 * {format_weight_percent(config.limit_up_turnover_weight)}`\n"
        f"- 综合容量分：`动态容量分 * {format_weight_percent(config.final_dynamic_weight)} + 静态容量分 * {format_weight_percent(config.final_static_weight)}`"
    )


def calculate_capacity_scores_compat(
    main_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: CapacityConfig,
) -> pd.DataFrame:
    config = ensure_capacity_config_compat(config)
    try:
        scored = calculate_capacity_scores(main_df, mapping_df, trade_date, config)
        if {"成交占比得分", "动态容量分", "综合容量分"}.issubset(scored.columns):
            return scored
    except Exception:
        pass

    selected_date = pd.Timestamp(trade_date).normalize()
    selected_main = main_df[main_df["交易日期"] == selected_date].copy()
    merged = selected_main.merge(mapping_df, on="股票代码", how="inner", suffixes=("_主表", "_映射"))
    mapping_name_col = "简称_映射" if "简称_映射" in merged.columns else "简称"
    merged["股票简称"] = merged[mapping_name_col].where(
        merged[mapping_name_col].astype("string").fillna("").str.strip() != "",
        merged["股票简称"],
    )
    merged["成交额(亿)"] = pd.to_numeric(merged["成交额"], errors="coerce").fillna(0) / 100000000
    total_turnover = pd.to_numeric(selected_main["成交额"], errors="coerce").fillna(0).sum() / 100000000

    static_df = (
        merged.groupby("板块", as_index=False)
        .agg(
            板块股票数=("股票代码", "nunique"),
            板块总市值_亿=("总市值(亿)", "sum"),
            板块成交额_亿=("成交额(亿)", "sum"),
        )
        .copy()
    )
    static_df["成交占比"] = 0.0
    if total_turnover > 0:
        static_df["成交占比"] = pd.to_numeric(static_df["板块成交额_亿"], errors="coerce").fillna(0) / total_turnover
    static_df["成交占比得分"] = clip_series_local(
        static_df["成交占比"] / config.turnover_share_full_score_ratio * 100,
        config.score_floor,
        config.score_ceiling,
    )
    static_df["静态容量分"] = clip_series_local(
        pd.to_numeric(static_df["板块总市值_亿"], errors="coerce").fillna(0) / config.static_market_cap_divisor * 100,
        config.score_floor,
        config.score_ceiling,
    )

    zt_df = merged[pd.to_numeric(merged["是否涨停"], errors="coerce").fillna(0) == 1].copy()
    if zt_df.empty:
        dynamic_df = static_df[["板块"]].copy()
        dynamic_df["涨停家数"] = 0
        dynamic_df["涨停股总市值_亿"] = 0.0
        dynamic_df["涨停股成交额_亿"] = 0.0
        dynamic_df["涨停股票简称列表"] = ""
    else:
        dynamic_df = (
            zt_df.groupby("板块", as_index=False)
            .agg(
                涨停家数=("股票代码", "nunique"),
                涨停股总市值_亿=("总市值(亿)", "sum"),
                涨停股成交额_亿=("成交额(亿)", "sum"),
                涨停股票简称列表=("股票简称", lambda s: ",".join(sorted(set(s)))),
            )
            .copy()
        )

    dynamic_df["涨停家数得分"] = clip_series_local(
        pd.to_numeric(dynamic_df["涨停家数"], errors="coerce").fillna(0) * config.limit_up_count_multiplier,
        config.score_floor,
        config.score_ceiling,
    )
    dynamic_df["涨停股总市值得分"] = clip_series_local(
        pd.to_numeric(dynamic_df["涨停股总市值_亿"], errors="coerce").fillna(0) / config.limit_up_market_cap_divisor * 100,
        config.score_floor,
        config.score_ceiling,
    )
    dynamic_df["涨停股成交额得分"] = clip_series_local(
        pd.to_numeric(dynamic_df["涨停股成交额_亿"], errors="coerce").fillna(0) / config.limit_up_turnover_divisor * 100,
        config.score_floor,
        config.score_ceiling,
    )

    score_df = static_df.merge(dynamic_df, on="板块", how="left")
    for col in [
        "板块成交额_亿",
        "成交占比",
        "成交占比得分",
        "涨停家数",
        "涨停股总市值_亿",
        "涨停股成交额_亿",
        "涨停家数得分",
        "涨停股总市值得分",
        "涨停股成交额得分",
    ]:
        score_df[col] = pd.to_numeric(score_df[col], errors="coerce").fillna(0)
    score_df["涨停股票简称列表"] = score_df["涨停股票简称列表"].fillna("")
    score_df["动态容量分"] = clip_series_local(
        weighted_sum_series(
            [
                (score_df["成交占比得分"], config.turnover_share_weight),
                (score_df["涨停家数得分"], config.limit_up_count_weight),
                (score_df["涨停股总市值得分"], config.limit_up_market_cap_weight),
                (score_df["涨停股成交额得分"], config.limit_up_turnover_weight),
            ]
        ),
        config.score_floor,
        config.score_ceiling,
    )
    score_df["综合容量分"] = clip_series_local(
        weighted_sum_series(
            [
                (score_df["动态容量分"], config.final_dynamic_weight),
                (score_df["静态容量分"], config.final_static_weight),
            ]
        ),
        config.score_floor,
        config.score_ceiling,
    )
    score_df.insert(0, "交易日期", selected_date)
    return score_df.sort_values(["综合容量分", "涨停家数", "板块"], ascending=[False, False, True]).reset_index(drop=True)


def _extract_date_from_cbd_filename(path: Path) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), errors="coerce")


def load_propagation_history() -> pd.DataFrame:
    cbd_dir = WORKSPACE_ROOT / "cbd"
    if not cbd_dir.exists():
        return pd.DataFrame(columns=["传播度日期", "板块", "传播度"])

    frames: list[pd.DataFrame] = []
    for path in sorted(cbd_dir.glob("t-*.csv")):
        propagation_date = _extract_date_from_cbd_filename(path)
        if pd.isna(propagation_date):
            continue

        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        rename_map = {}
        if "sector" in df.columns:
            rename_map["sector"] = "板块"
        if "total_score" in df.columns:
            rename_map["total_score"] = "传播度"
        if "得分" in df.columns and "传播度" not in df.columns:
            rename_map["得分"] = "传播度"
        df = df.rename(columns=rename_map)
        if not {"板块", "传播度"}.issubset(df.columns):
            continue

        part = df[["板块", "传播度"]].copy()
        part["传播度日期"] = pd.Timestamp(propagation_date).normalize()
        part["板块"] = part["板块"].astype("string").fillna("").str.strip()
        part["传播度"] = pd.to_numeric(part["传播度"], errors="coerce").fillna(0)
        frames.append(part[part["板块"] != ""][["传播度日期", "板块", "传播度"]])

    if not frames:
        return pd.DataFrame(columns=["传播度日期", "板块", "传播度"])

    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["传播度日期", "板块"], as_index=False)["传播度"]
        .max()
        .sort_values(["传播度日期", "传播度", "板块"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def calculate_total_scores_compat(
    coverage_board_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    config: TotalConfig,
    propagation_df: pd.DataFrame,
) -> pd.DataFrame:
    try:
        return calculate_total_scores(coverage_board_df, capacity_df, config, propagation_df)
    except TypeError:
        coverage_view = coverage_board_df[["板块", "板块加权包含度得分", "板块平均包含度得分"]].copy()
        capacity_view = capacity_df[["交易日期", "板块", "静态容量分", "动态容量分", "综合容量分"]].copy()
        merged = capacity_view.merge(coverage_view, on="板块", how="left")
        if not propagation_df.empty:
            propagation_view = propagation_df[["板块", "传播度"]].copy()
            propagation_view["板块"] = propagation_view["板块"].astype("string").fillna("").str.strip()
            propagation_view["传播度"] = pd.to_numeric(propagation_view["传播度"], errors="coerce").fillna(0)
            propagation_view = propagation_view.groupby("板块", as_index=False)["传播度"].max()
            merged = merged.merge(propagation_view, on="板块", how="left")
        else:
            merged["传播度"] = 0.0
        merged["传播度"] = pd.to_numeric(merged["传播度"], errors="coerce").fillna(0)
        propagation_multiplier = getattr(config, "propagation_multiplier", 1.0)
        merged["传播度放大分"] = (merged["传播度"] * float(propagation_multiplier)).clip(lower=0, upper=100)
        merged["总分"] = (
            merged["传播度放大分"] * float(config.propagation_weight)
            + pd.to_numeric(merged["板块加权包含度得分"], errors="coerce").fillna(0) * float(config.coverage_weight)
            + pd.to_numeric(merged["综合容量分"], errors="coerce").fillna(0) * float(config.capacity_weight)
        ).clip(lower=0, upper=100)
        return merged.sort_values(["总分", "板块"], ascending=[False, True]).reset_index(drop=True)


def coverage_config_signature(config: CoverageConfig) -> tuple[object, ...]:
    return (
        int(config.short_window),
        int(config.middle_window),
        int(config.long_window),
        float(config.score_base),
        float(config.penalty_slope),
        float(config.short_weight),
        float(config.middle_weight),
        float(config.long_weight),
        float(config.score_floor),
        float(config.score_ceiling),
    )


def capacity_config_signature(config: CapacityConfig) -> tuple[object, ...]:
    config = ensure_capacity_config_compat(config)
    return (
        float(config.static_market_cap_divisor),
        float(config.turnover_share_full_score_ratio),
        float(config.turnover_share_weight),
        float(config.limit_up_count_multiplier),
        float(config.limit_up_market_cap_divisor),
        float(config.limit_up_turnover_divisor),
        float(config.limit_up_count_weight),
        float(config.limit_up_market_cap_weight),
        float(config.limit_up_turnover_weight),
        float(config.final_dynamic_weight),
        float(config.final_static_weight),
        float(config.score_floor),
        float(config.score_ceiling),
    )


def total_config_signature(config: TotalConfig) -> tuple[object, ...]:
    return (
        float(config.coverage_weight),
        float(config.capacity_weight),
        float(config.propagation_weight),
        float(getattr(config, "propagation_multiplier", 1.0)),
    )


@st.cache_data(show_spinner="计算基准总分中...")
def build_total_snapshot_cached(
    propagation_history_df: pd.DataFrame,
    available_trade_date_keys: tuple[str, ...],
    propagation_date_key: str,
    dataset_signature: tuple[object, ...],
    coverage_signature: tuple[object, ...],
    capacity_signature: tuple[object, ...],
    total_signature: tuple[object, ...],
    _feature_df: pd.DataFrame,
    _raw_df: pd.DataFrame,
    _mapping_df: pd.DataFrame,
    _coverage_config: CoverageConfig,
    _capacity_config: CapacityConfig,
    _total_config: TotalConfig,
) -> pd.DataFrame:
    del dataset_signature, coverage_signature, capacity_signature, total_signature
    propagation_date = pd.Timestamp(propagation_date_key).normalize()
    available_trade_dates = [pd.Timestamp(item).normalize() for item in available_trade_date_keys]
    matched_trade_date = match_trade_date(propagation_date, available_trade_dates)
    propagation_df = propagation_history_df[
        pd.to_datetime(propagation_history_df["传播度日期"]).dt.normalize() == propagation_date
    ].copy()
    _, coverage_board = calculate_coverage_scores(_feature_df, _mapping_df, matched_trade_date, _coverage_config)
    capacity_df = calculate_capacity_scores_compat(_raw_df, _mapping_df, matched_trade_date, _capacity_config)
    total_df = calculate_total_scores_compat(coverage_board, capacity_df, _total_config, propagation_df)
    total_df["传播度日期"] = propagation_date
    total_df["匹配交易日"] = matched_trade_date
    return total_df


@st.cache_data(show_spinner="计算总分趋势中...")
def build_total_history_frame_cached(
    propagation_history_df: pd.DataFrame,
    available_trade_date_keys: tuple[str, ...],
    history_date_keys: tuple[str, ...],
    selected_boards: tuple[str, ...],
    start_date_key: str,
    end_date_key: str,
    dataset_signature: tuple[object, ...],
    coverage_signature: tuple[object, ...],
    capacity_signature: tuple[object, ...],
    total_signature: tuple[object, ...],
    _feature_df: pd.DataFrame,
    _raw_df: pd.DataFrame,
    _mapping_df: pd.DataFrame,
    _coverage_config: CoverageConfig,
    _capacity_config: CapacityConfig,
    _total_config: TotalConfig,
) -> pd.DataFrame:
    del dataset_signature, coverage_signature, capacity_signature, total_signature
    if not selected_boards:
        return pd.DataFrame(
            columns=["传播度日期", "匹配交易日", "板块", "传播度", "传播度放大分", "板块加权包含度得分", "综合容量分", "总分"]
        )

    start_date = pd.Timestamp(start_date_key).normalize()
    end_date = pd.Timestamp(end_date_key).normalize()
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    history_dates = [
        pd.Timestamp(item).normalize()
        for item in history_date_keys
        if start_date <= pd.Timestamp(item).normalize() <= end_date
    ]
    if not history_dates:
        return pd.DataFrame(
            columns=["传播度日期", "匹配交易日", "板块", "传播度", "传播度放大分", "板块加权包含度得分", "综合容量分", "总分"]
        )

    mapping_filtered = _mapping_df[_mapping_df["板块"].isin(selected_boards)].copy()
    if mapping_filtered.empty:
        return pd.DataFrame(
            columns=["传播度日期", "匹配交易日", "板块", "传播度", "传播度放大分", "板块加权包含度得分", "综合容量分", "总分"]
        )

    selected_codes = mapping_filtered["股票代码"].astype("string").dropna().unique().tolist()
    feature_filtered = _feature_df[_feature_df["股票代码"].astype("string").isin(selected_codes)].copy()
    available_trade_dates = [pd.Timestamp(item).normalize() for item in available_trade_date_keys]

    frames: list[pd.DataFrame] = []
    for propagation_date in history_dates:
        matched_trade_date = match_trade_date(propagation_date, available_trade_dates)
        propagation_df = propagation_history_df[
            pd.to_datetime(propagation_history_df["传播度日期"]).dt.normalize() == propagation_date
        ].copy()
        propagation_df = propagation_df[propagation_df["板块"].isin(selected_boards)].copy()
        _, coverage_board = calculate_coverage_scores(
            feature_filtered,
            mapping_filtered,
            matched_trade_date,
            _coverage_config,
        )
        capacity_df = calculate_capacity_scores_compat(
            _raw_df,
            mapping_filtered,
            matched_trade_date,
            _capacity_config,
        )
        total_df = calculate_total_scores_compat(coverage_board, capacity_df, _total_config, propagation_df)
        total_df = total_df[total_df["板块"].isin(selected_boards)].copy()
        total_df["传播度日期"] = propagation_date
        total_df["匹配交易日"] = matched_trade_date
        frames.append(
            total_df[
                ["传播度日期", "匹配交易日", "板块", "传播度", "传播度放大分", "板块加权包含度得分", "综合容量分", "总分"]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=["传播度日期", "匹配交易日", "板块", "传播度", "传播度放大分", "板块加权包含度得分", "综合容量分", "总分"]
        )

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["传播度日期", "总分", "板块"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="加载数据中...")
def load_sources_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ensure_mapping_file()
    if not APP_PARQUET_PATH.exists():
        build_app_dataset()
    return load_app_dataset(), load_mapping_table(), load_propagation_history(), load_dataset_summary()


@st.cache_data(show_spinner="计算历史窗口中...")
def build_feature_frame_cached(windows: tuple[int, ...], dataset_signature: tuple[object, ...]) -> pd.DataFrame:
    raw_df = load_app_dataset()
    return ensure_feature_columns(raw_df, set(windows))


def build_coverage_panel() -> CoverageConfig:
    panel = st.sidebar.expander("股价包含度参数", expanded=False)
    short_window = panel.number_input("短周期窗口", min_value=3, max_value=120, value=10, step=1, key="cov_short_window")
    middle_window = panel.number_input("中周期窗口", min_value=5, max_value=180, value=20, step=1, key="cov_middle_window")
    long_window = panel.number_input("长周期窗口", min_value=10, max_value=250, value=60, step=1, key="cov_long_window")
    score_base = panel.number_input("得分基准", min_value=0.0, max_value=200.0, value=100.0, step=1.0, key="cov_score_base")
    penalty_slope = panel.number_input("涨幅惩罚系数", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="cov_penalty")
    short_weight = panel.slider("短周期权重", min_value=0.0, max_value=2.0, value=0.5, step=0.05, key="cov_short_weight")
    middle_weight = panel.slider("中周期权重", min_value=0.0, max_value=2.0, value=0.3, step=0.05, key="cov_middle_weight")
    long_weight = panel.slider("长周期权重", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="cov_long_weight")
    score_floor = panel.number_input("单项得分下限", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="cov_score_floor")
    score_ceiling = panel.number_input("单项得分上限", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="cov_score_ceiling")
    return CoverageConfig(
        short_window=int(short_window),
        middle_window=int(middle_window),
        long_window=int(long_window),
        score_base=float(score_base),
        penalty_slope=float(penalty_slope),
        short_weight=float(short_weight),
        middle_weight=float(middle_weight),
        long_weight=float(long_weight),
        score_floor=float(score_floor),
        score_ceiling=float(score_ceiling),
    )


def build_capacity_panel() -> CapacityConfig:
    panel = st.sidebar.expander("板块容量参数", expanded=False)
    static_market_cap_divisor = panel.number_input("静态容量分分母", min_value=1.0, max_value=500000.0, value=125000.0, step=1000.0, key="cap_static_div")
    turnover_share_full_score_percent = panel.number_input("成交占比满分标准", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key="cap_turnover_share_full")
    turnover_share_weight = panel.slider("成交占比得分权重", min_value=0.0, max_value=2.0, value=0.5, step=0.05, key="cap_turnover_share_weight")
    limit_up_count_multiplier = panel.number_input("涨停家数得分系数", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key="cap_count_multiplier")
    limit_up_market_cap_divisor = panel.number_input("涨停股总市值分母", min_value=1.0, max_value=500000.0, value=50000.0, step=1000.0, key="cap_mc_div")
    limit_up_turnover_divisor = panel.number_input("涨停股成交额分母", min_value=1.0, max_value=50000.0, value=3000.0, step=100.0, key="cap_turnover_div")
    limit_up_count_weight = panel.slider("涨停家数权重", min_value=0.0, max_value=2.0, value=0.3, step=0.05, key="cap_count_weight")
    limit_up_market_cap_weight = panel.slider("涨停股总市值权重", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="cap_mc_weight")
    limit_up_turnover_weight = panel.slider("涨停股成交额权重", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="cap_turnover_weight")
    final_dynamic_weight = panel.slider("综合分中动态权重", min_value=0.0, max_value=2.0, value=0.8, step=0.05, key="cap_final_dynamic")
    final_static_weight = panel.slider("综合分中静态权重", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="cap_final_static")
    score_floor = panel.number_input("容量单项得分下限", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="cap_score_floor")
    score_ceiling = panel.number_input("容量单项得分上限", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="cap_score_ceiling")
    kwargs = {
        "static_market_cap_divisor": float(static_market_cap_divisor),
        "limit_up_count_multiplier": float(limit_up_count_multiplier),
        "limit_up_market_cap_divisor": float(limit_up_market_cap_divisor),
        "limit_up_turnover_divisor": float(limit_up_turnover_divisor),
        "limit_up_count_weight": float(limit_up_count_weight),
        "limit_up_market_cap_weight": float(limit_up_market_cap_weight),
        "limit_up_turnover_weight": float(limit_up_turnover_weight),
        "final_dynamic_weight": float(final_dynamic_weight),
        "final_static_weight": float(final_static_weight),
        "score_floor": float(score_floor),
        "score_ceiling": float(score_ceiling),
    }
    try:
        config = CapacityConfig(
            **kwargs,
            turnover_share_full_score_ratio=float(turnover_share_full_score_percent) / 100,
            turnover_share_weight=float(turnover_share_weight),
        )
    except TypeError:
        config = CapacityConfig(**kwargs)
        object.__setattr__(config, "turnover_share_full_score_ratio", float(turnover_share_full_score_percent) / 100)
        object.__setattr__(config, "turnover_share_weight", float(turnover_share_weight))
    return ensure_capacity_config_compat(config)


def build_total_panel() -> TotalConfig:
    panel = st.sidebar.expander("总分参数", expanded=False)
    propagation_multiplier = panel.number_input("传播度放大倍数", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="total_prop_multiplier")
    propagation_weight_percent = panel.slider("总分中传播度权重", min_value=0, max_value=100, value=55, step=1, key="total_prop_weight_percent")
    coverage_weight_percent = panel.slider("总分中包含度权重", min_value=0, max_value=100, value=20, step=1, key="total_cov_weight_percent")
    capacity_weight_percent = panel.slider("总分中容量权重", min_value=0, max_value=100, value=25, step=1, key="total_cap_weight_percent")
    kwargs = {
        "coverage_weight": float(coverage_weight_percent) / 100,
        "capacity_weight": float(capacity_weight_percent) / 100,
        "propagation_weight": float(propagation_weight_percent) / 100,
    }
    try:
        return TotalConfig(**kwargs, propagation_multiplier=float(propagation_multiplier))
    except TypeError:
        config = TotalConfig(**kwargs, propagation_score=0.0)
        object.__setattr__(config, "propagation_multiplier", float(propagation_multiplier))
        return config


def formula_reference_box(title: str, baseline_markdown: str, tuned_markdown: str) -> None:
    st.subheader(title)
    left, right = st.columns(2)
    with left:
        st.markdown("**原始基准公式**")
        st.markdown(baseline_markdown)
    with right:
        st.markdown("**当前调参公式**")
        st.markdown(tuned_markdown)


def render_coverage_tab(
    feature_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    selected_date: pd.Timestamp,
    tuned_config: CoverageConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_config = default_coverage_config()
    tuned_detail, tuned_board = calculate_coverage_scores(feature_df, mapping_df, selected_date, tuned_config)
    baseline_detail, baseline_board = calculate_coverage_scores(feature_df, mapping_df, selected_date, baseline_config)

    formula_reference_box(
        "股价包含度公式参考",
        coverage_formula_markdown(baseline_config),
        coverage_formula_markdown(tuned_config),
    )

    baseline_board = baseline_board.rename(
        columns={
            "板块加权包含度得分": "基准板块加权包含度得分",
            "板块平均包含度得分": "基准板块平均包含度得分",
        }
    )
    board_compare = tuned_board.merge(
        baseline_board[["板块", "基准板块加权包含度得分", "基准板块平均包含度得分"]],
        on="板块",
        how="left",
    )
    board_compare["板块加权得分变化"] = board_compare["板块加权包含度得分"] - board_compare["基准板块加权包含度得分"]
    board_compare["板块平均得分变化"] = board_compare["板块平均包含度得分"] - board_compare["基准板块平均包含度得分"]

    st.markdown("**板块层得分变化**")
    st.dataframe(
        board_compare[
            [
                "板块",
                "板块股票数",
                "板块总市值",
                "板块加权包含度得分",
                "基准板块加权包含度得分",
                "板块加权得分变化",
                "板块平均包含度得分",
                "基准板块平均包含度得分",
                "板块平均得分变化",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    baseline_stock_cols = baseline_detail[["板块", "代码", "股价包含度得分", "加权股价包含度得分"]].rename(
        columns={
            "股价包含度得分": "基准股价包含度得分",
            "加权股价包含度得分": "基准加权股价包含度得分",
        }
    )
    stock_compare = tuned_detail.merge(baseline_stock_cols, on=["板块", "代码"], how="left")
    stock_compare["股价包含度变化"] = stock_compare["股价包含度得分"] - stock_compare["基准股价包含度得分"]
    stock_compare["加权包含度变化"] = stock_compare["加权股价包含度得分"] - stock_compare["基准加权股价包含度得分"]

    board_options = board_compare["板块"].tolist()
    selected_board = st.selectbox("查看板块内个股明细", options=board_options, key="coverage_selected_board")
    board_stock = stock_compare[stock_compare["板块"] == selected_board].copy()
    board_stock = board_stock.sort_values(
        ["加权股价包含度得分", "股价包含度得分", "总市值(亿)", "代码"],
        ascending=[False, False, False, True],
    )
    st.markdown(f"**{selected_board} 个股得分变化**")
    st.dataframe(
        board_stock[
            [
                "代码",
                "简称",
                "股价包含度得分",
                "基准股价包含度得分",
                "股价包含度变化",
                "加权股价包含度得分",
                "基准加权股价包含度得分",
                "加权包含度变化",
                "总市值(亿)",
                "成交额(亿)",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    return tuned_board, baseline_board.rename(columns={"基准板块加权包含度得分": "板块加权包含度得分", "基准板块平均包含度得分": "板块平均包含度得分"})


def render_capacity_tab(
    raw_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    selected_date: pd.Timestamp,
    tuned_config: CapacityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_config = ensure_capacity_config_compat(default_capacity_config())
    tuned_config = ensure_capacity_config_compat(tuned_config)
    tuned_df = calculate_capacity_scores_compat(raw_df, mapping_df, selected_date, tuned_config)
    baseline_df = calculate_capacity_scores_compat(raw_df, mapping_df, selected_date, baseline_config)

    formula_reference_box(
        "板块容量公式参考",
        capacity_formula_markdown_compat(baseline_config),
        capacity_formula_markdown_compat(tuned_config),
    )

    baseline_renamed = baseline_df.rename(
        columns={
            "静态容量分": "基准静态容量分",
            "动态容量分": "基准动态容量分",
            "综合容量分": "基准综合容量分",
            "成交占比得分": "基准成交占比得分",
        }
    )
    compare_df = tuned_df.merge(
        baseline_renamed[["板块", "基准静态容量分", "基准动态容量分", "基准综合容量分", "基准成交占比得分"]],
        on="板块",
        how="left",
    )
    compare_df["静态容量分变化"] = compare_df["静态容量分"] - compare_df["基准静态容量分"]
    compare_df["动态容量分变化"] = compare_df["动态容量分"] - compare_df["基准动态容量分"]
    compare_df["综合容量分变化"] = compare_df["综合容量分"] - compare_df["基准综合容量分"]
    compare_df["成交占比得分变化"] = compare_df["成交占比得分"] - compare_df["基准成交占比得分"]

    st.markdown("**板块容量得分变化**")
    st.dataframe(
        compare_df[
            [
                "板块",
                "板块股票数",
                "板块总市值_亿",
                "板块成交额_亿",
                "成交占比",
                "成交占比得分",
                "基准成交占比得分",
                "成交占比得分变化",
                "静态容量分",
                "基准静态容量分",
                "静态容量分变化",
                "动态容量分",
                "基准动态容量分",
                "动态容量分变化",
                "综合容量分",
                "基准综合容量分",
                "综合容量分变化",
                "涨停家数",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    board_options = compare_df["板块"].tolist()
    selected_board = st.selectbox("查看单个板块容量拆解", options=board_options, key="capacity_selected_board")
    row = compare_df[compare_df["板块"] == selected_board].iloc[0]
    baseline_row = baseline_df[baseline_df["板块"] == selected_board].iloc[0]
    st.markdown(f"**{selected_board} 容量拆解**")
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "静态容量分", "当前值": row["静态容量分"], "基准值": row["基准静态容量分"], "变化": row["静态容量分变化"]},
                {"项目": "成交占比得分", "当前值": row["成交占比得分"], "基准值": row["基准成交占比得分"], "变化": row["成交占比得分变化"]},
                {"项目": "动态容量分", "当前值": row["动态容量分"], "基准值": row["基准动态容量分"], "变化": row["动态容量分变化"]},
                {"项目": "综合容量分", "当前值": row["综合容量分"], "基准值": row["基准综合容量分"], "变化": row["综合容量分变化"]},
                {"项目": "涨停家数得分", "当前值": row["涨停家数得分"], "基准值": baseline_row["涨停家数得分"], "变化": row["涨停家数得分"] - baseline_row["涨停家数得分"]},
                {"项目": "涨停股总市值得分", "当前值": row["涨停股总市值得分"], "基准值": baseline_row["涨停股总市值得分"], "变化": row["涨停股总市值得分"] - baseline_row["涨停股总市值得分"]},
                {"项目": "涨停股成交额得分", "当前值": row["涨停股成交额得分"], "基准值": baseline_row["涨停股成交额得分"], "变化": row["涨停股成交额得分"] - baseline_row["涨停股成交额得分"]},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    return tuned_df, baseline_df


def render_total_tab(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    propagation_history_df: pd.DataFrame,
    available_dates: list[pd.Timestamp],
    propagation_dates: list[pd.Timestamp],
    dataset_signature: tuple[object, ...],
    selected_propagation_date: pd.Timestamp,
    tuned_coverage_board: pd.DataFrame,
    baseline_coverage_board: pd.DataFrame,
    tuned_capacity_df: pd.DataFrame,
    baseline_capacity_df: pd.DataFrame,
    propagation_df: pd.DataFrame,
    coverage_config: CoverageConfig,
    capacity_config: CapacityConfig,
    total_config: TotalConfig,
) -> None:
    baseline_config = default_total_config()
    tuned_total = calculate_total_scores_compat(tuned_coverage_board, tuned_capacity_df, total_config, propagation_df)
    baseline_total = calculate_total_scores_compat(baseline_coverage_board, baseline_capacity_df, baseline_config, propagation_df)

    formula_reference_box(
        "总分公式参考",
        total_formula_markdown(baseline_config),
        total_formula_markdown(total_config),
    )

    baseline_renamed = baseline_total.rename(
        columns={
            "传播度": "基准传播度",
            "传播度放大分": "基准传播度放大分",
            "板块加权包含度得分": "基准板块加权包含度得分",
            "综合容量分": "基准综合容量分",
            "总分": "基准总分",
        }
    )
    compare_df = tuned_total.merge(
        baseline_renamed[["板块", "基准传播度", "基准传播度放大分", "基准板块加权包含度得分", "基准综合容量分", "基准总分"]],
        on="板块",
        how="left",
    )
    compare_df["总分变化"] = compare_df["总分"] - compare_df["基准总分"]
    compare_df["包含度变化"] = compare_df["板块加权包含度得分"] - compare_df["基准板块加权包含度得分"]
    compare_df["容量变化"] = compare_df["综合容量分"] - compare_df["基准综合容量分"]
    compare_df["传播度放大分变化"] = compare_df["传播度放大分"] - compare_df["基准传播度放大分"]

    st.markdown("**总分变化**")
    st.dataframe(
        compare_df[
            [
                "板块",
                "传播度",
                "传播度放大分",
                "基准传播度放大分",
                "传播度放大分变化",
                "板块加权包含度得分",
                "基准板块加权包含度得分",
                "包含度变化",
                "综合容量分",
                "基准综合容量分",
                "容量变化",
                "总分",
                "基准总分",
                "总分变化",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    history_date_options = propagation_dates or available_dates
    history_dates_desc = sorted([pd.Timestamp(item).normalize() for item in history_date_options], reverse=True)
    history_dates_asc = list(reversed(history_dates_desc))
    base_date_default = pd.Timestamp(selected_propagation_date).normalize()
    if base_date_default not in history_dates_desc and history_dates_desc:
        base_date_default = history_dates_desc[0]

    controls = st.columns([1, 1.4, 1.2, 1.2])
    top_n_default = min(5, max(len(tuned_total), 1))
    top_n = int(
        controls[0].number_input(
            "Top N",
            min_value=1,
            max_value=max(len(tuned_total), 1),
            value=top_n_default,
            step=1,
            key="total_history_top_n",
        )
    )
    topn_base_date = pd.Timestamp(
        controls[1].selectbox(
            "TopN基准日期",
            options=history_dates_desc,
            index=history_dates_desc.index(base_date_default) if history_dates_desc else 0,
            format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            key="total_history_base_date",
        )
    ).normalize()
    base_date_position = history_dates_asc.index(topn_base_date)
    default_start_position = max(0, base_date_position - 59)
    start_date = pd.Timestamp(
        controls[2].selectbox(
            "开始时间",
            options=history_dates_asc[: base_date_position + 1],
            index=default_start_position,
            format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            key="total_history_start_date",
        )
    ).normalize()
    end_date_candidates = [date for date in history_dates_asc if pd.Timestamp(date).normalize() >= start_date]
    end_date = pd.Timestamp(
        controls[3].selectbox(
            "结束时间",
            options=end_date_candidates,
            index=end_date_candidates.index(topn_base_date) if topn_base_date in end_date_candidates else len(end_date_candidates) - 1,
            format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            key="total_history_end_date",
        )
    ).normalize()

    coverage_signature = coverage_config_signature(coverage_config)
    capacity_signature = capacity_config_signature(capacity_config)
    total_signature = total_config_signature(total_config)
    available_trade_date_keys = tuple(pd.Timestamp(item).strftime("%Y-%m-%d") for item in sorted(available_dates))
    history_date_keys = tuple(pd.Timestamp(item).strftime("%Y-%m-%d") for item in history_dates_asc)

    if topn_base_date == pd.Timestamp(selected_propagation_date).normalize():
        base_total = tuned_total.copy()
        base_total["传播度日期"] = topn_base_date
    else:
        base_total = build_total_snapshot_cached(
            propagation_history_df,
            available_trade_date_keys,
            topn_base_date.strftime("%Y-%m-%d"),
            dataset_signature,
            coverage_signature,
            capacity_signature,
            total_signature,
            feature_df,
            raw_df,
            mapping_df,
            coverage_config,
            capacity_config,
            total_config,
        )

    selected_boards = tuple(base_total.head(top_n)["板块"].tolist())
    st.markdown(
        f"**总分趋势图**  基准日期：`{topn_base_date.strftime('%Y-%m-%d')}`  Top N：`{top_n}`  区间：`{start_date.strftime('%Y-%m-%d')}` 至 `{end_date.strftime('%Y-%m-%d')}`"
    )

    if not selected_boards:
        st.info("当前基准日期没有可用的板块总分数据。")
        return

    st.caption("基准日期入选板块：" + "、".join(selected_boards))
    history_df = build_total_history_frame_cached(
        propagation_history_df,
        available_trade_date_keys,
        history_date_keys,
        selected_boards,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        dataset_signature,
        coverage_signature,
        capacity_signature,
        total_signature,
        feature_df,
        raw_df,
        mapping_df,
        coverage_config,
        capacity_config,
        total_config,
    )

    if history_df.empty:
        st.info("当前时间区间没有可展示的总分历史。")
        return

    history_df["板块"] = pd.Categorical(history_df["板块"], categories=list(selected_boards), ordered=True)
    history_df = history_df.sort_values(["传播度日期", "板块"]).reset_index(drop=True)
    fig = px.line(
        history_df,
        x="传播度日期",
        y="总分",
        color="板块",
        markers=True,
        category_orders={"板块": list(selected_boards)},
        hover_data={
            "板块": True,
            "传播度日期": "|%Y-%m-%d",
            "匹配交易日": "|%Y-%m-%d",
            "总分": ":.2f",
            "传播度": ":.2f",
            "传播度放大分": ":.2f",
            "板块加权包含度得分": ":.2f",
            "综合容量分": ":.2f",
        },
    )
    fig.update_layout(
        height=460,
        margin=dict(l=16, r=16, t=24, b=16),
        legend_title_text="板块",
        xaxis_title="传播度日期",
        yaxis_title="总分",
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def match_trade_date(propagation_date: pd.Timestamp, trade_dates: list[pd.Timestamp]) -> pd.Timestamp:
    selected = pd.Timestamp(propagation_date).normalize()
    candidates = [pd.Timestamp(item).normalize() for item in trade_dates if pd.Timestamp(item).normalize() <= selected]
    if not candidates:
        return pd.Timestamp(trade_dates[-1]).normalize()
    return max(candidates)


def main() -> None:
    st.title("因子参数调整")

    raw_df, mapping_df, propagation_history_df, summary = load_sources_cached()
    available_dates = sorted(pd.to_datetime(raw_df["交易日期"]).dt.normalize().unique(), reverse=True)
    min_trade_date = min(available_dates)
    propagation_dates = [
        date
        for date in sorted(pd.to_datetime(propagation_history_df["传播度日期"]).dropna().dt.normalize().unique(), reverse=True)
        if pd.Timestamp(date).normalize() >= pd.Timestamp(min_trade_date).normalize()
    ]

    st.sidebar.subheader("全局设置")
    date_options = propagation_dates or available_dates
    selected_propagation_date = st.sidebar.selectbox(
        "选择传播度日期",
        options=date_options,
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
        index=0,
    )
    selected_date = match_trade_date(pd.Timestamp(selected_propagation_date), available_dates)
    st.caption(
        f"传播度日期：{pd.Timestamp(selected_propagation_date).strftime('%Y-%m-%d')} | "
        f"匹配交易日：{pd.Timestamp(selected_date).strftime('%Y-%m-%d')} | "
        f"主表最新：{summary.get('max_trade_date', '未知')}"
    )

    coverage_config = build_coverage_panel()
    capacity_config = build_capacity_panel()
    total_config = build_total_panel()

    feature_windows = {
        coverage_config.short_window,
        coverage_config.middle_window,
        coverage_config.long_window,
        default_coverage_config().short_window,
        default_coverage_config().middle_window,
        default_coverage_config().long_window,
    }
    dataset_signature = (
        summary.get("rows"),
        summary.get("max_trade_date"),
        summary.get("source_csv_mtime_ns"),
    )
    feature_df = build_feature_frame_cached(tuple(sorted(feature_windows)), dataset_signature)
    selected_propagation_df = propagation_history_df[
        pd.to_datetime(propagation_history_df["传播度日期"]).dt.normalize() == pd.Timestamp(selected_propagation_date).normalize()
    ].copy()

    tab1, tab2, tab3 = st.tabs(["股价包含度", "板块容量", "总分"])

    with tab1:
        tuned_coverage_board, baseline_coverage_board = render_coverage_tab(
            feature_df,
            mapping_df,
            pd.Timestamp(selected_date),
            coverage_config,
        )

    with tab2:
        tuned_capacity_df, baseline_capacity_df = render_capacity_tab(
            raw_df,
            mapping_df,
            pd.Timestamp(selected_date),
            capacity_config,
        )

    with tab3:
        render_total_tab(
            raw_df,
            feature_df,
            mapping_df,
            propagation_history_df,
            available_dates,
            propagation_dates,
            dataset_signature,
            pd.Timestamp(selected_propagation_date),
            tuned_coverage_board,
            baseline_coverage_board,
            tuned_capacity_df,
            baseline_capacity_df,
            selected_propagation_df,
            coverage_config,
            capacity_config,
            total_config,
        )


if __name__ == "__main__":
    main()
