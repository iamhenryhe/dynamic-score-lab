from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class CoverageConfig:
    short_window: int = 10
    middle_window: int = 20
    long_window: int = 60
    score_base: float = 100.0
    penalty_slope: float = 2.0
    short_weight: float = 0.5
    middle_weight: float = 0.3
    long_weight: float = 0.2
    score_floor: float = 0.0
    score_ceiling: float = 100.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityConfig:
    static_market_cap_divisor: float = 75000.0
    turnover_share_full_score_ratio: float = 0.10
    turnover_share_weight: float = 0.5
    limit_up_count_multiplier: float = 1.0
    limit_up_market_cap_divisor: float = 30000.0
    limit_up_turnover_divisor: float = 3000.0
    limit_up_count_weight: float = 0.1
    limit_up_market_cap_weight: float = 0.2
    limit_up_turnover_weight: float = 0.2
    final_dynamic_weight: float = 0.8
    final_static_weight: float = 0.2
    score_floor: float = 0.0
    score_ceiling: float = 100.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TotalConfig:
    coverage_weight: float = 0.2
    capacity_weight: float = 0.2
    propagation_weight: float = 0.6
    propagation_multiplier: float = 1.8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_coverage_config() -> CoverageConfig:
    return CoverageConfig()


def default_capacity_config() -> CapacityConfig:
    return CapacityConfig()


def default_total_config() -> TotalConfig:
    return TotalConfig()


def clip_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=lower, upper=upper)


def weighted_average_frame(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    valid_parts = [
        (pd.to_numeric(series, errors="coerce").fillna(0), float(weight))
        for series, weight in parts
        if float(weight) > 0
    ]
    if not valid_parts:
        reference = parts[0][0] if parts else pd.Series(dtype="float64")
        return pd.Series(0.0, index=reference.index, dtype="float64")
    numerator = sum(series * weight for series, weight in valid_parts)
    denominator = sum(weight for _, weight in valid_parts)
    return numerator / denominator


def weighted_sum_frame(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    valid_parts = [
        (pd.to_numeric(series, errors="coerce").fillna(0), float(weight))
        for series, weight in parts
        if float(weight) > 0
    ]
    if not valid_parts:
        reference = parts[0][0] if parts else pd.Series(dtype="float64")
        return pd.Series(0.0, index=reference.index, dtype="float64")
    return sum(series * weight for series, weight in valid_parts)


def ensure_feature_columns(df: pd.DataFrame, windows: set[int]) -> pd.DataFrame:
    out = df.sort_values(["股票代码", "交易日期"]).reset_index(drop=True).copy()
    grouped = out.groupby("股票代码", sort=False)

    for window in sorted({max(int(w), 1) for w in windows}):
        price_col = f"T-{window + 1}收盘价"
        return_col = f"近{window}日涨跌幅(%)"
        if return_col not in out.columns:
            out[price_col] = grouped["收盘价"].shift(window)
            out[return_col] = (out["收盘价"] / out[price_col] - 1) * 100

    if "T+1涨幅" not in out.columns:
        out["T+1涨幅"] = grouped["涨跌幅(%)"].shift(-1)
    if "T+2涨幅" not in out.columns:
        out["T+2涨幅"] = grouped["涨跌幅(%)"].shift(-2)
    if "成交额(亿)" not in out.columns:
        out["成交额(亿)"] = pd.to_numeric(out["成交额"], errors="coerce") / 100000000

    return out


def calculate_coverage_scores(
    feature_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: CoverageConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_date = pd.Timestamp(trade_date).normalize()
    merged = feature_df.merge(mapping_df, on="股票代码", how="inner", suffixes=("_主表", "_映射"))
    mapping_name_col = "简称_映射" if "简称_映射" in merged.columns else "简称"
    merged["简称"] = merged[mapping_name_col].where(
        merged[mapping_name_col].astype("string").fillna("").str.strip() != "",
        merged["股票简称"],
    )

    short_col = f"近{config.short_window}日涨跌幅(%)"
    middle_col = f"近{config.middle_window}日涨跌幅(%)"
    long_col = f"近{config.long_window}日涨跌幅(%)"
    short_score_col = f"{config.short_window}日股价得分"
    middle_score_col = f"{config.middle_window}日股价得分"
    long_score_col = f"{config.long_window}日股价得分"

    merged[short_score_col] = clip_series(
        config.score_base - config.penalty_slope * pd.to_numeric(merged[short_col], errors="coerce"),
        config.score_floor,
        config.score_ceiling,
    )
    merged[middle_score_col] = clip_series(
        config.score_base - config.penalty_slope * pd.to_numeric(merged[middle_col], errors="coerce"),
        config.score_floor,
        config.score_ceiling,
    )
    merged[long_score_col] = clip_series(
        config.score_base - config.penalty_slope * pd.to_numeric(merged[long_col], errors="coerce"),
        config.score_floor,
        config.score_ceiling,
    )
    merged["股价包含度得分"] = clip_series(
        weighted_average_frame(
            [
                (merged[short_score_col], config.short_weight),
                (merged[middle_score_col], config.middle_weight),
                (merged[long_score_col], config.long_weight),
            ]
        ),
        config.score_floor,
        config.score_ceiling,
    )
    merged["板块总市值"] = merged.groupby(["交易日期", "板块"])["总市值(亿)"].transform("sum")
    merged["加权股价包含度得分"] = merged["股价包含度得分"] * merged["总市值(亿)"] / merged["板块总市值"]
    merged.loc[~(merged["板块总市值"] > 0), "加权股价包含度得分"] = 0

    selected = merged[merged["交易日期"] == selected_date].copy()
    selected = selected.sort_values(
        ["板块", "加权股价包含度得分", "股价包含度得分", "总市值(亿)", "股票代码"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    board_summary = (
        selected.groupby("板块", as_index=False)
        .agg(
            板块股票数=("股票代码", "nunique"),
            板块总市值=("总市值(亿)", "sum"),
            板块加权包含度得分=("加权股价包含度得分", "sum"),
            板块平均包含度得分=("股价包含度得分", "mean"),
        )
        .sort_values(["板块加权包含度得分", "板块总市值", "板块"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    board_summary["板块加权包含度得分"] = clip_series(
        board_summary["板块加权包含度得分"],
        config.score_floor,
        config.score_ceiling,
    )
    board_summary["板块平均包含度得分"] = clip_series(
        board_summary["板块平均包含度得分"],
        config.score_floor,
        config.score_ceiling,
    )

    detail_cols = [
        "交易日期",
        "板块",
        "简称",
        "股票代码",
        "收盘价",
        short_col,
        short_score_col,
        middle_col,
        middle_score_col,
        long_col,
        long_score_col,
        "股价包含度得分",
        "总市值(亿)",
        "加权股价包含度得分",
        "成交额(亿)",
        "T+1涨幅",
        "T+2涨幅",
    ]
    detail_cols = [col for col in detail_cols if col in selected.columns]
    selected = selected[detail_cols].copy()
    selected = selected.rename(columns={"股票代码": "代码"})

    return selected, board_summary


def calculate_capacity_scores(
    main_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: CapacityConfig,
) -> pd.DataFrame:
    selected_date = pd.Timestamp(trade_date).normalize()
    selected_main = main_df[main_df["交易日期"] == selected_date].copy()
    merged = selected_main.merge(mapping_df, on="股票代码", how="inner", suffixes=("_主表", "_映射"))
    mapping_name_col = "简称_映射" if "简称_映射" in merged.columns else "简称"
    merged["股票简称"] = merged[mapping_name_col].where(
        merged[mapping_name_col].astype("string").fillna("").str.strip() != "",
        merged["股票简称"],
    )
    merged["成交额(亿)"] = pd.to_numeric(merged["成交额"], errors="coerce") / 100000000
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
    static_df["成交占比得分"] = clip_series(
        static_df["成交占比"] / config.turnover_share_full_score_ratio * 100,
        config.score_floor,
        config.score_ceiling,
    )
    static_df["静态容量分"] = clip_series(
        static_df["板块总市值_亿"] / config.static_market_cap_divisor * 100,
        config.score_floor,
        config.score_ceiling,
    )

    zt_df = merged[merged["是否涨停"] == 1].copy()
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

    dynamic_df["涨停家数得分"] = clip_series(
        pd.to_numeric(dynamic_df["涨停家数"], errors="coerce").fillna(0) * config.limit_up_count_multiplier,
        config.score_floor,
        config.score_ceiling,
    )
    dynamic_df["涨停股总市值得分"] = clip_series(
        pd.to_numeric(dynamic_df["涨停股总市值_亿"], errors="coerce").fillna(0) / config.limit_up_market_cap_divisor * 100,
        config.score_floor,
        config.score_ceiling,
    )
    dynamic_df["涨停股成交额得分"] = clip_series(
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
    score_df["动态容量分"] = clip_series(
        weighted_sum_frame(
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
    score_df["综合容量分"] = clip_series(
        weighted_sum_frame(
            [
                (score_df["动态容量分"], config.final_dynamic_weight),
                (score_df["静态容量分"], config.final_static_weight),
            ]
        ),
        config.score_floor,
        config.score_ceiling,
    )
    score_df.insert(0, "交易日期", selected_date)
    score_df = score_df.sort_values(["综合容量分", "涨停家数", "板块"], ascending=[False, False, True]).reset_index(drop=True)
    return score_df


def calculate_total_scores(
    coverage_board_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    config: TotalConfig,
    propagation_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    coverage_view = coverage_board_df[["板块", "板块加权包含度得分", "板块平均包含度得分"]].copy()
    capacity_view = capacity_df[
        [
            "交易日期",
            "板块",
            "静态容量分",
            "动态容量分",
            "综合容量分",
        ]
    ].copy()
    merged = capacity_view.merge(coverage_view, on="板块", how="left")
    if propagation_df is not None and not propagation_df.empty:
        propagation_view = propagation_df[["板块", "传播度"]].copy()
        propagation_view["板块"] = propagation_view["板块"].astype("string").fillna("").str.strip()
        propagation_view["传播度"] = pd.to_numeric(propagation_view["传播度"], errors="coerce").fillna(0)
        propagation_view = propagation_view.groupby("板块", as_index=False)["传播度"].max()
        merged = merged.merge(propagation_view, on="板块", how="left")
    else:
        merged["传播度"] = 0.0
    merged["传播度"] = pd.to_numeric(merged["传播度"], errors="coerce").fillna(0)
    merged["传播度放大分"] = clip_series(
        merged["传播度"] * float(config.propagation_multiplier),
        0.0,
        100.0,
    )
    merged["总分"] = clip_series(
        merged["传播度放大分"] * float(config.propagation_weight)
        + pd.to_numeric(merged["板块加权包含度得分"], errors="coerce").fillna(0) * float(config.coverage_weight)
        + pd.to_numeric(merged["综合容量分"], errors="coerce").fillna(0) * float(config.capacity_weight),
        0.0,
        100.0,
    )
    merged = merged.sort_values(["总分", "板块"], ascending=[False, True]).reset_index(drop=True)
    return merged


def summarize_top_n(scored_df: pd.DataFrame, top_n: int) -> dict[str, float]:
    if scored_df.empty:
        return {
            "top_n_avg_t1": 0.0,
            "top_n_avg_t2": 0.0,
            "top_n_win_rate_t1": 0.0,
            "all_avg_t1": 0.0,
            "all_avg_t2": 0.0,
        }

    top_n = max(int(top_n), 1)
    top_df = scored_df.head(top_n).copy()
    top_t1 = pd.to_numeric(top_df["T+1涨幅"], errors="coerce")
    top_t2 = pd.to_numeric(top_df["T+2涨幅"], errors="coerce")
    all_t1 = pd.to_numeric(scored_df["T+1涨幅"], errors="coerce")
    all_t2 = pd.to_numeric(scored_df["T+2涨幅"], errors="coerce")

    def safe_mean(series: pd.Series) -> float:
        value = series.mean()
        return 0.0 if pd.isna(value) else float(value)

    def safe_win_rate(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return 0.0
        return float((valid > 0).mean() * 100)

    return {
        "top_n_avg_t1": safe_mean(top_t1),
        "top_n_avg_t2": safe_mean(top_t2),
        "top_n_win_rate_t1": safe_win_rate(top_t1),
        "all_avg_t1": safe_mean(all_t1),
        "all_avg_t2": safe_mean(all_t2),
    }


def format_weight_percent(weight: float) -> str:
    return f"{float(weight) * 100:.0f}%"


def normalized_weight_percent(weight: float, weights: list[float]) -> str:
    total = sum(max(float(item), 0.0) for item in weights)
    if total <= 0:
        return "0%"
    return f"{max(float(weight), 0.0) / total * 100:.0f}%"


def coverage_formula_markdown(config: CoverageConfig) -> str:
    short_percent = normalized_weight_percent(
        config.short_weight,
        [config.short_weight, config.middle_weight, config.long_weight],
    )
    middle_percent = normalized_weight_percent(
        config.middle_weight,
        [config.short_weight, config.middle_weight, config.long_weight],
    )
    long_percent = normalized_weight_percent(
        config.long_weight,
        [config.short_weight, config.middle_weight, config.long_weight],
    )
    return (
        f"- 短周期窗口：`{config.short_window}` 日，得分公式：`{config.score_base} - {config.penalty_slope} * 近{config.short_window}日涨跌幅`\n"
        f"- 中周期窗口：`{config.middle_window}` 日，得分公式：`{config.score_base} - {config.penalty_slope} * 近{config.middle_window}日涨跌幅`\n"
        f"- 长周期窗口：`{config.long_window}` 日，得分公式：`{config.score_base} - {config.penalty_slope} * 近{config.long_window}日涨跌幅`\n"
        f"- 股价包含度得分：`{config.short_window}日股价得分 * {short_percent} + {config.middle_window}日股价得分 * {middle_percent} + {config.long_window}日股价得分 * {long_percent}`\n"
        "- 加权股价包含度得分：`股价包含度得分 * (总市值 / 板块总市值)`"
    )


def capacity_formula_markdown(config: CapacityConfig) -> str:
    return (
        f"- 静态容量分：`板块总市值_亿 / {config.static_market_cap_divisor} * 100`\n"
        f"- 成交占比得分：`成交占比 / {format_weight_percent(config.turnover_share_full_score_ratio)} * 100`\n"
        f"- 涨停家数得分：`涨停家数 * {config.limit_up_count_multiplier}`\n"
        f"- 涨停股总市值得分：`涨停股总市值_亿 / {config.limit_up_market_cap_divisor} * 100`\n"
        f"- 涨停股成交额得分：`涨停股成交额_亿 / {config.limit_up_turnover_divisor} * 100`\n"
        f"- 动态容量分：`成交占比得分 * {format_weight_percent(config.turnover_share_weight)} + 涨停家数得分 * {format_weight_percent(config.limit_up_count_weight)} + 涨停股总市值得分 * {format_weight_percent(config.limit_up_market_cap_weight)} + 涨停股成交额得分 * {format_weight_percent(config.limit_up_turnover_weight)}`\n"
        f"- 综合容量分：`动态容量分 * {format_weight_percent(config.final_dynamic_weight)} + 静态容量分 * {format_weight_percent(config.final_static_weight)}`"
    )


def total_formula_markdown(config: TotalConfig) -> str:
    return (
        f"- 传播度放大分：`传播度 * {config.propagation_multiplier}`\n"
        f"- 板块总分：`传播度放大分 * {format_weight_percent(config.propagation_weight)} + 板块加权包含度得分 * {format_weight_percent(config.coverage_weight)} + 板块容量得分 * {format_weight_percent(config.capacity_weight)}`"
    )
