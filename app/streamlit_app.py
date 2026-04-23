from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
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
    load_propagation_history,
)


st.set_page_config(page_title="因子参数调整", page_icon="📊", layout="wide")


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
    limit_up_count_multiplier = panel.number_input("涨停家数得分系数", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key="cap_count_multiplier")
    limit_up_market_cap_divisor = panel.number_input("涨停股总市值分母", min_value=1.0, max_value=500000.0, value=50000.0, step=1000.0, key="cap_mc_div")
    limit_up_turnover_divisor = panel.number_input("涨停股成交额分母", min_value=1.0, max_value=50000.0, value=3000.0, step=100.0, key="cap_turnover_div")
    limit_up_count_weight = panel.slider("涨停家数权重", min_value=0.0, max_value=2.0, value=0.4, step=0.05, key="cap_count_weight")
    limit_up_market_cap_weight = panel.slider("涨停股总市值权重", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="cap_mc_weight")
    limit_up_turnover_weight = panel.slider("涨停股成交额权重", min_value=0.0, max_value=2.0, value=0.4, step=0.05, key="cap_turnover_weight")
    final_dynamic_weight = panel.slider("综合分中动态权重", min_value=0.0, max_value=2.0, value=0.5, step=0.05, key="cap_final_dynamic")
    final_static_weight = panel.slider("综合分中静态权重", min_value=0.0, max_value=2.0, value=0.5, step=0.05, key="cap_final_static")
    score_floor = panel.number_input("容量单项得分下限", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="cap_score_floor")
    score_ceiling = panel.number_input("容量单项得分上限", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="cap_score_ceiling")
    return CapacityConfig(
        static_market_cap_divisor=float(static_market_cap_divisor),
        limit_up_count_multiplier=float(limit_up_count_multiplier),
        limit_up_market_cap_divisor=float(limit_up_market_cap_divisor),
        limit_up_turnover_divisor=float(limit_up_turnover_divisor),
        limit_up_count_weight=float(limit_up_count_weight),
        limit_up_market_cap_weight=float(limit_up_market_cap_weight),
        limit_up_turnover_weight=float(limit_up_turnover_weight),
        final_dynamic_weight=float(final_dynamic_weight),
        final_static_weight=float(final_static_weight),
        score_floor=float(score_floor),
        score_ceiling=float(score_ceiling),
    )


def build_total_panel() -> TotalConfig:
    panel = st.sidebar.expander("总分参数", expanded=False)
    propagation_multiplier = panel.number_input("传播度放大倍数", min_value=0.0, max_value=20.0, value=2.5, step=0.1, key="total_prop_multiplier")
    propagation_weight_percent = panel.slider("总分中传播度权重", min_value=0, max_value=100, value=70, step=1, key="total_prop_weight_percent")
    coverage_weight_percent = panel.slider("总分中包含度权重", min_value=0, max_value=100, value=15, step=1, key="total_cov_weight_percent")
    capacity_weight_percent = panel.slider("总分中容量权重", min_value=0, max_value=100, value=15, step=1, key="total_cap_weight_percent")
    return TotalConfig(
        coverage_weight=float(coverage_weight_percent) / 100,
        capacity_weight=float(capacity_weight_percent) / 100,
        propagation_weight=float(propagation_weight_percent) / 100,
        propagation_multiplier=float(propagation_multiplier),
    )


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
    baseline_config = default_capacity_config()
    tuned_df = calculate_capacity_scores(raw_df, mapping_df, selected_date, tuned_config)
    baseline_df = calculate_capacity_scores(raw_df, mapping_df, selected_date, baseline_config)

    formula_reference_box(
        "板块容量公式参考",
        capacity_formula_markdown(baseline_config),
        capacity_formula_markdown(tuned_config),
    )

    baseline_renamed = baseline_df.rename(
        columns={
            "静态容量分": "基准静态容量分",
            "动态容量分": "基准动态容量分",
            "综合容量分": "基准综合容量分",
        }
    )
    compare_df = tuned_df.merge(
        baseline_renamed[["板块", "基准静态容量分", "基准动态容量分", "基准综合容量分"]],
        on="板块",
        how="left",
    )
    compare_df["静态容量分变化"] = compare_df["静态容量分"] - compare_df["基准静态容量分"]
    compare_df["动态容量分变化"] = compare_df["动态容量分"] - compare_df["基准动态容量分"]
    compare_df["综合容量分变化"] = compare_df["综合容量分"] - compare_df["基准综合容量分"]

    st.markdown("**板块容量得分变化**")
    st.dataframe(
        compare_df[
            [
                "板块",
                "板块股票数",
                "板块总市值_亿",
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
    tuned_coverage_board: pd.DataFrame,
    baseline_coverage_board: pd.DataFrame,
    tuned_capacity_df: pd.DataFrame,
    baseline_capacity_df: pd.DataFrame,
    propagation_df: pd.DataFrame,
    total_config: TotalConfig,
) -> None:
    baseline_config = default_total_config()
    tuned_total = calculate_total_scores(tuned_coverage_board, tuned_capacity_df, total_config, propagation_df)
    baseline_total = calculate_total_scores(baseline_coverage_board, baseline_capacity_df, baseline_config, propagation_df)

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
            tuned_coverage_board,
            baseline_coverage_board,
            tuned_capacity_df,
            baseline_capacity_df,
            selected_propagation_df,
            total_config,
        )


if __name__ == "__main__":
    main()
