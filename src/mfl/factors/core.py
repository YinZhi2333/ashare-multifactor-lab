"""因子层的小型公共契约：键、方向、质量原因及配置。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NAMES = ["EP", "ROE_TTM", "MOM_12_1", "REV_1M", "TURN_20"]
DIRECTIONS = dict(zip(NAMES, [1, 1, 1, -1, -1]))
REASONS = {
    "missing_pit": "没有该月对应PIT记录或公告审计字段缺失",
    "pit_time_violation": "财务可用时间或依赖时间不符合信号时点",
    "stale_financial": "报告期末距信号日超过配置期限",
    "missing_profit_ttm": "PIT归母净利润TTM缺失",
    "invalid_profit_ttm": "PIT归母净利润TTM非有限值",
    "missing_market_cap": "当日总市值缺失，不以前值回填",
    "invalid_market_cap": "总市值非有限值或不大于零",
    "missing_equity": "PIT平均净资产缺失",
    "invalid_equity": "PIT平均净资产非有限值或不大于零",
    "missing_roe": "PIT的ROE_TTM缺失，不以年化指标替代",
    "invalid_roe": "PIT的ROE_TTM非有限值",
    "unresolved_price_event": "价格窗口触及未解决公司行动",
    "insufficient_history": "全局预热或该股本地历史不足",
    "missing_endpoint": "所需月末复权价格端点缺失或无效",
    "incomplete_price_window": "统一交易日历上的价格窗口不完整",
    "incomplete_turnover_window": "20个市场交易日的换手率不完整或有非法值",
    "invalid_result": "计算结果非有限值",
}


def load_settings(path: Path) -> dict:
    """读取第一阶段配置；不让预处理或任意方向悄悄进入本阶段。"""
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if settings.get("version") != 1 or settings.get("stage") != "raw_factors":
        raise ValueError("只支持第一阶段raw_factors配置")
    if settings.get("factors") != NAMES or settings.get("directions") != DIRECTIONS:
        raise ValueError("五因子清单或事先固定的方向不符")
    integers = [("financial", "max_report_age_days"), ("momentum", "lookback_months"),
                ("momentum", "skip_months"), ("reversal", "lookback_months"), ("turnover", "window")]
    for section, field in integers:
        value = settings[section][field]
        if type(value) is not int or value <= 0:
            raise ValueError(f"因子窗口/期限须为正整数：{section}.{field}")
    if settings["momentum"]["skip_months"] >= settings["momentum"]["lookback_months"]:
        raise ValueError("动量跳过月数须小于回看月数")
    if settings["price_windows"]["require_complete"] is not True:
        raise ValueError("首版价格窗口必须完整，不能静默放宽")
    for section, field in [("financial", "market_cap_unit"), ("turnover", "percent_unit")]:
        value = float(settings[section][field])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("单位换算须为有效正数")
    return settings


def normalize_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """复制并统一时间精度，拒绝缺失或重复的股票/日期键。"""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).astype("datetime64[ns]")
    if result[["date", "code"]].isna().any().any() or result.duplicated(["date", "code"]).any():
        raise ValueError(f"{label}主键缺失或重复")
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def quality_frame(targets: pd.DataFrame, name: str, base: pd.Series,
                  flags: dict[str, pd.Series], direction: int) -> pd.DataFrame:
    """生成每股每因子的质量行；原值与带方向值均只在有效时输出。"""
    result = targets[["date", "code"]].copy()
    result["factor"] = name
    result["eligible"] = targets["tradable"].fillna(False).astype(bool).to_numpy()
    result["primary_reason"] = "ok"
    result["reason_codes"] = ""
    finite = pd.Series(np.isfinite(base.to_numpy(dtype=float)), index=result.index)
    assigned = pd.Series(False, index=result.index)
    for reason, mask in [*flags.items(), ("invalid_result", ~finite)]:
        mask = mask.fillna(True).astype(bool)
        if reason == "invalid_result":
            mask &= ~assigned
        first = mask & ~assigned
        result.loc[first, "primary_reason"] = reason
        prefix = result.loc[mask, "reason_codes"]
        result.loc[mask, "reason_codes"] = prefix.where(prefix.eq(""), prefix + ";") + reason
        assigned |= mask
    result["valid"] = ~assigned
    result["base_value"] = base.where(result["valid"]).astype(float)
    result["direction"] = direction
    result["value"] = result["base_value"] * direction
    result["window_start"] = pd.NaT
    result["window_end"] = pd.NaT
    result["latest_input_date"] = pd.NaT
    result["financial_end_date"] = pd.NaT
    result["ann_date_eff"] = pd.NaT
    result["input_max_ann_date"] = pd.NaT
    result["report_age_days"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["expected_observations"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["valid_observations"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["negative_profit"] = False
    result["crosses_inferred_boundary"] = False
    return result

