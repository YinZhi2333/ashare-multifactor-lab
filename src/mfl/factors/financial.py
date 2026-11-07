"""从现有PIT月末面板计算EP与ROE，不读取原始财报或数据商PE。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import quality_frame


def financial_factors(targets: pd.DataFrame, pit: pd.DataFrame,
                      prices: pd.DataFrame, settings: dict) -> list[pd.DataFrame]:
    """EP采用同日总市值，ROE复用已验收的PIT值；亏损不是缺失。"""
    data = targets[["date", "code"]].merge(pit, on=["date", "code"], how="left", validate="one_to_one")
    data = data.merge(prices[["date", "code", "total_mv"]], on=["date", "code"], how="left", validate="one_to_one")
    fields = ["end_date", "ann_date", "ann_date_eff", "input_max_ann_date"]
    for col in fields:
        data[col] = pd.to_datetime(data[col]).astype("datetime64[ns]")
    missing = data[fields].isna().any(axis=1)
    time_bad = ((data["ann_date_eff"] > data["date"]) | (data["input_max_ann_date"] >= data["date"])
                | (data["input_max_ann_date"] > data["ann_date"]) | (data["end_date"] > data["ann_date"])
                | (data["ann_date_eff"] != data["ann_date"] + pd.Timedelta(days=1))) & ~missing
    age = (data["date"] - data["end_date"]).dt.days
    shared = {"missing_pit": missing, "pit_time_violation": time_bad,
              "stale_financial": age.gt(settings["financial"]["max_report_age_days"])}
    profit = data["net_profit_ttm"]
    cap = data["total_mv"]
    equity, roe = data["avg_equity"], data["roe_ttm"]
    profit_flags = {"missing_profit_ttm": profit.isna(), "invalid_profit_ttm": profit.notna() & ~np.isfinite(profit)}
    flags_ep = {**shared, **profit_flags, "missing_market_cap": cap.isna(),
                "invalid_market_cap": cap.notna() & (~np.isfinite(cap) | cap.le(0))}
    ep = profit / (cap.where(np.isfinite(cap) & cap.gt(0)) * settings["financial"]["market_cap_unit"])
    flags_roe = {**shared, **profit_flags, "missing_equity": equity.isna(),
                 "invalid_equity": equity.notna() & (~np.isfinite(equity) | equity.le(0)),
                 "missing_roe": roe.isna(), "invalid_roe": roe.notna() & ~np.isfinite(roe)}
    results = []
    for name, base, flags in [("EP", ep, flags_ep), ("ROE_TTM", roe, flags_roe)]:
        q = quality_frame(targets, name, base, flags, settings["directions"][name])
        q["financial_end_date"] = data["end_date"]
        q["ann_date_eff"] = data["ann_date_eff"]
        q["input_max_ann_date"] = data["input_max_ann_date"]
        q["report_age_days"] = age.astype("Int64")
        q["latest_input_date"] = data["date"] if name == "EP" else data["ann_date_eff"]
        q["negative_profit"] = profit.lt(0) & np.isfinite(profit)
        results.append(q)
    return results

