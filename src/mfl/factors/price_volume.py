"""按统一交易日历计算月末动量、反转和20日换手率，不补造报价。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from price_quality import price_window_validity

from .core import quality_frame


def _lookup(grid: pd.DataFrame, dates: pd.Series, codes: pd.Series) -> pd.Series:
    """按每条目标记录的日期和代码取二维网格值，不做asof或前向填充。"""
    rows, cols = grid.index.get_indexer(dates), grid.columns.get_indexer(codes)
    result = np.full(len(dates), np.nan)
    valid = (rows >= 0) & (cols >= 0)
    result[valid] = grid.to_numpy()[rows[valid], cols[valid]]
    return pd.Series(result, index=dates.index)


def _counts(good: pd.DataFrame, start: pd.Series, end: pd.Series,
            codes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """用累计计数核对闭区间中每一个市场交易日，不按该股报价条数滚动。"""
    left, right = good.index.get_indexer(start), good.index.get_indexer(end)
    cols = good.columns.get_indexer(codes)
    valid = (left >= 0) & (right >= left) & (cols >= 0)
    count = np.zeros(len(start), dtype=np.int64)
    expected = np.zeros(len(start), dtype=np.int64)
    cumulative = np.vstack([np.zeros((1, good.shape[1]), dtype=np.int32),
                            np.cumsum(good.to_numpy(), axis=0, dtype=np.int32)])
    count[valid] = cumulative[right[valid] + 1, cols[valid]] - cumulative[left[valid], cols[valid]]
    expected[valid] = right[valid] - left[valid] + 1
    return pd.Series(count, index=start.index), pd.Series(expected, index=start.index)


def price_volume_factors(targets: pd.DataFrame, prices: pd.DataFrame, days: pd.DatetimeIndex,
                         month_ends: pd.Series, rules: list[dict], settings: dict) -> list[pd.DataFrame]:
    """先计算入池前历史，再对齐月末目标；严格窗口可能因历史停牌而留空。"""
    codes = pd.Index(targets["code"].unique())
    prices = prices[prices["date"].le(targets["date"].max()) & prices["code"].isin(codes)].copy()
    first = targets["code"].map(prices.groupby("code", observed=True)["date"].min())
    price_grid = prices.pivot(index="date", columns="code", values="close_adj").reindex(index=days, columns=codes)
    price_grid = price_grid.where(np.isfinite(price_grid) & price_grid.gt(0))
    good = price_grid.notna()
    months = targets["date"].dt.to_period("M")
    specs = [("MOM_12_1", settings["momentum"]["lookback_months"], settings["momentum"]["skip_months"]),
             ("REV_1M", settings["reversal"]["lookback_months"], 0)]
    results = []
    for name, lookback, skip in specs:
        start = (months - lookback).map(month_ends).astype("datetime64[ns]")
        end = (months - skip).map(month_ends).astype("datetime64[ns]")
        defined = start.notna() & end.notna()
        history_bad = ~defined | first.isna() | start.lt(first)
        left = _lookup(price_grid, start, targets["code"])
        right = _lookup(price_grid, end, targets["code"])
        count, expected = _counts(good, start, end, targets["code"])
        windows = pd.DataFrame({"code": targets["code"], "start": start, "end": end})
        blocked = pd.Series(False, index=targets.index)
        if defined.any():
            blocked.loc[defined] = ~price_window_validity(windows.loc[defined], rules)
        flags = {"unresolved_price_event": blocked, "insufficient_history": history_bad,
                 "missing_endpoint": defined & ~history_bad & (left.isna() | right.isna()),
                 "incomplete_price_window": defined & ~history_bad & count.lt(expected)}
        base = right / left - 1
        q = quality_frame(targets, name, base, flags, settings["directions"][name])
        q["window_start"], q["window_end"], q["latest_input_date"] = start, end, end
        q["expected_observations"], q["valid_observations"] = expected.astype("Int64"), count.astype("Int64")
        for rule in rules:
            if rule["action"] == "normalize":
                event = pd.to_datetime(rule["date"], format="%Y%m%d")
                q["crosses_inferred_boundary"] |= targets["code"].eq(rule["code"]) & start.lt(event) & end.ge(event)
        results.append(q)
    # 换手率不依赖复权价格，不因价格公司行动自动丢弃；缺报价仍会形成窗口缺口。
    turnover = prices.pivot(index="date", columns="code", values="turnover_rate").reindex(index=days, columns=codes)
    turnover = turnover.where(np.isfinite(turnover) & turnover.ge(0)) / settings["turnover"]["percent_unit"]
    window = settings["turnover"]["window"]
    positions = days.get_indexer(targets["date"])
    start = pd.Series(pd.NaT, index=targets.index, dtype="datetime64[ns]")
    enough = positions >= window - 1
    start.loc[enough] = days[positions[enough] - window + 1].to_numpy()
    end = targets["date"]
    count, expected = _counts(turnover.notna(), start, end, targets["code"])
    history_bad = start.isna() | first.isna() | start.lt(first)
    base = _lookup(turnover.rolling(window, min_periods=window).mean(), end, targets["code"])
    flags = {"insufficient_history": history_bad,
             "incomplete_turnover_window": ~history_bad & count.lt(expected)}
    q = quality_frame(targets, "TURN_20", base, flags, settings["directions"]["TURN_20"])
    q["window_start"], q["window_end"], q["latest_input_date"] = start, end, end
    q["expected_observations"] = pd.Series(window, index=q.index, dtype="Int64")
    q["valid_observations"] = count.astype("Int64")
    results.append(q)
    return results

