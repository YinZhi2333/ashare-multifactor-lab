"""加工产物契约检查：分批检查行情，核对股票池与财务面板。

从项目根目录执行 python scripts/check_interim.py；不修改任何数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mfl" / "data"))
from common import INTERIM, setup_logging

LOG = setup_logging("check_interim")


def check_prices() -> None:
    parquet = pq.ParquetFile(INTERIM / "prices.parquet")
    seed = pd.DataFrame()
    last_key: tuple | None = None
    rows = missing = unavailable_returns = 0
    for batch in parquet.iter_batches(batch_size=250_000, columns=["date", "code", "close_adj", "adj_factor", "ret"]):
        df = batch.to_pandas()
        keys = pd.MultiIndex.from_frame(df[["date", "code"]])
        if not keys.is_unique or not keys.is_monotonic_increasing or (last_key is not None and keys[0] <= last_key):
            raise RuntimeError("prices 主键重复或日期/股票排序错误")
        last_key = keys[-1]
        bad = df["adj_factor"].isna()
        if df.loc[bad, ["close_adj", "ret"]].notna().any().any():
            raise RuntimeError("缺复权因子的行情产生了复权价或收益")
        # 分批维护每只股票上条观测，独立核对相邻收益，不把缺口变成零收益。
        df["_row"] = np.arange(len(df))
        work = pd.concat([seed, df], ignore_index=True).sort_values(["code", "date"])
        expected = work.groupby("code", observed=True)["close_adj"].pct_change(fill_method=None)
        comparable = work.groupby("code", observed=True).cumcount().gt(0) & work["_row"].ge(0)
        matches = np.isclose(work.loc[comparable, "ret"], expected.loc[comparable], rtol=1e-10, atol=1e-12, equal_nan=True)
        if not matches.all():
            raise RuntimeError("prices 相邻收益不一致，或跨过复权缺口")
        seed = work.drop_duplicates("code", keep="last").copy()
        seed["_row"] = -1
        rows += len(df)
        missing += int(bad.sum())
        unavailable_returns += int(df["ret"].isna().sum())
    LOG.info("prices %d 行；缺复权因子 %d 行；收益空值 %d 行；主键和相邻收益通过", rows, missing, unavailable_returns)


def check_panels() -> None:
    uni = pd.read_parquet(INTERIM / "universe.parquet")
    panel = pd.read_parquet(INTERIM / "fundamental_panel.parquet")
    events = pd.read_parquet(INTERIM / "fundamental_events.parquet")
    for name, df in [("universe", uni), ("fundamental_panel", panel)]:
        if df.duplicated(["date", "code"]).any():
            raise RuntimeError(f"{name} 主键重复")
    if not pd.MultiIndex.from_frame(uni[["date", "code"]]).equals(pd.MultiIndex.from_frame(panel[["date", "code"]])):
        raise RuntimeError("财务面板主键与股票池不一致")
    if events.duplicated(["code", "ann_date_eff"]).any():
        raise RuntimeError("财务事件主键重复")
    for name, df in [("events", events), ("panel", panel)]:
        known = df["ann_date"].notna()
        if df.loc[known, "input_max_ann_date"].isna().any():
            raise RuntimeError(f"{name} 缺少依赖时间审计")
        bad = (df["input_max_ann_date"] > df["ann_date"]) | (df["end_date"] > df["ann_date"])
        bad |= known & (df["ann_date_eff"] != df["ann_date"] + pd.Timedelta(days=1))
        if name == "panel":
            bad |= df["ann_date_eff"] > df["date"]
            ordered = df.sort_values(["code", "date"])
        else:
            ordered = df.sort_values(["code", "ann_date_eff"])
        if bad.any() or (ordered["end_date"] < ordered.groupby("code", observed=True)["end_date"].cummax()).any():
            raise RuntimeError(f"{name} 有未来依赖或报告期倒退")
    blocked = uni["suspended"].fillna(True) | uni["close"].isna()
    if uni.loc[blocked, ["can_buy", "can_sell"]].any().any():
        raise RuntimeError("停牌/状态未知/无行情的股票被允许成交")
    if uni.loc[~uni["name_known"], "can_buy"].any():
        raise RuntimeError("未知历史 ST 状态被允许新建仓")
    counts = uni.groupby("date").size()
    LOG.info("universe %d 行，%d 期，每期 %d~%d 只；上市年龄精确 %.2f%%", len(uni), len(counts), counts.min(), counts.max(), uni["listed_days_exact"].mean() * 100)
    LOG.info("历史名称覆盖 %.2f%%；平均可新建仓 %.1f 只；确认停牌 %d 条", uni["name_known"].mean() * 100, uni.groupby("date")["tradable"].sum().mean(), int(uni["suspended"].fillna(False).sum()))
    LOG.info("events %d 行；panel %d 行，ROE_TTM 覆盖 %.2f%%；未来依赖/报告期倒退 0 条", len(events), len(panel), panel["roe_ttm"].notna().mean() * 100)


def main() -> None:
    check_prices()
    check_panels()
    LOG.info("OK: interim 主键、收益、时间审计与交易标记检查通过")


if __name__ == "__main__":
    main()
