"""股票池：中证500 月度快照 + ST / 上市天数 / 停牌 / 涨跌停标记。

输出 data/interim/universe.parquet，每个调仓日一行（date, code）。
只保留当天属于中证500 的股票，避免把 5000 只全市场股票都展开。

用法：
    python src/mfl/data/universe.py --start 20150101 --end 20241231
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    INTERIM,
    month_end_dates,
    open_dates,
    read_calendar,
    read_optional,
    read_single,
    read_years,
    resolve_range,
    save_interim,
    setup_logging,
)

LOG = setup_logging("universe")

INDEX_CODE = "000905.SH"
MIN_LISTED_DAYS = 120


def load_index_members(start: str, end: str) -> pd.DataFrame:
    iw = read_single("index_weight", f"index_code={INDEX_CODE}", "data.parquet").copy()
    iw["date"] = pd.to_datetime(iw["trade_date"].astype(str), format="%Y%m%d")
    # 不截掉 start 之前的快照：2015 年 1 月要用 2014 年 12 月的成分股
    iw = iw[iw["date"] <= pd.to_datetime(end, format="%Y%m%d")]
    iw = iw.rename(columns={"con_code": "code"})
    iw = iw[["date", "code", "weight"]].drop_duplicates(["date", "code"])
    return iw.sort_values(["date", "code"]).reset_index(drop=True)


def snapshot_at_rebalance(iw: pd.DataFrame, rebalance: list[pd.Timestamp]) -> pd.DataFrame:
    """每个调仓日取 <= 该日的最近一期成分股快照。"""
    frames = []
    for d in rebalance:
        sub = iw[iw["date"] <= d]
        if sub.empty:
            continue
        snap = sub["date"].max()
        part = sub.loc[sub["date"] == snap, ["code", "weight"]].copy()
        part["date"] = d
        frames.append(part)
    if not frames:
        raise RuntimeError("index_weight 没有覆盖任何调仓日，检查成分股数据")
    return pd.concat(frames, ignore_index=True)


def attach_names(
    panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    namechange: pd.DataFrame | None,
) -> pd.DataFrame:
    base = stock_basic.drop_duplicates("ts_code").set_index("ts_code")["name"]
    panel = panel.copy()
    panel["name"] = panel["code"].map(base)

    if namechange is None or namechange.empty:
        LOG.warning("没有 namechange 数据，ST 判断退化为当前名称（有历史偏差）")
    else:
        nc = namechange.rename(columns={"ts_code": "code"}).copy()
        nc["start_date"] = pd.to_datetime(nc["start_date"], errors="coerce")
        nc["end_date"] = pd.to_datetime(nc["end_date"], errors="coerce")
        nc = nc.dropna(subset=["start_date"]).sort_values(["start_date", "code"])
        nc = nc.drop_duplicates(["code", "start_date"], keep="last")

        panel = pd.merge_asof(
            panel.sort_values(["date", "code"]),
            nc[["code", "name", "start_date", "end_date"]],
            by="code",
            left_on="date",
            right_on="start_date",
            direction="backward",
            suffixes=("", "_hist"),
        )
        # 名称区间已经结束的，说明当天没有有效更名记录
        valid = panel["end_date"].isna() | (panel["end_date"] >= panel["date"])
        panel.loc[~valid, "name_hist"] = np.nan
        panel["name"] = panel["name_hist"].fillna(panel["name"])
        panel = panel.drop(columns=["start_date", "end_date", "name_hist"])

    panel["name"] = panel["name"].fillna("")
    panel["is_st"] = panel["name"].str.contains("ST", case=False, regex=False)
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def attach_listed_days(
    panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    cal = pd.DatetimeIndex(dates)
    rank = pd.Series(np.arange(len(cal)), index=cal)

    ld = pd.to_datetime(
        stock_basic.drop_duplicates("ts_code").set_index("ts_code")["list_date"],
        format="%Y%m%d",
        errors="coerce",
    )
    ld = ld.dropna()
    # 上市日不是交易日时，取之后第一个交易日
    pos = cal.searchsorted(ld.values, side="left")
    list_rank = pd.Series(pos, index=ld.index)

    panel = panel.copy()
    panel["date_rank"] = panel["date"].map(rank)
    panel["list_rank"] = panel["code"].map(list_rank)
    panel["listed_days"] = panel["date_rank"] - panel["list_rank"] + 1
    return panel.drop(columns=["date_rank", "list_rank"])


def attach_price(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    prices_path = INTERIM / "prices.parquet"
    if prices_path.exists():
        px = pd.read_parquet(prices_path, columns=["date", "code", "close"])
    else:
        LOG.warning("没有 prices.parquet，改用原始 daily 判断当天是否交易")
        daily = read_years("daily", start, end).rename(columns={"ts_code": "code"}).copy()
        daily["date"] = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
        px = daily[["date", "code", "close"]]

    panel = panel.merge(px, on=["date", "code"], how="left", indicator=True)
    panel["suspended"] = panel["_merge"].eq("left_only")
    return panel.drop(columns="_merge")


def attach_limits(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    try:
        lim = read_years("stk_limit", start, end)
    except FileNotFoundError:
        LOG.warning("没有 stk_limit 数据，can_buy / can_sell 全部按 True 处理")
        panel = panel.copy()
        panel["up_limit"] = np.nan
        panel["down_limit"] = np.nan
        return panel

    lim = lim.rename(columns={"ts_code": "code"}).copy()
    lim["date"] = pd.to_datetime(lim["trade_date"], format="%Y%m%d")
    lim = lim[["date", "code", "up_limit", "down_limit"]].drop_duplicates(["date", "code"])
    return panel.merge(lim, on=["date", "code"], how="left")


def build_universe(start: str, end: str) -> pd.DataFrame:
    cal = read_calendar(start, end)
    dates = open_dates(cal)
    rebalance = month_end_dates(dates)
    LOG.info("交易日 %d 天，月末调仓日 %d 个", len(dates), len(rebalance))

    stock_basic = read_single("stock_basic", "data.parquet")
    namechange = read_optional("namechange", "data.parquet")
    iw = load_index_members(start, end)

    panel = snapshot_at_rebalance(iw, rebalance)
    LOG.info("成分股快照 %d 行，涉及 %d 只股票", len(panel), panel["code"].nunique())

    panel = attach_names(panel, stock_basic, namechange)
    panel = attach_listed_days(panel, stock_basic, dates)
    panel = attach_price(panel, start, end)
    panel = attach_limits(panel, start, end)

    panel["tradable"] = (
        (~panel["is_st"])
        & (panel["listed_days"] >= MIN_LISTED_DAYS)
        & (~panel["suspended"])
        & panel["close"].notna()
    )
    # 涨停买不进、跌停卖不出；limits 缺失时比较结果为 False，等价于“可以交易”
    panel["can_buy"] = ~(panel["close"] >= panel["up_limit"])
    panel["can_sell"] = ~(panel["close"] <= panel["down_limit"])

    cols = [
        "date",
        "code",
        "weight",
        "name",
        "is_st",
        "listed_days",
        "suspended",
        "close",
        "up_limit",
        "down_limit",
        "can_buy",
        "can_sell",
        "tradable",
    ]
    out = panel[[c for c in cols if c in panel.columns]].sort_values(["date", "code"])

    per_date = out.groupby("date").size()
    LOG.info(
        "每个调仓日 %.0f 只（最少 %d，最多 %d）；可交易 %.0f 只",
        per_date.mean(),
        int(per_date.min()),
        int(per_date.max()),
        out.groupby("date")["tradable"].sum().mean(),
    )
    LOG.info("ST 记录 %d 条，停牌记录 %d 条", int(out["is_st"].sum()), int(out["suspended"].sum()))
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建中证500 月度股票池")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()

    start, end = resolve_range(args.start, args.end)
    LOG.info("区间：%s ~ %s", start, end)

    universe = build_universe(start, end)
    path = save_interim(universe, "universe.parquet")
    LOG.info("universe 已写入 %s：%d 行", path, len(universe))

    rebalance = pd.DataFrame({"date": sorted(universe["date"].unique())})
    save_interim(rebalance, "rebalance_dates.parquet")


if __name__ == "__main__":
    main()
