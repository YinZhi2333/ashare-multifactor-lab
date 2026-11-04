"""复权与行情面板：raw -> data/interim/prices.parquet。

只做四件事：
1. daily + adj_factor 合成后复权价，算日收益；
2. 合并 daily_basic（市值、换手、PE/PB）；
3. 生成 vwap（成交额/成交量）；
4. 做一致性检查，把问题打在日志里。

用法：
    python src/mfl/data/adjust.py --start 20150101 --end 20241231
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    read_single,
    read_years,
    resolve_range,
    save_interim,
    setup_logging,
)

LOG = setup_logging("adjust")

BASIC_COLS = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "turnover_rate_f",
    "total_share",
    "float_share",
    "total_mv",
    "circ_mv",
    "pe_ttm",
    "pb",
]


def _dedupe(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    n = int(df.duplicated(keys).sum())
    if n:
        LOG.warning("原始数据有 %d 条重复的 %s，已去重", n, keys)
        df = df.drop_duplicates(keys)
    return df


def build_prices(start: str, end: str) -> pd.DataFrame:
    daily = _dedupe(read_years("daily", start, end), ["ts_code", "trade_date"])
    adj = _dedupe(read_years("adj_factor", start, end), ["ts_code", "trade_date"])
    try:
        basic = _dedupe(read_years("daily_basic", start, end), ["ts_code", "trade_date"])
    except FileNotFoundError:
        LOG.warning("没有 daily_basic：市值/换手/PE 字段留空，EP、换手类因子将不可用")
        basic = pd.DataFrame(columns=BASIC_COLS)

    LOG.info(
        "daily %d 行 | adj_factor %d 行 | daily_basic %d 行",
        len(daily),
        len(adj),
        len(basic),
    )

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    missing_adj = float(df["adj_factor"].isna().mean())
    if missing_adj > 0:
        LOG.warning("adj_factor 缺失 %.4f%%，相关行会丢弃", missing_adj * 100)
    df = df.dropna(subset=["adj_factor"]).copy()

    # 后复权价：收益用这个，市值/换手/涨跌停仍然用不复权价
    df["close_adj"] = df["close"] * df["adj_factor"]
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["ret"] = df.groupby("ts_code", observed=True)["close_adj"].pct_change(fill_method=None)

    use_cols = [c for c in BASIC_COLS if c in basic.columns]
    if use_cols and not basic.empty:
        df = df.merge(
            basic[use_cols],
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
    else:
        LOG.warning("跳过 daily_basic 合并：prices 里不会有 total_mv / turnover_rate / pe_ttm")

    df = df.rename(columns={"ts_code": "code", "trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Tushare 单位：vol=手（100 股），amount=千元。vwap(元/股) = amount*1000 / (vol*100)
    df["vwap"] = np.where(df["volume"] > 0, df["amount"] * 10.0 / df["volume"], np.nan)

    # --- 一致性检查 ---
    mask = df["ret"].notna() & df["pct_chg"].notna()
    if mask.any():
        n = int(mask.sum())
        sample = df.loc[mask, ["ret", "pct_chg"]].sample(min(200_000, n), random_state=0)
        corr = float(sample["ret"].corr(sample["pct_chg"] / 100.0))
        LOG.info("复权收益与 pct_chg 相关系数：%.5f（抽样 %d 行）", corr, len(sample))
        if corr < 0.99:
            LOG.warning("相关系数偏低，检查 adj_factor 是否和行情对齐")

    extreme = int((df["ret"].abs() > 0.5).sum())
    LOG.info("|ret| > 50%% 的记录：%d 条（除权、重组复牌附近可能出现）", extreme)

    cols = [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "vwap",
        "adj_factor",
        "close_adj",
        "ret",
        "turnover_rate",
        "turnover_rate_f",
        "total_share",
        "float_share",
        "total_mv",
        "circ_mv",
        "pe_ttm",
        "pb",
    ]
    out = df[[c for c in cols if c in df.columns]]
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def build_index(start: str, end: str) -> pd.DataFrame:
    idx = read_single("index_daily", "ts_code=000905.SH", "data.parquet")
    idx = idx[(idx["trade_date"] >= start) & (idx["trade_date"] <= end)].copy()
    idx["date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ret"] = idx["close"].pct_change(fill_method=None)
    out = idx[["date", "close", "ret"]].rename(columns={"close": "index_close"})
    save_interim(out, "index.parquet")
    LOG.info("index：%d 行", len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="复权与行情面板")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()

    start, end = resolve_range(args.start, args.end)
    LOG.info("区间：%s ~ %s", start, end)

    prices = build_prices(start, end)
    path = save_interim(prices, "prices.parquet")
    LOG.info("prices 已写入 %s：%d 行，%d 只股票", path, len(prices), prices["code"].nunique())

    try:
        build_index(start, end)
    except FileNotFoundError as exc:
        LOG.warning("跳过指数基准：%s", exc)


if __name__ == "__main__":
    main()
