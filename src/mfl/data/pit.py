"""PIT 财务对齐：把财报变成「什么时候能知道」的事件表，再 as-of 到调仓日。

两步：

1. 事件表 fundamental_events.parquet
   每行是一个 (code, end_date, ann_date) 的财报版本，字段包括 TTM 净利润、
   平均净资产、ROE_TTM 等。ann_date 之后（严格说是次日）才允许使用。

2. 面板 fundamental_panel.parquet
   用 merge_asof 把事件表对齐到 universe.parquet 的 (date, code) 上，
   每个调仓日只取 ann_date <= date 的最近一期。没有 universe 时只输出事件表。

累计值处理：
   单季 = 本期累计 - 上期累计（同一财年，季度连续）
   TTM  = 本期累计 + 上年年报 - 上年同期累计

用法：
    python src/mfl/data/pit.py --start 20150101 --end 20241231

依赖：
    data/raw/fina_indicator/data.parquet        必需
    data/raw/income/data.parquet                可选；缺省时退化为 roe_yearly
    data/raw/balancesheet/data.parquet          可选
    data/interim/universe.parquet               可选；缺省时只输出事件表

已知简化：
   同一个 (code, end_date) 若有多条公告（财报修正），保留 ann_date 最新的一条。
   严格的 PIT 应保留所有版本、按知识时间逐版本生效，这里为了复杂度做了折中。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    INTERIM,
    read_optional,
    resolve_range,
    save_interim,
    setup_logging,
)

LOG = setup_logging("pit")

FI_COLS = [
    "roe",
    "roe_waa",
    "roe_dt",
    "roe_yearly",
    "roa",
    "netprofit_yoy",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "eps",
    "bps",
    "profit_dedt",
]


def _to_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")


def _load_fina() -> pd.DataFrame | None:
    df = read_optional("fina_indicator", "data.parquet")
    if df is None:
        return None
    df = df.rename(columns={"ts_code": "code"})
    df["end_date"] = _to_date(df["end_date"])
    df["ann_date"] = _to_date(df["ann_date"])
    df = df.dropna(subset=["code", "end_date", "ann_date"])
    df = df[df["end_date"].dt.month.isin([3, 6, 9, 12])]

    for c in FI_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df.sort_values(["code", "end_date", "ann_date"])
    df = df.drop_duplicates(["code", "end_date"], keep="last")
    return df[["code", "end_date", "ann_date"] + FI_COLS].reset_index(drop=True)


def _load_statements(name: str, cols: list[str]) -> pd.DataFrame | None:
    df = read_optional(name, "data.parquet")
    if df is None:
        return None

    df = df.rename(columns={"ts_code": "code"}).copy()
    if "report_type" in df.columns:
        rt = pd.to_numeric(df["report_type"], errors="coerce")
        df = df[rt == 1]  # 1 = 合并报表

    df["end_date"] = _to_date(df["end_date"])
    df["ann_date"] = _to_date(df["ann_date"])
    df = df.dropna(subset=["code", "end_date", "ann_date"])

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df.sort_values(["code", "end_date", "ann_date"])
    df = df.drop_duplicates(["code", "end_date"], keep="last")
    return df[["code", "end_date", "ann_date"] + cols].reset_index(drop=True)


def _build_ttm(inc: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    df = inc.merge(bs, on=["code", "end_date"], how="inner", suffixes=("_inc", "_bs"))
    # 两个报表都公告后才算“知道”，取较晚的公告日
    df["ann_date"] = df[["ann_date_inc", "ann_date_bs"]].max(axis=1)
    df = df.sort_values(["code", "end_date"]).reset_index(drop=True)

    df["quarter"] = df["end_date"].dt.quarter
    df["year"] = df["end_date"].dt.year
    g = df.groupby("code", observed=True)
    df["prev_ytd"] = g["n_income_attr_p"].shift(1)
    df["prev_q"] = g["quarter"].shift(1)
    df["prev_y"] = g["year"].shift(1)

    # 累计 -> 单季
    df["q_profit"] = df["n_income_attr_p"] - df["prev_ytd"]
    q1 = df["quarter"] == 1
    df.loc[q1, "q_profit"] = df.loc[q1, "n_income_attr_p"]
    consecutive = (df["year"] == df["prev_y"]) & (df["quarter"] == df["prev_q"] + 1)
    df.loc[(~q1) & (~consecutive), "q_profit"] = np.nan

    # TTM = 本期累计 + 上年年报 - 上年同期累计
    df["prev_year_end"] = df["end_date"] - pd.DateOffset(years=1)
    df["prev_fy_end"] = pd.to_datetime(
        (df["year"] - 1).astype(str) + "1231", format="%Y%m%d"
    )

    annual = (
        df.loc[df["quarter"] == 4, ["code", "end_date", "n_income_attr_p"]]
        .rename(columns={"end_date": "fy_end", "n_income_attr_p": "last_annual"})
    )
    same = (
        df[["code", "end_date", "n_income_attr_p"]]
        .rename(columns={"end_date": "prev_year_end", "n_income_attr_p": "last_same_ytd"})
    )
    df = df.merge(annual, left_on=["code", "prev_fy_end"], right_on=["code", "fy_end"], how="left")
    df = df.merge(same, on=["code", "prev_year_end"], how="left")
    df["net_profit_ttm"] = df["n_income_attr_p"] + df["last_annual"] - df["last_same_ytd"]

    # 平均净资产 = (期末 + 上年同期) / 2
    eq = (
        df[["code", "end_date", "total_hldr_eqy_exc_min_int"]]
        .rename(columns={
            "end_date": "prev_year_end",
            "total_hldr_eqy_exc_min_int": "equity_prev",
        })
    )
    df = df.merge(eq, on=["code", "prev_year_end"], how="left")
    df["equity"] = df["total_hldr_eqy_exc_min_int"]
    df["avg_equity"] = (df["equity"] + df["equity_prev"]) / 2.0
    df["roe_ttm"] = np.where(
        df["avg_equity"] > 0,
        df["net_profit_ttm"] / df["avg_equity"],
        np.nan,
    )
    return df


def _attach_fina(df: pd.DataFrame, fi: pd.DataFrame) -> pd.DataFrame:
    add = fi[
        ["code", "end_date", "ann_date", "roe_yearly", "grossprofit_margin",
         "debt_to_assets", "eps", "bps"]
    ].rename(columns={"ann_date": "ann_date_fi"})

    df = df.merge(add, on=["code", "end_date"], how="left")
    df["ann_date"] = df[["ann_date", "ann_date_fi"]].max(axis=1)
    return df.drop(columns=["ann_date_fi"])


def build_events(start: str, end: str) -> pd.DataFrame:
    fi = _load_fina()
    if fi is not None:
        LOG.info("fina_indicator：%d 条（已按 end_date 去重）", len(fi))
    else:
        LOG.warning("没有 fina_indicator：roe_yearly / 毛利率 / 负债率等字段会留空")

    inc = _load_statements("income", ["n_income_attr_p", "revenue"])
    bs = _load_statements("balancesheet", ["total_hldr_eqy_exc_min_int", "total_assets"])

    use_ttm = (
        inc is not None
        and bs is not None
        and inc["n_income_attr_p"].notna().any()
        and bs["total_hldr_eqy_exc_min_int"].notna().any()
    )

    if not use_ttm:
        if fi is None:
            raise RuntimeError("income/balancesheet 和 fina_indicator 都不可用，无法构造财务事件")
        LOG.warning("缺 income/balancesheet，质量因子退化为 roe_yearly（不是 TTM）")
        events = fi.copy()
        events["roe_ttm"] = events["roe_yearly"]
        events["net_profit_ttm"] = np.nan
        events["avg_equity"] = np.nan
        events["source"] = "fina_yearly"
    else:
        events = _build_ttm(inc, bs)
        if fi is not None:
            events = _attach_fina(events, fi)
        else:
            for col in ["roe_yearly", "grossprofit_margin", "debt_to_assets", "eps", "bps"]:
                events[col] = np.nan
        events["source"] = "ttm"
        LOG.info(
            "TTM 事件：%d 条，ROE_TTM 非空 %.1f%%",
            len(events),
            events["roe_ttm"].notna().mean() * 100,
        )

    events["ann_date_eff"] = events["ann_date"] + pd.Timedelta(days=1)

    hi = pd.to_datetime(end, format="%Y%m%d")
    # 不截掉 start 之前的财报：2015 年初的 as-of 需要 2014 年报
    events = events[events["end_date"] <= hi]

    bad = int((events["ann_date"] < events["end_date"]).sum())
    if bad:
        LOG.warning("ann_date < end_date 的异常记录：%d 条", bad)

    keep = [
        "code", "end_date", "ann_date", "ann_date_eff", "source",
        "roe_ttm", "roe_yearly", "net_profit_ttm", "avg_equity",
        "grossprofit_margin", "debt_to_assets", "eps", "bps",
    ]
    events = events[[c for c in keep if c in events.columns]]
    return events.sort_values(["code", "ann_date_eff"]).reset_index(drop=True)


def build_panel(events: pd.DataFrame) -> pd.DataFrame | None:
    universe_path = INTERIM / "universe.parquet"
    if not universe_path.exists():
        LOG.warning("没有 universe.parquet，跳过面板；先跑 universe.py")
        return None

    uni = pd.read_parquet(universe_path, columns=["date", "code"])
    left = uni.drop_duplicates(["date", "code"]).sort_values("date")

    # 同一公告日可能有多期（年报+一季报），保留 end_date 最新的那条
    right = (
        events.sort_values(["ann_date_eff", "end_date"])
        .drop_duplicates(["code", "ann_date_eff"], keep="last")
    )

    panel = pd.merge_asof(
        left,
        right,
        by="code",
        left_on="date",
        right_on="ann_date_eff",
        direction="backward",
    )

    leak = int((panel["ann_date_eff"] > panel["date"]).sum())
    if leak:
        raise RuntimeError(f"PIT 对齐出现 {leak} 条未来数据，检查 merge_asof")

    cov = panel["roe_ttm"].notna().mean()
    LOG.info("PIT 面板：%d 行，ROE_TTM 覆盖率 %.1f%%", len(panel), cov * 100)

    panel["year"] = panel["date"].dt.year
    yearly = panel.groupby("year")["roe_ttm"].apply(lambda s: s.notna().mean())
    LOG.info("分年覆盖率：\n%s", yearly.round(2).to_string())
    panel = panel.drop(columns="year")

    keep = [
        "date", "code", "end_date", "ann_date", "roe_ttm", "roe_yearly",
        "net_profit_ttm", "avg_equity", "grossprofit_margin",
        "debt_to_assets", "eps", "bps",
    ]
    return panel[[c for c in keep if c in panel.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT 财务对齐")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()

    start, end = resolve_range(args.start, args.end)
    LOG.info("区间：%s ~ %s", start, end)

    events = build_events(start, end)
    path = save_interim(events, "fundamental_events.parquet")
    LOG.info("events 已写入 %s：%d 行", path, len(events))

    panel = build_panel(events)
    if panel is not None:
        path = save_interim(panel, "fundamental_panel.parquet")
        LOG.info("panel 已写入 %s：%d 行", path, len(panel))


if __name__ == "__main__":
    main()
