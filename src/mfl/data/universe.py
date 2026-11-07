"""中证500 月末信号股票池及当日状态；不代表下一交易日一定可成交。

历史名称、上市年龄或停牌状态不可确认时保留未知，不用当前名称回填历史。
只读 raw，输出 universe.parquet / rebalance_dates.parquet。

输出主键为 (date, code)，date 是月末信号日，snapshot_date 是所用成分股
快照日。weight 沿用原始指数权重，不是策略目标权重。is_st、suspended
使用可空布尔值：未知与 False 含义不同，不能直接把未知当成可交易。
tradable 表示新建仓资格；can_buy/can_sell 另外区分买入和卖出约束。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW, month_end_dates, open_dates, read_optional, read_single, resolve_range, save_interim, setup_logging

LOG = setup_logging("universe")
INDEX_CODE = "000905.SH"
MIN_LISTED_DAYS = 120  # 新建仓至少上市 120 个交易日，计数包含上市后的首个交易日。


def load_index_members(start: str, end: str) -> pd.DataFrame:
    """读取不晚于 end 的历史成分股快照，检查 (date, code) 权重唯一。

    start 保留在调用接口中，但不据此截断：首个调仓日可能需要 start 前
    的最近快照。历史成分股不可用当前成分股替代，否则会有幸存者偏差。
    """
    iw = read_single("index_weight", f"index_code={INDEX_CODE}", "data.parquet").copy()
    iw["date"] = pd.to_datetime(iw["trade_date"].astype(str), format="%Y%m%d").astype("datetime64[ns]")
    iw = iw[iw["date"] <= pd.to_datetime(end, format="%Y%m%d")]
    iw = iw.rename(columns={"con_code": "code"})[["date", "code", "weight"]].drop_duplicates()
    if iw.duplicated(["date", "code"]).any():
        raise ValueError("index_weight 同一 (date, code) 有冲突权重")
    return iw.sort_values(["date", "code"]).reset_index(drop=True)


def snapshot_at_rebalance(iw: pd.DataFrame, rebalance: list[pd.Timestamp]) -> pd.DataFrame:
    """每个调仓日用不晚于当日的最近快照，保留快照日期以便审计。"""
    frames = []
    stale = 0
    # 以调仓日为粒度选整期快照，不能逐股取最近记录，否则已剔除股票会残留。
    for d in rebalance:
        sub = iw[iw["date"] <= d]
        if sub.empty:
            raise RuntimeError(f"index_weight 未覆盖调仓日 {d.date()}，不能静默跳过")
        snap = sub["date"].max()
        # 月度数据允许上月快照；超过一个自然月的缺口必须告警。
        stale += int(snap.to_period("M") < d.to_period("M") - 1)
        part = sub.loc[sub["date"] == snap, ["code", "weight"]].copy()
        part["snapshot_date"] = snap
        part["date"] = d
        frames.append(part)
    if not frames:
        raise RuntimeError("指定区间没有可确认的完整月末调仓日")
    if stale:
        LOG.warning("%d 个调仓日使用超过一个月的旧成分股快照", stale)
    return pd.concat(frames, ignore_index=True)


def attach_names(panel: pd.DataFrame, stock_basic: pd.DataFrame, namechange: pd.DataFrame | None) -> pd.DataFrame:
    """按历史名称的知识时间连接面板，增加 name/name_known/is_st。

    namechange 必须有 ts_code、name、start_date、end_date，可另有 ann_date。
    stock_basic 参数保留以兼容调用，但不使用其中的当前名称回填历史。
    名称尚未生效、已失效、缺失或同日冲突时，ST 状态均保留 pd.NA。
    """
    panel = panel.copy()
    panel["date"] = panel["date"].astype("datetime64[ns]")
    panel["name"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    if namechange is not None and not namechange.empty:
        nc = namechange.rename(columns={"ts_code": "code"}).copy()
        nc["start_date"] = pd.to_datetime(nc["start_date"], errors="coerce").astype("datetime64[ns]")
        nc["end_date"] = pd.to_datetime(nc["end_date"], errors="coerce").astype("datetime64[ns]")
        nc["known_from"] = nc["start_date"]
        # 名称既要已生效，也要已公告；两者较晚者才是此处可知的起点。
        if "ann_date" in nc:
            nc["known_from"] = pd.concat([nc["start_date"], pd.to_datetime(nc["ann_date"], errors="coerce").astype("datetime64[ns]")], axis=1).max(axis=1)
        nc = nc.dropna(subset=["code", "start_date", "known_from", "name"])
        nc = nc.sort_values(["known_from", "start_date", "code"]).drop_duplicates()
        # 同一知识时点出现冲突名称，不能由文件行序决定历史 ST 状态。
        conflict = nc.groupby(["code", "known_from"])["name"].transform("nunique").gt(1)
        if conflict.any():
            LOG.warning("%d 个历史名称时点有冲突，名称和 ST 状态保留未知",
                        len(nc.loc[conflict, ["code", "known_from"]].drop_duplicates()))
            nc["name"] = nc["name"].astype("string")
            nc.loc[conflict, "name"] = pd.NA
        nc = nc.drop_duplicates(["code", "known_from"], keep="last")
        if not nc.empty:
            # merge_asof 要求时间键整体递增；by=code 只限制同股匹配，不替代排序。
            panel = pd.merge_asof(
                panel.drop(columns="name").sort_values(["date", "code"]),
                nc[["code", "name", "known_from", "end_date"]].sort_values("known_from"),
                by="code", left_on="date", right_on="known_from", direction="backward",
            )
            # 向后匹配只能保证已开始，还须检查有效期结束；空 end_date 表示未结束。
            valid = panel["known_from"].notna() & (panel["end_date"].isna() | (panel["end_date"] >= panel["date"]))
            panel["name"] = panel["name"].astype("string").where(valid)
            panel = panel.drop(columns=["known_from", "end_date"])
    panel["name_known"] = panel["name"].notna() & panel["name"].str.len().gt(0).fillna(False)
    panel["is_st"] = panel["name"].str.contains("ST", case=False, regex=False).astype("boolean")
    panel.loc[~panel["name_known"], "is_st"] = pd.NA
    LOG.info("历史名称已确认 %.2f%%；未知 %d 条", panel["name_known"].mean() * 100, int((~panel["name_known"]).sum()))
    if not panel["name_known"].all():
        LOG.warning("历史名称未知的股票不允许新建仓；不以当前名称回填历史")
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def _offline_calendar() -> pd.DatetimeIndex | None:
    """读取已有 AkShare 包内离线日历，不调用联网接口。"""
    spec = importlib.util.find_spec("akshare")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(spec.submodule_search_locations[0]) / "data" / "calendar.json"
    if not path.exists():
        return None
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return pd.DatetimeIndex(pd.to_datetime(values, format="%Y%m%d")).as_unit("ns").sort_values().unique()
    except (ValueError, TypeError, OSError) as exc:
        LOG.warning("离线日历不可读，仅使用 raw 日历：%s", exc)
        return None


def _age_calendar(raw_cal: pd.DataFrame) -> list[pd.Timestamp]:
    """raw 覆盖内以 raw 为准，离线日历仅补充上市年龄所需历史。"""
    dates = open_dates(raw_cal)
    offline = _offline_calendar()
    if offline is None:
        LOG.warning("没有离线历史日历，早于 raw 起点的上市年龄仅给已观察下界")
        return dates
    lo, hi = raw_cal["date"].min(), raw_cal["date"].max()
    overlap = offline[(offline >= lo) & (offline <= hi)]
    if set(overlap) != set(dates):
        LOG.warning("包内日历与 raw 重叠不一致，拒绝拼接，上市年龄使用 raw 下界")
        return dates
    # 只补 raw 起点之前的历史，既不覆盖本地权威区间，也不延伸研究终点。
    out = offline[offline < lo].union(pd.DatetimeIndex(dates)).sort_values()
    LOG.info("上市年龄使用 AkShare 包内离线 calendar.json + raw 日历，%s ~ %s；重叠一致", out.min().date(), out.max().date())
    return out.tolist()


def attach_listed_days(panel: pd.DataFrame, stock_basic: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """按交易日历增加上市年龄及其可信程度，保留输入面板的行。

    dates 应为经过一致性检查的交易日序列；非交易日上市从随后首个交易日
    起计数。listed_days 仅在日历覆盖上市日起点时给精确值，否则为空；
    listed_days_lower_bound 给可证明的下界，listed_days_exact 标识是否精确。
    下界已满 120 天即可确认年龄合格，无须伪造精确上市年龄。
    """
    cal = pd.DatetimeIndex(dates).as_unit("ns").sort_values().unique()
    if cal.empty:
        raise ValueError("上市年龄计算没有交易日历")
    listed = pd.to_datetime(stock_basic.drop_duplicates("ts_code").set_index("ts_code")["list_date"], format="%Y%m%d", errors="coerce")
    panel = panel.copy()
    panel["date"] = panel["date"].astype("datetime64[ns]")
    ld = panel["code"].map(listed)
    valid = ld.notna() & panel["date"].isin(cal) & ld.le(panel["date"])
    exact = valid & ld.ge(cal.min())
    # 覆盖起点之前 searchsorted 返回0，此时结果仅是已观察下界。
    positions = cal.searchsorted(ld.fillna(cal.max()).to_numpy(), side="left")
    # 左闭右闭区间用左端插入位置与右端插入位置之差，包含信号日自身。
    end_positions = cal.searchsorted(panel["date"].to_numpy(), side="right")
    lower = pd.Series(end_positions - positions, index=panel.index).where(valid)
    panel["listed_days_lower_bound"] = lower.astype("Int64")
    panel["listed_days_exact"] = exact
    panel["listed_days"] = lower.where(exact).astype("Int64")
    LOG.info("上市年龄精确 %d 条；仅有下界 %d 条；上市日无效 %d 条", int(exact.sum()), int((valid & ~exact).sum()), int((~valid).sum()))
    return panel


def _market_rows(table: str, columns: list[str], codes: list[str], end: str) -> pd.DataFrame:
    """只读现有按年 raw，列投影并限制股票；保留起点前状态用于停复牌判断。"""
    frames = []
    for path in sorted((RAW / table).glob("year=*/data.parquet")):
        year = path.parent.name.removeprefix("year=")
        if int(year) > int(end[:4]):
            continue
        frames.append(pd.read_parquet(path, columns=columns, filters=[("ts_code", "in", codes), ("trade_date", "<=", end)]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def attach_price(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """增加信号日不复权收盘价及可空的全天停牌状态。

    有当日 raw 行情即按非全天停牌处理；无行情时，只有仍有效的全天停牌
    事件才能确认 suspended=True，其他情况保留未知。start 为兼容参数，
    实际读取 end 之前的历史，以免截掉区间前开始、尚未结束的停牌状态。
    """
    panel = panel.copy()
    panel["date"] = panel["date"].astype("datetime64[ns]")
    codes = panel["code"].unique().tolist()
    daily = _market_rows("daily", ["ts_code", "trade_date", "close"], codes, end)
    if daily.empty:
        raise FileNotFoundError("没有匹配股票的 raw/daily，不能判断交易状态")
    px = daily.rename(columns={"ts_code": "code", "trade_date": "last_trade_date"})
    px["last_trade_date"] = pd.to_datetime(px["last_trade_date"], format="%Y%m%d").astype("datetime64[ns]")
    px = px.drop_duplicates()
    if px.duplicated(["code", "last_trade_date"]).any():
        raise ValueError("daily 同一 (date, code) 有冲突行情")
    panel = pd.merge_asof(panel.sort_values(["date", "code"]), px.sort_values("last_trade_date"), by="code", left_on="date", right_on="last_trade_date", direction="backward")
    # 最近成交日用于判断旧停牌事件是否仍有效，旧收盘价不能冒充当天行情。
    quote = panel["last_trade_date"].eq(panel["date"])
    panel["close"] = panel["close"].where(quote)
    panel["suspended"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    panel.loc[quote, "suspended"] = False
    susp = _market_rows("suspend_d", ["ts_code", "trade_date", "suspend_type", "suspend_timing"], codes, end)
    if not susp.empty:
        susp = susp.rename(columns={"ts_code": "code", "trade_date": "susp_date"})
        susp["susp_date"] = pd.to_datetime(susp["susp_date"], format="%Y%m%d").astype("datetime64[ns]")
        is_s = susp["suspend_type"].eq("S")
        timing = susp["suspend_timing"].fillna("").astype(str).str.strip()
        susp["full_s"] = is_s & (timing.eq("") | timing.eq("09:30-15:00"))
        susp["partial_s"] = is_s & ~susp["full_s"]
        susp["resume"] = susp["suspend_type"].eq("R")
        state = susp.groupby(["code", "susp_date"], as_index=False)[["full_s", "partial_s", "resume"]].any()
        # S/R 同日且没有顺序，或仅有盘中停牌，都不能认定为全天停牌。
        state["confirmed_full_s"] = state["full_s"] & ~state["partial_s"] & ~state["resume"]
        panel = pd.merge_asof(panel.sort_values(["date", "code"]), state[["code", "susp_date", "confirmed_full_s"]].sort_values("susp_date"), by="code", left_on="date", right_on="susp_date", direction="backward")
        # 最近事件必须是全天停牌，且其后没有实际成交；复牌事件会清除此证据。
        active = panel["confirmed_full_s"].fillna(False) & (panel["last_trade_date"].isna() | panel["susp_date"].gt(panel["last_trade_date"]))
        panel.loc[~quote & active, "suspended"] = True
        conflicts = quote & panel["susp_date"].eq(panel["date"]) & panel["confirmed_full_s"].fillna(False)
        if conflicts.any():
            LOG.warning("%d 条全天停牌标记与实际 daily 行情冲突，按行情标为非全天停牌", int(conflicts.sum()))
        panel = panel.drop(columns=["susp_date", "confirmed_full_s"])
    else:
        LOG.warning("没有 suspend_d，行情缺失时停牌状态保留未知")
    panel = panel.drop(columns="last_trade_date")
    LOG.info("停牌已确认 %d 条；无行情且状态未知 %d 条", int(panel["suspended"].fillna(False).sum()), int(panel["suspended"].isna().sum()))
    return panel


def attach_limits(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按 (date, code) 左连接原始涨跌停价；缺失保留空值、冲突则报错。

    限价与 close 均为不复权价格。start 为兼容参数，读取截至 end 的记录；
    连接仍仅匹配面板当天，不向前填充旧限价。
    """
    lim = _market_rows("stk_limit", ["ts_code", "trade_date", "up_limit", "down_limit"], panel["code"].unique().tolist(), end)
    if lim.empty:
        LOG.warning("没有 stk_limit，涨跌停状态未知，禁止相关方向成交")
        return panel.assign(up_limit=np.nan, down_limit=np.nan)
    lim = lim.rename(columns={"ts_code": "code", "trade_date": "date"})
    lim["date"] = pd.to_datetime(lim["date"], format="%Y%m%d").astype("datetime64[ns]")
    lim = lim.drop_duplicates()
    if lim.duplicated(["date", "code"]).any():
        raise ValueError("stk_limit 同一 (date, code) 有冲突价格")
    return panel.merge(lim, on=["date", "code"], how="left", validate="one_to_one")


def build_universe(start: str, end: str) -> pd.DataFrame:
    """构造区间内完整月末的历史中证500 信号股票池，不写盘。

    Parameters
    ----------
    start, end : str
        YYYYMMDD 格式的信号日闭区间，月中 end 不产生额外调仓日。

    Returns
    -------
    pd.DataFrame
        以 (date, code) 唯一标识的成分股、状态及方向性交易标记。
        未通过筛选的成分股仍保留，供审计和已有持仓的卖出判断使用。
        这些标记仅反映信号日，实际成交日必须再次检查行情与限制。
    """
    raw_cal = read_single("trade_cal", "data.parquet").copy()
    raw_cal["date"] = pd.to_datetime(raw_cal["cal_date"], format="%Y%m%d").astype("datetime64[ns]")
    raw_cal["is_open"] = pd.to_numeric(raw_cal["is_open"], errors="raise")
    raw_cal = raw_cal.sort_values("date").drop_duplicates("date")
    # 在完整 raw 日历中确定月末，月中 --end 不会制造假调仓日。
    dates = open_dates(raw_cal)
    hi, lo = pd.to_datetime(end, format="%Y%m%d"), pd.to_datetime(start, format="%Y%m%d")
    months = month_end_dates(dates)
    covered_months = raw_cal["date"].max().to_period("M")
    # 最后覆盖月只有日历延伸至自然月末才可确认完整，防止把下载中断当月末。
    full_last_month = raw_cal["date"].max().is_month_end
    rebalance = [d for d in months if lo <= d <= hi and (d.to_period("M") < covered_months or full_last_month)]
    LOG.info("完整月末调仓日 %d 个；raw 日历 %s ~ %s", len(rebalance), raw_cal["date"].min().date(), raw_cal["date"].max().date())
    stock_basic = read_single("stock_basic", "data.parquet")
    panel = snapshot_at_rebalance(load_index_members(start, end), rebalance)
    panel = attach_names(panel, stock_basic, read_optional("namechange", "data.parquet"))
    panel = attach_listed_days(panel, stock_basic, _age_calendar(raw_cal))
    panel = attach_price(panel, start, end)
    panel = attach_limits(panel, start, end)
    # 只有有效行情且明确非停牌才具备成交基础；未知状态在决策处保守转为 False。
    quote = panel["close"].notna() & panel["close"].gt(0) & panel["suspended"].eq(False).fillna(False)
    panel["tradable"] = (panel["name_known"] & panel["is_st"].eq(False).fillna(False) & panel["listed_days_lower_bound"].ge(MIN_LISTED_DAYS).fillna(False) & quote).astype(bool)
    # ST/年龄属于新建仓资格，不能禁止已有持仓卖出；缺限价不假定可成交。
    up_known = np.isfinite(panel["up_limit"]) & panel["up_limit"].gt(0)
    down_known = np.isfinite(panel["down_limit"]) & panel["down_limit"].gt(0)
    panel["can_buy"] = panel["tradable"] & up_known & panel["close"].lt(panel["up_limit"])
    panel["can_sell"] = quote & down_known & panel["close"].gt(panel["down_limit"])
    cols = ["date", "code", "weight", "name", "is_st", "listed_days", "suspended", "close", "up_limit", "down_limit", "can_buy", "can_sell", "tradable", "snapshot_date", "name_known", "listed_days_exact", "listed_days_lower_bound"]
    out = panel[cols].sort_values(["date", "code"]).reset_index(drop=True)
    if out.duplicated(["date", "code"]).any():
        raise RuntimeError("股票池 (date, code) 重复")
    counts = out.groupby("date").agg(stocks=("code", "size"), eligible=("tradable", "sum"))
    LOG.info("股票池 %d 行 / %d 只；每期 %d ~ %d 只，可新建仓均值 %.1f 只；%s ~ %s", len(out), out["code"].nunique(), counts.stocks.min(), counts.stocks.max(), counts.eligible.mean(), out.date.min().date(), out.date.max().date())
    LOG.info("ST %d 条，未知名称 %d 条，停牌未知 %d 条，涨停价缺失 %d 条，跌停价缺失 %d 条", int(out.is_st.fillna(False).sum()), int((~out.name_known).sum()), int(out.suspended.isna().sum()), int(out.up_limit.isna().sum()), int(out.down_limit.isna().sum()))
    return out


def main() -> None:
    """写入月末股票池及其实际日期清单，供后续 PIT 面板复用主键。"""
    parser = argparse.ArgumentParser(description="构建中证500 月度信号股票池")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()
    start, end = resolve_range(args.start, args.end, data=True)
    LOG.info("区间：%s ~ %s", start, end)
    universe = build_universe(start, end)
    path = save_interim(universe, "universe.parquet")
    save_interim(pd.DataFrame({"date": sorted(universe["date"].unique())}), "rebalance_dates.parquet")
    LOG.info("universe 已写入 %s：%d 行；成交约束仍须在实际执行日重新检查", path, len(universe))


if __name__ == "__main__":
    main()
