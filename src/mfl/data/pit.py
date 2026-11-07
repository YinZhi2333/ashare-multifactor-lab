"""财务事件与月度 PIT 面板；只读 raw，按当时已知版本重算。

可用时间为 max(ann_date, f_ann_date) + 1 个自然日；缺实际公告日时回退
ann_date。每次利润表、资产负债表或可选指标公告都触发当前报告期重算。
只保证 raw 已保留版本的知识时间；无法恢复数据商未保存的原始版本。

PIT（point-in-time）要求每个历史信号只能使用当时已经公开的数据：
end_date 是财报所属期间，ann_date 是本次快照的保守知识日，ann_date_eff
是可用于信号的首日。事件表主键为 (code, ann_date_eff)，每行是当前报告期
在该时点的完整快照；月度面板再把快照向后匹配到股票池的 (date, code)。
用法：python src/mfl/data/pit.py --start 20150101 --end 20241231
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import INTERIM, read_optional, resolve_range, save_interim, setup_logging

LOG = setup_logging("pit")
# 数据商指标保留原单位；本模块计算的 roe_ttm 是小数比例，不能直接混用。
FI_COLS = [
    "roe", "roe_waa", "roe_dt", "roe_yearly", "roa", "netprofit_yoy",
    "grossprofit_margin", "netprofit_margin", "debt_to_assets", "eps", "bps",
    "profit_dedt",
]
# 可选指标只补充事件，不作为利润 TTM 可用的前置条件。
OPTIONAL_COLS = ["roe_yearly", "grossprofit_margin", "debt_to_assets", "eps", "bps"]
EVENT_COLS = [
    "code", "end_date", "ann_date", "ann_date_eff", "source", "roe_ttm",
    "roe_yearly", "net_profit_ttm", "avg_equity", "grossprofit_margin",
    "debt_to_assets", "eps", "bps", "input_max_ann_date",
]
PANEL_COLS = ["date"] + EVENT_COLS


def _to_date(s: pd.Series) -> pd.Series:
    """将时间戳或 YYYYMMDD 列统一为纳秒精度；非法字符串日期变为 NaT。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).astype("datetime64[ns]")
    return pd.to_datetime(s.astype("string"), format="%Y%m%d", errors="coerce").astype("datetime64[ns]")


def _normalize_versions(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    """标准化 raw 财报，按 (code, end_date, ann_date) 保留知识版本。

    cols 为所需数值列，缺列补空；非法日期、非季末报告及提前公告记录排除。
    同键有不同值时只把冲突字段置空，仍保留该版本，防止 as-of 匹配错误地
    沿用早期旧值。这里无法恢复在 fetch 落盘前就已被覆盖的财报版本。
    """
    df = df.rename(columns={"ts_code": "code"}).copy()
    df["end_date"] = _to_date(df["end_date"])
    nominal = _to_date(df["ann_date"])
    actual = _to_date(df["f_ann_date"]) if "f_ann_date" in df else nominal
    # 实际公告晚于名义公告时取较晚日期；缺一个时用另一个，避免提前知晓。
    df["ann_date"] = pd.concat([nominal, actual], axis=1).max(axis=1)
    delayed = int((actual > nominal).sum())
    valid = (
        df["code"].notna() & df["end_date"].notna() & df["ann_date"].notna()
        & df["end_date"].dt.is_quarter_end & (df["ann_date"] >= df["end_date"])
    )
    if (~valid).any():
        LOG.warning("%s 排除无效日期/非季末/公告早于报告期记录 %d 条", name, int((~valid).sum()))
    df = df.loc[valid].copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    keys = ["code", "end_date", "ann_date"]
    # 将“空值 vs 已知值”也视为冲突；把分组计数映射回原行，以便逐字段阻断。
    counts = df.groupby(keys, observed=True)[cols].nunique(dropna=False)
    conflicts = counts.reindex(pd.MultiIndex.from_frame(df[keys])).set_axis(df.index) > 1
    if conflicts.any().any():
        groups = df.loc[conflicts.any(axis=1), keys].drop_duplicates()
        LOG.warning("%s 同一可用日有 %d 组冲突版本：冲突字段置空并保留阻断事件", name, len(groups))
        df[cols] = df[cols].mask(conflicts)
    df = df[keys + cols].drop_duplicates(keys).sort_values(["ann_date", "code", "end_date"])
    LOG.info("%s 版本 %d 条；实际公告日推迟 %d 条；不按报告期覆盖旧版本", name, len(df), delayed)
    return df.reset_index(drop=True)


def _load_fina() -> pd.DataFrame | None:
    """读取并标准化可选 fina_indicator；文件缺失时返回 None。"""
    df = read_optional("fina_indicator", "data.parquet")
    return None if df is None else _normalize_versions(df, FI_COLS, "fina_indicator")


def _load_statements(name: str, cols: list[str]) -> pd.DataFrame | None:
    """读取指定财务表，若有 report_type 则仅保留类型 1，再标准化版本。"""
    df = read_optional(name, "data.parquet")
    if df is None:
        return None
    if "report_type" in df:
        df = df.loc[pd.to_numeric(df["report_type"], errors="coerce").eq(1)].copy()
    return _normalize_versions(df, cols, name)


def _asof_values(
    grid: pd.DataFrame, source: pd.DataFrame, target: str, cols: list[str], prefix: str,
) -> pd.DataFrame:
    """为事件表查询同一股票、指定报告期在该知识日已公开的最新版本。

    grid 的 ann_date 是查询截止时间，target 指向要查的报告期列；source
    需含 code/end_date/ann_date 和 cols。返回原 grid，并增加 prefix_字段
    以及 prefix_ann（命中版本的公告日），后者用于追踪依赖是否来自未来。
    未命中或命中冲突版本时保持空值，不越过最近版本寻找更早非空值。
    """
    out = grid.copy()
    if source.empty:
        for col in cols:
            out[f"{prefix}_{col}"] = np.nan
        out[f"{prefix}_ann"] = pd.NaT
        return out
    left = grid[["code", "ann_date", target]].rename(columns={target: "_ref_end"}).copy()
    # as-of 需要按公告时间重排；保留行号，匹配后恢复 grid 原顺序再按位置赋值。
    left["_row"] = np.arange(len(left))
    right = source[["code", "end_date", "ann_date"] + cols].rename(columns={
        "end_date": "_ref_end", "ann_date": f"{prefix}_ann",
        **{col: f"{prefix}_{col}" for col in cols},
    })
    # pandas 3 默认推断到微秒；parquet/旧环境可能是纳秒，as-of 键必须一致。
    for frame, keys in [(left, ["_ref_end", "ann_date"]), (right, ["_ref_end", f"{prefix}_ann"])]:
        for key in keys:
            frame[key] = _to_date(frame[key])
    matched = pd.merge_asof(
        left.sort_values("ann_date"), right.sort_values(f"{prefix}_ann"),
        by=["code", "_ref_end"], left_on="ann_date", right_on=f"{prefix}_ann",
        direction="backward",
    ).sort_values("_row")
    for col in cols:
        out[f"{prefix}_{col}"] = matched[f"{prefix}_{col}"].to_numpy()
    out[f"{prefix}_ann"] = matched[f"{prefix}_ann"].to_numpy()
    return out


def _event_grid(primary: pd.DataFrame, sources: list[pd.DataFrame]) -> pd.DataFrame:
    """合并各来源的公告时点，生成每只股票当前报告期的重算网格。

    primary 决定已知的最新报告期，sources 决定哪些公告会触发重算。
    例如旧年报更正可能改变当前季度 TTM，但不能把当前报告期退回旧年报。
    首次主表公告之前的其他来源事件因尚无当前报告期而被丢弃。
    """
    latest = primary[["code", "end_date", "ann_date"]].sort_values(["code", "ann_date", "end_date"])
    # 按知识时间累计取最大的报告期，修订旧报告也继续计算最新报告期。
    latest["end_date"] = latest.groupby("code", observed=True)["end_date"].cummax()
    latest = latest.drop_duplicates(["code", "ann_date"], keep="last")
    timeline = pd.concat([s[["code", "ann_date"]] for s in sources if not s.empty], ignore_index=True)
    timeline = timeline.drop_duplicates().sort_values("ann_date")
    grid = pd.merge_asof(
        timeline, latest.sort_values("ann_date"), by="code", on="ann_date", direction="backward",
    )
    return grid.dropna(subset=["end_date"]).reset_index(drop=True)


def _attach_fina(df: pd.DataFrame, fi: pd.DataFrame) -> pd.DataFrame:
    """可选字段只取同报告期当时已公开的值，不推迟此前主财务事件。"""
    out = _asof_values(df, fi, "end_date", OPTIONAL_COLS, "fi")
    for col in OPTIONAL_COLS:
        out[col] = out.pop(f"fi_{col}")
    out["input_max_ann_date"] = out[["input_max_ann_date", "fi_ann"]].max(axis=1)
    return out.drop(columns="fi_ann")


def _build_ttm(inc: pd.DataFrame, bs: pd.DataFrame, fi: pd.DataFrame | None = None) -> pd.DataFrame:
    """按每个公告时点重算归母净利润 TTM、平均净资产及 ROE_TTM。

    inc/bs/fi 均须先标准化版本。利润是年初至报告期末的累计值，因此前三季
    TTM = 本期累计 + 上年全年 - 上年同期累计，Q4 直接等于本年全年利润。
    例如次年 Q1 累计为 12、上年全年为 40、上年 Q1 为 10，则 TTM=42。
    平均净资产取本期末和上年同期末的简单平均；分母非正时 ROE 留空。
    返回含计算中间列的事件表，最终公共字段由 build_events 统一筛选。
    """
    sources = [inc, bs] + ([] if fi is None else [fi])
    df = _event_grid(inc, sources)
    df["quarter"] = df["end_date"].dt.quarter
    df["prev_year_end"] = df["end_date"] - pd.DateOffset(years=1)
    df["prev_fy_end"] = pd.to_datetime((df["end_date"].dt.year - 1).astype(str) + "1231", format="%Y%m%d")
    df["prev_quarter_end"] = df["end_date"] - pd.offsets.QuarterEnd()
    # 每个历史依赖都按“本次事件的公告日”独立查版本，不能直接取事后最新版。
    for target, prefix in [("end_date", "cur"), ("prev_fy_end", "annual"),
                           ("prev_year_end", "same"), ("prev_quarter_end", "prevq")]:
        df = _asof_values(df, inc, target, ["n_income_attr_p"], prefix)
    for target, prefix in [("end_date", "eq"), ("prev_year_end", "eqprev")]:
        df = _asof_values(df, bs, target, ["total_hldr_eqy_exc_min_int"], prefix)

    # 单季利润为辅助中间量，Q1 已是年初累计；q_profit 不属于最终输出契约。
    df["q_profit"] = df["cur_n_income_attr_p"] - df["prevq_n_income_attr_p"]
    q1 = df["quarter"].eq(1)
    df.loc[q1, "q_profit"] = df.loc[q1, "cur_n_income_attr_p"]
    df["net_profit_ttm"] = df["cur_n_income_attr_p"] + df["annual_n_income_attr_p"] - df["same_n_income_attr_p"]
    q4 = df["quarter"].eq(4)
    df.loc[q4, "net_profit_ttm"] = df.loc[q4, "cur_n_income_attr_p"]
    df["avg_equity"] = (df["eq_total_hldr_eqy_exc_min_int"] + df["eqprev_total_hldr_eqy_exc_min_int"]) / 2
    df["roe_ttm"] = (df["net_profit_ttm"] / df["avg_equity"]).where(df["avg_equity"] > 0)
    # 年报 TTM 不用上年利润，Q1 单季不用前季；仅审计实际参与的依赖。
    df.loc[q4, ["annual_ann", "same_ann"]] = pd.NaT
    df.loc[q1, "prevq_ann"] = pd.NaT
    # 记录利润、单季辅助量、净资产及可选指标的最晚依赖日，供面板做泄漏断言。
    times = ["cur_ann", "annual_ann", "same_ann", "prevq_ann", "eq_ann", "eqprev_ann"]
    df["input_max_ann_date"] = df[times].max(axis=1)
    if fi is None:
        for col in OPTIONAL_COLS:
            df[col] = np.nan
    else:
        df = _attach_fina(df, fi)
    # 推迟一个自然日；若落在休市日，面板向后匹配会在后续信号日才使用它。
    df["ann_date_eff"] = df["ann_date"] + pd.Timedelta(days=1)
    df["source"] = "ttm"
    if (df["input_max_ann_date"] > df["ann_date"]).any():
        raise RuntimeError("TTM 引用了未来公告版本")
    return df


def build_events(start: str, end: str) -> pd.DataFrame:
    """构造截至 end 已生效的财务快照，保留 start 之前的历史种子。

    start/end 为 YYYYMMDD；返回 EVENT_COLS，按 (ann_date_eff, code) 排序，
    不写盘。start 用于检查区间先后，不作为事件下界，以保证起点信号有历史
    可匹配、早期 TTM 有历史依赖。缺利润表时降级为 fina_only，缺资产负债表
    时仅计算利润 TTM；两者都不会拿数据商年化 ROE 替代 roe_ttm。
    """
    hi = pd.to_datetime(end, format="%Y%m%d")
    if pd.to_datetime(start, format="%Y%m%d") > hi:
        raise ValueError("start 不能晚于 end")
    fi = _load_fina()
    inc = _load_statements("income", ["n_income_attr_p", "revenue"])
    bs = _load_statements("balancesheet", ["total_hldr_eqy_exc_min_int", "total_assets"])
    # 保留 start 前种子；end 后公告和报告不能参与本次加工。
    def available(df: pd.DataFrame | None) -> pd.DataFrame | None:
        """保留截止日内已生效的版本，防止截止日当天公告被提前使用。"""
        if df is None:
            return None
        return df.loc[(df["end_date"] <= hi) & (df["ann_date"] + pd.Timedelta(days=1) <= hi)].copy()
    fi, inc, bs = available(fi), available(inc), available(bs)
    if inc is None or inc.empty:
        if fi is None or fi.empty:
            raise RuntimeError("没有区间内可用的 income 或 fina_indicator，无法构造事件")
        LOG.warning("没有可用 income：只输出 fina_only，roe_ttm/net_profit_ttm 留空")
        events = _event_grid(fi, [fi])
        events["input_max_ann_date"] = pd.NaT
        events = _attach_fina(events, fi)
        events["ann_date_eff"] = events["ann_date"] + pd.Timedelta(days=1)
        events["source"] = "fina_only"
        for col in ["roe_ttm", "net_profit_ttm", "avg_equity"]:
            events[col] = np.nan
    else:
        if bs is None or bs.empty:
            LOG.warning("没有可用 balancesheet：保留利润 TTM，ROE_TTM 留空")
            bs = pd.DataFrame(columns=["code", "end_date", "ann_date", "total_hldr_eqy_exc_min_int"])
        events = _build_ttm(inc, bs, fi)
    events = events[EVENT_COLS].sort_values(["ann_date_eff", "code"]).reset_index(drop=True)
    if events.duplicated(["code", "ann_date_eff"]).any():
        raise RuntimeError("财务事件存在重复 (code, ann_date_eff)")
    LOG.info("财务快照 %d 条，%d 只；ROE_TTM 非空 %.2f%%；依赖时间泄漏 0 条", len(events), events["code"].nunique(), events["roe_ttm"].notna().mean() * 100)
    LOG.info("有效日期 %s ~ %s（含 start 前种子），source=%s", events["ann_date_eff"].min(), events["ann_date_eff"].max(), events["source"].value_counts().to_dict())
    return events


def build_panel(events: pd.DataFrame) -> pd.DataFrame | None:
    """将财务快照向后匹配到本地 universe 的全部 (date, code)，不写盘。

    events 必须符合公告次一自然日生效规则。返回与 universe 行数一致的面板，
    尚无已生效财报的股票保留空值；股票池文件不存在则告警并返回 None。
    此函数不另按 CLI 区间截断，所以须与同一研究区间生成的 universe 配套。
    """
    universe_path = INTERIM / "universe.parquet"
    if not universe_path.exists():
        LOG.warning("没有 universe.parquet，跳过面板；先跑 universe.py")
        return None
    uni = pd.read_parquet(universe_path, columns=["date", "code"])
    if uni.duplicated(["date", "code"]).any():
        raise ValueError("universe 存在重复 (date, code)")
    uni["date"] = pd.to_datetime(uni["date"]).astype("datetime64[ns]")
    left = uni.sort_values("date")
    right = events.reindex(columns=EVENT_COLS).copy()
    for col in ["end_date", "ann_date", "ann_date_eff", "input_max_ann_date"]:
        right[col] = _to_date(right[col])
    right = right.sort_values(["code", "ann_date_eff", "end_date"])
    # 同时校验快照本身和底层输入的知识时间；只检查 end_date 不足以阻止泄漏。
    required = ["code", "end_date", "ann_date", "ann_date_eff", "input_max_ann_date"]
    bad = (
        right[required].isna().any(axis=1)
        | (right["end_date"] > right["ann_date"])
        | (right["input_max_ann_date"] > right["ann_date"])
        | (right["ann_date_eff"] != right["ann_date"] + pd.Timedelta(days=1))
    )
    if bad.any():
        raise RuntimeError("财务事件的依赖时间/次日生效规则不一致")
    latest = right.groupby("code", observed=True)["end_date"].cummax()
    # 兼容旧报告修订事件；本脚本已在每次事件重算当前报告期。
    right = right.loc[right["end_date"] == latest].sort_values(["ann_date_eff", "end_date"])
    right = right.drop_duplicates(["code", "ann_date_eff"], keep="last")
    if right.empty:
        panel = left.copy()
        for col in EVENT_COLS:
            if col != "code":
                panel[col] = pd.NaT if col.endswith("date") or col == "ann_date_eff" else np.nan
        panel["source"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    else:
        # 按生效时间匹配，而非按报告期匹配；公告日晚于信号日的版本不会被选中。
        panel = pd.merge_asof(left, right, by="code", left_on="date", right_on="ann_date_eff", direction="backward")
    leak = (panel["ann_date_eff"] > panel["date"]) | (panel["input_max_ann_date"] >= panel["date"])
    if leak.any():
        raise RuntimeError(f"PIT 面板出现 {int(leak.sum())} 条未来依赖")
    if len(panel) != len(uni):
        raise RuntimeError("PIT 面板主键数量与 universe 不一致")
    LOG.info("PIT 面板 %d 行；ROE_TTM 覆盖率 %.2f%%；事件及依赖泄漏 0 条", len(panel), panel["roe_ttm"].notna().mean() * 100)
    yearly = panel.groupby(panel["date"].dt.year)["roe_ttm"].apply(lambda s: s.notna().mean())
    LOG.info("分年覆盖率：\n%s", yearly.round(4).to_string())
    return panel[PANEL_COLS].sort_values(["date", "code"]).reset_index(drop=True)


def main() -> None:
    """先保存完整事件快照，再在股票池存在时保存月末 PIT 面板。"""
    parser = argparse.ArgumentParser(description="按已知版本构造财务 PIT 事件与面板")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()
    start, end = resolve_range(args.start, args.end, data=True)
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
