"""复权与行情面板：raw -> data/interim/prices.parquet。

只做四件事：
1. daily + adj_factor 合成后复权价，算日收益；
2. 合并 daily_basic（市值、换手、PE/PB）；
3. 生成 vwap（成交额/成交量）；
4. 做一致性检查，把问题打在日志里。

prices 的主键为 (date, code)，ret 为小数收益率（0.01 表示 1%）。
行情缺失时不补造交易行；有行情但复权因子无效时保留行、复权价留空。
显式质量规则只在加工层校正已核对的基准边界，输出adj_factor为有效因子；
未解决公司行动隔离事件日，跨期消费者另须检查price_window_validity。
CLI 另外生成中证500 价格指数的 index.parquet，尚不含分红再投资。

用法：
    python src/mfl/data/adjust.py --start 20150101 --end 20241231
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from common import (
    INTERIM,
    RAW,
    read_single,
    read_years,
    resolve_range,
    save_interim,
    setup_logging,
)
from price_quality import apply_factor_rules, load_validated_rules

LOG = setup_logging("adjust")

# daily_basic 原始字段及单位不变：股本为万股、市值为万元，换手率为百分数。
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
    """去除完全重复行；同一主键仍有多行时拒绝任意挑选版本。"""
    # 完全相同的重复可以合并，数值冲突不能任意选第一行。
    df = df.drop_duplicates()
    n = int(df.duplicated(keys).sum())
    if n:
        raise ValueError(f"原始数据有 {n} 条主键冲突：{keys}")
    return df


# 固定输出列；缺少可选 daily_basic 时仍保留原有契约。
PRICE_COLS = [
    "date", "code", "open", "high", "low", "close", "pre_close", "volume",
    "amount", "vwap", "adj_factor", "close_adj", "ret", *BASIC_COLS[2:],
]


def _previous_observations(start: str, codes: pd.Series) -> pd.DataFrame:
    """仅从已有本地文件找每只股票在 start 前最后一条行情，绝不抓取数据。

    不能用最后一个非空复权价代替最后一条行情，否则会跨过数据缺口。
    """
    columns = ["ts_code", "trade_date", "close", "adj_factor"]
    needed = set(codes)
    frames = []
    files = sorted((RAW / "daily").glob("year=*/data.parquet"), reverse=True)
    # 从近到远按年扫描；每只股票只保留最近一条历史行情，找到后移出待查集合。
    for path in files:
        year = path.parent.name.removeprefix("year=")
        if year > start[:4] or not needed:
            continue
        if year == start[:4] and start.endswith("0101"):
            continue
        daily = pd.read_parquet(path, columns=["ts_code", "trade_date", "close"])
        daily = daily[(daily["trade_date"] < start) & daily["ts_code"].isin(needed)]
        if daily.empty:
            continue
        daily = _dedupe(daily, ["ts_code", "trade_date"])
        daily = daily.sort_values("trade_date").drop_duplicates("ts_code", keep="last")
        adj_path = RAW / "adj_factor" / path.parent.name / "data.parquet"
        if adj_path.exists():
            adj = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
            adj = _dedupe(adj, ["ts_code", "trade_date"])
            daily = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        else:
            daily["adj_factor"] = np.nan
        frames.append(daily[columns])
        needed.difference_update(daily["ts_code"])
    if needed:
        LOG.info("%d 只股票没有区间前本地行情，首条收益保持空值", len(needed))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _pearson(left: pd.Series, right: pd.Series) -> float:
    """逐元素求和计算 Pearson，避开部分本地 BLAS 环境的 dot 阻塞。"""
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else float("nan")


def build_prices(start: str, end: str) -> pd.DataFrame:
    """构造闭区间内的后复权行情面板，只返回结果、不写盘。

    Parameters
    ----------
    start, end : str
        YYYYMMDD 格式；daily 和 adj_factor 分区必需，daily_basic 可缺省。

    Returns
    -------
    pd.DataFrame
        按 (date, code) 排序、列顺序固定为 PRICE_COLS 的行情表。
        ret 使用每只股票相邻两条实际行情的复权价计算；停牌无行情的间隔
        可形成复牌跨期收益，复权因子缺口则使缺口当条和下一条收益均为空。
        已登记未解决事件同样留空；跨窗口收益须检查价格质量事件清单。
    """
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

    # 行情是主表；左连接保留复权因子缺失的记录，一对一校验防止意外扩行。
    df = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    del daily, adj
    # 仅对显式登记且raw锚点仍一致的事件做本地连续性校正；不写回raw。
    rules = load_validated_rules(RAW, df["ts_code"].drop_duplicates())
    original = df["adj_factor"]
    df = apply_factor_rules(df, rules)
    normalized = original.notna() & df["adj_factor"].notna() & ~np.isclose(
        original, df["adj_factor"], rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    isolated = original.notna() & df["adj_factor"].isna()
    LOG.info("本地质量规则：累计基准校正%d行，未确认事件隔离%d行；校正依据为显式连续性推断", int(normalized.sum()), int(isolated.sum()))
    del original
    bad_adj = df["adj_factor"].isna() | ~np.isfinite(df["adj_factor"]) | (df["adj_factor"] <= 0)
    LOG.info("adj_factor 缺失/无效 %d 行（%.4f%%），行情保留", int(bad_adj.sum()), bad_adj.mean() * 100)
    if bad_adj.any():
        LOG.warning("复权因子缺口两侧不能计算相邻收益，不填充、不跨缺口计算")
    df.loc[bad_adj, "adj_factor"] = np.nan

    # 后复权价：收益用这个，市值/换手/涨跌停仍然用不复权价。
    df["close_adj"] = df["close"] * df["adj_factor"]
    history = _previous_observations(start, df["ts_code"].drop_duplicates())
    history["adj_factor"] = pd.to_numeric(history["adj_factor"], errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    # 区间前预热也须使用同一规则，避免局部加工或跨年时引入第二次断点。
    history = apply_factor_rules(history, rules)
    history["close_adj"] = history["close"] * history["adj_factor"].where(
        np.isfinite(history["adj_factor"]) & history["adj_factor"].gt(0)
    )
    # 区间前一条行情仅用于收益预热，最终以 df 为左表合回，不会写入输出。
    returns = pd.concat(
        [history[["ts_code", "trade_date", "close_adj"]], df[["ts_code", "trade_date", "close_adj"]]],
        ignore_index=True,
    ).sort_values(["ts_code", "trade_date"])
    # 禁用前向填充，防止将缺失复权价当成零收益，并跨过缺口计算后续收益。
    returns["ret"] = returns.groupby("ts_code", observed=True)["close_adj"].pct_change(fill_method=None)
    df = df.merge(returns[["ts_code", "trade_date", "ret"]], on=["ts_code", "trade_date"], validate="one_to_one")
    del returns, history

    use_cols = [c for c in BASIC_COLS if c in basic.columns]
    if use_cols and not basic.empty:
        df = df.merge(
            basic[use_cols],
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
    else:
        LOG.warning("跳过 daily_basic 合并：市值/换手/估值字段保留并填空")
    del basic
    for c in BASIC_COLS[2:]:
        if c not in df.columns:
            df[c] = np.nan
    LOG.info("daily_basic 市值覆盖率 %.2f%%，换手率覆盖率 %.2f%%", df["total_mv"].notna().mean() * 100, df["turnover_rate"].notna().mean() * 100)

    df = df.rename(columns={"ts_code": "code", "trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Tushare 单位：vol=手（100 股），amount=千元。vwap(元/股) = amount*1000 / (vol*100)
    df["vwap"] = np.where(df["volume"] > 0, df["amount"] * 10.0 / df["volume"], np.nan)

    # --- 一致性检查 ---
    mask = df["ret"].notna() & df["pct_chg"].notna()
    if mask.any():
        # pct_chg 是百分数而 ret 是小数；相关性与尾部差异一起用于诊断，不改数据。
        error = (df.loc[mask, "ret"] - df.loc[mask, "pct_chg"] / 100).abs()
        LOG.info("复权收益差异：中位数 %.6g，P99 %.6g，最大 %.6g；超过 10bp 共 %d 条", error.median(), error.quantile(.99), error.max(), int((error > .001).sum()))
        n = int(mask.sum())
        # 抽样上限控制诊断开销，固定随机种子使日志可复现；不影响最终行情表。
        sample = df.loc[mask, ["ret", "pct_chg"]].sample(min(200_000, n), random_state=0)
        corr = _pearson(sample["ret"], sample["pct_chg"] / 100.0)
        LOG.info("复权收益与 pct_chg 相关系数：%.5f（抽样 %d 行）", corr, len(sample))
        if corr < 0.99:
            LOG.warning("相关系数偏低，检查 adj_factor 是否和行情对齐")

    extreme = int((df["ret"].abs() > 0.5).sum())
    LOG.info("|ret| > 50%% 的记录：%d 条（除权、重组复牌附近可能出现）", extreme)

    out = df.reindex(columns=PRICE_COLS)
    LOG.info("prices 日期 %s ~ %s，收益空值 %d 行", start, end, int(out["ret"].isna().sum()))
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def write_prices(start: str, end: str) -> Path:
    """CLI 按年加工，写单一 parquet，跨年收益仍从本地前序行情预热。

    每年数据已按 (date, code) 排序，按年顺序追加保持全局排序。
    临时文件完整关闭后才替换最终产物，异常时不留下半成品 prices。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # 显式固定 Arrow 类型，避免某年全空列被推断成 null 而无法继续追加。
    schema = pa.schema([
        pa.field(c, pa.timestamp("ns") if c == "date" else pa.string() if c == "code" else pa.float64())
        for c in PRICE_COLS
    ])
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "prices.parquet"
    # 临时文件与目标文件同目录，确保替换不跨盘；Windows 下先关闭文件句柄再写。
    with tempfile.NamedTemporaryFile(prefix="prices-", suffix=".parquet.tmp", dir=INTERIM, delete=False) as handle:
        temporary = Path(handle.name)
    rows = 0
    codes: set[str] = set()
    try:
        with pq.ParquetWriter(temporary, schema) as writer:
            for year in range(int(start[:4]), int(end[:4]) + 1):
                lo, hi = max(start, f"{year}0101"), min(end, f"{year}1231")
                prices = build_prices(lo, hi)
                rows += len(prices)
                codes.update(prices["code"])
                table = pa.Table.from_pandas(prices, schema=schema, preserve_index=False)
                writer.write_table(table)
                LOG.info("已加工并写入 %s 年：%d 行（累计 %d 行）", year, len(prices), rows)
                del prices, table
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("prices 已写入 %s：%d 行，%d 只股票", path, rows, len(codes))
    return path


def build_index(start: str, end: str) -> pd.DataFrame:
    """计算并写入中证500 价格指数的区间收益，同时返回 index 面板。

    与 build_prices 不同，此函数会写 interim/index.parquet。ret 为小数；
    当前源数据是价格指数，后续与股票收益比较时须另行处理分红口径。
    """
    idx = read_single("index_daily", "ts_code=000905.SH", "data.parquet")
    idx = _dedupe(idx, ["trade_date"])
    # 先在本地完整历史上算收益，再截取区间，保留起点前一日信息。
    idx = idx[idx["trade_date"] <= end].copy()
    idx["date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ret"] = idx["close"].pct_change(fill_method=None)
    idx = idx[idx["trade_date"] >= start]
    out = idx[["date", "close", "ret"]].rename(columns={"close": "index_close"})
    if not out.empty and pd.isna(out.iloc[0]["ret"]):
        LOG.info("指数没有区间前本地行情，起点收益为空")
    save_interim(out, "index.parquet")
    LOG.info("index：%d 行", len(out))
    return out


def main() -> None:
    """解析区间，写入行情面板；本地指数文件缺失时告警并跳过基准。"""
    parser = argparse.ArgumentParser(description="复权与行情面板")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()

    start, end = resolve_range(args.start, args.end, data=True)
    LOG.info("区间：%s ~ %s", start, end)

    write_prices(start, end)

    try:
        build_index(start, end)
    except FileNotFoundError as exc:
        LOG.warning("跳过指数基准：%s", exc)


if __name__ == "__main__":
    main()
