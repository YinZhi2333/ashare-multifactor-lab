#!/usr/bin/env python
"""检查 data/raw 是否齐全、覆盖是否正常。

在跑 adjust / universe / pit 之前先跑这个，避免拿半套数据做研究。

用法：
    python scripts/check_raw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

EXPECTED_YEARS = list(range(2015, 2025))
FAILS: list[str] = []
WARNS: list[str] = []


def ok(label: str, detail: str) -> None:
    print(f"[OK]   {label:<18} {detail}")


def warn(label: str, detail: str) -> None:
    print(f"[WARN] {label:<18} {detail}")
    WARNS.append(label)


def fail(label: str, detail: str) -> None:
    print(f"[FAIL] {label:<18} {detail}")
    FAILS.append(label)


def year_files(table: str) -> list[Path]:
    return sorted((RAW / table).glob("year=*/data.parquet"))


def scan_year_table(table: str, columns: list[str]) -> dict | None:
    files = year_files(table)
    if not files:
        return None

    rows = 0
    dup = 0
    codes: set[str] = set()
    dates: set[str] = set()
    years: list[str] = []
    missing: dict[str, int] = {c: 0 for c in columns if c not in ("ts_code", "trade_date")}

    for f in files:
        years.append(f.parent.name.replace("year=", ""))
        df = pd.read_parquet(f, columns=columns)
        rows += len(df)
        codes.update(df["ts_code"].astype(str).unique())
        dates.update(df["trade_date"].astype(str).unique())
        dup += int(df.duplicated(["ts_code", "trade_date"]).sum())
        for c in missing:
            missing[c] += int(df[c].isna().sum())

    return {
        "rows": rows,
        "dup": dup,
        "codes": len(codes),
        "dates": dates,
        "years": years,
        "missing": missing,
    }


def fmt(n: int) -> str:
    return f"{n:,}"


def main() -> int:
    print("=" * 78)
    print(f"检查目录：{RAW}")
    print("=" * 78)

    # ---------------- 日历 ----------------
    cal_path = RAW / "trade_cal" / "data.parquet"
    cal = None
    open_dates: set[str] = set()
    if not cal_path.exists():
        fail("trade_cal", "缺少 trade_cal/data.parquet")
    else:
        cal = pd.read_parquet(cal_path)
        is_open = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
        open_dates = set(cal.loc[is_open == 1, "cal_date"].astype(str))
        ok(
            "trade_cal",
            f"{fmt(len(cal))} 行，{cal['cal_date'].min()} ~ {cal['cal_date'].max()}，"
            f"交易日 {fmt(len(open_dates))}",
        )

    # ---------------- 股票列表 ----------------
    basic_path = RAW / "stock_basic" / "data.parquet"
    if not basic_path.exists():
        fail("stock_basic", "缺少 stock_basic/data.parquet")
    else:
        sb = pd.read_parquet(basic_path)
        counts = sb["list_status"].value_counts().to_dict() if "list_status" in sb else {}
        ok("stock_basic", f"{fmt(len(sb))} 只（{counts}）")
        if counts.get("D", 0) == 0:
            warn("stock_basic", "没有 D（退市）记录，可能有幸存者偏差")

    # ---------------- 行情 ----------------
    daily = scan_year_table("daily", ["ts_code", "trade_date", "close"])
    if daily is None:
        fail("daily", "没有 year=*/data.parquet")
    else:
        missing_years = sorted(set(EXPECTED_YEARS) - {int(y) for y in daily["years"]})
        if missing_years:
            warn("daily", f"缺年份 {missing_years}")
        miss_dates = sorted(open_dates - daily["dates"]) if open_dates else []
        if miss_dates:
            warn("daily", f"缺 {len(miss_dates)} 个交易日，例如 {miss_dates[:3]}")
        ok(
            "daily",
            f"{fmt(daily['rows'])} 行，{min(daily['dates'])} ~ {max(daily['dates'])}，"
            f"{fmt(daily['codes'])} 只，重复 {fmt(daily['dup'])}，"
            f"close 缺失 {fmt(daily['missing']['close'])}",
        )

    adj = scan_year_table("adj_factor", ["ts_code", "trade_date", "adj_factor"])
    if adj is None:
        fail("adj_factor", "没有 year=*/data.parquet")
    else:
        ok(
            "adj_factor",
            f"{fmt(adj['rows'])} 行，{fmt(adj['codes'])} 只，"
            f"缺失 {fmt(adj['missing']['adj_factor'])}",
        )
        if daily and adj["rows"] < daily["rows"] * 0.99:
            warn("adj_factor", "行数少于 daily，可能有交易日缺复权因子")
        elif daily and adj["rows"] > daily["rows"] * 1.01:
            print(
                f"[INFO] adj_factor     行数比 daily 多 {adj['rows'] - daily['rows']:,}"
                "（停牌股也有复权因子，正常）"
            )

    basic = scan_year_table(
        "daily_basic", ["ts_code", "trade_date", "total_mv", "turnover_rate"]
    )
    if basic is None:
        fail("daily_basic", "没有 year=*/data.parquet（EP/换手/市值都会缺）")
    else:
        ok(
            "daily_basic",
            f"{fmt(basic['rows'])} 行，{fmt(basic['codes'])} 只，"
            f"total_mv 缺失 {fmt(basic['missing']['total_mv'])}，"
            f"turnover 缺失 {fmt(basic['missing']['turnover_rate'])}",
        )

    # ---------------- 指数 ----------------
    idx_path = RAW / "index_daily" / "ts_code=000905.SH" / "data.parquet"
    if not idx_path.exists():
        fail("index_daily", "缺少 000905.SH 的日线")
    else:
        idx = pd.read_parquet(idx_path)
        ok(
            "index_daily",
            f"{fmt(len(idx))} 行，{idx['trade_date'].min()} ~ {idx['trade_date'].max()}",
        )

    iw_path = RAW / "index_weight" / "index_code=000905.SH" / "data.parquet"
    iw = None
    if not iw_path.exists():
        fail("index_weight", "缺少 000905.SH 的成分股（只能拿当前成分股回测）")
    else:
        iw = pd.read_parquet(iw_path)
        per_date = iw.groupby("trade_date").size()
        ok(
            "index_weight",
            f"{fmt(len(iw))} 行，{len(per_date)} 个快照，"
            f"每期 {per_date.min()} ~ {per_date.max()} 只",
        )
        if per_date.min() < 400:
            warn("index_weight", "有快照不足 400 只，检查是否截断")

    # ---------------- 财务 ----------------
    def check_statement(table: str, required: bool) -> set[str] | None:
        path = RAW / table / "data.parquet"
        if not path.exists():
            (fail if required else warn)(table, "缺少 data.parquet")
            return None
        df = pd.read_parquet(path, columns=["ts_code", "end_date", "ann_date"])
        codes = set(df["ts_code"].astype(str).unique())
        bad = int((df["ann_date"].astype(str) < df["end_date"].astype(str)).sum())
        ok(
            table,
            f"{fmt(len(df))} 行，{fmt(len(codes))} 只，"
            f"{df['end_date'].min()} ~ {df['end_date'].max()}",
        )
        if bad:
            warn(table, f"{bad} 条 ann_date < end_date")
        return codes

    income_codes = check_statement("income", required=True)
    bs_codes = check_statement("balancesheet", required=True)
    check_statement("cashflow", required=False)
    check_statement("fina_indicator", required=False)

    # ---------------- 财务覆盖率 ----------------
    if iw is not None and income_codes:
        expected = set(iw["con_code"].astype(str).unique())
        for name, codes in (("income", income_codes), ("balancesheet", bs_codes)):
            if not codes:
                continue
            cov = len(expected & codes) / max(len(expected), 1)
            detail = f"{len(expected & codes)}/{len(expected)} = {cov:.1%}（相对中证500历史成分股）"
            if cov < 0.95:
                warn(name, f"覆盖率 {detail}")
            else:
                ok(f"{name} 覆盖率", detail)

    # ---------------- 可选表 ----------------
    if not (RAW / "namechange" / "data.parquet").exists():
        warn("namechange", "缺失：universe 的 ST 判断会退化为当前名称")
    if not year_files("stk_limit"):
        warn("stk_limit", "缺失：universe 的 can_buy/can_sell 会全部为 True")
    if not year_files("suspend_d"):
        warn("suspend_d", "缺失：停牌改用 prices 是否缺失判断")

    # ---------------- 总结 ----------------
    print("-" * 78)
    if FAILS:
        print(f"结论：关键表缺失或异常 {len(FAILS)} 个：{', '.join(FAILS)}")
        print("先补齐这些表，再进入加工阶段。")
        return 1

    print("结论：关键表齐全，可以进入加工阶段。")
    if WARNS:
        print(f"注意：{len(WARNS)} 条 WARN，按需处理（多数是可接受的降级）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
