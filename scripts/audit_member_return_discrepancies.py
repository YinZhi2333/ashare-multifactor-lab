"""只读复核历史中证500成员的复权收益差异，不联网或重建加工数据。

从项目根目录运行：python scripts/audit_member_return_discrepancies.py
输出为results下的诊断快照，不能当作已批准的价格修复规则。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mfl" / "data"))
from common import INTERIM, RAW, ROOT, setup_logging
from price_quality import apply_factor_rules, read_rules, validate_anchors

LOG = setup_logging("audit_member_return_discrepancies")
OUT = ROOT / "results" / "adj_factor_audit" / "member_discrepancy_audit"


def _manifest() -> dict[str, str]:
    """对raw和interim逐文件计算内容哈希，用于证明诊断未改数据。"""
    result = {}
    for base in (RAW, INTERIM):
        for path in sorted(base.rglob("*.parquet")):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.relative_to(ROOT).as_posix()] = digest.hexdigest()
    return result


def main() -> None:
    """从raw独立重算差异、核对现有prices并保存19条事件的证据。"""
    OUT.mkdir(parents=True, exist_ok=True)
    before = _manifest()
    rules = read_rules()
    validate_anchors(rules, RAW)
    universe = pd.read_parquet(INTERIM / "universe.parquet", columns=["date", "code"])
    weights = pd.read_parquet(
        RAW / "index_weight" / "index_code=000905.SH" / "data.parquet",
        columns=["con_code"],
    )
    members = sorted(weights["con_code"].dropna().unique())
    snapshots = {date: set(g["code"]) for date, g in universe.groupby("date")}
    dates = sorted(snapshots)
    seed = pd.DataFrame()
    events = []
    longping = []
    first, last = int(universe["date"].dt.year.min()), int(universe["date"].dt.year.max())
    for year in range(first, last + 1):
        daily = pd.read_parquet(
            RAW / "daily" / f"year={year}" / "data.parquet",
            columns=["ts_code", "trade_date", "close", "pre_close", "pct_chg"],
            filters=[("ts_code", "in", members)],
        )
        adj = pd.read_parquet(
            RAW / "adj_factor" / f"year={year}" / "data.parquet",
            filters=[("ts_code", "in", members)],
        )
        daily = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        daily["_current"] = True
        work = pd.concat([seed, daily], ignore_index=True).sort_values(["ts_code", "trade_date"])
        group = work.groupby("ts_code", observed=True)
        for col in ("trade_date", "close", "adj_factor"):
            work[f"prev_{col}"] = group[col].shift()
        current = work[work["_current"].fillna(False)].copy()
        seed = work.drop_duplicates("ts_code", keep="last")[list(daily.columns)].copy()
        seed["_current"] = False
        current["factor_ratio"] = current["adj_factor"] / current["prev_adj_factor"]
        current["reference_ratio"] = current["prev_close"] / current["pre_close"]
        current["ret_recomputed"] = current["close"] * current["factor_ratio"] / current["prev_close"] - 1
        current["daily_reference_ret"] = current["close"] / current["pre_close"] - 1
        current["error"] = current["ret_recomputed"] - current["pct_chg"] / 100
        longping.append(current[current["ts_code"].eq("000998.SZ")])
        bad = current[current["error"].abs().gt(.001)].copy()
        bad["date"] = pd.to_datetime(bad["trade_date"], format="%Y%m%d")
        bad = bad.rename(columns={"ts_code": "code"})
        if bad.empty:
            LOG.info("%d年：成员行情%d条，raw收益差异超过10bp共0条", year, len(current))
            continue
        prices = pd.read_parquet(
            INTERIM / "prices.parquet", columns=["date", "code", "ret", "adj_factor"],
            filters=[("date", ">=", pd.Timestamp(f"{year}-01-01")),
                     ("date", "<=", pd.Timestamp(f"{year}-12-31")),
                     ("code", "in", bad["code"].unique().tolist())],
        )
        bad = bad.merge(prices.rename(columns={"adj_factor": "interim_factor"}),
                        on=["date", "code"], how="left", validate="one_to_one")
        # raw差异继续作为审计对象；当前产物应对应显式规则处理后的值。
        now = apply_factor_rules(bad.rename(columns={"code": "ts_code"}), rules)
        prev = bad[["code", "prev_trade_date", "prev_adj_factor"]].rename(
            columns={"code": "ts_code", "prev_trade_date": "trade_date", "prev_adj_factor": "adj_factor"},
        )
        prev = apply_factor_rules(prev, rules)
        expected = now["close"] * now["adj_factor"] / (bad["prev_close"] * prev["adj_factor"]) - 1
        assert np.isclose(bad["ret"], expected, atol=1e-12, rtol=1e-10, equal_nan=True).all()
        assert np.isclose(now["adj_factor"], bad["interim_factor"], equal_nan=True).all()
        bad["expected_effective_ret"] = expected
        bad["remaining_error"] = bad["ret"] - bad["pct_chg"] / 100
        events.append(bad)
        LOG.info("%d年：成员行情%d条，raw收益差异超过10bp共%d条", year, len(current), len(bad))
    audit = pd.concat(events, ignore_index=True).sort_values(["date", "code"])
    positions = np.searchsorted(dates, audit["date"], side="right") - 1
    audit["snapshot_date"] = [dates[pos] if pos >= 0 else pd.NaT for pos in positions]
    audit["member_at_latest_snapshot"] = [
        code in snapshots.get(date, set()) for code, date in zip(audit["code"], audit["snapshot_date"])
    ]
    # 这些是审计处置分类：数据不一致不等于已证明数据商内部具体成因。
    audit["review_class"] = np.where(
        audit["code"].eq("600733.SH"), "特殊股改事件，现有因子口径待确认",
        np.where(audit["code"].eq("000998.SZ"), "显著因子基准不连续", "小幅因子基准不连续"),
    )
    audit["abs_error_bp"] = audit["error"].abs() * 10000
    audit.to_csv(OUT / "member_events.csv", index=False, encoding="utf-8-sig")
    lp = pd.concat(longping, ignore_index=True)
    changes = lp[lp["adj_factor"].ne(lp["prev_adj_factor"])].copy()
    changes.to_csv(OUT / "000998_factor_changes.csv", index=False, encoding="utf-8-sig")
    windows = []
    for start, end in [("20180823", "20180827"), ("20200102", "20210106"), ("20230601", "20241231")]:
        q = lp[lp["trade_date"].between(start, end)]
        windows.append({"start": start, "end": end, "observations": len(q)})
    after = _manifest()
    assert before == after, "审计前后raw或interim发生变化，不能确认只读状态"
    ordinary = audit[~audit["code"].eq("600733.SH")]
    summary = {
        "historical_member_union": len(members), "events": len(audit), "stocks": audit["code"].nunique(),
        "snapshot_member_events": int(audit["member_at_latest_snapshot"].sum()),
        "class_counts": audit["review_class"].value_counts().to_dict(),
        "all_rule_adjusted_returns_match_interim": True,
        "normalized_events": int(audit["ret"].notna().sum()),
        "unresolved_events_isolated": int(audit["ret"].isna().sum()),
        "remaining_computable_error_gt_10bp": int(audit["remaining_error"].abs().gt(.001).sum()),
        "ordinary_events_reference_close_unchanged": bool(ordinary["pre_close"].eq(ordinary["prev_close"]).all()),
        "daily_pct_vs_close_reference_max_error": float((audit["daily_reference_ret"] - audit["pct_chg"] / 100).abs().max()),
        "000998_piecewise_level_windows_for_review": windows,
        "raw_and_interim_files_unchanged": len(after),
        "company_action_evidence": "参见reports/adj_factor_member_review_20261107.md；分类含推断，不是修复后的数据",
    }
    for name, value in [("summary.json", summary), ("manifest_before.json", before), ("manifest_after.json", after)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("审计完成：%d条/%d只，raw及interim的%d个文件哈希不变", len(audit), audit["code"].nunique(), len(after))
    LOG.info("当前加工结果：连续性校正%d条，隔离未解决%d条；可计算的剩余超10bp差异%d条", summary["normalized_events"], summary["unresolved_events_isolated"], summary["remaining_computable_error_gt_10bp"])


if __name__ == "__main__":
    main()
