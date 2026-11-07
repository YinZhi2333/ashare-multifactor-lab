"""逐行验证本次价格质量修复，基线由执行修复前的价格快照提供。

从项目根目录执行；不写raw或interim，仅生成results审计产物。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mfl" / "data"))
from common import INTERIM, RAW, ROOT, setup_logging
from price_quality import price_window_validity, read_rules, validate_anchors

LOG = setup_logging("verify_price_quality_repair")
OUT = ROOT / "results" / "adj_factor_repair"


def _same(left: pd.Series, right: pd.Series) -> np.ndarray:
    return np.isclose(left, right, equal_nan=True, rtol=1e-11, atol=1e-12)


def main() -> None:
    """独立构造累计倍率，核对全部价格列、边界收益和输入哈希。"""
    baseline = OUT / "prices_before.parquet"
    current = INTERIM / "prices.parquet"
    before = json.loads((OUT / "manifest_before.json").read_text(encoding="utf-8"))
    baseline_digest = hashlib.sha256()
    with baseline.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            baseline_digest.update(chunk)
    assert baseline_digest.hexdigest() == before["data/interim/prices.parquet"], "修复前基线与保存的哈希不符"
    rules = read_rules()
    validate_anchors(rules, RAW)
    assert pq.read_schema(baseline).equals(pq.read_schema(current)), "价格schema发生变化"
    normal_keys = {(r["date"], r["code"]) for r in rules if r["action"] == "normalize"}
    blocked_keys = {(r["date"], r["code"]) for r in rules if r["action"] == "quarantine"}
    allowed = normal_keys | blocked_keys
    # 事件后的第一条实际行情也必须留空，且不能跳过被隔离的复权价。
    for date, code in blocked_keys:
        quotes = pd.read_parquet(baseline, columns=["date", "code"], filters=[("code", "==", code)])
        later = quotes.loc[quotes["date"].gt(pd.to_datetime(date))].sort_values("date")
        if not later.empty:
            allowed.add((later["date"].iloc[0].strftime("%Y%m%d"), code))
    members = set(pd.read_parquet(INTERIM / "universe.parquet", columns=["code"])["code"])
    records, remaining, changed_returns = [], [], set()
    totals = {"rows": 0, "effective_factor_rows_changed": 0, "adjusted_close_rows_changed": 0,
              "return_rows_changed": 0, "factor_missing": 0, "return_missing": 0,
              "historical_member_factor_missing": 0, "computable_discrepancies_gt_10bp": 0,
              "historical_member_computable_discrepancies_gt_10bp": 0}
    for year in range(2015, 2025):
        filters = [("date", ">=", pd.Timestamp(f"{year}-01-01")), ("date", "<=", pd.Timestamp(f"{year}-12-31"))]
        old, new = (pd.read_parquet(path, filters=filters) for path in (baseline, current))
        assert old[["date", "code"]].equals(new[["date", "code"]]), "行情主键或排序改变"
        for col in old.columns.difference(["adj_factor", "close_adj", "ret"]):
            assert old[col].equals(new[col]), f"未经授权修改价格列：{col}"
        # 与生产实现不同：按股票预先计算累计倍率，再searchsorted映射到报价。
        multiplier = np.ones(len(old))
        for code in {r["code"] for r in rules if r["action"] == "normalize"}:
            steps = sorted([r for r in rules if r["code"] == code and r["action"] == "normalize"], key=lambda r: r["date"])
            boundaries = pd.to_datetime([r["date"] for r in steps]).to_numpy(dtype="datetime64[ns]")
            cumulative = np.cumprod([r["expected_previous_factor"] / r["expected_factor"] for r in steps])
            stock = old["code"].eq(code).to_numpy()
            pos = np.searchsorted(boundaries, old.loc[stock, "date"].to_numpy(dtype="datetime64[ns]"), side="right")
            multiplier[stock] = np.r_[1., cumulative][pos]
        expected = old["adj_factor"] * multiplier
        for date, code in blocked_keys:
            expected.loc[old["code"].eq(code) & old["date"].eq(pd.to_datetime(date))] = np.nan
        assert _same(new["adj_factor"], expected).all(), "因子不符合批准的分段倍率/隔离规则"
        assert _same(new["close_adj"], new["close"] * expected).all(), "复权价与有效因子不一致"
        changed = ~_same(old["ret"], new["ret"])
        keys = set(zip(old.loc[changed, "date"].dt.strftime("%Y%m%d"), old.loc[changed, "code"]))
        assert keys.issubset(allowed), f"登记边界以外的日收益发生变化：{keys - allowed}"
        changed_returns.update(keys)
        for date, code in normal_keys:
            mask = old["code"].eq(code) & old["date"].eq(pd.to_datetime(date))
            if mask.any():
                reference = old.loc[mask, "close"] / old.loc[mask, "pre_close"] - 1
                assert _same(new.loc[mask, "ret"], reference).all(), "普通边界未恢复参考价收益"
        for date, code in allowed - normal_keys:
            mask = old["code"].eq(code) & old["date"].eq(pd.to_datetime(date))
            assert new.loc[mask, "ret"].isna().all(), "未确认事件仍产生相邻收益"
        selected = pd.MultiIndex.from_frame(old[["date", "code"]]).isin(
            pd.MultiIndex.from_tuples([(pd.to_datetime(date), code) for date, code in normal_keys | blocked_keys]),
        )
        detail = new.loc[selected, ["date", "code", "adj_factor", "close_adj", "ret"]].copy()
        for col in ("adj_factor", "close_adj", "ret"):
            detail[f"before_{col}"] = old.loc[selected, col].to_numpy()
        records.append(detail)
        raw_daily = pd.read_parquet(RAW / "daily" / f"year={year}" / "data.parquet", columns=["ts_code", "trade_date", "pct_chg"])
        raw_daily = raw_daily.rename(columns={"ts_code": "code", "trade_date": "date"})
        raw_daily["date"] = pd.to_datetime(raw_daily["date"], format="%Y%m%d")
        diagnostic = new[["date", "code", "ret"]].merge(raw_daily, on=["date", "code"], validate="one_to_one")
        bad = diagnostic[(diagnostic["ret"] - diagnostic["pct_chg"] / 100).abs().gt(.001)].copy()
        remaining.append(bad)
        totals["rows"] += len(new)
        totals["effective_factor_rows_changed"] += int((~_same(old["adj_factor"], new["adj_factor"])).sum())
        totals["adjusted_close_rows_changed"] += int((~_same(old["close_adj"], new["close_adj"])).sum())
        totals["return_rows_changed"] += int(changed.sum())
        totals["factor_missing"] += int(new["adj_factor"].isna().sum())
        totals["return_missing"] += int(new["ret"].isna().sum())
        totals["historical_member_factor_missing"] += int((new["code"].isin(members) & new["adj_factor"].isna()).sum())
        totals["computable_discrepancies_gt_10bp"] += len(bad)
        totals["historical_member_computable_discrepancies_gt_10bp"] += int(bad["code"].isin(members).sum())
        LOG.info("%d年：%d条报价，边界外所有价格字段及日收益核对通过", year, len(new))
    assert changed_returns == allowed, "登记事件的实际变化范围不符合预期"
    after = {}
    for base in (RAW, INTERIM):
        for path in sorted(base.rglob("*.parquet")):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after[path.relative_to(ROOT).as_posix()] = digest.hexdigest()
    changed_files = [path for path in before if before[path] != after.get(path)]
    assert set(before) == set(after), "raw/interim文件集合改变"
    assert changed_files == ["data/interim/prices.parquet"], "存在计划外数据文件修改"
    window = pd.DataFrame({"code": ["600733.SH"] * 3,
                           "start": ["2018-02-01", "2018-09-28", "2018-09-27"],
                           "end": ["2018-09-28", "2018-10-31", "2018-10-31"]})
    assert price_window_validity(window, rules).tolist() == [False, True, False]
    summary = {**totals, "normalized_events": len(normal_keys), "unresolved_events": len(blocked_keys),
               "raw_files_unchanged": sum(path.startswith("data/raw/") for path in before),
               "only_changed_data_file": changed_files, "output_schema_unchanged": True,
               "window_guard_verified": True, "unlisted_returns_unchanged_within_tolerance": True}
    pd.concat(records).sort_values(["date", "code"]).to_csv(OUT / "events_before_after.csv", index=False, encoding="utf-8-sig")
    pd.concat(remaining).to_csv(OUT / "remaining_all_market_discrepancies.csv", index=False, encoding="utf-8-sig")
    for name, value in [("summary.json", summary), ("manifest_after.json", after)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("全量验证通过：%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
