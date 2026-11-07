"""独立验收五因子原值、窗口、覆盖报告及输入未变；不写任何数据。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mfl" / "data"))

from mfl.data.common import setup_logging
from mfl.factors.build import protected_manifest, sha256
from mfl.factors.core import NAMES
from price_quality import read_rules

LOG = setup_logging("check_factors")


def _same(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> None:
    assert np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True).all(), "因子数值核对失败"


def _take(frame: pd.DataFrame, rows: pd.Index | pd.Series, codes: pd.Series) -> np.ndarray:
    ri, ci = frame.index.get_indexer(rows), frame.columns.get_indexer(codes)
    result = np.full(len(rows), np.nan)
    ok = (ri >= 0) & (ci >= 0)
    result[ok] = frame.to_numpy()[ri[ok], ci[ok]]
    return result


def main() -> None:
    """以月度分段计数和逐日滞后求和复算，区别于构建器的日网格累计计数。"""
    audit = ROOT / "results" / "factors"
    manifest = json.loads((audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "raw_factors" and manifest["inputs_unchanged"]
    current = protected_manifest(ROOT)
    assert current == manifest["protected_before"] == manifest["protected_after"], "raw/interim不再匹配本轮输入"
    for name, digest in manifest["output_sha256"].items():
        assert sha256(ROOT / name) == digest, f"产物清单不一致：{name}"
    for name, digest in manifest["source_sha256"].items():
        assert sha256(ROOT / name) == digest, f"因子源码已变化，须重建：{name}"
    for name, digest in manifest["config_sha256"].items():
        assert sha256(Path(name)) == digest, f"因子或质量规则已变化，须重建：{name}"
    cfg = manifest["settings"]
    raw = pd.read_parquet(ROOT / "data" / "processed" / "factors" / "raw.parquet")
    q = pd.read_parquet(ROOT / "data" / "processed" / "factors" / "quality.parquet")
    u = pd.read_parquet(ROOT / "data" / "interim" / "universe.parquet")
    u = u[u.date.between(pd.Timestamp(manifest["requested_start"]), pd.Timestamp(manifest["requested_end"]))].sort_values(["date", "code"]).reset_index(drop=True)
    assert str(raw.date.min().date()) == manifest["signal_start"]
    assert str(raw.date.max().date()) == manifest["signal_end"]
    assert raw[["date", "code"]].equals(u[["date", "code"]])
    assert raw.columns.tolist() == ["date", "code", *NAMES]
    assert len(q) == len(raw) * 5 and not q.duplicated(["date", "code", "factor"]).any()
    assert q.groupby(["date", "code"]).size().eq(5).all() and set(q.factor) == set(NAMES)
    assert q.loc[q.valid, "value"].notna().all() and q.loc[~q.valid, "value"].isna().all()
    assert q.valid.eq(q.primary_reason.eq("ok")).all()
    assert q.loc[~q.valid, "reason_codes"].str.len().gt(0).all()
    assert (q.loc[q.valid, "latest_input_date"] <= q.loc[q.valid, "date"]).all()
    pivot = q.pivot(index=["date", "code"], columns="factor", values="value").reset_index()
    _same(raw[NAMES].to_numpy(), pivot[NAMES].to_numpy())
    _same(q.value, q.base_value * q.direction)
    eligibility = q.merge(u[["date", "code", "tradable"]], on=["date", "code"], validate="many_to_one")
    assert eligibility.eligible.eq(eligibility.tradable.fillna(False)).all()
    p = pd.read_parquet(ROOT / "data" / "interim" / "prices.parquet",
                        columns=["date", "code", "close_adj", "total_mv", "turnover_rate"],
                        filters=[("code", "in", u.code.unique().tolist()), ("date", "<=", u.date.max())])
    pit = pd.read_parquet(ROOT / "data" / "interim" / "fundamental_panel.parquet")
    financial = u[["date", "code"]].merge(pit, on=["date", "code"], how="left", validate="one_to_one")
    financial = financial.merge(p[["date", "code", "total_mv"]], on=["date", "code"], how="left", validate="one_to_one")
    time_ok = (financial[["end_date", "ann_date", "ann_date_eff", "input_max_ann_date"]].notna().all(axis=1)
               & financial.ann_date_eff.le(financial.date) & financial.input_max_ann_date.lt(financial.date)
               & financial.input_max_ann_date.le(financial.ann_date) & financial.end_date.le(financial.ann_date)
               & financial.ann_date_eff.eq(financial.ann_date + pd.Timedelta(days=1))
               & (financial.date - financial.end_date).dt.days.le(cfg["financial"]["max_report_age_days"]))
    profit_ok = np.isfinite(financial.net_profit_ttm)
    for name in ("EP", "ROE_TTM"):
        part = q[q.factor.eq(name)].sort_values(["date", "code"]).reset_index(drop=True)
        if name == "EP":
            ok = time_ok & profit_ok & np.isfinite(financial.total_mv) & financial.total_mv.gt(0)
            value = financial.net_profit_ttm / (financial.total_mv.where(ok) * cfg["financial"]["market_cap_unit"])
        else:
            ok = time_ok & profit_ok & np.isfinite(financial.avg_equity) & financial.avg_equity.gt(0) & np.isfinite(financial.roe_ttm)
            value = financial.roe_ttm.where(ok)
        assert part.valid.eq(ok).all(), f"{name}有效条件不一致"
        _same(part.value, value)
    calendar = pd.read_parquet(ROOT / "data" / "raw" / "trade_cal" / "data.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.eq(1) & calendar.exchange.eq("SSE"), "cal_date"], format="%Y%m%d")).astype("datetime64[ns]").sort_values()
    month_ends = pd.Series(days).groupby(days.to_period("M")).max()
    days = days[days <= u.date.max()]
    price_series = p.set_index(["date", "code"]).close_adj
    observed = p[np.isfinite(p.close_adj) & p.close_adj.gt(0)].assign(month=lambda x: x.date.dt.to_period("M"))
    monthly_count = observed.groupby(["month", "code"]).size().unstack("code").reindex(
        index=pd.period_range(days.min(), days.max(), freq="M"), columns=u.code.unique(), fill_value=0,
    ).fillna(0)
    calendar_count = pd.Series(1, index=days).groupby(days.to_period("M")).sum()
    complete = monthly_count.eq(calendar_count, axis=0)
    rules = read_rules(ROOT / "config" / "price_quality.yaml")
    for name, lookback, skip in [("MOM_12_1", cfg["momentum"]["lookback_months"], cfg["momentum"]["skip_months"]),
                                  ("REV_1M", cfg["reversal"]["lookback_months"], 0)]:
        part = q[q.factor.eq(name)].reset_index(drop=True)
        months = part.date.dt.to_period("M")
        start = (months - lookback).map(month_ends)
        end = (months - skip).map(month_ends)
        pd.testing.assert_series_equal(part.window_start, start, check_names=False)
        pd.testing.assert_series_equal(part.window_end, end, check_names=False)
        left = price_series.reindex(pd.MultiIndex.from_arrays([start, part.code])).to_numpy()
        right = price_series.reindex(pd.MultiIndex.from_arrays([end, part.code])).to_numpy()
        full_months = lookback - skip
        month_ok = complete.rolling(full_months, min_periods=full_months).sum().eq(full_months)
        all_months = _take(month_ok, end.dt.to_period("M"), part.code) == 1
        ok = np.isfinite(left) & (left > 0) & np.isfinite(right) & (right > 0) & all_months
        inferred = pd.Series(False, index=part.index)
        for rule in rules:
            event = pd.Timestamp(rule["date"])
            if rule["action"] == "quarantine":
                ok &= ~(part.code.eq(rule["code"]) & start.le(event) & end.ge(event)).to_numpy()
            else:
                inferred |= part.code.eq(rule["code"]) & start.lt(event) & end.ge(event)
        assert part.valid.eq(ok).all(), f"{name}严格月度窗口判定不一致"
        expected = np.full(len(part), np.nan)
        expected[ok] = (right[ok] / left[ok] - 1) * cfg["directions"][name]
        _same(part.value, expected)
        assert part.crosses_inferred_boundary.eq(inferred).all()
    part = q[q.factor.eq("TURN_20")].reset_index(drop=True)
    turnover = p.set_index(["date", "code"]).turnover_rate
    positions = days.get_indexer(part.date)
    window = cfg["turnover"]["window"]
    total, count = np.zeros(len(part)), np.zeros(len(part), dtype=int)
    for lag in range(window):
        valid_date = positions - lag >= 0
        query_dates = pd.DatetimeIndex([pd.NaT] * len(part)).to_numpy(dtype="datetime64[ns]")
        query_dates[valid_date] = days[positions[valid_date] - lag].to_numpy(dtype="datetime64[ns]")
        values = turnover.reindex(pd.MultiIndex.from_arrays([query_dates, part.code])).to_numpy()
        valid_value = np.isfinite(values) & (values >= 0)
        total += np.where(valid_value, values, 0)
        count += valid_value
    ok = count == window
    assert part.valid.eq(ok).all(), "TURN_20没有使用完整市场交易日窗口"
    expected = np.full(len(part), np.nan)
    expected[ok] = -total[ok] / window / cfg["turnover"]["percent_unit"]
    _same(part.value, expected)
    assert part.valid_observations.eq(count).all()
    coverage = pd.read_csv(audit / "coverage.csv").set_index("factor")
    for name in NAMES:
        part = q[q.factor.eq(name)]
        assert int(coverage.loc[name, "total"]) == len(part)
        assert int(coverage.loc[name, "valid"]) == int(part.valid.sum())
        assert int(coverage.loc[name, "eligible_valid"]) == int((part.eligible & part.valid).sum())
    reasons = pd.read_csv(audit / "missing_reasons.csv")
    assert reasons["count"].sum() == int((~q.valid).sum())
    LOG.info("OK：%d行五因子、%d行质量明细；独立公式/完整窗口/方向/PIT/覆盖/哈希检查通过；%d个输入未改", len(raw), len(q), len(current))


if __name__ == "__main__":
    main()
