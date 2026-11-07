"""用合成数据验证五因子公式、PIT、交易日窗口和未来数据不变性。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mfl.factors.build import compute_factors, coverage_tables
from mfl.factors.core import NAMES, load_settings


@pytest.fixture
def inputs():
    settings = load_settings(Path(__file__).resolve().parents[1] / "config" / "factors.yaml")
    days = pd.bdate_range("2014-01-01", "2016-03-31")
    calendar = pd.DataFrame({"cal_date": days.strftime("%Y%m%d"), "is_open": 1})
    frames = []
    for code, shift in [("X", 0.), ("Y", 5.)]:
        close = 10 + shift + np.arange(len(days)) * .01
        frames.append(pd.DataFrame({"date": days, "code": code, "close": close,
                                    "adj_factor": 1., "close_adj": close, "total_mv": 200000., "turnover_rate": 2.}))
    prices = pd.concat(frames, ignore_index=True)
    universe = pd.DataFrame({"date": pd.to_datetime(["2015-01-30", "2015-02-27", "2015-12-31", "2016-01-29", "2016-01-29"]),
                            "code": ["X", "X", "X", "X", "Y"], "tradable": [False, True, True, True, True]})
    pit = universe[["date", "code"]].copy()
    pit["end_date"] = pit.date.dt.to_period("Q").sub(1).dt.end_time.dt.normalize()
    pit["ann_date"] = pit.date - pd.Timedelta(days=2)
    pit["ann_date_eff"] = pit.date - pd.Timedelta(days=1)
    pit["input_max_ann_date"] = pit.ann_date
    pit["net_profit_ttm"], pit["avg_equity"], pit["roe_ttm"] = 100_000_000., 500_000_000., .2
    return prices, universe, pit, calendar, [], settings


def _build(inputs):
    return compute_factors(*inputs)


def _row(quality, name, date="2015-01-30", code="X"):
    return quality[quality.factor.eq(name) & quality.date.eq(pd.Timestamp(date)) & quality.code.eq(code)].iloc[0]


def test_hand_calculated_five_factors_and_direction_applied_once(inputs):
    prices, universe, *_ = inputs
    raw, q = _build(inputs)
    quotes = prices.set_index(["date", "code"]).close_adj
    p = lambda date: quotes.loc[(pd.Timestamp(date), "X")]
    row = raw[raw.date.eq(pd.Timestamp("2015-01-30")) & raw.code.eq("X")].iloc[0]
    assert row.EP == pytest.approx(.05)  # 一亿元利润/二十亿元总市值
    assert row.ROE_TTM == pytest.approx(.2)
    assert row.MOM_12_1 == pytest.approx(p("2014-12-31") / p("2014-01-31") - 1)
    assert row.REV_1M == pytest.approx(-(p("2015-01-30") / p("2014-12-31") - 1))
    assert row.TURN_20 == pytest.approx(-.02)  # 百分数2转小数.02，再取负
    assert np.allclose(q.value, q.base_value * q.direction)
    assert len(raw) == len(universe) and len(q) == len(universe) * 5
    assert not _row(q, "EP").eligible and _row(q, "EP").valid
    assert raw.columns.tolist() == ["date", "code", *NAMES]


def test_monthly_momentum_skips_last_month_without_extra_signal_shift(inputs):
    raw, _ = _build(inputs)
    altered = list(deepcopy(inputs))
    p = altered[0]
    mask = p.code.eq("X") & p.date.between("2015-01-01", "2015-01-30")
    p.loc[mask, ["close", "close_adj"]] *= 2
    changed, _ = _build(altered)
    assert raw.MOM_12_1.iloc[0] == changed.MOM_12_1.iloc[0]
    assert raw.REV_1M.iloc[0] != changed.REV_1M.iloc[0]


def test_pre_membership_history_is_used_and_partial_range_is_identical(inputs):
    full, fq = _build(inputs)
    partial = list(inputs)
    partial[1] = inputs[1].loc[inputs[1].date.eq(pd.Timestamp("2016-01-29"))].copy()
    raw, q = _build(partial)
    assert raw[NAMES].notna().all().all()  # Y首次出现在名单，仍有入池前行情
    expected = full[full.date.eq(pd.Timestamp("2016-01-29"))].reset_index(drop=True)
    pd.testing.assert_frame_equal(raw, expected)
    pd.testing.assert_frame_equal(q, fq[fq.date.eq(pd.Timestamp("2016-01-29"))].reset_index(drop=True))


def test_first_year_history_is_missing_without_dropping_other_factors(inputs):
    changed = list(inputs)
    changed[0] = inputs[0][inputs[0].date.ge("2015-01-01")].copy()
    changed[3] = inputs[3][inputs[3].cal_date.ge("20150101")].copy()
    raw, q = _build(changed)
    assert raw.loc[raw.date.dt.year.eq(2015), "MOM_12_1"].isna().all()
    assert raw.loc[raw.date.eq(pd.Timestamp("2016-01-29")), "MOM_12_1"].notna().all()
    assert raw.EP.notna().all()
    assert _row(q, "MOM_12_1").primary_reason == "insufficient_history"


def test_missing_trading_day_is_not_skipped_or_zero_filled(inputs):
    changed = list(inputs)
    changed[0] = inputs[0][~(inputs[0].code.eq("X") & inputs[0].date.eq("2015-01-15"))].copy()
    _, q = _build(changed)
    assert _row(q, "REV_1M").primary_reason == "incomplete_price_window"
    assert _row(q, "TURN_20").primary_reason == "incomplete_turnover_window"
    assert _row(q, "TURN_20").valid_observations == 19
    assert _row(q, "MOM_12_1").valid  # 动量窗口只到2014年末


def test_missing_month_end_does_not_use_last_available_quote(inputs):
    changed = list(inputs)
    changed[0] = inputs[0][~(inputs[0].code.eq("X") & inputs[0].date.eq("2014-12-31"))].copy()
    _, q = _build(changed)
    assert _row(q, "REV_1M").primary_reason == "missing_endpoint"
    assert _row(q, "MOM_12_1").primary_reason == "missing_endpoint"


def test_quarantined_event_blocks_valid_endpoints_but_not_turnover(inputs):
    changed = list(inputs)
    changed[4] = [{"code": "X", "date": "20150115", "action": "quarantine"}]
    _, q = _build(changed)
    assert _row(q, "REV_1M").primary_reason == "unresolved_price_event"
    assert _row(q, "MOM_12_1", "2015-02-27").primary_reason == "unresolved_price_event"
    assert _row(q, "MOM_12_1").valid and _row(q, "TURN_20").valid


def test_inferred_boundary_is_marked_without_reapplying_adjustment(inputs):
    changed = list(inputs)
    changed[4] = [{"code": "X", "date": "20150115", "action": "normalize"}]
    raw, q = _build(changed)
    original, _ = _build(inputs)
    pd.testing.assert_frame_equal(raw, original)
    assert _row(q, "REV_1M").crosses_inferred_boundary
    assert not _row(q, "TURN_20").crosses_inferred_boundary


def test_negative_profit_and_roe_are_valid_and_no_pe_fallback(inputs):
    changed = list(deepcopy(inputs))
    changed[2].loc[:, "net_profit_ttm"] = -100_000_000.
    changed[2].loc[:, "roe_ttm"] = -.2
    changed[0]["pe_ttm"] = 999.
    raw, q = _build(changed)
    assert np.allclose(raw.EP, -.05) and np.allclose(raw.ROE_TTM, -.2)
    assert _row(q, "EP").negative_profit and _row(q, "EP").valid


@pytest.mark.parametrize("cap,reason", [(0., "invalid_market_cap"), (np.inf, "invalid_market_cap"), (np.nan, "missing_market_cap")])
def test_missing_or_invalid_same_day_market_cap_has_no_fallback(inputs, cap, reason):
    changed = list(deepcopy(inputs))
    changed[0].loc[changed[0].date.eq("2015-01-30") & changed[0].code.eq("X"), "total_mv"] = cap
    raw, q = _build(changed)
    assert pd.isna(raw.EP.iloc[0])
    assert _row(q, "EP").primary_reason == reason
    assert _row(q, "ROE_TTM").valid


def test_nonpositive_equity_disables_only_roe(inputs):
    changed = list(deepcopy(inputs))
    changed[2].loc[0, "avg_equity"] = 0.
    _, q = _build(changed)
    assert _row(q, "ROE_TTM").primary_reason == "invalid_equity"
    assert _row(q, "EP").valid


@pytest.mark.parametrize("problem", ["future", "stale", "missing"])
def test_pit_time_staleness_and_missing_rows_are_explicit(inputs, problem):
    changed = list(deepcopy(inputs))
    if problem == "future":
        changed[2].loc[0, "ann_date_eff"] = pd.Timestamp("2015-02-01")
        reason = "pit_time_violation"
    elif problem == "stale":
        changed[2].loc[0, "end_date"] = pd.Timestamp("2012-12-31")
        reason = "stale_financial"
    else:
        changed[2] = changed[2].iloc[1:].copy()
        reason = "missing_pit"
    raw, q = _build(changed)
    assert len(raw) == len(inputs[1])
    assert _row(q, "EP").primary_reason == reason
    assert _row(q, "ROE_TTM").primary_reason == reason


def test_append_future_quotes_and_pit_does_not_change_past_factors(inputs):
    reduced = list(inputs)
    reduced[0] = inputs[0][inputs[0].date.le("2016-01-29")].copy()
    before, bq = _build(reduced)
    expanded = list(deepcopy(inputs))
    future = expanded[2].iloc[-1:].copy()
    future["date"] = pd.Timestamp("2016-03-31")
    future["ann_date"] = pd.Timestamp("2016-03-30")
    future["ann_date_eff"] = pd.Timestamp("2016-03-31")
    future["input_max_ann_date"] = future.ann_date
    future["net_profit_ttm"] = 999_000_000.
    expanded[2] = pd.concat([expanded[2], future], ignore_index=True)
    after, aq = _build(expanded)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(bq, aq)


def test_coverage_reason_counts_and_duplicate_keys(inputs):
    changed = list(inputs)
    changed[0] = inputs[0][inputs[0].date.ge("2015-01-01")].copy()
    _, q = _build(changed)
    tables = coverage_tables(q)
    assert (tables["coverage"].total == len(inputs[1])).all()
    assert tables["missing_reasons"]["count"].sum() == (~q.valid).sum()
    assert not q.duplicated(["date", "code", "factor"]).any()
    bad = list(inputs)
    bad[1] = pd.concat([inputs[1], inputs[1].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="主键"):
        _build(bad)


def test_midmonth_target_and_inconsistent_adjusted_price_are_rejected(inputs):
    changed = list(deepcopy(inputs))
    changed[1].loc[0, "date"] = pd.Timestamp("2015-01-29")
    with pytest.raises(ValueError, match="月末"):
        _build(changed)
    changed = list(deepcopy(inputs))
    changed[0].loc[0, "close_adj"] = 999.
    with pytest.raises(ValueError, match="复权价"):
        _build(changed)


@pytest.mark.parametrize("value", [np.nan, -1., np.inf])
def test_invalid_turnover_is_a_gap_not_a_zero(inputs, value):
    changed = list(deepcopy(inputs))
    p = changed[0]
    p.loc[p.date.eq("2015-01-30") & p.code.eq("X"), "turnover_rate"] = value
    _, q = _build(changed)
    assert _row(q, "TURN_20").primary_reason == "incomplete_turnover_window"
    assert _row(q, "EP").valid


def test_pure_build_does_not_mutate_inputs(inputs):
    copies = [value.copy(deep=True) for value in inputs[:4]]
    _build(inputs)
    for old, current in zip(copies, inputs[:4]):
        pd.testing.assert_frame_equal(old, current)


def _file_inputs(tmp_path, inputs):
    prices, universe, pit, calendar, rules, settings = inputs
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    for name, frame in [("prices", prices), ("universe", universe), ("fundamental_panel", pit)]:
        frame.to_parquet(interim / f"{name}.parquet", index=False)
    raw = tmp_path / "data" / "raw" / "trade_cal"
    raw.mkdir(parents=True)
    calendar.assign(exchange="SSE").to_parquet(raw / "data.parquet", index=False)
    config = tmp_path / "config"
    config.mkdir()
    (config / "factors.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    (config / "price_quality.yaml").write_text(yaml.safe_dump({"version": 1, "events": rules}), encoding="utf-8")
    (config / "base.yaml").write_text(yaml.safe_dump({"start": "2015-01-01", "end": "2016-03-31", "data_start": "2014-01-01"}), encoding="utf-8")


def test_file_pipeline_preserves_inputs_and_repeats_identical_factor_files(inputs, tmp_path):
    from mfl.factors.build import protected_manifest, run
    _file_inputs(tmp_path, inputs)
    before = protected_manifest(tmp_path)
    first = run(tmp_path)
    second = run(tmp_path)
    for name in ("raw.parquet", "quality.parquet"):
        key = f"data/processed/factors/{name}"
        assert first["output_sha256"][key] == second["output_sha256"][key]
    assert protected_manifest(tmp_path) == before
    assert (tmp_path / "reports" / "factor_coverage.md").exists()
    csv = tmp_path / "results" / "factors" / "coverage.csv"
    assert b"\r\r\n" not in csv.read_bytes()


def test_changed_input_during_build_prevents_publication(inputs, tmp_path, monkeypatch):
    import mfl.factors.build as build
    _file_inputs(tmp_path, inputs)
    original = build.compute_factors

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        # 只破坏本用例的合成临时文件，用于确认生产发布保护会阻止并发输入变化。
        (tmp_path / "data" / "raw" / "trade_cal" / "data.parquet").write_bytes(b"changed-synthetic-input")
        return result

    monkeypatch.setattr(build, "compute_factors", changed)
    with pytest.raises(RuntimeError, match="输入发生变化"):
        build.run(tmp_path)
    assert not (tmp_path / "results" / "factors" / "run_manifest.json").exists()


def test_file_default_range_reads_base_and_retains_full_prewarm(inputs, tmp_path):
    from mfl.factors.build import run
    _file_inputs(tmp_path, inputs)
    path = tmp_path / "config" / "base.yaml"
    path.write_text(yaml.safe_dump({"start": "2016-01-01", "end": "2016-03-31", "data_start": "2014-01-01"}), encoding="utf-8")
    manifest = run(tmp_path)
    raw = pd.read_parquet(tmp_path / "data" / "processed" / "factors" / "raw.parquet")
    expected, _ = _build(inputs)
    expected = expected[expected.date.ge("2016-01-01")].reset_index(drop=True)
    pd.testing.assert_frame_equal(raw, expected)
    assert raw.MOM_12_1.notna().all()
    assert manifest["requested_start"] == "20160101" and manifest["data_start"] == "20140101"
    assert str(path) in manifest["config_sha256"]
    overridden = run(tmp_path, start="2015-01-01", end="2015-02-28")
    assert overridden["requested_start"] == "20150101" and overridden["rows"] == 2


@pytest.mark.parametrize("edit", ["stage", "direction", "window", "completeness"])
def test_invalid_settings_are_rejected(inputs, tmp_path, edit):
    cfg = deepcopy(inputs[-1])
    if edit == "stage":
        cfg["stage"] = "preprocess"
    elif edit == "direction":
        cfg["directions"]["REV_1M"] = 1
    elif edit == "window":
        cfg["turnover"]["window"] = 0
    else:
        cfg["price_windows"]["require_complete"] = False
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path)
