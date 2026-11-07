"""显式价格规则：基准往返、正常除权保留、原始锚点和跨事件窗口。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

import adjust
import common
import price_quality as quality


def _rule(date="20231229", previous="20231228", before=2., after=1., action="normalize"):
    return {"id": f"X_{date}", "code": "X", "date": date, "previous_date": previous,
            "expected_previous_factor": before, "expected_factor": after, "action": action,
            "status": "synthetic_review", "reason": "合成用例", "evidence": ["synthetic"]}


def _setup(monkeypatch, tmp_path, rules):
    # 第三天为正常10%因子增长；最后一天的第二次基准切换将因子乘回2。
    daily = pd.DataFrame({
        "ts_code": ["X"] * 4, "trade_date": ["20231228", "20231229", "20240102", "20240103"],
        "close": [10., 11., 10., 12.], "pre_close": [10., 10., 10., 10.],
        "open": [10., 11., 10., 12.], "high": [10., 11., 10., 12.], "low": [10., 11., 10., 12.],
        "vol": [100.] * 4, "amount": [100.] * 4, "pct_chg": [0., 10., 0., 20.],
    })
    # 1.1的真实因子增长对应11元前收与10元除权参考前收。
    factors = [2., 1., 1.1, 2.2]
    raw = tmp_path / "raw"
    for table, frame in [("daily", daily), ("adj_factor", daily[["ts_code", "trade_date"]].assign(adj_factor=factors))]:
        for year, group in frame.groupby(frame.trade_date.str[:4]):
            path = raw / table / f"year={year}"
            path.mkdir(parents=True)
            group.to_parquet(path / "data.parquet", index=False)
    config = tmp_path / "rules.yaml"
    config.write_text(yaml.safe_dump({"version": 1, "events": rules}), encoding="utf-8")
    monkeypatch.setattr(quality, "RULES_PATH", config)
    monkeypatch.setattr(common, "RAW", raw)
    monkeypatch.setattr(adjust, "RAW", raw)
    monkeypatch.setattr(adjust, "INTERIM", tmp_path / "interim")
    return raw, config


def test_piecewise_scaling_preserves_real_dividend_and_other_stocks():
    frame = pd.DataFrame({"ts_code": ["X", "X", "Y", "X", "X"],
                          "trade_date": ["20231228", "20231229", "20231229", "20240102", "20240103"],
                          "adj_factor": [2., 1., 7., 1.1, 2.2]})
    original = frame.copy(deep=True)
    rules = [_rule(), _rule("20240103", "20240102", 1.1, 2.2)]
    result = quality.apply_factor_rules(frame, rules)
    assert result.adj_factor.tolist() == pytest.approx([2., 2., 7., 2.2, 2.2])
    pd.testing.assert_frame_equal(frame, original)
    assert result.index.equals(frame.index)


def test_scaling_does_not_fill_missing_factor():
    frame = pd.DataFrame({"ts_code": ["X", "X"], "trade_date": ["20231229", "20240102"], "adj_factor": [np.nan, 1.1]})
    result = quality.apply_factor_rules(frame, [_rule()])
    assert pd.isna(result.adj_factor.iloc[0])
    assert result.adj_factor.iloc[1] == pytest.approx(2.2)


def test_full_partial_and_streamed_returns_agree(monkeypatch, tmp_path):
    rules = [_rule(), _rule("20240103", "20240102", 1.1, 2.2)]
    _setup(monkeypatch, tmp_path, rules)
    full = adjust.build_prices("20231228", "20240103")
    partial = adjust.build_prices("20240102", "20240103")
    assert full.adj_factor.tolist() == pytest.approx([2., 2., 2.2, 2.2])
    assert full.close_adj.tolist() == pytest.approx([20., 22., 22., 26.4])
    assert full.ret.iloc[1:].tolist() == pytest.approx([.1, 0., .2])
    pd.testing.assert_frame_equal(partial, full.iloc[2:].reset_index(drop=True), check_dtype=False)
    streamed = pd.read_parquet(adjust.write_prices("20231228", "20240103"))
    pd.testing.assert_frame_equal(streamed, full, check_dtype=False)
    # 重跑仍从raw开始，不把已经校正的倍率再叠加到旧产物。
    pd.testing.assert_frame_equal(full, adjust.build_prices("20231228", "20240103"))


@pytest.mark.parametrize("change", ["factor", "previous_date", "ex_reference"])
def test_changed_raw_anchor_is_rejected_even_for_later_partial_range(monkeypatch, tmp_path, change):
    rule = _rule()
    raw, _ = _setup(monkeypatch, tmp_path, [rule])
    if change == "factor":
        path = raw / "adj_factor" / "year=2023" / "data.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.trade_date.eq("20231229"), "adj_factor"] = 1.5
    else:
        path = raw / "daily" / "year=2023" / "data.parquet"
        frame = pd.read_parquet(path)
        if change == "previous_date":
            frame.loc[frame.trade_date.eq("20231228"), "trade_date"] = "20231227"
        else:
            frame.loc[frame.trade_date.eq("20231229"), "pre_close"] = 9.5
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="锚点不符"):
        adjust.build_prices("20240102", "20240103")


def test_quarantine_preserves_quotes_and_breaks_both_adjacent_returns(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [_rule(action="quarantine")])
    out = adjust.build_prices("20231228", "20240103")
    assert out.close.tolist() == [10., 11., 10., 12.]
    assert out.loc[1, ["adj_factor", "close_adj", "ret"]].isna().all()
    assert pd.isna(out.ret.iloc[2])
    assert out.ret.iloc[3] == pytest.approx(1.4)
    # 起点的上一条实际行情被隔离时，不能用更早的有效价格跳过它。
    partial = adjust.build_prices("20240102", "20240103")
    assert pd.isna(partial.ret.iloc[0])


def test_window_guard_rejects_endpoints_that_skip_quarantined_date():
    windows = pd.DataFrame({"code": ["X", "X", "X", "X", "Y", "X"],
                            "start": ["20231228", "20231228", "20231229", "20240101", "20231228", "20231228"],
                            "end": ["20231228", "20240103", "20240103", "20240103", "20240103", "20231229"]})
    rules = [_rule(action="quarantine")]
    assert quality.price_window_validity(windows, rules).tolist() == [True, False, False, True, True, False]
    invalid = windows.iloc[:1].assign(end="20230101")
    with pytest.raises(ValueError, match="起止顺序"):
        quality.price_window_validity(invalid, rules)


@pytest.mark.parametrize("edit", ["duplicate", "action", "negative", "missing_evidence", "invalid_date"])
def test_invalid_rule_configuration_is_rejected(tmp_path, edit):
    rule = _rule()
    rules = [rule]
    if edit == "duplicate":
        rules.append(rule.copy())
    elif edit == "action":
        rule["action"] = "auto_fill"
    elif edit == "negative":
        rule["expected_factor"] = -1.
    elif edit == "missing_evidence":
        rule["evidence"] = []
    else:
        rule["date"] = "20230230"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "events": rules}), encoding="utf-8")
    with pytest.raises(ValueError):
        quality.read_rules(path)
