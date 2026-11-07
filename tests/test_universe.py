"""股票池边界与可交易性回归测试；全部使用小型合成数据。"""

from __future__ import annotations

import pandas as pd
import pytest

import universe as u


def basic(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": codes, "name": ["ST现在"] * len(codes), "list_date": ["20000103"] * len(codes)})


def test_names_never_fill_from_current_and_respect_announcement() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["20150130", "20150227", "20150331"]), "code": ["X"] * 3})
    names = pd.DataFrame({"ts_code": ["X"], "name": ["历史正常"], "start_date": ["20150201"], "end_date": ["20150228"], "ann_date": ["20150210"]})
    out = u.attach_names(panel, basic(["X"]), names)
    assert out.name_known.tolist() == [False, True, False]
    assert out.loc[1, "is_st"] == False
    assert out.loc[[0, 2], "is_st"].isna().all()
    assert out.loc[[0, 2], "name"].isna().all()
    before_ann = panel.iloc[[0]].assign(date=pd.Timestamp("20150205"))
    assert not u.attach_names(before_ann, basic(["X"]), names).name_known.any()


def test_missing_names_are_unknown() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["20150130"]), "code": ["X"]})
    out = u.attach_names(panel, basic(["X"]), None)
    assert out.is_st.isna().all()
    assert not out.name_known.any()


def test_conflicting_historical_names_remain_unknown(caplog: pytest.LogCaptureFixture) -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["20150130"]), "code": ["X"]})
    names = pd.DataFrame({"ts_code": ["X", "X"], "name": ["历史正常", "ST历史"], "start_date": ["20150101"] * 2, "end_date": [None] * 2})
    with caplog.at_level("WARNING"):
        out = u.attach_names(panel, basic(["X"]), names)
    assert out.is_st.isna().all() and not out.name_known.any()
    assert "冲突" in caplog.text


def test_asof_accepts_ns_panel_and_us_source(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["20150130"]).astype("datetime64[ns]"), "code": ["X"]})
    names = pd.DataFrame({"ts_code": ["X"], "name": ["历史正常"], "start_date": pd.to_datetime(["20150101"]).astype("datetime64[us]"), "end_date": [None]})
    out = u.attach_names(panel, basic(["X"]), names)
    assert out.name_known.all()
    daily = pd.DataFrame({"ts_code": ["X"], "trade_date": ["20150130"], "close": [10.]})
    monkeypatch.setattr(u, "_market_rows", lambda table, *args: daily if table == "daily" else pd.DataFrame())
    assert u.attach_price(panel, "20150101", "20150131").suspended.eq(False).all()


def test_age_uses_pre_start_calendar_and_marks_uncovered_age() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["20150130"] * 3), "code": ["OLD", "NEW", "UNKNOWN"]})
    sb = pd.DataFrame({"ts_code": ["OLD", "NEW", "UNKNOWN"], "list_date": ["20000103", "20150105", None]})
    full = pd.bdate_range("20000103", "20150130").tolist()
    out = u.attach_listed_days(panel, sb, full).set_index("code")
    assert out.loc["OLD", "listed_days"] > 120
    assert out.loc["NEW", "listed_days"] == 20
    assert pd.isna(out.loc["UNKNOWN", "listed_days"])
    limited = u.attach_listed_days(panel, sb, pd.bdate_range("20150101", "20150130").tolist()).set_index("code")
    assert pd.isna(limited.loc["OLD", "listed_days"])
    assert limited.loc["OLD", "listed_days_lower_bound"] == 22
    assert not limited.loc["OLD", "listed_days_exact"]


def test_offline_age_calendar_requires_matching_raw_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({"date": pd.date_range("20150101", "20150107"), "is_open": [1, 1, 0, 0, 1, 1, 1]})
    offline = pd.bdate_range("20140101", "20150107")
    monkeypatch.setattr(u, "_offline_calendar", lambda: offline)
    assert min(u._age_calendar(raw)) == pd.Timestamp("20140101")
    monkeypatch.setattr(u, "_offline_calendar", lambda: offline.drop(pd.Timestamp("20150105")))
    assert u._age_calendar(raw) == u.open_dates(raw)
    monkeypatch.setattr(u, "_offline_calendar", lambda: None)
    assert u._age_calendar(raw) == u.open_dates(raw)


def test_suspension_needs_evidence_and_handles_resumption(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = ["TRADE", "FULL", "PARTIAL", "BOTH", "RESUMED", "STALE", "PLAIN"]
    panel = pd.DataFrame({"code": codes, "date": pd.to_datetime(["20240131"] * len(codes))})
    daily = pd.DataFrame({"ts_code": codes + ["TRADE", "STALE"], "trade_date": ["20240102"] * len(codes) + ["20240131", "20240130"], "close": [10.] * (len(codes) + 2)})
    susp = pd.DataFrame({
        "ts_code": ["TRADE", "FULL", "PARTIAL", "BOTH", "BOTH", "RESUMED", "RESUMED", "STALE"],
        "trade_date": ["20240131", "20240130", "20240131", "20240131", "20240131", "20240129", "20240130", "20240129"],
        "suspend_type": ["S", "S", "S", "S", "R", "S", "R", "S"],
        "suspend_timing": [None, None, "13:00-15:00", None, None, None, None, None],
    })
    monkeypatch.setattr(u, "_market_rows", lambda table, *args: daily if table == "daily" else susp)
    out = u.attach_price(panel, "20240101", "20240131").set_index("code")
    assert out.loc["TRADE", "suspended"] == False
    assert out.loc["FULL", "suspended"] == True
    assert out.loc[["PARTIAL", "BOTH", "RESUMED", "STALE", "PLAIN"], "suspended"].isna().all()
    assert out.close.notna().sum() == 1


def test_missing_suspension_does_not_turn_missing_quote_into_suspension(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = pd.DataFrame({"code": ["X"], "date": pd.to_datetime(["20240131"])})
    daily = pd.DataFrame({"ts_code": ["X"], "trade_date": ["20240102"], "close": [10.]})
    monkeypatch.setattr(u, "_market_rows", lambda table, *args: daily if table == "daily" else pd.DataFrame())
    out = u.attach_price(panel, "20240101", "20240131")
    assert out.suspended.isna().all()


def _synthetic_build(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = ["NORMAL", "ST", "UNKNOWN", "FULL", "NO_LIMIT", "UP", "DOWN"]
    days = pd.date_range("20150101", "20150331")
    cal = pd.DataFrame({"cal_date": days.strftime("%Y%m%d"), "is_open": (days.day_of_week < 5).astype(int)})
    monkeypatch.setattr(u, "read_single", lambda table, *args: cal if table == "trade_cal" else basic(codes))
    names = pd.DataFrame({"ts_code": codes[:2] + codes[3:], "name": ["普通", "ST历史"] + ["普通"] * 4, "start_date": ["20000103"] * 6, "end_date": [None] * 6})
    monkeypatch.setattr(u, "read_optional", lambda *args: names)
    monkeypatch.setattr(u, "_offline_calendar", lambda: pd.bdate_range("19990101", "20150331"))
    snapshots = pd.DataFrame([(d, c, 100. / len(codes)) for d in pd.to_datetime(["20150130", "20150227", "20150331"]) for c in codes], columns=["date", "code", "weight"])
    monkeypatch.setattr(u, "load_index_members", lambda *args: snapshots)
    dates = ["20150130", "20150227", "20150331"]
    daily = pd.DataFrame([(c, d, 10.) for d in dates for c in codes if c != "FULL"] + [("FULL", "20150102", 10.)], columns=["ts_code", "trade_date", "close"])
    limits = pd.DataFrame([(c, d, 10. if c == "UP" else 11., 10. if c == "DOWN" else 9.) for d in dates for c in codes if c != "NO_LIMIT"], columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    suspend = pd.DataFrame({"ts_code": ["FULL"], "trade_date": ["20150105"], "suspend_type": ["S"], "suspend_timing": [None]})
    monkeypatch.setattr(u, "_market_rows", lambda table, *args: {"daily": daily, "stk_limit": limits, "suspend_d": suspend}[table])


def test_monthly_pipeline_keys_and_directional_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_build(monkeypatch)
    out = u.build_universe("20150101", "20150331")
    assert not out.duplicated(["date", "code"]).any()
    assert out.groupby("date").size().eq(7).all()
    assert out.is_st.fillna(False).any() and out.suspended.fillna(False).any()
    first = out[out.date == pd.Timestamp("20150130")].set_index("code")
    assert first.loc["NORMAL", "listed_days"] > 120
    assert first.loc["NORMAL", "can_buy"] and first.loc["NORMAL", "can_sell"]
    for c in ["ST", "UNKNOWN", "FULL", "NO_LIMIT", "UP"]:
        assert not first.loc[c, "can_buy"]
    assert first.loc["ST", "can_sell"] and first.loc["UNKNOWN", "can_sell"]
    assert not first.loc["FULL", "can_sell"]
    assert not first.loc["NO_LIMIT", "can_sell"]
    assert not first.loc["DOWN", "can_sell"]
    assert first.loc["DOWN", "can_buy"]


def test_midmonth_end_is_not_rebalance(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_build(monkeypatch)
    out = u.build_universe("20150101", "20150215")
    assert out.date.unique().tolist() == [pd.Timestamp("20150130")]


def test_invalid_limit_prices_do_not_allow_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_build(monkeypatch)
    reader = u._market_rows
    def invalid_limits(table, *args):
        df = reader(table, *args).copy()
        if table == "stk_limit":
            df.loc[df.ts_code.eq("NORMAL"), "up_limit"] = float("inf")
            df.loc[df.ts_code.eq("ST"), "down_limit"] = 0.
        return df
    monkeypatch.setattr(u, "_market_rows", invalid_limits)
    out = u.build_universe("20150101", "20150331").set_index("code")
    assert not out.loc["NORMAL", "can_buy"].any()
    assert not out.loc["ST", "can_sell"].any()


def test_missing_member_snapshot_raises_and_stale_snapshot_warns(caplog: pytest.LogCaptureFixture) -> None:
    iw = pd.DataFrame({"date": pd.to_datetime(["20150130"]), "code": ["X"], "weight": [100.]})
    with pytest.raises(RuntimeError, match="未覆盖"):
        u.snapshot_at_rebalance(iw, [pd.Timestamp("20141231")])
    with caplog.at_level("WARNING"):
        out = u.snapshot_at_rebalance(iw, [pd.Timestamp("20150430")])
    assert out.snapshot_date.iloc[0] == pd.Timestamp("20150130")
    assert "超过一个月" in caplog.text


def test_raw_column_projection_and_history_are_read_only(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(u, "RAW", tmp_path)
    path = tmp_path / "daily" / "year=2023" / "data.parquet"
    path.parent.mkdir(parents=True)
    raw = pd.DataFrame({"ts_code": ["X", "Y"], "trade_date": ["20231229", "20231229"], "close": [10., 20.], "unused": [1, 2]})
    raw.to_parquet(path, index=False)
    before = path.read_bytes()
    out = u._market_rows("daily", ["ts_code", "trade_date", "close"], ["X"], "20240131")
    assert out.ts_code.tolist() == ["X"]
    assert "unused" not in out
    assert path.read_bytes() == before
