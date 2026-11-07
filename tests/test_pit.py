"""财务 PIT 回归测试：全部使用小型合成数据，不读取真实 raw。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pit


def income(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["code", "end_date", "ann_date", "n_income_attr_p"])
    df["revenue"] = 1000.0
    for col in ["end_date", "ann_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d")
    return df


def balance(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["code", "end_date", "ann_date", "total_hldr_eqy_exc_min_int"],
    )
    df["total_assets"] = 2000.0
    for col in ["end_date", "ann_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d")
    return df


def fina(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["code", "end_date", "ann_date", "roe_yearly"])
    for col in ["end_date", "ann_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d")
    for col in pit.FI_COLS:
        if col not in df:
            df[col] = np.nan
    return df


def at(events: pd.DataFrame, announced: str, code: str = "X") -> pd.Series:
    chosen = events.loc[
        events["code"].eq(code) & events["ann_date"].eq(pd.Timestamp(announced))
    ]
    assert len(chosen) == 1
    return chosen.iloc[0]


def test_quarters_and_ttm_hand_calculated() -> None:
    periods = ["20230331", "20230630", "20230930", "20231231", "20240331"]
    announced = ["20230420", "20230820", "20231020", "20240320", "20240420"]
    inc = income([
        ("X", end, ann, profit)
        for end, ann, profit in zip(periods, announced, [10.0, 25.0, 45.0, 70.0, 20.0])
    ])
    bs = balance([
        ("X", end, ann, equity)
        for end, ann, equity in zip(periods, announced, [100.0, 110.0, 120.0, 130.0, 140.0])
    ])
    events = pit._build_ttm(inc, bs)

    assert [at(events, ann)["q_profit"] for ann in announced] == pytest.approx(
        [10.0, 15.0, 20.0, 25.0, 20.0]
    )
    assert at(events, "20240320")["net_profit_ttm"] == 70.0
    latest = at(events, "20240420")
    assert latest["net_profit_ttm"] == 20.0 + 70.0 - 10.0
    assert latest["avg_equity"] == 120.0
    assert latest["roe_ttm"] == pytest.approx(80.0 / 120.0)


def test_annual_ttm_does_not_need_previous_annual() -> None:
    events = pit._build_ttm(
        income([("X", "20231231", "20240320", 70.0)]), balance([])
    )
    row = at(events, "20240320")
    assert row["net_profit_ttm"] == 70.0
    assert pd.isna(row["roe_ttm"])


def test_historical_revision_recomputes_current_report_without_lookahead() -> None:
    inc = income([
        ("X", "20230331", "20230420", 10.0),
        ("X", "20231231", "20240320", 70.0),
        ("X", "20240331", "20240420", 20.0),
        ("X", "20230331", "20250510", 30.0),
    ])
    bs = balance([
        ("X", "20230331", "20230420", 100.0),
        ("X", "20231231", "20240320", 130.0),
        ("X", "20240331", "20240420", 140.0),
    ])
    events = pit._build_ttm(inc, bs)

    assert at(events, "20240420")["net_profit_ttm"] == 80.0
    revised = at(events, "20250510")
    assert revised["end_date"] == pd.Timestamp("20240331")
    assert revised["net_profit_ttm"] == 60.0
    assert (events["input_max_ann_date"] <= events["ann_date"]).all()


def test_future_dependency_is_not_used_before_it_is_announced() -> None:
    inc = income([
        ("X", "20231231", "20240320", 70.0),
        ("X", "20240331", "20240420", 20.0),
        ("X", "20230331", "20250510", 30.0),
    ])
    events = pit._build_ttm(inc, balance([]))
    assert pd.isna(at(events, "20240420")["net_profit_ttm"])
    assert at(events, "20250510")["net_profit_ttm"] == 60.0


def test_balance_announcement_does_not_delay_available_profit() -> None:
    inc = income([("X", "20231231", "20240320", 70.0)])
    bs = balance([
        ("X", "20221231", "20230320", 100.0),
        ("X", "20231231", "20240420", 140.0),
    ])
    events = pit._build_ttm(inc, bs)

    early = at(events, "20240320")
    assert early["net_profit_ttm"] == 70.0
    assert pd.isna(early["roe_ttm"])
    assert at(events, "20240420")["roe_ttm"] == pytest.approx(70.0 / 120.0)


def test_future_fina_does_not_delay_main_event() -> None:
    inc = income([("X", "20231231", "20240320", 70.0)])
    fi = fina([("X", "20231231", "20240520", 15.0)])
    fi["grossprofit_margin"] = 25.0
    events = pit._build_ttm(inc, balance([]), fi)

    early = at(events, "20240320")
    assert early["net_profit_ttm"] == 70.0
    assert pd.isna(early["grossprofit_margin"])
    assert at(events, "20240520")["grossprofit_margin"] == 25.0


def test_latest_report_does_not_fall_back_when_value_is_missing() -> None:
    inc = income([
        ("X", "20231231", "20240320", 70.0),
        ("X", "20240331", "20240420", np.nan),
    ])
    events = pit._build_ttm(inc, balance([]))
    latest = at(events, "20240420")
    assert latest["end_date"] == pd.Timestamp("20240331")
    assert pd.isna(latest["net_profit_ttm"])


def test_missing_quarter_does_not_turn_multiquarter_profit_into_one_quarter() -> None:
    inc = income([
        ("X", "20230331", "20230420", 10.0),
        ("X", "20230930", "20231020", 45.0),
    ])
    events = pit._build_ttm(inc, balance([]))
    assert pd.isna(at(events, "20231020")["q_profit"])


def test_same_day_annual_and_q1_choose_latest_report_period() -> None:
    inc = income([
        ("X", "20230331", "20230420", 10.0),
        ("X", "20231231", "20240420", 70.0),
        ("X", "20240331", "20240420", 20.0),
    ])
    latest = at(pit._build_ttm(inc, balance([])), "20240420")
    assert latest["end_date"] == pd.Timestamp("20240331")
    assert latest["net_profit_ttm"] == 80.0


def test_late_annual_dependency_recomputes_known_q1() -> None:
    inc = income([
        ("X", "20230331", "20230420", 10.0),
        ("X", "20240331", "20240420", 20.0),
        ("X", "20231231", "20240520", 70.0),
    ])
    events = pit._build_ttm(inc, balance([]))
    assert pd.isna(at(events, "20240420")["net_profit_ttm"])
    late = at(events, "20240520")
    assert late["end_date"] == pd.Timestamp("20240331")
    assert late["net_profit_ttm"] == 80.0


def test_lookups_do_not_cross_stock_codes() -> None:
    inc = income([
        ("X", "20231231", "20240320", 70.0),
        ("Y", "20231231", "20240320", 900.0),
        ("X", "20240331", "20240420", 20.0),
        ("Y", "20230331", "20230420", 200.0),
        ("Y", "20240331", "20240420", 250.0),
    ])
    events = pit._build_ttm(inc, balance([]))
    assert at(events, "20240320", "X")["net_profit_ttm"] == 70.0
    assert at(events, "20240420", "Y")["net_profit_ttm"] == 950.0
    assert pd.isna(at(events, "20240420", "X")["net_profit_ttm"])


def test_statement_loader_preserves_versions_and_uses_actual_date(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({
        "ts_code": ["X", "X"],
        "end_date": ["20230331", "20230331"],
        "ann_date": ["20230420", "20250510"],
        "f_ann_date": ["20230422", "20250509"],
        "report_type": ["1", "1"],
        "n_income_attr_p": [10.0, 30.0],
    })
    monkeypatch.setattr(pit, "read_optional", lambda *args: raw.copy())
    loaded = pit._load_statements("income", ["n_income_attr_p"])
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded["ann_date"].tolist() == [pd.Timestamp("20230422"), pd.Timestamp("20250510")]


def test_conflicting_same_day_version_blocks_then_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({
        "ts_code": ["X"] * 4,
        "end_date": ["20231231"] * 4,
        "ann_date": ["20240320", "20240420", "20240420", "20240520"],
        "f_ann_date": ["20240320", "20240420", "20240420", "20240520"],
        "report_type": ["1"] * 4,
        "n_income_attr_p": [70.0, 80.0, 90.0, 100.0],
    })
    monkeypatch.setattr(pit, "read_optional", lambda *args: raw.copy())
    loaded = pit._load_statements("income", ["n_income_attr_p"])
    assert loaded is not None
    assert len(loaded) == 3
    events = pit._build_ttm(loaded, balance([]))
    assert at(events, "20240320")["net_profit_ttm"] == 70.0
    assert pd.isna(at(events, "20240420")["net_profit_ttm"])
    assert at(events, "20240520")["net_profit_ttm"] == 100.0


def test_loader_rejects_non_quarter_end_and_early_announcement(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({
        "ts_code": ["X", "X", "X"],
        "end_date": ["20230331", "20230330", "20230630"],
        "ann_date": ["20230420", "20230420", "20230620"],
        "report_type": ["1", "1", "1"],
        "n_income_attr_p": [10.0, 20.0, 30.0],
    })
    monkeypatch.setattr(pit, "read_optional", lambda *args: raw.copy())
    loaded = pit._load_statements("income", ["n_income_attr_p"])
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.iloc[0]["end_date"] == pd.Timestamp("20230331")


def test_fina_loader_keeps_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({
        "ts_code": ["X", "X"], "end_date": ["20231231", "20231231"],
        "ann_date": ["20240320", "20240520"], "roe_yearly": [15.0, 20.0],
    })
    monkeypatch.setattr(pit, "read_optional", lambda *args: raw.copy())
    loaded = pit._load_fina()
    assert loaded is not None
    assert len(loaded) == 2


def test_build_events_filters_by_effective_date_and_keeps_ttm_without_bs(monkeypatch: pytest.MonkeyPatch) -> None:
    inc = income([
        ("X", "20231231", "20240320", 70.0),
        ("X", "20240331", "20240420", 20.0),
    ])
    monkeypatch.setattr(pit, "_load_fina", lambda: None)
    monkeypatch.setattr(pit, "_load_statements", lambda name, cols: inc.copy() if name == "income" else None)
    events = pit.build_events("20240101", "20240420")
    assert len(events) == 1
    assert events.iloc[0]["net_profit_ttm"] == 70.0
    assert events["roe_ttm"].isna().all()
    assert events["ann_date_eff"].le(pd.Timestamp("20240420")).all()


def test_fina_only_does_not_impersonate_ttm(monkeypatch: pytest.MonkeyPatch) -> None:
    fi = fina([("X", "20231231", "20240320", 15.0)])
    monkeypatch.setattr(pit, "_load_fina", lambda: fi.copy())
    monkeypatch.setattr(pit, "_load_statements", lambda name, cols: None)
    events = pit.build_events("20240101", "20241231")
    assert events["source"].eq("fina_only").all()
    assert events["roe_ttm"].isna().all()
    assert events["net_profit_ttm"].isna().all()


def test_panel_obeys_effective_day_and_preserves_audit_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({
        "date": pd.to_datetime(["20240320", "20240321", "20240321"]),
        "code": ["X", "X", "Y"],
    }).to_parquet(tmp_path / "universe.parquet", index=False)
    events = pit._build_ttm(income([("X", "20231231", "20240320", 70.0)]), balance([]))
    events["source"] = "ttm"
    monkeypatch.setattr(pit, "INTERIM", tmp_path)
    panel = pit.build_panel(events)
    assert panel is not None
    assert not panel.duplicated(["date", "code"]).any()
    assert {"ann_date_eff", "source", "input_max_ann_date"}.issubset(panel.columns)
    assert panel.loc[panel["date"].eq(pd.Timestamp("20240320")), "net_profit_ttm"].isna().all()
    available = panel["ann_date_eff"].notna()
    assert panel.loc[available, "ann_date_eff"].le(panel.loc[available, "date"]).all()
    assert panel.loc[panel["code"].eq("Y"), "net_profit_ttm"].isna().all()


def test_panel_does_not_regress_to_late_old_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"date": pd.to_datetime(["20241231"]), "code": ["X"]}).to_parquet(
        tmp_path / "universe.parquet", index=False
    )
    events = pd.DataFrame({
        "code": ["X", "X"],
        "end_date": pd.to_datetime(["20240930", "20231231"]),
        "ann_date": pd.to_datetime(["20241028", "20241128"]),
        "ann_date_eff": pd.to_datetime(["20241029", "20241129"]),
        "input_max_ann_date": pd.to_datetime(["20241028", "20241128"]),
        "source": ["ttm", "ttm"], "roe_ttm": [0.15, 0.05],
        "net_profit_ttm": [150.0, 50.0],
    })
    monkeypatch.setattr(pit, "INTERIM", tmp_path)
    panel = pit.build_panel(events)
    assert panel is not None
    assert panel.iloc[0]["end_date"] == pd.Timestamp("20240930")
    assert panel.iloc[0]["roe_ttm"] == 0.15


def test_panel_rejects_future_input_hidden_behind_past_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"date": pd.to_datetime(["20240321"]), "code": ["X"]}).to_parquet(
        tmp_path / "universe.parquet", index=False
    )
    events = pit._build_ttm(income([("X", "20231231", "20240320", 70.0)]), balance([]))
    events["source"] = "ttm"
    events["input_max_ann_date"] = pd.Timestamp("20240520")
    monkeypatch.setattr(pit, "INTERIM", tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        pit.build_panel(events)


@pytest.mark.parametrize("column,value", [
    ("input_max_ann_date", pd.NaT), ("end_date", pd.Timestamp("20261231")),
])
def test_panel_rejects_missing_audit_or_future_report(tmp_path, monkeypatch, column, value):
    pd.DataFrame({"date": pd.to_datetime(["20240321"]), "code": ["X"]}).to_parquet(
        tmp_path / "universe.parquet", index=False,
    )
    events = pit._build_ttm(income([("X", "20231231", "20240320", 70.0)]), balance([]))
    events[column] = value
    monkeypatch.setattr(pit, "INTERIM", tmp_path)
    with pytest.raises(RuntimeError):
        pit.build_panel(events)
