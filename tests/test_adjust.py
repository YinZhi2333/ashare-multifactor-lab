"""复权行情的缺口、单位、输出列和区间边界回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import adjust


def _daily(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["X"] * len(dates), "trade_date": dates,
        "close": closes, "open": closes, "high": closes, "low": closes,
        "pre_close": [np.nan] + closes[:-1], "vol": [100.] * len(dates),
        "amount": [100.] * len(dates), "pct_chg": [0.] * len(dates),
    })


def _mock_raw(monkeypatch, tmp_path, daily, factors, basic=None):
    monkeypatch.setattr(adjust, "RAW", tmp_path)
    adj = daily[["ts_code", "trade_date"]].assign(adj_factor=factors)

    def reader(table, start, end):
        if table == "daily":
            df = daily
        elif table == "adj_factor":
            df = adj
        elif basic is not None:
            df = basic
        else:
            raise FileNotFoundError("合成用例不含 daily_basic")
        return df.loc[df.trade_date.between(start, end)].copy()

    monkeypatch.setattr(adjust, "read_years", reader)


def test_missing_adjustment_keeps_rows_and_does_not_bridge(monkeypatch, tmp_path):
    daily = _daily(["20240102", "20240103", "20240104", "20240105"], [10., 11., 12., 15.])
    _mock_raw(monkeypatch, tmp_path, daily, [1., np.nan, 1., 1.])
    out = adjust.build_prices("20240101", "20240131")
    assert len(out) == 4
    assert out.ret.iloc[:3].isna().all()
    assert out.ret.iloc[3] == pytest.approx(.25)
    assert out.close.iloc[1] == 11.


@pytest.mark.parametrize("factor", [0., -1., np.inf])
def test_invalid_adjustment_is_not_a_return(monkeypatch, tmp_path, factor):
    daily = _daily(["20240102", "20240103", "20240104"], [10., 11., 12.])
    _mock_raw(monkeypatch, tmp_path, daily, [1., factor, 1.])
    out = adjust.build_prices("20240101", "20240131")
    assert out.ret.isna().all()
    assert pd.isna(out.adj_factor.iloc[1])


def test_missing_basic_keeps_schema_and_vwap_unit(monkeypatch, tmp_path):
    daily = _daily(["20240102", "20240103"], [10., 11.])
    _mock_raw(monkeypatch, tmp_path, daily, [2., 2.])
    out = adjust.build_prices("20240101", "20240131")
    assert out.columns.tolist() == adjust.PRICE_COLS
    assert out[adjust.BASIC_COLS[2:]].isna().all().all()
    assert out.vwap.tolist() == [10., 10.]
    assert out.ret.iloc[1] == pytest.approx(.1)
    assert not out.duplicated(["date", "code"]).any()


def test_partial_basic_columns_are_retained(monkeypatch, tmp_path):
    daily = _daily(["20240102"], [10.])
    basic = daily[["ts_code", "trade_date"]].assign(total_mv=123.)
    _mock_raw(monkeypatch, tmp_path, daily, [1.], basic)
    out = adjust.build_prices("20240101", "20240131")
    assert out.total_mv.iloc[0] == 123.
    assert out.turnover_rate.isna().all()


def _write_history(tmp_path, date, close, factor):
    for table, data in [
        ("daily", pd.DataFrame({"ts_code": ["X"], "trade_date": [date], "close": [close]})),
        ("adj_factor", pd.DataFrame({"ts_code": ["X"], "trade_date": [date], "adj_factor": [factor]})),
    ]:
        folder = tmp_path / table / f"year={date[:4]}"
        folder.mkdir(parents=True, exist_ok=True)
        data.to_parquet(folder / "data.parquet", index=False)


def test_start_return_uses_local_previous_year(monkeypatch, tmp_path):
    daily = _daily(["20240102"], [11.])
    _mock_raw(monkeypatch, tmp_path, daily, [2.])
    _write_history(tmp_path, "20231229", 10., 2.)
    out = adjust.build_prices("20240101", "20240131")
    assert out.ret.iloc[0] == pytest.approx(.1)
    assert len(out) == 1


def test_start_does_not_skip_latest_missing_history(monkeypatch, tmp_path):
    daily = _daily(["20240102"], [11.])
    _mock_raw(monkeypatch, tmp_path, daily, [2.])
    _write_history(tmp_path, "20231229", 10., np.nan)
    _write_history(tmp_path, "20221230", 5., 2.)
    out = adjust.build_prices("20240101", "20240131")
    assert pd.isna(out.ret.iloc[0])


def test_no_daily_between_observations_preserves_resumption_return(monkeypatch, tmp_path):
    daily = _daily(["20240102", "20240108"], [10., 11.])
    _mock_raw(monkeypatch, tmp_path, daily, [1., 1.])
    out = adjust.build_prices("20240101", "20240131")
    assert out.ret.iloc[1] == pytest.approx(.1)


def test_conflicting_duplicate_is_rejected():
    df = pd.DataFrame({"ts_code": ["X", "X"], "trade_date": ["20240102"] * 2, "close": [10., 11.]})
    with pytest.raises(ValueError, match="主键冲突"):
        adjust._dedupe(df, ["ts_code", "trade_date"])
    assert len(adjust._dedupe(pd.concat([df.iloc[:1]] * 2), ["ts_code", "trade_date"])) == 1


def test_index_computes_return_before_clipping(monkeypatch):
    raw = pd.DataFrame({"trade_date": ["20231229", "20240102"], "close": [100., 110.]})
    monkeypatch.setattr(adjust, "read_single", lambda *args: raw.copy())
    written = []
    monkeypatch.setattr(adjust, "save_interim", lambda df, filename: written.append((filename, df)))
    out = adjust.build_index("20240101", "20240131")
    assert len(out) == 1
    assert out.ret.iloc[0] == pytest.approx(.1)
    assert written[0][0] == "index.parquet"


def test_pearson_handles_missing_values_and_constant_series():
    assert adjust._pearson(pd.Series([1., 2., 3., np.nan]), pd.Series([1., 3., 2., 5.])) == pytest.approx(.5)
    assert np.isnan(adjust._pearson(pd.Series([1., 1.]), pd.Series([2., 3.])))
    assert np.isnan(adjust._pearson(pd.Series([1.]), pd.Series([2.])))


def test_streamed_prices_are_one_sorted_file_with_fixed_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(adjust, "INTERIM", tmp_path)
    calls = []

    def build(start, end):
        calls.append((start, end))
        frame = pd.DataFrame(np.nan, index=[0], columns=adjust.PRICE_COLS)
        frame["date"] = pd.to_datetime([end])
        frame["code"] = ["X"]
        frame["close"] = [10.]
        frame["ret"] = [np.nan if start[:4] == "2023" else .1]
        return frame

    monkeypatch.setattr(adjust, "build_prices", build)
    path = adjust.write_prices("20231229", "20240105")
    out = pd.read_parquet(path)
    assert calls == [("20231229", "20231231"), ("20240101", "20240105")]
    assert len(out) == 2
    assert out.columns.tolist() == adjust.PRICE_COLS
    assert out.ret.iloc[1] == pytest.approx(.1)
    assert out.date.is_monotonic_increasing
    assert not out.duplicated(["date", "code"]).any()
    assert not list(tmp_path.glob("*.tmp"))


def test_stream_failure_preserves_previous_output(monkeypatch, tmp_path):
    monkeypatch.setattr(adjust, "INTERIM", tmp_path)
    path = tmp_path / "prices.parquet"
    path.write_bytes(b"previous-complete-output")

    def fail(start, end):
        raise RuntimeError("合成加工失败")

    monkeypatch.setattr(adjust, "build_prices", fail)
    with pytest.raises(RuntimeError, match="合成加工失败"):
        adjust.write_prices("20240101", "20240105")
    assert path.read_bytes() == b"previous-complete-output"
    assert not list(tmp_path.glob("*.tmp"))
