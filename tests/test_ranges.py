"""研究起点与数据预热分离，防止默认加工丢失2015年行情。"""

from __future__ import annotations

import sys

import pandas as pd
import pytest
import yaml

import adjust
import common
import pit
import universe


def _config(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({"start": "2016-01-01", "end": "2024-12-31", "data_start": "2015-01-01"}), encoding="utf-8")
    return path


def test_default_research_and_data_ranges_are_separate(tmp_path):
    path = _config(tmp_path)
    assert common.resolve_range(None, None, config_path=path) == ("20160101", "20241231")
    assert common.resolve_range(None, None, data=True, config_path=path) == ("20150101", "20241231")


def test_explicit_range_has_priority_and_invalid_range_is_rejected(tmp_path):
    path = _config(tmp_path)
    assert common.resolve_range("2020-01-01", "20211231", config_path=path) == ("20200101", "20211231")
    assert common.resolve_range("20200101", None, data=True, config_path=path)[0] == "20200101"
    with pytest.raises(ValueError, match="start"):
        common.resolve_range("20250101", "20241231", config_path=path)


def test_missing_configuration_keeps_safe_prewarm_default(tmp_path):
    path = tmp_path / "absent.yaml"
    assert common.resolve_range(None, None, config_path=path)[0] == "20160101"
    assert common.resolve_range(None, None, data=True, config_path=path)[0] == "20150101"


@pytest.mark.parametrize("module", [adjust, universe, pit])
def test_data_cli_uses_prewarm_range_without_real_io(module, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "BASE_CONFIG", _config(tmp_path))
    monkeypatch.setattr(sys, "argv", [module.__name__])
    calls = []

    def record(start, end):
        calls.append((start, end))
        return pd.DataFrame({"date": pd.to_datetime(["2015-01-30"]), "code": ["X"]})

    if module is adjust:
        monkeypatch.setattr(module, "write_prices", record)
        monkeypatch.setattr(module, "build_index", lambda *args: None)
    elif module is universe:
        monkeypatch.setattr(module, "build_universe", record)
        monkeypatch.setattr(module, "save_interim", lambda *args: tmp_path / "unused.parquet")
    else:
        monkeypatch.setattr(module, "build_events", record)
        monkeypatch.setattr(module, "build_panel", lambda *args: None)
        monkeypatch.setattr(module, "save_interim", lambda *args: tmp_path / "unused.parquet")
    module.main()
    assert calls == [("20150101", "20241231")]

