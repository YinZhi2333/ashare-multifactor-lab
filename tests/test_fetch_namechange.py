"""名称抓取回归用例：模拟接口和临时文件，不依赖真实 raw 或 token。

范围白名单、清理和网络响应均使用临时文件/模拟对象验证。
"""

from __future__ import annotations

import pandas as pd
import pytest

import fetch


def names(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=fetch.NAMECHANGE_COLS)


class FakeClient:
    def __init__(self, responses: dict[str, pd.DataFrame | Exception]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict]] = []

    def call(self, api: str, **params) -> pd.DataFrame:
        self.requests.append((api, params))
        response = self.responses[params["ts_code"]]
        if isinstance(response, Exception):
            raise response
        return response.copy()


def target_rows(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": codes, "date": pd.to_datetime(["20150130"] * len(codes)).astype("datetime64[ns]")})


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(fetch, "_namechange_members", lambda: {"X", "Y"})


def test_full_history_request_and_single_file_merge(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    old = names([("X", "原名", "20190101", None, "20181220", "其他")])
    old.to_parquet(path, index=False)
    incoming = names([
        ("X", "早期名", "20000101", "20181231", None, "其他"),
        ("X", "原名", "20190101", None, "20181220", "其他"),
        ("X", "原名", "20210101", None, "20181220", "其他"),
    ])
    client = FakeClient({"X": incoming})
    fetch.fetch_namechange(client, "20150101", "20241231", codes=["X"])
    out = pd.read_parquet(path)
    assert len(out) == 3  # 同名同公告但不同区间必须保留。
    assert out.columns.tolist() == fetch.NAMECHANGE_COLS
    assert len(out.merge(old, on=fetch.NAMECHANGE_COLS)) == 1
    assert client.requests == [("namechange", {"ts_code": "X", "fields": ",".join(fetch.NAMECHANGE_COLS)})]
    assert list(path.parent.iterdir()) == [path]


def test_default_only_queries_research_gaps(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "_namechange_targets", lambda *args: target_rows(["X", "Y"]))
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    names([("X", "已覆盖", "20000101", None, "20000101", "其他")]).to_parquet(path, index=False)
    client = FakeClient({"Y": names([("Y", "补齐", "20000101", None, "20000101", "其他")])})
    fetch.fetch_namechange(client, "20150101", "20241231")
    assert [params["ts_code"] for _, params in client.requests] == ["Y"]
    # 同一区间重跑不以请求次数为准，而根据实际研究日期是否已覆盖跳过。
    fetch.fetch_namechange(client, "20150101", "20241231")
    assert len(client.requests) == 1


def test_gap_detection_includes_late_expired_and_conflicting_names():
    old = names([
        ("LATE", "后来名", "20160101", None, "20160101", "其他"),
        ("ENDED", "旧名", "20000101", "20141231", "20000101", "其他"),
        ("CONFLICT", "甲", "20000101", None, "20000101", "其他"),
        ("CONFLICT", "乙", "20000101", None, "20000101", "其他"),
    ])
    assert fetch._namechange_gaps(target_rows(["LATE", "ENDED", "CONFLICT", "MISSING"]), old) == ["CONFLICT", "ENDED", "LATE", "MISSING"]


@pytest.mark.parametrize("force", [False, True])
def test_empty_response_never_discards_old_file(tmp_path, monkeypatch, force, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    names([("X", "已有", "20000101", None, "20000101", "其他")]).to_parquet(path, index=False)
    before = path.read_bytes()
    fetch.fetch_namechange(FakeClient({"X": names([])}), "20150101", "20241231", force=force, codes=["X"])
    assert path.read_bytes() == before


def test_successful_stock_is_saved_before_later_permission_failure(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    client = FakeClient({
        "X": names([("X", "已返回", "20000101", None, "20000101", "其他")]),
        "Y": PermissionError("合成权限失败"),
    })
    with pytest.raises(PermissionError):
        fetch.fetch_namechange(client, "20150101", "20241231", codes=["X", "Y"])
    out = pd.read_parquet(tmp_path / "namechange" / "data.parquet")
    assert out.ts_code.tolist() == ["X"]


def test_name_command_accepts_explicit_small_sample():
    args = fetch.build_parser().parse_args(["names", "--codes", "600848.SH", "000001.SZ"])
    assert args.codes == ["600848.SH", "000001.SZ"]


def test_no_local_pool_does_not_default_to_all_stocks(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "INTERIM", tmp_path / "interim")
    with pytest.raises(FileNotFoundError, match="先抓 index"):
        fetch._namechange_targets("20150101", "20241231")


def test_whitelist_is_complete_historical_union_not_current_members(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    path = tmp_path / "index_weight" / f"index_code={fetch.INDEX_CODE}" / "data.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"con_code": ["X", "X", "Y"], "trade_date": ["20150130", "20241231", "20150130"]}).to_parquet(path, index=False)
    assert fetch._namechange_members() == {"X", "Y"}


def test_names_preserves_all_member_history_and_skips_covered_stocks(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "_namechange_targets", lambda *args: target_rows(["X", "Y"]))
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    member_history = names([
        ("X", "早期名", "19900101", "19991231", "19900101", "其他"),
        ("X", "当前名", "20000101", None, "20000101", "其他"),
        ("Y", "另一成员", "20000101", None, "20000101", "其他"),
    ])
    pd.concat([member_history, names([("OUT", "池外名", "20000101", None, "20000101", "其他")])]).to_parquet(path, index=False)
    client = FakeClient({})
    fetch.fetch_namechange(client, "20150101", "20241231")
    pd.testing.assert_frame_equal(pd.read_parquet(path), member_history)
    before = path.read_bytes()
    fetch.fetch_namechange(client, "20150101", "20241231")
    assert client.requests == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("force", [False, True])
def test_outside_code_is_rejected_without_request_or_write(tmp_path, monkeypatch, pool, force):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    names([("OUT", "尚未清理", "20000101", None, "20000101", "其他")]).to_parquet(path, index=False)
    before = path.read_bytes()
    client = FakeClient({})
    with pytest.raises(ValueError, match="池外股票"):
        fetch.fetch_namechange(client, "20150101", "20241231", force=force, codes=["OUT"])
    assert client.requests == []
    assert path.read_bytes() == before


def test_default_cleans_extras_even_when_no_download_is_needed(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "_namechange_targets", lambda *args: target_rows(["X"]))
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    # 请求只涉及X，也必须保留白名单内Y的整条历史。
    retained = names([
        ("X", "已覆盖", "20000101", None, "20000101", "其他"),
        ("Y", "更早成员", "19900101", "19991231", "19900101", "其他"),
    ])
    pd.concat([retained, names([("OUT", "池外", "20000101", None, "20000101", "其他")])]).to_parquet(path, index=False)
    client = FakeClient({})
    fetch.fetch_namechange(client, "20150101", "20241231")
    assert client.requests == []
    pd.testing.assert_frame_equal(pd.read_parquet(path), retained)


def test_missing_whitelist_never_cleans_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    path = tmp_path / "namechange" / "data.parquet"
    path.parent.mkdir()
    names([("X", "原记录", "20000101", None, "20000101", "其他")]).to_parquet(path, index=False)
    before = path.read_bytes()
    client = FakeClient({})
    with pytest.raises(FileNotFoundError, match="历史名单"):
        fetch.fetch_namechange(client, "20150101", "20241231", codes=["X"])
    assert client.requests == []
    assert path.read_bytes() == before


def test_response_for_other_stock_cannot_enter_file(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    client = FakeClient({"X": names([("OUT", "错误响应", "20000101", None, "20000101", "其他")])})
    with pytest.raises(ValueError, match="混入其他股票"):
        fetch.fetch_namechange(client, "20150101", "20241231", codes=["X"])
    assert not (tmp_path / "namechange" / "data.parquet").exists()


def test_removed_clean_only_option_is_rejected_before_client_init(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("参数错误时不能初始化客户端")
    monkeypatch.setattr(fetch, "Client", forbidden)
    with pytest.raises(SystemExit) as exc:
        fetch.main(["names", "--clean-only"])
    assert exc.value.code == 2
