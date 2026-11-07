"""只读排查行情与复权因子的缺口，并生成本地审计清单。

不导入抓数入口、不请求接口、不修改 raw/interim；所有输出位于 results。
成分身份用不晚于当日的最近月度快照近似，不冒充逐日精确调整记录。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mfl" / "data"))
from common import RAW, INTERIM, ROOT, setup_logging
from universe import attach_names

LOG = setup_logging("audit_adj_factor")
OUTPUT = ROOT / "results" / "adj_factor_audit"


def raw_manifest() -> dict:
    """逐文件内容哈希，证明只读排查没有改变原始数据。"""
    result = {}
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            result[path.relative_to(RAW).as_posix()] = digest.hexdigest()
    return result


def membership_at(frame: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """按最近月度快照添加成分身份，缺少更早快照时保持未知。"""
    right = weights[["snapshot_date"]].drop_duplicates().sort_values("snapshot_date")
    out = pd.merge_asof(frame.sort_values("date"), right, left_on="date", right_on="snapshot_date", direction="backward")
    marks = weights[["snapshot_date", "code"]].drop_duplicates().assign(_member=True)
    out = out.merge(marks, on=["snapshot_date", "code"], how="left", validate="many_to_one")
    out["member_at_latest_snapshot"] = out["_member"].fillna(False).astype("boolean")
    out.loc[out["snapshot_date"].isna(), "member_at_latest_snapshot"] = pd.NA
    return out.drop(columns="_member")


def main() -> None:
    before = raw_manifest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = pd.read_parquet(RAW / "index_weight" / "index_code=000905.SH" / "data.parquet")
    weights = weights.rename(columns={"con_code": "code"})
    weights["snapshot_date"] = pd.to_datetime(weights["trade_date"], format="%Y%m%d").astype("datetime64[ns]")
    members = set(weights["code"])
    missing_frames, date_frames, member_factor_frames = [], [], []
    total_daily = total_adj = raw_invalid = 0
    for path in sorted((RAW / "daily").glob("year=*/data.parquet")):
        daily = pd.read_parquet(path, columns=["ts_code", "trade_date"])
        factor_path = RAW / "adj_factor" / path.parent.name / "data.parquet"
        factors = pd.read_parquet(factor_path) if factor_path.exists() else pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        total_daily += len(daily)
        total_adj += len(factors)
        raw_invalid += int((factors["adj_factor"].isna() | ~np.isfinite(factors["adj_factor"]) | factors["adj_factor"].le(0)).sum())
        joined = daily.merge(factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one", indicator=True)
        bad = joined["adj_factor"].isna() | ~np.isfinite(joined["adj_factor"]) | joined["adj_factor"].le(0)
        missing = joined.loc[bad, ["ts_code", "trade_date", "_merge"]].copy()
        missing["reason"] = np.where(missing.pop("_merge").eq("left_only"), "no_matching_factor_row", "invalid_factor_value")
        missing_frames.append(missing)
        per_date = missing.groupby("trade_date").size().rename("missing_rows").to_frame()
        per_date["daily_rows"] = daily.groupby("trade_date").size()
        per_date["factor_rows"] = factors.groupby("trade_date").size().reindex(per_date.index).fillna(0).astype(int)
        per_date["historical_member_missing"] = missing[missing["ts_code"].isin(members)].groupby("trade_date").size().reindex(per_date.index).fillna(0).astype(int)
        date_frames.append(per_date.reset_index())
        member_factor_frames.append(factors.loc[factors["ts_code"].isin(members)])
        LOG.info("%s：daily %d 行，缺复权因子 %d 行", path.parent.name, len(daily), len(missing))
    missing = pd.concat(missing_frames, ignore_index=True).rename(columns={"ts_code": "code"})
    missing["date"] = pd.to_datetime(missing.pop("trade_date"), format="%Y%m%d").astype("datetime64[ns]")
    missing["historical_member"] = missing["code"].isin(members)
    date_summary = pd.concat(date_frames, ignore_index=True)
    affected = membership_at(missing.loc[missing["historical_member"]].copy(), weights)
    basic = pd.read_parquet(RAW / "stock_basic" / "data.parquet")
    affected = attach_names(affected, basic, pd.read_parquet(RAW / "namechange" / "data.parquet"))
    affected["list_date"] = affected["code"].map(basic.drop_duplicates("ts_code").set_index("ts_code")["list_date"])
    affected["latest_name_reference"] = affected["code"].map(basic.drop_duplicates("ts_code").set_index("ts_code")["name"])
    factor_history = pd.concat(member_factor_frames, ignore_index=True).rename(columns={"ts_code": "code"})
    factor_history["factor_date"] = pd.to_datetime(factor_history.pop("trade_date"), format="%Y%m%d").astype("datetime64[ns]")
    valid = factor_history["adj_factor"].gt(0) & np.isfinite(factor_history["adj_factor"])
    factor_history = factor_history.loc[valid].sort_values("factor_date")
    for direction, prefix in [("backward", "previous"), ("forward", "next")]:
        source = factor_history.rename(columns={"factor_date": f"{prefix}_factor_date", "adj_factor": f"{prefix}_factor"})
        affected = pd.merge_asof(affected.sort_values("date"), source, by="code", left_on="date", right_on=f"{prefix}_factor_date", direction=direction, allow_exact_matches=False)

    # 仅审计已有加工数据，不重建价格、不实现因子或组合回测。
    px = pd.read_parquet(INTERIM / "prices.parquet", columns=["date", "code", "adj_factor", "close_adj", "ret"], filters=[("code", "in", sorted(members))])
    px["date"] = px["date"].astype("datetime64[ns]")
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px["current_factor_missing"] = px["adj_factor"].isna()
    px["previous_observation_factor_missing"] = px.groupby("code", observed=True)["current_factor_missing"].shift(1).fillna(False)
    blocked = px["current_factor_missing"] | px["previous_observation_factor_missing"]
    assert px.loc[blocked, "ret"].isna().all(), "缺口两侧产生了收益"
    assert px.loc[px["current_factor_missing"], "close_adj"].isna().all()
    raw_keys = pd.MultiIndex.from_frame(affected[["date", "code"]])
    interim_keys = pd.MultiIndex.from_frame(px.loc[px["current_factor_missing"], ["date", "code"]])
    assert raw_keys.sort_values().equals(interim_keys.sort_values()), "raw缺口与interim不一致"
    all_interim_missing = pd.read_parquet(
        INTERIM / "prices.parquet", columns=["date", "code", "close_adj", "ret"],
        filters=ds.field("adj_factor").is_null(),
    )
    global_keys = pd.MultiIndex.from_frame(missing[["date", "code"]])
    assert global_keys.sort_values().equals(pd.MultiIndex.from_frame(all_interim_missing[["date", "code"]]).sort_values())
    assert all_interim_missing[["close_adj", "ret"]].isna().all().all()
    gaps = membership_at(px.loc[blocked].copy(), weights)
    gaps["cause"] = np.where(gaps["current_factor_missing"], "current_factor_missing", "previous_observation_factor_missing")
    successors = gaps.loc[~gaps["current_factor_missing"], ["code", "date"]].rename(columns={"date": "next_observation_return_missing_date"})
    affected = affected.merge(successors, on="code", how="left", validate="one_to_one")
    calendar = pd.read_parquet(RAW / "trade_cal" / "data.parquet")
    open_days = pd.to_datetime(calendar.loc[pd.to_numeric(calendar["is_open"]).eq(1), "cal_date"], format="%Y%m%d").astype("datetime64[ns]")
    gap_daily = gaps.groupby("date").agg(stocks=("code", "nunique"), latest_snapshot_members=("member_at_latest_snapshot", "sum")).reset_index()
    # 对当前缺口的相邻有效观测，只计算跨缺口区间的可核查涨跌，不当成单日收益。
    valid_prices = px.loc[px["close_adj"].notna(), ["code", "date", "close_adj"]]
    for direction, prefix in [("backward", "previous"), ("forward", "next")]:
        source = valid_prices.rename(columns={"date": f"{prefix}_quote_date", "close_adj": f"{prefix}_close_adj"}).sort_values(f"{prefix}_quote_date")
        affected = pd.merge_asof(affected.sort_values("date"), source, by="code", left_on="date", right_on=f"{prefix}_quote_date", direction=direction, allow_exact_matches=False)
    affected["known_endpoint_interval_return"] = affected["next_close_adj"] / affected["previous_close_adj"] - 1
    uni = pd.read_parquet(INTERIM / "universe.parquet", columns=["date", "code"])
    monthend_missing = affected.merge(uni, on=["date", "code"], how="inner")
    summary = {
        "daily_rows": total_daily, "factor_rows": total_adj, "missing_rows": len(missing),
        "missing_rate": len(missing) / total_daily, "missing_stocks": int(missing["code"].nunique()),
        "missing_dates": int(missing["date"].nunique()), "raw_invalid_factor_values": raw_invalid,
        "reason_counts": missing["reason"].value_counts().to_dict(),
        "exchange_counts": missing["code"].str.split(".").str[-1].value_counts().to_dict(),
        "historical_member_missing_rows": len(affected), "historical_member_missing_stocks": int(affected["code"].nunique()),
        "historical_member_missing_dates": sorted(affected["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "contemporaneous_snapshot_member_missing_rows": int(affected["member_at_latest_snapshot"].fillna(False).sum()),
        "member_returns_blocked_rows": len(gaps), "member_returns_blocked_stocks": int(gaps["code"].nunique()),
        "member_return_gap_dates": gap_daily.assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        "member_monthend_missing_rows": len(monthend_missing),
        "interim_global_factor_missing_rows": len(all_interim_missing),
        "other_missing_dates_exchange_counts": missing.loc[~missing["date"].isin(affected["date"]), "code"].str.split(".").str[-1].value_counts().to_dict(),
        "neighbor_factors_both_available": int(affected[["previous_factor", "next_factor"]].notna().all(axis=1).sum()),
        "neighbor_factors_equal": int(affected["previous_factor"].eq(affected["next_factor"]).sum()),
        "known_interval_return_nonzero": int(affected["known_endpoint_interval_return"].abs().gt(1e-8).sum()),
        "known_interval_return_median": float(affected["known_endpoint_interval_return"].median()),
        "open_calendar_missing_dates": sorted(set(affected["date"]) - set(open_days)),
        "daily_dates_without_any_factor_rows": date_summary.loc[date_summary["factor_rows"].eq(0), "trade_date"].tolist(),
        "factor_gap_blocking_passed": True,
    }
    missing.sort_values(["date", "code"]).to_csv(OUTPUT / "all_missing_pairs.csv", index=False, encoding="utf-8-sig")
    date_summary.sort_values("trade_date").to_csv(OUTPUT / "missing_dates_summary.csv", index=False, encoding="utf-8-sig")
    affected.sort_values(["date", "code"]).to_csv(OUTPUT / "historical_member_missing_pairs.csv", index=False, encoding="utf-8-sig")
    affected.loc[affected["member_at_latest_snapshot"].fillna(False)].sort_values(["date", "code"]).to_csv(OUTPUT / "latest_snapshot_member_missing_pairs.csv", index=False, encoding="utf-8-sig")
    gaps.sort_values(["date", "code"]).to_csv(OUTPUT / "historical_member_return_gaps.csv", index=False, encoding="utf-8-sig")
    missing.groupby("code").agg(missing_rows=("date", "size"), first_missing=("date", "min"), last_missing=("date", "max"), historical_member=("historical_member", "any")).to_csv(OUTPUT / "missing_stocks_summary.csv", encoding="utf-8-sig")
    after = raw_manifest()
    assert before == after, "审计期间raw发生变化"
    summary["raw_files_unchanged"] = len(after)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUTPUT / "raw_manifest.json").write_text(json.dumps(after, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
