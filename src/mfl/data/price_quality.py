"""价格质量规则：显式基准校正、未确认公司行动隔离与窗口检查。

只处理config/price_quality.yaml列出的事件，不按收益大小自动修数据。
normalize为本地连续性推断，原始因子始终保存在raw中。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import ROOT

RULES_PATH = ROOT / "config" / "price_quality.yaml"


def read_rules(path: Path | None = None) -> list[dict]:
    """读取可审阅的事件清单；缺文件或非法规则拒绝静默忽略。"""
    path = RULES_PATH if path is None else path
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or content.get("version") != 1:
        raise ValueError("价格质量规则版本无效")
    rules = content.get("events")
    if not isinstance(rules, list):
        raise ValueError("价格质量规则events须为列表")
    keys, ids = set(), set()
    for rule in rules:
        required = {"id", "code", "date", "previous_date", "expected_previous_factor",
                    "expected_factor", "action", "status", "reason", "evidence"}
        if not isinstance(rule, dict) or not required.issubset(rule):
            raise ValueError("价格质量规则字段缺失")
        if rule["action"] not in {"normalize", "quarantine"}:
            raise ValueError(f"未知价格质量动作：{rule['action']}")
        for name in ("date", "previous_date"):
            value = rule[name]
            if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
                raise ValueError("价格质量日期须为YYYYMMDD字符串")
            pd.to_datetime(value, format="%Y%m%d", errors="raise")
        if rule["previous_date"] >= rule["date"]:
            raise ValueError("价格质量事件的前序日期须早于事件日")
        for name in ("expected_previous_factor", "expected_factor"):
            value = float(rule[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("价格质量因子锚点须为有效正数")
        key = (rule["code"], rule["date"])
        if key in keys or rule["id"] in ids:
            raise ValueError("价格质量规则主键或id重复")
        keys.add(key)
        ids.add(rule["id"])
        if not rule["reason"] or not rule["evidence"] or not rule["status"]:
            raise ValueError("价格质量规则须保留证据、状态及原因")
    return sorted(rules, key=lambda r: (r["date"], r["code"]))


def validate_anchors(rules: list[dict], raw: Path) -> None:
    """核验raw事件及最近前序报价，防止旧规则误用于已更新的输入。"""
    if not rules:
        return
    daily_parts, adj_parts = [], []
    for year in sorted({d[:4] for r in rules for d in (r["date"], r["previous_date"])}):
        codes = sorted({r["code"] for r in rules if year in (r["date"][:4], r["previous_date"][:4])})
        filters = [("ts_code", "in", codes)]
        daily_parts.append(pd.read_parquet(
            raw / "daily" / f"year={year}" / "data.parquet",
            columns=["ts_code", "trade_date", "close", "pre_close"], filters=filters,
        ))
        adj_parts.append(pd.read_parquet(
            raw / "adj_factor" / f"year={year}" / "data.parquet", filters=filters,
        ))
    daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates()
    adj = pd.concat(adj_parts, ignore_index=True).drop_duplicates()
    work = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    work = work.sort_values(["ts_code", "trade_date"])
    group = work.groupby("ts_code", observed=True)
    for col in ("trade_date", "close", "adj_factor"):
        work[f"previous_{col}"] = group[col].shift()
    lookup = work.set_index(["ts_code", "trade_date"])
    for rule in rules:
        key = (rule["code"], rule["date"])
        if key not in lookup.index:
            raise ValueError(f"质量规则锚点缺失，须重新核验：{rule['id']}")
        row = lookup.loc[key]
        same = row["previous_trade_date"] == rule["previous_date"]
        same &= bool(np.isclose(row["adj_factor"], rule["expected_factor"], rtol=0, atol=1e-9))
        same &= bool(np.isclose(row["previous_adj_factor"], rule["expected_previous_factor"], rtol=0, atol=1e-9))
        if rule["action"] == "normalize":
            same &= bool(np.isfinite(row["close"]) and row["close"] > 0)
            same &= bool(np.isclose(row["pre_close"], row["previous_close"], rtol=0, atol=1e-9))
        if not same:
            raise ValueError(f"质量规则与raw锚点不符，须重新核验：{rule['id']}")


def load_validated_rules(raw: Path, codes: pd.Series) -> list[dict]:
    """只核验当前股票涉及的规则；局部区间也检查完整本地边界。"""
    available = set(codes)
    rules = [r for r in read_rules() if r["code"] in available]
    validate_anchors(rules, raw)
    return rules


def apply_factor_rules(frame: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """按边界累计倍率校正因子；仅隔离未确认事件当日，不填缺失因子。

    每个普通边界从当日起乘previous_raw_factor/current_raw_factor，
    多个边界累计相乘，其他日期相对倍率（真实分红送转）保持不变。
    输入与输出均为raw字段名，输出因子是加工后的有效因子。
    """
    result = frame.copy()
    for rule in sorted(rules, key=lambda r: (r["date"], r["code"])):
        stock = result["ts_code"].eq(rule["code"])
        if rule["action"] == "normalize":
            mask = stock & result["trade_date"].ge(rule["date"])
            scale = rule["expected_previous_factor"] / rule["expected_factor"]
            result.loc[mask, "adj_factor"] *= scale
        else:
            mask = stock & result["trade_date"].eq(rule["date"])
            result.loc[mask, "adj_factor"] = np.nan
    return result


def price_window_validity(windows: pd.DataFrame, rules: list[dict] | None = None) -> pd.Series:
    """判断价格窗口是否触及未确认事件；列须为code/start/end。

    后续任何用两端价格计算跨期收益或价格窗口的消费者都须调用。
    单行NaN不足以阻止只读端点的计算，因此显式拒绝start<=事件日<=end。
    此函数只检查已登记的未解决事件，不保证窗口没有其他数据缺口。
    """
    rules = read_rules() if rules is None else rules
    start, end = pd.to_datetime(windows["start"]), pd.to_datetime(windows["end"])
    if start.isna().any() or end.isna().any() or start.gt(end).any():
        raise ValueError("价格窗口日期缺失或起止顺序错误")
    valid = pd.Series(True, index=windows.index)
    for rule in rules:
        if rule["action"] == "quarantine":
            date = pd.to_datetime(rule["date"], format="%Y%m%d")
            valid &= ~(windows["code"].eq(rule["code"]) & start.le(date) & end.ge(date))
    return valid
