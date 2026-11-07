"""数据层公共工具：路径、配置、读写、日志。

约定：加工模块只读 raw，interim 可重建；只有 fetch 可以写入 raw。
raw 日期通常是 YYYYMMDD 字符串，加工表的 date 是 pandas 时间戳，
股票代码在加工层统一为 code。所有加工脚本从这里取路径。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

# 从 common.py 向上定位仓库根目录，路径不依赖终端当前所在目录。
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
BASE_CONFIG = ROOT / "config" / "base.yaml"


def load_config(path: Path | None = None) -> dict:
    """读取公共 YAML 配置；文件缺失或内容为空时返回空字典。

    此处只读 base.yaml，不读取含 token 的 local.yaml。
    """
    path = BASE_CONFIG if path is None else path
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def resolve_range(start: str | None, end: str | None, *, data: bool = False,
                  config_path: Path | None = None) -> tuple[str, str]:
    """按CLI → base.yaml → 默认值解析研究或含预热的数据区间。

    默认研究为2016—2024年；数据加工传data=True，从data_start（默认2015）
    开始保留预热，不能因研究起点后移而丢掉历史。显式CLI参数仍优先。
    config_path用于读取同一输入根目录配置；只读公共配置，不读取local.yaml。
    """
    cfg = load_config(config_path)
    key, fallback = ("data_start", "20150101") if data else ("start", "20160101")
    s = start or str(cfg.get(key, fallback))
    e = end or str(cfg.get("end", "20241231"))
    s, e = s.replace("-", ""), e.replace("-", "")
    lo, hi = pd.to_datetime(s, format="%Y%m%d"), pd.to_datetime(e, format="%Y%m%d")
    if lo > hi:
        raise ValueError("start不能晚于end")
    return s, e


def setup_logging(name: str) -> logging.Logger:
    """配置 INFO 级别的终端日志，并返回指定模块的 logger。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def read_years(table: str, start: str, end: str) -> pd.DataFrame:
    """读取按年分区的 raw 表，返回闭区间 [start, end] 内的记录。

    table 是表目录名；start/end 须为 YYYYMMDD，与原始 trade_date
    字符串直接比较。区间内没有任何分区文件时抛 FileNotFoundError；
    部分年份缺失会被跳过，因此返回成功本身不代表日期覆盖完整。
    """
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        path = RAW / table / f"year={year}" / "data.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"没有 {table} 的原始数据（{start} ~ {end}）。可能未运行 fetch.py 抓取数据。"
        )
    df = pd.concat(frames, ignore_index=True)
    return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()


def read_single(*parts: str) -> pd.DataFrame:
    """读取 RAW 下由 parts 拼出的必需文件；缺失时抛出带路径的错误。"""
    path = RAW.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}")
    return pd.read_parquet(path)


def read_optional(*parts: str) -> pd.DataFrame | None:
    """读取 RAW 下的可选文件；仅文件不存在时返回 None，不吞读盘错误。"""
    path = RAW.joinpath(*parts)
    return pd.read_parquet(path) if path.exists() else None


def read_calendar(start: str, end: str) -> pd.DataFrame:
    """读取区间内日历，增加 date 时间戳并将非法 is_open 按休市处理。

    返回表仍含休市日；需要交易日时再调用 open_dates。
    """
    cal = read_single("trade_cal", "data.parquet")
    cal = cal[(cal["cal_date"] >= start) & (cal["cal_date"] <= end)].copy()
    #cal的列名：["exchange","cal_date","is_open","pretrade_date"]
    cal["date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d")
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    return cal.sort_values("date").reset_index(drop=True)


def open_dates(cal: pd.DataFrame) -> list[pd.Timestamp]:
    """提取 is_open=1 的 date，沿用输入日历的行序。"""
    return cal.loc[cal["is_open"] == 1, "date"].tolist()


def month_end_dates(dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """取输入中每个自然月最大的交易日期，并按时间升序返回。

    调用方须保证日历覆盖完整月份；月中截断后调用会误把区间末日当月末。
    """
    s = pd.Series(pd.to_datetime(dates))
    return s.groupby(s.dt.to_period("M")).max().sort_values().tolist()


def save_interim(df: pd.DataFrame, filename: str) -> Path:
    """覆盖写入 interim 文件并返回路径。

    不保存 DataFrame 行索引。
    """
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / filename
    df.to_parquet(path, index=False)
    return path
