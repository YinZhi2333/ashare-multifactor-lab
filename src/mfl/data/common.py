"""数据层公共工具：路径、配置、读写、日志。

约定：raw 只读，interim 可重建。所有脚本都从这里取路径，避免各自拼字符串。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
BASE_CONFIG = ROOT / "config" / "base.yaml"


def load_config() -> dict:
    if BASE_CONFIG.exists():
        return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    return {}


def resolve_range(start: str | None, end: str | None) -> tuple[str, str]:
    """CLI 优先，否则读 config/base.yaml；统一成 YYYYMMDD。"""
    cfg = load_config()
    s = start or str(cfg.get("start", "20150101"))
    e = end or str(cfg.get("end", "20241231"))
    return s.replace("-", ""), e.replace("-", "")


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def read_years(table: str, start: str, end: str) -> pd.DataFrame:
    """读取 data/raw/<table>/year=YYYY/data.parquet，按年拼接并截取区间。"""
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        path = RAW / table / f"year={year}" / "data.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"没有 {table} 的原始数据（{start} ~ {end}）。先跑 fetch.py 抓数。"
        )
    df = pd.concat(frames, ignore_index=True)
    return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()


def read_single(*parts: str) -> pd.DataFrame:
    path = RAW.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}")
    return pd.read_parquet(path)


def read_optional(*parts: str) -> pd.DataFrame | None:
    path = RAW.joinpath(*parts)
    return pd.read_parquet(path) if path.exists() else None


def read_calendar(start: str, end: str) -> pd.DataFrame:
    cal = read_single("trade_cal", "data.parquet")
    cal = cal[(cal["cal_date"] >= start) & (cal["cal_date"] <= end)].copy()
    cal["date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d")
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    return cal.sort_values("date").reset_index(drop=True)


def open_dates(cal: pd.DataFrame) -> list[pd.Timestamp]:
    return cal.loc[cal["is_open"] == 1, "date"].tolist()


def month_end_dates(dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """每个自然月的最后一个交易日。"""
    s = pd.Series(pd.to_datetime(dates))
    return s.groupby(s.dt.to_period("M")).max().sort_values().tolist()


def save_interim(df: pd.DataFrame, filename: str) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / filename
    df.to_parquet(path, index=False)
    return path
