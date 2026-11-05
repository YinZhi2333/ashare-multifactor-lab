#!/usr/bin/env python
"""Tushare 原始数据抓取：只负责下载和落盘，不做清洗。

复权、PIT 对齐、股票池过滤都属于 transform 阶段，放在别的模块里。
这样 raw 数据永远可回溯，清洗逻辑改了也不用重新下载。

用法（在项目根目录执行）::

    python src/mfl/data/fetch.py probe                           # 先探测权限
    python src/mfl/data/fetch.py calendar     --start 20150101 --end 20241231
    python src/mfl/data/fetch.py basic
    python src/mfl/data/fetch.py price        --start 20150101 --end 20241231
    python src/mfl/data/fetch.py metrics      --start 20150101 --end 20241231
    python src/mfl/data/fetch.py financials   --start 20150101 --end 20241231
    python src/mfl/data/fetch.py index        --start 20150101 --end 20241231
    python src/mfl/data/fetch.py tradability  --start 20150101 --end 20241231
    python src/mfl/data/fetch.py all          --start 20150101 --end 20241231

Token 优先级：``--token`` > 环境变量 ``TUSHARE_TOKEN`` > ``config/local.yaml``。

落盘结构::

    data/raw/trade_cal/data.parquet
    data/raw/stock_basic/data.parquet
    data/raw/namechange/data.parquet
    data/raw/daily/year=2024/data.parquet
    data/raw/adj_factor/year=2024/data.parquet
    data/raw/daily_basic/year=2024/data.parquet
    data/raw/income/data.parquet
    data/raw/balancesheet/data.parquet
    data/raw/cashflow/data.parquet
    data/raw/stk_limit/year=2024/data.parquet
    data/raw/suspend_d/year=2024/data.parquet
    data/raw/index_daily/ts_code=000905.SH/data.parquet
    data/raw/index_weight/index_code=000905.SH/data.parquet
    data/raw/fina_indicator/data.parquet      # 可选
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
LOCAL_CONFIG = ROOT / "config" / "local.yaml"
INDEX_CODE = "000905.SH"  # 中证500

LOG = logging.getLogger("fetch")


# --------------------------------------------------------------------------
# token
# --------------------------------------------------------------------------
def load_token(cli_token: str | None = None) -> str:
    """按优先级取 token；都没有就给出清楚的设置说明。"""
    if cli_token:
        return cli_token.strip()

    env_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if env_token:
        return env_token

    if LOCAL_CONFIG.exists():
        cfg = yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8")) or {}
        token = str(cfg.get("tushare_token", "")).strip()
        if token and token != "your_token_here":
            return token

    raise SystemExit(
        "没有找到 Tushare token。三种方式任选一种：\n"
        '  1) 环境变量：setx TUSHARE_TOKEN "你的token"，然后重开终端\n'
        "  2) 配置文件：把 config/local.example.yaml 复制成 config/local.yaml，填入 token\n"
        "  3) 命令行：--token 你的token"
    )


# --------------------------------------------------------------------------
# Tushare 客户端：限速 + 重试
# --------------------------------------------------------------------------
class RateLimited(RuntimeError):
    """接口频次限制（例如 1 次/小时）。等一小时不现实，直接抛给上层降级。"""


class Client:
    """薄封装：统一处理限速、网络重试、权限错误。"""

    def __init__(self, token: str, sleep: float = 0.4, retries: int = 3) -> None:
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("未安装 tushare：pip install tushare") from exc

        ts.set_token(token)
        self.pro = ts.pro_api()
        self.sleep = sleep
        self.retries = retries
        self.calls = 0

    def call(self, api: str, _interval: float | None = None, **kwargs) -> pd.DataFrame:
        fn = getattr(self.pro, api, None)
        if fn is None:
            raise AttributeError(f"tushare 没有接口 {api}")

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                df = fn(**kwargs)
                self.calls += 1
                time.sleep(self.sleep if _interval is None else _interval)
                return df if df is not None else pd.DataFrame()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc)

                # 权限/积分问题重试没有意义，直接抛出
                if "权限" in msg or "积分" in msg:
                    raise PermissionError(f"{api}: {msg}") from exc

                # 参数缺失（如财务接口没给 ts_code）重试也没用
                if "必填参数" in msg:
                    raise ValueError(f"{api} 参数不完整：{msg}") from exc

                # 小时级限制（如 trade_cal 1次/小时）：等一小时不现实，抛出交给上层
                if "小时" in msg:
                    raise RateLimited(f"{api} 触发小时级频次限制：{msg}") from exc

                if "每分钟" in msg or "频率" in msg:
                    wait = 61
                else:
                    wait = min(2**attempt, 30)

                LOG.warning(
                    "%s 第 %d/%d 次失败：%s（%.0fs 后重试）",
                    api,
                    attempt,
                    self.retries,
                    msg[:100],
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"{api} 连续失败：{last_exc}") from last_exc


# --------------------------------------------------------------------------
# 读写小工具
# --------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------
# 日历 / 股票列表
# --------------------------------------------------------------------------
def _covers(df: pd.DataFrame | None, start: str, end: str) -> bool:
    if df is None or df.empty or "cal_date" not in df.columns:
        return False
    dates = df["cal_date"].astype(str)
    return dates.min() <= start and dates.max() >= end


def fetch_calendar_akshare(start: str, end: str) -> pd.DataFrame:
    """Tushare 日历接口受限时的兜底：AkShare 免费、不限频。"""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("trade_cal 不可用，且没有安装 akshare：pip install akshare") from exc

    raw = ak.tool_trade_date_hist_sina()
    open_days = set(pd.to_datetime(raw["trade_date"]).dt.strftime("%Y%m%d"))

    days = pd.date_range(
        pd.to_datetime(start, format="%Y%m%d"),
        pd.to_datetime(end, format="%Y%m%d"),
        freq="D",
    )
    cal = pd.DataFrame({"cal_date": days.strftime("%Y%m%d").tolist()})
    cal["is_open"] = cal["cal_date"].isin(open_days).astype(int)
    cal["exchange"] = "SSE"
    cal["pretrade_date"] = ""
    return cal[["exchange", "cal_date", "is_open", "pretrade_date"]]


def fetch_calendar(client: Client, start: str, end: str, force: bool = False) -> pd.DataFrame:
    path = RAW / "trade_cal" / "data.parquet"
    old = None if force else _read(path)

    # 日历几乎不变：本地已覆盖就直接用，别再浪费接口额度
    if old is not None and _covers(old, start, end):
        LOG.info("trade_cal：本地已覆盖 %s ~ %s，跳过接口请求", start, end)
        return old

    try:
        df = client.call(
            "trade_cal",
            exchange="SSE",
            start_date=start,
            end_date=end,
            fields="exchange,cal_date,is_open,pretrade_date",
        )
    except (PermissionError, RuntimeError) as exc:
        LOG.warning("trade_cal 接口不可用：%s", exc)
        LOG.warning("改用 AkShare 生成交易日历（不需要 token）")
        df = fetch_calendar_akshare(start, end)

    if df.empty and (old is None or old.empty):
        LOG.warning("trade_cal：没有数据")
        return pd.DataFrame(columns=["cal_date", "is_open"])

    if old is not None and not old.empty:
        df = pd.concat([old, df], ignore_index=True)

    df = df.drop_duplicates("cal_date").sort_values("cal_date").reset_index(drop=True)
    _write(df, path)
    LOG.info("trade_cal：%d 行（%s ~ %s）", len(df), df["cal_date"].min(), df["cal_date"].max())
    return df


def get_trade_dates(client: Client, start: str, end: str, force: bool = False) -> list[str]:
    cal = fetch_calendar(client, start, end, force=force)
    is_open = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    mask = (is_open == 1) & (cal["cal_date"] >= start) & (cal["cal_date"] <= end)
    return sorted(cal.loc[mask, "cal_date"].astype(str).tolist())


def fetch_basic(client: Client, force: bool = False) -> None:
    """L=上市，D=退市，P=暂停上市。必须把 D 也抓下来，否则有幸存者偏差。"""
    frames = []
    for status in ("L", "D", "P"):
        df = client.call(
            "stock_basic",
            exchange="",
            list_status=status,
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs",
        )
        if df.empty:
            continue
        df = df.copy()
        df["list_status"] = status
        frames.append(df)

    if not frames:
        LOG.warning("stock_basic：没有拿到数据")
        return

    out = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
    _write(out, RAW / "stock_basic" / "data.parquet")
    counts = out["list_status"].value_counts().to_dict()
    LOG.info("stock_basic：%d 只（%s）", len(out), counts)


def fetch_namechange(client: Client, start: str, end: str, force: bool = False) -> None:
    """更名记录，用来还原历史 ST 状态。"""
    path = RAW / "namechange" / "data.parquet"
    old = None if force else _read(path)
    frames = [] if old is None or old.empty else [old]

    for year in range(int(start[:4]), int(end[:4]) + 1):
        df = client.call("namechange", start_date=f"{year}0101", end_date=f"{year}1231")
        if not df.empty:
            frames.append(df)

    if not frames:
        LOG.warning("namechange：没有拿到数据")
        return

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["ts_code", "ann_date", "name"]).sort_values(["ts_code", "ann_date"])
    _write(out, path)
    LOG.info("namechange：%d 行", len(out))


# --------------------------------------------------------------------------
# 按交易日抓取（行情类）
# --------------------------------------------------------------------------
def fetch_by_date(
    client: Client,
    api: str,
    table: str,
    dates: list[str],
    force: bool = False,
) -> None:
    """按年分区，增量抓取。已有年份只补缺失的交易日。"""
    if not dates:
        LOG.warning("%s：没有交易日", table)
        return

    by_year: dict[int, list[str]] = {}
    for d in dates:
        by_year.setdefault(int(d[:4]), []).append(d)

    for year, days in sorted(by_year.items()):
        path = RAW / table / f"year={year}" / "data.parquet"
        old = None if force else _read(path)
        done = set(old["trade_date"].astype(str)) if old is not None and not old.empty else set()
        todo = [d for d in days if d not in done]

        if not todo:
            LOG.info("%s %d：已完整，跳过", table, year)
            continue

        frames = [old] if old is not None and not old.empty else []
        for i, d in enumerate(todo, 1):
            df = client.call(api, trade_date=d)
            if not df.empty:
                frames.append(df)
            if i % 50 == 0 or i == len(todo):
                LOG.info("%s %d：%d/%d", table, year, i, len(todo))

        if not frames:
            continue

        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(["ts_code", "trade_date"]).sort_values(["trade_date", "ts_code"])
        _write(out, path)
        LOG.info("%s %d：%d 行", table, year, len(out))


# --------------------------------------------------------------------------
# 指数：行情 + 历史成分股
# --------------------------------------------------------------------------
def fetch_index_daily(client: Client, start: str, end: str, force: bool = False) -> None:
    path = RAW / "index_daily" / f"ts_code={INDEX_CODE}" / "data.parquet"
    old = None if force else _read(path)

    df = client.call("index_daily", ts_code=INDEX_CODE, start_date=start, end_date=end)
    if df.empty and (old is None or old.empty):
        LOG.warning("index_daily：没有数据")
        return

    if old is not None and not old.empty:
        df = pd.concat([old, df], ignore_index=True)

    df = df.drop_duplicates(["ts_code", "trade_date"]).sort_values("trade_date")
    _write(df, path)
    LOG.info("index_daily：%d 行", len(df))


def fetch_index_weight(client: Client, start: str, end: str, force: bool = False) -> None:
    """按月抓成分股快照；一次抓十年很可能被单次返回上限截断。"""
    path = RAW / "index_weight" / f"index_code={INDEX_CODE}" / "data.parquet"
    old = None if force else _read(path)
    frames = [old] if old is not None and not old.empty else []

    have_months: set[str] = set()
    if old is not None and not old.empty:
        have_months = set(old["trade_date"].astype(str).str[:6])

    months = pd.period_range(
        pd.to_datetime(start, format="%Y%m%d"),
        pd.to_datetime(end, format="%Y%m%d"),
        freq="M",
    )
    fetched = 0
    for m in months:
        if not force and m.strftime("%Y%m") in have_months:
            continue
        df = client.call(
            "index_weight",
            index_code=INDEX_CODE,
            start_date=m.start_time.strftime("%Y%m%d"),
            end_date=m.end_time.strftime("%Y%m%d"),
        )
        if not df.empty:
            frames.append(df)
            fetched += 1

    if not frames:
        LOG.warning("index_weight：没有数据（可能积分不足）")
        return

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["index_code", "con_code", "trade_date"]).sort_values("trade_date")
    _write(out, path)
    LOG.info("index_weight：%d 行，本次抓 %d 个月", len(out), fetched)


def fetch_index(client: Client, start: str, end: str, force: bool = False) -> None:
    for fn in (fetch_index_daily, fetch_index_weight):
        try:
            fn(client, start, end, force)
        except PermissionError as exc:
            LOG.error("%s 权限不足，跳过：%s", fn.__name__, exc)


# --------------------------------------------------------------------------
# 财务：按单只股票抓（2000 积分档只允许单只，全市场要 *_vip 5000 分）
# --------------------------------------------------------------------------
INCOME_FIELDS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
    "basic_eps,n_income_attr_p,revenue,operate_profit,total_profit"
)
BS_FIELDS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,"
    "total_assets,total_liab,total_hldr_eqy_exc_min_int"
)
CF_FIELDS = "ts_code,ann_date,f_ann_date,end_date,report_type,n_cashflow_act"
FI_FIELDS = (
    "ts_code,ann_date,end_date,roe,roe_waa,roe_dt,roe_yearly,roa,"
    "netprofit_yoy,grossprofit_margin,netprofit_margin,debt_to_assets,eps,bps"
)

FIN_CHECKPOINT = 200


def _candidate_codes() -> list[str]:
    """优先用已生成的股票池，其次中证500历史成分股，最后全市场。"""
    uni = INTERIM / "universe.parquet"
    if uni.exists():
        codes = sorted(pd.read_parquet(uni, columns=["code"])["code"].astype(str).unique())
        LOG.info("财务按股票抓取：使用 universe.parquet，%d 只", len(codes))
        return codes

    iw = RAW / "index_weight" / f"index_code={INDEX_CODE}" / "data.parquet"
    if iw.exists():
        codes = sorted(pd.read_parquet(iw, columns=["con_code"])["con_code"].astype(str).unique())
        LOG.info("财务按股票抓取：使用中证500历史成分股并集，%d 只", len(codes))
        return codes

    sb = RAW / "stock_basic" / "data.parquet"
    if sb.exists():
        codes = sorted(pd.read_parquet(sb, columns=["ts_code"])["ts_code"].astype(str).unique())
        LOG.info("财务按股票抓取：没有股票池，退回全市场 %d 只（会很慢）", len(codes))
        return codes

    raise RuntimeError("需要先抓 stock_basic 或 index_weight，才能确定股票列表")


def _flush_statement(frames: list[pd.DataFrame], path: Path) -> None:
    if not frames:
        return
    old = _read(path)
    all_frames = ([old] if old is not None and not old.empty else []) + frames
    out = pd.concat(all_frames, ignore_index=True)
    out = out.drop_duplicates(["ts_code", "end_date", "ann_date"]).sort_values(["end_date", "ts_code"])
    _write(out, path)


def fetch_statement(
    client: Client,
    api: str,
    table: str,
    fields: str,
    start: str,
    end: str,
    force: bool = False,
    report_type: str | None = "1",
) -> None:
    """逐只股票抓财务报表。

    2000 积分档没有 *_vip 权限，income / balancesheet / cashflow 只接受
    ts_code，不能按 period 拉全市场。这里每 200 只落一次盘，断了重跑能续。
    """
    path = RAW / table / "data.parquet"
    if force and path.exists():
        path.unlink()

    old = _read(path)
    done = set(old["ts_code"].astype(str)) if old is not None and not old.empty else set()
    codes = [c for c in _candidate_codes() if c not in done]

    LOG.info(
        "%s：待抓 %d 只（已有 %d 只），预计 %.0f 分钟",
        table,
        len(codes),
        len(done),
        len(codes) * 0.8 / 60,
    )

    buffer: list[pd.DataFrame] = []
    try:
        for i, code in enumerate(codes, 1):
            params: dict = {"ts_code": code, "fields": fields}
            if report_type:
                params["report_type"] = report_type
            df = client.call(api, _interval=0.8, **params)
            if not df.empty:
                buffer.append(df)
            if i % FIN_CHECKPOINT == 0 or i == len(codes):
                _flush_statement(buffer, path)
                buffer = []
                LOG.info("%s：%d/%d（已落盘）", table, i, len(codes))
    finally:
        _flush_statement(buffer, path)


def fetch_financials(client: Client, start: str, end: str, force: bool = False) -> None:
    jobs = [
        ("income", "income", INCOME_FIELDS, "1"),
        ("balancesheet", "balancesheet", BS_FIELDS, "1"),
        ("cashflow", "cashflow", CF_FIELDS, "1"),
    ]
    for api, table, fields, report_type in jobs:
        try:
            fetch_statement(client, api, table, fields, start, end, force, report_type)
        except PermissionError as exc:
            LOG.error("%s 权限不足，跳过：%s", table, exc)
        except Exception:  # noqa: BLE001
            LOG.exception("%s 失败，继续后面的表", table)


def fetch_fundamental(client: Client, start: str, end: str, force: bool = False) -> None:
    try:
        fetch_statement(
            client,
            "fina_indicator",
            "fina_indicator",
            FI_FIELDS,
            start,
            end,
            force,
            report_type=None,
        )
    except PermissionError as exc:
        LOG.error("fina_indicator 权限不足，跳过：%s", exc)


# --------------------------------------------------------------------------
# 权限探测
# --------------------------------------------------------------------------
PROBE_CALLS: list[tuple[str, dict]] = [
    ("trade_cal", {"exchange": "SSE", "start_date": "20240101", "end_date": "20240110"}),
    ("stock_basic", {"list_status": "L", "fields": "ts_code,name,list_date"}),
    ("daily", {"trade_date": "20240102"}),
    ("adj_factor", {"trade_date": "20240102"}),
    ("daily_basic", {"trade_date": "20240102"}),
    ("income", {"ts_code": "600000.SH", "period": "20231231", "report_type": "1"}),
    ("balancesheet", {"ts_code": "600000.SH", "period": "20231231", "report_type": "1"}),
    ("cashflow", {"ts_code": "600000.SH", "period": "20231231", "report_type": "1"}),
    ("index_daily", {"ts_code": "000905.SH", "start_date": "20240101", "end_date": "20240110"}),
    ("index_weight", {"index_code": "000905.SH", "start_date": "20240101", "end_date": "20240131"}),
    ("stk_limit", {"trade_date": "20240102"}),
    ("suspend_d", {"trade_date": "20240102"}),
]

PROBE_CRITICAL = {"daily_basic", "income", "balancesheet", "index_weight"}


def probe(client: Client) -> None:
    """每个接口打一发，确认当前积分到底能调什么。不落盘。"""
    LOG.info("开始探测接口权限：%d 个接口，各 1 次调用", len(PROBE_CALLS))
    rows: list[tuple[str, str, str]] = []

    for api, params in PROBE_CALLS:
        try:
            df = client.call(api, _interval=0.1, **params)
            status, detail = "OK", f"{len(df)} 行"
        except PermissionError as exc:
            status, detail = "DENIED", str(exc)[:48]
        except RateLimited as exc:
            status, detail = "RATE", str(exc)[:48]
        except Exception as exc:  # noqa: BLE001
            status, detail = "FAIL", f"{type(exc).__name__}: {str(exc)[:40]}"
        rows.append((api, status, detail))
        LOG.info("%-14s %-7s %s", api, status, detail)

    ok = [r[0] for r in rows if r[1] == "OK"]
    missing = [r[0] for r in rows if r[1] != "OK" and r[0] in PROBE_CRITICAL]

    LOG.info("可用：%s", ", ".join(ok) if ok else "无")
    if missing:
        LOG.warning("关键接口不可用：%s", ", ".join(missing))
        LOG.warning("缺 daily_basic 就没有市值/换手，缺 income/balancesheet 就只能用 roe_yearly，缺 index_weight 就没有历史成分股")


# --------------------------------------------------------------------------
# 全量
# --------------------------------------------------------------------------
def run_all(client: Client, args: argparse.Namespace) -> None:
    cal = fetch_calendar(client, args.start, args.end, force=args.force)
    is_open = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    mask = (is_open == 1) & (cal["cal_date"] >= args.start) & (cal["cal_date"] <= args.end)
    dates = sorted(cal.loc[mask, "cal_date"].astype(str).tolist())
    LOG.info("交易日 %d 天，行情/可交易性约需 %d 次请求", len(dates), len(dates) * 5)

    steps = [
        ("stock_basic", lambda: fetch_basic(client, args.force)),
        ("namechange", lambda: fetch_namechange(client, args.start, args.end, args.force)),
        ("daily", lambda: fetch_by_date(client, "daily", "daily", dates, args.force)),
        ("adj_factor", lambda: fetch_by_date(client, "adj_factor", "adj_factor", dates, args.force)),
        ("daily_basic", lambda: fetch_by_date(client, "daily_basic", "daily_basic", dates, args.force)),
        ("stk_limit", lambda: fetch_by_date(client, "stk_limit", "stk_limit", dates, args.force)),
        ("suspend_d", lambda: fetch_by_date(client, "suspend_d", "suspend_d", dates, args.force)),
        ("index", lambda: fetch_index(client, args.start, args.end, args.force)),
        ("financials", lambda: fetch_financials(client, args.start, args.end, args.force)),
        ("fina_indicator", lambda: fetch_fundamental(client, args.start, args.end, args.force)),
    ]

    for name, fn in steps:
        try:
            fn()
        except PermissionError as exc:
            LOG.error("%s 权限不足，跳过：%s", name, exc)
        except Exception:  # noqa: BLE001
            LOG.exception("%s 失败，继续后面的任务", name)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", default="20150101", help="开始日期 YYYYMMDD")
    common.add_argument("--end", default="20241231", help="结束日期 YYYYMMDD")
    common.add_argument("--force", action="store_true", help="忽略已有文件，全部重抓")
    common.add_argument("--sleep", type=float, default=0.4, help="每次请求后的等待秒数")
    common.add_argument("--token", default=None, help="Tushare token（不推荐写在命令里）")

    parser = argparse.ArgumentParser(description="Tushare 原始数据抓取")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", parents=[common], help="探测各接口权限（不落盘）")
    sub.add_parser("calendar", parents=[common], help="交易日历")
    sub.add_parser("basic", parents=[common], help="股票列表 + 更名记录")
    sub.add_parser("price", parents=[common], help="daily + adj_factor")
    sub.add_parser("metrics", parents=[common], help="daily_basic（市值/换手/PE/PB）")
    sub.add_parser("financials", parents=[common], help="income + balancesheet + cashflow")
    sub.add_parser("tradability", parents=[common], help="涨跌停价 + 停牌")
    sub.add_parser("index", parents=[common], help="中证500 行情 + 历史成分股")
    sub.add_parser("fundamental", parents=[common], help="fina_indicator（可选，逐只股票）")
    sub.add_parser("all", parents=[common], help="按顺序抓全部（含涨跌停/停牌）")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    client = Client(load_token(args.token), sleep=args.sleep)
    LOG.info("开始 %s：%s ~ %s", args.command, args.start, args.end)

    if args.command == "probe":
        probe(client)

    elif args.command == "calendar":
        fetch_calendar(client, args.start, args.end, args.force)

    elif args.command == "basic":
        fetch_basic(client, args.force)
        try:
            fetch_namechange(client, args.start, args.end, args.force)
        except PermissionError as exc:
            LOG.error("namechange 权限不足，跳过：%s", exc)

    elif args.command == "price":
        dates = get_trade_dates(client, args.start, args.end, args.force)
        LOG.info("交易日 %d 天，预计 %d 次请求", len(dates), len(dates) * 2)
        fetch_by_date(client, "daily", "daily", dates, args.force)
        fetch_by_date(client, "adj_factor", "adj_factor", dates, args.force)

    elif args.command == "metrics":
        dates = get_trade_dates(client, args.start, args.end, args.force)
        LOG.info("交易日 %d 天，预计 %d 次请求", len(dates), len(dates))
        fetch_by_date(client, "daily_basic", "daily_basic", dates, args.force)

    elif args.command == "financials":
        fetch_financials(client, args.start, args.end, args.force)

    elif args.command == "tradability":
        dates = get_trade_dates(client, args.start, args.end, args.force)
        LOG.info("交易日 %d 天，预计 %d 次请求", len(dates), len(dates) * 2)
        fetch_by_date(client, "stk_limit", "stk_limit", dates, args.force)
        fetch_by_date(client, "suspend_d", "suspend_d", dates, args.force)

    elif args.command == "index":
        fetch_index(client, args.start, args.end, args.force)

    elif args.command == "fundamental":
        fetch_fundamental(client, args.start, args.end, args.force)

    elif args.command == "all":
        run_all(client, args)

    LOG.info("完成，共请求 %d 次", client.calls)


if __name__ == "__main__":
    try:
        main()
    except RateLimited as exc:
        LOG.error("%s", exc)
        LOG.error(
            "Tushare 当前积分对应的频次太低。120 分只能调非复权日线；"
            "学生认证可免费拿到 2000 分，解锁 daily_basic / index_weight / fina_indicator。"
        )
        raise SystemExit(2) from exc
    except PermissionError as exc:
        LOG.error("%s", exc)
        LOG.error(
            "该接口需要更高积分。多数指标/财务/指数接口要求 2000 分；"
            "先完成学生认证，或改用 AkShare。"
        )
        raise SystemExit(2) from exc
