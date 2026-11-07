"""第一阶段五因子构建、覆盖报告和输入只读审计，不生成未来收益标签。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from mfl.data.common import ROOT, resolve_range, setup_logging
from price_quality import read_rules

from .core import NAMES, REASONS, load_settings, normalize_keys
from .financial import financial_factors
from .price_volume import price_volume_factors

LOG = setup_logging("build_factors")


def compute_factors(prices: pd.DataFrame, universe: pd.DataFrame, pit: pd.DataFrame,
                    calendar: pd.DataFrame, rules: list[dict], settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """纯计算接口：所有输入由调用方提供，返回月末宽表及逐因子质量长表。"""
    targets = normalize_keys(universe, "universe")
    if targets.empty:
        raise ValueError("请求区间没有月末股票池记录")
    prices = normalize_keys(prices, "prices")
    pit = normalize_keys(pit, "PIT")
    opened = calendar.loc[pd.to_numeric(calendar["is_open"]).eq(1), "cal_date"]
    days = pd.DatetimeIndex(pd.to_datetime(opened, format="%Y%m%d")).astype("datetime64[ns]").sort_values()
    if days.empty or days.has_duplicates:
        raise ValueError("交易日历为空或有重复交易日")
    month_ends = pd.Series(days).groupby(days.to_period("M")).max()
    expected_dates = targets["date"].dt.to_period("M").map(month_ends)
    if not targets["date"].eq(expected_dates).all():
        raise ValueError("股票池日期不是本地完整日历的月末交易日")
    days = days[days <= targets["date"].max()]
    # 不截断区间前的行情历史，也不在股票入池前删除该股报价。
    prices = prices[prices["date"].le(targets["date"].max()) & prices["code"].isin(targets["code"])].copy()
    if not prices["date"].isin(days).all():
        raise ValueError("行情出现本地交易日历以外的日期")
    known = prices["close_adj"].notna()
    if not np.isclose(prices.loc[known, "close_adj"],
                      prices.loc[known, "close"] * prices.loc[known, "adj_factor"],
                      equal_nan=False, rtol=1e-10, atol=1e-10).all():
        raise ValueError("现有interim复权价与有效因子不一致，请先核查输入")
    parts = financial_factors(targets, pit, prices, settings)
    parts += price_volume_factors(targets, prices, days, month_ends, rules, settings)
    quality = pd.concat(parts, ignore_index=True).sort_values(["date", "code", "factor"]).reset_index(drop=True)
    raw = quality.pivot(index=["date", "code"], columns="factor", values="value").reset_index()
    raw.columns.name = None
    raw = targets[["date", "code"]].merge(raw, on=["date", "code"], validate="one_to_one")
    raw = raw[["date", "code", *NAMES]]
    if len(raw) != len(targets) or len(quality) != len(targets) * len(NAMES):
        raise RuntimeError("因子输出没有保留全部股票池记录")
    if quality.duplicated(["date", "code", "factor"]).any():
        raise RuntimeError("因子质量主键重复")
    valid = quality["valid"]
    if not np.isfinite(quality.loc[valid, "value"]).all() or quality.loc[~valid, "value"].notna().any():
        raise RuntimeError("因子有效标记与数值不一致")
    if (quality.loc[valid, "latest_input_date"] > quality.loc[valid, "date"]).any():
        raise RuntimeError("因子依赖了信号日之后的信息")
    return raw, quality


def coverage_tables(quality: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """仅统计可计算性和数值分布，不计算IC、排序收益或因子效果。"""
    data = quality.assign(eligible_valid=quality["eligible"] & quality["valid"])
    tables = {}
    for name, groups in [("coverage", ["factor"]), ("monthly_coverage", ["date", "factor"]),
                         ("yearly_coverage", ["year", "factor"])]:
        frame = data.assign(year=data["date"].dt.year) if "year" in groups else data
        table = frame.groupby(groups, observed=True).agg(
            total=("valid", "size"), valid=("valid", "sum"), eligible=("eligible", "sum"),
            eligible_valid=("eligible_valid", "sum"), inferred_boundary=("crosses_inferred_boundary", "sum"),
        ).reset_index()
        table["missing"] = table["total"] - table["valid"]
        table["coverage"] = table["valid"] / table["total"]
        table["eligible_coverage"] = table["eligible_valid"] / table["eligible"].replace(0, np.nan)
        tables[name] = table
    tables["missing_reasons"] = data.loc[~data["valid"]].groupby(
        ["factor", "primary_reason"], observed=True,
    ).size().rename("count").reset_index()
    tables["missing_reasons"]["description"] = tables["missing_reasons"]["primary_reason"].map(REASONS)
    rows = []
    for name in NAMES:
        part = quality[quality["factor"].eq(name) & quality["valid"]]
        values = part["value"]
        rows.append({"factor": name, "first_valid_date": part["date"].min(),
                     "negative_profit_valid": int(part["negative_profit"].sum()),
                     "mean": values.mean(), "std": values.std(), "min": values.min(),
                     "p01": values.quantile(.01), "median": values.median(),
                     "p99": values.quantile(.99), "max": values.max()})
    tables["distribution"] = pd.DataFrame(rows)
    return tables


def sha256(path: Path) -> str:
    """分块计算文件哈希，不读取或记录任何本地密钥配置。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_manifest(root: Path) -> dict:
    """记录全部raw/interim文件的内容、大小和修改时间，验证本轮未写输入。"""
    result = {}
    for base in (root / "data" / "raw", root / "data" / "interim"):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                stat = path.stat()
                result[path.relative_to(root).as_posix()] = {
                    "sha256": sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                }
    return result


def _atomic_write(path: Path, value: pd.DataFrame | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if isinstance(value, pd.DataFrame):
            value.to_parquet(temporary, index=False)
        else:
            temporary.write_text(value, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _report(tables: dict[str, pd.DataFrame], quality: pd.DataFrame, manifest: dict) -> str:
    cfg = manifest["settings"]
    lines = ["# 五因子第一阶段覆盖报告", "", f"生成时间：{manifest['created_at']}。仅描述原值和质量，不含预处理、评价或回测。", "",
             f"- 正式研究请求区间：{manifest['requested_start']}—{manifest['requested_end']}；实际信号：{manifest['signal_start']}—{manifest['signal_end']}。",
             f"- 数据预热起点：{manifest['data_start']}；历史行情继续保留，预热月份不计入以下覆盖率分母。",
             "- 信号日期为月末收盘后；下一交易日执行的规则由后续层落实。本阶段不机械shift，不生成未来收益标签。",
             f"- EP=PIT归母净利润TTM/(同日总市值×{cfg['financial']['market_cap_unit']:g})；ROE复用PIT。亏损值保留；报告期年龄超过{cfg['financial']['max_report_age_days']}天留空。",
             f"- MOM_12_1=P[m-{cfg['momentum']['skip_months']}]/P[m-{cfg['momentum']['lookback_months']}]-1；REV_1M=-(P[m]/P[m-{cfg['reversal']['lookback_months']}]-1)；TURN_20=-mean{cfg['turnover']['window']}(turnover_rate/{cfg['turnover']['percent_unit']:g})。",
             "- 保留全部股票池键；换手和价格窗口按市场交易日严格完整，不补停牌报价或缺失因子。",
             "- 跨未解决公司行动的价格窗口留空；跨推断校正边界另有标记。计算有效与tradable资格分开统计。", "",
             "## 总体覆盖", "", "| 因子 | 总记录 | 有效 | 覆盖率 | tradable样本覆盖率 | 缺失 |", "|---|---:|---:|---:|---:|---:|"]
    for row in tables["coverage"].set_index("factor").loc[NAMES].reset_index().itertuples():
        lines.append(f"| {row.factor} | {row.total:,} | {row.valid:,} | {row.coverage:.4%} | {row.eligible_coverage:.4%} | {row.missing:,} |")
    lines += ["", "## 年度覆盖率", "", "| 年份 | " + " | ".join(NAMES) + " |", "|---|" + "---:|" * len(NAMES)]
    annual = tables["yearly_coverage"].pivot(index="year", columns="factor", values="coverage")
    for year, values in annual.iterrows():
        lines.append(f"| {year} | " + " | ".join(f"{values[name]:.2%}" for name in NAMES) + " |")
    lines += ["", "## 缺失的首要原因", "", "同一条记录可能有多个问题；下表每条只计一次，完整原因见quality.parquet的reason_codes。", "",
              "| 因子 | 首要原因 | 条数 | 解释 |", "|---|---|---:|---|"]
    for row in tables["missing_reasons"].itertuples():
        lines.append(f"| {row.factor} | {row.primary_reason} | {row.count:,} | {row.description} |")
    lines += ["", "## 审计与局限", "",
              f"- 输入保护：{len(manifest['protected_before'])}个raw/interim文件内容、大小和修改时间一致。",
              "- 每个因子每月的总数、有效数、资格覆盖率见monthly_coverage.csv；数值分布见distribution.csv。",
              "- base_value为未加方向的经济量，value及raw.parquet已按因子定义统一方向；没有去极值、标准化、中性化或填充。",
              f"- 价格因子窗口跨推断校正边界的质量行共{int(quality['crosses_inferred_boundary'].sum())}条；这些修正尚非数据商逐笔确认。",
              f"- 本批实际目标窗口触及未解决公司行动共{int(quality['reason_codes'].str.contains('unresolved_price_event').sum())}条；规则仍保留，合成测试验证其会阻断相关窗口。",
              "- 严格完整窗口会排除有历史停牌的样本，覆盖不足不等于因子失效；不能据覆盖率推断选股能力。",
              "- 首年动量缺预热、部分财务TTM缺失、财务原始历史版本不完整等限制保留；未为本阶段抓数。",
              "- 行业中性化、因子收益检验、合成及回测均尚未执行；base.yaml中后续阶段参数未在这里启用。", ""]
    return "\n".join(lines)


def run(root: Path = ROOT, config_path: Path | None = None,
        start: str | None = None, end: str | None = None) -> dict:
    """从现有加工输入构建；只写processed/results/reports，清单最后发布。"""
    root = root.resolve()
    config_path = root / "config" / "factors.yaml" if config_path is None else config_path.resolve()
    quality_path = root / "config" / "price_quality.yaml"
    base_path = root / "config" / "base.yaml"
    settings = load_settings(config_path)
    rules = read_rules(quality_path)
    before = protected_manifest(root)
    config_paths = [config_path, quality_path] + ([base_path] if base_path.exists() else [])
    config_hashes = {str(path): sha256(path) for path in config_paths}
    start, end = resolve_range(start, end, config_path=base_path)
    data_start, _ = resolve_range(None, None, data=True, config_path=base_path)
    interim = root / "data" / "interim"
    universe = pd.read_parquet(interim / "universe.parquet")
    universe = universe[universe["date"].between(pd.to_datetime(start), pd.to_datetime(end))]
    if universe.empty:
        raise ValueError("请求区间没有月末股票池记录")
    codes = universe["code"].unique().tolist()
    prices = pd.read_parquet(interim / "prices.parquet",
                             columns=["date", "code", "close", "close_adj", "adj_factor", "total_mv", "turnover_rate"],
                             filters=[("code", "in", codes), ("date", "<=", universe["date"].max())])
    pit = pd.read_parquet(interim / "fundamental_panel.parquet")
    calendar = pd.read_parquet(root / "data" / "raw" / "trade_cal" / "data.parquet")
    if "exchange" in calendar:
        calendar = calendar[calendar["exchange"].eq("SSE")]
    LOG.info("研究%s—%s；预热数据自%s保留", start, end, data_start)
    LOG.info("输入：月末目标%d行/%d只，读取入池前行情%d行", len(universe), len(codes), len(prices))
    raw, quality = compute_factors(prices, universe, pit, calendar, rules, settings)
    tables = coverage_tables(quality)
    after = protected_manifest(root)
    if before != after or any(sha256(Path(path)) != digest for path, digest in config_hashes.items()):
        raise RuntimeError("构建期间输入发生变化，停止发布因子结果")
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / "build_factors.py"]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout.decode().strip()
    diff = subprocess.run(["git", "diff", "HEAD", "--", "src", "scripts", "tests", "config/base.yaml"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    manifest = {"stage": "raw_factors", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "requested_start": start, "requested_end": end, "data_start": data_start,
                "signal_start": str(raw["date"].min().date()), "signal_end": str(raw["date"].max().date()),
                "rows": len(raw), "quality_rows": len(quality), "settings": settings,
                "config_sha256": config_hashes, "git_commit": head,
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "source_sha256": source_hashes,
                "environment": {pkg: importlib.metadata.version(pkg) for pkg in ("pandas", "numpy", "pyarrow", "PyYAML")},
                "protected_before": before, "protected_after": after, "inputs_unchanged": True}
    processed = root / "data" / "processed" / "factors"
    audit = root / "results" / "factors"
    output = {processed / "raw.parquet": raw, processed / "quality.parquet": quality}
    for name, table in tables.items():
        output[audit / f"{name}.csv"] = table.to_csv(index=False, lineterminator="\n")
    output[root / "reports" / "factor_coverage.md"] = _report(tables, quality, manifest)
    for path, value in output.items():
        _atomic_write(path, value)
    if protected_manifest(root) != before:
        raise RuntimeError("输出后输入保护检查失败，未发布完成清单")
    manifest["output_sha256"] = {path.relative_to(root).as_posix(): sha256(path) for path in output}
    _atomic_write(audit / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    for row in tables["coverage"].itertuples():
        LOG.info("%s：%d/%d，覆盖%.2f%%；tradable样本覆盖%.2f%%", row.factor, row.valid, row.total,
                 row.coverage * 100, row.eligible_coverage * 100)
    LOG.info("完成：%d条因子面板，%d条质量明细；%d个输入文件未改", len(raw), len(quality), len(before))
    return manifest

