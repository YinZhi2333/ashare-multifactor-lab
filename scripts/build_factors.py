"""构建五个因子的原值、质量明细和覆盖报告；不抓数或改动interim。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mfl" / "data"))

from mfl.factors.build import run


def main() -> None:
    """可选区间只筛选信号月份，仍从完整本地行情获取历史预热。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="覆盖base.yaml的研究起日；默认2016，不截断2015预热")
    parser.add_argument("--end", help="覆盖base.yaml的研究止日；默认2024年底")
    args = parser.parse_args()
    if args.start and args.end and args.start.replace("-", "") > args.end.replace("-", ""):
        parser.error("start不能晚于end")
    run(start=args.start, end=args.end)


if __name__ == "__main__":
    main()

