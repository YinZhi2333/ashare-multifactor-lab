"""Environment self-check for ashare-multifactor-lab.

Usage:
    python scripts/check_env.py

Exit code 0 = 核心依赖齐全，parquet 读写正常。
"""

import importlib
import pathlib
import sys
import tempfile

CORE = ["numpy", "pandas", "pyarrow", "matplotlib", "yaml"]
OPTIONAL = ["scipy", "statsmodels", "tushare", "akshare"]


def get_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print("-" * 64)

    missing_core: list[str] = []
    for name in CORE:
        ver = get_version(name)
        print(f"{name:14s} {ver}")
        if ver.startswith("MISSING"):
            missing_core.append(name)

    print("-" * 64)
    for name in OPTIONAL:
        print(f"{name:14s} {get_version(name)}")

    print("-" * 64)

    # 冒烟测试：DataFrame -> parquet -> DataFrame
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "code": ["A", "B"],
            "ret": [0.01, -0.02],
        }
    )
    path = pathlib.Path(tempfile.gettempdir()) / "mfl_smoke.parquet"
    df.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    assert back.shape == df.shape, "parquet round-trip shape mismatch"
    assert str(back["date"].dtype).startswith("datetime64"), "date dtype lost"
    path.unlink(missing_ok=True)

    if missing_core:
        print(f"FAIL: missing core packages -> {', '.join(missing_core)}")
        return 1

    print("OK: imports + parquet round-trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
