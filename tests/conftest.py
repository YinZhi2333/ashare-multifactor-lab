"""让测试复用从项目根目录直接运行的数据脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mfl" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
