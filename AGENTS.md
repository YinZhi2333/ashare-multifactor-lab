# AGENTS.md

本文件是项目的**持久协作规则**，面向所有接手这个仓库的 AI/人类协作者。

它只规定「必须遵守什么」，不记录「当前做到哪一步」——项目说明、当前进度、验证结果、已知局限与下一步计划统一维护在 `README.md`。

`HANDOFF.md` 停止维护，仅保留为历史归档；不再要求读取、更新或同步，也不作为当前状态的依据。判断项目状态时，以仓库实际代码、数据和验证结果为准。

本项目的参考流程是`项目提案_A股多因子研究框架.md`，仅作参考，你可以对其提出质疑，并向我说明。

---

## 1. Project scope

- 项目目标：A 股中证500 多因子研究框架。最终产出是可复现的因子检验、扣费组合回测和一份研究报告，不是「一个高收益策略」。
- 代码分层：
  - `src/mfl/` — 可复用库代码。当前只有 `data/`（fetch / adjust / universe / pit），后续加 `factors/`、`preprocess/`、`evaluation/`、`portfolio/`、`backtest/`、`analytics/`。
  - `scripts/` — 一次性入口脚本（`check_env.py`、`check_raw.py`）。
  - `tests/` — pytest 测试。目前还没有用例，新增测试放这里。
  - `config/` — 参数与本地密钥。
  - `data/raw/` — 原始数据，只增不改。
  - `data/interim/` — 加工产物，可由 raw 重建。
  - `results/`、`reports/` — 每次实验的输出，不入库。
- **不要手工修改 `data/raw/` 下的任何文件**。只有 `fetch.py` 可以写 raw，且必须走增量逻辑。
- **不要手工编辑 `data/interim/` 或 `results/` 的产物**。要改就改脚本、重跑。
- 改动限定在请求范围内。不要顺手重构无关模块、不要重命名公共函数、不要改输出 schema。

## 2. Setup

- Python 3.11，conda 环境名 `mfl`。依赖唯一来源是 `environment.yml`：

  ```powershell
  conda env create -f environment.yml
  conda activate mfl
  python -m ipykernel install --user --name mfl --display-name "Python (mfl)"
  ```

- token 只允许放两处：
  1. 环境变量 `TUSHARE_TOKEN`；
  2. `config/local.yaml`（已被 `.gitignore` 忽略，从 `config/local.example.yaml` 复制）。
- 不要把 token 写进代码、README、notebook、命令历史或日志。
- 环境自检：`python scripts/check_env.py`。
- 数据自检：`python scripts/check_raw.py`。

## 3. Development commands

在项目根目录执行（`cd "E:\大学资料\1量化\ashare-multifactor-lab"`，并先 `conda activate mfl`）：

```powershell
# 环境与数据检查
python scripts/check_env.py
python scripts/check_raw.py

# 数据加工（按顺序）
python src/mfl/data/adjust.py   --start 20150101 --end 20241231
python src/mfl/data/universe.py --start 20150101 --end 20241231
python src/mfl/data/pit.py      --start 20150101 --end 20241231

# 测试（目前 tests/ 为空，新增用例后再跑）
pytest -q
```

数据抓取命令（**不要无故执行**，见第 5 节）：

```powershell
python src/mfl/data/fetch.py probe
python src/mfl/data/fetch.py calendar --start 20150101 --end 20241231
python src/mfl/data/fetch.py basic
python src/mfl/data/fetch.py price    --start 20150101 --end 20241231
python src/mfl/data/fetch.py metrics  --start 20150101 --end 20241231
python src/mfl/data/fetch.py index    --start 20150101 --end 20241231
python src/mfl/data/fetch.py financials --start 20150101 --end 20241231
```

- 没有配置 lint / typecheck；不要声称跑过不存在的命令。如果以后引入 ruff/mypy，先在 `environment.yml` 和 README 里写清楚。
- 脚本必须从项目根目录运行：`common.py` 靠「脚本同目录」被导入，不是安装过的包。

## 4. Code conventions

- Python 3.11；新文件用 `from __future__ import annotations`；公共函数写类型注解。
- 路径统一用 `pathlib.Path`，从 `common.py` 的 `ROOT` / `RAW` / `INTERIM` 派生；不要硬编码绝对路径。
- 保持 **raw 与 transform 分离**：
  - `fetch.py` 只下载、只落盘，不做复权、对齐、过滤；
  - `adjust.py` / `universe.py` / `pit.py` 只读 raw、写 interim。
- pandas 优先；横截面用 `groupby(...).transform`，时间对齐用 `merge_asof`，长宽表用 `pivot` / `stack`。除了回测的逐日递推，不要写逐行循环。
- 不要静默 `except`。捕获后要么重试、要么降级并打 `WARNING`、要么抛出带上下文的错误。
- 关键步骤要打日志：行数、缺失率、覆盖率、日期范围。没有日志的脚本视为没做完。
- 参数不要散落成魔法数字；能进 `config/base.yaml` 的就进配置，函数默认值要写注释。
- 保持现有的中文 docstring / 注释风格；不要为了「国际化」把已有注释全改成英文。
- 不要提前抽象。复用 `common.py` 的读写和路径；没有第二个调用方之前，不要为「未来可能用到」建基类或框架。

## 5. Data rules（本项目的核心）

### 5.1 数据不可变与增量

- `data/raw/` 只增不改。`fetch.py` 是增量的：行情按 `trade_date`、财务按 `ts_code`、指数成分股按月跳过。
- 不要删除 raw 来「重来」。确实要重抓时，先确认，再用 `--force`；`--force` 会重下整段，耗时且消耗配额。
- 抓数前先跑 `python scripts/check_raw.py`。已有数据就不要再跑 `all`。

### 5.2 Tushare 2000 积分档的硬约束

- `income` / `balancesheet` / `cashflow` / `fina_indicator` **只支持单只股票查询**；按报告期拉全市场要用 `*_vip`（5000 分），本项目不用。
- 财务接口限频 80 次/分钟；`fetch_statement` 用 `0.8` 秒间隔，不要调小。
- 财务股票池的优先级：`universe.parquet` → `index_weight` 历史成分股并集 → `stock_basic` 全市场。不要默认抓全市场。
- `trade_cal` 在低积分档可能限制为 1 次/小时；本地日历缓存是权威，缺失时走 AkShare 兜底。不要反复请求 `trade_cal`。
- 新接口先用 `probe` 确认权限，再写抓取逻辑。不要假设「官方文档说 2000 分」就一定能批量调用。

### 5.3 PIT 与研究诚信

- 财务数据只有在 `ann_date_eff = ann_date + 1 天` 之后才可用。**禁止用 `end_date` 直接对齐交易日**。
- 任何因子/回测如果用到财务字段，必须走 `pit.py` 的事件表或面板；不要在因子代码里重新 merge 原始财报。
- 收益率用后复权价 `close_adj = close × adj_factor`；市值、换手率、涨跌停价用不复权价。两者不要混。
- 股票池必须用**历史**中证500 成分股（`index_weight` 月度快照），不能用今天的成分股回溯。
- 回测必须扣成本（默认单边 15bp），报告净收益；不要只报毛收益。
- 基准目标是用中证500 **全收益**指数。当前 `index.parquet` 是价格指数；回测时必须处理分红差异，或在报告中明确声明。
- 不允许为了「结果好看」回头改样本外参数。样本外只验证一次。

### 5.4 输出契约

- `data/interim/prices.parquet`、`universe.parquet`、`fundamental_events.parquet`、`fundamental_panel.parquet` 是下游依赖的契约。
- 改列名、改主键、改日期含义，必须同时改所有消费方和 `README.md`，并说明迁移方式。
- 不要在 `fetch.py` 里重命名 Tushare 原始字段；字段清洗放在 transform 层。

## 6. Testing requirements

- 测试放 `tests/`，用 pytest；用**合成的小 DataFrame**，不要依赖 `data/raw/` 的真实数据。
- 数据层至少覆盖：
  - TTM：Q1–Q4 + 次年 Q1，手算期望值；
  - PIT：断言 `(panel["ann_date_eff"] > panel["date"]).sum() == 0`；
  - 股票池：`(date, code)` 无重复，每日数量在合理范围，ST/停牌标记不全是 False；
  - 回测（后续）：权重漂移、成本记账恒等式。
- 先跑最窄的测试（`pytest tests/test_pit.py -q`），再跑全量 `pytest -q`。
- **不要删除或 skip 失败的测试来让 CI 变绿**。要么修代码，要么在测试里写清 `xfail` 原因。

## 7. Security and data boundaries

- 不要打印、提交、记录 token；不要把它放进 URL、日志、报错信息。
- 不要提交 `data/`、`results/`、`reports/` 的产物、`config/local.yaml`、`.venv/`。
- 不要把原始数据或加工数据上传到外部服务（包括在线的 notebook、pastebin、AI 对话）。
- 以下操作**先问**，不要自行决定：
  - 新增生产依赖；
  - 修改 `data/interim` 的输出 schema；
  - 删除 `data/raw/` 或运行 `fetch.py --force`；
  - 跑全量 `fetch.py all`（数小时、消耗配额）；
  - 修改成本假设、基准、股票池定义。

## 8. Agent workflow

- 开始前：`git status`，读 `README.md` 和相关模块；不要覆盖用户手动改过的内容。
- 改动后：跑最相关的检查（`check_env` / `check_raw` / 单测 / 单脚本），确认日志里的关键数字，再谈「完成」。
- 不要为了「看起来完成了」而跳过验证；未验证的结论在 `README.md` 中标 `[UNVERIFIED]`。
- 一次只做一件事。数据层没验收，不要开始写因子层。
- 结束一轮工作时，仅在 `README.md` 更新有实质变化的当前状态、验证结果、已知局限与下一步计划；不再维护单独的交接文档。

## 9. Definition of done

一个数据/研究脚本算「完成」，至少满足：

- 从项目根目录、`mfl` 环境下可以直接运行；
- 日志给出关键 sanity 值（缺失率、覆盖率、行数、泄漏计数）；
- 不修改 `data/raw/`；
- 有对应的窄测试，或明确说明为什么暂时没有；
- 行为变化同步到 `README.md`；
- 没有把 token、数据、结果提交进 git。
