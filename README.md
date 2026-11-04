# ashare-multifactor-lab

A 股中证500 多因子选股研究框架（个人项目）。

目标：用公开数据跑通「PIT 清洗 → 因子构建 → 中性化 → IC/分层检验 → 组合构建 → 扣费回测 → 稳健性归因」的完整流水线，产出可复现的代码和一份研究报告。

配套文档：

- [项目提案](../项目提案_A股多因子研究框架.md)
- [量化流程入门手册](../量化流程入门手册.md)

## 环境

Python 3.11，用 conda 管理。完整依赖见 [`environment.yml`](environment.yml)。

## 快速开始

```powershell
cd "E:\大学资料\1量化\ashare-multifactor-lab"

# 1. 创建环境（第一次约 5–10 分钟）
conda env create -f environment.yml

# 2. 激活
conda activate mfl

# 3. 注册 Jupyter 内核（推荐，否则 notebook 可能用错解释器）
python -m ipykernel install --user --name mfl --display-name "Python (mfl)"

# 4. 自检
python scripts/check_env.py
```

自检脚本运行后，如果最后一行为 `OK: imports + parquet round-trip`，表示环境配置正常。

注：
1. **如果 `conda activate` 报错**

PowerShell 里第一次用 conda，需要先初始化：

```powershell
conda init powershell
```

关掉终端重新打开，再输入命令 `conda activate mfl`。

2. **如果 conda 下载太慢**

改用 pip + 清华镜像（同样在 `mfl` 环境里）：

```powershell
conda create -n mfl python=3.11 -y
conda activate mfl
pip install numpy pandas pyarrow matplotlib scipy statsmodels jupyterlab ipykernel pyyaml pytest tushare akshare -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据抓取

数据源用 Tushare Pro。需注册账号，在个人主页拿到 token。

> 需要 2000 积分以获得必要的权限（下载数据不扣积分）。2000积分的限制频次为 200 次/分钟，财务 80 次/分钟，每天调用总量 10 万次/接口。学生可以通过 [学生积分](https://tushare.pro/document/1?doc_id=360) 免费拿 2000 分。

### 0. 先探测权限

```powershell
cd "E:\大学资料\1量化\ashare-multifactor-lab"
python src/mfl/data/fetch.py probe
```

`probe` 每个接口打一发，输出 `OK / DENIED`，不落盘。重点看：

- `daily_basic`：没有就没有市值/换手/PE/PB；
- `income`、`balancesheet`：没有就只能用 `roe_yearly`，不是真 TTM；
- `index_weight`：没有就没有历史成分股，只能拿当前成分股回测，有幸存者偏差。

### 1. 配置 token

```powershell
# 方式 A：环境变量（设置后要重开终端）
setx TUSHARE_TOKEN "你的token"

# 方式 B：本地配置文件（推荐，不进 git）
copy config\local.example.yaml config\local.yaml
# copy后，把 config/local.yaml 里的 your_token_here 换成你的 token

# 方式 C：临时用一次
python src/mfl/data/fetch.py probe --token 你的token
```

### 2. 基础数据（约 15 次请求）

```powershell
python src/mfl/data/fetch.py calendar --start 20150101 --end 20241231
python src/mfl/data/fetch.py basic
```

### 3. 行情（约 4900 次请求，30–40 分钟）

```powershell
python src/mfl/data/fetch.py price --start 20150101 --end 20241231
```

`price` 只抓 `daily` 和 `adj_factor`。每日指标单独抓：

```powershell
python src/mfl/data/fetch.py metrics --start 20150101 --end 20241231
```

### 4. 指数（约 120 次请求，先跑）

```powershell
python src/mfl/data/fetch.py index --start 20150101 --end 20241231
```

抓中证500 日线和月度成分股。**先跑这一步**：后面的财务按股票抓，会优先用这里的成分股并集（约 1500–2000 只），比全市场少一个数量级。

### 5. 财务三大表（按股票抓，约 1–1.5 小时）

```powershell
python src/mfl/data/fetch.py financials --start 20150101 --end 20241231
```

2000 积分档的 `income` / `balancesheet` / `cashflow` **只支持单只股票查询**；按报告期拉全市场要用 `income_vip` 等接口（5000 分）。所以脚本改成逐只抓：

- 有 `index_weight`：只抓中证500 历史成分股并集，3 张表约 5000–6000 次请求，1–1.5 小时；
- 没有成分股：退回全市场约 5400 只，3 张表约 1.6 万次请求，3–4 小时；
- 每 200 只落一次盘，中断后重跑会从断点继续。

可选的财务指标（ROE、毛利率、负债率）也按股票抓：

```powershell
python src/mfl/data/fetch.py fundamental --start 20150101 --end 20241231
```

### 6. 一次性下载全部

```powershell
python src/mfl/data/fetch.py all --start 20150101 --end 20241231
```

`all` 按顺序抓基础数据、行情、每日指标、指数、三大表、fina_indicator；某个接口没权限会打 ERROR 并跳过，不会中断。总共约 1.4–1.5 万次请求，2.5–3 小时（没有成分股会到 5–6 小时）。

涨跌停和停牌单独抓（可选，约 4900 次请求）：

```powershell
python src/mfl/data/fetch.py tradability --start 20150101 --end 20241231
```

### 7. 落盘结构

`data/raw/` 是原始快照，只增不改；清洗逻辑改了也不用重新下载。

```text
data/raw/
├── trade_cal/data.parquet                  # 交易日历：cal_date、is_open、pretrade_date
├── stock_basic/data.parquet                # 全部股票（L 上市 / D 退市 / P 暂停），含 list_date
├── namechange/data.parquet                 # 历史更名记录，用来还原某天的 ST 状态
├── daily/year=2024/data.parquet            # 日线：OHLC、成交量、成交额、pct_chg，按年分区
├── adj_factor/year=2024/data.parquet       # 复权因子：close_adj = close × adj_factor
├── daily_basic/year=2024/data.parquet      # 每日指标：总/流通市值、换手率、PE_TTM、PB
├── income/data.parquet                     # 利润表：营收、净利润、EPS，按报告期
├── balancesheet/data.parquet               # 资产负债表：总资产、总负债、归母净资产
├── cashflow/data.parquet                   # 现金流量表：经营现金流净额
├── stk_limit/year=2024/data.parquet        # 每日涨跌停价，判断能不能买/卖
├── suspend_d/year=2024/data.parquet        # 停复牌记录
├── index_daily/ts_code=000905.SH/data.parquet      # 中证500 日线，做基准
├── index_weight/index_code=000905.SH/data.parquet  # 月度成分股与权重，防幸存者偏差
└── fina_indicator/data.parquet             # 财务指标（可选）：ROE、毛利率、负债率
```

加工后的数据在 `data/interim/`，都可以由 raw 重建：

```text
data/interim/
├── prices.parquet                  # 后复权价、日收益、vwap、市值/换手/PE/PB
├── index.parquet                   # 中证500 收盘与日收益
├── universe.parquet                # 每个调仓日的成分股 + ST/停牌/涨跌停/可交易
├── rebalance_dates.parquet         # 月末调仓日列表
├── fundamental_events.parquet      # 按公告日排列的财务事件（含 TTM）
└── fundamental_panel.parquet       # 对齐到 (date, code) 的 PIT 面板
```

### 8. 常见报错

- **`没有访问该接口的权限` / `积分不足`**：先用 `probe` 确认；`all` 会跳过没权限的表。
- **`频率超限(1次/小时)`**：小时级限制，脚本直接抛错；日历会自动改用 AkShare 生成并缓存。
- **`每分钟最多访问`**：脚本自动等 60 秒重试；嫌慢把 `--sleep` 调大。
- **空 DataFrame**：先确认那天是不是交易日，非交易日返回空是正常的。
- **`必填参数, ts_code`**：2000 分档的 `income` / `balancesheet` / `cashflow` 只能按单只股票查询；脚本已改成逐只抓。按报告期拉全市场需要 `*_vip` 接口（5000 分）。

## 数据加工：复权 → 股票池 → PIT

抓完 raw 数据后，按顺序跑三个脚本（都在项目根目录）：

```powershell
# 1. 复权 + 收益 + 市值/换手
python src/mfl/data/adjust.py --start 20150101 --end 20241231

# 2. 中证500 月度股票池：ST / 上市天数 / 停牌 / 涨跌停
python src/mfl/data/universe.py --start 20150101 --end 20241231

# 3. 财务 PIT：事件表 + as-of 面板
python src/mfl/data/pit.py --start 20150101 --end 20241231
```

输出：

| 文件 | 内容 |
|---|---|
| `data/interim/prices.parquet` | 后复权价、日收益、vwap、市值、换手、PE/PB |
| `data/interim/index.parquet` | 中证500 收盘与日收益 |
| `data/interim/universe.parquet` | 每个调仓日的成分股与可交易标记 |
| `data/interim/fundamental_events.parquet` | 按公告日排列的财务事件 |
| `data/interim/fundamental_panel.parquet` | 对齐到 (date, code) 的 PIT 面板 |

三个脚本都只读 raw、写 interim，可以单独重跑，不改原始数据。

### 复权脚本

- 收益用后复权价 `close × adj_factor`；市值、换手、涨跌停用不复权价；
- 生成 `vwap = amount × 10 / volume`（Tushare 的 vol 是手、amount 是千元）；
- 会打印复权收益与 `pct_chg` 的相关系数，低于 0.99 会警告。

### 股票池脚本

- 每个自然月最后一个交易日取一次成分股快照（最近一期 `index_weight`）；
- 用 `namechange` 还原历史名称，判断 ST；
- 按交易日历算上市天数，剔除上市不足 120 个交易日的次新股；
- 用 `prices.parquet` 判断当天是否停牌；
- 用 `stk_limit` 标记涨停买不进 / 跌停卖不出。

### PIT 脚本

- 先把累计净利润拆成单季，再按 `TTM = 本期累计 + 上年年报 − 上年同期累计` 算 TTM；
- 用 `ann_date + 1 天` 作为可用时间，`merge_asof` 对齐到调仓日；
- 同一报告期有修正公告时，保留最新版本（严格的逐版本 PIT 是后续改进项）；
- 没有 `income` / `balancesheet` 时退化为 `roe_yearly`，日志会明确警告。

## 目录结构

> 项目提案里数据层叫 `data_pipeline/`，实现时简化成了 `data/`，职责一样：下载、复权、PIT、股票池。

### 现在的目录

```text
ashare-multifactor-lab/
├── README.md
├── environment.yml               # conda 环境：Python 3.11 + 数据栈
├── .gitignore                    # 忽略数据、环境、token
├── config/
│   ├── base.yaml                 # 全局参数：区间、股票池、成本、调仓频率
│   ├── local.example.yaml        # token 模板
│   └── local.yaml                # 本地 token（自己创建，不入库）
├── data/                         # 所有数据，不入库
│   ├── raw/                      # 原始快照，只增不改（见第 7 节）
│   ├── interim/                  # 清洗后的 parquet，可由脚本重建
│   └── meta/                     # 交易日历、行业分类等元数据（后续用）
├── src/mfl/
│   ├── __init__.py
│   └── data/                     # 数据层（已完成）
│       ├── __init__.py
│       ├── common.py             # 公共路径、配置、读写、日志
│       ├── fetch.py              # 下载 Tushare 原始数据
│       ├── adjust.py             # 复权、收益、行情面板
│       ├── universe.py           # 中证500 股票池与可交易性
│       └── pit.py                # 财务 PIT 对齐
├── scripts/
│   └── check_env.py              # 环境自检：包版本 + parquet 读写
├── notebooks/                    # 探索用，不放核心逻辑（待建）
├── tests/                        # 单元测试（待建）
├── reports/                      # 图表与研究报告（待建）
└── results/                      # 每次实验的输出（待建）
```

### 每个脚本的用途

| 脚本 | 用途 | 读什么 | 写什么 |
|---|---|---|---|
| `src/mfl/data/fetch.py` | 下载原始数据，只落盘不清洗 | Tushare | `data/raw/` |
| `src/mfl/data/adjust.py` | 后复权、算日收益、合并市值/换手 | `raw/daily`、`adj_factor`、`daily_basic` | `prices.parquet`、`index.parquet` |
| `src/mfl/data/universe.py` | 月末中证500 股票池 + ST/停牌/涨跌停 | `raw/index_weight`、`stock_basic`、`namechange` | `universe.parquet`、`rebalance_dates.parquet` |
| `src/mfl/data/pit.py` | 财报按公告日对齐，算 TTM 与 ROE_TTM | `raw/income`、`balancesheet`、`fina_indicator` | `fundamental_events.parquet`、`fundamental_panel.parquet` |
| `src/mfl/data/common.py` | 公共路径、配置、读写、日志 | — | — |
| `scripts/check_env.py` | 检查环境是否装好 | — | 终端输出 |

### 后续要补的模块

按项目提案的分层，因子之后的部分还没写：

```text
src/mfl/
├── factors/                      # 因子层（待建）
│   ├── base.py                   # 因子基类与注册表
│   ├── value.py                  # EP、BP
│   ├── quality.py                # ROE_TTM
│   └── price_volume.py           # 12-1 动量、1 月反转、20 日换手
├── preprocess/                   # 预处理（待建）
│   ├── winsorize.py              # MAD 去极值
│   ├── standardize.py            # 横截面 z-score
│   └── neutralize.py             # 行业/市值中性化
├── evaluation/                   # 评价（待建）
│   ├── ic.py                     # IC / RankIC / ICIR
│   ├── quantile.py               # 分层回测
│   └── turnover.py               # 换手率
├── portfolio/                    # 组合（待建）
│   ├── combine.py                # 多因子合成
│   └── construct.py              # 打分 → 目标权重
├── backtest/                     # 回测（待建）
│   ├── engine.py                 # 权重漂移 + 成本
│   ├── costs.py                  # 佣金、印花税、滑点
│   └── constraints.py            # 涨跌停、停牌、单票上限
└── analytics/                    # 归因与报告（待建）
    ├── metrics.py                # 年化、Sharpe、回撤、IR
    ├── attribution.py            # Brinson 归因
    └── report.py                 # 生成图表与报告
```

工程化文件（可选，待建）：`pyproject.toml`（打包安装）、`Makefile`（一键 data/test/report）、`config/factors.yaml`（因子开关与参数）。

## 当前进度

- [x] 环境与项目骨架
- [x] Tushare 抓取脚本（`src/mfl/data/fetch.py`）
- [x] 复权与行情面板（`adjust.py`）
- [x] 中证500 月度股票池（`universe.py`）
- [x] 财务 PIT 对齐（`pit.py`，三大表 TTM）
- [ ] 第一个因子：EP
- [ ] IC / 分层检验
- [ ] 回测引擎（权重漂移 + 成本 + 涨跌停）
- [ ] 稳健性与报告

## 常用命令

```powershell
conda env list                 # 看所有环境
conda deactivate               # 退出当前环境
conda env remove -n mfl        # 删掉重来（谨慎）
python scripts/check_env.py    # 环境自检
pytest -q                      # 跑测试
```

## 数据说明

数据不入库。行情与财务通过 Tushare（或 AkShare）获取，仅用于个人研究，不对外分发。原始数据按表存在 `data/raw/<表名>/`，清洗产物 `data/interim/` 可由脚本重建。
