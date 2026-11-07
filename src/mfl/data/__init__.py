"""数据层：将原始快照加工成可复现、按历史时点可用的研究输入。

fetch 负责联网抓数及 raw 落盘；adjust、universe、pit 只读本地 raw，
分别产出行情、月末股票池、财务事件及 PIT（point-in-time）面板。
执行顺序为 adjust → universe → pit，其中 PIT 面板沿用股票池的主键。
这些模块目前作为脚本运行，公共工具通过同目录的 common 导入。
"""
