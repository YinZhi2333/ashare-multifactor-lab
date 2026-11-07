"""ashare-multifactor-lab：A 股中证500 多因子研究框架。

当前实现集中在 data 层；建议按 common → fetch → adjust → universe → pit
阅读。后续因子和回测消费 interim 面板，不直接拼接原始行情或财报。
"""

__version__ = "0.1.0"
