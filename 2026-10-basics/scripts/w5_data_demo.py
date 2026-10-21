# -*- coding: utf-8 -*-
# w5_data_demo.py —— 模拟实习任务的数据处理流水线
# 用法：conda activate aigc && cd "D:\Agent learning\aigc-intern-roadmap\2026-10-basics" && python scripts/w5_data_demo.py
import pandas as pd
import numpy as np

# ---------- Step 0: 造 20 行订单 + 5 行客户，存成 CSV ----------
# np.random.seed(0) 固定随机数种子：每次运行生成的数据一模一样（实验可复现）
np.random.seed(0)
n = 20
orders = pd.DataFrame({
    'order_id': range(101, 101 + n),                 # 101~120
    'customer_id': np.random.randint(1, 6, n),       # 客户 1~5 随机
    'amount': np.random.randint(50, 300, n)          # 金额 50~299 随机
})
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['张三', '李四', '王五', '赵六', '钱七'],
    'city': ['北京', '上海', '广州', '深圳', '杭州']
})
orders.to_csv('data/orders.csv', index=False)        # 落盘（index=False：不写行号列）
customers.to_csv('data/customers.csv', index=False)

# ---------- Step 1: 从 CSV 读回来（真实工作就是读现成的 CSV）----------
o = pd.read_csv('data/orders.csv')   # 20 行
c = pd.read_csv('data/customers.csv') # 5 行

# ---------- Step 2: 过滤 —— 只要金额 > 100 的订单（= SQL WHERE）----------
big = o[o['amount'] > 100]

# ---------- Step 3: 分组统计 —— 每个客户的总金额 + 订单数（= SQL GROUP BY）----------
stats = big.groupby('customer_id')['amount'].agg(['sum', 'count'])
stats.columns = ['total_amount', 'order_count']   # 给结果列改名
print(stats)
stats = stats.reset_index()   # 分组键从"索引"变成普通列（对应 SQL 里 GROUP BY 键作为 SELECT 列）
print('分组统计（按客户）:')
print(stats)

# ---------- Step 4: 拼上客户信息（= SQL LEFT JOIN）----------
# how='left'：以 stats（左表）为准，一行不丢；客户 1~5 都在 customers 里，所以都能匹配上
result = pd.merge(stats, c, on='customer_id', how='left')

# ---------- Step 5: 输出结果表 ----------
result.to_csv('data/customer_stats.csv', index=False)
print('\n最终结果:')
print(result)
