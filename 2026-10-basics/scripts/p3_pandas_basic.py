# -*- coding: utf-8 -*-
# p3_pandas_basic.py —— pandas 第 3 段教学脚本
# 用法：conda activate aigc && cd 2026-10-basics && python scripts/p3_pandas_basic.py
# 每段 print 都有标题，跑完对照下面讲解逐段理解。

import pandas as pd

# ========== 0. 心智模型 ==========
# pandas 只有两个核心东西：
#   Series   = 一列（带名字的一维数据）
#   DataFrame = 一张表（很多列拼起来 = 多个 Series 共享一个行索引）
# 索引 index = 每一行的"名字"，默认是 0,1,2,...（记住这个，后面 loc 要用）

print('=' * 50)
print('1. 读入数据')
print('=' * 50)

# lines=True 表示"每行一个 JSON 对象"，逐行解析
# 不加 lines=True 会把这 3 行当成一个整体 JSON 数组去解析 -> 报错或错乱
df = pd.read_json('data/clean.jsonl', lines=True)

print(df)
print('shape:', df.shape)          # (行数, 列数) -> (3, 2)
print('列名:', list(df.columns))   # ['id', 'text']

print()
print('=' * 50)
print('2. 认识你的表：info / describe / head')
print('=' * 50)

print(df.info())   # 体检报告：每列类型 + 非空个数
# 输出会显示：id -> int64（整数），text -> object（字符串）
# 为什么 text 是 object？因为 pandas 里字符串列统一叫 object

print(df.describe())  # 只统计【数值列】id：count/mean/std/min/max
# 为什么没有 text？因为 describe 默认只看数值列，字符串列统计无意义

print(df.head(2))   # 前 2 行（数据大时不刷屏；这里 3 行看不出区别，但养成习惯）

print()
print('=' * 50)
print('3. 选列：df[方括号] 的第一种用法')
print('=' * 50)

s = df['text']        # 传【单个字符串】-> 返回 Series（一列）
print('df[\'text\'] 类型:', type(s).__name__)

d = df[['id', 'text']]  # 传【字符串列表】-> 返回 DataFrame（多列）
print('df[[\'id\',\'text\']] 类型:', type(d).__name__)
# 关键区别：单括号=一列(Series)，双括号=多列(DataFrame)
# 这是 pandas 新手第一大坑：df['text'] 不是 df[['text']]

print()
print('=' * 50)
print('4. 布尔过滤：df[方括号] 的第二种用法')
print('=' * 50)

# 先造一个新列：每行文本的长度（str.len() 对整列逐个取长度）
df['text_len'] = df['text'].str.len()
print(df)

# 过滤：方括号里放一个【布尔 Series】——逐行问"长度 > 10 吗？"
mask = df['text_len'] > 10
print('布尔 Series（mask）：')
print(mask)            # 输出 True/False，每行一个判断结果
print('类型:', type(mask).__name__)

filtered = df[mask]    # 只保留 mask 为 True 的行
print('过滤后：')
print(filtered)
# 拆开看：df['text_len'] > 10 先产生 True/False 列表，
# 然后 df[这个列表] 把 True 对应的行挑出来 -> 和 SQL 的 WHERE 完全一样

print()
print('=' * 50)
print('5. 链式思维：pandas 一切操作都返回新对象，可以一直串')
print('=' * 50)

# 举例：先过滤出长文本，再按 id 排序，再取前 1 行
top = df[df['text_len'] > 5].sort_values('id', ascending=False).head(1)
print(top)
# 从左往右读：过滤 -> 排序 -> 截取，每步输入输出都是 DataFrame
# 这就是 pandas 的核心心智：把操作当"管道"，一段一段接

print()
print('=' * 50)
print('6. loc vs iloc：按标签 vs 按位置')
print('=' * 50)

# 正常情况下两者一样（索引恰好是 0,1,2）
print('loc[0]:', df.loc[0]['text'][:5], '...')
print('iloc[0]:', df.iloc[0]['text'][:5], '...')

# 关键场景：过滤后索引"缺了号"
sub = df[df['text_len'] > 10]
print('过滤后的索引:', list(sub.index))   # 比如 [2]——第 1、2 行被删了
# sub.loc[1] 会报 KeyError！因为索引里没有"标签 1"
# sub.iloc[0] 正常——它按【物理位置】取，永远从 0 开始数
print('iloc[0] 拿到的是原表的第几行？看它的 id:', sub.iloc[0]['id'])
# 结论：
#   df.loc[标签]   -> 按行名字取（索引可能乱）
#   df.iloc[位置]  -> 按行位置取（永远 0,1,2,...）
