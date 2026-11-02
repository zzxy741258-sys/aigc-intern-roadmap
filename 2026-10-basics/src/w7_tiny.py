# -*- coding: utf-8 -*-
"""
W7 第 4 段·降维版：把 nn.Module 的外壳剥掉，只用你最熟的张量操作，
把「字符 → 向量 → 打分 → loss」走一遍，一共 5 步，每步打印一个形状。

这一段刻意不用 nn.Embedding / nn.Linear / nn.Module ——
它们只是把下面这几行包装了一层，理解了这里，那些类就只是语法糖。

运行：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_tiny.py
"""
import math

import torch

torch.manual_seed(0)

V = 1220   # 词表大小（1220 个不同字符）
d = 8      # 每个字符用几维向量表示（真正的模型用 64，这里压到 8 便于看清）
B = 3      # 一次处理 3 个字符（真正的 batch 是 32）

print("=" * 62)
print("模型到底在干什么：5 步，全程只有张量乘法")
print("=" * 62)

# ---------------------------------------------------------------- ① 查表
# 一张 (1220, 8) 的表，第 i 行就是第 i 号字符的「身份证」。
# 所谓「嵌入层」，就是拿整数 id 去这张表里取对应的行。
emb = torch.randn(V, d) * 0.02       # 先随便初始化，尺度 0.02 稍后解释

ids = torch.tensor([5, 100, 7])      # 3 个字符的编号
x = emb[ids]                         # 查表：取第 5、100、7 行
print("\n① 查表（嵌入）")
print("   emb 形状      : %s   ← 1220 行 8 列" % (tuple(emb.shape),))
print("   ids           : %s" % ids.tolist())
print("   x = emb[ids]  : %s   ← 3 行 8 列" % (tuple(x.shape),))
print("   含义          : 3 个字符，每个变成一个 8 维向量")

# ---------------------------------------------------------------- ② 加位置
# Self-Attention 看不见顺序，所以再准备一张「座位表」，把座位向量加进去。
pos_emb = torch.randn(64, d) * 0.02  # 最多 64 个座位
pos = torch.arange(B)                # [0, 1, 2]
x = x + pos_emb[pos]
print("\n② 加位置编码")
print("   pos           : %s" % pos.tolist())
print("   pos_emb[pos]  : %s   ← 3 行 8 列" % (tuple(pos_emb[pos].shape),))
print("   x + pos_emb   : %s   ← 形状不变，只是每个向量被推了一下" % (tuple(x.shape),))

# ---------------------------------------------------------------- ③ 打分
# 输出头要和嵌入共享权重，所以复用同一张 emb。
# Linear 算的是 x @ W.T，共享后 W 就是 emb，所以这里是 x @ emb.T。
logits = x @ emb.T                   # (3, 8) @ (8, 1220) = (3, 1220)
print("\n③ 打分（输出头）")
print("   logits        : %s   ← 3 个字符，每个对 1220 个字符各有一个分数" % (tuple(logits.shape),))
print("   这就是「权重共享」：输出头没有自己的矩阵，直接用嵌入表转置")

# ---------------------------------------------------------------- ④ softmax
probs = torch.softmax(logits, dim=-1)
print("\n④ softmax 成概率")
print("   第 0 行前 5 个概率: %s" % [round(v, 6) for v in probs[0, :5].tolist()])
print("   均匀分布时每个应为 1/1220 = %.6f" % (1 / V))

# ---------------------------------------------------------------- ⑤ 交叉熵
targets = torch.tensor([6, 101, 8])          # 正确字符分别是 6、101、8 号
p_true = probs[torch.arange(B), targets]     # 取出「正确字符」那一个概率
loss = -torch.log(p_true).mean()             # 交叉熵 = -log(正确项概率) 的平均

print("\n⑤ 交叉熵 = -log(正确字符的概率)")
print("   正确字符的概率: %s" % [round(v, 6) for v in p_true.tolist()])
print("   loss          : %.4f" % loss.item())
print("   ln(1220)      : %.4f" % math.log(V))
print("   差值          : %.4f" % abs(loss.item() - math.log(V)))

# ---------------------------------------------------------------- 反面实验
print("\n" + "=" * 62)
print("反面实验：把初始化尺度从 0.02 改成 1.0，只改这一个数")
print("=" * 62)
emb2 = torch.randn(V, d) * 1.0
x2 = emb2[ids]
logits2 = x2 @ emb2.T
probs2 = torch.softmax(logits2, dim=-1)
p2 = probs2[torch.arange(B), targets]
loss2 = -torch.log(p2).mean()
print("   logits 跨度（最大-最小）: %.4f   （0.02 时是 %.4f）"
      % ((logits2.max() - logits2.min()).item(), (logits.max() - logits.min()).item()))
print("   正确字符的概率          : %s" % [round(v, 6) for v in p2.tolist()])
print("   loss                    : %.4f" % loss2.item())
print("\n   读法：概率被压到 0 附近 → -log(接近 0) 就会爆大。")
print("        所以初始化的标准差不是玄学，它决定 logits 的尺度。")
