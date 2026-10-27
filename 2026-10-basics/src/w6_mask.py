"""
W6 第 3 段补档：causal mask（因果掩码）固化脚本

背景：当时是交互式验证，代码没落盘。本文件把「mask 长什么样、遮完变成什么、
      为什么必须填 -inf 而不是 0」全部用真实数字固化下来。

运行（PowerShell）：
    & "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_mask.py
"""

import math
import torch

# ==== 沿用第 1—2 段那组手算输入，数字你应该眼熟 ====
X = torch.tensor([[1., 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 1]])                       # (3, 4)
W_Q = torch.tensor([[1., 0], [0, 1], [1, 0], [0, 1]])  # (4, 2)
W_K = torch.tensor([[1., 1], [0, 0], [0, 1], [1, 0]])  # (4, 2)
W_V = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])  # (4, 3)

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
n = X.shape[0]


def fmt(t):
    return [[round(v, 4) for v in row.tolist()] for row in t]


def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))  # noqa: E712
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights, scores


print("=" * 66)
print("PART 1  mask 长什么样")
print("=" * 66)
mask = torch.tril(torch.ones(n, n)).bool()
print("torch.tril(torch.ones(3,3)).bool() =")
for row in mask.tolist():
    print("   ", [int(v) for v in row])
print("  读法：行 = 谁在看（query），列 = 看谁（key）")
print("  第 0 行只有第 0 列是 1 → 第 0 个 token 只能看自己")
print("  第 2 行三列都是 1   → 第 2 个 token 能看前两个 + 自己")

print("\n" + "=" * 66)
print("PART 2  遮之前 vs 遮之后（缩放后的分数）")
print("=" * 66)
scores_raw = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
print("遮之前:", fmt(scores_raw))
_, w_no, s_masked = attention(Q, K, V, mask)
print("遮之后:", fmt(s_masked))
print("  → 右上三角全变成 -inf（负无穷），下三角原样保留")

print("\n" + "=" * 66)
print("PART 3  softmax 之后：每行权重和恒为 1")
print("=" * 66)
_, w_no, _ = attention(Q, K, V, None)
_, w_yes, _ = attention(Q, K, V, mask)
labels = ["第0个token（只能看自己）", "第1个token（能看0和自己）", "第2个token（全都能看）"]
for i in range(n):
    print(f"  {labels[i]}")
    print(f"    无 mask: {fmt(w_no)[i]}  行和={w_no[i].sum():.4f}")
    print(f"    有 mask: {fmt(w_yes)[i]}  行和={w_yes[i].sum():.4f}")
print("  → 第 0 行变成 [1,0,0]：只有一个可选，概率必然是 1")
print("  → 第 2 行完全没变：它本来就是最后一个，没有人被挡住")

print("\n" + "=" * 66)
print("PART 4  验尸：如果错误地用 0 填充（而不是 -inf）")
print("=" * 66)
mask_wrong = scores_raw.clone()
mask_wrong[~mask] = 0.0
w_wrong = torch.softmax(mask_wrong, dim=-1)
print("错误做法（填 0）第 0 行权重:", fmt(w_wrong)[0])
print("正确做法（填 -inf）第 0 行权重:", fmt(w_yes)[0])
print("  → 填 0 的版本，第 0 个 token 仍然把 16.4% + 16.4% = 32.7% 的注意力")
print("     分给了两个'未来'的 token —— 等于偷看答案。")
print("  → 原因：softmax(0) = e^0 = 1 ≠ 0。0 分不是'没有权重'，是'权重和第一名一样大'。")
print("  → -inf 才是真正归零：e^(-inf) = 0，权重严格为 0。")

print("\n" + "=" * 66)
print("PART 5  输出对比")
print("=" * 66)
print("无 mask 输出:")
for r in fmt(w_no @ V):
    print("   ", r)
print("有 mask 输出:")
for r in fmt(w_yes @ V):
    print("   ", r)
print("  → 第 0 行变成 V 的第 0 行 [1,0,1]：它只能看自己，输出就是自己的 value")
