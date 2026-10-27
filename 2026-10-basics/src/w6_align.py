"""
W6 第 5 段：把你自己手写的 MHA 对齐到 nn.MultiheadAttention
验收：随机输入下最大绝对误差 < 1e-5

运行（PowerShell）：
    & "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_align.py

TODO：4 行权重搬运。别急着抄，先想清楚形状为什么那样。
"""

import math
import torch
import torch.nn as nn

torch.manual_seed(0)

n, E, h = 3, 4, 2          # token 数 / d_model / 头数
d_k = E // h               # 2

# bias=False：第一遍先把变量减到一个（"一次只改一个变量"）
mha = nn.MultiheadAttention(embed_dim=E, num_heads=h, batch_first=True,
                            bias=False, dropout=0.0)
mha.eval()                 # 关掉 dropout，否则每次跑结果都不同

X = torch.randn(1, n, E)   # batch_first=True → (B, n, E)

# ---------- 官方实现 ----------
with torch.no_grad():
    out_off, attn_off = mha(X, X, X, need_weights=True, average_attn_weights=False)
out_off = out_off.squeeze(0)                            # (n, E)
print("官方输出形状:", tuple(out_off.shape))
print("官方注意力形状:", tuple(attn_off.shape), "← (B, h, n, n)，每个头一张 n×n")

# ---------- 把官方内部权重偷出来 ----------
print("\n=== 官方内部权重 ===")
print("in_proj_weight :", tuple(mha.in_proj_weight.shape), "← (3E, E) = (12, 4)")
print("out_proj.weight:", tuple(mha.out_proj.weight.shape), "← (E, E) = (4, 4)")

w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)      # 纵向切 3 块，顺序 Q/K/V
print("chunk(3) 后 → w_q", tuple(w_q.shape), " w_k", tuple(w_k.shape), " w_v", tuple(w_v.shape))

# ============================================================
# TODO：4 行。把官方权重转成"你的约定"。
#
# 你的实现：  Q = X @ W_Q          → W_Q 必须是 (in, out) = (E, E)
# 官方实现：  Q = X @ w_q.T + b    → nn.Linear 的权重存的是 (out, in)，所以它自己转置了
#
# w_q / w_k / w_v / out_proj.weight 全都是 (out, in) = (E, E)
# 你要的 W_Q / W_K / W_V / W_O 全都是 (in, out) = (E, E)
# 形状一样，数值不一样 —— 所以形状检查抓不到这个错。
#
# W_Q = ...
# W_K = ...
# W_V = ...
# W_O = ...
# ============================================================
W_Q = w_q.t()
W_K = w_k.t()
W_V = w_v.t()
W_O = mha.out_proj.weight.t()
def my_mha(X, W_Q, W_K, W_V, W_O, h, dk_override=None):
    """第 4 段那套，原样搬过来。dk_override 只用于后面的"验尸"。"""
    n, E = X.shape
    dk = E // h if dk_override is None else dk_override

    Q, K, V = X @ W_Q, X @ W_K, X @ W_V                 # (n, E)
    Qh, Kh, Vh = Q.view(n, h, d_k), K.view(n, h, d_k), V.view(n, h, d_k)

    heads, weights = [], []
    for i in range(h):
        qi, ki, vi = Qh[:, i, :], Kh[:, i, :], Vh[:, i, :]
        scores = qi @ ki.transpose(-2, -1) / math.sqrt(dk)
        w = torch.softmax(scores, dim=-1)
        weights.append(w)
        heads.append(w @ vi)                            # (n, d_k)

    cat = torch.cat(heads, dim=-1)                      # (n, E)
    return cat @ W_O, weights


out_mine, weights_mine = my_mha(X.squeeze(0), W_Q, W_K, W_V, W_O, h)
diff = (out_mine - out_off).abs().max().item()

print("\n=== 对齐结果 ===")
print("我的输出形状:", tuple(out_mine.shape), " 官方:", tuple(out_off.shape))
print(f"最大绝对误差 = {diff:.3e}")
print("验收线 1e-5 →", "PASS" if diff < 1e-5 else "FAIL")

print("\n我的输出第 0 行:", [round(v, 6) for v in out_mine[0].tolist()])
print("官方输出第 0 行:", [round(v, 6) for v in out_off[0].tolist()])

print("\n=== 逐头比对注意力权重（比总输出更细）===")
for i in range(h):
    mine = [round(v, 6) for v in weights_mine[i][0].tolist()]
    off = [round(v, 6) for v in attn_off[0, i, 0].tolist()]
    same = torch.allclose(weights_mine[i], attn_off[0, i], atol=1e-6)
    print(f"头{i} 第0行  我: {mine}   官方: {off}   一致: {same}")


# ============ 验尸：三种典型错法分别差多少 ============
print("\n=== 验尸：错了会差多少 ===")
bad1, _ = my_mha(X.squeeze(0), mha.in_proj_weight[0:E], W_K, W_V, W_O, h)
bad2, _ = my_mha(X.squeeze(0), W_Q, W_K, W_V, mha.out_proj.weight, h)
bad3, _ = my_mha(X.squeeze(0), W_Q, W_K, W_V, W_O, h, dk_override=E)
print(f"W_Q 没转置（形状照样 4×4） : {(bad1 - out_off).abs().max():.3e}")
print(f"W_O 没转置（形状照样 4×4） : {(bad2 - out_off).abs().max():.3e}")
print(f"√d_k 写成 √E（4 而不是 2） : {(bad3 - out_off).abs().max():.3e}")
print("→ 三种错都落在 1e-2 ~ 1e-1 量级，比 1e-5 大 3—4 个数量级。")
print("→ 所以'误差 < 1e-5'这条线不是形式主义：它把精度问题和算法问题分得干干净净。")
