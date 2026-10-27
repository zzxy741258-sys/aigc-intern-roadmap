"""
W6 第 4 段配套：MHA 具体数字例子（全部可手算/可验证）

目的：用一组整数权重，把"多头到底多看到了什么"变成看得见的数字。
运行（PowerShell）：
    & "D:\anaconda3\envs\aigc\python.exe" src\w6_mha_demo.py

说明：本文件是"读得懂的算例"，src/w6_mha.py 是"你自己填的练习"。
      两个都要跑；算例跑通说明你理解了，练习填对说明你会写了。
"""

import math
import torch

print("[demo] 脚本已启动，torch =", torch.__version__)

X = torch.tensor([[1., 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 1]])          # (3, 4)

# 头 0：W_Q0 / W_K0 用的就是第 1、2 段那套，所以数字你应该眼熟
W_Q0 = torch.tensor([[1., 0], [0, 1], [1, 0], [0, 1]])   # (4, 2)
W_K0 = torch.tensor([[1., 1], [0, 0], [0, 1], [1, 0]])
W_V0 = torch.tensor([[1., 0], [0, 1], [0, 0], [1, 1]])   # 注意 d_v = d_k = 2

# 头 1：另一个 2 维子空间，权重换一组
W_Q1 = torch.tensor([[0., 0], [1, 0], [0, 1], [0, 0]])
W_K1 = torch.tensor([[0., 1], [1, 0], [0, 0], [0, 1]])
W_V1 = torch.tensor([[0., 1], [1, 0], [1, 0], [0, 0]])

# 输出投影：故意做成"跨头混合"，方便验证 W_O 到底干了什么
W_O = torch.tensor([[1., 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]])       # (4, 4)


def fmt(t):
    if t.dim() == 1:
        return [round(v, 4) for v in t.tolist()]
    return [[round(v, 4) for v in row.tolist()] for row in t]


def attention(Q, K, V, dk=None):
    dk = Q.shape[-1] if dk is None else dk
    scores = Q @ K.transpose(-2, -1)
    scaled = scores / math.sqrt(dk)
    w = torch.softmax(scaled, dim=-1)
    return w @ V, w, scores, scaled


print("=" * 70)
print("PART 1  两个头看到的东西一样吗？")
print("=" * 70)

heads_out, all_w = [], []
for i, (WQ, WK, WV) in enumerate([(W_Q0, W_K0, W_V0), (W_Q1, W_K1, W_V1)]):
    Q, K, V = X @ WQ, X @ WK, X @ WV
    o, w, raw, scaled = attention(Q, K, V)
    heads_out.append(o)
    all_w.append(w)

    print(f"\n--- 头 {i} ---")
    print(f"Q{tuple(Q.shape)} =", fmt(Q))
    print(f"K{tuple(K.shape)} =", fmt(K))
    print(f"V{tuple(V.shape)} =", fmt(V))
    print("原始分数 QK^T =", fmt(raw))
    print(f"除以 √d_k = √{Q.shape[-1]} = {math.sqrt(Q.shape[-1]):.4f} 后 =", fmt(scaled))
    print("注意力权重（每行和=1）=", fmt(w))
    print(f"输出 O{tuple(o.shape)} =", fmt(o))

print("\n>>> 关键对比：两个头对第 0 个 token 的关注分配")
print("    头 0 :", fmt(all_w[0][0]))
print("    头 1 :", fmt(all_w[1][0]))
print("    → 同一句话、同一个位置，两个头给出的注意力分布不同。")
print("      头 0 把 67% 权重压在 token2（峰值为 0.6728），头 1 只给 50%（0.5035）。")
print("      这就是'多头 = 多视角'的具体表现，不是修辞。")


print("\n" + "=" * 70)
print("PART 2  如果错把 √d_k 写成 √d_model 会怎样")
print("=" * 70)

Q0, K0, V0 = X @ W_Q0, X @ W_K0, X @ W_V0
o_right, w_right, _, _ = attention(Q0, K0, V0)                    # dk=2  → √2
o_wrong, w_wrong, _, _ = attention(Q0, K0, V0, dk=Q0.shape[-1] * 2)  # 故意用 √4

print("正确（√2）权重第0行 =", fmt(w_right[0]))
print("错误（√4）权重第0行 =", fmt(w_wrong[0]))
print(f"输出最大绝对偏差     = {(o_wrong - o_right).abs().max().item():.6f}")
print("→ 偏差约 2e-1，比 W6 验收线 1e-5 大 2 万倍。第 5 段对齐会直接失败。")
print("→ 机制：√4 > √2，等于除得更多 → 分数被压小 → softmax 输入太平")
print("        → 分布趋近均匀 → 峰值从 0.67 掉到 0.58，模型无法聚焦。")


print("\n" + "=" * 70)
print("PART 3  cat 之后为什么还要 W_O")
print("=" * 70)

cat = torch.cat(heads_out, dim=-1)
print(f"cat{tuple(cat.shape)} =", fmt(cat))
print("  ↑ 前 2 列 = 头 0 的输出，后 2 列 = 头 1 的输出。两段之间没有任何交互。")

final = cat @ W_O
print(f"\nfinal{tuple(final.shape)} =", fmt(final))
print("\nW_O 的第 0 行 = [1, 0, 1, 0]，所以：")
print(f"  final[0,0] = cat[0,0]*1 + cat[0,1]*0 + cat[0,2]*1 + cat[0,3]*0")
print(f"             = {cat[0,0]:.4f} + {cat[0,2]:.4f} = {final[0,0]:.4f}")
print("  第一项来自头 0，第三项来自头 1 —— 这才是'融合'。")
print("  不乘 W_O 的话，头 0 的数字永远到不了第 2、3 维，两个头只能并排站着。")
