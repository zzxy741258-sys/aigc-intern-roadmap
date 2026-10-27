"""
W6 第 4 段：多头注意力（MHA）手写

规则：三个 TODO 是你必须自己填的（因为 view / cat 是本段唯一的新 API）。
其余代码已给，不要重敲，跑通即可。

运行（PowerShell）：
    & "D:\anaconda3\envs\aigc\python.exe" src\w6_mha.py

验收：打印出来的 "与手算例子的头0对比" 必须全部 True
"""

import math
import torch

torch.manual_seed(0)

# ============ 设定 ============
n, d_model, h = 3, 4, 2          # 3 个 token，模型维度 4，2 个头
d_k = d_model // h               # 每头维度 = 4 // 2 = 2

X = torch.randn(n, d_model)      # (3, 4)
W_Q = torch.randn(d_model, d_model)   # (4, 4) 注意：是 4x4，不是 4x2
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)
W_O = torch.randn(d_model, d_model)   # 输出投影，(4, 4)

print(f"n={n} d_model={d_model} h={h} d_k={d_k}")


def attention(Q, K, V):
    """单头 attention —— 就是你在第 2 段写过的那 3 行，一行都没变。"""
    dk = Q.shape[-1]                                   # 这里的 dk = 2，不是 4
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)    # (n, n)
    w = torch.softmax(scores, dim=-1)
    return w @ V


# ============ ① 三个线性投影 ============
Q = X @ W_Q    # (3,4)@(4,4) = (3,4)
K = X @ W_K    # (3,4)
V = X @ W_V    # (3,4)
print("\n① 投影后  Q/K/V:", tuple(Q.shape), tuple(K.shape), tuple(V.shape))


# ============ ② 拆头：把最后那个 4 维切成 2 份 ============
# TODO-1：用 view 把 Q/K/V 从 (n, d_model) 变成 (n, h, d_k)
#         提示：4 要拆成 2×2，所以是 Q.view(?, ?, ?)
#         填完把下面三行的注释去掉
Qh = Q.view(n, h, d_k)
Kh = K.view(n, h, d_k)
Vh = V.view(n, h, d_k)
# 下面这行用来验证你拆对没有：它必须等于第 2 段手算的 Q = [[2,0],[0,2],[1,2]]
Q_manual_ref = torch.tensor([[2., 0.], [0., 2.], [1., 2.]])
X_manual = torch.tensor([[1., 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1]])
W_Q4_manual = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0]])
Q4_manual = X_manual @ W_Q4_manual
head0_of_manual = Q4_manual.view(n, h, d_k)[:, 0, :]
print("② 手算构造里的头0 是否等于第2段的 Q:",
      torch.allclose(head0_of_manual, Q_manual_ref, atol=1e-6))


# ============ ③ 每个头各算各的 attention ============
heads_out = []
for i in range(h):
    qi = Qh[:, i, :]     # (3, 2)  第 i 个头的 Q
    ki = Kh[:, i, :]     # (3, 2)
    vi = Vh[:, i, :]     # (3, 2)
    oi = attention(qi, ki, vi)   # (3, 2)
    heads_out.append(oi)
    print(f"③ 头{i}: Q{tuple(qi.shape)} K{tuple(ki.shape)} V{tuple(vi.shape)} -> O{tuple(oi.shape)}")


# ============ ④ 拼接：把 h 个 (n, d_k) 拼回 (n, d_model) ============
# TODO-2：把 heads_out 这个列表沿"最后一维"拼接
#         提示：torch.cat(列表, dim=?)
cat = torch.cat(heads_out, dim=-1)

print("④ 拼接后:", tuple(cat.shape))     # 应该是 (3, 4)


# ============ ⑤ 输出投影 ============
final = cat @ W_O      # (3,4)@(4,4) = (3,4)

# TODO-3（写在纸上/回复里，不用写进代码）：
#   如果不乘 W_O，直接用 cat 当输出，形状也能跑通（也是 (3,4)）。
#   那 W_O 到底在做什么？不乘会丢什么？
#   提示：看 cat 的前 2 列全部来自头 0，后 2 列全部来自头 1 —— 头与头之间有交流吗？

print("⑤ final:", tuple(final.shape))
print("\nfinal =\n", final)
