import torch
import math

# ===== 第 1 步：复现手算例子的输入（和上一段完全一致）=====
X = torch.tensor([[1., 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 1]])            # (3, 4) 三个 token，每个 4 维

W_Q = torch.tensor([[1., 0], [0, 1], [1, 0], [0, 1]])  # (4, 2)
W_K = torch.tensor([[1., 1], [0, 0], [0, 1], [1, 0]])  # (4, 2)
W_V = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])  # (4, 3)

# 线性投影：矩阵乘法 @，形状法则 (3,4)@(4,2)=(3,2)
Q = X @ W_Q    # (3, 2)  我找什么
K = X @ W_K    # (3, 2)  我是什么
V = X @ W_V    # (3, 3)  我传什么

print("Q:\n", Q)          # 期待 [[2,0],[0,2],[1,2]]
print("K:\n", K)          # 期待 [[1,2],[1,0],[2,1]]
print("V:\n", V)          # 期待 [[1,0,1],[1,1,1],[2,1,1]]

# ===== 第 2 步：手写 attention 核心（就 3 行，对应公式）=====
def attention(Q, K, V):
    d_k = Q.shape[-1]                     # 特征维 d_k
    scores = Q @ K.T / math.sqrt(d_k)     # 相似度 (n,n)：Q@K.T 是 (3,2)@(2,3)=(3,3)
    weights = torch.softmax(scores, dim=-1)  # 按"行"归一化，每行和=1
    return weights @ V, weights           # 输出 (n,d_v)，顺带返回权重看分布

out, weights = attention(Q, K, V)

print("\n注意力权重 weights:\n", weights)   # 期待第 0 行 ≈ [0.16, 0.16, 0.67]
print("\n输出 out:\n", out)                # 期待 out[0] ≈ [1.67, 0.84, 1.00]
