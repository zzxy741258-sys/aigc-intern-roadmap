import numpy as np
np.random.seed(0)

# ── 1. 数据：三类高斯点 ──
def make_data():
    centers = [(0,0),(3,0),(0,3)]
    X_list, y_list = [], []
    for c in range(3):
        cx, cy = centers[c]
        pts = np.random.randn(200, 2)*0.5 + [cx, cy]   # 200 个点平移到中心附近
        X_list.append(pts)
        y_list.append(np.full(200, c))
    return np.vstack(X_list), np.concatenate(y_list)

X, y = make_data()       # X:(600,2)  y:(600,)
N, d = X.shape           # N=600, d=2
K = 3

# ── 2. one-hot：第 i 行在 y[i] 位置是 1 ──
Y = np.eye(K)[y]         # (600,3)

# ── 3. 参数与超参 ──
W = np.zeros((d, K))     # (2,3)：2 个特征 × 3 个类
b = np.zeros((K,))       # (3,)
lr = 0.1

# ── 4. 训练循环 ──
for i in range(200):
    z = X @ W + b                    # (600,3)，3 列 = 3 个类的分数
    # ★ 空1：softmax（数值稳定版，提示：先减每行最大值，再 exp，再除以行和）
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)

    # ★ 空2：交叉熵（提示：p[np.arange(N), y] 取出每个样本真实类的概率，再取 log 取负求平均）
    loss = -np.mean(np.log(p[np.arange(N), y]))

    # ★ 空3：梯度（提示：dW = X.T @ (p - Y) / N，db = (p - Y) 按列求和 / N）
    dW =  X.T @ (p - Y) / N
    db =  (p - Y).sum(axis=0) / N

    W -= lr * dW
    b -= lr * db

    if i % 50 == 0:
        acc = np.mean(np.argmax(p, axis=1) == y)
        print(i, "loss=%.4f" % loss, "acc=%.3f" % acc)
