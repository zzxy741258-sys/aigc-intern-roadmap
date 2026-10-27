"""
W6 八股配套：RoPE（Rotary Position Embedding，旋转位置编码）数字验证

目的：用真实数字证明 RoPE 最关键的性质——
      旋转后的内积只取决于"两个位置的差"，与它们各自在第几位无关。

运行（PowerShell）：
    & "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_rope.py
"""

import math

BASE = 10000.0      # 频率基数，原论文取值；决定各维度转得多快


def angle(pos, i, dim):
    """第 i 组（共 dim/2 组）在位置 pos 上要转的角度，单位弧度。

    pos  : 位置编号（第几个 token）
    i    : 第几组（0 到 dim/2-1）
    dim  : 向量维度 d
    """
    return pos * (BASE ** (-2 * i / dim))


def rope(x, pos):
    """把向量 x 在 pos 位置上做旋转。x 长度必须是偶数。"""
    out = []
    for i in range(len(x) // 2):
        t = angle(pos, i, len(x))
        c, s = math.cos(t), math.sin(t)
        x0, x1 = x[2 * i], x[2 * i + 1]
        out += [x0 * c - x1 * s,      # 二维旋转矩阵第一行
                x0 * s + x1 * c]      # 二维旋转矩阵第二行
    return out


def show(t):
    return [round(v, 6) for v in t]


print("=" * 64)
print("PART 1  不同维度的旋转角（d=4，所以有 2 组）")
print("=" * 64)
for i in range(2):
    print(f"  第 {i} 组：pos=1 时角度 {angle(1, i, 4):.6f} 弧度，"
          f"pos=2 时 {angle(2, i, 4):.6f} 弧度")
print("  → 第 0 组转得快（1 弧度），第 1 组转得慢（0.01 弧度）")
print("  → 这一快一慢组合起来，才能同时区分'相邻'和'远处'的位置")


print("\n" + "=" * 64)
print("PART 2  同一个向量 [1, 0, 1, 0] 在不同位置旋转后长什么样")
print("=" * 64)
v = [1.0, 0.0, 1.0, 0.0]
for p in range(4):
    print(f"  pos={p}: {show(rope(v, p))}")
print("  → pos=0 时原样不动（角度 0，cos0=1、sin0=0）")
print("  → 位置每加 1，向量就在各自平面上多转一个固定角度")


print("\n" + "=" * 64)
print("PART 3  核心性质：内积只依赖位置差")
print("=" * 64)
print("  取 x=[1,0]，分别放在位置 m 和 n，算旋转后的内积：")
w = [1.0, 0.0]
for m, n in [(1, 0), (5, 4), (50, 49), (100, 99)]:
    q, k = rope(w, m), rope(w, n)
    ip = sum(a * b for a, b in zip(q, k))
    print(f"    (m={m:3d}, n={n:3d})  位置差={m - n}  内积={ip:.6f}")

print(f"\n  理论值 cos(1 × 第0组角度) = cos(1.0) = {math.cos(1.0):.6f}")
print("  → 四组内积完全相同，即使把位置推到 100 和 99 也不变。")
print("  → 这就是'相对位置编码'：注意力只看两个 token 隔多远，不看它们在第几位。")
print("  → 对比：如果用的是位置向量相加，位置 1 和位置 101 的表示毫无关系。")
