# -*- coding: utf-8 -*-
"""
W7 第 4 段（下）· A：一个 Transformer block，裸张量版，核心只有 10 行。

W6 你已经手写过 attention。这里加的是三件新东西：
    ① 残差连接   x = x + 子层(x)
    ② LayerNorm  放在子层入口（pre-LN）
    ③ causal mask 在 batch 里的形状

最后做一个最硬的验证：**改最后一个 token，前面位置的输出必须一个字都不变**。
如果变了，说明 mask 漏了 —— 这是检查因果性最快的方法。

运行：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_block.py
"""
import math

import torch
import torch.nn.functional as F

# ============================================================
# 核心：一个 Transformer block。这 10 行就是全部。
# ============================================================
def block(x, p, mask):
    """x: (B, T, d)   p: 参数字典   mask: (T, T) bool，True 表示"不许看"
       返回: (B, T, d)，形状和输入完全一样"""
    d = x.shape[-1]

    # 子层 1：多头注意力（这里先用单头，第 6 段改多头）
    h = F.layer_norm(x, (d,))                        # ② pre-LN：先归一化再进子层
    q, k, v = (h @ p["Wqkv"]).chunk(3, dim=-1)       # 一次矩阵乘算完 Q/K/V
    s = q @ k.transpose(-2, -1) / math.sqrt(d)       # (B, T, T) 打分
    s = s.masked_fill(mask, float("-inf"))           # ③ 遮住未来
    a = torch.softmax(s, dim=-1) @ v                 # (B, T, d)
    x = x + a @ p["Wo"]                              # ① 残差：加回去

    # 子层 2：前馈网络（FFN），先扩 4 倍再压回来
    h = F.layer_norm(x, (d,))                        # ② pre-LN
    x = x + F.gelu(h @ p["W1"]) @ p["W2"]            # ① 残差
    return x


# ============================================================
# 下面是演示，不是核心
# ============================================================
def make_params(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = {
        "Wqkv": torch.randn(d, 3 * d, generator=g) * 0.02,
        "Wo": torch.randn(d, d, generator=g) * 0.02,
        "W1": torch.randn(d, 4 * d, generator=g) * 0.02,
        "W2": torch.randn(4 * d, d, generator=g) * 0.02,
    }
    return p


def count_params(p):
    return sum(v.numel() for v in p.values())


def main():
    torch.manual_seed(0)
    B, T, d = 2, 8, 32          # 2 条样本、每条 8 个字符、每个字符 32 维
    x = torch.randn(B, T, d)
    p = make_params(d)

    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    print("=" * 64)
    print("配置：B=%d  T=%d  d=%d" % (B, T, d))
    print("=" * 64)

    print("\n① causal mask 长什么样（True = 不许看）")
    for i in range(min(T, 5)):
        row = "".join("X" if mask[i, j] else "." for j in range(T))
        print("   第 %d 行: %s" % (i, row))
    print("   X = 被遮住（未来）   . = 可以看（自己 + 过去）")
    print("   形状 %s —— 只有 (T,T)，没有 batch 维" % (tuple(mask.shape),))
    print("   广播：masked_fill 会自动把它套到每一条样本上，不用手动 expand")

    print("\n② 一个 block 的参数量")
    print("   Wqkv %s  +  Wo %s" % (tuple(p["Wqkv"].shape), tuple(p["Wo"].shape)))
    print("   W1   %s  +  W2 %s" % (tuple(p["W1"].shape), tuple(p["W2"].shape)))
    print("   合计 %s 个参数 = 12 × d² = 12 × %d" % (format(count_params(p), ","), d * d))

    print("\n③ 前向一次")
    out = block(x, p, mask)
    print("   输入  : %s" % (tuple(x.shape),))
    print("   输出  : %s   ← 形状完全一样（残差要求必须一样）" % (tuple(out.shape),))

    print("\n" + "=" * 64)
    print("最硬的验证：因果性测试")
    print("=" * 64)
    print("做法：把第 7 个（最后一个）token 的某一维改大，再前向一次。")
    print("      如果 mask 是对的，位置 0—6 的输出必须一个字都不变。")

    x2 = x.clone()
    x2[:, -1, 0] += 5.0                    # 只改最后一个 token 的第 0 维
    out2 = block(x2, p, mask)

    diff_head = (out2[:, :-1] - out[:, :-1]).abs().max().item()
    diff_last = (out2[:, -1] - out[:, -1]).abs().max().item()

    print("\n   位置 0—6 的最大变化 : %.10f   ← 必须是 0" % diff_head)
    print("   位置 7   的最大变化 : %.10f   ← 必须大于 0" % diff_last)
    assert diff_head == 0.0, "因果性被破坏：前面的位置受到了未来 token 的影响"
    assert diff_last > 0.0, "最后一位居然没变，说明输入改动没生效"
    print("\n   结论：因果性成立。前面 7 个位置完全不知道最后一个字长什么样。")

    print("\n" + "=" * 64)
    print("对照：如果忘了加 mask，会怎样")
    print("=" * 64)
    no_mask = torch.zeros(T, T, dtype=torch.bool)   # 全 False = 一个都不遮
    out_nm = block(x2, p, no_mask)
    out_nm0 = block(x, p, no_mask)
    leak = (out_nm[:, :-1] - out_nm0[:, :-1]).abs().max().item()
    print("   不加 mask 时，位置 0—6 的最大变化 : %.3e" % leak)
    print("   也就是：最后一个字的信息泄漏到了前面的位置。")
    print("   训练时这等于让模型偷看答案，loss 会掉得很好看，上线立刻崩。")

    print("\n" + "=" * 64)
    print("顺带一个坑：为什么扰动不能用「所有维度都加 1」")
    print("=" * 64)
    x3 = x.clone()
    x3[:, -1, :] += 1.0                                  # 所有维度加同一个常数
    out3 = block(x3, p, mask)
    d_const = (out3[:, -1] - out[:, -1]).abs().max().item()
    print("   所有维度加 1.0，位置 7 的输出变化 : %.3e" % d_const)
    print("   只有 1.0 左右，而这 1.0 全部来自残差 x + ... 的直通，")
    print("   子层的贡献几乎没有 —— 因为 LayerNorm 会减掉均值，")
    print("   给所有维度加同一个常数，过完 LN 之后一模一样。")
    print("   （我第一次写这个验证时用的就是 +1.0，结果泄漏测不出来。）")


if __name__ == "__main__":
    main()
