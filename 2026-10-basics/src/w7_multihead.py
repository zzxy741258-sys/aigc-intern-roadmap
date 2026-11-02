# -*- coding: utf-8 -*-
"""
W7 补：单头 vs 多头对照（日历 W7 表 2 的原文要求）。

同一份语料、同一个种子、同一步数、同一批 batch，
只改一个数字：n_head。

    单头：n_head=1  →  每个头 64 维
    多头：n_head=4  →  每个头 16 维

参数量完全一样（W6 已经算过：多头不比单头多一个参数），
所以差别只来自「一次 64 维匹配」还是「四次 16 维匹配」。

跑：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_multihead.py
"""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from w7_corpus import BLOCK_SIZE, build_corpus, build_vocab, encode, get_batch, split_train_val
from w7_gpt import GPT
from w7_model import Config, lm_loss
import w7_gpt


# ============================================================
# 核心：多头 block。和单头只差 4 行（拆头、缩放、拼头）
# ============================================================
class MultiHeadBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.W1 = nn.Linear(d, 4 * d, bias=False)
        self.W2 = nn.Linear(4 * d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        B, T, d = x.shape
        H = self.n_head
        dk = d // H

        h = self.ln1(x)
        q, k, v = self.Wqkv(h).chunk(3, dim=-1)              # 各 (B,T,d)

        q = q.view(B, T, H, dk).transpose(1, 2)              # ① (B,H,T,dk)
        k = k.view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, H, dk).transpose(1, 2)

        s = q @ k.transpose(-2, -1) / math.sqrt(dk)          # ② (B,H,T,T)，除的是 dk
        s = s.masked_fill(mask, float("-inf"))               # ③ (T,T) 自动广播到 (B,H,T,T)
        a = torch.softmax(s, dim=-1) @ v                     # ④ (B,H,T,dk)

        a = a.transpose(1, 2).contiguous().view(B, T, d)      # ⑤ 拼回去 (B,T,d)
        x = x + self.drop(self.Wo(a))

        h = self.ln2(x)
        x = x + self.drop(self.W2(F.gelu(self.W1(h))))
        return x


def train_and_eval(n_head, train, val, vocab_size, steps=600):
    """训练一个 n_head 头的模型，返回 (参数量, 训练loss, 验证loss, 模型)。"""
    w7_gpt.Block = MultiHeadBlock              # 把 GPT 里的 Block 换掉

    torch.manual_seed(1337)                    # 同一个种子 → 同一批初始化、同一批 batch
    cfg = Config()
    cfg.vocab_size = vocab_size
    cfg.n_head = n_head

    model = GPT(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    t0 = time.time()
    for _ in range(steps):
        xb, yb = get_batch(train)
        loss = lm_loss(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    saved = torch.get_rng_state()
    torch.manual_seed(2024)
    res = {}
    with torch.no_grad():
        for name, data in (("train", train), ("val", val)):
            L = []
            for _ in range(20):
                xb, yb = get_batch(data)
                L.append(lm_loss(model(xb), yb).item())
            res[name] = sum(L) / len(L)
    torch.set_rng_state(saved)

    n = sum(p.numel() for p in model.parameters())
    return n, res["train"], res["val"], model, time.time() - t0


@torch.no_grad()
def sample(model, itos, stoi, n=120, temp=0.8, top_k=40):
    model.eval()
    idx = torch.tensor([[stoi["模"], stoi["型"]]])
    for _ in range(n):
        lg = model(idx[:, -BLOCK_SIZE:])[:, -1, :] / temp
        v, _ = torch.topk(lg, top_k)
        lg[lg < v[:, [-1]]] = -float("inf")
        p = F.softmax(lg, dim=-1)
        idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())


def main():
    text = build_corpus(verbose=False)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train, val = split_train_val(data)

    print("=" * 66)
    print("单头 vs 多头：只改 n_head 一个数字")
    print("=" * 66)
    print("  %-8s %-12s %10s %10s %8s %8s" % ("头数", "每头维度", "参数量", "训练loss", "验证loss", "困惑度"))
    print("  " + "-" * 60)

    rows = []
    for n_head in (1, 4):
        n, tl, vl, model, cost = train_and_eval(n_head, train, val, len(chars))
        rows.append((n_head, n, tl, vl, model))
        print("  %-8d %-12s %10s %10.4f %10.4f %8.2f   (%.0f秒)"
              % (n_head, "%d 维" % (64 // n_head), format(n, ","), tl, vl, math.exp(vl), cost))

    print()
    print("  参数量一样吗：", "一样" if rows[0][1] == rows[1][1] else "不一样（有问题）")

    print()
    print("=" * 66)
    print("同一个提示「模型」，各生成 120 字")
    print("=" * 66)
    for n_head, n, tl, vl, model in rows:
        s = sample(model, itos, stoi)
        print("\n  【n_head=%d】" % n_head)
        print("  " + s.replace("\n", " "))

    print()
    print("=" * 66)
    print("怎么读这组数")
    print("=" * 66)
    print("  ① 参数量必须完全一样 —— 多头不是加容量，是换组织方式")
    print("  ② 只比验证 loss，不比训练 loss —— 训练 loss 低可能只是背得多")
    print("  ③ 差距小于 0.05 就算「打平」，这个语料太小，不足以分出胜负")


if __name__ == "__main__":
    main()
