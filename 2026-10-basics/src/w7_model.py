# -*- coding: utf-8 -*-
"""
W7 第 4 段（上）：字符级语言模型的骨架 —— 嵌入层 + 位置编码 + 输出头。

这一段先不放 Transformer block，只把「整数 id → 词向量 → 加上位置 → 打分 → loss」这条
最小链路搭通，用一个可验证的数字收口：

    随机初始化时，loss 应该约等于 ln(词表大小)。
    词表 1220 → ln(1220) ≈ 7.11

如果初始 loss 明显偏离这个数，说明嵌入层、输出头、或者 loss 的形状某一处错了。
这是后面所有训练的前提：先能复现出这个数，再谈训练。

运行：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_model.py

只留 3 个 TODO。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from w7_corpus import BATCH_SIZE, BLOCK_SIZE, build_corpus, build_vocab, encode, get_batch, split_train_val


class Config:
    """所有超参数集中在这里，后面第 5 段训练脚本直接复用，不用再散落各处。"""
    d_model = 64        # 每个字符用多少维向量表示（W6 里的 d_model）
    n_head = 1          # 头数：单头版先跑通，第 6 段再改多头
    n_layer = 2         # Transformer block 的层数（本段先不用，占位）
    block_size = BLOCK_SIZE   # 64，序列长度上限
    dropout = 0.1
    vocab_size = None   # 运行时按语料填


class CharLM(nn.Module):
    """字符级语言模型的最小骨架。

    数据流（形状都写在注释里，B=batch，T=序列长度，V=词表大小）：
        idx  (B, T)  int64
         ↓ token_emb
        (B, T, d_model)
         ↓ + pos_emb          ← 关键：Self-Attention 本身不知道顺序，必须显式注入位置
        (B, T, d_model)
         ↓ Transformer blocks（本段先留空，第 4 段下半补上）
        (B, T, d_model)
         ↓ ln_f
        (B, T, d_model)
         ↓ lm_head（权重与 token_emb 共享）
        logits (B, T, V)      ← 每个位置对"下一个字符是谁"的打分
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1) 词嵌入：把 0..V-1 的整数 id 查表变成 d_model 维向量
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # 2) 位置嵌入：可学习的位置向量，形状 (block_size, d_model)
        #    注意：和 W6 讲的 RoPE 不同，这是最朴素的「绝对位置 + 加法」，正是论文 §3.3 的做法
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)

        # 3) 主干：本段先留空（第 4 段下半用 W6 手写的 MHA 填进来）
        self.blocks = nn.Identity()

        # 4) 输出前的归一化
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # 5) 输出头：d_model → V，为每个字符打分
        #    bias=False 是刻意的：权重共享要求两边形状完全一致
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # 初始化：必须显式指定尺度，不能用 PyTorch 默认值。
        # 默认 nn.Embedding / nn.Linear 的权重是 N(0, 1)，对这个模型来说太大了，
        # 会让 logits 数值爆炸，初始 loss 直接飙到 40 以上（实测 44.08）。
        # GPT-2 用的就是 std=0.02，这里照抄。
        self.apply(self._init_weights)

        # TODO-1  权重共享：让输出头和词嵌入用同一个矩阵。
        #         论文 §3.5 原话：三处共享同一个权重矩阵（输入嵌入、输出嵌入、softmax 前线性层）。
        #         写法：把 lm_head.weight 指向 token_emb.weight 那一块内存。
        #         提示：`self.lm_head.weight = self.token_emb.weight`
        #         （这一步必须在 __init__ 末尾做，前面已经定义好两个模块才行）
        self.lm_head.weight = self.token_emb.weight   # <-- 写这一行

        # 守卫：单独一个 `...` 在 Python 里是合法语句，不会报错，
        # 所以没法靠"跑不跑得起来"判断这一行做没做。这里直接检查两块内存是否同一处。
        if self.lm_head.weight.data_ptr() != self.token_emb.weight.data_ptr():
            raise SystemExit(
                "\n[TODO-1 未完成] 输出头还用自己的权重。\n"
                "  共享后 lm_head.weight 和 token_emb.weight 必须是同一块内存。\n"
                "  加一行：self.lm_head.weight = self.token_emb.weight\n"
            )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) int64 → logits: (B, T, V)"""
        B, T = idx.shape
        assert T <= self.cfg.block_size, "序列长度 %d 超过 block_size %d" % (T, self.cfg.block_size)

        tok = self.token_emb(idx)                       # (B, T, d_model)

        # TODO-2  造位置索引：0, 1, 2, ..., T-1，形状 (T,)，放到和 idx 同一个 device 上。
        #         提示：torch.arange(T, device=idx.device)
        pos = torch.arange(T, device=idx.device)                                      # <-- 改这里
        pos = self.pos_emb(pos)                         # (T, d_model)
        x = tok + pos                                   # 广播：(B,T,d) + (T,d) → (B,T,d)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, V)
        return logits

    def n_params(self):
        """只数「真正会被优化」的参数。

        注意：权重共享后，token_emb 和 lm_head 指向同一块内存，
        直接 sum(p.numel() for p in self.parameters()) 会重复计数吗？
        答案是「不会」——Parameters() 会去重。但 named_parameters() 会看到两个名字。
        这个细节第 6 段讲参数量时要用到。
        """
        return sum(p.numel() for p in self.parameters())


def lm_loss(logits, targets):
    """交叉熵。

    logits : (B, T, V)   每个位置对所有字符的打分
    targets: (B, T)      每个位置真正的下一个字符

    交叉熵要求「样本维在最前」，所以必须把 (B,T,V) 压成 (B*T, V)，
    对应的 targets 压成 (B*T,)。
    """
    B, T, V = logits.shape

    # TODO-3  把 logits 和 targets 都压平到二维/一维。
    #         提示：logits 用 .view(B * T, V)，targets 用 .view(B * T)
    #         这一步为什么必须有？因为 F.cross_entropy 不接受三维输入。
    flat_logits = logits.view(B * T, V)                               # <-- 改这里
    flat_targets = targets.view(B * T)                              # <-- 改这里
    return F.cross_entropy(flat_logits, flat_targets)


def main():
    torch.manual_seed(1337)

    text = build_corpus(verbose=False)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train, val = split_train_val(data)

    cfg = Config()
    cfg.vocab_size = len(chars)

    model = CharLM(cfg)
    print("=" * 66)
    print("模型配置")
    print("=" * 66)
    for k, v in vars(cfg).items():
        print("  %-12s = %s" % (k, v))
    print("  参数量       = %s" % format(model.n_params(), ","))

    print()
    print("=" * 66)
    print("前向一次，检查形状")
    print("=" * 66)
    xb, yb = get_batch(train)
    logits = model(xb)
    print("  xb     : %s" % (tuple(xb.shape),))
    print("  logits : %s" % (tuple(logits.shape),))
    print("  yb     : %s" % (tuple(yb.shape),))
    assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, cfg.vocab_size)

    loss = lm_loss(logits, yb)
    print()
    print("=" * 66)
    print("初始 loss 自检（最重要的一行）")
    print("=" * 66)
    print("  实际 loss      = %.4f" % loss.item())
    print("  理论值 ln(V)   = %.4f   （V = %d）" % (math.log(cfg.vocab_size), cfg.vocab_size))
    print("  两者差值       = %.4f" % abs(loss.item() - math.log(cfg.vocab_size)))
    print()
    print("  读法：随机初始化时模型对每个字符的打分几乎一样，相当于在 V 个字符里均匀猜，")
    print("        交叉熵 = ln(V)。差值在 0.3 以内算正常（权重共享会带来一点偏差）。")


if __name__ == "__main__":
    main()
