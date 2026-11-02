# -*- coding: utf-8 -*-
"""
W7 第 4 段（下）· B：把 A 里的裸张量 block 装进 nn.Module，堆 2 层，接回语言模型。

A 的 block() 能前向，但永远训不了 —— 参数放在字典里，优化器看不到它们。
这一段做的就是把「参数字典」换成「nn.Module」，只此一件事。

三个新 API（就这三个，别的都见过）：
    nn.ModuleList     让一堆子模块被正确注册（参数才会出现在 model.parameters() 里）
    register_buffer   把 mask 这类「不是参数、但要跟着模型走」的张量挂上去
    nn.Dropout        训练时随机丢一部分，防止死记

跑：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_gpt.py

只留 3 个 TODO。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from w7_corpus import build_corpus, build_vocab, encode, get_batch, split_train_val
from w7_model import CharLM, Config, lm_loss


# ============================================================
# 核心 1：一个 block。和 A 的 block() 逐行对应，只是参数换成了 nn.Linear。
# ============================================================
class Block(nn.Module):
    """x: (B, T, d)  →  (B, T, d)"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        # 两个 LN（pre-LN：放在子层入口，不是出口）
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        # 注意力：一次矩阵乘算出 Q/K/V，等价于 A 里的 Wqkv
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        # FFN：先扩 4 倍，再压回来
        self.W1 = nn.Linear(d, 4 * d, bias=False)
        self.W2 = nn.Linear(4 * d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        B, T, d = x.shape

        # 子层 1：多头注意力（单头版）
        h = self.ln1(x)
        q, k, v = self.Wqkv(h).chunk(3, dim=-1)          # 各自 (B, T, d)
        s = q @ k.transpose(-2, -1) / math.sqrt(d)       # (B, T, T)
        s = s.masked_fill(mask, float("-inf"))           # 遮未来
        a = torch.softmax(s, dim=-1) @ v                 # (B, T, d)
        x = x + self.drop(self.Wo(a))                    # 残差

        # 子层 2：FFN
        h = self.ln2(x)
        x = x + self.drop(self.W2(F.gelu(self.W1(h))))   # 残差
        return x

    def params_breakdown(self):
        """逐项报参数量，方便和 12d² 对账。"""
        return [
            ("Wqkv", self.Wqkv.weight.numel()),
            ("Wo", self.Wo.weight.numel()),
            ("W1", self.W1.weight.numel()),
            ("W2", self.W2.weight.numel()),
            ("ln1", sum(p.numel() for p in self.ln1.parameters())),
            ("ln2", sum(p.numel() for p in self.ln2.parameters())),
        ]


# ============================================================
# 核心 2：整个模型。和 CharLM 的差别只有一处 —— blocks 不再是 nn.Identity。
# ============================================================
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # TODO-1  堆 n_layer 个 Block，并用 nn.ModuleList 包起来。
        #         为什么不能直接用 python list？
        #           因为普通 list 里的子模块不会被 nn.Module 注册，
        #           它们的参数不会出现在 model.parameters() 里，优化器就收不到。
        #         提示：nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])   # <-- TODO-1

        if not isinstance(self.blocks, nn.ModuleList):
            raise SystemExit(
                "\n[TODO-1 未完成] blocks 必须是 nn.ModuleList，不能是普通 list 或 ...。\n"
                "  写：self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])\n"
            )

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # TODO-2  mask 只用造一次，之后每次前向都复用。
        #         它不是参数（不需要梯度、不参与训练），但必须跟着模型一起搬到 GPU，
        #         所以用 register_buffer 挂上去，不能用普通 attribute。
        #         persistent=False：存 checkpoint 时不写进文件 —— 反正能算出来，没必要占体积。
        #         提示：self.register_buffer("mask", 张量, persistent=False)
        #         张量 = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
         
        self.register_buffer("mask", torch.triu(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool), diagonal=1), persistent=False)   # <-- TODO-2

        if not hasattr(self, "mask"):
            raise SystemExit(
                "\n[TODO-2 未完成] 没有注册 mask。\n"
                "  写：self.register_buffer('mask',\n"
                "        torch.triu(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool), diagonal=1),\n"
                "        persistent=False)\n"
            )

        # 初始化：直接复用上一段写好的那份（std=0.02）
        self.apply(CharLM._init_weights)
        # 权重共享
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.block_size

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        # TODO-3  从整块 mask 里切出这次真正要用的 (T, T)。
        #         为什么要切？block_size 是上限（64），但这次可能只喂了 10 个字符，
        #         63×63 的 mask 套到 10×10 的分数上会形状不匹配。
        #         提示：self.mask[:T, :T]
        m = self.mask[:T, :T]   # <-- TODO-3

        if not isinstance(m, torch.Tensor):
            raise SystemExit(
                "\n[TODO-3 未完成] mask 没有切到实际长度。\n"
                "  写：m = self.mask[:T, :T]\n"
                "  切完之后 m.shape 应该是 (T, T)，其中 T 是这次输入的序列长度。\n"
            )

        for blk in self.blocks:
            x = blk(x, m)

        x = self.ln_f(x)
        return self.lm_head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 实验：残差到底在挡什么（第 4 段下 A 的自检第 2 题）
# ============================================================
def _stack_forward(x, blocks, mask, use_residual):
    """跑完一整摞 block。use_residual=False 时把 x = x + ... 的 x + 去掉。"""
    for b in blocks:
        h = b.ln1(x)
        q, k, v = b.Wqkv(h).chunk(3, dim=-1)
        s = (q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])).masked_fill(mask, float("-inf"))
        a = torch.softmax(s, dim=-1) @ v
        x = x + b.Wo(a) if use_residual else b.Wo(a)
        h = b.ln2(x)
        f = b.W2(F.gelu(b.W1(h)))
        x = x + f if use_residual else f
    return x


def residual_experiment(depths=(1, 2, 4, 8, 16), d=64, T=64, V=1220):
    """有残差 vs 无残差：输出还认不认得输入。

    两个指标：
      cos      输出与输入的余弦相似度。接近 1 = 原始信息活着；接近 0 = 已经面目全非
      rel      相对变化 ‖out-in‖ / ‖in‖。越小 = 改动越轻微

    注意这里**不测梯度范数**。实测过，有残差时梯度反而更小
    （因为子层对输出的影响力小了），拿它当证据会得出相反结论。
    """
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    rows = []

    for use_res in (True, False):
        for n in depths:
            torch.manual_seed(0)
            cfg = Config()
            cfg.d_model, cfg.n_layer, cfg.vocab_size = d, n, V
            blocks = nn.ModuleList([Block(cfg) for _ in range(n)])
            for b in blocks:
                b.eval()                       # 关掉 dropout，变量只留一个

            x = torch.randn(2, T, d)
            x0 = x.clone()
            out = _stack_forward(x, blocks, mask, use_res)

            cos = F.cosine_similarity(out.reshape(-1), x0.reshape(-1), dim=0).item()
            rel = ((out - x0).norm() / x0.norm()).item()
            rows.append((use_res, n, cos, rel))

    return rows


def main():
    torch.manual_seed(1337)

    text = build_corpus(verbose=False)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train, val = split_train_val(data)

    cfg = Config()
    cfg.vocab_size = len(chars)

    model = GPT(cfg)
    model.eval()                              # 关 dropout，下面所有数字才可复现

    print("=" * 66)
    print("① 参数量：从「骨架」变成「2 层 Transformer」")
    print("=" * 66)
    bd = model.blocks[0].params_breakdown()
    for name, n in bd:
        print("   一个 block 的 %-6s %10s" % (name, format(n, ",")))
    one_block = sum(n for _, n in bd)
    print("   " + "-" * 46)
    print("   一个 block 合计    %10s   （12d² = %s）" % (format(one_block, ","), format(12 * cfg.d_model ** 2, ",")))
    print("   × %d 层              %10s" % (cfg.n_layer, format(one_block * cfg.n_layer, ",")))
    print()
    print("   骨架（无 block）      %10s" % format(82304, ","))
    print("   现在（2 层 block）    %10s" % format(model.n_params(), ","))
    print("   差值              %10s   = 2 × 一个 block" % format(model.n_params() - 82304, ","))

    print()
    print("=" * 66)
    print("② 形状链")
    print("=" * 66)
    xb, yb = get_batch(train)
    logits = model(xb)
    print("   xb      : %s" % (tuple(xb.shape),))
    print("   logits  : %s" % (tuple(logits.shape),))
    print("   yb      : %s" % (tuple(yb.shape),))

    print()
    print("=" * 66)
    print("③ 初始 loss 还是 ln(V) 吗")
    print("=" * 66)
    loss = lm_loss(logits, yb)
    print("   实际 loss    = %.4f" % loss.item())
    print("   理论 ln(V)   = %.4f   （V = %d）" % (math.log(cfg.vocab_size), cfg.vocab_size))
    print("   差值         = %.4f" % abs(loss.item() - math.log(cfg.vocab_size)))
    print()
    print("   注意：加了 2 层 block 之后，这个值会略微偏离 ln(V)。")
    print("   原因是残差让第一层近似恒等，输出 ≈ 输入嵌入，")
    print("   而 logits = 输入嵌入 @ 嵌入表转置 —— 等于在跟自己比相似度。")
    print("   「预测当前字符」比「均匀乱猜」稍好一点，所以 loss 会低于 ln(V)。")

    print()
    print("=" * 66)
    print("④ 堆了 2 层之后，因果性还成立吗（最硬的那条）")
    print("=" * 66)
    # 输入是整数 id，不是浮点数，所以扰动方式改成「把最后一个字符换成另一个字符」
    xb2 = xb.clone()
    xb2[:, -1] = (xb2[:, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        out1 = model(xb)
        out2 = model(xb2)
    head = (out2[:, :-1] - out1[:, :-1]).abs().max().item()
    last = (out2[:, -1] - out1[:, -1]).abs().max().item()
    print("   位置 0—62 的最大变化 : %.10f   ← 必须精确为 0" % head)
    print("   位置 63  的最大变化 : %.10f   ← 必须大于 0" % last)
    assert head == 0.0, "因果性被破坏：2 层里有 mask 漏了"
    print("   结论：两层都遮住了。任何一层漏，这里就不可能是 0。")

    print()
    print("=" * 66)
    print("⑤ 残差的量化证据：输出还认不认得输入")
    print("=" * 66)
    print("   %-6s %-6s %10s %12s" % ("层数", "残差", "cos(出,入)", "相对变化"))
    print("   " + "-" * 42)
    for use_res, n, cos, rel in residual_experiment():
        print("   %-6d %-6s %10.4f %12.4f" % (n, "有" if use_res else "无", cos, rel))
    print()
    print("   读法：看「无」残差那一列，cos 从第 1 层就已经是 0 —— 输入信息被彻底冲掉了。")
    print("   「有」残差那一列即使堆到 16 层，cos 还有 0.71，原始信息一直活着。")
    print()
    print("   这就是残差的全部作用：留一条恒等直通，让每一层只需要学「改动量」，")
    print("   而不是把整份表示重新造一遍。")


if __name__ == "__main__":
    main()
