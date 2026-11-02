# -*- coding: utf-8 -*-
"""
W7 第 ⑥ 站后半：生成 200 字 + checkpoint 保存/恢复。

这是日历 W7 的硬验收项：
    「生成脚本可从 checkpoint 恢复」
意思是：关掉进程、重新打开，只凭磁盘上那个文件，就能继续生成。

核心生成循环只有 6 行（见 generate），其余都是装载和打印。

新 API 4 个：
    model.state_dict() / model.load_state_dict()
    torch.save() / torch.load()
    torch.multinomial()
    torch.topk()

跑：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_generate.py
"""
import math
import os

import torch
import torch.nn.functional as F

from w7_corpus import BLOCK_SIZE, build_corpus, build_vocab, encode, get_batch, split_train_val
from w7_gpt import GPT
from w7_model import Config, lm_loss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "data", "w7_ckpt.pt")

STEPS = 600
LR = 1e-3
SEED = 1337


# ============================================================
# 核心：生成 6 行
# ============================================================
@torch.no_grad()
def generate(model, idx, max_new=200, temperature=1.0, top_k=None):
    """idx: (1, T) 起始提示 → (1, T+max_new) 生成结果"""
    model.eval()
    for _ in range(max_new):
        idx_cond = idx[:, -BLOCK_SIZE:]              # ① 上下文最多 64 个，多了会越界
        logits = model(idx_cond)[:, -1, :]           # ② 只取最后一个位置
        logits = logits / temperature                # ③ 温度：<1 保守，>1 放飞
        if top_k is not None:                        # ④ 只保留分数最高的 k 个
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)            # ⑤ 分数变概率
        nxt = torch.multinomial(probs, num_samples=1)  # ⑥ 按概率抽一个，不是取最大
        idx = torch.cat([idx, nxt], dim=1)           # ⑦ 拼回去，当作下一轮的输入
    return idx


# ============================================================
# checkpoint
# ============================================================
def save_ckpt(model, opt, cfg, chars, step, tl, vl):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({
        "model": model.state_dict(),      # 所有权重
        "optimizer": opt.state_dict(),    # 动量等状态，续训要用
        "step": step,
        "config": vars(cfg),              # 架构参数：层数、维度、block_size
        "chars": chars,                   # ★ 词表。少了这个，id 就失去意义
        "train_loss": tl,
        "val_loss": vl,
    }, CKPT)


def load_ckpt(path=CKPT):
    ck = torch.load(path, map_location="cpu")
    cfg = Config()
    for k, v in ck["config"].items():
        setattr(cfg, k, v)
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return ck, model, cfg


# ============================================================
# 演示：checkpoint 里为什么必须存词表
# ============================================================
def demo_vocab_shift(text):
    """模拟真实场景：语料扩充了，重新算词表。"""
    old = sorted(set(text[: int(len(text) * 0.8)]))   # 以前只有 80% 语料时算出的词表
    new = sorted(set(text))                           # 现在语料补全了
    changed = sum(1 for ch in old if new.index(ch) != old.index(ch))
    print("   语料从 80%% 补到 100%%：旧词表 %d 个字，其中 %d 个编号变了（%.0f%%）"
          % (len(old), changed, 100 * changed / len(old)))
    for probe in "模型数训":
        if probe in old:
            print("   「%s」的编号：%d → %d" % (probe, old.index(probe), new.index(probe)))
    print("   也就是说：同一份权重，配上新词表会输出完全不同的字。")


def main():
    torch.manual_seed(SEED)

    text = build_corpus(verbose=False)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train, val = split_train_val(data)

    cfg = Config()
    cfg.vocab_size = len(chars)

    print("=" * 62)
    print("① 训练并保存 checkpoint")
    print("=" * 62)
    model = GPT(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(STEPS + 1):
        xb, yb = get_batch(train)
        logits = model(xb)
        loss = lm_loss(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(train)
        tl = lm_loss(model(xb), yb).item()
        xb, yb = get_batch(val)
        vl = lm_loss(model(xb), yb).item()

    save_ckpt(model, opt, cfg, chars, STEPS, tl, vl)
    size = os.path.getsize(CKPT) / 1024
    print("   已保存 %s （%.0f KB）" % (os.path.basename(CKPT), size))
    print("   文件里有什么：model / optimizer / step / config / chars / loss")

    print()
    print("=" * 62)
    print("② 从 checkpoint 恢复（模拟关掉进程后重开）")
    print("=" * 62)
    del model, opt                       # 把内存里的模型彻底丢掉
    ck, model, cfg2 = load_ckpt()
    print("   恢复的配置：d_model=%d  n_layer=%d  block_size=%d  vocab=%d"
          % (cfg2.d_model, cfg2.n_layer, cfg2.block_size, cfg2.vocab_size))
    print("   保存时的验证集 loss = %.4f （困惑度 %.2f）"
          % (ck["val_loss"], math.exp(ck["val_loss"])))
    print("   参数量 = %s" % format(sum(p.numel() for p in model.parameters()), ","))

    print()
    print("=" * 62)
    print("③ 生成 200 字（同一提示，三种采样）")
    print("=" * 62)
    prompt = "模型"
    idx0 = torch.tensor([[stoi[c] for c in prompt]])

    for temp, k, tag in ((0.5, None, "太保守"), (0.8, 40, "常用"), (1.0, None, "太放飞")):
        out = generate(model, idx0.clone(), max_new=200, temperature=temp, top_k=k)
        s = "".join(itos[i] for i in out[0].tolist())
        label = "温度 %.1f" % temp + (" + 只留前 %d 名" % k if k else "")
        print("\n   【%s　%s】" % (label, tag))
        print("   " + s.replace("\n", " ")[:190])

    print()
    print("=" * 62)
    print("④ 为什么 checkpoint 里必须存词表")
    print("=" * 62)
    demo_vocab_shift(text)
    print("   所以：光有权重不够，权重输出的只是一个编号。")
    print("   编号→字的对照表（chars）必须一起存，否则恢复出来全是乱码。")


if __name__ == "__main__":
    main()
