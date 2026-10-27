# -*- coding: utf-8 -*-
"""
W7 第 3 段：字符级语言模型（character-level LM）的语料与数据构造。

这一段不训练，只把「文本 → 训练样本」这条链路搭好：
    原始文本 → 清洗 → 字符表（词表）→ 整数 id → (x, y) 错位对 → 随机 batch

核心只有一句话：
    语言模型的训练信号，就是把同一段文本「向右挪一格」当成答案。
    模型看 "我们先做"，要它猜 "们先做数"。

运行：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_corpus.py

只留 3 个 TODO，都在这条链路的关键位置上。
"""
import glob
import os
import re

import torch

# ---------------------------------------------------------------- 配置
BLOCK_SIZE = 64     # 每条样本喂给模型的字符数（就是 W6 说的序列长度 n）
BATCH_SIZE = 32     # 一个 batch 里几条样本
TRAIN_RATIO = 0.9   # 训练 / 验证划分

OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "w7_corpus.txt")

# 语料来源：本地已有文本，零下载
SOURCES = (
    sorted(glob.glob("notes/*.md"))
    + ["README.md"]
    + [
        r"E:\workbuddy\2026-10-24-07-50-47\calendar_full.txt",
        r"E:\workbuddy\2026-10-24-07-50-47\mianjing_full.txt",
    ]
)

CJK = re.compile(r"[\u4e00-\u9fff]")
CJK_RATIO_MIN = 0.35   # 一行里中文字符占比低于这个值就丢掉（可调旋钮）


# ---------------------------------------------------------------- 1. 语料
def build_corpus(verbose=True):
    """把本地文本拼成一份干净的中文语料，写到 data/w7_corpus.txt。

    清洗只做三件事（不是追求完美，是够用）：
      1. 去掉代码块和行内反引号 —— 代码字符会污染字符表
      2. 去掉 markdown 表格行、标题井号、列表符号 —— 这些是格式不是文字
      3. 只保留「中文占比 >= 35%」的句子 —— 保证生成的文本是可读中文

    第 3 条是个可调旋钮：阈值调低 → 语料变大但混杂英文；调高 → 更纯但更少。
    """
    if os.path.exists(OUT_PATH):
        text = open(OUT_PATH, encoding="utf-8").read()
        if verbose:
            print("[语料] 复用已存在的 %s（%d 字符）" % (OUT_PATH, len(text)))
        return text

    raw = []
    for f in SOURCES:
        if os.path.exists(f):
            raw.append(open(f, encoding="utf-8").read())
    joined = "\n".join(raw)

    # 1. 去代码
    joined = re.sub(r"```.*?```", "", joined, flags=re.S)
    joined = re.sub(r"`([^`\n]{0,60})`", r"\1", joined)

    # 2. 逐行清洗
    keep = []
    for line in joined.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("|"):              # 表格行整行丢掉
            continue
        s = re.sub(r"^#+\s*", "", s)       # 标题井号
        s = re.sub(r"^[-*>]\s*", "", s)    # 列表 / 引用符号
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)   # 粗体
        s = s.replace("|", "")
        if len(s) < 8:
            continue
        # 3. 中文占比门槛
        if len(CJK.findall(s)) / len(s) < CJK_RATIO_MIN:
            continue
        keep.append(s)

    text = re.sub(r"\n{2,}", "\n", "\n".join(keep)).strip()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    if verbose:
        print("[语料] 原始 %d 字符 → 清洗后 %d 字符，已写入 %s"
              % (len(joined), len(text), OUT_PATH))
    return text


# ---------------------------------------------------------------- 2. 词表
def build_vocab(text):
    """字符级词表就是「文本里出现过的所有字符」，排序后编号。

    这就是字符级模型不需要 tokenizer 的原因：
    词表 = sorted(set(文本))，没有 BPE、没有合并规则、没有下载。
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


# ---------------------------------------------------------------- 3. 编码
def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def split_train_val(data, ratio=TRAIN_RATIO):
    # TODO-1  算出切分点下标：前 ratio 部分给训练，剩下的给验证。
    #         提示：data 是一个一维 tensor，切分点 = int(len(data) * ratio)
    #         注意：验证集必须是「文本的尾部」，不能随机抽 —— 随机抽会让
    #         相邻字符同时出现在训练集和验证集，等于提前泄题。
    split_idx = ...   # <-- 改这里
    return data[:split_idx], data[split_idx:]


# ---------------------------------------------------------------- 4. 采样
def get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """随机抽 batch_size 个起点，每个起点取 block_size 个字符。

    x = data[i   : i+block_size]
    y = data[i+1 : i+block_size+1]        ← 就是 x 整体右移一格
    """
    # TODO-2  算出 torch.randint 的「上界（不含）」。
    #         思路：起点 i 最大只能到 len(data) - block_size - 1，
    #         否则 y = data[i+1 : i+1+block_size] 会越界（y 比 x 多看一位）。
    #         而 torch.randint(hi, ...) 取的是 [0, hi)，所以 hi 要比最大起点大 1。
    hi = ...   # <-- 改这里
    ix = torch.randint(hi, (batch_size,))

    x = torch.stack([data[i: i + block_size] for i in ix])

    # TODO-3  构造答案 y：从每个起点 i+1 开始，取 block_size 个字符。
    #         写完后检查：y 的形状必须和 x 完全一样，都是 (batch_size, block_size)。
    y = ...   # <-- 改这里
    return x, y


# ---------------------------------------------------------------- 5. 展示
def main():
    text = build_corpus()
    chars, stoi, itos = build_vocab(text)

    print()
    print("=" * 66)
    print("语料与词表")
    print("=" * 66)
    print("总字符数        : %d" % len(text))
    print("字符集大小(词表): %d" % len(chars))

    from collections import Counter
    freq = Counter(text)
    singles = sum(1 for _, n in freq.items() if n == 1)
    print("只出现 1 次     : %d 个字符（占词表 %.1f%%）" % (singles, 100 * singles / len(chars)))
    print("最高频 8 个     : %s" % " ".join("%r=%d" % (c, n) for c, n in freq.most_common(8)))

    data = encode(text, stoi)
    train, val = split_train_val(data)
    print()
    print("训练集 / 验证集 : %d / %d 字符" % (len(train), len(val)))

    print()
    print("=" * 66)
    print("一条真实样本（block_size=%d）" % BLOCK_SIZE)
    print("=" * 66)
    x, y = get_batch(train, block_size=8, batch_size=1)   # 取短一点便于肉眼核对
    xs = "".join(itos[i] for i in x[0].tolist())
    ys = "".join(itos[i] for i in y[0].tolist())
    print("x = %r" % xs)
    print("y = %r" % ys)
    print("↑ 逐个字符比对，y 是不是 x 右移一格？")
    for k in range(len(xs) - 1):
        assert ys[k] == xs[k + 1], "第 %d 位不对：y=%r x的下一位=%r" % (k, ys[k], xs[k + 1])
    print("自检通过：y[k] == x[k+1] 对所有 k 成立")

    print()
    print("=" * 66)
    print("正式 batch 的形状")
    print("=" * 66)
    xb, yb = get_batch(train)
    print("x: %s   y: %s   dtype: %s" % (tuple(xb.shape), tuple(yb.shape), xb.dtype))
    print("每个 batch 覆盖 %d 个字符位置（%d × %d）"
          % (xb.numel(), BATCH_SIZE, BLOCK_SIZE))
    print()
    print("估算：1 个 epoch 需要 %d 个 batch（%d / %d）"
          % (len(train) // (BATCH_SIZE * BLOCK_SIZE) + 1, len(train), BATCH_SIZE * BLOCK_SIZE))


if __name__ == "__main__":
    main()
