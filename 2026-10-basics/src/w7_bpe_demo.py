"""
W7 第 1 段：BPE（Byte Pair Encoding，字节对编码）手算级演示

目的：不依赖任何库，把 BPE 的「训练」和「编码」两件事用真实数字走一遍。

运行（PowerShell）：
    & "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_bpe_demo.py
"""

from collections import Counter

# ============ 训练语料：4 个词 + 各自出现次数 ============
CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

# 每个词拆成字符，末尾加 </w> 表示"词结束"
words = {tuple(w) + ("</w>",): c for w, c in CORPUS.items()}


def count_pairs(words):
    """统计所有相邻字符对的加权出现次数（词频加权）。"""
    pairs = Counter()
    for w, freq in words.items():
        for i in range(len(w) - 1):
            pairs[(w[i], w[i + 1])] += freq
    return pairs


def merge(words, pair):
    """把所有出现 pair 的地方合并成一个新符号。"""
    out = {}
    for w, freq in words.items():
        buf, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                buf.append(w[i] + w[i + 1])
                i += 2
            else:
                buf.append(w[i])
                i += 1
        out[tuple(buf)] = out.get(tuple(buf), 0) + freq
    return out


print("=" * 68)
print("PART 0  初始状态：每个词被拆成单字符")
print("=" * 68)
for w, f in words.items():
    print(f"  {' '.join(w):<22} 出现 {f} 次")

# ============ 训练：反复合并最高频的相邻对 ============
print("\n" + "=" * 68)
print("PART 1  训练：每轮合并一个最高频对")
print("=" * 68)

merges = []
for step in range(1, 11):
    pairs = count_pairs(words)
    top3 = pairs.most_common(3)
    best, n = top3[0]

    print(f"\n第 {step} 轮")
    print("  候选前三:", {f"{a}+{b}": c for (a, b), c in top3})
    print(f"  → 合并 {best[0]}+{best[1]}（出现 {n} 次）")
    merges.append(best)
    words = merge(words, best)
    print("  合并后:", {" ".join(w): f for w, f in words.items()})


# ============ 编码：用学到的合并规则处理没见过的词 ============
print("\n" + "=" * 68)
print("PART 2  编码：拿训练好的规则去切一个「训练时没见过」的词")
print("=" * 68)


def encode(text, merges):
    """严格按 merges 的顺序依次替换。"""
    sym = list(text) + ["</w>"]
    for pair in merges:
        buf, i = [], 0
        while i < len(sym):
            if i < len(sym) - 1 and (sym[i], sym[i + 1]) == pair:
                buf.append(sym[i] + sym[i + 1])
                i += 2
            else:
                buf.append(sym[i])
                i += 1
        sym = buf
    return sym


for test in ["low", "lowest", "newer"]:
    toks = encode(test, merges)
    seen = "（训练集里有）" if test in CORPUS else "（训练集里没有）"
    print(f"  {test:<8}{seen:<14} → {toks}   共 {len(toks)} 个 token")

print("\n  关键：'lowest' 训练时从未出现，但被拆成 low + est</w>")
print("        两个片段都是训练时学到的，所以模型仍然认得它。")
print("        这就是 BPE 解决 OOV（out-of-vocabulary，未登录词）的方式。")
