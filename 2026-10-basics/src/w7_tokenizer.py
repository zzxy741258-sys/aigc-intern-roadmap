# -*- coding: utf-8 -*-
"""
W7 第 2 段：用 tokenizers 库从零训练一个 BPE 词表，和你第 1 段手算的结果对照。

对照的意义：
  第 1 段你手算了 10 条 merges，得到词表 21。
  这里让库用同一份语料自己训一遍，看它和你算的一样不一样 —— 不一样也很正常，
  因为 tie-break（并列取舍）和初始字母表的构成由库决定。
  这正是 BPE「结果依赖实现细节」的实证。

运行：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_tokenizer.py

只留 3 个 TODO，都是本段唯一的新 API 参数。
"""
import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ---------------------------------------------------------------- 语料
# 和第 1 段完全同一份：4 个词，频次不同
CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}


def build_lines(corpus):
    """把 {词: 频次} 展开成一份真实的文本行列表。

    注意：库训练时吃的是"文本"，不是"词频表"。
    所以出现 5 次的 low 就要真的给 5 行。
    """
    lines = []
    for word, freq in corpus.items():
        lines += [word] * freq
    return lines


# ---------------------------------------------------------------- 训练
def train_bpe(lines, vocab_size):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 空格切分：先把句子切成"词"，再对每个词内部做 BPE。
    # 真实场景还会加标点预切分，这里保持最小可运行。
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    # 解码器：负责把 token 列表还原成字符串。
    # 没有它，decode 只会用空格把 token 拼起来，得到 "lo w est</w>"。
    # 有了它，才会正确地把词尾的 </w> 换回空格并去掉，得到 "lowest"。
    # 这就是"往返验证"能通过的前提。
    tok.decoder = decoders.BPEDecoder(suffix="</w>")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,          # 词表上限：达到就停
        min_frequency=1,                # 频次低于这个的对不合并
        special_tokens=["[UNK]"],       # 特殊 token 占 1 个名额
        # TODO-1  这里要把"词结束标记"设成 "</w>"，和第 1 段手算的口径一致。
        #         参数名：end_of_word_suffix
        #         填完的效果：词尾的空格位置会被替换成 </w>，从而区分
        #         "词中的 est" 和 "词尾的 est"
        end_of_word_suffix='</w>',         # <-- 改这里
        show_progress=False,
    )

    tok.train_from_iterator(lines, trainer)
    return tok


# ---------------------------------------------------------------- 对照
def main():
    lines = build_lines(CORPUS)
    print("语料行数: %d（= 4 个词按频次展开）" % len(lines))

    # TODO-2  算出 vocab_size 该填多少。
    #         提示：库会自动把「单字符」和「词尾字符」都放进初始字母表。
    #         先按 vocab_size=200 跑一遍，看它实际产出的初始字母表有几个，
    #         再反推：要让合并恰好发生 7 次，vocab_size 应该填几？
    #         （第 1 段手算时初始字母表是 11 个，但库里会多出词尾版本，所以不一样）
    VOCAB_SIZE = 21   # <-- 改这里（200 只是探针值，不是答案）

    # 保险：没改就明确报错，不许"跑通了就等于做完了"
    if VOCAB_SIZE == 200:
        raise SystemExit(
            "\n[TODO-2 未完成] 200 只是用来探测初始字母表的探针值。\n"
            "  你上一轮跑出 词表 27 = 初始 14 + 合并 13，说明初始字母表是 14 个。\n"
            "  要把合并次数压到 7 次，VOCAB_SIZE 应该 = 14 + 7 = ?\n"
            "  改完再跑。\n"
        )

    tok = train_bpe(lines, vocab_size=VOCAB_SIZE)

    cfg = json.loads(tok.to_str())
    vocab = cfg["model"]["vocab"]
    merges = cfg["model"]["merges"]

    print("词表大小: %d" % len(vocab))
    print("合并条数: %d" % len(merges))
    for i, m in enumerate(merges, 1):
        print("  %2d  %s + %s" % (i, m[0], m[1]))

    print()
    print("=" * 66)
    print("编码结果（对照第 1 段手算：low=1 / lowest=2 / newer=4）")
    print("=" * 66)

    for word in ["low", "lower", "newest", "widest", "lowest", "newer"]:
        # TODO-3  用 tok.encode(word) 拿到编码对象，打印两样东西：
        #         .tokens  —— 切出来的 token 列表
        #         .ids     —— 对应的整数 id 列表
        #         再打印 token 数
        enc = tok.encode(word)                            # <-- 改这里
        print("%-8s -> %-42s %d token" % (word, str(enc.tokens), len(enc.tokens)))

    print()
    print("往返验证：解码后必须和原词一致")
    for word in ["lowest", "newer"]:
        enc = tok.encode(word)
        back = tok.decode(enc.ids)
        print("  %-8s -> %-30s -> %-10r  一致: %s"
              % (word, str(enc.tokens), back, back == word))

    print()
    print("整句编码（验证多词之间的空格能不能还原）")
    sent = "lowest newer"
    enc = tok.encode(sent)
    print("  %r -> %s -> %r" % (sent, str(enc.tokens), tok.decode(enc.ids)))


if __name__ == "__main__":
    main()
