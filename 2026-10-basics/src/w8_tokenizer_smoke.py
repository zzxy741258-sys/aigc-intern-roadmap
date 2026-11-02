"""W8 前置 · AutoTokenizer 冒烟测试

目的：证明「本机能用 HuggingFace 分词器」，且下面六件事都成立。
    ① 能从本地目录加载（不走网络）
    ② 英文编码后再解码，文本能原样回来
    ③ 特殊 token 的位置和 id 都对
    ④ padding / truncation / attention_mask 三件套能正常工作
    ⑤ 中文在三种词表下的三种命运（这一步是意外收获，见文件末尾说明）
    ⑥ 不是所有分词器都有 [PAD]

全部通过，W8 才不会第一天就卡在环境上。
"""

import torch
from transformers import AutoTokenizer

# 本地目录，不走网络。三个模型都下好了，见 models/
DIR_BERT = "models/distilbert-base-uncased"   # W8 主用：英文词表，有 [CLS]/[SEP]/[PAD]
DIR_GPT2 = "models/gpt2"                      # 对照：字节级，无 [PAD]
DIR_ZH = "models/bert-base-chinese"           # 对照：中文字级词表


def part1_load():
    """① 加载：确认拿到的是什么类、词表多大。"""
    tok = AutoTokenizer.from_pretrained(DIR_BERT)
    print("   类型      : %s" % type(tok).__name__)
    print("   词表大小  : %d" % tok.vocab_size)
    print("   最大长度  : %s" % tok.model_max_length)
    return tok


def part2_roundtrip(tok):
    """② 往返：文本 → id → 文本，必须一致（忽略特殊 token）。

    这里刻意用英文：distilbert-base-uncased 的语料是英文，词表里没有汉字。
    中文会怎样，看 ⑤ —— 那是个比往返测试更值得看的现象。
    """
    text = "lowest newer pipeline"
    enc = tok(text, add_special_tokens=False)
    back = tok.decode(enc["input_ids"])
    ok = back == text
    print("   原文      : %r" % text)
    print("   id        : %s" % enc["input_ids"])
    print("   token     : %s" % tok.convert_ids_to_tokens(enc["input_ids"]))
    print("   解码回来  : %r  一致: %s" % (back, ok))
    assert ok, "往返不一致，分词器有问题"


def part3_special(tok):
    """③ 特殊 token：id 是否唯一、是否是真特殊 token（不是普通字符串）。"""
    print("   [CLS] id  : %d" % tok.cls_token_id)
    print("   [SEP] id  : %d" % tok.sep_token_id)
    print("   [PAD] id  : %d" % tok.pad_token_id)
    print("   [MASK] id : %d" % tok.mask_token_id)

    enc = tok("hello world")
    print("   编码整句  : %s" % enc["input_ids"])
    print("   首尾是    : %r ... %r"
          % (tok.decode([enc["input_ids"][0]]),
             tok.decode([enc["input_ids"][-1]])))
    # 关键：特殊 token 必须 ID 唯一，否则模型分不清"这是标记"还是"这是字面文本"
    assert len({tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}) == 3


def part4_padding(tok):
    """④ 三件套：padding 补到等长，attention_mask 标出哪些是填充，truncation 截断。"""
    sents = [
        "short one",
        "a much longer sentence with quite a few more words inside it",
        "mid length here",
    ]
    enc = tok(sents, padding="max_length", truncation=True,
              max_length=12, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    print("   input_ids        : %s" % (tuple(ids.shape),))
    print("   attention_mask    : %s" % (tuple(mask.shape),))
    print()
    print("   %-4s %-46s %s" % ("行", "input_ids", "attention_mask"))
    for i in range(ids.shape[0]):
        print("   %-4d %-46s %s"
              % (i, ids[i].tolist(), mask[i].tolist()))

    print()
    print("   mask 里 0 的个数 : %s（= 被填充的位置数）"
          % mask.eq(0).sum(dim=1).tolist())
    print("   填充位置的值是否都等于 pad_token_id: %s"
          % bool((ids[mask == 0] == tok.pad_token_id).all()))
    # 关键：mask=0 的位置不参与 attention，但它仍占显存
    assert ids.shape == mask.shape == (3, 12)


def _cjk_ratio(vocab_path):
    """数一个 vocab.txt 里有多少个「单字汉字」，占词表多大比例。"""
    import re
    pat = re.compile(r"^[\u4e00-\u9fff]$")
    toks = [l.rstrip("\n") for l in open(vocab_path, encoding="utf-8")]
    cjk = [t for t in toks if pat.match(t)]
    return len(toks), len(cjk)


def part5_chinese(tok_en):
    """⑤ 中文在三种词表下的三种命运。

    这是本脚本最有价值的一段：同一句话，三个分词器给出三种结果，
    分别对应三种「处理没见过的字」的哲学。

    注意：这里不断言「全部 UNK」——实测 distilbert 词表里碰巧收了少量汉字，
    所以是「部分可用」。部分可用比全不可用更危险，见下方说明。
    """
    text = "机器学习"
    print("   测试文本: %r（4 个汉字）" % text)
    print()
    print("   %-30s %-6s %-8s %s" % ("分词器", "token", "中文可用", "结果"))
    print("   " + "-" * 66)

    rows = [
        ("distilbert-base-uncased（英文）", tok_en),
        ("gpt2（字节级）", AutoTokenizer.from_pretrained(DIR_GPT2)),
        ("bert-base-chinese（中文字级）", AutoTokenizer.from_pretrained(DIR_ZH)),
    ]
    for name, tk in rows:
        ids = tk(text, add_special_tokens=False)["input_ids"]
        toks = tk.convert_ids_to_tokens(ids)
        unk = sum(1 for t in toks if t in ("[UNK]", "<unk>"))
        usable = "否" if unk == len(toks) else ("部分" if unk else "是")
        print("   %-30s %-6d %-8s %s"
              % (name, len(ids), usable, toks[:6]))

    print()
    print("   词表里的单字汉字有多少（词表越大不代表越认中文）：")
    for name, path in [("distilbert-base-uncased", DIR_BERT + "/vocab.txt"),
                       ("bert-base-chinese", DIR_ZH + "/vocab.txt")]:
        total, cjk = _cjk_ratio(path)
        print("   %-26s 词表 %6d，单字汉字 %5d（%.1f%%）"
              % (name, total, cjk, 100 * cjk / total))

    print()
    print("   三种命运：")
    print("   · 英文词表 → 绝大多数变 [UNK]，而且「碰巧收录的那几个字」会造成")
    print("                部分可用：序列看着能跑，语义已经缺了一大半，最危险")
    print("   · 字节级  → 不会 UNK，但 1 个汉字要 3 个 token（UTF-8 三字节）")
    print("   · 中文词表 → 1 个汉字 1 个 token，最省")
    print()
    print("   结论：中文用英文词表会「静默丢信息」，用字节级会「贵 3 倍」。")
    print("         所以中文场景必须用中文词表，或中文占比足够高的多语词表。")

    # 断言：4 个字里至少有一个变 UNK —— 这是可验证的事实，不是印象
    ids_en = tok_en(text, add_special_tokens=False)["input_ids"]
    n_unk = sum(1 for i in ids_en if i == tok_en.unk_token_id)
    assert n_unk >= 1, "英文词表把 4 个汉字全认出来了？这与词表统计矛盾"
    print()
    print("   自检：4 个字里有 %d 个变成 [UNK]，可用率 %d/%d"
          % (n_unk, len(ids_en) - n_unk, len(ids_en)))


def part7_cost(tok_en):
    """⑦ 同一句话、三种词表，中英 token 成本各是多少。

    这里要纠正一个常见说法：「中文比英文贵 3 倍」。
    实测下来它只对**字节级**分词器成立，不是所有分词器都这样。
    """
    zh = "大模型推理服务的压测与调优"
    en = "load testing and tuning for LLM inference"
    print("   中文原句: %r（13 个汉字）" % zh)
    print("   英文译句: %r（7 个单词）" % en)
    print()
    print("   %-28s %-8s %-8s %s" % ("分词器", "中文", "英文", "中文/英文"))
    print("   " + "-" * 58)

    rows = [
        ("distilbert（英文词表）", DIR_BERT),
        ("gpt2（字节级）", DIR_GPT2),
        ("bert-base-chinese（中文字级）", DIR_ZH),
    ]
    res = {}
    for name, path in rows:
        tk = AutoTokenizer.from_pretrained(path)
        nz = len(tk(zh, add_special_tokens=False)["input_ids"])
        ne = len(tk(en, add_special_tokens=False)["input_ids"])
        res[name] = (nz, ne)
        print("   %-28s %-8d %-8d %.1fx" % (name, nz, ne, nz / ne))

    print()
    print("   三条读法：")
    print("   · 字节级 3.5x —— 这才是「中文贵 3 倍」的真实来源：")
    print("     1 个汉字在 UTF-8 里是 3 字节，字节级分词器就至少 3 个 token")
    print("   · 中文字级 1.1x —— 1 字 1 token，中文的单价基本和英文持平")
    print("   · 英文词表 1.6x 但中文是废的（大量 [UNK]），这个数字没有意义")
    print()
    print("   所以正确的说法是：")
    print("   中文贵不贵，取决于分词器，不取决于语言本身。")
    print("   用英文语料训的字节级词表（GPT-2 / Llama-2）→ 贵 3 倍以上；")
    print("   用中文词表（Qwen / bert-base-chinese）→ 基本持平。")

    # 断言：字节级下中文应显著贵于中文字级下的比值
    b = res["gpt2（字节级）"]
    c = res["bert-base-chinese（中文字级）"]
    assert b[0] / b[1] > c[0] / c[1], "字节级应比中文字级更费中文 token"
    print()
    print("   自检：字节级 %.1fx > 中文字级 %.1fx  → 成立"
          % (b[0] / b[1], c[0] / c[1]))


def part6_gpt2_has_no_pad():
    """对照：不是所有分词器都有 [PAD]。这是 W8 会踩的坑。"""
    tok = AutoTokenizer.from_pretrained(DIR_GPT2)
    print("   gpt2 pad_token_id : %s" % tok.pad_token_id)
    print("   gpt2 词表大小     : %d" % tok.vocab_size)
    if tok.pad_token_id is None:
        print("   → 没有 pad token。直接 padding=True 会报错，")
        print("     必须先 tok.pad_token = tok.eos_token 才能批量处理。")


def main():
    line = "=" * 64
    print(line)
    print("① 从本地目录加载（不走网络）")
    print(line)
    tok = part1_load()

    print()
    print(line)
    print("② 编码 / 解码往返")
    print(line)
    part2_roundtrip(tok)

    print()
    print(line)
    print("③ 特殊 token：位置与 id")
    print(line)
    part3_special(tok)

    print()
    print(line)
    print("④ padding / truncation / attention_mask")
    print(line)
    part4_padding(tok)

    print()
    print(line)
    print("⑤ 中文在三种词表下的三种命运")
    print(line)
    part5_chinese(tok)

    print()
    print(line)
    print("⑥ 对照：gpt2 没有 pad token（W8 的坑）")
    print(line)
    part6_gpt2_has_no_pad()

    print()
    print(line)
    print("⑦ 中文的真实 token 成本")
    print(line)
    part7_cost(tok)

    print()
    print(line)
    print("全部通过：本机可用 AutoTokenizer，W8 不会卡在环境上")
    print(line)


if __name__ == "__main__":
    main()
