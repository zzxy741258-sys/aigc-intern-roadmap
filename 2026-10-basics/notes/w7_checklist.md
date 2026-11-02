# W7 逐条对账清单（日历原文，W7 = 11.09—11.15）

> 生成时间：2026-10-27。依据 `calendar_full.txt` 第 34 行（表 2）、第 71 行（表 8）、第 9 行（补充模块①）。
> **规则：每周开始按此清单执行，每周收尾按此清单打勾；未完成项必须显式记录，不许以「段都过了」代替。**

---

## 一、表 2（第 34 行）· 具体任务与产出 + 验收标准

**主题**：Transformer 与 tokenizer

**任务原文拆成 6 条：**

| # | 任务 | 归属段 |
|---|---|---|
| T1 | HF LLM Course 第 1—2 章 | ⚠️ **阅读任务，单独排**（见第四节） |
| T2 | 学习 BPE | 第 1 段 |
| T3 | 学习 token 与 position embedding | 第 1 段（token）+ W6 已覆盖（position embedding） |
| T4 | 学习 causal LM | 第 4 段 |
| T5 | 训练字符级语言模型（小语料、**单头/多头**、生成 200 字） | 第 3—5 段 |
| T6 | 记录 perplexity | 第 5 段 |

**验收标准原文拆成 2 条：**

| # | 验收 | 判据 |
|---|---|---|
| A1 | **能解释 causal mask** | 口述 30 秒，三段式（是什么 → 为什么必须 → 填 0 会怎样）。W6 已学，**W7 要复验**，因为 W7 的模型必须用上它 |
| A2 | **生成脚本可从 checkpoint 恢复** | 关掉进程后，只给 checkpoint 路径就能继续生成，结果与不中断时逐字一致 |

---

## 二、表 8（第 71 行）· 怎么学 / 看什么 / 做什么 / 刷什么

| 维度 | 原文 | 落地动作 |
|---|---|---|
| **怎么学** | tokenizer 与 LM **分开学**；perplexity **随 checkpoint** 记录 | 第 1—2 段只碰 tokenizer，第 3 段起才碰模型；每存一个 checkpoint 同目录写一份 perplexity |
| **看什么资料** | HF LLM Course 第 1—2 章（`huggingface.co/learn/llm-course`）；HF Tokenizers 文档（`huggingface.co/docs`） | ⚠️ **两项阅读任务，见第四节** |
| **做什么项目** | 字符级语言模型（小语料，**单头/多头**），生成 200 字 | 第 3—5 段；**必须跑单头和多头两版做对照** |
| **刷什么题** | **704、35（二分）** | LeetCode 704 二分查找、35 搜索插入位置 |

---

## 三、补充模块①（第 9 行）· W7 的 0.5h 八股配额

原文：**W6—W8 学 RoPE / 位置编码、causal mask、注意力复杂度。**

| 八股主题 | 状态 |
|---|---|
| RoPE / 位置编码 | ✅ W6 已完成 |
| causal mask | ✅ W6 已完成 |
| 注意力复杂度 | ✅ W6 已完成 |

**结论**：日历给的 W7 八股配额**已被 W6 提前用完**。按「补什么最值钱」原则，W7 的 0.5h 转投 **tokenizer 类八股**（日历没列，属建议补充）：

- BPE vs WordPiece vs SentencePiece 的差别
- 为什么中文比英文费 token（UTF-8 一个汉字 3 字节）
- 词表大小怎么影响显存（嵌入矩阵 `vocab × d_model`）

---

## 四、⚠️ 三项最容易重演 W6 事故的漏项（开工前先标出来）

W6 的教训是：**「手写代码」类任务不会漏，「读文档 / 做对照」类任务必漏。** W7 有三项属于后者：

| # | 漏项 | 为什么容易漏 | 什么时候做 |
|---|---|---|---|
| **L1** | **HF LLM Course 第 1—2 章** | 是"看网页"，没有可运行的产出，容易觉得"讲完了就等于看了" | **明天上午，一次性 75 min；硬门禁：`notes/hf_course_guide.md` 末尾 5 道题答完才能开第 2 段** |
| **L2** | **HF Tokenizers 文档** | 同上，而且我讲义里会讲 BPE，容易以为不必看原文 | 第 2 段前扫一遍，约 30 min |
| **L3** | **单头 / 多头两版对照** | 我原先的七段计划里只写了"模型搭建"，**没有显式要求两版对照** —— 这是本次对账抓出来的真实缺口 | 第 4—5 段，多头版额外约 40 min |

**L3 的具体做法**（写清楚，避免又漏）：

```
同一份语料、同一个种子、同样的步数，只改两个参数：
  单头版：n_head=1，d_model=64
  多头版：n_head=4，d_model=64（d_head=16）
记录：最终 perplexity、生成 200 字的质量、参数量
结论必须落到一句可写进简历的话
```

---

## 五、W7 七段执行表（与上面逐条对齐）

| 段 | 内容 | 对应日历条目 | 预计 |
|---|---|---|---|
| 1 | BPE 原理与手算 | T2、T3（token） | 40 min |
| 2 | HF tokenizer 实操（`AutoTokenizer` / `tokenizers` 自训词表） | T3、L2 | 60 min |
| 3 | 字符级 LM：语料与数据构造 | T5 | 40 min |
| 4 | 模型搭建（复用 W6 MHA），**单头版** | T4、T5 | 60 min |
| 5 | 训练 + 生成 200 字 + perplexity（**单头**） | T5、T6 | 60 min |
| 6 | **多头版**同条件重跑 + 对照 | **L3** | 40 min |
| 7 | checkpoint 保存 / 恢复（验收 A2） | A2 | 30 min |
| 8 | 刷题 704、35 | 表 8 | 40 min |
| 9 | **CPU 小数据对照实验环境**（10.28 补充修订新增） | 补充修订（2026.10.28）第一节 | 60 min |
| — | 八股 0.5h（tokenizer 类） | 补充模块① | 30 min |
| — | 阅读 HF LLM Course 1—2 章 + Tokenizers 文档 | **L1、L2** | 90—120 min |

**合计约 8—9 小时**，在每周 20—25h 预算内，留有余量给系统基础支线（第 1 周：B+ 树与索引）。

---

## 六、收尾打勾区（周末填）

| 项 | 状态 | 证据（文件 / 数字） |
|---|---|---|
| T1 HF LLM Course 1—2 章 | ⬜ **顺延 W8** | 硬门禁：`hf_course_guide.md` 5 题 |
| T2 BPE | ✅ | `w7_bpe_demo.py` 10 轮 trace；手算 11+10=21 对比库 14+7=21 |
| T3 token 与 position embedding | ✅ | `w7_tiny.py` 五步；位置向量 64×64 |
| T4 causal LM | ✅ | `w7_gpt.py` 位置 0—62 变化 = 0.0000000000 |
| T5 字符级 LM（单头） | ✅ | `w7_train.py` 训练 3.4687 / 验证 4.2873 / ppl 72.77 |
| T5 字符级 LM（多头） | ✅ | `w7_multihead.py` 参数量同为 181,120，验证 4.2844 vs 4.2897 |
| T6 perplexity 随 checkpoint 记录 | ✅ | `w7_generate.py` 存 ckpt 同时写 ppl 66.48（单 batch） |
| A1 口述 causal mask | ✅ | 精确 0 的浮点解释 + 填 0 会泄漏 32.7% |
| A2 checkpoint 恢复 | ✅ | `del model, opt` 后从 `data/w7_ckpt.pt`（2,160 KB）重新生成 |
| 刷题 704 | ✅ | 二分查找 |
| 刷题 35 | ✅ | 搜索插入位置 |
| 八股 0.5h（tokenizer 类） | ✅ | BPE/WordPiece/SentencePiece + 中文 3.5x vs 1.1x |
| L2 Tokenizers 文档 | ⬜ **顺延 W8** | 30 分钟，只看三点 |
| **CPU 小数据实验循环** | ✅ | 600 步 15 秒；改超参 → 出 loss 循环 < 5 分钟 |
| 系统支线（B+ 树） | ⬜ | 本周主线满负荷，按「主线优先」规则跳过，不补 |
| **L1 + L2 冒烟（`AutoTokenizer`）** | ✅ | `w8_tokenizer_smoke.py` 七段全通过 |

---

## L1 / L2 的定位调整（2026-10-30）

**原定**：放在 W7 末尾作为「阅读任务」。
**问题**：W7 全程用字符级词表，`AutoTokenizer` 一次没碰（当时 HF 缓存坏了，绕过去了）。而 W8 第一件事就是 `AutoTokenizer.from_pretrained(...)` + `padding` + `attention_mask`，这三样目前是零基础。

**结论**：L1/L2 **不是 W7 的尾巴，是 W8 的门槛**。移到 W8 开工前置，不再挂在本周收尾。

顺序（按「先消除环境风险，再补概念」）：

1. **现在 10 分钟**：`AutoTokenizer` 冒烟跑通（见下）→ ✅ **2026.11.02 已完成，`src/w8_tokenizer_smoke.py`**
2. **L2 Tokenizers 文档 30 分钟**：只看三点 —— BPE / WordPiece / SentencePiece 区别；special tokens；`padding` 与 `truncation`
3. **L1 HF Course 1—2 章 75 分钟**：5 题验收

### 冒烟结果（2026.11.02 实测，全部通过）

`src/w8_tokenizer_smoke.py` 七段全通，三个分词器已落在本地：

| 目录 | 用途 | 词表 |
|---|---|---|
| `models/distilbert-base-uncased` | W8 主用（有 `[CLS]/[SEP]/[PAD]`） | 30522，其中单字汉字仅 244（0.8%） |
| `models/gpt2` | 字节级对照（**无 pad token**） | 50257 |
| `models/bert-base-chinese` | 中文字级对照 | 21128，单字汉字 7321（34.7%） |

两个必须记住的实测结论：

1. **中文在英文词表下是「部分可用」而非「全废」**：`机器学习` → `['[UNK]','[UNK]','学','[UNK]']`。比全 UNK 更危险，因为序列看着能跑，语义已缺一半。
2. **「中文比英文贵 3 倍」只对字节级分词器成立**：同一句 13 汉字 / 7 英文词，字节级 28 vs 8 = **3.5x**；中文字级 13 vs 12 = **1.1x**。**贵不贵取决于词表，不取决于语言。**

### 环境结论（2026-10-30 实测）

- `huggingface.co` 直连：**不通**
- `HF_ENDPOINT=https://hf-mirror.com` 走缓存：**卡死被杀**
- **可用路径**：`hf_hub_download(..., local_dir="models/gpt2")` → 14.5 秒完成 → 之后 `AutoTokenizer.from_pretrained("models/gpt2")` 正常

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    hf_hub_download("gpt2", f, local_dir="models/gpt2")
```

**以后所有 HF 模型一律走这个模式：先下到 `models/`，再用本地路径加载。** 不要依赖默认缓存。

