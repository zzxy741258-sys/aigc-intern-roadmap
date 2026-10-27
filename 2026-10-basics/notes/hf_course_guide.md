# HF LLM Course 第 1—2 章 导读与验收（W7 · L1/L2）

> 日历 W7 表 8 原文：「HF LLM Course 第 1—2 章（huggingface.co/learn/llm-course）」
> 这是 W7 唯一的**纯阅读任务**。W6 的教训是读文档类任务必漏，所以这份导读配了**硬验收**：5 道题答完才算完成。

---

## 0. 访问地址（关键：直连不通）

| 地址 | 状态 |
|---|---|
| `https://huggingface.co/learn/llm-course/chapter1/1` | ❌ 直连失败（代理 502） |
| **`https://hugging-face.cn/learn/llm-course/chapter1/1`** | ✅ **中文镜像，实测可达，内容同步** |

**用后者。** 章节编号两边一致，把 `huggingface.co` 换成 `hugging-face.cn` 即可。

---

## 1. 第 1 章「Transformer models」真实目录

| 节 | 标题 | 读不读 |
|---|---|---|
| 1/1 | Introduction | 扫 |
| 1/2 | Natural Language Processing and Large Language Models | 扫（概念综述） |
| 1/3 | Transformers, what can they do? | 快速看（pipeline 演示） |
| **1/4** | **How do Transformers work?** | **必读** |
| **1/5** | **How 🤗 Transformers solve tasks** | **必读** |
| **1/6** | **Transformer Architectures** | **必读（最重要）** |
| 1/7 | Quick quiz | **做** |
| **1/8** | **Inference with LLMs** | **必读** |
| 1/9 | Bias and limitations | 扫 |
| 1/10 | Summary | 扫 |
| 1/11 | Certification exam | 跳过 |

## 2. 第 2 章「Using 🤗 Transformers」真实目录

| 节 | 标题 | 读不读 |
|---|---|---|
| 2/1 | Introduction | 扫 |
| **2/2** | **Behind the pipeline** | **必读**（tokenizer → model → post-processing 三段流水线） |
| **2/3** | **Models** | **必读**（`AutoModel` 家族、保存/加载） |
| **2/4** | **Tokenizers** | **必读**（词级 / 字符级 / 子词级三种切法对比） |
| **2/5** | **Handling multiple sequences** | **必读**（padding、attention mask、batch） |
| 2/6 | Putting it all together | 扫 |
| 2/7 | Basic usage completed! | 跳 |
| 2/8 | Optimized Inference Deployment | **跳过**（W19 再读） |
| 2/9 | End-of-chapter quiz | **做** |

**时间预算**：Ch1 约 35 min（含 quiz），Ch2 约 40 min。**合计 75 min，一次做完。**

---

## 3. 五道必答题（答完写在本文件末尾，才算完成 L1）

> 要求按三段式：结论 → 机制 → 后果。**不许抄页面原文。**

1. **三种架构的本质区别**：encoder-only（如 BERT）、decoder-only（如 GPT）、encoder-decoder（如 T5）——**各自的 attention 能看到哪些位置？** 为什么 encoder-only 不能直接拿来做文本生成？

2. **第 2 章 2/4 讲了三种切词方式**（词级 / 字符级 / 子词级）。**子词级具体是哪个算法？** 它和你在第 1 段手算的 BPE 是同一件事吗？

3. **`padding` 和 `attention mask` 各解决什么问题？** 如果不给 attention mask，pad 出来的 token 会怎样影响结果？

4. **`AutoModel` / `AutoModelForSequenceClassification` / `AutoModelForCausalLM` 的区别是什么？** 同一个 checkpoint 能加载成不同 head 吗？为什么能？

5. **第 1 章 1/8「Inference with LLMs」讲了什么？** 用一句话说清“推理”和“微调”的分界线在哪。

---

## 4. 这 5 题与后面周次的对应

| 题 | 会用到它的地方 |
|---|---|
| 1 | W7 第 4 段（自己搭 causal LM，必须解释为什么用 masked self-attention） |
| 2 | W7 第 1—2 段（BPE 与库实现对照） |
| 3 | W8（DistilBERT 中文分类，`padding` 是必踩的坑） |
| 4 | W8 第 2 段（`Trainer` + `AutoModelForSequenceClassification`） |
| 5 | W9—W10（SFT 微调 vs 推理的边界） |

---

## 5. 答案区（自己写，不许复制页面）

**Q1**



**Q2**



**Q3**



**Q4**



**Q5**
