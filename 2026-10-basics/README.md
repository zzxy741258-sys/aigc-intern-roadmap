# aigc-intern-roadmap
AIGC 算法工程师实习学习个人仓库。
每周成果放在按月目录下（2026-10-basics 等），每个目录含
README / requirements / configs / src / scripts / tests。
9.26： 看完python 1-5节 学习git操作
9.27： 看完d2l 2.1 跟敲代码
9.30:  看完d2l 2.3 跟敲代码
10.1\10.2:看完d2l 2.5 2.6 跟敲代码 手写线性回归和softmax回归
10.3:看d2l 3.3 3.7 跟敲
10.5:# W3 FashionMNIST 分类器
MLP（784→256→10）图像分类

## 运行
conda activate aigc
python src/w3_fashion_mnist.py            # 从头训练，10 epoch + 早停
python src/w3_fashion_mnist.py --resume   # 从 best_model.pt 断点续跑

## 结果
- 验证集准确率：best_acc = 0.87（epoch X）
- 数据：FashionMNIST 60000 切 50000/10000
- 模型：MLP 784-256-10 / ReLU / CrossEntropyLoss / SGD lr=0.1
- 随机种子：torch.manual_seed(0)
- 断点恢复：已验证可续跑

## W4 CIFAR-10 CNN 与实验记录
- 模型：CNN(3-6-16, FC 400-120-84-10)，SGD lr=0.001 momentum=0.9，batch 64
- 结果：val_acc 0.62（Dropout 版）；混淆矩阵显示 cat↔dog 互混最严重（349 次）
- 过拟合实验：40 epoch 时 train 0.78 / val 0.64（gap 14%）→ 加 Dropout(0.5) 后 gap 缩至 3.6%
- 产物：w4_overfit_report.md、confusion_matrix.png、wrong_samples.png

## W5 训练诊断、迁移学习与 ML 八股（2026.10.14—10.21，提前于日历 10.26—11.01 完成）

### 本周任务（对照日历 W5）
- D2L 第 5 章阅读：4.4 欠拟合/过拟合、4.5 权重衰减、4.6 Dropout、4.8 数值稳定性（全部跟敲通过）
- MLP 学习率 3 档 × Dropout 3 档九组对照（src/w5_grid.py → w5_grid.csv）
- ResNet18 迁移学习：冻结 backbone vs 全量微调（src/w5_resnet_transfer.py）
- The Illustrated Transformer 阅读（英文原版看不懂，用中文讲解版建立 Attention 直觉）
- ML 经典八股：LR / GBDT-XGBoost / AUC / 过拟合处理
- 补充模块③：SQL（SQLBolt 全部 14 课）+ pandas（10 minutes 跟敲 + 验收流水线）一天集中补齐

### 九组对照实验（固定 seed=0、固定数据切分、固定模型结构，唯一变量 lr/dropout）
| lr \ dropout | 0 | 0.3 | 0.5 |
|---|---|---|---|
| 0.01 | 0.8372 (train 0.8484) | 0.8365 (0.8474) | 0.8343 (0.8451) |
| 0.1 | **0.8748** (0.8939) | 0.8735 (0.8905) | 0.8632 (0.8787) |
| 0.5 | 0.8725 (0.8961) | 0.8414 (0.8569) | 0.8356 (0.8432) |

**验收①影响最大的变量 = 学习率**：
- 固定 dropout=0 看 lr：val 0.8372 → 0.8748 → 0.8725，跨度 **3.76%**（0.01 收敛不足，0.5 开始震荡）
- 固定 lr=0.1 看 dropout：val 0.8748 → 0.8735 → 0.8632，跨度仅 **1.16%**
- 最佳组合 lr=0.1 + dropout=0（val=0.8748）= W3 原始配置
- 关键洞察：小模型（784-256-10）+ FashionMNIST 过拟合 gap 仅 ~2%，dropout 反而压容量；对照 W4 CIFAR CNN（gap 14%→3.6%）dropout 有效——**正则化要用在过拟合真发生处**
- lr=0.5 + dropout=0.5 双随机叠加训练不稳（train 掉到 0.84）

### 迁移学习：ResNet18 冻结 vs 全量（验收②）
| | 冻结 backbone | 全量微调 |
|---|---|---|
| 可训练参数 | 5,130 | 11,181,642（差 2000×） |
| val acc（5 epoch） | 0.77—0.80 震荡，终值 **0.7978** | 单调上升，终值 **0.9345** |
| 耗时 | 421s | 728s（仅 1.7×） |

- 冻结版 80% 上限：CIFAR 32×32 小图 resize 224 后分布偏离 ImageNet 自然大图，特征调不动
- 全量版 93.45%：backbone 微调适配 CIFAR 分布，域适配价值 +13.7 点
- 参数量差 2000× 但耗时仅 1.7×：GPU 瓶颈在前向卷积计算，反向相对便宜；真实代价是显存
- 真实工作流：先冻结出基线 → 解冻全量微调 → backbone 用更小 lr（discriminative lr）

### The Illustrated Transformer（W6 手写 attention 的预热）
- 架构：Encoder×6（MHA + FFN）+ Decoder×6（masked MHA + Cross-Attention + FFN）+ 残差
- Q=我找什么、K=我是什么、V=我传什么；三者是同一输入经三个线性层投影
- Attention = softmax(QKᵀ/√dₖ)V；除以 √dₖ 防止点积过大推入 softmax 饱和区
- 多头 = 多视角并行；Decoder mask 遮未来词；残差 x+SubLayer(x) 解决梯度消失

### ML 经典八股（验收③口述要点）
- **LR**：线性 z → sigmoid → 概率；交叉熵损失（梯度干净 = p-y）；决策边界线性；多分类升级为 softmax
- **GBDT / XGBoost**：串行树拟合负梯度（残差），降偏差；XGBoost = 二阶展开 + 正则 + 特征并行；vs 随机森林（Bagging 并行、降方差）
- **AUC**：随机正样本得分 > 负样本得分的概率；0.5 = 瞎猜；类别不平衡时 accuracy 会骗人，AUC 只看排序能力
- **过拟合处理**：L2 权重衰减（4.5）、Dropout（W4/W5 实操）、早停（W3）、train/val 切分；口述模板"加数据/简化模型 → 正则化 → 早停 + 看 gap"

### 补充模块③：SQL + pandas（一天集中补齐）
- SQL：SQLBolt 全部 14 课（SELECT/JOIN/GROUP BY/窗口函数 ROW_NUMBER OVER PARTITION BY）
- pandas：10 minutes 教程跟敲；read_json/read_csv、选列（单列 Series vs 多列 DataFrame）、布尔过滤、groupby + reset_index、merge how='left'
- 验收脚本 scripts/w5_data_demo.py：造 20 行订单 → 过滤 amount>100 → 按客户分组统计 → merge 客户信息 → 输出 customer_stats.csv，跑通
- 验收锚点：merge how='left' = SQL LEFT JOIN（主表流水必须 left，默认 inner 会丢孤儿数据）；GROUP BY 后不能选普通列（组内多值需聚合函数）

### 失败案例与教训（周记录模板）
- D2L 4.4 噪声 scale 误记为 0.01，实际原网页为 0.1（用户核对纠正）→ 对照实验必须核对源文档，不凭记忆
- sklearn 曾装错环境、pandas 未装：本会话 conda run 被沙箱劫持到错误 python，应直接用 D:\anaconda3\envs\aigc\python.exe；装包后必须验证 import 路径
- 脚本模式 matplotlib 不弹图：需 TkAgg 后端 + d2l.plt.show()

### 本周提交
- src/w5_grid.py、w5_grid.csv、src/w5_resnet_transfer.py、scripts/w5_data_demo.py、data/orders.csv、data/customers.csv、data/customer_stats.csv

### 下周问题清单（W6 手写 attention 待验证）
1. Encoder / Decoder 各几层、每层子模块组成
2. Q/K/V 的来源；Q=(n,d), K=(m,d), V=(m,v) 时输出形状推导
3. 为什么除以 √dₖ
4. 为什么多头
5. Decoder mask 遮什么
6. 残差解决什么
+ W5 待刷：LeetCode 242（有效字母异位词）、349（两数组交集），哈希分桶

## W6 Attention 手写实现（2026.10.22—10.26，提前于日历 11.02—11.08 完成）

### 本周任务（对照日历 W6）
- 手写 scaled dot-product attention、causal mask、multi-head attention
- 与 `nn.MultiheadAttention` 输出对齐，随机输入最大绝对误差 < 1e-5
- 读 Attention Is All You Need 第 3—5 节
- 八股补充：RoPE / 位置编码
- 刷题：LeetCode 167（两数之和 II）、15（三数之和），双指针

### 分段验收（6 段全过）
| 段 | 内容 | 验收锚点 | 结果 |
|---|---|---|---|
| 1 | 形状手推 | X=(3,4)、W_Q/W_K(4→2)、W_V(4→3)；Q/K=(3,2)、QKᵀ=(3,3)、O=(3,3) | ✅ |
| 2 | 单头手写 | 代码输出与手算逐项一致 | ✅ |
| 3 | causal mask | `tril(ones).bool()` + `masked_fill(mask==False, -inf)`（`src/w6_mask.py`） | ✅ |
| 4 | MHA 多头 | `view(n,h,d_k)` 拆头 + `cat` 拼接 + W_O 融合 | ✅ |
| 5 | 对齐官方实现 | 最大绝对误差 **2.980e-08**（验收线 1e-5） | ✅ |
| 6 | 刷题 167 / 15 | 双指针，从两端夹 | ✅ |
| 7 | 八股 RoPE | 位置差为 1 的四组内积全为 0.540302 | ✅ |
| 8 | 八股 causal mask | 填 0 会让 32.7% 注意力泄漏到未来 token | ✅ |
| 9 | 八股 注意力复杂度 | 临界点 `N=4D`（`D=4096` 时 16384）；3 题全对 | ✅ |
| 10 | 论文 §3—5 | 5 题验收 → 见「W6 后补：论文 §3—5 关键结论」 | ✅ |

### 双头对照（真实数字，d_model=4、h=2、d_k=2）
单头（第 1—2 段，d_k=2、d_v=3）：
- scores = `[[2,2,4],[4,0,2],[5,1,4]]`，除以 √2 = 1.4142
- weights 行 0 = `[0.1636, 0.1636, 0.6728]`（第 2 个 token 拿走 67%）
- O[0] = `[1.6728, 0.8364, 1.0000]`

MHA（第 4 段，两个头各看一个 2 维子空间）：
| | 注意力权重 行 0 | 输出 O[0] | 峰值 |
|---|---|---|---|
| 头 0 | `[0.1636, 0.1636, 0.6728]` | `[1.6728, 1.6728]` | 0.6728 |
| 头 1 | `[0.2483, 0.2483, 0.5035]` | `[1.0, 0.7517]` | 0.5035 |

- **同一句话、同一个位置，两个头给出完全不同的注意力分配** → 多头的实证，不是修辞
- 拼接 `cat[0] = [1.6728, 1.6728, 1.0000, 0.7517]`（前 2 列全来自头 0，后 2 列全来自头 1，中间无交互）
- 融合 `final[0] = [2.6728, 2.4246, 1.6728, 1.6728]`；验算 `final[0,0] = 1.6728(头0) + 1.0000(头1) = 2.6728`
- 结论：**cat 是"并列摆放"，W_O 才是"融合"**；W_O 不降维（4→4），做的是跨头混合

### 对齐官方实现（W6 核心验收）
- 配置：n=3、E(embed_dim)=4、h=2、batch=1、bias=False、dropout=0.0、seed=0
- 官方输出形状 (3,4)，注意力权重形状 (1,2,3,3) = (批量, 头数, 词数, 词数)
- **最大绝对误差 2.980e-08 → PASS**（比验收线小约 300 倍）
- 逐头权重完全一致：头 0 行 0 = `[0.283307, 0.42786, 0.288832]`；头 1 行 0 = `[0.499288, 0.256336, 0.244376]`
- 逐头对齐的意义：**证明 `view(n,h,d_k)` 的切头顺序与官方 `reshape+transpose` 等价**，而不只是总输出碰巧对上

**搬权重的关键坑**（本次最大的收获）：
- 官方 `in_proj_weight` 形状 `(3E, E)` = `(12,4)`，Q/K/V 三段**纵向堆叠**，用 `chunk(3, dim=0)` 切
- `nn.Linear` 的权重存的是 `(out, in)`，自己写的是 `(in, out)` → **四个矩阵都要加 `.T`**
- **`w_q` 和 `W_Q` 形状都是 4×4，数值互为转置 —— 形状检查抓不到，只能比数值**

错法代价实测（同一组输入）：
| 错法 | 最大绝对误差 | 相对验收线 |
|---|---|---|
| `W_Q` 没转置（形状仍 4×4） | 1.327e-01 | 大 1.3 万倍 |
| `W_O` 没转置（形状仍 4×4） | 3.789e-01 | 大 3.8 万倍 |
| `√d_k` 写成 `√E`（用 √4 而非 √2） | 3.110e-02 | 大 3 千倍 |

→ 三者都不报错、形状全正常。**"误差 < 1e-5"这条线的作用是把精度问题和算法问题切干净**：差 1e-8 是精度，差 1e-1 是算错。

### 八股：RoPE（旋转位置编码）
- `θ_i(pos) = pos × base^(-2i/d)`，`base = 10000`；d=4 时 i=0 → θ=1.000000 rad，i=1 → θ=0.010000 rad（转速差 100 倍）
- 对 `[1,0,1,0]` 在 pos=1 旋转得 `[0.540302, 0.841471, 0.99995, 0.01]`
- **相对位置性质实测**：`(m,n)=(1,0)/(5,4)/(50,49)/(100,99)` 位置差均为 1，旋转后内积**全部 = 0.540302 = cos(1)**，位置推到 100 也不变
- 旋转是正交变换 → 保模长 → 注意力分数尺度不随位置漂移，且可逆不丢信息
- 外推（4K→32K）：**把 base 调大**（转速变慢，把长序列角度压回训练见过的范围）。锚点：Llama 3 用 `rope_theta=500000` 支撑 128K；YaRN 对高频维不插值、只对低频维插值

### 失败案例与教训
1. **√dₖ 的作用答错**（第 2 段）：曾答"防过拟合"，实际是数值稳定性 —— 点积方差 ≈ d_k，不缩放会让 softmax 饱和、梯度趋 0
2. **"d_v 必须等于 d_k"答错**：反证就在自己的 `src/w6_attention.py` 里（`W_V` 是 `(4,3)`，d_v=3≠d_k=2，一直跑得好好的）。真正约束是 `W_O` 输入 = `h·d_v`、输出 = `d_model`（为了加残差）
3. **答题模式问题（反复出现）**：结论能答对，机制答不出。如 W_O 只说"融合"、RoPE 第 2 题只说"不再依赖位置差"。→ 后续一律按三段式作答：**结论 → 机制 → 后果**
4. **术语密度过高导致理解困难**：讲解中大量使用未展开的英文标识符 → 建 `notes/w6_glossary.md`（十节），并约定"缩略语首次出现必须附中文全称"
5. **脚本跑不出输出**：断点在当前目录 / 解释器路径，非脚本本身。→ 固定用全路径 `D:\anaconda3\envs\aigc\python.exe`，且必须 cd 到 `2026-10-basics`
6. **第 3 段 causal mask 未落盘**：当时为交互式验证，已于 10.26 补 `src/w6_mask.py` 固化（含"填 0 vs 填 -inf"的错误对照）

### causal mask 关键数字（src/w6_mask.py）
| 行 | 无 mask 权重 | 有 mask 权重 |
|---|---|---|
| 第 0 个 token | `[0.1636, 0.1636, 0.6728]` | `[1.0, 0.0, 0.0]` |
| 第 1 个 token | `[0.7679, 0.0454, 0.1867]` | `[0.9442, 0.0558, 0.0]` |
| 第 2 个 token | `[0.6443, 0.0381, 0.3177]` | `[0.6443, 0.0381, 0.3177]`（不变） |

- 第 0 行必然变成 `[1,0,0]`：只有一个可选，概率必为 1；输出直接等于 `V[0] = [1,0,1]`
- 第 2 行完全不变：它是最后一个，本来就没有未来可遮
- **必须填 `-inf` 不能填 `0`**：`softmax(0)=e⁰=1 ≠ 0`，填 0 会让第 0 个 token 仍把 32.7% 的注意力分给两个"未来"token（实测 `[0.6728, 0.1636, 0.1636]`），等于偷看答案；`e^(-inf)=0` 才是真正归零

### 本周提交
- `src/w6_attention.py`：单头 attention（含手算对照）
- `src/w6_mask.py`：causal mask 固化（含填 0 vs 填 -inf 的错误对照）
- `src/w6_mha.py`：MHA 手写（`view` 拆头 + `cat` 拼接 + `W_O`）
- `src/w6_mha_demo.py`：双头数字算例（含 √2 与 √4 的误差对比）
- `src/w6_align.py`：对齐 `nn.MultiheadAttention`（含三种错法的验尸对照）
- `src/w6_rope.py`：RoPE 旋转与相对位置性质验证
- `notes/w6_glossary.md`：W6 术语对照表
- `notes/interview_notes.md`：面经笔记（13 条面经 / 20+ 场面试，按主题归类 + 与日历对照 + 缺口分析）
- `notes/sys_basics.md`：计算机系统基础支线计划（W7—W14，每周 1.5h）

### 运行（W6 脚本）
```powershell
cd "D:\Agent learning\aigc-intern-roadmap\2026-10-basics"
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_attention.py   # 单头，对照手算
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_mask.py        # causal mask
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_mha.py         # MHA 手写
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_mha_demo.py    # 双头数字算例
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_align.py       # 对齐官方，看误差
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w6_rope.py        # RoPE 性质验证
```

### 漏项对账（2026.10.26 发现，10.27 已全部补齐）

日历 W6 要求 8 项 + 补充模块 3 项八股，当时漏掉 3 项：

| 漏项 | 来源 | 状态 |
|---|---|---|
| 读 Attention Is All You Need 第 3—5 节 | 日历表 2、表 8 | ✅ **10.27 补完**（导读 + 5 题验收，见下节） |
| D2L 注意力章节 | 日历表 8 资料清单 | ✅ **10.27 处理**：手写 attention/MHA/RoPE 已等价覆盖代码层；仅扫 §10.7 Transformer 核对三种注意力的 Q/K/V 来源 |
| 注意力复杂度（`L·(4N²D + 16ND²)`） | 补充模块① W6—W8 | ✅ **10.26 补讲 + 3 题全对** |

**根因**：执行时以交接文档的“6 段拆解”为蓝本，而交接文档本身把论文阅读与注意力复杂度漏掉了；手上有 `calendar_full.txt` 却没逐条比对。**“6 段全过”≠“周任务完成”**。
**修正**：自 W7 起，每周开始从 `calendar_full.txt` 抽表 2 + 表 8 逐条列清单；每周收尾按同一清单打勾，未完成项必须显式记录。

---

## W6 后补：论文 §3—5 关键结论（2026.10.27）

导读全文见 `notes/paper_attention_reading.md`。5 道验收题的结果与修正如下。

### 三种注意力的 Q/K/V 来源（对应日历“能画出 Q/K/V 与 mask 的形状”）

| 位置 | Q 来自 | K、V 来自 | mask |
|---|---|---|---|
| 编码器自注意力 | 上一层编码器输出 | 同左 | ❌ |
| 解码器自注意力 | 上一层解码器输出 | 同左 | ✅ causal |
| 编码器-解码器交叉注意力 | 上一层解码器输出 | 编码器最终输出 | ❌ |

口诀：**Q 来自“正在处理的那一侧”，K/V 来自“被查阅的那一侧”。**

### 多头与单头的算力严格相等（可举的证明）

```
单头  d=512：     QKᵀ 2n²·512  +  ×V 2n²·512   = 4n²·512
多头  h=8, d_k=64：每头 2n²·64，8 头 = 2n²·512；×V 同理 = 2n²·512  → 合计 4n²·512

代入 n=100：单头 20,480,000 = 多头 20,480,000（已验证相等）
```

**结论**：多头不是多算 8 倍，是把同样的算力切成 8 份并行用。

### 三个指标不能混（Q4 曾答错，重点记）

| 指标 | 衡量什么 | 决定什么 |
|---|---|---|
| 每层复杂度 | 算力 + 显存 | 要多少卡 |
| 串行操作数 | 能否并行 | GPU 利用率 |
| **最大路径长度** | **信号跨位置要经过几“跳”顺序运算** | **梯度连乘次数 → 长距离依赖好不好学** |

自注意力是 **1 跳**（任意两位置直连），RNN 是 **n 跳**（沿时间步走）。
**所以“Transformer 为什么比 RNN 强”的答案是路径长度，不是复杂度** —— 面试答复杂度会被“N<4D 时 FFN 才是大头”反制。

### post-LN vs pre-LN

```
post-LN:  x_{l+1} = LN( x_l + F(x_l) )     残差流每层被重新归一化，恒等路径被打断
pre-LN :  x_{l+1} = x_l + F( LN(x_l) )     残差流是纯恒等，梯度直达底层
```

- **post-LN 问题**：初始化时输出层附近梯度范数随深度增长 → 不稳定 → **必须 warmup（论文 4000 步）**
- **pre-LN 代价**：残差流范数随深度单调增长 → 深层子层贡献被稀释（curse of depth）；必须补 final LayerNorm；同深度表达力略弱
- 一句话：**post-LN 稳但难训，pre-LN 好训但深层稀释**

### label smoothing 为什么 perplexity 差、BLEU 好

ε=0.1 时平滑后的目标分布自带熵，**困惑度存在下限**（已验证）：

| 词表 V | 最低交叉熵 | 困惑度下界 |
|---|---|---|
| 32000 | 1.3624 nats | 3.906 |
| **37000（论文量级）** | **1.3769 nats** | **3.963** |
| 50257 | 1.4075 nats | 4.086 |

无平滑时理论下界是 **1.0**。所以“perplexity 变差”不是模型变弱，是指标被地板顶住了。
BLEU 变好是因为：翻译有多个正确版本，平滑后模型不再把其他译法压到 0，beam search 有机会选到别的合理路径。

**工程含义**：perplexity 到地板后就不再具备分辨率，**不能只看 perplexity 挑 checkpoint**（会停在 BLEU 还在涨的中途）。且两模型 label smoothing 配置不同时，perplexity 完全不可比。

### 论文已过时的 5 点（面试加分）

| 论文 2017 | 现在 | 原因 |
|---|---|---|
| `LayerNorm(x+Sub(x))` post-LN | `x+Sub(LayerNorm(x))` pre-LN | 见上 |
| 正弦绝对位置编码 | RoPE | 加法编码长上下文外推失效 |
| ReLU | GELU / SwiGLU | SwiGLU 用 2/3 的 `d_ff` 达到同等效果 |
| LayerNorm | RMSNorm | 去掉均值中心化，快 7—15% |
| Encoder-Decoder | Decoder-only | 一套参数做所有任务 |

### 论文配置（可背）

| | base | big |
|---|---|---|
| N / `d_model` / `h` | 6 / 512 / 8 | 6 / 1024 / 16 |
| `d_k = d_v` | 64 | 64 |
| `d_ff` | 2048 | 4096 |
| 参数量 | 65M | 213M |
| 训练 | 8×P100, 12h / 100k 步 | 8×P100, 3.5 天 / 300k 步 |

优化器：Adam，`β1=0.9`、`β2=0.98`、`ε=1e-9`、`warmup=4000`、dropout 0.1、label smoothing 0.1。

## W6 后补：注意力复杂度（2026.10.26）

单层 = 注意力 `4N²D` + FFN `16ND²`，L 层为 `L·(4N²D + 16ND²)`。
- 注意力：`QKᵀ` 为 `2N²D`，再乘 V 为 `2N²D`，合计 `4N²D`
- FFN：升维 `2N·D·4D = 8ND²`，降维同理 `8ND²`，合计 `16ND²`
- 系数 2 = 一次乘加算 2 次浮点运算；`4D` = FFN 隐层取 4 倍

**临界点**：`4N²D = 16ND²` → `N = 4D`。故 `N < 4D` 时瓶颈在 FFN，`N > 4D` 时瓶颈在注意力。

固定 `D=4096` 实测：

| N | 注意力 4N²D | FFN 16ND² | FFN/注意力 |
|---|---|---|---|
| 512 | 4.295e9 | 1.374e11 | 32× |
| 2048 | 6.872e10 | 5.498e11 | 8× |
| 8192 | 1.100e12 | 2.199e12 | 2× |
| **16384 = 4D** | 4.398e12 | 4.398e12 | **1×** |
| 32768 | 1.759e13 | 8.796e12 | 0.5×（注意力反超） |

**工程结论**：主流 `D` 为 4096—8192 → `4D` = 16K—32K。扩到 32K 以上注意力成主成本，必须改架构（MQA/GQA、FlashAttention、稀疏/线性注意力），不是加卡。

---

# W7 · Transformer 与 tokenizer（11.09—11.15，实际 10.27—11.02）

> 一句话：这周从「文本」走到了「能自己写字的小模型」，六站全部跑通，两条硬验收（生成 200 字 + checkpoint 恢复）都拿到。
> 全程 **CPU 完成**，不需要 GPU —— 6GB 显卡在这周完全没参与。

## 六站流水线（所有文件都挂在这六个格子里）

| 站 | 输入 → 输出 | 做什么 | 落在哪个文件 |
|---|---|---|---|
| ① 切分 | 一维文本 → `(B, T)` | 按字符/子词切，查表变整数 | `w7_bpe_demo.py`、`w7_tokenizer.py`、`w7_corpus.py` |
| ② 查表 | `(B, T)` → `(B, T, 64)` | 每个整数取一行向量，加位置向量 | `w7_tiny.py`、`w7_model.py` 前半 |
| ③ 混合 | `(B, T, 64)` → `(B, T, 64)` | attention 交换信息 + FFN 各自变换 | `w7_block.py`、`w7_gpt.py` |
| ④ 打分 | `(B, T, 64)` → `(B, T, 1220)` | 乘共享嵌入矩阵，对每个字之后的所有可能打分 | `w7_model.py` 的 `lm_head` |
| ⑤ 对答案 | `(B, T, 1220)` + `(B, T)` → 一个数 | 交叉熵，所有位置平均 | `w7_model.py` 的 `lm_loss` |
| ⑥ 更新 | 一个数 → 新参数 | 反向传播 + 优化器，回到 ② | `w7_train.py`、`w7_generate.py` |

**贯穿全流程的一句话：B 和 T 从头不动，只有最后一维在变：无 → 64 → 64 → 1220 → 一个数。**

## 本周任务（对照日历 W7 表 2 + 表 8）

| # | 日历要求 | 结果 |
|---|---|---|
| T1 | HF LLM Course 第 1—2 章 | ⬜ **顺延至 W8**（见未完成项） |
| T2 | 学习 BPE | ✅ 手算 10 轮 + 库训练对照 |
| T3 | token 与 position embedding | ✅ 手算 + 代码 |
| T4 | causal LM | ✅ 因果性测试精确 0 |
| T5 | 训练字符级 LM（单头/多头，生成 200 字） | ✅ 两版都跑 |
| T6 | 记录 perplexity | ✅ 随 checkpoint 记录 |
| A1 | 能解释 causal mask | ✅ 复验通过 |
| A2 | 生成脚本可从 checkpoint 恢复 | ✅ `del model` 后仍能生成 |
| — | 刷题 704、35 | ✅ |
| — | 八股 0.5h（tokenizer 类） | ✅ BPE/WordPiece/SentencePiece + 中文成本 |
| L2 | HF Tokenizers 文档 | ⬜ 顺延至 W8 |
| L3 | 单头 / 多头两版对照 | ✅ **对账抓出来的缺口，已补** |
| — | CPU 小数据对照实验环境（10.28 补充修订） | ✅ 600 步 15 秒，改超参循环 5 分钟内 |

## 关键数字

### ① BPE：手算与工业实现对照

语料只有 4 个词（low:5 / lower:2 / newest:6 / widest:3），第 1 轮有三个对并列 9 次，取第一个 `e+s`。

| | 初始符号 | 合并次数 | 词表 | `lowest` 切法 |
|---|---|---|---|---|
| 手算 | 11 | 10 | 21 | `low + est</w>` = 2 |
| `tokenizers` 库 | 14 | 7 | 21 | `lo + w + est</w>` = 3 |

**差 3 个初始符号，是因为库自动把「词尾字符」单列（`w</w>`、`r</w>`、`t</w>`）。**

> **工程铁律：不能用 A 库训的词表和 B 库的 token id 混用。** 换库、换版本、换语料顺序，id 全变。

### ② 语料与数据管道

```
原始文本（本地 notes + README + 日历 + 面经）85,905 字符
  → 去代码块 / 去表格行 / 只保留中文占比 ≥35% 的行
最终语料 32,953 字符，词表 1,220 个字符
训练 / 验证 = 29,657 / 3,296（按尾部切，不随机抽）
每条样本 64 字符 × 32 条 = 2,048 个预测位置 / 步
```

**「中文占比 35%」这个阈值扫了 5 档定下来的**：50% 只剩 2 万字（几分钟背下来），25% 混进大量英文符号（生成乱码）。

### ③ 模型与初始 loss

```
骨架（嵌入 + 位置 + LN + 输出头）   82,304
一个 block（12d² + 2×LN）          49,408
2 层 block                         98,816
合计                              181,120
```

**初始 loss = 7.0902，ln(1220) = 7.1066，差 0.0164。**

这条是整周的体检线：**随机初始化时模型只能在 1220 个字里均匀猜，交叉熵必然是 ln(V)。** 大于它 = 比乱猜还差 = 接线错了。

反面实测：把初始化从 `std=0.02` 改成 PyTorch 默认 `std=1.0`，同一份代码 loss 直接变成 **42.7730**。

### ④ 训练曲线（600 步，CPU 15 秒）

| 步 | 训练 | 验证 | 困惑度 |
|---|---|---|---|
| 0 | 7.1027 | 7.0903 | 1200.33 |
| 200 | 4.4639 | 4.7765 | 118.69 |
| 400 | 3.8198 | 4.4048 | 81.84 |
| 600 | 3.4576 | 4.3407 | 76.76 |

**最终：训练 3.4687 / 验证 4.2873 / 困惑度 72.77**（20 个 batch 平均）

**验证集在 500 步后不再下降（4.3406 → 4.3407），训练集还在降** —— 这是过拟合的起点，也决定了 checkpoint 存哪一步。

### ⑤ 训更久反而更差

| 步数 | 验证 loss | 困惑度 |
|---|---|---|
| **600** | **4.1970** | **66.48** |
| 1500 | 4.6743 | 107.16 |

**181,120 个参数对着 29,657 个字符，600 步之后开始背答案。** 结论：checkpoint 存「验证集最好」那一步，不存最后一步。

### ⑥ 因果性（堆 2 层后）

```
位置 0—62 的最大变化 : 0.0000000000   ← 精确 0
位置 63  的最大变化 : 0.9119579792   ← 动了
去掉 mask 时位置 0—62 变化 : 2.571e-03  ← 泄漏了
```

**堆 2 层仍是精确 0，比单层是 0 强得多** —— 任何一层漏 mask，上层就会把泄漏带下来。

**为什么是精确 0 而不是 1e-8**：`exp(-inf) = 0.0` 在 IEEE 754 里是精确值，整条链没有舍入。

### ⑦ 单头 vs 多头（L3 对照）

| n_head | 每头维度 | 参数量 | 训练 | 验证 |
|---|---|---|---|---|
| 1 | 64 | 181,120 | 3.4698 | **4.2844** |
| 4 | 16 | 181,120 | **3.4262** | 4.2897 |

**参数量完全相同，验证集打平（差 0.005，噪声范围内）。**

多头训练集更低、验证集更高 —— 拟合更好但没转化成泛化。原因：语料只有 3 万字、模型只有 2 层，**没有足够的多样性让 4 个头分工**。

> 面试话术：多头不是无条件更好，它需要「有东西可分」这个前提。业界 d_head=128 这个经验值就是在保证每头的维度足够。

### ⑧ 中文 token 成本（`w8_tokenizer_smoke.py`）

同一句 13 汉字 / 7 英文词，三种分词器：

| 分词器 | 中文 | 英文 | 比值 |
|---|---|---|---|
| distilbert（英文词表） | 13（大半 `[UNK]`） | 8 | 1.6x（数字无意义） |
| **gpt2（字节级）** | **28** | **8** | **3.5x** |
| **bert-base-chinese（中文字级）** | **13** | **12** | **1.1x** |

**「中文比英文贵 3 倍」只对字节级分词器成立。** 贵不贵取决于词表，不取决于语言。

另一个实测：distilbert 词表 30,522 个 token 里，**单字汉字只有 244 个（0.8%）**，所以 `机器学习` → `['[UNK]','[UNK]','学','[UNK]']`。

**「部分可用」比「全废」更危险** —— 序列看着能跑，语义已缺一半，不报错。

## 失败案例与教训（含我自己犯的错）

| # | 现象 | 根因 | 教训 |
|---|---|---|---|
| 1 | 残差实验测「第一层梯度范数」，无残差反而更大 | **测错了指标**：梯度范数反映子层影响力，有残差时子层本就只做小改动 | 改测「输出还认不认得输入」（cos），16 层后 0.72 vs 0.006 |
| 2 | `ln1` 写成 `ln2`，几乎没差别（差 0.007） | LayerNorm 初始 γ=1、β=0，两者初始是同一函数 | **不报错、形状对、查不出的错最危险** |
| 3 | `drop` 包住整个 `x + 子层` | 噪声撒到恒等直通路上 | 验证集差 0.25（4.4330 vs 4.0658） |
| 4 | 漏 `zero_grad()` | `.backward()` 是**累加**不是覆盖 | loss 停在 6.5 抖，600 步 6.5227 vs 3.4576，**不报错** |
| 5 | 我断言「英文词表下中文全 UNK」，脚本断言失败 | 词表里碰巧收了 244 个汉字 | **断言写错比代码写错更危险**，它把错误结论固化 |
| 6 | 我断言「中文贵 3 倍」 | 只对字节级成立，中文词表是 1.1x | 结论要标适用范围，不能一概而论 |
| 7 | 我在 `w7_generate.py` 里写了段训练 loss 的错表达式 | 顺手写的表达式没验证 | 演示脚本里混训练逻辑本身是设计问题 |

**共同点：前 4 条全部「不报错、形状对、浅层看不出」** —— 只能靠对照实验发现，不能靠读代码。

## 本周提交

```
src/w7_bpe_demo.py      BPE 手算全过程（纯标准库）
src/w7_tokenizer.py     tokenizers 库自训词表，与手算对照
src/w7_corpus.py        语料清洗 + 词表 + 训练/验证切分 + batch
src/w7_tiny.py          裸张量版五步：查表→加位置→打分→softmax→loss
src/w7_model.py         模型骨架（嵌入 + 位置 + 权重共享输出头）
src/w7_block.py         单个 Transformer block（裸张量，含因果性验证）
src/w7_gpt.py           装进 nn.Module，堆 2 层
src/w7_train.py         训练循环 + 体检函数 + 两道守卫
src/w7_generate.py      生成 + checkpoint 保存/恢复 + 词表偏移实证
src/w7_multihead.py     单头 / 多头对照实验
src/w8_tokenizer_smoke.py  W8 前置：AutoTokenizer 七段冒烟

notes/w7_bpe.md         BPE 完整讲义（10 轮真实 trace + 词表算术）
notes/w7_skeleton.md    六站骨架图（一页）
notes/w7_checklist.md   逐条对账清单（日历原文对齐）
notes/hf_course_guide.md  HF LLM Course 1—2 章导读 + 5 道必答题

data/w7_corpus.txt      清洗后语料（32,953 字符）
data/w7_ckpt.pt         checkpoint（2,160 KB）
```

## 运行（W7 脚本）

```powershell
cd "D:\Agent learning\aigc-intern-roadmap\2026-10-basics"

& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_bpe_demo.py      # BPE 十轮
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_tokenizer.py     # 库训练词表
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_corpus.py        # 语料与 batch
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_tiny.py          # 裸张量五步
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_model.py         # 骨架 + 初始 loss
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_block.py         # 单 block + 因果性
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_gpt.py           # 堆 2 层 + 残差实验
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_train.py         # 训练 600 步
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_generate.py      # 生成 + checkpoint
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w7_multihead.py     # 单头/多头对照
& "D:\anaconda3\envs\aigc\python.exe" -X utf8 src\w8_tokenizer_smoke.py  # 分词器冒烟
```

## 本周未完成项（对账发现，如实记录）

| 漏项 | 原因 | 去处 |
|---|---|---|
| **HF LLM Course 第 1—2 章**（T1） | 是"看网页"，没有可运行的产出，容易觉得"讲完了就等于看了" | **移到 W8 开工前置，75 分钟，硬门禁：`notes/hf_course_guide.md` 5 题答完才能开 W8** |
| **HF Tokenizers 文档**（L2） | 同上，而且讲义里已讲 BPE，容易以为不必看原文 | W8 前置，30 分钟，只看三点：BPE/WordPiece/SentencePiece 区别、special tokens、padding 与 truncation |

**这两项不是"忘了"，是"知道但一直往后放"。** 已确认它们不阻塞 W7 任何技术产出（W7 用字符级词表，全程没碰 `AutoTokenizer`），但会阻塞 W8 第一天。

## 环境记录：HF 下载的可用路径（重要）

| 路径 | 结果 |
|---|---|
| `huggingface.co` 直连 | ❌ 不通 |
| `HF_ENDPOINT=hf-mirror` 走默认缓存 | ❌ 卡死被杀（`snapshots/` 目录为空，`blobs/` 里有文件） |
| **`hf_hub_download(..., local_dir="models/xxx")`** | ✅ **可用**，14.5 秒下完 |

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    hf_hub_download("gpt2", f, local_dir="models/gpt2")
# 之后一律 AutoTokenizer.from_pretrained("models/gpt2")
```

**以后所有 HF 模型都走这个模式：先下到 `models/`，再用本地路径加载，不依赖默认缓存。**

已就位：`models/gpt2`（50257）、`models/distilbert-base-uncased`（30522）、`models/bert-base-chinese`（21128）。

## 下周问题清单（W8：DistilBERT 情感分类）

0. **W7 顺延（硬门禁）**：HF LLM Course 1—2 章 + Tokenizers 文档，`notes/hf_course_guide.md` 5 题答完才能开 W8
1. `attention_mask` 传进去是自动生效的吗？如果不是，它在模型内部做什么？
2. 分类任务为什么取第一个位置的输出？`[CLS]` 是分词器自动加的，GPT 系为什么没有？
3. 全参微调 DistilBERT（66M）在 6GB 上能不能做？不能的话第一步该冻哪一层？
4. W6/W7 遗留：LeetCode 242、349（W5 暂缓）仍未做
5. checkpoint 恢复为什么要同时存 optimizer state？（W7 自检第 4 题，到现在还没答）
