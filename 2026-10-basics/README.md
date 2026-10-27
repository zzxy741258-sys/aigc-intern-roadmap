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

### 下周问题清单（W7：Transformer 与 tokenizer）
0. **W6 顺延**：读 Attention Is All You Need 第 3—5 节（重点 §4 Why Self-Attention，即复杂度对比表）+ D2L 注意力章节
1. BPE 的 `merges` 为什么必须严格按学到的顺序执行？（W7 自检第 3 题，未答）
2. 真实模型的 base 词表是「字节」还是「字符」？一个汉字 UTF-8 占 3 字节，这能解释中文比英文更费 token 吗？
3. causal mask 在字符级语言模型里具体加在哪一步？
4. checkpoint 恢复为什么要同时存 optimizer state？只存模型权重会丢什么？
5. W6 遗留：LeetCode 242、349（W5 暂缓）仍未做
