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
