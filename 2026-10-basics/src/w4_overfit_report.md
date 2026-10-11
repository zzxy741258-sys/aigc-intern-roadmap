# W4 过拟合实验报告
## 1. 基线
CNN(3-6-16, FC 400-120-84-10), SGD lr=0.001 momentum=0.9, batch 64, 40 epoch
## 2. 过拟合现象
第23 epoch 起：train acc 继续升至 0.78，val acc 停在 0.64，gap 达 14%
## 3. 处理办法
在分类头两个 FC 后加 Dropout(0.5)（训练随机丢弃，eval 自动关闭）
## 4. 处理前后对比
加 Dropout 后：train 0.6586 / val 0.6228，gap 从 14% 缩至 3%
## 5. 结论
模型容量 > 数据可学信息时过拟合；Dropout 以牺牲少量训练精度换泛化
