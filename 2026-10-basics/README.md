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
