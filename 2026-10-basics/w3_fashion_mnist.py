import torch, torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
torch.manual_seed(0)


# 1. 数据（复用第 1 级的 ds，套上 DataLoader）
ds = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                       transform=transforms.ToTensor())

train_set, val_set = random_split(ds, [50000, 10000])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)   # 训练要打乱
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)  # 验证不打乱
# 为什么 shuffle：打乱顺序，防止模型学到样本排列规律

# 2. 模型（三问自己想）
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
# ① Flatten 干嘛的？→ 把 (1,28,28) 拉平成 784 一维，才能喂给 Linear
# ② 784 哪来的？→ 28×28
# ③ 256 是什么？→ 隐藏层宽度（超参数，先抄这个值）

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

best_acc, no_improve, start_epoch = 0.0, 0, 0

if args.resume:
    ckpt = torch.load('best_model.pt', weights_only=True)
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']          # 别漏，否则早停/覆盖逻辑全错
    print(f'从 epoch {start_epoch} 续跑')
# 3. 损失 + 优化器
criterion = nn.CrossEntropyLoss()          # 3.7 学的合体版
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def evaluate(model, loader):
    model.eval()                       # 切到评估模式
    correct, total = 0, 0
    with torch.no_grad():              # 不建计算图，省显存、加速
        for X, y in loader:
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += len(X)
    return correct / total


# 4. 训练：两层 for

    

for epoch in range(start_epoch, 10):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X, y in train_loader:              # 内层套你盲写的 5 行模板
        optimizer.zero_grad()
        out = model(X)                     # out 形状 (64,10)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X) # 累计 loss（乘 batch 大小还原总量）
        correct += (out.argmax(1) == y).sum().item()
        total += len(X)
    train_acc = evaluate(model, train_loader)   # 顺手也测训练集
    val_acc = evaluate(model, val_loader)
    print(f'epoch {epoch}: loss={total_loss/total:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
    if val_acc > best_acc:
        best_acc = val_acc
        no_improve = 0
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_acc': best_acc},
                    'best_model.pt')
    else:
        no_improve += 1
        if no_improve >= 3:
            print(f'早停 @ epoch {epoch}，best_acc={best_acc:.4f}')
            break



