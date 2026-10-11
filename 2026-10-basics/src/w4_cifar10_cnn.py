import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
train_loader = DataLoader(ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(test_set, batch_size=64, shuffle=False)


model = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(120, 84), nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(84, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += len(X)
    return correct / total

def predict(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X).argmax(1).tolist())
            labels.extend(y.tolist())
    return preds, labels



writer = SummaryWriter('runs/cifar_cnn')

for epoch in range(40):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in train_loader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct += (out.argmax(1) == y).sum().item()
        total += len(X)
    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f'epoch {epoch}: loss={total_loss/total:.4f} '
          f'train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
    writer.add_scalar('loss/train', total_loss / total, epoch)
    writer.add_scalar('acc/train', train_acc, epoch)
    writer.add_scalar('acc/val', val_acc, epoch)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
preds, labels = predict(model, val_loader)
cm = confusion_matrix(labels, preds)          # 10×10 矩阵：行=真实类，列=预测类
ConfusionMatrixDisplay(cm).plot()
import matplotlib.pyplot as plt
plt.savefig('confusion_matrix.png')            # 存图
raw_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        transform=transforms.ToTensor())   # 只 ToTensor，不 Normalize
raw_loader = DataLoader(raw_test, batch_size=64, shuffle=False)
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
wrong = []   # 每项 = (原始图, 真实标签, 预测标签)
for (Xr, y), (X, _) in zip(raw_loader, val_loader):   # 并行遍历，Xr 原始图 / X 标准化图
    p = model(X).argmax(1)
    for i in range(len(y)):
        if p[i] != y[i]:
            wrong.append((Xr[i], y[i].item(), p[i].item()))
fig, axes = plt.subplots(4, 5, figsize=(12, 10))   # 4 行 5 列，20 格
for ax, (img, true, pred) in zip(axes.flat, wrong[:20]):
    ax.imshow(img.permute(1, 2, 0))                # (3,32,32)→(32,32,3) 给 imshow
    ax.set_title(f'真{classes[true]}\n预{classes[pred]}', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('wrong_samples.png')
