import csv, torch, torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

device = 'cuda'

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(batch_size=64):
    ds = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor())
    train_set, val_set = random_split(ds, [50000, 10000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_model(dropout):
    layers = [nn.Flatten(), nn.Linear(784, 256), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers).to(device)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total

def train_one(lr, dropout, num_epochs=10):
    set_seed(0)                             # ① 每组重置种子
    train_loader, val_loader = load_data()  # ② 每组重载数据
    model = build_model(dropout)            # ③ 每组重建模型
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)
    return train_acc, val_acc

# 主循环（Step 3）
if __name__ == '__main__':
    lrs = [0.01, 0.1, 0.5]
    dropouts = [0.0, 0.3, 0.5]
    with open('w5_grid.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['lr','dropout','train_acc','val_acc'])
        w.writeheader()
        for lr in lrs:
            for dp in dropouts:
                tr, va = train_one(lr, dp)
                w.writerow({'lr':lr,'dropout':dp,'train_acc':round(tr,4),'val_acc':round(va,4)})
                print(f'lr={lr} dropout={dp} -> train={tr:.4f} val={va:.4f}')
