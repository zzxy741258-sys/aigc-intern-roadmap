import torch, torch.nn as nn, torchvision, time
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

device = 'cuda'

# ① ResNet 预训练时用的是 224x224 输入 + ImageNet 统计量，必须对齐
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_ds  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

def build_model(freeze):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)   # 新版 API，自动下载预训练权重
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)      # 锁住所有层
    model.fc = nn.Linear(512, 10)       # 换掉最后分类头：512特征 -> 10类
    return model.to(device)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.01, momentum=0.9)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                correct += (model(X).argmax(1) == y).sum().item()
                total += len(y)
        print(f'  epoch {epoch}: val_acc={correct/total:.4f}')
    print(f'  耗时 {time.time()-t0:.1f}s')


print('=== 冻结版（只训分类头）===')
m1 = build_model(freeze=True)
print('可训练参数:', count_trainable(m1))
train(m1, epochs=5)

print('=== 全量微调版 ===')
m2 = build_model(freeze=False)
print('可训练参数:', count_trainable(m2))
train(m2, epochs=5)
