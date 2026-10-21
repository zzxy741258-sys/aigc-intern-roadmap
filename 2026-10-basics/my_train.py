import torch
import matplotlib.pyplot as plt


def evaluate_accuracy(net, data_iter):
    """计算模型在给定数据集上的准确率"""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def evaluate_loss(net, data_iter, loss):
    """计算模型在给定数据集上的平均损失"""
    net.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            l = loss(out, y)
            if l.dim() > 0:
                l = l.mean()
            total_loss += l.item() * len(y)
            total += len(y)
    return total_loss / total


def train(net, train_iter, test_iter, loss, num_epochs, trainer,
          print_every=1, verbose=True, plot=True, save_path=None):
    """
    训练函数，等价于 d2l.train_ch3

    参数：
        net         : 模型
        train_iter  : 训练数据迭代器
        test_iter   : 测试数据迭代器
        loss        : 损失函数
        num_epochs  : 训练轮数
        trainer     : 优化器
        print_every : 每多少轮打印一次
        verbose     : 是否打印
        plot        : 训练结束后是否画图
        save_path   : 图片保存路径（None 则不保存）

    返回：
        history: {'train_loss': [...], 'train_acc': [...], 'test_acc': [...]}
    """
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        net.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_iter:
            trainer.zero_grad()
            out = net(X)
            l = loss(out, y)
            if l.dim() > 0:
                l = l.mean()
            l.backward()
            trainer.step()

            total_loss += l.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total

        # ===== 评估阶段 =====
        test_acc = evaluate_accuracy(net, test_iter)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if verbose and (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1:2d} | '
                  f'train loss {train_loss:.4f} | '
                  f'train acc {train_acc:.4f} | '
                  f'test acc {test_acc:.4f}')

    # ===== 训练结束，画图 =====
    if plot:
        _plot_history(history, save_path)

    return history


def _plot_history(history, save_path=None):
    """画训练/测试损失和准确率曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：损失
    axes[0].plot(epochs, history['train_loss'], 'o-', label='train loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Loss')
    axes[0].grid(True)
    axes[0].legend()

    # 右图：准确率
    axes[1].plot(epochs, history['train_acc'], 'o-', label='train acc')
    axes[1].plot(epochs, history['test_acc'], 's-', label='test acc')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].set_title('Accuracy')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f'图片已保存到 {save_path}')

    plt.show()