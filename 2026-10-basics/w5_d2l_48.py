import torch, matplotlib
matplotlib.use('TkAgg')
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
d2l.plt.plot(x.detach(), y.detach(), label='sigmoid')
d2l.plt.plot(x.detach(), x.grad, label='gradient')
d2l.plt.legend(); d2l.plt.show()

M = torch.normal(0, 1, size=(4, 4))
print('初始矩阵：\n', M)
for i in range(100):
    M = torch.matmul(M, torch.normal(0, 1, size=(4, 4)))
print('乘 100 次后：\n', M)
