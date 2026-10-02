import torch

x=torch.arange(4.0)
x.requires_grad_(True)
y=2*torch.dot(x,x)
y.backward()
x.grad.zero_()
y=x.sum()
y.backward()

x.grad.zero_()
y=x*x

y.backward(torch.ones(len(x)))
print(x.grad)
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x
z.backward(torch.ones(len(x)))
print(x.grad==u)