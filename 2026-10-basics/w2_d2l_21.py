import torch, numpy as np
# 1. 建一个 (3,4) 全零 torch 张量 和 同形状 np 数组
# 2. 用 arange 造 12 个数，reshape 成 (3,4)
# 3. 取第 2 行、第 1-2 列，打印出来
# 4. 造 (3,1) 和 (1,4) 两个数组，相加，打印形状——口述为什么是 (3,4)
# 5. 造 (3,4) @ (4,2)，打印形状——口述为什么是 (3,2)
# 6. b = a.clone() 改 b，看 a 变不变；b = a 再试——口述区别

# x=torch.zeros((3,4))
# xnp=x.numpy()
# print(x)
# print(xnp)
# a=torch.arange(12).reshape((3,4))
# print(a)
# print(a[1,0:2])

# a1=torch.arange(3).reshape((3,1))
# a2=torch.arange(4).reshape((1,4))
# a3=a1+a2
# print(a3.shape)

# b1=torch.arange(12).reshape((3,4))
# b2=torch.arange(8).reshape((4,2))
# b3=b1@b2
# print(b3.shape)

# a=torch.arange(12).reshape((3,4))
# b=a.clone()
# b[2,1]=0
# print(a)
# print(b)


a=torch.arange(12).reshape((3,4))
b=a
b[2,1]=0
print(a)
print(b)

