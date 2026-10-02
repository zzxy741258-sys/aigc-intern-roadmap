import numpy as np
np.random.seed(0)
N = 1000
x = np.random.randn(N, 1)
y = 2*x + 3 + 0.1*np.random.randn(N, 1)
w, b, lr = 0.0, 0.0, 0.03
for i in range(300):
    y_hat= x*w+b
    loss=np.mean((y_hat-y)**2)
    dw= (2/N)*x.T@(y_hat-y)
    db= (2/N)*np.sum(y_hat-y)
    w -= lr*dw
    b -= lr*db

    if i%50==0:
        print(i,loss,w,b)
