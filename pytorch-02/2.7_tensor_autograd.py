
import torch
import torch as t
# %matplotlib inline
from matplotlib import pyplot as plt

t.manual_seed(100) 
dtype = t.float
#生成x坐标数据，x为tenor，形状为100x1
x = t.unsqueeze(torch.linspace(-1, 1, 100), dim=1) 
#生成y坐标数据，y为tenor，形状为100x1，另加上一些噪音
y = 3*x.pow(2) +2+ 0.2*torch.rand(x.size())                 

# 画图，把tensor数据转换为numpy数据
plt.scatter(x.numpy(), y.numpy())
# plt.show()
plt.savefig('2.7_1.png')

# 随机初始化参数，参数w，b为需要学习的，故需requires_grad=True
w = t.randn(1,1, dtype=dtype,requires_grad=True)
b = t.zeros(1,1, dtype=dtype, requires_grad=True) 


lr =0.001 # 学习率

for ii in range(800):
    # forward：计算loss
    y_pred = x.pow(2).mm(w) + b
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    
    # backward：自动计算梯度
    loss.backward()
    
    # 手动更新参数，需要用torch.no_grad()更新参数
    with t.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 梯度清零
        w.grad.zero_()
        b.grad.zero_()



plt.plot(x.numpy(), y_pred.detach().numpy(),'r-',label='predict')#predict
plt.scatter(x.numpy(), y.numpy(),color='blue',marker='o',label='true') # true data
plt.xlim(-1,1)
plt.ylim(2,6)  
plt.legend()
# plt.show()
plt.savefig('2.7_2.png')
        
print(w, b)


