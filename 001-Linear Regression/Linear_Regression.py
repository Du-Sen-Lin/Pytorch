import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# pytorch里面的基本处理单元Tensor，我们需要将numpy转换成Tenso
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        # nn.Linear表示的是 y=w*x+b，里面的两个参数都是1，
        # 表示的是x是1维，y也是1维。当然这里是可以根据你想要的输入输出维度来更改
        self.linear = nn.Linear(in_features=1, out_features=1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = linearRegression()
#定义loss和优化函数:随机梯度下降
criterion = nn.MSELoss() #均方误差
# torch.optim是一个实现了多种优化算法的包，大多数通用的方法都已支持
# 使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，
# 并能够根据计算得到的梯度来更新参数。
# 注意需要将model的参数model.parameters()传进去让这个函数知道他要优化的参数是那些。
# 这里测试了一下，SGD效果比Adam好很多
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# start train
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)  # 前向传播
    loss = criterion(out, target)  # 计算loss
    # backward 每次反向传播的时候需要将参数的梯度归零
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # 打印loss结果
    if (epoch + 1) % 20 == 0:
        print(f'Epoch[{epoch + 1}/{num_epochs}], loss: {loss.item():.6f}')

# 测试模型：model.eval()，让model变成测试模式，
# 这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
model.eval()

# with torch.no_grad():，强制之后的内容不进行计算图构建。 减小显存占用
with torch.no_grad():
    predict = model(x_train)

predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
# 显示图例
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), './linear.pth')

