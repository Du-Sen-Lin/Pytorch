import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义超参数
batch_size = 64
learning_rate = 1e-2
num_epochs = 50
use_gpu = torch.cuda.is_available()

# 加载数据集
train_dataset = datasets.FashionMNIST(
    root = '../datasets/',
    train=True,
    transform=transforms.ToTensor(),
    target_transform=None,
    download=False,)
test_dataset = datasets.FashionMNIST(
    root = '../datasets/', train=False, transform=transforms.ToTensor())

len(train_dataset)


from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.imshow(train_dataset.data[0])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义简单的前馈神经网络
class neuralNetWork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetWork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = neuralNetWork(
    in_dim=28*28,
    n_hidden_1=300,
    n_hidden_2=100,
    out_dim=10)
if use_gpu:
    model = model.cuda()

# 定义loss和optimizer
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

import time

for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch + 1}')
    since = time.time()
    running_loss = 0.0  # 记录训练过程中的loss和acc
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        #         print(img.shape)  # torch.Size([64, 1, 28, 28])
        #         print(label) #tensor([9, 8, ..., 5]) 64个label,batch_size = 64
        img = img.view(img.size(0), -1)  # torch.Size([64, 784])
        #         print(img.shape)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item()
        # torch.max(a, 1)意思是指对tensor a而言取其行方向的最大值，结果为列（也就是1所示的维度）
        # 类似的，torch.max(a, 0)意思是指对tensor a而言取其列方向的最大值，结果为行（也就是0所示的维度）
        # 函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，
        # 所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().mean()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:  # 训练集中60000 / 64 = 937 所以每轮会打印3次
            print(f'[{epoch + 1}/{num_epochs}] Loss:{running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
    print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc; {running_acc / i:.6f}')

    # 每一轮验证一次
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()
    print(f'Test Loss: {eval_loss / len(test_loader):.6f}, Acc: {eval_acc / len(test_loader):.6f}\n')
    print(f'Time:{(time.time() - since):.1f} s')