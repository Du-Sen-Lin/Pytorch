import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

#下载训练集MNIST 手写数字训练集
#有一个常使用的是torchvision.datasets.ImageFolder()，这个可以让我们按文件夹来取图片
# torchvision.transforms里面的操作是对导入的图片做处理：
    # 比如可以随机取(50, 50)这样的窗框大小，或者随机翻转，
    # 或者去中间的(50, 50)的窗框大小部分等等，
    # 但是里面必须要用的是transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='../datasets',
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)
test_dataset = datasets.FashionMNIST(
    root='../datasets', train=False, transform=transforms.ToTensor())

# import matplotlib.pyplot as plt
# plt.subplot(111), plt.imshow(train_dataset.data[600])
# plt.show()

# pytorch 的数据加载到模型的操作顺序是这样的：
# ① 创建一个 Dataset 对象
# ② 创建一个 DataLoader 对象
# ③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
#     dataset,传入的数据集
#     batch_size=1, 每个batch有多少个样本
#     shuffle=False,在每个epoch开始的时候，对数据进行重新排序,训练集中True
#     sampler=None,自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
#     batch_sampler=None,与sampler类似，但是一次只返回一个batch的indices（索引）需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
#     num_workers=0,这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
#     collate_fn=None, 将一个list的sample组成一个mini-batch的函数
#     pin_memory=False,如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
#     drop_last=False,如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
                        # 如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。
#     timeout=0, 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0
#     worker_init_fn=None,每个worker初始化函数
#     multiprocessing_context=None,
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_lodaer = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Logistic Regression 模型
class Logistic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logistic_Regression, self).__init__()
        self.logistic = nn.Linear(in_features=in_dim, out_features=n_class)

    def forward(self, x):
        out = self.logistic(x)
        return out


# 向这个模型传入参数，第一个参数定义为数据的维度，第二维数是我们分类的数目。
model = Logistic_Regression(28*28, 10) #图片大小为28x28
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()


# 定义loss和optimizer
# 多分类问题，对于每一个数据，我们输出的维数是分类的总数，比如10分类，我们输出的就是一个10维的向量，然后我们使用另外一个激活函数，softmax
# 交叉熵损失函数：交叉熵主要是用来判定实际的输出与期望的输出的接近程度
# Pytorch中CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。
criterion = nn.CrossEntropyLoss()
# 随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

import time

# start train
for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch + 1}')
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)  # 将图片展开成 28 x 28
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        # 向前传播
        out = model(img)  # 前向传播
        loss = criterion(out, label)  # 计算loss
        running_loss += loss.item()  # ？记录训练过程中的loss,保存loss以供print
        _, pred = torch.max(out, 1)  # ? 计算正确率，
        running_acc += (pred == label).float().mean()  # ？记录训练过程中Acc,打印查看
        # 向后传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if i % 300 == 0:  # 训练集中60000 / 64 = 937 所以每轮会打印3次
            print(f'[{epoch + 1}/{num_epochs}] Loss:{running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
    print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc; {running_acc / i:.6f}')

    # 每一轮测试一次
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_lodaer:
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
    print(f'Test Loss:{eval_loss / len(test_lodaer):.6f}, Acc: {eval_acc / len(test_lodaer):.6f}')
    print(f'Time:{(time.time() - since):.1f} s')

torch.save(model.state_dict(), './logstic.pth')
