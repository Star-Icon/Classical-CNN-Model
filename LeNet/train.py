import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.nn as nn
import copy
import time

from model import LeNet

def train_val_data_process(batch_size=32,validation_split=0.2):
    data = FashionMNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=28),
                                                    transforms.ToTensor()]),
                        download=True)
    
    train_size = int((1 - validation_split) * len(data))
    val_size = len(data) - train_size
    train_data, val_data = Data.random_split(data, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(42)) # 随机种子保证可重复性

    train_dataloader = Data.DataLoader(dataset=train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2)
    
    val_dataloader = Data.DataLoader(dataset=val_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=2)
    return train_dataloader, val_dataloader

# # 定义类别名称
# classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

def train_process(model, train_dataloader, val_dataloader, num_epochs):

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:{}".format(device))

    model = model.to(device)

    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    # 保存当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("EPOCH {}/{}".format(epoch, num_epochs-1))
        print(10*"-")

        # 初始化参数
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):

            # inputss, labels放入到device
            inputs, labels = inputs.to(device), labels.to(device)

            # 设置模型为训练模式（这会启用dropout和batch normalization的训练行为）
            model.train()

            # 前向传播：讲输入数据传入模型得到输出
            outputs = model(inputs)

            # 计算损失：比较模型的输出和真实标签
            loss = criterion(outputs, labels) 

            # 反向传播
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播：计算梯度
            optimizer.step() # 更新参数

            # 统计信息
            train_loss += loss.item() * inputs.size(0)
            pred_lab = torch.argmax(outputs, dim=1)
            train_corrects += (pred_lab == labels).sum().item()
            train_num += labels.size(0)
        
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):

            # inputs, labels放入到device
            inputs, labels = inputs.to(device), labels.to(device)

            # 设置模型为评估模式（这会禁用dropout和batch normalization的训练行为）
            model.eval()

            # 前向传播：讲输入数据传入模型得到输出
            outputs = model(inputs)

            # 计算损失：比较模型的输出和真实标签
            loss = criterion(outputs, labels) 

            # 统计信息
            val_loss += loss.item() * inputs.size(0)
            pred_lab = torch.argmax(outputs, dim=1)
            val_corrects += (pred_lab == labels).sum().item()
            val_num += labels.size(0)

        # 计算每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)

        print("{} train loss:{:.4f} train acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if train_acc_all[-1] > best_acc:
            best_acc = train_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("The time consumed for training and validation{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, "best_model.pth")

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all,})

    return train_process

def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 加载需要的模型
    model_train = LeNet()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_process(model_train, train_data, val_data, num_epochs=20)
    matplot_acc_loss(train_process)


    






