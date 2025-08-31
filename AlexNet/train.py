import torch
from torchvision.datasets import FashionMNIST
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time

from model import AlexNet

def train_val_data_process(batch_size=32, validation_split=0.2):
    train_val_data = FashionMNIST(root="./data",
                                  train=True,
                                  transform = transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                                  download=True)
    
    train_size = int((1 - validation_split) * len(train_val_data))
    val_size = len(train_val_data) - train_size

    train_data, val_data = Data.random_split(train_val_data, [train_size, val_size],
                                                         generator=torch.Generator().manual_seed(42))
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=2)  
    return train_dataloader, val_dataloader

def train_val_process(model, train_dataloader, val_dataloader, num_epochs):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:{}".format(device))

    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    # 保存当前时间
    since_original = time.time()

    # 开始训练过程
    for epoch in range(num_epochs):
        
        since = time.time()

        print("EPOCH: {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # 初始化参数
        train_corrects = 0
        train_loss = 0.0
        train_num = 0

        val_corrects = 0
        val_loss = 0.0
        val_num = 0

        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            
            # 将数据放入设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            train_loss += loss.item() * inputs.size(0)
            train_num += labels.size(0)
            pre_lab = torch.argmax(outputs, dim=1)
            train_corrects += (pre_lab == labels).sum().item()

        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_dataloader):

                # 将数据放入设备
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)

                # 统计信息
                val_loss += loss.item() * inputs.size(0)
                val_num += labels.size(0)
                pre_lab = torch.argmax(outputs, dim=1)
                val_corrects += (pre_lab == labels).sum().item()
        
        # 统计信息
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)

        print("{} train loss:{:.4f} train acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if train_acc_all[-1] > best_acc:
            best_acc = train_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use_total = time.time() - since_original
        time_use_each_epoch = time.time() - since
        print("The toal time consumed for training and validation{:.0f}m{:.0f}s".format(time_use_total//60, time_use_total%60))
        print("The time consumed for each epoch{:.0f}m{:.0f}s".format(time_use_each_epoch//60, time_use_each_epoch%60))

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
    model_train = AlexNet()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_val_process = train_val_process(model_train, train_data, val_data, num_epochs=20)
    matplot_acc_loss(train_val_process)

