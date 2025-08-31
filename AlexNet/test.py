import torch
from torchvision.datasets import FashionMNIST
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt


from model import AlexNet

def test_data_process(batch_size=128):
    test_data = FashionMNIST(root="./data",
                                  train=False,
                                  transform = transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                                  download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=4)  
    return test_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:{}".format(device))

def test_process(model, test_dataloader):

    model = model.to(device)


    # 初始化参数
    test_corrects = 0
    test_num = 0


    for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        
        # 将数据放入设备
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():

            model.eval()

            # 前向传播
            outputs = model(inputs)

            # 统计信息

            test_num += labels.size(0)
            pre_lab = torch.argmax(outputs, dim=1)
            test_corrects += (pre_lab == labels).sum().item()

    test_acc = test_corrects / test_num
    print("测试准确率为：{:.4f}".format(test_acc))


if __name__ == "__main__":
    model = AlexNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    test_data = test_data_process()
    test_process = test_process(model, test_data)


