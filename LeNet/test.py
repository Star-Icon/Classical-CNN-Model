import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data

from model import LeNet

def test_data_process(batch_size=32):
    test_data = FashionMNIST(root="./data",
                             train=False,
                             transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]),
                             download=True)
    
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=0)
    return test_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_process(model, test_dataloader):
    
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        
        inputs, labels  = inputs.to(device), labels.to(device)

        with torch.no_grad():

            model.eval()

            outputs = model(inputs)
            pre_lab = torch.argmax(outputs, dim=1)
            test_corrects += (pre_lab == labels).sum().item()
            test_num += inputs.size(0)

    test_acc = test_corrects / test_num
    print("测试准确率为：{:.4f}".format(test_acc))

if __name__ == "__main__":
    model = LeNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    test_data = test_data_process()
    test_process = test_process(model, test_data)




    

