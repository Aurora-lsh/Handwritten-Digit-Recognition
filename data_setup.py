# 下载MNIST数据集并进行预处理

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def download_mnist():
    # 定义预处理步骤：转为张量并归一化（均值0.1307，标准差0.3081是MNIST的标准值）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载训练集
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 下载测试集
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(f"训练集样本数: {len(train_set)}")
    print(f"测试集样本数: {len(test_set)}")
    
    return train_set

if __name__ == "__main__":
    train_data = download_mnist()
    
    # 可视化部分：看看数据集长什么样
    sample_loader = torch.utils.data.DataLoader(train_data, batch_size=9, shuffle=True)
    images, labels = next(iter(sample_loader))
    
    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()