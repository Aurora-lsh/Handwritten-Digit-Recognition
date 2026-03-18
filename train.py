# 训练脚本：负责加载数据、定义模型、训练模型并保存权重文件
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet  # 导入我们刚刚定义的模型

def train():
    # 1. 自动选择硬件 (优先使用英伟达GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    # 2. 数据预处理与加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 官方标准归一化参数
    ])

    print(">>> 正在准备数据...")
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    # 3. 实例化模型和优化器
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # 损失函数

    # 4. 开始训练
    print(">>> 开始训练模型...")
    model.train()
    epochs = 5  # 训练5轮即可达到约98-99%准确率
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()           # 梯度清零
            output = model(data)            # 前向传播
            loss = criterion(output, target)# 计算误差
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新权重
            
            total_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f"--- 第 {epoch} 轮训练结束，平均 Loss: {avg_loss:.4f} ---")

    # 5. 保存模型权重文件
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("\n>>> 训练成功！模型已保存为: mnist_cnn.pth")

if __name__ == "__main__":
    train()