# Handwritten Digit Recognition System (PyTorch + GUI)

## 📖 项目简介
本项目是一个全栈式的手写数字识别解决方案。不同于标准的 MNIST 示例，本项目重点解决了**真实场景下**手写数字识别率低的问题。通过引入形态学处理，使得模型能更准确地识别用户在桌面 GUI 或网页端书写的数字。



## ✨ 核心功能
* **深度学习模型**: 使用卷积神经网络 (CNN) 在 MNIST 数据集上训练，准确率达到 99% 以上。
* **图像预处理流水线**: 
    * **智能反色**: 自动识别黑底白字或白底黑字。
    * **形态学膨胀 (Dilation)**: 动态增粗手写笔画，模拟训练集特征。
    * **高斯模糊 (Gaussian Blur)**: 平滑边缘，减少环境噪点。
* **交互式界面**:
    * **GUI 版**: 基于 Tkinter 开发的轻量级桌面应用程序。

## 🚀 快速开始

### 1. 环境准备
确保你的 Python 版本为 3.9+，并安装以下依赖：
```bash
pip install torch torchvision opencv-python pillow streamlit streamlit-drawable-canvas
