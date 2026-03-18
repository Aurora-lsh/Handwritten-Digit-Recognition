# Handwritten Digit Recognition System (PyTorch + GUI)

## 📖 项目简介
本项目是一个全栈式的手写数字识别解决方案。不同于标准的 MNIST 示例，本项目重点解决了**真实场景下**手写数字识别率低的问题。通过引入形态学处理，使得模型能更准确地识别用户在桌面 GUI 或网页端书写的数字。

![Handwritten digit recognition](https://github.com/user-attachments/assets/57f177c7-8652-4734-b540-99ba5be8894e)


## ✨ 核心功能
* **深度学习模型**: 使用卷积神经网络 (CNN) 在 MNIST 数据集上训练，准确率达到 99% 以上。
* **图像预处理流水线**: 
    * **智能反色**: 自动识别黑底白字或白底黑字。
    * **形态学膨胀 (Dilation)**: 动态增粗手写笔画，模拟训练集特征。
    * **高斯模糊 (Gaussian Blur)**: 平滑边缘，减少环境噪点。
* **交互式界面**:
    * **GUI 版**: 基于 Tkinter 开发的轻量级桌面应用程序。

---

## 📂 文件说明
- `gui_app.py`: **[推荐]** 基于 Tkinter 的桌面版绘图识别工具。
- `app.py`: 基于 Streamlit 的交互网页版。
- `predict.py`: 存放核心的图像预处理与推理逻辑。
- `model.py`: 定义神经网络结构。
- `train.py`: 自动化训练脚本。
- `mnist_cnn.pth`: 预训练好的模型权重文件。

---

## 🚀 快速开始
### 1. 安装环境
```bash
pip install torch torchvision opencv-python pillow streamlit

---

## 2. 运行桌面交互版 (GUI)
```bash
python gui_app.py


🛠️ 常见问题排查
ModuleNotFoundError: 请确保已激活对应的虚拟环境并执行了 pip install -r requirements.txt。

识别偏移: 如果笔画过细，可以在 gui_app.py 中尝试调大 dilation_iter 参数。

🤝 贡献与反馈
如果你有任何改进建议，欢迎提交 Issue 或 Pull Request！
