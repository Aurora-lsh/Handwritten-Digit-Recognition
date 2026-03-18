import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import MNISTNet

def smart_invert(img):
    """
    智能反色函数：确保返回黑底白字。
    MNIST数据集（黑底白字）：0代表黑色背景，255代表白色数字。
    """
    # 获取图像边缘像素的平均值（假设边缘属于背景）
    top_mean = np.mean(img[0, :])       # 上边
    bottom_mean = np.mean(img[-1, :])   # 下边
    left_mean = np.mean(img[:, 0])      # 左边
    right_mean = np.mean(img[:, -1])    # 右边
    
    border_avg = (top_mean + bottom_mean + left_mean + right_mean) / 4.0
    
    if border_avg > 127:  # 如果边缘是亮色（白底黑字）
        print(">>> 检测为白底黑字，执行反色。")
        _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    else:  # 如果边缘是暗色（黑底白字）
        print(">>> 检测为黑底白字，保持原样。")
        _, img_bin = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY) # 直接二值化以去除噪点
    return img_bin

def predict_optimized_v2(image_path):
    # --- 1. 初始化模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    # --- 2. 高级图像预处理 v2 (智能反色) ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("错误：无法读取图片！")
        return

    # A. 智能反色处理 -> 获得黑底白字的 img_bin
    img_bin = smart_invert(img)

    # B. 形态学操作：膨胀（Dilation）- 作用是增粗笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilated = cv2.dilate(img_bin, kernel, iterations=1)

    # C. 高斯模糊（Gaussian Blur）- 作用是平滑边缘，减少噪点
    img_blurred = cv2.GaussianBlur(img_dilated, (5, 5), 0)

    # D. 调整尺寸为 28x28
    img_resized = cv2.resize(img_blurred, (28, 28), interpolation=cv2.INTER_AREA)

    # E. 归一化处理（与训练时一致）
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).unsqueeze(0).to(device)

    # --- 3. 推理 ---
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1).max().item()

    # --- 4. 可视化优化过程 ---
    print(f">>> 优化后预测结果为: {prediction} (置信度: {confidence:.2%})")
    
    # 显示预处理的四个阶段
    titles = ['Original', 'Smart_Binary', 'Dilated (Fat)', 'Blurred & Resized']
    images = [img, img_bin, img_dilated, img_resized]
    
    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 继续使用你之前的黑底白字测试图
    predict_optimized_v2("test_digit.jpg")