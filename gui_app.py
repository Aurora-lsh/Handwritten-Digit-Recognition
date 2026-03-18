import tkinter as tk
from tkinter import font as tkfont
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 尝试导入自定义模型，如果失败则使用占位类防止报错（方便你直接运行看界面效果）
try:
    from model import MNISTNet
except ImportError:
    print("未找到 model.py，使用模拟模型类...")
    class MNISTNet(torch.nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x): return torch.randn(1, 10) # 模拟输出

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 v1.0")
        self.root.geometry("900x600")
        self.root.resizable(False, False)
        
        # --- 配色方案 (现代深色风) ---
        self.colors = {
            'bg': '#2b2b2b',          # 主背景
            'frame_bg': '#3c3f41',    # 卡片背景
            'canvas_bg': '#000000',   # 画布背景
            'text_main': '#ffffff',   # 主文字
            'text_sub': '#aaaaaa',    # 副文字
            'accent': '#4CAF50',      # 强调色 (绿色)
            'accent_hover': '#45a049',
            'danger': '#ff5252',      # 清除按钮色
            'danger_hover': '#ff1744',
            'brush': '#ffffff'        # 画笔颜色
        }

        self.root.configure(bg=self.colors['bg'])

        # 1. 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = MNISTNet().to(self.device)
            # 请确保 mnist_cnn.pth 在当前目录，否则这里会报错
            self.model.load_state_dict(torch.load("mnist_cnn.pth", map_location=self.device))
            self.model.eval()
            self.model_loaded = True
        except FileNotFoundError:
            print("警告：未找到 mnist_cnn.pth，模型将使用随机权重进行测试。")
            self.model = MNISTNet().to(self.device)
            self.model_loaded = False

        # 2. 初始化画布参数
        self.canvas_size = 400  # 显示大小
        self.image_size = 280   # 实际处理大小 (保持比例)
        self.brush_size = 20    # 默认笔刷大小
        
        # 后台 PIL 图像 (灰度模式)
        self.image = Image.new("L", (self.image_size, self.image_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 状态变量
        self.last_x = None
        self.last_y = None

        # 3. 构建 UI
        self.setup_ui()

    def setup_ui(self):
        # 自定义字体
        title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
        label_font = tkfont.Font(family="Helvetica", size=14)
        btn_font = tkfont.Font(family="Helvetica", size=12, weight="bold")

        # === 左侧：绘图区 ===
        left_frame = tk.Frame(self.root, bg=self.colors['frame_bg'], padx=20, pady=20, relief=tk.FLAT)
        left_frame.place(x=20, y=20, width=460, height=560)

        tk.Label(left_frame, text="请在下方书写数字", font=label_font, fg=self.colors['text_sub'], bg=self.colors['frame_bg']).pack(pady=(0, 10))

        # 画布容器 (加边框效果)
        canvas_container = tk.Frame(left_frame, bg=self.colors['text_main'], bd=0)
        canvas_container.pack()

        self.canvas = tk.Canvas(
            canvas_container, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg=self.colors['canvas_bg'],
            highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack()

        # 笔刷大小调节
        brush_frame = tk.Frame(left_frame, bg=self.colors['frame_bg'])
        brush_frame.pack(pady=15, fill='x')
        tk.Label(brush_frame, text="笔刷大小:", font=label_font, fg=self.colors['text_sub'], bg=self.colors['frame_bg']).pack(side=tk.LEFT)
        self.brush_slider = tk.Scale(
            brush_frame, from_=5, to=40, orient=tk.HORIZONTAL, 
            bg=self.colors['frame_bg'], fg=self.colors['text_main'], 
            troughcolor=self.colors['bg'], activebackground=self.colors['accent'],
            highlightthickness=0, command=self.update_brush_size
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=10, fill='x', expand=True)

        # 绑定事件 (同时绑定移动和按下)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_paint)

        # === 右侧：结果与控制区 ===
        right_frame = tk.Frame(self.root, bg=self.colors['frame_bg'], padx=30, pady=30, relief=tk.FLAT)
        right_frame.place(x=500, y=20, width=380, height=560)

        # 标题
        tk.Label(right_frame, text="识别结果", font=title_font, fg=self.colors['text_main'], bg=self.colors['frame_bg']).pack(pady=(0, 30))

        # 结果显示卡片
        result_card = tk.Frame(right_frame, bg=self.colors['bg'], pady=30)
        result_card.pack(fill='x', padx=10)

        self.res_label = tk.Label(
            result_card, 
            text="?", 
            font=("Helvetica", 80, "bold"), 
            fg=self.colors['accent'], 
            bg=self.colors['bg']
        )
        self.res_label.pack()

        self.prob_label = tk.Label(
            result_card, 
            text="置信度: --%", 
            font=("Helvetica", 14), 
            fg=self.colors['text_sub'], 
            bg=self.colors['bg']
        )
        self.prob_label.pack(pady=(10, 0))

        # 按钮区域
        btn_container = tk.Frame(right_frame, bg=self.colors['frame_bg'])
        btn_container.pack(pady=40, fill='both', expand=True)

        # 识别按钮
        self.btn_recognize = tk.Button(
            btn_container,
            text="🚀 开始识别",
            font=btn_font,
            fg="#ffffff",
            bg=self.colors['accent'],
            activebackground=self.colors['accent_hover'],
            activeforeground="#ffffff",
            bd=0,
            padx=20,
            pady=15,
            cursor="hand2",
            command=self.recognize
        )
        self.btn_recognize.pack(pady=10, fill='x')
        self.btn_recognize.bind("<Enter>", lambda e: self.btn_recognize.config(bg=self.colors['accent_hover']))
        self.btn_recognize.bind("<Leave>", lambda e: self.btn_recognize.config(bg=self.colors['accent']))

        # 清除按钮
        self.btn_clear = tk.Button(
            btn_container,
            text="🗑️ 清空画布",
            font=btn_font,
            fg="#ffffff",
            bg=self.colors['danger'],
            activebackground=self.colors['danger_hover'],
            activeforeground="#ffffff",
            bd=0,
            padx=20,
            pady=15,
            cursor="hand2",
            command=self.clear_canvas
        )
        self.btn_clear.pack(pady=10, fill='x')
        self.btn_clear.bind("<Enter>", lambda e: self.btn_clear.config(bg=self.colors['danger_hover']))
        self.btn_clear.bind("<Leave>", lambda e: self.btn_clear.config(bg=self.colors['danger']))

        # 底部提示
        tk.Label(right_frame, text="基于 MNIST 数据集训练 \n CNN 模型", font=("Helvetica", 9), fg=self.colors['text_sub'], bg=self.colors['frame_bg']).pack(side=tk.BOTTOM, pady=10)

    def update_brush_size(self, val):
        self.brush_size = int(val)

    def start_paint(self, event):
        self.last_x = event.x
        self.last_y = event.y
        # 点击时也画一个点
        self.paint(event)

    def paint(self, event):
        if self.last_x and self.last_y:
            # Tkinter 坐标转 PIL 坐标 (因为显示大小和实际大小可能不同)
            # 这里假设 canvas_size 和 image_size 比例一致，或者直接映射
            # 为了简单，我们让 canvas 和 image 逻辑尺寸一致，显示时缩放
            # 但本例中 canvas_size (400) != image_size (280)，需要转换坐标
            
            scale = self.image_size / self.canvas_size
            
            # 计算当前点和上一点的 PIL 坐标
            curr_x = int(event.x * scale)
            curr_y = int(event.y * scale)
            last_x_pil = int(self.last_x * scale)
            last_y_pil = int(self.last_y * scale)

            # 在 Tkinter 画布上画线 (视觉)
            # 使用 create_line 比 create_oval 连续拖动更平滑
            r = self.brush_size / 2
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=self.colors['brush'],
                width=self.brush_size,
                capstyle=tk.ROUND,
                smooth=True
            )

            # 在 PIL 图像上画线 (数据)
            self.draw.line(
                [last_x_pil, last_y_pil, curr_x, curr_y],
                fill=255,
                width=int(self.brush_size * scale), # 笔刷宽度也要按比例缩放
                joint="curve"
            )

        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        
        # 重置 UI 文本
        self.res_label.config(text="?")
        self.prob_label.config(text="置信度: --%")
        self.res_label.config(fg=self.colors['accent'])

    def recognize(self):
        # --- 图像预处理 ---
        img_np = np.array(self.image)
        
        # 检查是否全黑（没写字）
        if np.sum(img_np) == 0:
            messagebox.showwarning("提示", "请先在画布上书写数字！")
            return

        # 1. 形态学操作：轻微膨胀以连接断笔 (可选，根据笔刷大小调整)
        # 注意：如果在画布上已经画得很粗，这里可以省略或减小 kernel
        kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img_np, kernel, iterations=1)
        
        # 2. 高斯模糊
        img_blurred = cv2.GaussianBlur(img_dilated, (5, 5), 0)
        
        # 3. 缩放到 28x28 (MNIST 标准输入)
        img_resized = cv2.resize(img_blurred, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 4. 转为 Tensor 并归一化 (均值 0.1307, 标准差 0.3081 是 MNIST 统计值)
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = (img_tensor - 0.1307) / 0.3081
        img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).unsqueeze(0).to(self.device)

        # --- 模型推理 ---
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(prob, dim=1).item()
            confidence = torch.max(prob).item()

        # 更新 UI
        self.res_label.config(text=str(prediction))
        self.prob_label.config(text=f"置信度: {confidence:.2%}")
        
        # 根据置信度改变颜色
        if confidence > 0.9:
            self.res_label.config(fg=self.colors['accent'])
        elif confidence > 0.6:
            self.res_label.config(fg="#FFC107") # 黄色警告
        else:
            self.res_label.config(fg=self.colors['danger'])

if __name__ == "__main__":
    root = tk.Tk()
    # 设置 DPI 感知 (解决高分屏模糊问题)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
        
    app = DigitRecognizerGUI(root)
    root.mainloop()