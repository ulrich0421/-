import cv2
import torch
from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator, colors
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import time
from threading import Thread
import onnxruntime as ort
from torchvision import transforms
import numpy as np
import os
import logging

# ============================ 配置日志 ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================ YOLO 配置 ============================
# 检查是否有可用的 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用 YOLO Nano 模型并将其加载到 GPU（如果可用）
YOLO_MODEL_PATH = 'yolo11n.pt'
try:
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
    logging.info("YOLO模型加载成功。")
except Exception as e:
    logging.error(f"YOLO模型加载失败: {e}")
    yolo_model = None

# 定义车辆和行人类别 ID （根据 COCO 数据集的类别）
vehicle_classes = {2, 5, 7}  # 2:car, 5:bus, 7:truck
person_class = 0  # 0:person

# YOLO 要求输入尺寸是 32 的倍数，这里定义目标输入尺寸
yolo_input_size = 640  # 常见输入尺寸可以是 640x640

# ============================ 交通标志分类配置 ============================
# 定义类别名称
CLASS_NAMES = {
    0: "限速5km", 1: "限速15km", 2: "限速30km", 3: "限速40km", 5: "限速60km",
    6: "限速70km", 7: "限速80km", 8: "禁止左转和直行", 9: "禁止直行和右转",
    10: "禁止直行", 11: "禁止左转", 12: "禁止左右转弯", 14: "禁止超车",
    15: "禁止掉头", 16: "禁止机动车驶入", 17: "禁止鸣笛", 18: "解除40km限制",
    19: "解除50km限制", 20: "直行和右转", 21: "单直行", 22: "向左转弯",
    23: "向左向右转弯", 24: "向右转弯", 25: "靠左侧通道行驶", 26: "靠右侧道路行驶",
    27: "环岛行驶", 28: "机动车行驶", 29: "鸣喇叭", 30: "非机动车行驶",
    31: "允许掉头", 32: "左右绕行", 33: "注意红绿灯", 34: "注意危险",
    35: "注意行人", 36: "注意非机动车", 37: "注意儿童", 38: "向右急转弯",
    39: "向左急转弯", 40: "下陡坡", 41: "上陡坡", 42: "慢行", 43: "T形交叉",
    44: "T形交叉", 45: "村庄", 46: "反向弯路", 47: "无人看守铁路道口",
    48: "施工", 49: "连续弯路", 50: "有人看守铁路道口", 51: "事故易发生路段",
    52: "停车让行", 53: "禁止通行", 54: "禁止车辆临时或长时间停放", 55: "禁止输入",
    56: "减速让行", 57: "停车检查"
}

# 模型配置
ONNX_MODEL_PATH = './save_model/model.onnx'
CLASSIFY_CONFIDENCE_THRESHOLD = 0.75

def softmax(x):
    """计算Softmax函数。"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def preprocess_image(image):
    """对输入图像进行预处理，调整大小、裁剪、转换为张量并归一化。"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).numpy()

class TrafficSignClassifier:
    """交通标志分类器，负责加载模型和执行推理。"""
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            logging.error(f"模型文件未找到: {model_path}")
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        self.session = ort.InferenceSession(model_path)
        logging.info("交通标志分类模型加载成功。")

    def classify(self, image):
        """对输入图像进行分类，返回分类结果或置信度过低的提示。"""
        try:
            image_np = preprocess_image(image)
            inputs = {self.session.get_inputs()[0].name: image_np}
            outputs = self.session.run(None, inputs)
            outputs_softmax = softmax(outputs[0])
            probabilities = np.max(outputs_softmax, axis=1)
            predicted_idx = np.argmax(outputs[0], axis=1)

            if probabilities[0] < CLASSIFY_CONFIDENCE_THRESHOLD:
                return "置信度过低，无法分类"
            else:
                return CLASS_NAMES.get(predicted_idx[0], "类别未知")
        except Exception as e:
            logging.error(f"分类过程中出现错误: {e}")
            return "分类失败"

# ============================ 停车管理配置 ============================
# 定义停车管理模型路径和配置
PARKING_MODEL_PATH = 'yolo11n.pt'  # 停车管理使用与车辆检测相同的YOLO模型
PARKING_JSON_FILE = 'bounding_boxes.json'  # 停车注释文件路径

# ============================ 主应用程序 ============================
class IntegratedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能交通系统")
        self.root.geometry("1500x900")

        # 初始化Notebook（标签页）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 初始化六个标签页
        self.init_detection_tab()
        self.init_classification_tab()
        self.init_parking_management_tab()
        self.init_speed_estimation_tab()  # 新增速度估计标签页
        self.init_instance_segmentation_tab()# 初始化实例分割与追踪标签页
        self.init_lane_detection_tab()# 初始化车道检测标签页

        # 加载交通标志分类模型
        try:
            self.classifier = TrafficSignClassifier(ONNX_MODEL_PATH)
        except FileNotFoundError as e:
            messagebox.showerror("错误", str(e))
            self.classifier = None
        except Exception as e:
            logging.error(f"交通标志分类模型加载失败: {e}")
            self.classifier = None

        # 初始化停车管理模型
        try:
            self.parking_manager = solutions.ParkingManagement(
                model=PARKING_MODEL_PATH,
                json_file=PARKING_JSON_FILE
            )
            logging.info("停车管理模型加载成功。")
        except Exception as e:
            logging.error(f"停车管理模型加载失败: {e}")
            self.parking_manager = None

        # 初始化速度估计模型
        try:
            self.speed_estimator = solutions.SpeedEstimator(
                reg_pts=[(0, 360), (1280, 360)],  # 根据实际视频调整
                names=yolo_model.model.names if yolo_model else {},
                view_img=True,
            )
            logging.info("速度估计模型初始化成功。")
        except Exception as e:
            logging.error(f"速度估计模型初始化失败: {e}")
            self.speed_estimator = None

    # ==================== 车辆与行人检测标签页 ====================
    def init_detection_tab(self):
        self.detection_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.detection_frame, text="车辆与行人检测")

        # 视频输入选择
        video_select_frame = ttk.Frame(self.detection_frame)
        video_select_frame.pack(fill=tk.X, pady=5)

        self.video_path = tk.StringVar()
        ttk.Label(video_select_frame, text="视频路径:").pack(side=tk.LEFT, padx=5)
        self.video_entry = ttk.Entry(video_select_frame, textvariable=self.video_path, width=50)
        self.video_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(video_select_frame, text="选择视频", command=self.select_video).pack(side=tk.LEFT, padx=5)

        # 视频显示Canvas
        self.video_canvas = tk.Canvas(self.detection_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        # 控制面板
        control_frame = ttk.Frame(self.detection_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # 车辆检测复选框
        self.vehicle_detection_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="启用车辆检测", variable=self.vehicle_detection_enabled).pack(side=tk.LEFT, padx=5)

        # 行人检测复选框
        self.pedestrian_detection_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="启用行人检测", variable=self.pedestrian_detection_enabled).pack(side=tk.LEFT, padx=5)

        # 置信度阈值滑块
        # ttk.Label(control_frame, text="置信度阈值:").pack(side=tk.LEFT, padx=5)
        self.confidence_threshold = tk.IntVar(value=0)
        # self.confidence_scale = ttk.Scale(control_frame, from_=0, to=100, orient='horizontal', variable=self.confidence_threshold)
        # self.confidence_scale.pack(side=tk.LEFT, padx=5)

        # 启动/暂停检测按钮
        self.start_button = ttk.Button(control_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # 类别计数显示标签
        counts_frame = ttk.Frame(self.detection_frame)
        counts_frame.pack(fill=tk.X, pady=5)

        self.vehicle_count_label = ttk.Label(counts_frame, text="车辆数量: 0")
        self.vehicle_count_label.pack(side=tk.LEFT, padx=10)

        self.person_count_label = ttk.Label(counts_frame, text="行人数量: 0")
        self.person_count_label.pack(side=tk.LEFT, padx=10)

        # 帧率显示标签
        self.frame_rate_label = ttk.Label(counts_frame, text="帧率: 0.00 FPS")
        self.frame_rate_label.pack(side=tk.LEFT, padx=10)

        # 绑定窗口调整事件
        self.detection_frame.bind("<Configure>", self.resize_video_canvas)

        # 初始化视频处理相关变量
        self.cap = None
        self.orig_width = 640
        self.orig_height = 480
        self.is_playing = False
        self.is_paused = False
        self.frame_count = 0
        self.start_time_detection = time.time()
        self.frame_skip = 1

    def select_video(self):
        """选择视频文件。"""
        filepath = filedialog.askopenfilename(title="选择视频文件", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filepath:
            self.video_path.set(filepath)
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(filepath)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件。")
                return
            self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"已选择视频: {filepath}，分辨率: {self.orig_width}x{self.orig_height}")

    def resize_video_canvas(self, event):
        """根据窗口大小调整 Canvas 尺寸并保持视频比例。"""
        if self.cap:
            new_width = self.video_canvas.winfo_width()
            new_height = self.video_canvas.winfo_height()
            aspect_ratio = self.orig_width / self.orig_height
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)
            self.video_canvas.config(width=new_width, height=new_height)

    def start_detection(self):
        """启动或暂停检测。"""
        if not self.cap:
            messagebox.showwarning("警告", "请先选择一个视频文件！")
            return

        if not self.is_playing:
            self.is_playing = True
            self.is_paused = False
            self.start_button.config(text="暂停检测", command=self.pause_detection)
            Thread(target=self.video_processing, daemon=True).start()
            logging.info("开始视频检测。")
        else:
            if not self.is_paused:
                self.is_paused = True
                self.start_button.config(text="继续检测", command=self.resume_detection)
                logging.info("暂停视频检测。")
            else:
                self.is_paused = False
                self.start_button.config(text="暂停检测", command=self.pause_detection)
                logging.info("继续视频检测。")

    def pause_detection(self):
        """暂停检测。"""
        self.is_paused = True
        self.start_button.config(text="继续检测", command=self.resume_detection)
        logging.info("暂停视频检测。")

    def resume_detection(self):
        """继续检测。"""
        self.is_paused = False
        self.start_button.config(text="暂停检测", command=self.pause_detection)
        logging.info("继续视频检测。")

    def video_processing(self):
        """视频处理线程。"""
        while self.is_playing and self.cap.isOpened():
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("视频播放完毕。")
                    break

                self.frame_count += 1

                if self.frame_count % self.frame_skip == 0:
                    if yolo_model and (self.vehicle_detection_enabled.get() or self.pedestrian_detection_enabled.get()):
                        self.process_frame(frame)
                        self.update_class_counts()

                self.update_fps()

                # 调整帧大小以适应Canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                aspect_ratio = self.orig_width / self.orig_height
                if canvas_width / canvas_height > aspect_ratio:
                    canvas_width = int(canvas_height * aspect_ratio)
                else:
                    canvas_height = int(canvas_width / aspect_ratio)

                frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # 更新Canvas中的图像
                self.video_canvas.create_image(0, 0, anchor='nw', image=img_tk)
                self.video_canvas.image = img_tk

            time.sleep(0.01)

        self.is_playing = False
        self.start_button.config(text="开始检测", command=self.start_detection)
        self.cap.release()
        logging.info("视频检测线程已结束。")

    def process_frame(self, frame):
        """处理视频帧并返回检测结果。"""
        global yolo_model
        vehicle_count = 0
        person_count = 0

        # 调整输入帧的大小，并确保大小为 32 的倍数
        frame_resized = cv2.resize(frame, (yolo_input_size, yolo_input_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            results = yolo_model(frame_tensor)

        scale_x = self.orig_width / yolo_input_size
        scale_y = self.orig_height / yolo_input_size

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                if conf >= self.confidence_threshold.get() / 100:
                    if cls in vehicle_classes and self.vehicle_detection_enabled.get():
                        vehicle_count += 1
                    elif cls == person_class and self.pedestrian_detection_enabled.get():
                        person_count += 1

                    if (cls in vehicle_classes and self.vehicle_detection_enabled.get()) or (cls == person_class and self.pedestrian_detection_enabled.get()):
                        x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

                        color = (0, 255, 0) if cls in vehicle_classes else (255, 0, 0)
                        label = f"{yolo_model.names[cls]}: {conf:.2f}"

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.current_vehicle_count = vehicle_count
        self.current_person_count = person_count

    def update_class_counts(self):
        """更新GUI中的类别计数。"""
        self.vehicle_count_label.config(text=f"车辆数量: {self.current_vehicle_count}")
        self.person_count_label.config(text=f"行人数量: {self.current_person_count}")

    def update_fps(self):
        """更新GUI中的帧率。"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time_detection
        if elapsed_time > 0:
            frame_rate = self.frame_count / elapsed_time
            self.frame_rate_label.config(text=f"帧率: {frame_rate:.2f} FPS")
            self.start_time_detection = current_time
            self.frame_count = 0

    # ==================== 交通标志分类标签页 ====================
    def init_classification_tab(self):
        self.classification_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.classification_frame, text="交通标志分类")

        # 选择图片按钮
        btn_frame = ttk.Frame(self.classification_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_select = ttk.Button(btn_frame, text="选择图像", command=self.select_image)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_classify = ttk.Button(btn_frame, text="分类", command=self.classify_image, state=tk.DISABLED)
        self.btn_classify.pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress = ttk.Progressbar(btn_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 显示图片的标签
        self.lbl_image = ttk.Label(self.classification_frame, text="请选择一张图像", anchor="center")
        self.lbl_image.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # 显示结果的标签
        self.lbl_result = ttk.Label(self.classification_frame, text="", font=("Arial", 14), foreground="blue", anchor="center")
        self.lbl_result.pack(fill=tk.X, padx=5, pady=10)

        # 初始化分类相关变量
        self.image_path = None
        self.image = None

    def select_image(self):
        """选择图像文件。"""
        if not self.classifier:
            messagebox.showerror("错误", "交通标志分类模型未加载。")
            return

        filetypes = (
            ("图像文件", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("所有文件", "*.*")
        )
        filepath = filedialog.askopenfilename(title="选择图像", initialdir="./", filetypes=filetypes)
        if filepath:
            self.image_path = filepath
            try:
                pil_image = Image.open(filepath).convert("RGB")
                pil_image = pil_image.resize((400, 400), Image.Resampling.LANCZOS)
                self.image = ImageTk.PhotoImage(pil_image)
                self.lbl_image.config(image=self.image, text="")
                self.lbl_image.image = self.image  # 保持引用
                self.lbl_result.config(text="")
                self.btn_classify.config(state=tk.NORMAL)
                logging.info(f"选择图像: {filepath}")
            except AttributeError:
                # 兼容旧版本Pillow
                pil_image = pil_image.resize((400, 400), Image.LANCZOS)
                self.image = ImageTk.PhotoImage(pil_image)
                self.lbl_image.config(image=self.image, text="")
                self.lbl_image.image = self.image  # 保持引用
                self.lbl_result.config(text="")
                self.btn_classify.config(state=tk.NORMAL)
                logging.info(f"选择图像: {filepath}")
            except Exception as e:
                logging.error(f"加载图像失败: {e}")
                messagebox.showerror("错误", f"加载图像失败: {e}")
                self.image_path = None
                self.lbl_image.config(image='', text="请选择一张图像")
                self.btn_classify.config(state=tk.DISABLED)

    def classify_image(self):
        """启动分类线程，防止界面冻结。"""
        if not self.image_path:
            messagebox.showwarning("警告", "请先选择一张图像！")
            return

        # 禁用按钮，启动进度条
        self.btn_classify.config(state=tk.DISABLED)
        self.progress.start(10)
        self.lbl_result.config(text="正在分类...")

        # 启动线程进行分类
        Thread(target=self.run_classification, daemon=True).start()

    def run_classification(self):
        """执行图像分类并更新界面。"""
        try:
            pil_image = Image.open(self.image_path).convert("RGB")
            result = self.classifier.classify(pil_image)
            logging.info(f"分类结果: {result}")
        except Exception as e:
            logging.error(f"分类过程中出现错误: {e}")
            result = "分类失败"

        # 更新界面需要在主线程中执行
        self.root.after(0, self.update_result, result)

    def update_result(self, result):
        """更新分类结果显示，并恢复按钮和停止进度条。"""
        self.lbl_result.config(text=f"分类结果: {result}")
        self.progress.stop()
        self.btn_classify.config(state=tk.NORMAL)

    # ==================== 停车管理标签页 ====================
    def init_parking_management_tab(self):
        self.parking_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.parking_frame, text="停车管理")

        # 视频输入选择
        parking_video_select_frame = ttk.Frame(self.parking_frame)
        parking_video_select_frame.pack(fill=tk.X, pady=5)

        self.parking_video_path = tk.StringVar()
        ttk.Label(parking_video_select_frame, text="视频路径:").pack(side=tk.LEFT, padx=5)
        self.parking_video_entry = ttk.Entry(parking_video_select_frame, textvariable=self.parking_video_path, width=50)
        self.parking_video_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(parking_video_select_frame, text="选择视频", command=self.select_parking_video).pack(side=tk.LEFT, padx=5)

        # 停车视频显示Canvas
        self.parking_video_canvas = tk.Canvas(self.parking_frame, bg="black")
        self.parking_video_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        # 控制面板
        parking_control_frame = ttk.Frame(self.parking_frame)
        parking_control_frame.pack(fill=tk.X, pady=5)

        # 启动/暂停停车管理按钮
        self.parking_start_button = ttk.Button(parking_control_frame, text="开始停车管理", command=self.start_parking_management)
        self.parking_start_button.pack(side=tk.LEFT, padx=5)

        # 停车管理状态显示
        self.parking_status_label = ttk.Label(parking_control_frame, text="状态: 停止", foreground="red")
        self.parking_status_label.pack(side=tk.LEFT, padx=10)

        # 绑定窗口调整事件
        self.parking_frame.bind("<Configure>", self.resize_parking_video_canvas)

        # 初始化停车管理处理相关变量
        self.parking_cap = None
        self.parking_orig_width = 640
        self.parking_orig_height = 480
        self.parking_is_playing = False
        self.parking_is_paused = False
        self.parking_frame_count = 0
        self.parking_start_time = time.time()

    def select_parking_video(self):
        """选择停车管理视频文件。"""
        filepath = filedialog.askopenfilename(title="选择停车管理视频文件", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filepath:
            self.parking_video_path.set(filepath)
            if self.parking_cap:
                self.parking_cap.release()
            if not self.parking_manager:
                messagebox.showerror("错误", "停车管理模型未加载。")
                return
            self.parking_cap = cv2.VideoCapture(filepath)
            if not self.parking_cap.isOpened():
                messagebox.showerror("错误", "无法打开停车管理视频文件。")
                return
            self.parking_orig_width = int(self.parking_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.parking_orig_height = int(self.parking_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"已选择停车管理视频: {filepath}，分辨率: {self.parking_orig_width}x{self.parking_orig_height}")

    def resize_parking_video_canvas(self, event):
        """根据窗口大小调整 Parking Canvas 尺寸并保持视频比例。"""
        if self.parking_cap:
            new_width = self.parking_video_canvas.winfo_width()
            new_height = self.parking_video_canvas.winfo_height()
            aspect_ratio = self.parking_orig_width / self.parking_orig_height
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)
            self.parking_video_canvas.config(width=new_width, height=new_height)

    def start_parking_management(self):
        """启动或暂停停车管理。"""
        if not self.parking_cap:
            messagebox.showwarning("警告", "请先选择一个停车管理视频文件！")
            return

        if not self.parking_is_playing:
            self.parking_is_playing = True
            self.parking_is_paused = False
            self.parking_start_button.config(text="暂停停车管理", command=self.pause_parking_management)
            self.parking_status_label.config(text="状态: 运行中", foreground="green")
            Thread(target=self.parking_video_processing, daemon=True).start()
            logging.info("开始停车管理。")
        else:
            if not self.parking_is_paused:
                self.parking_is_paused = True
                self.parking_start_button.config(text="继续停车管理", command=self.resume_parking_management)
                self.parking_status_label.config(text="状态: 暂停", foreground="orange")
                logging.info("暂停停车管理。")
            else:
                self.parking_is_paused = False
                self.parking_start_button.config(text="暂停停车管理", command=self.pause_parking_management)
                self.parking_status_label.config(text="状态: 运行中", foreground="green")
                logging.info("继续停车管理。")

    def pause_parking_management(self):
        """暂停停车管理。"""
        self.parking_is_paused = True
        self.parking_start_button.config(text="继续停车管理", command=self.resume_parking_management)
        self.parking_status_label.config(text="状态: 暂停", foreground="orange")
        logging.info("暂停停车管理。")

    def resume_parking_management(self):
        """继续停车管理。"""
        self.parking_is_paused = False
        self.parking_start_button.config(text="暂停停车管理", command=self.pause_parking_management)
        self.parking_status_label.config(text="状态: 运行中", foreground="green")
        logging.info("继续停车管理。")

    def parking_video_processing(self):
        """停车管理视频处理线程。"""
        while self.parking_is_playing and self.parking_cap.isOpened():
            if not self.parking_is_paused:
                ret, frame = self.parking_cap.read()
                if not ret:
                    logging.info("停车管理视频播放完毕。")
                    break

                self.parking_frame_count += 1

                # 处理帧
                if self.parking_manager:
                    processed_frame = self.parking_manager.process_data(frame)
                else:
                    processed_frame = frame

                # 调整帧大小以适应Canvas
                canvas_width = self.parking_video_canvas.winfo_width()
                canvas_height = self.parking_video_canvas.winfo_height()
                aspect_ratio = self.parking_orig_width / self.parking_orig_height
                if canvas_width / canvas_height > aspect_ratio:
                    canvas_width = int(canvas_height * aspect_ratio)
                else:
                    canvas_height = int(canvas_width / aspect_ratio)

                frame_resized = cv2.resize(processed_frame, (canvas_width, canvas_height))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # 更新Canvas中的图像
                self.parking_video_canvas.create_image(0, 0, anchor='nw', image=img_tk)
                self.parking_video_canvas.image = img_tk

            time.sleep(0.01)

        self.parking_is_playing = False
        self.parking_start_button.config(text="开始停车管理", command=self.start_parking_management)
        self.parking_cap.release()
        logging.info("停车管理视频处理线程已结束。")

    # ==================== 速度估计标签页 ====================
    def init_speed_estimation_tab(self):
        self.speed_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.speed_frame, text="速度估计")

        # 视频输入选择
        speed_video_select_frame = ttk.Frame(self.speed_frame)
        speed_video_select_frame.pack(fill=tk.X, pady=5)

        self.speed_video_path = tk.StringVar()
        ttk.Label(speed_video_select_frame, text="视频路径:").pack(side=tk.LEFT, padx=5)
        self.speed_video_entry = ttk.Entry(speed_video_select_frame, textvariable=self.speed_video_path, width=50)
        self.speed_video_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(speed_video_select_frame, text="选择视频", command=self.select_speed_video).pack(side=tk.LEFT, padx=5)

        # 速度估计视频显示Canvas
        self.speed_video_canvas = tk.Canvas(self.speed_frame, bg="black")
        self.speed_video_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        # 控制面板
        speed_control_frame = ttk.Frame(self.speed_frame)
        speed_control_frame.pack(fill=tk.X, pady=5)

        # 启动/暂停速度估计按钮
        self.speed_start_button = ttk.Button(speed_control_frame, text="开始速度估计", command=self.start_speed_estimation)
        self.speed_start_button.pack(side=tk.LEFT, padx=5)

        # 速度估计状态显示
        self.speed_status_label = ttk.Label(speed_control_frame, text="状态: 停止", foreground="red")
        self.speed_status_label.pack(side=tk.LEFT, padx=10)

        # 置信度阈值滑块
        # ttk.Label(speed_control_frame, text="置信度阈值:").pack(side=tk.LEFT, padx=5)
        self.speed_confidence_threshold = tk.IntVar(value=0)
        # self.speed_confidence_scale = ttk.Scale(speed_control_frame, from_=0, to=100, orient='horizontal', variable=self.speed_confidence_threshold)
        # self.speed_confidence_scale.pack(side=tk.LEFT, padx=5)

        # 绑定窗口调整事件
        self.speed_frame.bind("<Configure>", self.resize_speed_video_canvas)

        # 初始化速度估计处理相关变量
        self.speed_cap = None
        self.speed_orig_width = 640
        self.speed_orig_height = 480
        self.speed_is_playing = False
        self.speed_is_paused = False
        self.speed_frame_count = 0
        self.speed_start_time = time.time()

    def select_speed_video(self):
        """选择速度估计视频文件。"""
        filepath = filedialog.askopenfilename(title="选择速度估计视频文件", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filepath:
            self.speed_video_path.set(filepath)
            if self.speed_cap:
                self.speed_cap.release()
            if not self.speed_estimator:
                messagebox.showerror("错误", "速度估计模型未初始化。")
                return
            self.speed_cap = cv2.VideoCapture(filepath)
            if not self.speed_cap.isOpened():
                messagebox.showerror("错误", "无法打开速度估计视频文件。")
                return
            self.speed_orig_width = int(self.speed_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.speed_orig_height = int(self.speed_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"已选择速度估计视频: {filepath}，分辨率: {self.speed_orig_width}x{self.speed_orig_height}")

    def resize_speed_video_canvas(self, event):
        """根据窗口大小调整速度估计 Canvas 尺寸并保持视频比例。"""
        if self.speed_cap:
            new_width = self.speed_video_canvas.winfo_width()
            new_height = self.speed_video_canvas.winfo_height()
            aspect_ratio = self.speed_orig_width / self.speed_orig_height
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)
            self.speed_video_canvas.config(width=new_width, height=new_height)

    def start_speed_estimation(self):
        """启动或暂停速度估计。"""
        if not self.speed_cap:
            messagebox.showwarning("警告", "请先选择一个速度估计视频文件！")
            return

        if not self.speed_is_playing:
            self.speed_is_playing = True
            self.speed_is_paused = False
            self.speed_start_button.config(text="暂停速度估计", command=self.pause_speed_estimation)
            self.speed_status_label.config(text="状态: 运行中", foreground="green")
            Thread(target=self.speed_video_processing, daemon=True).start()
            logging.info("开始速度估计。")
        else:
            if not self.speed_is_paused:
                self.speed_is_paused = True
                self.speed_start_button.config(text="继续速度估计", command=self.resume_speed_estimation)
                self.speed_status_label.config(text="状态: 暂停", foreground="orange")
                logging.info("暂停速度估计。")
            else:
                self.speed_is_paused = False
                self.speed_start_button.config(text="暂停速度估计", command=self.pause_speed_estimation)
                self.speed_status_label.config(text="状态: 运行中", foreground="green")
                logging.info("继续速度估计。")

    def pause_speed_estimation(self):
        """暂停速度估计。"""
        self.speed_is_paused = True
        self.speed_start_button.config(text="继续速度估计", command=self.resume_speed_estimation)
        self.speed_status_label.config(text="状态: 暂停", foreground="orange")
        logging.info("暂停速度估计。")

    def resume_speed_estimation(self):
        """继续速度估计。"""
        self.speed_is_paused = False
        self.speed_start_button.config(text="暂停速度估计", command=self.pause_speed_estimation)
        self.speed_status_label.config(text="状态: 运行中", foreground="green")
        logging.info("继续速度估计。")

    def speed_video_processing(self):
        """速度估计视频处理线程。"""
        while self.speed_is_playing and self.speed_cap.isOpened():
            if not self.speed_is_paused:
                ret, frame = self.speed_cap.read()
                if not ret:
                    logging.info("速度估计视频播放完毕。")
                    break

                self.speed_frame_count += 1

                # 处理帧
                if self.speed_estimator:
                    tracks = yolo_model.track(frame, persist=True) if yolo_model else []
                    frame = self.speed_estimator.estimate_speed(frame, tracks)
                else:
                    frame = frame

                # 调整帧大小以适应Canvas
                canvas_width = self.speed_video_canvas.winfo_width()
                canvas_height = self.speed_video_canvas.winfo_height()
                aspect_ratio = self.speed_orig_width / self.speed_orig_height
                if canvas_width / canvas_height > aspect_ratio:
                    canvas_width = int(canvas_height * aspect_ratio)
                else:
                    canvas_height = int(canvas_width / aspect_ratio)

                frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # 更新Canvas中的图像
                self.speed_video_canvas.create_image(0, 0, anchor='nw', image=img_tk)
                self.speed_video_canvas.image = img_tk

            time.sleep(0.01)

        self.speed_is_playing = False
        self.speed_start_button.config(text="开始速度估计", command=self.start_speed_estimation)
        self.speed_cap.release()
        logging.info("速度估计视频处理线程已结束。")
    # ==================== 实例分割标签页 ====================
    def init_instance_segmentation_tab(self):
        self.instance_segmentation_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.instance_segmentation_frame, text="实例分割与追踪")

        # 视频输入选择
        seg_video_select_frame = ttk.Frame(self.instance_segmentation_frame)
        seg_video_select_frame.pack(fill=tk.X, pady=5)

        self.seg_video_path = tk.StringVar()
        ttk.Label(seg_video_select_frame, text="视频路径:").pack(side=tk.LEFT, padx=5)
        self.seg_video_entry = ttk.Entry(seg_video_select_frame, textvariable=self.seg_video_path, width=50)
        self.seg_video_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(seg_video_select_frame, text="选择视频", command=self.select_seg_video).pack(side=tk.LEFT, padx=5)

        # 实例分割视频显示Canvas
        self.seg_video_canvas = tk.Canvas(self.instance_segmentation_frame, bg="black")
        self.seg_video_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        # 控制面板
        seg_control_frame = ttk.Frame(self.instance_segmentation_frame)
        seg_control_frame.pack(fill=tk.X, pady=5)

        # 启动/暂停实例分割追踪按钮
        self.seg_start_button = ttk.Button(seg_control_frame, text="开始分割与追踪", command=self.start_instance_segmentation)
        self.seg_start_button.pack(side=tk.LEFT, padx=5)

        # 状态显示
        self.seg_status_label = ttk.Label(seg_control_frame, text="状态: 停止", foreground="red")
        self.seg_status_label.pack(side=tk.LEFT, padx=10)

        # 绑定窗口调整事件
        self.instance_segmentation_frame.bind("<Configure>", self.resize_seg_video_canvas)

        # 初始化实例分割处理相关变量
        self.seg_cap = None
        self.seg_orig_width = 640
        self.seg_orig_height = 480
        self.seg_is_playing = False
        self.seg_is_paused = False
        self.seg_frame_count = 0
        self.seg_start_time = time.time()

        # 加载实例分割模型
        try:
            self.seg_model = YOLO("yolov8n-seg.pt")  # 确保模型路径正确
            logging.info("实例分割模型加载成功。")
        except Exception as e:
            logging.error(f"实例分割模型加载失败: {e}")
            self.seg_model = None
            messagebox.showerror("错误", f"实例分割模型加载失败: {e}")
    def select_seg_video(self):
        """选择实例分割视频文件。"""
        filepath = filedialog.askopenfilename(title="选择实例分割视频文件", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filepath:
            self.seg_video_path.set(filepath)
            if self.seg_cap:
                self.seg_cap.release()
            if not self.seg_model:
                messagebox.showerror("错误", "实例分割模型未加载。")
                return
            self.seg_cap = cv2.VideoCapture(filepath)
            if not self.seg_cap.isOpened():
                messagebox.showerror("错误", "无法打开实例分割视频文件。")
                return
            self.seg_orig_width = int(self.seg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.seg_orig_height = int(self.seg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"已选择实例分割视频: {filepath}，分辨率: {self.seg_orig_width}x{self.seg_orig_height}")
    def resize_seg_video_canvas(self, event):
        """根据窗口大小调整实例分割 Canvas 尺寸并保持视频比例。"""
        if self.seg_cap:
            new_width = self.seg_video_canvas.winfo_width()
            new_height = self.seg_video_canvas.winfo_height()
            aspect_ratio = self.seg_orig_width / self.seg_orig_height
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)
            self.seg_video_canvas.config(width=new_width, height=new_height)
    def start_instance_segmentation(self):
        """启动或暂停实例分割与追踪。"""
        if not self.seg_cap:
            messagebox.showwarning("警告", "请先选择一个实例分割视频文件！")
            return

        if not self.seg_is_playing:
            self.seg_is_playing = True
            self.seg_is_paused = False
            self.seg_start_button.config(text="暂停分割与追踪", command=self.pause_instance_segmentation)
            self.seg_status_label.config(text="状态: 运行中", foreground="green")
            Thread(target=self.seg_video_processing, daemon=True).start()
            logging.info("开始实例分割与追踪。")
        else:
            if not self.seg_is_paused:
                self.seg_is_paused = True
                self.seg_start_button.config(text="继续分割与追踪", command=self.resume_instance_segmentation)
                self.seg_status_label.config(text="状态: 暂停", foreground="orange")
                logging.info("暂停实例分割与追踪。")
            else:
                self.seg_is_paused = False
                self.seg_start_button.config(text="暂停分割与追踪", command=self.pause_instance_segmentation)
                self.seg_status_label.config(text="状态: 运行中", foreground="green")
                logging.info("继续实例分割与追踪。")

    def pause_instance_segmentation(self):
        """暂停实例分割与追踪。"""
        self.seg_is_paused = True
        self.seg_start_button.config(text="继续分割与追踪", command=self.resume_instance_segmentation)
        self.seg_status_label.config(text="状态: 暂停", foreground="orange")
        logging.info("暂停实例分割与追踪。")

    def resume_instance_segmentation(self):
        """继续实例分割与追踪。"""
        self.seg_is_paused = False
        self.seg_start_button.config(text="暂停分割与追踪", command=self.pause_instance_segmentation)
        self.seg_status_label.config(text="状态: 运行中", foreground="green")
        logging.info("继续实例分割与追踪。")
    def seg_video_processing(self):
        """实例分割与追踪视频处理线程。"""
        out = None  # 可选：如果需要保存输出视频

        while self.seg_is_playing and self.seg_cap.isOpened():
            if not self.seg_is_paused:
                ret, frame = self.seg_cap.read()
                if not ret:
                    logging.info("实例分割视频播放完毕。")
                    break

                self.seg_frame_count += 1

                if self.seg_model:
                    results = self.seg_model.track(frame, persist=True)
                    if results and results[0].boxes.id is not None and results[0].masks is not None:
                        masks = results[0].masks.xy
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        annotator = Annotator(frame, line_width=2)
                        for mask, track_id in zip(masks, track_ids):
                            color = colors(int(track_id), True)
                            txt_color = annotator.get_txt_color(color)
                            annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)
                        frame = annotator.result()

                # 可选：保存输出视频
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", fourcc, self.seg_cap.get(cv2.CAP_PROP_FPS), (self.seg_orig_width, self.seg_orig_height))

                if out:
                    out.write(frame)

                # 调整帧大小以适应Canvas
                canvas_width = self.seg_video_canvas.winfo_width()
                canvas_height = self.seg_video_canvas.winfo_height()
                aspect_ratio = self.seg_orig_width / self.seg_orig_height
                if canvas_width / canvas_height > aspect_ratio:
                    canvas_width = int(canvas_height * aspect_ratio)
                else:
                    canvas_height = int(canvas_width / aspect_ratio)

                frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # 更新Canvas中的图像
                self.seg_video_canvas.create_image(0, 0, anchor='nw', image=img_tk)
                self.seg_video_canvas.image = img_tk

                # 更新帧率显示（可选）
                elapsed_time = time.time() - self.seg_start_time
                if elapsed_time > 1:
                    frame_rate = self.seg_frame_count / elapsed_time
                    self.seg_status_label.config(text=f"状态: 运行中 ({frame_rate:.2f} FPS)", foreground="green")
                    self.seg_start_time = time.time()
                    self.seg_frame_count = 0

            time.sleep(0.01)

        self.seg_is_playing = False
        self.seg_start_button.config(text="开始分割与追踪", command=self.start_instance_segmentation)
        if out:
            out.release()
        self.seg_cap.release()
        logging.info("实例分割与追踪视频处理线程已结束。")

    # ==================== 车道识别标签页 ====================
    def init_lane_detection_tab(self):
        self.lane_detection_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.lane_detection_frame, text="车道检测")

        # 视频输入选择
        lane_video_select_frame = ttk.Frame(self.lane_detection_frame)
        lane_video_select_frame.pack(fill=tk.X, pady=5)

        self.lane_video_path = tk.StringVar()
        ttk.Label(lane_video_select_frame, text="视频路径:").pack(side=tk.LEFT, padx=5)
        self.lane_video_entry = ttk.Entry(lane_video_select_frame, textvariable=self.lane_video_path, width=50)
        self.lane_video_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(lane_video_select_frame, text="选择视频", command=self.select_lane_video).pack(side=tk.LEFT, padx=5)

        # 车道检测视频显示Canvas
        self.lane_video_canvas = tk.Canvas(self.lane_detection_frame, bg="black")
        self.lane_video_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        # 控制面板
        lane_control_frame = ttk.Frame(self.lane_detection_frame)
        lane_control_frame.pack(fill=tk.X, pady=5)

        # 启动/暂停车道检测按钮
        self.lane_start_button = ttk.Button(lane_control_frame, text="开始车道检测", command=self.start_lane_detection)
        self.lane_start_button.pack(side=tk.LEFT, padx=5)

        # 车道检测状态显示
        self.lane_status_label = ttk.Label(lane_control_frame, text="状态: 停止", foreground="red")
        self.lane_status_label.pack(side=tk.LEFT, padx=10)

        # 绑定窗口调整事件
        self.lane_detection_frame.bind("<Configure>", self.resize_lane_video_canvas)

        # 初始化车道检测处理相关变量
        self.lane_cap = None
        self.lane_orig_width = 640
        self.lane_orig_height = 480
        self.lane_is_playing = False
        self.lane_is_paused = False
        self.lane_frame_count = 0
        self.lane_start_time = time.time()

    # ==================== 车道检测相关方法 ====================
    def _grayscale(self, img):
        """将图像转换为灰度图像"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _gaussian_blur(self, img, kernel_size=5):
        """应用高斯模糊以减少噪声"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def _canny(self, img, low_threshold=50, high_threshold=150):
        """应用 Canny 边缘检测"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def _region_of_interest(self, img):
        """
        只保留图像的感兴趣区域（车道线通常位于图像下半部分的多边形区域）
        """
        height, width = img.shape
        mask = np.zeros_like(img)

        # 定义多边形的顶点
        polygon = np.array([
            [
                (int(0.1 * width), height),
                (int(0.45 * width), int(0.6 * height)),
                (int(0.55 * width), int(0.6 * height)),
                (int(0.9 * width), height)
            ]
        ], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def _hough_lines(self, img, rho=2, theta=np.pi/180, threshold=50, min_line_len=40, max_line_gap=100):
        """
        使用霍夫变换检测图像中的直线
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        return lines

    def _draw_lines(self, img, lines, color=[0, 255, 0], thickness=2):
        """
        绘制检测到的车道线，并进行左右车道线的平均
        同时绘制半透明的蓝色色块覆盖车道之间的路面
        """
        left_lines = []
        right_lines = []
        left_weights = []
        right_weights = []
        if lines is None:
            return

        for line in lines:
            for x1, y1, x2, y2 in line:
                # 计算每条线的斜率和截距
                if x2 - x1 == 0:  # 避免除以零
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if slope < -0.5:  # 左车道线
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                elif slope > 0.5:  # 右车道线
                    right_lines.append((slope, intercept))
                    right_weights.append(length)

        # 计算加权平均斜率和截距
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None

        y1 = img.shape[0]
        y2 = int(y1 * 0.6)

        def make_line(y1, y2, line):
            if line is None:
                return None
            slope, intercept = line
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return ((x1, y1), (x2, y2))

        left_line = make_line(y1, y2, left_lane)
        right_line = make_line(y1, y2, right_lane)

        # 存储车道线的顶点，用于绘制半透明的多边形
        polygon_points = []

        # 绘制车道线
        if left_line is not None:
            cv2.line(img, left_line[0], left_line[1], color, thickness)
            polygon_points.append(left_line[0])
            polygon_points.append(left_line[1])
        if right_line is not None:
            cv2.line(img, right_line[0], right_line[1], color, thickness)
            polygon_points.append(right_line[0])
            polygon_points.append(right_line[1])

        # 如果同时检测到左车道线和右车道线，绘制半透明的多边形
        if left_line is not None and right_line is not None:
            overlay = img.copy()
            # 定义多边形的四个顶点
            polygon = np.array([
                left_line[0],
                left_line[1],
                right_line[1],
                right_line[0]
            ], np.int32)
            # 绘制多边形
            cv2.fillPoly(overlay, [polygon], (255, 0, 0))  # 蓝色
            # 叠加半透明效果
            alpha = 0.3  # 透明度
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _process_frame(self, frame, fps):
        """处理每一帧图像，进行车道检测，并在帧上显示 FPS"""
        gray = self._grayscale(frame)
        blur = self._gaussian_blur(gray)
        edges = self._canny(blur)
        roi = self._region_of_interest(edges)
        lines = self._hough_lines(roi)
        self._draw_lines(frame, lines)

        # 显示 FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)

        return frame

    def select_lane_video(self):
        """选择车道检测视频文件。"""
        filepath = filedialog.askopenfilename(
            title="选择车道检测视频文件",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filepath:
            self.lane_video_path.set(filepath)
            if self.lane_cap:
                self.lane_cap.release()
            self.lane_cap = cv2.VideoCapture(filepath)
            if not self.lane_cap.isOpened():
                messagebox.showerror("错误", "无法打开车道检测视频文件。")
                return
            self.lane_orig_width = int(self.lane_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.lane_orig_height = int(self.lane_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"已选择车道检测视频: {filepath}，分辨率: {self.lane_orig_width}x{self.lane_orig_height}")

    def resize_lane_video_canvas(self, event):
        """根据窗口大小调整车道检测 Canvas 尺寸并保持视频比例。"""
        if self.lane_cap:
            new_width = self.lane_video_canvas.winfo_width()
            new_height = self.lane_video_canvas.winfo_height()
            aspect_ratio = self.lane_orig_width / self.lane_orig_height
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)
            self.lane_video_canvas.config(width=new_width, height=new_height)

    def start_lane_detection(self):
        """启动或暂停车道检测。"""
        if not self.lane_cap:
            messagebox.showwarning("警告", "请先选择一个车道检测视频文件！")
            return

        if not self.lane_is_playing:
            self.lane_is_playing = True
            self.lane_is_paused = False
            self.lane_start_button.config(text="暂停车道检测", command=self.pause_lane_detection)
            self.lane_status_label.config(text="状态: 运行中", foreground="green")
            Thread(target=self.lane_video_processing, daemon=True).start()
            logging.info("开始车道检测。")
        else:
            if not self.lane_is_paused:
                self.lane_is_paused = True
                self.lane_start_button.config(text="继续车道检测", command=self.resume_lane_detection)
                self.lane_status_label.config(text="状态: 暂停", foreground="orange")
                logging.info("暂停车道检测。")
            else:
                self.lane_is_paused = False
                self.lane_start_button.config(text="暂停车道检测", command=self.pause_lane_detection)
                self.lane_status_label.config(text="状态: 运行中", foreground="green")
                logging.info("继续车道检测。")

    def pause_lane_detection(self):
        """暂停车道检测。"""
        self.lane_is_paused = True
        self.lane_start_button.config(text="继续车道检测", command=self.resume_lane_detection)
        self.lane_status_label.config(text="状态: 暂停", foreground="orange")
        logging.info("暂停车道检测。")

    def resume_lane_detection(self):
        """继续车道检测。"""
        self.lane_is_paused = False
        self.lane_start_button.config(text="暂停车道检测", command=self.pause_lane_detection)
        self.lane_status_label.config(text="状态: 运行中", foreground="green")
        logging.info("继续车道检测。")

    def lane_video_processing(self):
        """车道检测视频处理线程。"""
        # 如果需要保存处理后的视频，初始化 VideoWriter
        output_path = "lane_detection_output.mp4"  # 可以根据需要修改路径
        out = None
        # 设置是否保存视频的条件，例如基于某个标志或参数
        save_video = False  # 设置为 True 以保存视频
        if save_video and output_path:
            width = int(self.lane_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.lane_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
            out = cv2.VideoWriter(output_path, fourcc, self.lane_cap.get(cv2.CAP_PROP_FPS), (width, height))

        while self.lane_is_playing and self.lane_cap.isOpened():
            if not self.lane_is_paused:
                ret, frame = self.lane_cap.read()
                if not ret:
                    logging.info("车道检测视频播放完毕。")
                    break

                self.lane_frame_count += 1

                # 处理帧
                processed_frame = self._process_frame(frame, self.lane_cap.get(cv2.CAP_PROP_FPS))

                # 调整帧大小以适应Canvas
                canvas_width = self.lane_video_canvas.winfo_width()
                canvas_height = self.lane_video_canvas.winfo_height()
                aspect_ratio = self.lane_orig_width / self.lane_orig_height
                if canvas_width / canvas_height > aspect_ratio:
                    canvas_width = int(canvas_height * aspect_ratio)
                else:
                    canvas_height = int(canvas_width / aspect_ratio)

                frame_resized = cv2.resize(processed_frame, (canvas_width, canvas_height))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # 更新Canvas中的图像
                self.lane_video_canvas.create_image(0, 0, anchor='nw', image=img_tk)
                self.lane_video_canvas.image = img_tk

                # 如果需要保存视频，写入帧
                if out:
                    out.write(processed_frame)

                # 更新帧率显示
                elapsed_time = time.time() - self.lane_start_time
                if elapsed_time > 1:
                    frame_rate = self.lane_frame_count / elapsed_time
                    self.lane_status_label.config(text=f"状态: 运行中 ({frame_rate:.2f} FPS)", foreground="green")
                    self.lane_start_time = time.time()
                    self.lane_frame_count = 0

            time.sleep(0.01)

        self.lane_is_playing = False
        self.lane_start_button.config(text="开始车道检测", command=self.start_lane_detection)
        if out:
            out.release()
        self.lane_cap.release()
        logging.info("车道检测视频处理线程已结束。")


    # ==================== 程序退出时释放资源 ====================
    def on_closing(self):
        """处理窗口关闭事件，释放资源。"""
        # 停止车辆与行人检测线程
        self.is_playing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # 停止停车管理线程
        self.parking_is_playing = False
        if self.parking_cap and self.parking_cap.isOpened():
            self.parking_cap.release()

        # 停止速度估计线程
        self.speed_is_playing = False
        if self.speed_cap and self.speed_cap.isOpened():
            self.speed_cap.release()

        # 停止实例分割与追踪线程
        self.seg_is_playing = False
        if self.seg_cap and self.seg_cap.isOpened():
            self.seg_cap.release()

        # 停止车道检测线程
        self.lane_is_playing = False
        if self.lane_cap and self.lane_cap.isOpened():
            self.lane_cap.release()

        self.root.destroy()

# ============================ 程序入口 ============================
def main():
    root = tk.Tk()
    app = IntegratedApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # 确保资源在关闭时释放
    root.mainloop()

if __name__ == "__main__":
    main()
