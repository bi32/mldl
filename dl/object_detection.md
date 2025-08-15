# YOLO系列目标检测完全指南 🎯

YOLO（You Only Look Once）将目标检测作为回归问题，一次前向传播即可得到所有检测结果。从YOLOv1到YOLOv8，让我们探索这个传奇系列的演进。

## 1. YOLOv5 - 工程化典范 🔧

### 安装和使用

```python
# 克隆YOLOv5仓库
# git clone https://github.com/ultralytics/yolov5
# cd yolov5
# pip install -r requirements.txt

import torch
import cv2
import numpy as np
from pathlib import Path

# 加载预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 推理
def detect_objects(image_path):
    """使用YOLOv5检测目标"""
    # 读取图像
    img = cv2.imread(image_path)
    
    # 推理
    results = model(img)
    
    # 解析结果
    detections = results.pandas().xyxy[0]  # 获取检测框
    
    # 可视化
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), \
                         int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = row['name']
        
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 标签
        label = f'{cls} {conf:.2f}'
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img, detections

# 训练自定义数据集
def train_custom_yolov5():
    """训练YOLOv5"""
    import yaml
    
    # 创建数据配置文件
    data_config = {
        'path': './datasets/custom',  # 数据集路径
        'train': 'images/train',
        'val': 'images/val',
        'nc': 10,  # 类别数
        'names': ['class1', 'class2', '...']  # 类别名称
    }
    
    with open('custom_data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    # 训练命令
    train_cmd = """
    python train.py \\
        --img 640 \\
        --batch 16 \\
        --epochs 100 \\
        --data custom_data.yaml \\
        --weights yolov5s.pt \\
        --cache
    """
    
    print(f"训练命令:\n{train_cmd}")
```

## 2. YOLOv8 - 最新SOTA 🚀

```python
# pip install ultralytics

from ultralytics import YOLO
import cv2
import numpy as np

# YOLOv8使用示例
class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """初始化YOLOv8检测器"""
        self.model = YOLO(model_path)
        
    def detect(self, image, conf_threshold=0.25):
        """检测目标"""
        results = self.model(image, conf=conf_threshold)
        return results
    
    def track(self, video_path):
        """目标跟踪"""
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跟踪
            results = self.model.track(frame, persist=True)
            
            # 可视化
            annotated_frame = results[0].plot()
            cv2.imshow('YOLOv8 Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def train(self, data_yaml, epochs=100):
        """训练模型"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device='cuda'
        )
        return results
    
    def export(self, format='onnx'):
        """导出模型"""
        self.model.export(format=format)

# 使用示例
detector = YOLOv8Detector('yolov8n.pt')

# 检测图像
image = cv2.imread('test.jpg')
results = detector.detect(image)

# 获取检测框
for r in results:
    boxes = r.boxes  # 检测框
    masks = r.masks  # 分割掩码（如果有）
    probs = r.probs  # 分类概率
    
    # 解析检测框
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        
        print(f"检测到: 类别{cls}, 置信度{conf:.2f}, "
              f"位置({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

# 实例分割（YOLOv8-Seg）
seg_model = YOLO('yolov8n-seg.pt')
results = seg_model(image)

# 姿态估计（YOLOv8-Pose）
pose_model = YOLO('yolov8n-pose.pt')
results = pose_model(image)
```

## 3. 自定义YOLO实现（教学版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3(nn.Module):
    """简化版YOLOv3实现"""
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        
        # Darknet-53骨干网络（简化版）
        self.backbone = self._make_darknet53()
        
        # FPN neck
        self.neck = self._make_fpn()
        
        # Detection heads
        self.heads = nn.ModuleList([
            self._make_detection_head(512, num_classes),  # 大目标
            self._make_detection_head(256, num_classes),  # 中目标
            self._make_detection_head(128, num_classes),  # 小目标
        ])
        
    def _make_darknet53(self):
        """构建Darknet-53骨干"""
        return nn.Sequential(
            # 省略具体实现，使用简化版
            ConvBlock(3, 32, 3, 1),
            ConvBlock(32, 64, 3, 2),
            ResidualBlock(64, 32, 64, num_blocks=1),
            ConvBlock(64, 128, 3, 2),
            ResidualBlock(128, 64, 128, num_blocks=2),
            ConvBlock(128, 256, 3, 2),
            ResidualBlock(256, 128, 256, num_blocks=8),
            ConvBlock(256, 512, 3, 2),
            ResidualBlock(512, 256, 512, num_blocks=8),
            ConvBlock(512, 1024, 3, 2),
            ResidualBlock(1024, 512, 1024, num_blocks=4),
        )
    
    def _make_fpn(self):
        """构建FPN"""
        return nn.ModuleList([
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 128, 1),
        ])
    
    def _make_detection_head(self, in_channels, num_classes):
        """构建检测头"""
        return nn.Sequential(
            ConvBlock(in_channels, in_channels * 2, 3, 1),
            nn.Conv2d(in_channels * 2, 3 * (5 + num_classes), 1)
            # 3个anchor × (4个坐标 + 1个置信度 + num_classes个类别)
        )
    
    def forward(self, x):
        # 骨干网络
        features = self.backbone(x)
        
        # FPN和检测
        outputs = []
        for i, head in enumerate(self.heads):
            feat = self.neck[i](features[-(i+1)])
            output = head(feat)
            outputs.append(output)
        
        return outputs

class ConvBlock(nn.Module):
    """卷积块：Conv + BN + LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks):
        super(ResidualBlock, self).__init__()
        self.blocks = nn.Sequential(
            *[self._make_layer(in_channels if i == 0 else out_channels,
                              hidden_channels, out_channels)
              for i in range(num_blocks)]
        )
    
    def _make_layer(self, in_channels, hidden_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, hidden_channels, 1, 1),
            ConvBlock(hidden_channels, out_channels, 3, 1)
        )
    
    def forward(self, x):
        return x + self.blocks(x)

# YOLO损失函数
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80, anchors=None):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors or [(10,13), (16,30), (33,23)]
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        """
        计算YOLO损失
        predictions: 模型输出
        targets: 真实标签 [batch, max_objects, 5] (x,y,w,h,class)
        """
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # 解析预测
        # predictions shape: [batch, 3*(5+num_classes), H, W]
        predictions = predictions.view(batch_size, 3, 5 + self.num_classes,
                                      predictions.shape[2], predictions.shape[3])
        predictions = predictions.permute(0, 1, 3, 4, 2)  # [batch, 3, H, W, 5+classes]
        
        # 提取预测值
        pred_xy = torch.sigmoid(predictions[..., :2])    # 中心坐标
        pred_wh = torch.exp(predictions[..., 2:4])       # 宽高
        pred_conf = torch.sigmoid(predictions[..., 4:5]) # 置信度
        pred_cls = predictions[..., 5:]                  # 类别
        
        # 计算损失
        loss_xy = self.mse_loss(pred_xy, targets[..., :2])
        loss_wh = self.mse_loss(pred_wh, targets[..., 2:4])
        loss_conf = self.bce_loss(pred_conf, targets[..., 4:5])
        loss_cls = self.ce_loss(pred_cls.reshape(-1, self.num_classes),
                               targets[..., 5].long().reshape(-1))
        
        # 总损失
        total_loss = loss_xy + loss_wh + loss_conf + loss_cls
        
        return total_loss, {
            'xy': loss_xy.item(),
            'wh': loss_wh.item(),
            'conf': loss_conf.item(),
            'cls': loss_cls.item()
        }
```

## 4. 数据准备和增强

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLODataset(torch.utils.data.Dataset):
    """YOLO数据集"""
    def __init__(self, images_dir, labels_dir, img_size=640, augment=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        
        self.images = list(self.images_dir.glob('*.jpg'))
        
        # 数据增强
        self.transform = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(cls))
        
        # 数据增强
        if self.augment and len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 转换标签格式
        targets = torch.zeros((len(bboxes), 6))
        for i, (box, cls) in enumerate(zip(bboxes, class_labels)):
            targets[i] = torch.tensor([0, cls, *box])  # [batch_idx, class, x, y, w, h]
        
        return image, targets

# Mosaic数据增强
def mosaic_augmentation(images, labels, img_size=640):
    """Mosaic数据增强：将4张图片拼接成一张"""
    mosaic_img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mosaic_labels = []
    
    # 随机选择分割点
    xc = np.random.randint(img_size * 0.25, img_size * 0.75)
    yc = np.random.randint(img_size * 0.25, img_size * 0.75)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        h, w = img.shape[:2]
        
        # 确定放置位置
        if i == 0:  # 左上
            x1, y1, x2, y2 = 0, 0, xc, yc
            sx1, sy1, sx2, sy2 = w - xc, h - yc, w, h
        elif i == 1:  # 右上
            x1, y1, x2, y2 = xc, 0, img_size, yc
            sx1, sy1, sx2, sy2 = 0, h - yc, w - xc, h
        elif i == 2:  # 左下
            x1, y1, x2, y2 = 0, yc, xc, img_size
            sx1, sy1, sx2, sy2 = w - xc, 0, w, h - yc
        else:  # 右下
            x1, y1, x2, y2 = xc, yc, img_size, img_size
            sx1, sy1, sx2, sy2 = 0, 0, w - xc, h - yc
        
        # 放置图片片段
        mosaic_img[y1:y2, x1:x2] = img[sy1:sy2, sx1:sx2]
        
        # 调整标签
        for box in label:
            cls, x, y, w, h = box
            # 转换坐标并裁剪
            # ... 坐标转换逻辑
            mosaic_labels.append([cls, x, y, w, h])
    
    return mosaic_img, mosaic_labels
```

## 5. 模型评估和可视化

```python
def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """计算mAP（mean Average Precision）"""
    from collections import defaultdict
    
    # 按类别组织预测和真实框
    detections = defaultdict(list)
    gt_boxes = defaultdict(list)
    
    for pred in predictions:
        detections[pred['class']].append({
            'bbox': pred['bbox'],
            'score': pred['score']
        })
    
    for gt in ground_truths:
        gt_boxes[gt['class']].append(gt['bbox'])
    
    # 计算每个类别的AP
    aps = []
    for cls in gt_boxes.keys():
        # 排序预测框
        dets = sorted(detections[cls], key=lambda x: x['score'], reverse=True)
        gts = gt_boxes[cls]
        
        # 计算precision和recall
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        
        for i, det in enumerate(dets):
            max_iou = 0
            max_idx = -1
            
            for j, gt in enumerate(gts):
                iou = calculate_iou(det['bbox'], gt)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold:
                tp[i] = 1
                gts.pop(max_idx)  # 移除已匹配的真实框
            else:
                fp[i] = 1
        
        # 计算AP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(gt_boxes[cls])
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 计算AP (11点插值)
        ap = 0
        for r in np.linspace(0, 1, 11):
            if np.any(recall >= r):
                ap += np.max(precision[recall >= r])
        ap /= 11
        
        aps.append(ap)
    
    return np.mean(aps)

def calculate_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# 可视化检测结果
def visualize_detections(image, detections, class_names):
    """可视化检测结果"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_id = det['class']
        score = det['score']
        
        # 画框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor=colors[cls_id],
                                facecolor='none')
        ax.add_patch(rect)
        
        # 标签
        label = f'{class_names[cls_id]}: {score:.2f}'
        ax.text(x1, y1-5, label, color='white',
               bbox=dict(facecolor=colors[cls_id], alpha=0.5))
    
    ax.axis('off')
    plt.title('YOLO Detection Results')
    plt.tight_layout()
    plt.show()
```

## 6. 部署优化

```python
# ONNX导出
def export_to_onnx(model, img_size=640):
    """导出YOLO模型到ONNX"""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        "yolo.onnx",
        opset_version=11,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    print("模型已导出到yolo.onnx")

# TensorRT加速
def tensorrt_inference():
    """使用TensorRT加速推理"""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # 构建TensorRT引擎
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX
    with open("yolo.onnx", 'rb') as f:
        parser.parse(f.read())
    
    # 构建引擎
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)
    
    return engine

# 量化
def quantize_model(model):
    """INT8量化"""
    import torch.quantization as quant
    
    # 准备量化
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)
    
    # 校准（需要代表性数据）
    # ... 运行一些推理
    
    # 转换为量化模型
    quant.convert(model, inplace=True)
    
    return model
```

## 最佳实践建议

### 1. 模型选择
- **速度优先**: YOLOv5n/YOLOv8n
- **精度优先**: YOLOv5x/YOLOv8x
- **平衡选择**: YOLOv5s/YOLOv8s

### 2. 训练技巧
- 使用预训练权重
- Mosaic和MixUp数据增强
- 多尺度训练
- 标签平滑

### 3. 部署优化
- ONNX/TensorRT部署
- INT8量化
- 批处理推理
- NMS优化

## 下一步学习
- [PyTorch部署](deployment.md) - 模型部署详解
- [NLP模型](nlp_models.md) - BERT、GPT等
- [LLM部署](llm_deployment.md) - 大模型部署