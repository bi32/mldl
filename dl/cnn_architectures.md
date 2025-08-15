# CNN架构演进 - 从VGG到EfficientNet 🏗️

卷积神经网络（CNN）就像人类的视觉系统，通过层层提取特征来理解图像。让我们探索CNN架构的演进历程。

## 1. VGG - 简单即美 🎯

### 核心思想
VGG证明了一个简单的道理：使用更小的卷积核（3×3）和更深的网络可以取得更好的效果。

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# VGG块的基本构建单元
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

# VGG16实现
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # VGG16的架构：[64,64,'M', 128,128,'M', 256,256,256,'M', 512,512,512,'M', 512,512,512,'M']
        self.features = nn.Sequential(
            # Block 1
            VGGBlock(3, 64, 2),     # 64x2 convolutions
            # Block 2
            VGGBlock(64, 128, 2),   # 128x2 convolutions
            # Block 3
            VGGBlock(128, 256, 3),  # 256x3 convolutions
            # Block 4
            VGGBlock(256, 512, 3),  # 512x3 convolutions
            # Block 5
            VGGBlock(512, 512, 3),  # 512x3 convolutions
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 简化版VGG（适合CIFAR-10）
class VGG11_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11_CIFAR, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 数据准备
def prepare_data():
    """准备CIFAR-10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# 训练函数
def train_model(model, trainloader, testloader, epochs=10):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': running_loss/len(trainloader), 
                             'Acc': 100.*correct/total})
        
        train_losses.append(running_loss/len(trainloader))
        scheduler.step()
        
        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.3f}, '
              f'Test Acc: {test_acc:.2f}%')
    
    return train_losses, test_accs

# 训练VGG
print("=== 训练VGG11 ===")
trainloader, testloader, classes = prepare_data()
vgg_model = VGG11_CIFAR(num_classes=10)
print(f"模型参数量: {sum(p.numel() for p in vgg_model.parameters())/1e6:.2f}M")

# 训练模型（演示用，实际训练需要更多epochs）
# train_losses, test_accs = train_model(vgg_model, trainloader, testloader, epochs=5)
```

## 2. ResNet - 残差革命 🔄

### 核心思想
ResNet通过残差连接解决了深层网络的退化问题，让网络可以轻松训练到上百层。

```python
# 基本残差块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = torch.relu(out)
        
        return out

# Bottleneck残差块（用于更深的网络）
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = torch.relu(out)
        
        return out

# ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 创建不同深度的ResNet
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

# 简化版ResNet for CIFAR-10
class ResNet20_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_CIFAR, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(BasicBlock(self.in_channels, out_channels, stride))
            else:
                layers.append(BasicBlock(out_channels, out_channels, 1))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

print("\n=== ResNet架构对比 ===")
models = {
    'ResNet18': ResNet18(),
    'ResNet34': ResNet34(),
    'ResNet50': ResNet50(),
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params/1e6:.2f}M parameters")
```

## 3. EfficientNet - 效率之王 ⚡

### 核心思想
EfficientNet通过复合缩放（同时缩放深度、宽度和分辨率）达到最佳的精度-效率平衡。

```python
import math

# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# MBConv块（EfficientNet的基本单元）
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, reduction=4, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        
        if self.expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )
        
        # Squeeze and Excitation
        self.se = SEBlock(hidden_dim, reduction)
        
        # Output phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if self.expand:
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            if self.drop_connect_rate > 0 and self.training:
                x = self._drop_connect(x)
            x = x + identity
        
        return x
    
    def _drop_connect(self, x):
        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x * binary_tensor / keep_prob

# EfficientNet-B0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetB0, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # Stage 1
            MBConvBlock(32, 16, 3, 1, expand_ratio=1),
            # Stage 2
            MBConvBlock(16, 24, 3, 2, expand_ratio=6),
            MBConvBlock(24, 24, 3, 1, expand_ratio=6),
            # Stage 3
            MBConvBlock(24, 40, 5, 2, expand_ratio=6),
            MBConvBlock(40, 40, 5, 1, expand_ratio=6),
            # Stage 4
            MBConvBlock(40, 80, 3, 2, expand_ratio=6),
            MBConvBlock(80, 80, 3, 1, expand_ratio=6),
            MBConvBlock(80, 80, 3, 1, expand_ratio=6),
            # Stage 5
            MBConvBlock(80, 112, 5, 1, expand_ratio=6),
            MBConvBlock(112, 112, 5, 1, expand_ratio=6),
            MBConvBlock(112, 112, 5, 1, expand_ratio=6),
            # Stage 6
            MBConvBlock(112, 192, 5, 2, expand_ratio=6),
            MBConvBlock(192, 192, 5, 1, expand_ratio=6),
            MBConvBlock(192, 192, 5, 1, expand_ratio=6),
            MBConvBlock(192, 192, 5, 1, expand_ratio=6),
            # Stage 7
            MBConvBlock(192, 320, 3, 1, expand_ratio=6),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

## 4. MobileNet - 移动设备优化 📱

```python
# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# MobileNetV2的Inverted Residual块
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        
        # 配置
        inverted_residual_setting = [
            # t, c, n, s (expand_ratio, out_channels, num_blocks, stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(32 * width_mult)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # 构建Inverted Residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        # 最后的卷积层
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## 5. 架构对比与选择指南

```python
# 模型对比分析
def compare_models():
    """对比不同CNN架构"""
    models = {
        'VGG11': VGG11_CIFAR(),
        'ResNet20': ResNet20_CIFAR(),
        'MobileNetV2': MobileNetV2(),
        'EfficientNetB0': EfficientNetB0()
    }
    
    # 统计模型信息
    model_info = []
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算FLOPs（简化版）
        input_tensor = torch.randn(1, 3, 32, 32)
        
        model_info.append({
            '模型': name,
            '总参数': f'{total_params/1e6:.2f}M',
            '可训练参数': f'{trainable_params/1e6:.2f}M'
        })
    
    import pandas as pd
    df = pd.DataFrame(model_info)
    print("\n=== CNN架构对比 ===")
    print(df.to_string(index=False))
    
    return models

# 速度测试
def benchmark_speed(models, input_size=(1, 3, 224, 224), num_iterations=100):
    """测试推理速度"""
    print("\n=== 推理速度测试 ===")
    
    for name, model in models.items():
        model.eval()
        input_tensor = torch.randn(input_size)
        
        # 预热
        for _ in range(10):
            _ = model(input_tensor)
        
        # 计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations * 1000  # ms
        
        print(f"{name}: {avg_time:.2f}ms per image")

# 可视化特征图
def visualize_feature_maps(model, input_image):
    """可视化CNN的中间特征图"""
    activation = {}
    
    def hook_fn(module, input, output):
        activation['output'] = output.detach()
    
    # 注册hook
    target_layer = model.features[0]  # 第一层卷积
    hook = target_layer.register_forward_hook(hook_fn)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(input_image)
    
    # 获取特征图
    feature_maps = activation['output'].squeeze(0)
    
    # 可视化
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for idx, ax in enumerate(axes.flat):
        if idx < feature_maps.shape[0]:
            ax.imshow(feature_maps[idx].cpu().numpy(), cmap='viridis')
            ax.axis('off')
    
    plt.suptitle('第一层卷积特征图')
    plt.tight_layout()
    plt.show()
    
    hook.remove()

# 执行对比
models = compare_models()

# 选择指南
print("\n=== 模型选择指南 ===")
selection_guide = """
1. VGG:
   - 优点：结构简单，易于理解和实现
   - 缺点：参数量大，计算量大
   - 适用：学习CNN基础，不考虑效率的场景

2. ResNet:
   - 优点：可以训练很深的网络，性能优秀
   - 缺点：相对复杂
   - 适用：需要高精度的任务，服务器部署

3. MobileNet:
   - 优点：轻量级，适合移动设备
   - 缺点：精度略低
   - 适用：移动端、嵌入式设备、实时应用

4. EfficientNet:
   - 优点：精度-效率平衡最佳
   - 缺点：结构复杂，训练技巧要求高
   - 适用：追求最佳性价比的场景
"""
print(selection_guide)
```

## 6. 实战：迁移学习

```python
def transfer_learning_example():
    """使用预训练模型进行迁移学习"""
    import torchvision.models as models
    
    # 加载预训练的ResNet18
    model = models.resnet18(pretrained=True)
    
    # 冻结特征提取层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)  # 10个类别
    )
    
    # 只训练新添加的层
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    print("迁移学习模型准备完成")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model

# 数据增强技巧
def get_augmentation_transforms():
    """获取数据增强变换"""
    from torchvision.transforms import v2
    
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        v2.RandomRotation(15),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms
```

## 最佳实践建议

### 1. 选择合适的架构
- **高精度需求**：ResNet50/101, EfficientNet-B4+
- **速度优先**：MobileNet, EfficientNet-B0
- **平衡选择**：ResNet18/34, EfficientNet-B1/B2

### 2. 训练技巧
- 使用预训练模型
- 渐进式解冻
- 学习率调度
- 数据增强

### 3. 优化策略
- 混合精度训练
- 梯度累积
- 知识蒸馏
- 模型剪枝

## 下一步学习
- [Vision Transformer](vision_transformer.md) - 计算机视觉的Transformer革命
- [目标检测](object_detection.md) - YOLO系列详解
- [PyTorch部署](deployment.md) - 模型部署实战