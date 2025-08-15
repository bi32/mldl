# 深度学习基础：从神经元到深度网络 🧠

全面掌握深度学习的基础概念、架构和实现。

## 1. 深度学习概述 🌟

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepLearningBasics:
    """深度学习基础概念"""
    
    def __init__(self):
        self.history = {
            "1943": "McCulloch-Pitts神经元模型",
            "1957": "Perceptron感知机",
            "1969": "Minsky证明XOR问题",
            "1986": "反向传播算法",
            "1989": "CNN (LeNet)",
            "1997": "LSTM",
            "2006": "深度信念网络",
            "2012": "AlexNet",
            "2014": "GAN",
            "2017": "Transformer",
            "2018": "BERT/GPT",
            "2020": "GPT-3",
            "2023": "GPT-4"
        }
    
    def demonstrate_neuron(self):
        """演示单个神经元"""
        # 简单的神经元实现
        class Neuron:
            def __init__(self, n_inputs):
                self.weights = np.random.randn(n_inputs)
                self.bias = np.random.randn()
            
            def forward(self, inputs):
                # 线性组合
                z = np.dot(inputs, self.weights) + self.bias
                # 激活函数（sigmoid）
                output = 1 / (1 + np.exp(-z))
                return output
        
        # 示例
        neuron = Neuron(3)
        inputs = np.array([1.0, 2.0, 3.0])
        output = neuron.forward(inputs)
        
        print("单个神经元示例:")
        print(f"输入: {inputs}")
        print(f"权重: {neuron.weights}")
        print(f"偏置: {neuron.bias:.4f}")
        print(f"输出: {output:.4f}")
        
        return neuron
    
    def gradient_descent_visualization(self):
        """梯度下降可视化"""
        # 创建简单的损失函数景观
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2  # 简单的二次函数
        
        # 梯度下降轨迹
        learning_rate = 0.1
        current_pos = np.array([4.0, 4.0])
        trajectory = [current_pos.copy()]
        
        for _ in range(20):
            gradient = 2 * current_pos  # 梯度
            current_pos = current_pos - learning_rate * gradient
            trajectory.append(current_pos.copy())
        
        trajectory = np.array(trajectory)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, levels=20, alpha=0.6)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', 
                markersize=8, linewidth=2, label='梯度下降路径')
        plt.plot(0, 0, 'g*', markersize=15, label='最优点')
        plt.xlabel('参数1')
        plt.ylabel('参数2')
        plt.title('梯度下降可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

## 2. 神经网络基础 🔗

```python
class NeuralNetworkBasics:
    """神经网络基础实现"""
    
    def __init__(self):
        self.models = {}
    
    def build_simple_network(self, input_size=784, hidden_size=128, output_size=10):
        """构建简单的全连接网络"""
        class SimpleNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        model = SimpleNet(input_size, hidden_size, output_size)
        self.models['simple_net'] = model
        
        # 打印模型结构
        print("简单神经网络结构:")
        print(model)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return model
    
    def forward_propagation_demo(self):
        """前向传播演示"""
        # 创建简单网络
        input_size = 3
        hidden_size = 4
        output_size = 2
        
        # 手动定义权重和偏置
        W1 = torch.randn(input_size, hidden_size)
        b1 = torch.randn(hidden_size)
        W2 = torch.randn(hidden_size, output_size)
        b2 = torch.randn(output_size)
        
        # 输入
        x = torch.randn(1, input_size)
        
        print("前向传播演示:")
        print(f"输入形状: {x.shape}")
        
        # 第一层
        z1 = torch.matmul(x, W1) + b1
        a1 = F.relu(z1)
        print(f"隐藏层输出形状: {a1.shape}")
        
        # 第二层
        z2 = torch.matmul(a1, W2) + b2
        output = F.softmax(z2, dim=1)
        print(f"输出形状: {output.shape}")
        print(f"输出值: {output}")
        
        return output
    
    def backpropagation_demo(self):
        """反向传播演示"""
        # 创建简单的计算图
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        w = torch.tensor([0.5, 0.5], requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        
        # 前向传播
        z = torch.sum(x * w) + b
        y = torch.sigmoid(z)
        
        # 定义损失
        target = torch.tensor(1.0)
        loss = (y - target) ** 2
        
        print("反向传播演示:")
        print(f"输入 x: {x.data}")
        print(f"权重 w: {w.data}")
        print(f"偏置 b: {b.data}")
        print(f"输出 y: {y.data:.4f}")
        print(f"损失: {loss.data:.4f}")
        
        # 反向传播
        loss.backward()
        
        print("\n梯度:")
        print(f"dx: {x.grad}")
        print(f"dw: {w.grad}")
        print(f"db: {b.grad:.4f}")
        
        return x, w, b
```

## 3. 激活函数详解 ⚡

```python
class ActivationFunctions:
    """激活函数实现与可视化"""
    
    def __init__(self):
        self.activations = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'swish': lambda x: x * torch.sigmoid(x)
        }
    
    def visualize_activations(self):
        """可视化激活函数"""
        x = torch.linspace(-5, 5, 100)
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, (name, func) in enumerate(self.activations.items()):
            y = func(x)
            
            # 计算导数（近似）
            x_grad = x.clone().requires_grad_(True)
            y_grad = func(x_grad)
            y_grad.sum().backward()
            grad = x_grad.grad
            
            axes[idx].plot(x.numpy(), y.numpy(), label=name, linewidth=2)
            axes[idx].plot(x.numpy(), grad.numpy(), '--', 
                          label=f'{name} gradient', alpha=0.7)
            axes[idx].set_title(name.upper())
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('f(x)')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
            axes[idx].axhline(y=0, color='k', linewidth=0.5)
            axes[idx].axvline(x=0, color='k', linewidth=0.5)
        
        # 隐藏多余的子图
        for idx in range(len(self.activations), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def compare_activations(self, input_tensor):
        """比较不同激活函数的输出"""
        results = {}
        
        for name, func in self.activations.items():
            output = func(input_tensor)
            results[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'zeros': (output == 0).sum().item()
            }
        
        # 转换为DataFrame
        import pandas as pd
        df = pd.DataFrame(results).T
        print("激活函数输出统计:")
        print(df.round(4))
        
        return df
```

## 4. 优化器实现 🎯

```python
class Optimizers:
    """优化器实现与比较"""
    
    def __init__(self):
        self.optimizers = {}
    
    def create_optimizers(self, model, lr=0.01):
        """创建不同的优化器"""
        self.optimizers = {
            'SGD': optim.SGD(model.parameters(), lr=lr),
            'SGD_Momentum': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
            'Adam': optim.Adam(model.parameters(), lr=lr),
            'AdamW': optim.AdamW(model.parameters(), lr=lr),
            'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
            'Adagrad': optim.Adagrad(model.parameters(), lr=lr)
        }
        
        return self.optimizers
    
    def compare_optimizers(self, loss_landscape):
        """比较优化器在损失景观上的表现"""
        # 简化的2D优化问题
        def rosenbrock(x, y):
            return (1 - x)**2 + 100 * (y - x**2)**2
        
        # 不同优化器的轨迹
        trajectories = {}
        
        for name in ['SGD', 'SGD_Momentum', 'Adam']:
            # 初始点
            x = torch.tensor([2.0], requires_grad=True)
            y = torch.tensor([2.0], requires_grad=True)
            
            # 选择优化器
            if name == 'SGD':
                optimizer = optim.SGD([x, y], lr=0.001)
            elif name == 'SGD_Momentum':
                optimizer = optim.SGD([x, y], lr=0.001, momentum=0.9)
            else:  # Adam
                optimizer = optim.Adam([x, y], lr=0.01)
            
            # 优化过程
            trajectory = []
            for _ in range(100):
                optimizer.zero_grad()
                loss = rosenbrock(x, y)
                loss.backward()
                optimizer.step()
                trajectory.append([x.item(), y.item()])
            
            trajectories[name] = np.array(trajectory)
        
        # 可视化
        x_range = np.linspace(-2, 3, 100)
        y_range = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = rosenbrock(X, Y)
        
        plt.figure(figsize=(12, 8))
        plt.contour(X, Y, Z, levels=50, alpha=0.6)
        
        colors = {'SGD': 'blue', 'SGD_Momentum': 'green', 'Adam': 'red'}
        for name, trajectory in trajectories.items():
            plt.plot(trajectory[:, 0], trajectory[:, 1], 
                    'o-', color=colors[name], label=name, 
                    markersize=4, alpha=0.7)
        
        plt.plot(1, 1, 'k*', markersize=15, label='最优点')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('优化器比较：Rosenbrock函数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return trajectories
```

## 5. 正则化技术 🛡️

```python
class RegularizationTechniques:
    """正则化技术实现"""
    
    def __init__(self):
        self.techniques = []
    
    def dropout_demo(self, p=0.5):
        """Dropout演示"""
        # 创建输入
        x = torch.randn(100, 10)
        
        # 训练模式的Dropout
        dropout = nn.Dropout(p=p)
        dropout.train()
        x_train = dropout(x)
        
        # 评估模式的Dropout
        dropout.eval()
        x_eval = dropout(x)
        
        print(f"Dropout演示 (p={p}):")
        print(f"原始输入均值: {x.mean():.4f}")
        print(f"训练模式输出均值: {x_train.mean():.4f}")
        print(f"评估模式输出均值: {x_eval.mean():.4f}")
        print(f"训练模式零值比例: {(x_train == 0).float().mean():.4f}")
        print(f"评估模式零值比例: {(x_eval == 0).float().mean():.4f}")
        
        return x_train, x_eval
    
    def batch_norm_demo(self):
        """批归一化演示"""
        # 创建批归一化层
        bn = nn.BatchNorm1d(10)
        
        # 创建输入（批次大小=32，特征数=10）
        x = torch.randn(32, 10) * 5 + 2  # 均值≈2，标准差≈5
        
        print("批归一化演示:")
        print(f"输入统计 - 均值: {x.mean():.4f}, 标准差: {x.std():.4f}")
        
        # 应用批归一化
        bn.train()
        x_normalized = bn(x)
        
        print(f"输出统计 - 均值: {x_normalized.mean():.4f}, "
              f"标准差: {x_normalized.std():.4f}")
        
        return x_normalized
    
    def weight_decay_comparison(self):
        """权重衰减比较"""
        # 创建简单模型
        model_no_wd = nn.Linear(10, 1)
        model_with_wd = nn.Linear(10, 1)
        
        # 复制初始权重
        model_with_wd.weight.data = model_no_wd.weight.data.clone()
        
        # 创建优化器
        optimizer_no_wd = optim.SGD(model_no_wd.parameters(), lr=0.1)
        optimizer_with_wd = optim.SGD(model_with_wd.parameters(), 
                                     lr=0.1, weight_decay=0.01)
        
        # 训练步骤
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        weight_norms_no_wd = []
        weight_norms_with_wd = []
        
        for _ in range(50):
            # 无权重衰减
            optimizer_no_wd.zero_grad()
            loss = F.mse_loss(model_no_wd(x), y)
            loss.backward()
            optimizer_no_wd.step()
            weight_norms_no_wd.append(model_no_wd.weight.norm().item())
            
            # 有权重衰减
            optimizer_with_wd.zero_grad()
            loss = F.mse_loss(model_with_wd(x), y)
            loss.backward()
            optimizer_with_wd.step()
            weight_norms_with_wd.append(model_with_wd.weight.norm().item())
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(weight_norms_no_wd, label='无权重衰减')
        plt.plot(weight_norms_with_wd, label='有权重衰减 (0.01)')
        plt.xlabel('训练步数')
        plt.ylabel('权重范数')
        plt.title('权重衰减效果比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return weight_norms_no_wd, weight_norms_with_wd
    
    def early_stopping(self, patience=5):
        """早停实现"""
        class EarlyStopping:
            def __init__(self, patience=5, delta=0):
                self.patience = patience
                self.delta = delta
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.Inf
            
            def __call__(self, val_loss, model):
                score = -val_loss
                
                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0
            
            def save_checkpoint(self, val_loss, model):
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                torch.save(model.state_dict(), 'checkpoint.pt')
                self.val_loss_min = val_loss
        
        return EarlyStopping(patience)
```

## 6. 训练循环实现 🔄

```python
class TrainingLoop:
    """完整的训练循环实现"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs, optimizer, criterion):
        """完整训练过程"""
        print("开始训练...")
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion
            )
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 打印进度
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        print("训练完成!")
        return self.history
    
    def plot_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## 7. 实战项目：MNIST分类 🔢

```python
class MNISTProject:
    """MNIST手写数字识别项目"""
    
    def __init__(self):
        self.model = None
        self.train_loader = None
        self.test_loader = None
    
    def prepare_data(self, batch_size=64):
        """准备MNIST数据"""
        from torchvision import datasets, transforms
        
        # 数据变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 下载和加载数据
        train_dataset = datasets.MNIST('./data', train=True, 
                                      download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, 
                                     transform=transform)
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                      shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                     shuffle=False)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        return self.train_loader, self.test_loader
    
    def create_cnn_model(self):
        """创建CNN模型"""
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)
                self.dropout = nn.Dropout(0.25)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 64 * 7 * 7)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)
        
        self.model = CNN()
        print("CNN模型结构:")
        print(self.model)
        
        return self.model
    
    def train_model(self, epochs=10, lr=0.001):
        """训练模型"""
        if self.model is None:
            self.create_cnn_model()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        trainer = TrainingLoop(self.model, device)
        history = trainer.train(
            self.train_loader, 
            self.test_loader,
            epochs,
            optimizer,
            criterion
        )
        
        trainer.plot_history()
        
        return history
    
    def evaluate_model(self):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f'测试集准确率: {accuracy:.2f}%')
        
        return accuracy
    
    def visualize_predictions(self, n_samples=10):
        """可视化预测结果"""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 获取一批数据
        data, target = next(iter(self.test_loader))
        data, target = data[:n_samples].to(device), target[:n_samples]
        
        # 预测
        with torch.no_grad():
            output = self.model(data)
            _, predicted = output.max(1)
        
        # 可视化
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(n_samples):
            img = data[i].cpu().numpy().squeeze()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {target[i].item()}, '
                            f'Pred: {predicted[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
```

## 8. 调试与优化技巧 🔧

```python
class DebuggingTips:
    """深度学习调试技巧"""
    
    @staticmethod
    def check_gradient_flow(model):
        """检查梯度流"""
        # 进行一次前向和反向传播
        x = torch.randn(1, *model.input_shape)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f'{name}: grad_norm = {grad_norm:.6f}')
                
                if grad_norm == 0:
                    print(f'  ⚠️ 警告: {name} 的梯度为0!')
                elif grad_norm > 100:
                    print(f'  ⚠️ 警告: {name} 的梯度可能爆炸!')
    
    @staticmethod
    def diagnose_training_issues(history):
        """诊断训练问题"""
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        # 检查是否学习
        if train_loss[-1] >= train_loss[0] * 0.95:
            print("⚠️ 模型似乎没有学习!")
            print("建议:")
            print("- 检查学习率（可能太小或太大）")
            print("- 检查数据和标签是否正确")
            print("- 检查损失函数是否合适")
        
        # 检查过拟合
        if val_loss[-1] > val_loss[min(5, len(val_loss)-1)] * 1.1:
            print("⚠️ 检测到过拟合!")
            print("建议:")
            print("- 增加正则化（dropout, weight decay）")
            print("- 减少模型容量")
            print("- 增加数据量或数据增强")
        
        # 检查欠拟合
        if train_loss[-1] > 0.5:  # 阈值取决于任务
            print("⚠️ 可能存在欠拟合!")
            print("建议:")
            print("- 增加模型容量")
            print("- 训练更长时间")
            print("- 减少正则化")
    
    @staticmethod
    def memory_optimization_tips():
        """内存优化技巧"""
        tips = """
        GPU内存优化技巧:
        
        1. 减小批次大小
        2. 使用梯度累积
        3. 使用混合精度训练 (AMP)
        4. 及时删除不需要的中间变量
        5. 使用 torch.no_grad() 在推理时
        6. 使用 checkpoint 技术（梯度检查点）
        7. 优化数据加载（num_workers, pin_memory）
        8. 使用更小的模型或模型剪枝
        """
        print(tips)
        
        # 显示当前GPU内存使用
        if torch.cuda.is_available():
            print(f"\n当前GPU内存使用:")
            print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
```

## 最佳实践总结 📋

```python
def deep_learning_best_practices():
    """深度学习最佳实践"""
    
    practices = {
        "数据准备": [
            "数据归一化/标准化",
            "数据增强提高泛化",
            "检查数据分布",
            "处理类别不平衡",
            "合理的训练/验证/测试分割"
        ],
        
        "模型设计": [
            "从简单模型开始",
            "逐步增加复杂度",
            "使用预训练模型",
            "注意感受野大小",
            "考虑计算效率"
        ],
        
        "训练技巧": [
            "使用合适的初始化",
            "选择合适的优化器",
            "学习率调度",
            "早停防止过拟合",
            "梯度裁剪防止爆炸"
        ],
        
        "调试方法": [
            "可视化中间层输出",
            "监控梯度流",
            "检查损失曲线",
            "使用TensorBoard",
            "小数据集快速迭代"
        ],
        
        "性能优化": [
            "混合精度训练",
            "多GPU并行",
            "优化数据管道",
            "模型量化",
            "知识蒸馏"
        ]
    }
    
    return practices

# 常见错误
common_mistakes = """
深度学习常见错误：

1. 忘记归一化输入数据
2. 学习率设置不当
3. 批次大小太大导致内存溢出
4. 忘记设置model.eval()在测试时
5. 数据泄露（测试数据泄露到训练）
6. 不正确的损失函数
7. 忘记梯度清零
8. 维度不匹配
9. 使用了错误的激活函数
10. 过早优化（应该先让模型工作）
"""

print("深度学习基础指南加载完成！")
```

## 下一步学习
- [CNN架构](cnn_architectures.md) - 卷积神经网络深入
- [训练技巧](training_tricks.md) - 高级训练技术
- [NLP模型](nlp_models.md) - 自然语言处理