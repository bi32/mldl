# 深度学习训练技巧大全 🚀

掌握让模型训练更快、更稳定、更准确的所有技巧。

## 1. 学习率调度策略 📈

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import numpy as np
import matplotlib.pyplot as plt

class LearningRateSchedulers:
    """学习率调度器实现"""
    
    def __init__(self, optimizer, initial_lr=0.1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.schedulers = {}
    
    def create_schedulers(self, epochs=100):
        """创建各种学习率调度器"""
        # 阶梯衰减
        self.schedulers['StepLR'] = StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )
        
        # 多阶梯衰减
        self.schedulers['MultiStepLR'] = MultiStepLR(
            self.optimizer, milestones=[30, 60, 90], gamma=0.1
        )
        
        # 指数衰减
        self.schedulers['ExponentialLR'] = ExponentialLR(
            self.optimizer, gamma=0.95
        )
        
        # 余弦退火
        self.schedulers['CosineAnnealingLR'] = CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        # 带热重启的余弦退火
        self.schedulers['CosineAnnealingWarmRestarts'] = \
            CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        
        # ReduceLROnPlateau（需要验证损失）
        self.schedulers['ReduceLROnPlateau'] = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # OneCycleLR
        self.schedulers['OneCycleLR'] = OneCycleLR(
            self.optimizer, max_lr=0.1, 
            steps_per_epoch=100, epochs=epochs
        )
        
        return self.schedulers
    
    def visualize_schedules(self, epochs=100):
        """可视化不同的学习率调度"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, (name, scheduler) in enumerate(self.schedulers.items()):
            if idx >= len(axes):
                break
            
            lrs = []
            optimizer_temp = optim.SGD([torch.randn(2, 2)], lr=self.initial_lr)
            
            if name == 'OneCycleLR':
                scheduler_temp = OneCycleLR(
                    optimizer_temp, max_lr=0.1,
                    steps_per_epoch=1, epochs=epochs
                )
            elif name == 'ReduceLROnPlateau':
                scheduler_temp = ReduceLROnPlateau(
                    optimizer_temp, mode='min', factor=0.5, patience=10
                )
            elif name == 'CosineAnnealingWarmRestarts':
                scheduler_temp = CosineAnnealingWarmRestarts(
                    optimizer_temp, T_0=10, T_mult=2
                )
            elif name == 'CosineAnnealingLR':
                scheduler_temp = CosineAnnealingLR(
                    optimizer_temp, T_max=epochs
                )
            elif name == 'StepLR':
                scheduler_temp = StepLR(
                    optimizer_temp, step_size=30, gamma=0.1
                )
            elif name == 'MultiStepLR':
                scheduler_temp = MultiStepLR(
                    optimizer_temp, milestones=[30, 60, 90], gamma=0.1
                )
            else:  # ExponentialLR
                scheduler_temp = ExponentialLR(
                    optimizer_temp, gamma=0.95
                )
            
            for epoch in range(epochs):
                lrs.append(optimizer_temp.param_groups[0]['lr'])
                if name == 'ReduceLROnPlateau':
                    # 模拟验证损失
                    val_loss = np.random.random()
                    scheduler_temp.step(val_loss)
                else:
                    scheduler_temp.step()
            
            axes[idx].plot(lrs)
            axes[idx].set_title(name)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Learning Rate')
            axes[idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(self.schedulers), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# Warmup实现
class WarmupScheduler:
    """带Warmup的学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 线性warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # warmup后的调度策略
            lr = self.target_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
```

## 2. 梯度处理技巧 🎯

```python
class GradientTechniques:
    """梯度处理技巧"""
    
    @staticmethod
    def gradient_clipping(model, max_norm=1.0):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # 检查梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        print(f"梯度范数: {total_norm:.4f}")
        if total_norm > max_norm:
            print(f"梯度被裁剪: {total_norm:.4f} -> {max_norm}")
        
        return total_norm
    
    @staticmethod
    def gradient_accumulation(model, data_loader, optimizer, 
                            accumulation_steps=4):
        """梯度累积"""
        model.train()
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(data_loader):
            # 前向传播
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # 标准化损失
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 每accumulation_steps步更新一次
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Step {i+1}: Updated weights")
    
    @staticmethod
    def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10):
        """梯度惩罚（用于WGAN-GP）"""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        
        # 插值
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算判别器输出
        d_interpolated = discriminator(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return lambda_gp * gradient_penalty
    
    @staticmethod
    def gradient_centralization(optimizer):
        """梯度中心化"""
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # 计算梯度均值
                    grad = p.grad.data
                    if len(grad.shape) > 1:
                        # 对每个输出通道中心化
                        grad -= grad.mean(dim=tuple(range(1, len(grad.shape))), 
                                        keepdim=True)
```

## 3. 混合精度训练 ⚡

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTraining:
    """混合精度训练"""
    
    def __init__(self):
        self.scaler = GradScaler()
    
    def train_with_amp(self, model, train_loader, optimizer, epochs=5):
        """使用自动混合精度训练"""
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                
                # 自动混合精度
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()
                
                # 更新权重
                self.scaler.step(optimizer)
                self.scaler.update()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    def manual_mixed_precision(self, model, x):
        """手动混合精度"""
        # 将输入转换为FP16
        x_fp16 = x.half()
        
        # 模型前向传播（FP16）
        with torch.cuda.amp.autocast():
            output_fp16 = model(x_fp16)
        
        # 损失计算（FP32）
        output_fp32 = output_fp16.float()
        loss = output_fp32.mean()
        
        return loss
    
    def compare_precision(self):
        """比较不同精度的影响"""
        # 创建测试数据
        x = torch.randn(100, 100).cuda()
        
        # FP32
        x_fp32 = x.float()
        result_fp32 = torch.matmul(x_fp32, x_fp32.T)
        
        # FP16
        x_fp16 = x.half()
        result_fp16 = torch.matmul(x_fp16, x_fp16.T)
        
        # BF16 (如果支持)
        if torch.cuda.is_bf16_supported():
            x_bf16 = x.bfloat16()
            result_bf16 = torch.matmul(x_bf16, x_bf16.T)
        
        # 比较结果
        print("精度比较:")
        print(f"FP32 范围: [{result_fp32.min():.6f}, {result_fp32.max():.6f}]")
        print(f"FP16 范围: [{result_fp16.min():.6f}, {result_fp16.max():.6f}]")
        
        # 误差分析
        error = (result_fp32 - result_fp16.float()).abs().mean()
        print(f"FP32 vs FP16 平均误差: {error:.6f}")
```

## 4. 数据增强技巧 🖼️

```python
import torchvision.transforms as transforms

class DataAugmentation:
    """数据增强技巧"""
    
    @staticmethod
    def create_augmentation_pipeline(task='classification'):
        """创建数据增强管道"""
        if task == 'classification':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                      saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        return train_transform, val_transform
    
    @staticmethod
    def mixup(x, y, alpha=1.0):
        """Mixup数据增强"""
        batch_size = x.size(0)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def cutmix(x, y, alpha=1.0):
        """CutMix数据增强"""
        batch_size = x.size(0)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(x.device)
        
        # 生成裁剪区域
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 应用CutMix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return x, y, y[index], lam
    
    @staticmethod
    def auto_augment():
        """AutoAugment策略"""
        from torchvision.transforms import autoaugment
        
        # 使用ImageNet策略
        augmenter = autoaugment.AutoAugment(
            autoaugment.AutoAugmentPolicy.IMAGENET
        )
        
        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform
```

## 5. 训练稳定性技巧 🔧

```python
class TrainingStability:
    """训练稳定性技巧"""
    
    @staticmethod
    def label_smoothing(targets, n_classes, epsilon=0.1):
        """标签平滑"""
        with torch.no_grad():
            targets = targets * (1 - epsilon) + epsilon / n_classes
        return targets
    
    @staticmethod
    def stochastic_depth(x, survival_prob=0.8, training=True):
        """随机深度（用于ResNet等）"""
        if not training:
            return x
        
        # 二项分布采样
        survival = torch.bernoulli(torch.tensor(survival_prob))
        
        if survival:
            return x / survival_prob
        else:
            return torch.zeros_like(x)
    
    @staticmethod
    def gradient_checkpointing_example():
        """梯度检查点示例"""
        import torch.utils.checkpoint as checkpoint
        
        class CheckpointedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1000, 1000)
                self.layer2 = nn.Linear(1000, 1000)
                self.layer3 = nn.Linear(1000, 10)
            
            def forward(self, x):
                # 使用checkpoint减少内存使用
                x = checkpoint.checkpoint(self.layer1, x)
                x = checkpoint.checkpoint(self.layer2, x)
                x = self.layer3(x)
                return x
        
        return CheckpointedModel()
    
    @staticmethod
    def ema_model(model, ema_decay=0.999):
        """指数移动平均模型"""
        class EMA:
            def __init__(self, model, decay=0.999):
                self.model = model
                self.decay = decay
                self.shadow = {}
                self.backup = {}
                
                # 初始化shadow参数
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()
            
            def update(self):
                """更新EMA参数"""
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        new_average = (1.0 - self.decay) * param.data + \
                                     self.decay * self.shadow[name]
                        self.shadow[name] = new_average.clone()
            
            def apply_shadow(self):
                """应用EMA参数"""
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.backup[name] = param.data
                        param.data = self.shadow[name]
            
            def restore(self):
                """恢复原始参数"""
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data = self.backup[name]
                self.backup = {}
        
        return EMA(model, ema_decay)
```

## 6. 分布式训练 🌐

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTraining:
    """分布式训练技巧"""
    
    @staticmethod
    def setup_distributed(rank, world_size):
        """设置分布式环境"""
        import os
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # 设置设备
        torch.cuda.set_device(rank)
    
    @staticmethod
    def cleanup_distributed():
        """清理分布式环境"""
        dist.destroy_process_group()
    
    @staticmethod
    def distributed_training_loop(rank, world_size, model, dataset):
        """分布式训练循环"""
        # 设置分布式
        DistributedTraining.setup_distributed(rank, world_size)
        
        # 将模型移到GPU
        model = model.to(rank)
        
        # 包装为DDP模型
        ddp_model = DDP(model, device_ids=[rank])
        
        # 创建分布式采样器
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        # 数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4
        )
        
        # 优化器
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
        
        # 训练
        for epoch in range(10):
            sampler.set_epoch(epoch)  # 重要：每个epoch都要设置
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(rank), target.to(rank)
                
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0 and rank == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 清理
        DistributedTraining.cleanup_distributed()
    
    @staticmethod
    def all_reduce_example():
        """All-Reduce示例"""
        # 假设在分布式环境中
        tensor = torch.ones(1).cuda()
        
        # All-reduce操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # 现在tensor包含所有进程的和
        world_size = dist.get_world_size()
        tensor = tensor / world_size  # 平均
        
        return tensor
```

## 7. 内存优化技巧 💾

```python
class MemoryOptimization:
    """内存优化技巧"""
    
    @staticmethod
    def optimize_dataloader(dataset):
        """优化数据加载器"""
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,  # 多进程加载
            pin_memory=True,  # 固定内存
            persistent_workers=True,  # 持久化工作进程
            prefetch_factor=2  # 预取因子
        )
        return dataloader
    
    @staticmethod
    def clear_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 打印内存使用情况
            print(f"GPU内存已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU内存已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    @staticmethod
    def inplace_operations():
        """原地操作示例"""
        x = torch.randn(1000, 1000)
        
        # 非原地操作（使用更多内存）
        y = x + 1
        y = F.relu(y)
        
        # 原地操作（节省内存）
        x.add_(1)  # 原地加法
        x.relu_()  # 原地ReLU
        
        return x
    
    @staticmethod
    def memory_efficient_attention(Q, K, V, chunk_size=1024):
        """内存高效的注意力机制"""
        batch_size, seq_len, d_model = Q.shape
        
        # 分块计算注意力
        attn_output = torch.zeros_like(Q)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            
            # 计算当前块的注意力
            Q_chunk = Q[:, i:end_i]
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / np.sqrt(d_model)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output[:, i:end_i] = torch.matmul(attn_weights, V)
        
        return attn_output
```

## 8. 调试和监控 📊

```python
class DebuggingMonitoring:
    """调试和监控技巧"""
    
    @staticmethod
    def hook_example(model):
        """使用钩子调试"""
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations[module.__class__.__name__] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            gradients[module.__class__.__name__] = grad_output[0].detach()
        
        # 注册钩子
        for name, module in model.named_modules():
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
        
        return activations, gradients
    
    @staticmethod
    def gradient_flow_check(model):
        """检查梯度流"""
        # 一次前向和反向传播
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        loss = y.mean()
        loss.backward()
        
        # 分析梯度
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        
        # 可视化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7)
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001)
        plt.xlabel("Layers")
        plt.ylabel("Average gradient")
        plt.title("Gradient flow")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.7)
        plt.hlines(0, 0, len(max_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(max_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(max_grads))
        plt.ylim(bottom=-0.001)
        plt.xlabel("Layers")
        plt.ylabel("Max gradient")
        plt.title("Gradient flow")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def profile_model(model, input_shape=(1, 3, 224, 224)):
        """性能分析"""
        from torch.profiler import profile, record_function, ProfilerActivity
        
        inputs = torch.randn(input_shape)
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True) as prof:
            with record_function("model_inference"):
                model(inputs)
        
        # 打印结果
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # 生成Chrome跟踪文件
        prof.export_chrome_trace("trace.json")
        
        return prof
```

## 9. 高级训练技巧 🎓

```python
class AdvancedTrainingTricks:
    """高级训练技巧"""
    
    @staticmethod
    def sam_optimizer(model, base_optimizer, rho=0.05):
        """Sharpness Aware Minimization (SAM)"""
        class SAM:
            def __init__(self, params, base_optimizer, rho=0.05):
                self.base_optimizer = base_optimizer
                self.rho = rho
                self.param_groups = self.base_optimizer.param_groups
            
            def first_step(self, zero_grad=False):
                grad_norm = self._grad_norm()
                for group in self.param_groups:
                    scale = self.rho / (grad_norm + 1e-12)
                    
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        e_w = p.grad * scale
                        p.add_(e_w)  # 爬到损失更高的地方
                        self.state[p]["e_w"] = e_w
                
                if zero_grad:
                    self.zero_grad()
            
            def second_step(self, zero_grad=False):
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        p.sub_(self.state[p]["e_w"])  # 恢复参数
                
                self.base_optimizer.step()
                
                if zero_grad:
                    self.zero_grad()
            
            def _grad_norm(self):
                norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
                )
                return norm
        
        return SAM(model.parameters(), base_optimizer, rho)
    
    @staticmethod
    def lookahead_optimizer(base_optimizer, k=5, alpha=0.5):
        """Lookahead优化器"""
        class Lookahead:
            def __init__(self, optimizer, k=5, alpha=0.5):
                self.optimizer = optimizer
                self.k = k
                self.alpha = alpha
                self.step_count = 0
                
                # 保存慢权重
                self.slow_weights = [[p.clone().detach() 
                                     for p in group['params']]
                                    for group in optimizer.param_groups]
            
            def step(self, closure=None):
                loss = self.optimizer.step(closure)
                self.step_count += 1
                
                if self.step_count % self.k == 0:
                    # 更新慢权重
                    for group_idx, group in enumerate(self.optimizer.param_groups):
                        for p_idx, p in enumerate(group['params']):
                            slow = self.slow_weights[group_idx][p_idx]
                            slow.add_(self.alpha * (p.data - slow))
                            p.data.copy_(slow)
                
                return loss
        
        return Lookahead(base_optimizer, k, alpha)
    
    @staticmethod
    def progressive_resizing(initial_size=64, final_size=224, stages=4):
        """渐进式调整图像大小"""
        sizes = np.linspace(initial_size, final_size, stages).astype(int)
        
        transforms_list = []
        for size in sizes:
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            transforms_list.append(transform)
        
        return transforms_list
```

## 10. 实战案例 💼

```python
class TrainingPipeline:
    """完整的训练管道"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 混合精度
        self.scaler = GradScaler()
        
        # EMA
        self.ema = None
        if config.get('use_ema', False):
            self.ema = self._create_ema()
        
        # 最佳模型
        self.best_acc = 0
        self.best_model_state = None
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['lr'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0)
            )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config['scheduler'] == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config['lr'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
    
    def _create_ema(self):
        """创建EMA模型"""
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        return ema_model
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixup/CutMix
            if self.config.get('mixup', False):
                inputs, targets_a, targets_b, lam = DataAugmentation.mixup(
                    inputs, targets, alpha=1.0
                )
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                outputs = self.model(inputs)
                
                if self.config.get('mixup', False):
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
            
            # 梯度缩放
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            # 优化器步骤
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 更新EMA
            if self.ema is not None:
                self._update_ema()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if not self.config.get('mixup', False):
                correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total if not self.config.get('mixup', False) else 0
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        model = self.ema if self.ema is not None else self.model
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        # 保存最佳模型
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model_state = copy.deepcopy(model.state_dict())
            print(f'Best model saved with accuracy: {acc:.2f}%')
        
        return avg_loss, acc
    
    def train(self):
        """完整训练流程"""
        print(f"Training on {self.device}")
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        print(f'Best validation accuracy: {self.best_acc:.2f}%')
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model
```

## 最佳实践总结 📋

```python
def training_best_practices():
    """训练最佳实践"""
    
    practices = {
        "初始化": [
            "使用合适的权重初始化（He, Xavier）",
            "BatchNorm后的层初始化为0",
            "残差连接的最后一层初始化为0",
            "检查初始损失是否合理"
        ],
        
        "学习率": [
            "使用学习率查找器找到最佳范围",
            "使用warmup避免初期不稳定",
            "余弦退火或OneCycle调度",
            "不同层使用不同学习率"
        ],
        
        "正则化": [
            "Dropout通常0.1-0.5",
            "权重衰减通常1e-4到1e-2",
            "标签平滑0.1效果好",
            "数据增强是最好的正则化"
        ],
        
        "优化技巧": [
            "混合精度训练加速50%+",
            "梯度累积模拟大batch",
            "梯度裁剪防止爆炸",
            "EMA提升测试性能"
        ],
        
        "调试": [
            "先在小数据集过拟合",
            "可视化梯度流",
            "监控激活值分布",
            "使用TensorBoard记录"
        ],
        
        "效率": [
            "num_workers=4*num_gpu",
            "pin_memory=True加速",
            "持久化workers减少开销",
            "使用torch.compile加速"
        ]
    }
    
    return practices

# 常见问题解决
troubleshooting = """
问题1: 损失不下降
- 检查学习率（太大或太小）
- 检查数据和标签
- 检查损失函数
- 尝试更简单的模型

问题2: 梯度爆炸/消失
- 使用梯度裁剪
- 检查初始化
- 使用BatchNorm
- 减少网络深度

问题3: 过拟合严重
- 增加数据增强
- 增加Dropout
- 减少模型容量
- 使用早停

问题4: 训练不稳定
- 减小学习率
- 使用梯度裁剪
- 增加批次大小
- 使用更稳定的优化器（如AdamW）

问题5: GPU内存不足
- 减小批次大小
- 使用梯度累积
- 使用混合精度
- 使用梯度检查点
"""

print("训练技巧指南加载完成！")
```

## 下一步学习
- [模型微调](finetuning.md) - 高效微调预训练模型
- [NLP模型](nlp_models.md) - 自然语言处理技术
- [模型部署](deployment.md) - 生产环境部署