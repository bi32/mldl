# Vision Transformer (ViT) 完全指南 👁️

Transformer在NLP领域取得巨大成功后，终于来到了计算机视觉领域。ViT证明了：图像也可以是一个序列！

## 1. ViT - 开创性工作 🎨

### 核心思想
将图像切分成小块（patches），把每个块当作一个token，然后用标准的Transformer处理。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768,
                 depth=12, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Patch embedding
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, dim // heads, mlp_dim, dropout)
              for _ in range(depth)]
        )
        
        self.norm = nn.LayerNorm(dim)
        self.to_cls_token = nn.Identity()
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        # Patch embedding
        x = self.patch_to_embedding(img)
        b, n, _ = x.shape
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Classification
        cls_token = x[:, 0]
        return self.mlp_head(cls_token)

# 创建ViT模型
def create_vit(variant='base'):
    """创建不同规模的ViT模型"""
    configs = {
        'tiny': dict(dim=192, depth=12, heads=3, mlp_dim=768),
        'small': dict(dim=384, depth=12, heads=6, mlp_dim=1536),
        'base': dict(dim=768, depth=12, heads=12, mlp_dim=3072),
        'large': dict(dim=1024, depth=24, heads=16, mlp_dim=4096),
        'huge': dict(dim=1280, depth=32, heads=16, mlp_dim=5120),
    }
    
    return VisionTransformer(**configs[variant])

# 测试
model = create_vit('base')
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"输出形状: {output.shape}")
```

## 2. CLIP - 视觉-语言对齐 🔗

### 核心思想
通过对比学习同时训练视觉和文本编码器，实现零样本图像分类。

```python
import clip  # pip install git+https://github.com/openai/CLIP.git

class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # 投影层
        self.vision_proj = nn.Linear(vision_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_image(self, image):
        features = self.vision_encoder(image)
        features = self.vision_proj(features)
        return F.normalize(features, dim=-1)
    
    def encode_text(self, text):
        features = self.text_encoder(text)
        features = self.text_proj(features)
        return F.normalize(features, dim=-1)
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

# CLIP训练
def train_clip(model, dataloader, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            images, texts = batch
            
            # 前向传播
            logits_per_image, logits_per_text = model(images, texts)
            
            # 对比学习损失
            batch_size = images.shape[0]
            labels = torch.arange(batch_size).to(images.device)
            
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 零样本分类
def zero_shot_classification(model, image, text_prompts):
    """使用CLIP进行零样本分类"""
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = torch.stack([
            model.encode_text(prompt) for prompt in text_prompts
        ])
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    return similarity
```

## 3. MAE - 自监督预训练 🎭

### 核心思想
遮盖图像的大部分patches（75%），让模型重建被遮盖的部分。

```python
class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
    
    def random_masking(self, x, mask_ratio):
        """随机遮盖patches"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # 生成随机噪声
        noise = torch.rand(N, L, device=x.device)
        
        # 排序获取要保留的索引
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留未遮盖的patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                               index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 生成mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, imgs):
        # Encoder
        latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
        
        # Decoder
        pred = self.decoder(latent, ids_restore)
        
        # 计算损失（只在被遮盖的部分）
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask
    
    def patchify(self, imgs):
        """将图像转换为patches"""
        p = self.encoder.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
```

## 4. DINO - 自蒸馏 🦕

```python
class DINO(nn.Module):
    def __init__(self, student, teacher, center_momentum=0.9):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.center = nn.Parameter(torch.zeros(1, student.output_dim))
        self.center_momentum = center_momentum
        
        # Teacher使用EMA更新
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        """EMA更新teacher网络"""
        for param_s, param_t in zip(self.student.parameters(), 
                                    self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """更新center"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)
    
    def forward(self, images):
        # 多尺度增强
        student_views = images[:2]  # 2个global views
        teacher_views = images[2:]  # 多个local views
        
        # Student前向
        student_output = torch.cat([
            self.student(view) for view in student_views
        ])
        
        # Teacher前向
        with torch.no_grad():
            teacher_output = torch.cat([
                self.teacher(view) for view in teacher_views
            ])
            teacher_output = F.softmax((teacher_output - self.center) / 0.04, dim=-1)
        
        # 计算损失
        student_output = student_output / 0.1
        student_log_sm = F.log_softmax(student_output, dim=-1)
        
        loss = -torch.mean(torch.sum(teacher_output * student_log_sm, dim=-1))
        
        self.update_center(teacher_output)
        
        return loss
```

## 5. 实战应用

```python
# 使用预训练ViT
from transformers import ViTForImageClassification, ViTImageProcessor

def use_pretrained_vit():
    # 加载模型和处理器
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # 处理图像
    from PIL import Image
    image = Image.open("test.jpg")
    inputs = processor(images=image, return_tensors="pt")
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
    
    print(f"预测类别: {model.config.id2label[predicted_class]}")

# 微调ViT
def finetune_vit(train_dataset, val_dataset, num_classes=10):
    from transformers import TrainingArguments, Trainer
    
    # 加载预训练模型
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./vit-finetuned',
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # 训练
    trainer.train()
    
    return model

# 可视化注意力
def visualize_attention(model, image):
    """可视化ViT的注意力图"""
    import matplotlib.pyplot as plt
    
    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        outputs = model(image, output_attentions=True)
        attentions = outputs.attentions  # List of attention matrices
    
    # 取最后一层的注意力
    attention = attentions[-1]  # [batch, heads, seq, seq]
    attention = attention.mean(dim=1)  # 平均所有heads
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原图
    axes[0].imshow(image[0].permute(1, 2, 0))
    axes[0].set_title('原图')
    axes[0].axis('off')
    
    # 注意力图
    axes[1].imshow(attention[0, 0, 1:].reshape(14, 14), cmap='hot')
    axes[1].set_title('注意力图')
    axes[1].axis('off')
    
    plt.show()
```

## 6. 性能优化技巧

```python
# Flash Attention
class FlashMultiHeadAttention(nn.Module):
    """使用Flash Attention加速"""
    def __init__(self, dim, heads=8):
        super().__init__()
        from flash_attn import flash_attn_func
        self.flash_attn = flash_attn_func
        # ... 其他初始化
    
    def forward(self, x):
        # 使用Flash Attention
        out = self.flash_attn(q, k, v)
        return out

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, dataloader):
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters())
    
    for batch in dataloader:
        with autocast():
            loss = model(batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Token Pruning
def token_pruning(x, keep_ratio=0.5):
    """动态剪枝不重要的tokens"""
    # 计算token重要性
    cls_attn = x[:, 0, 1:]  # CLS token对其他tokens的注意力
    
    # 保留最重要的tokens
    _, idx = torch.topk(cls_attn, int(keep_ratio * cls_attn.size(-1)))
    
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
```

## 模型选择指南

| 模型 | 参数量 | 特点 | 适用场景 |
|-----|--------|------|----------|
| ViT-B/16 | 86M | 标准ViT | 通用图像分类 |
| ViT-L/14 | 304M | 大模型 | 高精度需求 |
| DeiT | 86M | 数据高效 | 小数据集 |
| Swin Transformer | 88M | 层级结构 | 检测、分割 |
| CLIP | 428M | 多模态 | 零样本学习 |
| MAE | 86M | 自监督 | 预训练 |

## 最佳实践

1. **数据增强很重要**：ViT需要大量数据，强增强必不可少
2. **预训练权重**：尽量使用预训练模型
3. **位置编码**：可学习位置编码通常更好
4. **注意力可视化**：用于模型解释
5. **效率优化**：考虑Token pruning、Flash Attention

## 下一步学习
- [目标检测](object_detection.md) - YOLO系列和DETR
- [PyTorch部署](deployment.md) - 模型优化和部署
- [NLP模型](nlp_models.md) - BERT、GPT等