# 机器学习与深度学习实战手册

一本面向实践的机器学习与深度学习完整指南，包含理论讲解、代码实现和工程部署。

## 📚 内容结构

### 🤖 [机器学习篇](ml/)
- **回归算法**：线性回归、Lasso、Ridge、SVR
- **分类算法**：逻辑回归、SVM、朴素贝叶斯
- **集成学习**：XGBoost、LightGBM、CatBoost
- **超参数调优**：GridSearchCV、RandomizedSearchCV、贝叶斯优化
- **特征工程**：特征选择、特征提取、特征变换
- **模型评估**：交叉验证、评估指标、可视化

### 🧠 [深度学习篇](dl/)
- **CNN架构**：VGG、ResNet、EfficientNet、MobileNet
- **Vision Transformer**：ViT、CLIP、MAE、DINO
- **目标检测**：YOLO系列、Faster R-CNN、DETR
- **语义分割**：U-Net、DeepLab、Mask R-CNN
- **NLP模型**：BERT、GPT、T5、Transformer
- **生成模型**：GAN、VAE、Diffusion Models
- **PyTorch部署**：ONNX、TorchScript、模型优化

## 🚀 快速开始

### 环境配置
```bash
# 创建虚拟环境
python -m venv ml_dl_env
source ml_dl_env/bin/activate  # Linux/Mac
# ml_dl_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行示例
```python
# 机器学习示例
from ml.models import XGBoostClassifier
model = XGBoostClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 深度学习示例
from dl.models import ResNet50
model = ResNet50(num_classes=10)
model.train(train_loader, val_loader)
```

## 📖 章节导览

### 机器学习路线
1. [基础概念与环境搭建](ml/basics.md)
2. [回归算法详解](ml/regression.md)
3. [分类算法详解](ml/classification.md)
4. [集成学习方法](ml/ensemble.md)
5. [超参数调优技巧](ml/hyperparameter_tuning.md)
6. [特征工程实战](ml/feature_engineering.md)
7. [模型评估与选择](ml/evaluation.md)

### 深度学习路线
1. [深度学习基础](dl/basics.md)
2. [CNN架构演进](dl/cnn_architectures.md)
3. [Vision Transformer详解](dl/vision_transformer.md)
4. [目标检测算法](dl/object_detection.md)
5. [NLP模型架构](dl/nlp_models.md)
6. [模型训练技巧](dl/training_tricks.md)
7. [PyTorch部署实战](dl/deployment.md)

## 💻 代码示例

所有代码均经过测试，可直接运行。每个算法都包含：
- 完整的代码实现
- 详细的注释说明
- 实际应用案例
- 性能优化技巧

## 🎯 适合人群

- 机器学习初学者
- 想要深入理解算法原理的工程师
- 准备面试的求职者
- 需要实战经验的研究人员

## 📊 项目特色

- **理论与实践结合**：每个算法都有数学原理和代码实现
- **循序渐进**：从基础到高级，逐步深入
- **实战导向**：包含大量真实案例和最佳实践
- **性能优化**：提供生产环境的优化建议
- **持续更新**：跟踪最新的研究进展和工业应用

## 🔧 技术栈

- **Python 3.8+**
- **机器学习**：scikit-learn, XGBoost, LightGBM, CatBoost
- **深度学习**：PyTorch, torchvision, transformers
- **数据处理**：NumPy, Pandas, Polars
- **可视化**：Matplotlib, Seaborn, Plotly
- **部署**：ONNX, TorchScript, Docker

## 📈 学习建议

1. **循序渐进**：先掌握机器学习基础，再学习深度学习
2. **动手实践**：每个算法都要亲自实现一遍
3. **项目驱动**：通过实际项目巩固知识
4. **保持更新**：关注领域最新进展

## 🤝 贡献指南

欢迎提交问题和改进建议，让这本手册更加完善。

## 📝 License

MIT License

---

开始你的机器学习之旅：[进入机器学习篇](ml/) | [进入深度学习篇](dl/)