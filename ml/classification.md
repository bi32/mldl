# 分类算法详解 🎯

分类就像给事物贴标签，比如判断邮件是否垃圾邮件、诊断疾病、识别图片中的物体。

## 1. 逻辑回归 (Logistic Regression)

### 核心思想
虽然叫"回归"，但它是分类算法！它用S型曲线（Sigmoid）把线性回归的输出压缩到0-1之间，表示概率。

### 完整代码实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           roc_curve, classification_report)

# 创建二分类数据集
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],  # 类别不平衡
    random_state=42
)

# 转换为DataFrame
feature_names = [f'特征_{i+1}' for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['标签'] = y

print("数据集信息:")
print(f"样本数: {len(df)}")
print(f"特征数: {len(feature_names)}")
print(f"类别分布:\n{df['标签'].value_counts()}")

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 训练逻辑回归模型
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# 预测
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# 评估模型
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """全面评估分类模型"""
    metrics = {
        '准确率': accuracy_score(y_true, y_pred),
        '精确率': precision_score(y_true, y_pred),
        '召回率': recall_score(y_true, y_pred),
        'F1分数': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

metrics = evaluate_model(y_test, y_pred, y_pred_proba)
print("\n模型评估指标:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('混淆矩阵')
axes[0, 0].set_xlabel('预测值')
axes[0, 0].set_ylabel('真实值')

# 2. ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, 'b-', label=f'AUC = {metrics["AUC"]:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'r--')
axes[0, 1].set_xlabel('假阳性率')
axes[0, 1].set_ylabel('真阳性率')
axes[0, 1].set_title('ROC曲线')
axes[0, 1].legend()

# 3. 特征重要性
feature_importance = pd.DataFrame({
    '特征': feature_names,
    '系数': log_reg.coef_[0]
}).sort_values('系数', key=abs, ascending=False).head(10)

axes[0, 2].barh(range(10), feature_importance['系数'].values)
axes[0, 2].set_yticks(range(10))
axes[0, 2].set_yticklabels(feature_importance['特征'].values)
axes[0, 2].set_xlabel('系数值')
axes[0, 2].set_title('Top 10 重要特征')

# 4. 预测概率分布
axes[1, 0].hist(y_pred_proba[y_test == 0], alpha=0.5, 
                label='类别0', bins=20, color='blue')
axes[1, 0].hist(y_pred_proba[y_test == 1], alpha=0.5, 
                label='类别1', bins=20, color='red')
axes[1, 0].set_xlabel('预测概率')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('预测概率分布')
axes[1, 0].legend()

# 5. 阈值对性能的影响
thresholds_test = np.linspace(0, 1, 100)
precisions = []
recalls = []

for threshold in thresholds_test:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    precisions.append(precision_score(y_test, y_pred_threshold, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_threshold, zero_division=0))

axes[1, 1].plot(thresholds_test, precisions, label='精确率')
axes[1, 1].plot(thresholds_test, recalls, label='召回率')
axes[1, 1].set_xlabel('阈值')
axes[1, 1].set_ylabel('分数')
axes[1, 1].set_title('阈值对性能的影响')
axes[1, 1].legend()

# 6. 学习曲线
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(random_state=42, max_iter=1000),
    X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

axes[1, 2].plot(train_sizes, np.mean(train_scores, axis=1), 
                'o-', label='训练分数')
axes[1, 2].plot(train_sizes, np.mean(val_scores, axis=1), 
                'o-', label='验证分数')
axes[1, 2].set_xlabel('训练样本数')
axes[1, 2].set_ylabel('准确率')
axes[1, 2].set_title('学习曲线')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# 多分类逻辑回归
print("\n=== 多分类逻辑回归 ===")

# 创建多分类数据
X_multi, y_multi = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_classes=4,
    random_state=42
)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=42
)

# 训练多分类模型
log_reg_multi = LogisticRegression(
    multi_class='multinomial',  # 使用softmax
    solver='lbfgs',
    random_state=42,
    max_iter=1000
)
log_reg_multi.fit(X_train_m, y_train_m)

# 预测
y_pred_m = log_reg_multi.predict(X_test_m)
y_pred_proba_m = log_reg_multi.predict_proba(X_test_m)

# 评估
print(f"多分类准确率: {accuracy_score(y_test_m, y_pred_m):.4f}")
print("\n分类报告:")
print(classification_report(y_test_m, y_pred_m, 
                           target_names=[f'类别{i}' for i in range(4)]))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm_multi = confusion_matrix(y_test_m, y_pred_m)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues')
plt.title('多分类混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()
```

## 2. 支持向量机 (SVM)

### 核心思想
SVM寻找一个最优超平面，使得两类数据点之间的间隔最大化。就像在两群人之间画一条线，让线两边的空间最大。

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 创建非线性可分数据
from sklearn.datasets import make_moons, make_circles

# 生成月亮形数据
X_moon, y_moon = make_moons(n_samples=500, noise=0.15, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_moon_scaled = scaler.fit_transform(X_moon)

# 训练不同核函数的SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, kernel in enumerate(kernels):
    ax = axes[idx // 2, idx % 2]
    
    # 训练SVM
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3, random_state=42)
    else:
        svm = SVC(kernel=kernel, random_state=42)
    
    svm.fit(X_moon_scaled, y_moon)
    
    # 创建网格用于决策边界
    h = 0.02
    x_min, x_max = X_moon_scaled[:, 0].min() - 1, X_moon_scaled[:, 0].max() + 1
    y_min, y_max = X_moon_scaled[:, 1].min() - 1, X_moon_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X_moon_scaled[:, 0], X_moon_scaled[:, 1], 
               c=y_moon, cmap=plt.cm.RdYlBu, edgecolor='black')
    ax.set_title(f'SVM ({kernel} kernel)')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')

plt.tight_layout()
plt.show()

# SVM参数调优
print("=== SVM参数调优 ===")

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# 网格搜索
svm_base = SVC(random_state=42)
grid_search = GridSearchCV(
    svm_base, param_grid, cv=5, 
    scoring='accuracy', n_jobs=-1
)

# 使用更大的数据集
X_large, y_large = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)
X_large_scaled = StandardScaler().fit_transform(X_large)

X_train, X_test, y_train, y_test = train_test_split(
    X_large_scaled, y_large, test_size=0.2, random_state=42
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# SVM的支持向量
print(f"\n支持向量数量: {len(best_svm.support_)}")
print(f"总训练样本数: {len(X_train)}")
print(f"支持向量比例: {len(best_svm.support_) / len(X_train):.2%}")
```

## 3. 朴素贝叶斯 (Naive Bayes)

### 核心思想
基于贝叶斯定理，假设特征之间相互独立。就像垃圾邮件过滤器，通过统计词频来判断邮件类别。

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. 高斯朴素贝叶斯（连续特征）
print("=== 高斯朴素贝叶斯 ===")

# 使用之前的数据
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred_gnb):.4f}")

# 2. 多项式朴素贝叶斯（文本分类）
print("\n=== 文本分类示例 ===")

# 创建示例文本数据
texts = [
    "免费赢取iPhone，点击这里！",
    "恭喜您中奖了，请提供银行账号",
    "会议安排在明天下午3点",
    "项目进度报告已发送",
    "限时优惠，买一送一",
    "请查收附件中的合同",
    "您的包裹已发货",
    "紧急！账户即将到期",
    "季度财务报表分析",
    "团队建设活动通知"
]

labels = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]  # 1: 垃圾邮件, 0: 正常邮件

# 扩展数据集
texts_extended = texts * 50  # 重复以增加样本
labels_extended = labels * 50
# 添加随机性
np.random.seed(42)
shuffle_idx = np.random.permutation(len(texts_extended))
texts_extended = [texts_extended[i] for i in shuffle_idx]
labels_extended = [labels_extended[i] for i in shuffle_idx]

# 文本向量化
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts_extended)

# 划分数据
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, labels_extended, test_size=0.2, random_state=42
)

# 训练多项式朴素贝叶斯
mnb = MultinomialNB()
mnb.fit(X_train_text, y_train_text)

# 预测
y_pred_text = mnb.predict(X_test_text)
print(f"文本分类准确率: {accuracy_score(y_test_text, y_pred_text):.4f}")

# 测试新文本
new_texts = [
    "恭喜获得百万大奖",
    "明天的会议议程",
    "限时折扣活动"
]

new_texts_vec = vectorizer.transform(new_texts)
predictions = mnb.predict(new_texts_vec)
probabilities = mnb.predict_proba(new_texts_vec)

print("\n新文本预测结果:")
for text, pred, prob in zip(new_texts, predictions, probabilities):
    label = "垃圾邮件" if pred == 1 else "正常邮件"
    confidence = max(prob) * 100
    print(f"文本: {text}")
    print(f"  预测: {label} (置信度: {confidence:.1f}%)")

# 3. 比较不同的朴素贝叶斯
print("\n=== 朴素贝叶斯变体比较 ===")

# 创建二值化特征
X_binary = (X > 0).astype(int)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y, test_size=0.2, random_state=42
)

nb_models = {
    'Gaussian': GaussianNB(),
    'Bernoulli': BernoulliNB()
}

nb_results = {}
for name, model in nb_models.items():
    if name == 'Gaussian':
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    else:
        model.fit(X_train_b, y_train_b)
        score = model.score(X_test_b, y_test_b)
    
    nb_results[name] = score
    print(f"{name} NB 准确率: {score:.4f}")

# 可视化朴素贝叶斯决策边界（2D示例）
from sklearn.decomposition import PCA

# 降维到2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, model) in enumerate(nb_models.items()):
    if name == 'Gaussian':
        model.fit(X_2d, y)
        X_plot = X_2d
    else:
        X_2d_binary = (X_2d > 0).astype(int)
        model.fit(X_2d_binary, y)
        X_plot = X_2d_binary
    
    # 创建网格
    h = 0.02
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    if name == 'Bernoulli':
        Z = model.predict((np.c_[xx.ravel(), yy.ravel()] > 0).astype(int))
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    axes[idx].scatter(X_plot[:, 0], X_plot[:, 1], 
                      c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    axes[idx].set_title(f'{name} Naive Bayes')
    axes[idx].set_xlabel('PC1')
    axes[idx].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```

## 4. 实战项目：信用卡欺诈检测

```python
# 创建模拟的信用卡交易数据
def create_credit_card_data(n_samples=10000):
    """创建信用卡欺诈检测数据集"""
    np.random.seed(42)
    
    # 正常交易 (95%)
    n_normal = int(n_samples * 0.95)
    normal_amount = np.random.lognormal(3, 1.5, n_normal)
    normal_hour = np.random.normal(14, 6, n_normal) % 24
    normal_merchant = np.random.choice(100, n_normal)
    normal_country = np.random.choice(50, n_normal, p=np.concatenate([
        np.array([0.8]), np.ones(49) * 0.2/49
    ]))
    
    # 欺诈交易 (5%)
    n_fraud = n_samples - n_normal
    fraud_amount = np.concatenate([
        np.random.lognormal(5, 1, n_fraud//2),  # 大额欺诈
        np.random.uniform(1, 50, n_fraud//2)    # 小额测试
    ])
    fraud_hour = np.random.uniform(0, 24, n_fraud)  # 任意时间
    fraud_merchant = np.random.choice(100, n_fraud)
    fraud_country = np.random.choice(50, n_fraud)  # 随机国家
    
    # 合并数据
    features = np.vstack([
        np.column_stack([normal_amount, normal_hour, 
                        normal_merchant, normal_country]),
        np.column_stack([fraud_amount, fraud_hour, 
                        fraud_merchant, fraud_country])
    ])
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # 打乱数据
    shuffle_idx = np.random.permutation(n_samples)
    
    return features[shuffle_idx], labels[shuffle_idx].astype(int)

# 创建数据
X_credit, y_credit = create_credit_card_data()

# 添加更多特征
X_credit_enhanced = np.column_stack([
    X_credit,
    X_credit[:, 0] ** 2,  # 金额平方
    np.sin(X_credit[:, 1] * np.pi / 12),  # 时间周期性
    X_credit[:, 0] * X_credit[:, 1],  # 交互特征
])

print("信用卡数据集:")
print(f"样本数: {len(X_credit_enhanced)}")
print(f"特征数: {X_credit_enhanced.shape[1]}")
print(f"欺诈比例: {y_credit.mean():.2%}")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_credit_enhanced, y_credit, test_size=0.2, 
    stratify=y_credit, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练多个模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
plt.figure(figsize=(15, 10))

for idx, (name, model) in enumerate(models.items(), 1):
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 评估
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['模型'] = name
    results.append(metrics)
    
    # PR曲线（对不平衡数据更有意义）
    plt.subplot(2, 3, idx)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, lw=2)
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'{name}\nAUC: {metrics["AUC"]:.3f}')
    plt.grid(True, alpha=0.3)

# 比较表
plt.subplot(2, 3, 5)
results_df = pd.DataFrame(results)
results_df = results_df.set_index('模型')

# 绘制指标对比
metrics_to_plot = ['准确率', '精确率', '召回率', 'F1分数']
results_df[metrics_to_plot].plot(kind='bar', ax=plt.gca())
plt.title('模型性能对比')
plt.xlabel('模型')
plt.ylabel('分数')
plt.legend(loc='lower right')
plt.xticks(rotation=45)

# 特征重要性（Random Forest）
plt.subplot(2, 3, 6)
rf_model = models['Random Forest']
feature_names = ['金额', '时间', '商户', '国家', '金额²', 
                 '时间周期', '金额×时间']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:5]

plt.bar(range(5), importances[indices])
plt.xticks(range(5), [feature_names[i] for i in indices], rotation=45)
plt.title('Top 5 重要特征')
plt.xlabel('特征')
plt.ylabel('重要性')

plt.tight_layout()
plt.show()

print("\n模型性能总结:")
print(results_df.round(4))

# 成本敏感学习
print("\n=== 成本敏感学习 ===")

# 假设漏检欺诈的成本是误报的10倍
class_weight = {0: 1, 1: 10}

weighted_models = {
    'Weighted LR': LogisticRegression(class_weight=class_weight, random_state=42),
    'Weighted SVM': SVC(class_weight=class_weight, probability=True, random_state=42),
    'Weighted RF': RandomForestClassifier(class_weight=class_weight, 
                                          n_estimators=100, random_state=42)
}

for name, model in weighted_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n{name}:")
    print(f"  精确率: {precision_score(y_test, y_pred):.4f}")
    print(f"  召回率: {recall_score(y_test, y_pred):.4f}")
    
    # 计算总成本
    cm = confusion_matrix(y_test, y_pred)
    fn_cost = cm[1, 0] * 1000  # 漏检欺诈成本
    fp_cost = cm[0, 1] * 100   # 误报成本
    total_cost = fn_cost + fp_cost
    print(f"  总成本: ${total_cost:,.0f}")
```

## 5. 高级技巧：集成分类器

```python
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

# 创建集成分类器
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='soft'  # 使用概率投票
)

# 训练集成模型
ensemble.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble.predict(X_test_scaled)

print("集成分类器性能:")
print(f"准确率: {accuracy_score(y_test, y_pred_ensemble):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_ensemble):.4f}")

# 概率校准
print("\n=== 概率校准 ===")

# 校准SVM的概率输出
svm_calibrated = CalibratedClassifierCV(
    SVC(random_state=42), 
    method='sigmoid'
)
svm_calibrated.fit(X_train_scaled, y_train)

# 比较校准前后
svm_uncalibrated = SVC(probability=True, random_state=42)
svm_uncalibrated.fit(X_train_scaled, y_train)

# 获取概率
prob_uncalibrated = svm_uncalibrated.predict_proba(X_test_scaled)[:, 1]
prob_calibrated = svm_calibrated.predict_proba(X_test_scaled)[:, 1]

# 可视化校准效果
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 校准曲线
for probs, name in [(prob_uncalibrated, '未校准'), 
                     (prob_calibrated, '已校准')]:
    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    axes[0].plot(mean_pred, fraction_pos, 'o-', label=name)

axes[0].plot([0, 1], [0, 1], 'k--', label='完美校准')
axes[0].set_xlabel('平均预测概率')
axes[0].set_ylabel('正例比例')
axes[0].set_title('概率校准曲线')
axes[0].legend()

# 概率分布
axes[1].hist(prob_uncalibrated, bins=20, alpha=0.5, 
             label='未校准', density=True)
axes[1].hist(prob_calibrated, bins=20, alpha=0.5, 
             label='已校准', density=True)
axes[1].set_xlabel('预测概率')
axes[1].set_ylabel('密度')
axes[1].set_title('概率分布')
axes[1].legend()

plt.tight_layout()
plt.show()
```

## 最佳实践建议

### 1. 算法选择指南
- **逻辑回归**：线性可分、需要概率输出、可解释性要求高
- **SVM**：非线性问题、高维数据、样本量中等
- **朴素贝叶斯**：文本分类、特征独立、训练数据少

### 2. 处理不平衡数据
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# SMOTE过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"原始训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后类别分布: {np.bincount(y_resampled)}")
```

### 3. 特征工程
- 特征缩放（标准化/归一化）
- 特征选择（互信息、卡方检验）
- 特征创造（多项式特征、交互特征）

### 4. 评估指标选择
- 平衡数据：准确率
- 不平衡数据：F1分数、AUC-ROC
- 成本敏感：自定义成本函数

## 下一步学习
- [集成学习](ensemble.md) - 组合多个模型获得更好性能
- [超参数调优](hyperparameter_tuning.md) - 优化模型参数
- [特征工程](feature_engineering.md) - 提升模型输入质量