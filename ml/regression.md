# 回归算法详解 📈

回归分析就像是在寻找数据点之间的"最佳拟合线"，帮助我们预测连续的数值。

## 1. 线性回归 (Linear Regression)

### 核心思想
想象你在散点图上画一条直线，让所有点到这条线的距离之和最小。这就是线性回归！

### 数学原理
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

其中：
- y: 目标变量
- x: 特征变量
- β: 回归系数
- ε: 误差项
```

### 完整代码实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 生成示例数据
np.random.seed(42)
n_samples = 1000

# 创建特征
X = np.random.randn(n_samples, 3)
# 创建目标变量（有线性关系）
true_coefficients = [3.5, -2.1, 1.8]
y = X @ true_coefficients + np.random.randn(n_samples) * 0.5

# 转换为DataFrame便于查看
df = pd.DataFrame(X, columns=['特征1', '特征2', '特征3'])
df['目标值'] = y

print("数据集前5行：")
print(df.head())
print(f"\n数据集形状：{df.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估模型
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n模型评估结果：")
print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

print("\n学习到的系数：")
print(f"真实系数: {true_coefficients}")
print(f"学习系数: {model.coef_.tolist()}")
print(f"截距项: {model.intercept_:.4f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 预测值 vs 真实值
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('真实值')
axes[0, 0].set_ylabel('预测值')
axes[0, 0].set_title('预测值 vs 真实值')

# 2. 残差图
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('预测值')
axes[0, 1].set_ylabel('残差')
axes[0, 1].set_title('残差图')

# 3. 残差分布
axes[1, 0].hist(residuals, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('残差分布')

# 4. 特征重要性
feature_importance = pd.DataFrame({
    '特征': ['特征1', '特征2', '特征3'],
    '系数': model.coef_
})
axes[1, 1].bar(feature_importance['特征'], 
               feature_importance['系数'])
axes[1, 1].set_xlabel('特征')
axes[1, 1].set_ylabel('系数值')
axes[1, 1].set_title('特征重要性')

plt.tight_layout()
plt.show()
```

### 实战案例：房价预测

```python
# 使用真实场景的特征名
def create_house_price_data(n_samples=1000):
    """创建模拟的房价数据"""
    np.random.seed(42)
    
    # 生成特征
    area = np.random.uniform(50, 300, n_samples)  # 面积(平方米)
    rooms = np.random.randint(1, 6, n_samples)    # 房间数
    age = np.random.uniform(0, 50, n_samples)     # 房龄
    distance_to_center = np.random.uniform(1, 30, n_samples)  # 到市中心距离(km)
    
    # 计算房价（带有真实的逻辑关系）
    price = (
        area * 15000 +                    # 每平米15000元
        rooms * 50000 +                    # 每个房间加5万
        age * (-5000) +                    # 每年折旧5000
        distance_to_center * (-10000) +   # 每公里离市中心减1万
        np.random.randn(n_samples) * 50000  # 随机噪声
    )
    
    # 确保价格为正
    price = np.maximum(price, 100000)
    
    return pd.DataFrame({
        '面积': area,
        '房间数': rooms,
        '房龄': age,
        '到市中心距离': distance_to_center,
        '房价': price
    })

# 创建数据
house_data = create_house_price_data()
print("房价数据统计：")
print(house_data.describe())

# 准备特征和目标
X = house_data[['面积', '房间数', '房龄', '到市中心距离']]
y = house_data['房价']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 创建结果DataFrame
results = pd.DataFrame({
    '真实房价': y_test.values,
    '预测房价': y_pred,
    '误差': y_test.values - y_pred,
    '误差百分比': np.abs((y_test.values - y_pred) / y_test.values * 100)
})

print("\n预测结果示例（前10个）：")
print(results.head(10))

print(f"\n平均绝对误差: {np.mean(np.abs(results['误差'])):.2f}元")
print(f"平均误差百分比: {results['误差百分比'].mean():.2f}%")

# 特征影响分析
feature_impact = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_,
    '影响说明': [
        f'每平米影响{model.coef_[0]:.2f}元',
        f'每个房间影响{model.coef_[1]:.2f}元',
        f'每年房龄影响{model.coef_[2]:.2f}元',
        f'每公里距离影响{model.coef_[3]:.2f}元'
    ]
})
print("\n特征影响分析：")
print(feature_impact)
```

## 2. Lasso回归 (L1正则化)

### 核心思想
Lasso像是一个"节俭"的线性回归，它会把不重要的特征系数直接压缩到0，实现特征选择。

### 完整代码实现

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 创建一个有很多特征的数据集（包含无用特征）
np.random.seed(42)
n_samples = 500
n_features = 20
n_informative = 5  # 只有5个特征是有用的

# 生成数据
X = np.random.randn(n_samples, n_features)
# 只使用前5个特征生成y
true_coef = np.zeros(n_features)
true_coef[:n_informative] = [3, -2, 1.5, 4, -3.5]
y = X @ true_coef + np.random.randn(n_samples) * 0.1

# 特征标准化（Lasso对特征尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 使用交叉验证选择最佳alpha
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

print(f"最佳alpha值: {lasso_cv.alpha_:.4f}")

# 训练最终模型
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
lasso.fit(X_train, y_train)

# 评估
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型性能:")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 特征选择效果
selected_features = np.where(np.abs(lasso.coef_) > 0.01)[0]
print(f"\n特征选择结果:")
print(f"原始特征数: {n_features}")
print(f"选中特征数: {len(selected_features)}")
print(f"选中的特征索引: {selected_features.tolist()}")

# 可视化系数
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 真实系数
axes[0].bar(range(n_features), true_coef)
axes[0].set_title('真实系数')
axes[0].set_xlabel('特征索引')
axes[0].set_ylabel('系数值')

# 2. Lasso系数
axes[1].bar(range(n_features), lasso.coef_)
axes[1].set_title('Lasso学习的系数')
axes[1].set_xlabel('特征索引')
axes[1].set_ylabel('系数值')

# 3. 对比普通线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
axes[2].bar(range(n_features), lr.coef_)
axes[2].set_title('普通线性回归系数')
axes[2].set_xlabel('特征索引')
axes[2].set_ylabel('系数值')

plt.tight_layout()
plt.show()

# Lasso路径：展示不同alpha下的系数变化
alphas = np.logspace(-4, 1, 50)
coefs = []

for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(X_train, y_train)
    coefs.append(lasso_temp.coef_)

# 绘制Lasso路径
plt.figure(figsize=(10, 6))
for i in range(n_features):
    plt.plot(alphas, [coef[i] for coef in coefs], 
             label=f'特征{i}' if i < 5 else None)

plt.xscale('log')
plt.xlabel('Alpha (正则化强度)')
plt.ylabel('系数值')
plt.title('Lasso路径：系数随正则化强度的变化')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()
```

## 3. Ridge回归 (L2正则化)

### 核心思想
Ridge像是一个"谨慎"的线性回归，它会缩小所有系数但不会将它们压缩到0。

```python
from sklearn.linear_model import Ridge, RidgeCV

# 使用相同的数据
# Ridge回归
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Ridge最佳alpha: {ridge_cv.alpha_:.4f}")

ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train, y_train)

# 对比三种方法
models = {
    'Linear': LinearRegression(),
    'Lasso': Lasso(alpha=lasso_cv.alpha_, max_iter=10000),
    'Ridge': Ridge(alpha=ridge_cv.alpha_)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        '非零系数': np.sum(np.abs(model.coef_) > 0.01)
    }

# 创建对比表
comparison_df = pd.DataFrame(results).T
print("\n三种回归方法对比:")
print(comparison_df)

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 系数对比
models_coef = {
    'Linear': LinearRegression().fit(X_train, y_train).coef_,
    'Lasso': lasso.coef_,
    'Ridge': ridge.coef_
}

x_pos = np.arange(n_features)
width = 0.25

for idx, (name, coef) in enumerate(models_coef.items()):
    axes[0, 0].bar(x_pos + idx * width, coef, width, label=name)

axes[0, 0].set_xlabel('特征索引')
axes[0, 0].set_ylabel('系数值')
axes[0, 0].set_title('不同模型的系数对比')
axes[0, 0].legend()

# MSE对比
axes[0, 1].bar(results.keys(), [r['MSE'] for r in results.values()])
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('均方误差对比')

# R²对比
axes[1, 0].bar(results.keys(), [r['R²'] for r in results.values()])
axes[1, 0].set_ylabel('R²')
axes[1, 0].set_title('R²得分对比')

# 非零系数数量
axes[1, 1].bar(results.keys(), [r['非零系数'] for r in results.values()])
axes[1, 1].set_ylabel('非零系数数量')
axes[1, 1].set_title('特征选择效果')

plt.tight_layout()
plt.show()
```

## 4. 支持向量回归 (SVR)

### 核心思想
SVR在一个"管道"内拟合数据，只关心管道外的点，对异常值更鲁棒。

```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# 创建非线性数据
np.random.seed(42)
X_nonlinear = np.sort(5 * np.random.rand(100, 1), axis=0)
y_nonlinear = np.sin(X_nonlinear).ravel() + np.random.randn(100) * 0.1

# 尝试不同的核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svr_models = {}

plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels, 1):
    # 训练SVR
    if kernel == 'rbf':
        svr = SVR(kernel=kernel, C=100, gamma=0.1, epsilon=0.01)
    else:
        svr = SVR(kernel=kernel, C=100, epsilon=0.01)
    
    svr.fit(X_nonlinear, y_nonlinear)
    svr_models[kernel] = svr
    
    # 预测
    X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    y_pred = svr.predict(X_plot)
    
    # 可视化
    plt.subplot(2, 2, i)
    plt.scatter(X_nonlinear, y_nonlinear, alpha=0.5, label='数据点')
    plt.plot(X_plot, y_pred, 'r-', label=f'SVR ({kernel})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'SVR with {kernel} kernel')
    plt.legend()

plt.tight_layout()
plt.show()

# 使用网格搜索优化RBF核SVR
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.2]
}

svr_rbf = SVR(kernel='rbf')
grid_search = GridSearchCV(svr_rbf, param_grid, cv=5, 
                          scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_nonlinear, y_nonlinear)

print("SVR最佳参数:")
print(grid_search.best_params_)
print(f"最佳得分: {-grid_search.best_score_:.4f}")

# 使用最佳参数的模型
best_svr = grid_search.best_estimator_
X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
y_pred_best = best_svr.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_nonlinear, y_nonlinear, alpha=0.5, label='训练数据')
plt.plot(X_plot, y_pred_best, 'r-', label='优化后的SVR', linewidth=2)
plt.plot(X_plot, np.sin(X_plot), 'g--', label='真实函数', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('优化后的SVR拟合效果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 5. 实战项目：多算法对比

```python
def compare_regression_models(X, y, model_dict, test_size=0.2):
    """
    对比多个回归模型的性能
    """
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    results = []
    predictions = {}
    
    for name, model in model_dict.items():
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predictions[name] = y_test_pred
        
        # 评估
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            '模型': name,
            '训练MSE': train_mse,
            '测试MSE': test_mse,
            '训练R²': train_r2,
            '测试R²': test_r2,
            '过拟合度': train_mse - test_mse
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('测试R²', ascending=False)
    
    return results_df, predictions, y_test

# 创建示例数据
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, 
                       n_informative=10, noise=10, random_state=42)

# 定义模型
models = {
    'Linear': LinearRegression(),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Ridge': Ridge(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    'SVR_linear': SVR(kernel='linear', C=1.0),
    'SVR_rbf': SVR(kernel='rbf', C=100, gamma=0.01)
}

# 比较模型
results_df, predictions, y_test = compare_regression_models(X, y, models)

print("模型性能对比:")
print(results_df.to_string())

# 可视化结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (name, y_pred) in enumerate(predictions.items()):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    
    # 绘制预测vs真实
    ax.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'{name}\nR²={r2_score(y_test, y_pred):.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 绘制性能对比条形图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R²对比
axes[0].barh(results_df['模型'], results_df['测试R²'])
axes[0].set_xlabel('R² Score')
axes[0].set_title('模型R²得分对比')

# MSE对比
axes[1].barh(results_df['模型'], results_df['测试MSE'])
axes[1].set_xlabel('MSE')
axes[1].set_title('模型MSE对比')

plt.tight_layout()
plt.show()
```

## 最佳实践建议

### 1. 算法选择指南
- **线性回归**：数据线性关系明显，特征较少
- **Lasso**：特征很多，需要特征选择
- **Ridge**：特征之间有多重共线性
- **SVR**：数据有异常值，非线性关系

### 2. 调参技巧
```python
# 通用的超参数调优模板
from sklearn.model_selection import RandomizedSearchCV

def tune_model(model, param_distributions, X_train, y_train):
    """
    使用随机搜索调优模型
    """
    random_search = RandomizedSearchCV(
        model, 
        param_distributions,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳得分: {-random_search.best_score_:.4f}")
    
    return random_search.best_estimator_
```

### 3. 特征工程提示
- 标准化特征（特别是使用正则化时）
- 处理缺失值
- 创建交互特征
- 使用多项式特征扩展

### 4. 避免过拟合
- 使用交叉验证
- 增加正则化
- 减少特征数量
- 增加训练数据

## 下一步学习
- [分类算法](classification.md) - 学习如何预测类别
- [集成学习](ensemble.md) - 了解更强大的算法
- [特征工程](feature_engineering.md) - 提升模型性能的关键