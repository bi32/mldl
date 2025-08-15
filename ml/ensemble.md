# 集成学习方法详解 🌲

集成学习就像组建一个专家团队，每个专家有自己的专长，通过投票或加权的方式做出最终决策。

## 1. XGBoost - 竞赛之王 👑

### 核心思想
XGBoost是"极限梯度提升"，通过不断添加新树来纠正之前树的错误，就像不断请教新老师来补充知识盲点。

### 完整代码实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 安装命令：pip install xgboost

# 创建回归数据集
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, 
                               n_informative=15, noise=10, random_state=42)

# 创建分类数据集
from sklearn.datasets import make_classification
X_clf, y_clf = make_classification(n_samples=1000, n_features=20,
                                   n_informative=15, n_redundant=5,
                                   n_classes=3, random_state=42)

# XGBoost回归示例
print("=== XGBoost 回归 ===")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 基础模型
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

# 训练模型
xgb_reg.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_test_reg, y_test_reg)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=False
)

# 预测
y_pred_reg = xgb_reg.predict(X_test_reg)

# 评估
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"最佳迭代轮数: {xgb_reg.best_iteration}")

# XGBoost分类示例
print("\n=== XGBoost 分类 ===")

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42
)

xgb_clf.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_test_clf, y_test_clf)],
    eval_metric='mlogloss',
    early_stopping_rounds=10,
    verbose=False
)

y_pred_clf = xgb_clf.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"准确率: {accuracy:.4f}")

# 特征重要性可视化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 回归特征重要性
feature_importance_reg = pd.DataFrame({
    'feature': [f'F{i}' for i in range(X_reg.shape[1])],
    'importance': xgb_reg.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[0].barh(range(10), feature_importance_reg['importance'].values)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(feature_importance_reg['feature'].values)
axes[0].set_xlabel('重要性')
axes[0].set_title('XGBoost回归 - Top 10特征')

# 分类特征重要性
xgb.plot_importance(xgb_clf, max_num_features=10, ax=axes[1])
axes[1].set_title('XGBoost分类 - Top 10特征')

plt.tight_layout()
plt.show()

# 高级功能：自定义评估函数和目标函数
def custom_eval_metric(y_pred, dtrain):
    """自定义评估指标"""
    y_true = dtrain.get_label()
    error = np.mean(np.abs(y_true - y_pred))
    return 'custom_mae', error

# 使用DMatrix（XGBoost的数据结构）
dtrain = xgb.DMatrix(X_train_reg, label=y_train_reg)
dtest = xgb.DMatrix(X_test_reg, label=y_test_reg)

# 设置参数
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

# 训练模型
watchlist = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    params, dtrain, 
    num_boost_round=100,
    evals=watchlist,
    feval=custom_eval_metric,
    early_stopping_rounds=10,
    verbose_eval=False
)

print(f"\n自定义评估指标最佳轮数: {model.best_iteration}")

# 学习曲线
results = model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['train']['rmse'], label='训练RMSE')
plt.plot(results['test']['rmse'], label='测试RMSE')
plt.xlabel('迭代轮数')
plt.ylabel('RMSE')
plt.title('XGBoost学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 2. LightGBM - 速度之光 ⚡

### 核心思想
LightGBM使用基于直方图的算法和叶子生长策略，在保持精度的同时大幅提升训练速度。

```python
import lightgbm as lgb
# 安装命令：pip install lightgbm

print("=== LightGBM 示例 ===")

# LightGBM回归
lgb_reg = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    force_col_wise=True  # 避免警告
)

# 训练
lgb_reg.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_test_reg, y_test_reg)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

# 预测和评估
y_pred_lgb = lgb_reg.predict(X_test_reg)
mse_lgb = mean_squared_error(y_test_reg, y_pred_lgb)
r2_lgb = r2_score(y_test_reg, y_pred_lgb)

print(f"LightGBM MSE: {mse_lgb:.4f}")
print(f"LightGBM R²: {r2_lgb:.4f}")

# LightGBM分类
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    force_col_wise=True
)

lgb_clf.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_test_clf, y_test_clf)],
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

y_pred_lgb_clf = lgb_clf.predict(X_test_clf)
accuracy_lgb = accuracy_score(y_test_clf, y_pred_lgb_clf)
print(f"LightGBM 准确率: {accuracy_lgb:.4f}")

# LightGBM的独特功能：类别特征处理
# 创建包含类别特征的数据
np.random.seed(42)
n_samples = 1000

# 数值特征
num_features = np.random.randn(n_samples, 3)

# 类别特征
cat_features = np.column_stack([
    np.random.choice(['A', 'B', 'C'], n_samples),
    np.random.choice(['X', 'Y', 'Z'], n_samples),
    np.random.choice(range(10), n_samples)  # 数值型类别
])

# 合并特征
X_mixed = np.column_stack([num_features, cat_features])

# 创建DataFrame
df_mixed = pd.DataFrame(X_mixed, columns=['num1', 'num2', 'num3', 
                                          'cat1', 'cat2', 'cat3'])

# 转换类别特征为category类型
for col in ['cat1', 'cat2', 'cat3']:
    df_mixed[col] = df_mixed[col].astype('category')

# 生成目标变量
y_mixed = (df_mixed['num1'] > 0).astype(int) & \
          (df_mixed['cat1'] == 'A').astype(int)

# 划分数据
X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(
    df_mixed, y_mixed, test_size=0.2, random_state=42
)

# 训练LightGBM（自动处理类别特征）
lgb_cat = lgb.LGBMClassifier(
    n_estimators=100,
    random_state=42,
    force_col_wise=True
)

lgb_cat.fit(
    X_train_mixed, y_train_mixed,
    eval_set=[(X_test_mixed, y_test_mixed)],
    categorical_feature=['cat1', 'cat2', 'cat3'],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

print(f"\n包含类别特征的LightGBM准确率: "
      f"{lgb_cat.score(X_test_mixed, y_test_mixed):.4f}")

# 速度对比
import time

# 大数据集
X_large, y_large = make_regression(n_samples=10000, n_features=100, 
                                   n_informative=50, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

# XGBoost计时
start = time.time()
xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0).fit(
    X_train_l, y_train_l
)
xgb_time = time.time() - start

# LightGBM计时
start = time.time()
lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1,
                  force_col_wise=True).fit(X_train_l, y_train_l)
lgb_time = time.time() - start

print(f"\n训练时间对比（10000样本，100特征）:")
print(f"XGBoost: {xgb_time:.2f}秒")
print(f"LightGBM: {lgb_time:.2f}秒")
print(f"LightGBM加速比: {xgb_time/lgb_time:.1f}x")
```

## 3. CatBoost - 类别特征大师 🐱

### 核心思想
CatBoost专门优化了类别特征的处理，并使用对称树结构减少过拟合。

```python
import catboost as cb
# 安装命令：pip install catboost

print("\n=== CatBoost 示例 ===")

# CatBoost回归
cat_reg = cb.CatBoostRegressor(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_seed=42,
    verbose=False
)

# 训练
cat_reg.fit(
    X_train_reg, y_train_reg,
    eval_set=(X_test_reg, y_test_reg),
    early_stopping_rounds=10,
    verbose=False
)

# 预测
y_pred_cat = cat_reg.predict(X_test_reg)
mse_cat = mean_squared_error(y_test_reg, y_pred_cat)
r2_cat = r2_score(y_test_reg, y_pred_cat)

print(f"CatBoost MSE: {mse_cat:.4f}")
print(f"CatBoost R²: {r2_cat:.4f}")

# CatBoost处理类别特征的优势
# 使用之前创建的混合数据
print("\n=== CatBoost类别特征处理 ===")

# 需要指定类别特征的索引
cat_features_indices = [3, 4, 5]  # cat1, cat2, cat3的索引

# 准备数据
X_train_cat = X_train_mixed.values
X_test_cat = X_test_mixed.values

# 创建Pool（CatBoost的数据结构）
train_pool = cb.Pool(
    X_train_cat, y_train_mixed,
    cat_features=cat_features_indices
)
test_pool = cb.Pool(
    X_test_cat, y_test_mixed,
    cat_features=cat_features_indices
)

# 训练CatBoost
cat_clf = cb.CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_seed=42,
    verbose=False
)

cat_clf.fit(train_pool, eval_set=test_pool, 
           early_stopping_rounds=10, verbose=False)

# 预测
y_pred_catboost = cat_clf.predict(test_pool)
accuracy_cat = accuracy_score(y_test_mixed, y_pred_catboost)
print(f"CatBoost准确率（类别特征）: {accuracy_cat:.4f}")

# SHAP值解释（CatBoost内置）
shap_values = cat_clf.get_feature_importance(
    test_pool, 
    type='ShapValues'
)

# 可视化SHAP值
plt.figure(figsize=(10, 6))
shap_mean = np.abs(shap_values[:, :-1]).mean(axis=0)
feature_names = [f'num{i+1}' for i in range(3)] + \
                [f'cat{i+1}' for i in range(3)]
indices = np.argsort(shap_mean)[::-1]

plt.bar(range(len(indices)), shap_mean[indices])
plt.xticks(range(len(indices)), 
          [feature_names[i] for i in indices])
plt.xlabel('特征')
plt.ylabel('平均|SHAP|值')
plt.title('CatBoost特征重要性（SHAP）')
plt.show()
```

## 4. 三大框架对比实战

```python
# 综合对比项目：预测房价
def create_house_data(n_samples=2000):
    """创建房价数据（包含数值和类别特征）"""
    np.random.seed(42)
    
    # 数值特征
    area = np.random.uniform(30, 300, n_samples)
    rooms = np.random.randint(1, 6, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    floor = np.random.randint(1, 30, n_samples)
    
    # 类别特征
    district = np.random.choice(['朝阳', '海淀', '东城', '西城', '丰台'], 
                               n_samples)
    orientation = np.random.choice(['东', '南', '西', '北', '东南', '西南'], 
                                  n_samples)
    decoration = np.random.choice(['毛坯', '简装', '精装', '豪装'], 
                                 n_samples)
    
    # 价格计算（有逻辑关系）
    price = (
        area * 15000 +
        rooms * 30000 +
        age * (-3000) +
        floor * 1000 +
        (district == '海淀').astype(int) * 100000 +
        (district == '朝阳').astype(int) * 80000 +
        (orientation == '南').astype(int) * 20000 +
        (decoration == '豪装').astype(int) * 50000 +
        np.random.randn(n_samples) * 30000
    )
    
    # 创建DataFrame
    df = pd.DataFrame({
        '面积': area,
        '房间数': rooms,
        '房龄': age,
        '楼层': floor,
        '区域': district,
        '朝向': orientation,
        '装修': decoration,
        '价格': price
    })
    
    return df

# 创建数据
house_df = create_house_data()
print("房价数据集:")
print(house_df.head())
print(f"\n数据形状: {house_df.shape}")

# 准备特征和目标
X = house_df.drop('价格', axis=1)
y = house_df['价格']

# 处理类别特征
from sklearn.preprocessing import LabelEncoder

# 为XGBoost和通用模型编码
X_encoded = X.copy()
label_encoders = {}
cat_columns = ['区域', '朝向', '装修']

for col in cat_columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 为CatBoost准备（保留原始类别）
X_train_cat = X.loc[X_train.index]
X_test_cat = X.loc[X_test.index]

# 三大框架对比
models = {}
results = {}

print("\n=== 训练三大梯度提升框架 ===")

# 1. XGBoost
print("训练XGBoost...")
models['XGBoost'] = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
models['XGBoost'].fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)

# 2. LightGBM
print("训练LightGBM...")
models['LightGBM'] = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    force_col_wise=True
)
models['LightGBM'].fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

# 3. CatBoost
print("训练CatBoost...")
cat_features_idx = [X_train_cat.columns.get_loc(col) 
                   for col in cat_columns]

models['CatBoost'] = cb.CatBoostRegressor(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    cat_features=cat_features_idx,
    random_seed=42,
    verbose=False
)
models['CatBoost'].fit(
    X_train_cat, y_train,
    eval_set=(X_test_cat, y_test),
    early_stopping_rounds=20,
    verbose=False
)

# 评估所有模型
print("\n=== 模型性能对比 ===")
for name, model in models.items():
    if name == 'CatBoost':
        y_pred = model.predict(X_test_cat)
    else:
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

# 创建对比表
results_df = pd.DataFrame(results).T
print(results_df.round(2))

# 可视化对比
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 性能指标对比
metrics = ['RMSE', 'MAE', 'R²']
for idx, metric in enumerate(metrics):
    ax = axes[0, idx]
    values = [results[model][metric] for model in models.keys()]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(models.keys(), values, color=colors)
    ax.set_title(f'{metric}对比')
    ax.set_ylabel(metric)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')

# 2. 预测vs真实（前100个样本）
for idx, (name, model) in enumerate(models.items()):
    ax = axes[1, idx]
    if name == 'CatBoost':
        y_pred_sample = model.predict(X_test_cat.iloc[:100])
    else:
        y_pred_sample = model.predict(X_test.iloc[:100])
    
    y_test_sample = y_test.iloc[:100].values
    
    ax.scatter(y_test_sample, y_pred_sample, alpha=0.5, s=10)
    ax.plot([y_test_sample.min(), y_test_sample.max()],
            [y_test_sample.min(), y_test_sample.max()],
            'r--', lw=2)
    ax.set_xlabel('真实价格')
    ax.set_ylabel('预测价格')
    ax.set_title(f'{name}')

plt.tight_layout()
plt.show()

# 特征重要性对比
print("\n=== 特征重要性对比 ===")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    if name == 'XGBoost':
        importance = model.feature_importances_
        feature_names = X_train.columns
    elif name == 'LightGBM':
        importance = model.feature_importances_
        feature_names = X_train.columns
    else:  # CatBoost
        importance = model.feature_importances_
        feature_names = X_train_cat.columns
    
    # 排序并取前10
    indices = np.argsort(importance)[::-1][:7]
    
    ax.barh(range(7), importance[indices])
    ax.set_yticks(range(7))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('重要性')
    ax.set_title(f'{name}特征重要性')

plt.tight_layout()
plt.show()
```

## 5. 超参数调优最佳实践

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

# XGBoost超参数空间
xgb_param_dist = {
    'n_estimators': stats.randint(100, 500),
    'max_depth': stats.randint(3, 10),
    'learning_rate': stats.uniform(0.01, 0.29),
    'subsample': stats.uniform(0.6, 0.4),
    'colsample_bytree': stats.uniform(0.6, 0.4),
    'min_child_weight': stats.randint(1, 10),
    'gamma': stats.uniform(0, 0.5),
    'reg_alpha': stats.uniform(0, 1),
    'reg_lambda': stats.uniform(0, 2)
}

# LightGBM超参数空间
lgb_param_dist = {
    'n_estimators': stats.randint(100, 500),
    'max_depth': stats.randint(3, 10),
    'num_leaves': stats.randint(20, 100),
    'learning_rate': stats.uniform(0.01, 0.29),
    'feature_fraction': stats.uniform(0.6, 0.4),
    'bagging_fraction': stats.uniform(0.6, 0.4),
    'bagging_freq': stats.randint(1, 10),
    'min_child_samples': stats.randint(5, 30),
    'reg_alpha': stats.uniform(0, 1),
    'reg_lambda': stats.uniform(0, 2)
}

# CatBoost超参数空间
cat_param_dist = {
    'iterations': stats.randint(100, 500),
    'depth': stats.randint(4, 10),
    'learning_rate': stats.uniform(0.01, 0.29),
    'l2_leaf_reg': stats.uniform(1, 10),
    'bagging_temperature': stats.uniform(0, 1),
    'random_strength': stats.uniform(0, 1),
    'border_count': stats.randint(32, 255)
}

print("=== 超参数调优示例（使用小数据集演示）===")

# 使用较小的数据集进行演示
X_small, y_small = make_regression(n_samples=500, n_features=10, 
                                   n_informative=8, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

# XGBoost调优
print("\n调优XGBoost...")
xgb_random = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42),
    xgb_param_dist,
    n_iter=20,  # 实际使用时可以增加
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

xgb_random.fit(X_train_s, y_train_s)
print(f"最佳参数: {xgb_random.best_params_}")
print(f"最佳CV分数: {-xgb_random.best_score_:.4f}")

# 贝叶斯优化（更高效的调参方法）
# pip install scikit-optimize
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    
    # 定义贝叶斯优化空间
    bayes_space = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0)
    }
    
    bayes_search = BayesSearchCV(
        xgb.XGBRegressor(random_state=42),
        bayes_space,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    bayes_search.fit(X_train_s, y_train_s)
    print(f"\n贝叶斯优化最佳参数: {bayes_search.best_params_}")
    print(f"贝叶斯优化最佳分数: {-bayes_search.best_score_:.4f}")
    
except ImportError:
    print("\n贝叶斯优化需要安装: pip install scikit-optimize")
```

## 6. 实战技巧总结

### 选择指南
```python
def choose_gbm_framework(data_characteristics):
    """根据数据特点选择最合适的梯度提升框架"""
    
    recommendations = []
    
    if data_characteristics.get('has_categorical', False):
        recommendations.append("CatBoost - 自动处理类别特征")
    
    if data_characteristics.get('large_dataset', False):
        recommendations.append("LightGBM - 训练速度最快")
    
    if data_characteristics.get('need_interpretability', False):
        recommendations.append("XGBoost - 特征重要性分析成熟")
    
    if data_characteristics.get('competition', False):
        recommendations.append("XGBoost - 竞赛经验丰富，调参资料多")
    
    if data_characteristics.get('gpu_available', False):
        recommendations.append("XGBoost/LightGBM - GPU支持好")
    
    return recommendations

# 示例
data_chars = {
    'has_categorical': True,
    'large_dataset': True,
    'need_interpretability': False,
    'competition': False,
    'gpu_available': False
}

print("推荐的框架:")
for rec in choose_gbm_framework(data_chars):
    print(f"  - {rec}")
```

### 防止过拟合技巧
```python
# 1. 早停
# 所有框架都支持early_stopping_rounds

# 2. 正则化参数
overfitting_params = {
    'XGBoost': {
        'max_depth': 3,  # 限制树深度
        'min_child_weight': 5,  # 增加最小叶子权重
        'gamma': 0.1,  # 增加分裂所需最小损失减少
        'reg_alpha': 0.1,  # L1正则化
        'reg_lambda': 1.0,  # L2正则化
        'subsample': 0.8,  # 样本采样
        'colsample_bytree': 0.8  # 特征采样
    },
    'LightGBM': {
        'max_depth': 3,
        'num_leaves': 20,  # 限制叶子数
        'min_child_samples': 20,  # 增加最小叶子样本数
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    },
    'CatBoost': {
        'depth': 4,
        'l2_leaf_reg': 3,  # L2正则化
        'bagging_temperature': 0.1,  # 贝叶斯bootstrap
        'random_strength': 1  # 随机性强度
    }
}
```

### 加速训练
```python
# 1. 使用更少的特征
# 2. 减少数据精度（float32 vs float64）
# 3. 使用GPU（如果可用）

# GPU示例（需要GPU版本的库）
gpu_params = {
    'XGBoost': {'tree_method': 'gpu_hist', 'gpu_id': 0},
    'LightGBM': {'device': 'gpu', 'gpu_platform_id': 0},
    'CatBoost': {'task_type': 'GPU', 'devices': '0'}
}
```

## 最佳实践建议

1. **先用默认参数**：三大框架的默认参数都很好
2. **关注过拟合**：使用早停和交叉验证
3. **特征工程优先**：好特征比调参重要
4. **集成多个模型**：组合不同框架的预测

## 下一步学习
- [超参数调优](hyperparameter_tuning.md) - 系统化调参方法
- [特征工程](feature_engineering.md) - 提升模型输入质量
- [模型解释](model_interpretation.md) - 理解模型决策