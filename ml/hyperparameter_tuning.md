# 超参数调优完全指南 🎯

超参数调优就像调音师调钢琴，找到让模型"音色"最美的参数组合。本章将介绍从暴力搜索到智能优化的各种方法。

## 1. GridSearchCV - 网格搜索 📊

### 核心思想
遍历所有参数组合，像在网格上逐个尝试每个交叉点。虽然耗时但保证找到最优解（在给定范围内）。

### 完整代码实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# 创建示例数据集
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, 
                                   n_informative=15, n_redundant=5,
                                   n_classes=2, random_state=42)

X_reg, y_reg = make_regression(n_samples=1000, n_features=20,
                               n_informative=15, noise=10, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)

print("=== GridSearchCV 示例 ===")

# 1. 简单的网格搜索
from sklearn.tree import DecisionTreeClassifier

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# 计算总组合数
n_combinations = 1
for param, values in param_grid.items():
    n_combinations *= len(values)
print(f"参数组合总数: {n_combinations}")

# 创建GridSearchCV对象
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1  # 显示进度
)

# 训练
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\n网格搜索耗时: {grid_time:.2f}秒")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳CV分数: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 2. 可视化搜索结果
results_df = pd.DataFrame(grid_search.cv_results_)

# 创建热力图显示不同参数组合的性能
pivot_table = results_df.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_min_samples_split'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('GridSearchCV结果热力图')
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.show()

# 3. 多指标评估
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# 定义多个评分指标
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# 使用多指标的GridSearchCV
multi_grid = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit='f1',  # 使用F1分数选择最佳模型
    n_jobs=-1,
    return_train_score=True
)

multi_grid.fit(X_train, y_train)

print("\n多指标评估结果:")
for metric in scoring.keys():
    score = multi_grid.cv_results_[f'mean_test_{metric}'][multi_grid.best_index_]
    print(f"{metric}: {score:.4f}")
```

## 2. RandomizedSearchCV - 随机搜索 🎲

### 核心思想
随机采样参数组合，用更少的尝试找到接近最优的解。适合参数空间很大的情况。

```python
from scipy import stats

print("\n=== RandomizedSearchCV 示例 ===")

# 定义连续分布的参数空间
param_distributions = {
    'n_estimators': stats.randint(50, 200),
    'max_depth': stats.randint(3, 20),
    'min_samples_split': stats.randint(2, 20),
    'min_samples_leaf': stats.randint(1, 10),
    'max_features': stats.uniform(0.1, 0.9),  # 连续均匀分布
    'bootstrap': [True, False]
}

# 创建RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100,  # 尝试100个随机组合
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# 训练
start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"\n随机搜索耗时: {random_time:.2f}秒")
print(f"最佳参数: {random_search.best_params_}")
print(f"最佳CV分数: {random_search.best_score_:.4f}")

# 对比Grid和Random搜索
print(f"\n时间对比:")
print(f"GridSearchCV: {grid_time:.2f}秒 ({n_combinations}个组合)")
print(f"RandomizedSearchCV: {random_time:.2f}秒 (100个组合)")
print(f"加速比: {grid_time/random_time:.1f}x")

# 可视化参数重要性
def plot_param_importance(search_cv, top_n=20):
    """绘制参数组合的性能分布"""
    results = pd.DataFrame(search_cv.cv_results_)
    results = results.sort_values('mean_test_score', ascending=False).head(top_n)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    params_to_plot = ['param_n_estimators', 'param_max_depth', 
                      'param_min_samples_split', 'param_min_samples_leaf',
                      'param_max_features', 'mean_test_score']
    
    for idx, param in enumerate(params_to_plot):
        ax = axes[idx // 3, idx % 3]
        if param == 'mean_test_score':
            ax.bar(range(top_n), results[param].values)
            ax.set_xlabel('组合排名')
            ax.set_ylabel('CV分数')
            ax.set_title('Top 20组合的分数')
        else:
            ax.scatter(results[param].values, results['mean_test_score'].values)
            ax.set_xlabel(param.replace('param_', ''))
            ax.set_ylabel('CV分数')
            ax.set_title(f'{param.replace("param_", "")}对性能的影响')
    
    plt.tight_layout()
    plt.show()

plot_param_importance(random_search)

# 学习曲线：参数搜索的收敛
def plot_search_convergence(search_cv):
    """绘制搜索过程的收敛曲线"""
    results = pd.DataFrame(search_cv.cv_results_)
    results = results.sort_values('mean_test_score', ascending=False)
    
    # 计算到目前为止的最佳分数
    best_scores = []
    current_best = -np.inf
    for score in results['mean_test_score'].values:
        if score > current_best:
            current_best = score
        best_scores.append(current_best)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_scores)), best_scores, 'b-', linewidth=2)
    plt.xlabel('评估的参数组合数')
    plt.ylabel('最佳CV分数')
    plt.title('随机搜索收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.show()

plot_search_convergence(random_search)
```

## 3. 贝叶斯优化 - 智能搜索 🧠

### 核心思想
使用之前的评估结果来指导下一次尝试，像一个会学习的搜索算法。

```python
# 需要安装: pip install scikit-optimize
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    
    print("\n=== 贝叶斯优化 ===")
    
    # 定义搜索空间
    search_spaces = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Real(0.1, 0.9),
        'bootstrap': Categorical([True, False])
    }
    
    # 创建贝叶斯搜索
    bayes_search = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        search_spaces=search_spaces,
        n_iter=50,  # 比随机搜索更少的迭代
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    # 训练
    start_time = time.time()
    bayes_search.fit(X_train, y_train)
    bayes_time = time.time() - start_time
    
    print(f"\n贝叶斯优化耗时: {bayes_time:.2f}秒")
    print(f"最佳参数: {bayes_search.best_params_}")
    print(f"最佳CV分数: {bayes_search.best_score_:.4f}")
    
    # 三种方法对比
    comparison_data = {
        '方法': ['Grid Search', 'Random Search', 'Bayesian Optimization'],
        '耗时(秒)': [grid_time, random_time, bayes_time],
        '尝试次数': [n_combinations, 100, 50],
        'CV分数': [grid_search.best_score_, 
                  random_search.best_score_, 
                  bayes_search.best_score_]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n三种调参方法对比:")
    print(comparison_df)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 耗时对比
    axes[0].bar(comparison_df['方法'], comparison_df['耗时(秒)'])
    axes[0].set_ylabel('耗时(秒)')
    axes[0].set_title('运行时间对比')
    
    # 尝试次数对比
    axes[1].bar(comparison_df['方法'], comparison_df['尝试次数'])
    axes[1].set_ylabel('参数组合数')
    axes[1].set_title('尝试次数对比')
    
    # CV分数对比
    axes[2].bar(comparison_df['方法'], comparison_df['CV分数'])
    axes[2].set_ylabel('CV分数')
    axes[2].set_title('最佳分数对比')
    axes[2].set_ylim([min(comparison_df['CV分数']) * 0.95, 
                      max(comparison_df['CV分数']) * 1.02])
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\n贝叶斯优化需要安装scikit-optimize:")
    print("pip install scikit-optimize")
```

## 4. Optuna - 新一代优化框架 🚀

```python
# 需要安装: pip install optuna
try:
    import optuna
    from optuna import Trial
    from optuna.samplers import TPESampler
    
    print("\n=== Optuna 高级优化 ===")
    
    def objective(trial: Trial):
        """定义优化目标函数"""
        # 建议超参数
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 0.9),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        # 创建模型
        model = RandomForestClassifier(**params, random_state=42)
        
        # 交叉验证
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        return scores.mean()
    
    # 创建study
    study = optuna.create_study(
        direction='maximize',  # 最大化准确率
        sampler=TPESampler(seed=42)  # 使用TPE采样器
    )
    
    # 优化
    start_time = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    optuna_time = time.time() - start_time
    
    print(f"\nOptuna优化耗时: {optuna_time:.2f}秒")
    print(f"最佳参数: {study.best_params}")
    print(f"最佳分数: {study.best_value:.4f}")
    
    # 可视化优化历史
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 优化历史
    trials_df = study.trials_dataframe()
    axes[0].plot(trials_df.index, trials_df['value'], 'b-', alpha=0.5)
    axes[0].scatter(study.best_trial.number, study.best_value, 
                   color='red', s=100, zorder=5)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('Optuna优化历史')
    axes[0].grid(True, alpha=0.3)
    
    # 参数重要性
    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())
    values = list(importances.values())
    
    axes[1].barh(params, values)
    axes[1].set_xlabel('重要性')
    axes[1].set_title('参数重要性（Optuna）')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\nOptuna需要安装:")
    print("pip install optuna")
```

## 5. 实战案例：多模型联合调优

```python
print("\n=== 实战：多模型联合调优 ===")

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# 定义多个模型和它们的参数空间
models_and_params = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, use_label_encoder=False,
                                   eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
    }
}

# 对每个模型进行调优
best_models = {}
results_summary = []

for name, config in models_and_params.items():
    print(f"\n调优 {name}...")
    
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    
    # 测试集性能
    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    results_summary.append({
        '模型': name,
        'CV分数': grid.best_score_,
        '测试分数': test_acc,
        '最佳参数': grid.best_params_
    })
    
    print(f"  最佳CV分数: {grid.best_score_:.4f}")
    print(f"  测试集分数: {test_acc:.4f}")

# 结果汇总
results_df = pd.DataFrame(results_summary)
print("\n=== 所有模型调优结果 ===")
print(results_df[['模型', 'CV分数', '测试分数']])

# 集成最佳模型
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n集成模型准确率: {ensemble_acc:.4f}")

# 可视化对比
plt.figure(figsize=(10, 6))
models = results_df['模型'].tolist() + ['Ensemble']
cv_scores = results_df['CV分数'].tolist() + [ensemble_acc]
test_scores = results_df['测试分数'].tolist() + [ensemble_acc]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, cv_scores, width, label='CV分数', alpha=0.8)
plt.bar(x + width/2, test_scores, width, label='测试分数', alpha=0.8)

plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('模型性能对比')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 6. 高级技巧：自适应调参策略

```python
class AdaptiveHyperparameterTuner:
    """自适应超参数调优器"""
    
    def __init__(self, model_class, initial_params, param_ranges):
        self.model_class = model_class
        self.initial_params = initial_params
        self.param_ranges = param_ranges
        self.history = []
        
    def coarse_search(self, X, y, cv=5):
        """粗搜索：大范围快速搜索"""
        print("Phase 1: 粗搜索...")
        
        # 创建粗粒度参数网格
        coarse_grid = {}
        for param, range_vals in self.param_ranges.items():
            if isinstance(range_vals, list):
                # 离散参数
                coarse_grid[param] = range_vals[::2]  # 每隔一个取一个
            else:
                # 连续参数
                min_val, max_val = range_vals
                coarse_grid[param] = np.linspace(min_val, max_val, 5)
        
        # 执行网格搜索
        model = self.model_class(**self.initial_params)
        grid = GridSearchCV(model, coarse_grid, cv=cv, n_jobs=-1)
        grid.fit(X, y)
        
        self.coarse_best_params = grid.best_params_
        self.coarse_best_score = grid.best_score_
        
        print(f"  粗搜索最佳分数: {self.coarse_best_score:.4f}")
        return self.coarse_best_params
    
    def fine_search(self, X, y, cv=5):
        """细搜索：在最佳参数附近精细搜索"""
        print("Phase 2: 细搜索...")
        
        # 创建细粒度参数网格
        fine_grid = {}
        for param, best_val in self.coarse_best_params.items():
            range_vals = self.param_ranges[param]
            
            if isinstance(range_vals, list):
                # 离散参数：选择邻近值
                idx = range_vals.index(best_val)
                start_idx = max(0, idx - 1)
                end_idx = min(len(range_vals), idx + 2)
                fine_grid[param] = range_vals[start_idx:end_idx]
            else:
                # 连续参数：在最佳值附近搜索
                min_val, max_val = range_vals
                delta = (max_val - min_val) * 0.1
                fine_min = max(min_val, best_val - delta)
                fine_max = min(max_val, best_val + delta)
                fine_grid[param] = np.linspace(fine_min, fine_max, 5)
        
        # 执行细搜索
        model = self.model_class(**self.initial_params)
        grid = GridSearchCV(model, fine_grid, cv=cv, n_jobs=-1)
        grid.fit(X, y)
        
        self.fine_best_params = grid.best_params_
        self.fine_best_score = grid.best_score_
        
        print(f"  细搜索最佳分数: {self.fine_best_score:.4f}")
        return self.fine_best_params
    
    def adaptive_search(self, X, y, cv=5):
        """完整的自适应搜索流程"""
        # 粗搜索
        coarse_params = self.coarse_search(X, y, cv)
        
        # 细搜索
        fine_params = self.fine_search(X, y, cv)
        
        # 返回最终最佳参数
        return fine_params, self.fine_best_score

# 使用示例
print("\n=== 自适应超参数调优 ===")

tuner = AdaptiveHyperparameterTuner(
    model_class=RandomForestClassifier,
    initial_params={'random_state': 42},
    param_ranges={
        'n_estimators': (50, 300),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
)

best_params, best_score = tuner.adaptive_search(X_train, y_train)
print(f"\n最终最佳参数: {best_params}")
print(f"最终最佳分数: {best_score:.4f}")
```

## 7. 实用工具函数

```python
def plot_validation_curve(model, X, y, param_name, param_range, cv=5):
    """绘制验证曲线"""
    from sklearn.model_selection import validation_curve
    
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name,
        param_range=param_range, cv=cv, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'b-', label='训练分数')
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='b')
    plt.plot(param_range, val_mean, 'r-', label='验证分数')
    plt.fill_between(param_range, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='r')
    
    plt.xlabel(param_name)
    plt.ylabel('分数')
    plt.title(f'验证曲线: {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 示例
print("\n=== 验证曲线分析 ===")
plot_validation_curve(
    RandomForestClassifier(random_state=42),
    X_train, y_train,
    'n_estimators',
    [10, 50, 100, 150, 200, 250, 300]
)

def get_param_importance(model, X, y, param_ranges, n_iter=20):
    """评估参数重要性"""
    from sklearn.model_selection import RandomizedSearchCV
    
    random_search = RandomizedSearchCV(
        model, param_ranges, n_iter=n_iter,
        cv=5, n_jobs=-1, random_state=42
    )
    random_search.fit(X, y)
    
    results = pd.DataFrame(random_search.cv_results_)
    
    # 计算每个参数的重要性
    importances = {}
    for param in param_ranges.keys():
        param_col = f'param_{param}'
        if param_col in results.columns:
            # 计算参数值与分数的相关性
            correlation = results[[param_col, 'mean_test_score']].corr().iloc[0, 1]
            importances[param] = abs(correlation)
    
    return importances

# 示例
param_ranges = {
    'n_estimators': stats.randint(50, 200),
    'max_depth': stats.randint(3, 20),
    'min_samples_split': stats.randint(2, 20)
}

importances = get_param_importance(
    RandomForestClassifier(random_state=42),
    X_train, y_train,
    param_ranges
)

print("\n参数重要性:")
for param, importance in sorted(importances.items(), 
                               key=lambda x: x[1], reverse=True):
    print(f"  {param}: {importance:.3f}")
```

## 最佳实践建议

### 1. 调参流程
1. **建立基线**：使用默认参数
2. **单参数分析**：理解每个参数的影响
3. **粗调**：大范围快速搜索
4. **细调**：在最佳区域精细搜索
5. **验证**：确保没有过拟合

### 2. 选择调优方法
- **参数少(<10)**: GridSearchCV
- **参数中等(10-20)**: RandomizedSearchCV
- **参数多(>20)**: 贝叶斯优化或Optuna
- **计算资源有限**: 随机搜索或贝叶斯优化

### 3. 常见陷阱
- 在测试集上调参（数据泄露）
- 忽视计算成本
- 过度调优导致过拟合
- 不考虑参数间的相互作用

## 下一步学习
- [特征工程](feature_engineering.md) - 好特征比调参更重要
- [模型评估](evaluation.md) - 正确评估调参效果
- [AutoML](automl.md) - 自动化机器学习