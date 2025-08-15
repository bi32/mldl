# 机器学习基础：从零开始 🚀

全面掌握机器学习的基础概念、理论和实践。

## 1. 机器学习概述 🌟

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置可视化风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

class MachineLearningBasics:
    """机器学习基础概念"""
    
    def __init__(self):
        self.concepts = {
            "监督学习": "从标注数据学习输入到输出的映射",
            "无监督学习": "从无标注数据发现模式",
            "强化学习": "通过与环境交互学习最优策略",
            "半监督学习": "结合少量标注数据和大量无标注数据",
            "迁移学习": "将一个任务的知识迁移到另一个任务"
        }
    
    def demonstrate_ml_workflow(self):
        """演示机器学习工作流程"""
        print("=" * 50)
        print("机器学习标准工作流程")
        print("=" * 50)
        
        # 1. 数据收集
        print("\n1. 数据收集")
        print("   - 确定数据源")
        print("   - 收集原始数据")
        print("   - 数据质量检查")
        
        # 2. 数据预处理
        print("\n2. 数据预处理")
        print("   - 处理缺失值")
        print("   - 处理异常值")
        print("   - 特征编码")
        print("   - 特征缩放")
        
        # 3. 特征工程
        print("\n3. 特征工程")
        print("   - 特征选择")
        print("   - 特征提取")
        print("   - 特征创建")
        
        # 4. 模型选择
        print("\n4. 模型选择")
        print("   - 选择算法")
        print("   - 设置基准模型")
        
        # 5. 模型训练
        print("\n5. 模型训练")
        print("   - 分割数据集")
        print("   - 训练模型")
        print("   - 交叉验证")
        
        # 6. 模型评估
        print("\n6. 模型评估")
        print("   - 选择评估指标")
        print("   - 验证集评估")
        print("   - 测试集评估")
        
        # 7. 模型优化
        print("\n7. 模型优化")
        print("   - 超参数调优")
        print("   - 特征优化")
        print("   - 集成方法")
        
        # 8. 模型部署
        print("\n8. 模型部署")
        print("   - 模型序列化")
        print("   - API开发")
        print("   - 监控维护")
    
    def create_sample_dataset(self, n_samples=1000, n_features=4, 
                            task='classification'):
        """创建示例数据集"""
        np.random.seed(42)
        
        if task == 'classification':
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=3,
                n_redundant=1,
                n_clusters_per_class=2,
                random_state=42
            )
        else:
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=3,
                noise=10,
                random_state=42
            )
        
        # 转换为DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        return df
```

## 2. 数据预处理 🔧

```python
class DataPreprocessing:
    """数据预处理技术"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def handle_missing_data(self, df, strategy='mean'):
        """处理缺失数据"""
        df_processed = df.copy()
        
        # 数值型特征
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            for col in numeric_columns:
                df_processed[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in numeric_columns:
                df_processed[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in numeric_columns:
                df_processed[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'forward_fill':
            df_processed.fillna(method='ffill', inplace=True)
        elif strategy == 'backward_fill':
            df_processed.fillna(method='bfill', inplace=True)
        elif strategy == 'interpolate':
            df_processed[numeric_columns] = df[numeric_columns].interpolate()
        
        # 类别型特征
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_processed[col].fillna('Unknown', inplace=True)
        
        return df_processed
    
    def scale_features(self, X, method='standard'):
        """特征缩放"""
        from sklearn.preprocessing import (
            StandardScaler, MinMaxScaler, RobustScaler, Normalizer
        )
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'normalize':
            scaler = Normalizer()
        else:
            return X
        
        X_scaled = scaler.fit_transform(X)
        self.scalers[method] = scaler
        
        return X_scaled
    
    def encode_categorical(self, df, columns, method='onehot'):
        """编码类别变量"""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        df_encoded = df.copy()
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                self.encoders[col] = le
        
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df, columns=columns)
        
        elif method == 'target':
            # 目标编码
            for col in columns:
                mean_target = df.groupby(col)['target'].mean()
                df_encoded[col] = df[col].map(mean_target)
        
        return df_encoded
    
    def remove_outliers(self, df, columns, method='iqr', threshold=1.5):
        """移除异常值"""
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                                   (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df_clean = df_clean[z_scores < threshold]
        
        return df_clean
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """分割数据集"""
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 从剩余数据中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# 数据可视化
class DataVisualization:
    """数据可视化工具"""
    
    @staticmethod
    def plot_data_distribution(df, columns=None):
        """绘制数据分布"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            axes[i].hist(df[col], bins=30, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # 隐藏多余的子图
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df):
        """绘制相关性矩阵"""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation = numeric_df.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.show()
        
        return correlation
    
    @staticmethod
    def plot_feature_importance(feature_names, importance_scores):
        """绘制特征重要性"""
        indices = np.argsort(importance_scores)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance_scores)), importance_scores[indices])
        plt.xticks(range(len(importance_scores)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
```

## 3. 监督学习基础 📚

```python
class SupervisedLearning:
    """监督学习基础"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """线性回归示例"""
        from sklearn.linear_model import LinearRegression
        
        # 创建和训练模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        print("线性回归结果:")
        print(f"训练MSE: {train_mse:.4f}")
        print(f"测试MSE: {test_mse:.4f}")
        print(f"系数: {model.coef_}")
        print(f"截距: {model.intercept_:.4f}")
        
        self.models['linear_regression'] = model
        return model
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """逻辑回归示例"""
        from sklearn.linear_model import LogisticRegression
        
        # 创建和训练模型
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print("逻辑回归结果:")
        print(f"训练准确率: {train_acc:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        
        self.models['logistic_regression'] = model
        return model
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test, 
                          task='classification'):
        """决策树示例"""
        if task == 'classification':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
            metric = accuracy_score
            metric_name = "准确率"
        else:
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
            metric = mean_squared_error
            metric_name = "MSE"
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_score = metric(y_train, y_pred_train)
        test_score = metric(y_test, y_pred_test)
        
        print(f"决策树结果:")
        print(f"训练{metric_name}: {train_score:.4f}")
        print(f"测试{metric_name}: {test_score:.4f}")
        
        # 特征重要性
        if hasattr(X_train, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n特征重要性:")
            print(feature_importance.head())
        
        self.models['decision_tree'] = model
        return model
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """支持向量机示例"""
        from sklearn.svm import SVC
        
        # 创建和训练模型
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print("SVM结果:")
        print(f"训练准确率: {train_acc:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"支持向量数: {len(model.support_)}")
        
        self.models['svm'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """随机森林示例"""
        from sklearn.ensemble import RandomForestClassifier
        
        # 创建和训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print("随机森林结果:")
        print(f"训练准确率: {train_acc:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        
        self.models['random_forest'] = model
        return model
```

## 4. 无监督学习基础 🔍

```python
class UnsupervisedLearning:
    """无监督学习基础"""
    
    def __init__(self):
        self.models = {}
    
    def kmeans_clustering(self, X, n_clusters=3):
        """K均值聚类"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # 创建和训练模型
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        
        # 评估
        silhouette = silhouette_score(X, labels)
        inertia = model.inertia_
        
        print(f"K-Means聚类结果 (k={n_clusters}):")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Inertia: {inertia:.4f}")
        
        self.models['kmeans'] = model
        
        # 可视化（如果是2D）
        if X.shape[1] == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            plt.scatter(model.cluster_centers_[:, 0], 
                       model.cluster_centers_[:, 1],
                       marker='x', s=200, linewidths=3, 
                       color='red', label='Centroids')
            plt.title('K-Means Clustering')
            plt.legend()
            plt.show()
        
        return labels
    
    def hierarchical_clustering(self, X, n_clusters=3):
        """层次聚类"""
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        # 创建和训练模型
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        
        print(f"层次聚类结果 (n_clusters={n_clusters}):")
        print(f"聚类标签: {np.unique(labels)}")
        
        # 绘制树状图
        if len(X) < 100:  # 只对小数据集绘制
            plt.figure(figsize=(12, 6))
            linkage_matrix = linkage(X, method='ward')
            dendrogram(linkage_matrix)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            plt.show()
        
        self.models['hierarchical'] = model
        return labels
    
    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        from sklearn.cluster import DBSCAN
        
        # 创建和训练模型
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        # 统计
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"DBSCAN聚类结果:")
        print(f"聚类数: {n_clusters}")
        print(f"噪声点数: {n_noise}")
        
        self.models['dbscan'] = model
        return labels
    
    def pca_reduction(self, X, n_components=2):
        """主成分分析"""
        from sklearn.decomposition import PCA
        
        # 创建和训练模型
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # 解释方差
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"PCA降维结果:")
        print(f"解释方差比: {explained_variance_ratio}")
        print(f"累积解释方差: {cumulative_variance}")
        
        # 可视化
        if n_components >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
            plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance)')
            plt.title('PCA Visualization')
            plt.show()
        
        self.models['pca'] = pca
        return X_reduced
    
    def anomaly_detection(self, X, contamination=0.1):
        """异常检测"""
        from sklearn.ensemble import IsolationForest
        
        # 创建和训练模型
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X)
        
        # 统计
        n_outliers = list(predictions).count(-1)
        n_inliers = list(predictions).count(1)
        
        print(f"异常检测结果:")
        print(f"正常点: {n_inliers}")
        print(f"异常点: {n_outliers}")
        print(f"异常比例: {n_outliers/len(X):.2%}")
        
        self.models['isolation_forest'] = model
        return predictions
```

## 5. 模型选择与验证 ✅

```python
class ModelSelection:
    """模型选择与验证"""
    
    def __init__(self):
        self.best_model = None
        self.best_params = None
    
    def compare_models(self, X, y, models_dict, cv=5):
        """比较多个模型"""
        from sklearn.model_selection import cross_val_score
        
        results = []
        
        for name, model in models_dict.items():
            scores = cross_val_score(model, X, y, cv=cv, 
                                    scoring='accuracy')
            results.append({
                'Model': name,
                'Mean Score': scores.mean(),
                'Std Score': scores.std(),
                'Min Score': scores.min(),
                'Max Score': scores.max()
            })
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Mean Score', ascending=False)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.errorbar(results_df['Model'], results_df['Mean Score'],
                    yerr=results_df['Std Score'], fmt='o', capsize=5)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def grid_search(self, model, param_grid, X, y, cv=5):
        """网格搜索"""
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        print("网格搜索结果:")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳分数: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        return grid_search
    
    def random_search(self, model, param_distributions, X, y, 
                     n_iter=100, cv=5):
        """随机搜索"""
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            model, param_distributions, 
            n_iter=n_iter, cv=cv,
            scoring='accuracy', n_jobs=-1,
            random_state=42
        )
        random_search.fit(X, y)
        
        print("随机搜索结果:")
        print(f"最佳参数: {random_search.best_params_}")
        print(f"最佳分数: {random_search.best_score_:.4f}")
        
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        return random_search
    
    def learning_curve_analysis(self, model, X, y, cv=5):
        """学习曲线分析"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        
        # 计算均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 绘制学习曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 诊断
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val
        
        if gap > 0.1:
            print("诊断: 可能存在过拟合")
            print("建议: 增加正则化、减少模型复杂度或增加数据")
        elif final_val < 0.7:
            print("诊断: 可能存在欠拟合")
            print("建议: 增加模型复杂度或改进特征")
        else:
            print("诊断: 模型表现良好")
```

## 6. 实战项目示例 💼

```python
class MLProject:
    """完整的机器学习项目"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessing()
        self.visualizer = DataVisualization()
        self.supervised = SupervisedLearning()
        self.selector = ModelSelection()
    
    def run_classification_project(self):
        """运行分类项目"""
        print("=" * 50)
        print("机器学习分类项目示例")
        print("=" * 50)
        
        # 1. 创建数据
        print("\n1. 创建示例数据集...")
        basics = MachineLearningBasics()
        df = basics.create_sample_dataset(n_samples=1000, task='classification')
        
        # 2. 数据探索
        print("\n2. 数据探索...")
        print(df.head())
        print(f"\n数据形状: {df.shape}")
        print(f"类别分布:\n{df['target'].value_counts()}")
        
        # 3. 数据可视化
        print("\n3. 数据可视化...")
        self.visualizer.plot_correlation_matrix(df)
        
        # 4. 数据预处理
        print("\n4. 数据预处理...")
        X = df.drop('target', axis=1)
        y = df['target']
        
        # 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.split_data(X, y)
        
        # 特征缩放
        X_train_scaled = self.preprocessor.scale_features(X_train)
        X_val_scaled = self.preprocessor.scale_features(X_val)
        X_test_scaled = self.preprocessor.scale_features(X_test)
        
        # 5. 模型训练
        print("\n5. 训练多个模型...")
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        # 6. 模型比较
        print("\n6. 模型比较...")
        results = self.selector.compare_models(
            X_train_scaled, y_train, models, cv=5
        )
        
        # 7. 超参数调优
        print("\n7. 对最佳模型进行超参数调优...")
        best_model_name = results.iloc[0]['Model']
        print(f"最佳模型: {best_model_name}")
        
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = self.selector.grid_search(
                RandomForestClassifier(), param_grid,
                X_train_scaled, y_train
            )
        
        # 8. 最终评估
        print("\n8. 在测试集上评估...")
        if hasattr(self.selector, 'best_model'):
            final_model = self.selector.best_model
            test_score = final_model.score(X_test_scaled, y_test)
            print(f"测试集准确率: {test_score:.4f}")
        
        print("\n项目完成！")
        
        return results
```

## 7. 常见问题与解决方案 ❓

```python
class MLTroubleshooting:
    """机器学习常见问题解决"""
    
    @staticmethod
    def diagnose_overfitting(train_score, val_score, threshold=0.1):
        """诊断过拟合"""
        gap = train_score - val_score
        
        if gap > threshold:
            print("检测到过拟合!")
            print(f"训练分数: {train_score:.4f}")
            print(f"验证分数: {val_score:.4f}")
            print(f"差距: {gap:.4f}")
            print("\n解决方案:")
            print("1. 增加训练数据")
            print("2. 减少模型复杂度")
            print("3. 增加正则化")
            print("4. 使用Dropout（深度学习）")
            print("5. 早停（Early Stopping）")
            print("6. 数据增强")
            return True
        return False
    
    @staticmethod
    def diagnose_underfitting(train_score, val_score, target_score=0.8):
        """诊断欠拟合"""
        if train_score < target_score and val_score < target_score:
            print("检测到欠拟合!")
            print(f"训练分数: {train_score:.4f}")
            print(f"验证分数: {val_score:.4f}")
            print("\n解决方案:")
            print("1. 增加模型复杂度")
            print("2. 增加特征")
            print("3. 减少正则化")
            print("4. 更换更强大的模型")
            print("5. 特征工程")
            return True
        return False
    
    @staticmethod
    def handle_imbalanced_data():
        """处理不平衡数据"""
        print("处理不平衡数据的策略:")
        print("\n1. 重采样方法:")
        print("   - 过采样少数类（SMOTE）")
        print("   - 欠采样多数类")
        print("   - 组合采样")
        
        print("\n2. 算法层面:")
        print("   - 使用class_weight参数")
        print("   - 使用专门的算法（如BalancedRandomForest）")
        
        print("\n3. 评估指标:")
        print("   - 不要只看准确率")
        print("   - 使用F1、AUC、精确率、召回率")
        
        print("\n4. 其他方法:")
        print("   - 集成方法")
        print("   - 异常检测方法")
        print("   - 成本敏感学习")
    
    @staticmethod
    def feature_selection_guide():
        """特征选择指南"""
        print("特征选择方法:")
        print("\n1. 过滤法（Filter）:")
        print("   - 方差阈值")
        print("   - 相关系数")
        print("   - 互信息")
        print("   - 卡方检验")
        
        print("\n2. 包装法（Wrapper）:")
        print("   - 递归特征消除（RFE）")
        print("   - 前向选择")
        print("   - 后向消除")
        
        print("\n3. 嵌入法（Embedded）:")
        print("   - L1正则化")
        print("   - 树模型特征重要性")
        print("   - 随机森林特征重要性")
        
        print("\n选择建议:")
        print("- 特征很多（>100）：先用过滤法")
        print("- 特征中等（10-100）：包装法或嵌入法")
        print("- 特征很少（<10）：谨慎选择，可能都需要")
```

## 最佳实践总结 📋

```python
def ml_best_practices():
    """机器学习最佳实践"""
    
    practices = {
        "数据准备": [
            "始终检查数据质量",
            "处理缺失值和异常值",
            "正确编码类别变量",
            "特征缩放很重要",
            "保持训练/验证/测试集独立"
        ],
        
        "模型开发": [
            "从简单模型开始",
            "建立基准模型",
            "使用交叉验证",
            "避免数据泄露",
            "记录所有实验"
        ],
        
        "特征工程": [
            "理解业务背景",
            "创建有意义的特征",
            "考虑特征交互",
            "定期评估特征重要性",
            "避免过多特征"
        ],
        
        "模型评估": [
            "选择合适的评估指标",
            "不要只看单一指标",
            "在独立测试集上评估",
            "考虑业务指标",
            "进行错误分析"
        ],
        
        "部署考虑": [
            "模型大小和推理速度",
            "模型可解释性",
            "监控模型性能",
            "准备模型更新策略",
            "考虑边缘情况"
        ]
    }
    
    return practices

# 学习路径
learning_path = """
机器学习学习路径：

1. 基础阶段（1-2个月）
   - Python编程基础
   - NumPy、Pandas、Matplotlib
   - 统计学基础
   - 线性代数基础

2. 核心算法（2-3个月）
   - 监督学习算法
   - 无监督学习算法
   - 模型评估方法
   - 特征工程

3. 进阶技术（2-3个月）
   - 集成学习
   - 深度学习基础
   - 自然语言处理
   - 计算机视觉

4. 实战项目（持续）
   - Kaggle竞赛
   - 开源项目贡献
   - 个人项目
   - 论文复现

5. 专业化（根据兴趣）
   - 深度学习
   - 强化学习
   - 推荐系统
   - 时间序列
"""

print("机器学习基础指南加载完成！")
```

## 下一步学习
- [回归算法](regression.md) - 深入学习回归技术
- [分类算法](classification.md) - 掌握分类方法
- [特征工程](feature_engineering.md) - 提升模型性能的关键