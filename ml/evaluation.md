# 模型评估完全指南 📊

深入理解如何正确评估机器学习模型的性能。

## 1. 评估基础 🎯

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    # 分类指标
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    # 回归指标
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    # 聚类指标
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, task_type='classification'):
        """
        task_type: 'classification', 'regression', 'clustering'
        """
        self.task_type = task_type
        self.results = {}
        
    def evaluate_classification(self, y_true, y_pred, y_proba=None):
        """分类模型评估"""
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # 如果有概率预测
        if y_proba is not None:
            if len(np.unique(y_true)) == 2:  # 二分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:  # 多分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # 分类报告
        metrics['classification_report'] = classification_report(y_true, y_pred)
        
        self.results = metrics
        return metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """回归模型评估"""
        metrics = {}
        
        # 基础指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 高级指标
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # 残差分析
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skew'] = self._calculate_skewness(residuals)
        
        self.results = metrics
        return metrics
    
    def evaluate_clustering(self, X, labels):
        """聚类模型评估"""
        metrics = {}
        
        # 内部指标
        metrics['silhouette'] = silhouette_score(X, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        
        # 聚类统计
        unique_labels = np.unique(labels)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = {
            f'cluster_{i}': np.sum(labels == i) 
            for i in unique_labels
        }
        
        self.results = metrics
        return metrics
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        from scipy.stats import skew
        return skew(data)
    
    def plot_results(self):
        """可视化评估结果"""
        if self.task_type == 'classification':
            self._plot_classification_results()
        elif self.task_type == 'regression':
            self._plot_regression_results()
        elif self.task_type == 'clustering':
            self._plot_clustering_results()
    
    def _plot_classification_results(self):
        """分类结果可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 混淆矩阵
        if 'confusion_matrix' in self.results:
            sns.heatmap(self.results['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues',
                       ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
        
        # 指标条形图
        if 'accuracy' in self.results:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [self.results.get(m, 0) for m in metrics]
            axes[0, 1].bar(metrics, values)
            axes[0, 1].set_title('Classification Metrics')
            axes[0, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
```

## 2. 交叉验证策略 🔄

```python
class CrossValidationStrategy:
    """交叉验证策略"""
    
    def __init__(self, cv_type='kfold', n_splits=5, random_state=42):
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = self._get_cv_splitter()
    
    def _get_cv_splitter(self):
        """获取交叉验证分割器"""
        if self.cv_type == 'kfold':
            return KFold(n_splits=self.n_splits, 
                        shuffle=True, 
                        random_state=self.random_state)
        
        elif self.cv_type == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits,
                                  shuffle=True,
                                  random_state=self.random_state)
        
        elif self.cv_type == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        
        elif self.cv_type == 'leave_one_out':
            from sklearn.model_selection import LeaveOneOut
            return LeaveOneOut()
        
        elif self.cv_type == 'group':
            from sklearn.model_selection import GroupKFold
            return GroupKFold(n_splits=self.n_splits)
    
    def evaluate_model(self, model, X, y, scoring='accuracy'):
        """使用交叉验证评估模型"""
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            '95_ci': (np.mean(scores) - 1.96 * np.std(scores),
                     np.mean(scores) + 1.96 * np.std(scores))
        }
        
        return results
    
    def evaluate_multiple_metrics(self, model, X, y, metrics_dict):
        """评估多个指标"""
        results = cross_validate(model, X, y, cv=self.cv, 
                               scoring=metrics_dict,
                               return_train_score=True)
        
        # 整理结果
        summary = {}
        for key in results:
            if key.startswith('test_') or key.startswith('train_'):
                summary[key] = {
                    'mean': np.mean(results[key]),
                    'std': np.std(results[key])
                }
        
        return summary
    
    def nested_cross_validation(self, model, param_grid, X, y, 
                              scoring='accuracy'):
        """嵌套交叉验证"""
        outer_cv = self.cv
        inner_cv = KFold(n_splits=3, shuffle=True, 
                        random_state=self.random_state)
        
        # 外层交叉验证
        nested_scores = []
        
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 内层交叉验证用于超参数调优
            grid_search = GridSearchCV(model, param_grid, cv=inner_cv,
                                      scoring=scoring)
            grid_search.fit(X_train, y_train)
            
            # 使用最佳模型在测试集上评估
            score = grid_search.score(X_test, y_test)
            nested_scores.append(score)
        
        return {
            'nested_scores': nested_scores,
            'mean': np.mean(nested_scores),
            'std': np.std(nested_scores)
        }

# 自定义交叉验证
class CustomCrossValidation:
    """自定义交叉验证策略"""
    
    @staticmethod
    def time_based_split(df, date_column, n_splits=5):
        """基于时间的分割"""
        df = df.sort_values(date_column)
        split_size = len(df) // n_splits
        
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = min(test_start + split_size, len(df))
            
            train_idx = df.index[:train_end]
            test_idx = df.index[test_start:test_end]
            
            yield train_idx, test_idx
    
    @staticmethod
    def blocked_time_series_split(df, block_size=30):
        """块时间序列分割"""
        n_samples = len(df)
        n_blocks = n_samples // block_size
        
        for i in range(2, n_blocks):
            train_blocks = i
            test_blocks = 1
            
            train_end = train_blocks * block_size
            test_start = train_end
            test_end = test_start + test_blocks * block_size
            
            if test_end > n_samples:
                break
            
            yield range(train_end), range(test_start, test_end)
    
    @staticmethod
    def stratified_group_split(X, y, groups, n_splits=5):
        """分层分组分割"""
        from sklearn.model_selection import StratifiedGroupKFold
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        return sgkf.split(X, y, groups)
```

## 3. 高级评估指标 📈

```python
class AdvancedMetrics:
    """高级评估指标"""
    
    @staticmethod
    def cohen_kappa_score(y_true, y_pred):
        """Cohen's Kappa系数"""
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(y_true, y_pred)
    
    @staticmethod
    def matthews_corrcoef(y_true, y_pred):
        """Matthews相关系数"""
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y_true, y_pred)
    
    @staticmethod
    def log_loss(y_true, y_proba):
        """对数损失"""
        from sklearn.metrics import log_loss
        return log_loss(y_true, y_proba)
    
    @staticmethod
    def brier_score(y_true, y_proba):
        """Brier分数"""
        from sklearn.metrics import brier_score_loss
        return brier_score_loss(y_true, y_proba)
    
    @staticmethod
    def average_precision_score(y_true, y_score):
        """平均精度分数"""
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, y_score)
    
    @staticmethod
    def balanced_accuracy(y_true, y_pred):
        """平衡准确率"""
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred)
    
    @staticmethod
    def hamming_loss(y_true, y_pred):
        """汉明损失（多标签分类）"""
        from sklearn.metrics import hamming_loss
        return hamming_loss(y_true, y_pred)
    
    @staticmethod
    def jaccard_score(y_true, y_pred):
        """Jaccard相似系数"""
        from sklearn.metrics import jaccard_score
        return jaccard_score(y_true, y_pred, average='weighted')

class RegressionMetrics:
    """回归评估指标"""
    
    @staticmethod
    def mean_squared_log_error(y_true, y_pred):
        """均方对数误差"""
        from sklearn.metrics import mean_squared_log_error
        return mean_squared_log_error(y_true, y_pred)
    
    @staticmethod
    def median_absolute_error(y_true, y_pred):
        """中位数绝对误差"""
        from sklearn.metrics import median_absolute_error
        return median_absolute_error(y_true, y_pred)
    
    @staticmethod
    def max_error(y_true, y_pred):
        """最大误差"""
        from sklearn.metrics import max_error
        return max_error(y_true, y_pred)
    
    @staticmethod
    def mean_tweedie_deviance(y_true, y_pred, power=0):
        """Tweedie偏差"""
        from sklearn.metrics import mean_tweedie_deviance
        return mean_tweedie_deviance(y_true, y_pred, power=power)
    
    @staticmethod
    def mean_pinball_loss(y_true, y_pred, alpha=0.5):
        """Pinball损失（分位数损失）"""
        from sklearn.metrics import mean_pinball_loss
        return mean_pinball_loss(y_true, y_pred, alpha=alpha)
    
    @staticmethod
    def adjusted_r2_score(y_true, y_pred, n_features):
        """调整R²分数"""
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2

class RankingMetrics:
    """排序评估指标"""
    
    @staticmethod
    def ndcg_score(y_true, y_score, k=None):
        """NDCG分数"""
        from sklearn.metrics import ndcg_score
        return ndcg_score(y_true, y_score, k=k)
    
    @staticmethod
    def dcg_score(y_true, y_score, k=None):
        """DCG分数"""
        from sklearn.metrics import dcg_score
        return dcg_score(y_true, y_score, k=k)
    
    @staticmethod
    def mean_reciprocal_rank(y_true, y_score):
        """平均倒数排名"""
        n_samples = len(y_true)
        mrr = 0
        
        for i in range(n_samples):
            # 获取排序索引
            sorted_indices = np.argsort(y_score[i])[::-1]
            # 找到第一个正确答案的位置
            for rank, idx in enumerate(sorted_indices, 1):
                if y_true[i][idx] == 1:
                    mrr += 1.0 / rank
                    break
        
        return mrr / n_samples
```

## 4. 可视化评估 📊

```python
class EvaluationVisualizer:
    """评估结果可视化"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_roc_curve(self, y_true, y_proba, title='ROC Curve'):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    
    def plot_precision_recall_curve(self, y_true, y_proba):
        """绘制精确率-召回率曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {'precision': precision, 'recall': recall}
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        return cm
    
    def plot_learning_curve(self, model, X, y, cv=5):
        """绘制学习曲线"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 
                'o-', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 
                'o-', label='Validation Score')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1)
        plt.fill_between(train_sizes,
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_validation_curve(self, model, X, y, param_name, 
                            param_range, cv=5):
        """绘制验证曲线"""
        from sklearn.model_selection import validation_curve
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name,
            param_range=param_range, cv=cv
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, np.mean(train_scores, axis=1), 
                'o-', label='Training Score')
        plt.plot(param_range, np.mean(val_scores, axis=1), 
                'o-', label='Validation Score')
        plt.fill_between(param_range,
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1)
        plt.fill_between(param_range,
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1)
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title('Validation Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """绘制残差图"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 残差 vs 预测值
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # QQ图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # 残差直方图
        axes[1, 0].hist(residuals, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        # 实际值 vs 预测值
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 1].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()
```

## 5. 模型比较 🔍

```python
class ModelComparison:
    """模型比较工具"""
    
    def __init__(self):
        self.results = {}
    
    def compare_models(self, models, X, y, cv=5, scoring='accuracy'):
        """比较多个模型"""
        results = []
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results.append({
                'Model': name,
                'Mean Score': np.mean(scores),
                'Std Score': np.std(scores),
                'Min Score': np.min(scores),
                'Max Score': np.max(scores)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Mean Score', ascending=False)
        
        self.results = results_df
        return results_df
    
    def statistical_comparison(self, model1_scores, model2_scores):
        """统计比较两个模型"""
        from scipy import stats
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Wilcoxon符号秩检验
        w_stat, w_p_value = stats.wilcoxon(model1_scores, model2_scores)
        
        # 计算效应量
        effect_size = np.mean(model1_scores - model2_scores) / np.std(model1_scores - model2_scores)
        
        return {
            't_statistic': t_stat,
            't_p_value': p_value,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': w_p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }
    
    def plot_model_comparison(self):
        """可视化模型比较"""
        if self.results.empty:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 条形图
        axes[0].bar(self.results['Model'], self.results['Mean Score'])
        axes[0].errorbar(range(len(self.results)), 
                        self.results['Mean Score'],
                        yerr=self.results['Std Score'],
                        fmt='none', color='black', capsize=5)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 箱线图（如果有详细分数）
        if hasattr(self, 'detailed_scores'):
            axes[1].boxplot(self.detailed_scores.values(), 
                          labels=self.detailed_scores.keys())
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Score Distribution')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def mcnemar_test(self, y_true, model1_pred, model2_pred):
        """McNemar检验（用于分类模型）"""
        from statsmodels.stats.contingency_tables import mcnemar
        
        # 构建列联表
        correct1_correct2 = np.sum((model1_pred == y_true) & (model2_pred == y_true))
        correct1_wrong2 = np.sum((model1_pred == y_true) & (model2_pred != y_true))
        wrong1_correct2 = np.sum((model1_pred != y_true) & (model2_pred == y_true))
        wrong1_wrong2 = np.sum((model1_pred != y_true) & (model2_pred != y_true))
        
        table = [[correct1_correct2, correct1_wrong2],
                [wrong1_correct2, wrong1_wrong2]]
        
        result = mcnemar(table, exact=True)
        
        return {
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05
        }
```

## 6. 偏差方差分析 ⚖️

```python
class BiasVarianceAnalysis:
    """偏差方差分析"""
    
    def __init__(self, model, n_iterations=100):
        self.model = model
        self.n_iterations = n_iterations
    
    def decompose(self, X, y, test_size=0.3):
        """偏差方差分解"""
        from sklearn.model_selection import train_test_split
        from sklearn.base import clone
        
        predictions = []
        
        for i in range(self.n_iterations):
            # 随机分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i
            )
            
            # 克隆并训练模型
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            
            # 预测
            y_pred = model_clone.predict(X_test)
            predictions.append(y_pred)
        
        # 转换为数组
        predictions = np.array(predictions)
        
        # 计算偏差和方差
        # 偏差：平均预测与真实值的差异
        avg_predicted = np.mean(predictions, axis=0)
        bias = np.mean((avg_predicted - y_test) ** 2)
        
        # 方差：预测的变异性
        variance = np.mean(np.var(predictions, axis=0))
        
        # 总误差
        mse = np.mean([mean_squared_error(y_test, pred) 
                      for pred in predictions])
        
        return {
            'bias': bias,
            'variance': variance,
            'total_error': mse,
            'irreducible_error': mse - bias - variance
        }
    
    def plot_bias_variance_tradeoff(self, X, y, complexity_param, 
                                   param_range):
        """绘制偏差方差权衡"""
        bias_scores = []
        variance_scores = []
        
        for param in param_range:
            # 设置模型复杂度参数
            setattr(self.model, complexity_param, param)
            
            # 计算偏差和方差
            results = self.decompose(X, y)
            bias_scores.append(results['bias'])
            variance_scores.append(results['variance'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, bias_scores, 'o-', label='Bias²')
        plt.plot(param_range, variance_scores, 'o-', label='Variance')
        plt.plot(param_range, np.array(bias_scores) + np.array(variance_scores),
                'o-', label='Total Error')
        plt.xlabel(complexity_param)
        plt.ylabel('Error')
        plt.title('Bias-Variance Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

## 7. 业务指标评估 💼

```python
class BusinessMetrics:
    """业务指标评估"""
    
    @staticmethod
    def calculate_lift(y_true, y_proba, percentile=10):
        """计算提升度"""
        # 按概率排序
        sorted_indices = np.argsort(y_proba)[::-1]
        n_top = int(len(y_true) * percentile / 100)
        
        # 顶部percentile的正例率
        top_positive_rate = np.mean(y_true[sorted_indices[:n_top]])
        
        # 整体正例率
        overall_positive_rate = np.mean(y_true)
        
        # 提升度
        lift = top_positive_rate / overall_positive_rate
        
        return lift
    
    @staticmethod
    def calculate_gain(y_true, y_proba):
        """计算增益图数据"""
        # 按概率排序
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # 累积正例
        cumulative_positives = np.cumsum(y_true_sorted)
        total_positives = np.sum(y_true)
        
        # 增益
        gain = cumulative_positives / total_positives
        
        # 百分比
        percentile = np.arange(1, len(y_true) + 1) / len(y_true)
        
        return percentile, gain
    
    @staticmethod
    def expected_calibration_error(y_true, y_proba, n_bins=10):
        """期望校准误差"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def profit_curve(y_true, y_proba, cost_benefit_matrix):
        """利润曲线"""
        # cost_benefit_matrix: [[TP_benefit, FP_cost], [FN_cost, TN_benefit]]
        
        thresholds = np.sort(np.unique(y_proba))[::-1]
        profits = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            profit = (tp * cost_benefit_matrix[0][0] +
                     fp * cost_benefit_matrix[0][1] +
                     fn * cost_benefit_matrix[1][0] +
                     tn * cost_benefit_matrix[1][1])
            
            profits.append(profit)
        
        return thresholds, profits
```

## 8. 时间序列评估 📅

```python
class TimeSeriesEvaluation:
    """时间序列模型评估"""
    
    @staticmethod
    def walk_forward_validation(model, data, window_size, horizon):
        """向前滚动验证"""
        predictions = []
        actuals = []
        
        for i in range(window_size, len(data) - horizon):
            # 训练窗口
            train_data = data[i-window_size:i]
            
            # 训练模型
            model.fit(train_data)
            
            # 预测
            pred = model.predict(horizon)
            predictions.append(pred)
            
            # 实际值
            actual = data[i:i+horizon]
            actuals.append(actual)
        
        return np.array(predictions), np.array(actuals)
    
    @staticmethod
    def time_series_cv_score(model, data, cv_splits, scoring_func):
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = []
        
        for train_index, test_index in tscv.split(data):
            train, test = data[train_index], data[test_index]
            
            model.fit(train)
            predictions = model.predict(len(test))
            
            score = scoring_func(test, predictions)
            scores.append(score)
        
        return scores
    
    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """方向准确率"""
        # 计算变化方向
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # 计算准确率
        accuracy = np.mean(true_direction == pred_direction)
        
        return accuracy
    
    @staticmethod
    def mase(y_true, y_pred, y_train):
        """平均绝对标准误差"""
        n = len(y_true)
        d = np.abs(np.diff(y_train)).sum() / (len(y_train) - 1)
        
        errors = np.abs(y_true - y_pred)
        mase = errors.mean() / d
        
        return mase
```

## 最佳实践总结 📝

```python
def evaluation_best_practices():
    """评估最佳实践"""
    
    practices = {
        "数据分割": [
            "使用分层采样保持类别比例",
            "时间序列数据不能随机分割",
            "保持验证集和测试集独立",
            "考虑数据泄露风险"
        ],
        
        "指标选择": [
            "不平衡数据使用F1、AUC而非准确率",
            "回归问题同时考虑MAE和RMSE",
            "业务场景决定指标权重",
            "使用多个指标综合评估"
        ],
        
        "交叉验证": [
            "选择合适的CV策略",
            "嵌套CV用于超参数调优",
            "时间序列使用前向验证",
            "分组数据使用GroupKFold"
        ],
        
        "统计检验": [
            "使用配对t检验比较模型",
            "多重比较需要校正",
            "注意样本量对显著性的影响",
            "报告效应量而非仅p值"
        ],
        
        "可视化": [
            "始终绘制学习曲线",
            "检查残差分布",
            "使用混淆矩阵理解错误",
            "绘制校准曲线"
        ]
    }
    
    return practices

# 常见陷阱
common_pitfalls = """
1. 数据泄露：在预处理时使用了测试集信息
2. 过拟合验证集：多次调参导致对验证集过拟合
3. 忽视类别不平衡：使用不当的评估指标
4. 单一指标：只关注一个指标忽略其他
5. 忽视统计显著性：小样本的偶然性
6. 错误的CV策略：时间序列用了随机CV
7. 忽视业务指标：模型指标好但业务价值低
"""

print("模型评估指南加载完成！")
```

## 下一步学习
- [特征工程](feature_engineering.md) - 提升模型输入质量
- [超参数调优](hyperparameter_tuning.md) - 优化模型参数
- [集成学习](ensemble.md) - 组合多个模型