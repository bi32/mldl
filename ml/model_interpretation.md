# 模型解释与可解释性 🔍

理解机器学习模型的决策过程，让AI不再是黑盒。

## 1. 模型可解释性概述 🌟

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class ModelInterpretability:
    """模型可解释性基础"""
    
    def __init__(self):
        self.interpretation_methods = {
            "内在可解释": ["线性模型", "决策树", "规则学习"],
            "事后解释": ["LIME", "SHAP", "Anchor", "CounterfactualExplanations"],
            "全局解释": ["特征重要性", "部分依赖图", "ALE图"],
            "局部解释": ["LIME", "SHAP值", "反事实解释"]
        }
    
    def explain_linear_model(self, model, feature_names):
        """解释线性模型"""
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            if len(coefficients.shape) > 1:
                coefficients = coefficients[0]
            
            # 创建系数DataFrame
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            })
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            
            # 可视化
            plt.figure(figsize=(10, 6))
            plt.barh(coef_df['Feature'][:15], coef_df['Coefficient'][:15])
            plt.xlabel('Coefficient Value')
            plt.title('Linear Model Feature Coefficients')
            plt.tight_layout()
            plt.show()
            
            return coef_df
    
    def explain_tree_model(self, model, feature_names):
        """解释树模型"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 创建重要性DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # 可视化
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
            plt.xlabel('Feature Importance')
            plt.title('Tree Model Feature Importances')
            plt.tight_layout()
            plt.show()
            
            return importance_df
```

## 2. SHAP (SHapley Additive exPlanations) 💡

```python
import shap

class SHAPExplainer:
    """SHAP解释器"""
    
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self):
        """创建SHAP解释器"""
        model_type = type(self.model).__name__
        
        if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type:
            # 树模型使用TreeExplainer
            self.explainer = shap.TreeExplainer(self.model)
        elif 'Linear' in model_type:
            # 线性模型使用LinearExplainer
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        else:
            # 其他模型使用KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict, 
                shap.sample(self.X_train, 100)
            )
        
        return self.explainer
    
    def explain_global(self, X):
        """全局解释"""
        if self.explainer is None:
            self.create_explainer()
        
        # 计算SHAP值
        self.shap_values = self.explainer.shap_values(X)
        
        # 如果是多类分类，选择第一个类
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        # 摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X, show=False)
        plt.tight_layout()
        plt.show()
        
        # 特征重要性条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.show()
        
        return self.shap_values
    
    def explain_instance(self, instance_idx, X):
        """解释单个实例"""
        if self.shap_values is None:
            self.explain_global(X)
        
        # 瀑布图
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=X.iloc[instance_idx] if hasattr(X, 'iloc') else X[instance_idx],
                feature_names=X.columns.tolist() if hasattr(X, 'columns') else None
            )
        )
        
        # 力图
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[instance_idx],
            X.iloc[instance_idx] if hasattr(X, 'iloc') else X[instance_idx],
            matplotlib=True
        )
        plt.show()
    
    def dependence_plots(self, X, feature_names=None):
        """依赖图"""
        if self.shap_values is None:
            self.explain_global(X)
        
        if feature_names is None:
            feature_names = X.columns if hasattr(X, 'columns') else range(X.shape[1])
        
        # 为前4个最重要的特征绘制依赖图
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_features = np.argsort(feature_importance)[-4:]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, feature_idx in enumerate(top_features):
            shap.dependence_plot(
                feature_idx, 
                self.shap_values, 
                X,
                ax=axes[idx],
                show=False
            )
        
        plt.tight_layout()
        plt.show()
    
    def interaction_effects(self, X):
        """交互效应分析"""
        if self.explainer is None:
            self.create_explainer()
        
        # 计算交互SHAP值
        shap_interaction_values = self.explainer.shap_interaction_values(X)
        
        # 如果是多类分类，选择第一个类
        if isinstance(shap_interaction_values, list):
            shap_interaction_values = shap_interaction_values[0]
        
        # 绘制交互效应热图
        mean_interaction = np.abs(shap_interaction_values).mean(axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_interaction, annot=True, fmt='.3f', cmap='coolwarm')
        plt.title('SHAP Interaction Values')
        plt.show()
        
        return shap_interaction_values
```

## 3. LIME (Local Interpretable Model-agnostic Explanations) 🎯

```python
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer

class LIMEExplainer:
    """LIME解释器"""
    
    def __init__(self, training_data, feature_names, class_names=None, mode='classification'):
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.explainer = None
    
    def create_tabular_explainer(self):
        """创建表格数据解释器"""
        self.explainer = LimeTabularExplainer(
            self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            discretize_continuous=True
        )
        return self.explainer
    
    def explain_instance_tabular(self, model, instance, num_features=10):
        """解释表格数据实例"""
        if self.explainer is None:
            self.create_tabular_explainer()
        
        # 生成解释
        if self.mode == 'classification':
            exp = self.explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=num_features
            )
        else:
            exp = self.explainer.explain_instance(
                instance,
                model.predict,
                num_features=num_features
            )
        
        # 显示解释
        exp.show_in_notebook(show_table=True)
        
        # 获取解释作为列表
        explanation_list = exp.as_list()
        
        # 可视化
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.show()
        
        return exp
    
    def explain_text(self, model, text, num_features=10):
        """解释文本分类"""
        text_explainer = LimeTextExplainer(class_names=self.class_names)
        
        exp = text_explainer.explain_instance(
            text,
            model.predict_proba,
            num_features=num_features
        )
        
        # 显示解释
        exp.show_in_notebook(text=True)
        
        return exp
    
    def explain_image(self, model, image, num_samples=1000):
        """解释图像分类"""
        image_explainer = LimeImageExplainer()
        
        exp = image_explainer.explain_instance(
            image,
            model.predict_proba,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples
        )
        
        # 显示解释
        from skimage.segmentation import mark_boundaries
        
        temp, mask = exp.get_image_and_mask(
            exp.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        plt.figure(figsize=(10, 10))
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.axis('off')
        plt.show()
        
        return exp
```

## 4. 特征重要性分析 📊

```python
class FeatureImportanceAnalysis:
    """特征重要性分析"""
    
    def __init__(self):
        self.importance_scores = {}
    
    def permutation_importance(self, model, X, y, n_repeats=10):
        """排列重要性"""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42
        )
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns if hasattr(X, 'columns') else range(X.shape[1]),
            'Importance': result.importances_mean,
            'Std': result.importances_std
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            importance_df['Feature'][:15],
            importance_df['Importance'][:15],
            yerr=importance_df['Std'][:15],
            fmt='o',
            capsize=5
        )
        plt.xlabel('Feature')
        plt.ylabel('Permutation Importance')
        plt.title('Feature Permutation Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        self.importance_scores['permutation'] = importance_df
        return importance_df
    
    def drop_column_importance(self, model, X, y, cv=5):
        """删除列重要性"""
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        
        # 基准分数
        base_score = cross_val_score(
            clone(model), X, y, cv=cv, scoring='accuracy'
        ).mean()
        
        importances = []
        
        for col in X.columns if hasattr(X, 'columns') else range(X.shape[1]):
            # 删除该列
            X_dropped = X.drop(columns=[col]) if hasattr(X, 'drop') else np.delete(X, col, axis=1)
            
            # 计算分数
            score = cross_val_score(
                clone(model), X_dropped, y, cv=cv, scoring='accuracy'
            ).mean()
            
            # 重要性 = 基准分数 - 删除后分数
            importance = base_score - score
            importances.append(importance)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns if hasattr(X, 'columns') else range(X.shape[1]),
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        self.importance_scores['drop_column'] = importance_df
        return importance_df
    
    def compare_importance_methods(self):
        """比较不同重要性方法"""
        if len(self.importance_scores) < 2:
            print("需要至少两种重要性计算方法")
            return
        
        # 合并所有重要性分数
        merged_df = None
        for method, df in self.importance_scores.items():
            if merged_df is None:
                merged_df = df[['Feature', 'Importance']].rename(
                    columns={'Importance': method}
                )
            else:
                merged_df = merged_df.merge(
                    df[['Feature', 'Importance']].rename(
                        columns={'Importance': method}
                    ),
                    on='Feature'
                )
        
        # 归一化
        for col in merged_df.columns[1:]:
            merged_df[col] = (merged_df[col] - merged_df[col].min()) / \
                            (merged_df[col].max() - merged_df[col].min())
        
        # 可视化
        merged_df.set_index('Feature').plot(kind='bar', figsize=(12, 6))
        plt.title('Feature Importance Comparison')
        plt.ylabel('Normalized Importance')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.show()
        
        return merged_df
```

## 5. 部分依赖图 (PDP) 和 ALE 图 📈

```python
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

class PartialDependenceAnalysis:
    """部分依赖分析"""
    
    def __init__(self, model, X, feature_names):
        self.model = model
        self.X = X
        self.feature_names = feature_names
    
    def plot_partial_dependence(self, features, kind='both'):
        """绘制部分依赖图"""
        # 创建部分依赖显示
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            features,
            kind=kind,  # 'average', 'individual', or 'both'
            subsample=50,
            n_jobs=-1,
            grid_resolution=20
        )
        
        display.figure_.suptitle('Partial Dependence Plots')
        display.figure_.set_size_inches(14, 8)
        plt.tight_layout()
        plt.show()
        
        return display
    
    def plot_2d_partial_dependence(self, feature_pairs):
        """绘制2D部分依赖图"""
        fig, axes = plt.subplots(1, len(feature_pairs), figsize=(6*len(feature_pairs), 5))
        
        if len(feature_pairs) == 1:
            axes = [axes]
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            # 计算2D部分依赖
            pd_result = partial_dependence(
                self.model,
                X=self.X,
                features=[(feat1, feat2)],
                grid_resolution=20
            )
            
            # 绘制等高线图
            XX, YY = np.meshgrid(pd_result['values'][0], pd_result['values'][1])
            Z = pd_result['average'][0].T
            
            contour = axes[idx].contourf(XX, YY, Z, levels=20, cmap='RdBu_r')
            axes[idx].set_xlabel(self.feature_names[feat1])
            axes[idx].set_ylabel(self.feature_names[feat2])
            axes[idx].set_title(f'2D PDP: {self.feature_names[feat1]} vs {self.feature_names[feat2]}')
            plt.colorbar(contour, ax=axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    def accumulated_local_effects(self, feature_idx, n_bins=10):
        """累积局部效应（ALE）图"""
        feature_values = self.X[:, feature_idx] if isinstance(self.X, np.ndarray) else self.X.iloc[:, feature_idx]
        
        # 创建分箱
        bins = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(feature_values, bins) - 1
        
        ale_values = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            # 获取该箱中的样本
            X_bin = self.X[mask]
            
            # 计算上下边界的预测差异
            X_lower = X_bin.copy()
            X_upper = X_bin.copy()
            
            if isinstance(X_lower, pd.DataFrame):
                X_lower.iloc[:, feature_idx] = bins[i]
                X_upper.iloc[:, feature_idx] = bins[i + 1]
            else:
                X_lower[:, feature_idx] = bins[i]
                X_upper[:, feature_idx] = bins[i + 1]
            
            # 计算差异
            diff = self.model.predict(X_upper) - self.model.predict(X_lower)
            ale_values.append(diff.mean())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
        
        # 累积和中心化
        ale_cumsum = np.cumsum(ale_values)
        ale_centered = ale_cumsum - ale_cumsum.mean()
        
        # 绘制ALE图
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, ale_centered, 'o-', linewidth=2, markersize=8)
        plt.xlabel(self.feature_names[feature_idx] if self.feature_names else f'Feature {feature_idx}')
        plt.ylabel('ALE')
        plt.title(f'Accumulated Local Effects: {self.feature_names[feature_idx] if self.feature_names else f"Feature {feature_idx}"}')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return bin_centers, ale_centered
```

## 6. 反事实解释 🔄

```python
class CounterfactualExplanations:
    """反事实解释"""
    
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
    
    def find_counterfactual(self, instance, target_class, max_changes=3):
        """寻找反事实实例"""
        original_class = self.model.predict([instance])[0]
        
        if original_class == target_class:
            print("实例已经属于目标类别")
            return instance
        
        # 找到目标类别的最近邻
        target_instances = self.X_train[self.y_train == target_class]
        
        # 计算距离
        distances = np.sum((target_instances - instance) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_target = target_instances[nearest_idx]
        
        # 找到最小改变
        differences = nearest_target - instance
        important_features = np.argsort(np.abs(differences))[::-1]
        
        counterfactual = instance.copy()
        changes_made = []
        
        for i, feat_idx in enumerate(important_features):
            if i >= max_changes:
                break
            
            # 改变特征值
            old_value = counterfactual[feat_idx]
            counterfactual[feat_idx] = nearest_target[feat_idx]
            new_value = counterfactual[feat_idx]
            
            changes_made.append({
                'feature': feat_idx,
                'old_value': old_value,
                'new_value': new_value,
                'change': new_value - old_value
            })
            
            # 检查是否达到目标类别
            if self.model.predict([counterfactual])[0] == target_class:
                break
        
        return counterfactual, changes_made
    
    def diverse_counterfactuals(self, instance, target_class, n_counterfactuals=3):
        """生成多样化的反事实解释"""
        counterfactuals = []
        
        for _ in range(n_counterfactuals):
            # 添加随机性以获得不同的反事实
            noise = np.random.normal(0, 0.1, instance.shape)
            noisy_instance = instance + noise
            
            cf, changes = self.find_counterfactual(noisy_instance, target_class)
            counterfactuals.append({
                'counterfactual': cf,
                'changes': changes
            })
        
        return counterfactuals
    
    def actionable_recourse(self, instance, constraints):
        """可操作的补救措施"""
        # constraints: {'feature_name': {'min': val, 'max': val, 'mutable': bool}}
        
        original_prediction = self.model.predict([instance])[0]
        desired_class = 1 - original_prediction  # 假设二分类
        
        # 创建优化问题
        from scipy.optimize import minimize
        
        def objective(x):
            # 最小化改变
            return np.sum((x - instance) ** 2)
        
        def constraint(x):
            # 确保预测为期望类别
            pred_proba = self.model.predict_proba([x])[0]
            return pred_proba[desired_class] - 0.5
        
        # 应用约束
        bounds = []
        for i, (feat, const) in enumerate(constraints.items()):
            if const['mutable']:
                bounds.append((const['min'], const['max']))
            else:
                bounds.append((instance[i], instance[i]))
        
        # 优化
        result = minimize(
            objective,
            instance,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint}
        )
        
        if result.success:
            return result.x, self.model.predict([result.x])[0]
        else:
            return None, None
```

## 7. 模型诊断工具 🔧

```python
class ModelDiagnostics:
    """模型诊断工具"""
    
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
    
    def confusion_matrix_analysis(self, y_true, y_pred):
        """混淆矩阵分析"""
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 可视化
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
        
        # 详细分析
        n_classes = cm.shape[0]
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    print(f"类别 {i} 被误分类为类别 {j}: {cm[i, j]} 次")
        
        return cm
    
    def error_analysis(self, X, y_true, y_pred):
        """错误分析"""
        # 找出错误预测的样本
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            print("没有错误预测")
            return
        
        # 分析错误样本的特征
        error_samples = X[error_indices]
        correct_samples = X[~errors]
        
        # 特征分布比较
        if hasattr(X, 'columns'):
            feature_names = X.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_names[:6]):
            if hasattr(X, 'iloc'):
                error_feat = error_samples[feature]
                correct_feat = correct_samples[feature]
            else:
                error_feat = error_samples[:, idx]
                correct_feat = correct_samples[:, idx]
            
            axes[idx].hist([correct_feat, error_feat], 
                          label=['Correct', 'Error'],
                          alpha=0.7, bins=20)
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
        
        return error_indices
    
    def calibration_analysis(self, y_true, y_proba):
        """校准分析"""
        from sklearn.calibration import calibration_curve
        
        # 计算校准曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        # 绘制校准图
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 
                's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 计算ECE (Expected Calibration Error)
        ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
        print(f'Expected Calibration Error: {ece:.4f}')
        
        return ece
```

## 8. 实战示例 💼

```python
def interpretation_pipeline_example():
    """完整的模型解释流程示例"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # 加载数据
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("=" * 50)
    print("模型解释流程示例")
    print("=" * 50)
    
    # 1. 特征重要性
    print("\n1. 特征重要性分析")
    interpreter = ModelInterpretability()
    importance_df = interpreter.explain_tree_model(model, X.columns)
    print(importance_df.head(10))
    
    # 2. SHAP解释
    print("\n2. SHAP解释")
    shap_explainer = SHAPExplainer(model, X_train)
    shap_values = shap_explainer.explain_global(X_test)
    
    # 3. LIME解释
    print("\n3. LIME解释（单个实例）")
    lime_explainer = LIMEExplainer(
        X_train.values,
        X.columns.tolist(),
        class_names=['Malignant', 'Benign']
    )
    lime_exp = lime_explainer.explain_instance_tabular(
        model, X_test.iloc[0].values
    )
    
    # 4. 部分依赖图
    print("\n4. 部分依赖图")
    pdp_analyzer = PartialDependenceAnalysis(model, X_test, X.columns)
    pdp_analyzer.plot_partial_dependence([0, 1, 2, 3])
    
    # 5. 模型诊断
    print("\n5. 模型诊断")
    diagnostics = ModelDiagnostics(model, X_test, y_test)
    cm = diagnostics.confusion_matrix_analysis(y_test, y_pred)
    ece = diagnostics.calibration_analysis(y_test, y_proba)
    
    # 6. 反事实解释
    print("\n6. 反事实解释")
    cf_explainer = CounterfactualExplanations(model, X_train.values, y_train)
    cf_instance, changes = cf_explainer.find_counterfactual(
        X_test.iloc[0].values, 
        target_class=1-y_test.iloc[0]
    )
    print(f"需要的改变: {changes}")
    
    print("\n解释流程完成！")
    
    return model, shap_values

# 运行示例
model, shap_values = interpretation_pipeline_example()
```

## 最佳实践总结 📋

```python
def interpretability_best_practices():
    """可解释性最佳实践"""
    
    practices = {
        "选择方法": [
            "线性模型使用系数解释",
            "树模型使用特征重要性和SHAP",
            "深度学习使用LIME和注意力机制",
            "全局解释用SHAP，局部解释用LIME"
        ],
        
        "SHAP使用": [
            "TreeExplainer用于树模型（快速）",
            "KernelExplainer用于黑盒模型（慢）",
            "DeepExplainer用于深度学习",
            "使用summary_plot理解全局模式"
        ],
        
        "特征重要性": [
            "使用多种方法验证",
            "考虑特征相关性的影响",
            "排列重要性更可靠",
            "注意数据泄露"
        ],
        
        "可视化": [
            "部分依赖图展示特征效应",
            "SHAP瀑布图解释单个预测",
            "混淆矩阵分析错误模式",
            "校准图检查概率可靠性"
        ],
        
        "报告撰写": [
            "从业务角度解释",
            "提供具体案例",
            "使用简单的语言",
            "包含不确定性说明"
        ]
    }
    
    return practices

# 解释性检查清单
interpretation_checklist = """
模型解释检查清单：

□ 全局解释
  - 特征重要性排序
  - 特征之间的交互效应
  - 模型整体行为模式

□ 局部解释
  - 单个预测的解释
  - 关键特征的贡献
  - 反事实情况

□ 模型诊断
  - 校准性检查
  - 错误案例分析
  - 偏差检测

□ 可视化
  - 特征重要性图
  - 部分依赖图
  - SHAP摘要图
  - 混淆矩阵

□ 文档记录
  - 方法说明
  - 局限性说明
  - 业务解释
"""

print("模型解释指南加载完成！")
```

## 下一步学习
- [特征工程](feature_engineering.md) - 创建可解释的特征
- [模型评估](evaluation.md) - 全面评估模型性能
- [AutoML](automl.md) - 自动化机器学习