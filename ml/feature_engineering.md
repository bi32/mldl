# 特征工程实战：数据的艺术 🎨

特征工程是机器学习中最重要的环节之一。好的特征胜过复杂的算法！

## 1. 特征工程概述 🌟

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, KBinsDiscretizer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.decomposition import PCA, TruncatedSVD, NMF
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringPipeline:
    """特征工程完整流程"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def analyze_dataset(self, df):
        """数据集分析"""
        print("=" * 50)
        print("数据集基本信息")
        print("=" * 50)
        print(f"样本数: {len(df)}")
        print(f"特征数: {len(df.columns)}")
        print(f"内存使用: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        print("\n数据类型分布:")
        print(df.dtypes.value_counts())
        
        print("\n缺失值统计:")
        missing = df.isnull().sum()
        missing_pct = 100 * missing / len(df)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False))
        
        print("\n数值特征统计:")
        print(df.describe())
        
        print("\n类别特征唯一值:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"{col}: {df[col].nunique()} unique values")
    
    def handle_missing_values(self, df, strategy='smart'):
        """处理缺失值"""
        df_processed = df.copy()
        
        if strategy == 'smart':
            # 数值特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    # 缺失比例
                    missing_pct = df[col].isnull().sum() / len(df)
                    
                    if missing_pct > 0.5:
                        # 缺失太多，考虑删除
                        print(f"Warning: {col} has {missing_pct:.2%} missing values")
                    elif missing_pct > 0.2:
                        # 创建缺失指示器
                        df_processed[f'{col}_was_missing'] = df[col].isnull().astype(int)
                        # 用中位数填充
                        df_processed[col].fillna(df[col].median(), inplace=True)
                    else:
                        # 用均值填充
                        df_processed[col].fillna(df[col].mean(), inplace=True)
            
            # 类别特征
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    # 用众数或"Unknown"填充
                    mode = df[col].mode()
                    if len(mode) > 0:
                        df_processed[col].fillna(mode[0], inplace=True)
                    else:
                        df_processed[col].fillna('Unknown', inplace=True)
        
        elif strategy == 'drop':
            df_processed = df.dropna()
        
        elif strategy == 'forward_fill':
            df_processed = df.fillna(method='ffill')
        
        elif strategy == 'backward_fill':
            df_processed = df.fillna(method='bfill')
        
        return df_processed
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """异常值检测"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = df[outlier_mask].index.tolist()
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > threshold
                outliers[col] = df[col].dropna()[outlier_mask].index.tolist()
            
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(df[[col]].dropna())
                outliers[col] = df[[col]].dropna()[outlier_pred == -1].index.tolist()
        
        return outliers
    
    def remove_outliers(self, df, outliers, strategy='cap'):
        """处理异常值"""
        df_processed = df.copy()
        
        for col, outlier_indices in outliers.items():
            if strategy == 'remove':
                df_processed = df_processed.drop(outlier_indices)
            
            elif strategy == 'cap':
                # 封顶处理
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            
            elif strategy == 'transform':
                # 对数变换
                if df[col].min() > 0:
                    df_processed[col] = np.log1p(df[col])
        
        return df_processed
```

## 2. 数值特征工程 🔢

```python
class NumericalFeatureEngineering:
    """数值特征工程"""
    
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
    
    def scale_features(self, df, method='standard'):
        """特征缩放"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_scaled = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'maxabs':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
        else:
            return df
        
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers[method] = scaler
        
        return df_scaled
    
    def create_polynomial_features(self, df, degree=2, include_bias=False):
        """多项式特征"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[numeric_cols])
        
        # 获取特征名
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # 创建DataFrame
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # 合并原始特征和多项式特征
        result = pd.concat([df.drop(columns=numeric_cols), poly_df], axis=1)
        
        return result
    
    def create_interaction_features(self, df, features_pairs):
        """交互特征"""
        df_inter = df.copy()
        
        for feat1, feat2 in features_pairs:
            # 乘法交互
            df_inter[f'{feat1}_times_{feat2}'] = df[feat1] * df[feat2]
            
            # 除法交互（避免除零）
            df_inter[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
            
            # 加法交互
            df_inter[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
            
            # 减法交互
            df_inter[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        return df_inter
    
    def create_statistical_features(self, df, numeric_cols, group_col=None):
        """统计特征"""
        df_stats = df.copy()
        
        if group_col:
            # 分组统计
            for col in numeric_cols:
                df_stats[f'{col}_mean_by_{group_col}'] = df.groupby(group_col)[col].transform('mean')
                df_stats[f'{col}_std_by_{group_col}'] = df.groupby(group_col)[col].transform('std')
                df_stats[f'{col}_max_by_{group_col}'] = df.groupby(group_col)[col].transform('max')
                df_stats[f'{col}_min_by_{group_col}'] = df.groupby(group_col)[col].transform('min')
                
                # 与组内均值的差异
                df_stats[f'{col}_diff_from_mean_{group_col}'] = \
                    df[col] - df_stats[f'{col}_mean_by_{group_col}']
        else:
            # 行统计
            df_stats['row_mean'] = df[numeric_cols].mean(axis=1)
            df_stats['row_std'] = df[numeric_cols].std(axis=1)
            df_stats['row_max'] = df[numeric_cols].max(axis=1)
            df_stats['row_min'] = df[numeric_cols].min(axis=1)
            df_stats['row_range'] = df_stats['row_max'] - df_stats['row_min']
            df_stats['row_sum'] = df[numeric_cols].sum(axis=1)
        
        return df_stats
    
    def binning_features(self, df, columns, n_bins=5, strategy='quantile'):
        """特征分箱"""
        df_binned = df.copy()
        
        for col in columns:
            # 分箱
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            df_binned[f'{col}_binned'] = kbd.fit_transform(df[[col]])
            
            # 创建分箱边界特征
            bins = kbd.bin_edges_[0]
            for i in range(len(bins)-1):
                df_binned[f'{col}_bin_{i}'] = ((df[col] >= bins[i]) & (df[col] < bins[i+1])).astype(int)
        
        return df_binned
    
    def transform_skewed_features(self, df, skew_threshold=0.5):
        """偏度变换"""
        from scipy.stats import skew
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_transformed = df.copy()
        
        for col in numeric_cols:
            col_skew = skew(df[col].dropna())
            
            if abs(col_skew) > skew_threshold:
                if col_skew > 0:  # 右偏
                    # 尝试log变换
                    if df[col].min() > 0:
                        df_transformed[f'{col}_log'] = np.log1p(df[col])
                    # 尝试平方根变换
                    if df[col].min() >= 0:
                        df_transformed[f'{col}_sqrt'] = np.sqrt(df[col])
                    # Box-Cox变换
                    from scipy.stats import boxcox
                    if df[col].min() > 0:
                        df_transformed[f'{col}_boxcox'], _ = boxcox(df[col])
                else:  # 左偏
                    # 平方变换
                    df_transformed[f'{col}_square'] = df[col] ** 2
                    # 指数变换
                    df_transformed[f'{col}_exp'] = np.exp(df[col])
        
        return df_transformed

# 时间特征工程
class TimeFeatureEngineering:
    """时间特征工程"""
    
    @staticmethod
    def extract_datetime_features(df, datetime_col):
        """提取日期时间特征"""
        df_time = df.copy()
        
        # 确保是datetime类型
        df_time[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # 基础特征
        df_time[f'{datetime_col}_year'] = df_time[datetime_col].dt.year
        df_time[f'{datetime_col}_month'] = df_time[datetime_col].dt.month
        df_time[f'{datetime_col}_day'] = df_time[datetime_col].dt.day
        df_time[f'{datetime_col}_hour'] = df_time[datetime_col].dt.hour
        df_time[f'{datetime_col}_minute'] = df_time[datetime_col].dt.minute
        df_time[f'{datetime_col}_second'] = df_time[datetime_col].dt.second
        df_time[f'{datetime_col}_dayofweek'] = df_time[datetime_col].dt.dayofweek
        df_time[f'{datetime_col}_dayofyear'] = df_time[datetime_col].dt.dayofyear
        df_time[f'{datetime_col}_weekofyear'] = df_time[datetime_col].dt.isocalendar().week
        df_time[f'{datetime_col}_quarter'] = df_time[datetime_col].dt.quarter
        
        # 是否是周末
        df_time[f'{datetime_col}_is_weekend'] = (df_time[f'{datetime_col}_dayofweek'] >= 5).astype(int)
        
        # 是否是月初/月末
        df_time[f'{datetime_col}_is_month_start'] = df_time[datetime_col].dt.is_month_start.astype(int)
        df_time[f'{datetime_col}_is_month_end'] = df_time[datetime_col].dt.is_month_end.astype(int)
        
        # 循环特征编码
        df_time[f'{datetime_col}_hour_sin'] = np.sin(2 * np.pi * df_time[f'{datetime_col}_hour'] / 24)
        df_time[f'{datetime_col}_hour_cos'] = np.cos(2 * np.pi * df_time[f'{datetime_col}_hour'] / 24)
        df_time[f'{datetime_col}_day_sin'] = np.sin(2 * np.pi * df_time[f'{datetime_col}_day'] / 31)
        df_time[f'{datetime_col}_day_cos'] = np.cos(2 * np.pi * df_time[f'{datetime_col}_day'] / 31)
        df_time[f'{datetime_col}_month_sin'] = np.sin(2 * np.pi * df_time[f'{datetime_col}_month'] / 12)
        df_time[f'{datetime_col}_month_cos'] = np.cos(2 * np.pi * df_time[f'{datetime_col}_month'] / 12)
        
        return df_time
    
    @staticmethod
    def create_lag_features(df, target_col, lag_periods=[1, 7, 30]):
        """创建滞后特征"""
        df_lag = df.copy()
        
        for lag in lag_periods:
            df_lag[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # 移动平均
        for window in [3, 7, 30]:
            df_lag[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df_lag[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df_lag[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df_lag[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        
        # 指数加权移动平均
        for alpha in [0.1, 0.3, 0.5]:
            df_lag[f'{target_col}_ewm_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
        
        return df_lag
```

## 3. 类别特征工程 🏷️

```python
class CategoricalFeatureEngineering:
    """类别特征工程"""
    
    def __init__(self):
        self.encoders = {}
        self.target_encoders = {}
    
    def label_encoding(self, df, columns):
        """标签编码"""
        df_encoded = df.copy()
        
        for col in columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        
        return df_encoded
    
    def onehot_encoding(self, df, columns, max_categories=10):
        """独热编码"""
        df_encoded = df.copy()
        
        for col in columns:
            # 限制类别数量
            top_categories = df[col].value_counts().head(max_categories).index
            df_encoded[col] = df[col].where(df[col].isin(top_categories), 'Other')
            
            # 独热编码
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
        
        return df_encoded
    
    def target_encoding(self, df, columns, target, smooth=5):
        """目标编码"""
        df_encoded = df.copy()
        
        for col in columns:
            # 计算每个类别的目标均值
            mean_target = df.groupby(col)[target].mean()
            
            # 平滑处理
            global_mean = df[target].mean()
            counts = df[col].value_counts()
            smooth_mean = (counts * mean_target + smooth * global_mean) / (counts + smooth)
            
            # 映射
            df_encoded[f'{col}_target_enc'] = df[col].map(smooth_mean)
            self.target_encoders[col] = smooth_mean
        
        return df_encoded
    
    def frequency_encoding(self, df, columns):
        """频率编码"""
        df_encoded = df.copy()
        
        for col in columns:
            freq = df[col].value_counts(normalize=True)
            df_encoded[f'{col}_freq'] = df[col].map(freq)
        
        return df_encoded
    
    def count_encoding(self, df, columns):
        """计数编码"""
        df_encoded = df.copy()
        
        for col in columns:
            counts = df[col].value_counts()
            df_encoded[f'{col}_count'] = df[col].map(counts)
        
        return df_encoded
    
    def binary_encoding(self, df, columns):
        """二进制编码"""
        import category_encoders as ce
        
        df_encoded = df.copy()
        
        for col in columns:
            encoder = ce.BinaryEncoder(cols=[col])
            encoded = encoder.fit_transform(df[[col]])
            
            # 重命名列
            encoded.columns = [f'{col}_bin_{i}' for i in range(len(encoded.columns))]
            
            # 合并
            df_encoded = pd.concat([df_encoded, encoded], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
        
        return df_encoded
    
    def embedding_encoding(self, df, col, embedding_dim=5):
        """嵌入编码（用于深度学习）"""
        from sklearn.decomposition import TruncatedSVD
        
        # 先进行独热编码
        dummies = pd.get_dummies(df[col])
        
        # SVD降维
        svd = TruncatedSVD(n_components=embedding_dim)
        embeddings = svd.fit_transform(dummies)
        
        # 创建DataFrame
        embedding_df = pd.DataFrame(
            embeddings, 
            columns=[f'{col}_emb_{i}' for i in range(embedding_dim)],
            index=df.index
        )
        
        return pd.concat([df, embedding_df], axis=1)

# 高级编码技术
class AdvancedEncoding:
    """高级编码技术"""
    
    @staticmethod
    def woe_encoding(df, feature, target):
        """WOE编码（Weight of Evidence）"""
        # 计算每个类别的好坏比例
        crosstab = pd.crosstab(df[feature], df[target])
        crosstab['Total'] = crosstab.sum(axis=1)
        crosstab['Bad_Rate'] = crosstab[1] / crosstab['Total']
        crosstab['Good_Rate'] = crosstab[0] / crosstab['Total']
        
        # 计算WOE
        crosstab['WOE'] = np.log(crosstab['Bad_Rate'] / crosstab['Good_Rate'])
        
        # 映射
        woe_dict = crosstab['WOE'].to_dict()
        df[f'{feature}_woe'] = df[feature].map(woe_dict)
        
        return df
    
    @staticmethod
    def james_stein_encoding(df, feature, target):
        """James-Stein编码"""
        # 计算全局均值
        global_mean = df[target].mean()
        
        # 计算每个类别的均值和方差
        group_stats = df.groupby(feature)[target].agg(['mean', 'var', 'count'])
        
        # James-Stein收缩
        k = len(group_stats)
        grand_var = df[target].var()
        
        shrinkage = 1 - (k - 3) * grand_var / group_stats['var'].sum()
        shrinkage = max(0, shrinkage)
        
        group_stats['js_estimate'] = shrinkage * group_stats['mean'] + (1 - shrinkage) * global_mean
        
        # 映射
        js_dict = group_stats['js_estimate'].to_dict()
        df[f'{feature}_js'] = df[feature].map(js_dict)
        
        return df
```

## 4. 特征选择 🎯

```python
class FeatureSelection:
    """特征选择方法"""
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
    
    def filter_method(self, X, y, method='mutual_info', k=10):
        """过滤法"""
        if method == 'mutual_info':
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                selector = SelectKBest(mutual_info_classif, k=k)
            else:
                from sklearn.feature_selection import mutual_info_regression
                selector = SelectKBest(mutual_info_regression, k=k)
        
        elif method == 'chi2':
            from sklearn.feature_selection import chi2
            selector = SelectKBest(chi2, k=k)
        
        elif method == 'f_score':
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                selector = SelectKBest(f_classif, k=k)
            else:
                from sklearn.feature_selection import f_regression
                selector = SelectKBest(f_regression, k=k)
        
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
        
        X_selected = selector.fit_transform(X, y)
        
        # 获取选中的特征
        if hasattr(X, 'columns'):
            self.selected_features = X.columns[selector.get_support()].tolist()
        
        return X_selected
    
    def wrapper_method(self, X, y, estimator, n_features=10):
        """包装法"""
        # RFE
        rfe = RFE(estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        
        # 获取选中的特征
        if hasattr(X, 'columns'):
            self.selected_features = X.columns[rfe.support_].tolist()
        
        # 特征排名
        self.feature_scores = dict(zip(X.columns, rfe.ranking_))
        
        return X_selected
    
    def embedded_method(self, X, y, estimator):
        """嵌入法"""
        # 使用L1正则化
        from sklearn.linear_model import LassoCV
        
        if isinstance(estimator, LassoCV):
            estimator.fit(X, y)
            selector = SelectFromModel(estimator, prefit=True)
        else:
            selector = SelectFromModel(estimator)
            selector.fit(X, y)
        
        X_selected = selector.transform(X)
        
        # 获取选中的特征
        if hasattr(X, 'columns'):
            self.selected_features = X.columns[selector.get_support()].tolist()
        
        return X_selected
    
    def permutation_importance(self, X, y, estimator, n_repeats=10):
        """排列重要性"""
        from sklearn.inspection import permutation_importance
        
        # 训练模型
        estimator.fit(X, y)
        
        # 计算排列重要性
        perm_importance = permutation_importance(
            estimator, X, y, n_repeats=n_repeats, random_state=42
        )
        
        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.feature_scores = importance_df.set_index('feature')['importance_mean'].to_dict()
        
        return importance_df
    
    def boruta_selection(self, X, y, estimator, max_iter=100):
        """Boruta特征选择"""
        from boruta import BorutaPy
        
        # Boruta特征选择
        boruta = BorutaPy(estimator, n_estimators='auto', max_iter=max_iter)
        boruta.fit(X.values, y.values)
        
        # 获取选中的特征
        self.selected_features = X.columns[boruta.support_].tolist()
        
        # 获取特征排名
        self.feature_scores = dict(zip(X.columns, boruta.ranking_))
        
        return X[self.selected_features]
    
    def shap_selection(self, X, y, estimator, threshold=0.01):
        """基于SHAP值的特征选择"""
        import shap
        
        # 训练模型
        estimator.fit(X, y)
        
        # 计算SHAP值
        explainer = shap.Explainer(estimator, X)
        shap_values = explainer(X)
        
        # 计算特征重要性
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        
        # 选择重要特征
        important_features = X.columns[feature_importance > threshold]
        self.selected_features = important_features.tolist()
        
        # 保存特征分数
        self.feature_scores = dict(zip(X.columns, feature_importance))
        
        return X[self.selected_features]
```

## 5. 特征变换 🔄

```python
class FeatureTransformation:
    """特征变换"""
    
    def __init__(self):
        self.transformers = {}
    
    def pca_transform(self, X, n_components=0.95, plot=True):
        """PCA降维"""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        self.transformers['pca'] = pca
        
        if plot:
            import matplotlib.pyplot as plt
            
            # 解释方差比
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(len(pca.explained_variance_ratio_)), 
                   pca.explained_variance_ratio_)
            plt.xlabel('Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance Ratio')
            
            plt.subplot(1, 2, 2)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance')
            plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return X_pca
    
    def lda_transform(self, X, y, n_components=None):
        """LDA降维"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y)
        
        self.transformers['lda'] = lda
        
        return X_lda
    
    def kernel_pca_transform(self, X, n_components=2, kernel='rbf'):
        """核PCA"""
        from sklearn.decomposition import KernelPCA
        
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        X_kpca = kpca.fit_transform(X)
        
        self.transformers['kpca'] = kpca
        
        return X_kpca
    
    def autoencoder_transform(self, X, encoding_dim=10, epochs=50):
        """自编码器降维"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        
        input_dim = X.shape[1]
        
        # 编码器
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # 解码器
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # 自编码器模型
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # 训练
        autoencoder.fit(X, X, epochs=epochs, batch_size=32, shuffle=True, verbose=0)
        
        # 编码器模型
        encoder = Model(input_layer, encoded)
        
        # 变换
        X_encoded = encoder.predict(X)
        
        self.transformers['autoencoder'] = encoder
        
        return X_encoded
    
    def nmf_transform(self, X, n_components=10):
        """非负矩阵分解"""
        # 确保非负
        X_positive = X - X.min() + 1e-10
        
        nmf = NMF(n_components=n_components, random_state=42)
        X_nmf = nmf.fit_transform(X_positive)
        
        self.transformers['nmf'] = nmf
        
        return X_nmf
    
    def tsne_transform(self, X, n_components=2):
        """t-SNE降维（主要用于可视化）"""
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=n_components, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        return X_tsne
    
    def umap_transform(self, X, n_components=2):
        """UMAP降维"""
        import umap
        
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_umap = reducer.fit_transform(X)
        
        self.transformers['umap'] = reducer
        
        return X_umap
```

## 6. 特征组合与生成 🔨

```python
class FeatureGeneration:
    """特征生成"""
    
    @staticmethod
    def generate_ratio_features(df, numerator_cols, denominator_cols):
        """生成比率特征"""
        df_ratio = df.copy()
        
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col:
                    df_ratio[f'{num_col}_div_{den_col}'] = df[num_col] / (df[den_col] + 1e-8)
        
        return df_ratio
    
    @staticmethod
    def generate_aggregation_features(df, group_cols, agg_cols, agg_funcs=['mean', 'std', 'max', 'min']):
        """生成聚合特征"""
        df_agg = df.copy()
        
        for group_col in group_cols:
            for agg_col in agg_cols:
                for func in agg_funcs:
                    agg_name = f'{agg_col}_{func}_by_{group_col}'
                    df_agg[agg_name] = df.groupby(group_col)[agg_col].transform(func)
        
        return df_agg
    
    @staticmethod
    def generate_diff_features(df, columns, periods=[1]):
        """生成差分特征"""
        df_diff = df.copy()
        
        for col in columns:
            for period in periods:
                df_diff[f'{col}_diff_{period}'] = df[col].diff(period)
                df_diff[f'{col}_pct_change_{period}'] = df[col].pct_change(period)
        
        return df_diff
    
    @staticmethod
    def generate_text_features(df, text_col):
        """生成文本特征"""
        df_text = df.copy()
        
        # 基础统计
        df_text[f'{text_col}_length'] = df[text_col].str.len()
        df_text[f'{text_col}_word_count'] = df[text_col].str.split().str.len()
        df_text[f'{text_col}_unique_word_count'] = df[text_col].apply(lambda x: len(set(str(x).split())))
        
        # 特殊字符
        df_text[f'{text_col}_punctuation_count'] = df[text_col].str.count('[^\w\s]')
        df_text[f'{text_col}_digit_count'] = df[text_col].str.count('\d')
        df_text[f'{text_col}_upper_count'] = df[text_col].str.count('[A-Z]')
        
        # 平均词长
        df_text[f'{text_col}_avg_word_length'] = df[text_col].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
        )
        
        return df_text
    
    @staticmethod
    def generate_domain_features(df, domain='retail'):
        """生成领域特定特征"""
        df_domain = df.copy()
        
        if domain == 'retail':
            # 零售领域
            if 'price' in df.columns and 'quantity' in df.columns:
                df_domain['total_amount'] = df['price'] * df['quantity']
            
            if 'customer_id' in df.columns:
                df_domain['customer_frequency'] = df.groupby('customer_id')['customer_id'].transform('count')
            
            if 'date' in df.columns:
                df_domain['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek.isin([5, 6]).astype(int)
                df_domain['is_holiday'] = 0  # 需要假期日历
        
        elif domain == 'finance':
            # 金融领域
            if 'income' in df.columns and 'debt' in df.columns:
                df_domain['debt_to_income'] = df['debt'] / (df['income'] + 1e-8)
            
            if 'credit_limit' in df.columns and 'balance' in df.columns:
                df_domain['utilization_rate'] = df['balance'] / (df['credit_limit'] + 1e-8)
        
        return df_domain
```

## 7. 自动化特征工程 🤖

```python
class AutoFeatureEngineering:
    """自动化特征工程"""
    
    def __init__(self):
        self.feature_importance = {}
        self.generated_features = []
    
    def auto_generate_features(self, df, target=None, max_features=100):
        """自动生成特征"""
        original_features = df.columns.tolist()
        df_auto = df.copy()
        
        # 1. 识别特征类型
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 2. 数值特征
        if len(numeric_cols) > 1:
            # 交互特征
            for i, col1 in enumerate(numeric_cols[:10]):  # 限制数量
                for col2 in numeric_cols[i+1:10]:
                    df_auto[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                    df_auto[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    
                    if len(df_auto.columns) > max_features:
                        break
        
        # 3. 类别特征
        for col in categorical_cols[:5]:  # 限制数量
            # 频率编码
            freq = df[col].value_counts(normalize=True)
            df_auto[f'{col}_freq'] = df[col].map(freq)
            
            # 如果有目标变量，做目标编码
            if target is not None:
                mean_target = df.groupby(col)[target].mean()
                df_auto[f'{col}_target_mean'] = df[col].map(mean_target)
        
        # 4. 时间特征
        for col in datetime_cols:
            df_auto[f'{col}_year'] = pd.to_datetime(df[col]).dt.year
            df_auto[f'{col}_month'] = pd.to_datetime(df[col]).dt.month
            df_auto[f'{col}_dayofweek'] = pd.to_datetime(df[col]).dt.dayofweek
        
        # 5. 统计特征
        if len(numeric_cols) > 2:
            df_auto['numeric_mean'] = df[numeric_cols].mean(axis=1)
            df_auto['numeric_std'] = df[numeric_cols].std(axis=1)
            df_auto['numeric_max'] = df[numeric_cols].max(axis=1)
            df_auto['numeric_min'] = df[numeric_cols].min(axis=1)
        
        self.generated_features = [col for col in df_auto.columns if col not in original_features]
        
        return df_auto
    
    def feature_tools_engineering(self, df, target_col=None, max_depth=2):
        """使用Featuretools自动特征工程"""
        import featuretools as ft
        
        # 创建实体集
        es = ft.EntitySet(id="data")
        
        # 添加实体
        es = es.add_dataframe(
            dataframe_name="main",
            dataframe=df,
            index="index"
        )
        
        # 深度特征合成
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="main",
            max_depth=max_depth,
            verbose=True
        )
        
        return feature_matrix
    
    def genetic_feature_selection(self, X, y, n_features=20, n_generations=50):
        """遗传算法特征选择"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        import random
        
        n_total_features = X.shape[1]
        population_size = 50
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            chromosome = [random.randint(0, 1) for _ in range(n_total_features)]
            # 确保至少有n_features个特征
            while sum(chromosome) < n_features:
                chromosome[random.randint(0, n_total_features-1)] = 1
            population.append(chromosome)
        
        # 进化
        for generation in range(n_generations):
            # 评估适应度
            fitness_scores = []
            for chromosome in population:
                selected_features = [i for i, gene in enumerate(chromosome) if gene == 1]
                if len(selected_features) > 0:
                    X_selected = X.iloc[:, selected_features]
                    score = cross_val_score(
                        RandomForestClassifier(n_estimators=10, random_state=42),
                        X_selected, y, cv=3
                    ).mean()
                    fitness_scores.append(score)
                else:
                    fitness_scores.append(0)
            
            # 选择
            sorted_population = [x for _, x in sorted(
                zip(fitness_scores, population), 
                key=lambda pair: pair[0], 
                reverse=True
            )]
            
            # 交叉和变异
            new_population = sorted_population[:10]  # 精英保留
            
            while len(new_population) < population_size:
                # 选择父母
                parent1 = random.choice(sorted_population[:20])
                parent2 = random.choice(sorted_population[:20])
                
                # 交叉
                crossover_point = random.randint(1, n_total_features-1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                
                # 变异
                if random.random() < 0.1:
                    mutation_point = random.randint(0, n_total_features-1)
                    child[mutation_point] = 1 - child[mutation_point]
                
                new_population.append(child)
            
            population = new_population
        
        # 返回最佳特征集
        best_chromosome = sorted_population[0]
        selected_features = [i for i, gene in enumerate(best_chromosome) if gene == 1]
        
        return X.iloc[:, selected_features]
```

## 8. 特征工程实战示例 🎯

```python
def feature_engineering_pipeline_example():
    """完整的特征工程流程示例"""
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'numeric1': np.random.randn(n_samples),
        'numeric2': np.random.exponential(2, n_samples),
        'numeric3': np.random.uniform(0, 100, n_samples),
        'category1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'text': ['text_' + str(i) for i in range(n_samples)],
        'target': np.random.randint(0, 2, n_samples)
    })
    
    # 添加一些缺失值
    df.loc[np.random.choice(df.index, 50), 'numeric2'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'category1'] = np.nan
    
    print("原始数据集形状:", df.shape)
    
    # 1. 初始化特征工程pipeline
    fe_pipeline = FeatureEngineeringPipeline()
    
    # 2. 数据分析
    fe_pipeline.analyze_dataset(df)
    
    # 3. 处理缺失值
    df = fe_pipeline.handle_missing_values(df)
    
    # 4. 异常值检测和处理
    outliers = fe_pipeline.detect_outliers(df[['numeric1', 'numeric2', 'numeric3']])
    df = fe_pipeline.remove_outliers(df, outliers, strategy='cap')
    
    # 5. 数值特征工程
    num_fe = NumericalFeatureEngineering()
    
    # 创建交互特征
    df = num_fe.create_interaction_features(df, [('numeric1', 'numeric2'), ('numeric2', 'numeric3')])
    
    # 创建统计特征
    df = num_fe.create_statistical_features(df, ['numeric1', 'numeric2', 'numeric3'])
    
    # 6. 类别特征工程
    cat_fe = CategoricalFeatureEngineering()
    
    # 目标编码
    df = cat_fe.target_encoding(df, ['category1', 'category2'], 'target')
    
    # 频率编码
    df = cat_fe.frequency_encoding(df, ['category1', 'category2'])
    
    # 7. 时间特征工程
    time_fe = TimeFeatureEngineering()
    df = time_fe.extract_datetime_features(df, 'date')
    
    # 8. 特征选择
    X = df.drop(['target', 'date', 'text', 'category1', 'category2'], axis=1)
    y = df['target']
    
    selector = FeatureSelection()
    
    # 使用随机森林进行特征重要性评估
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    importance_df = selector.permutation_importance(X, y, rf)
    
    print("\n特征重要性 Top 10:")
    print(importance_df.head(10))
    
    # 选择最重要的特征
    X_selected = selector.filter_method(X, y, method='mutual_info', k=15)
    
    print(f"\n最终特征数量: {X_selected.shape[1]}")
    
    # 9. 特征缩放
    X_scaled = num_fe.scale_features(pd.DataFrame(X_selected), method='standard')
    
    # 10. 模型评估
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"\n交叉验证AUC分数: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return df

# 运行示例
result_df = feature_engineering_pipeline_example()
```

## 最佳实践总结 📋

```python
def feature_engineering_best_practices():
    """特征工程最佳实践"""
    
    practices = {
        "数据理解": [
            "深入理解业务背景和数据含义",
            "与领域专家合作",
            "进行EDA（探索性数据分析）",
            "识别数据质量问题"
        ],
        
        "特征创建": [
            "基于业务逻辑创建特征",
            "考虑特征之间的交互",
            "创建聚合特征和统计特征",
            "利用时间序列特征",
            "适当使用多项式特征"
        ],
        
        "特征编码": [
            "类别特征：根据基数选择编码方式",
            "高基数：目标编码、嵌入",
            "低基数：独热编码",
            "有序类别：序数编码"
        ],
        
        "特征选择": [
            "移除常数特征",
            "移除高度相关的特征",
            "使用多种方法验证特征重要性",
            "考虑特征的可解释性"
        ],
        
        "数据泄露预防": [
            "确保特征不包含未来信息",
            "正确处理时间序列数据",
            "验证集和测试集分开处理",
            "目标编码使用交叉验证"
        ],
        
        "性能优化": [
            "减少特征数量以加快训练",
            "使用稀疏矩阵存储独热编码",
            "并行化特征生成过程",
            "缓存计算密集的特征"
        ]
    }
    
    return practices

# 常见陷阱
common_pitfalls = """
1. 数据泄露：使用了包含目标信息的特征
2. 过度工程：创建太多无用特征
3. 忽视缺失值：不当处理缺失值
4. 标准化时机：在分割数据前进行标准化
5. 维度灾难：特征过多导致过拟合
6. 忽视特征分布：不处理偏态分布
7. 类别不一致：训练集和测试集类别不匹配
8. 时间特征处理不当：不考虑周期性
"""

print("特征工程最佳实践加载完成！")
```

## 下一步学习
- [模型评估](evaluation.md) - 正确评估模型性能
- [超参数调优](hyperparameter_tuning.md) - 优化模型参数
- [集成学习](ensemble.md) - 组合多个模型