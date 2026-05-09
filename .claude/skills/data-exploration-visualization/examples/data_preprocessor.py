#!/usr/bin/env python3
"""
数据预处理器 (Data Preprocessor) - 智能数据清洗和转换模块

提供全面的数据预处理功能，包括：
- 数据清洗（缺失值处理、异常值检测和处理）
- 数据类型转换和标准化
- 特征工程（特征选择、创建、转换）
- 数据编码（标签编码、独热编码）
- 数据分割和平衡
- 数据质量评估和改进建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理器 - 智能数据清洗和特征工程引擎"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据预处理器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.preprocessing_steps = []
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.preprocessing_report = {}

        # 默认配置
        self.default_config = {
            'missing_threshold': 0.5,  # 缺失值阈值
            'outlier_method': 'isolation_forest',  # 异常值检测方法
            'outlier_contamination': 0.1,  # 异常值比例
            'scaling_method': 'standard',  # 标准化方法
            'encoding_method': 'auto',  # 编码方法
            'feature_selection': False,  # 是否进行特征选择
            'k_features': 10,  # 选择的特征数量
            'test_size': 0.2,  # 测试集比例
            'random_state': 42,  # 随机种子
            'balance_data': False,  # 是否平衡数据
            'balance_method': 'smote'  # 数据平衡方法
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        分析数据质量

        Parameters:
        - data: 数据DataFrame

        Returns:
        - 数据质量报告
        """
        print("🔍 分析数据质量...")

        quality_report = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'columns': {},
            'overall_score': 0,
            'issues': [],
            'recommendations': []
        }

        total_issues = 0
        total_checks = 0

        for col in data.columns:
            col_info = {
                'dtype': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'null_percentage': data[col].isnull().sum() / len(data) * 100,
                'unique_count': data[col].nunique(),
                'duplicate_count': data[col].duplicated().sum(),
                'issues': []
            }

            # 检查缺失值
            total_checks += 1
            if col_info['null_percentage'] > 0:
                total_issues += 1
                col_info['issues'].append(f"缺失值: {col_info['null_percentage']:.1f}%")

                if col_info['null_percentage'] > self.config['missing_threshold'] * 100:
                    quality_report['issues'].append(
                        f"列 '{col}' 缺失值过高 ({col_info['null_percentage']:.1f}%)"
                    )

            # 检查重复值
            if col_info['duplicate_count'] > 0:
                col_info['issues'].append(f"重复值: {col_info['duplicate_count']}")

            # 检查数据类型
            if data[col].dtype == 'object':
                # 检查可能的数值型分类变量
                try:
                    pd.to_numeric(data[col], errors='raise')
                    col_info['issues'].append("可能是数值型但存储为字符串")
                    quality_report['recommendations'].append(
                        f"考虑将列 '{col}' 转换为数值类型"
                    )
                except:
                    pass
            elif pd.api.types.is_numeric_dtype(data[col]):
                # 检查异常值
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

                if outliers > 0:
                    outlier_percentage = outliers / len(data) * 100
                    col_info['issues'].append(f"异常值: {outliers} ({outlier_percentage:.1f}%)")

            quality_report['columns'][col] = col_info

        # 计算整体质量分数
        quality_report['overall_score'] = max(0, 100 - (total_issues / total_checks) * 100)

        # 生成建议
        if quality_report['overall_score'] < 80:
            quality_report['recommendations'].append("数据质量较低，建议进行数据清洗")

        print(f"   ✓ 数据质量分析完成，质量分数: {quality_report['overall_score']:.1f}")
        return quality_report

    def clean_data(self, data: pd.DataFrame,
                   handle_missing: str = 'auto',
                   handle_outliers: str = 'auto',
                   handle_duplicates: bool = True) -> pd.DataFrame:
        """
        数据清洗

        Parameters:
        - data: 原始数据
        - handle_missing: 缺失值处理方法
        - handle_outliers: 异常值处理方法
        - handle_duplicates: 是否处理重复值

        Returns:
        - 清洗后的数据
        """
        print("🧹 开始数据清洗...")
        cleaned_data = data.copy()
        original_shape = cleaned_data.shape

        # 1. 处理重复值
        if handle_duplicates:
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = before_count - len(cleaned_data)
            if removed_duplicates > 0:
                print(f"   ✓ 移除了 {removed_duplicates} 个重复行")
                self.preprocessing_steps.append(f"移除重复值: {removed_duplicates} 行")

        # 2. 处理缺失值
        if handle_missing != 'none':
            cleaned_data = self._handle_missing_values(cleaned_data, handle_missing)

        # 3. 处理异常值
        if handle_outliers != 'none':
            cleaned_data = self._handle_outliers(cleaned_data, handle_outliers)

        final_shape = cleaned_data.shape
        print(f"   ✓ 数据清洗完成: {original_shape} -> {final_shape}")

        return cleaned_data

    def _handle_missing_values(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """处理缺失值"""
        print("   处理缺失值...")
        cleaned_data = data.copy()

        for col in data.columns:
            missing_percentage = data[col].isnull().sum() / len(data) * 100

            if missing_percentage > 0:
                if missing_percentage > self.config['missing_threshold'] * 100:
                    # 删除缺失值过多的列
                    cleaned_data = cleaned_data.drop(columns=[col])
                    print(f"     - 删除列 '{col}' (缺失值 {missing_percentage:.1f}%)")
                    self.preprocessing_steps.append(f"删除列: {col} (缺失值过多)")
                    continue

                # 根据数据类型选择填充方法
                if method == 'auto':
                    if pd.api.types.is_numeric_dtype(data[col]):
                        fill_method = 'median'
                    else:
                        fill_method = 'mode'
                else:
                    fill_method = method

                if fill_method == 'mean' and pd.api.types.is_numeric_dtype(data[col]):
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
                elif fill_method == 'median' and pd.api.types.is_numeric_dtype(data[col]):
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                elif fill_method == 'mode':
                    mode_value = cleaned_data[col].mode()
                    if len(mode_value) > 0:
                        cleaned_data[col] = cleaned_data[col].fillna(mode_value[0])
                elif fill_method == 'knn' and pd.api.types.is_numeric_dtype(data[col]):
                    # 使用KNN填充
                    imputer = KNNImputer(n_neighbors=5)
                    cleaned_data[[col]] = imputer.fit_transform(cleaned_data[[col]])
                    self.imputers[col] = imputer
                elif fill_method == 'forward':
                    cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
                elif fill_method == 'backward':
                    cleaned_data[col] = cleaned_data[col].fillna(method='bfill')

                print(f"     - 填充列 '{col}' 缺失值 (方法: {fill_method})")

        self.preprocessing_steps.append(f"处理缺失值: {method}")
        return cleaned_data

    def _handle_outliers(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """处理异常值"""
        print("   处理异常值...")
        cleaned_data = data.copy()
        outlier_count = 0

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if method == 'auto':
            method = self.config['outlier_method']

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = ((data[col] < lower_bound) | (data[col] > upper_bound))
                col_outliers = outlier_mask.sum()
                if col_outliers > 0:
                    outlier_count += col_outliers
                    # 用边界值替换异常值
                    cleaned_data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                    cleaned_data[col] = np.where(data[col] > upper_bound, upper_bound, cleaned_data[col])

        elif method == 'isolation_forest':
            # 使用Isolation Forest检测异常值
            iso_forest = IsolationForest(contamination=self.config['outlier_contamination'],
                                       random_state=self.config['random_state'])

            # 只使用数值列进行检测
            numeric_data = data[numeric_cols].dropna()
            if len(numeric_data) > 0:
                outlier_labels = iso_forest.fit_predict(numeric_data)
                outlier_mask = outlier_labels == -1

                # 移除异常值行
                outlier_indices = numeric_data.index[outlier_mask]
                outlier_count = len(outlier_indices)
                cleaned_data = cleaned_data.drop(outlier_indices)

        print(f"     - 处理了 {outlier_count} 个异常值")
        self.preprocessing_steps.append(f"处理异常值: {method}")
        return cleaned_data

    def transform_data_types(self, data: pd.DataFrame, auto_detect: bool = True) -> pd.DataFrame:
        """
        转换数据类型

        Parameters:
        - data: 数据DataFrame
        - auto_detect: 是否自动检测数据类型

        Returns:
        - 类型转换后的数据
        """
        print("🔄 转换数据类型...")
        transformed_data = data.copy()
        type_conversions = []

        for col in data.columns:
            original_type = str(data[col].dtype)

            if auto_detect:
                # 尝试自动检测最佳类型
                if data[col].dtype == 'object':
                    # 尝试转换为数值类型
                    try:
                        numeric_data = pd.to_numeric(data[col], errors='raise')
                        if (numeric_data % 1 == 0).all():
                            transformed_data[col] = numeric_data.astype('int64')
                            type_conversions.append(f"{col}: {original_type} -> int64")
                        else:
                            transformed_data[col] = numeric_data.astype('float64')
                            type_conversions.append(f"{col}: {original_type} -> float64")
                    except:
                        # 尝试转换为日期时间
                        try:
                            transformed_data[col] = pd.to_datetime(data[col], errors='raise')
                            type_conversions.append(f"{col}: {original_type} -> datetime64")
                        except:
                            # 尝试转换为分类类型
                            unique_ratio = data[col].nunique() / len(data)
                            if unique_ratio < 0.5:  # 如果唯一值比例小于50%
                                transformed_data[col] = data[col].astype('category')
                                type_conversions.append(f"{col}: {original_type} -> category")

        if type_conversions:
            print("   数据类型转换:")
            for conversion in type_conversions:
                print(f"     ✓ {conversion}")
            self.preprocessing_steps.append("数据类型自动转换")

        return transformed_data

    def encode_categorical(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                          method: str = 'auto') -> pd.DataFrame:
        """
        编码分类变量

        Parameters:
        - data: 数据DataFrame
        - columns: 要编码的列名列表
        - method: 编码方法

        Returns:
        - 编码后的数据
        """
        print("🏷️ 编码分类变量...")
        encoded_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not columns:
            print("   ✓ 没有需要编码的分类变量")
            return encoded_data

        if method == 'auto':
            # 自动选择编码方法
            for col in columns:
                unique_count = data[col].nunique()

                if unique_count == 2:
                    # 二分类变量使用标签编码
                    encoder = LabelEncoder()
                    encoded_data[col] = encoder.fit_transform(data[col].astype(str))
                    self.encoders[col] = encoder
                    print(f"     ✓ 标签编码: {col} ({unique_count} 类别)")

                elif unique_count <= 10:
                    # 少量类别使用独热编码
                    dummies = pd.get_dummies(data[col], prefix=col)
                    encoded_data = pd.concat([encoded_data.drop(columns=[col]), dummies], axis=1)
                    print(f"     ✓ 独热编码: {col} ({unique_count} 类别)")

                else:
                    # 多类别使用标签编码
                    encoder = LabelEncoder()
                    encoded_data[col] = encoder.fit_transform(data[col].astype(str))
                    self.encoders[col] = encoder
                    print(f"     ✓ 标签编码: {col} ({unique_count} 类别)")

        else:
            # 使用指定的编码方法
            if method == 'label':
                for col in columns:
                    encoder = LabelEncoder()
                    encoded_data[col] = encoder.fit_transform(data[col].astype(str))
                    self.encoders[col] = encoder
                    print(f"     ✓ 标签编码: {col}")

            elif method == 'onehot':
                for col in columns:
                    dummies = pd.get_dummies(data[col], prefix=col)
                    encoded_data = pd.concat([encoded_data.drop(columns=[col]), dummies], axis=1)
                    print(f"     ✓ 独热编码: {col}")

        self.preprocessing_steps.append(f"分类变量编码: {method}")
        return encoded_data

    def scale_features(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                      method: str = None) -> pd.DataFrame:
        """
        特征缩放

        Parameters:
        - data: 数据DataFrame
        - columns: 要缩放的列名列表
        - method: 缩放方法

        Returns:
        - 缩放后的数据
        """
        print("📏 特征缩放...")
        scaled_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            print("   ✓ 没有需要缩放的数值变量")
            return scaled_data

        if method is None:
            method = self.config['scaling_method']

        # 选择缩放器
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")

        # 应用缩放
        scaled_data[columns] = scaler.fit_transform(data[columns])
        self.scalers['feature_scaler'] = scaler

        print(f"   ✓ 使用 {method} 方法缩放了 {len(columns)} 个特征")
        self.preprocessing_steps.append(f"特征缩放: {method}")

        return scaled_data

    def select_features(self, data: pd.DataFrame, target_col: str,
                       method: str = 'univariate', k: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        特征选择

        Parameters:
        - data: 数据DataFrame
        - target_col: 目标列名
        - method: 选择方法
        - k: 选择的特征数量

        Returns:
        - 选择后的数据和特征列表
        """
        print("🎯 特征选择...")

        if target_col not in data.columns:
            print(f"   ⚠️ 目标列 '{target_col}' 不存在，跳过特征选择")
            return data, data.columns.tolist()

        if k is None:
            k = self.config['k_features']

        # 准备数据
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # 只使用数值列进行选择
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]

        if len(numeric_cols) == 0:
            print("   ⚠️ 没有数值特征，跳过特征选择")
            return data, data.columns.tolist()

        if method == 'univariate':
            # 单变量统计选择
            if y.dtype == 'object' or len(y.unique()) < 10:
                # 分类问题
                selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
            else:
                # 回归问题
                selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_cols)))

            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()

        elif method == 'rfe':
            # 递归特征消除
            from sklearn.linear_model import LogisticRegression, LinearRegression

            if y.dtype == 'object' or len(y.unique()) < 10:
                estimator = LogisticRegression(max_iter=1000)
            else:
                estimator = LinearRegression()

            selector = RFE(estimator=estimator, n_features_to_select=min(k, len(numeric_cols)))
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()

        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        # 构建选择后的数据
        other_cols = [col for col in data.columns if col not in numeric_cols and col != target_col]
        selected_data = pd.concat([
            data[other_cols],
            pd.DataFrame(X_selected, columns=selected_features, index=data.index),
            data[[target_col]]
        ], axis=1)

        print(f"   ✓ 从 {len(numeric_cols)} 个特征中选择了 {len(selected_features)} 个")
        self.feature_selectors['feature_selector'] = selector
        self.preprocessing_steps.append(f"特征选择: {method} (选择了 {len(selected_features)} 个特征)")

        return selected_data, selected_features

    def split_data(self, data: pd.DataFrame, target_col: str,
                   test_size: float = None, stratify: bool = True) -> Dict:
        """
        数据分割

        Parameters:
        - data: 数据DataFrame
        - target_col: 目标列名
        - test_size: 测试集比例
        - stratify: 是否分层抽样

        Returns:
        - 分割后的数据字典
        """
        print("✂️ 分割数据...")

        if test_size is None:
            test_size = self.config['test_size']

        X = data.drop(columns=[target_col])
        y = data[target_col]

        # 分层抽样参数
        stratify_param = y if stratify and (y.dtype == 'object' or len(y.unique()) < 100) else None

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config['random_state'],
            stratify=stratify_param
        )

        split_info = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_ratio': len(X_train) / len(data),
            'test_ratio': len(X_test) / len(data),
            'feature_count': X.shape[1],
            'target_classes': y.nunique() if y.dtype == 'object' else 'continuous'
        }

        print(f"   ✓ 训练集: {len(X_train)} 样本 ({len(X_train)/len(data):.1%})")
        print(f"   ✓ 测试集: {len(X_test)} 样本 ({len(X_test)/len(data):.1%})")

        self.preprocessing_steps.append(f"数据分割: 测试集比例 {test_size}")
        return split_info

    def balance_data(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                    method: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        平衡数据

        Parameters:
        - X_train: 训练特征
        - y_train: 训练标签
        - method: 平衡方法

        Returns:
        - 平衡后的数据
        """
        if method is None:
            method = self.config['balance_method']

        # 检查是否需要平衡
        if y_train.dtype == 'object' or len(y_train.unique()) < 100:
            class_counts = y_train.value_counts()
            min_count = class_counts.min()
            max_count = class_counts.max()

            if max_count / min_count <= 2:  # 如果类别比例小于2:1，认为已经平衡
                print("   ✓ 数据已经平衡，无需处理")
                return X_train, y_train

        print(f"⚖️ 平衡数据 (方法: {method})...")

        if method == 'smote':
            # SMOTE过采样
            smote = SMOTE(random_state=self.config['random_state'])
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        elif method == 'oversample':
            # 随机过采样
            ros = RandomOverSampler(random_state=self.config['random_state'])
            X_balanced, y_balanced = ros.fit_resample(X_train, y_train)

        elif method == 'undersample':
            # 随机欠采样
            rus = RandomUnderSampler(random_state=self.config['random_state'])
            X_balanced, y_balanced = rus.fit_resample(X_train, y_train)

        else:
            raise ValueError(f"不支持的平衡方法: {method}")

        print(f"   ✓ 平衡前: {X_train.shape[0]} 样本")
        print(f"   ✓ 平衡后: {X_balanced.shape[0]} 样本")

        # 转换回DataFrame
        if hasattr(X_balanced, 'toarray'):
            X_balanced = pd.DataFrame(X_balanced.toarray(), columns=X_train.columns)
        else:
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)

        y_balanced = pd.Series(y_balanced, name=y_train.name)

        self.preprocessing_steps.append(f"数据平衡: {method}")
        return X_balanced, y_balanced

    def auto_preprocess(self, data: pd.DataFrame, target_col: str,
                       save_report: bool = True) -> Dict:
        """
        自动化预处理流程

        Parameters:
        - data: 原始数据
        - target_col: 目标列名
        - save_report: 是否保存报告

        Returns:
        - 预处理结果字典
        """
        print("🚀 开始自动化数据预处理...")

        results = {
            'original_data': data,
            'preprocessed_data': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'quality_report': None,
            'preprocessing_steps': [],
            'feature_info': {}
        }

        # 1. 数据质量分析
        quality_report = self.analyze_data_quality(data)
        results['quality_report'] = quality_report

        # 2. 数据清洗
        cleaned_data = self.clean_data(data)
        results['preprocessing_steps'].extend(self.preprocessing_steps)

        # 3. 数据类型转换
        transformed_data = self.transform_data_types(cleaned_data)

        # 4. 特征工程
        # 这里可以添加更多的特征工程步骤
        engineered_data = self._feature_engineering(transformed_data, target_col)

        # 5. 编码分类变量
        encoded_data = self.encode_categorical(engineered_data)

        # 6. 特征缩放
        numeric_cols = encoded_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if numeric_cols:
            scaled_data = self.scale_features(encoded_data, numeric_cols)
        else:
            scaled_data = encoded_data

        results['preprocessed_data'] = scaled_data

        # 7. 特征选择（可选）
        if self.config['feature_selection'] and target_col in scaled_data.columns:
            selected_data, selected_features = self.select_features(
                scaled_data, target_col, k=self.config['k_features']
            )
            results['feature_info']['selected_features'] = selected_features
            final_data = selected_data
        else:
            final_data = scaled_data
            results['feature_info']['all_features'] = [
                col for col in final_data.columns if col != target_col
            ]

        # 8. 数据分割
        if target_col in final_data.columns:
            split_result = self.split_data(final_data, target_col)
            results.update(split_result)

        # 9. 数据平衡（可选，仅对分类问题）
        if (self.config['balance_data'] and
            target_col in final_data.columns and
            (results['y_train'].dtype == 'object' or len(results['y_train'].unique()) < 100)):

            X_balanced, y_balanced = self.balance_data(
                results['X_train'], results['y_train'], method=self.config['balance_method']
            )
            results['X_train'] = X_balanced
            results['y_train'] = y_balanced

        # 保存预处理报告
        if save_report:
            self.preprocessing_report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'original_shape': data.shape,
                'final_shape': results['preprocessed_data'].shape,
                'preprocessing_steps': self.preprocessing_steps,
                'quality_score': quality_report['overall_score'],
                'feature_count': len(results['feature_info'].get('selected_features',
                                                results['feature_info'].get('all_features', [])))
            }

        print(f"\n🎉 自动化预处理完成！")
        print(f"   原始数据: {data.shape}")
        print(f"   预处理后: {results['preprocessed_data'].shape}")
        print(f"   预处理步骤: {len(self.preprocessing_steps)}")

        return results

    def _feature_engineering(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """基础特征工程"""
        engineered_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 创建交互特征（对于数值变量）
        if len(numeric_cols) >= 2:
            # 选择前几个重要变量创建交互项
            important_cols = numeric_cols[:min(3, len(numeric_cols))]

            for i, col1 in enumerate(important_cols):
                for col2 in important_cols[i+1:]:
                    # 乘积特征
                    engineered_data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                    # 比值特征（避免除零）
                    engineered_data[f'{col1}_div_{col2}'] = np.where(
                        data[col2] != 0, data[col1] / data[col2], 0
                    )

        # 创建多项式特征（对于重要变量）
        if len(important_cols) > 0:
            for col in important_cols[:2]:  # 只为前两个变量创建
                engineered_data[f'{col}_squared'] = data[col] ** 2
                engineered_data[f'{col}_sqrt'] = np.sqrt(np.abs(data[col]))

        print(f"   ✓ 创建了 {engineered_data.shape[1] - data.shape[1]} 个新特征")
        return engineered_data

    def get_preprocessing_summary(self) -> Dict:
        """
        获取预处理摘要

        Returns:
        - 预处理摘要信息
        """
        return {
            'preprocessing_steps': self.preprocessing_steps,
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'imputers': list(self.imputers.keys()),
            'feature_selectors': list(self.feature_selectors.keys()),
            'preprocessing_report': getattr(self, 'preprocessing_report', {}),
            'config': self.config
        }

    def save_preprocessing_objects(self, output_dir: str):
        """
        保存预处理对象

        Parameters:
        - output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存scalers
        if self.scalers:
            import joblib
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, output_path / f"{name}.pkl")

        # 保存encoders
        if self.encoders:
            import joblib
            for name, encoder in self.encoders.items():
                joblib.dump(encoder, output_path / f"{name}_encoder.pkl")

        # 保存预处理报告
        if hasattr(self, 'preprocessing_report'):
            with open(output_path / 'preprocessing_report.json', 'w', encoding='utf-8') as f:
                json.dump(self.preprocessing_report, f, ensure_ascii=False, indent=2)

        print(f"✅ 预处理对象已保存到 {output_dir}")

    def load_preprocessing_objects(self, input_dir: str):
        """
        加载预处理对象

        Parameters:
        - input_dir: 输入目录
        """
        input_path = Path(input_dir)

        # 加载scalers
        import joblib
        for scaler_file in input_path.glob("*.pkl"):
            if not str(scaler_file).endswith('_encoder.pkl'):
                name = scaler_file.stem
                self.scalers[name] = joblib.load(scaler_file)

        # 加载encoders
        for encoder_file in input_path.glob("*_encoder.pkl"):
            name = encoder_file.stem.replace('_encoder', '')
            self.encoders[name] = joblib.load(encoder_file)

        print(f"✅ 预处理对象已从 {input_dir} 加载")

    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        对新数据应用相同的预处理

        Parameters:
        - new_data: 新数据

        Returns:
        - 预处理后的新数据
        """
        transformed_data = new_data.copy()

        # 应用缺失值处理
        for col, imputer in self.imputers.items():
            if col in transformed_data.columns:
                if hasattr(imputer, 'transform'):
                    transformed_data[[col]] = imputer.transform(transformed_data[[col]])

        # 应用编码
        for col, encoder in self.encoders.items():
            if col in transformed_data.columns:
                transformed_data[col] = encoder.transform(transformed_data[col].astype(str))

        # 应用缩放
        if 'feature_scaler' in self.scalers:
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                transformed_data[numeric_cols] = self.scalers['feature_scaler'].transform(
                    transformed_data[numeric_cols]
                )

        return transformed_data