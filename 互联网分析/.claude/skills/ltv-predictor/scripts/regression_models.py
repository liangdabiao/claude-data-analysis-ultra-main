#!/usr/bin/env python3
"""
回归算法模块
基于第3课核心算法实现线性回归和随机森林等LTV预测算法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class RegressionModels:
    """
    回归算法集合

    基于第3课理论实现的LTV预测回归模型
    支持线性回归、随机森林等多种算法
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化回归模型

        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'scoring_metric': 'r2',
            'enable_hyperparameter_tuning': False,
            'n_iter_search': 50,
            'feature_columns': ['R值', 'F值', 'M值'],
            'target_column': '年度LTV'
        }

        # 更新配置
        if config:
            self.config.update(config)

        # 模型存储
        self.models = {}
        self.trained_models = {}
        self.model_performance = {}
        self.feature_importance = {}
        self.best_model_name = None

        # 数据存储
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def prepare_data(self, rfm_data: pd.DataFrame,
                    feature_columns: Optional[List[str]] = None,
                    target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据

        Args:
            rfm_data: RFM特征数据
            feature_columns: 特征列名列表
            target_column: 目标列名

        Returns:
            特征矩阵和目标向量
        """
        if feature_columns is None:
            feature_columns = self.config['feature_columns']
        if target_column is None:
            target_column = self.config['target_column']

        # 检查必需列是否存在
        missing_features = [col for col in feature_columns if col not in rfm_data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")

        if target_column not in rfm_data.columns:
            raise ValueError(f"缺少目标列: {target_column}")

        # 准备特征和目标
        X = rfm_data[feature_columns].copy()
        y = rfm_data[target_column].copy()

        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # 移除异常值（使用3σ原则）
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()
            outlier_mask = np.abs(X[col] - mean) > 3 * std
            if outlier_mask.any():
                print(f"  移除{col}列异常值: {outlier_mask.sum()} 个")
                X = X[~outlier_mask]
                y = y[~outlier_mask]

        # 移除LTV异常值
        ltv_mean = y.mean()
        ltv_std = y.std()
        ltv_outlier_mask = np.abs(y - ltv_mean) > 3 * ltv_std
        if ltv_outlier_mask.any():
            print(f"  移除LTV异常值: {ltv_outlier_mask.sum()} 个")
            X = X[~ltv_outlier_mask]
            y = y[~ltv_outlier_mask]

        self.feature_names = feature_columns
        print(f"✓ 数据准备完成: {X.shape}, 目标值范围: {y.min():.2f} ~ {y.max():.2f}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        分割训练和测试数据

        Args:
            X: 特征矩阵
            y: 目标向量

        Returns:
            训练和测试数据
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        print(f"✓ 数据分割完成:")
        print(f"  - 训练集: {self.X_train.shape}")
        print(f"  - 测试集: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_linear_regression_model(self) -> LinearRegression:
        """
        创建线性回归模型

        Returns:
            线性回归模型实例
        """
        model = LinearRegression()
        self.models['linear_regression'] = model
        return model

    def create_random_forest_model(self, **kwargs) -> RandomForestRegressor:
        """
        创建随机森林模型

        Args:
            **kwargs: 模型参数

        Returns:
            随机森林模型实例
        """
        # 默认参数（基于第3课优化）
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.config['random_state'],
            'n_jobs': -1
        }

        # 更新参数
        params = {**default_params, **kwargs}
        model = RandomForestRegressor(**params)

        self.models['random_forest'] = model
        return model

    def train_single_model(self,
                          model_name: str,
                          X_train: Optional[pd.DataFrame] = None,
                          y_train: Optional[pd.Series] = None,
                          tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        训练单个模型

        Args:
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练目标
            tune_hyperparameters: 是否调参

        Returns:
            训练结果字典
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError("训练数据未准备，请先调用split_data")

        if model_name not in self.models:
            if model_name == 'linear_regression':
                self.create_linear_regression_model()
            elif model_name == 'random_forest':
                self.create_random_forest_model()
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")

        model = self.models[model_name]

        print(f"🤖 训练{model_name}模型...")

        # 超参数调优
        if tune_hyperparameters and model_name == 'random_forest':
            model = self._tune_random_forest(model, X_train, y_train)
            self.models[model_name] = model

        # 训练模型
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model

        # 评估模型
        train_score = model.score(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train,
                                   cv=self.config['cv_folds'],
                                   scoring=self.config['scoring_metric'])

        results = {
            'model_name': model_name,
            'model': model,
            'train_score': train_score,
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'feature_importance': self._get_feature_importance(model) if hasattr(model, 'feature_importances_') else None
        }

        print(f"✓ {model_name}训练完成:")
        print(f"  - 训练集R²: {train_score:.4f}")
        print(f"  - 交叉验证R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        return results

    def _tune_random_forest(self, model: RandomForestRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        """
        随机森林超参数调优

        Args:
            model: 随机森林模型
            X_train: 训练特征
            y_train: 训练目标

        Returns:
            调优后的模型
        """
        print("  进行超参数调优...")

        # 参数搜索空间
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # 网格搜索
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.config['cv_folds'],
            scoring=self.config['scoring_metric'],
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"  最佳参数: {grid_search.best_params_}")
        print(f"  最佳分数: {grid_search.best_score_:.4f}")

        return best_model

    def _get_feature_importance(self, model) -> Optional[Dict[str, float]]:
        """
        获取特征重要性

        Args:
            model: 训练好的模型

        Returns:
            特征重要性字典
        """
        if hasattr(model, 'feature_importances_') and self.feature_names:
            importance_dict = dict(zip(self.feature_names, model.feature_importances_))
            # 按重要性排序
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_') and self.feature_names:
            importance_dict = dict(zip(self.feature_names, np.abs(model.coef_)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return None

    def evaluate_model(self,
                      model_name: str,
                      X_test: Optional[pd.DataFrame] = None,
                      y_test: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            model_name: 模型名称
            X_test: 测试特征
            y_test: 测试目标

        Returns:
            评估指标字典
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test

        if model_name not in self.trained_models:
            raise ValueError(f"模型{model_name}未训练")

        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # 相对误差指标
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        metrics = {
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

        # 存储评估结果
        self.model_performance[model_name] = metrics
        self.feature_importance[model_name] = self._get_feature_importance(model)

        print(f"📊 {model_name}模型评估结果:")
        print(f"  - R² 分数: {r2:.4f}")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - MAPE: {mape:.2f}%")

        return metrics

    def train_multiple_models(self,
                            model_names: List[str],
                            X_train: Optional[pd.DataFrame] = None,
                            y_train: Optional[pd.Series] = None,
                            X_test: Optional[pd.DataFrame] = None,
                            y_test: Optional[pd.Series] = None) -> Dict[str, Dict]:
        """
        训练多个模型并比较性能

        Args:
            model_names: 模型名称列表
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标

        Returns:
            所有模型的训练和评估结果
        """
        results = {}

        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"训练和评估模型: {model_name}")
            print(f"{'='*50}")

            try:
                # 训练模型
                train_result = self.train_single_model(model_name, X_train, y_train)

                # 评估模型
                test_result = self.evaluate_model(model_name, X_test, y_test)

                # 合并结果
                results[model_name] = {**train_result, **test_result}

            except Exception as e:
                print(f"❌ 模型{model_name}训练失败: {str(e)}")
                results[model_name] = {'error': str(e)}

        # 确定最佳模型
        self._select_best_model(results)

        # 打印模型比较
        self._print_model_comparison(results)

        return results

    def _select_best_model(self, results: Dict[str, Dict]):
        """选择最佳模型"""
        valid_models = {name: result for name, result in results.items()
                       if 'error' not in result and 'r2_score' in result}

        if valid_models:
            best_model_name = max(valid_models.keys(),
                                key=lambda x: valid_models[x]['r2_score'])
            self.best_model_name = best_model_name
            print(f"\n🏆 最佳模型: {best_model_name} (R²: {valid_models[best_model_name]['r2_score']:.4f})")

    def _print_model_comparison(self, results: Dict[str, Dict]):
        """打印模型比较结果"""
        print(f"\n📈 模型性能对比:")
        print(f"{'模型名称':<15} {'训练R²':<10} {'测试R²':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 70)

        for model_name, result in results.items():
            if 'error' in result:
                print(f"{model_name:<15} {'ERROR':<50}")
            else:
                train_score = result.get('train_score', 'N/A')
                test_r2 = result.get('r2_score', 'N/A')
                mae = result.get('mae', 'N/A')
                rmse = result.get('rmse', 'N/A')
                mape = result.get('mape', 'N/A')

                print(f"{model_name:<15} {train_score:<10.4f} {test_r2:<10.4f} {mae:<10.2f} {rmse:<10.2f} {mape:<10.2f}%")

    def predict(self,
               model_name: Optional[str] = None,
               X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        进行预测

        Args:
            model_name: 模型名称，如果为None则使用最佳模型
            X: 预测特征

        Returns:
            预测结果
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name is None:
            raise ValueError("没有可用的模型，请先训练模型")

        if model_name not in self.trained_models:
            raise ValueError(f"模型{model_name}未训练")

        model = self.trained_models[model_name]

        if X is None:
            X = self.X_test

        predictions = model.predict(X)
        return predictions

    def save_models(self, directory: str):
        """
        保存训练好的模型

        Args:
            directory: 保存目录
        """
        import os
        from pathlib import Path

        Path(directory).mkdir(parents=True, exist_ok=True)

        for model_name, model in self.trained_models.items():
            model_path = os.path.join(directory, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            print(f"✓ 模型已保存: {model_path}")

        # 保存元数据
        metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'best_model_name': self.best_model_name
        }

        metadata_path = os.path.join(directory, "metadata.joblib")
        joblib.dump(metadata, metadata_path)
        print(f"✓ 元数据已保存: {metadata_path}")

    def load_models(self, directory: str):
        """
        加载训练好的模型

        Args:
            directory: 模型目录
        """
        import os
        import glob

        # 加载模型文件
        model_files = glob.glob(os.path.join(directory, "*.joblib"))

        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.joblib', '')

            if model_name == 'metadata':
                # 加载元数据
                metadata = joblib.load(model_file)
                self.config = metadata.get('config', self.config)
                self.feature_names = metadata.get('feature_names')
                self.model_performance = metadata.get('model_performance', {})
                self.feature_importance = metadata.get('feature_importance', {})
                self.best_model_name = metadata.get('best_model_name')
            else:
                # 加载模型
                model = joblib.load(model_file)
                self.trained_models[model_name] = model
                self.models[model_name] = model  # also add to models dict
                print(f"✓ 模型已加载: {model_name}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息

        Returns:
            模型摘要字典
        """
        summary = {
            'total_models_trained': len(self.trained_models),
            'best_model': self.best_model_name,
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance
        }

        if self.best_model_name and self.best_model_name in self.model_performance:
            best_performance = self.model_performance[self.best_model_name]
            summary['best_performance'] = best_performance

        return summary

# 便利函数
def quick_model_training(rfm_data: pd.DataFrame,
                        model_names: List[str] = ['linear_regression', 'random_forest'],
                        feature_columns: List[str] = ['R值', 'F值', 'M值'],
                        target_column: str = '年度LTV',
                        test_size: float = 0.2) -> Dict[str, Any]:
    """
    快速模型训练

    Args:
        rfm_data: RFM数据
        model_names: 要训练的模型列表
        feature_columns: 特征列
        target_column: 目标列
        test_size: 测试集比例

    Returns:
        训练结果字典
    """
    # 初始化回归模型
    regression = RegressionModels({
        'test_size': test_size,
        'feature_columns': feature_columns,
        'target_column': target_column
    })

    # 准备数据
    X, y = regression.prepare_data(rfm_data, feature_columns, target_column)
    X_train, X_test, y_train, y_test = regression.split_data(X, y)

    # 训练多个模型
    results = regression.train_multiple_models(model_names, X_train, y_train, X_test, y_test)

    return {
        'regression': regression,
        'results': results,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

if __name__ == "__main__":
    # 示例使用
    print("🤖 回归算法模块测试")

    # 如果有数据，可以进行测试
    # 这里应该有测试数据，暂时跳过
    print("模块初始化完成，等待数据输入")