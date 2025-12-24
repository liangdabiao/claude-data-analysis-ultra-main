#!/usr/bin/env python3
"""
建模评估器 (Modeling Evaluator) - 机器学习模型训练和评估模块

提供全面的建模功能，包括：
- 多种分类算法支持（逻辑回归、随机森林、XGBoost、神经网络）
- 自动化模型选择和超参数调优
- 全面的模型评估指标
- 模型可解释性分析（特征重要性、SHAP值）
- 交叉验证和模型比较
- 医疗数据建模特化支持
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor
import shap
import warnings
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')


class ModelingEvaluator:
    """建模评估器 - 自动化机器学习模型训练和评估"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化建模评估器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.explainers = {}
        self.model_history = []

        # 默认配置
        self.default_config = {
            'cv_folds': 5,  # 交叉验证折数
            'scoring_metric': 'accuracy',  # 主要评估指标
            'test_size': 0.2,  # 测试集比例
            'random_state': 42,  # 随机种子
            'n_iter_search': 50,  # 随机搜索迭代次数
            'enable_hyperparameter_tuning': True,  # 是否启用超参数调优
            'enable_feature_importance': True,  # 是否计算特征重要性
            'enable_shap': True,  # 是否启用SHAP解释
            'model_timeout': 300,  # 模型训练超时时间（秒）
            'ensemble_models': True,  # 是否使用集成方法
            'medical_specialization': False  # 是否启用医疗数据特化
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

        # 初始化模型字典
        self._initialize_models()

    def _initialize_models(self):
        """初始化可用模型"""
        self.available_models = {
            # 分类模型
            'classification': {
                'logistic_regression': {
                    'model': LogisticRegression(random_state=self.config['random_state'], max_iter=1000),
                    'param_grid': {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2', 'elasticnet'],
                        'solver': ['liblinear', 'saga']
                    }
                },
                'random_forest': {
                    'model': RandomForestClassifier(random_state=self.config['random_state']),
                    'param_grid': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBClassifier(random_state=self.config['random_state']),
                    'param_grid': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMClassifier(random_state=self.config['random_state']),
                    'param_grid': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 62, 127]
                    }
                },
                'svm': {
                    'model': SVC(random_state=self.config['random_state'], probability=True),
                    'param_grid': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear', 'poly'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'knn': {
                    'model': KNeighborsClassifier(),
                    'param_grid': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree']
                    }
                },
                'naive_bayes': {
                    'model': GaussianNB(),
                    'param_grid': {}
                },
                'decision_tree': {
                    'model': DecisionTreeClassifier(random_state=self.config['random_state']),
                    'param_grid': {
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'mlp': {
                    'model': MLPClassifier(random_state=self.config['random_state'], max_iter=1000),
                    'param_grid': {
                        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                        'activation': ['relu', 'tanh'],
                        'alpha': [0.0001, 0.001, 0.01]
                    }
                }
            },
            # 回归模型
            'regression': {
                'linear_regression': {
                    'model': LinearRegression(),
                    'param_grid': {}
                },
                'ridge': {
                    'model': Ridge(random_state=self.config['random_state']),
                    'param_grid': {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                },
                'lasso': {
                    'model': Lasso(random_state=self.config['random_state']),
                    'param_grid': {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.config['random_state']),
                    'param_grid': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=self.config['random_state']),
                    'param_grid': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                },
                'svm': {
                    'model': SVR(),
                    'param_grid': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                }
            }
        }

    def detect_problem_type(self, y: pd.Series) -> str:
        """
        检测问题类型（分类或回归）

        Parameters:
        - y: 目标变量

        Returns:
        - 问题类型
        """
        if y.dtype == 'object' or len(y.unique()) < 20:
            return 'classification'
        else:
            return 'regression'

    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          model_name: str, tune_hyperparameters: bool = None) -> Dict:
        """
        训练单个模型

        Parameters:
        - X_train: 训练特征
        - y_train: 训练标签
        - X_test: 测试特征
        - y_test: 测试标签
        - model_name: 模型名称
        - tune_hyperparameters: 是否调参

        Returns:
        - 模型训练结果
        """
        if tune_hyperparameters is None:
            tune_hyperparameters = self.config['enable_hyperparameter_tuning']

        problem_type = self.detect_problem_type(y_train)
        models_dict = self.available_models[problem_type]

        if model_name not in models_dict:
            raise ValueError(f"模型 '{model_name}' 不适用于 {problem_type} 问题")

        print(f"🚀 训练模型: {model_name}")

        model_info = models_dict[model_name]
        model = model_info['model']

        # 超参数调优
        if tune_hyperparameters and model_info['param_grid']:
            print(f"   🔧 进行超参数调优...")
            search = RandomizedSearchCV(
                model, model_info['param_grid'],
                n_iter=self.config['n_iter_search'],
                cv=self.config['cv_folds'],
                scoring=self.config['scoring_metric'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            print(f"   ✓ 最佳参数: {best_params}")
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}

        # 预测
        if problem_type == 'classification':
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        else:
            y_pred = best_model.predict(X_test)
            metrics = self._calculate_regression_metrics(y_test, y_pred)

        # 交叉验证
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=self.config['cv_folds'],
            scoring=self.config['scoring_metric']
        )

        result = {
            'model': best_model,
            'model_name': model_name,
            'problem_type': problem_type,
            'best_params': best_params,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'feature_names': X_train.columns.tolist()
        }

        if problem_type == 'classification' and y_pred_proba is not None:
            result['predictions_proba'] = y_pred_proba

        self.models[model_name] = result
        print(f"   ✓ {model_name} 训练完成 - {self.config['scoring_metric']}: {cv_scores.mean():.4f}")

        return result

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """计算分类指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

        # 分类报告
        metrics['classification_report'] = classification_report(y_true, y_pred)

        return metrics

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算回归指标"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

        return metrics

    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            model_names: Optional[List[str]] = None) -> Dict:
        """
        训练多个模型

        Parameters:
        - X_train: 训练特征
        - y_train: 训练标签
        - X_test: 测试特征
        - y_test: 测试标签
        - model_names: 要训练的模型名称列表

        Returns:
        - 所有模型的训练结果
        """
        if model_names is None:
            problem_type = self.detect_problem_type(y_train)
            model_names = list(self.available_models[problem_type].keys())

        print(f"🎯 开始批量模型训练 ({len(model_names)} 个模型)...")
        results = {}

        for model_name in model_names:
            try:
                result = self.train_single_model(
                    X_train, y_train, X_test, y_test, model_name
                )
                results[model_name] = result
            except Exception as e:
                print(f"   ❌ 模型 {model_name} 训练失败: {str(e)}")
                continue

        # 选择最佳模型
        if results:
            best_model_name = max(
                results.keys(),
                key=lambda x: results[x]['cv_mean']
            )
            self.best_model = results[best_model_name]['model']
            self.best_model_name = best_model_name

            print(f"\n🏆 最佳模型: {best_model_name}")
            print(f"   CV {self.config['scoring_metric']}: {results[best_model_name]['cv_mean']:.4f}")

        self.model_results = results
        return results

    def auto_modeling(self, data: pd.DataFrame, target_col: str,
                     model_names: Optional[List[str]] = None) -> Dict:
        """
        自动化建模流程

        Parameters:
        - data: 数据DataFrame
        - target_col: 目标列名
        - model_names: 要训练的模型名称列表

        Returns:
        - 自动化建模结果
        """
        print("🤖 开始自动化建模...")

        # 数据分割
        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y if self.detect_problem_type(y) == 'classification' else None
        )

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # 训练模型
        model_results = self.train_multiple_models(
            X_train_scaled, y_train, X_test_scaled, y_test, model_names
        )

        # 特征重要性分析
        if self.config['enable_feature_importance']:
            print("\n📊 分析特征重要性...")
            importance_results = self.analyze_feature_importance(X_train_scaled)
        else:
            importance_results = {}

        # SHAP解释（仅对最佳模型）
        shap_results = {}
        if (self.config['enable_shap'] and
            self.best_model is not None and
            len(X_train.columns) <= 50):  # 限制特征数量避免计算过慢

            print("\n🔍 生成SHAP解释...")
            try:
                shap_results = self.generate_shap_explanations(
                    self.best_model, X_train_scaled, X_test_scaled
                )
            except Exception as e:
                print(f"   ⚠️ SHAP解释生成失败: {str(e)}")

        # 模型比较
        comparison_results = self.compare_models(model_results)

        results = {
            'data_info': {
                'shape': data.shape,
                'target_col': target_col,
                'problem_type': self.detect_problem_type(y),
                'feature_count': X.shape[1]
            },
            'split_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'scaler': scaler
            },
            'model_results': model_results,
            'best_model': {
                'name': self.best_model_name,
                'model': self.best_model,
                'metrics': model_results[self.best_model_name]['metrics'] if self.best_model_name else None
            },
            'feature_importance': importance_results,
            'shap_explanations': shap_results,
            'model_comparison': comparison_results
        }

        print(f"\n🎉 自动化建模完成！")
        print(f"   训练了 {len(model_results)} 个模型")
        print(f"   最佳模型: {self.best_model_name}")

        return results

    def analyze_feature_importance(self, X: pd.DataFrame) -> Dict:
        """
        分析特征重要性

        Parameters:
        - X: 特征数据

        Returns:
        - 特征重要性结果
        """
        importance_results = {}

        for model_name, result in self.models.items():
            model = result['model']
            feature_names = result['feature_names']

            try:
                if hasattr(model, 'feature_importances_'):
                    # 树模型的特征重要性
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # 线性模型的系数
                    importances = np.abs(model.coef_).flatten()
                else:
                    continue

                # 创建重要性DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                importance_results[model_name] = {
                    'importance_df': importance_df,
                    'top_features': importance_df.head(10).to_dict('records')
                }

                print(f"   ✓ {model_name} 特征重要性分析完成")

            except Exception as e:
                print(f"   ⚠️ {model_name} 特征重要性分析失败: {str(e)}")

        return importance_results

    def generate_shap_explanations(self, model: Any, X_train: pd.DataFrame,
                                 X_test: pd.DataFrame, sample_size: int = 100) -> Dict:
        """
        生成SHAP解释

        Parameters:
        - model: 训练好的模型
        - X_train: 训练特征
        - X_test: 测试特征
        - sample_size: 采样大小

        Returns:
        - SHAP解释结果
        """
        # 采样数据减少计算量
        if len(X_test) > sample_size:
            X_test_sample = X_test.sample(sample_size, random_state=self.config['random_state'])
        else:
            X_test_sample = X_test

        # 创建解释器
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X_train.sample(50))

            # 计算SHAP值
            shap_values = explainer.shap_values(X_test_sample)

            # 如果是多分类，取第一个类的SHAP值
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # 全局特征重要性
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'feature': X_test_sample.columns,
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending=False)

            results = {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_importance': importance_df,
                'sample_data': X_test_sample
            }

            print(f"   ✓ SHAP解释生成完成 (样本数: {len(X_test_sample)})")
            return results

        except Exception as e:
            print(f"   ❌ SHAP解释生成失败: {str(e)}")
            return {}

    def compare_models(self, model_results: Dict) -> Dict:
        """
        比较模型性能

        Parameters:
        - model_results: 模型训练结果字典

        Returns:
        - 模型比较结果
        """
        if not model_results:
            return {}

        # 创建比较DataFrame
        comparison_data = []

        for model_name, result in model_results.items():
            row = {
                'model': model_name,
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }

            # 添加主要指标
            metrics = result['metrics']
            if 'accuracy' in metrics:
                row['accuracy'] = metrics['accuracy']
            if 'f1' in metrics:
                row['f1'] = metrics['f1']
            if 'auc' in metrics:
                row['auc'] = metrics['auc']
            if 'r2' in metrics:
                row['r2'] = metrics['r2']

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # 排序
        if 'accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        elif 'r2' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('r2', ascending=False)

        # 排名
        comparison_df['rank'] = range(1, len(comparison_df) + 1)

        results = {
            'comparison_df': comparison_df,
            'best_model': comparison_df.iloc[0]['model'],
            'ranking': comparison_df[['model', 'rank']].to_dict('records')
        }

        return results

    def predict_new_data(self, new_data: pd.DataFrame, model_name: Optional[str] = None) -> Dict:
        """
        对新数据进行预测

        Parameters:
        - new_data: 新数据
        - model_name: 使用的模型名称（默认使用最佳模型）

        Returns:
        - 预测结果
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("没有可用的模型，请先训练模型")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                raise ValueError(f"模型 '{model_name}' 不存在")
            model = self.models[model_name]['model']

        # 确保特征顺序一致
        feature_names = self.models[model_name]['feature_names']
        if set(new_data.columns) != set(feature_names):
            missing_features = set(feature_names) - set(new_data.columns)
            extra_features = set(new_data.columns) - set(feature_names)

            if missing_features:
                raise ValueError(f"缺少特征: {missing_features}")
            if extra_features:
                print(f"⚠️ 多余特征将被忽略: {extra_features}")

        # 重新排序列
        new_data_aligned = new_data[feature_names]

        # 预测
        predictions = model.predict(new_data_aligned)

        results = {
            'model_name': model_name,
            'predictions': predictions,
            'input_data': new_data_aligned
        }

        # 添加概率预测（如果支持）
        if hasattr(model, 'predict_proba'):
            predictions_proba = model.predict_proba(new_data_aligned)
            results['predictions_proba'] = predictions_proba

        return results

    def generate_model_report(self, output_path: str, include_plots: bool = True):
        """
        生成模型报告

        Parameters:
        - output_path: 输出路径
        - include_plots: 是否包含图表
        """
        if not self.model_results:
            print("❌ 没有模型结果可以生成报告")
            return

        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': self.config,
            'best_model': self.best_model_name,
            'model_count': len(self.model_results),
            'model_results': {}
        }

        # 转换模型结果为可序列化格式
        for model_name, result in self.model_results.items():
            model_data = {
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'best_params': result['best_params'],
                'metrics': {}
            }

            # 转换指标
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float, np.number)):
                    model_data['metrics'][metric] = float(value)
                elif isinstance(value, str):
                    model_data['metrics'][metric] = value

            report_data['model_results'][model_name] = model_data

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 模型报告已保存到 {output_path}")

        # 生成图表
        if include_plots:
            self._generate_model_plots(output_path.replace('.json', '_plots.png'))

    def _generate_model_plots(self, output_path: str):
        """生成模型比较图表"""
        if not self.model_results:
            return

        # 准备数据
        model_names = []
        cv_means = []
        cv_stds = []

        for model_name, result in self.model_results.items():
            model_names.append(model_name)
            cv_means.append(result['cv_mean'])
            cv_stds.append(result['cv_std'])

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 模型性能比较
        bars = ax1.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        ax1.set_title('模型性能比较 (CV Score)', fontsize=14)
        ax1.set_ylabel('CV Score')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, mean in zip(bars, cv_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(cv_stds)*0.1,
                    f'{mean:.3f}', ha='center', va='bottom')

        # 模型稳定性比较
        ax2.bar(model_names, cv_stds, color='orange', alpha=0.7)
        ax2.set_title('模型稳定性比较 (CV Std)', fontsize=14)
        ax2.set_ylabel('CV Std')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 模型比较图表已保存到 {output_path}")

    def save_models(self, output_dir: str):
        """
        保存所有模型

        Parameters:
        - output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存每个模型
        for model_name, result in self.model_results.items():
            model_path = output_path / f"{model_name}.pkl"
            joblib.dump(result['model'], model_path)

        # 保存最佳模型
        if self.best_model:
            joblib.dump(self.best_model, output_path / "best_model.pkl")

        # 保存模型元数据
        metadata = {
            'best_model_name': self.best_model_name,
            'model_names': list(self.model_results.keys()),
            'config': self.config
        }

        with open(output_path / 'models_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ 模型已保存到 {output_dir}")

    def load_models(self, input_dir: str):
        """
        加载模型

        Parameters:
        - input_dir: 输入目录
        """
        input_path = Path(input_dir)

        # 加载元数据
        with open(input_path / 'models_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.best_model_name = metadata['best_model_name']

        # 加载最佳模型
        best_model_path = input_path / "best_model.pkl"
        if best_model_path.exists():
            self.best_model = joblib.load(best_model_path)

        print(f"✅ 模型已从 {input_dir} 加载")
        print(f"   最佳模型: {self.best_model_name}")

    def get_model_summary(self) -> Dict:
        """
        获取模型摘要信息

        Returns:
        - 模型摘要
        """
        summary = {
            'trained_models': list(self.model_results.keys()),
            'best_model': self.best_model_name,
            'model_count': len(self.model_results),
            'config': self.config
        }

        if self.model_results:
            # 性能统计
            cv_scores = [result['cv_mean'] for result in self.model_results.values()]
            summary['performance_stats'] = {
                'best_cv_score': max(cv_scores),
                'worst_cv_score': min(cv_scores),
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores)
            }

            # 最佳模型详细信息
            if self.best_model_name in self.model_results:
                best_result = self.model_results[self.best_model_name]
                summary['best_model_details'] = {
                    'cv_mean': float(best_result['cv_mean']),
                    'cv_std': float(best_result['cv_std']),
                    'metrics': best_result['metrics'],
                    'best_params': best_result['best_params']
                }

        return summary