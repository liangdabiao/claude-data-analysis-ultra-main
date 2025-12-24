#!/usr/bin/env python3
"""
模型优化器
提供模型性能优化和超参数调优功能
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    """模型优化器类"""

    def __init__(self):
        self.best_params = {}
        self.optimization_history = {}

    def optimize_random_forest(self, X_train, y_train, X_test=None, y_test=None,
                             optimization_method='grid', cv_folds=5, n_iter=50):
        """
        优化随机森林模型

        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据（可选）
            optimization_method: 优化方法 ('grid', 'random', 'bayesian')
            cv_folds: 交叉验证折数
            n_iter: 随机搜索迭代次数

        Returns:
            优化后的模型和参数
        """
        print("🔧 开始随机森林模型优化...")

        # 基础模型
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        # 参数搜索空间
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # 选择优化方法
        if optimization_method == 'grid':
            print("  使用网格搜索...")
            search = GridSearchCV(
                rf, param_grid, cv=cv_folds,
                scoring='r2', n_jobs=-1, verbose=1
            )
        elif optimization_method == 'random':
            print("  使用随机搜索...")
            search = RandomizedSearchCV(
                rf, param_grid, n_iter=n_iter, cv=cv_folds,
                scoring='r2', n_jobs=-1, verbose=1, random_state=42
            )
        else:
            raise ValueError(f"不支持的优化方法: {optimization_method}")

        # 执行搜索
        search.fit(X_train, y_train)

        # 获取最佳模型
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        print(f"✅ 优化完成!")
        print(f"  最佳参数: {best_params}")
        print(f"  交叉验证R²: {best_score:.4f}")

        # 在测试集上评估
        if X_test is not None and y_test is not None:
            test_score = best_model.score(X_test, y_test)
            print(f"  测试集R²: {test_score:.4f}")

        # 保存结果
        self.best_params['random_forest'] = best_params
        self.optimization_history['random_forest'] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_score': test_score if X_test is not None else None
        }

        return best_model, best_params, best_score

    def optimize_linear_regression(self, X_train, y_train, X_test=None, y_test=None):
        """
        优化线性回归模型（特征选择和正则化）

        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据（可选）

        Returns:
            优化后的模型和特征信息
        """
        print("🔧 开始线性回归模型优化...")

        # 基础模型
        lr = LinearRegression()

        # 交叉验证评估
        cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')

        print(f"  交叉验证R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 训练模型
        lr.fit(X_train, y_train)

        # 特征重要性分析（系数绝对值）
        feature_importance = np.abs(lr.coef_)
        feature_ranking = np.argsort(feature_importance)[::-1]

        print("  特征重要性排名:")
        for i, idx in enumerate(feature_ranking):
            if i < len(feature_importance):
                print(f"    {i+1}. 特征{idx}: {feature_importance[idx]:.4f}")

        # 在测试集上评估
        test_score = None
        if X_test is not None and y_test is not None:
            test_score = lr.score(X_test, y_test)
            print(f"  测试集R²: {test_score:.4f}")

        # 保存结果
        self.optimization_history['linear_regression'] = {
            'feature_importance': feature_importance.tolist(),
            'feature_ranking': feature_ranking.tolist(),
            'cv_scores': cv_scores.tolist(),
            'test_score': test_score
        }

        return lr, feature_importance, cv_scores.mean()

    def ensemble_models(self, models, X_test, y_test, weights=None):
        """
        模型集成

        Args:
            models: 模型列表
            X_test, y_test: 测试数据
            weights: 模型权重（可选）

        Returns:
            集成预测结果和性能
        """
        print("🔗 开始模型集成...")

        if weights is None:
            # 简单平均
            weights = [1.0 / len(models)] * len(models)

        # 获取各模型预测
        predictions = []
        model_scores = []

        for i, model in enumerate(models):
            pred = model.predict(X_test)
            predictions.append(pred)
            score = model.score(X_test, y_test)
            model_scores.append(score)
            print(f"  模型{i+1} R²: {score:.4f} (权重: {weights[i]:.3f})")

        # 加权平均预测
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight

        # 计算集成性能
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

        print(f"✅ 集成模型性能:")
        print(f"  R² 分数: {ensemble_r2:.4f}")
        print(f"  MAE: {ensemble_mae:.4f}")
        print(f"  RMSE: {ensemble_rmse:.4f}")

        return ensemble_pred, {
            'r2_score': ensemble_r2,
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'individual_scores': model_scores,
            'weights': weights
        }

    def feature_selection(self, X_train, y_train, X_test=None, y_test=None,
                        method='correlation', threshold=0.1):
        """
        特征选择

        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据（可选）
            method: 选择方法 ('correlation', 'mutual_info', 'rfe')
            threshold: 选择阈值

        Returns:
            选择的特征索引和重要性
        """
        print(f"🎯 开始特征选择 (方法: {method})...")

        if method == 'correlation':
            # 相关性分析
            correlations = []
            for i in range(X_train.shape[1]):
                corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
                correlations.append(abs(corr))

            selected_features = [i for i, corr in enumerate(correlations) if corr > threshold]
            feature_importance = correlations

        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X_train, y_train)
            selected_features = [i for i, score in enumerate(mi_scores) if score > threshold]
            feature_importance = mi_scores

        elif method == 'rfe':
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LinearRegression

            estimator = LinearRegression()
            selector = RFE(estimator, n_features_to_select=max(1, int(threshold * X_train.shape[1])))
            selector.fit(X_train, y_train)

            selected_features = [i for i, selected in enumerate(selector.support_) if selected]
            feature_importance = selector.ranking_

        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        print(f"  原始特征数: {X_train.shape[1]}")
        print(f"  选择特征数: {len(selected_features)}")
        print(f"  选择特征索引: {selected_features}")

        # 在测试集上评估特征选择效果
        if X_test is not None and y_test is not None and len(selected_features) > 0:
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]

            lr = LinearRegression()
            lr.fit(X_train_selected, y_train)
            test_score = lr.score(X_test_selected, y_test)

            # 与原始特征对比
            lr_full = LinearRegression()
            lr_full.fit(X_train, y_train)
            full_score = lr_full.score(X_test, y_test)

            print(f"  选择后模型R²: {test_score:.4f}")
            print(f"  原始模型R²: {full_score:.4f}")
            print(f"  性能变化: {((test_score - full_score) / full_score * 100):+.2f}%")

        return selected_features, feature_importance

    def automated_hyperparameter_tuning(self, X_train, y_train, X_test, y_test,
                                      model_types=['random_forest'], time_limit=300):
        """
        自动化超参数调优

        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            model_types: 要调优的模型类型
            time_limit: 时间限制（秒）

        Returns:
            调优结果摘要
        """
        print("🤖 开始自动化超参数调优...")
        print(f"  时间限制: {time_limit}秒")
        print(f"  调优模型: {model_types}")

        import time
        start_time = time.time()
        results = {}

        for model_type in model_types:
            if time.time() - start_time > time_limit:
                print(f"⏰ 时间限制达到，停止调优")
                break

            print(f"\n  调优 {model_type}...")

            try:
                if model_type == 'random_forest':
                    # 快速随机搜索
                    model, params, score = self.optimize_random_forest(
                        X_train, y_train, X_test, y_test,
                        optimization_method='random', n_iter=20, cv_folds=3
                    )
                    results[model_type] = {
                        'model': model,
                        'best_params': params,
                        'best_score': score
                    }

                elif model_type == 'linear_regression':
                    model, importance, score = self.optimize_linear_regression(
                        X_train, y_train, X_test, y_test
                    )
                    results[model_type] = {
                        'model': model,
                        'feature_importance': importance,
                        'score': score
                    }

            except Exception as e:
                print(f"    ❌ 调优失败: {str(e)}")
                continue

        # 选择最佳模型
        best_model_type = None
        best_score = -np.inf

        for model_type, result in results.items():
            score = result.get('best_score', result.get('score', 0))
            if score > best_score:
                best_score = score
                best_model_type = model_type

        elapsed_time = time.time() - start_time

        print(f"\n✅ 自动调优完成 (耗时: {elapsed_time:.1f}秒)")
        if best_model_type:
            print(f"🏆 最佳模型: {best_model_type} (R²: {best_score:.4f})")

        return results, best_model_type, elapsed_time

    def generate_optimization_report(self, output_path='optimization_report.md'):
        """生成优化报告"""
        print("📋 生成优化报告...")

        report = "# 模型优化报告\n\n"
        report += f"生成时间: {pd.Timestamp.now()}\n\n"

        # 随机森林优化结果
        if 'random_forest' in self.optimization_history:
            rf_result = self.optimization_history['random_forest']
            report += "## 随机森林优化\n\n"
            report += f"最佳参数: {rf_result['best_params']}\n\n"
            report += f"交叉验证R²: {rf_result['best_cv_score']:.4f}\n\n"
            if rf_result.get('test_score'):
                report += f"测试集R²: {rf_result['test_score']:.4f}\n\n"

        # 线性回归优化结果
        if 'linear_regression' in self.optimization_history:
            lr_result = self.optimization_history['linear_regression']
            report += "## 线性回归优化\n\n"
            cv_scores = lr_result['cv_scores']
            report += f"交叉验证R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})\n\n"
            if lr_result.get('test_score'):
                report += f"测试集R²: {lr_result['test_score']:.4f}\n\n"

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 优化报告已保存: {output_path}")
        return output_path