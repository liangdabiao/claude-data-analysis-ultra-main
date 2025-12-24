#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Model Evaluation and Comparison
Comprehensive model evaluation with diagnostics and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                           mean_absolute_percentage_error, explained_variance_score)
from sklearn.model_selection import learning_curve, validation_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Advanced model evaluation toolkit with comprehensive diagnostics"""

    def __init__(self, chinese_font='SimHei'):
        """Initialize model evaluator"""
        self.chinese_font = chinese_font
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False

        self.evaluation_results = {}
        self.comparison_results = {}

    def calculate_comprehensive_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model

        Returns:
            dict: Comprehensive metrics
        """
        # Basic metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)

        # Additional metrics
        mape = mean_absolute_percentage_error(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        # Custom metrics
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(residuals) / (np.abs(y_true) + np.abs(y_pred))) * 100

        # Relative metrics
        y_mean = np.mean(y_true)
        relative_mae = mae / y_mean * 100
        relative_rmse = rmse / y_mean * 100

        metrics = {
            'model_name': model_name,
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'mape': mape,
            'smape': smape,
            'explained_variance': explained_var,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'relative_mae': relative_mae,
            'relative_rmse': relative_rmse,
            'n_samples': len(y_true),
            'mean_target': y_mean,
            'std_target': np.std(y_true)
        }

        self.evaluation_results[model_name] = metrics
        return metrics

    def perform_residual_analysis(self, y_true, y_pred, model_name="Model", save_plots=True):
        """
        Perform comprehensive residual analysis

        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
            save_plots (bool): Whether to save plots

        Returns:
            dict: Residual analysis results
        """
        print(f"\n=== {model_name} 残差分析 ===")

        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals)
        predicted_values = y_pred

        # Statistical tests
        # 1. Normality test for residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)

        # 2. Homoscedasticity test (Breusch-Pagan)
        # Simplified version - check correlation between residuals and predicted values
        homoscedasticity_corr = np.corrcoef(np.abs(residuals), predicted_values)[0, 1]

        # 3. Independence test (Durbin-Watson approximation)
        if len(residuals) > 1:
            dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        else:
            dw_stat = 2.0  # Perfect independence

        # Create diagnostic plots
        if save_plots:
            self._create_residual_plots(y_true, y_pred, model_name)

        results = {
            'model_name': model_name,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'residuals_skewness': stats.skew(residuals),
            'residuals_kurtosis': stats.kurtosis(residuals),
            'shapiro_stat': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'normality_test': 'Pass' if shapiro_p > 0.05 else 'Fail',
            'homoscedasticity_correlation': homoscedasticity_corr,
            'homoscedasticity_test': 'Pass' if abs(homoscedasticity_corr) < 0.3 else 'Fail',
            'durbin_watson_stat': dw_stat,
            'independence_test': 'Pass' if 1.5 < dw_stat < 2.5 else 'Fail'
        }

        # Print summary
        print(f"残差均值: {results['residuals_mean']:.4f}")
        print(f"残差标准差: {results['residuals_std']:.4f}")
        print(f"正态性检验: {results['normality_test']} (p-value: {shapiro_p:.4f})")
        print(f"同方差性检验: {results['homoscedasticity_test']} (相关性: {homoscedasticity_corr:.4f})")
        print(f"独立性检验: {results['independence_test']} (DW统计量: {dw_stat:.4f})")

        return results

    def _create_residual_plots(self, y_true, y_pred, model_name):
        """Create comprehensive residual diagnostic plots"""
        residuals = y_true - y_pred
        predicted_values = y_pred

        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - 残差诊断图', fontsize=16, fontweight='bold')

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(predicted_values, residuals, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('预测值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差 vs 拟合值')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q图 (正态性检验)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Scale-Location Plot
        axes[0, 2].scatter(predicted_values, np.sqrt(np.abs(residuals)), alpha=0.6, s=20)
        axes[0, 2].set_xlabel('预测值')
        axes[0, 2].set_ylabel('√|残差|')
        axes[0, 2].set_title('Scale-Location图')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Histogram of Residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('残差直方图')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Actual vs Predicted
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 1].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('实际值')
        axes[1, 1].set_ylabel('预测值')
        axes[1, 1].set_title('实际值 vs 预测值')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Residuals vs Order (if index is available)
        axes[1, 2].plot(range(len(residuals)), residuals, alpha=0.6)
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].set_xlabel('观测顺序')
        axes[1, 2].set_ylabel('残差')
        axes[1, 2].set_title('残差 vs 观测顺序')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{model_name}_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_models(self, model_results, save_comparison=True):
        """
        Compare multiple models with comprehensive metrics

        Args:
            model_results (dict): Dictionary with model results
            save_comparison (bool): Whether to save comparison plots

        Returns:
            pd.DataFrame: Model comparison results
        """
        print("\n=== 模型比较分析 ===")

        # Create comparison DataFrame
        comparison_data = []

        for model_name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
            elif hasattr(results, '__dict__'):
                # Handle model objects
                metrics = self.calculate_comprehensive_metrics(
                    results['y_test'], results['predictions'], model_name
                )
            else:
                continue

            comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)

        if len(comparison_df) == 0:
            print("没有可比较的模型结果")
            return pd.DataFrame()

        # Rank models by different metrics
        ranking_metrics = ['r2_score', 'mae', 'rmse', 'mape']
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                if metric == 'r2_score':
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
                else:
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=True)

        # Calculate overall rank
        rank_columns = [col for col in comparison_df.columns if col.endswith('_rank')]
        if rank_columns:
            comparison_df['overall_rank'] = comparison_df[rank_columns].mean(axis=1)
            comparison_df = comparison_df.sort_values('overall_rank')

        self.comparison_results = comparison_df

        # Print comparison summary
        print("\n模型性能排名:")
        for idx, row in comparison_df.iterrows():
            print(f"{idx + 1}. {row['model_name']}: "
                  f"R²={row.get('r2_score', 'N/A'):.4f}, "
                  f"MAE={row.get('mae', 'N/A'):.4f}, "
                  f"RMSE={row.get('rmse', 'N/A'):.4f}")

        if save_comparison:
            self._create_comparison_plots(comparison_df)

        return comparison_df

    def _create_comparison_plots(self, comparison_df):
        """Create model comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能比较', fontsize=16, fontweight='bold')

        # 1. R² Score Comparison
        if 'r2_score' in comparison_df.columns:
            bars = axes[0, 0].bar(comparison_df['model_name'], comparison_df['r2_score'])
            axes[0, 0].set_title('R² 分数比较')
            axes[0, 0].set_ylabel('R² 分数')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, comparison_df['r2_score']):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.4f}', ha='center', va='bottom')

        # 2. MAE Comparison
        if 'mae' in comparison_df.columns:
            bars = axes[0, 1].bar(comparison_df['model_name'], comparison_df['mae'], color='orange')
            axes[0, 1].set_title('平均绝对误差比较')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, comparison_df['mae']):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.2f}', ha='center', va='bottom')

        # 3. RMSE Comparison
        if 'rmse' in comparison_df.columns:
            bars = axes[1, 0].bar(comparison_df['model_name'], comparison_df['rmse'], color='green')
            axes[1, 0].set_title('均方根误差比较')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, comparison_df['rmse']):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.2f}', ha='center', va='bottom')

        # 4. Overall Ranking
        if 'overall_rank' in comparison_df.columns:
            bars = axes[1, 1].bar(comparison_df['model_name'], comparison_df['overall_rank'], color='purple')
            axes[1, 1].set_title('综合排名 (越小越好)')
            axes[1, 1].set_ylabel('排名分数')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].invert_yaxis()  # Lower rank is better

            # Add value labels
            for bar, value in zip(bars, comparison_df['overall_rank']):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_learning_curves(self, model, X, y, cv=5, train_sizes=None, save_plot=True):
        """
        Analyze learning curves to diagnose model performance

        Args:
            model: Trained model object
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Cross-validation folds
            train_sizes (array): Training sizes to evaluate
            save_plot (bool): Whether to save the plot

        Returns:
            dict: Learning curve analysis results
        """
        print("\n=== 学习曲线分析 ===")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Calculate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='r2', random_state=42, n_jobs=-1
        )

        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Analyze curve characteristics
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        overfitting_gap = final_train_score - final_val_score

        # Determine if model is overfitting, underfitting, or well-balanced
        if overfitting_gap > 0.1:
            model_status = "过拟合"
            recommendation = "增加正则化、获取更多数据、简化模型"
        elif final_val_score < 0.5:
            model_status = "欠拟合"
            recommendation = "增加特征复杂度、使用更强大的模型"
        else:
            model_status = "平衡良好"
            recommendation = "模型性能良好，可以考虑微调"

        results = {
            'model_status': model_status,
            'final_train_score': final_train_score,
            'final_val_score': final_val_score,
            'overfitting_gap': overfitting_gap,
            'recommendation': recommendation,
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_mean,
            'train_scores_std': train_std,
            'val_scores_mean': val_mean,
            'val_scores_std': val_std
        }

        print(f"模型状态: {model_status}")
        print(f"最终训练分数: {final_train_score:.4f}")
        print(f"最终验证分数: {final_val_score:.4f}")
        print(f"过拟合程度: {overfitting_gap:.4f}")
        print(f"建议: {recommendation}")

        if save_plot:
            self._create_learning_curve_plot(results)

        return results

    def _create_learning_curve_plot(self, results):
        """Create learning curve visualization"""
        plt.figure(figsize=(10, 6))

        # Plot training scores
        plt.plot(results['train_sizes'], results['train_scores_mean'], 'o-',
                color='blue', label='训练分数')
        plt.fill_between(results['train_sizes'],
                        results['train_scores_mean'] - results['train_scores_std'],
                        results['train_scores_mean'] + results['train_scores_std'],
                        alpha=0.1, color='blue')

        # Plot validation scores
        plt.plot(results['train_sizes'], results['val_scores_mean'], 'o-',
                color='red', label='验证分数')
        plt.fill_between(results['train_sizes'],
                        results['val_scores_mean'] - results['val_scores_std'],
                        results['val_scores_mean'] + results['val_scores_std'],
                        alpha=0.1, color='red')

        plt.xlabel('训练样本数')
        plt.ylabel('R² 分数')
        plt.title('学习曲线分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_evaluation_report(self, model_results, save_to_file=True):
        """
        Generate comprehensive evaluation report

        Args:
            model_results (dict): Dictionary with model results
            save_to_file (bool): Whether to save report to file

        Returns:
            str: Evaluation report
        """
        print("\n=== 生成评估报告 ===")

        # Compare models
        comparison_df = self.compare_models(model_results, save_comparison=False)

        if len(comparison_df) == 0:
            return "没有模型结果可用于生成报告"

        # Generate report
        report = f"""
# 回归模型评估报告
# Regression Model Evaluation Report

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 模型性能排名 | Model Performance Ranking

| 排名 | 模型名称 | R² 分数 | MAE | RMSE | MAPE | 综合评分 |
|------|----------|---------|-----|------|------|----------|
"""

        for idx, (_, row) in enumerate(comparison_df.iterrows(), 1):
            report += f"| {idx} | {row['model_name']} | {row.get('r2_score', 'N/A'):.4f} | {row.get('mae', 'N/A'):.2f} | {row.get('rmse', 'N/A'):.2f} | {row.get('mape', 'N/A'):.4f} | {row.get('overall_rank', 'N/A'):.2f} |\n"

        # Best model analysis
        best_model = comparison_df.iloc[0]
        report += f"""

## 最佳模型分析 | Best Model Analysis

**模型名称:** {best_model['model_name']}

### 性能指标 | Performance Metrics
- **R² 分数:** {best_model.get('r2_score', 'N/A'):.4f}
- **平均绝对误差 (MAE):** {best_model.get('mae', 'N/A'):.2f}
- **均方根误差 (RMSE):** {best_model.get('rmse', 'N/A'):.2f}
- **平均绝对百分比误差 (MAPE):** {best_model.get('mape', 'N/A'):.4f}
- **解释方差比:** {best_model.get('explained_variance', 'N/A'):.4f}

### 模型特征 | Model Characteristics
- **样本数量:** {best_model.get('n_samples', 'N/A')}
- **目标变量均值:** {best_model.get('mean_target', 'N/A'):.2f}
- **相对误差 (MAE):** {best_model.get('relative_mae', 'N/A'):.2f}%

## 诊断建议 | Diagnostic Recommendations

"""

        # Add specific recommendations based on model performance
        if best_model.get('r2_score', 0) > 0.8:
            report += "✅ **模型表现优秀** - 模型具有很强的预测能力\n"
        elif best_model.get('r2_score', 0) > 0.6:
            report += "⚠️ **模型表现良好** - 模型具有较好的预测能力，仍有改进空间\n"
        else:
            report += "❌ **模型表现需要改进** - 建议尝试特征工程、更复杂的模型或数据预处理\n"

        # Overfitting analysis
        if 'train_r2' in best_model and 'test_r2' in best_model:
            overfitting = best_model['train_r2'] - best_model['test_r2']
            if overfitting > 0.2:
                report += f"\n⚠️ **存在过拟合** - 训练和测试分数差距较大 ({overfitting:.3f})\n"
                report += "   建议: 增加正则化、使用交叉验证或获取更多训练数据\n"

        # Feature engineering suggestions
        report += f"""

## 改进建议 | Improvement Recommendations

### 1. 数据层面 | Data Level
- 考虑添加更多相关特征
- 处理异常值和缺失值
- 检查数据质量和一致性

### 2. 特征工程 | Feature Engineering
- 尝试特征交互和多项式特征
- 使用特征选择方法
- 考虑特征缩放和标准化

### 3. 模型层面 | Model Level
- 尝试不同的回归算法
- 调整超参数
- 使用集成学习方法

### 4. 验证方法 | Validation
- 使用时间序列交叉验证（如果数据有时间依赖）
- 考虑分层抽样
- 增加验证的稳定性

## 可视化文件 | Generated Visualizations

以下可视化文件已生成:
- `model_comparison.png`: 模型性能比较图
- `learning_curves.png`: 学习曲线分析图
- `*_residual_analysis.png`: 各模型残差诊断图

---

*报告由回归模型评估系统自动生成*
"""

        if save_to_file:
            with open('model_evaluation_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            print("评估报告已保存到: model_evaluation_report.md")

        return report

def main():
    """Example usage"""
    evaluator = ModelEvaluator()

    # Create sample data
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 1000)
    y_pred_1 = y_true + np.random.normal(0, 5, 1000)  # Good model
    y_pred_2 = y_true + np.random.normal(0, 15, 1000)  # Poor model

    # Calculate metrics
    metrics_1 = evaluator.calculate_comprehensive_metrics(y_true, y_pred_1, "Good Model")
    metrics_2 = evaluator.calculate_comprehensive_metrics(y_true, y_pred_2, "Poor Model")

    print("Good Model Metrics:")
    for key, value in metrics_1.items():
        if key != 'model_name':
            print(f"  {key}: {value:.4f}")

    # Residual analysis
    residual_results_1 = evaluator.perform_residual_analysis(y_true, y_pred_1, "Good Model")

if __name__ == "__main__":
    main()