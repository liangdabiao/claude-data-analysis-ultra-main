#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Prediction Visualization Tools
Comprehensive visualization for regression analysis and model interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class PredictionVisualizer:
    """Advanced visualization toolkit for regression analysis"""

    def __init__(self, chinese_font='SimHei', style='seaborn-v0_8'):
        """Initialize the visualizer"""
        self.chinese_font = chinese_font
        self.style = style

        # Set up plotting style
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

    def create_comprehensive_dashboard(self, model_results, feature_importance=None,
                                     save_path='regression_dashboard.png'):
        """
        Create comprehensive regression analysis dashboard

        Args:
            model_results (dict): Dictionary containing model results
            feature_importance (pd.DataFrame): Feature importance data
            save_path (str): Path to save the dashboard
        """
        print("创建综合回归分析仪表板...")

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Model Performance Comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_model_performance_comparison(model_results, ax1)

        # 2. Feature Importance (top right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        if feature_importance is not None:
            self._plot_feature_importance(feature_importance, ax2)
        else:
            self._plot_placeholder(ax2, "特征重要性数据不可用")

        # 3. Prediction vs Actual (middle left)
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_prediction_vs_actual(model_results, ax3)

        # 4. Residual Analysis (middle right)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_residual_patterns(model_results, ax4)

        # 5. Error Distribution (bottom left)
        ax5 = fig.add_subplot(gs[2, 0:2])
        self._plot_error_distribution(model_results, ax5)

        # 6. Performance Metrics Summary (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:4])
        self._plot_metrics_summary(model_results, ax6)

        # 7. Model Rankings (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_model_rankings(model_results, ax7)

        # Add main title
        best_model_name = self._get_best_model_name(model_results)
        fig.suptitle(f'回归分析综合仪表板\nRegression Analysis Dashboard (最佳模型: {best_model_name})',
                     fontsize=20, fontweight='bold', y=0.95)

        # Add footer with analysis info
        fig.text(0.5, 0.02, f'分析时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                              f'模型数量: {len(model_results)} | 样本数量: {self._get_sample_count(model_results)}',
                 ha='center', fontsize=10, style='italic')

        # Save the dashboard
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"仪表板已保存: {save_path}")

    def _plot_model_performance_comparison(self, model_results, ax):
        """Plot model performance comparison"""
        models = []
        r2_scores = []
        maes = []

        for name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                models.append(name)
                r2_scores.append(metrics.get('r2_score', 0))
                maes.append(metrics.get('mae', 0))

        if not models:
            self._plot_placeholder(ax, "没有模型性能数据")
            return

        # Create dual-axis plot
        x = np.arange(len(models))
        width = 0.35

        # R² scores (left axis)
        bars1 = ax.bar(x - width/2, r2_scores, width, label='R² 分数', alpha=0.7, color='skyblue')
        ax.set_ylabel('R² 分数', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # MAE (right axis)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, maes, width, label='MAE', alpha=0.7, color='orange')
        ax2.set_ylabel('平均绝对误差', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Customize
        ax.set_xlabel('模型')
        ax.set_title('模型性能比较', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, feature_importance, ax):
        """Plot feature importance"""
        if feature_importance is None or len(feature_importance) == 0:
            self._plot_placeholder(ax, "特征重要性数据不可用")
            return

        # Take top 15 features
        top_features = feature_importance.head(15)

        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('重要性分数')
        ax.set_title('特征重要性排名 (Top 15)', fontweight='bold')

        # Add importance values
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)

        ax.grid(True, alpha=0.3, axis='x')

    def _plot_prediction_vs_actual(self, model_results, ax):
        """Plot prediction vs actual scatter plots"""
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        for i, (name, results) in enumerate(model_results.items()):
            if 'y_test' in results and 'predictions' in results:
                y_true = results['y_test']
                y_pred = results['predictions']

                # Sample points if too many
                if len(y_true) > 1000:
                    indices = np.random.choice(len(y_true), 1000, replace=False)
                    y_true = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
                    y_pred = y_pred[indices]

                color = colors[i % len(colors)]
                ax.scatter(y_true, y_pred, alpha=0.5, s=20, label=name, color=color)

        # Add perfect prediction line
        if model_results:
            # Find data range
            all_values = []
            for results in model_results.values():
                if 'y_test' in results and 'predictions' in results:
                    all_values.extend(results['y_test'])
                    all_values.extend(results['predictions'])

            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8,
                       label='完美预测线')

        ax.set_xlabel('实际值')
        ax.set_ylabel('预测值')
        ax.set_title('预测值 vs 实际值', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_residual_patterns(self, model_results, ax):
        """Plot residual patterns"""
        best_model = None
        best_results = None
        best_r2 = -1

        # Find best model
        for name, results in model_results.items():
            if 'metrics' in results:
                r2 = results['metrics'].get('r2_score', -1)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
                    best_results = results

        if best_results is None or 'y_test' not in best_results:
            self._plot_placeholder(ax, "没有残差分析数据")
            return

        y_true = best_results['y_test']
        y_pred = best_results['predictions']
        residuals = y_true - y_pred

        # Create residual plot
        ax.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('预测值')
        ax.set_ylabel('残差')
        ax.set_title(f'残差分析 ({best_model})', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(y_pred) > 1:
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            ax.plot(y_pred, p(y_pred), "r--", alpha=0.8, linewidth=2)

    def _plot_error_distribution(self, model_results, ax):
        """Plot error distribution histogram"""
        for name, results in model_results.items():
            if 'y_test' in results and 'predictions' in results:
                y_true = results['y_test']
                y_pred = results['predictions']
                errors = y_true - y_pred

                # Filter extreme errors for better visualization
                q95 = np.percentile(np.abs(errors), 95)
                filtered_errors = errors[np.abs(errors) <= q95]

                ax.hist(filtered_errors, alpha=0.6, bins=30, label=name)

        ax.set_xlabel('预测误差')
        ax.set_ylabel('频次')
        ax.set_title('误差分布', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)

    def _plot_metrics_summary(self, model_results, ax):
        """Plot metrics summary table"""
        if not model_results:
            self._plot_placeholder(ax, "没有指标数据")
            return

        # Prepare data
        models = []
        metrics_data = {'R²': [], 'MAE': [], 'RMSE': []}

        for name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                models.append(name)
                metrics_data['R²'].append(metrics.get('r2_score', 0))
                metrics_data['MAE'].append(metrics.get('mae', 0))
                metrics_data['RMSE'].append(metrics.get('rmse', 0))

        if not models:
            self._plot_placeholder(ax, "没有有效的指标数据")
            return

        # Create table
        table_data = []
        table_data.append(['模型'] + list(metrics_data.keys()))

        for i, model in enumerate(models):
            row = [model[:15]]  # Truncate long model names
            for metric in metrics_data.keys():
                if metric == 'R²':
                    row.append(f"{metrics_data[metric][i]:.4f}")
                else:
                    row.append(f"{metrics_data[metric][i]:.2f}")
            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3] + [0.2] * len(metrics_data))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('性能指标汇总', fontweight='bold')
        ax.axis('off')

    def _plot_model_rankings(self, model_results, ax):
        """Plot model rankings visualization"""
        # Calculate overall scores
        model_scores = {}
        for name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                # Simple scoring: R² (60%) + (1-normalized_MAE) (20%) + (1-normalized_RMSE) (20%)
                r2 = metrics.get('r2_score', 0)
                mae = metrics.get('mae', float('inf'))
                rmse = metrics.get('rmse', float('inf'))

                # Normalize MAE and RMSE (lower is better)
                mae_scores = [results['metrics'].get('mae', float('inf'))
                             for results in model_results.values() if 'metrics' in results]
                rmse_scores = [results['metrics'].get('rmse', float('inf'))
                               for results in model_results.values() if 'metrics' in results]

                if mae_scores and rmse_scores and max(mae_scores) > 0 and max(rmse_scores) > 0:
                    mae_norm = 1 - (mae / max(mae_scores))
                    rmse_norm = 1 - (rmse / max(rmse_scores))
                    overall_score = 0.6 * r2 + 0.2 * mae_norm + 0.2 * rmse_norm
                else:
                    overall_score = r2

                model_scores[name] = overall_score

        if not model_scores:
            self._plot_placeholder(ax, "无法计算模型排名")
            return

        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        # Create ranking visualization
        models = [model[0] for model in sorted_models]
        scores = [model[1] for model in sorted_models]

        bars = ax.bar(range(len(models)), scores, color=plt.cm.RdYlGn(np.array(scores)))
        ax.set_xlabel('模型')
        ax.set_ylabel('综合评分')
        ax.set_title('模型综合排名 (得分越高越好)', fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels and rankings
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}\n#{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight top model
        if bars:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkgoldenrod')
            bars[0].set_linewidth(2)

    def _plot_placeholder(self, ax, message):
        """Plot placeholder text"""
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12,
               transform=ax.transAxes, style='italic', color='gray')
        ax.set_title('数据不可用', fontweight='bold')
        ax.axis('off')

    def _get_best_model_name(self, model_results):
        """Get the name of the best performing model"""
        best_r2 = -1
        best_model = "Unknown"

        for name, results in model_results.items():
            if 'metrics' in results:
                r2 = results['metrics'].get('r2_score', -1)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name

        return best_model

    def _get_sample_count(self, model_results):
        """Get sample count from model results"""
        for results in model_results.values():
            if 'metrics' in results:
                return results['metrics'].get('n_samples', 'Unknown')
        return 'Unknown'

    def create_individual_analysis_plots(self, model_results, output_dir='analysis_plots'):
        """
        Create individual detailed analysis plots for each model

        Args:
            model_results (dict): Model results dictionary
            output_dir (str): Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("生成详细分析图表...")

        for name, results in model_results.items():
            if 'y_test' in results and 'predictions' in results:
                self._create_detailed_model_analysis(name, results, output_dir)

        print(f"详细图表已保存到: {output_dir}/")

    def _create_detailed_model_analysis(self, model_name, results, output_dir):
        """Create detailed analysis for a specific model"""
        y_true = results['y_test']
        y_pred = results['predictions']

        # Create detailed subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - 详细模型分析', fontsize=16, fontweight='bold')

        # 1. Prediction vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('实际值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('预测值 vs 实际值')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals vs Predicted
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差 vs 预测值')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals Histogram
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('残差')
        axes[0, 2].set_ylabel('频次')
        axes[0, 2].set_title('残差分布')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Error Percentiles
        error_percentiles = np.percentile(np.abs(residuals), [25, 50, 75, 90, 95, 99])
        percentiles = [25, 50, 75, 90, 95, 99]
        bars = axes[1, 0].bar(range(len(percentiles)), error_percentiles)
        axes[1, 0].set_xlabel('百分位数')
        axes[1, 0].set_ylabel('绝对误差')
        axes[1, 0].set_title('误差百分位数分析')
        axes[1, 0].set_xticks(range(len(percentiles)))
        axes[1, 0].set_xticklabels([f'{p}%' for p in percentiles])
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, value in zip(bars, error_percentiles):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # 5. Prediction Intervals
        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        residual_std = np.std(residuals)

        axes[1, 1].plot(range(len(y_true_sorted)), y_true_sorted, 'b-', alpha=0.7, label='实际值')
        axes[1, 1].plot(range(len(y_pred_sorted)), y_pred_sorted, 'r-', alpha=0.7, label='预测值')
        axes[1, 1].fill_between(range(len(y_pred_sorted)),
                                y_pred_sorted - 1.96 * residual_std,
                                y_pred_sorted + 1.96 * residual_std,
                                alpha=0.2, color='red', label='95% 置信区间')
        axes[1, 1].set_xlabel('样本索引 (按实际值排序)')
        axes[1, 1].set_ylabel('值')
        axes[1, 1].set_title('预测区间分析')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Model Metrics Summary
        if 'metrics' in results:
            metrics = results['metrics']
            metrics_text = f"""
            R² 分数: {metrics.get('r2_score', 'N/A'):.4f}
            MAE: {metrics.get('mae', 'N/A'):.4f}
            RMSE: {metrics.get('rmse', 'N/A'):.4f}
            MAPE: {metrics.get('mape', 'N/A'):.4f}
            样本数量: {metrics.get('n_samples', 'N/A')}
            """
            axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                           transform=axes[1, 2].transAxes, family='monospace')
            axes[1, 2].set_title('模型指标摘要')
            axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage"""
    visualizer = PredictionVisualizer()

    # Create sample model results
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.normal(100, 20, n_samples)

    model_results = {
        'Linear Regression': {
            'y_test': pd.Series(y_true),
            'predictions': y_true + np.random.normal(0, 5, n_samples),
            'metrics': {
                'r2_score': 0.94,
                'mae': 4.1,
                'rmse': 5.2,
                'mape': 0.041,
                'n_samples': n_samples
            }
        },
        'Random Forest': {
            'y_test': pd.Series(y_true),
            'predictions': y_true + np.random.normal(0, 4, n_samples),
            'metrics': {
                'r2_score': 0.96,
                'mae': 3.2,
                'rmse': 4.1,
                'mape': 0.032,
                'n_samples': n_samples
            }
        }
    }

    # Create feature importance
    feature_importance = pd.DataFrame({
        'feature': ['特征1', '特征2', '特征3', '特征4', '特征5'],
        'importance': [0.4, 0.3, 0.15, 0.1, 0.05]
    })

    # Create dashboard
    visualizer.create_comprehensive_dashboard(model_results, feature_importance)
    print("示例仪表板已生成: regression_dashboard.png")

if __name__ == "__main__":
    main()