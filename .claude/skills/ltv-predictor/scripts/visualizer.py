#!/usr/bin/env python3
"""
可视化模块
基于第3课理论实现的LTV分析可视化功能
支持RFM分布图、模型性能对比、特征重要性分析等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataVisualizer:
    """
    数据可视化器

    专门用于LTV分析结果的可视化展示
    支持RFM分析、模型性能、客户分群等多维度可视化
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化可视化器

        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            'figure_size': (12, 8),
            'style': 'whitegrid',
            'color_palette': 'husl',
            'save_format': 'png',
            'dpi': 300,
            'customer_column': '用户码',
            'feature_columns': ['R值', 'F值', 'M值'],
            'target_column': '年度LTV'
        }

        # 更新配置
        if config:
            self.config.update(config)

        # 设置样式
        sns.set_style(self.config['style'])
        sns.set_palette(self.config['color_palette'])

        # 存储图表
        self.charts = {}
        self.chart_count = 0

        print("📈 数据可视化器初始化完成")

    def plot_rfm_distribution(self, rfm_data: pd.DataFrame,
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        绘制RFM特征分布图

        Args:
            rfm_data: RFM特征数据
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RFM特征分布分析', fontsize=16, fontweight='bold')

        # R值分布
        axes[0, 0].hist(rfm_data['R值'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('R值分布 (最近消费时间间隔)', fontsize=14)
        axes[0, 0].set_xlabel('天数')
        axes[0, 0].set_ylabel('客户数量')
        axes[0, 0].axvline(rfm_data['R值'].mean(), color='red', linestyle='--',
                          label=f'平均值: {rfm_data["R值"].mean():.1f}')
        axes[0, 0].legend()

        # F值分布
        axes[0, 1].hist(rfm_data['F值'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('F值分布 (消费频率)', fontsize=14)
        axes[0, 1].set_xlabel('消费次数')
        axes[0, 1].set_ylabel('客户数量')
        axes[0, 1].axvline(rfm_data['F值'].mean(), color='red', linestyle='--',
                          label=f'平均值: {rfm_data["F值"].mean():.1f}')
        axes[0, 1].legend()

        # M值分布
        axes[1, 0].hist(rfm_data['M值'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].set_title('M值分布 (消费金额)', fontsize=14)
        axes[1, 0].set_xlabel('总金额')
        axes[1, 0].set_ylabel('客户数量')
        axes[1, 0].axvline(rfm_data['M值'].mean(), color='red', linestyle='--',
                          label=f'平均值: {rfm_data["M值"].mean():.2f}')
        axes[1, 0].legend()

        # LTV分布
        axes[1, 1].hist(rfm_data[self.config['target_column']], bins=30, alpha=0.7,
                       color='gold', edgecolor='black')
        axes[1, 1].set_title(f'{self.config["target_column"]}分布', fontsize=14)
        axes[1, 1].set_xlabel('生命周期价值')
        axes[1, 1].set_ylabel('客户数量')
        axes[1, 1].axvline(rfm_data[self.config['target_column']].mean(), color='red', linestyle='--',
                          label=f'平均值: {rfm_data[self.config["target_column"]].mean():.2f}')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"✓ RFM分布图已保存: {save_path}")

        if show_plot:
            plt.show()

        self.chart_count += 1
        self.charts['rfm_distribution'] = fig

        return fig

    def plot_customer_segments(self, rfm_data: pd.DataFrame,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
        """
        绘制客户分群分析图

        Args:
            rfm_data: 包含客户分群的RFM数据
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            matplotlib图表对象
        """
        if '客户价值分层' not in rfm_data.columns:
            print("⚠️ 数据中缺少客户分群信息，跳过分群可视化")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('客户价值分层分析', fontsize=16, fontweight='bold')

        # 1. 各层级客户数量分布
        segment_counts = rfm_data['客户价值分层'].value_counts()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=sns.color_palette('viridis', len(segment_counts)))
        axes[0, 0].set_title('客户价值分层分布')

        # 2. 各层级平均LTV
        segment_ltv = rfm_data.groupby('客户价值分层')[self.config['target_column']].mean().sort_values(ascending=False)
        bars = axes[0, 1].bar(range(len(segment_ltv)), segment_ltv.values,
                             color=sns.color_palette('viridis', len(segment_ltv)))
        axes[0, 1].set_title('各层级平均LTV')
        axes[0, 1].set_xticks(range(len(segment_ltv)))
        axes[0, 1].set_xticklabels(segment_ltv.index, rotation=45)
        axes[0, 1].set_ylabel('平均LTV')

        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.0f}', ha='center', va='bottom')

        # 3. RFM散点图 (F值 vs M值)
        scatter = axes[1, 0].scatter(rfm_data['F值'], rfm_data['M值'],
                                   c=rfm_data[self.config['target_column']],
                                   cmap='viridis', alpha=0.6, s=50)
        axes[1, 0].set_xlabel('F值 (消费频率)')
        axes[1, 0].set_ylabel('M值 (消费金额)')
        axes[1, 0].set_title('F值 vs M值 (颜色表示LTV)')
        plt.colorbar(scatter, ax=axes[1, 0], label='LTV')

        # 4. 各层级RFM特征对比
        if all(col in rfm_data.columns for col in self.config['feature_columns']):
            segment_rfm = rfm_data.groupby('客户价值分层')[self.config['feature_columns']].mean()

            x = np.arange(len(segment_rfm.columns))
            width = 0.8 / len(segment_rfm)

            for i, segment in enumerate(segment_rfm.index):
                offset = (i - len(segment_rfm)/2 + 0.5) * width
                axes[1, 1].bar(x + offset, segment_rfm.loc[segment], width,
                             label=segment, alpha=0.8)

            axes[1, 1].set_xlabel('RFM特征')
            axes[1, 1].set_ylabel('平均值')
            axes[1, 1].set_title('各层级RFM特征对比')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(['R值', 'F值', 'M值'])
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"✓ 客户分群图已保存: {save_path}")

        if show_plot:
            plt.show()

        self.chart_count += 1
        self.charts['customer_segments'] = fig

        return fig

    def plot_model_performance(self, model_results: Dict[str, Dict],
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
        """
        绘制模型性能对比图

        Args:
            model_results: 模型训练结果
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            matplotlib图表对象
        """
        # 过滤有效的模型结果
        valid_models = {name: result for name, result in model_results.items()
                       if 'error' not in result and 'r2_score' in result}

        if not valid_models:
            print("⚠️ 没有有效的模型结果，跳过性能对比可视化")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型性能对比分析', fontsize=16, fontweight='bold')

        model_names = list(valid_models.keys())

        # 1. R²分数对比
        r2_scores = [valid_models[name]['r2_score'] for name in model_names]
        bars1 = axes[0, 0].bar(model_names, r2_scores, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('R²分数对比')
        axes[0, 0].set_ylabel('R²分数')
        axes[0, 0].set_ylim(0, max(r2_scores) * 1.1)

        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.4f}', ha='center', va='bottom')

        # 2. MAE对比
        mae_scores = [valid_models[name].get('mae', 0) for name in model_names]
        bars2 = axes[0, 1].bar(model_names, mae_scores, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('平均绝对误差(MAE)对比')
        axes[0, 1].set_ylabel('MAE')

        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.2f}', ha='center', va='bottom')

        # 3. RMSE对比
        rmse_scores = [valid_models[name].get('rmse', 0) for name in model_names]
        bars3 = axes[1, 0].bar(model_names, rmse_scores, alpha=0.7, color='salmon')
        axes[1, 0].set_title('均方根误差(RMSE)对比')
        axes[1, 0].set_ylabel('RMSE')

        # 添加数值标签
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.2f}', ha='center', va='bottom')

        # 4. MAPE对比
        mape_scores = [valid_models[name].get('mape', 0) for name in model_names]
        bars4 = axes[1, 1].bar(model_names, mape_scores, alpha=0.7, color='gold')
        axes[1, 1].set_title('平均绝对百分比误差(MAPE)对比')
        axes[1, 1].set_ylabel('MAPE (%)')

        # 添加数值标签
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.2f}%', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"✓ 模型性能对比图已保存: {save_path}")

        if show_plot:
            plt.show()

        self.chart_count += 1
        self.charts['model_performance'] = fig

        return fig

    def plot_feature_importance(self, feature_importance: Dict[str, Dict[str, float]],
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> plt.Figure:
        """
        绘制特征重要性分析图

        Args:
            feature_importance: 特征重要性字典
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            matplotlib图表对象
        """
        if not feature_importance:
            print("⚠️ 没有特征重要性数据，跳过可视化")
            return None

        # 选择最佳模型的特征重要性
        best_model = None
        for model_name, importance in feature_importance.items():
            if importance and best_model is None:
                best_model = model_name
            elif importance and model_name == 'random_forest':
                best_model = model_name
                break

        if best_model is None or not feature_importance[best_model]:
            print("⚠️ 没有有效的特征重要性数据，跳过可视化")
            return None

        best_importance = feature_importance[best_model]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'特征重要性分析 ({best_model})', fontsize=16, fontweight='bold')

        # 1. 特征重要性条形图
        features = list(best_importance.keys())
        importances = list(best_importance.values())

        bars = axes[0].barh(features, importances, alpha=0.7, color='skyblue')
        axes[0].set_title('特征重要性排序')
        axes[0].set_xlabel('重要性分数')

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0].text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center')

        # 2. 特征重要性饼图
        colors = sns.color_palette('viridis', len(features))
        wedges, texts, autotexts = axes[1].pie(importances, labels=features, autopct='%1.1f%%',
                                              startangle=90, colors=colors)
        axes[1].set_title('特征重要性占比')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"✓ 特征重要性图已保存: {save_path}")

        if show_plot:
            plt.show()

        self.chart_count += 1
        self.charts['feature_importance'] = fig

        return fig

    def plot_prediction_analysis(self, y_true: pd.Series, y_pred: np.ndarray,
                               model_name: str = "模型",
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> plt.Figure:
        """
        绘制预测结果分析图

        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} 预测结果分析', fontsize=16, fontweight='bold')

        # 1. 预测值 vs 真实值散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                       'r--', lw=2, label='完美预测线')
        axes[0, 0].set_xlabel('真实LTV')
        axes[0, 0].set_ylabel('预测LTV')
        axes[0, 0].set_title('预测值 vs 真实值')
        axes[0, 0].legend()

        # 计算R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 2. 残差分析图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测LTV')
        axes[0, 1].set_ylabel('残差 (真实值 - 预测值)')
        axes[0, 1].set_title('残差分析')

        # 3. 真实值分布
        axes[1, 0].hist(y_true, bins=30, alpha=0.7, label='真实值', color='blue', density=True)
        axes[1, 0].hist(y_pred, bins=30, alpha=0.7, label='预测值', color='red', density=True)
        axes[1, 0].set_xlabel('LTV值')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('LTV分布对比')
        axes[1, 0].legend()

        # 4. 误差分布
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('残差')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('残差分布')
        axes[1, 1].axvline(0, color='red', linestyle='--', label='零误差线')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"✓ 预测分析图已保存: {save_path}")

        if show_plot:
            plt.show()

        self.chart_count += 1
        self.charts['prediction_analysis'] = fig

        return fig

    def plot_correlation_matrix(self, rfm_data: pd.DataFrame,
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> plt.Figure:
        """
        绘制特征相关性矩阵热图

        Args:
            rfm_data: RFM特征数据
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            matplotlib图表对象
        """
        # 选择数值型特征
        numeric_features = self.config['feature_columns'] + [self.config['target_column']]
        available_features = [col for col in numeric_features if col in rfm_data.columns]

        if len(available_features) < 2:
            print("⚠️ 可用数值特征不足，跳过相关性分析")
            return None

        correlation_matrix = rfm_data[available_features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        # 创建热图
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)

        ax.set_title('特征相关性矩阵', fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"✓ 相关性矩阵图已保存: {save_path}")

        if show_plot:
            plt.show()

        self.chart_count += 1
        self.charts['correlation_matrix'] = fig

        return fig

    def generate_dashboard(self, rfm_data: pd.DataFrame,
                          model_results: Optional[Dict[str, Dict]] = None,
                          feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                          predictions: Optional[Tuple[pd.Series, np.ndarray, str]] = None,
                          output_dir: str = './charts') -> Dict[str, str]:
        """
        生成完整的分析仪表板

        Args:
            rfm_data: RFM特征数据
            model_results: 模型训练结果
            feature_importance: 特征重要性
            predictions: 预测结果 (y_true, y_pred, model_name)
            output_dir: 输出目录

        Returns:
            生成的图表路径字典
        """
        import os
        from pathlib import Path

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        chart_paths = {}

        print("📊 生成分析仪表板...")

        # 1. RFM分布分析
        try:
            path = os.path.join(output_dir, '01_rfm_distribution.png')
            self.plot_rfm_distribution(rfm_data, save_path=path, show_plot=False)
            chart_paths['rfm_distribution'] = path
        except Exception as e:
            print(f"⚠️ RFM分布图生成失败: {str(e)}")

        # 2. 客户分群分析
        try:
            path = os.path.join(output_dir, '02_customer_segments.png')
            self.plot_customer_segments(rfm_data, save_path=path, show_plot=False)
            chart_paths['customer_segments'] = path
        except Exception as e:
            print(f"⚠️ 客户分群图生成失败: {str(e)}")

        # 3. 相关性矩阵
        try:
            path = os.path.join(output_dir, '03_correlation_matrix.png')
            self.plot_correlation_matrix(rfm_data, save_path=path, show_plot=False)
            chart_paths['correlation_matrix'] = path
        except Exception as e:
            print(f"⚠️ 相关性矩阵图生成失败: {str(e)}")

        # 4. 模型性能对比
        if model_results:
            try:
                path = os.path.join(output_dir, '04_model_performance.png')
                self.plot_model_performance(model_results, save_path=path, show_plot=False)
                chart_paths['model_performance'] = path
            except Exception as e:
                print(f"⚠️ 模型性能图生成失败: {str(e)}")

        # 5. 特征重要性分析
        if feature_importance:
            try:
                path = os.path.join(output_dir, '05_feature_importance.png')
                self.plot_feature_importance(feature_importance, save_path=path, show_plot=False)
                chart_paths['feature_importance'] = path
            except Exception as e:
                print(f"⚠️ 特征重要性图生成失败: {str(e)}")

        # 6. 预测结果分析
        if predictions:
            try:
                y_true, y_pred, model_name = predictions
                path = os.path.join(output_dir, '06_prediction_analysis.png')
                self.plot_prediction_analysis(y_true, y_pred, model_name,
                                            save_path=path, show_plot=False)
                chart_paths['prediction_analysis'] = path
            except Exception as e:
                print(f"⚠️ 预测分析图生成失败: {str(e)}")

        print(f"✓ 分析仪表板生成完成，共生成 {len(chart_paths)} 个图表")
        print(f"  - 输出目录: {output_dir}")

        return chart_paths

    def save_charts_summary(self, output_path: str):
        """
        保存图表生成摘要

        Args:
            output_path: 输出路径
        """
        summary = {
            'total_charts': self.chart_count,
            'generated_charts': list(self.charts.keys()),
            'config': self.config,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ 图表摘要已保存: {output_path}")

# 便利函数
def quick_visualization(rfm_data: pd.DataFrame,
                       model_results: Optional[Dict[str, Dict]] = None,
                       output_dir: str = './charts',
                       show_plots: bool = False) -> Dict[str, str]:
    """
    快速生成所有可视化图表

    Args:
        rfm_data: RFM特征数据
        model_results: 模型训练结果
        output_dir: 输出目录
        show_plots: 是否显示图表

    Returns:
        生成的图表路径字典
    """
    visualizer = DataVisualizer()

    return visualizer.generate_dashboard(
        rfm_data=rfm_data,
        model_results=model_results,
        feature_importance=None,  # 可以从model_results中提取
        predictions=None,         # 可以从model_results中提取
        output_dir=output_dir
    )

if __name__ == "__main__":
    # 示例使用
    print("📊 可视化模块测试")

    # 创建示例数据
    np.random.seed(42)
    n_customers = 100

    sample_rfm = pd.DataFrame({
        'R值': np.random.randint(1, 90, n_customers),
        'F值': np.random.randint(1, 100, n_customers),
        'M值': np.random.uniform(100, 10000, n_customers),
        '年度LTV': np.random.uniform(500, 20000, n_customers),
        '客户价值分层': np.random.choice(['铜牌客户', '银牌客户', '金牌客户', '白金客户'], n_customers)
    })

    # 测试可视化
    visualizer = DataVisualizer()
    visualizer.plot_rfm_distribution(sample_rfm, show_plot=False)
    visualizer.plot_customer_segments(sample_rfm, show_plot=False)

    print("✓ 可视化模块测试完成")