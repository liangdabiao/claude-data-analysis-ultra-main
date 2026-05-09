"""
AB测试可视化模块

提供全面的AB测试可视化功能，包括：
- 转化率对比图
- 留存率曲线图
- 用户分群热力图
- 交互效应可视化
- 统计检验结果图
- 交互式仪表板
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ABTestVisualizer:
    """AB测试可视化器"""

    def __init__(self, style: str = 'seaborn', palette: str = 'viridis'):
        self.style = style
        self.palette = palette
        self.fig_size = (12, 8)
        self.dpi = 300

        # 设置样式
        if style == 'seaborn':
            sns.set_style("whitegrid")
        elif style == 'matplotlib':
            plt.style.use('default')

    def set_style(self, style: str, palette: str = None):
        """设置可视化样式"""
        self.style = style
        if palette:
            self.palette = palette

        if style == 'seaborn':
            sns.set_style("whitegrid")
            if palette:
                sns.set_palette(palette)

    def plot_conversion_comparison(self,
                                 conversion_results: Dict,
                                 title: str = 'AB测试转化率对比',
                                 show_confidence_interval: bool = True,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制转化率对比图"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        groups = list(conversion_results.keys())
        rates = [conversion_results[group]['conversion_rate'] for group in groups]

        # 绘制柱状图
        bars = ax.bar(groups, rates, alpha=0.7, color=sns.color_palette(self.palette)[:len(groups)])

        # 添加数值标签
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 添加置信区间
        if show_confidence_interval:
            errors = []
            for group in groups:
                ci_lower, ci_upper = conversion_results[group]['confidence_interval']
                error = ci_upper - conversion_results[group]['conversion_rate']
                errors.append(error)

            ax.errorbar(groups, rates, yerr=errors, fmt='none', color='red', capsize=5, capthick=2)

        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('转化率', fontsize=14)
        ax.set_xlabel('实验组别', fontsize=14)
        ax.set_ylim(0, max(rates) * 1.2)

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_retention_curves(self,
                             retention_results: Dict,
                             title: str = '用户留存率对比',
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制留存率曲线图"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        groups = list(retention_results.keys())
        rates = [retention_results[group]['retention_rate'] for group in groups]

        # 创建阶梯图模拟留存曲线
        x_points = np.arange(0, 8)  # 0-7天
        colors = sns.color_palette(self.palette)[:len(groups)]

        for i, group in enumerate(groups):
            base_rate = retention_results[group]['retention_rate']
            # 模拟留存率衰减
            retention_curve = [base_rate * (0.9 ** day) for day in x_points]
            ax.plot(x_points, retention_curve, marker='o', linewidth=2,
                   label=group, color=colors[i])

        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('天数', fontsize=14)
        ax.set_ylabel('留存率', fontsize=14)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 设置x轴标签
        ax.set_xticks(x_points)
        ax.set_xticklabels([f'Day {i}' for i in x_points])

        # 添加百分比标签
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_segment_heatmap(self,
                             segment_data: pd.DataFrame,
                             segment_col: str,
                             metric_col: str,
                             title: str = '用户分群热力图',
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制用户分群热力图"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        # 创建透视表
        pivot_table = segment_data.pivot_table(
            index=segment_col,
            columns=metric_col if segment_data[metric_col].nunique() < 20 else None,
            values=metric_col,
            aggfunc='mean',
            fill_value=0
        )

        # 绘制热力图
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd',
                   square=True, ax=ax, cbar_kws={'label': metric_col})

        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('')  # 移除x轴标签
        ax.set_ylabel('')  # 移除y轴标签

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_interaction_effects(self,
                                interaction_results: Dict,
                                title: str = '交互效应分析',
                                save_path: Optional[str] = None) -> plt.Figure:
        """绘制交互效应图"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        segments = [k for k in interaction_results.keys() if k != 'overall_interaction']

        if not segments:
            ax.text(0.5, 0.5, '无交互效应数据', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            return fig

        # 准备数据
        group1_conversions = []
        group2_conversions = []
        group1_names = []
        group2_names = []

        for segment in segments:
            if 'group1_conversion' in interaction_results[segment]:
                group1_conversions.append(interaction_results[segment]['group1_conversion'])
                group2_conversions.append(interaction_results[segment]['group2_conversion'])
                group1_names.append(interaction_results[segment]['group1'])
                group2_names.append(interaction_results[segment]['group2'])

        if not group1_conversions:
            ax.text(0.5, 0.5, '无有效的交互效应数据', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            return fig

        # 设置x轴位置
        x_pos = np.arange(len(segments))
        width = 0.35

        # 绘制柱状图
        colors = sns.color_palette(self.palette, 2)
        bars1 = ax.bar(x_pos - width/2, group1_conversions, width,
                      label=group1_names[0] if group1_names else 'Group 1', color=colors[0])
        bars2 = ax.bar(x_pos + width/2, group2_conversions, width,
                      label=group2_names[0] if group2_names else 'Group 2', color=colors[1])

        # 添加数值标签和显著性标记
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()

            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                   f'{height1:.1%}', ha='center', va='bottom', fontsize=10)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                   f'{height2:.1%}', ha='center', va='bottom', fontsize=10)

            # 添加显著性标记
            if 'is_significant_t' in interaction_results[segments[i]]:
                if interaction_results[segments[i]]['is_significant_t']:
                    ax.text(i, max(height1, height2) + 0.03, '*',
                           ha='center', va='bottom', fontsize=16, fontweight='bold')

        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('用户分群', fontsize=14)
        ax.set_ylabel('转化率', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(segments, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # 设置y轴为百分比格式
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_confidence_intervals(self,
                                 statistical_results: Dict,
                                 metric: str = 'conversion_rate',
                                 title: str = '置信区间分析',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制置信区间图"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        groups = list(statistical_results.keys())

        # 准备数据
        means = []
        cis_lower = []
        cis_upper = []

        for group in groups:
            if 'confidence_interval' in statistical_results[group]:
                means.append(statistical_results[group]['conversion_rate'])
                ci_lower, ci_upper = statistical_results[group]['confidence_interval']
                cis_lower.append(ci_lower)
                cis_upper.append(ci_upper)

        if not means:
            ax.text(0.5, 0.5, '无置信区间数据', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            return fig

        # 绘制误差线图
        x_pos = np.arange(len(groups))
        errors_lower = [means[i] - cis_lower[i] for i in range(len(means))]
        errors_upper = [cis_upper[i] - means[i] for i in range(len(means))]
        errors = [errors_lower, errors_upper]

        colors = sns.color_palette(self.palette, len(groups))
        bars = ax.bar(x_pos, means, yerr=errors, capsize=5, capthick=2,
                     color=colors, alpha=0.7)

        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2., mean + errors[1][i] + 0.01,
                   f'{mean:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('实验组别', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups)
        ax.grid(True, alpha=0.3, axis='y')

        # 设置y轴为百分比格式
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_bayesian_comparison(self,
                                bayesian_results: Dict,
                                title: str = '贝叶斯AB测试分析',
                                save_path: Optional[str] = None) -> plt.Figure:
        """绘制贝叶斯AB测试结果图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)

        groups = [k for k in bayesian_results.keys() if k != 'comparison']
        colors = sns.color_palette(self.palette, len(groups))

        # 1. 后验分布图
        for i, group in enumerate(groups):
            samples = bayesian_results[group]['posterior_samples']
            ax1.hist(samples, bins=50, alpha=0.5, label=group, color=colors[i], density=True)

        ax1.set_title('后验分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('转化率', fontsize=12)
        ax1.set_ylabel('密度', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. 获胜概率
        if 'comparison' in bayesian_results:
            prob_data = []
            prob_labels = []

            for key, value in bayesian_results['comparison'].items():
                if key.startswith('prob_') and key.endswith('_better'):
                    group_name = key.replace('prob_', '').replace('_better', '')
                    prob_data.append(value)
                    prob_labels.append(group_name)

            if prob_data:
                bars = ax2.bar(prob_labels, prob_data, color=colors[:len(prob_data)], alpha=0.7)

                # 添加数值标签
                for bar, prob in zip(bars, prob_data):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax2.set_title('获胜概率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('概率', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def create_interactive_dashboard(self,
                                   analysis_results: Dict,
                                   output_file: str = 'ab_test_dashboard.html') -> str:
        """创建交互式仪表板"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('转化率对比', '留存率对比', '统计检验结果', '分群分析'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "table"}, {"type": "bar"}]]
        )

        # 1. 转化率对比
        if 'conversion_analysis' in analysis_results:
            conversion_data = analysis_results['conversion_analysis']
            groups = list(conversion_data.keys())
            rates = [conversion_data[group]['conversion_rate'] for group in groups]

            fig.add_trace(
                go.Bar(x=groups, y=rates, name='转化率'),
                row=1, col=1
            )

        # 2. 留存率对比
        if 'retention_analysis' in analysis_results:
            retention_data = analysis_results['retention_analysis']
            groups = list(retention_data.keys())
            retention_rates = [retention_data[group]['retention_rate'] for group in groups]

            fig.add_trace(
                go.Scatter(x=groups, y=retention_rates, mode='lines+markers', name='留存率'),
                row=1, col=2
            )

        # 3. 统计检验结果表
        if 'statistical_tests' in analysis_results:
            test_data = analysis_results['statistical_tests']
            table_data = []

            for test_name, results in test_data.items():
                table_data.append([
                    test_name,
                    f"{results.get('statistic', 0):.4f}",
                    f"{results.get('p_value', 1):.4f}",
                    "显著" if results.get('is_significant', False) else "不显著"
                ])

            fig.add_trace(
                go.Table(
                    header=dict(values=['检验方法', '统计量', 'P值', '显著性']),
                    cells=dict(values=list(zip(*table_data)))
                ),
                row=2, col=1
            )

        # 4. 分群分析
        if 'segment_analysis' in analysis_results:
            segment_data = analysis_results['segment_analysis']
            segments = list(segment_data.keys())

            # 为每个分群计算平均转化率
            segment_rates = []
            for segment in segments:
                if 'conversion_rate' in segment_data[segment]:
                    segment_rates.append(segment_data[segment]['conversion_rate'])
                else:
                    segment_rates.append(0)

            fig.add_trace(
                go.Bar(x=segments, y=segment_rates, name='分群转化率'),
                row=2, col=2
            )

        # 更新布局
        fig.update_layout(
            title_text="AB测试分析仪表板",
            showlegend=False,
            height=800
        )

        # 保存为HTML文件
        fig.write_html(output_file)

        return output_file

    def plot_with_config(self,
                        data: Union[pd.DataFrame, Dict],
                        config: Dict,
                        plot_type: str = 'bar') -> plt.Figure:
        """使用自定义配置绘制图表"""
        # 设置配置
        if 'figsize' in config:
            self.fig_size = config['figsize']
        if 'dpi' in config:
            self.dpi = config['dpi']
        if 'style' in config:
            self.set_style(config['style'])
        if 'palette' in config:
            self.set_style(self.style, config['palette'])

        # 根据图表类型调用相应的绘图方法
        if plot_type == 'bar':
            if isinstance(data, dict):
                return self.plot_conversion_comparison(data, save_path=config.get('filename'))
        elif plot_type == 'heatmap':
            if isinstance(data, pd.DataFrame):
                segment_col = config.get('segment_col', 'segment')
                metric_col = config.get('metric_col', 'metric')
                return self.plot_segment_heatmap(data, segment_col, metric_col, save_path=config.get('filename'))
        elif plot_type == 'interaction':
            if isinstance(data, dict):
                return self.plot_interaction_effects(data, save_path=config.get('filename'))
        elif plot_type == 'bayesian':
            if isinstance(data, dict):
                return self.plot_bayesian_comparison(data, save_path=config.get('filename'))
        else:
            raise ValueError(f"不支持的图表类型: {plot_type}")

    def save_all_plots(self,
                      analysis_results: Dict,
                      output_dir: str = './plots/') -> List[str]:
        """批量保存所有图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        # 保存转化率对比图
        if 'conversion_analysis' in analysis_results:
            fig = self.plot_conversion_comparison(
                analysis_results['conversion_analysis'],
                save_path=os.path.join(output_dir, 'conversion_comparison.png')
            )
            plt.close(fig)
            saved_files.append('conversion_comparison.png')

        # 保存留存率图
        if 'retention_analysis' in analysis_results:
            fig = self.plot_retention_curves(
                analysis_results['retention_analysis'],
                save_path=os.path.join(output_dir, 'retention_curves.png')
            )
            plt.close(fig)
            saved_files.append('retention_curves.png')

        # 保存交互效应图
        if 'interaction_analysis' in analysis_results:
            fig = self.plot_interaction_effects(
                analysis_results['interaction_analysis'],
                save_path=os.path.join(output_dir, 'interaction_effects.png')
            )
            plt.close(fig)
            saved_files.append('interaction_effects.png')

        # 保存贝叶斯分析图
        if 'bayesian_analysis' in analysis_results:
            fig = self.plot_bayesian_comparison(
                analysis_results['bayesian_analysis'],
                save_path=os.path.join(output_dir, 'bayesian_comparison.png')
            )
            plt.close(fig)
            saved_files.append('bayesian_comparison.png')

        # 生成交互式仪表板
        dashboard_file = self.create_interactive_dashboard(
            analysis_results,
            output_file=os.path.join(output_dir, 'ab_test_dashboard.html')
        )
        saved_files.append('ab_test_dashboard.html')

        return saved_files