"""
推荐系统可视化展示器

提供全面的推荐系统可视化功能：
- 推荐结果可视化
- 评估指标图表
- 用户-商品交互热力图
- 算法性能对比图
- 数据洞察分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class RecommenderVisualizer:
    """推荐系统可视化展示器类"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器

        Args:
            figsize: 默认图形大小
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        self.figures = {}  # 存储生成的图形

    def plot_recommendation_results(self, recommendations: List[Tuple[str, float]],
                                  user_id: str, title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        可视化推荐结果

        Args:
            recommendations: 推荐结果列表，格式为 [(商品ID, 预测评分), ...]
            user_id: 用户ID
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        if not recommendations:
            print("没有推荐结果可以可视化")
            return None

        # 提取数据
        items = [item[0] for item in recommendations]
        scores = [item[1] for item in recommendations]

        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)

        # 创建水平柱状图
        bars = ax.barh(range(len(items)), scores, color=self.color_palette[0], alpha=0.7)

        # 设置标签
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(items)
        ax.set_xlabel('预测评分')
        ax.set_title(title or f'用户 {user_id} 的推荐结果')

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center')

        # 调整布局
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures[f'recommendations_{user_id}'] = fig
        return fig

    def plot_user_item_heatmap(self, user_item_matrix: pd.DataFrame,
                              sample_size: Tuple[int, int] = (50, 50),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制用户-商品交互热力图

        Args:
            user_item_matrix: 用户-商品交互矩阵
            sample_size: 采样大小 (用户数, 商品数)
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        # 采样数据
        n_users, n_items = sample_size
        users_sample = user_item_matrix.index[:n_users]
        items_sample = user_item_matrix.columns[:n_items]

        sample_matrix = user_item_matrix.loc[users_sample, items_sample]

        # 创建图形
        fig, ax = plt.subplots(figsize=(max(8, n_items//2), max(6, n_users//2)))

        # 绘制热力图
        sns.heatmap(sample_matrix, annot=False, cmap='YlOrRd',
                   cbar_kws={'label': '评分'}, ax=ax)

        ax.set_title('用户-商品交互热力图')
        ax.set_xlabel('商品ID')
        ax.set_ylabel('用户ID')

        # 调整布局
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['user_item_heatmap'] = fig
        return fig

    def plot_evaluation_metrics(self, evaluation_results: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制评估指标图表

        Args:
            evaluation_results: 评估结果字典
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        # 提取指标数据
        metrics_data = {}

        for key, value in evaluation_results.items():
            if any(metric in key for metric in ['precision', 'recall', 'f1']) and not key.endswith('_std'):
                if '@' in key:
                    metric_name, k_value = key.split('@')
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = {'k_values': [], 'values': []}
                    metrics_data[metric_name]['k_values'].append(int(k_value))
                    metrics_data[metric_name]['values'].append(value)

        if not metrics_data:
            print("没有可绘制的评估指标")
            return None

        # 创建子图
        fig, axes = plt.subplots(1, len(metrics_data), figsize=(5*len(metrics_data), 5))
        if len(metrics_data) == 1:
            axes = [axes]

        # 绘制每个指标
        for i, (metric_name, data) in enumerate(metrics_data.items()):
            ax = axes[i]

            # 绘制折线图
            ax.plot(data['k_values'], data['values'],
                   marker='o', linewidth=2, markersize=8,
                   color=self.color_palette[i], label=metric_name.capitalize())

            ax.set_xlabel('K值')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()}@K')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # 添加数值标签
            for j, (k, v) in enumerate(zip(data['k_values'], data['values'])):
                ax.annotate(f'{v:.3f}', (k, v), textcoords="offset points",
                           xytext=(0,10), ha='center')

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['evaluation_metrics'] = fig
        return fig

    def plot_algorithm_comparison(self, comparison_df: pd.DataFrame,
                                 metrics: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制算法比较图

        Args:
            comparison_df: 算法比较结果DataFrame
            metrics: 要比较的指标列表
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        if metrics is None:
            # 自动选择指标
            metrics = [col for col in comparison_df.columns
                      if any(x in col for x in ['precision', 'recall', 'f1'])
                      and not col.endswith('_rank') and not col.endswith('_std')]

        if not metrics:
            print("没有可比较的指标")
            return None

        # 创建子图
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # 提取数据
            algorithms = comparison_df['Algorithm'].tolist()
            values = comparison_df[metric].tolist()

            # 绘制柱状图
            bars = ax.bar(algorithms, values, color=self.color_palette[i], alpha=0.7)

            ax.set_title(metric.replace('@', '@').upper())
            ax.set_ylabel('分数')
            ax.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')

            # 添加平均线
            avg_value = np.mean(values)
            ax.axhline(y=avg_value, color='red', linestyle='--',
                      alpha=0.7, label=f'平均值: {avg_value:.3f}')
            ax.legend()

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['algorithm_comparison'] = fig
        return fig

    def plot_user_behavior_analysis(self, user_data: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制用户行为分析图

        Args:
            user_data: 用户行为数据
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 用户活跃度分布
        user_activity = user_data.groupby('用户ID').size().sort_values(ascending=False)
        axes[0, 0].bar(range(len(user_activity)), user_activity.values,
                      color=self.color_palette[0], alpha=0.7)
        axes[0, 0].set_title('用户活跃度分布')
        axes[0, 0].set_xlabel('用户排名')
        axes[0, 0].set_ylabel('行为次数')

        # 2. 评分分布
        if '评分' in user_data.columns:
            rating_dist = user_data['评分'].value_counts().sort_index()
            axes[0, 1].bar(rating_dist.index, rating_dist.values,
                          color=self.color_palette[1], alpha=0.7)
            axes[0, 1].set_title('评分分布')
            axes[0, 1].set_xlabel('评分')
            axes[0, 1].set_ylabel('频次')

        # 3. 时间分布（如果有时间戳）
        if '时间戳' in user_data.columns:
            user_data['时间戳'] = pd.to_datetime(user_data['时间戳'])
            daily_activity = user_data.groupby(user_data['时间戳'].dt.date).size()
            axes[1, 0].plot(daily_activity.index, daily_activity.values,
                           color=self.color_palette[2], linewidth=2)
            axes[1, 0].set_title('每日用户活动')
            axes[1, 0].set_xlabel('日期')
            axes[1, 0].set_ylabel('活动次数')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 商品类别分布
        if '商品类别' in user_data.columns:
            category_dist = user_data['商品类别'].value_counts()
            axes[1, 1].pie(category_dist.values, labels=category_dist.index,
                           autopct='%1.1f%%', colors=self.color_palette)
            axes[1, 1].set_title('商品类别分布')

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['user_behavior_analysis'] = fig
        return fig

    def plot_item_popularity_analysis(self, item_data: pd.DataFrame,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制商品热度分析图

        Args:
            item_data: 商品数据或用户行为数据
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 如果是用户行为数据，计算商品热度
        if '用户ID' in item_data.columns and '商品ID' in item_data.columns:
            item_popularity = item_data.groupby('商品ID').size().sort_values(ascending=False)

            # 1. 商品热度分布
            axes[0, 0].bar(range(len(item_popularity)), item_popularity.values,
                          color=self.color_palette[0], alpha=0.7)
            axes[0, 0].set_title('商品热度分布')
            axes[0, 0].set_xlabel('商品排名')
            axes[0, 0].set_ylabel('交互次数')

            # 2. 热度TOP10商品
            top_items = item_popularity.head(10)
            axes[0, 1].barh(range(len(top_items)), top_items.values,
                           color=self.color_palette[1], alpha=0.7)
            axes[0, 1].set_yticks(range(len(top_items)))
            axes[0, 1].set_yticklabels(top_items.index)
            axes[0, 1].set_title('TOP10 热门商品')
            axes[0, 1].set_xlabel('交互次数')

        # 如果是商品信息数据
        if '商品类别' in item_data.columns:
            # 3. 商品类别分布
            category_dist = item_data['商品类别'].value_counts()
            axes[1, 0].pie(category_dist.values, labels=category_dist.index,
                           autopct='%1.1f%%', colors=self.color_palette)
            axes[1, 0].set_title('商品类别分布')

        if '价格' in item_data.columns:
            # 4. 价格分布
            axes[1, 1].hist(item_data['价格'], bins=20, color=self.color_palette[2], alpha=0.7)
            axes[1, 1].set_title('商品价格分布')
            axes[1, 1].set_xlabel('价格')
            axes[1, 1].set_ylabel('商品数量')

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['item_popularity_analysis'] = fig
        return fig

    def plot_similarity_matrix(self, similarity_matrix: pd.DataFrame,
                             sample_size: int = 50,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制相似度矩阵热力图

        Args:
            similarity_matrix: 相似度矩阵
            sample_size: 采样大小
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        # 采样数据
        sample_matrix = similarity_matrix.iloc[:sample_size, :sample_size]

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制热力图
        sns.heatmap(sample_matrix, annot=False, cmap='coolwarm',
                   center=0, cbar_kws={'label': '相似度'}, ax=ax)

        ax.set_title('相似度矩阵热力图')
        ax.set_xlabel('项目ID')
        ax.set_ylabel('项目ID')

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['similarity_matrix'] = fig
        return fig

    def plot_learning_curve(self, training_data: Dict[str, List[float]],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制学习曲线

        Args:
            training_data: 训练数据，格式为 {'metric': [values]}
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 绘制每个指标的学习曲线
        for i, (metric_name, values) in enumerate(training_data.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=6,
                   color=self.color_palette[i], label=metric_name)

        ax.set_xlabel('训练轮次')
        ax.set_ylabel('指标值')
        ax.set_title('学习曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures['learning_curve'] = fig
        return fig

    def create_interactive_recommendations(self, recommendations: List[Tuple[str, float]],
                                         user_id: str) -> go.Figure:
        """
        创建交互式推荐结果图表

        Args:
            recommendations: 推荐结果列表
            user_id: 用户ID

        Returns:
            plotly Figure对象
        """
        items = [item[0] for item in recommendations]
        scores = [item[1] for item in recommendations]

        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=items,
                orientation='h',
                marker=dict(
                    color=scores,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'{score:.3f}' for score in scores],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title=f'用户 {user_id} 的推荐结果',
            xaxis_title='预测评分',
            yaxis_title='商品',
            height=max(400, len(recommendations) * 30),
            font=dict(size=12)
        )

        return fig

    def create_interactive_comparison(self, comparison_df: pd.DataFrame,
                                    metrics: Optional[List[str]] = None) -> go.Figure:
        """
        创建交互式算法比较图表

        Args:
            comparison_df: 算法比较结果DataFrame
            metrics: 要比较的指标列表

        Returns:
            plotly Figure对象
        """
        if metrics is None:
            metrics = [col for col in comparison_df.columns
                      if any(x in col for x in ['precision', 'recall', 'f1'])
                      and not col.endswith('_rank') and not col.endswith('_std')]

        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[metric.replace('@', '@').upper() for metric in metrics]
        )

        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Algorithm'],
                    y=comparison_df[metric],
                    name=metric,
                    marker_color=self.color_palette[i]
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            title='算法性能比较',
            height=500,
            showlegend=False
        )

        return fig

    def save_all_figures(self, output_dir: str = './figures/') -> None:
        """
        保存所有生成的图形

        Args:
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for name, fig in self.figures.items():
            file_path = os.path.join(output_dir, f'{name}.png')
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存: {file_path}")

    def show_all_figures(self) -> None:
        """显示所有生成的图形"""
        for name, fig in self.figures.items():
            print(f"\n显示图形: {name}")
            plt.show()

    def clear_figures(self) -> None:
        """清除所有缓存的图形"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()

    def get_figure_summary(self) -> Dict[str, str]:
        """
        获取图形摘要信息

        Returns:
            图形摘要字典
        """
        summary = {}
        for name, fig in self.figures.items():
            summary[name] = f"图形类型: {type(fig).__name__}, 大小: {fig.get_size_inches()}"
        return summary