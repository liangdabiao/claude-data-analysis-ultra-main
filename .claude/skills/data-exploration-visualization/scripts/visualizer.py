"""
数据可视化生成器 (Data Visualizer) - 专业可视化图表自动生成模块

提供全面的可视化功能，包括：
- 分布可视化（直方图、密度图、小提琴图）
- 统计可视化（箱线图、误差条图、QQ图）
- 关系可视化（散点图、热图、配对图）
- 专门可视化（ROC曲线、混淆矩阵、特征重要性）
- 交互式可视化（Plotly图表）
- 中文支持和自定义样式
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')


class DataVisualizer:
    """数据可视化生成器 - 专业图表自动生成引擎"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化可视化器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.charts = {}
        self.figure_style = None

        # 默认配置
        self.default_config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'style': 'seaborn-v0_8',
            'color_palette': 'husl',
            'chinese_font': 'SimHei',
            'interactive_charts': True,
            'save_format': 'png',
            'output_dir': 'charts',
            'max_categories': 20,  # 分类变量最大显示类别数
            'sample_size': 10000,  # 大数据集采样大小
            'grid_alpha': 0.3,
            'title_fontsize': 14,
            'label_fontsize': 12
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

        # 设置中文字体
        self._setup_chinese_font()

        # 设置matplotlib样式
        self._setup_matplotlib_style()

    def _setup_chinese_font(self):
        """设置中文字体支持"""
        try:
            # 尝试设置中文字体
            plt.rcParams['font.sans-serif'] = [self.config['chinese_font'], 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("⚠️ 无法设置中文字体，图表可能无法正常显示中文")

    def _setup_matplotlib_style(self):
        """设置matplotlib样式"""
        try:
            plt.style.use(self.config['style'])
        except:
            # 如果指定的样式不存在，使用默认样式
            plt.style.use('default')

        # 设置图表参数
        plt.rcParams['figure.figsize'] = self.config['figure_size']
        plt.rcParams['savefig.dpi'] = self.config['dpi']
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = self.config['grid_alpha']

    def plot_distribution(self, data: pd.DataFrame, column: str,
                         plot_type: str = 'auto', interactive: bool = None) -> Union[plt.Figure, go.Figure]:
        """
        绘制分布图

        Parameters:
        - data: 数据DataFrame
        - column: 列名
        - plot_type: 图表类型 ('auto', 'histogram', 'density', 'violin', 'box')
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        if column not in data.columns:
            raise ValueError(f"列 '{column}' 不存在于数据中")

        if interactive is None:
            interactive = self.config['interactive_charts']

        # 数据预处理
        plot_data = data[column].dropna()

        if len(plot_data) == 0:
            raise ValueError(f"列 '{column}' 没有有效数据")

        # 采样大数据集
        if len(plot_data) > self.config['sample_size']:
            plot_data = plot_data.sample(self.config['sample_size'], random_state=42)

        # 自动选择图表类型
        if plot_type == 'auto':
            if pd.api.types.is_numeric_dtype(plot_data):
                # 数值型数据：根据分布特征选择
                if len(plot_data.unique()) < 20:
                    plot_type = 'histogram'
                else:
                    plot_type = 'density'
            else:
                # 分类型数据
                plot_type = 'bar'

        if interactive:
            return self._plot_distribution_interactive(plot_data, column, plot_type)
        else:
            return self._plot_distribution_static(plot_data, column, plot_type)

    def _plot_distribution_interactive(self, data: pd.Series, column: str, plot_type: str) -> go.Figure:
        """绘制交互式分布图"""
        fig = go.Figure()

        if pd.api.types.is_numeric_dtype(data):
            if plot_type == 'histogram':
                fig.add_trace(go.Histogram(
                    x=data,
                    name=column,
                    nbinsx=min(50, len(data.unique())),
                    opacity=0.7
                ))

            elif plot_type == 'density':
                # 计算核密度估计
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                kde_values = kde(x_range)

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name=f'{column} 密度曲线',
                    fill='tozeroy',
                    opacity=0.7
                ))

            elif plot_type == 'violin':
                fig.add_trace(go.Violin(
                    y=data,
                    name=column,
                    box_visible=True,
                    meanline_visible=True
                ))

            elif plot_type == 'box':
                fig.add_trace(go.Box(
                    y=data,
                    name=column,
                    boxpoints='outliers'
                ))

        else:
            # 分类型数据
            value_counts = data.value_counts().head(self.config['max_categories'])
            fig.add_trace(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=column
            ))

        fig.update_layout(
            title=f'{column} 分布图',
            xaxis_title=column,
            yaxis_title='频数' if pd.api.types.is_numeric_dtype(data) else '计数',
            showlegend=False,
            height=500
        )

        return fig

    def _plot_distribution_static(self, data: pd.Series, column: str, plot_type: str) -> plt.Figure:
        """绘制静态分布图"""
        fig, ax = plt.subplots(figsize=self.config['figure_size'])

        if pd.api.types.is_numeric_dtype(data):
            if plot_type == 'histogram':
                ax.hist(data, bins=min(50, len(data.unique())), alpha=0.7, edgecolor='black')
                ax.set_ylabel('频数')

            elif plot_type == 'density':
                sns.kdeplot(data=data, ax=ax, fill=True, alpha=0.7)
                ax.set_ylabel('密度')

            elif plot_type == 'violin':
                sns.violinplot(y=data, ax=ax)

            elif plot_type == 'box':
                sns.boxplot(y=data, ax=ax)

        else:
            # 分类型数据
            value_counts = data.value_counts().head(self.config['max_categories'])
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45)
            ax.set_ylabel('计数')

        ax.set_title(f'{column} 分布图', fontsize=self.config['title_fontsize'])
        ax.set_xlabel(column, fontsize=self.config['label_fontsize'])
        ax.grid(True, alpha=self.config['grid_alpha'])

        plt.tight_layout()
        return fig

    def plot_correlation(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'pearson', interactive: bool = None) -> Union[plt.Figure, go.Figure]:
        """
        绘制相关性热图

        Parameters:
        - data: 数据DataFrame
        - columns: 要分析的列名列表
        - method: 相关性计算方法
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        if columns is None:
            # 选择数值列
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) < 2:
            raise ValueError("至少需要2个数值列来计算相关性")

        if interactive is None:
            interactive = self.config['interactive_charts']

        # 计算相关性矩阵
        corr_matrix = data[columns].corr(method=method)

        if interactive:
            return self._plot_correlation_interactive(corr_matrix, method)
        else:
            return self._plot_correlation_static(corr_matrix, method)

    def _plot_correlation_interactive(self, corr_matrix: pd.DataFrame, method: str) -> go.Figure:
        """绘制交互式相关性热图"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=f'{method.capitalize()} 相关性热图',
            width=800,
            height=700
        )

        return fig

    def _plot_correlation_static(self, corr_matrix: pd.DataFrame, method: str) -> plt.Figure:
        """绘制静态相关性热图"""
        fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix.columns)),
                                        max(8, len(corr_matrix.columns))))

        # 创建mask用于只显示下三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)

        ax.set_title(f'{method.capitalize()} 相关性热图',
                    fontsize=self.config['title_fontsize'], pad=20)

        plt.tight_layout()
        return fig

    def plot_scatter(self, data: pd.DataFrame, x: str, y: str,
                    color_col: Optional[str] = None, size_col: Optional[str] = None,
                    interactive: bool = None) -> Union[plt.Figure, go.Figure]:
        """
        绘制散点图

        Parameters:
        - data: 数据DataFrame
        - x: X轴列名
        - y: Y轴列名
        - color_col: 颜色分组列名
        - size_col: 大小列名
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        if x not in data.columns or y not in data.columns:
            raise ValueError(f"列 '{x}' 或 '{y}' 不存在于数据中")

        if interactive is None:
            interactive = self.config['interactive_charts']

        # 数据预处理
        plot_data = data[[x, y]].dropna()
        if color_col:
            plot_data[color_col] = data[color_col]
        if size_col:
            plot_data[size_col] = data[size_col]

        # 采样大数据集
        if len(plot_data) > self.config['sample_size']:
            plot_data = plot_data.sample(self.config['sample_size'], random_state=42)

        if interactive:
            return self._plot_scatter_interactive(plot_data, x, y, color_col, size_col)
        else:
            return self._plot_scatter_static(plot_data, x, y, color_col, size_col)

    def _plot_scatter_interactive(self, data: pd.DataFrame, x: str, y: str,
                                 color_col: Optional[str], size_col: Optional[str]) -> go.Figure:
        """绘制交互式散点图"""
        scatter_args = {
            'x': data[x],
            'y': data[y],
            'mode': 'markers',
            'marker': {'size': 8, 'opacity': 0.7}
        }

        if color_col and color_col in data.columns:
            scatter_args['marker']['color'] = data[color_col]
            scatter_args['marker']['colorscale'] = 'Viridis'
            scatter_args['marker']['showscale'] = True

        if size_col and size_col in data.columns:
            # 标准化大小
            size_normalized = (data[size_col] - data[size_col].min()) / (data[size_col].max() - data[size_col].min())
            scatter_args['marker']['size'] = 5 + size_normalized * 15

        fig = go.Figure(data=[go.Scatter(**scatter_args)])

        fig.update_layout(
            title=f'{x} vs {y} 散点图',
            xaxis_title=x,
            yaxis_title=y,
            height=600
        )

        return fig

    def _plot_scatter_static(self, data: pd.DataFrame, x: str, y: str,
                           color_col: Optional[str], size_col: Optional[str]) -> plt.Figure:
        """绘制静态散点图"""
        fig, ax = plt.subplots(figsize=self.config['figure_size'])

        # 使用seaborn的scatterplot
        scatter_kwargs = {
            'x': x,
            'y': y,
            'data': data,
            'ax': ax,
            'alpha': 0.7
        }

        if color_col and color_col in data.columns:
            scatter_kwargs['hue'] = color_col
        if size_col and size_col in data.columns:
            scatter_kwargs['size'] = size_col

        sns.scatterplot(**scatter_kwargs)

        ax.set_title(f'{x} vs {y} 散点图', fontsize=self.config['title_fontsize'])
        ax.grid(True, alpha=self.config['grid_alpha'])

        plt.tight_layout()
        return fig

    def plot_boxplot(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                     group_col: Optional[str] = None, interactive: bool = None) -> Union[plt.Figure, go.Figure]:
        """
        绘制箱线图

        Parameters:
        - data: 数据DataFrame
        - columns: 要绘制的列名列表
        - group_col: 分组列名
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            raise ValueError("没有找到数值列")

        if interactive is None:
            interactive = self.config['interactive_charts']

        if interactive:
            return self._plot_boxplot_interactive(data, columns, group_col)
        else:
            return self._plot_boxplot_static(data, columns, group_col)

    def _plot_boxplot_interactive(self, data: pd.DataFrame, columns: List[str],
                                group_col: Optional[str]) -> go.Figure:
        """绘制交互式箱线图"""
        fig = go.Figure()

        if group_col and group_col in data.columns:
            # 分组箱线图
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]
                for col in columns:
                    fig.add_trace(go.Box(
                        y=group_data[col],
                        name=f'{group} - {col}',
                        boxpoints='outliers'
                    ))
        else:
            # 单列箱线图
            for col in columns:
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col,
                    boxpoints='outliers'
                ))

        fig.update_layout(
            title='箱线图',
            xaxis_title='变量',
            yaxis_title='值',
            height=600
        )

        return fig

    def _plot_boxplot_static(self, data: pd.DataFrame, columns: List[str],
                           group_col: Optional[str]) -> plt.Figure:
        """绘制静态箱线图"""
        # 重新组织数据用于seaborn
        if group_col and group_col in data.columns:
            # 分组箱线图
            fig, ax = plt.subplots(figsize=(max(12, len(columns) * 2), 8))

            # 使用melt重组数据
            melt_cols = columns + [group_col]
            plot_data = data[melt_cols].melt(id_vars=[group_col], var_name='Variable', value_name='Value')

            sns.boxplot(data=plot_data, x='Variable', y='Value', hue=group_col, ax=ax)
            plt.xticks(rotation=45)
        else:
            # 单列箱线图
            fig, ax = plt.subplots(figsize=(max(10, len(columns)), 6))
            sns.boxplot(data=data[columns], ax=ax)
            plt.xticks(rotation=45)

        ax.set_title('箱线图', fontsize=self.config['title_fontsize'])
        ax.grid(True, alpha=self.config['grid_alpha'])

        plt.tight_layout()
        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                       model_name: str = 'Model', interactive: bool = None) -> Union[plt.Figure, go.Figure]:
        """
        绘制ROC曲线

        Parameters:
        - y_true: 真实标签
        - y_scores: 预测概率
        - model_name: 模型名称
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        from sklearn.metrics import roc_curve, auc

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        if interactive is None:
            interactive = self.config['interactive_charts']

        if interactive:
            return self._plot_roc_interactive(fpr, tpr, roc_auc, model_name)
        else:
            return self._plot_roc_static(fpr, tpr, roc_auc, model_name)

    def _plot_roc_interactive(self, fpr: np.ndarray, tpr: np.ndarray,
                            roc_auc: float, model_name: str) -> go.Figure:
        """绘制交互式ROC曲线"""
        fig = go.Figure()

        # ROC曲线
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))

        # 对角线
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='随机分类器',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title='ROC曲线',
            xaxis_title='假阳性率 (FPR)',
            yaxis_title='真阳性率 (TPR)',
            width=700,
            height=600
        )

        return fig

    def _plot_roc_static(self, fpr: np.ndarray, tpr: np.ndarray,
                       roc_auc: float, model_name: str) -> plt.Figure:
        """绘制静态ROC曲线"""
        fig, ax = plt.subplots(figsize=self.config['figure_size'])

        # ROC曲线
        ax.plot(fpr, tpr, color='blue', lw=2,
               label=f'{model_name} (AUC = {roc_auc:.3f})')

        # 对角线
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机分类器')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)', fontsize=self.config['label_fontsize'])
        ax.set_ylabel('真阳性率 (TPR)', fontsize=self.config['label_fontsize'])
        ax.set_title('ROC曲线', fontsize=self.config['title_fontsize'])
        ax.legend(loc="lower right")
        ax.grid(True, alpha=self.config['grid_alpha'])

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             normalize: bool = False, interactive: bool = None) -> Union[plt.Figure, go.Figure]:
        """
        绘制混淆矩阵

        Parameters:
        - y_true: 真实标签
        - y_pred: 预测标签
        - labels: 标签名称
        - normalize: 是否标准化
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        from sklearn.metrics import confusion_matrix

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if interactive is None:
            interactive = self.config['interactive_charts']

        if interactive:
            return self._plot_confusion_matrix_interactive(cm, labels, normalize)
        else:
            return self._plot_confusion_matrix_static(cm, labels, normalize)

    def _plot_confusion_matrix_interactive(self, cm: np.ndarray, labels: Optional[List[str]],
                                        normalize: bool) -> go.Figure:
        """绘制交互式混淆矩阵"""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels if labels else range(cm.shape[1]),
            y=labels if labels else range(cm.shape[0]),
            colorscale='Blues',
            text=np.around(cm, decimals=2) if normalize else cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))

        title = '标准化混淆矩阵' if normalize else '混淆矩阵'
        fig.update_layout(
            title=title,
            xaxis_title='预测标签',
            yaxis_title='真实标签',
            width=600,
            height=500
        )

        return fig

    def _plot_confusion_matrix_static(self, cm: np.ndarray, labels: Optional[List[str]],
                                    normalize: bool) -> plt.Figure:
        """绘制静态混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # 使用seaborn绘制热图
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)

        title = '标准化混淆矩阵' if normalize else '混淆矩阵'
        ax.set_title(title, fontsize=self.config['title_fontsize'])
        ax.set_xlabel('预测标签', fontsize=self.config['label_fontsize'])
        ax.set_ylabel('真实标签', fontsize=self.config['label_fontsize'])

        plt.tight_layout()
        return fig

    def auto_visualize(self, data: pd.DataFrame, target_col: Optional[str] = None,
                      save_charts: bool = True, output_dir: Optional[str] = None) -> Dict[str, Union[plt.Figure, go.Figure]]:
        """
        自动化可视化分析

        Parameters:
        - data: 数据DataFrame
        - target_col: 目标列名
        - save_charts: 是否保存图表
        - output_dir: 输出目录

        Returns:
        - 生成的图表字典
        """
        if output_dir is None:
            output_dir = self.config['output_dir']

        if save_charts:
            Path(output_dir).mkdir(exist_ok=True)

        charts = {}
        print("🚀 开始自动化可视化分析...")

        # 1. 分布可视化
        print("   1. 生成分布图...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # 数值变量分布
        for col in numeric_cols[:6]:  # 限制数量避免过多图表
            try:
                fig = self.plot_distribution(data, col)
                charts[f'{col}_distribution'] = fig
                if save_charts:
                    self._save_chart(fig, f'{col}_distribution', output_dir)
                print(f"      ✓ {col} 分布图")
            except Exception as e:
                print(f"      ❌ {col} 分布图生成失败: {str(e)}")

        # 分类变量分布
        for col in categorical_cols[:6]:  # 限制数量
            try:
                fig = self.plot_distribution(data, col)
                charts[f'{col}_distribution'] = fig
                if save_charts:
                    self._save_chart(fig, f'{col}_distribution', output_dir)
                print(f"      ✓ {col} 分布图")
            except Exception as e:
                print(f"      ❌ {col} 分布图生成失败: {str(e)}")

        # 2. 相关性分析
        if len(numeric_cols) > 1:
            print("   2. 生成相关性图...")
            try:
                fig = self.plot_correlation(data, numeric_cols)
                charts['correlation_heatmap'] = fig
                if save_charts:
                    self._save_chart(fig, 'correlation_heatmap', output_dir)
                print("      ✓ 相关性热图")
            except Exception as e:
                print(f"      ❌ 相关性热图生成失败: {str(e)}")

        # 3. 散点图分析
        if len(numeric_cols) >= 2:
            print("   3. 生成散点图...")
            # 选择最重要的几个变量
            important_cols = numeric_cols[:min(4, len(numeric_cols))]

            for i, col1 in enumerate(important_cols):
                for col2 in important_cols[i+1:]:
                    try:
                        fig = self.plot_scatter(data, col1, col2, color_col=target_col)
                        charts[f'{col1}_vs_{col2}_scatter'] = fig
                        if save_charts:
                            self._save_chart(fig, f'{col1}_vs_{col2}_scatter', output_dir)
                        print(f"      ✓ {col1} vs {col2} 散点图")
                    except Exception as e:
                        print(f"      ❌ {col1} vs {col2} 散点图生成失败: {str(e)}")

        # 4. 箱线图分析
        if len(numeric_cols) > 0:
            print("   4. 生成箱线图...")
            try:
                fig = self.plot_boxplot(data, numeric_cols[:6], group_col=target_col)
                charts['boxplot'] = fig
                if save_charts:
                    self._save_chart(fig, 'boxplot', output_dir)
                print("      ✓ 箱线图")
            except Exception as e:
                print(f"      ❌ 箱线图生成失败: {str(e)}")

        self.charts = charts
        print(f"\n🎉 自动化可视化完成！共生成 {len(charts)} 个图表")

        return charts

    def create_dashboard(self, data: pd.DataFrame, target_col: Optional[str] = None,
                        chart_types: Optional[List[str]] = None) -> go.Figure:
        """
        创建交互式仪表板

        Parameters:
        - data: 数据DataFrame
        - target_col: 目标列名
        - chart_types: 要包含的图表类型

        Returns:
        - 仪表板图表
        """
        if chart_types is None:
            chart_types = ['distribution', 'correlation', 'scatter', 'boxplot']

        # 计算子图布局
        n_charts = len(chart_types)
        cols = min(2, n_charts)
        rows = (n_charts + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'{chart_type.title()}' for chart_type in chart_types],
            specs=[[{"secondary_y": False}] * cols] * rows
        )

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for i, chart_type in enumerate(chart_types):
            row = (i // cols) + 1
            col = (i % cols) + 1

            if chart_type == 'distribution' and len(numeric_cols) > 0:
                # 分布图（第一个数值列）
                col_data = data[numeric_cols[0]].dropna()
                fig.add_trace(
                    go.Histogram(x=col_data, name='分布'),
                    row=row, col=col
                )

            elif chart_type == 'correlation' and len(numeric_cols) > 1:
                # 相关性热图
                corr_matrix = data[numeric_cols].corr()
                fig.add_trace(
                    go.Heatmap(z=corr_matrix.values,
                              x=corr_matrix.columns,
                              y=corr_matrix.columns,
                              colorscale='RdBu',
                              showscale=False),
                    row=row, col=col
                )

            elif chart_type == 'scatter' and len(numeric_cols) >= 2:
                # 散点图
                fig.add_trace(
                    go.Scatter(x=data[numeric_cols[0]],
                              y=data[numeric_cols[1]],
                              mode='markers',
                              name='散点图'),
                    row=row, col=col
                )

            elif chart_type == 'boxplot' and len(numeric_cols) > 0:
                # 箱线图
                for j, col in enumerate(numeric_cols[:3]):  # 最多3个变量
                    fig.add_trace(
                        go.Box(y=data[col], name=col),
                        row=row, col=col
                    )

        fig.update_layout(
            title_text="数据分析仪表板",
            height=300 * rows,
            showlegend=True
        )

        return fig

    def _save_chart(self, fig: Union[plt.Figure, go.Figure], filename: str, output_dir: str):
        """保存图表"""
        if isinstance(fig, plt.Figure):
            # matplotlib图表
            filepath = Path(output_dir) / f"{filename}.{self.config['save_format']}"
            fig.savefig(filepath, dpi=self.config['dpi'], bbox_inches='tight')
        elif isinstance(fig, go.Figure):
            # plotly图表
            filepath = Path(output_dir) / f"{filename}.html"
            fig.write_html(str(filepath))

    def export_charts_report(self, output_path: str, format: str = 'html'):
        """
        导出图表报告

        Parameters:
        - output_path: 输出路径
        - format: 输出格式 ('html', 'pdf')
        """
        if not self.charts:
            print("❌ 没有可导出的图表")
            return

        if format == 'html':
            self._export_html_report(output_path)
        elif format == 'pdf':
            self._export_pdf_report(output_path)
        else:
            raise ValueError("不支持的格式，请使用 'html' 或 'pdf'")

    def _export_html_report(self, output_path: str):
        """导出HTML报告"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据分析可视化报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .chart-container { margin: 20px 0; text-align: center; }
                h1 { color: #333; }
                h2 { color: #666; }
            </style>
        </head>
        <body>
            <h1>数据分析可视化报告</h1>
            <p>生成时间: {timestamp}</p>
            <p>图表总数: {chart_count}</p>
        """.format(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            chart_count=len(self.charts)
        )

        for name, fig in self.charts.items():
            if isinstance(fig, go.Figure):
                # plotly图表
                chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{name}")
                html_content += f"""
                <div class="chart-container">
                    <h2>{name.replace('_', ' ').title()}</h2>
                    {chart_html}
                </div>
                """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML报告已导出到 {output_path}")

    def _export_pdf_report(self, output_path: str):
        """导出PDF报告"""
        try:
            # 需要安装 additional packages: pip install weasyprint
            from weasyprint import HTML

            # 先生成HTML然后转换为PDF
            html_path = output_path.replace('.pdf', '.html')
            self._export_html_report(html_path)

            # 转换为PDF
            HTML(filename=html_path).write_pdf(output_path)
            print(f"✅ PDF报告已导出到 {output_path}")

        except ImportError:
            print("❌ 需要安装 weasyprint 来生成PDF报告: pip install weasyprint")
        except Exception as e:
            print(f"❌ PDF导出失败: {str(e)}")