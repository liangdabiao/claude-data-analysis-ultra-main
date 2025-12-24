"""
增长可视化模块 - 增长模型图表和仪表板

提供全面的增长分析可视化功能，包括：
- 转化漏斗可视化
- 用户分群图表
- 增长趋势分析
- ROI分析图表
- 交互式仪表板
- Qini曲线和Uplift模型可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GrowthVisualizer:
    """增长可视化器 - 专业图表和仪表板生成工具"""

    def __init__(self, style: str = 'seaborn', palette: str = 'husl'):
        """
        初始化可视化器

        Parameters:
        - style: 图表风格
        - palette: 色彩方案
        """
        self.style = style
        self.palette = palette
        self.setup_style()

    def setup_style(self):
        """设置图表样式"""
        if self.style == 'seaborn':
            sns.set_style("whitegrid")
            sns.set_palette(self.palette)

    def plot_conversion_funnel(self,
                             funnel_data: List[Dict],
                             title: str = "用户转化漏斗",
                             interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        绘制转化漏斗图

        Parameters:
        - funnel_data: 漏斗数据列表
        - title: 图表标题
        - interactive: 是否使用交互式图表

        Returns:
        - 图表对象
        """
        if interactive:
            return self._plot_interactive_funnel(funnel_data, title)
        else:
            return self._plot_static_funnel(funnel_data, title)

    def _plot_interactive_funnel(self, funnel_data: List[Dict], title: str) -> go.Figure:
        """绘制交互式漏斗图"""
        fig = go.Figure(go.Funnel(
            y=[item['stage'] for item in funnel_data],
            x=[item['users'] for item in funnel_data],
            textinfo="value+percent initial",
            textposition="inside",
            textfont={"size": 14},
            marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(funnel_data)]},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 2}}
        ))

        fig.update_layout(
            title=f"{title}<br><sup>总用户数: {funnel_data[0]['users']:,}</sup>",
            font_size=12,
            height=600
        )

        return fig

    def _plot_static_funnel(self, funnel_data: List[Dict], title: str) -> plt.Figure:
        """绘制静态漏斗图"""
        fig, ax = plt.subplots(figsize=(10, 8))

        stages = [item['stage'] for item in funnel_data]
        users = [item['users'] for item in funnel_data]
        colors = sns.color_palette(self.palette, len(funnel_data))

        # 绘制漏斗条形图
        bars = ax.barh(stages, users, color=colors, alpha=0.8)

        # 添加转化率标签
        for i, (stage, bar) in enumerate(zip(funnel_data, bars)):
            if i < len(funnel_data) - 1:
                conversion_rate = funnel_data[i+1]['users'] / funnel_data[i]['users']
                ax.text(bar.get_width() + max(users) * 0.01, bar.get_y() + bar.get_height()/2,
                       f"转化率: {conversion_rate:.1%}",
                       va='center', fontsize=10)

        # 美化图表
        ax.set_xlabel('用户数', fontsize=12)
        ax.set_title(f"{title}\n总用户数: {funnel_data[0]['users']:,}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (stage, bar) in enumerate(zip(funnel_data, bars)):
            ax.text(bar.get_width() - max(users) * 0.05, bar.get_y() + bar.get_height()/2,
                   f"{int(bar.get_width()):,}",
                   ha='right', va='center', fontsize=11, fontweight='bold', color='white')

        plt.tight_layout()
        return fig

    def plot_rfm_segments(self,
                         rfm_data: pd.DataFrame,
                         title: str = "RFM用户分群分析") -> go.Figure:
        """
        绘制RFM分群散点图

        Parameters:
        - rfm_data: RFM分析数据
        - title: 图表标题

        Returns:
        - 交互式散点图
        """
        fig = px.scatter(
            rfm_data,
            x='Recency',
            y='Frequency',
            size='Monetary',
            color='RFM_Segment',
            hover_data=['RFM_Total', 'Cluster'],
            title=title,
            labels={
                'Recency': '近度 (R值)',
                'Frequency': '频度 (F值)',
                'Monetary': '金额 (M值)',
                'RFM_Segment': '用户分群'
            },
            color_discrete_map={
                'Champions': '#FF6B6B',
                'Loyal Customers': '#4ECDC4',
                'Potential Loyalists': '#45B7D1',
                'New Customers': '#96CEB4',
                'At Risk': '#FFEAA7',
                'Cannot Lose Them': '#DDA0DD',
                'Lost': '#FFA07A'
            }
        )

        fig.update_layout(
            height=600,
            xaxis_title="近度 (R值) - 越小越好",
            yaxis_title="频度 (F值) - 越大越好"
        )

        return fig

    def plot_campaign_comparison(self,
                               campaign_data: Dict,
                               metrics: List[str] = ['转化率', 'ROI', 'CPA'],
                               title: str = "营销活动对比分析") -> go.Figure:
        """
        绘制营销活动对比雷达图

        Parameters:
        - campaign_data: 活动数据
        - metrics: 对比指标
        - title: 图表标题

        Returns:
        - 雷达图
        """
        if isinstance(campaign_data, dict) and 'campaign_metrics' in campaign_data:
            df = pd.DataFrame(campaign_data['campaign_metrics']).T
        else:
            df = pd.DataFrame(campaign_data).T

        # 标准化数据到0-1范围
        normalized_df = df[metrics].copy()
        for metric in metrics:
            if metric == 'CPA':  # CPA越小越好
                normalized_df[metric] = 1 - (normalized_df[metric] - normalized_df[metric].min()) / (normalized_df[metric].max() - normalized_df[metric].min())
            else:
                normalized_df[metric] = (normalized_df[metric] - normalized_df[metric].min()) / (normalized_df[metric].max() - normalized_df[metric].min())

        fig = go.Figure()

        colors = px.colors.qualitative.Set1
        for i, (campaign, row) in enumerate(normalized_df.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + row.values.tolist()[:1],  # 闭合雷达图
                theta=metrics + metrics[:1],
                fill='toself',
                name=campaign,
                line_color=colors[i % len(colors)]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=title,
            height=600
        )

        return fig

    def plot_qini_curve(self,
                       qini_data: Dict,
                       title: str = "Qini曲线分析") -> go.Figure:
        """
        绘制Qini曲线

        Parameters:
        - qini_data: Qini曲线数据
        - title: 图表标题

        Returns:
        - Qini曲线图
        """
        fractions = qini_data['fraction']
        qini_gains = qini_data['qini_gain']
        random_qini = qini_data['random_qini']

        fig = go.Figure()

        # 模型Qini曲线
        fig.add_trace(go.Scatter(
            x=fractions,
            y=qini_gains,
            mode='lines',
            name='Uplift模型',
            line=dict(color='blue', width=3)
        ))

        # 随机模型曲线
        fig.add_trace(go.Scatter(
            x=fractions,
            y=random_qini,
            mode='lines',
            name='随机模型',
            line=dict(color='gray', width=2, dash='dash')
        ))

        # 填充区域
        fig.add_trace(go.Scatter(
            x=fractions + fractions[::-1],
            y=qini_gains + random_qini[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='增益区域'
        ))

        fig.update_layout(
            title=f"{title}<br><sup>AUQC: {qini_data.get('auqc', 0):.4f}</sup>",
            xaxis_title='人口比例',
            yaxis_title='累积增量',
            height=500
        )

        return fig

    def plot_uplift_distribution(self,
                                uplift_data: pd.DataFrame,
                                title: str = "增量分数分布") -> go.Figure:
        """
        绘制增量分数分布图

        Parameters:
        - uplift_data: 增量分析数据
        - title: 图表标题

        Returns:
        - 分布图
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('增量分数分布', '分群用户数', '平均增量分数', '累积分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. 增量分数直方图
        fig.add_trace(
            go.Histogram(x=uplift_data['uplift_score'], name='增量分数', nbinsx=50),
            row=1, col=1
        )

        # 2. 分群用户数
        if 'uplift_segment' in uplift_data.columns:
            segment_counts = uplift_data['uplift_segment'].value_counts()
            fig.add_trace(
                go.Bar(x=segment_counts.index, y=segment_counts.values, name='分群用户数'),
                row=1, col=2
            )

        # 3. 各分群平均增量分数
        if 'uplift_segment' in uplift_data.columns:
            segment_uplift = uplift_data.groupby('uplift_segment')['uplift_score'].mean()
            fig.add_trace(
                go.Bar(x=segment_uplift.index, y=segment_uplift.values, name='平均增量分数'),
                row=2, col=1
            )

        # 4. 累积分布
        sorted_uplift = np.sort(uplift_data['uplift_score'])
        cumulative = np.arange(1, len(sorted_uplift) + 1) / len(sorted_uplift)
        fig.add_trace(
            go.Scatter(x=sorted_uplift, y=cumulative, name='累积分布', mode='lines'),
            row=2, col=2
        )

        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )

        return fig

    def plot_roi_dashboard(self,
                          roi_data: Dict,
                          title: str = "ROI分析仪表板") -> go.Figure:
        """
        绘制ROI分析仪表板

        Parameters:
        - roi_data: ROI分析数据
        - title: 仪表板标题

        Returns:
        - 仪表板图表
        """
        if isinstance(roi_data, dict) and 'campaign_metrics' in roi_data:
            df = pd.DataFrame(roi_data['campaign_metrics']).T
        else:
            df = pd.DataFrame(roi_data).T

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI对比', '转化率vs成本', '收入vs投入', '利润率分析'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # 1. ROI对比
        fig.add_trace(
            go.Bar(x=df.index, y=df['ROI'], name='ROI', marker_color='lightblue'),
            row=1, col=1
        )

        # 2. 转化率vs成本散点图
        fig.add_trace(
            go.Scatter(
                x=df['CPA'], y=df['转化率'],
                mode='markers+text',
                text=df.index,
                textposition="top center",
                name='转化率vsCPA',
                marker=dict(size=df['触达用户数']/100, color=df['ROI'], colorscale='Viridis', showscale=True)
            ),
            row=1, col=2
        )

        # 3. 收入vs投入
        fig.add_trace(
            go.Bar(x=df.index, y=df['总收入'], name='总收入', marker_color='lightgreen'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['总成本'], name='总成本', marker_color='lightcoral'),
            row=2, col=1
        )

        # 4. 利润率分析
        fig.add_trace(
            go.Bar(x=df.index, y=df['利润率'], name='利润率', marker_color='gold'),
            row=2, col=2
        )

        fig.update_layout(
            title=f"{title}<br><sup>整体ROI: {roi_data.get('overall_roi', 0):.2%}</sup>",
            height=800,
            showlegend=True
        )

        return fig

    def plot_growth_trend(self,
                         trend_data: Dict,
                         title: str = "增长趋势分析") -> go.Figure:
        """
        绘制增长趋势图

        Parameters:
        - trend_data: 趋势数据
        - title: 图表标题

        Returns:
        - 趋势图
        """
        if isinstance(trend_data, dict) and 'time_series' in trend_data:
            df = pd.DataFrame(trend_data['time_series']).T
            df.index = pd.to_datetime(df.index)
        else:
            df = trend_data

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('收入成本趋势', 'ROI趋势', '月环比变化'),
            vertical_spacing=0.08
        )

        # 1. 收入成本趋势
        fig.add_trace(
            go.Scatter(x=df.index, y=df['总收入'], name='总收入', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['总成本'], name='总成本', line=dict(color='red')),
            row=1, col=1
        )

        # 2. ROI趋势
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ROI'], name='ROI', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ROI_3MAvg'], name='3月平均', line=dict(color='orange', dash='dash')),
            row=2, col=1
        )

        # 3. 月环比变化
        fig.add_trace(
            go.Bar(x=df.index, y=df['ROI_MoM'], name='月环比变化'),
            row=3, col=1
        )

        fig.update_layout(
            title=title,
            height=900,
            showlegend=True
        )

        return fig

    def plot_budget_optimization(self,
                                optimization_data: Dict,
                                title: str = "预算优化分配") -> go.Figure:
        """
        绘制预算优化图

        Parameters:
        - optimization_data: 优化数据
        - title: 图表标题

        Returns:
        - 预算优化图
        """
        allocations = optimization_data['allocations']
        campaigns = list(allocations.keys())
        budgets = [allocations[c]['分配预算'] for c in campaigns]
        expected_returns = [allocations[c]['预期回报'] for c in campaigns]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('预算分配', '预期回报'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )

        # 1. 预算分配饼图
        fig.add_trace(
            go.Pie(labels=campaigns, values=budgets, name="预算分配"),
            row=1, col=1
        )

        # 2. 预期回报柱状图
        fig.add_trace(
            go.Bar(x=campaigns, y=expected_returns, name='预期回报', marker_color='lightblue'),
            row=1, col=2
        )

        fig.update_layout(
            title=f"{title}<br><sup>总预期回报: {optimization_data.get('total_expected_return', 0):,.0f}</sup>",
            height=500
        )

        return fig

    def create_growth_dashboard(self,
                               data_dict: Dict,
                               title: str = "增长分析综合仪表板") -> go.Figure:
        """
        创建综合增长分析仪表板

        Parameters:
        - data_dict: 包含所有分析数据的字典
        - title: 仪表板标题

        Returns:
        - 综合仪表板
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '转化漏斗', 'RFM分群', 'ROI对比',
                'Qini曲线', '增量分布', '增长趋势',
                '预算分配', '活动效果', '关键指标'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "indicator"}]
            ]
        )

        # 根据可用数据添加图表
        # 这里简化处理，实际使用时需要根据具体数据结构调整

        fig.update_layout(
            title=title,
            height=1200,
            showlegend=False
        )

        return fig

    def export_charts(self, figs: List[go.Figure], output_dir: str = "charts"):
        """
        导出图表到文件

        Parameters:
        - figs: 图表列表
        - output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, fig in enumerate(figs):
            filename = f"chart_{i+1}.html"
            filepath = os.path.join(output_dir, filename)
            fig.write_html(filepath)

        print(f"✅ 已导出 {len(figs)} 个图表到 {output_dir} 目录")

    def set_chinese_font(self):
        """设置中文字体支持"""
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False