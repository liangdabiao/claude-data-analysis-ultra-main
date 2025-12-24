#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFM Customer Segmentation Visualization Tools
Generate comprehensive visualizations and dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class RFMVisualizer:
    """Visualization engine for RFM customer segmentation analysis"""

    def __init__(self, chinese_font='SimHei'):
        """Initialize visualizer with Chinese font support"""
        self.chinese_font = chinese_font
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False

        # Set style
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')

    def create_comprehensive_dashboard(self, rfm_df, save_path='rfm_dashboard.png'):
        """
        Create comprehensive customer segmentation dashboard

        Args:
            rfm_df (pd.DataFrame): Complete RFM analysis results
            save_path (str): Path to save the dashboard image
        """
        print("生成分群分析仪表板...")

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Customer segment distribution (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_segment_distribution(rfm_df, ax1)

        # 2. Revenue share by segment (pie chart)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_revenue_share(rfm_df, ax2)

        # 3. RFM scatter plot (3D view projected)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_rfm_scatter(rfm_df, ax3)

        # 4. Box plots by segment
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_rfm_boxplots(rfm_df, ax4)

        # 5. Customer value heatmap
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_value_heatmap(rfm_df, ax5)

        # 6. Cluster distribution
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_cluster_distribution(rfm_df, ax6)

        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_summary_table(rfm_df, ax7)

        # Add main title
        fig.suptitle('RFM客户分群分析仪表板\nRFM Customer Segmentation Analysis Dashboard',
                     fontsize=20, fontweight='bold', y=0.95)

        # Add footer with data info
        fig.text(0.5, 0.02, f'分析时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                              f'客户总数: {len(rfm_df)}',
                 ha='center', fontsize=10, style='italic')

        # Save the dashboard
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"仪表板已保存: {save_path}")

    def _plot_segment_distribution(self, df, ax):
        """Plot customer segment distribution pie chart"""
        segment_counts = df['客户价值'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']  # Red, Teal, Blue

        wedges, texts, autotexts = ax.pie(
            segment_counts.values,
            labels=[f'{seg}\n({count:,}人)' for seg, count in segment_counts.items()],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )

        ax.set_title('客户分群分布\nCustomer Segment Distribution',
                    fontsize=12, fontweight='bold', pad=10)

    def _plot_revenue_share(self, df, ax):
        """Plot revenue share by segment pie chart"""
        revenue_by_segment = df.groupby('客户价值')['金额性'].sum()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

        wedges, texts, autotexts = ax.pie(
            revenue_by_segment.values,
            labels=[f'{seg}\n￥{revenue:,.0f}' for seg, revenue in revenue_by_segment.items()],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )

        ax.set_title('收入份额分布\nRevenue Share by Segment',
                    fontsize=12, fontweight='bold', pad=10)

    def _plot_rfm_scatter(self, df, ax):
        """Plot RFM scatter plot (Frequency vs Monetary, colored by segment)"""
        segment_colors = {'High': '#45b7d1', 'Medium': '#4ecdc4', 'Low': '#ff6b6b'}

        for segment in ['High', 'Medium', 'Low']:
            segment_data = df[df['客户价值'] == segment]
            if len(segment_data) > 0:
                ax.scatter(segment_data['频率性'], segment_data['金额性'],
                          c=segment_colors[segment], label=f'{segment} Value',
                          alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('购买频率 (Frequency)', fontsize=10)
        ax.set_ylabel('消费金额 (Monetary)', fontsize=10)
        ax.set_title('RFM散点图\nRFM Scatter Plot', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    def _plot_rfm_boxplots(self, ax):
        """Plot RFM box plots by customer segment"""
        # This would need the full rfm_df with segment info
        # For now, create a template
        ax.text(0.5, 0.5, 'RFM指标箱线图\n(需要完整数据)\nRFM Box Plots by Segment',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('RFM指标分布\nRFM Metrics Distribution', fontsize=12, fontweight='bold')

    def _plot_value_heatmap(self, df, ax):
        """Plot customer value correlation heatmap"""
        # Select numeric columns for correlation
        numeric_cols = ['最近性', '频率性', '金额性', 'RFM总分']
        correlation_matrix = df[numeric_cols].corr()

        sns.heatmap(correlation_matrix,
                   annot=True,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)

        ax.set_title('RFM指标相关性\nRFM Correlation Heatmap',
                    fontsize=12, fontweight='bold')

    def _plot_cluster_distribution(self, df, ax):
        """Plot cluster distribution bar chart"""
        if 'cluster' in df.columns:
            cluster_counts = df['cluster'].value_counts().sort_index()

            bars = ax.bar(cluster_counts.index, cluster_counts.values,
                         color='skyblue', alpha=0.7, edgecolor='navy')

            # Add value labels on bars
            for bar, count in zip(bars, cluster_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('聚类编号 (Cluster ID)', fontsize=10)
            ax.set_ylabel('客户数量 (Customer Count)', fontsize=10)
            ax.set_title('聚类分布\nCluster Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, '聚类数据不可用\nCluster data not available',
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)

    def _plot_summary_table(self, df, ax):
        """Create summary statistics table"""
        # Calculate key metrics
        total_customers = len(df)
        total_revenue = df['金额性'].sum()
        avg_customer_value = df['金额性'].mean()

        segment_stats = []
        for segment in ['High', 'Medium', 'Low']:
            segment_data = df[df['客户价值'] == segment]
            if len(segment_data) > 0:
                stats = {
                    '分群': segment,
                    '客户数': len(segment_data),
                    '占比': f"{len(segment_data)/total_customers*100:.1f}%",
                    '平均消费': f"￥{segment_data['金额性'].mean():.0f}",
                    '收入贡献': f"￥{segment_data['金额性'].sum():,.0f}"
                }
                segment_stats.append(stats)

        # Create table data
        table_data = []
        headers = ['指标', '数值']

        table_data.extend([
            ['总客户数', f'{total_customers:,}'],
            ['总收入', f'￥{total_revenue:,.0f}'],
            ['平均客户价值', f'￥{avg_customer_value:.0f}'],
            ['', ''],  # Separator
        ])

        for stats in segment_stats:
            table_data.extend([
                [f"{stats['分群']} 客户数", f"{stats['客户数']} ({stats['占比']})"],
                [f"{stats['分群']} 平均消费", stats['平均消费']],
                ['', ''],  # Separator
            ])

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('关键指标摘要\nKey Metrics Summary', fontsize=12, fontweight='bold')
        ax.axis('off')

    def create_individual_charts(self, rfm_df, output_dir='charts'):
        """
        Create individual charts for detailed analysis

        Args:
            rfm_df (pd.DataFrame): Complete RFM analysis results
            output_dir (str): Directory to save individual charts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("生成详细分析图表...")

        # 1. Customer segment distribution
        plt.figure(figsize=(10, 8))
        self._create_segment_distribution_detailed(rfm_df)
        plt.savefig(f'{output_dir}/segment_distribution_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. RFM score distribution
        plt.figure(figsize=(15, 5))
        self._create_rfm_score_distribution(rfm_df)
        plt.savefig(f'{output_dir}/rfm_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Customer lifetime value analysis
        plt.figure(figsize=(12, 8))
        self._create_clv_analysis(rfm_df)
        plt.savefig(f'{output_dir}/customer_lifetime_value.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"详细图表已保存到: {output_dir}/")

    def _create_segment_distribution_detailed(self, df):
        """Create detailed segment distribution chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Customer count by segment
        segment_counts = df['客户价值'].value_counts()
        colors = ['#45b7d1', '#4ecdc4', '#ff6b6b']
        ax1.pie(segment_counts.values, labels=segment_counts.index, colors=colors,
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('客户数量分布', fontweight='bold')

        # Revenue by segment
        revenue_by_segment = df.groupby('客户价值')['金额性'].sum()
        ax2.pie(revenue_by_segment.values, labels=revenue_by_segment.index, colors=colors,
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('收入分布', fontweight='bold')

        # Average order value by segment
        avg_order = df.groupby('客户价值')['金额性'].mean()
        bars = ax3.bar(avg_order.index, avg_order.values, color=colors)
        ax3.set_title('平均客户价值', fontweight='bold')
        ax3.set_ylabel('金额 (￥)')

        for bar, value in zip(bars, avg_order.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'￥{value:,.0f}', ha='center', va='bottom')

        # Customer count by RFM score
        score_counts = df['RFM总分'].value_counts().sort_index()
        ax4.bar(score_counts.index, score_counts.values, color='lightcoral', alpha=0.7)
        ax4.set_title('RFM总分分布', fontweight='bold')
        ax4.set_xlabel('RFM总分')
        ax4.set_ylabel('客户数')

        plt.tight_layout()

    def _create_rfm_score_distribution(self, df):
        """Create RFM score distribution charts"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Recency score distribution
        r_scores = df['R_评分'].value_counts().sort_index()
        ax1.bar(r_scores.index, r_scores.values, color='lightblue', alpha=0.7)
        ax1.set_title('最近性评分分布 (R)')
        ax1.set_xlabel('R评分')
        ax1.set_ylabel('客户数')

        # Frequency score distribution
        f_scores = df['F_评分'].value_counts().sort_index()
        ax2.bar(f_scores.index, f_scores.values, color='lightgreen', alpha=0.7)
        ax2.set_title('频率性评分分布 (F)')
        ax2.set_xlabel('F评分')
        ax2.set_ylabel('客户数')

        # Monetary score distribution
        m_scores = df['M_评分'].value_counts().sort_index()
        ax3.bar(m_scores.index, m_scores.values, color='lightcoral', alpha=0.7)
        ax3.set_title('金额性评分分布 (M)')
        ax3.set_xlabel('M评分')
        ax3.set_ylabel('客户数')

        plt.tight_layout()

    def _create_clv_analysis(self, df):
        """Create customer lifetime value analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Customer value vs Recency
        for segment in ['High', 'Medium', 'Low']:
            segment_data = df[df['客户价值'] == segment]
            if len(segment_data) > 0:
                ax1.scatter(segment_data['最近性'], segment_data['金额性'],
                          label=segment, alpha=0.6, s=20)
        ax1.set_xlabel('最近购买天数')
        ax1.set_ylabel('总消费金额')
        ax1.set_title('客户价值 vs 最近购买')
        ax1.legend()
        ax1.invert_xaxis()  # Lower recency is better

        # Customer value vs Frequency
        for segment in ['High', 'Medium', 'Low']:
            segment_data = df[df['客户价值'] == segment]
            if len(segment_data) > 0:
                ax2.scatter(segment_data['频率性'], segment_data['金额性'],
                          label=segment, alpha=0.6, s=20)
        ax2.set_xlabel('购买频率')
        ax2.set_ylabel('总消费金额')
        ax2.set_title('客户价值 vs 购买频率')
        ax2.legend()
        ax2.set_xscale('log')

        # RFM total score distribution
        score_dist = df['RFM总分'].value_counts().sort_index()
        ax3.bar(score_dist.index, score_dist.values, color='purple', alpha=0.7)
        ax3.set_xlabel('RFM总分')
        ax3.set_ylabel('客户数')
        ax3.set_title('RFM总分分布')

        # Customer segment by total score
        segment_scores = df.groupby('客户价值')['RFM总分'].mean()
        bars = ax4.bar(segment_scores.index, segment_scores.values,
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax4.set_title('各分群平均RFM得分')
        ax4.set_ylabel('平均RFM总分')

        for bar, score in zip(bars, segment_scores.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

def main():
    """Example usage of visualizer"""
    # Example - would use actual data
    visualizer = RFMVisualizer()

    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        '用户码': range(100),
        '最近性': np.random.randint(1, 365, 100),
        '频率性': np.random.randint(1, 50, 100),
        '金额性': np.random.randint(100, 10000, 100),
        '客户价值': np.random.choice(['High', 'Medium', 'Low'], 100),
        'RFM总分': np.random.randint(3, 9, 100),
        'R_评分': np.random.randint(1, 4, 100),
        'F_评分': np.random.randint(1, 4, 100),
        'M_评分': np.random.randint(1, 4, 100),
        'cluster': np.random.randint(0, 3, 100)
    })

    # Create dashboard
    visualizer.create_comprehensive_dashboard(sample_data)
    print("示例仪表板已生成: rfm_dashboard.png")

if __name__ == "__main__":
    main()