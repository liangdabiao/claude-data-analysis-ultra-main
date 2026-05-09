#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attribution Analysis Visualization Tools
Comprehensive visualization for marketing attribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class AttributionVisualizer:
    """Comprehensive visualization toolkit for attribution analysis"""

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

    def create_attribution_dashboard(self, attribution_results, save_path='attribution_dashboard.png'):
        """
        Create comprehensive attribution analysis dashboard

        Args:
            attribution_results (dict): Attribution analysis results
            save_path (str): Path to save the dashboard
        """
        print("创建归因分析综合仪表板...")

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Attribution weights comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_attribution_weights_comparison(attribution_results, ax1)

        # 2. Channel performance metrics (top right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_channel_performance_metrics(attribution_results, ax2)

        # 3. Customer journey path analysis (middle left)
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_journey_path_analysis(attribution_results, ax3)

        # 4. Conversion funnel analysis (middle right)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_conversion_funnel(attribution_results, ax4)

        # 5. ROI vs Attribution weights (bottom left)
        ax5 = fig.add_subplot(gs[2, 0:2])
        self._plot_roi_attribution_correlation(attribution_results, ax5)

        # 6. Channel interaction heatmap (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:4])
        self._plot_channel_interaction_heatmap(attribution_results, ax6)

        # 7. Attribution model comparison (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_model_comparison(attribution_results, ax7)

        # Add main title
        fig.suptitle('营销归因分析综合仪表板\nMarketing Attribution Analysis Dashboard',
                     fontsize=20, fontweight='bold', y=0.95)

        # Add footer with analysis info
        fig.text(0.5, 0.02, f'分析时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                              f'渠道数量: {self._get_channel_count(attribution_results)} | '
                              f'归因模型数: {self._get_model_count(attribution_results)}',
                 ha='center', fontsize=10, style='italic')

        # Save the dashboard
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"仪表板已保存: {save_path}")

    def _plot_attribution_weights_comparison(self, results, ax):
        """Plot attribution weights comparison across models"""
        if 'attribution_models' not in results:
            self._plot_placeholder(ax, "归因模型数据不可用")
            return

        models_data = results['attribution_models']
        all_channels = set()
        for model, channels in models_data.items():
            all_channels.update(channels.keys())

        # Prepare data for plotting
        plot_data = []
        for channel in all_channels:
            row = {'channel': channel}
            for model, channels in models_data.items():
                row[model] = channels.get(channel, 0.0)
            plot_data.append(row)

        df_plot = pd.DataFrame(plot_data)
        df_plot = df_plot.set_index('channel')

        # Create stacked bar chart
        df_plot.T.plot(kind='bar', stacked=True, ax=ax, figsize=(12, 8))
        ax.set_title('各归因模型的渠道权重对比', fontweight='bold', fontsize=12)
        ax.set_xlabel('归因模型')
        ax.set_ylabel('归因权重')
        ax.legend(title='归因模型', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    def _plot_channel_performance_metrics(self, results, ax):
        """Plot channel performance metrics"""
        if 'channel_performance' not in results:
            self._plot_placeholder(ax, "渠道性能数据不可用")
            return

        perf_df = pd.DataFrame(results['channel_performance'])

        # Select key metrics for visualization
        metrics = ['conversion_rate', 'cpa', 'roi']
        metric_labels = {
            'conversion_rate': '转化率',
            'cpa': 'CPA (获客成本)',
            'roi': 'ROI (投资回报率)'
        }

        # Create subplots
        n_metrics = len(metrics)
        for i, metric in enumerate(metrics):
            sub_ax = ax.inset_axes([0.1, 0.8 - i*0.25, 0.8, 0.2])
            perf_df_sorted = perf_df.sort_values(metric, ascending=(metric == 'cpa'))

            bars = sub_ax.barh(range(len(perf_df_sorted)), perf_df_sorted[metric],
                               color=plt.cm.RdYlGn_r([0.3, 0.7, 0.9]))

            sub_ax.set_yticks(range(len(perf_df_sorted)))
            sub_ax.set_yticklabels([ch[:15] for ch in perf_df_sorted['channel']], fontsize=8)
            sub_ax.set_xlabel(metric_labels[metric], fontsize=9)
            sub_ax.grid(True, alpha=0.3)

            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, perf_df_sorted[metric])):
                if metric == 'roi':
                    sub_ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{value:.1f}x', ha='left', va='center', fontsize=7)
                else:
                    sub_ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{value:.2f}', ha='left', va='center', fontsize=7)

        ax.set_title('渠道关键性能指标', fontweight='bold', fontsize=12)
        ax.axis('off')

    def _plot_journey_path_analysis(self, results, ax):
        """Plot customer journey path analysis"""
        if 'customer_paths' not in results:
            self._plot_placeholder(ax, "客户路径数据不可用")
            return

        paths_df = pd.DataFrame(results['customer_paths'])

        # Analyze path lengths
        path_lengths = [len(path) for path in paths_df['path']]
        path_lengths_by_conversion = {
            '转化路径': [len(path) for path in paths_df[paths_df['converted'] == 1]['path']],
            '未转化路径': [len(path) for path in paths_df[paths_df['converted'] == 0]['path']]
        }

        # Create histograms
        colors = ['green', 'red']
        labels = ['转化路径', '未转化路径']

        for i, (label, lengths) in enumerate(path_lengths_by_conversion.items()):
            ax.hist(lengths, alpha=0.6, bins=15, label=label, color=colors[i])

        ax.set_xlabel('路径长度')
        ax.set_ylabel('频次')
        ax.set_title('客户旅程路径长度分布', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        avg_converted_len = np.mean(path_lengths_by_conversion['转化路径'])
        avg_non_converted_len = np.mean(path_lengths_by_conversion['未转化路径'])

        ax.text(0.7, 0.8, f'转化路径平均长度: {avg_converted_len:.1f}',
               transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor='lightgreen'))
        ax.text(0.7, 0.6, f'未转化路径平均长度: {avg_non_converted_len:.1f}',
               transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor='lightcoral'))

    def _plot_conversion_funnel(self, results, ax):
        """Plot conversion funnel analysis"""
        if 'customer_paths' not in results:
            self._plot_placeholder(ax, "漏斗分析数据不可用")
            return

        paths_df = pd.DataFrame(results['customer_paths'])

        # Calculate funnel stages
        total_users = len(paths_df)
        converted_users = paths_df['converted'].sum()
        non_converted_users = total_users - converted_users

        # Funnel stages based on path length
        funnel_stages = []
        stage_labels = []

        # Define stages based on path length distribution
        max_length = max([len(path) for path in paths_df['path']])
        for length in range(2, min(max_length + 1, 7)):
            stage_users = len([path for path in paths_df['path'] if len(path) >= length])
            funnel_stages.append(stage_users)
            stage_labels.append(f'{length}+个触点')

        # Create funnel plot
        funnel_stages.insert(0, total_users)
        stage_labels.insert(0, '总用户数')

        # Calculate conversion rates
        conversion_rates = [converted_users / stage if stage > 0 else 0 for stage in funnel_stages]

        # Create funnel bars
        bars = ax.barh(range(len(funnel_stages)), funnel_stages, color=plt.cm.Blues_r(np.linspace(0.3, 0.9, len(funnel_stages))))

        ax.set_yticks(range(len(funnel_stages)))
        ax.set_yticklabels(stage_labels)
        ax.set_xlabel('用户数')
        ax.set_title('转化漏斗分析', fontweight='bold')

        # Add conversion rate labels
        for i, (bar, rate) in enumerate(zip(bars, conversion_rates)):
            ax.text(bar.get_width() + total_users * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{rate:.1%}', ha='left', va='center', fontsize=9)

        ax.grid(True, alpha=0.3, axis='x')

    def _plot_roi_attribution_correlation(self, results, ax):
        """Plot ROI vs Attribution weights correlation"""
        if 'channel_performance' not in results or 'attribution_models' not in results:
            self._plot_placeholder(ax, "ROI和归因权重数据不可用")
            return

        perf_df = pd.DataFrame(results['channel_performance'])
        models_data = results['attribution_models']

        # Use the average attribution weight across models
        channel_weights = {}
        for model, channels in models_data.items():
            for channel, weight in channels.items():
                if channel not in channel_weights:
                    channel_weights[channel] = []
                channel_weights[channel].append(weight)

        avg_weights = {ch: np.mean(weights) for ch, weights in channel_weights.items()}

        # Create scatter plot
        channels = list(set(perf_df['channel']) & set(avg_weights.keys()))
        roi_data = []
        weight_data = []

        for channel in channels:
            channel_perf = perf_df[perf_df['channel'] == channel].iloc[0]
            if channel_perf['roi'] != float('inf'):  # Exclude infinite ROI
                roi_data.append(channel_perf['roi'])
                weight_data.append(avg_weights[channel])

        if len(roi_data) > 0:
            scatter = ax.scatter(weight_data, roi_data, alpha=0.6, s=60)

            # Add trend line
            z = np.polyfit(weight_data, roi_data, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(weight_data), max(weight_data), 100)
            y_trend = p(x_trend)
            ax.plot(x_trend, y_trend, "r--", alpha=0.8)

            # Calculate correlation
            correlation = np.corrcoef(weight_data, roi_data)[0, 1]

            # Add channel labels
            for i, channel in enumerate(channels):
                if i < len(channels) // 2:  # Label only half of the points to avoid overcrowding
                    ax.annotate(channel[:10], (weight_data[i], roi_data[i]),
                                 fontsize=8, alpha=0.7)

            ax.set_xlabel('平均归因权重')
            ax.set_ylabel('ROI')
            ax.set_title(f'ROI vs 归因权重相关性 (r={correlation:.3f})', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add correlation interpretation
            if correlation > 0.7:
                interpretation = "强正相关"
            elif correlation > 0.3:
                interpretation = "中等正相关"
            elif correlation > -0.3:
                interpretation = "弱相关"
            else:
                interpretation = "负相关"

            ax.text(0.7, 0.9, f'相关性: {interpretation}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))

    def _plot_channel_interaction_heatmap(self, results, ax):
        """Plot channel interaction heatmap"""
        if 'attribution_models' not in results or 'shapley_values' not in results:
            self._plot_placeholder(ax, "渠道协同数据不可用")
            return

        # For simplicity, we'll create a correlation matrix of attribution weights across models
        models_data = results['attribution_models']

        if len(models_data) < 2:
            self._plot_placeholder(ax, "需要多个归因模型进行比较")
            return

        # Create channel x model matrix
        all_channels = set()
        for model in models_data.values():
            all_channels.update(model.keys())

        # Create correlation matrix
        matrix_data = []
        model_names = list(models_data.keys())

        for channel in all_channels:
            row = []
            for model in model_names:
                weight = models_data[model].get(channel, 0.0)
                row.append(weight)
            matrix_data.append(row)

        if len(matrix_data) > 0:
            df_matrix = pd.DataFrame(matrix_data, index=list(all_channels), columns=model_names)

            # Calculate correlation matrix
            corr_matrix = df_matrix.corr()

            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('渠道归因权重相关性热力图', fontweight='bold')

            # Rotate labels if needed
            if len(model_names) > 5:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_model_comparison(self, results, ax):
        """Plot comparison of different attribution models"""
        if 'attribution_comparison' not in results:
            self._plot_placeholder(ax, "模型比较数据不可用")
            return

        comp_df = pd.DataFrame(results['attribution_comparison'])

        if comp_df.empty:
            self._plot_placeholder(ax, "模型比较数据为空")
            return

        # Select key models for comparison
        model_columns = ['first_touch', 'last_touch', 'linear', 'markov_chain']
        available_models = [col for col in model_columns if col in comp_df.columns]

        if not available_models:
            self._plot_placeholder(ax, "没有可用的归因模型数据")
            return

        # Create comparison plot
        plot_data = comp_df[['channel'] + available_models].set_index('channel')

        # Create horizontal bar plot
        plot_data.plot(kind='barh', ax=ax, figsize=(14, 8))
        ax.set_xlabel('归因权重')
        ax.set_title('多归因模型权重对比', fontweight='bold', fontsize=12)
        ax.legend(title='归因模型')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y')

        # Add statistical summary
        model_stats = {}
        for model in available_models:
            weights = comp_df[model].fillna(0)
            model_stats[model] = {
                'mean': weights.mean(),
                'std': weights.std(),
                'max': weights.max(),
                'min': weights.min()
            }

        stats_text = "模型统计:\n"
        for model, stats in model_stats.items():
            stats_text += f"{model}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}\n"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))

    def _plot_placeholder(self, ax, message):
        """Plot placeholder text when data is not available"""
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12,
               transform=ax.transAxes, style='italic', color='gray')
        ax.set_title('数据不可用', fontweight='bold')
        ax.axis('off')

    def _get_channel_count(self, results):
        """Get channel count from results"""
        if 'attribution_models' in results:
            return len(set().union(*[set(model.keys()) for model in results['attribution_models'].values()]))
        elif 'channel_performance' in results:
            return len(results['channel_performance'])
        else:
            return 0

    def _get_model_count(self, results):
        """Get model count from results"""
        if 'attribution_models' in results:
            return len(results['attribution_models'])
        else:
            return 0

    def create_markov_visualization(self, markov_results, save_path='markov_analysis.png'):
        """
        Create Markov chain analysis visualization

        Args:
            markov_results (dict): Markov chain analysis results
            save_path (str): Path to save the visualization
        """
        print("创建马尔可夫链可视化...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('马尔可夫链归因分析', fontsize=16, fontweight='bold')

        # 1. Transition matrix heatmap
        if 'transition_matrix' in markov_results:
            transition_matrix = markov_results['transition_matrix']

            # Focus on channels only (remove start/end states for clarity)
            channels = [state for state in transition_matrix.columns
                       if state not in ['开始', '未转化', '成功转化']]

            if channels:
                channel_matrix = transition_matrix.loc[channels, channels]
                sns.heatmap(channel_matrix, annot=True, cmap='Blues', fmt='.3f', ax=ax1)
                ax1.set_title('渠道转移概率矩阵', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, '渠道数据不可用', ha='center', va='center',
                         transform=ax1.transAxes, style='italic')
                ax1.axis('off')

        # 2. Removal effects
        if 'removal_effects' in markov_results:
            removal_effects = markov_results['removal_effects']

            channels = list(removal_effects.keys())
            effects = list(removal_effects.values())

            bars = ax2.barh(channels, effects)
            ax2.set_xlabel('移除效应')
            ax2.set_title('渠道移除效应', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, effects):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{value:.3f}', ha='left', va='center', fontsize=9)

        # 3. Channel network graph
        if 'channel_graph' in markov_results:
            graph = markov_results['channel_graph']

            if graph and len(graph.nodes()) > 0:
                pos = nx.spring_layout(graph, k=1, iterations=50)

                # Calculate node sizes based on out-degree
                out_degrees = [graph.out_degree(node) for node in graph.nodes()]
                node_sizes = [deg * 500 + 100 for deg in out_degrees]

                # Calculate edge widths based on weight
                edges = graph.edges()
                edge_widths = [graph[u][v]['weight'] * 5 for u, v in edges]

                nx.draw_networkx_nodes(graph, pos, ax=ax3, node_size=node_sizes,
                                     node_color='lightblue', alpha=0.7)
                nx.draw_networkx_edges(graph, pos, ax=ax3, width=edge_widths,
                                      edge_color='gray', alpha=0.5)
                nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=8)

                ax3.set_title('渠道转换网络图', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '网络图数据不可用', ha='center', va='center',
                         transform=ax3.transAxes, style='italic')
                ax3.axis('off')

        # 4. Attribution weights comparison
        if 'attribution_weights' in markov_results:
            attribution_weights = markov_results['attribution_weights']

            channels = list(attribution_weights.keys())
            weights = list(attribution_weights.values())

            bars = ax4.barh(channels, weights)
            ax4.set_xlabel('归因权重')
            ax4.set_title('马尔可夫链归因权重', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Add percentage labels
            for bar, weight in zip(bars, weights):
                ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{weight:.1%}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"马尔可夫链可视化已保存: {save_path}")

    def create_shapley_visualization(self, shapley_results, save_path='shapley_analysis.png'):
        """
        Create Shapley value analysis visualization

        Args:
            shapley_results (dict): Shapley value analysis results
            save_path (str): Path to save the visualization
        """
        print("创建Shapley值可视化...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Shapley值归因分析', fontsize=16, fontweight='bold')

        # 1. Shapley value distribution
        if 'attribution_weights' in shapley_results:
            attribution_weights = shapley_results['attribution_weights']

            channels = list(attribution_weights.keys())
            weights = list(attribution_weights.values())

            bars = ax1.barh(channels, weights)
            ax1.set_xlabel('归因权重')
            ax1.set_title('Shapley值归因权重', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Add percentage labels
            for bar, weight in zip(bars, weights):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{weight:.1%}', ha='left', va='center', fontsize=9)

        # 2. Marginal contribution analysis
        if 'marginal_analysis' in shapley_results:
            marginal_df = pd.DataFrame(shapley_results['marginal_analysis'])

            # Plot by performance tier
            tier_groups = marginal_df.groupby('performance_tier')['shapley_value'].mean()
            tier_groups = tier_groups.sort_values(ascending=False)

            colors = {'top_performer': '#2ecc71', 'strong_performer': '#3498db',
                       'moderate_performer': '#f1c40f', 'low_performer': '#e74c3c'}

            bars = ax2.bar(range(len(tier_groups)), tier_groups.values(),
                          color=[colors.get(tier, 'gray') for tier in tier_groups.index])
            ax2.set_xticks(range(len(tier_groups)))
            ax2.set_xticklabels([tier.replace('_', ' ').title() for tier in tier_groups.index],
                               rotation=45, ha='right')
            ax2.set_ylabel('平均Shapley值')
            ax2.set_title('按性能层级的边际贡献', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # 3. Channel synergy analysis
        if 'channel_synergy' in shapley_results:
            synergy_data = shapley_results['channel_synergy']

            if synergy_data:
                # Select top synergies for visualization
                top_synergies = dict(list(synergy_data.items())[:10])
                channel_pairs = [f"{data['channel1']} + {data['channel2']}"
                                 for data in top_synergies.values()]
                synergy_ratios = [data['synergy_ratio'] for data in top_synergies.values()]
                synergy_types = [data['synergy_type'] for data in top_synergies.values()]

                bars = ax3.barh(range(len(channel_pairs)), synergy_ratios,
                              color=['green' if st == 'positive' else 'orange' if st == 'neutral' else 'red'
                                    for st in synergy_types])
                ax3.set_yticks(range(len(channel_pairs)))
                ax3.set_yticklabels([pair[:20] for pair in channel_pairs], fontsize=8)
                ax3.set_xlabel('协同比')
                ax3.set_title('渠道协同效应 (前10个)', fontweight='bold')
                ax3.grid(True, alpha=0.3)

                # Add vertical line at x=1
                ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5)

        # 4. Optimization recommendations
        if 'optimization' in shapley_results:
            opt_results = shapley_results['optimization']

            if 'recommendations' in opt_results:
                recommendations = opt_results['recommendations']

                # Group recommendations by action type
                actions = {}
                for rec in recommendations:
                    action_type = rec['action']
                    if action_type not in actions:
                        actions[action_type] = []
                    actions[action_type].append(rec['channel'])

                # Create pie chart
                action_counts = {action: len(channels) for action, channels in actions.items()}
                colors = {'increase': '#2ecc71', 'decrease': '#e74c3c', 'optimize': '#f39c12'}

                if action_counts:
                    wedges, texts, autotexts = ax4.pie(
                        action_counts.values(),
                        labels=list(action_counts.keys()),
                        colors=[colors.get(action, 'gray') for action in action_counts.keys()],
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax4.set_title('优化建议分布', fontweight='bold')
                else:
                    ax4.text(0.5, 0.5, '无优化建议数据', ha='center', va='center',
                             transform=ax4.transAxes, style='italic')
                    ax4.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Shapley值可视化已保存: {save_path}")

def main():
    """Example usage of attribution visualizer"""
    visualizer = AttributionVisualizer()

    # Create sample attribution results
    sample_results = {
        'attribution_models': {
            'first_touch': {'付费搜索': 0.3, '社交媒体': 0.25, '邮件营销': 0.2, '内容营销': 0.15, '线下广告': 0.1},
            'last_touch': {'付费搜索': 0.2, '社交媒体': 0.35, '邮件营销': 0.25, '内容营销': 0.15, '线下广告': 0.05},
            'markov_chain': {'付费搜索': 0.28, '社交媒体': 0.30, '邮件营销': 0.22, '内容营销': 0.14, '线下广告': 0.06}
        },
        'channel_performance': [
            {'channel': '付费搜索', 'conversion_rate': 0.025, 'cpa': 50.0, 'roi': 2.5},
            {'channel': '社交媒体', 'conversion_rate': 0.035, 'cpa': 40.0, 'roi': 3.0},
            {'channel': '邮件营销', 'conversion_rate': 0.020, 'cpa': 30.0, 'roi': 4.0},
            {'channel': '内容营销', 'conversion_rate': 0.015, 'cpa': 60.0, 'roi': 1.5},
            {'channel': '线下广告', 'conversion_rate': 0.010, 'cpa': 80.0, 'roi': 1.2}
        ]
    }

    # Create dashboard
    visualizer.create_attribution_dashboard(sample_results)
    print("示例归因仪表板已生成: attribution_dashboard.png")

if __name__ == "__main__":
    main()