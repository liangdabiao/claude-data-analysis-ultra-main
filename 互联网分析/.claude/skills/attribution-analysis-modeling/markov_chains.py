#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markov Chain Attribution Analysis
Advanced attribution analysis using Markov chain models and removal effects
"""

import pandas as pd
import numpy as np
from scipy.linalg import inv
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class MarkovChainAttributor:
    """Markov chain-based attribution analysis"""

    def __init__(self):
        """Initialize the Markov chain attributor"""
        self.transition_matrix = None
        self.removal_effects = {}
        self.attribution_weights = {}
        self.channel_graph = None

    def build_transition_matrix(self, paths_df):
        """
        Build transition probability matrix from customer paths

        Args:
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            pd.DataFrame: Transition probability matrix
        """
        print("\n=== 构建马尔可夫转移矩阵 ===")

        # Extract all unique states (channels + start + end states)
        all_states = set()
        for path in paths_df['path']:
            all_states.update(path)

        # Initialize transition counts
        transition_counts = {}
        for state1 in all_states:
            for state2 in all_states:
                transition_counts[f"{state1}>{state2}"] = 0

        # Count transitions
        for path in paths_df['path']:
            for i in range(len(path) - 1):
                current_state = path[i]
                next_state = path[i + 1]
                transition_counts[f"{current_state}>{next_state}"] += 1

        # Calculate transition probabilities
        transition_probabilities = {}
        state_totals = {}

        # Calculate totals for each state (excluding end states)
        for transition, count in transition_counts.items():
            state = transition.split('>')[0]
            if state not in ['成功转化', '未转化']:
                state_totals[state] = state_totals.get(state, 0) + count

        # Calculate probabilities
        for transition, count in transition_counts.items():
            state = transition.split('>')[0]
            if state in state_totals and state_totals[state] > 0:
                transition_probabilities[transition] = count / state_totals[state]
            else:
                transition_probabilities[transition] = 0

        # Create transition matrix
        states_list = sorted(list(all_states))
        transition_matrix = pd.DataFrame(0.0, index=states_list, columns=states_list)

        # Fill transition matrix
        for transition, prob in transition_probabilities.items():
            from_state, to_state = transition.split('>')
            transition_matrix.at[from_state, to_state] = prob

        # Set diagonal elements for absorbing states
        for state in ['成功转化', '未转化']:
            transition_matrix.at[state, state] = 1.0

        # Set diagonal elements for other states to ensure rows sum to 1
        for state in states_list:
            if state not in ['成功转化', '未转化']:
                row_sum = transition_matrix.loc[state].sum()
                if row_sum < 1.0:
                    # Add remaining probability to '未转化'
                    transition_matrix.at[state, '未转化'] = 1.0 - row_sum

        self.transition_matrix = transition_matrix

        print(f"转移矩阵形状: {transition_matrix.shape}")
        print(f"状态数量: {len(states_list)}")
        print(f"状态列表: {states_list}")

        return transition_matrix

    def calculate_removal_effects(self, paths_df, base_conversion_rate):
        """
        Calculate removal effects for each channel

        Args:
            paths_df (pd.DataFrame): Customer journey paths
            base_conversion_rate (float): Base conversion rate

        Returns:
            dict: Removal effects for each channel
        """
        print("\n=== 计算移除效应 ===")

        transition_matrix = self.transition_matrix.copy()
        channels = [state for state in transition_matrix.columns
                   if state not in ['开始', '未转化', '成功转化']]

        removal_effects = {}
        base_cvr = base_conversion_rate

        for channel in channels:
            # Create matrix without the channel
            reduced_matrix = transition_matrix.drop(channel, axis=0).drop(channel, axis=1)

            # Adjust probabilities for remaining channels
            for col in reduced_matrix.columns:
                if col not in ['未转化', '成功转化']:
                    row_sum = reduced_matrix.loc[col].sum()
                    if row_sum < 1.0:
                        # Add remaining probability to '未转化'
                        reduced_matrix.at[col, '未转化'] = 1.0 - row_sum

            # Ensure '未转化' state absorbs properly
            reduced_matrix.at['未转化', '未转化'] = 1.0

            # Calculate conversion rate with channel removed
            try:
                # Split matrix into transient and absorbing states
                transient_states = [state for state in reduced_matrix.index
                                  if state not in ['未转化', '成功转化']]
                absorbing_states = ['未转化', '成功转化']

                if len(transient_states) > 0:
                    Q = reduced_matrix.loc[transient_states, transient_states].values
                    R = reduced_matrix.loc[transient_states, absorbing_states].values

                    # Calculate fundamental matrix
                    I = np.identity(len(Q))
                    N = inv(I - Q)

                    # Calculate absorption probabilities
                    B = np.dot(N, R)

                    # Get probability of successful conversion from start
                    start_index = transient_states.index('开始') if '开始' in transient_states else 0
                    success_index = absorbing_states.index('成功转化')
                    new_cvr = B[start_index, success_index]
                else:
                    new_cvr = 0.0

            except:
                # Fallback calculation
                new_cvr = 0.0

            # Calculate removal effect
            removal_effect = 1 - (new_cvr / base_cvr) if base_cvr > 0 else 0
            removal_effects[channel] = removal_effect

            print(f"{channel}: 移除效应 = {removal_effect:.4f}")

        self.removal_effects = removal_effects
        return removal_effects

    def calculate_attribution_weights(self):
        """
        Calculate attribution weights based on removal effects

        Returns:
            dict: Attribution weights for each channel
        """
        print("\n=== 计算马尔可夫归因权重 ===")

        if not self.removal_effects:
            print("错误: 需要先计算移除效应")
            return {}

        # Normalize removal effects to get attribution weights
        total_effect = sum(self.removal_effects.values())

        if total_effect == 0:
            print("警告: 总移除效应为0，使用均匀权重")
            channels = list(self.removal_effects.keys())
            attribution_weights = {channel: 1.0/len(channels) for channel in channels}
        else:
            attribution_weights = {
                channel: effect / total_effect
                for channel, effect in self.removal_effects.items()
            }

        # Sort by attribution weight
        sorted_weights = dict(sorted(attribution_weights.items(),
                                     key=lambda x: x[1], reverse=True))

        print("马尔可夫归因权重:")
        for channel, weight in sorted_weights.items():
            print(f"  {channel}: {weight:.4f} ({weight*100:.1f}%)")

        self.attribution_weights = sorted_weights
        return sorted_weights

    def analyze_channel_transitions(self):
        """
        Analyze channel transition patterns

        Returns:
            dict: Channel transition analysis
        """
        print("\n=== 渠道转换分析 ===")

        if self.transition_matrix is None:
            print("错误: 需要先构建转移矩阵")
            return {}

        channels = [state for state in self.transition_matrix.columns
                   if state not in ['开始', '未转化', '成功转化']]

        transition_analysis = {}

        for channel in channels:
            # Outgoing transitions
            outgoing = self.transition_matrix.loc[channel].drop(
                index=['开始', '未转化', '成功转化', channel], errors='ignore'
            )
            outgoing_transitions = outgoing[outgoing > 0].sort_values(ascending=False)

            # Incoming transitions
            incoming = self.transition_matrix[channel].drop(
                index=['开始', '未转化', '成功转化', channel], errors='ignore'
            )
            incoming_transitions = incoming[incoming > 0].sort_values(ascending=False)

            transition_analysis[channel] = {
                'outgoing_transitions': outgoing_transitions.to_dict(),
                'incoming_transitions': incoming_transitions.to_dict(),
                'total_outgoing_prob': outgoing_transitions.sum(),
                'total_incoming_prob': incoming_transitions.sum()
            }

        # Find most common transitions
        all_transitions = []
        for channel in channels:
            outgoing = transition_analysis[channel]['outgoing_transitions']
            for target, prob in outgoing.items():
                all_transitions.append({
                    'from': channel,
                    'to': target,
                    'probability': prob
                })

        all_transitions.sort(key=lambda x: x['probability'], reverse=True)

        print("最常见的渠道转换:")
        for i, transition in enumerate(all_transitions[:10]):
            print(f"  {i+1}. {transition['from']} → {transition['to']}: {transition['probability']:.4f}")

        return {
            'channel_transitions': transition_analysis,
            'top_transitions': all_transitions[:10]
        }

    def build_channel_graph(self):
        """
        Build network graph of channel transitions

        Returns:
            networkx.DiGraph: Channel transition graph
        """
        print("\n=== 构建渠道转换图 ===")

        if self.transition_matrix is None:
            print("错误: 需要先构建转移矩阵")
            return None

        channels = [state for state in self.transition_matrix.columns
                   if state not in ['开始', '未转化', '成功转化']]

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for channel in channels:
            G.add_node(channel)

        # Add edges based on transition probabilities
        for channel in channels:
            outgoing = self.transition_matrix.loc[channel].drop(
                index=['开始', '未转化', '成功转化', channel], errors='ignore'
            )
            for target, prob in outgoing.items():
                if prob > 0.01:  # Only include significant transitions
                    G.add_edge(channel, target, weight=prob)

        # Calculate network metrics
        network_metrics = {}
        for node in G.nodes():
            network_metrics[node] = {
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
                'clustering': nx.clustering(G, node),
                'betweenness': nx.betweenness_centrality(G).get(node, 0)
            }

        print(f"渠道转换图构建完成:")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")
        print(f"  强连通分量: {nx.number_strongly_connected_components(G)}")

        self.channel_graph = G
        return G

    def simulate_attribution_scenarios(self, scenarios):
        """
        Simulate different attribution scenarios

        Args:
            scenarios (dict): Dictionary of scenario configurations

        Returns:
            dict: Scenario simulation results
        """
        print("\n=== 模拟归因场景 ===")

        if not self.transition_matrix:
            print("错误: 需要先构建转移矩阵")
            return {}

        results = {}
        base_matrix = self.transition_matrix.copy()

        for scenario_name, config in scenarios.items():
            print(f"\n模拟场景: {scenario_name}")

            # Modify transition matrix based on scenario
            modified_matrix = base_matrix.copy()

            for channel, modifications in config.items():
                if channel in modified_matrix.index:
                    for target_channel, new_prob in modifications.items():
                        if target_channel in modified_matrix.columns:
                            modified_matrix.at[channel, target_channel] = new_prob

            # Calculate new conversion rate
            try:
                transient_states = [state for state in modified_matrix.index
                                  if state not in ['未转化', '成功转化']]
                absorbing_states = ['未转化', '成功转化']

                if len(transient_states) > 0:
                    Q = modified_matrix.loc[transient_states, transient_states].values
                    R = modified_matrix.loc[transient_states, absorbing_states].values

                    I = np.identity(len(Q))
                    N = inv(I - Q)
                    B = np.dot(N, R)

                    start_index = transient_states.index('开始') if '开始' in transient_states else 0
                    success_index = absorbing_states.index('成功转化')
                    simulated_cvr = B[start_index, success_index]
                else:
                    simulated_cvr = 0.0

            except Exception as e:
                print(f"场景计算失败: {str(e)}")
                simulated_cvr = 0.0

            results[scenario_name] = {
                'conversion_rate': simulated_cvr,
                'configuration': config
            }

        return results

    def generate_markov_insights(self):
        """
        Generate insights from Markov chain analysis

        Returns:
            dict: Markov chain analysis insights
        """
        print("\n=== 生成马尔可夫链洞察 ===")

        insights = {}

        if self.attribution_weights:
            # Identify dominant channels
            sorted_channels = sorted(self.attribution_weights.items(),
                                    key=lambda x: x[1], reverse=True)

            insights['dominant_channels'] = sorted_channels[:3]
            insights['underperforming_channels'] = sorted_channels[-3:]

        if self.transition_matrix is not None:
            # Analyze conversion bottlenecks
            channels = [state for state in self.transition_matrix.columns
                       if state not in ['开始', '未转化', '成功转化']]

            bottlenecks = []
            for channel in channels:
                direct_to_conversion = self.transition_matrix.at[channel, '成功转化']
                if direct_to_conversion > 0:
                    bottlenecks.append((channel, direct_to_conversion))

            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            insights['conversion_bottlenecks'] = bottlenecks

        if self.channel_graph is not None:
            # Network insights
            centrality = nx.betweenness_centrality(self.channel_graph)
            insights['most_influential_channels'] = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[:3]

        print("马尔可夫链分析洞察:")
        if 'dominant_channels' in insights:
            print("  主导渠道:")
            for channel, weight in insights['dominant_channels']:
                print(f"    {channel}: {weight:.4f}")

        return insights

    def run_complete_markov_analysis(self, paths_df):
        """
        Run complete Markov chain attribution analysis

        Args:
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            dict: Complete Markov chain analysis results
        """
        print("🔗 开始马尔可夫链归因分析")
        print("=" * 50)

        # Calculate base conversion rate
        base_conversion_rate = paths_df['converted'].mean()

        # 1. Build transition matrix
        transition_matrix = self.build_transition_matrix(paths_df)

        # 2. Calculate removal effects
        removal_effects = self.calculate_removal_effects(paths_df, base_conversion_rate)

        # 3. Calculate attribution weights
        attribution_weights = self.calculate_attribution_weights()

        # 4. Analyze channel transitions
        transition_analysis = self.analyze_channel_transitions()

        # 5. Build channel graph
        channel_graph = self.build_channel_graph()

        # 6. Generate insights
        insights = self.generate_markov_insights()

        results = {
            'transition_matrix': transition_matrix,
            'removal_effects': removal_effects,
            'attribution_weights': attribution_weights,
            'transition_analysis': transition_analysis,
            'channel_graph': channel_graph,
            'insights': insights,
            'base_conversion_rate': base_conversion_rate
        }

        print(f"\n✅ 马可夫链分析完成！")
        print(f"基础转化率: {base_conversion_rate:.2%}")
        print(f"识别了 {len(attribution_weights)} 个渠道的归因权重")
        print(f"分析了 {len(transition_analysis['channel_transitions'])} 个渠道的转换模式")

        return results

def main():
    """Example usage of Markov chain attributor"""
    attributor = MarkovChainAttributor()

    # Create sample data for demonstration
    sample_paths = [
        ['开始', '付费搜索', '社交媒体', '成功转化'],
        ['开始', '社交媒体', '付费搜索', '成功转化'],
        ['开始', '邮件营销', '社交媒体', '付费搜索', '成功转化'],
        ['开始', '付费搜索', '未转化'],
        ['开始', '社交媒体', '未转化'],
        ['开始', '邮件营销', '未转化']
    ]

    paths_df = pd.DataFrame({
        'user_id': range(len(sample_paths)),
        'path': sample_paths,
        'converted': [1, 1, 1, 0, 0, 0],
        'conversion_value': [100, 150, 80, 0, 0, 0]
    })

    results = attributor.run_complete_markov_analysis(paths_df)

    print(f"\n马尔可夫归因权重:")
    for channel, weight in results['attribution_weights'].items():
        print(f"  {channel}: {weight:.4f}")

if __name__ == "__main__":
    main()