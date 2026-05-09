#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Attribution Analysis Engine
Comprehensive marketing attribution analysis with multiple attribution models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AttributionAnalyzer:
    """Core attribution analysis engine with multiple attribution models"""

    def __init__(self):
        """Initialize the attribution analyzer"""
        self.attribution_models = {}
        self.attribution_results = {}
        self.channel_performance = {}
        self.customer_paths = None

    def load_and_validate_data(self, file_path, **kwargs):
        """
        Load and validate marketing attribution data

        Args:
            file_path (str): Path to the CSV file
            **kwargs: Additional parameters for pd.read_csv()

        Returns:
            pd.DataFrame: Validated attribution data
        """
        print("=== 数据加载与验证 ===")

        # Load data with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8', **kwargs)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk', **kwargs)
            except:
                df = pd.read_csv(file_path, encoding='latin-1', **kwargs)

        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # Validate required columns
        required_columns = ['user_id', 'timestamp', 'channel', 'conversion_status']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            # Try to find similar column names
            column_mapping = {}
            for required_col in missing_columns:
                possible_matches = [col for col in df.columns
                                  if required_col.lower() in col.lower() or
                                  col.lower() in required_col.lower()]
                if possible_matches:
                    column_mapping[possible_matches[0]] = required_col
                    print(f"自动识别列: {possible_matches[0]} -> {required_col}")

            # Rename columns
            df = df.rename(columns=column_mapping)
            missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")

        # Data validation
        print(f"\n数据质量检查:")
        print(f"- 唯一用户数: {df['user_id'].nunique():,}")
        print(f"- 唯一渠道数: {df['channel'].nunique()}")
        print(f"- 总触点数: {len(df):,}")

        # Check conversion rates
        conversion_count = df['conversion_status'].sum()
        conversion_rate = conversion_count / len(df) if len(df) > 0 else 0
        print(f"- 转化触点数: {conversion_count:,}")
        print(f"- 转化率: {conversion_rate:.2%}")

        # Convert timestamp to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"- 时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        except:
            print("- 警告: 时间戳格式可能不标准")

        # Add cost and value columns if not present
        if 'cost' not in df.columns:
            df['cost'] = 1.0  # Default cost
        if 'conversion_value' not in df.columns:
            df['conversion_value'] = np.where(df['conversion_status'] == 1, 1.0, 0.0)

        return df

    def build_customer_paths(self, df):
        """
        Build customer journey paths from touchpoint data

        Args:
            df (pd.DataFrame): Validated attribution data

        Returns:
            pd.DataFrame: Customer journey paths
        """
        print("\n=== 构建客户路径 ===")

        # Sort by user and timestamp
        df_sorted = df.sort_values(['user_id', 'timestamp'], ascending=[True, True])

        # Add visit sequence number
        df_sorted['visit_sequence'] = df_sorted.groupby('user_id').cumcount() + 1

        # Build paths for each user
        user_paths = []
        user_conversions = {}

        for user_id, user_data in df_sorted.groupby('user_id'):
            # Get unique channels in chronological order
            channels = user_data['channel'].unique().tolist()

            # Determine if user converted
            converted = user_data['conversion_status'].max()
            conversion_value = user_data.loc[user_data['conversion_status'] == 1, 'conversion_value'].sum() if converted else 0

            # Get total cost for this user
            total_cost = user_data['cost'].sum()

            # Build path with start and end markers
            if converted:
                path = ['开始'] + channels + ['成功转化']
            else:
                path = ['开始'] + channels + ['未转化']

            user_paths.append({
                'user_id': user_id,
                'path': path,
                'path_length': len(path),
                'converted': converted,
                'conversion_value': conversion_value,
                'total_cost': total_cost,
                'touchpoints': len(channels)
            })

            user_conversions[user_id] = converted

        paths_df = pd.DataFrame(user_paths)
        self.customer_paths = paths_df

        print(f"构建了 {len(paths_df)} 条客户路径")
        print(f"转化用户数: {paths_df['converted'].sum():,}")
        print(f"转化率: {paths_df['converted'].mean():.2%}")
        print(f"平均路径长度: {paths_df['path_length'].mean():.1f}")

        return paths_df

    def first_touch_attribution(self, paths_df):
        """
        First-touch attribution model

        Args:
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            dict: Channel attribution weights
        """
        print("\n=== 首次接触归因分析 ===")

        attribution = defaultdict(float)
        total_conversions = 0
        total_value = 0.0

        for _, row in paths_df.iterrows():
            if row['converted'] and len(row['path']) > 2:
                # Get the first real channel (after '开始')
                first_channel = row['path'][1]
                attribution[first_channel] += row['conversion_value']
                total_conversions += 1
                total_value += row['conversion_value']

        # Normalize
        for channel in attribution:
            attribution[channel] /= total_value if total_value > 0 else 1

        print(f"首次接触归因完成，总转化: {total_conversions:,}, 总价值: {total_value:,.2f}")

        self.attribution_models['first_touch'] = dict(attribution)
        return dict(attribution)

    def last_touch_attribution(self, paths_df):
        """
        Last-touch attribution model

        Args:
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            dict: Channel attribution weights
        """
        print("\n=== 最后接触归因分析 ===")

        attribution = defaultdict(float)
        total_conversions = 0
        total_value = 0.0

        for _, row in paths_df.iterrows():
            if row['converted'] and len(row['path']) > 2:
                # Get the last real channel (before '成功转化')
                last_channel = row['path'][-2]
                attribution[last_channel] += row['conversion_value']
                total_conversions += 1
                total_value += row['conversion_value']

        # Normalize
        for channel in attribution:
            attribution[channel] /= total_value if total_value > 0 else 1

        print(f"最后接触归因完成，总转化: {total_conversions:,}, 总价值: {total_value:,.2f}")

        self.attribution_models['last_touch'] = dict(attribution)
        return dict(attribution)

    def linear_attribution(self, paths_df):
        """
        Linear attribution model (equal distribution)

        Args:
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            dict: Channel attribution weights
        """
        print("\n=== 线性归因分析 ===")

        attribution = defaultdict(float)
        total_conversions = 0
        total_value = 0.0

        for _, row in paths_df.iterrows():
            if row['converted'] and len(row['path']) > 2:
                # Get all real channels (exclude '开始' and '成功转化')
                channels = [ch for ch in row['path'] if ch not in ['开始', '成功转化', '未转化']]
                if channels:
                    weight = row['conversion_value'] / len(channels)
                    for channel in channels:
                        attribution[channel] += weight
                    total_conversions += 1
                    total_value += row['conversion_value']

        # Normalize
        for channel in attribution:
            attribution[channel] /= total_value if total_value > 0 else 1

        print(f"线性归因完成，总转化: {total_conversions:,}, 总价值: {total_value:,.2f}")

        self.attribution_models['linear'] = dict(attribution)
        return dict(attribution)

    def time_decay_attribution(self, paths_df, decay_factor=0.5):
        """
        Time-decay attribution model

        Args:
            paths_df (pd.DataFrame): Customer journey paths
            decay_factor (float): Decay factor for time weighting

        Returns:
            dict: Channel attribution weights
        """
        print(f"\n=== 时间衰减归因分析 (衰减因子: {decay_factor}) ===")

        attribution = defaultdict(float)
        total_conversions = 0
        total_value = 0.0

        for _, row in paths_df.iterrows():
            if row['converted'] and len(row['path']) > 2:
                # Get all real channels
                channels = [ch for ch in row['path'] if ch not in ['开始', '成功转化', '未转化']]
                if channels:
                    # Calculate weights based on position (last channel gets highest weight)
                    positions = range(len(channels))
                    weights = [decay_factor ** (len(channels) - 1 - pos) for pos in positions]
                    total_weight = sum(weights)

                    # Normalize weights
                    weights = [w / total_weight for w in weights]

                    # Distribute conversion value
                    for channel, weight in zip(channels, weights):
                        attribution[channel] += weight * row['conversion_value']

                    total_conversions += 1
                    total_value += row['conversion_value']

        # Normalize
        for channel in attribution:
            attribution[channel] /= total_value if total_value > 0 else 1

        print(f"时间衰减归因完成，总转化: {total_conversions:,}, 总价值: {total_value:,.2f}")

        self.attribution_models['time_decay'] = dict(attribution)
        return dict(attribution)

    def position_based_attribution(self, paths_df, first_weight=0.4, last_weight=0.4):
        """
        Position-based attribution model

        Args:
            paths_df (pd.DataFrame): Customer journey paths
            first_weight (float): Weight for first touch
            last_weight (float): Weight for last touch

        Returns:
            dict: Channel attribution weights
        """
        print(f"\n=== 位置归因分析 (首次: {first_weight}, 最后: {last_weight}) ===")

        attribution = defaultdict(float)
        total_conversions = 0
        total_value = 0.0

        for _, row in paths_df.iterrows():
            if row['converted'] and len(row['path']) > 2:
                # Get all real channels
                channels = [ch for ch in row['path'] if ch not in ['开始', '成功转化', '未转化']]
                if channels:
                    middle_weight = 1 - first_weight - last_weight
                    num_middle = max(0, len(channels) - 2)

                    weights = []
                    for i, channel in enumerate(channels):
                        if i == 0:  # First channel
                            weights.append(first_weight)
                        elif i == len(channels) - 1:  # Last channel
                            weights.append(last_weight)
                        else:  # Middle channels
                            weights.append(middle_weight / num_middle if num_middle > 0 else 0)

                    # Normalize weights
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]

                        # Distribute conversion value
                        for channel, weight in zip(channels, weights):
                            attribution[channel] += weight * row['conversion_value']

                    total_conversions += 1
                    total_value += row['conversion_value']

        # Normalize
        for channel in attribution:
            attribution[channel] /= total_value if total_value > 0 else 1

        print(f"位置归因完成，总转化: {total_conversions:,}, 总价值: {total_value:,.2f}")

        self.attribution_models['position_based'] = dict(attribution)
        return dict(attribution)

    def calculate_channel_performance(self, df, paths_df):
        """
        Calculate comprehensive channel performance metrics

        Args:
            df (pd.DataFrame): Original touchpoint data
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            pd.DataFrame: Channel performance metrics
        """
        print("\n=== 渠道性能分析 ===")

        # Basic channel metrics
        channel_metrics = []

        for channel in df['channel'].unique():
            # Get all users who touched this channel
            channel_users = set(df[df['channel'] == channel]['user_id'])

            # Get conversions from this channel
            converted_users = set(paths_df[paths_df['converted']]['user_id'])
            channel_converted_users = channel_users.intersection(converted_users)

            # Calculate metrics
            total_touchpoints = len(df[df['channel'] == channel])
            total_cost = df[df['channel'] == channel]['cost'].sum()
            conversion_value = paths_df[paths_df['user_id'].isin(channel_converted_users)]['conversion_value'].sum()

            channel_metrics.append({
                'channel': channel,
                'total_touchpoints': total_touchpoints,
                'unique_users': len(channel_users),
                'conversions': len(channel_converted_users),
                'conversion_rate': len(channel_converted_users) / len(channel_users) if len(channel_users) > 0 else 0,
                'total_cost': total_cost,
                'conversion_value': conversion_value,
                'cpa': total_cost / len(channel_converted_users) if len(channel_converted_users) > 0 else float('inf'),
                'roi': (conversion_value - total_cost) / total_cost if total_cost > 0 else 0,
                'avg_touchpoints_per_user': total_touchpoints / len(channel_users) if len(channel_users) > 0 else 0
            })

        performance_df = pd.DataFrame(channel_metrics)
        performance_df = performance_df.sort_values('conversion_value', ascending=False)

        self.channel_performance = performance_df

        print("渠道性能排名:")
        for idx, row in performance_df.iterrows():
            print(f"{row['channel']}: 转化率={row['conversion_rate']:.2%}, CPA={row['cpa']:.2f}, ROI={row['roi']:.2f}")

        return performance_df

    def compare_attribution_models(self, paths_df):
        """
        Run and compare multiple attribution models

        Args:
            paths_df (pd.DataFrame): Customer journey paths

        Returns:
            pd.DataFrame: Comparison of all attribution models
        """
        print("\n=== 多模型归因比较 ===")

        # Run all attribution models
        self.first_touch_attribution(paths_df)
        self.last_touch_attribution(paths_df)
        self.linear_attribution(paths_df)
        self.time_decay_attribution(paths_df)
        self.position_based_attribution(paths_df)

        # Create comparison table
        all_channels = set()
        for model_result in self.attribution_models.values():
            all_channels.update(model_result.keys())

        comparison_data = []
        for channel in all_channels:
            row = {'channel': channel}
            for model_name, attribution in self.attribution_models.items():
                row[model_name] = attribution.get(channel, 0.0)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.fillna(0.0)

        # Calculate statistics
        print("\n各模型归因权重比较:")
        print(comparison_df.set_index('channel'))

        # Calculate average attribution
        model_columns = list(self.attribution_models.keys())
        comparison_df['average_attribution'] = comparison_df[model_columns].mean(axis=1)
        comparison_df['std_deviation'] = comparison_df[model_columns].std(axis=1)

        print(f"\n归因权重标准差 (模型间差异):")
        print(comparison_df[['channel', 'average_attribution', 'std_deviation']].sort_values('std_deviation', ascending=False))

        self.attribution_results = comparison_df
        return comparison_df

    def generate_attribution_summary(self, df):
        """
        Generate comprehensive attribution analysis summary

        Args:
            df (pd.DataFrame): Original touchpoint data

        Returns:
            dict: Comprehensive attribution summary
        """
        print("\n=== 生成归因分析摘要 ===")

        if self.customer_paths is None:
            paths_df = self.build_customer_paths(df)
        else:
            paths_df = self.customer_paths

        if not self.attribution_results.empty:
            performance_df = self.calculate_channel_performance(df, paths_df)
        else:
            performance_df = self.compare_attribution_models(paths_df)
            performance_df = self.calculate_channel_performance(df, paths_df)

        # Generate summary statistics
        total_users = df['user_id'].nunique()
        total_conversions = paths_df['converted'].sum()
        overall_conversion_rate = total_conversions / total_users if total_users > 0 else 0
        total_conversion_value = paths_df[paths_df['converted']]['conversion_value'].sum()
        total_cost = df['cost'].sum()

        summary = {
            'analysis_metadata': {
                'total_users': total_users,
                'total_conversions': total_conversions,
                'overall_conversion_rate': overall_conversion_rate,
                'total_conversion_value': total_conversion_value,
                'total_cost': total_cost,
                'overall_roi': (total_conversion_value - total_cost) / total_cost if total_cost > 0 else 0,
                'unique_channels': df['channel'].nunique(),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'attribution_models': self.attribution_models,
            'channel_performance': performance_df.to_dict('records'),
            'model_comparison': self.attribution_results.to_dict('records') if hasattr(self.attribution_results, 'to_dict') else {},
            'recommended_actions': self._generate_recommendations(performance_df)
        }

        print("\n=== 分析摘要 ===")
        print(f"总用户数: {total_users:,}")
        print(f"转化用户数: {total_conversions:,}")
        print(f"整体转化率: {overall_conversion_rate:.2%}")
        print(f"总转化价值: {total_conversion_value:,.2f}")
        print(f"总营销成本: {total_cost:,.2f}")
        print(f"整体ROI: {(total_conversion_value - total_cost) / total_cost * 100:.1f}%")

        return summary

    def _generate_recommendations(self, performance_df):
        """
        Generate actionable recommendations based on channel performance

        Args:
            performance_df (pd.DataFrame): Channel performance metrics

        Returns:
            list: Actionable recommendations
        """
        recommendations = []

        # High ROI channels (increase investment)
        high_roi_channels = performance_df[performance_df['roi'] > 2.0]
        if not high_roi_channels.empty:
            for _, row in high_roi_channels.iterrows():
                recommendations.append({
                    'type': 'increase_investment',
                    'channel': row['channel'],
                    'reason': f"高ROI ({row['roi']:.1f}x), 考虑增加投资",
                    'priority': 'high' if row['roi'] > 3.0 else 'medium'
                })

        # Low CPA channels (efficient acquisition)
        low_cpa_channels = performance_df[performance_df['cpa'] < performance_df['cpa'].median()]
        if not low_cpa_channels.empty:
            for _, row in low_cpa_channels.iterrows():
                recommendations.append({
                    'type': 'efficient_acquisition',
                    'channel': row['channel'],
                    'reason': f"低CPA ({row['cpa']:.2f}), 获客效率高",
                    'priority': 'medium'
                })

        # High conversion rate channels (optimize funnel)
        high_conv_channels = performance_df[performance_df['conversion_rate'] > performance_df['conversion_rate'].median()]
        if not high_conv_channels.empty:
            for _, row in high_conv_channels.iterrows():
                recommendations.append({
                    'type': 'optimize_funnel',
                    'channel': row['channel'],
                    'reason': f"高转化率 ({row['conversion_rate']:.2%}), 优化转化漏斗",
                    'priority': 'medium'
                })

        # Negative ROI channels (reduce investment)
        negative_roi_channels = performance_df[performance_df['roi'] < 0]
        if not negative_roi_channels.empty:
            for _, row in negative_roi_channels.iterrows():
                recommendations.append({
                    'type': 'reduce_investment',
                    'channel': row['channel'],
                    'reason': f"负ROI ({row['roi']:.1f}x), 考虑减少或停止投资",
                    'priority': 'high'
                })

        # Underperforming channels (investigate issues)
        low_conv_channels = performance_df[performance_df['conversion_rate'] < 0.01]
        if not low_conv_channels.empty:
            for _, row in low_conv_channels.iterrows():
                recommendations.append({
                    'type': 'investigate_issues',
                    'channel': row['channel'],
                    'reason': f"转化率低 ({row['conversion_rate']:.2%}), 调查渠道表现问题",
                    'priority': 'low'
                })

        return recommendations

    def run_complete_analysis(self, file_path, **kwargs):
        """
        Run complete attribution analysis pipeline

        Args:
            file_path (str): Path to attribution data file
            **kwargs: Additional parameters for data loading

        Returns:
            dict: Complete attribution analysis results
        """
        print("🚀 开始完整归因分析")
        print("=" * 50)

        # 1. Load and validate data
        df = self.load_and_validate_data(file_path, **kwargs)

        # 2. Build customer paths
        paths_df = self.build_customer_paths(df)

        # 3. Calculate channel performance
        performance_df = self.calculate_channel_performance(df, paths_df)

        # 4. Compare attribution models
        comparison_df = self.compare_attribution_models(paths_df)

        # 5. Generate comprehensive summary
        summary = self.generate_attribution_summary(df)

        print(f"\n✅ 归因分析完成！")
        print(f"分析了 {df['user_id'].nunique():,} 个用户的 {len(df):,} 个触点")
        print(f"识别了 {df['channel'].nunique()} 个营销渠道")
        print(f"运行了 {len(self.attribution_models)} 种归因模型")

        return {
            'raw_data': df,
            'customer_paths': paths_df,
            'channel_performance': performance_df,
            'attribution_comparison': comparison_df,
            'attribution_models': self.attribution_models,
            'summary': summary
        }

def main():
    """Example usage of attribution analyzer"""
    analyzer = AttributionAnalyzer()

    # Example with sample data
    file_path = "渠道转化.csv"

    if pd.io.common.file_exists(file_path):
        results = analyzer.run_complete_analysis(file_path)

        print(f"\n归因权重对比:")
        for model, attribution in results['attribution_models'].items():
            print(f"{model}: {attribution}")

    else:
        print(f"数据文件未找到: {file_path}")
        print("请提供正确的营销触点数据文件")

if __name__ == "__main__":
    main()