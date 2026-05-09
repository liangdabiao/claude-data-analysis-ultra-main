#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级归因分析示例
Advanced Attribution Analysis Example

演示如何使用马尔可夫链和Shapley值进行高级归因分析
"""

import pandas as pd
import sys
import os
import time

# 添加父目录到路径以导入技能模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_attribution import AttributionAnalyzer
from markov_chains import MarkovChainAttributor
from shapley_values import ShapleyValueAttributor
from attribution_visualizer import AttributionVisualizer

def main():
    """高级归因分析示例主函数"""

    print("🎮 高级归因分析示例")
    print("=" * 60)

    # 1. 加载和预处理数据
    print("📊 第1步: 数据加载和预处理")
    print("-" * 40)

    analyzer = AttributionAnalyzer()
    data_path = os.path.join(os.path.dirname(__file__), 'sample_channel_data.csv')

    df = analyzer.load_and_validate_data(data_path)
    if df is None:
        print("❌ 数据加载失败")
        return

    print(f"✅ 数据加载成功: {len(df)} 条记录")

    # 构建客户路径
    paths_df = analyzer.build_customer_paths(df)
    print(f"✅ 客户路径构建完成: {len(paths_df)} 条路径")

    # 2. 马尔可夫链归因分析
    print(f"\n🔗 第2步: 马尔可夫链归因分析")
    print("-" * 40)

    start_time = time.time()

    try:
        markov_attributor = MarkovChainAttributor()
        print("🔄 构建马尔可夫链模型...")

        # 构建转移矩阵
        transition_matrix = markov_attributor.build_transition_matrix(paths_df)
        print("✅ 转移矩阵构建完成")

        # 计算归因权重
        markov_weights = markov_attributor.calculate_attribution_weights()
        print("✅ 马尔可夫链归因权重计算完成")

        # 分析渠道转换
        transition_analysis = markov_attributor.analyze_channel_transitions(transition_matrix)
        print("✅ 渠道转换分析完成")

        # 构建渠道网络图
        channel_graph = markov_attributor.build_channel_graph(transition_matrix)
        print("✅ 渠道网络图构建完成")

        markov_time = time.time() - start_time
        print(f"⏱️ 马尔可夫链分析耗时: {markov_time:.2f} 秒")

        # 显示马尔可夫链结果
        print(f"\n📊 马尔可夫链归因权重:")
        sorted_markov = sorted(markov_weights.items(), key=lambda x: x[1], reverse=True)
        for channel, weight in sorted_markov:
            print(f"  {channel:<15}: {weight:.4f} ({weight*100:.1f}%)")

        # 显示关键转换路径
        print(f"\n🔀 关键渠道转换路径 (前5个):")
        for path in transition_analysis['top_paths'][:5]:
            print(f"  {' → '.join(path['path'])}: 概率={path['probability']:.4f}")

    except Exception as e:
        print(f"❌ 马尔可夫链分析失败: {e}")
        markov_weights = {}
        transition_analysis = {}
        channel_graph = None

    # 3. Shapley值归因分析
    print(f"\n🎮 第3步: Shapley值归因分析")
    print("-" * 40)

    start_time = time.time()

    try:
        shapley_attributor = ShapleyValueAttributor()
        print("🔄 计算Shapley值...")

        # 运行完整的Shapley值分析
        shapley_results = shapley_attributor.run_complete_shapley_analysis(paths_df)
        print("✅ Shapley值分析完成")

        shapley_time = time.time() - start_time
        print(f"⏱️ Shapley值分析耗时: {shapley_time:.2f} 秒")

        # 显示Shapley值结果
        print(f"\n📊 Shapley值归因权重:")
        shapley_weights = shapley_results['attribution_weights']
        sorted_shapley = sorted(shapley_weights.items(), key=lambda x: x[1], reverse=True)
        for channel, weight in sorted_shapley:
            print(f"  {channel:<15}: {weight:.4f} ({weight*100:.1f}%)")

        # 显示渠道协同效应
        print(f"\n🤝 渠道协同效应分析 (前3个最佳组合):")
        synergy_analysis = shapley_results['channel_synergy']
        for i, (pair_key, synergy) in enumerate(list(synergy_analysis.items())[:3]):
            synergy_type = synergy['synergy_type']
            print(f"  {i+1}. {synergy['channel1']} + {synergy['channel2']}: "
                  f"协同比={synergy['synergy_ratio']:.3f} ({synergy_type})")

        # 显示边际贡献分析
        print(f"\n📈 边际贡献分析:")
        marginal_df = shapley_results['marginal_analysis']
        for _, row in marginal_df.iterrows():
            tier_icon = "🌟" if row['performance_tier'] == 'top_performer' else \
                       "⭐" if row['performance_tier'] == 'strong_performer' else \
                       "✨" if row['performance_tier'] == 'moderate_performer' else "💫"
            print(f"  {tier_icon} {row['channel']:<15}: "
                  f"Shapley值={row['shapley_value']:.6f}, "
                  f"层级={row['performance_tier']}")

        # 显示优化建议
        print(f"\n🎯 渠道优化建议:")
        optimization = shapley_results['optimization']
        recommendations = optimization.get('recommendations', [])

        for i, rec in enumerate(recommendations[:5]):
            action_icon = "📈" if rec['action'] == 'increase' else "📉"
            priority_icon = "🔥" if rec['priority'] == 'high' else "⚡"
            print(f"  {i+1}. {action_icon} {priority_icon} {rec['channel']}: {rec['reason']}")

    except Exception as e:
        print(f"❌ Shapley值分析失败: {e}")
        shapley_results = {}

    # 4. 模型对比分析
    print(f"\n📊 第4步: 归因模型对比分析")
    print("-" * 40)

    # 收集所有模型的结果
    model_results = {}

    # 基础模型
    try:
        basic_results = analyzer.run_basic_attribution_analysis(paths_df)
        model_results.update(basic_results)
    except:
        pass

    # 高级模型
    if 'markov_weights' in locals() and markov_weights:
        model_results['马尔可夫链归因'] = markov_weights

    if 'shapley_weights' in locals() and shapley_weights:
        model_results['Shapley值归因'] = shapley_weights

    # 创建对比表
    if model_results:
        print(f"\n📋 归因模型对比表:")
        print("-" * 80)

        # 获取所有渠道
        all_channels = set()
        for weights in model_results.values():
            all_channels.update(weights.keys())

        # 表头
        print(f"{'渠道':<15}", end="")
        for model_name in model_results.keys():
            print(f"{model_name:<12}", end="")
        print(f"{'平均权重':<10} {'标准差':<10}")
        print("-" * 80)

        # 计算每个渠道的统计信息
        for channel in sorted(all_channels):
            weights = []
            print(f"{channel:<15}", end="")

            for model_name, model_weights in model_results.items():
                weight = model_weights.get(channel, 0)
                weights.append(weight)
                print(f"{weight*100:>6.1f}%{' '*6}", end="")

            avg_weight = sum(weights) / len(weights)
            std_weight = (sum((w - avg_weight)**2 for w in weights) / len(weights))**0.5

            print(f"{avg_weight*100:>6.1f}%{' '*4} {std_weight*100:>6.1f}%")

    # 5. 生成高级可视化
    print(f"\n📊 第5步: 生成高级可视化")
    print("-" * 40)

    try:
        visualizer = AttributionVisualizer()

        # 创建马尔可夫链可视化
        if 'channel_graph' in locals() and channel_graph:
            markov_viz_path = visualizer.create_markov_visualization({
                'transition_matrix': transition_matrix,
                'attribution_weights': markov_weights,
                'channel_graph': channel_graph,
                'transition_analysis': transition_analysis
            })
            print(f"✅ 马尔可夫链可视化已保存: {markov_viz_path}")

        # 创建Shapley值可视化
        if 'shapley_results' in locals() and shapley_results:
            shapley_viz_path = visualizer.create_shapley_visualization(shapley_results)
            print(f"✅ Shapley值可视化已保存: {shapley_viz_path}")

        # 创建综合对比可视化
        if model_results:
            comparison_viz_path = visualizer.create_attribution_dashboard(model_results)
            print(f"✅ 归因模型对比可视化已保存: {comparison_viz_path}")

    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")

    # 6. 总结和业务洞察
    print(f"\n💡 第6步: 总结和业务洞察")
    print("-" * 40)

    print(f"🎯 高级归因分析完成!")
    print(f"   • 分析了 {len(paths_df)} 条客户路径")
    print(f"   • 评估了 {len(model_results)} 种归因模型")
    print(f"   • 识别了 {len(df['channel'].unique())} 个营销渠道")

    if model_results:
        # 找出在不同模型中表现一致的渠道
        print(f"\n🏆 稳定表现渠道 (在所有模型中权重排名前3):")

        # 计算每个渠道的平均排名
        channel_rankings = {}
        for model_name, weights in model_results.items():
            sorted_channels = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for rank, (channel, weight) in enumerate(sorted_channels):
                if channel not in channel_rankings:
                    channel_rankings[channel] = []
                channel_rankings[channel].append(rank + 1)

        # 找出平均排名最好的渠道
        avg_rankings = {
            channel: sum(ranks) / len(ranks)
            for channel, ranks in channel_rankings.items()
        }

        top_channels = sorted(avg_rankings.items(), key=lambda x: x[1])[:3]
        for i, (channel, avg_rank) in enumerate(top_channels):
            print(f"  {i+1}. {channel}: 平均排名 {avg_rank:.1f}")

    print(f"\n📈 分析建议:")
    print(f"  1. 使用多种归因模型进行交叉验证")
    print(f"  2. 关注马尔可夫链识别的关键转换路径")
    print(f"  3. 利用Shapley值分析优化渠道组合")
    print(f"  4. 基于协同效应设计联合营销策略")
    print(f"  5. 定期重新评估归因模型的有效性")

    print(f"\n✅ 高级归因分析示例完成！")
    print(f"📁 结果文件和可视化图表已保存")

if __name__ == "__main__":
    main()