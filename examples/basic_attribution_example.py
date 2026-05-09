#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础归因分析示例
Basic Attribution Analysis Example

演示如何使用归因分析技能进行基础的营销渠道归因分析
"""

import pandas as pd
import sys
import os

# Windows环境下控制台编码设置
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# 添加父目录到路径以导入技能模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_attribution import AttributionAnalyzer
from attribution_visualizer import AttributionVisualizer

def main():
    """基础归因分析示例主函数"""

    print("🎯 基础归因分析示例")
    print("=" * 60)

    # 1. 初始化分析器
    analyzer = AttributionAnalyzer()

    # 2. 加载示例数据
    data_path = os.path.join(os.path.dirname(__file__), 'sample_channel_data.csv')
    print(f"📊 加载数据: {data_path}")

    df = analyzer.load_and_validate_data(data_path)
    if df is None:
        print("❌ 数据加载失败，请检查文件路径")
        return

    print(f"✅ 数据加载成功: {len(df)} 条记录")
    print(f"📊 数据概览:")
    print(f"   - 用户数量: {df['user_id'].nunique()}")
    print(f"   - 渠道数量: {df['channel'].nunique()}")
    print(f"   - 转化数量: {df['conversion_status'].sum()}")
    print(f"   - 总转化价值: {df['conversion_value'].sum():,.2f}")
    print(f"   - 总成本: {df['cost'].sum():,.2f}")

    # 3. 构建客户路径
    print("\n🛤️ 构建客户路径...")
    paths_df = analyzer.build_customer_paths(df)

    if paths_df.empty:
        print("❌ 客户路径构建失败")
        return

    print(f"✅ 客户路径构建完成: {len(paths_df)} 条路径")
    print(f"   - 转化路径: {paths_df['converted'].sum()}")
    print(f"   - 平均路径长度: {paths_df['path_length'].mean():.2f}")

    # 4. 运行基础归因分析
    print("\n📈 运行基础归因分析...")
    attribution_results = analyzer.run_basic_attribution_analysis(paths_df)

    if not attribution_results:
        print("❌ 归因分析失败")
        return

    # 5. 显示归因结果
    print("\n📊 归因分析结果:")
    print("-" * 40)

    for model_name, weights in attribution_results.items():
        print(f"\n{model_name}:")
        # 按权重排序显示前5个渠道
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for channel, weight in sorted_weights:
            print(f"  {channel:<12}: {weight:.4f} ({weight*100:.1f}%)")

    # 6. 计算ROI分析
    print("\n💰 ROI分析:")
    print("-" * 40)

    # 计算各渠道的成本和收益
    channel_costs = df.groupby('channel')['cost'].sum()
    channel_revenues = df[df['conversion_status'] == 1].groupby('channel')['conversion_value'].sum()

    roi_analysis = []
    for channel in df['channel'].unique():
        cost = channel_costs.get(channel, 0)
        revenue = channel_revenues.get(channel, 0)
        roi = (revenue - cost) / cost * 100 if cost > 0 else 0
        roi_analysis.append({
            'channel': channel,
            'cost': cost,
            'revenue': revenue,
            'roi': roi
        })

    roi_df = pd.DataFrame(roi_analysis)
    roi_df = roi_df.sort_values('roi', ascending=False)

    for _, row in roi_df.iterrows():
        print(f"{row['channel']:<12}: 成本={row['cost']:6.0f}, 收益={row['revenue']:7.0f}, ROI={row['roi']:6.1f}%")

    # 7. 生成归因对比表
    print("\n📋 归因模型对比:")
    print("-" * 60)

    # 创建所有渠道的归因权重对比表
    all_channels = set()
    for weights in attribution_results.values():
        all_channels.update(weights.keys())

    print(f"{'渠道':<15}", end="")
    for model_name in attribution_results.keys():
        print(f"{model_name[:8]:<10}", end="")
    print()
    print("-" * 60)

    for channel in sorted(all_channels):
        print(f"{channel:<15}", end="")
        for model_name, weights in attribution_results.items():
            weight = weights.get(channel, 0)
            print(f"{weight*100:>6.1f}%{' '*4}", end="")
        print()

    # 8. 可视化分析
    print("\n📊 生成归因可视化...")
    try:
        visualizer = AttributionVisualizer()

        # 保存分析结果
        analyzer.save_attribution_results(attribution_results, paths_df, 'examples/')

        # 创建可视化
        output_path = visualizer.create_attribution_dashboard(attribution_results)
        print(f"✅ 可视化已保存至: {output_path}")

    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")
        print("提示: 请确保已安装所需的可视化依赖包")

    # 9. 业务洞察和建议
    print("\n💡 业务洞察和建议:")
    print("-" * 40)

    # 基于最后接触归因（最常用）给出建议
    last_touch_weights = attribution_results.get('最后接触归因', {})

    # 找出表现最好和最差的渠道
    if last_touch_weights:
        sorted_channels = sorted(last_touch_weights.items(), key=lambda x: x[1], reverse=True)

        top_channels = sorted_channels[:3]
        bottom_channels = sorted_channels[-3:]

        print("🏆 表现最佳的渠道 (按最后接触归因):")
        for channel, weight in top_channels:
            cost = channel_costs.get(channel, 0)
            revenue = channel_revenues.get(channel, 0)
            roi = (revenue - cost) / cost * 100 if cost > 0 else 0
            print(f"  • {channel}: 权重={weight*100:.1f}%, ROI={roi:.1f}%")

        print("\n⚠️ 需要优化的渠道:")
        for channel, weight in bottom_channels:
            if weight > 0:  # 忽略权重为0的渠道
                cost = channel_costs.get(channel, 0)
                revenue = channel_revenues.get(channel, 0)
                roi = (revenue - cost) / cost * 100 if cost > 0 else 0
                print(f"  • {channel}: 权重={weight*100:.1f}%, ROI={roi:.1f}%")

        # 优化建议
        print("\n📈 优化建议:")
        print("  1. 考虑增加对高ROI、高权重渠道的投入")
        print("  2. 分析低权重渠道的原因：定位问题、创意问题还是受众问题")
        print("  3. 测试不同归因模型以获得更全面的视角")
        print("  4. 定期监控和调整渠道策略")

    print(f"\n✅ 基础归因分析完成！")
    print(f"📁 结果文件已保存至 examples/ 目录")
    print(f"📊 可视化图表已生成")

if __name__ == "__main__":
    main()