#!/usr/bin/env python3
"""
增长分析完整示例

演示增长模型分析技能的核心功能：
- 营销策略效果评估
- RFM用户分群
- 用户行为分析
- 转化漏斗分析
- 增长策略洞察
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path))

from scripts.growth_analyzer import GrowthModelAnalyzer
from scripts.growth_visualizer import GrowthVisualizer


def create_comprehensive_sample_data():
    """创建全面的样本数据"""
    np.random.seed(42)
    n_users = 2000

    # 创建更真实的用户数据
    data = {
        '用户码': [f'USER_{i:06d}' for i in range(n_users)],
        '裂变类型': np.random.choice(
            ['无裂变页面', '助力砍价', '拼团狂买', '裂变海报', '好友助力'],
            n_users, p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        '城市类型': np.random.choice(
            ['一线城市', '二线城市', '三线城市', '四线城市'],
            n_users, p=[0.2, 0.3, 0.3, 0.2]
        ),
        '设备类型': np.random.choice(['iOS', 'Android'], n_users, p=[0.55, 0.45]),
        '年龄': np.random.randint(18, 65, n_users),
        '性别': np.random.choice(['男', '女'], n_users, p=[0.48, 0.52])
    }

    # 创建行为数据
    df = pd.DataFrame(data)

    # 根据用户特征调整行为
    def calculate_conversion_probability(row):
        base_prob = 0.15

        # 裂变类型影响
        campaign_boost = {
            '无裂变页面': 0.0,
            '助力砍价': 0.12,
            '拼团狂买': 0.15,
            '裂变海报': 0.08,
            '好友助力': 0.20
        }

        # 城市影响
        city_boost = {
            '一线城市': 0.05,
            '二线城市': 0.03,
            '三线城市': 0.01,
            '四线城市': -0.02
        }

        # 年龄影响
        age_boost = -0.001 * (row['年龄'] - 35) ** 2 / 100

        # 设备影响
        device_boost = 0.02 if row['设备类型'] == 'iOS' else 0.0

        prob = base_prob + campaign_boost[row['裂变类型']] + city_boost[row['城市类型']] + age_boost + device_boost
        return max(0, min(1, prob))

    df['转化概率'] = df.apply(calculate_conversion_probability, axis=1)
    df['是否转化'] = np.random.random(len(df)) < df['转化概率']

    # 生成RFM数据
    df['R值'] = np.where(
        df['是否转化'],
        np.random.exponential(7, len(df)).astype(int) + 1,
        np.random.exponential(30, len(df)).astype(int) + 1
    )

    df['F值'] = np.where(
        df['是否转化'],
        np.random.poisson(3, len(df)) + 1,
        np.random.poisson(1, len(df)) + 1
    )

    df['M值'] = np.where(
        df['是否转化'],
        np.random.gamma(2, 50, len(df)),
        np.random.gamma(1, 20, len(df))
    )

    # 生成其他行为数据
    df['曾助力'] = np.where(
        df['裂变类型'].isin(['助力砍价', '好友助力']),
        np.random.random(len(df)) < 0.6,
        np.random.random(len(df)) < 0.15
    )

    df['曾拼团'] = np.where(
        df['裂变类型'] == '拼团狂买',
        np.random.random(len(df)) < 0.7,
        np.random.random(len(df)) < 0.1
    )

    df['曾推荐'] = np.random.random(len(df)) < (df['是否转化'] * 0.5 + 0.05)

    # 生成财务数据
    df['收入'] = np.where(
        df['是否转化'],
        np.random.gamma(2, 80, len(df)),
        np.random.gamma(1, 15, len(df))
    )

    df['成本'] = np.random.uniform(20, 120, len(df))

    return df.drop('转化概率', axis=1)


def main():
    """主函数：演示完整增长分析流程"""
    print("=" * 80)
    print("增长模型分析技能 - 完整增长分析示例")
    print("=" * 80)

    # 1. 初始化分析器
    print("\n🚀 1. 初始化增长分析组件...")
    analyzer = GrowthModelAnalyzer()
    visualizer = GrowthVisualizer()

    # 2. 创建和加载样本数据
    print("\n📊 2. 创建和加载样本数据...")
    data = create_comprehensive_sample_data()
    print(f"✅ 样本数据创建成功：{len(data):,} 条用户记录")
    print(f"   - 裂变策略类型: {data['裂变类型'].nunique()} 种")
    print(f"   - 整体转化率: {data['是否转化'].mean():.2%}")
    print(f"   - 平均用户价值: {data['M值'].mean():.2f}")

    # 3. 数据质量检查
    print("\n🔍 3. 数据质量检查...")
    quality_report = analyzer.data_quality_check(
        data,
        required_cols=['用户码', '裂变类型', '是否转化', 'R值', 'F值', 'M值']
    )

    print(f"   - 总记录数: {quality_report['total_rows']:,}")
    print(f"   - 总字段数: {quality_report['total_columns']}")
    print(f"   - 重复记录: {quality_report['duplicate_rows']}")
    print(f"   - 数据完整性: {'✅ 良好' if quality_report['required_columns_check'] else '❌ 有问题'}")

    # 4. 营销活动效果分析
    print("\n📈 4. 营销活动效果分析...")
    campaign_results = analyzer.analyze_campaign_effectiveness(
        data=data,
        campaign_col='裂变类型',
        conversion_col='是否转化',
        control_group='无裂变页面'
    )

    print("\n   各策略转化效果:")
    for campaign, stats in campaign_results['campaign_statistics'].items():
        print(f"   • {campaign}:")
        print(f"     - 用户数: {stats['用户数']:,}")
        print(f"     - 转化率: {stats['转化率']:.2%}")
        print(f"     - 转化数: {stats['转化数']:,}")

    if campaign_results['lift_analysis']:
        print("\n   策略提升效果:")
        for campaign, lift in campaign_results['lift_analysis'].items():
            print(f"   • {campaign} vs 对照组:")
            print(f"     - 绝对提升: {lift['absolute_lift']:.2%}")
            print(f"     - 相对提升: {lift['relative_lift']:.2%}")
            print(f"     - 提升百分比: {lift['lift_percentage']:.1f}%")

    print(f"\n   统计显著性检验:")
    stats_test = campaign_results['statistical_test']
    print(f"   • 卡方统计量: {stats_test['chi2_statistic']:.4f}")
    print(f"   • P值: {stats_test['p_value']:.6f}")
    print(f"   • 显著性: {'✅ 显著' if stats_test['is_significant'] else '❌ 不显著'}")
    print(f"   • 效应量: {campaign_results['effect_size']['cramers_v']:.4f} ({campaign_results['effect_size']['interpretation']})")

    # 5. RFM用户分群分析
    print("\n👥 5. RFM用户分群分析...")
    rfm_results = analyzer.rfm_segmentation(
        data=data,
        user_col='用户码',
        recency_col='R值',
        frequency_col='F值',
        monetary_col='M值',
        n_clusters=5
    )

    print(f"   - 聚类数量: {rfm_results['n_clusters']}")
    print(f"   - 轮廓系数: {rfm_results['silhouette_score']:.3f}")

    print("\n   RFM分群统计:")
    for segment, count in rfm_results['rfm_segments'].items():
        print(f"   • {segment}: {count:,} 用户 ({count/len(data):.1%})")

    print("\n   聚类特征分析:")
    for cluster, stats in rfm_results['cluster_statistics'].items():
        print(f"   • {cluster}:")
        print(f"     - 用户数: {stats['count']:,} ({stats['percentage']:.1f}%)")
        print(f"     - 平均R值: {stats['avg_recency']:.1f}")
        print(f"     - 平均F值: {stats['avg_frequency']:.1f}")
        print(f"     - 平均M值: {stats['avg_monetary']:.2f}")

    # 6. 用户行为分析
    print("\n🎯 6. 用户行为分析...")
    behavior_results = analyzer.user_behavior_analysis(
        data=data,
        user_col='用户码',
        behavior_cols=['曾助力', '曾拼团', '曾推荐']
    )

    print("   单项行为参与率:")
    for behavior, stats in behavior_results['individual_behavior'].items():
        print(f"   • {behavior}: {stats['behavior_rate']:.1%} ({stats['behavior_users']:,} 用户)")

    if 'behavior_insights' in behavior_results:
        print("\n   行为洞察:")
        for insight in behavior_results['behavior_insights']:
            print(f"   • {insight}")

    # 7. 转化漏斗分析
    print("\n🔥 7. 转化漏斗分析...")
    funnel_results = analyzer.conversion_funnel_analysis(
        data=data,
        funnel_stages=['曾助力', '曾拼团', '是否转化']
    )

    print("   转化漏斗数据:")
    for stage_data in funnel_results['funnel_data']:
        print(f"   • {stage_data['stage']}: {stage_data['users']:,} 用户 ({stage_data['conversion_rate']:.1%})")
        if 'stage_conversion_rate' in stage_data:
            print(f"     - 阶段转化率: {stage_data['stage_conversion_rate']:.1%}")
            print(f"     - 阶段流失率: {stage_data['stage_dropoff_rate']:.1%}")

    print(f"\n   整体转化率: {funnel_results['overall_conversion_rate']:.1%}")

    if funnel_results['insights']:
        print("\n   漏斗洞察:")
        for insight in funnel_results['insights']:
            print(f"   • {insight}")

    # 8. 队列分析
    print("\n⏰ 8. 队列分析...")
    cohort_results = analyzer.cohort_analysis(
        data=data,
        user_col='用户码',
        time_col='R值',
        conversion_col='是否转化'
    )

    print("   时间队列转化率:")
    for cohort, stats in cohort_results['cohort_conversion'].items():
        print(f"   • {cohort}: {stats['conversion_rate']:.1%} ({stats['conversions']:,}/{stats['total_users']:,})")

    print(f"\n   整体转化率: {cohort_results['overall_conversion_rate']:.1%}")

    # 9. 生成综合洞察报告
    print("\n💡 9. 生成综合洞察报告...")
    insights_report = analyzer.generate_insights_report()

    print("   关键发现:")
    for finding in insights_report['key_findings']:
        print(f"   • {finding}")

    print("\n   策略建议:")
    for recommendation in insights_report['recommendations']:
        print(f"   • {recommendation}")

    # 10. 创建可视化图表
    print("\n📊 10. 创建可视化图表...")
    output_dir = Path(__file__).parent / "growth_analysis_output"
    output_dir.mkdir(exist_ok=True)

    try:
        # 转化漏斗图
        funnel_fig = visualizer.plot_conversion_funnel(
            funnel_results['funnel_data'],
            title="用户转化漏斗分析",
            interactive=False
        )
        if funnel_fig:
            funnel_fig.savefig(output_dir / "conversion_funnel.png", dpi=300, bbox_inches='tight')
            print("   ✅ 转化漏斗图已保存")
            import matplotlib.pyplot as plt
            plt.close(funnel_fig)

        # RFM分群散点图
        rfm_fig = visualizer.plot_rfm_segments(
            rfm_results['rfm_data'],
            title="RFM用户分群分析"
        )
        rfm_fig.write_html(str(output_dir / "rfm_segments.html"))
        print("   ✅ RFM分群图已保存")

        # 营销活动对比雷达图
        campaign_fig = visualizer.plot_campaign_comparison(
            campaign_results,
            title="营销活动效果对比"
        )
        campaign_fig.write_html(str(output_dir / "campaign_comparison.html"))
        print("   ✅ 营销活动对比图已保存")

    except Exception as e:
        print(f"   ⚠️ 图表生成遇到问题: {str(e)}")

    # 11. 保存分析结果
    print("\n💾 11. 保存分析结果...")

    # 保存营销活动分析结果
    campaign_df = pd.DataFrame(campaign_results['campaign_statistics']).T
    campaign_df.to_csv(output_dir / "campaign_analysis.csv", encoding='utf-8-sig')

    # 保存RFM分群结果
    rfm_data = rfm_results['rfm_data']
    rfm_data.to_csv(output_dir / "rfm_segments.csv", index=False, encoding='utf-8-sig')

    # 保存转化漏斗结果
    funnel_df = pd.DataFrame(funnel_results['funnel_data'])
    funnel_df.to_csv(output_dir / "conversion_funnel.csv", index=False, encoding='utf-8-sig')

    # 保存队列分析结果
    cohort_df = pd.DataFrame(cohort_results['cohort_conversion']).T
    cohort_df.to_csv(output_dir / "cohort_analysis.csv", encoding='utf-8-sig')

    print("   ✅ 分析结果已保存到 growth_analysis_output/ 目录")

    # 12. 总结报告
    print("\n" + "=" * 80)
    print("🎉 增长分析完成!")
    print("=" * 80)

    print(f"\n📁 生成的文件:")
    print(f"   - 营销活动分析: {output_dir}/campaign_analysis.csv")
    print(f"   - RFM分群结果: {output_dir}/rfm_segments.csv")
    print(f"   - 转化漏斗分析: {output_dir}/conversion_funnel.csv")
    print(f"   - 队列分析结果: {output_dir}/cohort_analysis.csv")
    print(f"   - 转化漏斗图: {output_dir}/conversion_funnel.png")
    print(f"   - RFM分群图: {output_dir}/rfm_segments.html")
    print(f"   - 营销对比图: {output_dir}/campaign_comparison.html")

    print(f"\n🎯 关键发现:")

    # 获取最佳和最差策略
    if campaign_results['lift_analysis']:
        best_campaign = max(campaign_results['lift_analysis'].items(),
                          key=lambda x: x[1]['lift_percentage'])
        print(f"   - 最佳策略: {best_campaign[0]} (提升 {best_campaign[1]['lift_percentage']:.1f}%)")
        print(f"   - 策略显著性: {'显著' if campaign_results['statistical_test']['is_significant'] else '不显著'}")

    # RFM分群洞察
    top_segment = max(rfm_results['segment_analysis'].items(),
                     key=lambda x: x[1]['Monetary'])
    print(f"   - 最有价值用户群: {top_segment[0]}")
    print(f"   - 高价值用户平均消费: {top_segment[1]['Monetary']:.2f}")

    # 转化漏斗洞察
    print(f"   - 整体转化率: {funnel_results['overall_conversion_rate']:.1%}")
    if len(funnel_results['funnel_data']) > 1:
        max_dropoff = max(funnel_results['funnel_data'][1:],
                         key=lambda x: x.get('stage_dropoff_rate', 0))
        print(f"   - 最大流失环节: {max_dropoff['stage']} ({max_dropoff.get('stage_dropoff_rate', 0)*100:.1f}%)")

    print(f"\n💡 增长模型分析技能特性:")
    print(f"   ✅ 全面的营销策略效果评估")
    print(f"   ✅ 智能RFM用户分群和价值分析")
    print(f"   ✅ 深度用户行为洞察和画像")
    print(f"   ✅ 专业的转化漏斗和队列分析")
    print(f"   ✅ 统计显著性检验和效应量分析")
    print(f"   ✅ 丰富的可视化图表和报告")
    print(f"   ✅ 数据驱动的增长策略建议")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 增长模型分析技能验证成功！可以开始使用。")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)