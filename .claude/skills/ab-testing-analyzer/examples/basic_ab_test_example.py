#!/usr/bin/env python3
"""
基础AB测试示例

演示AB测试分析技能的核心功能：
- 数据加载和预处理
- 转化率分析
- 统计显著性检验
- 基础可视化
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path))

from scripts.ab_test_analyzer import ABTestAnalyzer
from scripts.statistical_tests import StatisticalTests
from scripts.visualizer import ABTestVisualizer


def main():
    """主函数：演示基础AB测试分析流程"""
    print("=" * 60)
    print("AB测试分析技能 - 基础示例")
    print("=" * 60)

    # 1. 初始化组件
    print("\n1. 初始化AB测试分析组件...")
    analyzer = ABTestAnalyzer()
    stats_tests = StatisticalTests()
    visualizer = ABTestVisualizer()

    # 2. 加载样本数据
    print("\n2. 加载AB测试样本数据...")
    data_dir = Path(__file__).parent / "sample_data"
    data_file = data_dir / "sample_ab_test_data.csv"

    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return False

    data = analyzer.load_data(str(data_file))
    if data is None:
        return False

    print(f"✅ 数据加载成功：{len(data)} 条用户记录")
    print(f"   - 实验组别: {data['实验组别'].unique()}")
    print(f"   - 总转化率: {data['转化状态'].apply(lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0).mean():.2%}")

    # 3. 数据质量检查
    print("\n3. 数据质量检查...")
    quality_report = analyzer.data_quality_check(
        data,
        required_cols=['用户ID', '实验组别', '转化状态']
    )

    print(f"   - 总行数: {quality_report['total_rows']:,}")
    print(f"   - 总列数: {quality_report['total_columns']}")
    print(f"   - 重复行数: {quality_report['duplicate_rows']}")
    print(f"   - 必需列检查: {quality_report['required_columns_check']}")

    if quality_report['missing_values']:
        print("   - 缺失值统计:")
        for col, info in quality_report['missing_values'].items():
            print(f"     * {col}: {info['count']} ({info['percentage']:.1f}%)")

    # 4. 基础转化率分析
    print("\n4. 基础转化率分析...")
    conversion_results = analyzer.analyze_conversion(
        data=data,
        group_col='实验组别',
        conversion_col='转化状态',
        control_group='对照组'
    )

    print("   - 各组转化率:")
    for group, stats in conversion_results['conversion_analysis'].items():
        print(f"     * {group}: {stats['conversion_rate']:.2%} "
              f"({stats['conversions']}/{stats['sample_size']})")

    # 5. 提升率分析
    print("\n5. 提升率分析...")
    if conversion_results['lift_analysis']:
        for group, lift in conversion_results['lift_analysis'].items():
            print(f"   - {group} vs 对照组:")
            print(f"     * 绝对提升: {lift['absolute_lift']:.2%}")
            print(f"     * 相对提升: {lift['relative_lift']:.2%}")
            print(f"     * 提升百分比: {lift['lift_percentage']:.1f}%")
            ci_lower, ci_upper = lift['lift_confidence_interval']
            print(f"     * 置信区间: [{ci_lower:.2%}, {ci_upper:.2%}]")

    # 6. 统计显著性检验
    print("\n6. 统计显著性检验...")

    # 转化状态数值化
    data_converted = data.copy()
    data_converted['转化状态_数值'] = data_converted['转化状态'].apply(
        lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0
    )

    # t检验
    t_test_result = stats_tests.t_test(
        data=data_converted,
        group_col='实验组别',
        metric_col='转化状态_数值',
        test_type='independent'
    )

    print("   - t检验结果:")
    print(f"     * 统计量: {t_test_result['statistic']:.4f}")
    print(f"     * P值: {t_test_result['p_value']:.4f}")
    print(f"     * 效应量 (Cohen's d): {t_test_result['effect_size']:.4f}")
    print(f"     * 效应量解释: {t_test_result['effect_size_interpretation']}")
    print(f"     * 显著性: {'是' if t_test_result['is_significant'] else '否'}")

    # 卡方检验
    chi2_result = stats_tests.chi_square_test(
        data=data,
        group_col='实验组别',
        outcome_col='转化状态',
        test_type='independence'
    )

    print("   - 卡方检验结果:")
    print(f"     * 卡方统计量: {chi2_result['statistic']:.4f}")
    print(f"     * P值: {chi2_result['p_value']:.4f}")
    print(f"     * 效应量 (Cramer's V): {chi2_result['effect_size']:.4f}")
    print(f"     * 效应量解释: {chi2_result['effect_size_interpretation']}")
    print(f"     * 显著性: {'是' if chi2_result['is_significant'] else '否'}")

    # 7. 留存率分析
    print("\n7. 留存率分析...")
    retention_results = analyzer.analyze_retention(
        data=data,
        group_col='实验组别',
        retention_col='留存状态'
    )

    print("   - 各组留存率:")
    for group, stats in retention_results['retention_analysis'].items():
        print(f"     * {group}: {stats['retention_rate']:.2%} "
              f"({stats['retained_users']}/{stats['sample_size']})")

    # 8. 随机分组验证
    print("\n8. 随机分组验证...")
    numeric_features = ['累计消费金额', '年龄']
    randomization_check = analyzer.check_randomization(
        data=data,
        group_col='实验组别',
        feature_cols=numeric_features
    )

    print("   - 分组均衡性检查:")
    for feature, result in randomization_check.items():
        print(f"     * {feature}:")
        print(f"       - 检验方法: {result['test']}")
        print(f"       - P值: {result['p_value']:.4f}")
        print(f"       - 分组均衡: {'是' if result['is_balanced'] else '否'}")

    # 9. 样本量计算
    print("\n9. 样本量计算...")
    baseline_rate = conversion_results['conversion_analysis']['对照组']['conversion_rate']
    expected_lift = 0.10  # 期望提升10%

    required_sample_size = analyzer.calculate_sample_size(
        baseline_rate=baseline_rate,
        expected_lift=expected_lift,
        alpha=0.05,
        power=0.80
    )

    print(f"   - 基准转化率: {baseline_rate:.2%}")
    print(f"   - 期望提升: {expected_lift:.0%}")
    print(f"   - 所需样本量: {required_sample_size:,} (每组)")
    print(f"   - 当前样本量: {len(data[data['实验组别'] == '对照组']):,} (对照组)")

    if len(data[data['实验组别'] == '对照组']) >= required_sample_size:
        print("   - ✅ 样本量充足")
    else:
        print("   - ⚠️ 样本量不足，建议增加样本")

    # 10. 生成可视化
    print("\n10. 生成可视化图表...")
    output_dir = Path(__file__).parent / "basic_output"
    output_dir.mkdir(exist_ok=True)

    # 转化率对比图
    fig1 = visualizer.plot_conversion_comparison(
        conversion_results['conversion_analysis'],
        title='AB测试转化率对比',
        show_confidence_interval=True,
        save_path=str(output_dir / "conversion_comparison.png")
    )
    print("   ✅ 转化率对比图已保存")

    # 留存率曲线图
    fig2 = visualizer.plot_retention_curves(
        retention_results['retention_analysis'],
        title='用户留存率对比',
        save_path=str(output_dir / "retention_curves.png")
    )
    print("   ✅ 留存率曲线图已保存")

    # 置信区间图
    fig3 = visualizer.plot_confidence_intervals(
        conversion_results['conversion_analysis'],
        title='转化率置信区间分析',
        save_path=str(output_dir / "confidence_intervals.png")
    )
    print("   ✅ 置信区间图已保存")

    # 关闭图形以避免内存泄漏
    import matplotlib.pyplot as plt
    plt.close('all')

    # 11. 保存分析结果
    print("\n11. 保存分析结果...")

    # 保存转化率分析结果
    conversion_df = pd.DataFrame([
        {
            '实验组别': group,
            '转化率': stats['conversion_rate'],
            '样本量': stats['sample_size'],
            '转化数': stats['conversions']
        }
        for group, stats in conversion_results['conversion_analysis'].items()
    ])
    conversion_df.to_csv(output_dir / "conversion_results.csv", index=False, encoding='utf-8-sig')

    # 保存统计检验结果
    stats_results = pd.DataFrame([
        {
            '检验方法': 't检验',
            '统计量': t_test_result['statistic'],
            'P值': t_test_result['p_value'],
            '效应量': t_test_result['effect_size'],
            '显著性': t_test_result['is_significant']
        },
        {
            '检验方法': '卡方检验',
            '统计量': chi2_result['statistic'],
            'P值': chi2_result['p_value'],
            '效应量': chi2_result['effect_size'],
            '显著性': chi2_result['is_significant']
        }
    ])
    stats_results.to_csv(output_dir / "statistical_tests.csv", index=False, encoding='utf-8-sig')

    print("✅ 分析结果已保存到 basic_output/ 目录")

    # 12. 总结报告
    print("\n" + "=" * 60)
    print("🎉 AB测试基础分析完成!")
    print("=" * 60)

    print(f"\n📁 生成的文件:")
    print(f"   - 转化率结果: {output_dir}/conversion_results.csv")
    print(f"   - 统计检验结果: {output_dir}/statistical_tests.csv")
    print(f"   - 转化率对比图: {output_dir}/conversion_comparison.png")
    print(f"   - 留存率曲线图: {output_dir}/retention_curves.png")
    print(f"   - 置信区间图: {output_dir}/confidence_intervals.png")

    print(f"\n🎯 关键发现:")

    # 获取测试组和对照组的转化率
    if '测试组' in conversion_results['conversion_analysis'] and '对照组' in conversion_results['conversion_analysis']:
        test_rate = conversion_results['conversion_analysis']['测试组']['conversion_rate']
        control_rate = conversion_results['conversion_analysis']['对照组']['conversion_rate']
        lift = (test_rate - control_rate) / control_rate if control_rate > 0 else 0

        print(f"   - 测试组转化率: {test_rate:.2%}")
        print(f"   - 对照组转化率: {control_rate:.2%}")
        print(f"   - 相对提升: {lift:.1%}")
        print(f"   - 统计显著性: {'显著' if t_test_result['is_significant'] else '不显著'}")

        if t_test_result['is_significant'] and lift > 0:
            recommendation = "建议推广测试组方案"
        elif t_test_result['is_significant'] and lift < 0:
            recommendation = "建议维持对照组方案"
        else:
            recommendation = "需要更多数据或重新设计实验"

        print(f"   - 建议: {recommendation}")

    print(f"\n💡 AB测试分析技能特性:")
    print(f"   ✅ 完整的转化率和留存率分析")
    print(f"   ✅ 多种统计显著性检验方法")
    print(f"   ✅ 效应量计算和解释")
    print(f"   ✅ 随机分组验证")
    print(f"   ✅ 样本量计算和功效分析")
    print(f"   ✅ 丰富的可视化图表")
    print(f"   ✅ 专业的分析报告生成")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 AB测试分析技能验证成功！可以开始使用。")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)