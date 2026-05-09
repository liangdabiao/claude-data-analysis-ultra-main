#!/usr/bin/env python3
"""
快速测试AB测试分析技能的核心功能
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent
sys.path.append(str(skill_path))

def main():
    """快速测试主要功能"""
    print("🚀 AB测试分析技能快速测试")

    try:
        # 1. 测试模块导入
        print("\n1. 测试模块导入...")
        from scripts.ab_test_analyzer import ABTestAnalyzer
        from scripts.statistical_tests import StatisticalTests
        from scripts.segment_analyzer import SegmentAnalyzer
        from scripts.visualizer import ABTestVisualizer
        print("   ✓ 核心模块导入成功")

        # 2. 测试数据加载和分析
        print("\n2. 测试AB测试分析器...")
        analyzer = ABTestAnalyzer()

        data_dir = skill_path / "examples" / "sample_data"
        data_file = data_dir / "sample_ab_test_data.csv"

        if data_file.exists():
            data = analyzer.load_data(str(data_file))
            if data is not None:
                print(f"   ✓ 数据加载成功: {len(data)} 条记录")

                # 测试转化率分析
                conversion_results = analyzer.analyze_conversion(
                    data, group_col='实验组别', conversion_col='转化状态'
                )
                print(f"   ✓ 转化率分析成功: {len(conversion_results['conversion_analysis'])} 个组")

                # 测试统计检验
                from scripts.statistical_tests import StatisticalTests
                stats_tests = StatisticalTests()

                # 准备数据用于统计检验
                data_converted = data.copy()
                data_converted['转化状态_数值'] = data_converted['转化状态'].apply(
                    lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0
                )

                t_test_result = stats_tests.t_test(
                    data=data_converted,
                    group_col='实验组别',
                    metric_col='转化状态_数值'
                )
                print(f"   ✓ t检验成功: p值 = {t_test_result['p_value']:.4f}")

                # 测试分群分析
                segment_analyzer = SegmentAnalyzer()
                value_segments = segment_analyzer.value_based_segmentation(
                    data, value_col='累计消费金额', n_tiers=3
                )
                print(f"   ✓ 价值分群成功: {value_segments['n_segments']} 个分群")

                # 测试可视化
                visualizer = ABTestVisualizer()
                fig = visualizer.plot_conversion_comparison(
                    conversion_results['conversion_analysis'],
                    title='快速测试 - 转化率对比'
                )
                if fig is not None:
                    print("   ✓ 转化率可视化成功")
                    # 关闭图形以避免内存泄漏
                    import matplotlib.pyplot as plt
                    plt.close(fig)

        # 3. 测试样本量计算
        print("\n3. 测试样本量计算...")
        sample_size = analyzer.calculate_sample_size(
            baseline_rate=0.10,
            expected_lift=0.20,
            alpha=0.05,
            power=0.80
        )
        print(f"   ✓ 样本量计算成功: {sample_size:,} (每组)")

        # 4. 测试贝叶斯分析
        print("\n4. 测试贝叶斯AB测试...")
        bayesian_result = analyzer.bayesian_ab_test(
            data,
            group_col='实验组别',
            conversion_col='转化状态',
            n_simulations=1000  # 使用较小的模拟数量以加快测试
        )
        if bayesian_result:
            print(f"   ✓ 贝叶斯分析成功: {len(bayesian_result)} 个结果")

        print("\n🎉 核心功能测试通过！")
        print("\nAB测试分析技能已就绪，可以使用以下命令运行完整示例:")
        print("   python examples/basic_ab_test_example.py")
        print("   python examples/advanced_segmentation_example.py")
        print("   python examples/comprehensive_analysis_example.py")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)