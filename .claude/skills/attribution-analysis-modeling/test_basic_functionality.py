#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归因分析技能基础功能测试
Basic Functionality Test for Attribution Analysis Skill
"""

import pandas as pd
import sys
import os

# 添加父目录到路径以导入技能模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_attribution import AttributionAnalyzer
from markov_chains import MarkovChainAttributor
from shapley_values import ShapleyValueAttributor

def test_core_functionality():
    """测试核心功能"""

    print("=== Attribution Analysis Skill Test ===")
    print()

    # 1. 测试基础归因分析
    print("1. Testing Basic Attribution Analysis...")
    try:
        analyzer = AttributionAnalyzer()

        # 加载示例数据
        data_path = os.path.join('examples', 'sample_channel_data.csv')
        df = analyzer.load_and_validate_data(data_path)

        if df is not None:
            print(f"   Data loaded successfully: {len(df)} records")

            # 构建客户路径
            paths_df = analyzer.build_customer_paths(df)
            print(f"   Customer paths built: {len(paths_df)} paths")

            # 运行基础归因分析
            results = analyzer.compare_attribution_models(paths_df)

            if results is not None and not results.empty:
                print("   Basic attribution models executed successfully:")
                print(f"   Results shape: {results.shape}")

                # 获取各模型结果
                model_names = ['首次接触归因', '最后接触归因', '线性归因', '时间衰减归因', '位置归因']
                for model_name in model_names:
                    if model_name in results.columns:
                        weights = results[model_name].to_dict()
                        if weights:
                            top_channel = max(weights.items(), key=lambda x: x[1])
                            print(f"   - {model_name}: Top channel is {top_channel[0]} with weight {top_channel[1]:.3f}")

                print("   Basic attribution test: PASSED")
            else:
                print("   Basic attribution test: FAILED")
        else:
            print("   Data loading test: FAILED")

    except Exception as e:
        print(f"   Basic attribution test: FAILED with error: {e}")

    print()

    # 2. 测试马尔可夫链分析
    print("2. Testing Markov Chain Analysis...")
    try:
        if 'paths_df' in locals() and not paths_df.empty:
            markov_attributor = MarkovChainAttributor()

            # 构建转移矩阵
            transition_matrix = markov_attributor.build_transition_matrix(paths_df)
            print(f"   Transition matrix built: {len(transition_matrix)} states")

            # 计算基础转化率
            base_conversion_rate = paths_df['converted'].sum() / len(paths_df) if len(paths_df) > 0 else 0

            # 计算移除效应
            removal_effects = markov_attributor.calculate_removal_effects(paths_df, base_conversion_rate)

            # 计算归因权重
            markov_weights = markov_attributor.calculate_attribution_weights()
            print(f"   Markov attribution weights calculated for {len(markov_weights)} channels")

            if markov_weights:
                top_channel = max(markov_weights.items(), key=lambda x: x[1])
                print(f"   Top Markov channel: {top_channel[0]} with weight {top_channel[1]:.3f}")
                print("   Markov chain test: PASSED")
            else:
                print("   Markov chain test: FAILED")

        else:
            print("   Markov chain test: SKIPPED (no path data)")

    except Exception as e:
        print(f"   Markov chain test: FAILED with error: {e}")

    print()

    # 3. 测试Shapley值分析
    print("3. Testing Shapley Value Analysis...")
    try:
        if 'paths_df' in locals() and not paths_df.empty:
            shapley_attributor = ShapleyValueAttributor()

            # 计算Shapley值
            shapley_weights = shapley_attributor.calculate_shapley_values(paths_df)
            print(f"   Shapley values calculated for {len(shapley_weights)} channels")

            if shapley_weights:
                top_channel = max(shapley_weights.items(), key=lambda x: x[1])
                print(f"   Top Shapley channel: {top_channel[0]} with weight {top_channel[1]:.3f}")
                print("   Shapley value test: PASSED")
            else:
                print("   Shapley value test: FAILED")

        else:
            print("   Shapley value test: SKIPPED (no path data)")

    except Exception as e:
        print(f"   Shapley value test: FAILED with error: {e}")

    print()

    # 4. 测试模块导入
    print("4. Testing Module Imports...")
    try:
        from attribution_visualizer import AttributionVisualizer
        print("   AttributionVisualizer import: PASSED")
    except Exception as e:
        print(f"   AttributionVisualizer import: FAILED with error: {e}")

    print()

    # 5. 测试数据验证功能
    print("5. Testing Data Validation...")
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'timestamp': ['2024-01-01T10:00:00Z', '2024-01-02T15:00:00Z',
                         '2024-01-03T09:00:00Z', '2024-01-04T14:00:00Z'],
            'channel': ['paid_search', 'email', 'social_media', 'paid_search'],
            'conversion_status': [0, 1, 0, 1],
            'conversion_value': [0, 100, 0, 200],
            'cost': [50, 10, 30, 60]
        })

        # 保存测试数据
        test_data.to_csv('test_data.csv', index=False, encoding='utf-8')

        # 测试加载和验证
        analyzer_test = AttributionAnalyzer()
        validated_data = analyzer_test.load_and_validate_data('test_data.csv')

        if validated_data is not None and len(validated_data) == 4:
            print("   Data validation test: PASSED")

            # 清理测试文件
            os.remove('test_data.csv')
        else:
            print("   Data validation test: FAILED")

    except Exception as e:
        print(f"   Data validation test: FAILED with error: {e}")

    print()
    print("=== Test Summary ===")
    print("Core functionality testing completed.")
    print("Check the output above for specific test results.")

if __name__ == "__main__":
    test_core_functionality()