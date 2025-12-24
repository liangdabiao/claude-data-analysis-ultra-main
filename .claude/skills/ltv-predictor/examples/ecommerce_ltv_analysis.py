#!/usr/bin/env python3
"""
电商LTV分析完整示例
演示如何使用LTV预测技能进行完整的电商客户生命周期价值分析
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 添加技能模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / 'scripts'))

from quick_analysis import quick_ltv_analysis

def main():
    """电商LTV分析示例"""
    print("🛒 电商客户生命周期价值分析示例")
    print("=" * 60)

    # 示例数据文件路径
    sample_data_path = current_dir.parent / 'data' / 'sample_orders.csv'
    output_dir = current_dir.parent / 'examples' / 'ecommerce_analysis_results'

    print(f"📁 示例数据: {sample_data_path}")
    print(f"📁 输出目录: {output_dir}")
    print()

    # 检查数据文件是否存在
    if not sample_data_path.exists():
        print(f"❌ 示例数据文件不存在: {sample_data_path}")
        print("请先运行数据准备脚本或检查文件路径。")
        return

    try:
        # 1. 基础LTV分析
        print("1️⃣ 执行基础LTV分析...")
        print("-" * 30)

        basic_config = {
            'data_processor_config': {
                'feature_period_months': 3,  # 使用前3个月数据
                'prediction_period_months': 12,  # 预测后12个月
                'remove_outliers': True,
                'min_orders_per_customer': 1
            },
            'regression_config': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scoring_metric': 'r2',
                'enable_hyperparameter_tuning': False
            },
            'models_to_train': ['linear_regression', 'random_forest']
        }

        results = quick_ltv_analysis(
            file_path=str(sample_data_path),
            feature_period_months=3,
            prediction_period_months=12,
            output_dir=str(output_dir / 'basic_analysis'),
            config=basic_config,
            generate_charts=True,
            generate_reports=True
        )

        print("✅ 基础LTV分析完成")

        # 2. 高级LTV分析（包含调优）
        print("\n2️⃣ 执行高级LTV分析...")
        print("-" * 30)

        advanced_config = {
            'data_processor_config': {
                'feature_period_months': 6,  # 使用前6个月数据
                'prediction_period_months': 12,  # 预测后12个月
                'remove_outliers': True,
                'min_orders_per_customer': 2  # 至少2个订单
            },
            'regression_config': {
                'test_size': 0.2,
                'cv_folds': 10,  # 更多折交叉验证
                'scoring_metric': 'r2',
                'enable_hyperparameter_tuning': True  # 启用超参数调优
            },
            'models_to_train': ['linear_regression', 'random_forest']
        }

        advanced_results = quick_ltv_analysis(
            file_path=str(sample_data_path),
            feature_period_months=6,
            prediction_period_months=12,
            output_dir=str(output_dir / 'advanced_analysis'),
            config=advanced_config,
            generate_charts=True,
            generate_reports=True
        )

        print("✅ 高级LTV分析完成")

        # 3. 结果比较
        print("\n3️⃣ 分析结果比较...")
        print("-" * 30)

        basic_summary = results.get('summary', {})
        advanced_summary = advanced_results.get('summary', {})

        print("基础分析结果:")
        print(f"  - 最佳模型: {basic_summary.get('model_summary', {}).get('best_model', 'Unknown')}")
        print(f"  - R²分数: {basic_summary.get('model_summary', {}).get('best_r2_score', 0):.4f}")
        print(f"  - 分析客户数: {basic_summary.get('data_summary', {}).get('total_customers', 0)}")
        print(f"  - 平均LTV: {basic_summary.get('data_summary', {}).get('avg_ltv', 0):.0f}")

        print("\n高级分析结果:")
        print(f"  - 最佳模型: {advanced_summary.get('model_summary', {}).get('best_model', 'Unknown')}")
        print(f"  - R²分数: {advanced_summary.get('model_summary', {}).get('best_r2_score', 0):.4f}")
        print(f"  - 分析客户数: {advanced_summary.get('data_summary', {}).get('total_customers', 0)}")
        print(f"  - 平均LTV: {advanced_summary.get('data_summary', {}).get('avg_ltv', 0):.0f}")

        # 4. 业务洞察
        print("\n4️⃣ 业务洞察...")
        print("-" * 30)

        # 获取RFM数据进行分析
        basic_rfm_path = results['output_paths']['rfm_features']
        if os.path.exists(basic_rfm_path):
            rfm_data = pd.read_csv(basic_rfm_path)

            # 客户价值分层统计
            if '客户价值分层' in rfm_data.columns:
                segment_counts = rfm_data['客户价值分层'].value_counts()
                print("客户价值分层分布:")
                for segment, count in segment_counts.items():
                    percentage = (count / len(rfm_data)) * 100
                    print(f"  - {segment}: {count}人 ({percentage:.1f}%)")

            # 城市分析
            if '城市' in rfm_data.columns:
                city_ltv = rfm_data.groupby('城市')['年度LTV'].agg(['count', 'mean']).sort_values('mean', ascending=False)
                print("\n城市LTV排名:")
                for city, stats in city_ltv.head(5).iterrows():
                    print(f"  - {city}: {stats['count']}人, 平均LTV {stats['mean']:,.0f}")

        # 5. 推荐策略
        print("\n5️⃣ 推荐营销策略...")
        print("-" * 30)

        recommendations = basic_summary.get('recommendations_summary', [])
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # 显示前5条建议
                print(f"  {i}. {rec}")
        else:
            print("  基于分析结果，建议:")
            print("  1. 针对高价值客户设计VIP维护计划")
            print("  2. 对中价值客户进行交叉销售和向上销售")
            print("  3. 对低价值客户制定激活和留存策略")
            print("  4. 重点关注一线城市客户的个性化服务")
            print("  5. 建立客户生命周期管理流程")

        # 6. 文件生成总结
        print("\n6️⃣ 生成文件总结...")
        print("-" * 30)

        print("基础分析生成文件:")
        print(f"  📊 RFM特征数据: {results['output_paths']['rfm_features']}")
        print(f"  🤖 训练模型: {results['output_paths']['models']}")
        print(f"  📋 分析摘要: {results['summary_path']}")

        chart_count = len(results.get('chart_paths', {}))
        report_count = len(results.get('report_paths', {}))
        print(f"  📈 可视化图表: {chart_count}个")
        print(f"  📄 分析报告: {report_count}个")

        print("\n高级分析生成文件:")
        print(f"  📊 RFM特征数据: {advanced_results['output_paths']['rfm_features']}")
        print(f"  🤖 训练模型: {advanced_results['output_paths']['models']}")
        print(f"  📋 分析摘要: {advanced_results['summary_path']}")

        chart_count = len(advanced_results.get('chart_paths', {}))
        report_count = len(advanced_results.get('report_paths', {}))
        print(f"  📈 可视化图表: {chart_count}个")
        print(f"  📄 分析报告: {report_count}个")

        print("\n" + "=" * 60)
        print("🎉 电商LTV分析示例完成！")
        print("=" * 60)
        print(f"📁 所有结果已保存至: {output_dir}")
        print("\n📖 后续步骤:")
        print("1. 查看生成的HTML报告了解详细分析结果")
        print("2. 使用可视化图表洞察客户行为模式")
        print("3. 基于推荐策略制定营销计划")
        print("4. 定期更新数据重新分析以跟踪变化")

    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()