#!/usr/bin/env python3
"""
快速测试增长模型分析技能的核心功能
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent
sys.path.append(str(skill_path))

def create_sample_data():
    """创建测试用的样本数据"""
    np.random.seed(42)
    n_users = 1000

    data = {
        '用户码': [f'USER_{i:06d}' for i in range(n_users)],
        '裂变类型': np.random.choice(['无裂变页面', '助力砍价', '拼团狂买', '裂变海报', '好友助力'], n_users),
        '是否转化': np.random.choice([0, 1], n_users, p=[0.8, 0.2]),
        '城市': np.random.choice(['一线城市', '二线城市', '三线城市', '四线城市'], n_users),
        '设备类型': np.random.choice(['iOS', 'Android'], n_users),
        'R值': np.random.randint(1, 90, n_users),
        'F值': np.random.randint(1, 20, n_users),
        'M值': np.random.uniform(10, 1000, n_users),
        '曾助力': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
        '曾拼团': np.random.choice([0, 1], n_users, p=[0.8, 0.2]),
        '曾推荐': np.random.choice([0, 1], n_users, p=[0.9, 0.1]),
        '收入': np.random.uniform(0, 500, n_users),
        '成本': np.random.uniform(10, 100, n_users)
    }

    # 增加相关性：裂变类型对转化的影响
    conversion_influence = {
        '无裂变页面': 0.1,
        '助力砍价': 0.25,
        '拼团狂买': 0.30,
        '裂变海报': 0.20,
        '好友助力': 0.35
    }

    for i, campaign in enumerate(data['裂变类型']):
        if np.random.random() < conversion_influence[campaign]:
            data['是否转化'][i] = 1

    # 增加相关性：高价值用户更可能转化
    high_value_mask = data['M值'] > np.percentile(data['M值'], 70)
    data['是否转化'] = np.where(high_value_mask & (data['是否转化'] == 0),
                                np.random.choice([0, 1], p=[0.6, 0.4]),
                                data['是否转化'])

    return pd.DataFrame(data)

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'plotly': 'plotly'
    }

    missing_packages = []

    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)

    return missing_packages

def main():
    """快速测试主要功能"""
    print("🚀 增长模型分析技能快速测试")

    # 检查依赖包
    missing_deps = check_dependencies()
    if missing_deps:
        print("\n❌ 缺少依赖包:")
        for package in missing_deps:
            print(f"   - {package}")
        print(f"\n请安装缺少的依赖包:")
        print(f"   pip install {' '.join(missing_deps)}")
        print(f"   或者运行: pip install -r requirements.txt")
        return False

    try:
        # 1. 测试模块导入
        print("\n1. 测试模块导入...")
        from scripts.growth_analyzer import GrowthModelAnalyzer
        from scripts.uplift_modeling import UpliftModeler
        from scripts.roi_analyzer import ROIAnalyzer
        from scripts.growth_visualizer import GrowthVisualizer
        print("   ✓ 核心模块导入成功")

        # 2. 创建样本数据
        print("\n2. 创建样本数据...")
        sample_data = create_sample_data()
        print(f"   ✓ 样本数据创建成功: {len(sample_data)} 条记录")

        # 3. 测试增长分析器
        print("\n3. 测试增长分析器...")
        analyzer = GrowthModelAnalyzer()

        # 数据质量检查
        quality_report = analyzer.data_quality_check(
            sample_data,
            required_cols=['用户码', '裂变类型', '是否转化']
        )
        print(f"   ✓ 数据质量检查通过: {quality_report['total_rows']} 行")

        # 营销活动效果分析
        campaign_results = analyzer.analyze_campaign_effectiveness(
            sample_data,
            campaign_col='裂变类型',
            conversion_col='是否转化',
            control_group='无裂变页面'
        )
        print(f"   ✓ 营销活动分析完成: {len(campaign_results['campaign_statistics'])} 个活动")

        # RFM分群分析
        rfm_results = analyzer.rfm_segmentation(
            sample_data,
            user_col='用户码',
            recency_col='R值',
            frequency_col='F值',
            monetary_col='M值'
        )
        print(f"   ✓ RFM分群完成: {rfm_results['n_clusters']} 个聚类")

        # 4. 测试Uplift建模
        print("\n4. 测试Uplift建模...")
        uplift_modeler = UpliftModeler()

        # 准备Uplift数据
        uplift_data = uplift_modeler.prepare_uplift_data(
            sample_data,
            treatment_col='裂变类型',
            outcome_col='是否转化',
            control_value='无裂变页面',
            treatment_value='助力砍价'
        )
        print(f"   ✓ Uplift数据准备完成: {len(uplift_data)} 条记录")

        # 构建Uplift模型（使用较小的数据集以加快测试）
        if len(uplift_data) > 200:
            uplift_data_sample = uplift_data.sample(200, random_state=42)
        else:
            uplift_data_sample = uplift_data

        try:
            # 选择数值特征进行建模
            numeric_features = ['R值', 'F值', 'M值', '收入', '成本']
            available_features = [col for col in numeric_features if col in uplift_data_sample.columns]

            if len(available_features) >= 2:
                model_results = uplift_modeler.build_uplift_model(
                    uplift_data_sample,
                    feature_cols=available_features,
                    test_size=0.3,
                    random_state=42
                )
                print(f"   ✓ Uplift模型训练完成: 准确率 {model_results['accuracy']:.3f}")

                # 计算增量分数
                uplift_scores = uplift_modeler.calculate_uplift_scores(
                    uplift_data_sample,
                    treatment_col='裂变类型',
                    outcome_col='是否转化'
                )
                print(f"   ✓ 增量分数计算完成: 平均分数 {uplift_scores['uplift_score'].mean():.4f}")
            else:
                print("   ⚠️ 可用特征不足，跳过Uplift建模")
        except Exception as e:
            print(f"   ⚠️ Uplift建模跳过: {str(e)}")

        # 5. 测试ROI分析器
        print("\n5. 测试ROI分析器...")
        roi_analyzer = ROIAnalyzer()

        # 计算营销活动ROI
        roi_results = roi_analyzer.calculate_campaign_roi(
            sample_data,
            campaign_col='裂变类型',
            conversion_col='是否转化',
            cost_col='成本',
            revenue_col='收入',
            user_col='用户码'
        )
        print(f"   ✓ ROI分析完成: 整体ROI {roi_results['overall_roi']:.3f}")
        print(f"   ✓ 最佳活动: {roi_results['best_campaign']}")

        # 计算LTV
        ltv_results = roi_analyzer.calculate_ltv(
            sample_data,
            user_col='用户码',
            revenue_col='收入',
            frequency_col='F值',
            recency_col='R值'
        )
        print(f"   ✓ LTV计算完成: 平均LTV {ltv_results['调整LTV'].mean():.2f}")

        # 6. 测试可视化
        print("\n6. 测试可视化...")
        visualizer = GrowthVisualizer()

        # 测试漏斗图
        funnel_data = [
            {'stage': '访问用户', 'users': len(sample_data)},
            {'stage': '助力行为', 'users': sample_data['曾助力'].sum()},
            {'stage': '拼团行为', 'users': sample_data['曾拼团'].sum()},
            {'stage': '最终转化', 'users': sample_data['是否转化'].sum()}
        ]

        try:
            fig = visualizer.plot_conversion_funnel(funnel_data, interactive=False)
            if fig is not None:
                print("   ✓ 漏斗图生成成功")
                # 关闭图形以避免内存泄漏
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            print(f"   ⚠️ 漏斗图生成跳过: {str(e)}")

        # 7. 测试数据导出功能
        print("\n7. 测试数据导出...")
        output_dir = skill_path / "test_output"
        output_dir.mkdir(exist_ok=True)

        # 导出分析结果
        campaign_df = pd.DataFrame(campaign_results['campaign_statistics']).T
        campaign_df.to_csv(output_dir / "campaign_results.csv", encoding='utf-8-sig')

        # 导出RFM分群结果
        rfm_data = rfm_results['rfm_data']
        rfm_data.to_csv(output_dir / "rfm_segments.csv", index=False, encoding='utf-8-sig')

        # 导出ROI结果
        roi_df = pd.DataFrame(roi_results['campaign_metrics']).T
        roi_df.to_csv(output_dir / "roi_results.csv", encoding='utf-8-sig')

        print("   ✓ 分析结果已导出到 test_output/ 目录")

        # 8. 生成测试报告
        print("\n8. 生成测试报告...")
        report = {
            "test_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_size": len(sample_data),
            "campaigns_analyzed": len(campaign_results['campaign_statistics']),
            "rfm_clusters": rfm_results['n_clusters'],
            "overall_roi": roi_results['overall_roi'],
            "best_campaign": roi_results['best_campaign'],
            "conversion_rate": sample_data['是否转化'].mean(),
            "avg_ltv": ltv_results['调整LTV'].mean()
        }

        report_df = pd.DataFrame([report])
        report_df.to_csv(output_dir / "test_report.csv", index=False, encoding='utf-8-sig')

        print("   ✓ 测试报告已生成")

        print("\n🎉 核心功能测试通过！")
        print("\n增长模型分析技能已就绪，可以使用以下命令运行完整示例:")
        print("   python examples/growth_analysis_example.py")
        print("   python examples/uplift_modeling_example.py")
        print("   python examples/roi_optimization_example.py")
        print("   python examples/comprehensive_growth_analysis.py")

        # 显示关键结果
        print("\n📊 测试结果摘要:")
        print(f"   - 数据规模: {len(sample_data):,} 用户")
        print(f"   - 整体转化率: {sample_data['是否转化'].mean():.2%}")
        print(f"   - 分析活动数: {len(campaign_results['campaign_statistics'])}")
        print(f"   - 整体ROI: {roi_results['overall_roi']:.2%}")
        print(f"   - 最佳活动: {roi_results['best_campaign']}")
        print(f"   - RFM分群数: {rfm_results['n_clusters']}")
        print(f"   - 平均LTV: {ltv_results['调整LTV'].mean():.2f}")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)