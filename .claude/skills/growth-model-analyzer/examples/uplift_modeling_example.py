#!/usr/bin/env python3
"""
Uplift建模完整示例

演示Uplift建模技能的核心功能：
- XGBoost Uplift模型训练
- 增量分数计算
- Qini曲线分析
- 用户分群和策略优化
- 模型效果评估
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path))

from scripts.uplift_modeling import UpliftModeler
from scripts.growth_visualizer import GrowthVisualizer


def create_uplift_sample_data():
    """创建Uplift建模样本数据"""
    np.random.seed(42)
    n_users = 3000

    # 创建具有明显处理效应的数据
    data = {
        '用户码': [f'USER_{i:06d}' for i in range(n_users)],
        '裂变类型': np.random.choice(['无裂变页面', '助力砍价', '拼团狂买'], n_users, p=[0.4, 0.35, 0.25]),
        '城市类型': np.random.choice(['一线城市', '二线城市', '三线城市'], n_users, p=[0.3, 0.4, 0.3]),
        '设备类型': np.random.choice(['iOS', 'Android', 'Web'], n_users, p=[0.5, 0.4, 0.1]),
        '年龄段': np.random.choice(['18-25', '26-35', '36-45', '46+'], n_users, p=[0.2, 0.4, 0.3, 0.1]),
        '用户等级': np.random.choice(['新用户', '普通用户', '活跃用户', 'VIP用户'], n_users, p=[0.25, 0.35, 0.3, 0.1])
    }

    df = pd.DataFrame(data)

    # 创建数值特征
    df['历史订单数'] = np.random.poisson(5, n_users)
    df['平均订单金额'] = np.random.gamma(2, 50, n_users)
    df['注册天数'] = np.random.exponential(100, n_users).astype(int) + 1
    df['上次购买天数'] = np.random.exponential(30, n_users).astype(int) + 1
    df['活跃度'] = np.random.beta(2, 5, n_users)
    df['价格敏感度'] = np.random.beta(1.5, 3, n_users)

    # 模拟处理效应
    def calculate_treatment_effect(row):
        """根据用户特征计算处理效应"""
        base_prob = 0.15

        # 处理效应
        if row['裂变类型'] == '助力砍价':
            treatment_effect = 0.20
            # 对活跃用户效果更好
            if row['用户等级'] in ['活跃用户', 'VIP用户']:
                treatment_effect += 0.10
            # 对价格敏感用户效果更好
            if row['价格敏感度'] > 0.7:
                treatment_effect += 0.08
        elif row['裂变类型'] == '拼团狂买':
            treatment_effect = 0.15
            # 对新用户效果更好
            if row['用户等级'] == '新用户':
                treatment_effect += 0.12
            # 对历史订单少的用户效果更好
            if row['历史订单数'] < 3:
                treatment_effect += 0.10
        else:  # 无裂变页面
            treatment_effect = 0.0

        # 基础转化概率
        base_conversion = base_prob + 0.05 * (row['活跃度']) + 0.02 * np.log1p(row['历史订单数'])

        return base_conversion + treatment_effect

    df['转化概率'] = df.apply(calculate_treatment_effect, axis=1)
    df['是否转化'] = np.random.random(len(df)) < df['转化概率']

    return df.drop('转化概率', axis=1)


def main():
    """主函数：演示完整Uplift建模流程"""
    print("=" * 80)
    print("Uplift建模技能 - 完整示例")
    print("=" * 80)

    # 1. 初始化Uplift建模器
    print("\n🚀 1. 初始化Uplift建模组件...")
    uplift_modeler = UpliftModeler()
    visualizer = GrowthVisualizer()

    # 2. 创建和加载样本数据
    print("\n📊 2. 创建Uplift建模样本数据...")
    data = create_uplift_sample_data()
    print(f"✅ 样本数据创建成功：{len(data):,} 条用户记录")
    print(f"   - 处理策略类型: {data['裂变类型'].nunique()} 种")
    print(f"   - 整体转化率: {data['是否转化'].mean():.2%}")

    # 各策略转化率
    strategy_conversion = data.groupby('裂变类型')['是否转化'].mean()
    print("\n   各策略基础转化率:")
    for strategy, rate in strategy_conversion.items():
        print(f"   • {strategy}: {rate:.2%}")

    # 3. 准备Uplift数据
    print("\n🔧 3. 准备Uplift建模数据...")
    uplift_data = uplift_modeler.prepare_uplift_data(
        data,
        treatment_col='裂变类型',
        outcome_col='是否转化',
        control_value='无裂变页面',
        treatment_value='助力砍价'  # 专注于助力砍价策略
    )

    print(f"✅ Uplift数据准备完成:")
    print(f"   - 总记录数: {len(uplift_data):,}")
    print(f"   - 处理组记录: {len(uplift_data[uplift_data['裂变类型'] != '无裂变页面']):,}")
    print(f"   - 对照组记录: {len(uplift_data[uplift_data['裂变类型'] == '无裂变页面']):,}")

    # 4. 构建Uplift模型
    print("\n🤖 4. 构建XGBoost Uplift模型...")

    # 选择数值特征
    numeric_features = [
        '历史订单数', '平均订单金额', '注册天数', '上次购买天数',
        '活跃度', '价格敏感度'
    ]

    # 确保特征存在
    available_features = [col for col in numeric_features if col in uplift_data.columns]
    print(f"   使用特征: {', '.join(available_features)}")

    # 添加处理组编码
    uplift_data['裂变_type'] = (uplift_data['裂变类型'] != '无裂变页面').astype(int)

    model_results = uplift_modeler.build_uplift_model(
        uplift_data,
        feature_cols=available_features,
        treatment_col='裂变_type',
        outcome_col='是否转化',
        model_type='xgboost',
        test_size=0.25,
        random_state=42
    )

    print(f"✅ Uplift模型训练完成:")
    print(f"   - 模型准确率: {model_results['accuracy']:.3f}")
    print(f"   - 训练集大小: {len(model_results['X_train']):,}")
    print(f"   - 测试集大小: {len(model_results['X_test']):,}")

    # 5. 计算增量分数
    print("\n📈 5. 计算增量分数...")
    uplift_scores_df = uplift_modeler.calculate_uplift_scores(
        model_results['X_test'].copy(),
        treatment_col='裂变类型',
        outcome_col='是否转化'
    )

    # 恢复原始用户信息
    test_indices = model_results['X_test'].index
    uplift_scores_df = uplift_scores_df.reset_index(drop=True)
    uplift_scores_df['用户码'] = data.loc[test_indices, '用户码'].values
    uplift_scores_df['裂变类型'] = data.loc[test_indices, '裂变类型'].values

    print(f"✅ 增量分数计算完成:")
    print(f"   - 平均增量分数: {uplift_scores_df['uplift_score'].mean():.4f}")
    print(f"   - 增量分数标准差: {uplift_scores_df['uplift_score'].std():.4f}")
    print(f"   - 正增量用户比例: {(uplift_scores_df['uplift_score'] > 0).mean():.1%}")

    # 6. Qini曲线分析
    print("\n📊 6. Qini曲线分析...")
    qini_results = uplift_modeler.analyze_qini_curve(
        uplift_scores_df,
        treatment_col='裂变类型',
        outcome_col='是否转化',
        control_value='无裂变页面'
    )

    print(f"✅ Qini曲线分析完成:")
    print(f"   - Qini AUC: {qini_results['qini_auc']:.4f}")
    print(f"   - Random AUC: {qini_results['random_auc']:.4f}")
    print(f"   - AUQC (调整后): {qini_results['auqc']:.4f}")
    print(f"   - 模型表现: {qini_results['model_performance']}")

    # 7. 用户分群分析
    print("\n👥 7. 基于增量分数的用户分群...")
    segmented_users = uplift_modeler.uplift_segmentation(
        uplift_scores_df,
        n_segments=5
    )

    # 分群统计
    segment_stats = segmented_users.groupby('uplift_segment').agg({
        'uplift_score': ['mean', 'std', 'count'],
        '是否转化': 'mean'
    }).round(4)

    segment_stats.columns = ['平均增量分数', '标准差', '用户数', '实际转化率']
    print("   分群统计:")
    for segment, stats in segment_stats.iterrows():
        print(f"   • {segment}:")
        print(f"     - 用户数: {stats['用户数']:,}")
        print(f"     - 平均增量分数: {stats['平均增量分数']:.4f}")
        print(f"     - 实际转化率: {stats['实际转化率']:.2%}")

    # 8. 生成Uplift分析报告
    print("\n💡 8. 生成Uplift分析报告...")
    uplift_report = uplift_modeler.generate_uplift_report(uplift_scores_df)

    print("   增量分数分析摘要:")
    summary = uplift_report['summary']
    print(f"   • 总用户数: {summary['total_users']:,}")
    print(f"   • 平均增量分数: {summary['avg_uplift_score']:.4f}")
    print(f"   • 正增量用户比例: {summary['positive_uplift_ratio']:.1%}")

    print("\n   关键洞察:")
    for insight in uplift_report['insights']:
        print(f"   • {insight}")

    print("\n   策略建议:")
    for recommendation in uplift_report['recommendations']:
        print(f"   • {recommendation}")

    # 9. 创建可视化图表
    print("\n📊 9. 创建可视化图表...")
    output_dir = Path(__file__).parent / "uplift_modeling_output"
    output_dir.mkdir(exist_ok=True)

    try:
        # Qini曲线
        qini_fig = visualizer.plot_qini_curve(
            qini_results['qini_data'],
            title=f"Qini曲线分析 (AUQC: {qini_results['auqc']:.4f})"
        )
        qini_fig.write_html(str(output_dir / "qini_curve.html"))
        print("   ✅ Qini曲线图已保存")

        # 增量分数分布
        uplift_fig = visualizer.plot_uplift_distribution(
            uplift_scores_df,
            title="增量分数分布分析"
        )
        uplift_fig.write_html(str(output_dir / "uplift_distribution.html"))
        print("   ✅ 增量分布图已保存")

        # 传统Qini曲线（静态）
        static_qini = uplift_modeler.plot_qini_curve(
            qini_results,
            title="Qini曲线分析 (静态图)"
        )
        static_qini.savefig(output_dir / "qini_curve_static.png", dpi=300, bbox_inches='tight')
        print("   ✅ 静态Qini曲线图已保存")
        import matplotlib.pyplot as plt
        plt.close(static_qini)

    except Exception as e:
        print(f"   ⚠️ 图表生成遇到问题: {str(e)}")

    # 10. 保存模型和结果
    print("\n💾 10. 保存模型和分析结果...")

    # 保存模型
    model_path = uplift_modeler.save_model('uplift_main', str(output_dir / "uplift_model.pkl"))
    print("   ✅ Uplift模型已保存")

    # 保存增量分数结果
    results_df = uplift_scores_df[['用户码', '裂变类型', '是否转化', 'uplift_score',
                                   'P_TR', 'P_TN', 'P_CR', 'P_CN']].copy()
    if 'uplift_segment' in uplift_scores_df.columns:
        results_df['分群'] = uplift_scores_df['uplift_segment']

    results_df.to_csv(output_dir / "uplift_scores.csv", index=False, encoding='utf-8-sig')
    print("   ✅ 增量分数结果已保存")

    # 保存Qini曲线数据
    qini_df = pd.DataFrame(qini_results['qini_data'])
    qini_df.to_csv(output_dir / "qini_curve_data.csv", index=False, encoding='utf-8-sig')
    print("   ✅ Qini曲线数据已保存")

    # 保存分群统计
    segment_stats.to_csv(output_dir / "segment_statistics.csv", encoding='utf-8-sig')
    print("   ✅ 分群统计已保存")

    # 11. 业务应用建议
    print("\n🎯 11. 业务应用建议...")

    # 计算高增量用户特征
    high_uplift_threshold = uplift_scores_df['uplift_score'].quantile(0.8)
    high_uplift_users = uplift_scores_df[uplift_scores_df['uplift_score'] >= high_uplift_threshold]

    print(f"   高增量用户特征分析 (前20%用户):")
    if len(high_uplift_users) > 0:
        high_uplift_original = data.loc[data['用户码'].isin(high_uplift_users['用户码'])]

        print(f"   • 用户数: {len(high_uplift_users):,} ({len(high_uplift_users)/len(uplift_scores_df):.1%})")
        print(f"   • 平均增量分数: {high_uplift_users['uplift_score'].mean():.4f}")

        # 城市类型分布
        if '城市类型' in high_uplift_original.columns:
            city_dist = high_uplift_original['城市类型'].value_counts()
            print("   • 城市类型分布:")
            for city, count in city_dist.head(3).items():
                print(f"     - {city}: {count:,} ({count/len(high_uplift_original):.1%})")

        # 用户等级分布
        if '用户等级' in high_uplift_original.columns:
            level_dist = high_uplift_original['用户等级'].value_counts()
            print("   • 用户等级分布:")
            for level, count in level_dist.head(3).items():
                print(f"     - {level}: {count:,} ({count/len(high_uplift_original):.1%})")

    print("\n   实施建议:")
    print("   1. 优先针对高增量分数用户实施助力砍价策略")
    print("   2. 为不同分群设计个性化的营销方案")
    print("   3. 定期更新模型以适应用户行为变化")
    print("   4. 结合业务经验解释和应用模型结果")
    print("   5. 建立监控机制评估策略实际效果")

    print("\n" + "=" * 80)
    print("🎉 Uplift建模分析完成!")
    print("=" * 80)

    print(f"\n📁 生成的文件:")
    print(f"   - Uplift模型: {output_dir}/uplift_model.pkl")
    print(f"   - 增量分数结果: {output_dir}/uplift_scores.csv")
    print(f"   - Qini曲线数据: {output_dir}/qini_curve_data.csv")
    print(f"   - 分群统计: {output_dir}/segment_statistics.csv")
    print(f"   - Qini曲线图: {output_dir}/qini_curve.html")
    print(f"   - 增量分布图: {output_dir}/uplift_distribution.html")
    print(f"   - 静态Qini图: {output_dir}/qini_curve_static.png")

    print(f"\n🎯 关键结果:")
    print(f"   - 模型表现: {qini_results['model_performance']}")
    print(f"   - AUQC值: {qini_results['auqc']:.4f}")
    print(f"   - 正增量用户比例: {(uplift_scores_df['uplift_score'] > 0).mean():.1%}")
    print(f"   - 高增量用户平均分数: {high_uplift_users['uplift_score'].mean():.4f}")

    print(f"\n💡 Uplift建模技能特性:")
    print(f"   ✅ 基于XGBoost的先进增量建模")
    print(f"   ✅ 精确的增量分数计算和解释")
    print(f"   ✅ 专业的Qini曲线模型评估")
    print(f"   ✅ 智能用户分群和价值排序")
    print(f"   ✅ 可解释的模型结果和应用指导")
    print(f"   ✅ 完整的模型保存和加载功能")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 Uplift建模技能验证成功！可以开始使用。")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)