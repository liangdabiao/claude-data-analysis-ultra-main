#!/usr/bin/env python3
"""
快速LTV预测示例
演示如何使用已训练的模型快速预测新客户的LTV
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 添加技能模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / 'scripts'))

from quick_analysis import quick_ltv_analysis, predict_new_customers, batch_predict_ltv

def create_sample_new_customers():
    """创建示例新客户数据"""
    np.random.seed(42)

    # 模拟新客户订单数据
    new_customers = []

    # 生成10个新客户的近期订单
    for i in range(10, 20):  # 客户ID从10开始
        customer_id = f"CUST{i:03d}"
        city = np.random.choice(['北京', '上海', '深圳', '广州', '杭州', '成都', '武汉'])

        # 每个客户1-3个订单
        num_orders = np.random.randint(1, 4)

        for j in range(num_orders):
            order_id = 2000 + i * 10 + j
            product_code = f"PROD{np.random.randint(1, 15):03d}"
            order_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 90))
            product_desc = f"测试产品{product_code}"
            quantity = np.random.randint(1, 5)
            price = np.random.uniform(20, 200)

            new_customers.append({
                '订单号': order_id,
                '产品码': product_code,
                '消费日期': order_date.strftime('%Y-%m-%d %H:%M'),
                '产品说明': product_desc,
                '数量': quantity,
                '单价': round(price, 2),
                '用户码': customer_id,
                '城市': city
            })

    return pd.DataFrame(new_customers)

def main():
    """快速LTV预测示例"""
    print("⚡ 快速LTV预测示例")
    print("=" * 60)

    # 路径设置
    sample_data_path = current_dir.parent / 'data' / 'sample_orders.csv'
    model_output_dir = current_dir.parent / 'examples' / 'quick_prediction' / 'models'
    new_customers_output_dir = current_dir.parent / 'examples' / 'quick_prediction'

    print(f"📁 训练数据: {sample_data_path}")
    print(f"🤖 模型目录: {model_output_dir}")
    print(f"📁 输出目录: {new_customers_output_dir}")
    print()

    try:
        # 第一步：训练基础模型
        print("1️⃣ 训练LTV预测模型...")
        print("-" * 30)

        if not sample_data_path.exists():
            print(f"❌ 示例数据文件不存在: {sample_data_path}")
            return

        # 快速训练配置
        quick_config = {
            'data_processor_config': {
                'feature_period_months': 3,
                'prediction_period_months': 12,
                'remove_outliers': False,  # 快速训练不去除异常值
                'min_orders_per_customer': 1
            },
            'regression_config': {
                'test_size': 0.3,  # 更大的测试集
                'cv_folds': 3,  # 更少的交叉验证折数
                'scoring_metric': 'r2',
                'enable_hyperparameter_tuning': False  # 不调以节省时间
            },
            'models_to_train': ['random_forest']  # 只训练随机森林
        }

        # 训练模型
        training_results = quick_ltv_analysis(
            file_path=str(sample_data_path),
            feature_period_months=3,
            prediction_period_months=12,
            output_dir=str(model_output_dir),
            config=quick_config,
            generate_charts=False,  # 不生成图表以节省时间
            generate_reports=False   # 不生成报告以节省时间
        )

        print("✅ 模型训练完成")

        # 第二步：创建新客户数据
        print("\n2️⃣ 创建新客户数据...")
        print("-" * 30)

        new_customers_df = create_sample_new_customers()
        new_customers_file = new_customers_output_dir / 'new_customers.csv'
        new_customers_df.to_csv(new_customers_file, index=False, encoding='utf-8-sig')

        print(f"✅ 新客户数据已创建: {new_customers_file}")
        print(f"   - 客户数: {new_customers_df['用户码'].nunique()}")
        print(f"   - 订单数: {len(new_customers_df)}")
        print(f"   - 时间范围: {new_customers_df['消费日期'].min()} 至 {new_customers_df['消费日期'].max()}")

        # 第三步：预测新客户LTV
        print("\n3️⃣ 预测新客户LTV...")
        print("-" * 30)

        predictions_file = new_customers_output_dir / 'new_customer_predictions.csv'
        predictions_df = predict_new_customers(
            model_dir=str(model_output_dir),
            new_orders_file=str(new_customers_file),
            output_path=str(predictions_file)
        )

        print("✅ 新客户LTV预测完成")

        # 第四步：分析预测结果
        print("\n4️⃣ 分析预测结果...")
        print("-" * 30)

        if predictions_df is not None:
            print("预测结果统计:")
            print(f"  - 预测客户数: {len(predictions_df)}")
            print(f"  - 平均预测LTV: {predictions_df['预测LTV'].mean():,.0f}")
            print(f"  - 最高预测LTV: {predictions_df['预测LTV'].max():,.0f}")
            print(f"  - 最低预测LTV: {predictions_df['预测LTV'].min():,.0f}")

            # 按城市分组分析（如果有城市信息）
            if '城市' in predictions_df.columns:
                city_analysis = predictions_df.groupby('城市')['预测LTV'].agg(['count', 'mean', 'std']).round(2)
                print("\n各城市LTV预测:")
                for city, stats in city_analysis.iterrows():
                    print(f"  - {city}: {stats['count']}人, 平均{stats['mean']:,.0f}, 标准差{stats['std']:,.0f}")
            else:
                print("\n注：预测结果中不包含城市信息，无法按城市分析")

            # 客户分群
            def categorize_ltv(ltv):
                if ltv >= 1000:
                    return "高价值客户"
                elif ltv >= 500:
                    return "中价值客户"
                else:
                    return "低价值客户"

            predictions_df['价值分层'] = predictions_df['预测LTV'].apply(categorize_ltv)
            segment_counts = predictions_df['价值分层'].value_counts()
            print("\n客户价值分层:")
            for segment, count in segment_counts.items():
                percentage = (count / len(predictions_df)) * 100
                print(f"  - {segment}: {count}人 ({percentage:.1f}%)")

        # 第五步：生成快速建议
        print("\n5️⃣ 营销建议...")
        print("-" * 30)

        if predictions_df is not None:
            # 高价值客户建议
            high_value = predictions_df[predictions_df['预测LTV'] >= 1000]
            if not high_value.empty:
                print("高价值客户策略:")
                print("  - 提供VIP专属服务")
                print("  - 安排客户经理专人对接")
                print("  - 定制个性化营销方案")
                print(f"  - 重点维护的{len(high_value)}位客户: {', '.join(high_value['用户码'].tolist())}")

            # 中价值客户建议
            medium_value = predictions_df[(predictions_df['预测LTV'] >= 500) & (predictions_df['预测LTV'] < 1000)]
            if not medium_value.empty:
                print("\n中价值客户策略:")
                print("  - 推荐升级产品和套餐")
                print("  - 提供优惠券刺激消费")
                print("  - 增加互动频率")
                print(f"  - 重点关注的{len(medium_value)}位客户: {', '.join(medium_value['用户码'].tolist())}")

            # 低价值客户建议
            low_value = predictions_df[predictions_df['预测LTV'] < 500]
            if not low_value.empty:
                print("\n低价值客户策略:")
                print("  - 发放新人礼包和优惠券")
                print("  - 推荐性价比高的产品")
                print("  - 增加产品体验机会")
                print(f"  - 需要激活的{len(low_value)}位客户: {', '.join(low_value['用户码'].tolist())}")

        # 第六步：总结
        print("\n6️⃣ 预测总结...")
        print("-" * 30)

        print("✅ 快速LTV预测流程完成")
        print(f"📁 模型文件: {model_output_dir}")
        print(f"📊 新客户数据: {new_customers_file}")
        print(f"🎯 预测结果: {predictions_file}")

        print("\n📖 使用说明:")
        print("1. 模型已训练并保存，可直接用于新客户预测")
        print("2. 新客户数据格式需与训练数据保持一致")
        print("3. 预测结果可用于制定个性化营销策略")
        print("4. 建议定期重新训练模型以保持预测准确性")

        print("\n🔧 批量预测:")
        print("如需批量预测大量客户，可使用:")
        print(f"python -c \"from quick_analysis import batch_predict_ltv; batch_predict_ltv('{model_output_dir}', 'rfm_file.csv', 'batch_predictions.csv')\"")

    except Exception as e:
        print(f"\n❌ 预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()