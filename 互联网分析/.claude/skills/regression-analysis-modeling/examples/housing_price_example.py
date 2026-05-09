#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Housing Price Prediction Example
Complete example of housing price prediction using multiple regression models
"""

import pandas as pd
import numpy as np
from core_regression import RegressionAnalyzer
from feature_engineering import FeatureEngineering
from model_evaluation import ModelEvaluator
from prediction_visualizer import PredictionVisualizer

def create_sample_housing_data():
    """Create realistic sample housing data"""
    print("创建示例房价数据...")

    np.random.seed(42)
    n_samples = 1000

    # Generate realistic housing features
    data = []

    for i in range(n_samples):
        # Basic features
        area = np.random.lognormal(4.5, 0.3)  # Log-normal distribution for area
        area = np.clip(area, 40, 300)  # Clip to reasonable range

        # Number of rooms depends on area
        max_rooms = int(area / 25)
        min_rooms = max(1, max_rooms - 2)
        rooms = np.random.randint(min_rooms, max_rooms + 1)

        # Bathrooms
        bathrooms = np.random.randint(1, min(rooms + 1, 4))

        # Floor information
        total_floors = np.random.choice([6, 12, 18, 30, 40], p=[0.3, 0.3, 0.2, 0.1, 0.1])
        floor = np.random.randint(1, total_floors + 1)

        # Building age
        build_year = np.random.choice(np.arange(1990, 2024))
        age = 2024 - build_year

        # Location features (distance in meters)
        subway_distance = np.random.exponential(800) * (1 + 0.3 * np.random.randn())
        subway_distance = np.clip(subway_distance, 50, 5000)

        school_distance = np.random.exponential(600) * (1 + 0.3 * np.random.randn())
        school_distance = np.clip(school_distance, 100, 4000)

        mall_distance = np.random.exponential(1000) * (1 + 0.3 * np.random.randn())
        mall_distance = np.clip(mall_distance, 200, 6000)

        # Categorical features with price impact
        decoration = np.random.choice(['毛坯', '简装修', '精装修', '豪华装修'],
                                   p=[0.1, 0.3, 0.4, 0.2])

        direction = np.random.choice(['南', '东南', '东', '西南', '西', '北'],
                                   p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])

        estate_type = np.random.choice(['普通小区', '高档小区', '豪华小区'],
                                     p=[0.4, 0.4, 0.2])

        # Calculate base price using realistic factors
        # Base price per square meter
        base_price_per_sqm = 8000

        # Location adjustments
        location_multiplier = 1.0
        location_multiplier *= (1 - subway_distance / 10000)  # Closer to subway = higher price
        location_multiplier *= (1 - school_distance / 15000)  # Closer to school = higher price
        location_multiplier *= (1 - mall_distance / 20000)   # Closer to mall = higher price

        # Building quality adjustments
        decoration_multiplier = {'毛坯': 0.7, '简装修': 0.85, '精装修': 1.0, '豪华装修': 1.3}[decoration]
        direction_multiplier = {'南': 1.1, '东南': 1.05, '东': 1.0, '西南': 0.95, '西': 0.9, '北': 0.85}[direction]
        estate_multiplier = {'普通小区': 0.8, '高档小区': 1.0, '豪华小区': 1.25}[estate_type]

        # Age adjustment (newer buildings are more expensive)
        age_multiplier = 1.0 - (age / 100) * 0.3  # Max 30% reduction for very old buildings

        # Floor adjustment (higher floors in tall buildings are more expensive)
        floor_ratio = floor / total_floors
        floor_multiplier = 1.0 + (floor_ratio - 0.5) * 0.1 * (total_floors / 30)

        # Calculate final price
        price_per_sqm = (base_price_per_sqm *
                        location_multiplier *
                        decoration_multiplier *
                        direction_multiplier *
                        estate_multiplier *
                        age_multiplier *
                        floor_multiplier *
                        (1 + 0.1 * np.random.randn()))  # Add some noise

        total_price = price_per_sqm * area

        # Ensure reasonable price ranges
        total_price = np.clip(total_price, 300000, 15000000)

        data.append({
            '房屋ID': i + 1,
            '面积': round(area, 1),
            '房间数': rooms,
            '卫生间数': bathrooms,
            '楼层': floor,
            '总楼层': total_floors,
            '建造年份': build_year,
            '地铁距离': round(subway_distance),
            '学校距离': round(school_distance),
            '商场距离': round(mall_distance),
            '装修等级': decoration,
            '朝向': direction,
            '小区类型': estate_type,
            '房价': round(total_price, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv('sample_housing_data.csv', index=False, encoding='utf-8-sig')

    print(f"生成了 {len(df)} 条房屋数据")
    print(f"价格范围: ￥{df['房价'].min():,.0f} - ￥{df['房价'].max():,.0f}")
    print(f"平均价格: ￥{df['房价'].mean():,.0f}")

    return df

def run_housing_price_example():
    """Run complete housing price prediction example"""
    print("🏠 开始房价预测示例")
    print("=" * 50)

    # 1. Create sample data
    housing_df = create_sample_housing_data()

    # 2. Initialize analyzers
    analyzer = RegressionAnalyzer()
    fe = FeatureEngineering()
    evaluator = ModelEvaluator()
    visualizer = PredictionVisualizer()

    # 3. Advanced feature engineering
    print("\n=== 高级特征工程 ===")

    # Create derived features
    housing_df['房龄'] = 2024 - housing_df['建造年份']
    housing_df['楼层比例'] = housing_df['楼层'] / housing_df['总楼层']
    housing_df['房间密度'] = housing_df['面积'] / housing_df['房间数']

    # Distance score (lower is better, so we invert)
    housing_df['交通便利性'] = (housing_df['地铁距离'].max() - housing_df['地铁距离']) / housing_df['地铁距离'].max()
    housing_df['学区便利性'] = (housing_df['学校距离'].max() - housing_df['学校距离']) / housing_df['学校距离'].max()
    housing_df['购物便利性'] = (housing_df['商场距离'].max() - housing_df['商场距离']) / housing_df['商场距离'].max()

    # Combined convenience score
    housing_df['综合便利性'] = (housing_df['交通便利性'] +
                               housing_df['学区便利性'] +
                               housing_df['购物便利性']) / 3

    # Price per square meter for analysis
    housing_df['单价'] = housing_df['房价'] / housing_df['面积']

    print(f"新增特征数量: {housing_df.shape[1] - 13}")
    print("新增特征包括: 房龄、楼层比例、房间密度、交通便利性、学区便利性、购物便利性、综合便利性、单价")

    # 4. Run regression analysis with interaction features
    print("\n=== 回归模型训练 ===")
    analysis_results = analyzer.run_complete_analysis(
        housing_df,
        '房价',
        create_interactions=True
    )

    # 5. Detailed model evaluation
    print("\n=== 模型详细评估 ===")

    # Residual analysis for best model
    best_model_name = analyzer.best_model_name
    best_results = analysis_results['results'][best_model_name]

    residual_analysis = evaluator.perform_residual_analysis(
        best_results['y_test'],
        best_results['predictions'],
        best_model_name
    )

    # Learning curve analysis
    learning_analysis = evaluator.analyze_learning_curves(
        analyzer.best_model,
        analysis_results['X_final'],
        analysis_results['y_final']
    )

    # 6. Feature importance analysis
    print("\n=== 特征重要性分析 ===")
    feature_importance = analysis_results['feature_importance']

    print("Top 10 影响房价的关键因素:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f} ({row['importance_pct']:.1f}%)")

    # 7. Create visualizations
    print("\n=== 生成可视化分析 ===")

    # Comprehensive dashboard
    visualizer.create_comprehensive_dashboard(
        analysis_results['results'],
        feature_importance,
        save_path='housing_price_dashboard.png'
    )

    # Individual analysis plots
    visualizer.create_individual_analysis_plots(
        analysis_results['results'],
        output_dir='housing_analysis_plots'
    )

    # 8. Generate comprehensive report
    print("\n=== 生成分析报告 ===")
    evaluation_report = evaluator.generate_evaluation_report(
        analysis_results['results'],
        save_to_file=True
    )

    # 9. Business insights and price predictions
    print("\n=== 房价预测业务洞察 ===")

    # Analyze price predictions by different segments
    y_pred = best_results['predictions']
    y_true = best_results['y_test']

    # Calculate prediction accuracy by price ranges
    price_analysis = pd.DataFrame({
        '实际价格': y_true,
        '预测价格': y_pred,
        '绝对误差': np.abs(y_true - y_pred),
        '相对误差': np.abs(y_true - y_pred) / y_true * 100
    })

    # Create price segments
    price_analysis['价格区间'] = pd.cut(
        price_analysis['实际价格'],
        bins=[0, 500000, 1000000, 2000000, np.inf],
        labels=['经济型(≤50万)', '中档型(50-100万)', '高档型(100-200万)', '豪华型(>200万)']
    )

    print("\n不同价格区间的预测准确性:")
    for segment in price_analysis['价格区间'].cat.categories:
        segment_data = price_analysis[price_analysis['价格区间'] == segment]
        if len(segment_data) > 0:
            avg_error = segment_data['相对误差'].mean()
            print(f"  {segment}: 平均预测误差 {avg_error:.1f}%")

    # 10. Feature impact analysis
    print(f"\n=== 关键特征对房价的影响 ===")

    # Analyze categorical features impact
    categorical_analysis = {}

    # Decoration level impact
    decoration_impact = housing_df.groupby('装修等级')['单价'].mean().sort_values(ascending=False)
    print("\n装修等级对单价的影响:")
    for level, price in decoration_impact.items():
        print(f"  {level}: ￥{price:,.0f}/m²")

    # Direction impact
    direction_impact = housing_df.groupby('朝向')['单价'].mean().sort_values(ascending=False)
    print("\n朝向对单价的影响:")
    for direction, price in direction_impact.items():
        print(f"  {direction}: ￥{price:,.0f}/m²")

    # Estate type impact
    estate_impact = housing_df.groupby('小区类型')['单价'].mean().sort_values(ascending=False)
    print("\n小区类型对单价的影响:")
    for estate, price in estate_impact.items():
        print(f"  {estate}: ￥{price:,.0f}/m²")

    # 11. Model performance summary
    print(f"\n=== 模型性能总结 ===")
    best_metrics = analysis_results['results'][best_model_name]['metrics']
    print(f"最佳模型: {best_model_name}")
    print(f"R² 分数: {best_metrics['test_r2']:.4f}")
    print(f"平均绝对误差: ￥{best_metrics['test_mae']:,.0f}")
    print(f"均方根误差: ￥{best_metrics['test_rmse']:,.0f}")

    # Business interpretation
    mean_price = housing_df['房价'].mean()
    mae_percentage = (best_metrics['test_mae'] / mean_price) * 100
    print(f"平均预测误差百分比: {mae_percentage:.2f}%")

    print(f"\n=== 投资建议 ===")
    print("基于模型分析的投资建议:")
    print("1. 关注靠近地铁、学校、商圈的地段")
    print("2. 优先选择精装修或豪华装修的房产")
    print("3. 南向和东南朝向的房产具有更高价值")
    print("4. 高档小区和豪华小区有更好的增值潜力")
    print("5. 中高楼层（特别是电梯房）价格优势明显")

    print(f"\n✅ 房价预测分析完成！")
    print(f"生成的文件:")
    print(f"- sample_housing_data.csv: 示例房价数据")
    print(f"- housing_price_dashboard.png: 综合分析仪表板")
    print(f"- housing_analysis_plots/: 详细分析图表")
    print(f"- model_evaluation_report.md: 评估报告")

if __name__ == "__main__":
    run_housing_price_example()