"""
高级推荐系统示例

演示推荐系统技能的高级功能：
- 多种评估方法对比
- 交互式可视化
- 用户画像分析
- 冷启动策略
- 实时推荐演示
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path))

from scripts.recommendation_engine import RecommendationEngine
from scripts.recommender_evaluator import RecommenderEvaluator
from scripts.data_analyzer import DataAnalyzer
from scripts.recommender_visualizer import RecommenderVisualizer


def simulate_real_time_recommendation(engine, user_id, new_ratings):
    """
    模拟实时推荐场景

    Args:
        engine: 推荐引擎
        user_id: 用户ID
        new_ratings: 新评分列表 [(商品ID, 评分), ...]
    """
    print(f"\n🔄 模拟实时推荐场景 - 用户 {user_id}")

    # 显示原始推荐
    original_recs = engine.recommend_hybrid(user_id, top_k=5)
    print("原始推荐:")
    for i, (item_id, score) in enumerate(original_recs, 1):
        print(f"   {i}. {item_id}: {score:.3f}")

    # 模拟用户添加新评分
    print(f"\n📝 用户添加了 {len(new_ratings)} 个新评分:")
    for item_id, rating in new_ratings:
        print(f"   - {item_id}: {rating} 分")

    # 在实际应用中，这里应该更新用户-商品矩阵并重新计算相似度
    # 为了演示，我们简单地模拟推荐的变化
    print("\n🔄 更新后的推荐:")
    updated_recs = engine.recommend_hybrid(user_id, top_k=5)
    for i, (item_id, score) in enumerate(updated_recs, 1):
        # 模拟推荐分数的变化
        new_score = score + np.random.uniform(-0.1, 0.2)
        print(f"   {i}. {item_id}: {new_score:.3f}")


def demonstrate_cold_start_strategies(engine, evaluator):
    """
    演示冷启动解决策略

    Args:
        engine: 推荐引擎
        evaluator: 评估器
    """
    print("\n❄️ 冷启动问题解决策略演示")

    # 1. 新用户冷启动
    print("\n1. 新用户冷启动:")
    new_user_id = "NEW_U001"

    try:
        # 尝试为新用户生成推荐
        recommendations = engine.recommend_hybrid(new_user_id, top_k=5)
        if recommendations:
            print("   ✅ 使用热门商品推荐策略成功")
            for i, (item_id, score) in enumerate(recommendations, 1):
                print(f"      {i}. {item_id}: {score:.3f}")
    except Exception as e:
        print(f"   ❌ 新用户推荐失败: {str(e)}")

    # 2. 新商品冷启动
    print("\n2. 新商品冷启动:")
    new_item_id = "NEW_P001"
    print(f"   - 新商品 {new_item_id} 需要通过基于内容的推荐或推广策略")

    # 3. 冷启动策略建议
    print("\n3. 冷启动解决策略建议:")
    print("   a) 热门商品推荐：为新用户推荐全局热门商品")
    print("   b) 基于内容的推荐：根据商品属性和用户人口统计学信息")
    print("   c) 主动学习：让用户对一些热门商品进行评分")
    print("   d) 混合策略：结合多种方法提高冷启动性能")


def analyze_user_segments(analyzer, user_data):
    """
    分析用户分群

    Args:
        analyzer: 数据分析器
        user_data: 用户行为数据
    """
    print("\n👥 用户分群分析")

    # 用户画像分析
    profiling = analyzer.analyze_user_profiling(user_data)

    if 'user_segments' in profiling:
        print("\n用户价值分层:")
        for segment, count in profiling['user_segments'].items():
            print(f"   - {segment}: {count:,} 用户")

    if 'demographics' in profiling:
        demographics = profiling['demographics']

        if 'age' in demographics:
            age_info = demographics['age']
            print(f"\n年龄分布:")
            print(f"   - 平均年龄: {age_info.get('mean', 0):.1f} 岁")
            print(f"   - 年龄范围: {age_info.get('min', 0)} - {age_info.get('max', 0)} 岁")

        if 'gender' in demographics:
            gender_info = demographics['gender']
            print(f"\n性别分布:")
            for gender, count in gender_info.items():
                print(f"   - {gender}: {count:,} 用户")

        if 'cities' in demographics:
            city_info = demographics['cities']
            print(f"\n城市分布:")
            print(f"   - 覆盖城市: {city_info.get('total_cities', 0)} 个")
            print(f"   - 最活跃城市: {city_info.get('most_active_city', 'N/A')}")


def compare_evaluation_methods(evaluator, engine, user_item_matrix):
    """
    对比不同评估方法

    Args:
        evaluator: 评估器
        engine: 推荐引擎
        user_item_matrix: 用户-商品矩阵
    """
    print("\n📊 评估方法对比分析")

    # 限制评估规模以提高演示速度
    max_users = min(15, len(user_item_matrix))

    # 1. 留一法评估
    print("\n1. 留一法评估 (Leave-One-Out):")
    loo_results = evaluator.leave_one_out_evaluation(
        engine, user_item_matrix,
        k_values=[5, 10],
        num_users=max_users
    )

    print(f"   - 评估用户数: {loo_results.get('evaluated_users', 0)}")
    print(f"   - Precision@5: {loo_results.get('precision@5', 0):.4f}")
    print(f"   - Recall@5: {loo_results.get('recall@5', 0):.4f}")

    # 2. 交叉验证评估
    print("\n2. 交叉验证评估:")
    cv_results = evaluator.cross_validation_evaluation(
        engine, user_item_matrix,
        cv_folds=3,
        k_values=[5, 10]
    )

    print(f"   - 评估折数: {cv_results.get('cv_folds', 0)}")
    print(f"   - Precision@5: {cv_results.get('precision@5', 0):.4f}")
    print(f"   - Recall@5: {cv_results.get('recall@5', 0):.4f}")

    # 3. 评估方法对比
    print("\n📈 评估方法对比:")
    methods = ['留一法', '交叉验证']
    precision_scores = [loo_results.get('precision@5', 0), cv_results.get('precision@5', 0)]
    recall_scores = [loo_results.get('recall@5', 0), cv_results.get('recall@5', 0)]

    for i, method in enumerate(methods):
        print(f"   {method}:")
        print(f"     - Precision@5: {precision_scores[i]:.4f}")
        print(f"     - Recall@5: {recall_scores[i]:.4f}")


def create_interactive_dashboard(visualizer, recommendations, evaluation_results):
    """
    创建交互式仪表板

    Args:
        visualizer: 可视化器
        recommendations: 推荐结果
        evaluation_results: 评估结果
    """
    print("\n📊 创建交互式仪表板")

    # 注意：这里只是示例代码框架
    # 实际的交互式仪表板需要使用 plotly dash 或 streamlit

    try:
        # 创建交互式推荐图表
        if recommendations:
            interactive_rec_fig = visualizer.create_interactive_recommendations(
                recommendations[:10], "U001"
            )
            print("   ✅ 交互式推荐图表创建成功")

        # 创建算法比较图表
        comparison_data = {
            'Algorithm': ['User-CF', 'Item-CF', 'SVD', 'Hybrid'],
            'Precision@5': [0.15, 0.18, 0.22, 0.25],
            'Recall@5': [0.12, 0.14, 0.18, 0.20]
        }
        comparison_df = pd.DataFrame(comparison_data)

        interactive_comp_fig = visualizer.create_interactive_comparison(comparison_df)
        print("   ✅ 交互式算法比较图表创建成功")

        print("\n💡 交互式仪表板功能包括:")
        print("   - 动态筛选推荐算法")
        print("   - 实时调整推荐参数")
        print("   - 交互式探索用户行为模式")
        print("   - 自定义评估指标选择")

    except Exception as e:
        print(f"   ❌ 交互式仪表板创建失败: {str(e)}")


def generate_comprehensive_report(evaluator, analysis_results, evaluation_results):
    """
    生成综合分析报告

    Args:
        evaluator: 评估器
        analysis_results: 数据分析结果
        evaluation_results: 评估结果
    """
    print("\n📋 生成综合分析报告")

    # 生成评估报告
    report = evaluator.generate_evaluation_report(
        evaluation_results,
        "推荐系统技能高级示例"
    )

    # 添加数据分析结果
    data_summary = "\n## 数据洞察分析\n"

    if 'user_behavior' in analysis_results:
        ub = analysis_results['user_behavior']
        data_summary += f"- 用户规模: {ub.get('total_users', 0):,} 用户\n"
        data_summary += f"- 商品规模: {ub.get('total_items', 0):,} 商品\n"
        data_summary += f"- 交互规模: {ub.get('total_interactions', 0):,} 次交互\n"

    if 'sparsity' in analysis_results:
        sp = analysis_results['sparsity']
        data_summary += f"- 数据稀疏度: {sp.get('sparsity_ratio', 0):.2%}\n"
        data_summary += f"- 平均每用户交互: {sp.get('avg_interactions_per_user', 0):.1f} 次\n"

    if 'cold_start' in analysis_results:
        cs = analysis_results['cold_start']
        data_summary += f"- 冷启动严重程度: {cs.get('cold_start_severity', '未知')}\n"

    # 添加建议和结论
    recommendations = "\n## 优化建议\n"
    recommendations += "1. 算法优化：\n"
    recommendations += "   - 采用混合推荐策略提高准确率\n"
    recommendations += "   - 优化相似度计算方法\n"
    recommendations += "   - 引入深度学习模型\n\n"

    recommendations += "2. 冷启动问题：\n"
    recommendations += "   - 实施热门商品推荐策略\n"
    recommendations += "   - 结合基于内容的推荐\n"
    recommendations += "   - 设计主动学习机制\n\n"

    recommendations += "3. 系统架构：\n"
    recommendations += "   - 实现增量学习机制\n"
    recommendations += "   - 建立A/B测试框架\n"
    recommendations += "   - 优化实时推荐性能\n"

    # 合并报告
    full_report = report + data_summary + recommendations

    print("✅ 综合分析报告生成完成")
    print("\n📄 报告主要内容:")
    print("   - 推荐算法性能评估")
    print("   - 数据质量和洞察分析")
    print("   - 冷启动问题分析")
    print("   - 优化建议和改进方向")

    return full_report


def main():
    """主函数：演示高级推荐系统功能"""
    print("=" * 80)
    print("推荐系统技能 - 高级示例")
    print("=" * 80)

    # 1. 初始化组件
    print("\n🚀 初始化推荐系统组件...")
    engine = RecommendationEngine()
    evaluator = RecommenderEvaluator()
    analyzer = DataAnalyzer()
    visualizer = RecommenderVisualizer()

    # 2. 加载和分析数据
    print("\n📊 加载和深度分析数据...")
    data_dir = Path(__file__).parent / "sample_data"
    user_behavior_path = data_dir / "sample_user_behavior.csv"
    item_info_path = data_dir / "sample_item_info.csv"

    user_data, item_data = engine.load_data(str(user_behavior_path), str(item_info_path))

    if user_data is None:
        print("❌ 数据加载失败")
        return

    # 全面数据分析
    print("   - 用户行为分析...")
    user_analysis = analyzer.analyze_user_behavior(user_data)

    print("   - 商品热度分析...")
    item_analysis = analyzer.analyze_item_popularity(user_data, item_data)

    print("   - 数据稀疏度分析...")
    sparsity_analysis = analyzer.calculate_sparsity(engine.user_item_matrix)

    print("   - 冷启动问题检测...")
    cold_start_analysis = analyzer.detect_cold_start(user_data)

    print("   - 用户画像分析...")
    profiling_analysis = analyzer.analyze_user_profiling(user_data, item_data)

    print("   - 数据质量检查...")
    quality_analysis = analyzer.generate_data_quality_report(user_data, item_data)

    # 3. 训练和优化模型
    print("\n🤖 训练和优化推荐模型...")

    # 训练多种推荐算法
    engine.train_user_based_cf(similarity_metric='cosine', normalize=True)
    engine.train_item_based_cf(similarity_metric='cosine')
    engine.train_svd(n_components=30, random_state=42)

    print("✅ 模型训练完成")

    # 4. 高级评估分析
    print("\n📈 高级评估分析...")
    evaluation_results = compare_evaluation_methods(evaluator, engine, engine.user_item_matrix)

    # 5. 用户分群分析
    analyze_user_segments(analyzer, user_data)

    # 6. 冷启动策略演示
    demonstrate_cold_start_strategies(engine, evaluator)

    # 7. 实时推荐模拟
    print("\n⚡ 实时推荐场景模拟...")
    new_ratings = [('P001', 5), ('P015', 4), ('P045', 5)]
    simulate_real_time_recommendation(engine, 'U001', new_ratings)

    # 8. 生成交互式仪表板
    print("\n📊 交互式仪表板演示...")
    target_user = 'U001'
    recommendations = engine.recommend_hybrid(target_user, top_k=20)
    create_interactive_dashboard(visualizer, recommendations, evaluation_results)

    # 9. 生成综合报告
    print("\n📋 生成综合分析报告...")

    # 收集所有分析结果
    all_analysis_results = {
        'user_behavior': user_analysis,
        'item_popularity': item_analysis,
        'sparsity': sparsity_analysis,
        'cold_start': cold_start_analysis,
        'user_profiling': profiling_analysis,
        'data_quality': quality_analysis
    }

    # 生成综合报告
    comprehensive_report = generate_comprehensive_report(
        evaluator, all_analysis_results, evaluation_results
    )

    # 10. 保存高级示例结果
    print("\n💾 保存高级示例结果...")
    output_dir = Path(__file__).parent / "advanced_output"
    output_dir.mkdir(exist_ok=True)

    # 保存综合报告
    report_path = output_dir / "comprehensive_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(comprehensive_report)

    # 保存详细分析结果
    analyzer.save_analysis_results(
        output_dir / "advanced_analysis_results.json",
        format='json'
    )

    # 保存评估结果
    evaluator.save_evaluation_results(
        evaluation_results,
        output_dir / "advanced_evaluation_results.json",
        format='json'
    )

    # 保存高级推荐结果
    advanced_recs = []
    for method in ['user_based_cf', 'item_based_cf', 'svd', 'hybrid']:
        try:
            if method == 'user_based_cf':
                recs = engine.recommend_user_based_cf(target_user, top_k=10)
            elif method == 'item_based_cf':
                recs = engine.recommend_item_based_cf(target_user, top_k=10)
            elif method == 'svd':
                recs = engine.recommend_svd(target_user, top_k=10)
            else:  # hybrid
                recs = engine.recommend_hybrid(target_user, top_k=10)

            for i, (item_id, score) in enumerate(recs, 1):
                advanced_recs.append({
                    'Method': method,
                    'Rank': i,
                    'Item_ID': item_id,
                    'Score': score,
                    'Timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            print(f"   ⚠️ {method} 推荐失败: {str(e)}")

    if advanced_recs:
        recs_df = pd.DataFrame(advanced_recs)
        recs_df.to_csv(output_dir / "advanced_recommendations.csv", index=False, encoding='utf-8-sig')

    print("✅ 高级示例结果已保存")

    # 11. 总结和展望
    print("\n" + "=" * 80)
    print("🎉 推荐系统高级示例完成!")
    print("=" * 80)

    print(f"\n📁 生成的文件:")
    print(f"   - 综合报告: {output_dir}/comprehensive_report.md")
    print(f"   - 高级分析: {output_dir}/advanced_analysis_results.json")
    print(f"   - 高级评估: {output_dir}/advanced_evaluation_results.json")
    print(f"   - 高级推荐: {output_dir}/advanced_recommendations.csv")

    print(f"\n🎯 关键洞察:")
    print(f"   - 数据稀疏度: {sparsity_analysis.get('sparsity_ratio', 0):.2%}")
    print(f"   - 冷启动严重程度: {cold_start_analysis.get('cold_start_severity', '未知')}")
    print(f"   - 最佳评估方法: {'留一法' if evaluation_results.get('precision@5', 0) > 0 else '交叉验证'}")

    if 'user_segments' in profiling_analysis:
        segments = profiling_analysis['user_segments']
        dominant_segment = max(segments.items(), key=lambda x: x[1])[0] if segments else '未知'
        print(f"   - 主要用户群体: {dominant_segment}")

    print(f"\n🚀 未来改进方向:")
    print(f"   1. 引入深度学习推荐模型")
    print(f"   2. 实现真正的实时推荐系统")
    print(f"   3. 构建完整的A/B测试框架")
    print(f"   4. 开发可解释的推荐算法")
    print(f"   5. 集成多模态推荐（文本、图像、音频）")


if __name__ == "__main__":
    main()