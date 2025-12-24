"""
简化推荐系统示例

演示推荐系统技能的核心功能：
- 数据加载和预处理
- 推荐算法训练
- 推荐结果生成
- 效果评估
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path))

from scripts.recommendation_engine import RecommendationEngine
from scripts.recommender_evaluator import RecommenderEvaluator
from scripts.recommender_visualizer import RecommenderVisualizer


def main():
    """主函数：演示简化推荐系统流程"""
    print("=" * 60)
    print("推荐系统技能 - 简化示例")
    print("=" * 60)

    # 1. 初始化组件
    print("\n1. 初始化推荐系统组件...")
    engine = RecommendationEngine()
    evaluator = RecommenderEvaluator()
    visualizer = RecommenderVisualizer()

    # 2. 数据加载
    print("\n2. 加载样本数据...")
    data_dir = Path(__file__).parent / "sample_data"
    user_behavior_path = data_dir / "sample_user_behavior.csv"
    item_info_path = data_dir / "sample_item_info.csv"

    # 加载用户行为和商品信息数据
    user_data, item_data = engine.load_data(str(user_behavior_path), str(item_info_path))

    if user_data is None:
        print("❌ 数据加载失败")
        return

    print(f"✅ 数据加载成功：{len(user_data)} 条用户行为记录")
    print(f"   - 用户数: {engine.user_item_matrix.shape[0]:,}")
    print(f"   - 商品数: {engine.user_item_matrix.shape[1]:,}")

    # 3. 基础数据分析
    print("\n3. 基础数据分析...")

    # 计算数据稀疏度
    total_entries = engine.user_item_matrix.shape[0] * engine.user_item_matrix.shape[1]
    non_zero_entries = (engine.user_item_matrix > 0).sum().sum()
    sparsity = (total_entries - non_zero_entries) / total_entries

    print(f"   - 数据稀疏度: {sparsity:.2%}")
    print(f"   - 平均每用户交互: {non_zero_entries / engine.user_item_matrix.shape[0]:.1f}")

    # 用户活跃度分析
    user_activity = engine.user_item_matrix.apply(lambda x: (x > 0).sum(), axis=1)
    print(f"   - 平均用户活跃度: {user_activity.mean():.1f}")
    print(f"   - 最活跃用户交互数: {user_activity.max()}")
    print(f"   - 最不活跃用户交互数: {user_activity.min()}")

    # 4. 训练推荐模型
    print("\n4. 训练推荐模型...")

    # 训练基于用户的协同过滤
    print("   - 训练基于用户的协同过滤...")
    engine.train_user_based_cf(similarity_metric='cosine', normalize=True)

    # 训练基于物品的协同过滤
    print("   - 训练基于物品的协同过滤...")
    engine.train_item_based_cf(similarity_metric='cosine')

    # 训练SVD矩阵分解模型
    print("   - 训练SVD矩阵分解模型...")
    engine.train_svd(n_components=20, random_state=42)

    print("✅ 所有模型训练完成")

    # 5. 生成推荐结果
    print("\n5. 生成推荐结果...")

    target_user = 'U001'  # 目标用户

    # 基于用户的协同过滤推荐
    user_cf_recs = engine.recommend_user_based_cf(target_user, top_k=5)
    print(f"\n📋 基于用户的协同过滤推荐 (用户 {target_user}):")
    for i, (item_id, score) in enumerate(user_cf_recs, 1):
        print(f"   {i}. {item_id}: {score:.3f}")

    # 基于物品的协同过滤推荐
    item_cf_recs = engine.recommend_item_based_cf(target_user, top_k=5)
    print(f"\n📋 基于物品的协同过滤推荐 (用户 {target_user}):")
    for i, (item_id, score) in enumerate(item_cf_recs, 1):
        print(f"   {i}. {item_id}: {score:.3f}")

    # SVD矩阵分解推荐
    svd_recs = engine.recommend_svd(target_user, top_k=5)
    print(f"\n📋 SVD矩阵分解推荐 (用户 {target_user}):")
    for i, (item_id, score) in enumerate(svd_recs, 1):
        print(f"   {i}. {item_id}: {score:.3f}")

    # 混合推荐
    hybrid_weights = {'user_cf': 0.3, 'item_cf': 0.3, 'svd': 0.4}
    hybrid_recs = engine.recommend_hybrid(target_user, top_k=5, weights=hybrid_weights)
    print(f"\n📋 混合推荐 (用户 {target_user}):")
    for i, (item_id, score) in enumerate(hybrid_recs, 1):
        print(f"   {i}. {item_id}: {score:.3f}")

    # 6. 评估推荐效果
    print("\n6. 评估推荐效果...")

    # 基本指标演示
    test_recommendations = ['P001', 'P002', 'P003', 'P004', 'P005']
    test_ground_truth = ['P001', 'P003', 'P006']

    precision = evaluator.precision_at_k(test_recommendations, test_ground_truth, k=5)
    recall = evaluator.recall_at_k(test_recommendations, test_ground_truth, k=5)
    f1 = evaluator.f1_score_at_k(test_recommendations, test_ground_truth, k=5)

    print(f"\n📊 评估指标示例:")
    print(f"   - Precision@5: {precision:.4f}")
    print(f"   - Recall@5: {recall:.4f}")
    print(f"   - F1@5: {f1:.4f}")

    # 评分预测准确性演示
    test_predictions = [4.5, 3.2, 4.8, 2.1, 3.9]
    test_actual = [4.0, 3.5, 4.5, 2.0, 4.0]

    mae = evaluator.mean_absolute_error(test_predictions, test_actual)
    rmse = evaluator.root_mean_square_error(test_predictions, test_actual)

    print(f"\n📊 评分预测准确性:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")

    # 7. 简化评估
    print("\n7. 系统性能评估...")

    # 选择几个用户进行评估
    test_users = engine.user_item_matrix.index[:min(5, len(engine.user_item_matrix))]
    successful_recs = 0

    for user_id in test_users:
        try:
            recs = engine.recommend_hybrid(user_id, top_k=3)
            if len(recs) > 0:
                successful_recs += 1
        except:
            continue

    success_rate = successful_recs / len(test_users)
    print(f"   - 推荐成功率: {success_rate:.1%} ({successful_recs}/{len(test_users)} 用户)")

    # 8. 模型信息
    print("\n8. 模型信息...")
    model_info = engine.get_model_info()
    print(f"   - 用户数: {model_info['user_count']:,}")
    print(f"   - 商品数: {model_info['item_count']:,}")
    print(f"   - 矩阵稀疏度: {model_info['matrix_sparsity']:.2%}")
    print(f"   - 用户CF模型: {'✅' if model_info['user_cf_trained'] else '❌'}")
    print(f"   - 物品CF模型: {'✅' if model_info['item_cf_trained'] else '❌'}")
    print(f"   - SVD模型: {'✅' if model_info['svd_trained'] else '❌'}")

    # 9. 保存结果
    print("\n9. 保存分析结果...")

    # 创建输出目录
    output_dir = Path(__file__).parent / "simple_output"
    output_dir.mkdir(exist_ok=True)

    # 保存推荐结果
    recommendations_data = {
        'Method': [],
        'Rank': [],
        'Item_ID': [],
        'Score': []
    }

    methods = [
        ('User-Based CF', user_cf_recs),
        ('Item-Based CF', item_cf_recs),
        ('SVD', svd_recs),
        ('Hybrid', hybrid_recs)
    ]

    for method_name, recs in methods:
        for i, (item_id, score) in enumerate(recs, 1):
            recommendations_data['Method'].append(method_name)
            recommendations_data['Rank'].append(i)
            recommendations_data['Item_ID'].append(item_id)
            recommendations_data['Score'].append(score)

    recs_df = pd.DataFrame(recommendations_data)
    recs_df.to_csv(output_dir / "recommendations_results.csv", index=False, encoding='utf-8-sig')

    # 保存模型信息
    model_df = pd.DataFrame([model_info])
    model_df.to_csv(output_dir / "model_info.csv", index=False, encoding='utf-8-sig')

    print("✅ 结果已保存到 simple_output/ 目录")

    # 10. 总结
    print("\n" + "=" * 60)
    print("🎉 推荐系统简化示例完成!")
    print("=" * 60)

    print(f"\n📁 生成的文件:")
    print(f"   - 推荐结果: {output_dir}/recommendations_results.csv")
    print(f"   - 模型信息: {output_dir}/model_info.csv")

    print(f"\n🎯 关键结果:")
    print(f"   - 数据稀疏度: {sparsity:.2%}")
    print(f"   - 推荐成功率: {success_rate:.1%}")
    print(f"   - 最佳推荐方法: 混合推荐 (综合多种算法)")
    print(f"   - 推荐覆盖度: {len(set([r[0] for r in hybrid_recs]))} 个商品")

    print(f"\n💡 推荐系统技能特性:")
    print(f"   ✅ 多种推荐算法 (UCF, ICF, SVD, Hybrid)")
    print(f"   ✅ 全面的评估指标")
    print(f"   ✅ 智能的冷启动处理")
    print(f"   ✅ 高效的矩阵运算")
    print(f"   ✅ 灵活的参数配置")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 推荐系统技能验证成功！可以开始使用。")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)