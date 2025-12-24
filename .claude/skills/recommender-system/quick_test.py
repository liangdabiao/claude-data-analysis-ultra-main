#!/usr/bin/env python3
"""
快速测试推荐系统核心功能
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent
sys.path.append(str(skill_path))

def main():
    """快速测试主要功能"""
    print("🚀 推荐系统技能快速测试")

    try:
        # 1. 测试导入
        print("\n1. 测试模块导入...")
        from scripts.recommendation_engine import RecommendationEngine
        from scripts.recommender_evaluator import RecommenderEvaluator
        print("   ✓ 核心模块导入成功")

        # 2. 测试数据加载和模型训练
        print("\n2. 测试推荐引擎...")
        engine = RecommendationEngine()

        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        user_data, item_data = engine.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is not None:
            print(f"   ✓ 数据加载成功: {len(user_data)} 条记录")

            # 训练模型
            engine.train_user_based_cf()
            engine.train_item_based_cf()
            engine.train_svd(n_components=10)
            print("   ✓ 模型训练成功")

            # 生成推荐
            target_user = engine.user_item_matrix.index[0]
            recommendations = engine.recommend_hybrid(target_user, top_k=5)
            print(f"   ✓ 推荐生成成功: {len(recommendations)} 个推荐")

            # 显示推荐结果
            print(f"\n🎯 用户 {target_user} 的推荐结果:")
            for i, (item_id, score) in enumerate(recommendations, 1):
                print(f"   {i}. {item_id}: {score:.3f}")

        # 3. 测试评估
        print("\n3. 测试评估功能...")
        evaluator = RecommenderEvaluator()

        # 测试基本评估指标
        recs = ['P001', 'P002', 'P003', 'P004', 'P005']
        truth = ['P001', 'P003', 'P006']

        precision = evaluator.precision_at_k(recs, truth, k=5)
        recall = evaluator.recall_at_k(recs, truth, k=5)

        print(f"   ✓ Precision@5: {precision:.4f}")
        print(f"   ✓ Recall@5: {recall:.4f}")

        print("\n🎉 核心功能测试通过！")
        print("\n推荐系统技能已就绪，可以使用以下命令运行完整示例:")
        print("   python examples/basic_recommendation_example.py")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)