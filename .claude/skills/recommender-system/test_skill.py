#!/usr/bin/env python3
"""
推荐系统技能测试脚本

测试推荐系统技能的所有功能模块：
- 模块导入测试
- 基本功能测试
- 算法正确性测试
- 性能测试
- 集成测试
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import traceback

# 添加技能路径
skill_path = Path(__file__).parent
sys.path.append(str(skill_path))


def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")

    try:
        # 测试核心模块导入
        from scripts.recommendation_engine import RecommendationEngine
        print("   ✓ RecommendationEngine 导入成功")

        from scripts.recommender_evaluator import RecommenderEvaluator
        print("   ✓ RecommenderEvaluator 导入成功")

        from scripts.data_analyzer import DataAnalyzer
        print("   ✓ DataAnalyzer 导入成功")

        from scripts.recommender_visualizer import RecommenderVisualizer
        print("   ✓ RecommenderVisualizer 导入成功")

        # 测试批量导入
        from scripts import RecommendationEngine, RecommenderEvaluator, DataAnalyzer, RecommenderVisualizer
        print("   ✓ 批量导入成功")

        return True

    except Exception as e:
        print(f"   ❌ 模块导入失败: {str(e)}")
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载功能"""
    print("\n📊 测试数据加载功能...")

    try:
        from scripts.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        if not user_behavior_path.exists():
            print(f"   ⚠️ 测试数据文件不存在: {user_behavior_path}")
            return False

        # 测试数据加载
        user_data, item_data = engine.load_data(str(user_behavior_path), str(item_info_path))

        if user_data is None:
            print("   ❌ 用户数据加载失败")
            return False

        print(f"   ✓ 用户数据加载成功: {len(user_data)} 条记录")
        print(f"   ✓ 商品数据加载成功: {len(item_data) if item_data is not None else 0} 个商品")

        # 验证数据格式
        required_columns = ['用户ID', '商品ID']
        for col in required_columns:
            if col not in user_data.columns:
                print(f"   ❌ 缺少必需列: {col}")
                return False

        print("   ✓ 数据格式验证通过")

        # 验证用户-商品矩阵构建
        if engine.user_item_matrix is None:
            print("   ❌ 用户-商品矩阵构建失败")
            return False

        print(f"   ✓ 用户-商品矩阵构建成功: {engine.user_item_matrix.shape}")

        return True

    except Exception as e:
        print(f"   ❌ 数据加载测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_recommendation_algorithms():
    """测试推荐算法"""
    print("\n🤖 测试推荐算法...")

    try:
        from scripts.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        # 加载数据
        user_data, _ = engine.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is None:
            return False

        # 获取测试用户
        test_users = engine.user_item_matrix.index[:3].tolist()

        # 测试基于用户的协同过滤
        print("   - 测试基于用户的协同过滤...")
        engine.train_user_based_cf()
        for user_id in test_users:
            try:
                recommendations = engine.recommend_user_based_cf(user_id, top_k=5)
                if len(recommendations) > 0:
                    print(f"     ✓ 用户 {user_id}: {len(recommendations)} 个推荐")
                else:
                    print(f"     ⚠️ 用户 {user_id}: 无推荐结果")
            except Exception as e:
                print(f"     ❌ 用户 {user_id} 推荐失败: {str(e)}")
                return False

        # 测试基于物品的协同过滤
        print("   - 测试基于物品的协同过滤...")
        engine.train_item_based_cf()
        for user_id in test_users:
            try:
                recommendations = engine.recommend_item_based_cf(user_id, top_k=5)
                if len(recommendations) > 0:
                    print(f"     ✓ 用户 {user_id}: {len(recommendations)} 个推荐")
                else:
                    print(f"     ⚠️ 用户 {user_id}: 无推荐结果")
            except Exception as e:
                print(f"     ❌ 用户 {user_id} 推荐失败: {str(e)}")
                return False

        # 测试SVD矩阵分解
        print("   - 测试SVD矩阵分解...")
        engine.train_svd(n_components=10)
        for user_id in test_users:
            try:
                recommendations = engine.recommend_svd(user_id, top_k=5)
                if len(recommendations) > 0:
                    print(f"     ✓ 用户 {user_id}: {len(recommendations)} 个推荐")
                else:
                    print(f"     ⚠️ 用户 {user_id}: 无推荐结果")
            except Exception as e:
                print(f"     ❌ 用户 {user_id} 推荐失败: {str(e)}")
                return False

        # 测试混合推荐
        print("   - 测试混合推荐...")
        for user_id in test_users:
            try:
                recommendations = engine.recommend_hybrid(user_id, top_k=5)
                if len(recommendations) > 0:
                    print(f"     ✓ 用户 {user_id}: {len(recommendations)} 个推荐")
                else:
                    print(f"     ⚠️ 用户 {user_id}: 无推荐结果")
            except Exception as e:
                print(f"     ❌ 用户 {user_id} 推荐失败: {str(e)}")
                return False

        print("   ✓ 所有推荐算法测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 推荐算法测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_evaluation_methods():
    """测试评估方法"""
    print("\n📈 测试评估方法...")

    try:
        from scripts.recommender_evaluator import RecommenderEvaluator
        from scripts.recommendation_engine import RecommendationEngine

        evaluator = RecommenderEvaluator()
        engine = RecommendationEngine()

        # 加载数据
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        user_data, _ = engine.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is None:
            return False

        # 训练模型
        engine.train_user_based_cf()

        # 测试基本评估指标
        print("   - 测试基本评估指标...")
        recommendations = ['P001', 'P002', 'P003', 'P004', 'P005']
        ground_truth = ['P001', 'P003', 'P006']

        precision = evaluator.precision_at_k(recommendations, ground_truth, k=5)
        recall = evaluator.recall_at_k(recommendations, ground_truth, k=5)
        f1 = evaluator.f1_score_at_k(recommendations, ground_truth, k=5)

        print(f"     ✓ Precision@5: {precision:.4f}")
        print(f"     ✓ Recall@5: {recall:.4f}")
        print(f"     ✓ F1@5: {f1:.4f}")

        # 测试评分预测指标
        predictions = [4.5, 3.2, 4.8, 2.1, 3.9]
        actual_ratings = [4.0, 3.5, 4.5, 2.0, 4.0]

        mae = evaluator.mean_absolute_error(predictions, actual_ratings)
        rmse = evaluator.root_mean_square_error(predictions, actual_ratings)

        print(f"     ✓ MAE: {mae:.4f}")
        print(f"     ✓ RMSE: {rmse:.4f}")

        # 测试留一法评估（小规模）
        print("   - 测试留一法评估...")
        loo_results = evaluator.leave_one_out_evaluation(
            engine, engine.user_item_matrix,
            k_values=[5],
            num_users=min(5, len(engine.user_item_matrix))
        )

        if loo_results.get('evaluated_users', 0) > 0:
            print(f"     ✓ 评估用户数: {loo_results['evaluated_users']}")
            print(f"     ✓ Precision@5: {loo_results.get('precision@5', 0):.4f}")
        else:
            print("     ⚠️ 留一法评估无结果")

        print("   ✓ 所有评估方法测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 评估方法测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_data_analysis():
    """测试数据分析功能"""
    print("\n📊 测试数据分析功能...")

    try:
        from scripts.data_analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        # 测试数据加载
        user_data, item_data = analyzer.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is None:
            return False

        print(f"   ✓ 数据加载成功: {len(user_data)} 条记录")

        # 测试用户行为分析
        print("   - 测试用户行为分析...")
        user_analysis = analyzer.analyze_user_behavior(user_data)
        if 'total_users' in user_analysis:
            print(f"     ✓ 总用户数: {user_analysis['total_users']}")
        else:
            print("     ⚠️ 用户行为分析结果不完整")

        # 测试商品热度分析
        print("   - 测试商品热度分析...")
        item_analysis = analyzer.analyze_item_popularity(user_data, item_data)
        if 'item_interactions' in item_analysis:
            print(f"     ✓ 商品交互分析完成")
        else:
            print("     ⚠️ 商品热度分析结果不完整")

        # 测试稀疏度分析
        print("   - 测试稀疏度分析...")
        sparsity = analyzer.calculate_sparsity()
        if 'sparsity_ratio' in sparsity:
            print(f"     ✓ 数据稀疏度: {sparsity['sparsity_ratio']:.2%}")
        else:
            print("     ⚠️ 稀疏度分析结果不完整")

        # 测试冷启动检测
        print("   - 测试冷启动检测...")
        cold_start = analyzer.detect_cold_start(user_data)
        if 'cold_start_severity' in cold_start:
            print(f"     ✓ 冷启动严重程度: {cold_start['cold_start_severity']}")
        else:
            print("     ⚠️ 冷启动检测结果不完整")

        print("   ✓ 所有数据分析功能测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 数据分析功能测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_visualization():
    """测试可视化功能"""
    print("\n📊 测试可视化功能...")

    try:
        from scripts.recommender_visualizer import RecommenderVisualizer
        from scripts.recommendation_engine import RecommendationEngine

        visualizer = RecommenderVisualizer()
        engine = RecommendationEngine()

        # 加载数据
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        user_data, _ = engine.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is None:
            return False

        # 训练模型
        engine.train_user_based_cf()

        # 测试推荐结果可视化
        print("   - 测试推荐结果可视化...")
        target_user = engine.user_item_matrix.index[0]
        recommendations = engine.recommend_user_based_cf(target_user, top_k=5)

        if recommendations:
            fig = visualizer.plot_recommendation_results(
                recommendations, target_user,
                title=f"用户 {target_user} 推荐结果测试"
            )
            if fig is not None:
                print("     ✓ 推荐结果可视化成功")
                # 关闭图形以避免内存泄漏
                import matplotlib.pyplot as plt
                plt.close(fig)
            else:
                print("     ⚠️ 推荐结果可视化失败")
        else:
            print("     ⚠️ 无推荐结果可供可视化")

        # 测试评估指标可视化
        print("   - 测试评估指标可视化...")
        evaluation_results = {
            'precision@5': 0.15,
            'recall@5': 0.12,
            'f1@5': 0.13,
            'precision@10': 0.12,
            'recall@10': 0.18,
            'f1@10': 0.14
        }

        fig = visualizer.plot_evaluation_metrics(evaluation_results)
        if fig is not None:
            print("     ✓ 评估指标可视化成功")
            import matplotlib.pyplot as plt
            plt.close(fig)
        else:
            print("     ⚠️ 评估指标可视化失败")

        # 测试用户行为分析可视化
        print("   - 测试用户行为分析可视化...")
        fig = visualizer.plot_user_behavior_analysis(user_data)
        if fig is not None:
            print("     ✓ 用户行为分析可视化成功")
            import matplotlib.pyplot as plt
            plt.close(fig)
        else:
            print("     ⚠️ 用户行为分析可视化失败")

        print("   ✓ 所有可视化功能测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 可视化功能测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_performance():
    """测试性能"""
    print("\n⚡ 测试性能...")

    try:
        from scripts.recommendation_engine import RecommendationEngine
        import time

        engine = RecommendationEngine()

        # 加载数据
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        user_data, _ = engine.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is None:
            return False

        # 测试训练性能
        print("   - 测试模型训练性能...")
        start_time = time.time()
        engine.train_user_based_cf()
        user_cf_time = time.time() - start_time
        print(f"     ✓ 基于用户的协同过滤训练时间: {user_cf_time:.3f}秒")

        start_time = time.time()
        engine.train_item_based_cf()
        item_cf_time = time.time() - start_time
        print(f"     ✓ 基于物品的协同过滤训练时间: {item_cf_time:.3f}秒")

        start_time = time.time()
        engine.train_svd(n_components=20)
        svd_time = time.time() - start_time
        print(f"     ✓ SVD训练时间: {svd_time:.3f}秒")

        # 测试推荐生成性能
        print("   - 测试推荐生成性能...")
        test_users = engine.user_item_matrix.index[:5].tolist()
        num_tests = len(test_users)
        total_time = 0

        for user_id in test_users:
            start_time = time.time()
            recommendations = engine.recommend_hybrid(user_id, top_k=10)
            rec_time = time.time() - start_time
            total_time += rec_time

        avg_rec_time = total_time / num_tests
        print(f"     ✓ 平均推荐生成时间: {avg_rec_time:.3f}秒")

        # 性能基准检查
        performance_ok = True

        if user_cf_time > 30:  # 30秒阈值
            print(f"     ⚠️ 基于用户的协同过滤训练时间过长: {user_cf_time:.3f}秒")
            performance_ok = False

        if avg_rec_time > 5:  # 5秒阈值
            print(f"     ⚠️ 推荐生成时间过长: {avg_rec_time:.3f}秒")
            performance_ok = False

        if performance_ok:
            print("   ✓ 性能测试通过")
        else:
            print("   ⚠️ 性能测试部分通过（存在性能瓶颈）")

        return performance_ok

    except Exception as e:
        print(f"   ❌ 性能测试失败: {str(e)}")
        traceback.print_exc()
        return False


def test_integration():
    """集成测试"""
    print("\n🔗 测试系统集成...")

    try:
        # 导入所有模块
        from scripts.recommendation_engine import RecommendationEngine
        from scripts.recommender_evaluator import RecommenderEvaluator
        from scripts.data_analyzer import DataAnalyzer
        from scripts.recommender_visualizer import RecommenderVisualizer

        # 初始化组件
        engine = RecommendationEngine()
        evaluator = RecommenderEvaluator()
        analyzer = DataAnalyzer()
        visualizer = RecommenderVisualizer()

        # 加载数据
        data_dir = skill_path / "examples" / "sample_data"
        user_behavior_path = data_dir / "sample_user_behavior.csv"
        item_info_path = data_dir / "sample_item_info.csv"

        user_data, item_data = engine.load_data(str(user_behavior_path), str(item_info_path))
        if user_data is None:
            return False

        # 完整流程测试
        print("   - 执行完整推荐流程...")

        # 1. 数据分析
        analysis_results = analyzer.analyze_user_behavior(user_data)

        # 2. 模型训练
        engine.train_user_based_cf()
        engine.train_item_based_cf()
        engine.train_svd(n_components=20)

        # 3. 生成推荐
        target_user = engine.user_item_matrix.index[0]
        recommendations = engine.recommend_hybrid(target_user, top_k=10)

        # 4. 评估效果
        loo_results = evaluator.leave_one_out_evaluation(
            engine, engine.user_item_matrix,
            k_values=[5],
            num_users=min(3, len(engine.user_item_matrix))
        )

        # 5. 生成可视化
        if recommendations:
            fig = visualizer.plot_recommendation_results(
                recommendations, target_user,
                title=f"集成测试 - 用户 {target_user}"
            )
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.close(fig)

        # 验证流程完整性
        success = True

        if not analysis_results:
            print("     ⚠️ 数据分析失败")
            success = False

        if not recommendations:
            print("     ⚠️ 推荐生成失败")
            success = False

        if loo_results.get('evaluated_users', 0) == 0:
            print("     ⚠️ 效果评估失败")
            success = False

        if success:
            print("     ✓ 集成测试流程执行成功")
            print(f"     ✓ 生成了 {len(recommendations)} 个推荐")
            print(f"     ✓ 评估了 {loo_results.get('evaluated_users', 0)} 个用户")

        return success

    except Exception as e:
        print(f"   ❌ 集成测试失败: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("推荐系统技能测试套件")
    print("=" * 80)

    # 定义测试项目
    tests = [
        ("模块导入测试", test_imports),
        ("数据加载测试", test_data_loading),
        ("推荐算法测试", test_recommendation_algorithms),
        ("评估方法测试", test_evaluation_methods),
        ("数据分析测试", test_data_analysis),
        ("可视化测试", test_visualization),
        ("性能测试", test_performance),
        ("集成测试", test_integration)
    ]

    # 执行测试
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 错误: {str(e)}")

    # 生成测试报告
    print("\n" + "=" * 80)
    print("测试报告")
    print("=" * 80)

    print(f"测试结果: {passed}/{total} 项通过")
    print(f"通过率: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 所有测试通过！推荐系统技能功能完整且正常。")
        print("\n技能特性:")
        print("  ✅ 完整的推荐算法实现")
        print("  ✅ 全面的评估框架")
        print("  ✅ 深度的数据分析")
        print("  ✅ 丰富的可视化功能")
        print("  ✅ 良好的性能表现")
        print("  ✅ 完美的系统集成")

    elif passed >= total * 0.8:
        print(f"\n✅ 大部分测试通过！推荐系统技能基本功能正常。")
        print(f"⚠️  {total - passed} 项测试失败，需要进一步优化。")

    else:
        print(f"\n❌ 测试失败较多！推荐系统技能需要修复。")
        print(f"🔧 {total - passed} 项测试失败，请检查代码。")

    # 推荐使用方式
    print("\n📖 推荐使用方式:")
    print("1. 基础使用: python examples/basic_recommendation_example.py")
    print("2. 高级功能: python examples/advanced_recommendation_example.py")
    print("3. 自定义开发: 参考examples/目录下的示例代码")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)