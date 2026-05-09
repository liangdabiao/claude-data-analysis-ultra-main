#!/usr/bin/env python3
"""
LTV预测技能测试验证脚本
全面测试LTV预测技能的各个模块和功能
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# 添加技能模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / 'scripts'))

try:
    from data_processor import DataProcessor
    from regression_models import RegressionModels
    from ltv_predictor import LTVPredictor, complete_ltv_analysis
    from visualizer import DataVisualizer
    from report_generator import generate_comprehensive_report
    from quick_analysis import quick_ltv_analysis
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

class TestLTVPredictor(unittest.TestCase):
    """LTV预测技能测试类"""

    @classmethod
    def setUpClass(cls):
        """测试环境设置"""
        cls.sample_data_path = current_dir.parent / 'data' / 'sample_orders.csv'
        cls.test_output_dir = current_dir.parent / 'examples' / 'test_results'
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)

        if not cls.sample_data_path.exists():
            cls.skipTest(cls, "示例数据文件不存在")

    def setUp(self):
        """每个测试前的设置"""
        self.test_data = pd.read_csv(self.sample_data_path)

    def test_data_processor(self):
        """测试数据处理器"""
        print("\n🧪 测试数据处理器...")

        processor = DataProcessor()

        # 测试数据加载
        loaded_data = processor.load_order_data(str(self.sample_data_path))
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), 50)  # 示例数据有50行

        # 测试RFM特征计算
        rfm_data = processor.calculate_rfm_features(
            loaded_data,
            feature_period_months=3,
            prediction_period_months=12
        )
        self.assertIsNotNone(rfm_data)
        self.assertTrue(len(rfm_data) > 0)

        # 检查RFM特征列是否存在
        expected_columns = ['R值', 'F值', 'M值', '年度LTV']
        for col in expected_columns:
            self.assertIn(col, rfm_data.columns, f"缺少RFM特征列: {col}")

        print("✅ 数据处理器测试通过")

    def test_regression_models(self):
        """测试回归模型"""
        print("\n🧪 测试回归模型...")

        # 创建测试数据
        X = np.random.rand(100, 3)  # 100个样本，3个特征
        y = X[:, 0] * 2 + X[:, 1] * 3 + X[:, 2] * 1 + np.random.randn(100) * 0.1

        models = RegressionModels()

        # 测试线性回归
        linear_model = models.train_linear_regression(X, y)
        self.assertIsNotNone(linear_model)

        # 测试随机森林
        rf_model = models.train_random_forest(X, y)
        self.assertIsNotNone(rf_model)

        # 测试模型评估
        evaluation = models.evaluate_model(linear_model, X, y)
        self.assertIn('r2_score', evaluation)
        self.assertIn('mae', evaluation)
        self.assertIn('rmse', evaluation)

        print("✅ 回归模型测试通过")

    def test_ltv_predictor_integration(self):
        """测试LTV预测器集成"""
        print("\n🧪 测试LTV预测器集成...")

        try:
            # 快速配置以节省测试时间
            test_config = {
                'data_processor_config': {
                    'feature_period_months': 3,
                    'prediction_period_months': 6,  # 缩短预测期
                    'remove_outliers': False,
                    'min_orders_per_customer': 1
                },
                'regression_config': {
                    'test_size': 0.3,
                    'cv_folds': 3,
                    'scoring_metric': 'r2',
                    'enable_hyperparameter_tuning': False
                },
                'models_to_train': ['linear_regression']  # 只测试线性回归
            }

            # 执行完整分析
            results = complete_ltv_analysis(
                str(self.sample_data_path),
                output_dir=str(self.test_output_dir / 'integration_test'),
                config=test_config
            )

            # 验证结果结构
            self.assertIn('predictor', results)
            self.assertIn('rfm_data', results)
            self.assertIn('training_results', results)
            self.assertIn('summary_report', results)
            self.assertIn('output_paths', results)

            # 验证RFM数据
            rfm_data = results['rfm_data']
            self.assertIsInstance(rfm_data, pd.DataFrame)
            self.assertTrue(len(rfm_data) > 0)

            # 验证模型结果
            training_results = results['training_results']
            self.assertIn('model_results', training_results)

            print("✅ LTV预测器集成测试通过")

        except Exception as e:
            self.fail(f"LTV预测器集成测试失败: {str(e)}")

    def test_visualizer(self):
        """测试可视化模块"""
        print("\n🧪 测试可视化模块...")

        try:
            # 创建测试数据
            test_rfm = pd.DataFrame({
                'R值': np.random.randint(1, 365, 50),
                'F值': np.random.randint(1, 20, 50),
                'M值': np.random.uniform(100, 5000, 50),
                '年度LTV': np.random.uniform(500, 5000, 50),
                '客户价值分层': np.random.choice(['青铜客户', '白银客户', '黄金客户', '白金客户'], 50)
            })

            # 创建模型结果
            model_results = {
                'linear_regression': {
                    'r2_score': 0.75,
                    'mae': 250.5,
                    'rmse': 350.2,
                    'mape': 15.3
                }
            }

            visualizer = DataVisualizer()

            # 测试RFM分布图
            chart_path = self.test_output_dir / 'test_rfm_distribution.png'
            visualizer.plot_rfm_distribution(
                test_rfm, save_path=str(chart_path), show_plot=False
            )
            self.assertTrue(chart_path.exists())

            # 测试模型性能图
            perf_chart_path = self.test_output_dir / 'test_model_performance.png'
            visualizer.plot_model_performance(
                model_results, save_path=str(perf_chart_path), show_plot=False
            )
            self.assertTrue(perf_chart_path.exists())

            print("✅ 可视化模块测试通过")

        except Exception as e:
            self.fail(f"可视化模块测试失败: {str(e)}")

    def test_quick_analysis(self):
        """测试快速分析功能"""
        print("\n🧪 测试快速分析功能...")

        try:
            # 使用最小配置进行快速测试
            results = quick_ltv_analysis(
                file_path=str(self.sample_data_path),
                feature_period_months=3,
                prediction_period_months=6,
                output_dir=str(self.test_output_dir / 'quick_test'),
                generate_charts=False,  # 不生成图表以节省时间
                generate_reports=False   # 不生成报告以节省时间
            )

            # 验证结果
            self.assertIsNotNone(results)
            self.assertIn('summary', results)

            summary = results['summary']
            self.assertIn('data_summary', summary)
            self.assertIn('model_summary', summary)

            print("✅ 快速分析功能测试通过")

        except Exception as e:
            self.fail(f"快速分析功能测试失败: {str(e)}")

    def test_data_quality(self):
        """测试数据质量验证"""
        print("\n🧪 测试数据质量验证...")

        # 检查示例数据结构
        expected_columns = ['订单号', '产品码', '消费日期', '产品说明', '数量', '单价', '用户码', '城市']
        for col in expected_columns:
            self.assertIn(col, self.test_data.columns, f"缺少必要列: {col}")

        # 检查数据类型
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['订单号']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['数量']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['单价']))

        # 检查数据完整性
        self.assertFalse(self.test_data['订单号'].duplicated().any(), "存在重复订单号")
        self.assertTrue(self.test_data['消费日期'].notna().all(), "存在空消费日期")
        self.assertTrue(self.test_data['用户码'].notna().all(), "存在空用户码")

        # 检查数值合理性
        self.assertTrue((self.test_data['数量'] > 0).all(), "存在非正数量")
        self.assertTrue((self.test_data['单价'] > 0).all(), "存在非正单价")

        print("✅ 数据质量验证通过")

def run_performance_test():
    """运行性能测试"""
    print("\n🚀 运行性能测试...")

    import time

    sample_data_path = current_dir.parent / 'data' / 'sample_orders.csv'
    test_output_dir = current_dir.parent / 'examples' / 'performance_test'

    if not sample_data_path.exists():
        print("❌ 示例数据文件不存在，跳过性能测试")
        return

    # 测试快速分析性能
    start_time = time.time()

    try:
        results = quick_ltv_analysis(
            file_path=str(sample_data_path),
            feature_period_months=3,
            prediction_period_months=6,
            output_dir=str(test_output_dir),
            generate_charts=False,
            generate_reports=False
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"✅ 性能测试完成")
        print(f"   - 执行时间: {execution_time:.2f}秒")
        print(f"   - 处理数据: {len(pd.read_csv(sample_data_path))}行")
        print(f"   - 分析客户: {results['summary']['data_summary']['total_customers']}人")

        # 性能基准
        if execution_time < 30:
            print("🎉 性能表现优秀！")
        elif execution_time < 60:
            print("👍 性能表现良好！")
        else:
            print("⚠️ 性能有待优化")

    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("🔬 LTV预测技能全面测试")
    print("=" * 60)

    # 环境检查
    print("1️⃣ 环境检查...")
    print("-" * 30)

    # 检查依赖包
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("请安装缺少的包后重新运行测试")
        return

    # 检查文件
    sample_data_path = current_dir.parent / 'data' / 'sample_orders.csv'
    if sample_data_path.exists():
        print(f"✅ 示例数据: {sample_data_path}")
    else:
        print(f"❌ 示例数据不存在: {sample_data_path}")
        return

    # 运行单元测试
    print("\n2️⃣ 运行单元测试...")
    print("-" * 30)

    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestLTVPredictor)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)

    # 输出测试结果摘要
    print(f"\n📊 测试结果摘要:")
    print(f"   - 总测试数: {test_result.testsRun}")
    print(f"   - 成功: {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}")
    print(f"   - 失败: {len(test_result.failures)}")
    print(f"   - 错误: {len(test_result.errors)}")

    if test_result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in test_result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if test_result.errors:
        print("\n❌ 错误的测试:")
        for test, traceback in test_result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")

    # 运行性能测试
    run_performance_test()

    # 最终总结
    print("\n" + "=" * 60)
    if test_result.wasSuccessful():
        print("🎉 所有测试通过！LTV预测技能功能正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    print("=" * 60)

if __name__ == "__main__":
    main()