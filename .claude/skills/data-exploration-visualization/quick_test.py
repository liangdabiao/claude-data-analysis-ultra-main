#!/usr/bin/env python3
"""
数据探索可视化技能快速测试
快速验证所有核心功能是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback

# 添加技能路径
skill_path = Path(__file__).parent
sys.path.append(str(skill_path / "scripts"))

def create_test_data():
    """创建测试数据"""
    print("📊 创建测试数据...")

    np.random.seed(42)
    n_samples = 200

    data = {
        'id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'score': np.random.normal(75, 15, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # 添加一些缺失值
    missing_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[missing_indices, 'income'] = np.nan

    print(f"   ✓ 测试数据创建完成: {df.shape}")
    return df

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")

    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'plotly': 'plotly',
        'jinja2': 'jinja2'
    }

    optional_packages = {
        'shap': 'shap',
        'lightgbm': 'lightgbm',
        'imblearn': 'imbalanced-learn',
        'weasyprint': 'weasyprint'
    }

    missing_required = []
    missing_optional = []

    # 检查必需包
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_required.append(package_name)

    # 检查可选包
    for module_name, package_name in optional_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_optional.append(package_name)

    if missing_required:
        print("   ❌ 缺少必需依赖包:")
        for package in missing_required:
            print(f"      - {package}")
        print(f"\n   请安装: pip install {' '.join(missing_required)}")
        return False

    if missing_optional:
        print("   ⚠️ 缺少可选依赖包 (某些功能可能不可用):")
        for package in missing_optional:
            print(f"      - {package}")

    print("   ✓ 所有必要依赖包已安装")
    return True

def test_eda_analyzer():
    """测试EDA分析器"""
    print("\n🔍 测试EDA分析器...")

    try:
        from eda_analyzer import EDAAnalyzer

        # 创建分析器
        analyzer = EDAAnalyzer()

        # 创建测试数据
        data = create_test_data()

        # 测试基本功能
        print("   测试数据质量检查...")
        quality_report = analyzer.data_quality_check(data)
        assert quality_report is not None, "数据质量检查失败"
        print("     ✓ 数据质量检查")

        print("   测试统计摘要...")
        stats_summary = analyzer.generate_statistical_summary(data)
        assert stats_summary is not None, "统计摘要生成失败"
        print("     ✓ 统计摘要生成")

        print("   测试相关性分析...")
        corr_matrix = analyzer.correlation_analysis(data)
        assert corr_matrix is not None, "相关性分析失败"
        print("     ✓ 相关性分析")

        print("   测试自动EDA...")
        eda_results = analyzer.auto_eda(data)
        assert eda_results is not None, "自动EDA失败"
        print("     ✓ 自动EDA分析")

        print("   ✓ EDA分析器测试通过")
        return True, eda_results

    except Exception as e:
        print(f"   ❌ EDA分析器测试失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_visualizer():
    """测试可视化器"""
    print("\n📈 测试可视化器...")

    try:
        from visualizer import DataVisualizer

        # 创建可视化器
        visualizer = DataVisualizer()

        # 创建测试数据
        data = create_test_data()

        # 测试分布图
        print("   测试分布图...")
        fig = visualizer.plot_distribution(data, 'age', interactive=False)
        assert fig is not None, "分布图生成失败"
        print("     ✓ 分布图生成")

        # 测试相关性热图
        print("   测试相关性热图...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig = visualizer.plot_correlation(data, numeric_cols, interactive=False)
            assert fig is not None, "相关性热图生成失败"
            print("     ✓ 相关性热图生成")

        # 测试散点图
        print("   测试散点图...")
        if len(numeric_cols) >= 2:
            fig = visualizer.plot_scatter(data, numeric_cols[0], numeric_cols[1], interactive=False)
            assert fig is not None, "散点图生成失败"
            print("     ✓ 散点图生成")

        # 测试自动可视化
        print("   测试自动可视化...")
        charts = visualizer.auto_visualize(
            data,
            target_col='target',
            save_charts=False
        )
        assert charts is not None, "自动可视化失败"
        print("     ✓ 自动可视化")

        print("   ✓ 可视化器测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 可视化器测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_preprocessor():
    """测试数据预处理器"""
    print("\n🧹 测试数据预处理器...")

    try:
        from data_preprocessor import DataPreprocessor

        # 创建预处理器
        preprocessor = DataPreprocessor({
            'missing_threshold': 0.5,
            'feature_selection': False,  # 关闭特征选择加快测试
            'test_size': 0.2
        })

        # 创建测试数据
        data = create_test_data()

        # 测试数据质量分析
        print("   测试数据质量分析...")
        quality_report = preprocessor.analyze_data_quality(data)
        assert quality_report is not None, "数据质量分析失败"
        print("     ✓ 数据质量分析")

        # 测试数据清洗
        print("   测试数据清洗...")
        cleaned_data = preprocessor.clean_data(data)
        assert cleaned_data is not None, "数据清洗失败"
        print("     ✓ 数据清洗")

        # 测试类型转换
        print("   测试类型转换...")
        transformed_data = preprocessor.transform_data_types(cleaned_data)
        assert transformed_data is not None, "类型转换失败"
        print("     ✓ 类型转换")

        # 测试自动预处理
        print("   测试自动预处理...")
        results = preprocessor.auto_preprocess(data, target_col='target', save_report=False)
        assert results is not None, "自动预处理失败"
        assert 'preprocessed_data' in results, "预处理数据缺失"
        print("     ✓ 自动预处理")

        print("   ✓ 数据预处理器测试通过")
        return True, results

    except Exception as e:
        print(f"   ❌ 数据预处理器测试失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_modeling_evaluator():
    """测试建模评估器"""
    print("\n🤖 测试建模评估器...")

    try:
        from modeling_evaluator import ModelingEvaluator

        # 创建建模器
        modeler = ModelingEvaluator({
            'cv_folds': 3,  # 减少折数加快测试
            'enable_hyperparameter_tuning': False,  # 关闭调加快测试
            'n_iter_search': 5
        })

        # 创建测试数据
        data = create_test_data()

        # 测试单个模型训练
        print("   测试单个模型训练...")
        # 准备数据
        X = data[['age', 'income', 'score']]
        y = data['target']

        # 分割数据
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        result = modeler.train_single_model(
            X_train, y_train, X_test, y_test,
            'logistic_regression', tune_hyperparameters=False
        )
        assert result is not None, "单个模型训练失败"
        print("     ✓ 单个模型训练")

        # 测试多模型训练
        print("   测试多模型训练...")
        results = modeler.train_multiple_models(
            X_train, y_train, X_test, y_test,
            model_names=['logistic_regression', 'random_forest']
        )
        assert results is not None, "多模型训练失败"
        print("     ✓ 多模型训练")

        # 测试自动建模
        print("   测试自动建模...")
        modeling_results = modeler.auto_modeling(
            data,
            target_col='target',
            model_names=['logistic_regression', 'random_forest']
        )
        assert modeling_results is not None, "自动建模失败"
        print("     ✓ 自动建模")

        print("   ✓ 建模评估器测试通过")
        return True, modeling_results

    except Exception as e:
        print(f"   ❌ 建模评估器测试失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_report_generator():
    """测试报告生成器"""
    print("\n📋 测试报告生成器...")

    try:
        from report_generator import ReportGenerator

        # 创建报告生成器
        generator = ReportGenerator({
            'report_title': '测试报告',
            'author': '测试用户'
        })

        # 创建测试数据
        data = create_test_data()

        # 测试快速报告生成
        print("   测试快速报告生成...")
        report_path = generator.generate_quick_report(
            data=data,
            target_col='target',
            output_path='test_quick_report.html'
        )
        assert report_path is not None, "快速报告生成失败"
        assert os.path.exists(report_path), "报告文件不存在"
        print("     ✓ 快速报告生成")

        # 测试综合报告生成
        print("   测试综合报告生成...")
        # 创建模拟的EDA和建模结果
        eda_results = {
            'data_quality': {'overall_score': 85.5},
            'insights': ['测试洞察1', '测试洞察2']
        }

        model_results = {
            'best_model': {
                'name': 'logistic_regression',
                'metrics': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85}
            }
        }

        comprehensive_path = generator.generate_comprehensive_report(
            data=data,
            eda_results=eda_results,
            model_results=model_results,
            output_path='test_comprehensive_report.html',
            format='html'
        )
        assert comprehensive_path is not None, "综合报告生成失败"
        assert os.path.exists(comprehensive_path), "综合报告文件不存在"
        print("     ✓ 综合报告生成")

        print("   ✓ 报告生成器测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 报告生成器测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_integration():
    """集成测试"""
    print("\n🔗 测试模块集成...")

    try:
        # 创建测试数据
        data = create_test_data()

        # 1. EDA分析
        print("   1. 执行EDA分析...")
        from eda_analyzer import EDAAnalyzer
        analyzer = EDAAnalyzer()
        eda_results = analyzer.auto_eda(data)

        # 2. 数据预处理
        print("   2. 执行数据预处理...")
        from data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessing_results = preprocessor.auto_preprocess(data, target_col='target')

        # 3. 可视化
        print("   3. 生成可视化...")
        from visualizer import DataVisualizer
        visualizer = DataVisualizer()
        charts = visualizer.auto_visualize(data, target_col='target', save_charts=False)

        # 4. 建模
        print("   4. 执行建模...")
        from modeling_evaluator import ModelingEvaluator
        modeler = ModelingEvaluator({'enable_hyperparameter_tuning': False})
        model_results = modeler.auto_modeling(data, target_col='target', model_names=['logistic_regression'])

        # 5. 报告生成
        print("   5. 生成报告...")
        from report_generator import ReportGenerator
        generator = ReportGenerator()
        report_path = generator.generate_comprehensive_report(
            data=data,
            eda_results=eda_results,
            model_results=model_results,
            output_path='test_integration_report.html'
        )

        # 验证所有步骤都成功
        assert eda_results is not None, "EDA分析失败"
        assert preprocessing_results is not None, "数据预处理失败"
        assert charts is not None, "可视化失败"
        assert model_results is not None, "建模失败"
        assert report_path is not None and os.path.exists(report_path), "报告生成失败"

        print("   ✓ 集成测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 集成测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_performance():
    """性能测试"""
    print("\n⚡ 性能测试...")

    try:
        import time

        # 创建较大的测试数据
        print("   创建性能测试数据...")
        n_samples = 1000
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })

        # 测试EDA性能
        print("   测试EDA性能...")
        start_time = time.time()
        from eda_analyzer import EDAAnalyzer
        analyzer = EDAAnalyzer()
        analyzer.auto_eda(data)
        eda_time = time.time() - start_time
        print(f"     ✓ EDA耗时: {eda_time:.2f}秒")

        # 测试建模性能
        print("   测试建模性能...")
        start_time = time.time()
        from modeling_evaluator import ModelingEvaluator
        modeler = ModelingEvaluator({'enable_hyperparameter_tuning': False})
        modeler.auto_modeling(data, target_col='target', model_names=['logistic_regression'])
        modeling_time = time.time() - start_time
        print(f"     ✓ 建模耗时: {modeling_time:.2f}秒")

        # 性能断言
        assert eda_time < 30, f"EDA耗时过长: {eda_time}秒"
        assert modeling_time < 60, f"建模耗时过长: {modeling_time}秒"

        print("   ✓ 性能测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 性能测试失败: {str(e)}")
        return False

def cleanup_test_files():
    """清理测试文件"""
    print("\n🧹 清理测试文件...")

    test_files = [
        'test_quick_report.html',
        'test_comprehensive_report.html',
        'test_integration_report.html',
        'quick_start_charts',
        'quick_start_output'
    ]

    cleaned = 0
    for file_path in test_files:
        path = Path(file_path)
        try:
            if path.is_file():
                path.unlink()
                cleaned += 1
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
                cleaned += 1
        except:
            pass

    print(f"   ✓ 清理了 {cleaned} 个测试文件")

def main():
    """主测试函数"""
    print("🧪 数据探索可视化技能 - 快速测试")
    print("=" * 60)

    start_time = time.time()

    # 测试结果记录
    test_results = {
        'dependencies': False,
        'eda_analyzer': False,
        'visualizer': False,
        'preprocessor': False,
        'modeling_evaluator': False,
        'report_generator': False,
        'integration': False,
        'performance': False
    }

    try:
        # 1. 检查依赖
        test_results['dependencies'] = check_dependencies()
        if not test_results['dependencies']:
            print("\n❌ 依赖检查失败，无法继续测试")
            return

        # 2. 测试各个模块
        test_results['eda_analyzer'], eda_results = test_eda_analyzer()
        test_results['visualizer'] = test_visualizer()
        test_results['preprocessor'], preprocessing_results = test_preprocessor()
        test_results['modeling_evaluator'], modeling_results = test_modeling_evaluator()
        test_results['report_generator'] = test_report_generator()

        # 3. 集成测试
        test_results['integration'] = test_integration()

        # 4. 性能测试
        test_results['performance'] = test_performance()

        # 5. 生成测试报告
        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("📋 测试结果摘要")
        print("=" * 60)

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        for test_name, result in test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:20} : {status}")

        print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
        print(f"测试耗时: {total_time:.2f}秒")

        if passed_tests == total_tests:
            print("\n🎉 所有测试通过！数据探索可视化技能已就绪。")
            print("\n💡 下一步:")
            print("   1. 运行 examples/quick_start_example.py 体验完整功能")
            print("   2. 运行 examples/medical_data_analysis.py 查看医疗数据示例")
            print("   3. 运行 examples/financial_data_analysis.py 查看金融数据示例")
        else:
            failed_tests = [name for name, result in test_results.items() if not result]
            print(f"\n⚠️  {len(failed_tests)} 个测试失败: {', '.join(failed_tests)}")
            print("   请检查错误信息并修复问题后重新测试。")

    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n\n💥 测试过程中发生异常: {str(e)}")
        traceback.print_exc()
    finally:
        # 清理测试文件
        cleanup_test_files()

if __name__ == "__main__":
    main()