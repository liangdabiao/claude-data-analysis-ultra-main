#!/usr/bin/env python3
"""
快速开始示例
演示如何快速使用数据探索可视化技能进行数据分析
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path / "scripts"))

def create_sample_data():
    """创建示例数据集"""
    print("📊 创建示例数据集...")

    np.random.seed(42)
    n_samples = 800

    data = {
        'id': range(1, n_samples + 1),
        'age': np.random.randint(18, 75, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
        'income': np.random.lognormal(10.5, 0.6, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'],
                                   n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'experience_years': np.random.exponential(8, n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples),
        'performance_score': np.random.normal(75, 15, n_samples),
        'team_size': np.random.randint(2, 15, n_samples),
        'hours_per_week': np.random.normal(40, 5, n_samples),
        'projects_completed': np.random.poisson(8, n_samples),
        'training_hours': np.random.randint(0, 100, n_samples),
        'salary': np.random.lognormal(10.8, 0.4, n_samples),
    }

    # 创建相关性
    df = pd.DataFrame(data)

    # 经验与薪资的相关性
    df['salary'] = df['salary'] * (0.7 + 0.3 * df['experience_years'] / df['experience_years'].max())

    # 绩效与满意度相关性
    df['performance_score'] = df['performance_score'] + df['satisfaction'] * 3

    # 项目完成数量与经验相关性
    df['projects_completed'] = df['projects_completed'] + (df['experience_years'] / 2).astype(int)

    # 创建目标变量：高绩效员工（绩效分数 > 85）
    df['high_performer'] = (df['performance_score'] > 85).astype(int)

    # 添加一些缺失值
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'training_hours'] = np.nan

    print(f"   ✓ 数据集创建完成: {df.shape}")
    print(f"   ✓ 高绩效员工比例: {df['high_performer'].mean():.2%}")

    return df

def quick_eda_example():
    """快速EDA示例"""
    print("\n🔍 快速EDA分析示例...")

    # 创建数据
    data = create_sample_data()

    try:
        from eda_analyzer import EDAAnalyzer

        # 初始化分析器
        analyzer = EDAAnalyzer()

        # 自动化EDA分析
        print("   执行自动化EDA...")
        results = analyzer.auto_eda(data)

        print(f"   ✓ EDA分析完成")
        print(f"   - 数据质量分数: {results.get('data_quality', {}).get('overall_score', 'N/A')}")
        print(f"   - 发现的洞察: {len(results.get('insights', []))}")

        # 显示前3个洞察
        insights = results.get('insights', [])[:3]
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")

        return data, results

    except Exception as e:
        print(f"   ❌ EDA分析失败: {str(e)}")
        return data, None

def quick_visualization_example(data):
    """快速可视化示例"""
    print("\n📈 快速可视化示例...")

    try:
        from visualizer import DataVisualizer

        # 初始化可视化器
        visualizer = DataVisualizer()

        # 自动生成图表
        print("   自动生成可视化图表...")
        charts = visualizer.auto_visualize(
            data,
            target_col='high_performer',
            save_charts=True,
            output_dir='quick_start_charts'
        )

        print(f"   ✓ 生成了 {charts['charts_generated']} 个图表")

        # 生成特定图表
        print("   生成特定图表...")

        # 1. 年龄分布
        age_chart = visualizer.plot_distribution(data, 'age', interactive=False)
        print("     ✓ 年龄分布图")

        # 2. 薪资 vs 经验散点图
        scatter_chart = visualizer.plot_scatter(
            data, 'experience_years', 'salary',
            color_col='high_performer',
            interactive=False
        )
        print("     ✓ 薪资-经验散点图")

        # 3. 教育水平分布
        education_chart = visualizer.plot_categorical(data, 'education', interactive=False)
        print("     ✓ 教育水平分布图")

        return charts

    except Exception as e:
        print(f"   ❌ 可视化失败: {str(e)}")
        return None

def quick_preprocessing_example(data):
    """快速数据预处理示例"""
    print("\n🧹 快速数据预处理示例...")

    try:
        from data_preprocessor import DataPreprocessor

        # 初始化预处理器
        preprocessor = DataPreprocessor({
            'missing_threshold': 0.3,
            'feature_selection': False,  # 关闭特征选择以加快演示
            'test_size': 0.2
        })

        # 自动预处理
        print("   执行自动预处理...")
        results = preprocessor.auto_preprocess(
            data,
            target_col='high_performer',
            save_report=True
        )

        print(f"   ✓ 预处理完成")
        print(f"   - 原始数据: {results['original_data'].shape}")
        print(f"   - 预处理后: {results['preprocessed_data'].shape}")
        print(f"   - 预处理步骤: {len(results['preprocessing_steps'])}")

        # 显示预处理步骤
        for step in results['preprocessing_steps'][:5]:
            print(f"     - {step}")

        return results

    except Exception as e:
        print(f"   ❌ 预处理失败: {str(e)}")
        return None

def quick_modeling_example(data):
    """快速建模示例"""
    print("\n🤖 快速建模示例...")

    try:
        from modeling_evaluator import ModelingEvaluator

        # 初始化建模器
        modeler = ModelingEvaluator({
            'cv_folds': 3,  # 减少折数加快演示
            'enable_hyperparameter_tuning': False,  # 关闭调参加快演示
            'n_iter_search': 5
        })

        # 自动建模
        print("   执行自动建模...")
        results = modeler.auto_modeling(
            data,
            target_col='high_performer',
            model_names=['logistic_regression', 'random_forest']  # 使用较少的模型
        )

        print(f"   ✓ 建模完成")
        print(f"   - 训练模型数: {len(results['model_results'])}")
        print(f"   - 最佳模型: {results['best_model']['name']}")

        best_metrics = results['best_model']['metrics']
        print(f"   - 最佳准确率: {best_metrics.get('accuracy', 0):.3f}")

        return results

    except Exception as e:
        print(f"   ❌ 建模失败: {str(e)}")
        return None

def quick_report_example(data, eda_results=None, model_results=None):
    """快速报告生成示例"""
    print("\n📋 快速报告生成示例...")

    try:
        from report_generator import ReportGenerator

        # 初始化报告生成器
        generator = ReportGenerator({
            'report_title': '员工绩效分析报告',
            'author': '数据分析助手',
            'include_toc': True,
            'include_summary': True
        })

        # 生成快速报告
        print("   生成快速分析报告...")
        report_path = generator.generate_quick_report(
            data=data,
            target_col='high_performer',
            output_path='quick_analysis_report.html'
        )

        print(f"   ✓ 报告已生成: {report_path}")

        # 如果有完整结果，生成综合报告
        if eda_results or model_results:
            print("   生成综合分析报告...")
            comprehensive_path = generator.generate_comprehensive_report(
                data=data,
                eda_results=eda_results,
                model_results=model_results,
                output_path='comprehensive_analysis_report.html',
                format='html'
            )

            print(f"   ✓ 综合报告已生成: {comprehensive_path}")

        return report_path

    except Exception as e:
        print(f"   ❌ 报告生成失败: {str(e)}")
        return None

def run_complete_pipeline():
    """运行完整分析流程"""
    print("🚀 运行完整数据分析流程...")

    # 创建输出目录
    output_dir = Path('quick_start_output')
    output_dir.mkdir(exist_ok=True)

    # 1. 数据创建和EDA
    data, eda_results = quick_eda_example()

    # 2. 可视化
    charts = quick_visualization_example(data)

    # 3. 预处理
    preprocessing_results = quick_preprocessing_example(data)

    # 4. 建模
    model_results = quick_modeling_example(data)

    # 5. 报告生成
    report_path = quick_report_example(data, eda_results, model_results)

    # 6. 总结
    print("\n🎉 快速开始示例完成！")

    # 显示统计信息
    print(f"\n📊 数据摘要:")
    print(f"   - 样本数量: {len(data):,}")
    print(f"   - 特征数量: {len(data.columns)}")
    print(f"   - 高绩效员工: {data['high_performer'].sum()} ({data['high_performer'].mean():.1%})")

    print(f"\n📈 分析结果:")
    if eda_results:
        quality_score = eda_results.get('data_quality', {}).get('overall_score', 0)
        print(f"   - 数据质量分数: {quality_score:.1f}")

    if model_results:
        best_accuracy = model_results['best_model']['metrics'].get('accuracy', 0)
        print(f"   - 最佳模型准确率: {best_accuracy:.3f}")

    if charts:
        print(f"   - 生成图表数: {charts['charts_generated']}")

    # 显示生成的文件
    print(f"\n📁 生成的文件:")
    for file_path in Path('.').glob('quick_start_*'):
        if file_path.is_file():
            print(f"   📄 {file_path}")

    if report_path:
        print(f"\n📋 分析报告: {report_path}")
        print("   请在浏览器中打开HTML文件查看完整报告。")

def demonstrate_specific_features():
    """演示特定功能"""
    print("\n🔧 演示特定功能...")

    try:
        # 1. 数据质量检查
        from eda_analyzer import EDAAnalyzer
        data = create_sample_data()
        analyzer = EDAAnalyzer()

        print("   1. 数据质量检查...")
        quality_report = analyzer.data_quality_check(data)
        print(f"      - 数据行数: {quality_report['total_rows']}")
        print(f"      - 数据列数: {quality_report['total_columns']}")
        print(f"      - 缺失值: {quality_report['missing_values']}")

        # 2. 异常值检测
        print("\n   2. 异常值检测...")
        outliers = analyzer.detect_outliers(data, 'salary')
        print(f"      - 薪资异常值: {outliers.sum()} 个")

        # 3. 相关性分析
        print("\n   3. 相关性分析...")
        correlation_matrix = analyzer.correlation_analysis(data)
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append(
                        f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {corr_val:.2f}"
                    )
        print(f"      - 强相关性特征对: {len(strong_corr)}")
        for corr in strong_corr[:3]:
            print(f"        • {corr}")

        # 4. 特征重要性
        if model_results := quick_modeling_example(data):
            print("\n   4. 特征重要性分析...")
            if 'feature_importance' in model_results:
                top_features = list(model_results['feature_importance'].keys())[:5]
                print(f"      - 最重要的5个特征:")
                for i, feature in enumerate(top_features, 1):
                    print(f"        {i}. {feature}")

    except Exception as e:
        print(f"   ❌ 功能演示失败: {str(e)}")

def main():
    """主函数"""
    print("🎯 数据探索可视化技能 - 快速开始示例")
    print("=" * 60)

    try:
        # 检查依赖
        print("🔍 检查依赖包...")
        required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"   ❌ 缺少依赖包: {', '.join(missing_packages)}")
            print(f"   请安装: pip install {' '.join(missing_packages)}")
            return

        print("   ✓ 所有依赖包已安装")

        # 运行完整流程
        run_complete_pipeline()

        # 演示特定功能
        demonstrate_specific_features()

        # 使用建议
        print("\n💡 使用建议:")
        print("   1. 将您自己的CSV数据替换示例数据")
        print("   2. 调整配置参数以适应您的需求")
        print("   3. 查看生成的HTML报告获取详细分析结果")
        print("   4. 尝试不同的模型和预处理方法")
        print("   5. 使用图表功能创建自定义可视化")

        print("\n📚 更多示例:")
        print("   - medical_data_analysis.py: 医疗数据分析示例")
        print("   - financial_data_analysis.py: 金融数据分析示例")

    except Exception as e:
        print(f"\n❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()