#!/usr/bin/env python3
"""
医疗数据分析示例
演示如何使用数据探索可视化技能进行医疗数据分析
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path / "scripts"))

def create_medical_sample_data():
    """创建医疗数据样本"""
    print("🏥 创建医疗数据样本...")

    np.random.seed(42)
    n_patients = 500

    # 基础患者信息
    data = {
        'patient_id': [f'P{i:04d}' for i in range(1, n_patients + 1)],
        'age': np.random.randint(18, 85, n_patients),
        'gender': np.random.choice(['男', '女'], n_patients, p=[0.48, 0.52]),
        'blood_type': np.random.choice(['A', 'B', 'AB', 'O'], n_patients, p=[0.3, 0.25, 0.1, 0.35]),
        'blood_pressure_systolic': np.random.normal(120, 15, n_patients),
        'blood_pressure_diastolic': np.random.normal(80, 10, n_patients),
        'heart_rate': np.random.normal(75, 10, n_patients),
        'cholesterol': np.random.normal(200, 40, n_patients),
        'glucose': np.random.normal(100, 25, n_patients),
        'bmi': np.random.normal(25, 5, n_patients),
        'smoking_status': np.random.choice(['从不', '曾经', '现在'], n_patients, p=[0.4, 0.3, 0.3]),
        'alcohol_consumption': np.random.choice(['无', '偶尔', '经常'], n_patients, p=[0.3, 0.5, 0.2]),
        'exercise_frequency': np.random.choice(['从不', '偶尔', '经常'], n_patients, p=[0.2, 0.4, 0.4]),
        'family_history': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'medications_count': np.random.randint(0, 8, n_patients),
        'doctor_visits_last_year': np.random.randint(0, 15, n_patients),
    }

    # 创建相关性：年龄和某些健康指标的关系
    data['cholesterol'] += np.random.normal(0, data['age'] * 0.5, n_patients)
    data['blood_pressure_systolic'] += np.random.normal(0, data['age'] * 0.3, n_patients)

    # BMI计算
    data['bmi'] = np.maximum(15, data['bmi'])  # 确保BMI合理

    # 诊断结果（二分类：0=健康，1=有疾病风险）
    risk_score = (
        (data['age'] > 60) * 0.3 +
        (data['cholesterol'] > 240) * 0.2 +
        (data['blood_pressure_systolic'] > 140) * 0.2 +
        (data['bmi'] > 30) * 0.15 +
        (data['family_history']) * 0.1 +
        (data['smoking_status'] == '现在') * 0.15 +
        (data['alcohol_consumption'] == '经常') * 0.1
    )

    # 添加噪声并转换为二分类
    risk_score += np.random.normal(0, 0.1, n_patients)
    data['disease_risk'] = (risk_score > 0.4).astype(int)

    # 疾病类型（针对有风险的患者）
    disease_types = ['无', '高血压', '糖尿病', '心脏病', '综合风险']
    data['disease_type'] = '无'
    risk_mask = data['disease_risk'] == 1
    data.loc[risk_mask, 'disease_type'] = np.random.choice(
        ['高血压', '糖尿病', '心脏病', '综合风险'],
        risk_mask.sum(),
        p=[0.35, 0.25, 0.25, 0.15]
    )

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 添加一些缺失值模拟真实数据
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'cholesterol'] = np.nan

    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'glucose'] = np.nan

    # 保存数据
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    data_path = output_dir / "medical_data_sample.csv"
    df.to_csv(data_path, index=False, encoding='utf-8-sig')

    print(f"   ✓ 医疗数据样本已保存: {data_path}")
    print(f"   ✓ 数据形状: {df.shape}")
    print(f"   ✓ 疾病风险分布: {df['disease_risk'].value_counts().to_dict()}")

    return df, data_path

def run_medical_eda_analysis(data_path):
    """运行医疗数据EDA分析"""
    print("\n🔍 运行医疗数据探索性分析...")

    try:
        from eda_analyzer import EDAAnalyzer

        analyzer = EDAAnalyzer()

        # 加载数据
        print("   加载数据...")
        data = analyzer.load_data(data_path)

        # 自动化EDA分析
        print("   执行自动化EDA分析...")
        eda_results = analyzer.auto_eda(data)

        # 保存EDA结果
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        eda_path = output_dir / "medical_eda_results.json"
        analyzer.export_results(eda_results, eda_path)

        print(f"   ✓ EDA分析完成，结果已保存: {eda_path}")

        # 显示关键发现
        print("\n📊 关键发现:")
        if 'insights' in eda_results:
            for insight in eda_results['insights'][:3]:
                print(f"   - {insight}")

        return eda_results, data

    except Exception as e:
        print(f"   ❌ EDA分析失败: {str(e)}")
        return None, None

def run_medical_visualization(data):
    """运行医疗数据可视化"""
    print("\n📈 生成医疗数据可视化图表...")

    try:
        from visualizer import DataVisualizer

        visualizer = DataVisualizer()

        # 自动可视化
        charts = visualizer.auto_visualize(
            data,
            target_col='disease_risk',
            save_charts=True,
            output_dir=str(Path(__file__).parent / "results" / "charts")
        )

        print(f"   ✓ 可视化完成，生成了 {charts['charts_generated']} 个图表")

        # 生成医疗专项图表
        medical_charts = {}

        # 1. 年龄分布 vs 疾病风险
        fig = visualizer.plot_distribution(
            data[data['disease_risk'] == 1],
            'age',
            interactive=True
        )
        medical_charts['high_risk_age_distribution'] = fig

        # 2. BMI vs 疾病风险散点图
        fig = visualizer.plot_scatter(
            data, 'age', 'bmi',
            color_col='disease_risk',
            interactive=True
        )
        medical_charts['age_bmi_risk_scatter'] = fig

        # 3. 疾病类型分布
        disease_counts = data['disease_type'].value_counts()
        fig = visualizer.plot_categorical(
            data, 'disease_type',
            interactive=True
        )
        medical_charts['disease_type_distribution'] = fig

        print(f"   ✓ 医疗专项图表生成完成: {len(medical_charts)} 个")

        return medical_charts

    except Exception as e:
        print(f"   ❌ 可视化生成失败: {str(e)}")
        return None

def run_medical_modeling(data, data_path):
    """运行医疗数据建模"""
    print("\n🤖 运行医疗数据建模...")

    try:
        from data_preprocessor import DataPreprocessor
        from modeling_evaluator import ModelingEvaluator

        # 数据预处理
        print("   数据预处理...")
        preprocessor = DataPreprocessor({
            'missing_threshold': 0.3,
            'feature_selection': True,
            'k_features': 10
        })

        # 预处理数据
        preprocessing_results = preprocessor.auto_preprocess(
            data,
            target_col='disease_risk',
            save_report=True
        )

        # 模型训练
        print("   模型训练...")
        modeler = ModelingEvaluator({
            'cv_folds': 5,
            'enable_hyperparameter_tuning': True,
            'n_iter_search': 20  # 减少搜索次数加快演示
        })

        # 自动建模
        model_results = modeler.auto_modeling(
            data,
            target_col='disease_risk',
            model_names=['logistic_regression', 'random_forest', 'xgboost']
        )

        print(f"   ✓ 模型训练完成，最佳模型: {model_results['best_model']['name']}")

        # 保存模型
        output_dir = Path(__file__).parent / "results" / "models"
        modeler.save_models(str(output_dir))

        return preprocessing_results, model_results

    except Exception as e:
        print(f"   ❌ 建模失败: {str(e)}")
        return None, None

def generate_medical_report(data, eda_results, model_results):
    """生成医疗数据分析报告"""
    print("\n📋 生成医疗数据分析报告...")

    try:
        from report_generator import ReportGenerator

        # 配置医疗特化报告
        generator = ReportGenerator({
            'report_title': '医疗数据分析报告',
            'author': '医疗数据分析助手',
            'company': '医疗机构',
            'medical_specialization': True
        })

        # 生成报告
        output_dir = Path(__file__).parent / "results"
        output_path = output_dir / "medical_analysis_report.html"

        report_path = generator.generate_comprehensive_report(
            data=data,
            eda_results=eda_results,
            model_results=model_results,
            output_path=str(output_path),
            format="html"
        )

        print(f"   ✓ 医疗分析报告已生成: {report_path}")

        # 生成快速报告
        quick_report_path = output_dir / "medical_quick_report.html"
        generator.generate_quick_report(
            data=data,
            target_col='disease_risk',
            output_path=str(quick_report_path)
        )

        print(f"   ✓ 快速报告已生成: {quick_report_path}")

        return report_path

    except Exception as e:
        print(f"   ❌ 报告生成失败: {str(e)}")
        return None

def main():
    """主函数"""
    print("🏥 医疗数据分析示例")
    print("=" * 50)

    # 创建输出目录
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 1. 创建样本数据
    data, data_path = create_medical_sample_data()

    # 2. EDA分析
    eda_results, processed_data = run_medical_eda_analysis(data_path)

    if processed_data is None:
        processed_data = data

    # 3. 可视化
    charts = run_medical_visualization(processed_data)

    # 4. 建模
    preprocessing_results, model_results = run_medical_modeling(processed_data, data_path)

    # 5. 生成报告
    report_path = generate_medical_report(processed_data, eda_results, model_results)

    # 6. 总结
    print("\n🎉 医疗数据分析完成！")
    print("\n📁 生成的文件:")

    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        for file_path in results_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(results_dir)
                print(f"   📄 {relative_path}")

    print(f"\n📊 主要发现:")
    if eda_results and 'data_quality' in eda_results:
        print(f"   - 数据质量分数: {eda_results['data_quality'].get('overall_score', 'N/A')}")

    if model_results and 'best_model' in model_results:
        best_model = model_results['best_model']
        if best_model and 'metrics' in best_model:
            accuracy = best_model['metrics'].get('accuracy', 0)
            print(f"   - 最佳模型准确率: {accuracy:.3f}")

    disease_risk_counts = processed_data['disease_risk'].value_counts()
    print(f"   - 高风险患者比例: {disease_risk_counts.get(1, 0) / len(processed_data) * 100:.1f}%")

    if report_path:
        print(f"\n📋 详细分析报告: {report_path}")
        print("   请在浏览器中打开HTML文件查看完整的交互式报告。")

if __name__ == "__main__":
    main()