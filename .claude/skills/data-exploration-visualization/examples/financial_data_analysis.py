#!/usr/bin/env python3
"""
金融数据分析示例
演示如何使用数据探索可视化技能进行金融数据分析
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加技能路径
skill_path = Path(__file__).parent.parent
sys.path.append(str(skill_path / "scripts"))

def create_financial_sample_data():
    """创建金融数据样本"""
    print("💰 创建金融数据样本...")

    np.random.seed(123)
    n_customers = 1000

    # 基础客户信息
    data = {
        'customer_id': [f'C{i:06d}' for i in range(1, n_customers + 1)],
        'age': np.random.randint(18, 80, n_customers),
        'gender': np.random.choice(['男', '女'], n_customers, p=[0.52, 0.48]),
        'income': np.random.lognormal(10.5, 0.5, n_customers),  # 收入分布
        'credit_score': np.random.normal(650, 100, n_customers),
        'employment_years': np.random.exponential(5, n_customers),
        'home_ownership': np.random.choice(['租房', '按揭', '自有'], n_customers, p=[0.35, 0.45, 0.20]),
        'marital_status': np.random.choice(['单身', '已婚', '离异'], n_customers, p=[0.3, 0.6, 0.1]),
        'education_level': np.random.choice(['高中', '本科', '硕士', '博士'], n_customers, p=[0.3, 0.4, 0.25, 0.05]),
        'debt_to_income_ratio': np.random.beta(2, 5, n_customers),  # 负债收入比
        'savings_amount': np.random.exponential(10000, n_customers),
        'credit_cards_count': np.random.poisson(2, n_customers),
        'late_payments_last_year': np.random.poisson(1, n_customers),
        'bankruptcy_history': np.random.choice([0, 1], n_customers, p=[0.95, 0.05]),
        'loan_amount': np.random.lognormal(9, 1, n_customers),
        'loan_purpose': np.random.choice(
            ['购房', '购车', '教育', '装修', '债务整合', '其他'],
            n_customers, p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
        ),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_customers, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        'interest_rate': np.random.uniform(3.5, 15.0, n_customers),
    }

    # 创建相关性
    df = pd.DataFrame(data)

    # 收入与信用分数的相关性
    df['credit_score'] = np.clip(
        df['credit_score'] + (df['income'] - df['income'].mean()) / df['income'].std() * 20,
        300, 850
    )

    # 年龄与工作年限的相关性
    df['employment_years'] = np.minimum(df['employment_years'], df['age'] - 18)

    # 收入与贷款金额的相关性
    df['loan_amount'] = df['loan_amount'] * (0.5 + 0.5 * df['income'] / df['income'].mean())

    # 负债收入比与利率的相关性
    df['interest_rate'] = df['interest_rate'] + df['debt_to_income_ratio'] * 5

    # 信用评分与利率的相关性（负相关）
    df['interest_rate'] = df['interest_rate'] - (df['credit_score'] - 650) / 100

    # 计算违约概率（目标变量）
    default_probability = (
        (df['credit_score'] < 600) * 0.4 +
        (df['debt_to_income_ratio'] > 0.4) * 0.3 +
        (df['late_payments_last_year'] > 3) * 0.2 +
        (df['bankruptcy_history'] == 1) * 0.3 +
        (df['employment_years'] < 1) * 0.2 +
        (df['income'] < 30000) * 0.15 +
        np.random.normal(0, 0.1, n_customers)
    )

    # 转换为二分类
    df['loan_default'] = (default_probability > 0.3).astype(int)

    # 风险等级
    risk_conditions = [
        df['credit_score'] >= 750,
        (df['credit_score'] >= 700) & (df['credit_score'] < 750),
        (df['credit_score'] >= 650) & (df['credit_score'] < 700),
        (df['credit_score'] >= 600) & (df['credit_score'] < 650),
        df['credit_score'] < 600
    ]

    risk_labels = ['AAA', 'AA', 'A', 'BBB', 'BB']
    df['risk_rating'] = np.select(risk_conditions, risk_labels, default='BB')

    # 添加一些缺失值
    missing_indices = np.random.choice(df.index, size=int(0.08 * len(df)), replace=False)
    df.loc[missing_indices, 'savings_amount'] = np.nan

    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'employment_years'] = np.nan

    # 保存数据
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    data_path = output_dir / "financial_data_sample.csv"
    df.to_csv(data_path, index=False, encoding='utf-8-sig')

    print(f"   ✓ 金融数据样本已保存: {data_path}")
    print(f"   ✓ 数据形状: {df.shape}")
    print(f"   ✓ 违约率: {df['loan_default'].mean():.2%}")
    print(f"   ✓ 平均信用分数: {df['credit_score'].mean():.1f}")

    return df, data_path

def run_financial_eda_analysis(data_path):
    """运行金融数据EDA分析"""
    print("\n🔍 运行金融数据探索性分析...")

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

        eda_path = output_dir / "financial_eda_results.json"
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

def run_financial_visualization(data):
    """运行金融数据可视化"""
    print("\n📈 生成金融数据可视化图表...")

    try:
        from visualizer import DataVisualizer

        visualizer = DataVisualizer()

        # 自动可视化
        charts = visualizer.auto_visualize(
            data,
            target_col='loan_default',
            save_charts=True,
            output_dir=str(Path(__file__).parent / "results" / "charts")
        )

        print(f"   ✓ 可视化完成，生成了 {charts['charts_generated']} 个图表")

        # 生成金融专项图表
        financial_charts = {}

        # 1. 信用分数分布
        fig = visualizer.plot_distribution(
            data, 'credit_score',
            interactive=True
        )
        financial_charts['credit_score_distribution'] = fig

        # 2. 收入 vs 贷款金额
        fig = visualizer.plot_scatter(
            data, 'income', 'loan_amount',
            color_col='loan_default',
            interactive=True
        )
        financial_charts['income_loan_scatter'] = fig

        # 3. 风险等级分布
        fig = visualizer.plot_categorical(
            data, 'risk_rating',
            interactive=True
        )
        financial_charts['risk_rating_distribution'] = fig

        # 4. 违约率 vs 特征分析
        default_by_purpose = data.groupby('loan_purpose')['loan_default'].mean().sort_values()
        fig = visualizer.plot_categorical(
            data, 'loan_purpose',
            interactive=True
        )
        financial_charts['default_by_purpose'] = fig

        print(f"   ✓ 金融专项图表生成完成: {len(financial_charts)} 个")

        return financial_charts

    except Exception as e:
        print(f"   ❌ 可视化生成失败: {str(e)}")
        return None

def run_financial_modeling(data, data_path):
    """运行金融数据建模"""
    print("\n🤖 运行金融数据建模...")

    try:
        from data_preprocessor import DataPreprocessor
        from modeling_evaluator import ModelingEvaluator

        # 数据预处理
        print("   数据预处理...")
        preprocessor = DataPreprocessor({
            'missing_threshold': 0.2,
            'feature_selection': True,
            'k_features': 15,
            'balance_data': True,  # 平衡违约样本
            'balance_method': 'smote'
        })

        # 预处理数据
        preprocessing_results = preprocessor.auto_preprocess(
            data,
            target_col='loan_default',
            save_report=True
        )

        # 模型训练
        print("   模型训练...")
        modeler = ModelingEvaluator({
            'cv_folds': 5,
            'enable_hyperparameter_tuning': True,
            'n_iter_search': 15,  # 减少搜索次数加快演示
            'scoring_metric': 'roc_auc'  # 使用ROC AUC作为评估指标
        })

        # 自动建模
        model_results = modeler.auto_modeling(
            data,
            target_col='loan_default',
            model_names=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        )

        print(f"   ✓ 模型训练完成，最佳模型: {model_results['best_model']['name']}")

        # 保存模型
        output_dir = Path(__file__).parent / "results" / "models"
        modeler.save_models(str(output_dir))

        return preprocessing_results, model_results

    except Exception as e:
        print(f"   ❌ 建模失败: {str(e)}")
        return None, None

def calculate_credit_risk_score(data, model_results):
    """计算综合信用风险评分"""
    print("\n📊 计算综合信用风险评分...")

    try:
        # 基于模型结果计算风险评分
        if model_results and 'best_model' in model_results:
            best_model_name = model_results['best_model']['name']
            print(f"   使用模型: {best_model_name}")

            # 创建风险评分表
            risk_scores = []

            for _, row in data.iterrows():
                # 基础信用分数
                base_score = row['credit_score']

                # 调整因子
                adjustments = 0

                # 收入调整
                if row['income'] > 100000:
                    adjustments += 20
                elif row['income'] < 30000:
                    adjustments -= 30

                # 负债收入比调整
                if row['debt_to_income_ratio'] > 0.4:
                    adjustments -= 40
                elif row['debt_to_income_ratio'] < 0.2:
                    adjustments += 15

                # 逾期记录调整
                if row['late_payments_last_year'] > 2:
                    adjustments -= 25

                # 破产历史调整
                if row['bankruptcy_history'] == 1:
                    adjustments -= 100

                # 就业稳定性调整
                if row['employment_years'] > 5:
                    adjustments += 10
                elif row['employment_years'] < 1:
                    adjustments -= 20

                final_score = base_score + adjustments
                final_score = np.clip(final_score, 300, 850)

                risk_scores.append(final_score)

            # 添加到数据中
            data = data.copy()
            data['comprehensive_risk_score'] = risk_scores

            # 风险等级
            score_conditions = [
                data['comprehensive_risk_score'] >= 780,
                (data['comprehensive_risk_score'] >= 740) & (data['comprehensive_risk_score'] < 780),
                (data['comprehensive_risk_score'] >= 700) & (data['comprehensive_risk_score'] < 740),
                (data['comprehensive_risk_score'] >= 660) & (data['comprehensive_risk_score'] < 700),
                (data['comprehensive_risk_score'] >= 620) & (data['comprehensive_risk_score'] < 660),
                data['comprehensive_risk_score'] < 620
            ]

            risk_labels = ['AA+', 'AA', 'A', 'BBB', 'BB', 'B']
            data['final_risk_rating'] = np.select(score_conditions, risk_labels, default='B')

            print(f"   ✓ 风险评分计算完成")
            print(f"   ✓ 平均风险评分: {data['comprehensive_risk_score'].mean():.1f}")

            # 保存风险评分结果
            output_dir = Path(__file__).parent / "results"
            data.to_csv(output_dir / "credit_risk_scores.csv", index=False, encoding='utf-8-sig')

            return data

        else:
            print("   ⚠️ 无法计算风险评分，缺少模型结果")
            return data

    except Exception as e:
        print(f"   ❌ 风险评分计算失败: {str(e)}")
        return data

def generate_financial_report(data, eda_results, model_results):
    """生成金融数据分析报告"""
    print("\n📋 生成金融数据分析报告...")

    try:
        from report_generator import ReportGenerator

        # 配置金融特化报告
        generator = ReportGenerator({
            'report_title': '金融信贷风险分析报告',
            'author': '金融风险分析助手',
            'company': '金融机构',
            'include_recommendations': True
        })

        # 生成报告
        output_dir = Path(__file__).parent / "results"
        output_path = output_dir / "financial_analysis_report.html"

        report_path = generator.generate_comprehensive_report(
            data=data,
            eda_results=eda_results,
            model_results=model_results,
            output_path=str(output_path),
            format="html"
        )

        print(f"   ✓ 金融分析报告已生成: {report_path}")

        # 生成快速报告
        quick_report_path = output_dir / "financial_quick_report.html"
        generator.generate_quick_report(
            data=data,
            target_col='loan_default',
            output_path=str(quick_report_path)
        )

        print(f"   ✓ 快速报告已生成: {quick_report_path}")

        return report_path

    except Exception as e:
        print(f"   ❌ 报告生成失败: {str(e)}")
        return None

def main():
    """主函数"""
    print("💰 金融数据分析示例")
    print("=" * 50)

    # 创建输出目录
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 1. 创建样本数据
    data, data_path = create_financial_sample_data()

    # 2. EDA分析
    eda_results, processed_data = run_financial_eda_analysis(data_path)

    if processed_data is None:
        processed_data = data

    # 3. 可视化
    charts = run_financial_visualization(processed_data)

    # 4. 建模
    preprocessing_results, model_results = run_financial_modeling(processed_data, data_path)

    # 5. 计算风险评分
    scored_data = calculate_credit_risk_score(processed_data, model_results)

    # 6. 生成报告
    report_path = generate_financial_report(scored_data, eda_results, model_results)

    # 7. 总结
    print("\n🎉 金融数据分析完成！")
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
            auc = best_model['metrics'].get('auc', 0)
            print(f"   - 最佳模型AUC: {auc:.3f}")

    default_rate = processed_data['loan_default'].mean()
    print(f"   - 整体违约率: {default_rate:.2%}")

    avg_credit_score = processed_data['credit_score'].mean()
    print(f"   - 平均信用分数: {avg_credit_score:.1f}")

    if 'comprehensive_risk_score' in scored_data.columns:
        avg_risk_score = scored_data['comprehensive_risk_score'].mean()
        print(f"   - 综合风险评分: {avg_risk_score:.1f}")

    if report_path:
        print(f"\n📋 详细分析报告: {report_path}")
        print("   请在浏览器中打开HTML文件查看完整的交互式报告。")

if __name__ == "__main__":
    main()