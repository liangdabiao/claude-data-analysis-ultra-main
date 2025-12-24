#!/usr/bin/env python3
"""
快速分析脚本
一键完成LTV预测分析流程的便利工具
基于第3课理论实现的完整自动化分析
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ltv_predictor import LTVPredictor, complete_ltv_analysis
from visualizer import DataVisualizer, quick_visualization
from report_generator import generate_comprehensive_report

def quick_ltv_analysis(file_path: str,
                      feature_period_months: int = 3,
                      prediction_period_months: int = 12,
                      output_dir: str = './ltv_analysis_results',
                      config: Optional[Dict] = None,
                      generate_charts: bool = True,
                      generate_reports: bool = True) -> Dict[str, Any]:
    """
    快速LTV分析 - 一键完成完整的LTV预测流程

    Args:
        file_path: 订单数据文件路径
        feature_period_months: 特征计算时间窗口（月）
        prediction_period_months: 预测时间窗口（月）
        output_dir: 输出目录
        config: 自定义配置
        generate_charts: 是否生成图表
        generate_reports: 是否生成报告

    Returns:
        完整分析结果字典
    """
    print("🚀 启动快速LTV分析...")
    print("=" * 60)
    print(f"📁 输入文件: {file_path}")
    print(f"📊 特征计算期: {feature_period_months}个月")
    print(f"🎯 预测期: {prediction_period_months}个月")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 准备配置
    if config is None:
        config = {
            'data_processor_config': {
                'feature_period_months': feature_period_months,
                'prediction_period_months': prediction_period_months,
                'remove_outliers': True,
                'min_orders_per_customer': 1
            },
            'regression_config': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scoring_metric': 'r2',
                'enable_hyperparameter_tuning': False
            },
            'models_to_train': ['linear_regression', 'random_forest']
        }

    try:
        # 1. 完整LTV分析流程
        results = complete_ltv_analysis(file_path, output_dir, config)

        # 提取结果
        predictor = results['predictor']
        rfm_data = results['rfm_data']
        training_results = results['training_results']
        summary_report = results['summary_report']

        # 2. 生成可视化图表
        chart_paths = {}
        if generate_charts:
            print("\n📈 生成可视化图表...")
            try:
                # 获取预测结果用于可视化
                X_test = predictor.training_data['X_test']
                y_test = predictor.training_data['y_test']
                y_pred = predictor.predict(predictor.regression_models.best_model_name, X_test)

                charts_dir = os.path.join(output_dir, 'charts')
                chart_paths = quick_visualization(
                    rfm_data=rfm_data,
                    model_results=training_results['model_results'],
                    output_dir=charts_dir,
                    show_plots=False
                )

                # 添加预测分析图
                visualizer = DataVisualizer()
                pred_chart_path = os.path.join(charts_dir, '06_prediction_analysis.png')
                visualizer.plot_prediction_analysis(
                    y_test, y_pred,
                    model_name=predictor.regression_models.best_model_name,
                    save_path=pred_chart_path,
                    show_plot=False
                )
                chart_paths['prediction_analysis'] = pred_chart_path

            except Exception as e:
                print(f"⚠️ 图表生成失败: {str(e)}")

        # 3. 生成分析报告
        report_paths = {}
        if generate_reports:
            print("\n📋 生成分析报告...")
            try:
                reports_dir = os.path.join(output_dir, 'reports')
                Path(reports_dir).mkdir(parents=True, exist_ok=True)

                feature_importance = training_results.get('feature_importance', {})
                report_paths = generate_comprehensive_report(
                    rfm_data=rfm_data,
                    model_results=training_results['model_results'],
                    feature_importance=feature_importance,
                    output_dir=reports_dir,
                    report_title=f'客户生命周期价值分析报告 - {Path(file_path).stem}'
                )

            except Exception as e:
                print(f"⚠️ 报告生成失败: {str(e)}")

        # 4. 生成分析摘要
        summary = generate_analysis_summary(results, chart_paths, report_paths)

        # 保存摘要
        summary_path = os.path.join(output_dir, 'analysis_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        # 5. 打印最终结果
        print("\n" + "=" * 60)
        print("🎉 快速LTV分析完成！")
        print("=" * 60)

        if summary_report and 'best_model_performance' in summary_report:
            best_perf = summary_report['best_model_performance']
            print(f"\n📊 核心结果:")
            print(f"  ✅ 最佳模型: {best_perf.get('model_name', 'Unknown')}")
            print(f"  📈 模型R²: {best_perf.get('r2_score', 0):.4f}")
            print(f"  👥 分析客户: {summary['data_summary']['total_customers']:,}")
            print(f"  💰 平均LTV: {summary['data_summary']['avg_ltv']:,.0f}")

        print(f"\n📁 生成文件:")
        print(f"  📄 RFM特征数据: {results['output_paths']['rfm_features']}")
        print(f"  🤖 训练模型: {results['output_paths']['models']}")
        print(f"  📋 分析摘要: {summary_path}")

        if chart_paths:
            print(f"  📈 可视化图表: {len(chart_paths)}个")

        if report_paths:
            print(f"  📄 分析报告: {len(report_paths)}个")

        # 组合完整结果
        complete_results = {
            **results,
            'chart_paths': chart_paths,
            'report_paths': report_paths,
            'summary': summary,
            'summary_path': summary_path
        }

        return complete_results

    except Exception as e:
        print(f"\n❌ 快速LTV分析失败: {str(e)}")
        raise

def generate_analysis_summary(results: Dict[str, Any], chart_paths: Dict[str, str], report_paths: Dict[str, str]) -> Dict[str, Any]:
    """生成分析摘要"""
    predictor = results['predictor']
    rfm_data = results['rfm_data']
    training_results = results['training_results']
    summary_report = results['summary_report']

    # 基础统计
    data_summary = {
        'total_customers': len(rfm_data),
        'total_orders': summary_report.get('data_summary', {}).get('total_orders', 'Unknown'),
        'date_range': summary_report.get('data_summary', {}).get('date_range', ['Unknown', 'Unknown']),
        'avg_ltv': rfm_data['年度LTV'].mean() if '年度LTV' in rfm_data.columns else 0,
        'median_ltv': rfm_data['年度LTV'].median() if '年度LTV' in rfm_data.columns else 0
    }

    # 模型性能摘要
    model_summary = {}
    if summary_report and 'best_model_performance' in summary_report:
        best_perf = summary_report['best_model_performance']
        model_summary = {
            'best_model': best_perf.get('model_name', 'Unknown'),
            'best_r2_score': best_perf.get('r2_score', 0),
            'best_mae': best_perf.get('mae', 0),
            'best_rmse': best_perf.get('rmse', 0),
            'best_mape': best_perf.get('mape', 0)
        }

    # 特征重要性摘要
    feature_summary = {}
    if 'feature_analysis' in summary_report and 'best_model_features' in summary_report['feature_analysis']:
        best_features = summary_report['feature_analysis']['best_model_features']
        if best_features:
            feature_summary = {
                'most_important_feature': summary_report['feature_analysis'].get('most_important_feature', 'Unknown'),
                'feature_importance': best_features
            }

    # 客户分群摘要
    segment_summary = {}
    if '客户价值分层' in rfm_data.columns:
        segment_counts = rfm_data['客户价值分层'].value_counts()
        segment_summary = {
            'segment_counts': segment_counts.to_dict(),
            'high_value_customers': segment_counts.get('白金客户', 0) + segment_counts.get('钻石客户', 0)
        }

    # 业务建议摘要
    recommendations_summary = []
    if summary_report and 'recommendations' in summary_report:
        recommendations_summary = summary_report['recommendations']

    summary = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_summary': data_summary,
        'model_summary': model_summary,
        'feature_summary': feature_summary,
        'segment_summary': segment_summary,
        'recommendations_summary': recommendations_summary,
        'generated_files': {
            'charts': chart_paths,
            'reports': report_paths
        }
    }

    return summary

def predict_new_customers(model_dir: str, new_orders_file: str, output_path: str = 'new_customer_predictions.csv'):
    """
    为新客户预测LTV

    Args:
        model_dir: 训练好的模型目录
        new_orders_file: 新客户订单数据文件
        output_path: 预测结果输出路径
    """
    print("🔮 为新客户预测LTV...")

    # 初始化预测器并加载模型
    predictor = LTVPredictor()
    predictor.load_results(model_dir)

    # 加载新客户数据
    new_customers_data = pd.read_csv(new_orders_file)
    print(f"  - 新客户数据: {new_customers_data.shape}")

    # 预测LTV
    predictions = predictor.predict_new_customers(new_customers_data)

    # 保存预测结果
    predictions.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 预测结果已保存: {output_path}")

    return predictions

def batch_predict_ltv(model_dir: str, rfm_features_file: str, output_path: str = 'batch_predictions.csv'):
    """
    批量预测LTV

    Args:
        model_dir: 训练好的模型目录
        rfm_features_file: RFM特征文件
        output_path: 预测结果输出路径
    """
    print("📊 批量LTV预测...")

    # 初始化预测器并加载模型
    predictor = LTVPredictor()
    predictor.load_results(model_dir)

    # 加载RFM特征数据
    rfm_data = pd.read_csv(rfm_features_file)
    print(f"  - 预测客户数: {len(rfm_data)}")

    # 预测LTV
    feature_columns = predictor.config['feature_columns']
    X = rfm_data[feature_columns]
    predictions = predictor.predict(X)

    # 保存预测结果
    result_df = rfm_data.copy()
    result_df['预测LTV'] = predictions
    result_df['预测时间'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 批量预测结果已保存: {output_path}")

    return result_df

def compare_models_performance(model_dir: str):
    """
    比较已训练模型的性能

    Args:
        model_dir: 模型目录
    """
    print("📈 模型性能比较...")

    # 加载模型
    predictor = LTVPredictor()
    predictor.load_results(model_dir)

    # 获取模型摘要
    summary = predictor.regression_models.get_model_summary()

    print("\n模型性能排名:")
    print("-" * 50)

    if summary and 'model_performance' in summary:
        # 按R²分数排序
        model_perf = summary['model_performance']
        sorted_models = sorted(model_perf.items(), key=lambda x: x[1].get('r2_score', 0), reverse=True)

        for rank, (model_name, performance) in enumerate(sorted_models, 1):
            r2_score = performance.get('r2_score', 0)
            mae = performance.get('mae', 0)
            rmse = performance.get('rmse', 0)
            mape = performance.get('mape', 0)

            print(f"{rank}. {model_name}")
            print(f"   R² 分数: {r2_score:.4f}")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   MAPE: {mape:.2f}%")
            print()

    # 最佳模型
    if summary.get('best_model'):
        print(f"🏆 最佳模型: {summary['best_model']}")
        if 'best_performance' in summary:
            best_perf = summary['best_performance']
            print(f"   R² 分数: {best_perf.get('r2_score', 0):.4f}")

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='LTV预测分析工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='执行完整的LTV分析')
    analyze_parser.add_argument('input_file', help='订单数据文件路径')
    analyze_parser.add_argument('--output-dir', '-o', default='./ltv_results', help='输出目录')
    analyze_parser.add_argument('--feature-period', type=int, default=3, help='特征计算期（月）')
    analyze_parser.add_argument('--prediction-period', type=int, default=12, help='预测期（月）')
    analyze_parser.add_argument('--no-charts', action='store_true', help='不生成图表')
    analyze_parser.add_argument('--no-reports', action='store_true', help='不生成报告')

    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测新客户LTV')
    predict_parser.add_argument('model_dir', help='训练好的模型目录')
    predict_parser.add_argument('input_file', help='新客户订单数据文件')
    predict_parser.add_argument('--output', '-o', default='predictions.csv', help='输出文件路径')

    # 批量预测命令
    batch_parser = subparsers.add_parser('batch', help='批量预测LTV')
    batch_parser.add_argument('model_dir', help='训练好的模型目录')
    batch_parser.add_argument('rfm_file', help='RFM特征文件')
    batch_parser.add_argument('--output', '-o', default='batch_predictions.csv', help='输出文件路径')

    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较模型性能')
    compare_parser.add_argument('model_dir', help='模型目录')

    args = parser.parse_args()

    if args.command == 'analyze':
        # 执行分析
        results = quick_ltv_analysis(
            file_path=args.input_file,
            feature_period_months=args.feature_period,
            prediction_period_months=args.prediction_period,
            output_dir=args.output_dir,
            generate_charts=not args.no_charts,
            generate_reports=not args.no_reports
        )

    elif args.command == 'predict':
        # 预测新客户
        predict_new_customers(args.model_dir, args.input_file, args.output)

    elif args.command == 'batch':
        # 批量预测
        batch_predict_ltv(args.model_dir, args.rfm_file, args.output)

    elif args.command == 'compare':
        # 比较模型
        compare_models_performance(args.model_dir)

    else:
        parser.print_help()

if __name__ == "__main__":
    # 示例使用
    print("🔧 快速分析工具测试")

    # 如果通过命令行运行，使用argparse
    if len(sys.argv) > 1:
        main()
    else:
        print("使用 'python quick_analysis.py --help' 查看使用说明")

        # 简单测试
        sample_file = '../data/sample_orders.csv'
        if os.path.exists(sample_file):
            print(f"发现示例数据文件: {sample_file}")
            print("可以使用以下命令进行测试:")
            print(f"python quick_analysis.py analyze {sample_file} --output-dir ./test_results")
        else:
            print("⚠️ 未找到示例数据文件")