# -*- coding: utf-8 -*-
"""
Stage 1: 数据质量检查 (Data Quality Assessment)
完整的数据质量检查脚本
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    """记录日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Stage 1] {msg}", flush=True)

def main():
    log("="*60)
    log("Stage 1: 数据质量检查开始")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data_storage")
    output_dir = os.path.join(base_dir, "data_quality_report")

    # 数据文件列表
    datasets = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "product_category_name_translation.csv"
    ]

    overall_quality = {}
    all_issues = []

    for filename in datasets:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            log(f"警告: 找不到文件 {filename}")
            continue

        log(f"正在分析: {filename}")
        try:
            df = pd.read_csv(filepath)
            quality = assess_dataset_quality(df, filename)
            overall_quality[filename] = quality
            all_issues.extend(quality['issues'])
            log(f"  完成: {filename}, 质量分数: {quality['overall_score']:.1f}")
        except Exception as e:
            log(f"  错误: {filename} - {str(e)}")

    # 生成总体报告
    generate_quality_report(overall_quality, all_issues, output_dir)

    log("="*60)
    log("Stage 1: 数据质量检查完成")
    log("="*60)

def convert_types(obj):
    """转换numpy类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(item) for item in obj]
    else:
        return obj

def assess_dataset_quality(df, filename):
    """评估单个数据集质量"""
    quality = {
        'filename': filename,
        'shape': (int(df.shape[0]), int(df.shape[1])),
        'columns': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict(),
        'completeness': {},
        'uniqueness': {},
        'validity': {},
        'issues': []
    }

    # 1. 完整性分析
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    quality['completeness'] = {
        'total_missing': missing.sum(),
        'missing_by_column': missing.to_dict(),
        'missing_pct_by_column': missing_pct.to_dict(),
        'score': max(0, 100 - (missing_pct.mean() * 2))
    }

    # 2. 唯一性分析
    duplicates = df.duplicated().sum()
    unique_ratios = {}
    for col in df.columns:
        unique_ratios[col] = df[col].nunique() / len(df) if len(df) > 0 else 0

    quality['uniqueness'] = {
        'duplicate_rows': duplicates,
        'duplicate_pct': (duplicates / len(df)) * 100 if len(df) > 0 else 0,
        'unique_ratios': unique_ratios,
        'score': max(0, 100 - (duplicates / len(df) * 50) if len(df) > 0 else 100)
    }

    # 3. 有效性分析 (简单检查)
    validity_issues = []
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_strings = (df[col] == '').sum()
            if empty_strings > 0:
                validity_issues.append(f"{col} 有 {empty_strings} 个空字符串")

    quality['validity'] = {
        'validity_issues': validity_issues,
        'score': max(0, 100 - len(validity_issues) * 5)
    }

    # 收集问题
    issues = []
    for col, pct in quality['completeness']['missing_pct_by_column'].items():
        if pct > 10:
            issues.append(f"高缺失值: {col} ({pct:.1f}%)")
    if quality['uniqueness']['duplicate_pct'] > 5:
        issues.append(f"重复行: {quality['uniqueness']['duplicate_pct']:.1f}%")
    issues.extend(validity_issues)

    quality['issues'] = issues

    # 计算总体质量分数
    scores = [
        quality['completeness']['score'],
        quality['uniqueness']['score'],
        quality['validity']['score']
    ]
    quality['overall_score'] = np.mean(scores)

    return quality

def generate_quality_report(overall_quality, all_issues, output_dir):
    """生成质量报告"""

    # 1. 保存JSON结果
    json_path = os.path.join(output_dir, "quality_assessment.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_types(overall_quality), f, indent=2, ensure_ascii=False)
    log(f"质量评估JSON已保存: {json_path}")

    # 2. 保存问题日志
    issues_path = os.path.join(output_dir, "data_issues.log")
    with open(issues_path, 'w', encoding='utf-8') as f:
        for issue in all_issues:
            f.write(f"[{datetime.now().isoformat()}] {issue}\n")
    log(f"问题日志已保存: {issues_path}")

    # 3. 生成Markdown报告
    md_path = os.path.join(output_dir, "quality_improvement_recommendations.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 数据质量评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 整体评分
        avg_score = np.mean([q['overall_score'] for q in overall_quality.values()])
        f.write(f"## 整体质量评分: {avg_score:.1f}/100\n\n")

        # 各数据集评分
        f.write("### 各数据集质量\n\n")
        f.write("| 数据集 | 行数 | 列数 | 质量分数 | 问题数 |\n")
        f.write("|--------|------|------|----------|--------|\n")
        for filename, q in overall_quality.items():
            f.write(f"| {filename} | {q['shape'][0]} | {q['shape'][1]} | {q['overall_score']:.1f} | {len(q['issues'])} |\n")

        f.write("\n## 详细分析\n\n")

        for filename, q in overall_quality.items():
            f.write(f"### {filename}\n\n")
            f.write(f"- 形状: {q['shape'][0]} 行 × {q['shape'][1]} 列\n")
            f.write(f"- 质量分数: {q['overall_score']:.1f}/100\n")
            f.write(f"- 完整性分数: {q['completeness']['score']:.1f}\n")
            f.write(f"- 唯一性分数: {q['uniqueness']['score']:.1f}\n")
            f.write(f"- 有效性分数: {q['validity']['score']:.1f}\n")
            if q['issues']:
                f.write("\n#### 发现的问题:\n")
                for issue in q['issues']:
                    f.write(f"- {issue}\n")
            f.write("\n")

        # 改进建议
        f.write("## 改进建议\n\n")
        f.write("1. **处理缺失值**: 对于缺失率>10%的字段，考虑删除、填充或标记\n")
        f.write("2. **删除重复**: 检查并删除重复行\n")
        f.write("3. **数据验证**: 确保数据类型和格式正确\n")
        f.write("4. **持续监控**: 建立数据质量监控机制\n")

    log(f"质量报告已保存: {md_path}")

    # 保存执行摘要到workflow_log
    summary_path = os.path.join(base_dir, "workflow_log", "stage1_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Stage 1: 数据质量检查\n\n")
        f.write(f"- 分析数据集: {len(overall_quality)}\n")
        f.write(f"- 整体质量评分: {avg_score:.1f}/100\n")
        f.write(f"- 发现问题总数: {len(all_issues)}\n")
        f.write(f"- 状态: ✅ 完成\n")

if __name__ == "__main__":
    main()
