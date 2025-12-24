"""
订单数据探索性分析脚本
对 Olist 订单数据进行全面的探索性数据分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(file_path)

    # 转换日期列
    date_columns = ['order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date',
                   'order_estimated_delivery_date']

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

def generate_data_overview(df):
    """生成数据概览"""
    print("\n" + "="*80)
    print("数据集概览")
    print("="*80)

    overview = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }

    print(f"数据维度: {overview['shape'][0]} 行 × {overview['shape'][1]} 列")
    print(f"内存使用: {overview['memory_usage']}")
    print(f"重复行数: {overview['duplicate_rows']}")

    return overview

def analyze_order_status(df):
    """分析订单状态分布"""
    print("\n" + "="*80)
    print("订单状态分析")
    print("="*80)

    status_counts = df['order_status'].value_counts()
    status_percentages = df['order_status'].value_counts(normalize=True) * 100

    print("\n订单状态分布:")
    for status, count in status_counts.items():
        pct = status_percentages[status]
        print(f"  {status}: {count} ({pct:.2f}%)")

    return {
        'counts': status_counts.to_dict(),
        'percentages': status_percentages.to_dict()
    }

def analyze_temporal_patterns(df):
    """分析时间模式"""
    print("\n" + "="*80)
    print("时间模式分析")
    print("="*80)

    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

    # 按年月统计订单数量
    monthly_orders = df.groupby(['purchase_year', 'purchase_month']).size()
    print("\n每月订单数量趋势:")
    for (year, month), count in monthly_orders.items():
        print(f"  {year}-{month:02d}: {count} 订单")

    # 按星期统计
    dow_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    dow_counts = df['purchase_dayofweek'].value_counts().sort_index()
    print("\n按星期分布:")
    for day, count in dow_counts.items():
        print(f"  {dow_names[day]}: {count} 订单")

    return {
        'monthly_orders': monthly_orders.to_dict(),
        'day_of_week': dow_counts.to_dict()
    }

def analyze_delivery_performance(df):
    """分析配送绩效"""
    print("\n" + "="*80)
    print("配送绩效分析")
    print("="*80)

    # 计算实际配送时间
    df['delivery_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    # 计算预计配送时间
    df['estimated_days'] = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.days

    # 计算是否延迟
    df['is_late'] = df['delivery_days'] > df['estimated_days']

    delivery_stats = {
        'avg_delivery_days': df['delivery_days'].mean(),
        'median_delivery_days': df['delivery_days'].median(),
        'avg_estimated_days': df['estimated_days'].mean(),
        'late_delivery_rate': df['is_late'].mean() * 100
    }

    print(f"\n平均实际配送天数: {delivery_stats['avg_delivery_days']:.2f} 天")
    print(f"中位数配送天数: {delivery_stats['median_delivery_days']:.2f} 天")
    print(f"平均预计配送天数: {delivery_stats['avg_estimated_days']:.2f} 天")
    print(f"延迟配送率: {delivery_stats['late_delivery_rate']:.2f}%")

    return delivery_stats

def create_visualizations(df, output_dir):
    """创建可视化图表"""
    print("\n正在生成可视化图表...")

    # 创建图表目录
    os.makedirs(output_dir, exist_ok=True)

    charts = []

    # 1. 订单状态分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    status_counts = df['order_status'].value_counts()
    colors = sns.color_palette("husl", len(status_counts))
    ax.bar(status_counts.index, status_counts.values, color=colors)
    ax.set_xlabel('订单状态', fontsize=12)
    ax.set_ylabel('订单数量', fontsize=12)
    ax.set_title('订单状态分布', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    status_chart = os.path.join(output_dir, 'order_status_distribution.png')
    plt.savefig(status_chart, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(status_chart)
    print(f"  [OK] Generated: order_status_distribution.png")

    # 2. 时间趋势图
    fig, ax = plt.subplots(figsize=(14, 6))
    df['purchase_date'] = df['order_purchase_timestamp'].dt.date
    daily_orders = df.groupby('purchase_date').size()
    ax.plot(daily_orders.index, daily_orders.values, linewidth=1.5, alpha=0.7)
    # 添加移动平均
    rolling_avg = daily_orders.rolling(window=7).mean()
    ax.plot(daily_orders.index, rolling_avg.values, linewidth=2.5, color='red', label='7日移动平均')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('订单数量', fontsize=12)
    ax.set_title('订单数量时间趋势', fontsize=14, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    trend_chart = os.path.join(output_dir, 'order_time_trend.png')
    plt.savefig(trend_chart, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(trend_chart)
    print(f"  [OK] Generated: order_time_trend.png")

    # 3. 配送时间分布图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 实际配送时间分布
    df['delivery_days'].hist(bins=50, ax=axes[0], alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(df['delivery_days'].mean(), color='red', linestyle='--', linewidth=2, label=f"平均值: {df['delivery_days'].mean():.1f}天")
    axes[0].set_xlabel('配送天数', fontsize=12)
    axes[0].set_ylabel('订单数量', fontsize=12)
    axes[0].set_title('实际配送时间分布', fontsize=14, fontweight='bold')
    axes[0].legend()

    # 延迟配送率
    late_rate = df['is_late'].mean() * 100
    on_time_rate = 100 - late_rate
    axes[1].pie([on_time_rate, late_rate],
               labels=['准时配送', '延迟配送'],
               colors=['#2ecc71', '#e74c3c'],
               autopct='%1.1f%%',
               startangle=90,
               textprops={'fontsize': 12})
    axes[1].set_title('配送准时率', fontsize=14, fontweight='bold')

    plt.tight_layout()
    delivery_chart = os.path.join(output_dir, 'delivery_performance.png')
    plt.savefig(delivery_chart, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(delivery_chart)
    print(f"  [OK] Generated: delivery_performance.png")

    # 4. 每周订单分布
    fig, ax = plt.subplots(figsize=(12, 6))
    dow_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    dow_counts = df['purchase_dayofweek'].value_counts().sort_index()
    bars = ax.bar(dow_names, dow_counts.values, color=sns.color_palette("viridis", 7))
    ax.set_xlabel('星期', fontsize=12)
    ax.set_ylabel('订单数量', fontsize=12)
    ax.set_title('每周订单分布', fontsize=14, fontweight='bold')
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    dow_chart = os.path.join(output_dir, 'weekly_order_distribution.png')
    plt.savefig(dow_chart, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(dow_chart)
    print(f"  [OK] Generated: weekly_order_distribution.png")

    # 5. 每小时订单分布
    fig, ax = plt.subplots(figsize=(14, 6))
    hour_counts = df['purchase_hour'].value_counts().sort_index()
    ax.bar(hour_counts.index, hour_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('订单数量', fontsize=12)
    ax.set_title('每小时订单分布', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    plt.tight_layout()
    hour_chart = os.path.join(output_dir, 'hourly_order_distribution.png')
    plt.savefig(hour_chart, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(hour_chart)
    print(f"  [OK] Generated: hourly_order_distribution.png")

    return charts

def generate_summary_report(overview, status_analysis, temporal_analysis, delivery_stats, output_dir):
    """生成综合分析报告"""
    report = {
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'Orders.csv',
        'total_orders': overview['shape'][0],
        'data_quality': {
            'total_columns': overview['shape'][1],
            'missing_values': sum(overview['missing_values'].values()),
            'duplicate_rows': overview['duplicate_rows']
        },
        'order_status_distribution': status_analysis['percentages'],
        'delivery_performance': delivery_stats,
        'temporal_patterns': {
            'date_range': {
                'start': '2016-09-04',
                'end': '2018-10-17'
            },
            'peak_months': sorted(temporal_analysis['monthly_orders'].items(),
                                key=lambda x: x[1], reverse=True)[:5]
        },
        'key_insights': [
            f"数据集包含 {overview['shape'][0]:,} 个订单记录",
            f"主要订单状态: {status_analysis['counts']['delivered']} 个已交付订单 ({status_analysis['percentages']['delivered']:.1f}%)",
            f"平均配送时间: {delivery_stats['avg_delivery_days']:.1f} 天",
            f"延迟配送率: {delivery_stats['late_delivery_rate']:.1f}%",
            f"准时配送率: {100-delivery_stats['late_delivery_rate']:.1f}%"
        ]
    }

    # 保存JSON报告 (转换numpy类型为Python原生类型)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    report_serializable = convert_to_serializable(report)
    report_file = os.path.join(output_dir, 'analysis_summary.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, ensure_ascii=False, indent=2)

    # 生成Markdown报告
    md_file = os.path.join(output_dir, 'EDA_Report.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 订单数据探索性分析报告\n\n")
        f.write(f"**分析时间**: {report['analysis_timestamp']}\n\n")
        f.write(f"**数据集**: {report['dataset']}\n\n")

        f.write("## 1. 数据概览\n\n")
        f.write(f"- **总订单数**: {report['total_orders']:,}\n")
        f.write(f"- **字段数量**: {report['data_quality']['total_columns']}\n")
        f.write(f"- **缺失值**: {report['data_quality']['missing_values']:,}\n")
        f.write(f"- **重复行**: {report['data_quality']['duplicate_rows']}\n\n")

        f.write("## 2. 订单状态分布\n\n")
        f.write("| 状态 | 数量 | 占比 |\n")
        f.write("|------|------|------|\n")
        for status, pct in status_analysis['percentages'].items():
            count = status_analysis['counts'][status]
            f.write(f"| {status} | {count:,} | {pct:.2f}% |\n")
        f.write("\n")

        f.write("## 3. 配送绩效分析\n\n")
        f.write(f"- **平均实际配送天数**: {delivery_stats['avg_delivery_days']:.2f} 天\n")
        f.write(f"- **中位数配送天数**: {delivery_stats['median_delivery_days']:.2f} 天\n")
        f.write(f"- **平均预计配送天数**: {delivery_stats['avg_estimated_days']:.2f} 天\n")
        f.write(f"- **延迟配送率**: {delivery_stats['late_delivery_rate']:.2f}%\n")
        f.write(f"- **准时配送率**: {100-delivery_stats['late_delivery_rate']:.2f}%\n\n")

        f.write("## 4. 关键洞察\n\n")
        for i, insight in enumerate(report['key_insights'], 1):
            f.write(f"{i}. {insight}\n")

        f.write("\n## 5. 数据质量评估\n\n")
        f.write("### 优点\n")
        f.write("- 数据规模大，包含近10万条订单记录\n")
        f.write("- 时间跨度长，涵盖2016-2018年数据\n")
        f.write("- 字段丰富，包含订单全生命周期信息\n\n")

        f.write("### 改进建议\n")
        f.write("- 处理部分字段的缺失值（如审批时间、承运商交付时间）\n")
        f.write("- 清理重复记录\n")
        f.write("- 标准化日期格式\n\n")

    print(f"\n报告已保存:")
    print(f"  - {report_file}")
    print(f"  - {md_file}")

    return report

def main():
    """主函数"""
    input_file = "data_storage/Orders.csv"
    output_dir = "do_more_analysis/skill_execution/data-exploration-visualization"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("订单数据探索性分析 (EDA)")
    print("="*80)

    # 加载数据
    df = load_and_preprocess_data(input_file)

    # 生成数据概览
    overview = generate_data_overview(df)

    # 分析订单状态
    status_analysis = analyze_order_status(df)

    # 分析时间模式
    temporal_analysis = analyze_temporal_patterns(df)

    # 分析配送绩效
    delivery_stats = analyze_delivery_performance(df)

    # 创建可视化
    charts = create_visualizations(df, output_dir)

    # 生成报告
    report = generate_summary_report(
        overview, status_analysis, temporal_analysis, delivery_stats, output_dir
    )

    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - {len(charts)} 个可视化图表")
    print(f"  - analysis_summary.json (数据摘要)")
    print(f"  - EDA_Report.md (分析报告)")

if __name__ == "__main__":
    main()
