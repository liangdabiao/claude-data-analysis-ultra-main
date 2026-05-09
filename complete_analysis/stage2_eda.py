# -*- coding: utf-8 -*-
"""
Stage 2: 探索性数据分析 (Exploratory Data Analysis)
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Stage 2] {msg}", flush=True)

def main():
    log("="*60)
    log("Stage 2: 探索性数据分析开始")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data_storage")
    output_dir = os.path.join(base_dir, "exploratory_analysis")

    # 加载数据
    log("正在加载数据集...")
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))

    log(f"订单: {len(orders)}, 订单项: {len(order_items)}, 客户: {len(customers)}")

    # 1. 统计摘要
    log("正在生成统计摘要...")
    stats_summary = generate_statistical_summary(orders, order_items, payments, reviews, customers)

    # 2. 模式发现
    log("正在发现数据模式...")
    patterns = discover_patterns(orders, order_items, payments, reviews)

    # 3. 相关性分析
    log("正在进行相关性分析...")
    correlations = analyze_correlations(orders, order_items, payments, reviews)

    # 4. 异常检测
    log("正在检测异常...")
    anomalies = detect_anomalies(orders, order_items, payments)

    # 保存结果
    save_exploratory_results(stats_summary, patterns, correlations, anomalies, output_dir)

    log("="*60)
    log("Stage 2: 探索性数据分析完成")
    log("="*60)

def generate_statistical_summary(orders, order_items, payments, reviews, customers):
    """生成统计摘要"""
    summary = {}

    # 订单统计
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    summary['orders'] = {
        'total_orders': len(orders),
        'status_distribution': orders['order_status'].value_counts().to_dict(),
        'date_range': {
            'min': orders['order_purchase_timestamp'].min().isoformat(),
            'max': orders['order_purchase_timestamp'].max().isoformat()
        }
    }

    # 订单金额统计 (正确聚合!)
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    amounts = order_amounts.values

    summary['order_values'] = {
        'mean': float(amounts.mean()),
        'median': float(np.median(amounts)),
        'std': float(amounts.std()),
        'min': float(amounts.min()),
        'max': float(amounts.max()),
        'q25': float(np.percentile(amounts, 25)),
        'q75': float(np.percentile(amounts, 75)),
        'total': float(amounts.sum())
    }

    # 支付统计
    summary['payments'] = {
        'payment_types': payments['payment_type'].value_counts().to_dict(),
        'installments_stats': {
            'mean': float(payments['payment_installments'].mean()),
            'max': int(payments['payment_installments'].max())
        }
    }

    # 评论统计
    if 'review_score' in reviews.columns:
        summary['reviews'] = {
            'score_distribution': reviews['review_score'].value_counts().to_dict(),
            'avg_score': float(reviews['review_score'].mean())
        }

    # 客户统计
    summary['customers'] = {
        'unique_customers': orders['customer_id'].nunique(),
        'states': customers['customer_state'].value_counts().head(10).to_dict()
    }

    return summary

def discover_patterns(orders, order_items, payments, reviews):
    """发现数据模式"""
    patterns = {
        'temporal_patterns': [],
        'category_patterns': [],
        'behavioral_patterns': []
    }

    # 时间模式
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['month'] = orders['order_purchase_timestamp'].dt.to_period('M')
    monthly_orders = orders.groupby('month').size()
    patterns['temporal_patterns'].append({
        'type': 'monthly_trend',
        'description': '月度订单趋势',
        'data': {str(k): int(v) for k, v in monthly_orders.items()}
    })

    # 周几模式
    orders['weekday'] = orders['order_purchase_timestamp'].dt.day_name()
    weekday_orders = orders.groupby('weekday').size().sort_values(ascending=False)
    patterns['temporal_patterns'].append({
        'type': 'weekday_pattern',
        'description': '周几订单分布',
        'data': {k: int(v) for k, v in weekday_orders.items()}
    })

    # 支付模式
    payment_type_pct = (payments['payment_type'].value_counts() / len(payments) * 100).to_dict()
    patterns['category_patterns'].append({
        'type': 'payment_preference',
        'description': '支付方式偏好',
        'data': {k: float(v) for k, v in payment_type_pct.items()}
    })

    return patterns

def analyze_correlations(orders, order_items, payments, reviews):
    """相关性分析"""
    correlations = {
        'strong_correlations': [],
        'weak_correlations': [],
        'insights': []
    }

    # 订单金额与评论评分 (简单分析)
    order_review = orders.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')
    order_amounts = order_items.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum'}).sum(axis=1)
    order_review['total_amount'] = order_review['order_id'].map(order_amounts)

    valid_data = order_review.dropna(subset=['review_score', 'total_amount'])
    if len(valid_data) > 100:
        corr = valid_data['review_score'].corr(valid_data['total_amount'])
        correlations['insights'].append({
            'variable1': 'review_score',
            'variable2': 'total_amount',
            'correlation': float(corr),
            'interpretation': '订单金额与评分的相关性'
        })

    return correlations

def detect_anomalies(orders, order_items, payments):
    """异常检测"""
    anomalies = {
        'outliers': [],
        'unusual_patterns': []
    }

    # 订单金额异常值检测 (IQR方法)
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    amounts = order_amounts.values

    q1 = np.percentile(amounts, 25)
    q3 = np.percentile(amounts, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    high_outliers = order_amounts[order_amounts > upper]
    low_outliers = order_amounts[order_amounts < lower]

    if len(high_outliers) > 0:
        anomalies['outliers'].append({
            'type': 'high_value_orders',
            'count': len(high_outliers),
            'threshold': float(upper),
            'max_value': float(high_outliers.max())
        })

    # 支付异常
    high_installments = payments[payments['payment_installments'] > 12]
    if len(high_installments) > 0:
        anomalies['unusual_patterns'].append({
            'type': 'high_installment_payments',
            'count': len(high_installments),
            'max_installments': int(high_installments['payment_installments'].max())
        })

    return anomalies

def save_exploratory_results(stats_summary, patterns, correlations, anomalies, output_dir):
    """保存探索性分析结果"""

    # 1. 统计摘要
    stats_path = os.path.join(output_dir, "statistical_summary.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Category,Metric,Value\n")
        for category, data in stats_summary.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    if isinstance(value, dict):
                        for subk, subv in value.items():
                            f.write(f"{category},{metric}.{subk},{subv}\n")
                    else:
                        f.write(f"{category},{metric},{value}\n")
    log(f"统计摘要已保存: {stats_path}")

    # 2. JSON结果
    json_path = os.path.join(output_dir, "correlation_analysis.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats_summary': stats_summary,
            'patterns': patterns,
            'correlations': correlations,
            'anomalies': anomalies
        }, f, indent=2, ensure_ascii=False)
    log(f"分析结果JSON已保存: {json_path}")

    # 3. Markdown报告
    md_path = os.path.join(output_dir, "pattern_analysis.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 探索性数据分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 核心统计\n\n")
        f.write(f"- 总订单数: {stats_summary['orders']['total_orders']:,}\n")
        f.write(f"- 总营收: R$ {stats_summary['order_values']['total']:,.2f}\n")
        f.write(f"- 平均订单金额: R$ {stats_summary['order_values']['mean']:.2f}\n")
        f.write(f"- 中位数订单金额: R$ {stats_summary['order_values']['median']:.2f}\n")
        if 'reviews' in stats_summary:
            f.write(f"- 平均评分: {stats_summary['reviews']['avg_score']:.2f}/5\n")

        f.write("\n## 发现的模式\n\n")
        for pattern in patterns['temporal_patterns'] + patterns['category_patterns']:
            f.write(f"### {pattern['description']}\n")
            f.write(f"- 类型: {pattern['type']}\n\n")

        if correlations['insights']:
            f.write("\n## 相关性洞察\n\n")
            for insight in correlations['insights']:
                f.write(f"- {insight['interpretation']}: {insight['correlation']:.3f}\n")

        if anomalies['outliers'] or anomalies['unusual_patterns']:
            f.write("\n## 检测到的异常\n\n")
            for outlier in anomalies['outliers']:
                f.write(f"- {outlier['type']}: {outlier['count']} 个\n")
            for unusual in anomalies['unusual_patterns']:
                f.write(f"- {unusual['type']}: {unusual['count']} 个\n")

    log(f"模式分析报告已保存: {md_path}")

    # 保存执行摘要
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "workflow_log", "stage2_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Stage 2: 探索性数据分析\n\n")
        f.write(f"- 总订单数: {stats_summary['orders']['total_orders']:,}\n")
        f.write(f"- 总营收: R$ {stats_summary['order_values']['total']:,.2f}\n")
        f.write(f"- 发现模式数: {len(patterns['temporal_patterns']) + len(patterns['category_patterns'])}\n")
        f.write(f"- 检测异常数: {len(anomalies['outliers']) + len(anomalies['unusual_patterns'])}\n")
        f.write(f"- 状态: ✅ 完成\n")

if __name__ == "__main__":
    main()
