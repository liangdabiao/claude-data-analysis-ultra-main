"""
客户留存分析 (Retention Analysis)
分析客户重复购买行为和留存模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    output_dir = 'do_more_analysis/skill_execution/retention-analysis'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Customer Retention Analysis")
    print("="*80)

    # Load data
    print("\nLoading data...")
    orders = pd.read_csv('data_storage/Orders.csv', parse_dates=['order_purchase_timestamp'])
    orders = orders[orders['order_status'] == 'delivered'].copy()

    print(f"Total delivered orders: {len(orders):,}")
    print(f"Unique customers: {orders['customer_id'].nunique():,}")

    # Calculate customer metrics
    print("\nCalculating retention metrics...")

    # First and last purchase per customer
    customer_metrics = orders.groupby('customer_id').agg({
        'order_purchase_timestamp': ['min', 'max', 'count']
    }).reset_index()
    customer_metrics.columns = ['customer_id', 'first_purchase', 'last_purchase', 'purchase_count']

    # Calculate customer lifespan in days
    customer_metrics['lifespan_days'] = (
        customer_metrics['last_purchase'] - customer_metrics['first_purchase']
    ).dt.days + 1

    # Calculate repeat purchase rate
    repeat_customers = (customer_metrics['purchase_count'] > 1).sum()
    total_customers = len(customer_metrics)
    repeat_rate = (repeat_customers / total_customers) * 100

    print(f"Repeat purchase rate: {repeat_rate:.2f}%")
    print(f"Average purchases per customer: {customer_metrics['purchase_count'].mean():.2f}")
    print(f"Average customer lifespan: {customer_metrics['lifespan_days'].mean():.1f} days")

    # Cohort analysis by month
    print("\nPerforming cohort analysis...")
    orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
    orders['cohort'] = orders.groupby('customer_id')['order_purchase_timestamp'].transform('min').dt.to_period('M')

    cohort_data = orders.groupby(['cohort', 'order_month'])['customer_id'].nunique().reset_index()
    cohort_data['period_number'] = (
        cohort_data['order_month'] - cohort_data['cohort']
    ).apply(lambda x: x.n)

    cohort_pivot = cohort_data.pivot_table(
        index='cohort',
        columns='period_number',
        values='customer_id'
    )

    # Calculate retention rates
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100

    print(f"Cohorts analyzed: {len(retention)}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Retention heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(retention.iloc[:, :12], annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Retention Rate (%)'}, ax=ax)
    ax.set_title('Customer Retention Cohort Analysis (First 12 Months)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Months Since First Purchase', fontsize=12)
    ax.set_ylabel('Cohort (Month)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retention_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] retention_heatmap.png")

    # 2. Purchase distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(customer_metrics['purchase_count'], bins=range(1, 20),
                 color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(customer_metrics['purchase_count'].mean(),
                    color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {customer_metrics['purchase_count'].mean():.2f}")
    axes[0].set_xlabel('Number of Purchases', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].set_title('Purchase Frequency Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Customer segments
    segments = ['One-time', '2-3 purchases', '4-5 purchases', '6+ purchases']
    segment_counts = [
        (customer_metrics['purchase_count'] == 1).sum(),
        ((customer_metrics['purchase_count'] >= 2) & (customer_metrics['purchase_count'] <= 3)).sum(),
        ((customer_metrics['purchase_count'] >= 4) & (customer_metrics['purchase_count'] <= 5)).sum(),
        (customer_metrics['purchase_count'] >= 6).sum()
    ]

    axes[1].bar(segments, segment_counts, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    axes[1].set_ylabel('Number of Customers', fontsize=12)
    axes[1].set_title('Customer Purchase Frequency Segments', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)

    for i, v in enumerate(segment_counts):
        axes[1].text(i, v + max(segment_counts)*0.01, f'{v:,}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'purchase_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] purchase_distribution.png")

    # 3. Retention rate by cohort
    fig, ax = plt.subplots(figsize=(12, 6))

    # Average retention by period
    avg_retention = retention.mean(axis=0)[:12]

    ax.plot(range(1, len(avg_retention) + 1), avg_retention.values,
            marker='o', linewidth=2.5, markersize=8, color='#3498db')
    ax.fill_between(range(1, len(avg_retention) + 1), avg_retention.values, alpha=0.3, color='#3498db')
    ax.set_xlabel('Months Since First Purchase', fontsize=12)
    ax.set_ylabel('Average Retention Rate (%)', fontsize=12)
    ax.set_title('Average Customer Retention Rate Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 12.5)

    for i, v in enumerate(avg_retention.values):
        ax.text(i + 1, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retention_trend.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] retention_trend.png")

    # Export results
    print("\nExporting results...")

    # Customer metrics
    customer_metrics.to_csv(os.path.join(output_dir, 'customer_metrics.csv'),
                           index=False, encoding='utf-8-sig')
    print("  [OK] customer_metrics.csv")

    # Retention summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_customers': int(total_customers),
        'repeat_customers': int(repeat_customers),
        'repeat_purchase_rate': float(repeat_rate),
        'avg_purchases_per_customer': float(customer_metrics['purchase_count'].mean()),
        'avg_customer_lifespan_days': float(customer_metrics['lifespan_days'].mean()),
        'median_purchases': float(customer_metrics['purchase_count'].median())
    }

    with open(os.path.join(output_dir, 'retention_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  [OK] retention_summary.json")

    # Markdown report
    report = f"""# Customer Retention Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of customer retention analysis for the e-commerce platform.

### Key Metrics

- **Total Customers**: {total_customers:,}
- **Repeat Customers**: {repeat_customers:,}
- **Repeat Purchase Rate**: {repeat_rate:.2f}%
- **Average Purchases per Customer**: {customer_metrics['purchase_count'].mean():.2f}
- **Average Customer Lifespan**: {customer_metrics['lifespan_days'].mean():.1f} days

## Customer Purchase Frequency

| Segment | Count | Percentage |
|---------|-------|------------|
| One-time buyers | {segment_counts[0]:,} | {segment_counts[0]/total_customers*100:.1f}% |
| 2-3 purchases | {segment_counts[1]:,} | {segment_counts[1]/total_customers*100:.1f}% |
| 4-5 purchases | {segment_counts[2]:,} | {segment_counts[2]/total_customers*100:.1f}% |
| 6+ purchases | {segment_counts[3]:,} | {segment_counts[3]/total_customers*100:.1f}% |

## Retention Analysis

### Cohort Analysis

- Cohorts based on first purchase month
- Retention tracked for up to 12 months
- Average retention rates calculated

### Key Findings

1. **Purchase Frequency**
   - Only {repeat_rate:.1f}% of customers made more than one purchase
   - Median purchases: {customer_metrics['purchase_count'].median():.0f}
   - This indicates a one-time purchase dominant behavior

2. **Customer Lifespan**
   - Average active period: {customer_metrics['lifespan_days'].mean():.0f} days
   - Most customers have short engagement windows

3. **Retention Challenge**
   - Low repeat purchase rate suggests need for retention strategies
   - Focus on converting first-time buyers to repeat customers

## Recommendations

### Short-term Actions
1. **Post-Purchase Engagement**
   - Send follow-up emails 7-14 days after purchase
   - Offer discounts on second purchase
   - Implement loyalty program from first purchase

2. **Product Recommendations**
   - Personalized product suggestions based on purchase history
   - Cross-sell complementary products
   - Reminder emails for consumable/replenishable items

### Long-term Strategy
1. **Customer Lifecycle Management**
   - Segment customers by purchase behavior
   - Develop targeted retention campaigns
   - Implement customer win-back programs

2. **Loyalty Program**
   - Points-based rewards system
   - Tiered membership benefits
   - Exclusive offers for repeat customers

3. **Community Building**
   - User reviews and ratings
   - Social media engagement
   - User-generated content campaigns

## Files Generated

- customer_metrics.csv: Individual customer retention metrics
- retention_summary.json: Analysis summary in JSON format
- retention_heatmap.png: Cohort retention heatmap
- purchase_distribution.png: Purchase frequency distribution
- retention_trend.png: Average retention rate trend

---

*This analysis provides insights for improving customer retention and increasing lifetime value.*
"""

    with open(os.path.join(output_dir, 'Retention_Analysis_Report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("  [OK] Retention_Analysis_Report.md")

    print("\n" + "="*80)
    print("Retention Analysis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
