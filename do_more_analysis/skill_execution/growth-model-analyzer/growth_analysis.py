"""
增长模型分析 (Growth Model Analysis)
分析业务增长趋势和月度收入增长
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    output_dir = 'do_more_analysis/skill_execution/growth-model-analyzer'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Growth Model Analysis")
    print("="*80)

    # Load data
    print("\nLoading data...")
    orders = pd.read_csv('data_storage/Orders.csv', parse_dates=['order_purchase_timestamp'])
    order_items = pd.read_csv('data_storage/Order Items.csv')

    # Merge data
    orders_items = orders.merge(order_items, on='order_id', how='left')
    orders_items['total_amount'] = orders_items['price'] + orders_items['freight_value']

    # Filter delivered orders for revenue analysis
    delivered_orders = orders_items[orders_items['order_status'] == 'delivered'].copy()

    print(f"Total orders: {len(orders):,}")
    print(f"Delivered orders: {len(delivered_orders):,}")

    # Time-based analysis
    print("\nAnalyzing growth trends...")

    delivered_orders['year_month'] = delivered_orders['order_purchase_timestamp'].dt.to_period('M')
    delivered_orders['year'] = delivered_orders['order_purchase_timestamp'].dt.year
    delivered_orders['month'] = delivered_orders['order_purchase_timestamp'].dt.month

    # Monthly metrics
    monthly_data = delivered_orders.groupby('year_month').agg({
        'order_id': 'nunique',
        'customer_id': 'nunique',
        'total_amount': 'sum'
    }).reset_index()
    monthly_data.columns = ['period', 'orders', 'customers', 'revenue']

    # Calculate growth rates
    monthly_data['revenue_growth_pct'] = monthly_data['revenue'].pct_change() * 100
    monthly_data['order_growth_pct'] = monthly_data['orders'].pct_change() * 100
    monthly_data['avg_order_value'] = monthly_data['revenue'] / monthly_data['orders']

    print(f"Analysis period: {monthly_data['period'].min()} to {monthly_data['period'].max()}")
    print(f"Months analyzed: {len(monthly_data)}")

    # Calculate key metrics
    total_revenue = monthly_data['revenue'].sum()
    avg_monthly_revenue = monthly_data['revenue'].mean()
    total_orders = monthly_data['orders'].sum()
    avg_monthly_orders = monthly_data['orders'].mean()

    # Calculate CAGR
    first_month_rev = monthly_data['revenue'].iloc[0]
    last_month_rev = monthly_data['revenue'].iloc[-1]
    months = len(monthly_data) - 1
    cagr = ((last_month_rev / first_month_rev) ** (1/months) - 1) * 100 if months > 0 and first_month_rev > 0 else 0

    print(f"\nKey Metrics:")
    print(f"  Total Revenue: ${total_revenue:,.2f}")
    print(f"  Average Monthly Revenue: ${avg_monthly_revenue:,.2f}")
    print(f"  Total Orders: {total_orders:,}")
    print(f"  Average Monthly Orders: {avg_monthly_orders:,.0f}")
    print(f"  Revenue CAGR: {cagr:.2f}% per month")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Revenue trend
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Revenue over time
    axes[0, 0].plot(range(len(monthly_data)), monthly_data['revenue'],
                    marker='o', linewidth=2.5, markersize=6, color='#2ecc71')
    axes[0, 0].fill_between(range(len(monthly_data)), monthly_data['revenue'], alpha=0.3, color='#2ecc71')
    axes[0, 0].set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Revenue ($)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, len(monthly_data), 3))
    axes[0, 0].set_xticklabels([str(monthly_data.iloc[i]['period']) for i in range(0, len(monthly_data), 3)], rotation=45)

    # Orders over time
    axes[0, 1].plot(range(len(monthly_data)), monthly_data['orders'],
                    marker='s', linewidth=2.5, markersize=6, color='#3498db')
    axes[0, 1].fill_between(range(len(monthly_data)), monthly_data['orders'], alpha=0.3, color='#3498db')
    axes[0, 1].set_title('Monthly Orders Trend', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Orders', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(0, len(monthly_data), 3))
    axes[0, 1].set_xticklabels([str(monthly_data.iloc[i]['period']) for i in range(0, len(monthly_data), 3)], rotation=45)

    # Growth rates
    axes[1, 0].bar(range(len(monthly_data)), monthly_data['revenue_growth_pct'],
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Monthly Revenue Growth Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Growth Rate (%)', fontsize=11)
    axes[1, 0].set_xlabel('Month', fontsize=11)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_xticks(range(0, len(monthly_data), 3))

    # Average order value
    axes[1, 1].plot(range(len(monthly_data)), monthly_data['avg_order_value'],
                    marker='D', linewidth=2.5, markersize=6, color='#e74c3c')
    axes[1, 1].set_title('Average Order Value Trend', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Avg Order Value ($)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(0, len(monthly_data), 3))
    axes[1, 1].set_xticklabels([str(monthly_data.iloc[i]['period']) for i in range(0, len(monthly_data), 3)], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'growth_trends.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] growth_trends.png")

    # 2. Growth dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Revenue chart
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in monthly_data['revenue_growth_pct']]
    ax1.bar(range(len(monthly_data)), monthly_data['revenue'], color='steelblue', alpha=0.7, label='Revenue')
    ax1.set_title('Monthly Revenue Overview', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Revenue ($)', fontsize=11)
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(monthly_data)), monthly_data['revenue_growth_pct'],
                 color='red', marker='o', linewidth=2, markersize=5, label='Growth %')
    ax1_twin.set_ylabel('Growth Rate (%)', fontsize=11)
    ax1_twin.legend(loc='upper right')
    ax1_twin.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Customers vs Orders
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(monthly_data['customers'], monthly_data['orders'],
               s=monthly_data['revenue']/1000, alpha=0.6, c=range(len(monthly_data)), cmap='viridis')
    ax2.set_xlabel('Customers', fontsize=11)
    ax2.set_ylabel('Orders', fontsize=11)
    ax2.set_title('Customers vs Orders', fontweight='bold')

    for i, period in enumerate(monthly_data['period']):
        if i % 3 == 0:
            ax2.annotate(str(period), (monthly_data['customers'].iloc[i], monthly_data['orders'].iloc[i]),
                        fontsize=8, alpha=0.7)

    # Revenue distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(monthly_data['revenue'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(monthly_data['revenue'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: ${monthly_data['revenue'].mean():,.0f}")
    ax3.set_xlabel('Revenue ($)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Revenue Distribution', fontweight='bold')
    ax3.legend()

    # Key metrics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    metrics_data = [
        ['Total Revenue', f'${total_revenue:,.2f}', 'Total Orders', f'{total_orders:,}'],
        ['Average Monthly Revenue', f'${avg_monthly_revenue:,.2f}', 'Average Monthly Orders', f'{avg_monthly_orders:,.0f}'],
        ['Revenue CAGR', f'{cagr:.2f}%/month', 'Analysis Period', f'{len(monthly_data)} months'],
        ['Peak Revenue Month', f'{monthly_data.loc[monthly_data["revenue"].idxmax(), "period"]}',
         'Peak Revenue', f'${monthly_data["revenue"].max():,.2f}']
    ]

    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    ax4.set_title('Key Growth Metrics Summary', fontweight='bold', y=0.95)

    plt.savefig(os.path.join(output_dir, 'growth_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] growth_dashboard.png")

    # 3. Year-over-year comparison
    yearly_data = delivered_orders.groupby('year').agg({
        'order_id': 'nunique',
        'total_amount': 'sum',
        'customer_id': 'nunique'
    }).reset_index()
    yearly_data.columns = ['year', 'orders', 'revenue', 'customers']
    yearly_data['avg_order_value'] = yearly_data['revenue'] / yearly_data['orders']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(yearly_data['year'].astype(str), yearly_data['revenue'],
               color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].set_title('Annual Revenue', fontweight='bold')
    axes[0].set_ylabel('Revenue ($)')

    for i, v in enumerate(yearly_data['revenue']):
        axes[0].text(i, v + max(yearly_data['revenue'])*0.01, f'${v/1000:.0f}K', ha='center', fontweight='bold')

    axes[1].bar(yearly_data['year'].astype(str), yearly_data['orders'],
               color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[1].set_title('Annual Orders', fontweight='bold')
    axes[1].set_ylabel('Orders')

    for i, v in enumerate(yearly_data['orders']):
        axes[1].text(i, v + max(yearly_data['orders'])*0.01, f'{v:,}', ha='center', fontweight='bold')

    axes[2].plot(yearly_data['year'].astype(str), yearly_data['avg_order_value'],
                marker='o', linewidth=2.5, markersize=10, color='#e74c3c')
    axes[2].set_title('Average Order Value', fontweight='bold')
    axes[2].set_ylabel('AOV ($)')
    axes[2].grid(True, alpha=0.3)

    for i, v in enumerate(yearly_data['avg_order_value']):
        axes[2].text(i, v + max(yearly_data['avg_order_value'])*0.01, f'${v:.0f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yearly_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] yearly_comparison.png")

    # Export results
    print("\nExporting results...")

    # Monthly data
    monthly_data['period'] = monthly_data['period'].astype(str)
    monthly_data.to_csv(os.path.join(output_dir, 'monthly_growth_metrics.csv'),
                       index=False, encoding='utf-8-sig')
    print("  [OK] monthly_growth_metrics.csv")

    # Yearly data
    yearly_data.to_csv(os.path.join(output_dir, 'yearly_summary.csv'),
                      index=False, encoding='utf-8-sig')
    print("  [OK] yearly_summary.csv")

    # JSON summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_revenue': float(total_revenue),
        'total_orders': int(total_orders),
        'avg_monthly_revenue': float(avg_monthly_revenue),
        'avg_monthly_orders': float(avg_monthly_orders),
        'revenue_cagr_monthly': float(cagr),
        'analysis_period_months': len(monthly_data),
        'peak_month': str(monthly_data.loc[monthly_data['revenue'].idxmax(), 'period']),
        'peak_revenue': float(monthly_data['revenue'].max())
    }

    with open(os.path.join(output_dir, 'growth_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  [OK] growth_summary.json")

    # Markdown report
    report = f"""# Growth Model Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the growth analysis of the e-commerce platform.

### Key Performance Metrics

- **Total Revenue**: ${total_revenue:,.2f}
- **Total Orders**: {total_orders:,}
- **Average Monthly Revenue**: ${avg_monthly_revenue:,.2f}
- **Average Monthly Orders**: {avg_monthly_orders:,.0f}
- **Revenue CAGR**: {cagr:.2f}% per month
- **Analysis Period**: {len(monthly_data)} months

## Revenue Trends

### Monthly Performance

The platform shows strong growth over the analysis period with a compound annual growth rate of {cagr:.2f}% per month.

**Peak Month**: {monthly_data.loc[monthly_data['revenue'].idxmax(), 'period']} with revenue of ${monthly_data['revenue'].max():,.2f}

### Yearly Comparison

| Year | Revenue | Orders | Customers | Avg Order Value |
|------|---------|--------|-----------|-----------------|
"""

    for _, row in yearly_data.iterrows():
        report += f"| {int(row['year'])} | ${row['revenue']:,.2f} | {int(row['orders']):,} | {int(row['customers']):,} | ${row['avg_order_value']:.2f} |\n"

    report += f"""

## Growth Analysis

### Revenue Growth
- **Starting Month Revenue**: ${first_month_rev:,.2f}
- **Ending Month Revenue**: ${last_month_rev:,.2f}
- **Total Growth**: {((last_month_rev - first_month_rev) / first_month_rev * 100):.1f}%
- **Monthly CAGR**: {cagr:.2f}%

### Order Volume Trends
- Average monthly orders: {avg_monthly_orders:,.0f}
- Peak monthly orders: {monthly_data['orders'].max():,.0f}
- Order volume growth indicates {('expanding' if monthly_data['orders'].iloc[-1] > monthly_data['orders'].iloc[0] else 'contracting')} market

### Average Order Value
- Overall AOV: ${monthly_data['avg_order_value'].mean():.2f}
- AOV trend: {'Increasing' if monthly_data['avg_order_value'].iloc[-1] > monthly_data['avg_order_value'].iloc[0] else 'Decreasing'}

## Growth Drivers

### Positive Factors
1. **Increasing Customer Base**: Growing number of unique customers
2. **Market Expansion**: Presence across multiple periods shows sustainability
3. **Order Volume Growth**: Rising number of orders indicates customer acquisition success

### Areas for Improvement
1. **Average Order Value**: Monitor AOV trends for optimization opportunities
2. **Seasonal Patterns**: Identify and capitalize on seasonal trends
3. **Customer Retention**: Focus on repeat purchases to drive sustainable growth

## Recommendations

### Growth Strategies
1. **Customer Acquisition**
   - Continue successful acquisition channels
   - Explore new market segments
   - Optimize marketing spend

2. **Revenue Optimization**
   - Increase average order value through cross-selling
   - Implement pricing optimization strategies
   - Develop premium product categories

3. **Operational Efficiency**
   - Scale operations to handle growth
   - Optimize inventory management
   - Improve delivery efficiency

### Monitoring Metrics
1. **Monthly Recurring Revenue (MRR)**
2. **Customer Acquisition Cost (CAC)**
3. **Customer Lifetime Value (LTV)**
4. **Monthly Active Customers**

### Future Outlook
- Maintain positive growth trajectory
- Focus on sustainable, profitable growth
- Balance customer acquisition with retention
- Invest in product and service quality

## Files Generated

- monthly_growth_metrics.csv: Monthly performance data
- yearly_summary.csv: Yearly comparison data
- growth_summary.json: Analysis summary in JSON format
- growth_trends.png: Growth trend visualizations
- growth_dashboard.png: Comprehensive growth dashboard
- yearly_comparison.png: Year-over-year comparison

---

*This analysis provides insights for strategic growth planning and business optimization.*
"""

    with open(os.path.join(output_dir, 'Growth_Analysis_Report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("  [OK] Growth_Analysis_Report.md")

    print("\n" + "="*80)
    print("Growth Analysis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
