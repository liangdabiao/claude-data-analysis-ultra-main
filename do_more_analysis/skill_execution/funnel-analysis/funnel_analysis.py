"""
转化漏斗分析 (Funnel Analysis)
分析订单转化流程和各阶段转化率
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
    output_dir = 'do_more_analysis/skill_execution/funnel-analysis'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Order Conversion Funnel Analysis")
    print("="*80)

    # Load data
    print("\nLoading data...")
    orders = pd.read_csv('data_storage/Orders.csv', parse_dates=[
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date'
    ])

    print(f"Total orders: {len(orders):,}")

    # Define funnel stages
    print("\nBuilding conversion funnel...")

    funnel_stages = {
        'Order Created': len(orders),
        'Order Approved': orders['order_approved_at'].notna().sum(),
        'Shipped to Carrier': orders['order_delivered_carrier_date'].notna().sum(),
        'Delivered to Customer': orders['order_delivered_customer_date'].notna().sum()
    }

    # Calculate conversion rates
    stage_names = list(funnel_stages.keys())
    stage_counts = list(funnel_stages.values())

    conversion_rates = []
    for i in range(1, len(stage_counts)):
        rate = (stage_counts[i] / stage_counts[i-1]) * 100
        conversion_rates.append(rate)

    overall_conversion = (stage_counts[-1] / stage_counts[0]) * 100

    print("\nFunnel Stages:")
    for i, stage in enumerate(stage_names):
        print(f"  {i+1}. {stage}: {stage_counts[i]:,} orders")

    print("\nConversion Rates:")
    for i in range(len(stage_names) - 1):
        print(f"  {stage_names[i]} -> {stage_names[i+1]}: {conversion_rates[i]:.2f}%")

    print(f"\nOverall Conversion Rate: {overall_conversion:.2f}%")

    # Analyze by status
    print("\nAnalyzing order status distribution...")
    status_counts = orders['order_status'].value_counts()

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Funnel visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create funnel chart
    y_positions = range(len(stage_names)-1, -1, -1)
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    bars = ax.barh(y_positions, stage_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add count and percentage labels
    for i, (bar, count) in enumerate(zip(bars, stage_counts)):
        width = bar.get_width()
        pct = (count / stage_counts[0] * 100)
        ax.text(width + max(stage_counts)*0.01, bar.get_y() + bar.get_height()/2,
               f'{count:,} ({pct:.1f}%)', va='center', fontweight='bold', fontsize=11)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(stage_names, fontsize=12)
    ax.set_xlabel('Number of Orders', fontsize=12)
    ax.set_title('Order Conversion Funnel', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conversion_funnel.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] conversion_funnel.png")

    # 2. Conversion rates by stage
    fig, ax = plt.subplots(figsize=(10, 6))

    stage_labels = [f"{stage_names[i]}\n->\n{stage_names[i+1]}" for i in range(len(stage_names)-1)]
    colors_grad = ['#ff6b6b', '#feca57', '#48dbfb']

    bars = ax.bar(stage_labels, conversion_rates, color=colors_grad, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Conversion Rate (%)', fontsize=12)
    ax.set_title('Stage-by-Stage Conversion Rates', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, conversion_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{rate:.2f}%', ha='center', fontweight='bold', fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage_conversion_rates.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] stage_conversion_rates.png")

    # 3. Order status distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    status_colors = plt.cm.Set3(np.linspace(0, 1, len(status_counts)))
    bars = ax.bar(status_counts.index, status_counts.values, color=status_colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Order Status', fontsize=12)
    ax.set_ylabel('Number of Orders', fontsize=12)
    ax.set_title('Order Status Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    max_count = status_counts.values.max()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_count*0.01,
               f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'status_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] status_distribution.png")

    # 4. Funnel dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Funnel stages
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(stage_names[::-1], stage_counts[::-1], color=colors[::-1], alpha=0.8)
    ax1.set_xlabel('Orders', fontsize=11)
    ax1.set_title('Conversion Funnel Stages', fontweight='bold')
    ax1.invert_yaxis()

    # Conversion rates
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, len(conversion_rates)+1), conversion_rates,
            marker='o', linewidth=2.5, markersize=10, color='#e74c3c')
    ax2.fill_between(range(1, len(conversion_rates)+1), conversion_rates, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Stage Transition', fontsize=11)
    ax2.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax2.set_title('Conversion Rate Trend', fontweight='bold')
    ax2.set_xticks(range(1, len(conversion_rates)+1))
    ax2.grid(True, alpha=0.3)
    for i, rate in enumerate(conversion_rates):
        ax2.text(i+1, rate+1, f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Status distribution (pie)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
            startangle=90, colors=status_colors)
    ax3.set_title('Order Status Distribution', fontweight='bold')

    # Funnel metrics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    funnel_data = []
    for i in range(len(stage_names)):
        funnel_data.append([
            stage_names[i],
            f"{stage_counts[i]:,}",
            "100.00%" if i == 0 else f"{stage_counts[i]/stage_counts[0]*100:.2f}%",
            "N/A" if i == 0 else f"{conversion_rates[i-1]:.2f}%"
        ])

    table = ax4.table(cellText=funnel_data,
                     colLabels=['Stage', 'Orders', 'Cumulative %', 'Stage Conversion %'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    ax4.set_title('Funnel Metrics Summary', fontweight='bold', y=0.95)

    plt.savefig(os.path.join(output_dir, 'funnel_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] funnel_dashboard.png")

    # Export results
    print("\nExporting results...")

    # Funnel metrics
    funnel_df = pd.DataFrame({
        'Stage': stage_names,
        'Orders': stage_counts,
        'Cumulative_Percent': [count / stage_counts[0] * 100 for count in stage_counts],
        'Stage_Conversion_Percent': [100] + conversion_rates
    })
    funnel_df.to_csv(os.path.join(output_dir, 'funnel_metrics.csv'),
                    index=False, encoding='utf-8-sig')
    print("  [OK] funnel_metrics.csv")

    # JSON summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'funnel_stages': {stage: int(count) for stage, count in funnel_stages.items()},
        'conversion_rates': {f'stage_{i+1}_to_{i+2}': float(rate)
                            for i, rate in enumerate(conversion_rates)},
        'overall_conversion_rate': float(overall_conversion),
        'order_status_distribution': {k: int(v) for k, v in status_counts.items()}
    }

    with open(os.path.join(output_dir, 'funnel_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  [OK] funnel_summary.json")

    # Markdown report
    report = f"""# Order Conversion Funnel Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the analysis of order conversion funnel for the e-commerce platform.

### Key Metrics

- **Total Orders Created**: {stage_counts[0]:,}
- **Overall Conversion Rate**: {overall_conversion:.2f}%
- **Delivered Orders**: {stage_counts[-1]:,}

## Funnel Stages Analysis

| Stage | Orders | Cumulative % | Stage Conversion % |
|-------|--------|--------------|-------------------|
"""

    for i in range(len(stage_names)):
        cumulative_pct = stage_counts[i] / stage_counts[0] * 100
        stage_conv = "N/A" if i == 0 else f"{conversion_rates[i-1]:.2f}%"
        report += f"| {stage_names[i]} | {stage_counts[i]:,} | {cumulative_pct:.2f}% | {stage_conv} |\n"

    report += f"""

## Stage-by-Stage Conversion Rates

"""

    for i in range(len(stage_names) - 1):
        report += f"- **{stage_names[i]} to {stage_names[i+1]}**: {conversion_rates[i]:.2f}%\n"

    report += f"""

## Order Status Distribution

| Status | Count | Percentage |
|--------|-------|------------|
"""

    for status, count in status_counts.items():
        pct = count / len(orders) * 100
        report += f"| {status} | {count:,} | {pct:.2f}% |\n"

    report += f"""

## Key Findings

### Conversion Performance
1. **High Approval Rate**: {conversion_rates[0]:.2f}% of created orders are approved
2. **Shipping Efficiency**: {conversion_rates[1]:.2f}% of approved orders reach the carrier
3. **Delivery Success**: {conversion_rates[2]:.2f}% of shipped orders are delivered to customers
4. **Overall Funnel Efficiency**: {overall_conversion:.2f}% of created orders complete the full journey

### Bottlenecks Identified
"""

    # Identify weakest stage
    weakest_stage_idx = np.argmin(conversion_rates)
    report += f"- **Weakest Stage**: {stage_names[weakest_stage_idx]} to {stage_names[weakest_stage_idx+1]} ({conversion_rates[weakest_stage_idx]:.2f}%)\n"

    report += f"""
- **Non-Delivered Orders**: {stage_counts[0] - stage_counts[-1]:,} orders ({100-overall_conversion:.1f}%) do not reach delivery

### Order Status Insights
- **Delivered**: {status_counts.get('delivered', 0):,} orders ({status_counts.get('delivered', 0)/len(orders)*100:.1f}%)
- **Shipped**: {status_counts.get('shipped', 0):,} orders in transit
- **Canceled**: {status_counts.get('canceled', 0):,} orders canceled ({status_counts.get('canceled', 0)/len(orders)*100:.1f}%)

## Recommendations

### Process Optimization
1. **Approval Process**
   - Automate approval checks where possible
   - Reduce manual review time
   - Implement real-time validation

2. **Shipping Operations**
   - Improve carrier integration
   - Reduce handoff delays
   - Better tracking and notifications

3. **Delivery Completion**
   - Address common delivery failure reasons
   - Improve customer communication
   - Optimize last-mile delivery

### Customer Experience
1. **Order Transparency**
   - Real-time order status updates
   - Proactive delay notifications
   - Clear delivery expectations

2. **Failed Order Handling**
   - Quick resolution for canceled orders
   - Automated retry mechanisms
   - Customer feedback collection

3. **Process Monitoring**
   - Set up alerts for low conversion stages
   - Monitor stage performance in real-time
   - Continuously optimize weak points

## Files Generated

- funnel_metrics.csv: Detailed funnel metrics
- funnel_summary.json: Analysis summary in JSON format
- conversion_funnel.png: Visual funnel representation
- stage_conversion_rates.png: Stage conversion rates
- status_distribution.png: Order status distribution
- funnel_dashboard.png: Comprehensive funnel dashboard

---

*This analysis provides insights for optimizing the order conversion process.*
"""

    with open(os.path.join(output_dir, 'Funnel_Analysis_Report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("  [OK] Funnel_Analysis_Report.md")

    print("\n" + "="*80)
    print("Funnel Analysis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
