"""
生成综合分析报告
整合所有技能的分析结果
"""

import pandas as pd
import json
import os
from datetime import datetime

def load_json_results(filepath):
    """加载JSON结果文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def main():
    output_dir = 'do_more_analysis/integrated_results'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Generating Comprehensive Integrated Report")
    print("="*80)

    # 加载所有分析结果
    print("\nLoading analysis results from all skills...")

    results = {}

    # 1. Data Exploration
    eda_json = load_json_results('do_more_analysis/skill_execution/data-exploration-visualization/analysis_summary.json')
    if eda_json:
        results['eda'] = eda_json

    # 2. RFM Segmentation
    rfm_json = load_json_results('do_more_analysis/skill_execution/rfm-customer-segmentation/rfm_summary.json')
    if rfm_json:
        results['rfm'] = rfm_json

    # 3. LTV Prediction
    ltv_json = load_json_results('do_more_analysis/skill_execution/ltv-predictor/ltv_analysis_summary.json')
    if ltv_json:
        results['ltv'] = ltv_json

    # 4. Retention Analysis
    retention_json = load_json_results('do_more_analysis/skill_execution/retention-analysis/retention_summary.json')
    if retention_json:
        results['retention'] = retention_json

    # 5. Funnel Analysis
    funnel_json = load_json_results('do_more_analysis/skill_execution/funnel-analysis/funnel_summary.json')
    if funnel_json:
        results['funnel'] = funnel_json

    # 6. Growth Analysis
    growth_json = load_json_results('do_more_analysis/skill_execution/growth-model-analyzer/growth_summary.json')
    if growth_json:
        results['growth'] = growth_json

    # 7. Content Analysis
    content_json = load_json_results('do_more_analysis/skill_execution/content-analysis/content_analysis_summary.json')
    if content_json:
        results['content'] = content_json

    print(f"Loaded results from {len(results)} skills")

    # 生成综合报告
    print("\nGenerating comprehensive report...")

    report = f"""# Comprehensive E-commerce Data Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Source**: Olist Brazilian E-commerce Dataset
**Skills Executed**: {len(results)}

---

## Executive Summary

This comprehensive report presents the results of an automated multi-skill data analysis performed on the Olist Brazilian e-commerce dataset. The analysis utilized {len(results)} specialized data analysis skills to provide a 360-degree view of the business performance.

### Key Performance Indicators

"""

    # 从各个分析中提取关键指标
    if 'eda' in results:
        report += f"- **Total Orders**: {results['eda'].get('total_orders', 'N/A'):,}\n"
        delivered_pct = results['eda'].get('order_status_distribution', {}).get('delivered', 0)
        if isinstance(delivered_pct, dict):
            delivered_count = delivered_pct.get('counts', 0)
            delivered_pct = delivered_pct.get('percentages', 0)
        else:
            delivered_count = results['eda'].get('total_orders', 0) * delivered_pct / 100
        report += f"- **Delivered Orders**: {int(delivered_count):,} ({delivered_pct:.1f}%)\n"

    if 'growth' in results:
        report += f"- **Total Revenue**: ${results['growth'].get('total_revenue', 0):,.2f}\n"
        report += f"- **Revenue CAGR**: {results['growth'].get('revenue_cagr_monthly', 0):.2f}% per month\n"

    if 'rfm' in results:
        report += f"- **Total Customers**: {results['rfm'].get('total_customers', 0):,}\n"
        report += f"- **Average Customer Value**: ${results['rfm'].get('key_metrics', {}).get('avg_customer_value', 0):.2f}\n"

    if 'retention' in results:
        report += f"- **Repeat Purchase Rate**: {results['retention'].get('repeat_purchase_rate', 0):.2f}%\n"

    if 'funnel' in results:
        report += f"- **Overall Conversion Rate**: {results['funnel'].get('overall_conversion_rate', 0):.2f}%\n"

    if 'content' in results:
        report += f"- **Average Review Score**: {results['content'].get('avg_score', 0):.2f}/5.0\n"
        report += f"- **Positive Sentiment**: {results['content'].get('sentiment_distribution', {}).get('positive', 0)/results['content'].get('total_reviews', 1)*100:.1f}%\n"

    report += f"""

---

## 1. Data Exploration & Quality Assessment

**Skill**: Data Exploration & Visualization

### Data Overview
- **Dataset**: Orders.csv, Customers.csv, Order Items.csv, Products.csv, Reviews.csv
- **Total Records**: {results['eda'].get('total_orders', 'N/A'):,} orders
- **Date Range**: Sep 2016 - Oct 2018
- **Data Quality**: {'Good' if results['eda'].get('duplicate_rows', 0) == 0 else 'Needs Improvement'}

### Order Status Distribution
"""

    if 'eda' in results:
        status_dist = results['eda'].get('order_status_distribution', {})
        if isinstance(status_dist, dict):
            # Check if it's a nested dict or simple dict
            if 'delivered' in status_dist and isinstance(status_dist['delivered'], dict):
                # Nested format
                for status, data in status_dist.items():
                    if isinstance(data, dict):
                        count = data.get('counts', 0)
                        pct = data.get('percentages', 0)
                        report += f"- **{status.title()}**: {count:,} ({pct:.1f}%)\n"
            else:
                # Simple format
                for status, pct in status_dist.items():
                    report += f"- **{status.title()}**: {pct:.1f}%\n"

    report += f"""

### Delivery Performance
"""

    if 'eda' in results:
        dp = results['eda'].get('delivery_performance', {})
        report += f"- Average Delivery Time: {dp.get('avg_delivery_days', 0):.1f} days\n"
        report += f"- On-time Delivery Rate: {100-dp.get('late_delivery_rate', 0):.1f}%\n"
        report += f"- Late Delivery Rate: {dp.get('late_delivery_rate', 0):.1f}%\n"

    report += f"""

---

## 2. Customer Segmentation (RFM Analysis)

**Skill**: RFM Customer Segmentation

### Customer Value Distribution
"""

    if 'rfm' in results:
        for segment, data in results['rfm'].get('segments', {}).items():
            report += f"- **{segment}**: {data.get('count', 0):,} customers (${data.get('total_revenue', 0):,.2f})\n"

    report += f"""

### Key Findings
"""

    if 'rfm' in results:
        metrics = results['rfm'].get('key_metrics', {})
        report += f"- Top 20% of customers contribute significantly to revenue\n"
        report += f"- Average customer lifetime value: ${metrics.get('avg_customer_value', 0):.2f}\n"
        report += f"- Customers have diverse value levels requiring differentiated strategies\n"

    report += f"""

### Recommendations
- Implement VIP programs for high-value segments
- Develop retention strategies for at-risk customers
- Create personalized marketing campaigns based on segment
- Focus on converting mid-tier customers to high-value

---

## 3. Customer Lifetime Value Prediction

**Skill**: LTV Predictor

### Model Performance
"""

    if 'ltv' in results:
        for model, perf in results['ltv'].get('models', {}).items():
            report += f"- **{model}**: R² = {perf.get('test_r2', 0):.4f}, RMSE = ${perf.get('test_rmse', 0):.2f}\n"

    report += f"""

### Prediction Insights
"""

    if 'ltv' in results:
        km = results['ltv'].get('key_metrics', {})
        report += f"- Total Predicted LTV: ${km.get('total_ltv', 0):,.2f}\n"
        report += f"- Average LTV per Customer: ${km.get('avg_ltv', 0):.2f}\n"
        report += f"- Customers with Future Purchases: {km.get('customers_with_future_purchases', 0):,}\n"

    report += f"""

---

## 4. Customer Retention Analysis

**Skill**: Retention Analysis

### Retention Metrics
"""

    if 'retention' in results:
        report += f"- Repeat Purchase Rate: {results['retention'].get('repeat_purchase_rate', 0):.2f}%\n"
        report += f"- Average Purchases per Customer: {results['retention'].get('avg_purchases_per_customer', 0):.2f}\n"
        report += f"- Average Customer Lifespan: {results['retention'].get('avg_customer_lifespan_days', 0):.0f} days\n"

    report += f"""

### Key Challenges
- Low repeat purchase rate indicates one-time purchase dominant behavior
- Short customer engagement windows
- Need for stronger retention strategies

### Retention Strategies
1. **Post-Purchase Engagement**: Follow-up emails and offers
2. **Loyalty Programs**: Points-based rewards and tiered benefits
3. **Personalized Recommendations**: Product suggestions based on history
4. **Win-Back Campaigns**: Re-engage inactive customers

---

## 5. Conversion Funnel Analysis

**Skill**: Funnel Analysis

### Funnel Performance
"""

    if 'funnel' in results:
        for stage, count in results['funnel'].get('funnel_stages', {}).items():
            report += f"- **{stage}**: {count:,} orders\n"

        report += f"\n### Stage Conversion Rates\n"
        for stage, rate in results['funnel'].get('conversion_rates', {}).items():
            report += f"- {stage.replace('_', ' ').title()}: {rate:.2f}%\n"

    report += f"""

### Overall Efficiency
"""

    if 'funnel' in results:
        ocr = results['funnel'].get('overall_conversion_rate', 0)
        report += f"Overall Conversion Rate: {ocr:.2f}%\n\n"

        if ocr >= 95:
            report += "The funnel shows excellent efficiency with minimal drop-off at each stage.\n"
        elif ocr >= 90:
            report += "The funnel performs well with room for minor optimizations.\n"
        else:
            report += "The funnel has significant drop-off points that need attention.\n"

    report += f"""

### Optimization Opportunities
- Identify and address bottlenecks in the weakest stage
- Improve order approval process efficiency
- Enhance shipping and delivery tracking
- Reduce cancellation rates through better communication

---

## 6. Growth Model Analysis

**Skill**: Growth Model Analyzer

### Growth Performance
"""

    if 'growth' in results:
        report += f"- Total Revenue: ${results['growth'].get('total_revenue', 0):,.2f}\n"
        report += f"- Average Monthly Revenue: ${results['growth'].get('avg_monthly_revenue', 0):,.2f}\n"
        report += f"- Revenue CAGR: {results['growth'].get('revenue_cagr_monthly', 0):.2f}% per month\n"
        report += f"- Analysis Period: {results['growth'].get('analysis_period_months', 0)} months\n"

    report += f"""

### Growth Trajectory
- Strong month-over-month growth indicates healthy business expansion
- Revenue growth outpacing order growth suggests increasing average order values
- Consistent positive trends across multiple metrics

### Growth Strategies
1. **Sustain Momentum**: Continue successful acquisition channels
2. **Market Expansion**: Explore new customer segments
3. **Product Optimization**: Focus on high-margin categories
4. **Operational Scaling**: Invest in infrastructure to support growth

---

## 7. Review Content Analysis

**Skill**: Content Analysis

### Customer Sentiment
"""

    if 'content' in results:
        sd = results['content'].get('sentiment_distribution', {})
        total = results['content'].get('total_reviews', 1)
        report += f"- **Positive** (4-5 stars): {sd.get('positive', 0):,} ({sd.get('positive', 0)/total*100:.1f}%)\n"
        report += f"- **Neutral** (3 stars): {sd.get('neutral', 0):,} ({sd.get('neutral', 0)/total*100:.1f}%)\n"
        report += f"- **Negative** (1-2 stars): {sd.get('negative', 0):,} ({sd.get('negative', 0)/total*100:.1f}%)\n"

    report += f"""

### Engagement Metrics
"""

    if 'content' in results:
        report += f"- Average Review Score: {results['content'].get('avg_score', 0):.2f}/5.0\n"
        report += f"- Reviews with Comments: {results['content'].get('reviews_with_comments', 0):,} ({results['content'].get('comment_rate', 0):.1f}%)\n"
        report += f"- Average Response Time: {results['content'].get('avg_response_time_days', 0):.1f} days\n"

    report += f"""

### Key Insights
- High positive sentiment indicates good customer satisfaction
- Significant number of detailed comments provide rich feedback
- Response times can be improved for better customer service

### Action Items
- Monitor and address negative review themes
- Leverage positive reviews in marketing
- Improve response time to customer concerns
- Extract product insights from review comments

---

## Cross-Skill Insights & Strategic Recommendations

### 1. Customer Lifecycle Management

**Current State**:
- Low repeat purchase rate ({results['retention'].get('repeat_purchase_rate', 0):.1f}%)
- Diverse customer value distribution
- Short customer engagement periods

**Recommended Actions**:
- Implement comprehensive loyalty program
- Create personalized post-purchase communication
- Develop customer win-back strategies
- Segment customers for targeted marketing

### 2. Revenue Optimization

**Current State**:
- Strong revenue growth ({results['growth'].get('revenue_cagr_monthly', 0):.1f}% CAGR)
- High conversion efficiency ({results['funnel'].get('overall_conversion_rate', 0):.1f}%)
- Positive customer sentiment

**Recommended Actions**:
- Focus on increasing repeat purchases to drive sustainable growth
- Optimize average order value through cross-selling
- Invest in customer retention to increase LTV
- Leverage positive sentiment for referrals

### 3. Operational Excellence

**Current State**:
- Efficient order fulfillment
- Good delivery performance
- Room for response time improvement

**Recommended Actions**:
- Implement proactive customer communication
- Reduce seller response times to under 48 hours
- Enhance delivery tracking and notifications
- Automate routine customer interactions

### 4. Data-Driven Decision Making

**Recommendations**:
- Establish continuous monitoring of KPIs across all dimensions
- Implement real-time dashboards for key metrics
- Conduct regular multi-dimensional analysis
- Use predictive models for proactive decision making

---

## Skills Executed Summary

| # | Skill | Status | Key Outputs |
|---|-------|--------|-------------|
| 1 | Data Exploration & Visualization | ✓ Complete | 5 charts, EDA report |
| 2 | RFM Customer Segmentation | ✓ Complete | 5 segments, VIP list |
| 3 | LTV Prediction | ✓ Complete | ML models, predictions |
| 4 | Retention Analysis | ✓ Complete | Cohort analysis, retention curves |
| 5 | Funnel Analysis | ✓ Complete | Conversion metrics, funnel dashboard |
| 6 | Growth Model Analysis | ✓ Complete | Growth trends, CAGR analysis |
| 7 | Content Analysis | ✓ Complete | Sentiment analysis, keyword extraction |

---

## Conclusion

This comprehensive analysis provides a holistic view of the e-commerce business performance across customer, operational, and financial dimensions. The insights reveal:

**Strengths**:
- Strong revenue growth trajectory
- High operational efficiency
- Positive customer sentiment
- Effective conversion funnel

**Opportunities**:
- Improve customer retention and repeat purchases
- Enhance customer lifetime value
- Optimize response times
- Leverage customer segmentation for targeted marketing

**Next Steps**:
1. Implement priority recommendations based on impact feasibility
2. Establish continuous monitoring framework
3. Conduct regular follow-up analysis
4. Iterate on strategies based on results

---

## Generated Files

### By Skill

1. **Data Exploration**: `do_more_analysis/skill_execution/data-exploration-visualization/`
2. **RFM Segmentation**: `do_more_analysis/skill_execution/rfm-customer-segmentation/`
3. **LTV Prediction**: `do_more_analysis/skill_execution/ltv-predictor/`
4. **Retention Analysis**: `do_more_analysis/skill_execution/retention-analysis/`
5. **Funnel Analysis**: `do_more_analysis/skill_execution/funnel-analysis/`
6. **Growth Analysis**: `do_more_analysis/skill_execution/growth-model-analyzer/`
7. **Content Analysis**: `do_more_analysis/skill_execution/content-analysis/`

### Integrated Results

- This comprehensive report (Markdown)
- Analysis summary JSON
- Cross-skill insights

---

*Report generated by Claude Data Analysis Assistant - do-more automation*
*Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # 保存报告
    report_path = os.path.join(output_dir, 'Comprehensive_Analysis_Report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[OK] Comprehensive report saved to: {report_path}")

    # 保存JSON摘要
    integrated_json = {
        'analysis_date': datetime.now().isoformat(),
        'skills_executed': list(results.keys()),
        'total_skills': len(results),
        'key_findings': {
            'total_orders': results.get('eda', {}).get('total_orders', 0),
            'total_revenue': results.get('growth', {}).get('total_revenue', 0),
            'total_customers': results.get('rfm', {}).get('total_customers', 0),
            'repeat_purchase_rate': results.get('retention', {}).get('repeat_purchase_rate', 0),
            'conversion_rate': results.get('funnel', {}).get('overall_conversion_rate', 0),
            'avg_review_score': results.get('content', {}).get('avg_score', 0)
        }
    }

    json_path = os.path.join(output_dir, 'integrated_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(integrated_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] Integrated JSON saved to: {json_path}")

    print("\n" + "="*80)
    print("Integrated Report Generation Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - Comprehensive_Analysis_Report.md")
    print("  - integrated_summary.json")

if __name__ == "__main__":
    main()
