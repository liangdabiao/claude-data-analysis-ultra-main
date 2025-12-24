#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的RFM客户分群分析
Simplified RFM Customer Segmentation Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class SimpleRFMAnalyzer:
    """简化的RFM分析引擎"""

    def __init__(self, output_dir='rfm_output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, orders_path, order_items_path=None):
        """加载订单数据"""
        print("Loading data...")

        orders = pd.read_csv(orders_path, parse_dates=['order_purchase_timestamp'])

        if order_items_path:
            order_items = pd.read_csv(order_items_path)
            orders = orders.merge(order_items, on='order_id', how='left')

        orders = orders[orders['order_status'] == 'delivered'].copy()

        if 'price' in orders.columns and 'freight_value' in orders.columns:
            orders['total_amount'] = orders['price'] + orders['freight_value']
        else:
            orders['total_amount'] = orders['price'] if 'price' in orders.columns else 0

        return orders

    def calculate_rfm(self, orders):
        """计算RFM指标"""
        print("Calculating RFM...")

        analysis_date = orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

        rfm = orders.groupby('customer_id').agg({
            'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,
            'order_id': 'count',
            'total_amount': 'sum'
        }).reset_index()

        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        return rfm

    def score_rfm(self, rfm):
        """RFM评分 (1-4分制)"""
        print("Scoring RFM...")

        rfm['R_score'] = pd.qcut(rfm['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
        rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
        rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

        return rfm

    def create_segments(self, rfm):
        """创建客户分群"""
        print("Creating segments...")

        def create_segment(row):
            if row['RFM_score'] >= 10:
                return 'VIP_Clients'
            elif row['RFM_score'] >= 8:
                return 'Loyal_Customers'
            elif row['RFM_score'] >= 6:
                return 'Potential_Customers'
            elif row['RFM_score'] >= 4:
                return 'Regular_Customers'
            else:
                return 'At_Risk_Customers'

        rfm['segment'] = rfm.apply(create_segment, axis=1)
        return rfm

    def apply_clustering(self, rfm, n_clusters=4):
        """K-means聚类"""
        print("Performing K-means clustering...")

        rfm_data = rfm[['recency', 'frequency', 'monetary']].copy()
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

        return rfm

    def generate_visualizations(self, rfm):
        """生成可视化图表"""
        print("Generating visualizations...")

        # 1. 分群分布饼图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        segment_counts = rfm['segment'].value_counts()
        axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')

        segment_revenue = rfm.groupby('segment')['monetary'].sum()
        axes[1].pie(segment_revenue.values, labels=segment_revenue.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Revenue Contribution by Segment', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'segment_distribution.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 2. RFM箱线图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        rfm.boxplot(column='recency', by='segment', ax=axes[0])
        axes[0].set_title('Recency by Segment')

        rfm.boxplot(column='frequency', by='segment', ax=axes[1])
        axes[1].set_title('Frequency by Segment')

        rfm.boxplot(column='monetary', by='segment', ax=axes[2])
        axes[2].set_title('Monetary by Segment')

        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rfm_boxplots.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 3. 仪表板
        fig = plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 3, 1)
        segment_counts.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Customer Count by Segment', fontweight='bold')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax2 = plt.subplot(2, 3, 2)
        avg_rfm = rfm.groupby('segment')[['recency', 'frequency', 'monetary']].mean()
        avg_rfm.plot(kind='bar', ax=ax2, rot=0)
        ax2.set_title('Average RFM by Segment', fontweight='bold')
        ax2.legend(['Recency', 'Frequency', 'Monetary'])

        ax3 = plt.subplot(2, 3, 3)
        segment_revenue.plot(kind='bar', ax=ax3, color='green')
        ax3.set_title('Total Revenue by Segment', fontweight='bold')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax4 = plt.subplot(2, 1, 2)
        pivot = rfm.pivot_table(values='customer_id', index='R_score', columns='F_score', aggfunc='count')
        sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd', ax=ax4)
        ax4.set_title('RFM Customer Distribution Heatmap', fontweight='bold')
        ax4.set_xlabel('Frequency Score')
        ax4.set_ylabel('Recency Score')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rfm_dashboard.png'), dpi=200, bbox_inches='tight')
        plt.close()

    def export_results(self, rfm):
        """导出分析结果"""
        print("Exporting results...")

        customer_segments = rfm[['customer_id', 'recency', 'frequency', 'monetary',
                                  'R_score', 'F_score', 'M_score', 'RFM_score',
                                  'segment', 'cluster']].copy()
        customer_segments = customer_segments.sort_values('monetary', ascending=False)
        customer_segments.to_csv(os.path.join(self.output_dir, 'customer_segments.csv'),
                                 index=False, encoding='utf-8-sig')

        vip_customers = customer_segments.head(500).copy()
        vip_customers['rank'] = range(1, len(vip_customers) + 1)
        vip_customers.to_csv(os.path.join(self.output_dir, 'vip_customers_list.csv'),
                             index=False, encoding='utf-8-sig')

        summary = rfm.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        summary.columns = ['count', 'avg_recency', 'avg_frequency', 'avg_monetary']
        summary['total_revenue'] = rfm.groupby('segment')['monetary'].sum()
        summary.to_csv(os.path.join(self.output_dir, 'segment_summary_statistics.csv'))

        summary_json = {
            'timestamp': datetime.now().isoformat(),
            'total_customers': int(len(rfm)),
            'segments': {}
        }

        for seg in rfm['segment'].unique():
            seg_data = rfm[rfm['segment'] == seg]
            summary_json['segments'][seg] = {
                'count': int(len(seg_data)),
                'avg_monetary': float(seg_data['monetary'].mean()),
                'total_revenue': float(seg_data['monetary'].sum()),
                'avg_recency': float(seg_data['recency'].mean()),
                'avg_frequency': float(seg_data['frequency'].mean())
            }

        summary_json['key_metrics'] = {
            'total_revenue': float(rfm['monetary'].sum()),
            'avg_customer_value': float(rfm['monetary'].mean()),
            'avg_frequency': float(rfm['frequency'].mean()),
            'avg_recency': float(rfm['recency'].mean())
        }

        with open(os.path.join(self.output_dir, 'rfm_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

        return summary

    def generate_report(self, rfm, summary):
        """生成分析报告"""
        insights = f"""# RFM Customer Segmentation Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Customers Analyzed**: {len(rfm):,}
- **Total Revenue**: ${rfm['monetary'].sum():,.2f}
- **Average Customer Value**: ${rfm['monetary'].mean():.2f}
- **Segments Created**: {rfm['segment'].nunique()}

## Segment Overview

| Segment | Count | Avg Recency | Avg Frequency | Avg Monetary | Total Revenue |
|---------|-------|-------------|---------------|--------------|---------------|
"""
        for seg in summary.index:
            insights += f"| {seg} | {summary.loc[seg, 'count']:,} | {summary.loc[seg, 'avg_recency']:.1f} days | {summary.loc[seg, 'avg_frequency']:.2f} | ${summary.loc[seg, 'avg_monetary']:.2f} | ${summary.loc[seg, 'total_revenue']:,.2f} |\n"

        insights += f"""

## Key Insights

1. **High-Value Customers**: Top 20% contribute ${(rfm.nlargest(int(len(rfm)*0.2), 'monetary')['monetary'].sum() / rfm['monetary'].sum() * 100):.1f}% of revenue
2. **Customer Engagement**: Average purchase frequency is {rfm['frequency'].mean():.2f} orders
3. **Recency**: Customers purchased {rfm['recency'].mean():.0f} days ago on average

## Marketing Recommendations

### VIP Clients
- Provide exclusive offers and priority support
- Invite to new product launches
- Create loyalty programs
- Personalized recommendations

### Loyal Customers
- Cross-sell and up-sell opportunities
- Referral program incentives
- Premium service upgrades
- Early access to sales

### Potential Customers
- Re-engagement campaigns
- Special discount offers
- Product recommendations
- Email marketing sequences

### Regular Customers
- Seasonal promotions
- Brand awareness campaigns
- Educational content
- Community building

### At Risk Customers
- Win-back campaigns
- Special recovery offers
- Feedback surveys
- Understanding churn reasons
"""

        with open(os.path.join(self.output_dir, 'marketing_insights.md'), 'w', encoding='utf-8') as f:
            f.write(insights)

    def run_analysis(self, orders_path, order_items_path=None, n_clusters=4):
        """运行完整的RFM分析流程"""
        print("="*60)
        print("RFM Customer Segmentation Analysis")
        print("="*60)

        orders = self.load_data(orders_path, order_items_path)
        rfm = self.calculate_rfm(orders)
        rfm = self.score_rfm(rfm)
        rfm = self.create_segments(rfm)
        rfm = self.apply_clustering(rfm, n_clusters)

        print(f"Total customers analyzed: {len(rfm)}")
        print(f"Segments created: {rfm['segment'].nunique()}")

        self.generate_visualizations(rfm)
        summary = self.export_results(rfm)
        self.generate_report(rfm, summary)

        print("\n" + "="*60)
        print("RFM Analysis Complete!")
        print("="*60)
        print(f"Output directory: {self.output_dir}")

        return rfm, summary

def main():
    analyzer = SimpleRFMAnalyzer(output_dir='rfm_output')
    analyzer.run_analysis('data_storage/Orders.csv', 'data_storage/Order Items.csv')

if __name__ == "__main__":
    main()
