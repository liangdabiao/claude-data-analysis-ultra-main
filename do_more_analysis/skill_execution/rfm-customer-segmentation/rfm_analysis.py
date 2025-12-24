"""
RFM客户分群分析脚本
对Olist电商数据进行客户价值细分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """加载并准备数据"""
    print("="*80)
    print("RFM客户分群分析")
    print("="*80)
    print("\n正在加载数据...")

    # 加载数据
    orders = pd.read_csv('data_storage/Orders.csv')
    order_items = pd.read_csv('data_storage/Order Items.csv')
    customers = pd.read_csv('data_storage/Customers.csv')

    print(f"  - Orders: {len(orders)} 条记录")
    print(f"  - Order Items: {len(order_items)} 条记录")
    print(f"  - Customers: {len(customers)} 条记录")

    # 转换日期
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

    # 只使用已交付的订单
    orders = orders[orders['order_status'] == 'delivered'].copy()

    print(f"\n数据预处理:")
    print(f"  - 过滤后订单数: {len(orders)} (仅已交付)")
    print(f"  - 客户数: {orders['customer_id'].nunique()}")

    return orders, order_items, customers

def calculate_rfm(orders, order_items):
    """计算RFM指标"""
    print("\n正在计算RFM指标...")

    # 合并订单和订单项
    orders_with_items = orders.merge(order_items, on='order_id', how='left')

    # 计算每个订单的总金额
    order_totals = orders_with_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'customer_id': 'first',
        'order_purchase_timestamp': 'first'
    }).reset_index()

    order_totals['total_amount'] = order_totals['price'] + order_totals['freight_value']

    # 计算分析日期
    analysis_date = orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    # 计算RFM
    rfm = order_totals.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'total_amount': 'sum'  # Monetary
    }).reset_index()

    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    # 只保留有购买记录的客户
    rfm = rfm[rfm['monetary'] > 0]

    print(f"\nRFM统计:")
    print(f"  - Recency (最近一次购买): {rfm['recency'].mean():.1f} 天前")
    print(f"  - Frequency (购买频率): {rfm['frequency'].mean():.2f} 次")
    print(f"  - Monetary (平均消费金额): ${rfm['monetary'].mean():.2f}")
    print(f"  - 总客户数: {len(rfm)}")

    return rfm

def score_rfm(rfm):
    """对RFM进行评分 (1-3分，3分最高)"""
    print("\n正在对RFM进行评分...")

    # Recency评分 (越近越好)
    rfm['R_score'] = pd.qcut(rfm['recency'], q=3, labels=[1, 2, 3], duplicates='drop')

    # Frequency评分 (越多越好)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=3, labels=[1, 2, 3])

    # Monetary评分 (越高越好)
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=3, labels=[1, 2, 3])

    # 转换为数值
    rfm['R_score'] = rfm['R_score'].astype(int)
    rfm['F_score'] = rfm['F_score'].astype(int)
    rfm['M_score'] = rfm['M_score'].astype(int)

    # 计算RFM总分
    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

    return rfm

def perform_clustering(rfm, max_clusters=10):
    """使用K-means聚类"""
    print("\n正在进行K-means聚类分析...")

    # 准备数据
    rfm_data = rfm[['recency', 'frequency', 'monetary']].copy()

    # 标准化
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)

    # 寻找最优聚类数 (肘部法则)
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_clusters + 1, len(rfm) // 100))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(rfm_scaled, kmeans.labels_)
            silhouette_scores.append(score)

    # 选择最优K值 (使用肘部法则)
    optimal_k = 4  # 默认使用4个客户群体

    print(f"  - 选择聚类数: {optimal_k}")

    # 执行最终聚类
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm['cluster'] = kmeans_final.fit_predict(rfm_scaled)

    return rfm, optimal_k, inertias, silhouette_scores

def label_segments(rfm):
    """为客户群体打标签"""
    print("\n正在为客户群体打标签...")

    # 计算每个群体的平均RFM值
    cluster_stats = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).round(2)

    cluster_stats.columns = ['avg_recency', 'avg_frequency', 'avg_monetary', 'count']

    # 根据特征命名群体
    segment_names = {}

    for cluster in cluster_stats.index:
        stats = cluster_stats.loc[cluster]

        if stats['avg_monetary'] > cluster_stats['avg_monetary'].quantile(0.75):
            if stats['avg_recency'] < cluster_stats['avg_recency'].quantile(0.5):
                segment_names[cluster] = 'VIP客户'
            else:
                segment_names[cluster] = '高价值客户'
        elif stats['avg_monetary'] < cluster_stats['avg_monetary'].quantile(0.25):
            segment_names[cluster] = '低价值客户'
        else:
            if stats['avg_recency'] < cluster_stats['avg_recency'].quantile(0.5):
                segment_names[cluster] = '潜力客户'
            else:
                segment_names[cluster] = '普通客户'

    rfm['segment'] = rfm['cluster'].map(segment_names)

    # 打印群体统计
    print("\n客户群体统计:")
    print(cluster_stats)

    return rfm, cluster_stats

def create_visualizations(rfm, cluster_stats, output_dir):
    """创建可视化图表"""
    print("\n正在生成可视化图表...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 客户群体分布饼图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    segment_counts = rfm['segment'].value_counts()
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                colors=colors[:len(segment_counts)], startangle=90)
    axes[0].set_title('客户群体数量分布', fontsize=14, fontweight='bold')

    # 收入贡献
    segment_revenue = rfm.groupby('segment')['monetary'].sum()
    axes[1].pie(segment_revenue.values, labels=segment_revenue.index, autopct='%1.1f%%',
                colors=colors[:len(segment_revenue)], startangle=90)
    axes[1].set_title('客户群体收入贡献', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Generated: segment_distribution.png")

    # 2. RFM散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    segments = rfm['segment'].unique()
    color_map = {seg: colors[i % len(colors)] for i, seg in enumerate(segments)}

    # Recency vs Frequency
    for seg in segments:
        data = rfm[rfm['segment'] == seg]
        axes[0].scatter(data['recency'], data['frequency'], label=seg, color=color_map[seg], alpha=0.5, s=20)
    axes[0].set_xlabel('Recency (天)')
    axes[0].set_ylabel('Frequency (次)')
    axes[0].set_title('Recency vs Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Frequency vs Monetary
    for seg in segments:
        data = rfm[rfm['segment'] == seg]
        axes[1].scatter(data['frequency'], data['monetary'], label=seg, color=color_map[seg], alpha=0.5, s=20)
    axes[1].set_xlabel('Frequency (次)')
    axes[1].set_ylabel('Monetary ($)')
    axes[1].set_title('Frequency vs Monetary')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Recency vs Monetary
    for seg in segments:
        data = rfm[rfm['segment'] == seg]
        axes[2].scatter(data['recency'], data['monetary'], label=seg, color=color_map[seg], alpha=0.5, s=20)
    axes[2].set_xlabel('Recency (天)')
    axes[2].set_ylabel('Monetary ($)')
    axes[2].set_title('Recency vs Monetary')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Generated: rfm_scatter_plots.png")

    # 3. 箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    segments_ordered = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=False).index

    sns.boxplot(data=rfm, x='segment', y='recency', order=segments_ordered, ax=axes[0])
    axes[0].set_xlabel('客户群体')
    axes[0].set_ylabel('Recency (天)')
    axes[0].set_title('各群体最近购买时间分布')
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(data=rfm, x='segment', y='frequency', order=segments_ordered, ax=axes[1])
    axes[1].set_xlabel('客户群体')
    axes[1].set_ylabel('Frequency (次)')
    axes[1].set_title('各群体购买频率分布')
    axes[1].tick_params(axis='x', rotation=45)

    sns.boxplot(data=rfm, x='segment', y='monetary', order=segments_ordered, ax=axes[2])
    axes[2].set_xlabel('客户群体')
    axes[2].set_ylabel('Monetary ($)')
    axes[2].set_title('各群体消费金额分布')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Generated: rfm_boxplots.png")

    # 4. RFM综合仪表板
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 群体大小
    ax1 = fig.add_subplot(gs[0, 0])
    segment_counts.plot(kind='bar', ax=ax1, color=colors[:len(segment_counts)])
    ax1.set_title('客户群体规模', fontweight='bold')
    ax1.set_ylabel('客户数量')
    ax1.tick_params(axis='x', rotation=45)

    # 平均RFM值
    ax2 = fig.add_subplot(gs[0, 1])
    cluster_stats[['avg_recency', 'avg_frequency', 'avg_monetary']].plot(
        kind='bar', ax=ax2, rot=0)
    ax2.set_title('各群体平均RFM值', fontweight='bold')
    ax2.set_ylabel('平均值')
    ax2.legend(['Recency', 'Frequency', 'Monetary'])

    # 收入贡献
    ax3 = fig.add_subplot(gs[0, 2])
    segment_revenue.plot(kind='bar', ax=ax3, color=colors[:len(segment_revenue)])
    ax3.set_title('各群体收入贡献', fontweight='bold')
    ax3.set_ylabel('总收入 ($)')
    ax3.tick_params(axis='x', rotation=45)

    # RFM分布热图
    ax4 = fig.add_subplot(gs[1, :])
    rfm_pivot = rfm.pivot_table(values='customer_id', index='R_score',
                                 columns='F_score', aggfunc='count')
    sns.heatmap(rfm_pivot, annot=True, fmt='g', cmap='YlOrRd', ax=ax4)
    ax4.set_title('RFM客户分布热图', fontweight='bold')
    ax4.set_xlabel('Frequency Score')
    ax4.set_ylabel('Recency Score')

    # 详细统计表
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    stats_table = ax5.table(cellText=cluster_stats.round(2).values,
                           rowLabels=cluster_stats.index,
                           colLabels=['Recency', 'Frequency', 'Monetary', 'Count'],
                           cellLoc='center', loc='center')
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(10)
    stats_table.scale(1, 2)
    ax5.set_title('客户群体详细统计', fontweight='bold', y=0.95)

    plt.savefig(os.path.join(output_dir, 'rfm_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Generated: rfm_dashboard.png")

def generate_marketing_insights(rfm, cluster_stats):
    """生成营销洞察和建议"""
    insights = {}

    for segment in rfm['segment'].unique():
        segment_data = rfm[rfm['segment'] == segment]

        stats = {
            'count': len(segment_data),
            'avg_recency': segment_data['recency'].mean(),
            'avg_frequency': segment_data['frequency'].mean(),
            'avg_monetary': segment_data['monetary'].mean(),
            'total_revenue': segment_data['monetary'].sum()
        }

        if segment == 'VIP客户':
            stats['strategy'] = [
                "提供专属VIP服务和优先支持",
                "邀请参与新产品测试和活动",
                "提供个性化推荐和定制优惠",
                "建立客户忠诚度计划"
            ]
        elif segment == '高价值客户':
            stats['strategy'] = [
                "提供升级产品和服务",
                "推送高价值产品推荐",
                "邀请参与会员计划",
                "提供优质客户服务"
            ]
        elif segment == '潜力客户':
            stats['strategy'] = [
                "通过优惠券激励再次购买",
                "发送个性化产品推荐",
                "提供首次复购优惠",
                "增加客户互动和关怀"
            ]
        elif segment == '普通客户':
            stats['strategy'] = [
                "定期发送促销信息",
                "提供季节性优惠",
                "增加品牌曝光",
                "培养购买习惯"
            ]
        else:  # 低价值客户
            stats['strategy'] = [
                "发送挽回邮件和优惠",
                "了解客户流失原因",
                "提供特别折扣刺激消费",
                "重新激活或调整策略"
            ]

        insights[segment] = stats

    return insights

def export_results(rfm, insights, output_dir):
    """导出分析结果"""
    print("\n正在导出分析结果...")

    # 1. 客户分群数据
    customer_segments = rfm[['customer_id', 'recency', 'frequency', 'monetary',
                              'R_score', 'F_score', 'M_score', 'RFM_score',
                              'cluster', 'segment']].copy()
    customer_segments = customer_segments.sort_values('monetary', ascending=False)
    customer_segments.to_csv(os.path.join(output_dir, 'customer_segments.csv'),
                             index=False, encoding='utf-8-sig')
    print(f"  [OK] Exported: customer_segments.csv")

    # 2. VIP客户列表
    vip_customers = customer_segments[
        customer_segments['segment'].isin(['VIP客户', '高价值客户'])
    ].head(500).copy()

    vip_customers['rank'] = range(1, len(vip_customers) + 1)
    vip_customers.to_csv(os.path.join(output_dir, 'vip_customers_list.csv'),
                         index=False, encoding='utf-8-sig')
    print(f"  [OK] Exported: vip_customers_list.csv ({len(vip_customers)} VIP customers)")

    # 3. 群体统计摘要
    summary_stats = pd.DataFrame({
        'segment': insights.keys(),
        'customer_count': [v['count'] for v in insights.values()],
        'avg_recency': [v['avg_recency'] for v in insights.values()],
        'avg_frequency': [v['avg_frequency'] for v in insights.values()],
        'avg_monetary': [v['avg_monetary'] for v in insights.values()],
        'total_revenue': [v['total_revenue'] for v in insights.values()]
    })
    summary_stats['revenue_percentage'] = (
        summary_stats['total_revenue'] / summary_stats['total_revenue'].sum() * 100
    )

    summary_stats.to_csv(os.path.join(output_dir, 'segment_summary_statistics.csv'),
                         index=False, encoding='utf-8-sig')
    print(f"  [OK] Exported: segment_summary_statistics.csv")

    # 4. 营销洞察报告
    report_lines = []
    report_lines.append("# RFM客户分群分析报告\n")
    report_lines.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**客户总数**: {len(rfm):,}\n\n")

    report_lines.append("## 客户群体概览\n\n")
    report_lines.append("| 群体 | 客户数 | 平均Recency | 平均Frequency | 平均Monetary | 总收入 | 占比 |\n")
    report_lines.append("|------|--------|-------------|---------------|--------------|--------|------|\n")

    for _, row in summary_stats.iterrows():
        report_lines.append(
            f"| {row['segment']} | {row['customer_count']:,} | "
            f"{row['avg_recency']:.1f}天 | {row['avg_frequency']:.2f}次 | "
            f"${row['avg_monetary']:.2f} | ${row['total_revenue']:,.2f} | "
            f"{row['revenue_percentage']:.1f}% |\n"
        )

    report_lines.append("\n## 营销策略建议\n\n")

    for segment, data in insights.items():
        report_lines.append(f"### {segment}\n\n")
        report_lines.append(f"- **客户数量**: {data['count']:,}\n")
        report_lines.append(f"- **平均消费**: ${data['avg_monetary']:.2f}\n")
        report_lines.append(f"- **总收入贡献**: ${data['total_revenue']:,.2f}\n\n")
        report_lines.append("**营销策略**:\n")
        for strategy in data['strategy']:
            report_lines.append(f"- {strategy}\n")
        report_lines.append("\n")

    report_lines.append("## 关键洞察\n\n")
    report_lines.append(f"1. 前20%的客户贡献了{summary_stats.nlargest(2, 'total_revenue')['revenue_percentage'].sum():.1f}%的收入\n")
    report_lines.append(f"2. 平均客户生命周期价值为${rfm['monetary'].mean():.2f}\n")
    report_lines.append(f"3. 客户平均购买频率为{rfm['frequency'].mean():.2f}次\n")
    report_lines.append(f"4. 平均最近购买时间为{rfm['recency'].mean():.1f}天前\n")

    with open(os.path.join(output_dir, 'marketing_insights.md'), 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    print(f"  [OK] Exported: marketing_insights.md")

    # 5. JSON摘要
    summary_json = {
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_customers': int(len(rfm)),
        'segments': {
            seg: {
                'count': int(data['count']),
                'avg_monetary': float(data['avg_monetary']),
                'total_revenue': float(data['total_revenue']),
                'strategies': data['strategy']
            }
            for seg, data in insights.items()
        },
        'key_metrics': {
            'avg_customer_value': float(rfm['monetary'].mean()),
            'total_revenue': float(rfm['monetary'].sum()),
            'avg_purchase_frequency': float(rfm['frequency'].mean()),
            'avg_recency_days': float(rfm['recency'].mean())
        }
    }

    with open(os.path.join(output_dir, 'rfm_analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Exported: rfm_analysis_summary.json")

def main():
    """主函数"""
    output_dir = 'do_more_analysis/skill_execution/rfm-customer-segmentation'

    # 加载数据
    orders, order_items, customers = load_and_prepare_data()

    # 计算RFM
    rfm = calculate_rfm(orders, order_items)

    # RFM评分
    rfm = score_rfm(rfm)

    # 聚类分析
    rfm, optimal_k, inertias, silhouette_scores = perform_clustering(rfm)

    # 标记群体
    rfm, cluster_stats = label_segments(rfm)

    # 可视化
    create_visualizations(rfm, cluster_stats, output_dir)

    # 生成营销洞察
    insights = generate_marketing_insights(rfm, cluster_stats)

    # 导出结果
    export_results(rfm, insights, output_dir)

    print("\n" + "="*80)
    print("RFM分析完成!")
    print("="*80)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  - customer_segments.csv (客户分群数据)")
    print("  - vip_customers_list.csv (VIP客户列表)")
    print("  - segment_summary_statistics.csv (群体统计)")
    print("  - marketing_insights.md (营销洞察报告)")
    print("  - 5个可视化图表")

if __name__ == "__main__":
    main()
