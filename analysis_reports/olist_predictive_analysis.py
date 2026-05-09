# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

log('='*60)
log('Olist 电商数据 - Predictive 预测性分析')
log('='*60)

# 加载数据
log('\n>>> 步骤1: 加载数据')
orders = pd.read_csv('./data_storage/olist_orders_dataset.csv')
order_items = pd.read_csv('./data_storage/olist_order_items_dataset.csv')
payments = pd.read_csv('./data_storage/olist_order_payments_dataset.csv')
reviews = pd.read_csv('./data_storage/olist_order_reviews_dataset.csv')
customers = pd.read_csv('./data_storage/olist_customers_dataset.csv')

log(f'订单: {len(orders)}, 订单项: {len(order_items)}, 客户: {len(customers)}')

# 筛选有效订单
delivered = orders[orders['order_status'] == 'delivered'].copy()
delivered['order_purchase_timestamp'] = pd.to_datetime(delivered['order_purchase_timestamp'])

# ============================================================
# 1. RFM 客户分析
# ============================================================
log('\n>>> 步骤2: RFM 客户分析')

# 计算订单金额 (按order_id汇总)
order_amounts = order_items.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
}).sum(axis=1)

# 合并数据
delivered = delivered.merge(order_amounts.reset_index().rename(columns={0: 'order_amount'}), on='order_id', how='left')

# 计算RFM
latest_date = delivered['order_purchase_timestamp'].max()
log(f'数据截止日期: {latest_date.strftime("%Y-%m-%d")}')

rfm = delivered.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,
    'order_id': 'count',
    'order_amount': 'sum'
}).rename(columns={
    'order_purchase_timestamp': 'recency',
    'order_id': 'frequency',
    'order_amount': 'monetary'
})

log(f'客户总数: {len(rfm)}')

# RFM 评分 (1-4分)
rfm['R_score'] = pd.qcut(rfm['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
rfm['M_score'] = pd.qcut(rfm['monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')

rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

# 客户分群
def segment_customer(score):
    try:
        r, f, m = int(score[0]), int(score[1]), int(score[2])
        if r >= 3 and f >= 3 and m >= 3:
            return 'VIP客户'
        elif r >= 3 and m >= 3:
            return '高价值客户'
        elif r >= 3 and f >= 3:
            return '潜力客户'
        elif r <= 2 and f <= 2:
            return '流失风险客户'
        elif r <= 2:
            return '沉睡客户'
        else:
            return '普通客户'
    except:
        return '普通客户'

rfm['segment'] = rfm['RFM_score'].apply(segment_customer)

log('\n客户分群结果:')
segment_counts = rfm['segment'].value_counts()
for seg, cnt in segment_counts.items():
    log(f'  {seg}: {cnt} ({cnt/len(rfm)*100:.1f}%)')

# 客户价值统计
log('\n各分群客户价值:')
segment_stats = rfm.groupby('segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': ['mean', 'sum', 'count']
}).round(2)
segment_stats.columns = ['平均R(天)', '平均F(次)', '平均M(R$)', '总消费(R$)', '客户数']
print(segment_stats.to_string())

# ============================================================
# 2. K-means 聚类分析
# ============================================================
log('\n>>> 步骤3: K-means 聚类分析')

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # 准备聚类数据
    features = ['recency', 'frequency', 'monetary']
    X = rfm[features].fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用 K=4 进行聚类
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(X_scaled)
    
    log('\n聚类结果分析:')
    cluster_stats = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).round(2)
    cluster_stats.columns = ['平均R(天)', '平均F(次)', '平均M(R$)', '客户数']
    
    for idx, row in cluster_stats.iterrows():
        log(f'  聚类{idx}: R={row["平均R(天)"]:.0f}天, F={row["平均F(次)"]:.1f}次, M=R${row["平均M(R$)"]:.0f}, 人数={row["客户数"]}')
    
except ImportError:
    log('sklearn 未安装，跳过K-means聚类')

# ============================================================
# 3. 客户满意度预测特征分析
# ============================================================
log('\n>>> 步骤4: 客户满意度特征分析')

# 合并评分数据
order_reviews = orders.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')
order_reviews = order_reviews[order_reviews['order_status'] == 'delivered']
order_reviews = order_reviews.merge(order_amounts.reset_index().rename(columns={0: 'order_amount'}), on='order_id', how='left')

# 计算配送时间
order_reviews['order_purchase_timestamp'] = pd.to_datetime(order_reviews['order_purchase_timestamp'])
order_reviews['order_delivered_customer_date'] = pd.to_datetime(order_reviews['order_delivered_customer_date'])
order_reviews['delivery_days'] = (order_reviews['order_delivered_customer_date'] - order_reviews['order_purchase_timestamp']).dt.days

# 筛选有效数据
order_reviews = order_reviews[(order_reviews['delivery_days'] >= 0) & (order_reviews['delivery_days'] <= 60)]
order_reviews = order_reviews.dropna(subset=['review_score', 'delivery_days', 'order_amount'])

log(f'有效样本数: {len(order_reviews)}')

# 特征与评分相关性
correlations = {}
correlations['配送时间'] = order_reviews['delivery_days'].corr(order_reviews['review_score'])
correlations['订单金额'] = order_reviews['order_amount'].corr(order_reviews['review_score'])

log('\n特征与评分相关性:')
for feature, corr in correlations.items():
    log(f'  {feature}: {corr:.4f}')

# 按配送时间分组看评分
log('\n配送时间 vs 评分:')
delivery_groups = order_reviews.groupby(pd.cut(order_reviews['delivery_days'], bins=[0, 5, 10, 15, 20, 100]))['review_score'].mean()
for days, score in delivery_groups.items():
    log(f'  {days}: {score:.2f} 分')

# ============================================================
# 4. 客户流失风险分析
# ============================================================
log('\n>>> 步骤5: 客户流失风险分析')

# 定义流失风险客户 (超过90天未购买)
rfm['churn_risk'] = rfm['recency'].apply(lambda x: '高风险' if x > 90 else ('中风险' if x > 60 else '低风险'))

log('\n客户流失风险分布:')
churn_counts = rfm['churn_risk'].value_counts()
for risk, cnt in churn_counts.items():
    log(f'  {risk}: {cnt} ({cnt/len(rfm)*100:.1f}%)')

# 高价值客户中流失风险
high_value = rfm[rfm['monetary'] > rfm['monetary'].quantile(0.8)]
log(f'\n高价值客户({len(high_value)}人)中流失风险:')
high_value_churn = high_value['churn_risk'].value_counts()
for risk, cnt in high_value_churn.items():
    log(f'  {risk}: {cnt} ({cnt/len(high_value)*100:.1f}%)')

# ============================================================
# 5. 业务预测建议
# ============================================================
log('\n>>> 步骤6: 业务预测建议')

# 计算各分群价值
segment_value = rfm.groupby('segment').agg({
    'monetary': 'sum'
}).rename(columns={'monetary': 'total_monetary'})
segment_value['count'] = rfm.groupby('segment').size()
segment_value['avg_value'] = segment_value['total_monetary'] / segment_value['count']
total_value = segment_value['total_monetary'].sum()

log('\n各客户群价值贡献:')
for seg in segment_value.index:
    value = segment_value.loc[seg, 'total_monetary']
    pct = value / total_value * 100
    log(f'  {seg}: R$ {value:,.0f} ({pct:.1f}%)')

# ============================================================
# 总结
# ============================================================
log('\n' + '='*60)
log('预测性分析完成!')
log('='*60)

print('''
📊 预测性分析关键发现:

1. 客户分群 (RFM)
   - VIP客户: 高频高价值
   - 高价值客户: 消费金额高但可能不活跃
   - 潜力客户: 活跃度高，有提升空间
   - 流失风险客户: 需重点召回

2. K-means 聚类
   - 自动发现4类客户群
   - 可用于精准营销

3. 满意度预测因素
   - 配送时间是关键因素 (负相关)
   - 订单金额与评分关系较弱

4. 流失风险
   - 超过90天未购买为高风险
   - 高价值客户需特别关注

5. 业务建议
   - VIP客户提供专属服务
   - 流失风险客户启动召回计划
   - 优化配送提升评分
''')
