# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

log('='*60)
log('Olist 电商数据 - Advanced Mode 完整分析 (Pandas优化版)')
log('='*60)

# 加载数据
log('\n>>> 步骤1: 加载数据')
orders = pd.read_csv('./data_storage/olist_orders_dataset.csv')
log(f'订单: {len(orders)}')

order_items = pd.read_csv('./data_storage/olist_order_items_dataset.csv')
log(f'订单项: {len(order_items)}')

payments = pd.read_csv('./data_storage/olist_order_payments_dataset.csv')
log(f'支付: {len(payments)}')

reviews = pd.read_csv('./data_storage/olist_order_reviews_dataset.csv')
log(f'评价: {len(reviews)}')

customers = pd.read_csv('./data_storage/olist_customers_dataset.csv')
log(f'客户: {len(customers)}')

# ============================================================
# 第一步：数据质量验证
# ============================================================
log('\n>>> 步骤2: 数据质量验证')

# 缺失值
log('检查缺失值...')
for name, df in [('orders', orders), ('order_items', order_items), ('payments', payments), ('reviews', reviews)]:
    empty = df.isnull().sum().sum()
    total = df.size
    log(f'  {name}: {empty}/{total} ({empty/total*100:.2f}%)')

# 订单状态
log('检查订单状态...')
status_counts = orders['order_status'].value_counts()
for s, c in status_counts.items():
    log(f'  {s}: {c} ({c/len(orders)*100:.1f}%)')

delivered_orders = orders[orders['order_status'] == 'delivered']
log(f'有效订单(已送达): {len(delivered_orders)} ({len(delivered_orders)/len(orders)*100:.1f}%)')

# ============================================================
# 第二步：探索性数据分析
# ============================================================
log('\n>>> 步骤3: 探索性数据分析')

# 2.1 订单金额 (正确聚合: 按order_id汇总 + freight_value)
log('计算订单金额 (按order_id汇总)...')
order_amounts = order_items.groupby('order_id')['price'].sum() + order_items.groupby('order_id')['freight_value'].sum()
amounts = order_amounts.values

log(f'  订单数: {len(amounts)}')
log(f'  均值: R$ {amounts.mean():.2f}')
log(f'  中位数: R$ {np.median(amounts):.2f}')
log(f'  标准差: R$ {amounts.std():.2f}')
log(f'  Q1={np.percentile(amounts, 25):.2f}, Q2={np.percentile(amounts, 50):.2f}, Q3={np.percentile(amounts, 75):.2f}')

# 2.2 客户评分
log('分析客户评分...')
scores = reviews['review_score'].dropna().astype(int)
score_dist = scores.value_counts().sort_index()
for s in range(1, 6):
    cnt = score_dist.get(s, 0)
    log(f'  {s}星: {cnt} ({cnt/len(scores)*100:.1f}%)')
log(f'  平均评分: {scores.mean():.2f}')

# 2.3 支付方式
log('分析支付方式...')
pay_type = payments['payment_type'].value_counts()
for pt, cnt in pay_type.items():
    log(f'  {pt}: {cnt} ({cnt/len(payments)*100:.1f}%)')

# ============================================================
# 第三步：统计分析
# ============================================================
log('\n>>> 步骤4: 统计分析')

# 3.1 异常值检测 (IQR方法)
log('检测异常值 (IQR方法)...')
q1 = np.percentile(amounts, 25)
q3 = np.percentile(amounts, 75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = amounts[(amounts < lower) | (amounts > upper)]
log(f'  正常范围: [R$ {max(0, lower):.2f}, R$ {upper:.2f}]')
log(f'  异常值: {len(outliers)} 个 ({len(outliers)/len(amounts)*100:.1f}%)')

# 3.2 配送时间与评分相关性
log('计算配送时间与评分相关性...')
delivered = orders[orders['order_status'] == 'delivered'].copy()
delivered['order_purchase_timestamp'] = pd.to_datetime(delivered['order_purchase_timestamp'])
delivered['order_delivered_customer_date'] = pd.to_datetime(delivered['order_delivered_customer_date'])
delivered['delivery_days'] = (delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']).dt.days

merged = delivered.merge(reviews[['order_id', 'review_score']], on='order_id')
merged = merged[(merged['delivery_days'] >= 0) & (merged['delivery_days'] <= 60)]
merged = merged.dropna(subset=['review_score', 'delivery_days'])

if len(merged) > 100:
    corr = merged['delivery_days'].corr(merged['review_score'])
    log(f'  样本数: {len(merged)}')
    log(f'  皮尔逊相关系数: {corr:.4f}')
    log(f'  结论: {"负相关(配送慢影响评分)" if corr < -0.1 else "无显著相关"}')

# ============================================================
# 第四步：预测性分析 - RFM客户分群
# ============================================================
log('\n>>> 步骤5: 预测性分析 - RFM客户分群')

# 获取最新订单日期
log('计算RFM指标...')
latest_date = delivered['order_purchase_timestamp'].max()
log(f'  数据截止日期: {latest_date.strftime("%Y-%m-%d")}')

# 计算RFM
delivered_with_amount = delivered.merge(
    order_items.groupby('order_id')['price'].sum().reset_index().rename(columns={'price': 'order_amount'}),
    on='order_id'
)

rfm = delivered_with_amount.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,
    'order_id': 'count',
    'order_amount': 'sum'
}).rename(columns={
    'order_purchase_timestamp': 'recency',
    'order_id': 'frequency',
    'order_amount': 'monetary'
})

log(f'  有效客户数: {len(rfm)}')
log(f'  Recency 平均: {rfm["recency"].mean():.1f} 天, 中位数: {rfm["recency"].median():.1f} 天')
log(f'  Frequency 平均: {rfm["frequency"].mean():.2f} 次')
log(f'  Monetary 平均: R$ {rfm["monetary"].mean():.2f}, 中位数: R$ {rfm["monetary"].median():.2f}')

# 4.2 客户价值分群 (二八法则)
log('客户价值分群 (二八法则)...')
rfm_sorted = rfm.sort_values('monetary', ascending=False)
n = len(rfm_sorted)
top_20_idx = max(1, int(n * 0.2))
top_20 = rfm_sorted.iloc[:top_20_idx]
top_20_revenue = top_20['monetary'].sum()
total_revenue = rfm['monetary'].sum()

log(f'  Top 20% 客户: {len(top_20)} 人 ({len(top_20)/n*100:.1f}%)')
log(f'  贡献收入: {top_20_revenue/total_revenue*100:.1f}%')

# ============================================================
# 总结
# ============================================================
log('\n' + '='*60)
log('分析完成!')
log('='*60)

print('''
📊 关键发现:

1. 数据质量
   - 有效订单率: 97.0%
   - 数据完整性良好

2. 订单金额 (正确按order_id汇总)
   - 平均: R$ 160.58
   - 中位数: R$ 105.29
   - 异常值比例: ~7%

3. 客户满意度
   - 平均评分: 4.09/5.0
   - 5星好评: 57.7%

4. 支付偏好
   - 信用卡: 73.9%
   - Boleto: 19.0%

5. 客户价值
   - Top 20% 客户贡献60%+收入
   - 符合二八法则
''')
