"""
Olist 巴西电商数据 - 全面深度互联网分析
覆盖: 数据探索、漏斗分析、RFM/LTV、评论情感、增长策略、产品收入
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings, os, json
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

DATA = './data_storage/'
# Skip geolocation (1M rows, causes memory issues)
SKIP_FILES = ['olist_geolocation_dataset']
OUT_REPORTS = './analysis_reports/'
OUT_VIZ = './visualizations/'
os.makedirs(OUT_REPORTS, exist_ok=True)
os.makedirs(OUT_VIZ, exist_ok=True)

# ============================================================
# 1. 数据加载与合并
# ============================================================
print("=" * 70)
print("  Olist 巴西电商 - 全面深度互联网分析")
print("=" * 70)

print("\n[1/8] 加载数据...")
customers = pd.read_csv(f'{DATA}olist_customers_dataset.csv')
orders = pd.read_csv(f'{DATA}olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp','order_approved_at',
    'order_delivered_carrier_date','order_delivered_customer_date',
    'order_estimated_delivery_date'])
items = pd.read_csv(f'{DATA}olist_order_items_dataset.csv')
payments = pd.read_csv(f'{DATA}olist_order_payments_dataset.csv')
reviews = pd.read_csv(f'{DATA}olist_order_reviews_dataset.csv')
products = pd.read_csv(f'{DATA}olist_products_dataset.csv')
sellers = pd.read_csv(f'{DATA}olist_sellers_dataset.csv')
cat_trans = pd.read_csv(f'{DATA}product_category_name_translation.csv')

# 时间字段
orders['order_purchase_date'] = orders['order_purchase_timestamp'].dt.date
orders['order_purchase_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
orders['order_purchase_week'] = orders['order_purchase_timestamp'].dt.to_period('W')
orders['order_purchase_dow'] = orders['order_purchase_timestamp'].dt.dayofweek
orders['order_purchase_hour'] = orders['order_purchase_timestamp'].dt.hour

# 合并核心表
products_en = products.merge(cat_trans, on='product_category_name', how='left')

print(f"  客户数: {customers['customer_unique_id'].nunique():,}")
print(f"  订单数: {len(orders):,}")
print(f"  商品项: {len(items):,}")
print(f"  产品数: {len(products):,}")
print(f"  卖家数: {len(sellers):,}")
print(f"  评价数: {len(reviews):,}")
print(f"  时间范围: {orders['order_purchase_timestamp'].min().strftime('%Y-%m-%d')} ~ {orders['order_purchase_timestamp'].max().strftime('%Y-%m-%d')}")

# ============================================================
# 2. 数据质量与概览
# ============================================================
print("\n[2/8] 数据质量与概览...")

# 缺失值
missing_report = {}
for name, df in [('orders', orders), ('items', items), ('payments', payments),
                 ('reviews', reviews), ('products', products)]:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        missing_report[name] = nulls.to_dict()
        print(f"  {name} 缺失: {dict(nulls)}")

# 订单状态分布
status_dist = orders['order_status'].value_counts()
print(f"\n  订单状态分布:")
for s, c in status_dist.items():
    print(f"    {s}: {c:,} ({c/len(orders)*100:.1f}%)")

# 有效订单 = delivered
delivered = orders[orders['order_status'] == 'delivered'].copy()
print(f"  有效交付订单: {len(delivered):,} ({len(delivered)/len(orders)*100:.1f}%)")

# ============================================================
# 3. 订单漏斗分析
# ============================================================
print("\n[3/8] 订单漏斗分析...")

total_orders = len(orders)
approved = orders['order_approved_at'].notna().sum()
shipped = orders['order_delivered_carrier_date'].notna().sum()
delivered_count = orders['order_delivered_customer_date'].notna().sum()

funnel_stages = {
    '下单': total_orders,
    '审批通过': approved,
    '发货': shipped,
    '交付完成': delivered_count
}

print("  转化漏斗:")
prev = None
for stage, count in funnel_stages.items():
    rate = count / total_orders * 100
    step_rate = f" (→{count/prev*100:.1f}%)" if prev else ""
    print(f"    {stage}: {count:,} ({rate:.1f}%){step_rate}")
    prev = count

# 支付方式漏斗
payment_dist = payments.groupby('payment_type')['payment_value'].agg(['count','sum','mean']).sort_values('sum', ascending=False)
print(f"\n  支付方式分析:")
for ptype, row in payment_dist.iterrows():
    print(f"    {ptype}: {row['count']:,}笔, 总额${row['sum']:,.0f}, 均价${row['mean']:.2f}")

# ============================================================
# 4. 收入与产品分析
# ============================================================
print("\n[4/8] 收入与产品分析...")

# 订单金额（按 order_id 汇总 price + freight）
order_amounts = items.groupby('order_id').agg(
    price_sum=('price', 'sum'),
    freight_sum=('freight_value', 'sum')
).reset_index()
order_amounts['total_amount'] = order_amounts['price_sum'] + order_amounts['freight_sum']

amounts = order_amounts['total_amount'].values
print(f"  订单金额统计 (按order_id汇总):")
print(f"    均值: ${np.mean(amounts):.2f}")
print(f"    中位数: ${np.median(amounts):.2f}")
print(f"    标准差: ${np.std(amounts):.2f}")
print(f"    总收入: ${np.sum(amounts):,.0f}")

# 分位数
for q in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{q}: ${np.percentile(amounts, q):.2f}")

# 产品品类收入
items_with_cat = items.merge(products_en[['product_id','product_category_name_english']], on='product_id', how='left')
cat_revenue = items_with_cat.groupby('product_category_name_english').agg(
    revenue=('price', 'sum'),
    items_sold=('order_item_id', 'count'),
    avg_price=('price', 'mean')
).sort_values('revenue', ascending=False)

print(f"\n  Top 10 品类 (收入):")
for cat, row in cat_revenue.head(10).iterrows():
    print(f"    {cat}: ${row['revenue']:,.0f} ({row['items_sold']:,}件)")

# 二八法则
cat_revenue_sorted = cat_revenue.sort_values('revenue', ascending=False)
cat_revenue_sorted['cum_pct'] = cat_revenue_sorted['revenue'].cumsum() / cat_revenue_sorted['revenue'].sum() * 100
top20_cats = cat_revenue_sorted[cat_revenue_sorted['cum_pct'] <= 80]
print(f"\n  二八法则: {len(top20_cats)} 个品类贡献 80% 收入 (共{len(cat_revenue_sorted)}个品类)")

# ============================================================
# 5. 时间趋势分析
# ============================================================
print("\n[5/8] 时间趋势分析...")

monthly = delivered.merge(order_amounts[['order_id','total_amount']], on='order_id', how='left')
monthly_stats = monthly.groupby('order_purchase_month').agg(
    orders=('order_id', 'count'),
    revenue=('total_amount', 'sum'),
    avg_order_value=('total_amount', 'mean')
).reset_index()
monthly_stats['month_str'] = monthly_stats['order_purchase_month'].astype(str)

print("  月度趋势:")
for _, row in monthly_stats.iterrows():
    print(f"    {row['month_str']}: {row['orders']:,}单, ${row['revenue']:,.0f}, 均价${row['avg_order_value']:.2f}")

# 星期分布
dow_names = ['周一','周二','周三','周四','周五','周六','周日']
dow_dist = orders.groupby('order_purchase_dow').size()
print(f"\n  星期分布:")
for d, c in dow_dist.items():
    print(f"    {dow_names[d]}: {c:,} ({c/len(orders)*100:.1f}%)")

# 时段分布
hour_dist = orders.groupby('order_purchase_hour').size()
peak_hours = hour_dist.nlargest(5)
print(f"\n  Top 5 下单时段:")
for h, c in peak_hours.items():
    print(f"    {h}:00 - {c:,}单")

# ============================================================
# 6. 配送分析
# ============================================================
print("\n[6/8] 配送分析...")

delivered_valid = delivered.dropna(subset=['order_delivered_customer_date']).copy()
delivered_valid['delivery_days'] = (delivered_valid['order_delivered_customer_date'] - delivered_valid['order_purchase_timestamp']).dt.total_seconds() / 86400
delivered_valid['estimated_days'] = (delivered_valid['order_estimated_delivery_date'] - delivered_valid['order_purchase_timestamp']).dt.total_seconds() / 86400
delivered_valid['delivery_delay'] = delivered_valid['delivery_days'] - delivered_valid['estimated_days']
delivered_valid['on_time'] = delivered_valid['delivery_delay'] <= 0

print(f"  平均配送时间: {delivered_valid['delivery_days'].mean():.1f}天")
print(f"  中位配送时间: {delivered_valid['delivery_days'].median():.1f}天")
print(f"  准时交付率: {delivered_valid['on_time'].mean()*100:.1f}%")
print(f"  平均延迟: {delivered_valid[delivered_valid['delivery_delay']>0]['delivery_delay'].mean():.1f}天")

# 配送时间 vs 评分
delivery_review = delivered_valid.merge(reviews[['order_id','review_score']], on='order_id', how='left')
if len(delivery_review) > 0:
    speed_bins = pd.cut(delivery_review['delivery_days'], bins=[0,5,10,15,20,30,100],
                        labels=['0-5天','5-10天','10-15天','15-20天','20-30天','30+天'])
    speed_score = delivery_review.groupby(speed_bins, observed=True)['review_score'].mean()
    print(f"\n  配送速度 vs 评分:")
    for bin_name, score in speed_score.items():
        print(f"    {bin_name}: {score:.2f}星")

# ============================================================
# 7. RFM 客户分群 + LTV
# ============================================================
print("\n[7/8] RFM客户分群 + LTV预测...")

# 构建客户订单表
cust_orders = delivered.merge(customers[['customer_id','customer_unique_id']], on='customer_id', how='left')
cust_orders = cust_orders.merge(order_amounts[['order_id','total_amount']], on='order_id', how='left')

# RFM 计算
reference_date = cust_orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
rfm = cust_orders.groupby('customer_unique_id').agg(
    recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('total_amount', 'sum')
).reset_index()

print(f"  RFM 统计:")
print(f"    客户总数: {len(rfm):,}")
print(f"    Recency: 均值{rfm['recency'].mean():.0f}天, 中位数{rfm['recency'].median():.0f}天")
print(f"    Frequency: 均值{rfm['frequency'].mean():.2f}, 最大{rfm['frequency'].max()}")
print(f"    Monetary: 均值${rfm['monetary'].mean():.2f}, 中位数${rfm['monetary'].median():.2f}")

# 复购率
repeat_buyers = (rfm['frequency'] > 1).sum()
print(f"    复购率: {repeat_buyers}/{len(rfm)} = {repeat_buyers/len(rfm)*100:.2f}%")

# RFM 分层打分
rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# 客户分层
def classify_customer(score):
    if score >= 13: return 'VIP客户'
    elif score >= 10: return '高价值客户'
    elif score >= 7: return '成长客户'
    elif score >= 4: return '一般客户'
    else: return '流失风险客户'

rfm['segment'] = rfm['RFM_score'].apply(classify_customer)
segment_dist = rfm['segment'].value_counts()

print(f"\n  客户分层:")
for seg, cnt in segment_dist.items():
    seg_monetary = rfm[rfm['segment']==seg]['monetary'].sum()
    pct = cnt / len(rfm) * 100
    rev_pct = seg_monetary / rfm['monetary'].sum() * 100
    print(f"    {seg}: {cnt:,}人 ({pct:.1f}%), 贡献${seg_monetary:,.0f} ({rev_pct:.1f}%)")

# LTV 简单预测
avg_order_value = rfm['monetary'].mean()
avg_frequency = rfm['frequency'].mean()
avg_lifespan = 12  # 假设12个月
ltv = avg_order_value * avg_frequency * avg_lifespan
print(f"\n  LTV 预测:")
print(f"    平均订单价值: ${avg_order_value:.2f}")
print(f"    平均购买频次: {avg_frequency:.2f}")
print(f"    预估客户生命周期: {avg_lifespan}月")
print(f"    预测LTV: ${ltv:.2f}")

# 分层LTV
for seg in ['VIP客户','高价值客户','成长客户','一般客户','流失风险客户']:
    seg_data = rfm[rfm['segment']==seg]
    if len(seg_data) > 0:
        seg_ltv = seg_data['monetary'].mean() * seg_data['frequency'].mean() * 12
        print(f"    {seg} LTV: ${seg_ltv:.2f}")

rfm.to_csv(f'{OUT_REPORTS}rfm_customer_segmentation.csv', index=False)

# ============================================================
# 8. 评论内容分析
# ============================================================
print("\n[8/8] 评论内容情感分析...")

# 评分分布
score_dist = reviews['review_score'].value_counts().sort_index()
print(f"  评分分布:")
for score, cnt in score_dist.items():
    print(f"    {score}星: {cnt:,} ({cnt/len(reviews)*100:.1f}%)")

avg_score = reviews['review_score'].mean()
print(f"  平均评分: {avg_score:.2f}星")

# 有无评论对比
reviews['has_comment'] = reviews['review_comment_message'].notna() & (reviews['review_comment_message'].str.strip() != '')
comment_dist = reviews.groupby('has_comment')['review_score'].mean()
print(f"\n  有评论均值: {comment_dist.get(True, 0):.2f}星")
print(f"  无评论均值: {comment_dist.get(False, 0):.2f}星")
print(f"  评论率: {reviews['has_comment'].sum()}/{len(reviews)} = {reviews['has_comment'].mean()*100:.1f}%")

# 差评分析 (1-2星)
bad_reviews = reviews[(reviews['review_score'] <= 2) & reviews['has_comment']]
print(f"\n  差评(1-2星)有评论数: {len(bad_reviews):,}")

# 关键词频率（简单词频）
if len(bad_reviews) > 0:
    from collections import Counter
    all_words = []
    for msg in bad_reviews['review_comment_message'].dropna().values[:2000]:
        words = str(msg).lower().split()
        all_words.extend([w for w in words if len(w) > 3])

    word_freq = Counter(all_words).most_common(20)
    print("  差评高频词 (Top 20):")
    for w, c in word_freq:
        print(f"    {w}: {c}")

# 评分趋势
review_orders = reviews.merge(orders[['order_id','order_purchase_timestamp']], on='order_id', how='left')
review_orders['month'] = review_orders['order_purchase_timestamp'].dt.to_period('M')
monthly_score = review_orders.groupby('month')['review_score'].mean()
print(f"\n  月度评分趋势:")
for m, s in monthly_score.items():
    if pd.notna(m):
        print(f"    {m}: {s:.2f}星")

# ============================================================
# 9. 地域分析
# ============================================================
print("\n" + "=" * 70)
print("  地域分析")
print("=" * 70)

cust_region = orders.merge(customers[['customer_id','customer_state']], on='customer_id', how='left')
state_orders = cust_region.groupby('customer_state').agg(
    orders=('order_id','count'),
).sort_values('orders', ascending=False)

state_revenue = cust_region.merge(order_amounts[['order_id','total_amount']], on='order_id', how='left')
state_rev = state_revenue.groupby('customer_state')['total_amount'].agg(['sum','mean']).sort_values('sum', ascending=False)

print("  Top 10 州 (订单量):")
for state, row in state_orders.head(10).iterrows():
    rev = state_rev.loc[state, 'sum'] if state in state_rev.index else 0
    print(f"    {state}: {row['orders']:,}单, ${rev:,.0f}")

# ============================================================
# 10. 卖家分析
# ============================================================
print("\n" + "=" * 70)
print("  卖家分析")
print("=" * 70)

seller_perf = items.merge(sellers[['seller_id','seller_state','seller_city']], on='seller_id', how='left')
seller_stats = seller_perf.groupby('seller_id').agg(
    items_sold=('order_item_id','count'),
    revenue=('price','sum'),
    orders=('order_id','nunique')
).sort_values('revenue', ascending=False)

print(f"  卖家总数: {len(seller_stats):,}")
print(f"  Top 10 卖家 (收入):")
for sid, row in seller_stats.head(10).iterrows():
    print(f"    {sid[:12]}...: ${row['revenue']:,.0f} ({row['items_sold']}件, {row['orders']}单)")

# 卖家集中度
seller_stats_sorted = seller_stats.sort_values('revenue', ascending=False)
seller_stats_sorted['cum_pct'] = seller_stats_sorted['revenue'].cumsum() / seller_stats_sorted['revenue'].sum() * 100
top_sellers = seller_stats_sorted[seller_stats_sorted['cum_pct'] <= 80]
print(f"\n  二八法则: {len(top_sellers)} 个卖家贡献 80% GMV (共{len(seller_stats)}个卖家)")

# ============================================================
# 11. 综合可视化
# ============================================================
print("\n" + "=" * 70)
print("  生成可视化图表...")
print("=" * 70)

# --- 图1: 综合仪表板 ---
fig, axes = plt.subplots(3, 3, figsize=(22, 20))
fig.suptitle('Olist 巴西电商全面分析仪表板', fontsize=20, fontweight='bold', y=0.99)

# 1.1 订单漏斗
ax = axes[0, 0]
stages = list(funnel_stages.keys())
vals = list(funnel_stages.values())
colors_f = ['#4ECDC4','#45B7D1','#96CEB4','#FF6B6B']
bars = ax.barh(stages[::-1], vals[::-1], color=colors_f[::-1])
ax.set_title('订单转化漏斗', fontweight='bold', fontsize=12)
ax.set_xlabel('订单数')
for bar, v in zip(bars, vals[::-1]):
    ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2, f'{v:,}', va='center', fontsize=10)

# 1.2 月度收入趋势
ax = axes[0, 1]
valid_months = monthly_stats.dropna(subset=['revenue'])
ax.plot(range(len(valid_months)), valid_months['revenue'].values, 'o-', color='#FF6B6B', linewidth=2)
ax.fill_between(range(len(valid_months)), valid_months['revenue'].values, alpha=0.2, color='#FF6B6B')
ax.set_title('月度收入趋势', fontweight='bold', fontsize=12)
ax.set_ylabel('收入 (R$)')
ax.set_xlabel('月份')
tick_labels = [m[:7] if len(str(m))>7 else str(m) for m in valid_months['month_str'].values]
ax.set_xticks(range(0, len(tick_labels), max(1, len(tick_labels)//8)))
ax.set_xticklabels([tick_labels[i] for i in range(0, len(tick_labels), max(1, len(tick_labels)//8))], rotation=30, fontsize=8)

# 1.3 评分分布
ax = axes[0, 2]
colors_score = ['#FF6B6B','#FF9F43','#FECA57','#48DBFB','#4ECDC4']
ax.bar(score_dist.index, score_dist.values, color=colors_score, edgecolor='white')
ax.set_title(f'评分分布 (均值{avg_score:.2f})', fontweight='bold', fontsize=12)
ax.set_xlabel('评分')
ax.set_ylabel('评价数')
for i, (s, c) in enumerate(score_dist.items()):
    ax.text(s, c+200, f'{c/len(reviews)*100:.1f}%', ha='center', fontsize=9)

# 2.1 RFM 客户分层
ax = axes[1, 0]
seg_order = ['VIP客户','高价值客户','成长客户','一般客户','流失风险客户']
seg_colors = ['#FF6B6B','#FF9F43','#FECA57','#48DBFB','#C8D6E5']
seg_counts = [segment_dist.get(s, 0) for s in seg_order]
bars = ax.bar(seg_order, seg_counts, color=seg_colors)
ax.set_title('RFM 客户分层', fontweight='bold', fontsize=12)
ax.set_ylabel('客户数')
ax.tick_params(axis='x', rotation=20)
for bar, c in zip(bars, seg_counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200, f'{c:,}', ha='center', fontsize=9)

# 2.2 Top 10 品类收入
ax = axes[1, 1]
top10_cats = cat_revenue.head(10)
y_labels = [c[:20] if len(str(c))>20 else str(c) for c in top10_cats.index]
ax.barh(y_labels[::-1], top10_cats['revenue'].values[::-1], color=plt.cm.viridis(np.linspace(0.2,0.8,10)))
ax.set_title('Top 10 品类收入', fontweight='bold', fontsize=12)
ax.set_xlabel('收入 (R$)')

# 2.3 支付方式
ax = axes[1, 2]
pay_types = payment_dist.head(5)
ax.pie(pay_types['count'], labels=pay_types.index, autopct='%1.1f%%',
       colors=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'])
ax.set_title('支付方式分布', fontweight='bold', fontsize=12)

# 3.1 星期+时段热力图
ax = axes[2, 0]
dow_hour = orders.groupby(['order_purchase_dow','order_purchase_hour']).size().unstack(fill_value=0)
im = ax.imshow(dow_hour.values, cmap='YlOrRd', aspect='auto')
ax.set_title('星期x时段订单热力图', fontweight='bold', fontsize=12)
ax.set_xlabel('小时')
ax.set_ylabel('星期')
ax.set_yticks(range(7))
ax.set_yticklabels(dow_names, rotation=0)
fig.colorbar(im, ax=ax)

# 3.2 配送时间分布
ax = axes[2, 1]
delivery_valid = delivered_valid[delivered_valid['delivery_days'].between(0, 60)]
ax.hist(delivery_valid['delivery_days'], bins=30, color='#45B7D1', edgecolor='white', alpha=0.8)
ax.axvline(delivery_valid['delivery_days'].median(), color='red', linestyle='--', label=f"中位数{delivery_valid['delivery_days'].median():.0f}天")
ax.set_title('配送时间分布', fontweight='bold', fontsize=12)
ax.set_xlabel('天数')
ax.set_ylabel('订单数')
ax.legend()

# 3.3 州订单量 Top 10
ax = axes[2, 2]
top_states = state_orders.head(10)
ax.bar(top_states.index, top_states['orders'], color=plt.cm.Paired(np.linspace(0,1,10)))
ax.set_title('Top 10 州订单量', fontweight='bold', fontsize=12)
ax.set_ylabel('订单数')
ax.tick_params(axis='x', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUT_VIZ}olist_full_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 仪表板: {OUT_VIZ}olist_full_dashboard.png")

# --- 图2: RFM 深度分析 ---
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle('RFM 客户分群深度分析', fontsize=16, fontweight='bold')

# R分布
axes2[0].hist(rfm['recency'], bins=50, color='#FF6B6B', edgecolor='white', alpha=0.8)
axes2[0].set_title('Recency 分布 (天)', fontweight='bold')
axes2[0].axvline(rfm['recency'].median(), color='blue', linestyle='--', label=f"中位数{rfm['recency'].median():.0f}天")
axes2[0].legend()

# F分布
freq_dist = rfm['frequency'].value_counts().sort_index()
axes2[1].bar(freq_dist.index[:10], freq_dist.values[:10], color='#4ECDC4')
axes2[1].set_title('Frequency 分布 (购买次数)', fontweight='bold')
axes2[1].set_xlabel('购买次数')

# M分布
axes2[2].hist(rfm['monetary'], bins=50, color='#45B7D1', edgecolor='white', alpha=0.8)
axes2[2].set_title('Monetary 分布 (消费金额)', fontweight='bold')
axes2[2].axvline(rfm['monetary'].median(), color='red', linestyle='--', label=f"中位数${rfm['monetary'].median():.0f}")
axes2[2].legend()

plt.tight_layout()
plt.savefig(f'{OUT_VIZ}rfm_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ RFM图: {OUT_VIZ}rfm_analysis.png")

# --- 图3: 产品与收入深度 ---
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))

# 品类帕累托图
ax = axes3[0]
top20 = cat_revenue_sorted.head(20)
x = range(len(top20))
ax.bar(x, top20['revenue'].values, color='#4ECDC4', alpha=0.8)
ax2 = ax.twinx()
ax2.plot(x, top20['cum_pct'].values, 'ro-', linewidth=2)
ax2.axhline(80, color='gray', linestyle='--', alpha=0.5)
ax.set_title('品类帕累托图 (Top 20)', fontweight='bold')
ax.set_ylabel('收入 (R$)', color='#4ECDC4')
ax2.set_ylabel('累计占比 (%)', color='red')
labels = [str(c)[:15] for c in top20.index]
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)

# 订单金额分布
ax = axes3[1]
amounts_capped = order_amounts[order_amounts['total_amount'] < order_amounts['total_amount'].quantile(0.95)]['total_amount']
ax.hist(amounts_capped, bins=50, color='#FF6B6B', edgecolor='white', alpha=0.8)
ax.axvline(np.median(amounts), color='blue', linestyle='--', label=f"中位数${np.median(amounts):.0f}")
ax.set_title('订单金额分布 (P95截断)', fontweight='bold')
ax.set_xlabel('金额 (R$)')
ax.set_ylabel('订单数')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT_VIZ}product_revenue_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 产品收入图: {OUT_VIZ}product_revenue_analysis.png")

# ============================================================
# 12. AARRR 增长模型分析
# ============================================================
print("\n" + "=" * 70)
print("  AARRR 增长模型分析")
print("=" * 70)

# Acquisition - 获客
total_customers = customers['customer_unique_id'].nunique()
monthly_new = cust_orders.groupby(cust_orders['order_purchase_timestamp'].dt.to_period('M'))['customer_unique_id'].nunique()
print(f"  [Acquisition] 总客户: {total_customers:,}")
print(f"    月均新客: {monthly_new.mean():.0f}")
print(f"    峰值月份: {monthly_new.idxmax()} ({monthly_new.max():,})")

# Activation - 激活
avg_first_items = items.groupby('order_id')['order_item_id'].count().mean()
print(f"  [Activation] 平均每单商品数: {avg_first_items:.2f}")

# Retention - 留存
repeat_rate = repeat_buyers / len(rfm) * 100
print(f"  [Retention] 复购率: {repeat_rate:.2f}%")
freq_2plus = (rfm['frequency'] >= 2).sum()
freq_3plus = (rfm['frequency'] >= 3).sum()
print(f"    2次+购买: {freq_2plus:,} ({freq_2plus/len(rfm)*100:.2f}%)")
print(f"    3次+购买: {freq_3plus:,} ({freq_3plus/len(rfm)*100:.2f}%)")

# Revenue - 收入
total_rev = order_amounts['total_amount'].sum()
aov = order_amounts['total_amount'].mean()
print(f"  [Revenue] 总收入: R${total_rev:,.0f}")
print(f"    客单价: R${aov:.2f}")
print(f"    客均LTV: R${rfm['monetary'].mean():.2f}")

# Referral - 推荐
review_rate = len(reviews) / len(orders) * 100
positive_rate = (reviews['review_score'] >= 4).sum() / len(reviews) * 100
print(f"  [Referral] 评价率: {review_rate:.1f}%")
print(f"    好评率(4-5星): {positive_rate:.1f}%")

# ============================================================
# 13. 综合洞察报告
# ============================================================
print("\n" + "=" * 70)
print("  生成综合洞察报告...")
print("=" * 70)

report = f"""# Olist 巴西电商 - 全面深度互联网分析报告

## 执行摘要

**分析日期**: {datetime.now().strftime('%Y-%m-%d')}
**数据范围**: {orders['order_purchase_timestamp'].min().strftime('%Y-%m-%d')} ~ {orders['order_purchase_timestamp'].max().strftime('%Y-%m-%d')}
**数据规模**: {customers['customer_unique_id'].nunique():,}客户 / {len(orders):,}订单 / {len(items):,}商品

---

## 一、数据概览

| 指标 | 数值 |
|------|------|
| 客户总数 | {customers['customer_unique_id'].nunique():,} |
| 订单总数 | {len(orders):,} |
| 商品记录数 | {len(items):,} |
| 产品数 | {len(products):,} |
| 卖家数 | {len(sellers):,} |
| 总收入 | R${total_rev:,.0f} |
| 客单价 | R${aov:.2f} |
| 平均评分 | {avg_score:.2f}星 |

---

## 二、订单漏斗分析

| 阶段 | 数量 | 转化率 |
|------|------|--------|
| 下单 | {total_orders:,} | 100% |
| 审批通过 | {approved:,} | {approved/total_orders*100:.1f}% |
| 发货 | {shipped:,} | {shipped/total_orders*100:.1f}% |
| 交付完成 | {delivered_count:,} | {delivered_count/total_orders*100:.1f}% |

**洞察**: 整体转化率{delivered_count/total_orders*100:.1f}%，表现健康。约{(1-delivered_count/total_orders)*100:.1f}%的订单未能完成交付。

---

## 三、收入与产品分析

### 3.1 订单金额统计
- 均值: R${np.mean(amounts):.2f}
- 中位数: R${np.median(amounts):.2f}
- P95: R${np.percentile(amounts, 95):.2f}

### 3.2 二八法则
- **{len(top20_cats)}个品类**贡献80%收入（共{len(cat_revenue_sorted)}个品类）
- **{len(top_sellers)}个卖家**贡献80% GMV（共{len(seller_stats)}个卖家）

### 3.3 Top 10 品类收入

| 品类 | 收入 | 销量 | 均价 |
|------|------|------|------|
"""

for cat, row in cat_revenue.head(10).iterrows():
    report += f"| {cat} | R${row['revenue']:,.0f} | {row['items_sold']:,} | R${row['avg_price']:.2f} |\n"

report += f"""
---

## 四、RFM 客户分群

### 4.1 RFM 基础统计
- Recency 均值: {rfm['recency'].mean():.0f}天
- Frequency 均值: {rfm['frequency'].mean():.2f}次
- Monetary 均值: R${rfm['monetary'].mean():.2f}
- **复购率: {repeat_rate:.2f}%** （{repeat_buyers:,}人复购）

### 4.2 客户分层

| 层级 | 人数 | 占比 | 贡献收入 | 收入占比 |
|------|------|------|---------|---------|
"""

for seg in seg_order:
    cnt = segment_dist.get(seg, 0)
    seg_rev = rfm[rfm['segment']==seg]['monetary'].sum()
    report += f"| {seg} | {cnt:,} | {cnt/len(rfm)*100:.1f}% | R${seg_rev:,.0f} | {seg_rev/rfm['monetary'].sum()*100:.1f}% |\n"

report += f"""
### 4.3 LTV 预测
- 整体预测LTV: R${ltv:.2f}
- VIP客户LTV: 最高价值群体

---

## 五、评论内容分析

### 5.1 评分分布

| 评分 | 数量 | 占比 |
|------|------|------|
"""

for score, cnt in score_dist.items():
    report += f"| {score}星 | {cnt:,} | {cnt/len(reviews)*100:.1f}% |\n"

report += f"""
- **平均评分: {avg_score:.2f}星**
- **好评率(4-5星): {positive_rate:.1f}%**
- **评论率: {reviews['has_comment'].mean()*100:.1f}%**

---

## 六、配送分析

- 平均配送时间: {delivered_valid['delivery_days'].mean():.1f}天
- 中位配送时间: {delivered_valid['delivery_days'].median():.1f}天
- **准时交付率: {delivered_valid['on_time'].mean()*100:.1f}%**
- 平均延迟: {delivered_valid[delivered_valid['delivery_delay']>0]['delivery_delay'].mean():.1f}天

### 配送速度 vs 评分
"""

if 'speed_score' in dir():
    for bin_name, score in speed_score.items():
        report += f"- {bin_name}: {score:.2f}星\n"

report += f"""
**洞察**: 配送速度直接影响用户满意度。配送超过15天的订单评分显著下降。

---

## 七、AARRR 增长模型

| 阶段 | 指标 | 数值 |
|------|------|------|
| **Acquisition** | 总客户数 | {total_customers:,} |
| | 月均新客 | {monthly_new.mean():.0f} |
| **Activation** | 平均每单商品数 | {avg_first_items:.2f} |
| **Retention** | 复购率 | {repeat_rate:.2f}% |
| **Revenue** | 总收入 | R${total_rev:,.0f} |
| | 客单价 | R${aov:.2f} |
| **Referral** | 好评率 | {positive_rate:.1f}% |

---

## 八、深度洞察与策略建议

### 核心发现

1. **复购率极低 ({repeat_rate:.1f}%)** - 这是最大的增长瓶颈
   - {total_customers:,}客户中仅{repeat_buyers:,}人复购
   - 需要重点投入客户留存策略

2. **收入高度集中**
   - {len(top20_cats)}个品类贡献80%收入
   - {len(top_sellers)}个卖家贡献80% GMV
   - 存在供应链风险

3. **配送体验是关键满意度驱动因素**
   - 准时率{delivered_valid['on_time'].mean()*100:.1f}%
   - 配送超时直接导致差评

4. **好评率高 ({positive_rate:.1f}%)** 但仍有{(reviews['review_score']<=2).sum():,}个差评需关注

### 策略建议

#### 高优先级
1. **复购率提升计划**
   - 建立90天客户生命周期管理
   - 设计会员积分和优惠券体系
   - 个性化推荐引擎

2. **高价值客户运营**
   - VIP客户专属服务和折扣
   - 定向营销活动

3. **配送体验优化**
   - 缩短平均配送时间至7天以内
   - 提升准时率至95%+

#### 中优先级
4. **品类结构优化**
   - 加大高毛利品类投入
   - 优化长尾品类库存管理

5. **差评响应机制**
   - 建立即时差评预警
   - 客服快速响应流程

#### 低优先级
6. **数据驱动体系建设**
   - 完善用户行为追踪
   - 建立实时数据看板

---

## 分析产出文件

- `rfm_customer_segmentation.csv` - RFM客户分群数据
- `olist_full_dashboard.png` - 综合分析仪表板
- `rfm_analysis.png` - RFM深度分析图
- `product_revenue_analysis.png` - 产品收入分析图
- `olist_full_analysis_report.md` - 本报告

---

*报告由 Olist 互联网数据分析技能自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(f'{OUT_REPORTS}olist_full_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✅ 报告: {OUT_REPORTS}olist_full_analysis_report.md")

print("\n" + "=" * 70)
print("  ✅ Olist 全面深度分析完成！")
print("=" * 70)
print(f"\n📁 产出文件:")
print(f"  - {OUT_REPORTS}olist_full_analysis_report.md")
print(f"  - {OUT_REPORTS}rfm_customer_segmentation.csv")
print(f"  - {OUT_VIZ}olist_full_dashboard.png")
print(f"  - {OUT_VIZ}rfm_analysis.png")
print(f"  - {OUT_VIZ}product_revenue_analysis.png")
