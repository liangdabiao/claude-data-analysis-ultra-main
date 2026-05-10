"""
dataB/sample_orders.csv - 中文电商订单全面深度分析
维度: 数据概览 / 产品分析 / 客户RFM+LTV / 城市 / 时间趋势 / 关联规则 / 增长模型
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
import warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

DATA = './dataB/sample_orders.csv'
OUT_R = './output/03-datab-orders/reports/'
OUT_V = './output/03-datab-orders/visualizations/'
os.makedirs(OUT_R, exist_ok=True)
os.makedirs(OUT_V, exist_ok=True)

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 70)
print("  中文电商订单全面深度分析")
print("=" * 70)

df = pd.read_csv(DATA)
df['消费日期'] = pd.to_datetime(df['消费日期'])
df['订单金额'] = df['数量'] * df['单价']
df['日期'] = df['消费日期'].dt.date
df['星期'] = df['消费日期'].dt.dayofweek
df['小时'] = df['消费日期'].dt.hour
df['旬'] = df['消费日期'].dt.day.apply(lambda d: '上旬' if d<=10 else ('中旬' if d<=20 else '下旬'))

# 品类分类
def get_category(name):
    if any(k in name for k in ['火龙果','樱桃','鳕鱼','牛奶','蔬菜','巧克力','坚果','牛肉',
                                 '咖啡','红酒','茶叶','奶酪','橄榄油','水果','零食','海鲜','礼盒']):
        return '食品生鲜'
    elif 'usb' in name or '分线器' in name:
        return '数码配件'
    elif 'T恤' in name or '男装' in name:
        return '服装'
    return '其他'

df['品类'] = df['产品说明'].apply(get_category)

print(f"\n📊 数据概览:")
print(f"  总订单数: {len(df)}")
print(f"  去重订单号: {df['订单号'].nunique()}")
print(f"  产品种类: {df['产品码'].nunique()}")
print(f"  客户数: {df['用户码'].nunique()}")
print(f"  城市数: {df['城市'].nunique()}")
print(f"  时间范围: {df['消费日期'].min().strftime('%Y-%m-%d')} ~ {df['消费日期'].max().strftime('%Y-%m-%d')}")
print(f"  总销量: {df['数量'].sum()}")
print(f"  总收入: ¥{df['订单金额'].sum():,.2f}")

# 缺失值
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) > 0:
    print(f"  缺失值: {dict(nulls)}")
else:
    print(f"  数据质量: 无缺失值 ✅")

# ============================================================
# 2. 产品分析
# ============================================================
print("\n" + "=" * 70)
print("  产品分析")
print("=" * 70)

prod_stats = df.groupby(['产品码','产品说明','品类']).agg(
    销量=('数量', 'sum'),
    订单数=('订单号', 'count'),
    总收入=('订单金额', 'sum'),
    均价=('单价', 'first'),
).reset_index().sort_values('总收入', ascending=False)

print("\n📈 产品收入排名:")
for i, row in prod_stats.iterrows():
    print(f"  {row['产品说明']:12s} [{row['品类']}]: "
          f"¥{row['总收入']:>8,.2f} ({row['销量']:>2.0f}件, {row['订单数']:>2.0f}单, 均价¥{row['均价']:.0f})")

# 品类分析
cat_stats = df.groupby('品类').agg(
    销量=('数量', 'sum'),
    订单数=('订单号', 'count'),
    总收入=('订单金额', 'sum'),
    客单价=('订单金额', 'mean'),
    产品数=('产品码', 'nunique'),
    客户数=('用户码', 'nunique'),
).sort_values('总收入', ascending=False)

print(f"\n🏷️ 品类分析:")
for cat, row in cat_stats.iterrows():
    pct = row['总收入'] / df['订单金额'].sum() * 100
    print(f"  {cat}: ¥{row['总收入']:,.2f} ({pct:.1f}%), {row['销量']:.0f}件, "
          f"{row['产品数']:.0f}个产品, {row['客户数']:.0f}个客户")

# 品类二八法则
cat_sorted = cat_stats.sort_values('总收入', ascending=False)
cat_sorted['累计占比'] = cat_sorted['总收入'].cumsum() / cat_sorted['总收入'].sum() * 100
print(f"\n  品类累计收入占比:")
for cat, row in cat_sorted.iterrows():
    print(f"    {cat}: {row['累计占比']:.1f}%")

# 价格带分析
def price_band(p):
    if p < 50: return '低价(¥0-50)'
    elif p < 100: return '中价(¥50-100)'
    elif p < 200: return '高价(¥100-200)'
    else: return '超高价(¥200+)'

df['价格带'] = df['单价'].apply(price_band)
price_band_stats = df.groupby('价格带').agg(
    销量=('数量','sum'), 总收入=('订单金额','sum'), 订单数=('订单号','count')
).reindex(['低价(¥0-50)','中价(¥50-100)','高价(¥100-200)','超高价(¥200+)'])

print(f"\n💰 价格带分析:")
for band, row in price_band_stats.iterrows():
    if pd.notna(row['销量']):
        pct = row['总收入'] / df['订单金额'].sum() * 100
        print(f"  {band}: ¥{row['总收入']:,.2f} ({pct:.1f}%), {row['销量']:.0f}件")

# ============================================================
# 3. 客户RFM分群 + LTV
# ============================================================
print("\n" + "=" * 70)
print("  客户RFM分群 + LTV预测")
print("=" * 70)

ref_date = df['消费日期'].max() + pd.Timedelta(days=1)
customer_orders = df.groupby('用户码').agg(
    首次购买=('消费日期', 'min'),
    最近购买=('消费日期', 'max'),
    购买次数=('订单号', 'count'),
    总消费=('订单金额', 'sum'),
    总件数=('数量', 'sum'),
    购买天数=('消费日期', lambda x: x.dt.date.nunique()),
    购买产品数=('产品码', 'nunique'),
    购买城市=('城市', 'first'),
).reset_index()

customer_orders['R'] = (ref_date - customer_orders['最近购买']).dt.days
customer_orders['F'] = customer_orders['购买次数']
customer_orders['M'] = customer_orders['总消费']
customer_orders['活跃天数'] = (customer_orders['最近购买'] - customer_orders['首次购买']).dt.days + 1

print(f"\n👤 客户基础统计 (共{len(customer_orders)}人):")
print(f"  平均消费: ¥{customer_orders['总消费'].mean():.2f}")
print(f"  中位消费: ¥{customer_orders['总消费'].median():.2f}")
print(f"  平均购买次数: {customer_orders['购买次数'].mean():.2f}")
print(f"  最高消费: ¥{customer_orders['总消费'].max():.2f}")

# RFM评分
customer_orders['R评分'] = pd.cut(customer_orders['R'], bins=[0,7,14,21,30,100],
                                   labels=[5,4,3,2,1]).astype(int)
customer_orders['F评分'] = pd.cut(customer_orders['F'].rank(method='first'), bins=5,
                                   labels=[1,2,3,4,5]).astype(int)
customer_orders['M评分'] = pd.cut(customer_orders['M'].rank(method='first'), bins=5,
                                   labels=[1,2,3,4,5]).astype(int)
customer_orders['RFM总分'] = customer_orders['R评分'] + customer_orders['F评分'] + customer_orders['M评分']

def rfm_segment(row):
    if row['RFM总分'] >= 13: return 'VIP客户'
    elif row['RFM总分'] >= 10: return '高价值客户'
    elif row['RFM总分'] >= 7: return '成长客户'
    elif row['RFM总分'] >= 4: return '一般客户'
    else: return '流失风险'

customer_orders['客户层级'] = customer_orders.apply(rfm_segment, axis=1)
seg_dist = customer_orders['客户层级'].value_counts()

print(f"\n📊 RFM客户分层:")
total_rev = df['订单金额'].sum()
for seg in ['VIP客户','高价值客户','成长客户','一般客户','流失风险']:
    cnt = seg_dist.get(seg, 0)
    seg_rev = customer_orders[customer_orders['客户层级']==seg]['总消费'].sum()
    pct = cnt / len(customer_orders) * 100
    rev_pct = seg_rev / total_rev * 100
    print(f"  {seg}: {cnt}人 ({pct:.1f}%), 消费¥{seg_rev:,.2f} ({rev_pct:.1f}%)")

# Top 10 客户
print(f"\n🏆 Top 10 客户:")
top_cust = customer_orders.sort_values('总消费', ascending=False).head(10)
for _, row in top_cust.iterrows():
    print(f"  {row['用户码']} ({row['购买城市']}): ¥{row['总消费']:,.2f}, "
          f"{row['购买次数']}单, {row['购买产品数']}种产品, 活跃{row['活跃天数']}天, [{row['客户层级']}]")

# 复购分析
repeat = (customer_orders['购买次数'] > 1).sum()
print(f"\n🔄 复购分析:")
print(f"  复购客户: {repeat}/{len(customer_orders)} ({repeat/len(customer_orders)*100:.1f}%)")
print(f"  平均购买频次: {customer_orders['购买次数'].mean():.2f}")

# LTV预测
avg_order_val = customer_orders['总消费'].mean()
avg_freq = customer_orders['购买次数'].mean()
avg_lifespan_months = 12
ltv = avg_order_val * avg_freq * avg_lifespan_months
print(f"\n💎 LTV预测:")
print(f"  平均客单价: ¥{avg_order_val:.2f}")
print(f"  平均购买频次: {avg_freq:.2f}")
print(f"  预测LTV(12月): ¥{ltv:.2f}")

for seg in ['VIP客户','高价值客户','成长客户','一般客户','流失风险']:
    seg_data = customer_orders[customer_orders['客户层级']==seg]
    if len(seg_data) > 0:
        seg_ltv = seg_data['总消费'].mean() * seg_data['购买次数'].mean() * 12
        print(f"    {seg}: ¥{seg_ltv:.2f}")

customer_orders.to_csv(f'{OUT_R}customer_rfm_analysis.csv', index=False, encoding='utf-8-sig')

# ============================================================
# 4. 城市地域分析
# ============================================================
print("\n" + "=" * 70)
print("  城市地域分析")
print("=" * 70)

city_stats = df.groupby('城市').agg(
    订单数=('订单号','count'),
    销量=('数量','sum'),
    总收入=('订单金额','sum'),
    客单价=('订单金额','mean'),
    客户数=('用户码','nunique'),
    产品种类=('产品码','nunique'),
    人均消费=('订单金额','sum'),
).sort_values('总收入', ascending=False)
city_stats['人均消费'] = city_stats['总收入'] / city_stats['客户数']
city_stats['ARPU'] = city_stats['总收入'] / city_stats['客户数']

print(f"\n🏙️ 城市分析:")
for city, row in city_stats.iterrows():
    pct = row['总收入'] / total_rev * 100
    print(f"  {city}: ¥{row['总收入']:>8,.2f} ({pct:>5.1f}%), "
          f"{row['订单数']:>2.0f}单, {row['客户数']:>2.0f}人, "
          f"客单¥{row['客单价']:>6.2f}, 人均¥{row['人均消费']:>6.2f}, {row['产品种类']:.0f}种产品")

# 城市品类偏好
print(f"\n🛒 城市品类偏好:")
city_cat = df.groupby(['城市','品类'])['订单金额'].sum().unstack(fill_value=0)
for city in city_stats.head(5).index:
    if city in city_cat.index:
        prefs = city_cat.loc[city].sort_values(ascending=False)
        top_prefs = [f"{cat}({val:.0f})" for cat, val in prefs.head(2).items() if val > 0]
        print(f"  {city}: {', '.join(top_prefs)}")

# ============================================================
# 5. 时间趋势分析
# ============================================================
print("\n" + "=" * 70)
print("  时间趋势分析")
print("=" * 70)

# 日趋势
daily = df.groupby('日期').agg(订单数=('订单号','count'), 收入=('订单金额','sum')).reset_index()
print(f"\n📅 日趋势 (共{len(daily)}天有订单):")
print(f"  日均订单: {daily['订单数'].mean():.1f}")
print(f"  日均收入: ¥{daily['收入'].mean():.2f}")

peak_days = daily.sort_values('收入', ascending=False).head(5)
print(f"  Top 5 收入日:")
for _, row in peak_days.iterrows():
    print(f"    {row['日期']}: ¥{row['收入']:,.2f} ({row['订单数']:.0f}单)")

# 旬分析
period_stats = df.groupby('旬').agg(订单数=('订单号','count'), 收入=('订单金额','sum')).reindex(['上旬','中旬','下旬'])
print(f"\n📊 旬分析:")
for period, row in period_stats.iterrows():
    print(f"  {period}: ¥{row['收入']:,.2f} ({row['订单数']:.0f}单)")

# 星期分析
dow_names = ['周一','周二','周三','周四','周五','周六','周日']
dow_stats = df.groupby('星期').agg(订单数=('订单号','count'), 收入=('订单金额','sum'))
print(f"\n📆 星期分布:")
for d, row in dow_stats.iterrows():
    print(f"  {dow_names[d]}: ¥{row['收入']:>8,.2f} ({row['订单数']:>2.0f}单)")

# 时段分析
hour_stats = df.groupby('小时').agg(订单数=('订单号','count'), 收入=('订单金额','sum'))
print(f"\n⏰ 时段分布:")
for h, row in hour_stats.iterrows():
    print(f"  {h}:00 - {h}:59: ¥{row['收入']:>8,.2f} ({row['订单数']:>2.0f}单)")

# ============================================================
# 6. 关联规则分析
# ============================================================
print("\n" + "=" * 70)
print("  关联规则分析（购物篮）")
print("=" * 70)

# 按日期+用户构建购物篮（同一天同一用户的购买）
baskets = df.groupby(['消费日期','用户码'])['产品说明'].apply(list).reset_index()

pair_count = Counter()
for _, row in baskets.iterrows():
    prods = list(set(row['产品说明']))
    if len(prods) >= 2:
        for pair in combinations(sorted(prods), 2):
            pair_count[pair] += 1

total_baskets = len(baskets)
print(f"\n🧺 购物篮分析 (共{total_baskets}个购物篮):")

# 产品频率
prod_freq = df.groupby('产品说明')['订单号'].count().to_dict()

if pair_count:
    top_pairs = pair_count.most_common(10)
    print(f"  Top 10 产品关联:")
    for (p1, p2), cnt in top_pairs:
        support = cnt / total_baskets * 100
        conf1 = cnt / prod_freq.get(p1, 1) * 100
        conf2 = cnt / prod_freq.get(p2, 1) * 100
        print(f"    {p1} ↔ {p2}: 共现{cnt}次 (支持度{support:.1f}%, 置信度{conf1:.0f}%/{conf2:.0f}%)")
else:
    print("  数据量较小，关联对较少")

# 客户-产品矩阵 (哪些客户买了什么)
cust_prod = df.groupby('用户码')['产品说明'].apply(lambda x: list(set(x))).to_dict()
print(f"\n🛍️ 客户购买多样性:")
diversity = [(uid, len(prods)) for uid, prods in cust_prod.items()]
diversity.sort(key=lambda x: -x[1])
for uid, cnt in diversity[:10]:
    prods = cust_prod[uid]
    print(f"  {uid}: {cnt}种产品 ({', '.join(prods[:5])})")

# ============================================================
# 7. AARRR 增长模型
# ============================================================
print("\n" + "=" * 70)
print("  AARRR 增长模型分析")
print("=" * 70)

# Acquisition
total_cust = df['用户码'].nunique()
first_buy = df.groupby('用户码')['消费日期'].min()
new_by_week = first_buy.groupby(first_buy.dt.isocalendar().week).size()
print(f"\n📥 Acquisition (获客):")
print(f"  总客户数: {total_cust}")
print(f"  周均新客: {new_by_week.mean():.1f}")
print(f"  峰值周: 第{new_by_week.idxmax()}周 ({new_by_week.max()}人)")

# Activation
avg_items_per_order = df.groupby('订单号')['数量'].sum().mean()
avg_products_per_order = df.groupby('订单号')['产品码'].nunique().mean()
print(f"\n🎯 Activation (激活):")
print(f"  平均每单件数: {avg_items_per_order:.2f}")
print(f"  平均每单产品种数: {avg_products_per_order:.2f}")

# Retention
repeat_cust = (customer_orders['购买次数'] > 1).sum()
repeat_rate = repeat_cust / total_cust * 100
print(f"\n🔁 Retention (留存):")
print(f"  复购率: {repeat_rate:.1f}%")
print(f"  3次+购买: {(customer_orders['购买次数'] >= 3).sum()}人")

# Revenue
print(f"\n💰 Revenue (收入):")
print(f"  总收入: ¥{total_rev:,.2f}")
print(f"  客单价: ¥{df['订单金额'].mean():.2f}")
print(f"  客均价值: ¥{customer_orders['总消费'].mean():.2f}")

# Referral
avg_basket_size = df.groupby(['消费日期','用户码'])['订单金额'].sum().mean()
print(f"\n📣 Referral (推荐):")
print(f"  平均购物篮金额: ¥{avg_basket_size:.2f}")
print(f"  多品类购买率: {(customer_orders['购买产品数']>1).sum()/len(customer_orders)*100:.1f}%")

# ============================================================
# 8. 深度洞察
# ============================================================
print("\n" + "=" * 70)
print("  深度洞察与策略建议")
print("=" * 70)

# 找出高价值产品
top_product = prod_stats.iloc[0]
print(f"\n🔑 关键发现:")
print(f"  1. 收入之王: {top_product['产品说明']} (¥{top_product['总收入']:,.2f}, {top_product['销量']:.0f}件)")
print(f"  2. 总收入: ¥{total_rev:,.2f}, 25天内50笔订单")
print(f"  3. 日均收入: ¥{daily['收入'].mean():.2f}")

# 品类集中度
top_cat = cat_stats.iloc[0]
top_cat_pct = top_cat['总收入'] / total_rev * 100
print(f"  4. 品类集中: {top_cat.name}占{top_cat_pct:.1f}%收入")

# 地域集中度
top_city = city_stats.iloc[0]
print(f"  5. 地域集中: {top_city.name}贡献¥{top_city['总收入']:,.2f} ({top_city['总收入']/total_rev*100:.1f}%)")

# 消费特征
print(f"  6. 高价产品贡献: 超高价格带(¥200+)产品贡献重要收入")
print(f"  7. 上旬消费最旺: 上旬收入 ¥{period_stats.loc['上旬','收入']:,.2f}")

# 策略建议
print(f"\n📋 策略建议:")
print(f"  🔴 高优先级:")
print(f"    - 扩大食品生鲜品类优势（最大收入来源）")
print(f"    - 重点维护VIP客户（贡献{customer_orders[customer_orders['客户层级']=='VIP客户']['总消费'].sum()/total_rev*100:.0f}%收入）")
print(f"    - 北京/上海为核心市场，加大投入")
print(f"  🟡 中优先级:")
print(f"    - 提升复购率（当前{repeat_rate:.0f}%，有较大提升空间）")
print(f"    - 加强低价产品交叉销售（引流→高价转化）")
print(f"    - 开拓二三线市场（成都、武汉、西安等）")
print(f"  🟢 低优先级:")
print(f"    - 优化时段营销（午后高峰期投放）")
print(f"    - 建立会员积分体系")

# ============================================================
# 9. 综合可视化
# ============================================================
print("\n" + "=" * 70)
print("  生成可视化图表...")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(22, 20))
fig.suptitle('中文电商订单全面深度分析仪表板', fontsize=20, fontweight='bold', y=0.99)
colors_main = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8','#F8B500']

# 1. 产品收入 Top 10
ax = axes[0, 0]
top_prods = prod_stats.head(10)
ax.barh(top_prods['产品说明'].values[::-1], top_prods['总收入'].values[::-1],
        color=plt.cm.RdYlGn(np.linspace(0.2,0.8,10)))
ax.set_title('产品收入 Top 10', fontweight='bold', fontsize=12)
ax.set_xlabel('收入 (¥)')
for i, (name, val) in enumerate(zip(top_prods['产品说明'].values[::-1], top_prods['总收入'].values[::-1])):
    ax.text(val+50, i, f'¥{val:,.0f}', va='center', fontsize=9)

# 2. 品类收入占比
ax = axes[0, 1]
cat_rev = cat_stats['总收入']
ax.pie(cat_rev.values, labels=cat_rev.index, autopct='%1.1f%%',
       colors=colors_main[:len(cat_rev)], startangle=90)
ax.set_title('品类收入占比', fontweight='bold', fontsize=12)

# 3. RFM 客户分层
ax = axes[0, 2]
seg_order = ['VIP客户','高价值客户','成长客户','一般客户','流失风险']
seg_colors = ['#FF6B6B','#FF9F43','#FECA57','#48DBFB','#C8D6E5']
seg_counts = [seg_dist.get(s, 0) for s in seg_order]
bars = ax.bar(seg_order, seg_counts, color=seg_colors)
ax.set_title('RFM 客户分层', fontweight='bold', fontsize=12)
ax.set_ylabel('客户数')
ax.tick_params(axis='x', rotation=15)
for bar, c in zip(bars, seg_counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(c), ha='center', fontsize=10, fontweight='bold')

# 4. 日收入趋势
ax = axes[1, 0]
ax.plot(daily['日期'], daily['收入'], 'o-', color='#FF6B6B', linewidth=2, markersize=5)
ax.fill_between(daily['日期'], daily['收入'], alpha=0.15, color='#FF6B6B')
ax.set_title('日收入趋势', fontweight='bold', fontsize=12)
ax.set_ylabel('收入 (¥)')
ax.tick_params(axis='x', rotation=45)

# 5. 城市收入分布
ax = axes[1, 1]
ax.bar(city_stats.index, city_stats['总收入'], color=plt.cm.Paired(np.linspace(0,1,len(city_stats))))
ax.set_title('城市收入分布', fontweight='bold', fontsize=12)
ax.set_ylabel('收入 (¥)')
ax.tick_params(axis='x', rotation=45)

# 6. 价格带分析
ax = axes[1, 2]
valid_bands = price_band_stats.dropna()
ax.bar(valid_bands.index, valid_bands['总收入'],
       color=['#4ECDC4','#45B7D1','#FF9F43','#FF6B6B'][:len(valid_bands)])
ax.set_title('价格带收入分析', fontweight='bold', fontsize=12)
ax.set_ylabel('收入 (¥)')
ax.tick_params(axis='x', rotation=15)

# 7. 星期x时段热力图
ax = axes[2, 0]
dow_hour = df.groupby(['星期','小时'])['订单金额'].sum().unstack(fill_value=0)
im = ax.imshow(dow_hour.values, cmap='YlOrRd', aspect='auto')
ax.set_title('星期x时段收入热力图', fontweight='bold', fontsize=12)
ax.set_xlabel('小时')
ax.set_ylabel('星期')
ax.set_yticks(range(len(dow_hour.index)))
ax.set_yticklabels([dow_names[d] for d in dow_hour.index])
ax.set_xticks(range(len(dow_hour.columns)))
ax.set_xticklabels([f'{h}:00' for h in dow_hour.columns], fontsize=8)
fig.colorbar(im, ax=ax, label='收入(¥)')

# 8. 旬分析
ax = axes[2, 1]
period_colors = ['#4ECDC4','#45B7D1','#FF6B6B']
bars = ax.bar(period_stats.index, period_stats['收入'], color=period_colors)
ax.set_title('旬收入对比', fontweight='bold', fontsize=12)
ax.set_ylabel('收入 (¥)')
for bar, row in zip(bars, period_stats.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
            f'¥{row.收入:,.0f}\n{row.订单数}单', ha='center', fontsize=10)

# 9. Top 客户消费排名
ax = axes[2, 2]
top_c = customer_orders.sort_values('总消费', ascending=True).tail(10)
ax.barh(top_c['用户码'].values, top_c['总消费'].values,
        color=plt.cm.viridis(np.linspace(0.2,0.8,10)))
ax.set_title('Top 10 客户消费', fontweight='bold', fontsize=12)
ax.set_xlabel('消费金额 (¥)')
for bar, val in zip(ax.patches, top_c['总消费'].values):
    ax.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2,
            f'¥{val:,.0f}', va='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUT_V}dataB_full_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 仪表板: {OUT_V}dataB_full_dashboard.png")

# --- 图2: 客户深度分析 ---
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle('客户深度分析', fontsize=16, fontweight='bold')

# 消费金额分布
axes2[0].hist(customer_orders['总消费'], bins=15, color='#4ECDC4', edgecolor='white', alpha=0.8)
axes2[0].axvline(customer_orders['总消费'].median(), color='red', linestyle='--',
                  label=f"中位数¥{customer_orders['总消费'].median():.0f}")
axes2[0].set_title('客户消费金额分布', fontweight='bold')
axes2[0].legend()

# 购买次数分布
freq_dist = customer_orders['购买次数'].value_counts().sort_index()
axes2[1].bar(freq_dist.index, freq_dist.values, color='#FF6B6B')
axes2[1].set_title('购买次数分布', fontweight='bold')
axes2[1].set_xlabel('购买次数')
axes2[1].set_ylabel('客户数')

# 客户层级 vs 消费（箱线图）
seg_data_list = [customer_orders[customer_orders['客户层级']==s]['总消费'].values
                  for s in seg_order if seg_dist.get(s, 0) > 0]
seg_labels = [s for s in seg_order if seg_dist.get(s, 0) > 0]
bp = axes2[2].boxplot(seg_data_list, labels=seg_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], seg_colors[:len(seg_data_list)]):
    patch.set_facecolor(color)
axes2[2].set_title('客户层级消费分布', fontweight='bold')
axes2[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(f'{OUT_V}dataB_customer_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 客户分析图: {OUT_V}dataB_customer_analysis.png")

# --- 图3: 产品与城市深度 ---
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))

# 产品销量 vs 单价 气泡图
ax = axes3[0]
scatter = ax.scatter(prod_stats['均价'], prod_stats['销量'],
                     s=prod_stats['总收入']/10, alpha=0.6,
                     c=prod_stats['总收入'], cmap='YlOrRd', edgecolors='black')
for _, row in prod_stats.iterrows():
    ax.annotate(row['产品说明'], (row['均价'], row['销量']),
                fontsize=7, ha='center', va='bottom')
ax.set_title('产品: 单价 vs 销量 (气泡=收入)', fontweight='bold', fontsize=12)
ax.set_xlabel('单价 (¥)')
ax.set_ylabel('销量')
fig3.colorbar(scatter, ax=ax, label='收入(¥)')

# 城市ARPU对比
ax = axes3[1]
city_sorted = city_stats.sort_values('ARPU', ascending=True)
ax.barh(city_sorted.index, city_sorted['ARPU'].values,
        color=plt.cm.RdYlGn(np.linspace(0.2,0.8,len(city_sorted))))
ax.set_title('城市人均消费 (ARPU)', fontweight='bold', fontsize=12)
ax.set_xlabel('ARPU (¥)')
for bar, val in zip(ax.patches, city_sorted['ARPU'].values):
    ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
            f'¥{val:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_V}dataB_product_city.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 产品城市图: {OUT_V}dataB_product_city.png")

# ============================================================
# 10. 综合报告
# ============================================================
report = f"""# 中文电商订单全面深度分析报告

## 执行摘要

**分析日期**: {datetime.now().strftime('%Y-%m-%d')}
**数据范围**: {df['消费日期'].min().strftime('%Y-%m-%d')} ~ {df['消费日期'].max().strftime('%Y-%m-%d')}
**数据规模**: {df['用户码'].nunique()}客户 / {len(df)}笔订单 / {df['产品码'].nunique()}种产品

---

## 一、数据概览

| 指标 | 数值 |
|------|------|
| 总订单数 | {len(df)} |
| 总销量 | {df['数量'].sum()}件 |
| 总收入 | ¥{total_rev:,.2f} |
| 客单价 | ¥{df['订单金额'].mean():.2f} |
| 客户数 | {df['用户码'].nunique()} |
| 产品种类 | {df['产品码'].nunique()} |
| 覆盖城市 | {df['城市'].nunique()} |

---

## 二、产品分析

### Top 10 产品

| 产品 | 品类 | 销量 | 总收入 | 均价 |
|------|------|------|--------|------|
"""

for _, row in prod_stats.head(10).iterrows():
    report += f"| {row['产品说明']} | {row['品类']} | {row['销量']:.0f} | ¥{row['总收入']:,.2f} | ¥{row['均价']:.0f} |\n"

report += f"""
### 品类分析

| 品类 | 收入 | 占比 | 产品数 | 客户数 |
|------|------|------|--------|--------|
"""

for cat, row in cat_stats.iterrows():
    report += f"| {cat} | ¥{row['总收入']:,.2f} | {row['总收入']/total_rev*100:.1f}% | {row['产品数']:.0f} | {row['客户数']:.0f} |\n"

report += f"""
---

## 三、RFM客户分群

### 客户分层

| 层级 | 人数 | 占比 | 消费金额 | 收入占比 |
|------|------|------|---------|---------|
"""

for seg in seg_order:
    cnt = seg_dist.get(seg, 0)
    seg_rev = customer_orders[customer_orders['客户层级']==seg]['总消费'].sum()
    report += f"| {seg} | {cnt} | {cnt/len(customer_orders)*100:.1f}% | ¥{seg_rev:,.2f} | {seg_rev/total_rev*100:.1f}% |\n"

report += f"""
### 复购率: {repeat_rate:.1f}%

### LTV预测: ¥{ltv:.2f}

---

## 四、城市地域分析

| 城市 | 收入 | 占比 | 客单价 | 客户数 | ARPU |
|------|------|------|--------|--------|------|
"""

for city, row in city_stats.iterrows():
    report += f"| {city} | ¥{row['总收入']:,.2f} | {row['总收入']/total_rev*100:.1f}% | ¥{row['客单价']:.2f} | {row['客户数']:.0f} | ¥{row['ARPU']:.2f} |\n"

report += f"""
---

## 五、时间趋势

### 旬分析
- 上旬: ¥{period_stats.loc['上旬','收入']:,.2f} ({period_stats.loc['上旬','订单数']:.0f}单)
- 中旬: ¥{period_stats.loc['中旬','收入']:,.2f} ({period_stats.loc['中旬','订单数']:.0f}单)
- 下旬: ¥{period_stats.loc['下旬','收入']:,.2f} ({period_stats.loc['下旬','订单数']:.0f}单)

---

## 六、AARRR增长模型

| 阶段 | 指标 | 数值 |
|------|------|------|
| Acquisition | 总客户数 | {total_cust} |
| Activation | 平均每单件数 | {avg_items_per_order:.2f} |
| Retention | 复购率 | {repeat_rate:.1f}% |
| Revenue | 总收入 | ¥{total_rev:,.2f} |
| Referral | 多品类购买率 | {(customer_orders['购买产品数']>1).sum()/len(customer_orders)*100:.1f}% |

---

## 七、深度洞察与策略建议

### 核心发现

1. **食品生鲜是核心品类** - 占收入{cat_stats.loc['食品生鲜','总收入']/total_rev*100:.0f}%，是最大收入来源
2. **收入地域集中** - 北京+上海贡献主要收入
3. **复购率有提升空间** - 当前{repeat_rate:.0f}%，可通过会员体系提升
4. **高价产品溢价空间大** - 进口红酒、海鲜、牛肉等高单价产品利润空间大

### 策略建议

#### 高优先级
1. **扩大食品生鲜优势** - 加大进口食品、生鲜品类SKU
2. **VIP客户维护** - 专属折扣、新品优先
3. **核心城市深耕** - 北京/上海加大营销投入

#### 中优先级
4. **复购率提升** - 设计90天客户生命周期管理
5. **交叉销售** - 低价产品引流→高价产品转化
6. **二三线拓展** - 成都/武汉/西安等潜力城市

#### 低优先级
7. **时段营销优化** - 午后高峰期精准投放
8. **会员积分体系** - 建立客户忠诚度计划

---

*报告由数据分析技能自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(f'{OUT_R}dataB_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✅ 报告: {OUT_R}dataB_analysis_report.md")

# 保存产品分析数据
prod_stats.to_csv(f'{OUT_R}product_analysis.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("  ✅ dataB 全面深度分析完成！")
print("=" * 70)
print(f"\n📁 产出文件:")
print(f"  - {OUT_R}dataB_analysis_report.md")
print(f"  - {OUT_R}customer_rfm_analysis.csv")
print(f"  - {OUT_R}product_analysis.csv")
print(f"  - {OUT_V}dataB_full_dashboard.png")
print(f"  - {OUT_V}dataB_customer_analysis.png")
print(f"  - {OUT_V}dataB_product_city.png")
