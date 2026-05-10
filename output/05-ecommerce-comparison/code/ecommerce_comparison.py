"""
5大电商平台跨平台全面深度对比分析
Amazon / Lazada / SHEIN / Shopee / Walmart
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT_R = './output/05-ecommerce-comparison/reports/'
OUT_V = './output/05-ecommerce-comparison/visualizations/'
os.makedirs(OUT_R, exist_ok=True)
os.makedirs(OUT_V, exist_ok=True)

# ============================================================
# 1. 数据加载与统一化
# ============================================================
print("=" * 70)
print("  5大电商平台跨平台全面深度对比分析")
print("=" * 70)

FX_RATES = {'USD':1,'INR':0.012,'GBP':1.27,'IDR':0.000064,'THB':0.028,
            'PHP':0.018,'SGD':0.74,'MYR':0.22,'MXN':0.059,'CLP':0.0011,
            'COP':0.00025,'VND':0.000042,'BRL':0.20,'TWD':0.032}

def safe_float(v):
    try: return float(v)
    except: return np.nan

def to_usd(val, currency):
    v = safe_float(val)
    return v * FX_RATES.get(currency, 1) if pd.notna(v) else np.nan

# --- Load all ---
print("\n[1/8] 加载并统一化数据...")

amazon = pd.read_csv('eCommerce-dataset-samples/amazon-products.csv', engine='python', on_bad_lines='skip')
lazada = pd.read_csv('eCommerce-dataset-samples/lazada-products.csv')
shein = pd.read_csv('eCommerce-dataset-samples/shein-products.csv')
shopee = pd.read_csv('eCommerce-dataset-samples/shopee-products.csv')
walmart = pd.read_csv('eCommerce-dataset-samples/walmart-products.csv')

# --- Build unified tables ---
records = []

# Amazon
for _, r in amazon.iterrows():
    fx = FX_RATES.get(str(r.get('currency','USD')), 1)
    fp = to_usd(r.get('final_price', np.nan), str(r.get('currency','USD')))
    ip = to_usd(r.get('initial_price', np.nan), str(r.get('currency','USD')))
    records.append({
        'platform': 'Amazon', 'title': str(r.get('title',''))[:80],
        'brand': str(r.get('brand','')), 'category': str(r.get('categories',''))[:60],
        'final_price_usd': fp, 'initial_price_usd': ip,
        'currency': str(r.get('currency','USD')),
        'rating': safe_float(r.get('rating', np.nan)),
        'reviews_count': safe_float(r.get('reviews_count', 0)),
        'has_discount': pd.notna(ip) and pd.notna(fp) and ip > fp,
    })

# Lazada
for _, r in lazada.iterrows():
    fx = FX_RATES.get(str(r.get('currency','USD')), 1)
    fp = to_usd(r.get('final_price', np.nan), str(r.get('currency','USD')))
    ip = to_usd(r.get('initial_price', np.nan), str(r.get('currency','USD')))
    records.append({
        'platform': 'Lazada', 'title': str(r.get('title',''))[:80],
        'brand': str(r.get('brand','')), 'category': str(r.get('breadcrumb',''))[:60],
        'final_price_usd': fp, 'initial_price_usd': ip,
        'currency': str(r.get('currency','USD')),
        'rating': safe_float(r.get('rating', np.nan)),
        'reviews_count': safe_float(r.get('reviews', 0)),
        'has_discount': pd.notna(ip) and pd.notna(fp) and ip > fp,
    })

# SHEIN
for _, r in shein.iterrows():
    fp = safe_float(r.get('final_price', np.nan))
    ip = safe_float(r.get('initial_price', np.nan))
    records.append({
        'platform': 'SHEIN', 'title': str(r.get('product_name',''))[:80],
        'brand': str(r.get('brand','')), 'category': str(r.get('root_category','')),
        'final_price_usd': fp, 'initial_price_usd': ip,
        'currency': 'USD', 'rating': safe_float(r.get('rating', np.nan)),
        'reviews_count': safe_float(r.get('reviews_count', 0)),
        'has_discount': pd.notna(ip) and pd.notna(fp) and ip > fp,
    })

# Shopee
for _, r in shopee.iterrows():
    fx = FX_RATES.get(str(r.get('currency','USD')), 1)
    fp = to_usd(r.get('final_price', np.nan), str(r.get('currency','USD')))
    ip = to_usd(r.get('initial_price', np.nan), str(r.get('currency','USD')))
    records.append({
        'platform': 'Shopee', 'title': str(r.get('title',''))[:80],
        'brand': str(r.get('brand','')), 'category': str(r.get('breadcrumb',''))[:60],
        'final_price_usd': fp, 'initial_price_usd': ip,
        'currency': str(r.get('currency','USD')),
        'rating': safe_float(r.get('rating', np.nan)),
        'reviews_count': safe_float(r.get('reviews', 0)),
        'has_discount': pd.notna(ip) and pd.notna(fp) and ip > fp,
    })

# Walmart
for _, r in walmart.iterrows():
    fp = safe_float(r.get('final_price', np.nan))
    ip = safe_float(r.get('initial_price', np.nan))
    records.append({
        'platform': 'Walmart', 'title': str(r.get('product_name',''))[:80],
        'brand': str(r.get('brand','')), 'category': str(r.get('root_category_name','')),
        'final_price_usd': fp, 'initial_price_usd': ip,
        'currency': 'USD', 'rating': safe_float(r.get('rating', np.nan)),
        'reviews_count': safe_float(r.get('review_count', 0)),
        'has_discount': pd.notna(ip) and pd.notna(fp) and ip > fp,
    })

df = pd.DataFrame(records)

# 折扣计算
df['discount_pct'] = np.where(
    df['initial_price_usd'] > 0,
    ((df['initial_price_usd'] - df['final_price_usd']) / df['initial_price_usd'] * 100),
    0
)
df['discount_pct'] = df['discount_pct'].clip(0, 100)

# 过滤极端价格（>$2000 USD 视为异常）
df_valid = df[(df['final_price_usd'] > 0) & (df['final_price_usd'] < 2000)].copy()

platforms = ['Amazon', 'Lazada', 'SHEIN', 'Shopee', 'Walmart']
colors_plat = ['#FF9900', '#0F146D', '#E11E1E', '#EE4D2D', '#0071DC']

print(f"  统一数据集: {len(df)} 产品")
for p in platforms:
    n = len(df_valid[df_valid['platform']==p])
    print(f"    {p}: {n} 产品")

# ============================================================
# 2. 平台概览对比
# ============================================================
print("\n" + "=" * 70)
print("  平台概览对比")
print("=" * 70)

overview = df_valid.groupby('platform').agg(
    产品数=('final_price_usd', 'count'),
    均价=('final_price_usd', 'mean'),
    中位价=('final_price_usd', 'median'),
    最低价=('final_price_usd', 'min'),
    最高价=('final_price_usd', 'max'),
    P25=('final_price_usd', lambda x: x.quantile(0.25)),
    P75=('final_price_usd', lambda x: x.quantile(0.75)),
    平均评分=('rating', 'mean'),
    平均评论数=('reviews_count', 'mean'),
    折扣率=('has_discount', 'mean'),
    平均折扣=('discount_pct', 'mean'),
).reindex(platforms)

print(f"\n📊 平台核心指标对比:")
print(f"{'平台':10s} {'产品数':>6s} {'均价$':>8s} {'中位价$':>8s} {'评分':>5s} {'折扣率':>6s} {'均折%':>6s}")
for p in platforms:
    r = overview.loc[p]
    print(f"{p:10s} {r['产品数']:>6.0f} {r['均价']:>8.2f} {r['中位价']:>8.2f} "
          f"{r['平均评分']:>5.2f} {r['折扣率']*100:>5.1f}% {r['平均折扣']:>5.1f}%")

# ============================================================
# 3. 定价策略深度对比
# ============================================================
print("\n" + "=" * 70)
print("  定价策略深度对比")
print("=" * 70)

def price_band_usd(p):
    if p < 5: return '超低价(<$5)'
    elif p < 15: return '低价($5-15)'
    elif p < 30: return '中价($15-30)'
    elif p < 50: return '中高价($30-50)'
    elif p < 100: return '高价($50-100)'
    else: return '超高价(>$100)'

df_valid['price_band'] = df_valid['final_price_usd'].apply(price_band_usd)
band_order = ['超低价(<$5)','低价($5-15)','中价($15-30)','中高价($30-50)','高价($50-100)','超高价(>$100)']

print(f"\n💰 各平台价格带分布 (%):")
band_cross = pd.crosstab(df_valid['platform'], df_valid['price_band'], normalize='index') * 100
band_cross = band_cross.reindex(platforms)[band_order]
print(band_cross.round(1).to_string())

# 价格定位
print(f"\n🏷️ 平台价格定位:")
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]['final_price_usd']
    top_band = pdata.apply(price_band_usd).value_counts().index[0]
    print(f"  {p}: 主力价格带={top_band}, 均价${pdata.mean():.2f}, IQR=${pdata.quantile(0.25):.2f}~${pdata.quantile(0.75):.2f}")

# ============================================================
# 4. 折扣策略对比
# ============================================================
print("\n" + "=" * 70)
print("  折扣策略深度对比")
print("=" * 70)

print(f"\n💸 折扣策略对比:")
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    disc = pdata[pdata['has_discount']==True]
    disc_rate = len(disc) / len(pdata) * 100
    avg_disc = disc['discount_pct'].mean() if len(disc) > 0 else 0
    max_disc = disc['discount_pct'].max() if len(disc) > 0 else 0
    print(f"  {p}: 折扣率{disc_rate:.1f}%, 均折{avg_disc:.1f}%, 最大折{max_disc:.1f}%")

# 折扣力度分布
def disc_level(pct):
    if pct == 0: return '无折扣'
    elif pct < 15: return '轻度(<15%)'
    elif pct < 30: return '中度(15-30%)'
    else: return '重度(>30%)'

df_valid['disc_level'] = df_valid['discount_pct'].apply(disc_level)
disc_cross = pd.crosstab(df_valid['platform'], df_valid['disc_level'], normalize='index') * 100
disc_cross = disc_cross.reindex(platforms)
print(f"\n  折扣力度分布 (%):")
print(disc_cross[['无折扣','轻度(<15%)','中度(15-30%)','重度(>30%)']].round(1).to_string())

# ============================================================
# 5. 评分与口碑对比
# ============================================================
print("\n" + "=" * 70)
print("  评分与口碑对比")
print("=" * 70)

print(f"\n⭐ 评分对比:")
for p in platforms:
    pdata = df_valid[(df_valid['platform']==p) & (df_valid['rating']>0)]
    if len(pdata) > 0:
        print(f"  {p}: 均分{pdata['rating'].mean():.2f}, "
              f"有评分产品{len(pdata)}/{len(df_valid[df_valid['platform']==p])} "
              f"({len(pdata)/len(df_valid[df_valid['platform']==p])*100:.0f}%)")
    else:
        print(f"  {p}: 无评分数据")

# 评论活跃度
print(f"\n💬 评论活跃度:")
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    avg_rev = pdata['reviews_count'].mean()
    med_rev = pdata['reviews_count'].median()
    has_rev = (pdata['reviews_count'] > 0).sum()
    print(f"  {p}: 均值{avg_rev:.0f}条, 中位{med_rev:.0f}条, "
          f"有评论{has_rev}/{len(pdata)} ({has_rev/len(pdata)*100:.0f}%)")

# ============================================================
# 6. 品牌分布对比
# ============================================================
print("\n" + "=" * 70)
print("  品牌分布对比")
print("=" * 70)

print(f"\n🏢 品牌多样性:")
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    n_brands = pdata['brand'].nunique()
    top_brands = pdata['brand'].value_counts().head(3)
    top_str = ", ".join([f"{b}({c})" for b, c in top_brands.items() if b != 'nan'])
    print(f"  {p}: {n_brands}个品牌, Top: {top_str}")

# ============================================================
# 7. 市场覆盖与竞争格局
# ============================================================
print("\n" + "=" * 70)
print("  市场覆盖与竞争格局")
print("=" * 70)

# 货币分布
print(f"\n🌍 市场覆盖 (货币分布):")
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    curr_dist = pdata['currency'].value_counts()
    curr_str = ", ".join([f"{c}({n})" for c, n in curr_dist.items()])
    print(f"  {p}: {curr_str}")

# ============================================================
# 8. 综合竞争力评分
# ============================================================
print("\n" + "=" * 70)
print("  综合竞争力评分")
print("=" * 70)

scores = {}
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    # 价格竞争力 (越低越好, 1-10)
    price_score = max(1, 10 - pdata['final_price_usd'].median() / 10)
    # 折扣力度 (越大越好, 1-10)
    disc_score = min(10, pdata['discount_pct'].mean() / 3)
    # 评分 (越高越好, 1-10)
    rated = pdata[pdata['rating']>0]
    rating_score = rated['rating'].mean() * 2 if len(rated) > 0 else 0
    # 评论活跃 (1-10)
    rev_score = min(10, pdata['reviews_count'].mean() / 500)
    # 品牌多样性 (1-10)
    brand_score = min(10, pdata['brand'].nunique() / 20)
    # 市场覆盖 (1-10)
    market_score = min(10, pdata['currency'].nunique() * 2)

    scores[p] = {
        '价格竞争力': round(price_score, 1),
        '折扣力度': round(disc_score, 1),
        '用户评分': round(rating_score, 1),
        '评论活跃度': round(rev_score, 1),
        '品牌多样性': round(brand_score, 1),
        '市场覆盖': round(market_score, 1),
        '综合得分': round((price_score + disc_score + rating_score + rev_score + brand_score + market_score) / 6, 1)
    }

print(f"\n🏆 综合竞争力评分 (10分制):")
dims = ['价格竞争力','折扣力度','用户评分','评论活跃度','品牌多样性','市场覆盖','综合得分']
print(f"{'维度':12s}", end='')
for p in platforms:
    print(f"  {p:>10s}", end='')
print()
for dim in dims:
    print(f"{dim:12s}", end='')
    for p in platforms:
        print(f"  {scores[p][dim]:>10.1f}", end='')
    print()

# ============================================================
# 9. 可视化
# ============================================================
print("\n" + "=" * 70)
print("  生成可视化图表...")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(22, 20))
fig.suptitle('5大电商平台跨平台全面对比分析', fontsize=20, fontweight='bold', y=0.99)

# 1. 产品数对比
ax = axes[0, 0]
counts = [overview.loc[p, '产品数'] for p in platforms]
bars = ax.bar(platforms, counts, color=colors_plat)
ax.set_title('产品样本量', fontweight='bold', fontsize=12)
ax.set_ylabel('产品数')
for bar, c in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f'{c:.0f}', ha='center', fontsize=10)

# 2. 价格分布箱线图
ax = axes[0, 1]
box_data = [df_valid[df_valid['platform']==p]['final_price_usd'].values for p in platforms]
bp = ax.boxplot(box_data, labels=platforms, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors_plat):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_title('价格分布 (USD, 无异常值)', fontweight='bold', fontsize=12)
ax.set_ylabel('价格 (USD)')
ax.tick_params(axis='x', rotation=15)

# 3. 均价 vs 中位价
ax = axes[0, 2]
x = range(len(platforms))
avg_p = [overview.loc[p, '均价'] for p in platforms]
med_p = [overview.loc[p, '中位价'] for p in platforms]
width = 0.35
ax.bar([i-width/2 for i in x], avg_p, width, label='均价', color=[c+'99' for c in colors_plat], edgecolor=colors_plat, linewidth=2)
ax.bar([i+width/2 for i in x], med_p, width, label='中位价', color=colors_plat)
ax.set_xticks(x)
ax.set_xticklabels(platforms, rotation=15)
ax.set_title('均价 vs 中位价 (USD)', fontweight='bold', fontsize=12)
ax.set_ylabel('价格 (USD)')
ax.legend()

# 4. 价格带堆叠柱
ax = axes[1, 0]
band_cross.plot(kind='bar', stacked=True, ax=ax,
                color=['#4ECDC4','#45B7D1','#96CEB4','#FECA57','#FF9F43','#FF6B6B'])
ax.set_title('价格带分布 (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('占比 (%)')
ax.set_xticklabels(platforms, rotation=15)
ax.legend(fontsize=7, loc='upper right')

# 5. 折扣策略对比
ax = axes[1, 1]
disc_rates = [overview.loc[p, '折扣率']*100 for p in platforms]
avg_discs = [overview.loc[p, '平均折扣'] for p in platforms]
ax2 = ax.twinx()
ax.bar(x, disc_rates, 0.35, label='折扣产品比例%', color=colors_plat, alpha=0.6)
ax2.bar([i+0.35 for i in x], avg_discs, 0.35, label='平均折扣%',
        color=colors_plat, edgecolor='black', linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(platforms, rotation=15)
ax.set_title('折扣策略对比', fontweight='bold', fontsize=12)
ax.set_ylabel('折扣产品比例 (%)')
ax2.set_ylabel('平均折扣幅度 (%)')

# 6. 评分对比
ax = axes[1, 2]
ratings = []
for p in platforms:
    pdata = df_valid[(df_valid['platform']==p) & (df_valid['rating']>0)]
    ratings.append(pdata['rating'].mean() if len(pdata)>0 else 0)
bars = ax.bar(platforms, ratings, color=colors_plat)
ax.set_title('平均用户评分', fontweight='bold', fontsize=12)
ax.set_ylabel('评分')
ax.set_ylim(0, 5)
ax.axhline(4.0, color='gray', linestyle='--', alpha=0.5)
for bar, r in zip(bars, ratings):
    if r > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{r:.2f}', ha='center', fontsize=10)
    else:
        ax.text(bar.get_x()+bar.get_width()/2, 0.1, 'N/A', ha='center', fontsize=10, color='gray')

# 7. 评论量对比 (log)
ax = axes[2, 0]
rev_data = []
for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    rev_data.append(pdata[pdata['reviews_count']>0]['reviews_count'].median() if (pdata['reviews_count']>0).any() else 0)
bars = ax.bar(platforms, [r+1 for r in rev_data], color=colors_plat)
ax.set_title('中位评论数', fontweight='bold', fontsize=12)
ax.set_ylabel('评论数 (中位数)')
ax.set_yscale('log')
for bar, r in zip(bars, rev_data):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.2, f'{r:.0f}', ha='center', fontsize=9)

# 8. 雷达图 - 综合竞争力
ax = axes[2, 1]
dims_radar = ['价格竞争力','折扣力度','用户评分','评论活跃度','品牌多样性','市场覆盖']
angles = np.linspace(0, 2*np.pi, len(dims_radar), endpoint=False).tolist()
angles += angles[:1]

for i, p in enumerate(platforms):
    values = [scores[p][d] for d in dims_radar]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=p, color=colors_plat[i])
    ax.fill(angles, values, alpha=0.05, color=colors_plat[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims_radar, fontsize=8)
ax.set_ylim(0, 10)
ax.set_title('综合竞争力雷达图', fontweight='bold', fontsize=12)
ax.legend(loc='upper right', fontsize=7)

# 9. 综合得分排名
ax = axes[2, 2]
sorted_platforms = sorted(platforms, key=lambda p: scores[p]['综合得分'], reverse=True)
sorted_scores = [scores[p]['综合得分'] for p in sorted_platforms]
sorted_colors = [colors_plat[platforms.index(p)] for p in sorted_platforms]
bars = ax.barh(sorted_platforms[::-1], sorted_scores[::-1], color=sorted_colors[::-1])
ax.set_title('综合竞争力排名', fontweight='bold', fontsize=12)
ax.set_xlabel('综合得分 (10分制)')
for bar, s in zip(bars, sorted_scores[::-1]):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f'{s:.1f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUT_V}ecommerce_comparison_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 仪表板: {OUT_V}ecommerce_comparison_dashboard.png")

# --- 图2: 价格分布深度对比 ---
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle('电商平台定价策略深度对比', fontsize=16, fontweight='bold')

# 价格直方图叠加
ax = axes2[0]
for i, p in enumerate(platforms):
    pdata = df_valid[(df_valid['platform']==p) & (df_valid['final_price_usd'] < 200)]
    ax.hist(pdata['final_price_usd'], bins=30, alpha=0.4, label=p, color=colors_plat[i])
ax.set_title('价格分布叠加 (USD<$200)', fontweight='bold')
ax.set_xlabel('价格 (USD)')
ax.set_ylabel('产品数')
ax.legend()

# 折扣力度箱线图
ax = axes2[1]
disc_data = [df_valid[(df_valid['platform']==p) & (df_valid['discount_pct']>0)]['discount_pct'].values for p in platforms]
bp = ax.boxplot(disc_data, labels=platforms, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors_plat):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_title('折扣幅度分布 (有折扣产品)', fontweight='bold')
ax.set_ylabel('折扣率 (%)')
ax.tick_params(axis='x', rotation=15)

# 价格分位数对比
ax = axes2[1]
# reuse ax[1] already done, do price quantiles on ax[2]
ax = axes2[2]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
for i, p in enumerate(platforms):
    pdata = df_valid[df_valid['platform']==p]['final_price_usd']
    qvals = [pdata.quantile(q) for q in quantiles]
    ax.plot(quantiles, qvals, 'o-', label=p, color=colors_plat[i], linewidth=2)
ax.set_title('价格分位数曲线', fontweight='bold')
ax.set_xlabel('分位数')
ax.set_ylabel('价格 (USD)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT_V}ecommerce_pricing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 定价对比图: {OUT_V}ecommerce_pricing_comparison.png")

# ============================================================
# 10. 综合报告
# ============================================================
report = f"""# 5大电商平台跨平台全面深度对比分析报告

## 执行摘要

**分析日期**: {datetime.now().strftime('%Y-%m-%d')}
**数据规模**: Amazon({len(amazon)}) + Lazada({len(lazada)}) + SHEIN({len(shein)}) + Shopee({len(shopee)}) + Walmart({len(walmart)}) = **{len(df)}产品**
**统一基准**: 所有价格已转换为美元 (USD)

---

## 一、平台核心指标对比

| 指标 | Amazon | Lazada | SHEIN | Shopee | Walmart |
|------|--------|--------|-------|--------|---------|
"""

for dim_name in ['产品数','均价($)','中位价($)','平均评分','折扣率','平均折扣(%)']:
    report += f"| {dim_name} |"
    for p in platforms:
        r = overview.loc[p]
        if dim_name == '产品数': report += f" {r['产品数']:.0f} |"
        elif dim_name == '均价($)': report += f" ${r['均价']:.2f} |"
        elif dim_name == '中位价($)': report += f" ${r['中位价']:.2f} |"
        elif dim_name == '平均评分': report += f" {r['平均评分']:.2f} |"
        elif dim_name == '折扣率': report += f" {r['折扣率']*100:.0f}% |"
        elif dim_name == '平均折扣(%)': report += f" {r['平均折扣']:.1f}% |"
    report += "\n"

report += f"""
---

## 二、定价策略对比

### 价格定位
"""

for p in platforms:
    pdata = df_valid[df_valid['platform']==p]['final_price_usd']
    top_band = pdata.apply(price_band_usd).value_counts().index[0]
    report += f"- **{p}**: 主力价格带={top_band}, 均价${pdata.mean():.2f}\n"

report += f"""
### 价格带分布 (%)

| 价格带 | Amazon | Lazada | SHEIN | Shopee | Walmart |
|--------|--------|--------|-------|--------|---------|
"""

for band in band_order:
    report += f"| {band} |"
    for p in platforms:
        val = band_cross.loc[p, band] if band in band_cross.columns else 0
        report += f" {val:.1f}% |"
    report += "\n"

report += f"""
---

## 三、折扣策略对比

"""

for p in platforms:
    pdata = df_valid[df_valid['platform']==p]
    disc = pdata[pdata['has_discount']==True]
    report += f"- **{p}**: 折扣率{len(disc)/len(pdata)*100:.0f}%, 平均折扣{disc['discount_pct'].mean():.1f}%\n"

report += f"""
---

## 四、评分与口碑对比

"""

for p in platforms:
    pdata = df_valid[(df_valid['platform']==p) & (df_valid['rating']>0)]
    if len(pdata) > 0:
        report += f"- **{p}**: {pdata['rating'].mean():.2f}星 ({len(pdata)}产品有评分)\n"
    else:
        report += f"- **{p}**: 无评分数据\n"

report += f"""
---

## 五、综合竞争力评分 (10分制)

| 维度 | Amazon | Lazada | SHEIN | Shopee | Walmart |
|------|--------|--------|-------|--------|---------|
"""

for dim in dims_radar:
    report += f"| {dim} |"
    for p in platforms:
        report += f" {scores[p][dim]:.1f} |"
    report += "\n"

report += f"| **综合得分** |"
for p in platforms:
    report += f" **{scores[p]['综合得分']:.1f}** |"
report += "\n"

report += f"""
---

## 六、深度洞察与策略建议

### 核心发现

1. **价格鸿沟显著**: SHEIN中位价仅${overview.loc['SHEIN','中位价']:.2f}，Walmart ${overview.loc['Walmart','中位价']:.2f}，Amazon ${overview.loc['Amazon','中位价']:.2f}
2. **折扣策略分化**: SHEIN折扣率最高({overview.loc['SHEIN','折扣率']*100:.0f}%)，Unbeatablesale(Amazon第三方)零折扣
3. **口碑差异明显**: Amazon评分最高({overview.loc['Amazon','平均评分']:.2f})，Shopee最低({overview.loc['Shopee','平均评分']:.2f})
4. **市场定位清晰**:
   - Amazon/Walmart = 品质+信任 (高价、高评分)
   - SHEIN = 极致性价比 (超低价、高折扣)
   - Lazada/Shopee = 东南亚市场覆盖 (多货币)

### 平台竞争力画像

| 平台 | 定位 | 优势 | 劣势 |
|------|------|------|------|
| Amazon | 品质电商 | 高评分、高评论量 | 价格偏高 |
| Lazada | 东南亚综合 | 多市场覆盖 | 评分中等 |
| SHEIN | 快时尚低价 | 极致低价、高折扣 | 无评分数据 |
| Shopee | 东南亚社交 | 最广市场覆盖 | 价格波动大 |
| Walmart | 全品类零售 | 品质+合理价格 | 折扣力度低 |

### 策略建议

1. **Amazon卖家**: 利用高信任度溢价，注重产品质量和评论积累
2. **SHEIN卖家**: 需要极致供应链效率，薄利多销
3. **东南亚市场**: Lazada+Shopee双平台覆盖，注意本地化定价
4. **Walmart卖家**: 品质与价格平衡，适合中端品牌

---

*报告由跨平台数据分析技能自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(f'{OUT_R}ecommerce_comparison_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

# 保存统一数据
df_valid.to_csv(f'{OUT_R}ecommerce_unified_data.csv', index=False, encoding='utf-8-sig')

print(f"  ✅ 报告: {OUT_R}ecommerce_comparison_report.md")

print("\n" + "=" * 70)
print("  ✅ 5大电商平台跨平台全面对比分析完成！")
print("=" * 70)
print(f"\n📁 产出文件:")
print(f"  - {OUT_R}ecommerce_comparison_report.md")
print(f"  - {OUT_R}ecommerce_unified_data.csv")
print(f"  - {OUT_V}ecommerce_comparison_dashboard.png")
print(f"  - {OUT_V}ecommerce_pricing_comparison.png")
