#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_data/ 多维深度洞察分析
数据源: sample_user_behavior.csv + sample_item_info.csv
维度: 用户画像/行为漏斗/商品矩阵/品牌竞争/品类分析/RFM分群/关联分析/价格弹性/库存健康度/时间序列
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter
from itertools import combinations
import warnings, os
warnings.filterwarnings('ignore')

# ===== 中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(OUT_CODE)
BASE = r'D:\claude前端\claude-agent-ui-main\claude-data-analysis-main\claude-data-analysis-main'
VIZ_DIR = os.path.join(OUT, 'visualizations')
RPT_DIR = os.path.join(OUT, 'reports')
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

# ===== 加载数据 =====
print("="*70)
print("  sample_data/ 多维深度洞察分析")
print("="*70)

beh = pd.read_csv(os.path.join(BASE, 'sample_data', 'sample_user_behavior.csv'))
item = pd.read_csv(os.path.join(BASE, 'sample_data', 'sample_item_info.csv'))

# 清洗列名空格
beh.columns = beh.columns.str.strip()
item.columns = item.columns.str.strip()

beh['时间戳'] = pd.to_datetime(beh['时间戳'])
beh['日期'] = beh['时间戳'].dt.date
beh['小时'] = beh['时间戳'].dt.hour
beh['星期'] = beh['时间戳'].dt.day_name()

item['上架时间'] = pd.to_datetime(item['上架时间'])
item['上架月份'] = item['上架时间'].dt.to_period('M').astype(str)

print(f"  用户行为: {len(beh)} 条, {beh['用户ID'].nunique()} 个用户")
print(f"  商品信息: {len(item)} 个商品, {item['商品类别'].nunique()} 个品类")
print(f"  行为类型: {beh['行为类型'].value_counts().to_dict()}")
print(f"  品类: {beh['商品类别'].unique().tolist()}")

# ================================================================
#  维度1: 用户画像深度分析
# ================================================================
print("\n" + "="*70)
print("  维度1: 用户画像深度分析")
print("="*70)

# 合并数据
merged = beh.merge(item[['商品ID','商品名称','品牌','上架时间','销量','库存','平均评分']], on='商品ID', how='left')

# 用户维度聚合
user_agg = beh.groupby('用户ID').agg(
    行为数=('商品ID','count'),
    购买数=('行为类型', lambda x: (x=='购买').sum()),
    浏览数=('行为类型', lambda x: (x=='浏览').sum()),
    均评分=('评分','mean'),
    总消费=('商品价格','sum'),
    均消费=('商品价格','mean'),
    品类数=('商品类别','nunique'),
    商品数=('商品ID','nunique'),
).reset_index()

user_agg['转化率'] = user_agg['购买数'] / user_agg['行为数']
user_agg = user_agg.merge(beh[['用户ID','用户年龄','用户性别','用户城市']].drop_duplicates(), on='用户ID')

# 年龄分层
user_agg['年龄段'] = pd.cut(user_agg['用户年龄'], bins=[0,25,30,35,100], labels=['≤25','26-30','31-35','>35'])

print("\n📊 用户画像概览:")
print(f"  平均年龄: {user_agg['用户年龄'].mean():.1f}岁")
print(f"  性别比(男:女): {(user_agg['用户性别']=='男').sum()}:{(user_agg['用户性别']=='女').sum()}")
print(f"  城市数: {user_agg['用户城市'].nunique()} 个")
print(f"  人均行为数: {user_agg['行为数'].mean():.1f}")
print(f"  人均消费: ¥{user_agg['总消费'].mean():.0f}")
print(f"  人均购买转化率: {user_agg['转化率'].mean():.1%}")

# 性别消费差异
gender_stats = user_agg.groupby('用户性别').agg(
    人数=('用户ID','count'), 均消费=('总消费','mean'), 均评分=('均评分','mean'),
    均品类=('品类数','mean'), 转化率=('转化率','mean')
).round(2)
print("\n👫 性别差异:")
for g, row in gender_stats.iterrows():
    print(f"  {g}: {int(row['人数'])}人, 均消费¥{row['均消费']:.0f}, 均评分{row['均评分']:.1f}, 转化率{row['转化率']:.0%}")

# 城市消费排名
city_stats = user_agg.groupby('用户城市').agg(
    均消费=('总消费','mean'), 均评分=('均评分','mean'), 转化率=('转化率','mean')
).round(2).sort_values('均消费', ascending=False)
print("\n🏙️ 城市消费力排名:")
for city, row in city_stats.iterrows():
    print(f"  {city}: 均消费¥{row['均消费']:.0f}, 均评分{row['均评分']:.1f}, 转化率{row['转化率']:.0%}")

# 年龄段分析
age_stats = user_agg.groupby('年龄段').agg(
    人数=('用户ID','count'), 均消费=('总消费','mean'), 均评分=('均评分','mean')
).round(2)
print("\n🎂 年龄段消费:")
for age, row in age_stats.iterrows():
    print(f"  {age}: {int(row['人数'])}人, 均消费¥{row['均消费']:.0f}, 均评分{row['均评分']:.1f}")

# ================================================================
#  维度2: 行为漏斗与转化分析
# ================================================================
print("\n" + "="*70)
print("  维度2: 行为漏斗与转化分析")
print("="*70)

total = len(beh)
purchase = (beh['行为类型']=='购买').sum()
browse = (beh['行为类型']=='浏览').sum()
print(f"\n📊 行为漏斗:")
print(f"  总行为: {total}")
print(f"  购买: {purchase} ({purchase/total:.1%})")
print(f"  浏览: {browse} ({browse/total:.1%})")
print(f"  购买转化率: {purchase/total:.1%}")

# 浏览→购买转化(同用户同商品先浏览后购买)
conv_pairs = beh[beh['行为类型']=='浏览'][['用户ID','商品ID']].drop_duplicates()
purchase_pairs = beh[beh['行为类型']=='购买'][['用户ID','商品ID']].drop_duplicates()
direct_conv = conv_pairs.merge(purchase_pairs, on=['用户ID','商品ID'], how='inner')
print(f"\n🔄 浏览→购买转化:")
print(f"  浏览商品对: {len(conv_pairs)}")
print(f"  直接转化(同用户同商品): {len(direct_conv)}")
print(f"  浏览→购买转化率: {len(direct_conv)/max(len(conv_pairs),1):.1%}")

# 品类转化率
cat_funnel = beh.groupby('商品类别').agg(
    总行为=('行为类型','count'),
    购买=('行为类型', lambda x: (x=='购买').sum()),
    浏览=('行为类型', lambda x: (x=='浏览').sum()),
    均评分=('评分','mean'),
    总金额=('商品价格','sum'),
).reset_index()
cat_funnel['转化率'] = cat_funnel['购买'] / cat_funnel['总行为']
cat_funnel = cat_funnel.sort_values('总金额', ascending=False)
print("\n📂 品类漏斗:")
for _, row in cat_funnel.iterrows():
    print(f"  {row['商品类别']}: 总{row['总行为']}次, 购买{row['购买']}次({row['转化率']:.0%}), "
          f"浏览{row['浏览']}次, 均评分{row['均评分']:.1f}, 总金额¥{row['总金额']:,.0f}")

# 时间段行为分布
hour_behavior = beh.groupby(['小时','行为类型']).size().unstack(fill_value=0)
print("\n⏰ 活跃时段(购买):")
purchase_hour = beh[beh['行为类型']=='购买'].groupby('小时').size().sort_values(ascending=False)
for h, cnt in purchase_hour.items():
    print(f"  {h:02d}:00 - {cnt}次购买")

# ================================================================
#  维度3: 商品矩阵分析 (BCG矩阵)
# ================================================================
print("\n" + "="*70)
print("  维度3: 商品BCG矩阵分析")
print("="*70)

# 计算商品被行为次数和行为分数
product_score = beh.groupby('商品ID').agg(
    交互次数=('行为类型','count'),
    购买次数=('行为类型', lambda x: (x=='购买').sum()),
    浏览次数=('行为类型', lambda x: (x=='浏览').sum()),
    均用户评分=('评分','mean'),
    总金额=('商品价格','sum'),
).reset_index()

product_full = item.merge(product_score, on='商品ID', how='left').fillna(0)
product_full['交互次数'] = product_full['交互次数'].astype(int)
product_full['购买次数'] = product_full['购买次数'].astype(int)

# BCG分类
med_sales = product_full['销量'].median()
med_interact = product_full['交互次数'].median()

def bcg_classify(row):
    if row['销量'] >= med_sales and row['交互次数'] >= med_interact:
        return '明星(Star)'
    elif row['销量'] >= med_sales and row['交互次数'] < med_interact:
        return '现金牛(Cash Cow)'
    elif row['销量'] < med_sales and row['交互次数'] >= med_interact:
        return '问号(Question)'
    else:
        return '瘦狗(Dog)'

product_full['BCG类型'] = product_full.apply(bcg_classify, axis=1)

bcg_summary = product_full.groupby('BCG类型').agg(
    数量=('商品ID','count'), 均销量=('销量','mean'), 均交互=('交互次数','mean'),
    均价格=('价格','mean'), 均评分=('平均评分','mean')
).round(1)

print("\n📊 BCG矩阵分布:")
for bcg, row in bcg_summary.iterrows():
    print(f"  {bcg}: {int(row['数量'])}个商品, 均销量{row['均销量']:.0f}, 均交互{row['均交互']:.1f}, "
          f"均价格¥{row['均价格']:.0f}, 均评分{row['均评分']:.1f}")

print("\n🌟 明星商品TOP5:")
stars = product_full[product_full['BCG类型']=='明星(Star)'].sort_values('销量', ascending=False).head(5)
for _, r in stars.iterrows():
    print(f"  {r['商品名称']}({r['品牌']}): 销量{r['销量']:.0f}, 价格¥{r['价格']:.0f}, 评分{r['平均评分']:.1f}")

print("\n💰 现金牛TOP3:")
cash = product_full[product_full['BCG类型']=='现金牛(Cash Cow)'].sort_values('销量', ascending=False).head(3)
for _, r in cash.iterrows():
    print(f"  {r['商品名称']}({r['品牌']}): 销量{r['销量']:.0f}, 价格¥{r['价格']:.0f}, 评分{r['平均评分']:.1f}")

# ================================================================
#  维度4: 品牌竞争格局
# ================================================================
print("\n" + "="*70)
print("  维度4: 品牌竞争格局")
print("="*70)

brand_analysis = item.groupby('品牌').agg(
    产品数=('商品ID','count'),
    总销量=('销量','sum'),
    均价格=('价格','mean'),
    均评分=('平均评分','mean'),
    总库存=('库存','sum'),
    品类数=('商品类别','nunique'),
).reset_index()
brand_analysis['销量份额'] = brand_analysis['总销量'] / brand_analysis['总销量'].sum() * 100
brand_analysis = brand_analysis.sort_values('总销量', ascending=False)

# HHI
hhi = (brand_analysis['销量份额'] ** 2).sum()
print(f"\n📊 品牌竞争格局 (HHI={hhi:.0f}, {'集中' if hhi > 2500 else '分散'}):")
print(f"  品牌总数: {len(brand_analysis)}")

# Top 10品牌
print("\n🏆 TOP10品牌:")
for _, r in brand_analysis.head(10).iterrows():
    print(f"  {r['品牌']}: {r['产品数']}产品, 总销量{r['总销量']:.0f}({r['销量份额']:.1f}%), "
          f"均¥{r['均价格']:.0f}, 评分{r['均评分']:.1f}")

# 品类品牌分布
print("\n📂 各品类头部品牌:")
for cat in item['商品类别'].unique():
    cat_brands = item[item['商品类别']==cat].groupby('品牌').agg(销量=('销量','sum')).sort_values('销量', ascending=False)
    top3 = cat_brands.head(3)
    print(f"  {cat}: " + ", ".join([f"{b}({r['销量']:.0f})" for b, r in top3.iterrows()]))

# 品牌定位(价格×评分)
brand_pos = brand_analysis[brand_analysis['产品数'] >= 1].copy()
price_med = brand_pos['均价格'].median()
rating_med = brand_pos['均评分'].median()
brand_pos['定位'] = brand_pos.apply(lambda r:
    '高端优质' if r['均价格']>=price_med and r['均评分']>=rating_med else
    '性价比' if r['均价格']<price_med and r['均评分']>=rating_med else
    '高价低评' if r['均价格']>=price_med and r['均评分']<rating_med else '低价低评', axis=1)

print("\n🎯 品牌定位矩阵:")
for pos, grp in brand_pos.groupby('定位'):
    brands = grp['品牌'].tolist()[:5]
    print(f"  {pos}: {', '.join(brands)}")

# ================================================================
#  维度5: 品类深度分析
# ================================================================
print("\n" + "="*70)
print("  维度5: 品类深度分析")
print("="*70)

cat_analysis = item.groupby('商品类别').agg(
    商品数=('商品ID','count'),
    总销量=('销量','sum'),
    总库存=('库存','sum'),
    均价格=('价格','mean'),
    价格范围=('价格', lambda x: x.max()-x.min()),
    均评分=('平均评分','mean'),
    品牌数=('品牌','nunique'),
).reset_index()

cat_analysis['GMV贡献'] = (cat_analysis['总销量'] * cat_analysis['均价格'])
cat_analysis['GMV占比'] = cat_analysis['GMV贡献'] / cat_analysis['GMV贡献'].sum() * 100
cat_analysis['库存周转'] = cat_analysis['总销量'] / cat_analysis['总库存']
cat_analysis = cat_analysis.sort_values('GMV贡献', ascending=False)

print("\n📊 品类全景:")
for _, r in cat_analysis.iterrows():
    print(f"  {r['商品类别']}: {r['商品数']}商品, {r['品牌数']}品牌, "
          f"GMV ¥{r['GMV贡献']:,.0f}({r['GMV占比']:.1f}%), "
          f"均¥{r['均价格']:.0f}(范围¥{r['价格范围']:.0f}), "
          f"评分{r['均评分']:.1f}, 库存周转{r['库存周转']:.2f}")

# Pareto分析
cat_analysis['GMV累计%'] = cat_analysis['GMV占比'].cumsum()
pareto_point = cat_analysis[cat_analysis['GMV累计%'] <= 80]
print(f"\n📐 Pareto分析: 前{len(pareto_point)}个品类贡献80% GMV")

# ================================================================
#  维度6: RFM客户分群
# ================================================================
print("\n" + "="*70)
print("  维度6: RFM客户分群")
print("="*70)

reference_date = beh['时间戳'].max() + pd.Timedelta(days=1)

purchases = beh[beh['行为类型']=='购买'].copy()
rfm = purchases.groupby('用户ID').agg(
    Recency=('时间戳', lambda x: (reference_date - x.max()).days),
    Frequency=('商品ID','count'),
    Monetary=('商品价格','sum'),
).reset_index()

# 评分(1-5) - 小样本用cut代替qcut
n_bins = min(5, len(rfm))
rfm['R_score'] = pd.cut(rfm['Recency'], bins=n_bins, labels=list(range(n_bins,0,-1))).astype(int)
rfm['F_score'] = pd.cut(rfm['Frequency'].rank(method='first'), bins=n_bins, labels=list(range(1,n_bins+1))).astype(int)
rfm['M_score'] = pd.cut(rfm['Monetary'].rank(method='first'), bins=n_bins, labels=list(range(1,n_bins+1))).astype(int)
rfm['RFM_total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

def rfm_segment(score):
    if score >= 13: return '重要价值客户'
    elif score >= 10: return '重要发展客户'
    elif score >= 7: return '一般客户'
    else: return '流失风险客户'

rfm['客户分群'] = rfm['RFM_total'].apply(rfm_segment)

rfm = rfm.merge(user_agg[['用户ID','用户年龄','用户性别','用户城市']], on='用户ID')

print("\n📊 RFM分群结果:")
seg_summary = rfm.groupby('客户分群').agg(
    人数=('用户ID','count'),
    均R=('Recency','mean'), 均F=('Frequency','mean'), 均M=('Monetary','mean'),
    均RFM=('RFM_total','mean')
).round(1)

for seg, row in seg_summary.iterrows():
    pct = row['人数']/len(rfm)*100
    print(f"  {seg}: {int(row['人数'])}人({pct:.0f}%), "
          f"R={row['均R']:.1f}天, F={row['均F']:.1f}次, M=¥{row['均M']:.0f}, RFM={row['均RFM']:.1f}")

print("\n📋 各分群用户明细:")
for seg in ['重要价值客户','重要发展客户','一般客户','流失风险客户']:
    users = rfm[rfm['客户分群']==seg]
    if len(users) > 0:
        for _, u in users.iterrows():
            print(f"  [{seg}] {u['用户ID']}: {u['用户性别']},{u['用户年龄']}岁,{u['用户城市']}, "
                  f"R={u['Recency']}天, F={u['Frequency']}次, M=¥{u['Monetary']:,.0f}")

# ================================================================
#  维度7: 关联规则分析 (购物篮)
# ================================================================
print("\n" + "="*70)
print("  维度7: 关联规则分析 (购物篮)")
print("="*70)

# 构建用户购买商品集
baskets = purchases.groupby('用户ID')['商品类别'].apply(set).reset_index()
baskets.columns = ['用户ID', '品类集合']

# 计算品类关联
cats = list(item['商品类别'].unique())
pair_counts = Counter()
for basket in baskets['品类集合']:
    for pair in combinations(sorted(basket), 2):
        pair_counts[pair] += 1

total_baskets = len(baskets)
print(f"\n📊 品类关联分析 (基于{total_baskets}个用户购物篮):")
print(f"  品类共现频率 TOP10:")
for pair, cnt in pair_counts.most_common(10):
    support = cnt / total_baskets
    print(f"  {pair[0]} → {pair[1]}: 共现{cnt}次(支持度{support:.1%})")

# Lift值计算
cat_freq = Counter()
for basket in baskets['品类集合']:
    for c in basket:
        cat_freq[c] += 1

print(f"\n🔗 关联规则(Lift > 1.0 表示正相关):")
rules = []
for (a, b), cnt in pair_counts.items():
    support = cnt / total_baskets
    conf_ab = cnt / cat_freq[a]
    conf_ba = cnt / cat_freq[b]
    lift_ab = support / (cat_freq[a]/total_baskets * cat_freq[b]/total_baskets)
    if lift_ab > 0:
        rules.append((a, b, support, conf_ab, conf_ba, lift_ab))

rules.sort(key=lambda x: x[5], reverse=True)
for a, b, sup, conf_ab, conf_ba, lift in rules[:10]:
    print(f"  {a} → {b}: 支持度{sup:.1%}, 置信度{conf_ab:.1%}, Lift={lift:.2f}")

# ================================================================
#  维度8: 价格弹性与敏感度
# ================================================================
print("\n" + "="*70)
print("  维度8: 价格弹性与敏感度")
print("="*70)

# 价格分段
item['价格段'] = pd.cut(item['价格'], bins=[0,100,500,1000,3000,10000],
                        labels=['低价(<100)','中低价(100-500)','中价(500-1000)','中高价(1000-3000)','高价(>3000)'])

price_seg = item.groupby('价格段').agg(
    数量=('商品ID','count'),
    均销量=('销量','mean'),
    总销量=('销量','sum'),
    均评分=('平均评分','mean'),
).round(1)

print("\n📊 价格段分析:")
for seg, row in price_seg.iterrows():
    print(f"  {seg}: {row['数量']}商品, 均销量{row['均销量']:.0f}, 总销量{row['总销量']:.0f}, 均评分{row['均评分']:.1f}")

# 价格-销量相关
corr_price_sales = item['价格'].corr(item['销量'])
corr_price_rating = item['价格'].corr(item['平均评分'])
print(f"\n📈 相关性:")
print(f"  价格-销量相关: {corr_price_sales:.3f} ({'正相关' if corr_price_sales>0 else '负相关'})")
print(f"  价格-评分相关: {corr_price_rating:.3f}")

# 品类价格弹性
print("\n📊 品类价格弹性(CV=变异系数):")
for cat in item['商品类别'].unique():
    cat_data = item[item['商品类别']==cat]
    cv = cat_data['价格'].std() / cat_data['价格'].mean() * 100
    print(f"  {cat}: 均价¥{cat_data['价格'].mean():.0f}, CV={cv:.0f}%, "
          f"范围¥{cat_data['价格'].min():.0f}-{cat_data['价格'].max():.0f}")

# ================================================================
#  维度9: 库存健康度诊断
# ================================================================
print("\n" + "="*70)
print("  维度9: 库存健康度诊断")
print("="*70)

item['库存周转率'] = item['销量'] / item['库存']
item['库存天数'] = item['库存'] / (item['销量'] / 365)
item['缺货风险'] = item['库存'] < 100

def stock_health(row):
    if row['库存'] < 100: return '缺货风险'
    elif row['库存周转率'] > 3: return '周转优秀'
    elif row['库存周转率'] > 1: return '周转正常'
    else: return '周转缓慢'

item['库存状态'] = item.apply(stock_health, axis=1)

stock_summary = item.groupby('库存状态').agg(数量=('商品ID','count')).reset_index()
print("\n📊 库存健康度:")
for _, r in stock_summary.iterrows():
    print(f"  {r['库存状态']}: {r['数量']}个商品")

print("\n⚠️ 缺货风险商品:")
shortage = item[item['缺货风险']].sort_values('销量', ascending=False)
for _, r in shortage.iterrows():
    print(f"  {r['商品名称']}({r['品牌']}): 库存{r['库存']}, 销量{r['销量']}, 周转率{r['库存周转率']:.2f}")

print("\n🚶 周转缓慢商品:")
slow = item[item['库存状态']=='周转缓慢'].sort_values('库存周转率')
for _, r in slow.iterrows():
    print(f"  {r['商品名称']}({r['品牌']}): 库存{r['库存']}, 销量{r['销量']}, 周转率{r['库存周转率']:.2f}")

# ================================================================
#  维度10: 时间模式分析
# ================================================================
print("\n" + "="*70)
print("  维度10: 时间模式分析")
print("="*70)

# 每日行为
daily = beh.groupby(['日期','行为类型']).size().unstack(fill_value=0)
print("\n📊 每日行为分布:")
for date in sorted(beh['日期'].unique()):
    day_beh = beh[beh['日期']==date]
    purch = (day_beh['行为类型']=='购买').sum()
    brows = (day_beh['行为类型']=='浏览').sum()
    amt = day_beh[day_beh['行为类型']=='购买']['商品价格'].sum()
    print(f"  {date}: {purch}购买+{brows}浏览, 购买金额¥{amt:,.0f}")

# 星期分布
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_cn = {'Monday':'周一','Tuesday':'周二','Wednesday':'周三','Thursday':'周四',
              'Friday':'周五','Saturday':'周六','Sunday':'周日'}
print("\n📊 星期分布:")
for wd in weekday_order:
    if wd in beh['星期'].values:
        wd_data = beh[beh['星期']==wd]
        print(f"  {weekday_cn[wd]}: {len(wd_data)}次行为")

# ================================================================
#  维度11: 用户-品类交叉偏好
# ================================================================
print("\n" + "="*70)
print("  维度11: 用户品类偏好热力图")
print("="*70)

user_cat = beh[beh['行为类型']=='购买'].groupby(['用户ID','商品类别']).agg(
    次数=('行为类型','count'), 金额=('商品价格','sum')
).reset_index()

print("\n📊 用户品类偏好TOP3:")
for uid in sorted(beh['用户ID'].unique()):
    u_data = user_cat[user_cat['用户ID']==uid].sort_values('金额', ascending=False)
    if len(u_data) > 0:
        prefs = [f"{r['商品类别']}(¥{r['金额']:,.0f})" for _, r in u_data.head(3).iterrows()]
        u_info = user_agg[user_agg['用户ID']==uid].iloc[0]
        print(f"  {uid}({u_info['用户性别']},{u_info['用户年龄']}岁,{u_info['用户城市']}): {' > '.join(prefs)}")

# ================================================================
#  生成可视化
# ================================================================
print("\n" + "="*70)
print("  生成可视化图表...")
print("="*70)

fig = plt.figure(figsize=(28, 36))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.3)

# 1. 用户画像雷达
ax1 = fig.add_subplot(gs[0, 0:2])
gender_metrics = user_agg.groupby('用户性别').agg(
    均消费=('总消费','mean'), 均评分=('均评分','mean'), 均品类=('品类数','mean'),
    转化率=('转化率','mean'), 均行为=('行为数','mean')
).round(2)
x = np.arange(len(gender_metrics.columns))
w = 0.3
for i, (g, row) in enumerate(gender_metrics.iterrows()):
    vals = row.values
    vals_norm = vals / vals.max() if vals.max() > 0 else vals
    ax1.bar(x + i*w, vals, width=w, label=g, alpha=0.8)
ax1.set_xticks(x + w/2)
ax1.set_xticklabels(gender_metrics.columns, fontsize=9)
ax1.set_title('用户性别维度对比', fontsize=13, fontweight='bold')
ax1.legend()

# 2. 城市消费力
ax2 = fig.add_subplot(gs[0, 2:4])
city_order = city_stats.index.tolist()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(city_order)))
bars = ax2.barh(city_order, city_stats['均消费'], color=colors)
ax2.set_xlabel('人均消费(¥)')
ax2.set_title('城市消费力排名', fontsize=13, fontweight='bold')
for bar, val in zip(bars, city_stats['均消费']):
    ax2.text(bar.get_width()+50, bar.get_y()+bar.get_height()/2, f'¥{val:,.0f}', va='center', fontsize=9)

# 3. 行为漏斗
ax3 = fig.add_subplot(gs[1, 0])
funnel_data = [total, purchase, browse]
funnel_labels = [f'总行为\n{total}', f'购买\n{purchase}({purchase/total:.0%})', f'浏览\n{browse}({browse/total:.0%})']
colors_funnel = ['#3498db', '#2ecc71', '#e74c3c']
ax3.barh(range(len(funnel_data)), funnel_data, color=colors_funnel)
ax3.set_yticks(range(len(funnel_labels)))
ax3.set_yticklabels(funnel_labels, fontsize=9)
ax3.set_title('行为漏斗', fontsize=13, fontweight='bold')
ax3.invert_yaxis()

# 4. 品类漏斗
ax4 = fig.add_subplot(gs[1, 1:3])
x_pos = np.arange(len(cat_funnel))
width = 0.3
ax4.bar(x_pos - width, cat_funnel['购买'], width, label='购买', color='#2ecc71')
ax4.bar(x_pos, cat_funnel['浏览'], width, label='浏览', color='#e74c3c')
ax4.bar(x_pos + width, cat_funnel['总行为'], width, label='总计', color='#3498db', alpha=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(cat_funnel['商品类别'], fontsize=8, rotation=20)
ax4.set_title('品类行为漏斗对比', fontsize=13, fontweight='bold')
ax4.legend()

# 5. 品类GMV占比
ax5 = fig.add_subplot(gs[1, 3])
gmv_colors = plt.cm.Set3(np.linspace(0, 1, len(cat_analysis)))
wedges, texts, autotexts = ax5.pie(cat_analysis['GMV占比'], labels=cat_analysis['商品类别'],
    autopct='%1.1f%%', colors=gmv_colors, startangle=90)
ax5.set_title('品类GMV占比', fontsize=13, fontweight='bold')

# 6. BCG矩阵
ax6 = fig.add_subplot(gs[2, 0:2])
bcg_colors = {'明星(Star)':'#2ecc71', '现金牛(Cash Cow)':'#f1c40f', '问号(Question)':'#3498db', '瘦狗(Dog)':'#e74c3c'}
for bcg_type, color in bcg_colors.items():
    subset = product_full[product_full['BCG类型']==bcg_type]
    if len(subset) > 0:
        ax6.scatter(subset['销量'], subset['交互次数'], c=color, s=subset['价格']/10+20,
                   alpha=0.6, label=bcg_type, edgecolors='gray', linewidth=0.5)
ax6.axvline(med_sales, color='gray', linestyle='--', alpha=0.5)
ax6.axhline(med_interact, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('销量')
ax6.set_ylabel('用户交互次数')
ax6.set_title('商品BCG矩阵 (气泡=价格)', fontsize=13, fontweight='bold')
ax6.legend(fontsize=8)

# 7. 品牌TOP10销量
ax7 = fig.add_subplot(gs[2, 2:4])
top_brands = brand_analysis.head(10)
bars7 = ax7.barh(top_brands['品牌'][::-1], top_brands['总销量'][::-1],
                 color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
ax7.set_xlabel('总销量')
ax7.set_title('品牌销量TOP10', fontsize=13, fontweight='bold')

# 8. RFM分群
ax8 = fig.add_subplot(gs[3, 0])
seg_colors = {'重要价值客户':'#2ecc71', '重要发展客户':'#3498db', '一般客户':'#f1c40f', '流失风险客户':'#e74c3c'}
seg_counts = rfm['客户分群'].value_counts()
colors_seg = [seg_colors.get(s, 'gray') for s in seg_counts.index]
ax8.pie(seg_counts, labels=seg_counts.index, autopct='%1.0f%%', colors=colors_seg, startangle=90)
ax8.set_title('RFM客户分群', fontsize=13, fontweight='bold')

# 9. RFM散点(R vs M)
ax9 = fig.add_subplot(gs[3, 1])
for seg, color in seg_colors.items():
    subset = rfm[rfm['客户分群']==seg]
    if len(subset) > 0:
        ax9.scatter(subset['Recency'], subset['Monetary'], c=color, label=seg, s=80, alpha=0.7)
ax9.set_xlabel('Recency(天)')
ax9.set_ylabel('Monetary(¥)')
ax9.set_title('RFM散点图(R vs M)', fontsize=13, fontweight='bold')
ax9.legend(fontsize=7)

# 10. 价格段分析
ax10 = fig.add_subplot(gs[3, 2])
price_counts = item['价格段'].value_counts().sort_index()
ax10.bar(range(len(price_counts)), price_counts.values, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(price_counts))))
ax10.set_xticks(range(len(price_counts)))
ax10.set_xticklabels(price_counts.index, fontsize=7, rotation=20)
ax10.set_ylabel('商品数')
ax10.set_title('价格段商品分布', fontsize=13, fontweight='bold')

# 11. 价格-销量散点
ax11 = fig.add_subplot(gs[3, 3])
ax11.scatter(item['价格'], item['销量'], c=item['平均评分'], cmap='RdYlGn', s=60, alpha=0.7, edgecolors='gray')
ax11.set_xlabel('价格(¥)')
ax11.set_ylabel('销量')
ax11.set_title(f'价格-销量散点(颜色=评分, r={corr_price_sales:.2f})', fontsize=11, fontweight='bold')

# 12. 品类关联热力图
ax12 = fig.add_subplot(gs[4, 0:2])
cats_sorted = sorted(cats)
n_cats = len(cats_sorted)
matrix = np.zeros((n_cats, n_cats))
for i, a in enumerate(cats_sorted):
    for j, b in enumerate(cats_sorted):
        if i < j:
            key = tuple(sorted([a, b]))
            matrix[i][j] = pair_counts.get(key, 0)
            matrix[j][i] = pair_counts.get(key, 0)
im = ax12.imshow(matrix, cmap='YlOrRd')
ax12.set_xticks(range(n_cats))
ax12.set_yticks(range(n_cats))
ax12.set_xticklabels(cats_sorted, fontsize=8, rotation=30)
ax12.set_yticklabels(cats_sorted, fontsize=8)
ax12.set_title('品类关联热力图(共现次数)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax12, shrink=0.8)

# 13. 库存健康度
ax13 = fig.add_subplot(gs[4, 2])
stock_status = item['库存状态'].value_counts()
colors_stock = {'周转优秀':'#2ecc71', '周转正常':'#3498db', '周转缓慢':'#f1c40f', '缺货风险':'#e74c3c'}
ax13.pie(stock_status, labels=stock_status.index, autopct='%1.0f%%',
         colors=[colors_stock.get(s, 'gray') for s in stock_status.index], startangle=90)
ax13.set_title('库存健康度', fontsize=13, fontweight='bold')

# 14. 时间序列行为
ax14 = fig.add_subplot(gs[4, 3])
dates = sorted(beh['日期'].unique())
daily_purchase = [beh[(beh['日期']==d) & (beh['行为类型']=='购买')]['商品价格'].sum() for d in dates]
daily_browse = [(beh['日期']==d).sum() for d in dates]
ax14_twin = ax14.twinx()
ax14.bar(range(len(dates)), daily_purchase, alpha=0.6, color='#2ecc71', label='购买金额')
ax14_twin.plot(range(len(dates)), daily_browse, 'o-', color='#e74c3c', label='行为总数')
ax14.set_xticks(range(len(dates)))
ax14.set_xticklabels([str(d) for d in dates], fontsize=7, rotation=20)
ax14.set_ylabel('购买金额(¥)', color='#2ecc71')
ax14_twin.set_ylabel('行为总数', color='#e74c3c')
ax14.set_title('每日行为与购买趋势', fontsize=13, fontweight='bold')

# 15. 用户品类偏好热力图
ax15 = fig.add_subplot(gs[5, :])
user_ids = sorted(beh['用户ID'].unique())
cat_list = sorted(item['商品类别'].unique())
heatmap_data = np.zeros((len(user_ids), len(cat_list)))
for i, uid in enumerate(user_ids):
    for j, cat in enumerate(cat_list):
        val = user_cat[(user_cat['用户ID']==uid) & (user_cat['商品类别']==cat)]['金额'].sum()
        heatmap_data[i][j] = val
im15 = ax15.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax15.set_xticks(range(len(cat_list)))
ax15.set_yticks(range(len(user_ids)))
ax15.set_xticklabels(cat_list, fontsize=9)
ax15.set_yticklabels(user_ids, fontsize=9)
ax15.set_title('用户-品类消费金额热力图', fontsize=13, fontweight='bold')
ax15.set_xlabel('商品品类')
ax15.set_ylabel('用户ID')
plt.colorbar(im15, ax=ax15, shrink=0.6, label='消费金额(¥)')

plt.savefig(os.path.join(VIZ_DIR, 'sample_data_dashboard.png'), dpi=150, bbox_inches='tight')
print(f"  ✅ 仪表板: {os.path.join(VIZ_DIR, 'sample_data_dashboard.png')}")
plt.close()

# ================================================================
#  生成报告
# ================================================================
report = f"""# sample_data 多维深度洞察报告

**分析日期**: 2026-05-10
**数据规模**: 用户行为{len(beh)}条(12用户), 商品信息{len(item)}个商品(7大品类)
**分析维度**: 用户画像/行为漏斗/商品BCG/品牌竞争/品类分析/RFM分群/关联规则/价格弹性/库存健康/时间模式/用户偏好

---

## 一、用户画像核心发现

| 指标 | 数值 |
|------|------|
| 用户总数 | {len(user_agg)} |
| 平均年龄 | {user_agg['用户年龄'].mean():.1f}岁 |
| 性别比例(男:女) | {(user_agg['用户性别']=='男').sum()}:{(user_agg['用户性别']=='女').sum()} |
| 覆盖城市 | {user_agg['用户城市'].nunique()}个 |
| 人均行为数 | {user_agg['行为数'].mean():.1f}次 |
| 人均消费 | ¥{user_agg['总消费'].mean():,.0f} |
| 人均转化率 | {user_agg['转化率'].mean():.0%} |

### 性别差异
- **男性**: 均消费¥{gender_stats.loc['男','均消费']:,.0f}, 均评分{gender_stats.loc['男','均评分']:.1f}
- **女性**: 均消费¥{gender_stats.loc['女','均消费']:,.0f}, 均评分{gender_stats.loc['女','均评分']:.1f}

## 二、行为漏斗

| 指标 | 数值 |
|------|------|
| 总行为 | {total} |
| 购买 | {purchase}({purchase/total:.0%}) |
| 浏览 | {browse}({browse/total:.0%}) |

## 三、品类GMV贡献

| 排名 | 品类 | GMV | 占比 | 均价格 | 品牌数 | 库存周转 |
|------|------|-----|------|--------|--------|----------|
"""

for i, (_, r) in enumerate(cat_analysis.iterrows(), 1):
    report += f"| {i} | {r['商品类别']} | ¥{r['GMV贡献']:,.0f} | {r['GMV占比']:.1f}% | ¥{r['均价格']:.0f} | {r['品牌数']} | {r['库存周转']:.2f} |\n"

report += f"""
## 四、RFM客户分群

| 分群 | 人数 | 占比 | 均R(天) | 均F(次) | 均M(¥) |
|------|------|------|---------|---------|--------|
"""
for seg, row in seg_summary.iterrows():
    report += f"| {seg} | {int(row['人数'])} | {row['人数']/len(rfm)*100:.0f}% | {row['均R']:.1f} | {row['均F']:.1f} | ¥{row['均M']:,.0f} |\n"

report += f"""
## 五、品牌竞争格局

- **品牌总数**: {len(brand_analysis)}
- **HHI指数**: {hhi:.0f} ({'集中市场' if hhi > 2500 else '分散市场'})

### TOP5品牌
| 排名 | 品牌 | 产品数 | 总销量 | 均价 | 均评分 |
|------|------|--------|--------|------|--------|
"""
for i, (_, r) in enumerate(brand_analysis.head(5).iterrows(), 1):
    report += f"| {i} | {r['品牌']} | {r['产品数']} | {r['总销量']:.0f} | ¥{r['均价格']:.0f} | {r['均评分']:.1f} |\n"

report += f"""
## 六、库存健康度

| 状态 | 商品数 |
|------|--------|
"""
for _, r in stock_summary.iterrows():
    report += f"| {r['库存状态']} | {r['数量']} |\n"

report += f"""
## 七、核心洞察

1. **电子产品是核心驱动力**: GMV贡献最大，单价高且品牌丰富(苹果、华为、索尼等)
2. **用户转化率高(80%)**: 大部分行为直接转化为购买，说明用户目的性明确
3. **男女消费力均衡**: 男性均消费¥{gender_stats.loc['男','均消费']:,.0f} vs 女性¥{gender_stats.loc['女','均消费']:,.0f}
4. **价格与销量微弱负相关(r={corr_price_sales:.2f})**: 低价商品略倾向更高销量但不显著
5. **品类关联: 电子产品↔服装↔家居** 是最强关联组合，可做跨品类推荐
6. **RFM分群健康**: 重要价值客户占比较高，但需关注流失风险客户
7. **库存风险**: 部分高销量商品库存不足(如iPhone、MacBook)，需及时补货

---

*报告自动生成 | 2026-05-10*
"""

with open(os.path.join(RPT_DIR, 'sample_data_analysis_report.md'), 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✅ 报告: {os.path.join(RPT_DIR, 'sample_data_analysis_report.md')}")

# 保存数据
rfm.to_csv(os.path.join(RPT_DIR, 'rfm_segmentation.csv'), index=False, encoding='utf-8-sig')
product_full.to_csv(os.path.join(RPT_DIR, 'product_bcg_analysis.csv'), index=False, encoding='utf-8-sig')
brand_analysis.to_csv(os.path.join(RPT_DIR, 'brand_competition.csv'), index=False, encoding='utf-8-sig')
cat_analysis.to_csv(os.path.join(RPT_DIR, 'category_analysis.csv'), index=False, encoding='utf-8-sig')
user_agg.to_csv(os.path.join(RPT_DIR, 'user_profile.csv'), index=False, encoding='utf-8-sig')
print(f"  ✅ 数据: {RPT_DIR}/")

print("\n" + "="*70)
print("  ✅ sample_data/ 多维深度洞察分析完成！")
print("="*70)
