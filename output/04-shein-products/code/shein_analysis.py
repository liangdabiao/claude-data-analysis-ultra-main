"""
SHEIN 产品目录全面深度分析
维度: 定价策略/折扣/品类/品牌/颜色/尺码/属性/价格带/视觉资产
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, ast, re, os, warnings
from collections import Counter, defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT_R = './output/04-shein-products/reports/'
OUT_V = './output/04-shein-products/visualizations/'
os.makedirs(OUT_R, exist_ok=True)
os.makedirs(OUT_V, exist_ok=True)

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 70)
print("  SHEIN 产品目录全面深度分析")
print("=" * 70)

df = pd.read_csv('eCommerce-dataset-samples/shein-products.csv')

# 折扣计算
df['discount_pct'] = ((df['initial_price'] - df['final_price']) / df['initial_price'] * 100).round(2)
df['has_discount'] = df['discount_pct'] > 0
df['discount_amount'] = df['initial_price'] - df['final_price']

# 价格带
def price_band(p):
    if p < 5: return '超低价($0-5)'
    elif p < 10: return '低价($5-10)'
    elif p < 20: return '中价($10-20)'
    elif p < 50: return '中高价($20-50)'
    elif p < 100: return '高价($50-100)'
    else: return '超高价($100+)'

df['price_band'] = df['final_price'].apply(price_band)

# 解析 other_attributes
def parse_attributes(attr_str):
    try:
        attrs = ast.literal_eval(attr_str)
        return {a['name']: a['value'] for a in attrs}
    except:
        return {}

df['attr_dict'] = df['other_attributes'].apply(parse_attributes)

# 材质提取
def extract_material(attr_dict):
    return attr_dict.get('Material', attr_dict.get('Fabric', attr_dict.get('Composition', '')))

df['material'] = df['attr_dict'].apply(extract_material)

print(f"\n📊 数据概览:")
print(f"  产品总数: {len(df)}")
print(f"  品牌数: {df['brand'].nunique()}")
print(f"  根品类数: {df['root_category'].nunique()}")
print(f"  子品类数: {df['category'].nunique()}")
print(f"  颜色种类: {df['color'].nunique()}")
print(f"  价格范围: ${df['final_price'].min():.2f} ~ ${df['final_price'].max():.2f}")
print(f"  平均售价: ${df['final_price'].mean():.2f}")
print(f"  中位售价: ${df['final_price'].median():.2f}")

# ============================================================
# 2. 定价策略与折扣分析
# ============================================================
print("\n" + "=" * 70)
print("  定价策略与折扣分析")
print("=" * 70)

discounted = df[df['has_discount']]
non_discounted = df[~df['has_discount']]

print(f"\n💸 折扣概况:")
print(f"  有折扣产品: {len(discounted)} ({len(discounted)/len(df)*100:.1f}%)")
print(f"  无折扣产品: {len(non_discounted)} ({len(non_discounted)/len(df)*100:.1f}%)")
print(f"  平均折扣幅度: {discounted['discount_pct'].mean():.1f}%")
print(f"  最大折扣幅度: {discounted['discount_pct'].max():.1f}%")
print(f"  中位折扣幅度: {discounted['discount_pct'].median():.1f}%")

# 折扣分档
def discount_level(pct):
    if pct == 0: return '无折扣'
    elif pct < 10: return '轻度(0-10%)'
    elif pct < 20: return '中度(10-20%)'
    elif pct < 30: return '重度(20-30%)'
    elif pct < 50: return '深度(30-50%)'
    else: return '极端(50%+)'

df['discount_level'] = df['discount_pct'].apply(discount_level)
disc_dist = df['discount_level'].value_counts()

print(f"\n📈 折扣力度分布:")
for level in ['无折扣','轻度(0-10%)','中度(10-20%)','重度(20-30%)','深度(30-50%)','极端(50%+)']:
    cnt = disc_dist.get(level, 0)
    print(f"  {level}: {cnt} ({cnt/len(df)*100:.1f}%)")

# 品牌折扣策略
print(f"\n🏢 品牌折扣策略对比:")
brand_disc = df.groupby('brand').agg(
    产品数=('product_id','count'),
    有折扣数=('has_discount','sum'),
    平均折扣率=('discount_pct','mean'),
    最大折扣率=('discount_pct','max'),
    平均原价=('initial_price','mean'),
    平均售价=('final_price','mean'),
).sort_values('产品数', ascending=False)
brand_disc['折扣比例'] = (brand_disc['有折扣数'] / brand_disc['产品数'] * 100).round(1)

for brand, row in brand_disc.iterrows():
    print(f"  {brand}: {row['产品数']:.0f}产品, 折扣率{row['折扣比例']:.0f}%, "
          f"均折{row['平均折扣率']:.1f}%, 原价${row['平均原价']:.2f}→售价${row['平均售价']:.2f}")

# 品类折扣对比
print(f"\n🏷️ 品类折扣分析 (Top 10 品类):")
cat_disc = df.groupby('root_category').agg(
    产品数=('product_id','count'),
    平均折扣率=('discount_pct','mean'),
    折扣比例=('has_discount','mean'),
    平均售价=('final_price','mean'),
).sort_values('产品数', ascending=False)

for cat, row in cat_disc.head(10).iterrows():
    print(f"  {cat}: {row['产品数']:.0f}产品, 折扣率{row['折扣比例']*100:.0f}%, "
          f"均折{row['平均折扣率']:.1f}%, 均价${row['平均售价']:.2f}")

# ============================================================
# 3. 品类深度分析
# ============================================================
print("\n" + "=" * 70)
print("  品类深度分析")
print("=" * 70)

root_cat = df.groupby('root_category').agg(
    产品数=('product_id','count'),
    平均售价=('final_price','mean'),
    中位售价=('final_price','median'),
    最高价=('final_price','max'),
    最低价=('final_price','min'),
    子品类数=('category','nunique'),
    平均图片数=('image_count','mean'),
    平均折扣=('discount_pct','mean'),
).sort_values('产品数', ascending=False)

print(f"\n📂 根品类分析 (共{df['root_category'].nunique()}个):")
for cat, row in root_cat.iterrows():
    pct = row['产品数'] / len(df) * 100
    print(f"  {cat}: {row['产品数']:.0f}产品 ({pct:.1f}%), "
          f"均价${row['平均售价']:.2f}, {row['子品类数']:.0f}子品类, 均折{row['平均折扣']:.1f}%")

# 子品类 Top 15
sub_cat = df.groupby('category').agg(
    产品数=('product_id','count'),
    平均售价=('final_price','mean'),
    平均折扣=('discount_pct','mean'),
).sort_values('产品数', ascending=False)

print(f"\n📂 子品类 Top 15 (共{df['category'].nunique()}个):")
for cat, row in sub_cat.head(15).iterrows():
    print(f"  {cat}: {row['产品数']:.0f}产品, 均价${row['平均售价']:.2f}, 均折{row['平均折扣']:.1f}%")

# 品类价格差异
print(f"\n💰 品类价格区间:")
for cat in ['Home & Living','Jewelry & Watches','Bags & Luggage','Beauty & Health','Sports & Outdoor']:
    cat_data = df[df['root_category']==cat]['final_price']
    print(f"  {cat}: ${cat_data.min():.2f} ~ ${cat_data.max():.2f} "
          f"(IQR: ${cat_data.quantile(0.25):.2f} ~ ${cat_data.quantile(0.75):.2f})")

# ============================================================
# 4. 品牌竞争格局
# ============================================================
print("\n" + "=" * 70)
print("  品牌竞争格局")
print("=" * 70)

brand_stats = df.groupby('brand').agg(
    产品数=('product_id','count'),
    平均售价=('final_price','mean'),
    中位售价=('final_price','median'),
    价格范围=('final_price', lambda x: f"${x.min():.2f}~${x.max():.2f}"),
    品类数=('root_category','nunique'),
    子品类数=('category','nunique'),
    颜色数=('color','nunique'),
    平均图片=('image_count','mean'),
    平均折扣=('discount_pct','mean'),
).sort_values('产品数', ascending=False)

print(f"\n🏢 品牌对比:")
for brand, row in brand_stats.iterrows():
    pct = row['产品数'] / len(df) * 100
    print(f"  {brand}: {row['产品数']:.0f}产品 ({pct:.1f}%), "
          f"均价${row['平均售价']:.2f}, {row['品类数']:.0f}品类, "
          f"{row['子品类数']:.0f}子品类, 均折{row['平均折扣']:.1f}%")

# 品牌品类分布
print(f"\n📊 品牌品类分布:")
for brand in ['SHEIN','Unbeatablesale','Jepeak','ROMWE']:
    brand_data = df[df['brand']==brand]
    top_cats = brand_data['root_category'].value_counts().head(3)
    cat_str = ", ".join([f"{c}({n})" for c, n in top_cats.items()])
    print(f"  {brand}: {cat_str}")

# 品牌价格定位
print(f"\n🏷️ 品牌价格定位:")
for brand in ['SHEIN','Unbeatablesale','Jepeak','ROMWE','Nike']:
    bd = df[df['brand']==brand]['final_price']
    if len(bd) > 0:
        bands = bd.apply(price_band).value_counts()
        top_band = bands.index[0]
        print(f"  {brand}: 主力价格带={top_band} ({bands.iloc[0]}/{len(bd)})")

# ============================================================
# 5. 颜色趋势分析
# ============================================================
print("\n" + "=" * 70)
print("  颜色趋势分析")
print("=" * 70)

color_stats = df.groupby('color').agg(
    产品数=('product_id','count'),
    平均售价=('final_price','mean'),
    平均折扣=('discount_pct','mean'),
).sort_values('产品数', ascending=False)

print(f"\n🎨 颜色分布 (共{df['color'].nunique()}种颜色):")
for color, row in color_stats.head(15).iterrows():
    pct = row['产品数'] / len(df) * 100
    print(f"  {color}: {row['产品数']:.0f}产品 ({pct:.1f}%), 均价${row['平均售价']:.2f}")

# 颜色 x 品类交叉
print(f"\n🎨 品类颜色偏好:")
top_colors = ['Black','White','Multicolor','Red','Pink']
for cat in ['Home & Living','Jewelry & Watches','Bags & Luggage','Beauty & Health']:
    cat_data = df[df['root_category']==cat]
    top_c = cat_data['color'].value_counts().head(3)
    print(f"  {cat}: {', '.join([f'{c}({n})' for c, n in top_c.items()])}")

# 颜色与价格
print(f"\n💰 颜色溢价分析:")
for color in ['Black','White','Gold','Silver','Red']:
    color_data = df[df['color']==color]['final_price']
    if len(color_data) > 0:
        print(f"  {color}: 均价${color_data.mean():.2f}, 中位${color_data.median():.2f}")

# ============================================================
# 6. 尺码策略分析
# ============================================================
print("\n" + "=" * 70)
print("  尺码策略分析")
print("=" * 70)

# 尺码类型分类
def classify_size(s):
    if s == 'one-size': return '均码'
    elif any(c.isdigit() for c in str(s)):
        if 'cm' in str(s) or 'inch' in str(s): return '尺寸规格'
        elif 'pcs' in str(s) or 'PC' in str(s): return '数量装'
        else: return '其他规格'
    else: return '字母尺码'

df['size_type'] = df['size'].apply(classify_size)
size_dist = df['size_type'].value_counts()

print(f"\n📏 尺码类型分布:")
for st, cnt in size_dist.items():
    print(f"  {st}: {cnt} ({cnt/len(df)*100:.1f}%)")

# 尺码与价格
print(f"\n📏 尺码与价格关系:")
for st in size_dist.index:
    st_data = df[df['size_type']==st]['final_price']
    print(f"  {st}: 均价${st_data.mean():.2f}, 中位${st_data.median():.2f}")

# 解析 all_available_sizes
def count_sizes(sizes_str):
    try:
        sizes = ast.literal_eval(sizes_str)
        return len(sizes)
    except:
        return 1

df['size_count'] = df['all_available_sizes'].apply(count_sizes)
print(f"\n📏 可用尺码数量分布:")
size_count_dist = df['size_count'].value_counts().head(10)
for sc, cnt in size_count_dist.items():
    print(f"  {sc}个尺码: {cnt}产品 ({cnt/len(df)*100:.1f}%)")

# ============================================================
# 7. 产品属性深度挖掘
# ============================================================
print("\n" + "=" * 70)
print("  产品属性深度挖掘")
print("=" * 70)

# 材质分析
material_counter = Counter()
for attr in df['attr_dict']:
    for key in ['Material', 'Fabric', 'Composition']:
        if key in attr:
            material_counter[attr[key]] += 1

print(f"\n🧵 材质分布 (Top 15):")
for mat, cnt in material_counter.most_common(15):
    print(f"  {mat}: {cnt}产品")

# 属性名称频率
attr_name_counter = Counter()
for attr in df['attr_dict']:
    for name in attr.keys():
        attr_name_counter[name] += 1

print(f"\n📋 产品属性频率 (Top 15):")
for name, cnt in attr_name_counter.most_common(15):
    print(f"  {name}: {cnt}产品 ({cnt/len(df)*100:.1f}%)")

# Style 属性分析
style_counter = Counter()
for attr in df['attr_dict']:
    if 'Style' in attr:
        style_counter[attr['Style']] += 1

if style_counter:
    print(f"\n✨ 风格分布 (Top 10):")
    for style, cnt in style_counter.most_common(10):
        print(f"  {style}: {cnt}产品")

# ============================================================
# 8. 价格带分析
# ============================================================
print("\n" + "=" * 70)
print("  价格带分析")
print("=" * 70)

band_order = ['超低价($0-5)','低价($5-10)','中价($10-20)','中高价($20-50)','高价($50-100)','超高价($100+)']
band_stats = df.groupby('price_band').agg(
    产品数=('product_id','count'),
    平均售价=('final_price','mean'),
    平均折扣=('discount_pct','mean'),
    折扣比例=('has_discount','mean'),
).reindex(band_order)

print(f"\n💰 价格带分析:")
total_rev_est = (df['final_price'] * 100).sum()  # 估计总收入(假设各100销量)
for band in band_order:
    row = band_stats.loc[band]
    pct = row['产品数'] / len(df) * 100
    print(f"  {band}: {row['产品数']:.0f}产品 ({pct:.1f}%), "
          f"均价${row['平均售价']:.2f}, 折扣率{row['折扣比例']*100:.0f}%")

# 品牌价格带交叉
print(f"\n🏢 品牌价格带分布:")
for brand in ['SHEIN','Unbeatablesale']:
    brand_data = df[df['brand']==brand]
    bands = brand_data['price_band'].value_counts().reindex(band_order).fillna(0)
    top3 = bands.nlargest(3)
    print(f"  {brand}: {', '.join([f'{b}({int(c)})' for b, c in top3.items()])}")

# ============================================================
# 9. 视觉资产分析
# ============================================================
print("\n" + "=" * 70)
print("  视觉资产(image_count)分析")
print("=" * 70)

print(f"\n🖼️ 图片数量统计:")
print(f"  平均图片数: {df['image_count'].mean():.1f}")
print(f"  中位图片数: {df['image_count'].median():.0f}")
print(f"  最少: {df['image_count'].min()}")
print(f"  最多: {df['image_count'].max()}")

# 图片数分档
def img_level(cnt):
    if cnt <= 3: return '较少(≤3)'
    elif cnt <= 8: return '标准(4-8)'
    elif cnt <= 15: return '丰富(9-15)'
    else: return '极丰富(16+)'

df['img_level'] = df['image_count'].apply(img_level)
img_dist = df['img_level'].value_counts()
print(f"\n  图片数量分档:")
for level in ['较少(≤3)','标准(4-8)','丰富(9-15)','极丰富(16+)']:
    cnt = img_dist.get(level, 0)
    print(f"    {level}: {cnt} ({cnt/len(df)*100:.1f}%)")

# 图片数 vs 价格
print(f"\n  图片数 vs 价格:")
for level in ['较少(≤3)','标准(4-8)','丰富(9-15)','极丰富(16+)']:
    level_data = df[df['img_level']==level]['final_price']
    if len(level_data) > 0:
        print(f"    {level}: 均价${level_data.mean():.2f}, 中位${level_data.median():.2f}")

# 品牌图片策略
print(f"\n  品牌图片策略:")
for brand in ['SHEIN','Unbeatablesale','Jepeak','ROMWE']:
    brand_img = df[df['brand']==brand]['image_count']
    if len(brand_img) > 0:
        print(f"    {brand}: 平均{brand_img.mean():.1f}张, 中位{brand_img.median():.0f}张")

# ============================================================
# 10. 产品名称关键词分析
# ============================================================
print("\n" + "=" * 70)
print("  产品名称关键词分析")
print("=" * 70)

# 提取关键词
stopwords = {'the','and','for','with','in','on','at','to','of','a','an','is','it',
             'from','by','as','or','1pc','2pc','3pc','set','free','shipping','returns'}

word_counter = Counter()
for name in df['product_name']:
    words = re.findall(r'[A-Za-z]+', str(name).lower())
    words = [w for w in words if w not in stopwords and len(w) > 2]
    word_counter.update(words)

print(f"\n🔤 产品名称高频词 (Top 30):")
for word, cnt in word_counter.most_common(30):
    print(f"  {word}: {cnt}")

# ============================================================
# 11. 综合可视化
# ============================================================
print("\n" + "=" * 70)
print("  生成可视化图表...")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(22, 20))
fig.suptitle('SHEIN 产品目录全面深度分析仪表板', fontsize=20, fontweight='bold', y=0.99)

# 1. 根品类产品数
ax = axes[0, 0]
top_root = root_cat.head(10)
ax.barh(top_root.index[::-1], top_root['产品数'].values[::-1],
        color=plt.cm.viridis(np.linspace(0.2,0.8,10)))
ax.set_title('根品类产品数 Top 10', fontweight='bold', fontsize=12)
ax.set_xlabel('产品数')

# 2. 价格分布
ax = axes[0, 1]
price_capped = df[df['final_price'] < df['final_price'].quantile(0.95)]['final_price']
ax.hist(price_capped, bins=40, color='#4ECDC4', edgecolor='white', alpha=0.8)
ax.axvline(df['final_price'].median(), color='red', linestyle='--',
           label=f"中位数${df['final_price'].median():.2f}")
ax.set_title('售价分布 (P95截断)', fontweight='bold', fontsize=12)
ax.set_xlabel('价格 ($)')
ax.legend()

# 3. 折扣力度分布
ax = axes[0, 2]
disc_levels = ['无折扣','轻度(0-10%)','中度(10-20%)','重度(20-30%)','深度(30-50%)','极端(50%+)']
disc_colors = ['#C8D6E5','#48DBFB','#45B7D1','#FECA57','#FF9F43','#FF6B6B']
disc_vals = [disc_dist.get(l, 0) for l in disc_levels]
ax.bar(disc_levels, disc_vals, color=disc_colors)
ax.set_title('折扣力度分布', fontweight='bold', fontsize=12)
ax.set_ylabel('产品数')
ax.tick_params(axis='x', rotation=25)

# 4. 品牌对比
ax = axes[1, 0]
brands = brand_stats.head(5).index.tolist()
brand_prices = [df[df['brand']==b]['final_price'].mean() for b in brands]
brand_counts = [brand_stats.loc[b, '产品数'] for b in brands]
ax.scatter(brand_counts, brand_prices, s=[c*3 for c in brand_counts],
           c=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'][:len(brands)],
           alpha=0.7, edgecolors='black')
for b, c, p in zip(brands, brand_counts, brand_prices):
    ax.annotate(b, (c, p), fontsize=9, ha='center', va='bottom')
ax.set_title('品牌: 产品数 vs 平均售价', fontweight='bold', fontsize=12)
ax.set_xlabel('产品数')
ax.set_ylabel('平均售价 ($)')

# 5. 颜色分布 Top 10
ax = axes[1, 1]
top_colors_df = color_stats.head(10)
color_map = {'Black':'#333333','White':'#EEEEEE','Red':'#FF0000','Pink':'#FF69B4',
             'Blue':'#0000FF','Green':'#00FF00','Gold':'#FFD700','Silver':'#C0C0C0',
             'Multicolor':'#FF6B6B','Beige':'#F5F5DC','Brown':'#8B4513','Grey':'#808080',
             'Purple':'#800080','Orange':'#FFA500','Yellow':'#FFFF00'}
bar_colors = [color_map.get(c, '#4ECDC4') for c in top_colors_df.index]
ax.bar(top_colors_df.index, top_colors_df['产品数'], color=bar_colors, edgecolor='gray')
ax.set_title('颜色分布 Top 10', fontweight='bold', fontsize=12)
ax.set_ylabel('产品数')
ax.tick_params(axis='x', rotation=30)

# 6. 价格带分布
ax = axes[1, 2]
band_vals = [band_stats.loc[b, '产品数'] for b in band_order]
band_colors_list = ['#4ECDC4','#45B7D1','#96CEB4','#FECA57','#FF9F43','#FF6B6B']
ax.bar(range(len(band_order)), band_vals, color=band_colors_list)
ax.set_xticks(range(len(band_order)))
ax.set_xticklabels([b.split('(')[0] for b in band_order], rotation=20)
ax.set_title('价格带产品分布', fontweight='bold', fontsize=12)
ax.set_ylabel('产品数')

# 7. 品牌折扣策略
ax = axes[2, 0]
top_brands = ['SHEIN','Unbeatablesale','Jepeak','ROMWE','Nike']
brand_disc_pcts = [df[df['brand']==b]['discount_pct'].mean() for b in top_brands]
ax.bar(top_brands, brand_disc_pcts, color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'])
ax.set_title('品牌平均折扣率', fontweight='bold', fontsize=12)
ax.set_ylabel('平均折扣率 (%)')
for i, v in enumerate(brand_disc_pcts):
    ax.text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=10)

# 8. 子品类 Top 15
ax = axes[2, 1]
top15_sub = sub_cat.head(15)
ax.barh(top15_sub.index[::-1], top15_sub['产品数'].values[::-1],
        color=plt.cm.RdYlGn(np.linspace(0.2,0.8,15)))
ax.set_title('子品类产品数 Top 15', fontweight='bold', fontsize=12)
ax.set_xlabel('产品数')

# 9. 图片数量 vs 价格散点
ax = axes[2, 2]
ax.scatter(df['image_count'], df['final_price'], alpha=0.3, s=15, c='#45B7D1')
ax.set_title('图片数量 vs 售价', fontweight='bold', fontsize=12)
ax.set_xlabel('图片数量')
ax.set_ylabel('售价 ($)')
ax.set_ylim(0, 200)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUT_V}shein_full_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 仪表板: {OUT_V}shein_full_dashboard.png")

# --- 图2: 价格与折扣深度 ---
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle('SHEIN 定价与折扣深度分析', fontsize=16, fontweight='bold')

# 品类价格箱线图
ax = axes2[0]
top5_cats = root_cat.head(6).index.tolist()
box_data = [df[df['root_category']==c]['final_price'].values for c in top5_cats]
short_names = [c[:12] for c in top5_cats]
bp = ax.boxplot(box_data, labels=short_names, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#4ECDC4')
    patch.set_alpha(0.6)
ax.set_title('品类价格分布', fontweight='bold')
ax.set_ylabel('售价 ($)')
ax.tick_params(axis='x', rotation=20)

# 折扣 vs 原价
ax = axes2[1]
has_disc = df[df['has_discount']]
ax.scatter(has_disc['initial_price'], has_disc['discount_pct'], alpha=0.4, s=20, c='#FF6B6B')
ax.set_title('原价 vs 折扣幅度', fontweight='bold')
ax.set_xlabel('原价 ($)')
ax.set_ylabel('折扣率 (%)')

# 价格带 vs 折扣率
ax = axes2[2]
band_disc = [band_stats.loc[b, '平均折扣'] for b in band_order]
ax.bar(range(len(band_order)), band_disc, color=band_colors_list)
ax.set_xticks(range(len(band_order)))
ax.set_xticklabels([b.split('(')[0] for b in band_order], rotation=20)
ax.set_title('价格带平均折扣率', fontweight='bold')
ax.set_ylabel('平均折扣率 (%)')

plt.tight_layout()
plt.savefig(f'{OUT_V}shein_pricing_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 定价分析图: {OUT_V}shein_pricing_analysis.png")

# --- 图3: 品牌竞争格局 ---
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle('SHEIN 品牌竞争格局', fontsize=16, fontweight='bold')

# 品牌份额饼图
ax = axes3[0]
brand_counts_pie = df['brand'].value_counts()
ax.pie(brand_counts_pie.values, labels=brand_counts_pie.index, autopct='%1.1f%%',
       colors=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'][:len(brand_counts_pie)])
ax.set_title('品牌产品份额', fontweight='bold')

# 品牌品类热力图
ax = axes3[1]
top5_brands = brand_stats.head(5).index.tolist()
top5_cats_list = root_cat.head(6).index.tolist()
heat_data = np.zeros((len(top5_brands), len(top5_cats_list)))
for i, b in enumerate(top5_brands):
    for j, c in enumerate(top5_cats_list):
        heat_data[i, j] = len(df[(df['brand']==b) & (df['root_category']==c)])

im = ax.imshow(heat_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(top5_cats_list)))
ax.set_xticklabels([c[:10] for c in top5_cats_list], rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(len(top5_brands)))
ax.set_yticklabels(top5_brands)
ax.set_title('品牌x品类热力图', fontweight='bold')
fig3.colorbar(im, ax=ax, label='产品数')

plt.tight_layout()
plt.savefig(f'{OUT_V}shein_brand_competition.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 品牌竞争图: {OUT_V}shein_brand_competition.png")

# ============================================================
# 12. 综合报告
# ============================================================
report = f"""# SHEIN 产品目录全面深度分析报告

## 执行摘要

**分析日期**: {datetime.now().strftime('%Y-%m-%d')}
**数据规模**: {len(df)}个产品 / {df['brand'].nunique()}个品牌 / {df['root_category'].nunique()}个根品类
**市场**: 美国站 (us.shein.com) | USD

---

## 一、数据概览

| 指标 | 数值 |
|------|------|
| 产品总数 | {len(df)} |
| 品牌数 | {df['brand'].nunique()} |
| 根品类数 | {df['root_category'].nunique()} |
| 子品类数 | {df['category'].nunique()} |
| 价格范围 | ${df['final_price'].min():.2f} ~ ${df['final_price'].max():.2f} |
| 平均售价 | ${df['final_price'].mean():.2f} |
| 中位售价 | ${df['final_price'].median():.2f} |
| 有折扣产品 | {len(discounted)} ({len(discounted)/len(df)*100:.1f}%) |
| 平均折扣率 | {discounted['discount_pct'].mean():.1f}% |

---

## 二、定价策略与折扣

### 折扣概况
- 有折扣产品: {len(discounted)} ({len(discounted)/len(df)*100:.1f}%)
- 平均折扣率: {discounted['discount_pct'].mean():.1f}%
- 最大折扣率: {discounted['discount_pct'].max():.1f}%

### 品牌折扣策略
"""

for brand, row in brand_disc.iterrows():
    report += f"- **{brand}**: 折扣比例{row['折扣比例']:.0f}%, 平均折扣{row['平均折扣率']:.1f}%\n"

report += f"""
---

## 三、品类分析

### 根品类 Top 10

| 品类 | 产品数 | 占比 | 均价 | 子品类 |
|------|--------|------|------|--------|
"""

for cat, row in root_cat.head(10).iterrows():
    report += f"| {cat} | {row['产品数']:.0f} | {row['产品数']/len(df)*100:.1f}% | ${row['平均售价']:.2f} | {row['子品类数']:.0f} |\n"

report += f"""
---

## 四、品牌竞争格局

| 品牌 | 产品数 | 占比 | 均价 | 品类数 | 折扣率 |
|------|--------|------|------|--------|--------|
"""

for brand, row in brand_stats.iterrows():
    report += f"| {brand} | {row['产品数']:.0f} | {row['产品数']/len(df)*100:.1f}% | ${row['平均售价']:.2f} | {row['品类数']:.0f} | {row['平均折扣']:.1f}% |\n"

report += f"""
---

## 五、颜色与尺码

### 颜色 Top 10
"""

for color, row in color_stats.head(10).iterrows():
    report += f"- **{color}**: {row['产品数']:.0f}产品, 均价${row['平均售价']:.2f}\n"

report += f"""
### 尺码策略
- 均码占比: {df[df['size']=='one-size'].shape[0]/len(df)*100:.1f}%

---

## 六、价格带分析

| 价格带 | 产品数 | 占比 | 折扣率 |
|--------|--------|------|--------|
"""

for band in band_order:
    row = band_stats.loc[band]
    report += f"| {band} | {row['产品数']:.0f} | {row['产品数']/len(df)*100:.1f}% | {row['折扣比例']*100:.0f}% |\n"

report += f"""
---

## 七、深度洞察与策略建议

### 核心发现

1. **低价主导** - 中位售价仅${df['final_price'].median():.2f}，SHEIN极致低价策略明显
2. **品类广泛** - 21个根品类覆盖家居、珠宝、美妆、箱包等全品类生态
3. **折扣策略激进** - {len(discounted)/len(df)*100:.0f}%产品有折扣，部分折扣超30%
4. **品牌集中** - SHEIN自有品牌占{df[df['brand']=='SHEIN'].shape[0]/len(df)*100:.0f}%，Unbeatablesale为第二大品牌
5. **视觉资产丰富** - 平均{df['image_count'].mean():.0f}张图片/产品，重视产品展示
6. **颜色偏好** - 多色(Multicolor)最多({df[df['color']=='Multicolor'].shape[0]}个)，Black/White紧随
7. **均码为主** - {df[df['size']=='one-size'].shape[0]/len(df)*100:.0f}%产品为均码，降低库存复杂度

### 策略建议

#### 定价优化
1. 高价品类(家具/汽配)可维持较高毛利
2. 低单价品类(饰品/小件)走量策略
3. 中间价格带($10-50)是竞争红海

#### 品类拓展
4. Home & Living是最大品类，可深化子品类
5. Jewelry和Bags增长空间大
6. 服装品类占比低，可考虑扩大

#### 品牌策略
7. SHEIN自有品牌绝对主导
8. Unbeatablesale专注高价细分市场
9. ROMWE定位年轻潮流

---

*报告由数据分析技能自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(f'{OUT_R}shein_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✅ 报告: {OUT_R}shein_analysis_report.md")

# 保存数据
df[['product_id','product_name','brand','root_category','category','final_price',
    'initial_price','discount_pct','price_band','color','size','image_count']].to_csv(
    f'{OUT_R}shein_products_analyzed.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("  ✅ SHEIN 全面深度分析完成！")
print("=" * 70)
print(f"\n📁 产出文件:")
print(f"  - {OUT_R}shein_analysis_report.md")
print(f"  - {OUT_R}shein_products_analyzed.csv")
print(f"  - {OUT_V}shein_full_dashboard.png")
print(f"  - {OUT_V}shein_pricing_analysis.png")
print(f"  - {OUT_V}shein_brand_competition.png")
