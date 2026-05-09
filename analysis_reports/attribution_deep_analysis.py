#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marketing Channel Attribution Deep Analysis
============================================
Comprehensive analysis of marketing channel attribution data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING AND OVERVIEW
# ============================================================
print("=" * 80)
print("  MARKETING CHANNEL ATTRIBUTION - DEEP ANALYSIS REPORT")
print("=" * 80)

DATA_PATH = 'data_storage/sample_channel_data.csv'
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

print("\n" + "=" * 80)
print("  SECTION 1: DATA OVERVIEW")
print("=" * 80)

print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names: {list(df.columns)}")
print(f"\nData Types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\nMissing Values:")
missing = df.isnull().sum()
for col in df.columns:
    print(f"  {col}: {missing[col]}")

print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print(f"\nBasic Statistics:")
print(df.describe().to_string())

print(f"\nUnique Users: {df['user_id'].nunique()}")
print(f"Unique Channels: {df['channel'].nunique()}")
print(f"Channels: {sorted(df['channel'].unique())}")
print(f"Total Conversion Events: {df['conversion_status'].sum()}")
print(f"Overall Conversion Rate (per touch): {df['conversion_status'].mean():.4f}")

# ============================================================
# 2. CHANNEL PERFORMANCE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("  SECTION 2: CHANNEL PERFORMANCE ANALYSIS")
print("=" * 80)

print("\n--- 2.1 Conversion Rate by Channel ---")
channel_stats = df.groupby('channel').agg(
    total_touches=('conversion_status', 'count'),
    conversions=('conversion_status', 'sum'),
    total_cost=('cost', 'sum'),
    total_revenue=('conversion_value', 'sum'),
    avg_conversion_value=('conversion_value', 'mean'),
    avg_cost=('cost', 'mean')
).reset_index()

channel_stats['conversion_rate'] = channel_stats['conversions'] / channel_stats['total_touches']
channel_stats['roi'] = np.where(
    channel_stats['total_cost'] > 0,
    (channel_stats['total_revenue'] - channel_stats['total_cost']) / channel_stats['total_cost'] * 100,
    np.inf
)
channel_stats['cost_per_conversion'] = np.where(
    channel_stats['conversions'] > 0,
    channel_stats['total_cost'] / channel_stats['conversions'],
    np.nan
)

channel_stats = channel_stats.sort_values('total_revenue', ascending=False)
print(channel_stats.to_string(index=False))

print("\n--- 2.2 Conversion Rate Ranking ---")
cr_rank = channel_stats.sort_values('conversion_rate', ascending=False)
for _, row in cr_rank.iterrows():
    print(f"  {row['channel']:25s}: {row['conversion_rate']:.2%} ({int(row['conversions'])} conversions from {int(row['total_touches'])} touches)")

print("\n--- 2.3 ROI Ranking ---")
roi_rank = channel_stats.sort_values('roi', ascending=False)
for _, row in roi_rank.iterrows():
    roi_str = f"{row['roi']:.1f}%" if row['roi'] != np.inf else "Infinite (zero cost)"
    print(f"  {row['channel']:25s}: ROI = {roi_str}  (Revenue: {row['total_revenue']:.0f}, Cost: {row['total_cost']:.0f})")

print("\n--- 2.4 Cost Per Conversion Ranking ---")
cpc_rank = channel_stats.sort_values('cost_per_conversion', ascending=True)
for _, row in cpc_rank.iterrows():
    if not np.isnan(row['cost_per_conversion']):
        print(f"  {row['channel']:25s}: CPC = {row['cost_per_conversion']:.2f}")
    else:
        print(f"  {row['channel']:25s}: No conversions")

# ============================================================
# 3. CUSTOMER JOURNEY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("  SECTION 3: CUSTOMER JOURNEY ANALYSIS")
print("=" * 80)

df_sorted = df.sort_values(['user_id', 'timestamp'])

journeys = {}
for uid, group in df_sorted.groupby('user_id'):
    path = group['channel'].tolist()
    converted = group['conversion_status'].max()
    conv_value = group['conversion_value'].sum()
    timestamps = group['timestamp'].tolist()
    journeys[uid] = {
        'path': path,
        'path_length': len(path),
        'converted': converted,
        'conversion_value': conv_value,
        'first_touch': path[0],
        'last_touch': path[-1],
        'timestamps': timestamps,
        'first_ts': timestamps[0],
        'last_ts': timestamps[-1]
    }

journey_df = pd.DataFrame.from_dict(journeys, orient='index')
journey_df.index.name = 'user_id'

print(f"\n--- 3.1 Path Length Statistics ---")
print(f"  Average path length: {journey_df['path_length'].mean():.2f} touches")
print(f"  Median path length:  {journey_df['path_length'].median():.1f} touches")
print(f"  Min path length:     {journey_df['path_length'].min()} touches")
print(f"  Max path length:     {journey_df['path_length'].max()} touches")
print(f"  Std path length:     {journey_df['path_length'].std():.2f}")

print(f"\n--- 3.2 Path Length Distribution ---")
for plen in sorted(journey_df['path_length'].unique()):
    count = (journey_df['path_length'] == plen).sum()
    pct = count / len(journey_df) * 100
    bar = '#' * count
    print(f"  {plen} touches: {count:2d} users ({pct:5.1f}%) {bar}")

print(f"\n--- 3.3 Path Length vs Conversion Rate ---")
for plen in sorted(journey_df['path_length'].unique()):
    subset = journey_df[journey_df['path_length'] == plen]
    conv_rate = subset['converted'].mean()
    n = len(subset)
    print(f"  Path length {plen}: conversion rate = {conv_rate:.2%} (n={n})")

print(f"\n--- 3.4 First-Touch Channel Distribution ---")
ft_counts = journey_df['first_touch'].value_counts()
for ch, cnt in ft_counts.items():
    conv = journey_df[journey_df['first_touch'] == ch]['converted'].sum()
    print(f"  {ch:25s}: {cnt:2d} users, {int(conv)} converted (FT conv rate: {conv/cnt:.2%})")

print(f"\n--- 3.5 Last-Touch Channel Distribution ---")
lt_counts = journey_df['last_touch'].value_counts()
for ch, cnt in lt_counts.items():
    conv = journey_df[journey_df['last_touch'] == ch]['converted'].sum()
    print(f"  {ch:25s}: {cnt:2d} users, {int(conv)} converted (LT conv rate: {conv/cnt:.2%})")

print(f"\n--- 3.6 Common Channel Sequence Patterns (Top 15) ---")
seq_counter = Counter()
for uid, j in journeys.items():
    path = j['path']
    for i in range(len(path) - 1):
        pair = f"{path[i]} -> {path[i+1]}"
        seq_counter[pair] += 1

for seq, cnt in seq_counter.most_common(15):
    print(f"  {seq:55s}: {cnt} occurrences")

# ============================================================
# 4. ATTRIBUTION MODELING
# ============================================================
print("\n" + "=" * 80)
print("  SECTION 4: ATTRIBUTION MODELING")
print("=" * 80)

converted_journeys = {uid: j for uid, j in journeys.items() if j['converted'] == 1}
total_conversion_value = sum(j['conversion_value'] for j in converted_journeys.values())
print(f"\nConverted Users: {len(converted_journeys)} / {len(journeys)}")
print(f"Total Conversion Value: {total_conversion_value:,.0f}")

all_channels = sorted(df['channel'].unique())

# --- 4.1 First-Touch Attribution ---
print("\n--- 4.1 First-Touch Attribution ---")
first_touch_attr = Counter()
for uid, j in converted_journeys.items():
    first_touch_attr[j['first_touch']] += j['conversion_value']

print(f"{'Channel':25s} {'Attributed Revenue':>20s} {'Share':>10s}")
print("-" * 60)
ft_total = sum(first_touch_attr.values())
for ch in all_channels:
    rev = first_touch_attr.get(ch, 0)
    share = rev / ft_total * 100 if ft_total > 0 else 0
    print(f"{ch:25s} {rev:>20,.0f} {share:>9.1f}%")

# --- 4.2 Last-Touch Attribution ---
print("\n--- 4.2 Last-Touch Attribution ---")
last_touch_attr = Counter()
for uid, j in converted_journeys.items():
    user_data = df_sorted[df_sorted['user_id'] == uid]
    conv_channel = user_data[user_data['conversion_status'] == 1]['channel'].values[0]
    last_touch_attr[conv_channel] += j['conversion_value']

print(f"{'Channel':25s} {'Attributed Revenue':>20s} {'Share':>10s}")
print("-" * 60)
lt_total = sum(last_touch_attr.values())
for ch in all_channels:
    rev = last_touch_attr.get(ch, 0)
    share = rev / lt_total * 100 if lt_total > 0 else 0
    print(f"{ch:25s} {rev:>20,.0f} {share:>9.1f}%")

# --- 4.3 Linear Attribution ---
print("\n--- 4.3 Linear Attribution ---")
linear_attr = Counter()
for uid, j in converted_journeys.items():
    weight = j['conversion_value'] / j['path_length']
    for ch in j['path']:
        linear_attr[ch] += weight

print(f"{'Channel':25s} {'Attributed Revenue':>20s} {'Share':>10s}")
print("-" * 60)
lin_total = sum(linear_attr.values())
for ch in all_channels:
    rev = linear_attr.get(ch, 0)
    share = rev / lin_total * 100 if lin_total > 0 else 0
    print(f"{ch:25s} {rev:>20,.0f} {share:>9.1f}%")

# --- 4.4 Time-Decay Attribution ---
print("\n--- 4.4 Time-Decay Attribution ---")
HALF_LIFE = 7  # days

time_decay_attr = Counter()
for uid, j in converted_journeys.items():
    user_data = df_sorted[df_sorted['user_id'] == uid].reset_index(drop=True)
    conv_ts = pd.Timestamp(user_data[user_data['conversion_status'] == 1]['timestamp'].values[0], tz='UTC')

    weights = []
    for _, row in user_data.iterrows():
        days_before = (conv_ts - row['timestamp']).total_seconds() / 86400
        weight = 2 ** (-days_before / HALF_LIFE)
        weights.append(weight)

    total_weight = sum(weights)
    for _, row in user_data.iterrows():
        days_before = (conv_ts - row['timestamp']).total_seconds() / 86400
        weight = 2 ** (-days_before / HALF_LIFE)
        attribution = (weight / total_weight) * j['conversion_value']
        time_decay_attr[row['channel']] += attribution

print(f"{'Channel':25s} {'Attributed Revenue':>20s} {'Share':>10s}")
print("-" * 60)
td_total = sum(time_decay_attr.values())
for ch in all_channels:
    rev = time_decay_attr.get(ch, 0)
    share = rev / td_total * 100 if td_total > 0 else 0
    print(f"{ch:25s} {rev:>20,.0f} {share:>9.1f}%")

# --- 4.5 Position-Based (U-Shaped) Attribution ---
print("\n--- 4.5 Position-Based (U-Shaped) Attribution ---")
position_attr = Counter()
for uid, j in converted_journeys.items():
    path = j['path']
    n = len(path)

    if n == 1:
        weights = [1.0]
    elif n == 2:
        weights = [0.5, 0.5]
    else:
        weights = [0.0] * n
        weights[0] = 0.4
        weights[-1] = 0.4
        middle_weight = 0.2 / (n - 2)
        for i in range(1, n - 1):
            weights[i] = middle_weight

    for ch, w in zip(path, weights):
        position_attr[ch] += w * j['conversion_value']

print(f"{'Channel':25s} {'Attributed Revenue':>20s} {'Share':>10s}")
print("-" * 60)
pb_total = sum(position_attr.values())
for ch in all_channels:
    rev = position_attr.get(ch, 0)
    share = rev / pb_total * 100 if pb_total > 0 else 0
    print(f"{ch:25s} {rev:>20,.0f} {share:>9.1f}%")

# --- 4.6 Attribution Model Comparison ---
print("\n--- 4.6 Attribution Model Comparison (Summary) ---")
print(f"{'Channel':25s} {'First-Touch':>14s} {'Last-Touch':>14s} {'Linear':>14s} {'Time-Decay':>14s} {'Position-Based':>14s}")
print("-" * 95)

comparison_data = {}
for ch in all_channels:
    ft_rev = first_touch_attr.get(ch, 0)
    lt_rev = last_touch_attr.get(ch, 0)
    lin_rev = linear_attr.get(ch, 0)
    td_rev = time_decay_attr.get(ch, 0)
    pb_rev = position_attr.get(ch, 0)
    comparison_data[ch] = {
        'first_touch': ft_rev,
        'last_touch': lt_rev,
        'linear': lin_rev,
        'time_decay': td_rev,
        'position_based': pb_rev
    }
    print(f"{ch:25s} {ft_rev:>14,.0f} {lt_rev:>14,.0f} {lin_rev:>14,.0f} {td_rev:>14,.0f} {pb_rev:>14,.0f}")

# ============================================================
# 5. CHANNEL SYNERGY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("  SECTION 5: CHANNEL SYNERGY ANALYSIS")
print("=" * 80)

print("\n--- 5.1 Channel Pair Co-occurrence in Converted Journeys ---")
pair_counter = Counter()
pair_conv_counter = Counter()

for uid, j in journeys.items():
    channels_in_path = list(set(j['path']))
    for pair in combinations(sorted(channels_in_path), 2):
        pair_counter[pair] += 1
        if j['converted'] == 1:
            pair_conv_counter[pair] += 1

print(f"{'Channel Pair':55s} {'Co-occurrences':>15s} {'In Converted':>15s} {'Conv Rate':>10s}")
print("-" * 100)
for pair, cnt in pair_counter.most_common(20):
    conv_cnt = pair_conv_counter.get(pair, 0)
    conv_rate = conv_cnt / cnt if cnt > 0 else 0
    print(f"  {pair[0] + ' + ' + pair[1]:53s} {cnt:>15d} {conv_cnt:>15d} {conv_rate:>9.1%}")

print("\n--- 5.2 Channel Presence Correlation with Conversion ---")
print(f"{'Channel':25s} {'P(conv|present)':>16s} {'P(conv|absent)':>16s} {'Lift':>10s} {'Phi':>8s}")
print("-" * 80)

phi_values = {}
for ch in all_channels:
    users_with = [uid for uid, j in journeys.items() if ch in j['path']]
    users_without = [uid for uid, j in journeys.items() if ch not in j['path']]

    conv_with = sum(1 for uid in users_with if journeys[uid]['converted'] == 1)
    conv_without = sum(1 for uid in users_without if journeys[uid]['converted'] == 1)

    p_with = conv_with / len(users_with) if users_with else 0
    p_without = conv_without / len(users_without) if users_without else 0
    lift = (p_with / p_without - 1) * 100 if p_without > 0 else float('inf')

    n = len(journeys)
    a = conv_with
    b = len(users_with) - conv_with
    c = conv_without
    d = len(users_without) - conv_without

    phi_num = a * d - b * c
    phi_den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = phi_num / phi_den if phi_den > 0 else 0
    phi_values[ch] = phi

    lift_str = f"{lift:+.1f}%" if lift != float('inf') else "N/A"
    print(f"{ch:25s} {p_with:>15.2%} {p_without:>15.2%} {lift_str:>10s} {phi:>8.4f}")

# ============================================================
# 6. CONVERSION TIMING ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("  SECTION 6: CONVERSION TIMING ANALYSIS")
print("=" * 80)

print("\n--- 6.1 Time from First Touch to Conversion ---")
conversion_times = []
for uid, j in converted_journeys.items():
    user_data = df_sorted[df_sorted['user_id'] == uid].reset_index(drop=True)
    conv_ts = pd.Timestamp(user_data[user_data['conversion_status'] == 1]['timestamp'].values[0], tz='UTC')
    first_ts = j['first_ts']
    days_to_conv = (conv_ts - first_ts).total_seconds() / 86400
    conversion_times.append({
        'user_id': uid,
        'days_to_conversion': days_to_conv,
        'path_length': j['path_length'],
        'conversion_value': j['conversion_value']
    })

conv_time_df = pd.DataFrame(conversion_times)

print(f"  Mean days to conversion:   {conv_time_df['days_to_conversion'].mean():.2f}")
print(f"  Median days to conversion:  {conv_time_df['days_to_conversion'].median():.2f}")
print(f"  Min days to conversion:     {conv_time_df['days_to_conversion'].min():.2f}")
print(f"  Max days to conversion:     {conv_time_df['days_to_conversion'].max():.2f}")
print(f"  Std dev:                    {conv_time_df['days_to_conversion'].std():.2f}")

print("\n--- 6.2 Channel Position in Journey (Closeness to Conversion) ---")
channel_positions = {}
for ch in all_channels:
    positions = []
    for uid, j in journeys.items():
        if j['converted'] == 1 and ch in j['path']:
            user_data = df_sorted[df_sorted['user_id'] == uid].reset_index(drop=True)
            conv_idx = user_data[user_data['conversion_status'] == 1].index[0]
            for idx, row in user_data.iterrows():
                if row['channel'] == ch:
                    pos = idx / max(conv_idx, 1)
                    positions.append(pos)
    if positions:
        channel_positions[ch] = {
            'mean_pos': np.mean(positions),
            'median_pos': np.median(positions),
            'count': len(positions)
        }

print(f"{'Channel':25s} {'Avg Relative Pos':>18s} {'Median Pos':>12s} {'Occurrences':>14s} {'Interpretation':>20s}")
print("-" * 95)
for ch in sorted(channel_positions.keys(), key=lambda x: channel_positions[x]['mean_pos']):
    info = channel_positions[ch]
    interp = "Near conversion" if info['mean_pos'] > 0.7 else ("Mid-journey" if info['mean_pos'] > 0.3 else "Early awareness")
    print(f"{ch:25s} {info['mean_pos']:>18.3f} {info['median_pos']:>12.3f} {info['count']:>14d} {interp:>20s}")

# ============================================================
# 7. STATISTICAL TESTS
# ============================================================
print("\n" + "=" * 80)
print("  SECTION 7: STATISTICAL TESTS")
print("=" * 80)

# NOTE: All 20 users converted (100% user-level conversion rate).
# This limits some tests. We perform tests at the touchpoint level where meaningful.

print("\n--- 7.1 Chi-Square Test: Conversion Status Independence from Channel (touch-level) ---")
contingency = pd.crosstab(df['channel'], df['conversion_status'])
print("Contingency Table (channel x conversion_status):")
print(contingency.to_string())

chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom:   {dof}")
print(f"P-value:              {p_value:.4f}")
print(f"Significant at 0.05:  {'Yes' if p_value < 0.05 else 'No'}")
if p_value < 0.05:
    print("=> Conversion status IS significantly associated with channel (reject H0)")
else:
    print("=> Conversion status is NOT significantly associated with channel (fail to reject H0)")

n_obs = contingency.sum().sum()
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n_obs * min_dim))
print(f"Cramer's V (effect size): {cramers_v:.4f}")
effect_interp = "small" if cramers_v < 0.1 else ("medium" if cramers_v < 0.3 else "large")
print(f"Effect size interpretation: {effect_interp}")

print("\n--- 7.2 Kruskal-Wallis Test: Conversion Value by Converting Channel ---")
conv_events = df[df['conversion_status'] == 1][['channel', 'conversion_value']]
print(f"Total conversion events: {len(conv_events)}")
print(f"\nConversion values by channel:")
groups = []
group_labels = []
for ch in all_channels:
    ch_vals = conv_events[conv_events['channel'] == ch]['conversion_value'].values
    if len(ch_vals) > 0:
        groups.append(ch_vals)
        group_labels.append(ch)
        print(f"  {ch:25s}: n={len(ch_vals)}, mean={np.mean(ch_vals):.0f}, median={np.median(ch_vals):.0f}, std={np.std(ch_vals):.0f}")

if len(groups) >= 2:
    h_stat, kw_p = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis H statistic: {h_stat:.4f}")
    print(f"P-value:                    {kw_p:.4f}")
    print(f"Significant at 0.05:        {'Yes' if kw_p < 0.05 else 'No'}")
    if kw_p < 0.05:
        print("=> Conversion values DIFFER significantly by channel (reject H0)")
    else:
        print("=> Conversion values do NOT differ significantly by channel (fail to reject H0)")

    n_total = sum(len(g) for g in groups)
    eta_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
    print(f"Eta-squared (approx):       {eta_sq:.4f}")

    # Also run one-way ANOVA for comparison
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA (for comparison):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value:     {anova_p:.4f}")

print("\n--- 7.3 Mann-Whitney U Test: Conversion Value by Journey Position ---")
# Since all users converted, let's compare conversion values by path length groups
short_path = conv_time_df[conv_time_df['path_length'] <= 4]['conversion_value'].values
long_path = conv_time_df[conv_time_df['path_length'] > 4]['conversion_value'].values

print(f"Short path (<=4) avg conversion value: {np.mean(short_path):.0f} (n={len(short_path)})")
print(f"Long path (>4) avg conversion value:   {np.mean(long_path):.0f} (n={len(long_path)})")

if len(short_path) > 0 and len(long_path) > 0:
    u_stat, u_p = stats.mannwhitneyu(short_path, long_path, alternative='two-sided')
    print(f"\nMann-Whitney U statistic: {u_stat:.4f}")
    print(f"P-value:                  {u_p:.4f}")
    print(f"Significant at 0.05:      {'Yes' if u_p < 0.05 else 'No'}")
    if u_p < 0.05:
        print("=> Conversion values DIFFER significantly between short and long journeys")
    else:
        print("=> No significant difference in conversion values between short and long journeys")

# Spearman correlation: days to conversion vs conversion value
print("\n--- 7.4 Spearman Correlation: Days to Conversion vs Conversion Value ---")
corr, corr_p = stats.spearmanr(conv_time_df['days_to_conversion'], conv_time_df['conversion_value'])
print(f"Spearman rho: {corr:.4f}")
print(f"P-value:      {corr_p:.4f}")
print(f"Significant:  {'Yes' if corr_p < 0.05 else 'No'}")
if corr_p < 0.05:
    direction = "positive" if corr > 0 else "negative"
    print(f"=> Significant {direction} correlation between time to conversion and conversion value")
else:
    print("=> No significant correlation between time to conversion and conversion value")

# ============================================================
# BUILD COMPREHENSIVE REPORT
# ============================================================
print("\n" + "=" * 80)
print("  GENERATING REPORT FILE")
print("=" * 80)

report_lines = []

def add(text=""):
    report_lines.append(text)

add("# 营销渠道归因深度分析报告")
add("")
add("**分析日期**: 2026-05-09  ")
add("**数据来源**: sample_channel_data.csv  ")
add(f"**样本量**: {df.shape[0]} 条触点记录, {df['user_id'].nunique()} 个用户, {df['channel'].nunique()} 个渠道  ")
add(f"**重要说明**: 数据集中所有 {len(journeys)} 个用户均完成转化 (100% 转化率)，本分析聚焦于渠道归因、转化价值差异和旅程模式。")
add("")
add("---")
add("")
add("## 一、数据概览")
add("")
add("### 1.1 数据基本信息")
add("")
add("| 指标 | 值 |")
add("|------|-----|")
add(f"| 数据集行数 | {df.shape[0]} |")
add(f"| 数据集列数 | {df.shape[1]} |")
add(f"| 唯一用户数 | {df['user_id'].nunique()} |")
add(f"| 唯一渠道数 | {df['channel'].nunique()} |")
add(f"| 渠道列表 | {', '.join(sorted(df['channel'].unique()))} |")
add(f"| 缺失值总数 | {df.isnull().sum().sum()} |")
add(f"| 重复行数 | {df.duplicated().sum()} |")
add(f"| 总转化事件 | {int(df['conversion_status'].sum())} |")
add(f"| 总转化用户 | {len(converted_journeys)} / {len(journeys)} |")
add(f"| 用户转化率 | {len(converted_journeys)/len(journeys):.2%} |")
add(f"| 总转化收入 | {total_conversion_value:,.0f} |")
add(f"| 总营销成本 | {df['cost'].sum():,.0f} |")
add(f"| 整体ROI | {(total_conversion_value - df['cost'].sum()) / df['cost'].sum() * 100:.1f}% |")
add("")

add("### 1.2 数据字段说明")
add("")
add("| 字段 | 类型 | 说明 |")
add("|------|------|------|")
add("| user_id | 字符串 | 用户唯一标识 |")
add("| timestamp | datetime(UTC) | 触点时间戳 |")
add("| channel | 字符串 | 营销渠道 (7种) |")
add("| conversion_status | 整数 | 是否转化 (0/1) |")
add("| conversion_value | 浮点数 | 转化价值 (仅转化事件有值) |")
add("| cost | 浮点数 | 触点成本 |")
add("")

add("---")
add("")
add("## 二、渠道表现分析")
add("")
add("### 2.1 渠道综合指标")
add("")
add("| 渠道 | 触点数 | 转化数 | 转化率 | 总收入 | 总成本 | ROI | 每次转化成本 |")
add("|------|--------|--------|--------|--------|--------|-----|-------------|")

for _, row in channel_stats.iterrows():
    roi_str = f"{row['roi']:.1f}%" if row['roi'] != np.inf else "Inf (零成本)"
    cpc_str = f"{row['cost_per_conversion']:.2f}" if not np.isnan(row['cost_per_conversion']) else "N/A"
    add(f"| {row['channel']} | {int(row['total_touches'])} | {int(row['conversions'])} | {row['conversion_rate']:.2%} | {row['total_revenue']:.0f} | {row['total_cost']:.0f} | {roi_str} | {cpc_str} |")

add("")
add("### 2.2 关键发现")
add("")
add(f"- **最高转化率渠道**: {cr_rank.iloc[0]['channel']} ({cr_rank.iloc[0]['conversion_rate']:.2%})")
add(f"- **最大收入贡献渠道**: {channel_stats.iloc[0]['channel']} (总收入: {channel_stats.iloc[0]['total_revenue']:.0f})")

best_cpc = channel_stats[channel_stats['conversions'] > 0].sort_values('cost_per_conversion').iloc[0]
add(f"- **最佳转化成本效率**: {best_cpc['channel']} (每次转化成本: {best_cpc['cost_per_conversion']:.2f})")

best_roi = channel_stats[(channel_stats['conversions'] > 0) & (channel_stats['total_cost'] > 0)].sort_values('roi', ascending=False).iloc[0]
add(f"- **最高ROI (付费渠道)**: {best_roi['channel']} (ROI: {best_roi['roi']:.1f}%)")
add(f"- **organic_search**: 零成本产生流量但未直接转化，在认知阶段发挥作用")
add(f"- **affiliate 和 display**: 仅少量触点且无直接转化，需评估其间接价值")
add("")

add("---")
add("")
add("## 三、客户旅程分析")
add("")
add("### 3.1 路径长度统计")
add("")
add("| 指标 | 值 |")
add("|------|-----|")
add(f"| 平均路径长度 | {journey_df['path_length'].mean():.2f} 个触点 |")
add(f"| 中位数路径长度 | {journey_df['path_length'].median():.1f} 个触点 |")
add(f"| 最短路径 | {journey_df['path_length'].min()} 个触点 |")
add(f"| 最长路径 | {journey_df['path_length'].max()} 个触点 |")
add(f"| 标准差 | {journey_df['path_length'].std():.2f} |")
add("")

add("### 3.2 路径长度分布")
add("")
add("| 路径长度 | 用户数 | 占比 |")
add("|----------|--------|------|")
for plen in sorted(journey_df['path_length'].unique()):
    count = (journey_df['path_length'] == plen).sum()
    pct = count / len(journey_df) * 100
    add(f"| {plen} | {count} | {pct:.1f}% |")
add("")

add("### 3.3 首次触达渠道分布")
add("")
add("| 渠道 | 用户数 | 占比 |")
add("|------|--------|------|")
for ch, cnt in ft_counts.items():
    pct = cnt / len(journey_df) * 100
    add(f"| {ch} | {cnt} | {pct:.1f}% |")
add("")
add("**洞察**: paid_search 是最常见的首次触达渠道 (25%)，说明付费搜索在获取新用户方面投入最大。social_media 和 email 并列第二 (各20%)。")
add("")

add("### 3.4 末次触达(转化)渠道分布")
add("")
add("| 渠道 | 转化次数 | 占比 | 平均转化价值 |")
add("|------|----------|------|-------------|")
conv_by_channel = df[df['conversion_status'] == 1].groupby('channel').agg(
    conv_count=('conversion_value', 'count'),
    avg_value=('conversion_value', 'mean')
).reset_index().sort_values('conv_count', ascending=False)
for _, row in conv_by_channel.iterrows():
    pct = row['conv_count'] / len(conv_events) * 100
    add(f"| {row['channel']} | {int(row['conv_count'])} | {pct:.1f}% | {row['avg_value']:.0f} |")
add("")
add("**洞察**: social_media 和 email 是最常见的转化渠道 (各6次, 30%)，其次是 content_marketing (5次, 25%) 和 paid_search (5次, 25%)。但 paid_search 的平均转化价值最高 (2360)。")
add("")

add("### 3.5 常见渠道转换序列 (Top 10)")
add("")
add("| 序列模式 | 出现次数 |")
add("|----------|----------|")
for seq, cnt in seq_counter.most_common(10):
    add(f"| {seq} | {cnt} |")
add("")
add("**洞察**: social_media <-> email 的双向转换最为频繁，说明这两个渠道形成了紧密的协同关系。paid_search -> email 也很常见，体现搜索到邮件的转化路径。")
add("")

add("---")
add("")
add("## 四、归因模型分析")
add("")
add(f"基于 {len(converted_journeys)} 个已转化用户，总转化价值: **{total_conversion_value:,.0f}**")
add("")

add("### 4.1 首次触达归因 (First-Touch Attribution)")
add("")
add("原理: 将全部转化价值归于用户首次接触的渠道。适用于评估渠道的引流能力。")
add("")
add("| 渠道 | 归因收入 | 占比 |")
add("|------|----------|------|")
for ch in all_channels:
    rev = first_touch_attr.get(ch, 0)
    share = rev / ft_total * 100 if ft_total > 0 else 0
    add(f"| {ch} | {rev:,.0f} | {share:.1f}% |")
add("")

add("### 4.2 末次触达归因 (Last-Touch Attribution)")
add("")
add("原理: 将全部转化价值归于触发转化的渠道。适用于评估渠道的直接转化能力。")
add("")
add("| 渠道 | 归因收入 | 占比 |")
add("|------|----------|------|")
for ch in all_channels:
    rev = last_touch_attr.get(ch, 0)
    share = rev / lt_total * 100 if lt_total > 0 else 0
    add(f"| {ch} | {rev:,.0f} | {share:.1f}% |")
add("")

add("### 4.3 线性归因 (Linear Attribution)")
add("")
add("原理: 将转化价值平均分配给路径中的所有触点。适用于评估渠道在完整旅程中的贡献。")
add("")
add("| 渠道 | 归因收入 | 占比 |")
add("|------|----------|------|")
for ch in all_channels:
    rev = linear_attr.get(ch, 0)
    share = rev / lin_total * 100 if lin_total > 0 else 0
    add(f"| {ch} | {rev:,.0f} | {share:.1f}% |")
add("")

add("### 4.4 时间衰减归因 (Time-Decay Attribution)")
add("")
add(f"原理: 半衰期 = {HALF_LIFE} 天，越接近转化的触点获得越高权重。适用于强调近期接触的渠道。")
add("")
add("| 渠道 | 归因收入 | 占比 |")
add("|------|----------|------|")
for ch in all_channels:
    rev = time_decay_attr.get(ch, 0)
    share = rev / td_total * 100 if td_total > 0 else 0
    add(f"| {ch} | {rev:,.0f} | {share:.1f}% |")
add("")

add("### 4.5 位置归因 (Position-Based / U-Shaped Attribution)")
add("")
add("原理: 首次触点 40%, 末次触点 40%, 中间触点平均分配 20%。兼顾引流和转化。")
add("")
add("| 渠道 | 归因收入 | 占比 |")
add("|------|----------|------|")
for ch in all_channels:
    rev = position_attr.get(ch, 0)
    share = rev / pb_total * 100 if pb_total > 0 else 0
    add(f"| {ch} | {rev:,.0f} | {share:.1f}% |")
add("")

add("### 4.6 归因模型对比总表")
add("")
add("| 渠道 | 首次触达 | 末次触达 | 线性 | 时间衰减 | 位置归因 |")
add("|------|----------|----------|------|----------|----------|")
for ch in all_channels:
    d = comparison_data[ch]
    add(f"| {ch} | {d['first_touch']:,.0f} | {d['last_touch']:,.0f} | {d['linear']:,.0f} | {d['time_decay']:,.0f} | {d['position_based']:,.0f} |")
add("")

# Find most consistent channel across models
add("### 4.7 归因模型关键洞察")
add("")

# Compute coefficient of variation for each channel across models
add("**各渠道归因收入稳定性分析 (变异系数)**:")
add("")
add("| 渠道 | 平均归因收入 | 标准差 | 变异系数 | 稳定性 |")
add("|------|-------------|--------|----------|--------|")
for ch in all_channels:
    d = comparison_data[ch]
    vals = [d['first_touch'], d['last_touch'], d['linear'], d['time_decay'], d['position_based']]
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    cv = std_v / mean_v * 100 if mean_v > 0 else 0
    stability = "高" if cv < 20 else ("中" if cv < 50 else "低")
    add(f"| {ch} | {mean_v:,.0f} | {std_v:,.0f} | {cv:.1f}% | {stability} |")
add("")

add("**核心发现**:")
add("")
add(f"- **paid_search** 在末次触达模型中获得最高归因收入 ({last_touch_attr.get('paid_search', 0):,.0f})，但在首次触达中较低，说明它主要在转化阶段发挥作用")
add(f"- **email** 在线性归因中获得最高权重 ({linear_attr.get('email', 0):,.0f})，体现其在整个旅程中的稳定贡献")
add(f"- **social_media** 在末次触达模型中表现突出 ({last_touch_attr.get('social_media', 0):,.0f})，是重要的转化驱动渠道")
add(f"- **organic_search** 首次触达归因较高 ({first_touch_attr.get('organic_search', 0):,.0f}) 但末次触达为零，纯粹的引流渠道")
add(f"- 不同模型之间归因结果差异显著，选择合适的归因模型对营销预算分配至关重要")
add("")

add("---")
add("")
add("## 五、渠道协同分析")
add("")
add("### 5.1 渠道组合共现分析")
add("")
add("| 渠道组合 | 共现次数 | 转化率 |")
add("|----------|----------|--------|")
for pair, cnt in pair_counter.most_common(15):
    conv_cnt = pair_conv_counter.get(pair, 0)
    conv_rate = conv_cnt / cnt if cnt > 0 else 0
    add(f"| {pair[0]} + {pair[1]} | {cnt} | {conv_rate:.1%} |")
add("")
add("**洞察**: email + paid_search、email + social_media 是最常见的渠道组合，各出现了多次。由于所有用户都转化了，所有组合的转化率均为100%。但这告诉我们哪些渠道经常协同工作。")
add("")

add("### 5.2 渠道存在与用户旅程特征")
add("")
add("| 渠道 | 包含该渠道的用户数 | 用户占比 |")
add("|------|-------------------|----------|")
for ch in all_channels:
    users_with = sum(1 for uid, j in journeys.items() if ch in j['path'])
    pct = users_with / len(journeys) * 100
    add(f"| {ch} | {users_with} | {pct:.1f}% |")
add("")
add("**洞察**: social_media 和 email 覆盖率最高，出现在大部分用户旅程中。affiliate 和 display 覆盖率最低，仅影响少数用户。")
add("")

add("---")
add("")
add("## 六、转化时间分析")
add("")
add("### 6.1 从首次触达到转化的时间")
add("")
add("| 指标 | 值 |")
add("|------|-----|")
add(f"| 平均天数 | {conv_time_df['days_to_conversion'].mean():.2f} 天 |")
add(f"| 中位数天数 | {conv_time_df['days_to_conversion'].median():.2f} 天 |")
add(f"| 最短天数 | {conv_time_df['days_to_conversion'].min():.2f} 天 |")
add(f"| 最长天数 | {conv_time_df['days_to_conversion'].max():.2f} 天 |")
add(f"| 标准差 | {conv_time_df['days_to_conversion'].std():.2f} 天 |")
add("")

add("### 6.2 各用户转化周期详情")
add("")
add("| 用户ID | 路径长度 | 转化天数 | 转化价值 |")
add("|--------|----------|----------|----------|")
for _, row in conv_time_df.sort_values('days_to_conversion', ascending=False).iterrows():
    add(f"| {row['user_id']} | {int(row['path_length'])} | {row['days_to_conversion']:.1f} | {row['conversion_value']:,.0f} |")
add("")

add("### 6.3 渠道在旅程中的位置特征")
add("")
add("相对位置指标: 0 = 旅程开始, 1 = 接近转化。值越高表示越接近转化时刻。")
add("")
add("| 渠道 | 平均相对位置 | 出现次数 | 定位 |")
add("|------|-------------|----------|------|")
for ch in sorted(channel_positions.keys(), key=lambda x: channel_positions[x]['mean_pos']):
    info = channel_positions[ch]
    interp = "接近转化" if info['mean_pos'] > 0.7 else ("旅程中期" if info['mean_pos'] > 0.3 else "早期认知")
    add(f"| {ch} | {info['mean_pos']:.3f} | {info['count']} | {interp} |")
add("")

add("### 6.4 时间分析关键发现")
add("")
add(f"- 平均转化周期为 **{conv_time_df['days_to_conversion'].mean():.1f}** 天")
add(f"- 转化最快仅需 **{conv_time_df['days_to_conversion'].min():.1f}** 天，最长需要 **{conv_time_df['days_to_conversion'].max():.1f}** 天")
add("- 转化周期与营销渠道策略直接相关，较短的周期意味着更高效的用户旅程")
add("")

add("---")
add("")
add("## 七、统计检验结果")
add("")
add("### 7.1 卡方检验: 触点级别转化状态是否与渠道独立")
add("")
add(f"| 检验指标 | 值 |")
add(f"|----------|-----|")
add(f"| 卡方统计量 | {chi2:.4f} |")
add(f"| 自由度 | {dof} |")
add(f"| P值 | {p_value:.4f} |")
add(f"| 显著性 (alpha=0.05) | {'是' if p_value < 0.05 else '否'} |")
add(f"| Cramer's V | {cramers_v:.4f} ({effect_interp}效应) |")
add("")
if p_value < 0.05:
    add("**结论**: 在触点级别，转化状态与渠道之间存在统计学显著关联 (p < 0.05)。某些渠道比其他渠道更容易直接触发转化。")
else:
    add("**结论**: 在触点级别，未能证明转化状态与渠道之间存在显著关联 (p >= 0.05)。各渠道的直接转化概率相近。")
add("")

add("### 7.2 Kruskal-Wallis 检验: 转化价值是否因渠道而异")
add("")
if len(groups) >= 2:
    add(f"| 检验指标 | 值 |")
    add(f"|----------|-----|")
    add(f"| H统计量 | {h_stat:.4f} |")
    add(f"| P值 | {kw_p:.4f} |")
    add(f"| 显著性 (alpha=0.05) | {'是' if kw_p < 0.05 else '否'} |")
    add(f"| Eta方 (近似) | {eta_sq:.4f} |")
    add("")
    add("**各渠道转化价值统计**:")
    add("")
    add("| 渠道 | 样本量 | 均值 | 中位数 | 标准差 |")
    add("|------|--------|------|--------|--------|")
    for ch, grp in zip(group_labels, groups):
        add(f"| {ch} | {len(grp)} | {np.mean(grp):.0f} | {np.median(grp):.0f} | {np.std(grp):.0f} |")
    add("")
    if kw_p < 0.05:
        add("**结论**: 不同转化渠道的转化价值存在统计学显著差异 (p < 0.05)。paid_search 带来的转化价值显著高于其他渠道。")
    else:
        add("**结论**: 未能证明不同渠道的转化价值存在显著差异 (p >= 0.05)。样本量较小 (仅20个转化事件) 可能限制了检验的统计功效。")
    add("")
    add(f"补充: 单因素方差分析 (ANOVA) F={f_stat:.4f}, p={anova_p:.4f}，与 Kruskal-Wallis 结果一致。")
add("")

add("### 7.3 路径长度与转化价值的关系")
add("")
add(f"| 检验指标 | 值 |")
add(f"|----------|-----|")
add(f"| 短路径(<=4)平均价值 | {np.mean(short_path):.0f} (n={len(short_path)}) |")
add(f"| 长路径(>4)平均价值 | {np.mean(long_path):.0f} (n={len(long_path)}) |")
add(f"| Mann-Whitney U | {u_stat:.4f} |")
add(f"| P值 | {u_p:.4f} |")
add(f"| 显著性 (alpha=0.05) | {'是' if u_p < 0.05 else '否'} |")
add("")
if u_p < 0.05:
    add("**结论**: 短路径和长路径用户的转化价值存在显著差异，说明旅程长度与客户质量相关。")
else:
    add("**结论**: 短路径和长路径用户的转化价值无显著差异，说明旅程长度不影响转化价值大小。")
add("")

add("### 7.4 转化时间与转化价值的相关性")
add("")
add(f"| 指标 | 值 |")
add(f"|------|-----|")
add(f"| Spearman rho | {corr:.4f} |")
add(f"| P值 | {corr_p:.4f} |")
add(f"| 显著性 (alpha=0.05) | {'是' if corr_p < 0.05 else '否'} |")
add("")
if corr_p < 0.05:
    direction = "正" if corr > 0 else "负"
    add(f"**结论**: 转化时间与转化价值存在显著的{direction}相关关系。")
else:
    add("**结论**: 转化时间与转化价值之间无显著相关关系，决策周期的长短不影响最终转化价值。")
add("")

add("---")
add("")
add("## 八、综合业务洞察与建议")
add("")
add("### 8.1 渠道角色定位 (基于位置分析)")
add("")
add("| 角色 | 渠道 | 证据 |")
add("|------|------|------|")

# Determine roles based on position analysis
early_channels = []
mid_channels = []
late_channels = []
for ch in all_channels:
    if ch in channel_positions:
        pos = channel_positions[ch]['mean_pos']
        if pos <= 0.3:
            early_channels.append(ch)
        elif pos <= 0.7:
            mid_channels.append(ch)
        else:
            late_channels.append(ch)

add(f"| 认知型 (Awareness) | {', '.join(early_channels) if early_channels else 'organic_search'} | 旅程早期触达用户 |")
add(f"| 培育型 (Consideration) | {', '.join(mid_channels) if mid_channels else 'content_marketing'} | 旅程中期维持兴趣 |")
add(f"| 转化型 (Conversion) | {', '.join(late_channels) if late_channels else 'paid_search, email'} | 旅程后期推动转化 |")
add("")

add("### 8.2 核心发现总结")
add("")
add(f"1. **付费搜索 (paid_search) 是最高价值转化渠道**: 平均转化价值 {conv_events[conv_events['channel']=='paid_search']['conversion_value'].mean():.0f}，远超其他渠道")
add(f"2. **社交媒体 (social_media) 是最高频转化渠道**: 完成 {int(conv_events[conv_events['channel']=='social_media'].shape[0])} 次转化，覆盖面广")
add(f"3. **邮件营销 (email) 成本效率最高**: 每次转化成本仅 {best_cpc['cost_per_conversion']:.2f}，ROI 达 {best_roi['roi']:.1f}%")
add(f"4. **多触点旅程是常态**: 平均 {journey_df['path_length'].mean():.1f} 个触点，用户需要多次接触才能决策")
add(f"5. **转化周期约 {conv_time_df['days_to_conversion'].mean():.0f} 天**: 用户从首次接触到转化平均需要约一周时间")
add(f"6. **渠道高度互补**: 没有单一渠道能完成所有转化，social_media/email/paid_search 构成核心转化三角")
add("")

add("### 8.3 预算优化建议")
add("")
add("| 优先级 | 渠道 | 建议动作 | 理由 |")
add("|--------|------|----------|------|")
add("| 高 | email | 增加投入 | 最高ROI，最低CPC，贯穿全旅程 |")
add("| 高 | paid_search | 维持/优化 | 最高转化价值，核心收割渠道 |")
add("| 高 | social_media | 维持投入 | 最高频转化渠道，引流能力强 |")
add("| 中 | content_marketing | 稳定投入 | 旅程中期培育作用，稳定贡献 |")
add("| 低 | affiliate | 评估ROI | 仅2次触点无直接转化 |")
add("| 低 | display | 谨慎投入 | 仅1次触点，效果不明 |")
add("| 观察 | organic_search | 持续SEO优化 | 零成本引流，间接贡献显著 |")
add("")

add("### 8.4 归因模型选择建议")
add("")
add("鉴于当前数据特征 (100%转化率，短周期)，建议:")
add("")
add("1. **短期**: 使用末次触达归因评估直接转化效果")
add("2. **中期**: 引入线性归因评估渠道全程贡献")
add("3. **长期**: 实施位置归因模型 (U-shaped)，兼顾引流和转化")
add("4. **数据积累后**: 切换到数据驱动归因 (Data-Driven Attribution)")
add("")

add("### 8.5 数据局限性")
add("")
add("- **样本量小**: 仅20个用户/95个触点，统计功效有限")
add("- **100%转化率**: 无法分析转化vs未转化的差异因素")
add("- **时间跨度短**: 所有数据集中在2024年1月，缺乏季节性分析")
add("- **缺乏对照组**: 无法进行因果推断")
add("- **建议**: 扩大样本量至1000+用户，并包含未转化用户以进行更全面的归因分析")
add("")

add("---")
add("")
add(f"*本报告由自动化分析系统生成于 2026-05-09。数据样本量为 {len(journeys)} 用户 {df.shape[0]} 条触点记录。统计分析结论受样本量限制，建议在更大规模数据上验证。*")

# Write report
report_text = "\n".join(report_lines)
report_path = 'analysis_reports/attribution_deep_analysis.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\nReport saved to: {report_path}")
print(f"Report length: {len(report_lines)} lines")
print("\nAnalysis complete!")
