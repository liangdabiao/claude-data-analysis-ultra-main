"""
Marketing Channel Attribution Analysis - Comprehensive Visualizations
Generates 8 professional charts for multi-touch attribution analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data_storage', 'sample_channel_data.csv')
OUTPUT_DIR = BASE_DIR

# Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Professional color palettes
CHANNEL_COLORS = {
    'paid_search': '#E74C3C',
    'social_media': '#3498DB',
    'email': '#2ECC71',
    'organic_search': '#F39C12',
    'content_marketing': '#9B59B6',
    'affiliate': '#1ABC9C',
    'display': '#E67E22',
}

PALETTE_SET2 = sns.color_palette('Set2', 8)
PALETTE_TAB10 = sns.color_palette('tab10', 10)

# ============================================================
# Data Loading & Preprocessing
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by user and timestamp to establish journey order
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Add journey position (1-indexed) for each touchpoint within a user journey
df['journey_position'] = df.groupby('user_id').cumcount() + 1
df['journey_length'] = df.groupby('user_id')['journey_position'].transform('max')
df['relative_position'] = df['journey_position'] / df['journey_length']

# Identify converted users
converted_users = df[df['conversion_status'] == 1]['user_id'].unique()
df['is_converted_user'] = df['user_id'].isin(converted_users)

print(f"Total records: {len(df)}")
print(f"Total users: {df['user_id'].nunique()}")
print(f"Converted users: {len(converted_users)}")
print(f"Channels: {sorted(df['channel'].unique())}")

# ============================================================
# Helper computations
# ============================================================

# 1. Channel-level metrics
channel_metrics = {}
for ch in df['channel'].unique():
    ch_data = df[df['channel'] == ch]
    total_touches = len(ch_data)
    conversions = ch_data['conversion_status'].sum()
    total_cost = ch_data['cost'].sum()
    total_value = ch_data.loc[ch_data['conversion_status'] == 1, 'conversion_value'].sum()
    conv_rate = conversions / total_touches if total_touches > 0 else 0
    cost_per_conv = total_cost / conversions if conversions > 0 else np.nan
    roi = (total_value - total_cost) / total_cost if total_cost > 0 else np.inf

    channel_metrics[ch] = {
        'total_touches': total_touches,
        'conversions': int(conversions),
        'total_cost': total_cost,
        'total_value': total_value,
        'conv_rate': conv_rate,
        'cost_per_conv': cost_per_conv,
        'roi': roi,
    }

metrics_df = pd.DataFrame(channel_metrics).T
metrics_df.index.name = 'channel'
metrics_df = metrics_df.reset_index()
metrics_df = metrics_df.sort_values('channel')

print("\nChannel Metrics:")
print(metrics_df[['channel', 'conversions', 'conv_rate', 'roi', 'cost_per_conv']].to_string(index=False))

# 2. Attribution models
all_channels = sorted(df['channel'].unique())

def first_touch_attribution(user_df):
    return user_df.iloc[0]['channel']

def last_touch_attribution(user_df):
    converted = user_df[user_df['conversion_status'] == 1]
    if len(converted) > 0:
        return converted.iloc[0]['channel']
    return user_df.iloc[-1]['channel']

def linear_attribution(user_df):
    n = len(user_df)
    return {ch: user_df[user_df['channel'] == ch].shape[0] / n for ch in user_df['channel'].unique()}

def time_decay_attribution(user_df, decay_rate=0.5):
    n = len(user_df)
    weights = []
    for i in range(n):
        w = decay_rate ** (n - 1 - i)
        weights.append(w)
    total_w = sum(weights)
    result = {}
    for i, row in user_df.iterrows():
        ch = row['channel']
        idx = user_df.index.get_loc(i)
        result[ch] = result.get(ch, 0) + weights[idx] / total_w
    return result

def position_based_attribution(user_df):
    n = len(user_df)
    if n == 1:
        return {user_df.iloc[0]['channel']: 1.0}
    result = {}
    first_ch = user_df.iloc[0]['channel']
    last_ch = user_df.iloc[-1]['channel']
    result[first_ch] = result.get(first_ch, 0) + 0.4
    result[last_ch] = result.get(last_ch, 0) + 0.4
    middle_weight = 0.2 / max(n - 2, 1) if n > 2 else 0
    for idx in range(1, n - 1):
        ch = user_df.iloc[idx]['channel']
        result[ch] = result.get(ch, 0) + middle_weight
    return result


# Compute attribution weights for converted users
attribution_results = {
    '首次触达': defaultdict(float),
    '末次触达': defaultdict(float),
    '线性归因': defaultdict(float),
    '时间衰减': defaultdict(float),
    '位置归因': defaultdict(float),
}

for user in converted_users:
    user_df = df[df['user_id'] == user].reset_index(drop=True)

    # First touch
    ft = first_touch_attribution(user_df)
    attribution_results['首次触达'][ft] += 1

    # Last touch
    lt = last_touch_attribution(user_df)
    attribution_results['末次触达'][lt] += 1

    # Linear
    lin = linear_attribution(user_df)
    for ch, w in lin.items():
        attribution_results['线性归因'][ch] += w

    # Time decay
    td = time_decay_attribution(user_df)
    for ch, w in td.items():
        attribution_results['时间衰减'][ch] += w

    # Position based
    pb = position_based_attribution(user_df)
    for ch, w in pb.items():
        attribution_results['位置归因'][ch] += w

# Build attribution heatmap data
attr_matrix = []
for model_name in ['首次触达', '末次触达', '线性归因', '时间衰减', '位置归因']:
    row = []
    for ch in all_channels:
        row.append(attribution_results[model_name].get(ch, 0))
    attr_matrix.append(row)

attr_matrix = np.array(attr_matrix)
attr_labels = ['首次触达', '末次触达', '线性归因', '时间衰减', '位置归因']

# Normalize each model to percentages
attr_pct = attr_matrix / attr_matrix.sum(axis=1, keepdims=True) * 100

print("\nAttribution Weights (%):")
for i, model in enumerate(attr_labels):
    print(f"  {model}: {dict(zip(all_channels, [round(v, 1) for v in attr_pct[i]]))}")


# ============================================================
# Chart 1: Channel Performance (Grouped Bar)
# ============================================================
print("\n[1/8] Generating channel_performance.png...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('渠道综合绩效分析', fontsize=18, fontweight='bold', y=1.02)

ch_labels = metrics_df['channel'].values
x = np.arange(len(ch_labels))
bar_width = 0.6
colors = [CHANNEL_COLORS.get(ch, '#95A5A6') for ch in ch_labels]

# Conversion Rate
conv_rates = metrics_df['conv_rate'].values * 100
bars1 = axes[0].bar(x, conv_rates, bar_width, color=colors, edgecolor='white', linewidth=1.2)
axes[0].set_title('转化率 (%)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('转化率 (%)', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(ch_labels, rotation=30, ha='right', fontsize=10)
for bar, val in zip(bars1, conv_rates):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, max(conv_rates) * 1.25)

# ROI
roi_vals = metrics_df['roi'].values
# Cap display for organic_search (zero cost)
roi_display = np.where(np.isinf(roi_vals), 0, roi_vals)
bars2 = axes[1].bar(x, roi_display, bar_width, color=colors, edgecolor='white', linewidth=1.2)
axes[1].set_title('投资回报率 (ROI)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('ROI', fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(ch_labels, rotation=30, ha='right', fontsize=10)
for bar, val in zip(bars2, roi_display):
    if val > 0:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[1].text(bar.get_x() + bar.get_width() / 2, 0.3,
                     'N/A\n(零成本)', ha='center', va='bottom', fontsize=9, color='gray')

# Cost per Conversion
cpc_vals = metrics_df['cost_per_conv'].values
# Handle NaN for channels with no conversions
cpc_display = np.where(np.isnan(cpc_vals), 0, cpc_vals)
bars3 = axes[2].bar(x, cpc_display, bar_width, color=colors, edgecolor='white', linewidth=1.2)
axes[2].set_title('每次转化成本 (元)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('成本 (元)', fontsize=12)
axes[2].set_xticks(x)
axes[2].set_xticklabels(ch_labels, rotation=30, ha='right', fontsize=10)
for bar, val in zip(bars3, cpc_display):
    if val > 0:
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'channel_performance.png'), bbox_inches='tight')
plt.close()
print("  -> Saved channel_performance.png")


# ============================================================
# Chart 2: Attribution Weights Comparison (Heatmap)
# ============================================================
print("[2/8] Generating attribution_weights_comparison.png...")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(attr_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=all_channels, yticklabels=attr_labels,
            linewidths=2, linecolor='white',
            cbar_kws={'label': '归因权重 (%)', 'shrink': 0.8},
            ax=ax)
ax.set_title('五种归因模型渠道权重对比 (%)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('渠道', fontsize=13)
ax.set_ylabel('归因模型', fontsize=13)
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'attribution_weights_comparison.png'), bbox_inches='tight')
plt.close()
print("  -> Saved attribution_weights_comparison.png")


# ============================================================
# Chart 3: Customer Journey Flow Diagram
# ============================================================
print("[3/8] Generating customer_journey_flow.png...")

# Count first-touch entries per channel
first_touch_counts = defaultdict(int)
for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user].sort_values('timestamp')
    first_touch_counts[user_df.iloc[0]['channel']] += 1

# Count last-touch exits per channel (conversions)
last_touch_conv = defaultdict(int)
for user in converted_users:
    user_df = df[df['user_id'] == user].sort_values('timestamp')
    conv_rows = user_df[user_df['conversion_status'] == 1]
    if len(conv_rows) > 0:
        conv_ch = conv_rows.iloc[0]['channel']
        last_touch_conv[conv_ch] += 1

# Build transition matrix
transition_counts = defaultdict(int)
for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user].sort_values('timestamp').reset_index(drop=True)
    for i in range(len(user_df) - 1):
        src = user_df.iloc[i]['channel']
        dst = user_df.iloc[i + 1]['channel']
        transition_counts[(src, dst)] += 1

fig, ax = plt.subplots(figsize=(16, 10))

x_left = 0
x_mid = 2.5
x_right = 5

# Channel node height proportional to total touches
total_touches_per_ch = df.groupby('channel').size().to_dict()
max_touches = max(total_touches_per_ch.values())

# Compute y positions and heights for channel nodes
ch_node_info = {}
y_cursor = 0
gap = 0.5
for ch in all_channels:
    h = total_touches_per_ch.get(ch, 1) / max_touches * 2.0
    ch_node_info[ch] = {'y': y_cursor, 'h': h, 'y_center': y_cursor + h / 2}
    y_cursor += h + gap

total_height = y_cursor

# Draw channel nodes
for ch in all_channels:
    info = ch_node_info[ch]
    rect = plt.Rectangle((x_mid - 0.4, info['y']), 0.8, info['h'],
                          facecolor=CHANNEL_COLORS.get(ch, '#95A5A6'),
                          edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x_mid, info['y_center'], ch.replace('_', '\n'),
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Draw START node
start_h = total_height
rect_start = plt.Rectangle((x_left - 0.4, 0), 0.8, start_h,
                             facecolor='#34495E', edgecolor='white', linewidth=2, alpha=0.85)
ax.add_patch(rect_start)
ax.text(x_left, start_h / 2, f'用户\n进入\n(n={df["user_id"].nunique()})',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Draw CONVERT node
conv_h = total_height * 0.6
rect_conv = plt.Rectangle((x_right - 0.4, (total_height - conv_h) / 2), 0.8, conv_h,
                            facecolor='#E74C3C', edgecolor='white', linewidth=2, alpha=0.85)
ax.add_patch(rect_conv)
ax.text(x_right, total_height / 2, f'转化\n完成\n(n={len(converted_users)})',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

from matplotlib.path import Path

# Draw flows: START -> channels (first touch)
for ch in all_channels:
    count = first_touch_counts.get(ch, 0)
    if count > 0:
        info = ch_node_info[ch]
        flow_width = count / df['user_id'].nunique() * 5
        y_src = info['y_center']
        y_dst = info['y_center']
        verts = [
            (x_left + 0.4, y_src),
            (x_left + 1.2, y_src),
            (x_mid - 0.4, y_dst),
            (x_mid - 0.4, y_dst),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = matplotlib.patches.PathPatch(
            path, facecolor='none', edgecolor=CHANNEL_COLORS.get(ch, '#95A5A6'),
            lw=flow_width, alpha=0.4)
        ax.add_patch(patch)
        ax.text((x_left + x_mid) / 2, y_src + 0.3, str(count),
                ha='center', va='bottom', fontsize=9,
                color=CHANNEL_COLORS.get(ch, '#333'))

# Draw flows: channels -> CONVERT (conversions)
for ch in all_channels:
    count = last_touch_conv.get(ch, 0)
    if count > 0:
        info = ch_node_info[ch]
        flow_width = count / len(converted_users) * 5
        verts = [
            (x_mid + 0.4, info['y_center']),
            (x_mid + 1.2, info['y_center']),
            (x_right - 0.4, total_height / 2),
            (x_right - 0.4, total_height / 2),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = matplotlib.patches.PathPatch(
            path, facecolor='none', edgecolor=CHANNEL_COLORS.get(ch, '#95A5A6'),
            lw=flow_width, alpha=0.4)
        ax.add_patch(patch)
        ax.text((x_mid + x_right) / 2, info['y_center'] + 0.3, str(count),
                ha='center', va='bottom', fontsize=9,
                color=CHANNEL_COLORS.get(ch, '#333'))

# Draw inter-channel transition arrows
max_trans = max(transition_counts.values()) if transition_counts else 1
for (src, dst), count in transition_counts.items():
    if src != dst and count > 0:
        src_info = ch_node_info[src]
        dst_info = ch_node_info[dst]
        lw = count / max_trans * 3 + 0.3
        # Draw curved arrow alongside the node column
        ax.annotate('',
                    xy=(x_mid + 0.5, dst_info['y_center']),
                    xytext=(x_mid + 0.5, src_info['y_center']),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D',
                                   lw=lw, alpha=0.35,
                                   connectionstyle='arc3,rad=0.3'))

# Stage labels
ax.text(x_left, total_height + 1.2, '第一步', ha='center', fontsize=13, fontweight='bold')
ax.text(x_mid, total_height + 1.2, '渠道触达', ha='center', fontsize=13, fontweight='bold')
ax.text(x_right, total_height + 1.2, '转化触点', ha='center', fontsize=13, fontweight='bold')

ax.set_xlim(-1.5, 6)
ax.set_ylim(-1.5, total_height + 2.5)
ax.set_title('用户转化旅程流程图', fontsize=18, fontweight='bold', pad=15)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'customer_journey_flow.png'), bbox_inches='tight')
plt.close()
print("  -> Saved customer_journey_flow.png")


# ============================================================
# Chart 4: Conversion Funnel
# ============================================================
print("[4/8] Generating conversion_funnel.png...")

fig, ax = plt.subplots(figsize=(14, 8))

# Build funnel data per channel
funnel_data = []
for ch in all_channels:
    ch_data = df[df['channel'] == ch]
    total_users_ch = ch_data['user_id'].nunique()
    engaged_users = set()
    for user in ch_data['user_id'].unique():
        user_all = df[df['user_id'] == user]
        if len(user_all) > 1:
            engaged_users.add(user)
    n_engaged = len(engaged_users)

    converted_touched = set()
    for user in converted_users:
        user_all = df[df['user_id'] == user]
        if ch in user_all['channel'].values:
            converted_touched.add(user)
    n_converted = len(converted_touched)

    funnel_data.append({
        'channel': ch,
        'total_users': total_users_ch,
        'engaged': n_engaged,
        'converted': n_converted,
    })

funnel_df = pd.DataFrame(funnel_data)

# Draw grouped funnel bars (horizontal)
stages = ['触达用户', '深度互动', '完成转化']
y_pos = np.arange(len(all_channels))
bar_h = 0.25

colors_funnel = ['#3498DB', '#F39C12', '#2ECC71']

for i, stage in enumerate(stages):
    col = ['total_users', 'engaged', 'converted'][i]
    vals = funnel_df[col].values
    bars = ax.barh(y_pos + i * bar_h, vals, bar_h,
                    color=colors_funnel[i], label=stage,
                    edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(val), ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos + bar_h)
ax.set_yticklabels(all_channels, fontsize=11)
ax.set_xlabel('用户数', fontsize=13)
ax.set_title('渠道转化漏斗分析', fontsize=18, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'conversion_funnel.png'), bbox_inches='tight')
plt.close()
print("  -> Saved conversion_funnel.png")


# ============================================================
# Chart 5: Channel Synergy Heatmap (Co-occurrence)
# ============================================================
print("[5/8] Generating channel_synergy_heatmap.png...")

# Build co-occurrence matrix for converted user journeys
cooccurrence = np.zeros((len(all_channels), len(all_channels)))

for user in converted_users:
    user_channels = df[df['user_id'] == user]['channel'].unique()
    for i, ch1 in enumerate(user_channels):
        for j, ch2 in enumerate(user_channels):
            idx1 = all_channels.index(ch1)
            idx2 = all_channels.index(ch2)
            cooccurrence[idx1, idx2] += 1

fig, ax = plt.subplots(figsize=(10, 8))

# Create a mask for the upper triangle (symmetric matrix)
mask = np.triu(np.ones_like(cooccurrence, dtype=bool), k=1)

sns.heatmap(cooccurrence, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=all_channels, yticklabels=all_channels,
            linewidths=2, linecolor='white',
            mask=mask,
            cbar_kws={'label': '共现次数', 'shrink': 0.8},
            ax=ax)
ax.set_title('转化用户渠道协同热力图\n(渠道在用户旅程中的共现频次)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('渠道', fontsize=13)
ax.set_ylabel('渠道', fontsize=13)
ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'channel_synergy_heatmap.png'), bbox_inches='tight')
plt.close()
print("  -> Saved channel_synergy_heatmap.png")


# ============================================================
# Chart 6: Timing Analysis (Box Plot - Position in Journey)
# ============================================================
print("[6/8] Generating timing_analysis.png...")

fig, ax = plt.subplots(figsize=(14, 7))

# Collect relative positions per channel
timing_data = {ch: [] for ch in all_channels}
for ch in all_channels:
    ch_positions = df[df['channel'] == ch]['relative_position'].values
    timing_data[ch] = ch_positions

# Create box plot
bp_data = [timing_data[ch] for ch in all_channels]
bp = ax.boxplot(bp_data, patch_artist=True, widths=0.6,
                medianprops=dict(color='#E74C3C', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=6))

for patch, ch in zip(bp['boxes'], all_channels):
    patch.set_facecolor(CHANNEL_COLORS.get(ch, '#95A5A6'))
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')
    patch.set_linewidth(1.5)

# Add scatter overlay (jittered)
np.random.seed(42)
for i, ch in enumerate(all_channels):
    positions = timing_data[ch]
    jitter = np.random.normal(0, 0.04, len(positions))
    ax.scatter(np.full(len(positions), i + 1) + jitter, positions,
               alpha=0.4, s=30, color=CHANNEL_COLORS.get(ch, '#95A5A6'),
               edgecolors='white', linewidth=0.5, zorder=5)

ax.set_xticklabels(all_channels, rotation=30, ha='right', fontsize=11)
ax.set_ylabel('相对旅程位置 (0=开始, 1=结束)', fontsize=13)
ax.set_title('渠道触达时机分析\n(各渠道在用户旅程中的位置分布)', fontsize=16, fontweight='bold', pad=15)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='旅程中点')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(-0.05, 1.05)

# Add interpretation annotations
ax.text(0.02, 0.02, '靠近0 = 渠道多出现在旅程早期 (认知阶段)\n靠近1 = 渠道多出现在旅程后期 (决策阶段)',
        transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'timing_analysis.png'), bbox_inches='tight')
plt.close()
print("  -> Saved timing_analysis.png")


# ============================================================
# Chart 7: ROI Analysis (Scatter + Bubble)
# ============================================================
print("[7/8] Generating roi_analysis.png...")

fig, ax = plt.subplots(figsize=(12, 8))

for _, row in metrics_df.iterrows():
    ch = row['channel']
    cost = row['total_cost']
    value = row['total_value']
    convs = row['conversions']
    color = CHANNEL_COLORS.get(ch, '#95A5A6')

    # Skip organic_search for meaningful display (zero cost)
    if cost == 0:
        ax.annotate(f'{ch}\n(零成本渠道)',
                    xy=(10, value),
                    xytext=(150, value + 200),
                    fontsize=10, color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        continue

    bubble_size = max(convs * 80, 100)
    ax.scatter(cost, value, s=bubble_size, c=color, alpha=0.7,
               edgecolors='white', linewidth=2, zorder=5)

    roi_val = (value - cost) / cost
    ax.annotate(f'{ch}\nROI: {roi_val:.1f}x\n转化: {int(convs)}次',
                xy=(cost, value),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2),
                arrowprops=dict(arrowstyle='->', color=color, lw=1))

# Add break-even line
max_cost = metrics_df['total_cost'].max()
ax.plot([0, max_cost * 1.2], [0, max_cost * 1.2], 'k--', alpha=0.3, lw=1.5, label='盈亏平衡线')

ax.set_xlabel('总成本 (元)', fontsize=13)
ax.set_ylabel('总收入 (元)', fontsize=13)
ax.set_title('渠道ROI分析\n(气泡大小 = 转化次数)', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roi_analysis.png'), bbox_inches='tight')
plt.close()
print("  -> Saved roi_analysis.png")


# ============================================================
# Chart 8: Conversion Value Distribution (Violin/Box)
# ============================================================
print("[8/8] Generating conversion_value_distribution.png...")

fig, ax = plt.subplots(figsize=(14, 7))

# Get conversion values by converting channel
conv_by_channel = {}
for user in converted_users:
    user_df = df[(df['user_id'] == user) & (df['conversion_status'] == 1)]
    for _, row in user_df.iterrows():
        ch = row['channel']
        val = row['conversion_value']
        if ch not in conv_by_channel:
            conv_by_channel[ch] = []
        conv_by_channel[ch].append(val)

# Prepare data for box plot
plot_channels = [ch for ch in all_channels if ch in conv_by_channel and len(conv_by_channel[ch]) > 0]
plot_data = [conv_by_channel[ch] for ch in plot_channels]

if len(plot_data) > 0:
    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='#E74C3C', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    for patch, ch in zip(bp['boxes'], plot_channels):
        patch.set_facecolor(CHANNEL_COLORS.get(ch, '#95A5A6'))
        patch.set_alpha(0.6)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)

    # Overlay individual data points with jitter
    np.random.seed(42)
    for i, ch in enumerate(plot_channels):
        values = conv_by_channel[ch]
        jitter = np.random.normal(0, 0.05, len(values))
        ax.scatter(np.full(len(values), i + 1) + jitter, values,
                   s=60, color=CHANNEL_COLORS.get(ch, '#95A5A6'),
                   edgecolors='white', linewidth=1, zorder=5, alpha=0.8)

    # Add mean markers
    for i, ch in enumerate(plot_channels):
        mean_val = np.mean(conv_by_channel[ch])
        ax.scatter(i + 1, mean_val, marker='D', s=80, color='#E74C3C',
                   edgecolors='white', linewidth=1, zorder=6)

ax.set_xticklabels(plot_channels, rotation=30, ha='right', fontsize=11)
ax.set_ylabel('转化价值 (元)', fontsize=13)
ax.set_title('各渠道转化价值分布\n(菱形=均值, 红线=中位数)', fontsize=16, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add summary stats as text
stats_text = ''
for ch in plot_channels:
    vals = conv_by_channel[ch]
    stats_text += f'{ch}: 均值={np.mean(vals):.0f}, 中位数={np.median(vals):.0f}, n={len(vals)}\n'
ax.text(0.98, 0.98, stats_text.strip(), transform=ax.transAxes,
        fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'conversion_value_distribution.png'), bbox_inches='tight')
plt.close()
print("  -> Saved conversion_value_distribution.png")


print("\n" + "=" * 60)
print("All 8 visualizations generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)
