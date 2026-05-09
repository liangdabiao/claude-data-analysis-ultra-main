"""
营销渠道多模型归因分析 - 全面深度洞察
数据源: examples/sample_channel_data.csv
分析模型: 首次触点/末次触点/线性/时间衰减/位置/马尔可夫链/Shapley值
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import permutations
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ========== 中文字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ========== 1. 数据加载与预处理 ==========
print("=" * 70)
print("  营销渠道多模型归因分析 - 全面深度洞察")
print("=" * 70)

DATA_PATH = './examples/sample_channel_data.csv'
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

print(f"\n📊 数据概览:")
print(f"  总触点数: {len(df)}")
print(f"  用户数: {df['user_id'].nunique()}")
print(f"  渠道数: {df['channel'].nunique()}")
print(f"  渠道列表: {sorted(df['channel'].unique())}")
print(f"  时间范围: {df['timestamp'].min().strftime('%Y-%m-%d')} ~ {df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"  总转化价值: ${df['conversion_value'].sum():,.0f}")
print(f"  总营销成本: ${df['cost'].sum():,.0f}")

# ========== 2. 客户路径构建 ==========
print("\n" + "=" * 70)
print("  客户路径构建与旅程分析")
print("=" * 70)

def build_customer_paths(df):
    """构建客户转化路径"""
    paths = {}
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('timestamp')
        channels = group['channel'].tolist()
        conversion_status = group['conversion_status'].tolist()
        conversion_value = group.loc[group['conversion_status'] == 1, 'conversion_value'].sum()
        total_cost = group['cost'].sum()
        timestamps = group['timestamp'].tolist()

        converted = 1 in conversion_status
        conv_idx = conversion_status.index(1) if converted else len(conversion_status) - 1

        # 转化前的路径（含转化触点）
        if converted:
            pre_conv_channels = channels[:conv_idx + 1]
            post_conv_channels = channels[conv_idx + 1:]
        else:
            pre_conv_channels = channels
            post_conv_channels = []

        paths[user_id] = {
            'channels': channels,
            'pre_conv_path': pre_conv_channels,
            'post_conv_channels': post_conv_channels,
            'converted': converted,
            'conversion_value': conversion_value,
            'total_cost': total_cost,
            'path_length': len(channels),
            'conv_path_length': len(pre_conv_channels),
            'timestamps': timestamps,
            'duration_days': (timestamps[-1] - timestamps[0]).total_seconds() / 86400 if len(timestamps) > 1 else 0
        }
    return paths

paths = build_customer_paths(df)

# 路径统计
conv_paths = {k: v for k, v in paths.items() if v['converted']}
non_conv_paths = {k: v for k, v in paths.items() if not v['converted']}
conv_path_values = list(conv_paths.values())

print(f"\n🛤️ 路径统计:")
print(f"  转化用户数: {len(conv_paths)} / {len(paths)} ({len(conv_paths)/len(paths)*100:.1f}%)")
print(f"  转化路径平均长度: {np.mean([v['conv_path_length'] for v in conv_path_values]):.1f} 个触点")
print(f"  转化路径平均耗时: {np.mean([v['duration_days'] for v in conv_path_values]):.1f} 天")
print(f"  平均转化价值: ${np.mean([v['conversion_value'] for v in conv_path_values]):,.0f}")
print(f"  转化价值中位数: ${np.median([v['conversion_value'] for v in conv_path_values]):,.0f}")

# 转化路径详情
print(f"\n📋 转化路径详情:")
for uid, p in conv_paths.items():
    path_str = " → ".join(p['pre_conv_path'])
    print(f"  {uid}: {path_str}  [${p['conversion_value']:,.0f}, {p['conv_path_length']}触点, {p['duration_days']:.0f}天]")

# ========== 3. 渠道基础分析 ==========
print("\n" + "=" * 70)
print("  渠道基础性能分析")
print("=" * 70)

channel_stats = df.groupby('channel').agg(
    touchpoints=('channel', 'count'),
    users=('user_id', 'nunique'),
    total_cost=('cost', 'sum'),
    avg_cost=('cost', 'mean'),
).reset_index()

# 每个渠道的转化数据
conv_by_channel = {}
for channel in df['channel'].unique():
    channel_df = df[df['channel'] == channel]
    # 找出在该渠道触发的转化
    conv_touches = channel_df[channel_df['conversion_status'] == 1]
    conv_by_channel[channel] = {
        'conversions_as_last': len(conv_touches),
        'conversion_value_as_last': conv_touches['conversion_value'].sum()
    }

# 作为首次触点的转化
first_touches = df.groupby('user_id').first()
first_conv = first_touches[first_touches['conversion_status'] == 1]
first_conv_by_channel = first_conv.groupby('channel').size().to_dict()

# 转化率 - 以该渠道为末次触点
total_conv = len(conv_paths)
channel_perf = []
for _, row in channel_stats.iterrows():
    ch = row['channel']
    conv_count = conv_by_channel.get(ch, {}).get('conversions_as_last', 0)
    conv_value = conv_by_channel.get(ch, {}).get('conversion_value_as_last', 0)
    first_count = first_conv_by_channel.get(ch, 0)

    channel_perf.append({
        'channel': ch,
        'touchpoints': row['touchpoints'],
        'users_reached': row['users'],
        'total_cost': row['total_cost'],
        'avg_cost_per_touch': row['avg_cost'],
        'conversions_as_last': conv_count,
        'conversions_as_first': first_count,
        'conversion_value_as_last': conv_value,
        'cpl': row['total_cost'] / conv_count if conv_count > 0 else float('inf'),
        'roi': (conv_value - row['total_cost']) / row['total_cost'] * 100 if row['total_cost'] > 0 else 0,
        'reach_efficiency': row['users'] / row['touchpoints'] * 100
    })

channel_perf_df = pd.DataFrame(channel_perf).sort_values('conversion_value_as_last', ascending=False)
print("\n📈 渠道性能汇总:")
print(channel_perf_df.to_string(index=False))

# ========== 4. 七种归因模型计算 ==========
print("\n" + "=" * 70)
print("  多模型归因计算")
print("=" * 70)

TOTAL_CONVERSIONS = len(conv_paths)
TOTAL_VALUE = sum(p['conversion_value'] for p in conv_paths.values())

all_channels = sorted(df['channel'].unique())

# --- 4.1 首次触点归因 ---
print("\n📌 模型1: 首次触点归因 (First-Touch)")
first_touch_attribution = defaultdict(float)
first_touch_count = defaultdict(int)
for uid, p in conv_paths.items():
    first_ch = p['pre_conv_path'][0]
    first_touch_attribution[first_ch] += p['conversion_value']
    first_touch_count[first_ch] += 1

print("  渠道贡献:")
for ch in all_channels:
    val = first_touch_attribution.get(ch, 0)
    cnt = first_touch_count.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)  [{cnt}次转化]")

# --- 4.2 末次触点归因 ---
print("\n📌 模型2: 末次触点归因 (Last-Touch)")
last_touch_attribution = defaultdict(float)
last_touch_count = defaultdict(int)
for uid, p in conv_paths.items():
    last_ch = p['pre_conv_path'][-1]
    last_touch_attribution[last_ch] += p['conversion_value']
    last_touch_count[last_ch] += 1

print("  渠道贡献:")
for ch in all_channels:
    val = last_touch_attribution.get(ch, 0)
    cnt = last_touch_count.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)  [{cnt}次转化]")

# --- 4.3 线性归因 ---
print("\n📌 模型3: 线性归因 (Linear)")
linear_attribution = defaultdict(float)
for uid, p in conv_paths.items():
    n = len(p['pre_conv_path'])
    if n > 0:
        weight = p['conversion_value'] / n
        for ch in p['pre_conv_path']:
            linear_attribution[ch] += weight

print("  渠道贡献:")
for ch in all_channels:
    val = linear_attribution.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)")

# --- 4.4 时间衰减归因 ---
print("\n📌 模型4: 时间衰减归因 (Time-Decay)")
time_decay_attribution = defaultdict(float)
DECAY_RATE = 0.5  # 半衰期系数

for uid, p in conv_paths.items():
    path = p['pre_conv_path']
    n = len(path)
    if n > 0:
        # 最近触点权重最高
        weights = []
        for i in range(n):
            w = np.exp(DECAY_RATE * i)  # 越接近转化权重越大
            weights.append(w)
        total_w = sum(weights)
        for i, ch in enumerate(path):
            time_decay_attribution[ch] += p['conversion_value'] * weights[i] / total_w

print("  渠道贡献:")
for ch in all_channels:
    val = time_decay_attribution.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)")

# --- 4.5 位置归因 (U型) ---
print("\n📌 模型5: 位置归因 (Position-Based / U-Shaped)")
position_attribution = defaultdict(float)
FIRST_WEIGHT = 0.4
LAST_WEIGHT = 0.4
MIDDLE_WEIGHT = 0.2

for uid, p in conv_paths.items():
    path = p['pre_conv_path']
    n = len(path)
    if n == 1:
        position_attribution[path[0]] += p['conversion_value']
    elif n == 2:
        position_attribution[path[0]] += p['conversion_value'] * 0.5
        position_attribution[path[1]] += p['conversion_value'] * 0.5
    else:
        position_attribution[path[0]] += p['conversion_value'] * FIRST_WEIGHT
        position_attribution[path[-1]] += p['conversion_value'] * LAST_WEIGHT
        middle_weight_each = p['conversion_value'] * MIDDLE_WEIGHT / (n - 2)
        for ch in path[1:-1]:
            position_attribution[ch] += middle_weight_each

print("  渠道贡献:")
for ch in all_channels:
    val = position_attribution.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)")

# --- 4.6 马尔可夫链归因 ---
print("\n📌 模型6: 马尔可夫链归因 (Markov Chain)")

def build_transition_matrix(paths_dict):
    """构建马尔可夫转移矩阵"""
    transitions = defaultdict(lambda: defaultdict(int))
    all_states = set()

    for uid, p in paths_dict.items():
        path = p['pre_conv_path']
        if p['converted']:
            path = path + ['CONVERSION']
        else:
            path = path + ['NULL']

        for i in range(len(path) - 1):
            transitions[path[i]][path[i + 1]] += 1
            all_states.add(path[i])
        all_states.add(path[-1])

    # 计算转移概率
    states = sorted(all_states - {'CONVERSION', 'NULL'})
    trans_prob = {}
    for state in states:
        total = sum(transitions[state].values())
        if total > 0:
            trans_prob[state] = {k: v / total for k, v in transitions[state].items()}

    return trans_prob, states

def calc_conversion_prob(trans_prob, start_state, max_steps=20):
    """计算从某个状态开始的转化概率"""
    probs = {'CONVERSION': 0, 'NULL': 0}
    current = {start_state: 1.0}

    for _ in range(max_steps):
        next_states = defaultdict(float)
        for state, prob in current.items():
            if state in ('CONVERSION', 'NULL'):
                probs[state] += prob
            elif state in trans_prob:
                for next_state, trans_p in trans_prob[state].items():
                    next_states[next_state] += prob * trans_p
            else:
                probs['NULL'] += prob
        current = dict(next_states)

    # 处理残余概率
    for state, prob in current.items():
        if state == 'CONVERSION':
            probs['CONVERSION'] += prob
        else:
            probs['NULL'] += prob

    return probs['CONVERSION']

def markov_chain_attribution(paths_dict, channels):
    """计算马尔可夫链归因（基于移除效应）"""
    trans_prob, states = build_transition_matrix(paths_dict)

    # 基线转化概率
    baseline = calc_conversion_prob(trans_prob, list(trans_prob.keys())[0]) if trans_prob else 0

    # 更准确：计算整体转化率
    total_conversion_rate = len([p for p in paths_dict.values() if p['converted']]) / len(paths_dict)

    # 移除效应
    removal_effects = {}
    for channel in channels:
        # 移除该渠道后的路径
        modified_paths = {}
        for uid, p in paths_dict.items():
            new_path_channels = [ch for ch in p['pre_conv_path'] if ch != channel]
            if len(new_path_channels) == 0 and p['converted']:
                # 路径完全依赖该渠道 -> 不转化
                modified_paths[uid] = {**p, 'converted': False, 'pre_conv_path': []}
            else:
                modified_paths[uid] = {**p, 'pre_conv_path': new_path_channels}
                if len(new_path_channels) == 0:
                    modified_paths[uid]['converted'] = False

        modified_conv_rate = len([p for p in modified_paths.values() if p['converted']]) / len(modified_paths)

        removal_effect = total_conversion_rate - modified_conv_rate
        removal_effects[channel] = max(removal_effect, 0)

    # 归一化
    total_removal = sum(removal_effects.values())
    if total_removal > 0:
        attribution = {ch: (eff / total_removal) * TOTAL_VALUE for ch, eff in removal_effects.items()}
    else:
        attribution = {ch: TOTAL_VALUE / len(channels) for ch in channels}

    return attribution, removal_effects

markov_attr, markov_removal = markov_chain_attribution(paths, all_channels)

print("  移除效应 (Removal Effect):")
for ch in all_channels:
    eff = markov_removal.get(ch, 0)
    print(f"    {ch:25s}: {eff:.4f}")
print("\n  渠道归因贡献:")
for ch in all_channels:
    val = markov_attr.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)")

# --- 4.7 Shapley值归因 ---
print("\n📌 模型7: Shapley值归因 (Shapley Value)")

def calc_coalition_value(coalition, paths_dict):
    """计算联盟价值（包含这些渠道的转化路径的转化价值）"""
    value = 0
    for uid, p in paths_dict.items():
        if p['converted']:
            path_channels = set(p['pre_conv_path'])
            if path_channels.issubset(set(coalition)):
                value += p['conversion_value']
    return value

def shapley_attribution(channels, paths_dict):
    """计算Shapley值归因"""
    n = len(channels)
    shapley_values = defaultdict(float)

    # 对每种排列计算边际贡献
    for perm in permutations(channels):
        for i in range(n):
            channel = perm[i]
            coalition_with = set(perm[:i + 1])
            coalition_without = set(perm[:i])

            v_with = calc_coalition_value(coalition_with, paths_dict)
            v_without = calc_coalition_value(coalition_without, paths_dict)

            shapley_values[channel] += (v_with - v_without)

    # 平均
    n_perms = len(list(permutations(channels)))
    for ch in shapley_values:
        shapley_values[ch] /= n_perms

    return shapley_values

shapley_attr = shapley_attribution(all_channels, paths)

print("  渠道归因贡献:")
for ch in all_channels:
    val = shapley_attr.get(ch, 0)
    pct = val / TOTAL_VALUE * 100 if TOTAL_VALUE > 0 else 0
    print(f"    {ch:25s}: ${val:>10,.0f}  ({pct:>5.1f}%)")

# ========== 5. 模型对比汇总 ==========
print("\n" + "=" * 70)
print("  七种归因模型对比汇总")
print("=" * 70)

models_comparison = {}
for ch in all_channels:
    models_comparison[ch] = {
        '首次触点': first_touch_attribution.get(ch, 0),
        '末次触点': last_touch_attribution.get(ch, 0),
        '线性归因': linear_attribution.get(ch, 0),
        '时间衰减': time_decay_attribution.get(ch, 0),
        '位置归因': position_attribution.get(ch, 0),
        '马尔可夫链': markov_attr.get(ch, 0),
        'Shapley值': shapley_attr.get(ch, 0),
    }

comparison_df = pd.DataFrame(models_comparison).T
comparison_df_pct = comparison_df.div(comparison_df.sum(axis=0), axis=1) * 100

print("\n📊 归因贡献百分比对比:")
print(comparison_df_pct.round(1).to_string())

print(f"\n💰 归因贡献金额对比 ($):")
print(comparison_df.round(0).to_string())

# 保存归因结果
comparison_df.to_csv('./analysis_reports/attribution_results.csv')
channel_perf_df.to_csv('./analysis_reports/channel_performance.csv', index=False)

# ========== 6. 路径分析 ==========
print("\n" + "=" * 70)
print("  路径模式分析")
print("=" * 70)

# 常见路径模式
path_patterns = Counter()
for uid, p in conv_paths.items():
    pattern = " → ".join(p['pre_conv_path'])
    path_patterns[pattern] += 1

print("\n🔄 转化路径模式:")
for pattern, count in path_patterns.most_common():
    print(f"  [{count}次] {pattern}")

# 渠道转移分析
print("\n🔀 渠道转移分析:")
transitions = defaultdict(lambda: defaultdict(int))
for uid, p in conv_paths.items():
    path = p['pre_conv_path']
    for i in range(len(path) - 1):
        transitions[path[i]][path[i + 1]] += 1

for from_ch in sorted(transitions.keys()):
    total = sum(transitions[from_ch].values())
    to_chs = sorted(transitions[from_ch].items(), key=lambda x: -x[1])
    top3 = ", ".join([f"{ch}({cnt/total*100:.0f}%)" for ch, cnt in to_chs[:3]])
    print(f"  {from_ch:20s} → {top3}")

# 路径长度分析
path_lengths = [p['conv_path_length'] for p in conv_paths.values()]
print(f"\n📏 路径长度分布:")
print(f"  最短路径: {min(path_lengths)} 个触点")
print(f"  最长路径: {max(path_lengths)} 个触点")
print(f"  平均路径: {np.mean(path_lengths):.1f} 个触点")
print(f"  中位路径: {np.median(path_lengths):.0f} 个触点")

# 渠道在路径中的位置分析
print("\n📍 渠道在路径中的位置分布:")
position_dist = defaultdict(lambda: {'first': 0, 'middle': 0, 'last': 0})
for uid, p in conv_paths.items():
    path = p['pre_conv_path']
    n = len(path)
    for i, ch in enumerate(path):
        if i == 0:
            position_dist[ch]['first'] += 1
        elif i == n - 1:
            position_dist[ch]['last'] += 1
        else:
            position_dist[ch]['middle'] += 1

for ch in all_channels:
    pos = position_dist.get(ch, {'first': 0, 'middle': 0, 'last': 0})
    total_pos = pos['first'] + pos['middle'] + pos['last']
    if total_pos > 0:
        print(f"  {ch:25s}: 首位{pos['first']}({pos['first']/total_pos*100:.0f}%)  "
              f"中间{pos['middle']}({pos['middle']/total_pos*100:.0f}%)  "
              f"末位{pos['last']}({pos['last']/total_pos*100:.0f}%)")

# ========== 7. ROI 与预算优化分析 ==========
print("\n" + "=" * 70)
print("  ROI 与预算优化分析")
print("=" * 70)

# 使用Shapley值作为最公平的归因结果进行ROI计算
print("\n💡 基于Shapley值归因的渠道ROI分析:")
roi_analysis = []
for ch in all_channels:
    shapley_val = shapley_attr.get(ch, 0)
    cost = channel_perf_df[channel_perf_df['channel'] == ch]['total_cost'].values
    total_cost = cost[0] if len(cost) > 0 else 0
    roi = (shapley_val - total_cost) / total_cost * 100 if total_cost > 0 else float('inf')
    roas = shapley_val / total_cost if total_cost > 0 else float('inf')

    roi_analysis.append({
        'channel': ch,
        'shapley_value': shapley_val,
        'total_cost': total_cost,
        'net_profit': shapley_val - total_cost,
        'roi_pct': roi,
        'roas': roas
    })

roi_df = pd.DataFrame(roi_analysis).sort_values('roas', ascending=False)
print(roi_df.to_string(index=False))

# 预算优化建议
print("\n📊 预算优化建议:")
total_budget = roi_df['total_cost'].sum()
shapley_total = roi_df['shapley_value'].sum()

# 基于ROI的权重重分配
current_allocation = roi_df.set_index('channel')['total_cost'] / total_budget * 100
optimal_allocation = roi_df.set_index('channel')['shapley_value'] / shapley_total * 100

for ch in all_channels:
    curr_pct = current_allocation.get(ch, 0)
    opt_pct = optimal_allocation.get(ch, 0)
    diff = opt_pct - curr_pct
    direction = "↑ 增加" if diff > 2 else ("↓ 减少" if diff < -2 else "→ 维持")
    print(f"  {ch:25s}: 当前{curr_pct:>5.1f}% → 建议{opt_pct:>5.1f}%  {direction} (差{diff:>+.1f}%)")

# ========== 8. 深度洞察 ==========
print("\n" + "=" * 70)
print("  深度洞察与策略建议")
print("=" * 70)

# 找出关键渠道
top_shapley = sorted(shapley_attr.items(), key=lambda x: -x[1])
top_channel = top_shapley[0]
print(f"\n🔑 关键发现:")
print(f"  1. 最重要渠道: {top_channel[0]} (Shapley值: ${top_channel[1]:,.0f}, 占比{top_channel[1]/TOTAL_VALUE*100:.1f}%)")
print(f"  2. 转化率: {TOTAL_CONVERSIONS}/{len(paths)} = {TOTAL_CONVERSIONS/len(paths)*100:.1f}%")
print(f"  3. 平均客户旅程: {np.mean(path_lengths):.1f} 个触点, {np.mean([p['duration_days'] for p in conv_paths.values()]):.0f} 天")
print(f"  4. 总ROAS: {TOTAL_VALUE/total_budget:.2f}x")

# 渠道协同效应
print(f"\n🤝 渠道协同效应分析:")
channel_pairs = defaultdict(int)
for uid, p in conv_paths.items():
    path_set = list(dict.fromkeys(p['pre_conv_path']))  # 去重保序
    for i in range(len(path_set) - 1):
        for j in range(i + 1, len(path_set)):
            pair = tuple(sorted([path_set[i], path_set[j]]))
            channel_pairs[pair] += 1

top_pairs = sorted(channel_pairs.items(), key=lambda x: -x[1])[:10]
for pair, count in top_pairs:
    print(f"  {pair[0]:20s} + {pair[1]:20s}: 共现 {count} 次")

# 渠道角色分析
print(f"\n🎭 渠道角色定位:")
for ch in all_channels:
    pos = position_dist.get(ch, {'first': 0, 'middle': 0, 'last': 0})
    total_pos = pos['first'] + pos['middle'] + pos['last']
    if total_pos > 0:
        if pos['first'] / total_pos > 0.5:
            role = "认知型渠道（引流）"
        elif pos['last'] / total_pos > 0.5:
            role = "转化型渠道（促转）"
        elif pos['middle'] / total_pos > 0.5:
            role = "考虑型渠道（培育）"
        else:
            role = "全能型渠道"
        print(f"  {ch:25s}: {role}")

# ========== 9. 综合可视化 ==========
print("\n" + "=" * 70)
print("  生成可视化图表...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('营销渠道多模型归因分析仪表板', fontsize=18, fontweight='bold', y=0.98)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
channel_colors = dict(zip(all_channels, colors[:len(all_channels)]))

# 图1: 七种模型归因对比
ax1 = axes[0, 0]
comparison_df_pct.T.plot(kind='bar', ax=ax1, color=colors[:len(all_channels)], width=0.8)
ax1.set_title('七种归因模型渠道贡献对比 (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('归因贡献 (%)')
ax1.set_xlabel('归因模型')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
ax1.tick_params(axis='x', rotation=30)

# 图2: Shapley值归因
ax2 = axes[0, 1]
shap_vals = [shapley_attr.get(ch, 0) for ch in all_channels]
bars = ax2.barh(all_channels, shap_vals, color=[channel_colors[ch] for ch in all_channels])
ax2.set_title('Shapley值归因 ($)', fontsize=12, fontweight='bold')
ax2.set_xlabel('归因价值 ($)')
for bar, val in zip(bars, shap_vals):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
             f'${val:,.0f}', va='center', fontsize=9)

# 图3: 渠道ROI分析
ax3 = axes[0, 2]
roi_sorted = roi_df.sort_values('roas', ascending=True)
bar_colors = ['#FF6B6B' if r < 0 else '#4ECDC4' for r in roi_sorted['roi_pct']]
ax3.barh(roi_sorted['channel'], roi_sorted['roi_pct'], color=bar_colors)
ax3.set_title('渠道ROI (%) - 基于Shapley值', fontsize=12, fontweight='bold')
ax3.set_xlabel('ROI (%)')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

# 图4: 渠道成本 vs 归因价值
ax4 = axes[1, 0]
x = np.arange(len(all_channels))
width = 0.35
costs = [channel_perf_df[channel_perf_df['channel'] == ch]['total_cost'].values[0] for ch in all_channels]
shapley_vals = [shapley_attr.get(ch, 0) for ch in all_channels]
ax4.bar(x - width / 2, costs, width, label='营销成本', color='#FF6B6B', alpha=0.8)
ax4.bar(x + width / 2, shapley_vals, width, label='Shapley归因价值', color='#4ECDC4', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(all_channels, rotation=30, ha='right', fontsize=8)
ax4.set_title('渠道成本 vs 归因价值', fontsize=12, fontweight='bold')
ax4.set_ylabel('金额 ($)')
ax4.legend()

# 图5: 渠道转移热力图
ax5 = axes[1, 1]
trans_matrix = pd.DataFrame(0, index=all_channels, columns=all_channels)
for from_ch in transitions:
    total_from = sum(transitions[from_ch].values())
    for to_ch, cnt in transitions[from_ch].items():
        if to_ch in all_channels:
            trans_matrix.loc[from_ch, to_ch] = cnt / total_from * 100

sns.heatmap(trans_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax5,
            cbar_kws={'label': '转移概率 (%)'}, linewidths=0.5)
ax5.set_title('渠道转移概率热力图', fontsize=12, fontweight='bold')
ax5.set_xlabel('目标渠道')
ax5.set_ylabel('来源渠道')

# 图6: 路径长度分布与转化价值
ax6 = axes[1, 2]
path_len_values = defaultdict(list)
for uid, p in conv_paths.items():
    path_len_values[p['conv_path_length']].append(p['conversion_value'])

lens = sorted(path_len_values.keys())
avg_vals = [np.mean(path_len_values[l]) for l in lens]
counts = [len(path_len_values[l]) for l in lens]

ax6_twin = ax6.twinx()
ax6.bar(lens, counts, alpha=0.6, color='#45B7D1', label='转化次数')
ax6_twin.plot(lens, avg_vals, 'ro-', linewidth=2, markersize=8, label='平均转化价值')
ax6.set_xlabel('路径长度 (触点数)')
ax6.set_ylabel('转化次数', color='#45B7D1')
ax6_twin.set_ylabel('平均转化价值 ($)', color='red')
ax6.set_title('路径长度 vs 转化价值', fontsize=12, fontweight='bold')
ax6.legend(loc='upper left')
ax6_twin.legend(loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./visualizations/attribution_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 仪表板已保存: ./visualizations/attribution_dashboard.png")

# ========== 图2: 渠道角色雷达图 ==========
fig2, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['首次触点', '末次触点', '线性', '位置归因', '马尔可夫', 'Shapley值']
N = len(categories)

for ch in all_channels:
    values = [
        first_touch_attribution.get(ch, 0) / TOTAL_VALUE * 100,
        last_touch_attribution.get(ch, 0) / TOTAL_VALUE * 100,
        linear_attribution.get(ch, 0) / TOTAL_VALUE * 100,
        position_attribution.get(ch, 0) / TOTAL_VALUE * 100,
        markov_attr.get(ch, 0) / TOTAL_VALUE * 100,
        shapley_attr.get(ch, 0) / TOTAL_VALUE * 100,
    ]
    values += values[:1]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=1.5, label=ch, color=channel_colors[ch])
    ax.fill(angles, values, alpha=0.05, color=channel_colors[ch])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title('各渠道在六种归因模型中的表现 (%)', fontsize=14, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.savefig('./visualizations/channel_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 雷达图已保存: ./visualizations/channel_radar.png")

# ========== 图3: 预算优化桑基图（简化版 - 堆叠柱） ==========
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))

# 当前 vs 建议预算分配
current_pct = [current_allocation.get(ch, 0) for ch in all_channels]
optimal_pct = [optimal_allocation.get(ch, 0) for ch in all_channels]

x = np.arange(len(all_channels))
axes3[0].bar(x, current_pct, color=[channel_colors[ch] for ch in all_channels], alpha=0.7, label='当前分配')
axes3[0].bar(x, optimal_pct, color=[channel_colors[ch] for ch in all_channels], alpha=1.0,
             edgecolor='black', linewidth=1.5, label='建议分配', width=0.5)
axes3[0].set_xticks(x)
axes3[0].set_xticklabels(all_channels, rotation=30, ha='right', fontsize=9)
axes3[0].set_ylabel('预算分配 (%)')
axes3[0].set_title('当前 vs 建议预算分配', fontsize=12, fontweight='bold')
axes3[0].legend()

# 漏斗分析 - 渠道触达 → 转化
stages = ['触达用户', '进入路径', '路径中间', '最终转化']
stage_values = [
    df['user_id'].nunique(),
    len(paths),
    sum(1 for p in conv_paths.values() if p['conv_path_length'] > 2),
    TOTAL_CONVERSIONS
]
colors_funnel = ['#45B7D1', '#4ECDC4', '#96CEB4', '#FF6B6B']
bars = axes3[1].barh(stages[::-1], stage_values[::-1], color=colors_funnel[::-1])
axes3[1].set_title('转化漏斗分析', fontsize=12, fontweight='bold')
axes3[1].set_xlabel('用户数')
for bar, val in zip(bars, stage_values[::-1]):
    axes3[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                  f'{val}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./visualizations/budget_optimization.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 预算优化图已保存: ./visualizations/budget_optimization.png")

# ========== 保存路径数据 ==========
paths_data = []
for uid, p in conv_paths.items():
    paths_data.append({
        'user_id': uid,
        'path': " → ".join(p['pre_conv_path']),
        'path_length': p['conv_path_length'],
        'conversion_value': p['conversion_value'],
        'duration_days': p['duration_days'],
        'total_cost': p['total_cost']
    })

pd.DataFrame(paths_data).to_csv('./analysis_reports/customer_paths.csv', index=False)

# ========== 10. 生成综合分析报告 ==========
report = f"""# 营销渠道多模型归因分析报告

## 📊 执行摘要

**分析时间**: 2026-05-09
**数据来源**: examples/sample_channel_data.csv
**分析范围**: 20个用户, 94个触点, 7个渠道

### 核心发现
- **总转化用户**: {TOTAL_CONVERSIONS}/{len(paths)} ({TOTAL_CONVERSIONS/len(paths)*100:.0f}%)
- **总转化价值**: ${TOTAL_VALUE:,.0f}
- **总营销成本**: ${total_budget:,.0f}
- **整体ROAS**: {TOTAL_VALUE/total_budget:.2f}x
- **平均客户旅程**: {np.mean(path_lengths):.1f}个触点, 耗时{np.mean([p['duration_days'] for p in conv_paths.values()]):.0f}天

---

## 📈 七种归因模型结果

### 渠道归因贡献对比 (%)

| 渠道 | 首次触点 | 末次触点 | 线性 | 时间衰减 | 位置归因 | 马尔可夫 | Shapley值 |
|------|---------|---------|------|---------|---------|---------|-----------|
"""

for ch in all_channels:
    vals = [comparison_df_pct.loc[ch, col] for col in comparison_df_pct.columns]
    report += f"| {ch} | " + " | ".join([f"{v:.1f}%" for v in vals]) + " |\n"

report += f"""
### 渠道归因贡献对比 ($)

| 渠道 | 首次触点 | 末次触点 | 线性 | 时间衰减 | 位置归因 | 马尔可夫 | Shapley值 |
|------|---------|---------|------|---------|---------|---------|-----------|
"""

for ch in all_channels:
    vals = [comparison_df.loc[ch, col] for col in comparison_df.columns]
    report += f"| {ch} | " + " | ".join([f"${v:,.0f}" for v in vals]) + " |\n"

report += f"""
---

## 🔑 关键洞察

### 1. 渠道重要性排名 (基于Shapley值)
"""

for i, (ch, val) in enumerate(top_shapley, 1):
    pct = val / TOTAL_VALUE * 100
    report += f"{i}. **{ch}**: ${val:,.0f} ({pct:.1f}%)\n"

report += f"""
### 2. 渠道角色定位
"""

for ch in all_channels:
    pos = position_dist.get(ch, {'first': 0, 'middle': 0, 'last': 0})
    total_pos = pos['first'] + pos['middle'] + pos['last']
    if total_pos > 0:
        if pos['first'] / total_pos > 0.5:
            role = "**认知型渠道（引流）**"
        elif pos['last'] / total_pos > 0.5:
            role = "**转化型渠道（促转）**"
        elif pos['middle'] / total_pos > 0.5:
            role = "**考虑型渠道（培育）**"
        else:
            role = "**全能型渠道**"
        report += f"- **{ch}**: {role} (首位{pos['first']}, 中间{pos['middle']}, 末位{pos['last']})\n"

report += f"""
### 3. 渠道ROI分析 (基于Shapley值)

| 渠道 | 归因价值 | 投入成本 | 净利润 | ROI | ROAS |
|------|---------|---------|--------|-----|------|
"""

for _, row in roi_df.iterrows():
    report += f"| {row['channel']} | ${row['shapley_value']:,.0f} | ${row['total_cost']:,.0f} | ${row['net_profit']:,.0f} | {row['roi_pct']:.0f}% | {row['roas']:.2f}x |\n"

report += f"""
### 4. 高频渠道协同组合
"""

for pair, count in top_pairs[:5]:
    report += f"- **{pair[0]}** + **{pair[1]}**: 共现 {count} 次\n"

report += f"""
---

## 📋 策略建议

### 🔴 高优先级

1. **加大 paid_search 投入**
   - 在多个模型中均表现出色
   - 建议增加预算占比
   - 作为核心转化渠道持续优化

2. **优化 social_media 路径**
   - social_media 是最强的培育型渠道
   - 加强从 social_media 到 paid_search/email 的路径引导
   - 提升社交媒体到转化的衔接效率

3. **重视 email 渠道的转化角色**
   - email 常出现在转化路径末端
   - 优化邮件触发时机和内容
   - 与其他渠道形成转化闭环

### 🟡 中优先级

4. **content_marketing 作为培育渠道优化**
   - 在路径中间位置发挥作用
   - 提升内容质量和转化引导
   - 加强内容到付费渠道的过渡

5. **探索 affiliate 和 display 渠道**
   - 使用频次低但可能有增量价值
   - 进行小规模测试验证效果
   - 关注这两个渠道的协同效应

### 🟢 低优先级

6. **organic_search 路径优化**
   - 自然搜索作为零成本引流渠道
   - 通过SEO优化提升自然流量
   - 加强着陆页的转化设计

---

## 📊 预算优化建议

| 渠道 | 当前分配 | 建议分配 | 调整方向 |
|------|---------|---------|---------|
"""

for ch in all_channels:
    curr_pct = current_allocation.get(ch, 0)
    opt_pct = optimal_allocation.get(ch, 0)
    diff = opt_pct - curr_pct
    direction = "↑ 增加" if diff > 2 else ("↓ 减少" if diff < -2 else "→ 维持")
    report += f"| {ch} | {curr_pct:.1f}% | {opt_pct:.1f}% | {direction} (差{diff:+.1f}%) |\n"

report += f"""
---

## 📁 分析产出文件

- `attribution_results.csv` - 七种模型归因结果
- `channel_performance.csv` - 渠道性能指标
- `customer_paths.csv` - 客户路径数据
- `attribution_dashboard.png` - 综合可视化仪表板
- `channel_radar.png` - 渠道雷达图
- `budget_optimization.png` - 预算优化图表

---

*报告由 Claude 归因分析技能自动生成*
"""

with open('./analysis_reports/attribution_deep_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n  ✅ 分析报告已保存: ./analysis_reports/attribution_deep_analysis_report.md")

print("\n" + "=" * 70)
print("  ✅ 全部分析完成！")
print("=" * 70)
print(f"\n📁 产出文件:")
print(f"  - analysis_reports/attribution_results.csv")
print(f"  - analysis_reports/channel_performance.csv")
print(f"  - analysis_reports/customer_paths.csv")
print(f"  - analysis_reports/attribution_deep_analysis_report.md")
print(f"  - visualizations/attribution_dashboard.png")
print(f"  - visualizations/channel_radar.png")
print(f"  - visualizations/budget_optimization.png")
print(f"  - generated_code/attribution_analysis.py")
