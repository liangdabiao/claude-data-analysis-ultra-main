# 归因分析技能使用指南
# Attribution Analysis Skill Usage Guide

## 📋 目录
- [快速开始](#快速开始)
- [基础归因分析](#基础归因分析)
- [高级归因分析](#高级归因分析)
- [数据准备指南](#数据准备指南)
- [结果解读](#结果解读)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import pandas, numpy, matplotlib, seaborn, scipy, networkx; print('✅ 依赖安装成功')"
```

### 2. 基础使用示例

```python
from core_attribution import AttributionAnalyzer
from attribution_visualizer import AttributionVisualizer

# 1. 初始化分析器
analyzer = AttributionAnalyzer()

# 2. 加载数据
df = analyzer.load_and_validate_data('your_data.csv')

# 3. 构建客户路径
paths_df = analyzer.build_customer_paths(df)

# 4. 运行基础归因分析
results = analyzer.run_basic_attribution_analysis(paths_df)

# 5. 生成可视化
visualizer = AttributionVisualizer()
visualizer.create_attribution_dashboard(results)
```

## 📊 基础归因分析

### 数据格式要求

```csv
user_id,timestamp,channel,conversion_status,conversion_value,cost
USER001,2024-01-15T10:30:00Z,paid_search,0,0,50
USER001,2024-01-16T14:20:00Z,social_media,0,0,30
USER001,2024-01-18T09:15:00Z,email,1,1000,10
```

### 必需字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| user_id | 字符串 | 唯一客户标识 | "USER001" |
| timestamp | 时间戳 | 触点时间 | "2024-01-15T10:30:00Z" |
| channel | 字符串 | 营销渠道 | "paid_search" |
| conversion_status | 整数 | 转化状态(0/1) | 1 |
| conversion_value | 数值 | 转化价值 | 1000 |
| cost | 数值 | 营销成本 | 50 |

### 运行基础归因分析

```python
# 基础归因分析示例
from core_attribution import AttributionAnalyzer

analyzer = AttributionAnalyzer()

# 加载和预处理数据
df = analyzer.load_and_validate_data('marketing_data.csv')
paths_df = analyzer.build_customer_paths(df)

# 运行所有基础归因模型
results = analyzer.run_basic_attribution_analysis(paths_df)

# 查看结果
for model_name, weights in results.items():
    print(f"\n{model_name}:")
    for channel, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {channel}: {weight:.4f} ({weight*100:.1f}%)")
```

### 输出说明

基础归因分析包含以下模型：

1. **首次接触归因**：100%功劳给第一个渠道
2. **最后接触归因**：100%功劳给转化前最后一个渠道
3. **线性归因**：平均分配给所有渠道
4. **时间衰减归因**：时间越近权重越高
5. **位置归因**：首尾40%，中间平均分配

## 🎮 高级归因分析

### 马尔可夫链归因

```python
from markov_chains import MarkovChainAttributor

# 初始化马尔可夫链归因器
markov_attributor = MarkovChainAttributor()

# 构建转移矩阵
transition_matrix = markov_attributor.build_transition_matrix(paths_df)

# 计算归因权重
markov_weights = markov_attributor.calculate_attribution_weights()

# 分析渠道转换
transition_analysis = markov_attributor.analyze_channel_transitions(transition_matrix)

# 构建渠道网络图
channel_graph = markov_attributor.build_channel_graph(transition_matrix)
```

### Shapley值归因

```python
from shapley_values import ShapleyValueAttributor

# 初始化Shapley值归因器
shapley_attributor = ShapleyValueAttributor()

# 运行完整分析
shapley_results = shapley_attributor.run_complete_shapley_analysis(paths_df)

# 获取归因权重
attribution_weights = shapley_results['attribution_weights']

# 分析渠道协同效应
synergy_analysis = shapley_results['channel_synergy']

# 获取优化建议
optimization = shapley_results['optimization']
```

### 高级可视化

```python
from attribution_visualizer import AttributionVisualizer

visualizer = AttributionVisualizer()

# 创建马尔可夫链可视化
markov_viz = visualizer.create_markov_visualization(markov_results)

# 创建Shapley值可视化
shapley_viz = visualizer.create_shapley_visualization(shapley_results)

# 创建综合对比仪表板
dashboard = visualizer.create_attribution_dashboard(all_results)
```

## 📁 数据准备指南

### 数据收集最佳实践

1. **统一用户标识**
   ```python
   # 标准化用户ID格式
   df['user_id'] = df['user_id'].astype(str).str.lower()
   ```

2. **时间戳标准化**
   ```python
   # 转换为统一时间格式
   df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
   ```

3. **渠道命名规范**
   ```python
   # 统一渠道名称
   channel_mapping = {
       'google_ads': 'paid_search',
       'facebook': 'social_media',
       'email_newsletter': 'email'
   }
   df['channel'] = df['channel'].map(channel_mapping).fillna(df['channel'])
   ```

### 数据质量检查

```python
def validate_attribution_data(df):
    """验证归因分析数据质量"""
    issues = []

    # 检查必需字段
    required_columns = ['user_id', 'timestamp', 'channel', 'conversion_status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"缺少必需字段: {missing_columns}")

    # 检查数据完整性
    if df['user_id'].isnull().any():
        issues.append("存在空的用户ID")

    if df['timestamp'].isnull().any():
        issues.append("存在空的时间戳")

    # 检查转化数据
    if df['conversion_status'].notna().sum() == 0:
        issues.append("没有转化数据")

    return issues

# 使用示例
issues = validate_attribution_data(df)
if issues:
    for issue in issues:
        print(f"⚠️ {issue}")
else:
    print("✅ 数据质量检查通过")
```

### 数据预处理

```python
def preprocess_attribution_data(df):
    """预处理归因分析数据"""

    # 1. 数据类型转换
    df['user_id'] = df['user_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['conversion_status'] = pd.to_numeric(df['conversion_status'], errors='coerce').fillna(0)
    df['conversion_value'] = pd.to_numeric(df.get('conversion_value', 0), errors='coerce').fillna(0)
    df['cost'] = pd.to_numeric(df.get('cost', 0), errors='coerce').fillna(0)

    # 2. 异常值处理
    df['conversion_value'] = df['conversion_value'].clip(lower=0)
    df['cost'] = df['cost'].clip(lower=0)

    # 3. 时间排序
    df = df.sort_values(['user_id', 'timestamp'])

    # 4. 去除重复记录
    df = df.drop_duplicates(subset=['user_id', 'timestamp', 'channel'], keep='first')

    return df

# 使用示例
df_clean = preprocess_attribution_data(df)
print(f"预处理后数据: {len(df_clean)} 条记录")
```

## 📈 结果解读

### 归因权重解读

```python
def interpret_attribution_weights(weights):
    """解读归因权重结果"""

    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    print("📊 归因权重解读:")
    print("-" * 40)

    # 权重分类
    for i, (channel, weight) in enumerate(sorted_weights):
        percentage = weight * 100

        if percentage >= 25:
            tier = "🥇 核心渠道"
            insight = "主要贡献者，重点投入"
        elif percentage >= 15:
            tier = "🥈 重要渠道"
            insight = "重要贡献者，保持投入"
        elif percentage >= 5:
            tier = "🥉 辅助渠道"
            insight = "辅助作用，适度投入"
        else:
            tier = "💫 长尾渠道"
            insight = "微小贡献，考虑优化"

        print(f"{i+1:2d}. {channel:<15} {percentage:>5.1f}% - {tier}")
        print(f"     {insight}")

# 使用示例
interpret_attribution_weights(results['最后接触归因'])
```

### 渠道协同效应解读

```python
def interpret_synergy_effects(synergy_analysis):
    """解读渠道协同效应"""

    print("\n🤝 渠道协同效应解读:")
    print("-" * 50)

    # 分类协同效应
    positive_synergy = []
    neutral_synergy = []
    negative_synergy = []

    for pair_key, synergy in synergy_analysis.items():
        if synergy['synergy_ratio'] > 1.1:
            positive_synergy.append(synergy)
        elif synergy['synergy_ratio'] < 0.9:
            negative_synergy.append(synergy)
        else:
            neutral_synergy.append(synergy)

    if positive_synergy:
        print("🌟 正协同效应 (1+1>2):")
        for synergy in sorted(positive_synergy, key=lambda x: x['synergy_ratio'], reverse=True)[:3]:
            print(f"  • {synergy['channel1']} + {synergy['channel2']}: "
                  f"协同比 {synergy['synergy_ratio']:.2f}")
            print(f"    建议: 加强这两个渠道的联合营销")

    if negative_synergy:
        print("⚠️ 负协同效应 (1+1<2):")
        for synergy in sorted(negative_synergy, key=lambda x: x['synergy_ratio'])[:3]:
            print(f"  • {synergy['channel1']} + {synergy['channel2']}: "
                  f"协同比 {synergy['synergy_ratio']:.2f}")
            print(f"    建议: 避免同时投放或调整时间间隔")

    if neutral_synergy:
        print("📊 中性协同效应:")
        print(f"  • {len(neutral_synergy)} 个渠道组合表现正常")

# 使用示例
if 'channel_synergy' in shapley_results:
    interpret_synergy_effects(shapley_results['channel_synergy'])
```

## 💡 最佳实践

### 1. 模型选择策略

```python
def select_attribution_model(business_context):
    """根据业务场景选择归因模型"""

    recommendations = {
        'new_product_launch': {
            'recommended': ['首次接触归因', '线性归因'],
            'reason': '新产品需要关注认知建立，首次触达很重要'
        },
        'mature_business': {
            'recommended': ['最后接触归因', '位置归因'],
            'reason': '成熟业务关注最终转化决策环节'
        },
        'complex_journey': {
            'recommended': ['马尔可夫链归因', 'Shapley值归因'],
            'reason': '复杂客户旅程需要考虑路径依赖和协同效应'
        },
        'budget_optimization': {
            'recommended': ['Shapley值归因'],
            'reason': '预算优化需要精确的边际贡献分析'
        }
    }

    context_key = business_context.lower()
    if context_key in recommendations:
        rec = recommendations[context_key]
        print(f"推荐归因模型: {', '.join(rec['recommended'])}")
        print(f"选择理由: {rec['reason']}")
    else:
        print("建议使用多种模型进行对比分析")

# 使用示例
select_attribution_model('complex_journey')
```

### 2. 性能监控建议

```python
def monitor_attribution_performance(current_results, benchmark_results=None):
    """监控归因分析性能"""

    print("📊 归因分析性能监控:")
    print("-" * 40)

    # 模型稳定性检查
    if benchmark_results:
        print("🔍 模型稳定性分析:")
        for model_name in current_results.keys():
            if model_name in benchmark_results:
                current_weights = current_results[model_name]
                benchmark_weights = benchmark_results[model_name]

                # 计算权重变化
                channels = set(current_weights.keys()) | set(benchmark_weights.keys())
                max_change = 0
                for channel in channels:
                    current_w = current_weights.get(channel, 0)
                    benchmark_w = benchmark_weights.get(channel, 0)
                    change = abs(current_w - benchmark_w)
                    max_change = max(max_change, change)

                if max_change > 0.1:
                    stability = "⚠️ 不稳定"
                elif max_change > 0.05:
                    stability = "🟡 中等稳定"
                else:
                    stability = "✅ 稳定"

                print(f"  {model_name}: {stability} (最大变化: {max_change:.3f})")

    # 渠道集中度分析
    for model_name, weights in current_results.items():
        # 计算赫芬达尔指数（HHI）
        hhi = sum(w**2 for w in weights.values())

        if hhi > 0.25:
            concentration = "高度集中"
            risk = "高风险"
        elif hhi > 0.15:
            concentration = "中度集中"
            risk = "中等风险"
        else:
            concentration = "分散"
            risk = "低风险"

        print(f"  {model_name}: {concentration} (HHI: {hhi:.3f}, {risk})")

# 使用示例
monitor_attribution_performance(results)
```

### 3. 报告生成模板

```python
def generate_attribution_report(results, business_context):
    """生成归因分析报告"""

    report = f"""
# 归因分析报告
# Attribution Analysis Report

## 📊 执行摘要
分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
业务场景: {business_context}
分析模型: {', '.join(results.keys())}

## 🎯 关键发现

### 核心渠道识别
"""

    # 添加各模型的核心渠道
    for model_name, weights in results.items():
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_channels = sorted_weights[:3]

        report += f"\n#### {model_name}\n"
        for i, (channel, weight) in enumerate(top_channels):
            report += f"{i+1}. **{channel}**: {weight*100:.1f}%\n"

    # 添加业务建议
    report += """
## 💡 业务建议

### 短期行动 (1-3个月)
1. 增加对高权重、高ROI渠道的投入
2. 优化低效渠道的投放策略
3. 加强正协同效应渠道的联合投放

### 中期规划 (3-12个月)
1. 建立多模型归因监控体系
2. 优化客户旅程设计
3. 实施动态预算分配机制

### 长期战略 (1年以上)
1. 构建预测性归因模型
2. 建立渠道协同效应评估框架
3. 实施AI驱动的归因优化

---
*报告由归因分析技能自动生成*
"""

    return report

# 使用示例
report = generate_attribution_report(results, "电商转化分析")
print(report)
```

## ❓ 常见问题

### Q1: 如何处理缺失的成本数据？

```python
# 方法1: 使用行业平均水平
def estimate_missing_costs(df, industry_benchmarks):
    """估算缺失的成本数据"""
    for channel, benchmark_cpa in industry_benchmarks.items():
        mask = (df['channel'] == channel) & (df['cost'].isna() | (df['cost'] == 0))
        if mask.any():
            # 基于转化数量估算成本
            conversions = df[mask & (df['conversion_status'] == 1)]['conversion_status'].sum()
            estimated_cost = conversions * benchmark_cpa
            df.loc[mask, 'cost'] = estimated_cost / mask.sum()
    return df

# 方法2: 使用渠道内部平均值
def fill_missing_costs_with_average(df):
    """使用渠道平均值填充缺失成本"""
    for channel in df['channel'].unique():
        channel_mask = df['channel'] == channel
        avg_cost = df.loc[channel_mask, 'cost'].mean()
        df.loc[channel_mask & df['cost'].isna(), 'cost'] = avg_cost
    return df
```

### Q2: 如何处理非常长的客户路径？

```python
def handle_long_paths(paths_df, max_path_length=10):
    """处理过长的客户路径"""

    # 统计路径长度分布
    path_lengths = paths_df['path'].apply(len)
    print(f"路径长度统计:")
    print(f"  平均长度: {path_lengths.mean():.1f}")
    print(f"  中位数: {path_lengths.median()}")
    print(f"  95%分位数: {path_lengths.quantile(0.95)}")

    # 截断过长路径
    def truncate_path(path):
        if len(path) <= max_path_length:
            return path
        else:
            # 保留首尾重要触点
            important_count = 4  # 开始2个 + 结束2个
            middle_count = max_path_length - important_count

            return (path[:2] +
                   path[2:2+middle_count] +
                   path[-2:])

    paths_df['path'] = paths_df['path'].apply(truncate_path)
    paths_df['path_length'] = paths_df['path'].apply(len)

    return paths_df
```

### Q3: 如何验证归因结果的准确性？

```python
def validate_attribution_results(attribution_results, holdout_data):
    """使用留置数据验证归因结果"""

    validation_scores = {}

    for model_name, weights in attribution_results.items():
        # 在留置数据上测试归因权重
        predicted_performance = 0
        actual_performance = 0

        for channel, weight in weights.items():
            channel_mask = holdout_data['channel'] == channel
            predicted_contribution = weight * holdout_data.loc[channel_mask, 'conversion_value'].sum()

            # 简化的验证逻辑（实际应用中需要更复杂的方法）
            actual_contribution = holdout_data.loc[
                (holdout_data['channel'] == channel) &
                (holdout_data['conversion_status'] == 1),
                'conversion_value'
            ].sum()

            predicted_performance += predicted_contribution
            actual_performance += actual_contribution

        # 计算预测准确度
        if actual_performance > 0:
            accuracy = 1 - abs(predicted_performance - actual_performance) / actual_performance
            validation_scores[model_name] = accuracy

    print("🔍 归因模型验证结果:")
    for model_name, score in sorted(validation_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name}: {score:.3f}")

    return validation_scores
```

### Q4: 如何处理小样本数据？

```python
def handle_small_sample_data(paths_df, min_conversions=50):
    """处理小样本数据的策略"""

    total_conversions = paths_df['converted'].sum()

    if total_conversions < min_conversions:
        print(f"⚠️ 样本量较小 (总转化数: {total_conversions})")
        print("建议采用以下策略:")

        # 策略1: 使用简单的归因模型
        print("1. 使用简单归因模型 (首次/最后接触)")

        # 策略2: 增加数据时间范围
        print("2. 延长数据收集时间范围")

        # 策略3: 合并相似渠道
        print("3. 合并相似渠道减少复杂性")

        # 策略4: 使用贝叶斯方法
        print("4. 考虑使用贝叶斯归因方法")

        # 自动合并相似渠道示例
        def merge_similar_channels(channel):
            mapping = {
                'google_search': 'paid_search',
                'bing_search': 'paid_search',
                'facebook_ads': 'social_media',
                'instagram_ads': 'social_media',
                'newsletter': 'email',
                'marketing_email': 'email'
            }
            return mapping.get(channel, channel)

        paths_df['path'] = paths_df['path'].apply(
            lambda path: [merge_similar_channels(touch) for touch in path]
        )

        print(f"✅ 渠道合并后路径数量: {len(paths_df)}")

    return paths_df
```

---

🎉 **恭喜！** 您已完成归因分析技能使用指南的学习。

如需更多帮助，请参考示例代码或查看技术文档。