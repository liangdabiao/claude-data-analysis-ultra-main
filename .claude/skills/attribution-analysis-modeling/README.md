# Attribution Analysis & Modeling Skill

一个用于营销渠道归因分析和效果评估的综合性Claude Code技能，支持多种归因模型和可视化分析。

## 功能特性 | Features

- 🎯 **多归因模型**: 首次接触、最后接触、线性、时间衰减、位置归因、马尔可夫链、Shapley值
- 🔗 **马尔可夫链分析**: 基于转移概率和移除效应的高级归因模型
- 🎮 **Shapley值计算**: 博弈理论基础的公平归因分配算法
- 📊 **可视化仪表板**: 综合归因分析和渠道效果可视化
- 📈 **渠道协同分析**: 渠道间协同效应和相互作用分析
- 🚀 **营销优化建议**: 基于归因结果的预算优化和策略建议
- 🌏 **中文支持**: 完整支持中文数据和可视化显示

## 安装依赖 | Installation

```bash
pip install -r requirements.txt
```

## 快速开始 | Quick Start

### 基本归因分析

```python
from core_attribution import AttributionAnalyzer
from attribution_visualizer import AttributionVisualizer

# 1. 数据分析和归因
analyzer = AttributionAnalyzer()
results = analyzer.run_complete_analysis('marketing_data.csv')

# 2. 可视化分析
visualizer = AttributionVisualizer()
visualizer.create_attribution_dashboard(results)
```

### 马尔可夫链归因分析

```python
from markov_chains import MarkovChainAttributor

# 1. 构建马尔可夫链模型
markov_attributor = MarkovChainAttributor()
markov_results = markov_attributor.run_complete_markov_analysis(customer_paths)

# 2. 生成马尔可夫可视化
visualizer.create_markov_visualization(markov_results)
```

### Shapley值归因分析

```python
from shapley_values import ShapleyValueAttributor

# 1. 计算Shapley值
shapley_attributor = ShapleyValueAttributor()
shapley_results = shapley_attributor.run_complete_shapley_analysis(customer_paths)

# 2. 生成Shapley可视化
visualizer.create_shapley_visualization(shapley_results)
```

## 数据格式要求 | Data Format Requirements

### 标准营销触点数据 | Standard Touchpoint Data

```csv
user_id,timestamp,channel,conversion_status,conversion_value,cost
USER001,2024-01-15T10:30:00Z,paid_search,0,0,50
USER001,2024-01-16T14:20:00Z,social_media,0,0,30
USER001,2024-01-18T09:15:00Z,email,1,1000,10
```

### 必需字段 | Required Fields
- **user_id**: 唯一客户标识符
- **timestamp**: 触点时间戳 (ISO格式推荐)
- **channel**: 营销渠道或触点
- **conversion_status**: 转化指标 (0/1)
- **conversion_value**: 转化价值 (可选)
- **cost**: 营销成本 (可选，用于ROI分析)

### 支持的渠道类型 | Supported Channel Types
- **数字渠道**: paid_search, organic_search, social_media, email, display, video
- **传统渠道**: tv, radio, print, outdoor, direct_mail
- **电商平台**: marketplace, affiliate, referral
- **自定义**: 任意渠道名称

## 输出文件 | Output Files

### 数据文件 | Data Files
- `attribution_results.csv`: 完整归因分析结果
- `channel_performance.csv`: 渠道性能指标和ROI分析
- `customer_paths.csv`: 重构的客户旅程路径
- `transition_matrix.csv`: 马尔可夫链转移概率矩阵

### 可视化文件 | Visualization Files
- `attribution_dashboard.png`: 综合归因分析仪表板
- `markov_analysis.png`: 马可夫链分析可视化
- `shapley_analysis.png`: Shapley值分析可视化
- `channel_network_graph.png`: 渠道转换网络图

### 报告文件 | Report Files
- `attribution_report.md`: 详细归因分析报告
- `optimization_recommendations.md`: 渠道优化建议报告

## 核心功能模块 | Core Modules

### 1. 归因分析引擎 (core_attribution.py)
```python
class AttributionAnalyzer:
    def load_and_validate_data()      # 数据加载与验证
    def build_customer_paths()         # 构建客户路径
    def first_touch_attribution()       # 首次接触归因
    def last_touch_attribution()        # 最后接触归因
    def linear_attribution()           # 线性归因
    def time_decay_attribution()       # 时间衰减归因
    def position_based_attribution()    # 位置归因
    def compare_attribution_models()    # 模型比较分析
```

### 2. 马尔可夫链分析 (markov_chains.py)
```python
class MarkovChainAttributor:
    def build_transition_matrix()      # 构建转移矩阵
    def calculate_removal_effects()    # 计算移除效应
    def calculate_attribution_weights() # 计算归因权重
    def analyze_channel_transitions() # 渠道转换分析
    def build_channel_graph()         # 构建渠道转换图
    def simulate_attribution_scenarios() # 场景模拟分析
```

### 3. Shapley值分析 (shapley_values.py)
```python
class ShapleyValueAttributor:
    def calculate_shapley_values()     # 计算Shapley值
    def calculate_channel_synergy()     # 计算渠道协同效应
    def analyze_marginal_contributions() # 分析边际贡献
    def optimize_channel_mix()         # 优化渠道组合
    def run_complete_shapley_analysis() # 完整Shapley分析
```

### 4. 可视化工具 (attribution_visualizer.py)
```python
class AttributionVisualizer:
    def create_attribution_dashboard()  # 综合分析仪表板
    def create_markov_visualization()    # 马尔可夫链可视化
    def create_shapley_visualization()    # Shapley值可视化
    def _plot_attribution_weights_comparison() # 归因权重对比
    def _plot_channel_performance_metrics() # 渠道性能指标
    def _plot_journey_path_analysis() # 客户旅程分析
```

## 支持的归因模型 | Supported Attribution Models

### 基础模型 | Basic Models
- **首次接触归因**: 将功劳完全分配给路径中的第一个渠道
- **最后接触归因**: 将功劳完全分配给转化前的最后一个渠道
- **线性归因**: 平均分配给路径中的所有渠道

### 高级模型 | Advanced Models
- **时间衰减归因**: 根据时间远近分配权重
- **位置归因**: 首尾接触权重更高，中间接触权重平均
- **马尔可夫链归因**: 基于概率转移的动态归因
- **Shapley值归因**: 基于博弈理论的公平归因分配

## 业务应用场景 | Business Applications

### 营销渠道分析 | Marketing Channel Analysis
- 数字营销渠道效果评估 (搜索、社交、邮件、展示广告等)
- 线上线下渠道归因分析
- 跨平台营销活动效果分析
- 多渠道营销活动归因

### 电商转化分析 | E-commerce Conversion Analysis
- 用户购买路径归因
- 多触点转化贡献分析
- 营销活动ROI评估
- 产品推荐系统优化

### 客户旅程分析 | Customer Journey Analysis
- 客户获取渠道归因
- 客户生命周期价值归因
- 流失原因分析和预防
- 个性化营销策略制定

## 高级用法 | Advanced Usage

### 自定义归因模型

```python
# 扩展基础归因分析器
class CustomAttributor(AttributionAnalyzer):
    def custom_weighted_attribution(self, paths_df, weight_config):
        """自定义权重归因模型"""
        # 实现自定义权重逻辑
        pass

    def business_rule_attribution(self, paths_df, business_rules):
        """业务规则归因模型"""
        # 基于业务规则的归因逻辑
        pass
```

### 多归因模型比较

```python
# 运行所有可用模型
results = analyzer.compare_attribution_models(paths_df)

# 选择最佳模型
best_model = self._select_best_model(results)
```

### 渠道优化建议

```python
# 基于归因结果优化预算
optimization_results = analyzer.generate_attribution_summary(df)

# 获取优化建议
for recommendation in optimization_results['recommended_actions']:
    print(f"{recommendation['type']}: {recommendation['channel']}")
```

## 模型比较 | Model Comparison

### 归因模型特点对比 | Model Characteristics Comparison

| 模型类型 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| 首次接触 | 简单直接 | 忽略后续触点 | 新产品推广 |
| 最后接触 | 考虑最终决策 | 忽略前期影响 | 销售转化 |
| 线性 | 公平分配 | 不考虑差异 | 平均效果 |
| 时间衰减 | 考虑时间因素 | 需要调参 | 时间敏感 |
| 马可夫链 | 动态概率 | 计算复杂 | 复杂路径 |
| Shapley值 | 理论最优 | 计算量大 | 精确归因 |

### 模型选择建议 | Model Selection Guidelines

- **快速评估**: 使用首次接触和最后接触
- **平衡分析**: 使用线性或位置归因
- **精确归因**: 使用马尔可夫链或Shapley值
- **A/B测试**: 比较不同模型的结果

## 最佳实践 | Best Practices

### 数据质量 | Data Quality
- 确保用户ID在所有触点保持一致
- 维护准确的时间戳数据
- 包含成本数据用于ROI分析
- 处理数据缺失和异常值

### 模型选择 | Model Selection
- 根据业务目标选择归因模型
- 比较多个模型进行验证
- 考虑客户旅程复杂性
- 与业务相关方验证结果

### 实施建议 | Implementation Recommendations
- 从简单模型开始，逐步升级
- 测试归因结果与已知业务结果
- 基于归因洞察进行渐进式调整
- 监控归因模型性能变化

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues

1. **用户标识不一致**
   ```python
   # 标准化用户ID
   df['user_id'] = df['user_id'].astype(str).str.lower()
   ```

2. **时间戳格式问题**
   ```python
   # 转换时间戳格式
   df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
   ```

3. **数据量过大**
   ```python
   # 分批处理大数据
   chunk_size = 10000
   for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
       # 处理每个数据块
       pass
   ```

4. **模型计算时间过长**
   ```python
   # 对于Shapley值计算，限制渠道数量
   channels = df['channel'].unique()[:10]  # 只分析前10个渠道
   ```

## 性能优化 | Performance Optimization

### 大数据处理 | Big Data Processing
```python
# 使用更高效的数据类型
dtypes = {
    'user_id': 'category',
    'channel': 'category',
    'cost': 'float32'
}
df = pd.read_csv('data.csv', dtype=dtypes)
```

### 并行计算 | Parallel Computing
```python
# 多进程处理Shapley值计算
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(calculate_shapley_for_channel, channels)
```

## 版本历史 | Version History

- **v1.0** (2024-12): 初始版本发布
  - 多种归因模型实现
  - 马可夫链和Shapley值算法
  - 综合可视化系统
  - 渠道优化建议引擎

---

*由归因分析与建模系统支持 | Powered by Attribution Analysis Engine*