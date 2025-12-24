# AB测试分析技能使用指南

一个功能完整的智能AB测试分析工具，为产品经理、数据分析师和营销人员提供专业的AB测试分析能力。

## 🚀 快速上手

### 1. 环境准备

确保安装了必要的Python包：

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
```

### 2. 快速验证

运行快速测试来验证技能功能：

```bash
python quick_test.py
```

### 3. 基础使用示例

```python
from scripts.ab_test_analyzer import ABTestAnalyzer
from scripts.statistical_tests import StatisticalTests
from scripts.visualizer import ABTestVisualizer

# 初始化分析器
analyzer = ABTestAnalyzer()
stats_tests = StatisticalTests()
visualizer = ABTestVisualizer()

# 加载数据
data = analyzer.load_data('examples/sample_data/sample_ab_test_data.csv')

# 基础分析
conversion_results = analyzer.analyze_conversion(
    data, group_col='页面版本', conversion_col='是否购买'
)

print("转化率分析结果:", conversion_results)
```

## 📊 核心功能概览

| 功能模块 | 主要能力 | 适用场景 |
|---------|---------|---------|
| **基础分析** | 转化率、留存率、提升率计算 | 产品改版、功能优化 |
| **统计检验** | t检验、卡方检验、置信区间 | 验证实验效果显著性 |
| **用户分群** | 价值分群、行为分群、交互效应 | 精细化运营、个性化推荐 |
| **可视化** | 图表生成、仪表板、报告导出 | 结果展示、决策支持 |
| **高级分析** | 贝叶斯方法、多变量、因果推断 | 复杂实验设计、深度洞察 |

## 🔧 典型使用流程

### 场景1：网页改版效果评估

```python
# 1. 数据准备
data = analyzer.load_data('website_ab_test.csv')

# 2. 基础转化率分析
conversion_results = analyzer.analyze_conversion(
    data=data,
    group_col='页面版本',
    conversion_col='是否购买'
)

# 3. 统计显著性检验
t_test_result = stats_tests.t_test(
    data=data,
    group_col='页面版本',
    metric_col='是否购买'
)

# 4. 生成报告
fig = visualizer.plot_conversion_comparison(conversion_results)
plt.show()

print(f"新页面转化率: {conversion_results['test_group_rate']:.2%}")
print(f"旧页面转化率: {conversion_results['control_group_rate']:.2%}")
print(f"提升幅度: {conversion_results['lift']:.2%}")
print(f"统计显著性: {'显著' if t_test_result['p_value'] < 0.05 else '不显著'}")
```

### 场景2：用户分群深度分析

```python
from scripts.segment_analyzer import SegmentAnalyzer

segment_analyzer = SegmentAnalyzer()

# 1. 价值分群
value_segments = segment_analyzer.value_based_segmentation(
    data=data,
    value_col='累计消费金额',
    n_tiers=3  # 高、中、低价值
)

# 2. 分群转化率分析
segment_conversion = segment_analyzer.segment_conversion_analysis(
    data=data,
    segment_col='价值组别',
    group_col='页面版本',
    conversion_col='是否购买'
)

# 3. 交互效应分析
interaction_analysis = segment_analyzer.interaction_analysis(
    data=data,
    group_col='页面版本',
    segment_col='价值组别',
    outcome_col='是否购买'
)

# 4. 可视化交互效应
fig = visualizer.plot_interaction_effects(interaction_analysis)
plt.show()
```

### 场景3：营销活动效果评估

```python
# 1. 留存率分析
retention_results = analyzer.analyze_retention(
    data=data,
    group_col='活动组别',
    retention_col='retention_7'  # 7日留存
)

# 2. 留存率可视化
fig = visualizer.plot_retention_curves(retention_results)

# 3. 长期效果分析
long_term_analysis = analyzer.long_term_impact_analysis(
    data=data,
    time_col='日期',
    group_col='活动组别',
    metric_col='日活跃用户'
)
```

## 📈 常见问题解答

### Q: 如何选择合适的统计检验方法？

**A: 根据数据类型和研究目标选择：**

- **连续变量 + 两组比较** → t检验
- **分类变量 + 两组比较** → 卡方检验
- **连续变量 + 多组比较** → 方差分析 (ANOVA)
- **相关关系分析** → 相关分析
- **小样本 + 需要先验信息** → 贝叶斯方法

### Q: 样本量不够怎么办？

**A: 解决方案：**

1. **计算最小样本量**：使用功效分析计算所需样本量
2. **延长实验时间**：收集更多数据
3. **使用贝叶斯方法**：在小样本情况下更有效
4. **考虑效应量**：关注实际业务意义而非单纯统计显著性

### Q: 如何解释p值？

**A: p值的正确理解：**

- p值是**在原假设为真**的情况下，观察到当前结果或更极端结果的概率
- p值小 ≠ 效应量大
- p值大 ≠ 没有效果
- 需要结合效应量和业务背景来解释

### Q: 多重比较如何处理？

**A: 使用校正方法：**

```python
# Bonferroni校正 (保守)
analyzer.set_multiple_comparison_correction('bonferroni')

# FDR校正 (较宽松)
analyzer.set_multiple_comparison_correction('fdr_bh')

# Holm校正
analyzer.set_multiple_comparison_correction('holm')
```

## 🎯 最佳实践

### 1. 实验设计阶段

```python
# 样本量计算
sample_size = analyzer.calculate_sample_size(
    baseline_rate=0.10,  # 基准转化率
    expected_lift=0.20,   # 期望提升20%
    alpha=0.05,          # 显著性水平
    power=0.80           # 统计功效
)

print(f"建议样本量: {sample_size} (每组)")
```

### 2. 数据质量检查

```python
# 随机分组验证
randomization_check = analyzer.check_randomization(
    data=data,
    group_col='实验组别',
    feature_cols=['年龄', '性别', '历史消费']
)

# 数据完整性检查
data_quality = analyzer.data_quality_check(
    data=data,
    required_cols=['用户ID', '实验组别', '转化状态']
)
```

### 3. 结果解释

```python
# 综合分析报告
comprehensive_report = analyzer.generate_comprehensive_report(
    data=data,
    group_col='页面版本',
    conversion_col='是否购买',
    segment_cols=['价值组别', '性别'],
    metrics=['conversion_rate', 'retention_rate', 'revenue']
)

# 导出报告
report_html = analyzer.export_report(
    results=comprehensive_report,
    format='html',
    filename='ab_test_report.html'
)
```

## 📚 进阶学习

### 统计学基础概念

- **假设检验**: 原假设 vs 备择假设
- **效应量**: Cohen's d, Cramer's V
- **置信区间**: 参数估计的范围
- **统计功效**: 检验出真实效应的能力

### 高级分析方法

- **贝叶斯AB测试**: 融入先验信息
- **多变量实验**: 同时测试多个因素
- **时间序列分析**: 考虑时间趋势
- **因果推断**: 建立因果关系

## 🛠️ 故障排除

### 常见错误及解决方案

1. **数据格式错误**
   ```python
   # 检查数据格式
   print(data.dtypes)
   print(data.head())
   ```

2. **样本量不足**
   ```python
   # 计算最小样本量
   min_sample_size = analyzer.calculate_sample_size(...)
   if len(data) < min_sample_size:
       print("警告：样本量不足")
   ```

3. **分组不均衡**
   ```python
   # 检查分组均衡性
   group_balance = analyzer.check_group_balance(data, '实验组别')
   print(group_balance)
   ```

## 📞 获取帮助

- 查看示例代码：`examples/`目录
- 运行测试脚本：`python test_skill.py`
- 阅读详细文档：`SKILL.md`

## 🎉 开始你的第一个AB测试分析

```bash
# 克隆或下载技能
cd .claude/skills/ab-testing-analyzer/

# 运行基础示例
python examples/basic_ab_test_example.py

# 查看更多示例
ls examples/
```

现在你可以开始专业的AB测试分析了！🚀