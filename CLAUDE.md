# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述 (Project Overview)

Claude Data Analysis Skills 是一个基于 Skill 架构的智能数据分析平台。提供两套完整的技能体系：

1. **通用数据分析技能** - 6阶段完整分析流程
2. **互联网数据分析技能** - 7个专业分析模块 + 1个入口技能

**主要特性**:
- Skills 模块化架构
- 支持 Pandas 优化性能
- 正确的数据聚合规则
- 专业的互联网分析能力

## 快速开始 (Quick Start)

### 1. 准备数据
将数据集放入 `data_storage/` 目录：
```
data_storage/
├── olist_orders_dataset.csv
├── olist_customers_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_reviews_dataset.csv
└── house.csv
```

### 2. 调用 Skills

#### 方式1: 互联网数据分析（推荐）
```bash
# 通过入口技能，一站式互联网分析
@internet-data-analysis 对 olist 电商数据进行全面分析

# 或直接使用专业技能
@ltv-predictor 进行客户生命周期价值分析
@content-analysis 分析用户评论情感
@funnel-analysis 分析转化漏斗
@growth-model-analyzer 制定增长策略
@ab-testing-analyzer 设计AB测试
@attribution-analysis-modeling 渠道归因分析
@data-exploration-visualization 探索性分析
```

#### 方式2: 通用数据分析
```bash
# 直接使用 Skill
@data-explorer 分析 olist 电商数据
@visualization-specialist 创建销售可视化
@report-writer 生成分析报告

# 通过主 Skill (支持 Advanced Mode)
@data-analysis 对 olist数据进行 statistical 分析
```

## Skills 架构

### 互联网分析技能（新增）
```
.claude/skills/
├── internet-data-analysis/           # 入口技能（协调者）
├── ab-testing-analyzer/             # AB测试分析
├── attribution-analysis-modeling/    # 归因分析建模
├── content-analysis/                # 内容分析（NLP）
├── data-exploration-visualization/  # 数据探索与可视化
├── funnel-analysis/                 # 漏斗分析
├── growth-model-analyzer/           # 增长模型分析
└── ltv-predictor/                   # LTV预测
```

### 各互联网分析 Skill 功能

| Skill | 功能 | 关键输出 |
|-------|------|---------|
| internet-data-analysis | 统筹全部互联网分析技能 | 完整分析方案、技能组合建议 |
| ab-testing-analyzer | A/B测试设计与分析 | 样本量计算、统计检验、分群分析 |
| attribution-analysis-modeling | 营销渠道归因 | 首次/末次/线性/马尔可夫/Shapley值 |
| content-analysis | 文本内容分析 | 情感分析、主题提取、关键词分析 |
| data-exploration-visualization | 探索性分析与可视化 | 数据概览、统计图表、分布分析 |
| funnel-analysis | 转化漏斗分析 | 漏斗图、流失节点、转化率优化 |
| growth-model-analyzer | 增长策略分析 | AARRR框架、Uplift建模、增长杠杆 |
| ltv-predictor | 客户生命周期价值预测 | RFM分群、LTV分层、价值预测 |

### 通用数据分析技能（保留）
```
.claude/skills/
├── data-analysis/           # 主 Skill（协调者）
├── data-explorer/          # 数据分析
├── visualization-specialist/ # 数据可视化
├── report-writer/          # 报告生成
├── code-generator/         # 代码生成
├── hypothesis-generator/    # 假设生成
└── quality-assurance/      # 质量保证
```

### 各通用分析 Skill 功能

| Skill | 功能 | 支持类型 |
|-------|------|----------|
| data-analysis | 协调完整分析流程 | exploratory, statistical, predictive, complete |
| data-explorer | 探索性/统计分析 | EDA, 统计检验, RFM |
| visualization-specialist | 数据可视化 | all, trends, distribution, correlation, comparison |
| report-writer | 报告生成 | summary, complete, executive, technical |
| code-generator | 代码生成 | data-cleaning, statistical, visualization, machine-learning |
| hypothesis-generator | 假设生成 | A/B测试, 样本量计算 |
| quality-assurance | 质量验证 | 缺失值, 重复, 一致性 |

## Advanced Mode 高级分析模式

### data-analysis 分析类型

| 类型 | 深度 | 适用场景 |
|------|------|----------|
| exploratory | ⭐ | 快速了解数据结构 |
| statistical | ⭐⭐ | 需要深度统计分析 |
| predictive | ⭐⭐⭐ | 需要预测模型 |
| complete | ⭐⭐⭐⭐ | 全面分析 |

### visualization-specialist 图表类型

| 类型 | 内容 |
|------|------|
| all | 完整仪表板 |
| trends | 趋势分析图 |
| distribution | 分布分析图 |
| correlation | 相关性分析图 |
| comparison | 对比分析图 |

### report-writer 报告类型

| 类型 | 适用对象 |
|------|----------|
| summary | 快速摘要 |
| complete | 全面报告 |
| executive | 高管决策 |
| technical | 技术团队 |

## 数据处理规范 (CRITICAL)

### 1. Pandas vs Pure Python

系统会自动检测环境，优先使用 Pandas：

```python
try:
    import pandas as pd
    import numpy as np
    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False
```

**性能对比**:
- Pandas: 数据量大时快 5-10 倍（向量化操作，C底层）
- Pure Python: 小数据量或无 pandas 环境时备用

### 2. 电商数据处理 (必须遵守)

```python
# ❌ 错误：直接平均 order_items (会重复计算多商品订单)
prices = [float(item['price']) for item in order_items]
avg_price = sum(prices) / len(prices)

# ✅ 正确：按 order_id 汇总
order_amounts = order_items.groupby('order_id')['price'].sum() + \
                order_items.groupby('order_id')['freight_value'].sum()
amounts = order_amounts.values
avg_amount = amounts.mean()
```

### 3. 数据聚合规则

| 数据类型 | 正确处理方式 |
|----------|--------------|
| 订单金额 | 按 order_id 汇总 (price + freight_value) |
| 评分 | 按 order_id 取平均值 |
| 配送时间 | delivered_date - purchase_date |

### 4. 样本量要求

- **始终使用全量数据**（除非数据量 > 100万行）
- 在报告中明确标注样本量
- 避免因样本限制导致统计偏差

## 使用示例 (Usage Examples)

### 示例 1: 互联网数据分析 - 电商综合分析
```bash
# 场景：对 Olist 电商数据进行全面分析
用户: 对 olist 电商数据进行全面互联网分析
调用: @internet-data-analysis
```

**分析流程**（由 internet-data-analysis 自动协调）：
1. 数据探索 → @data-exploration-visualization
2. 漏斗分析 → @funnel-analysis
3. LTV预测 → @ltv-predictor
4. 内容分析 → @content-analysis
5. 增长策略 → @growth-model-analyzer

---

### 示例 2: 互联网数据分析 - 专项分析
```bash
# 场景1：只做 LTV 分析
用户: 分析 olist 客户生命周期价值
调用: @ltv-predictor

# 场景2：只做评论分析
用户: 分析 olist 用户评论
调用: @content-analysis

# 场景3：只做增长分析
用户: 制定 olist 增长策略
调用: @growth-model-analyzer

# 场景4：只做漏斗分析
用户: 分析 olist 转化漏斗
调用: @funnel-analysis
```

---

### 示例 3: 互联网数据分析 - AB测试与归因
```bash
# 场景：准备营销活动
用户: 设计一个提升复购的 AB 测试
调用: @ab-testing-analyzer

用户: 分析不同渠道的转化贡献
调用: @attribution-analysis-modeling
```

---

### 示例 4: 通用数据分析 - 基础 EDA
```bash
用户: 分析 house.csv 数据
调用: data-explorer
```

### 示例 5: 通用数据分析 - 指定分析类型
```bash
用户: 对 olist 电商数据进行 statistical 分析
调用: data-analysis (analysis_type=statistical)
```

### 示例 6: 通用数据分析 - 完整分析流程
```bash
# 1. 数据质量验证
@quality-assurance 检查 olist 数据

# 2. 探索性分析
@data-explorer 分析 olist 订单

# 3. 统计检验
@data-analysis 对 olist 进行 statistical 分析

# 4. 客户分群
@data-analysis 对 olist 进行 predictive 分析

# 5. 可视化
@visualization-specialist 创建 olist 仪表板

# 6. 生成报告
@report-writer 生成 olist 完整报告
```

---

## 互联网分析场景指南

### 电商业务典型分析场景

| 业务目标 | 推荐技能组合 | 输出结果 |
|---------|-------------|---------|
| **提升复购** | ltv-predictor + growth-model-analyzer + ab-testing-analyzer | RFM分群、复购策略、A/B测试方案 |
| **优化转化** | funnel-analysis + data-exploration-visualization | 漏斗分析、流失节点、优化建议 |
| **评论分析** | content-analysis | 情感分析、关键词、主题洞察 |
| **渠道优化** | attribution-analysis-modeling | 渠道归因、预算优化建议 |
| **客户价值** | ltv-predictor | LTV分层、价值预测、差异化策略 |
| **全面诊断** | internet-data-analysis (入口) | 完整分析报告、综合增长方案 |

### 新技能特性

- **templates/**: 每个技能都提供标准化分析报告模板
- **guide/**: 详细的操作指南和最佳实践
- **examples/**: 可直接运行的示例代码
- **独立方法论**: 新技能不依赖现有6阶段框架，采用独立分析方法

### 示例 7: Python 代码 (Pandas 推荐)
```python
import pandas as pd
import numpy as np

# 读取数据
orders = pd.read_csv('./data_storage/olist_orders_dataset.csv')
order_items = pd.read_csv('./data_storage/olist_order_items_dataset.csv')

# 正确聚合订单金额
order_amounts = order_items.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
}).sum(axis=1)

# 描述性统计
amounts = order_amounts.values
print(f"均值: {amounts.mean():.2f}")
print(f"中位数: {np.median(amounts):.2f}")

# 异常值检测 (IQR)
q1, q3 = np.percentile(amounts, [25, 75])
iqr = q3 - q1
outliers = amounts[(amounts < q1-1.5*iqr) | (amounts > q3+1.5*iqr)]
```

## 输出目录

| 目录 | 用途 |
|------|------|
| `./data_storage/` | 输入数据 |
| `./analysis_reports/` | 分析报告 |
| `./visualizations/` | 可视化图表 |
| `./generated_code/` | 生成的代码 |

## 注意事项 (Important Notes)

1. **报告和文档使用中文**
2. **可视化注意中文字体问题**
3. **数据聚合必须按 order_id 汇总**
4. **优先使用 Pandas 提升性能**
5. **全量数据分析，标注样本量**

## 关键工作流

### 通用数据分析工作流
1. **数据探索**: 上传数据 → 探索性分析 → 发现模式
2. **统计分析**: 假设检验 → 相关性分析 → 异常检测
3. **客户分群**: RFM分析 → 价值分群 → 二八法则
4. **可视化**: 图表创建 → 仪表板 → 交互式展示
5. **报告生成**: 执行摘要 → 详细分析 → 业务建议

### 互联网数据分析工作流

#### 电商综合分析流程
1. **数据探索** (@data-exploration-visualization)
   - 数据概览与质量检查
   - 基础统计与分布分析
   - 可视化探索

2. **转化分析** (@funnel-analysis)
   - 订单漏斗分析
   - 流失节点识别
   - 转化率优化建议

3. **客户价值** (@ltv-predictor)
   - RFM分群
   - LTV计算与预测
   - 价值分层策略

4. **内容分析** (@content-analysis)
   - 评论情感分析
   - 关键词与主题提取
   - 用户洞察

5. **增长策略** (@growth-model-analyzer)
   - AARRR框架分析
   - 增长杠杆识别
   - 策略建议

#### 专项分析场景

**场景A: 新客获取优化**
```
数据探索 → 渠道归因 (@attribution-analysis-modeling) → AB测试 (@ab-testing-analyzer)
```

**场景B: 老客留存提升**
```
LTV预测 (@ltv-predictor) → 漏斗分析 (@funnel-analysis) → 增长策略 (@growth-model-analyzer)
```

**场景C: 产品与内容优化**
```
内容分析 (@content-analysis) → 数据探索 → AB测试 (@ab-testing-analyzer)
```

---

## 项目产出记录

### Olist 电商数据分析（已完成）

**分析时间**: 2026-05-09

**使用技能**:
- @internet-data-analysis (入口)
- @ltv-predictor (LTV预测)
- @content-analysis (内容分析)
- @growth-model-analyzer (增长分析)
- @funnel-analysis (漏斗分析)

**关键发现**:
1. 订单转化漏斗健康：97% 完成率
2. 用户满意度高：平均 4.09 星，77% 用户给 4-5 星
3. 复购率有待提升：当前仅 3.0%
4. 高价值客户（25%）贡献 50% 收入
5. 物流和产品质量是用户最关心的话题

**分析产出文件**:
- `olist_internet_analysis.py` - 综合分析脚本
- `analyze_olist.py` - 基础分析脚本
- `deep_ltv_analysis.py` - LTV深度分析
- `deep_content_analysis.py` - 内容深度分析
- `deep_growth_analysis.py` - 增长深度分析
- `customer_ltv_analysis.csv` - LTV分析结果
- `reviews_content_analysis.csv` - 评论分析结果
- `customer_growth_analysis.csv` - 增长分析结果

**策略建议优先级**:
1. 🔴 高优先级：复购率提升、高价值客户运营、负面评论响应
2. 🟡 中优先级：休眠客户召回、产品品类优化、物流服务优化
3. 🟢 低优先级：数据驱动体系建设、NLP内容分析升级
