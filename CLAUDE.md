# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述 (Project Overview)

Claude Data Analysis Skills 是一个基于 Trae Skill 架构的智能数据分析平台。通过模块化的 Skills 实现完整的数据分析工作流，支持 Advanced Mode 高级分析模式。

**主要特性**:
- Skills 模块化架构
- Advanced Mode 多种分析类型
- 自动检测 Pandas 优化性能
- 正确的数据聚合规则

## 快速开始 (Quick Start)

### 1. 准备数据
将数据集放入 `data_storage/` 目录：
```
data_storage/
├── olist_orders_dataset.csv
├── olist_customers_dataset.csv
└── house.csv
```

### 2. 调用 Skills
```bash
# 方式1: 直接使用 Skill
@data-explorer 分析 olist 电商数据
@visualization-specialist 创建销售可视化
@report-writer 生成分析报告

# 方式2: 通过主 Skill (支持 Advanced Mode)
@data-analysis 对 olist数据进行 statistical 分析
```

## Skills 架构

```
.trae/skills/
├── data-analysis/           # 主 Skill（协调者）
├── data-explorer/          # 数据分析
├── visualization-specialist/ # 数据可视化
├── report-writer/          # 报告生成
├── code-generator/         # 代码生成
├── hypothesis-generator/    # 假设生成
└── quality-assurance/      # 质量保证
```

### 各 Skill 功能

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

### 示例 1: 基础 EDA
```bash
用户: 分析 house.csv 数据
调用: data-explorer
```

### 示例 2: 指定分析类型
```bash
用户: 对 olist 电商数据进行 statistical 分析
调用: data-analysis (analysis_type=statistical)
```

### 示例 3: 完整分析流程
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

### 示例 4: Python 代码 (Pandas 推荐)
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

1. **数据探索**: 上传数据 → 探索性分析 → 发现模式
2. **统计分析**: 假设检验 → 相关性分析 → 异常检测
3. **客户分群**: RFM分析 → 价值分群 → 二八法则
4. **可视化**: 图表创建 → 仪表板 → 交互式展示
5. **报告生成**: 执行摘要 → 详细分析 → 业务建议
