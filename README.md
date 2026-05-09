# Claude Data Analysis Skills

基于 Claude Skill 架构的智能数据分析平台。提供两套完整的技能体系：

1. **通用数据分析技能** - 6阶段完整分析流程
2. **互联网数据分析技能** - 7个专业分析模块 + 1个入口技能

---

## 🚀 快速开始

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

### 2. 调用 Skills 分析

#### 方式0: 小白一键模式（推荐）
```bash
❯ data_storage/ 是需要进行数据分析的文件， 请你先浏览一下，然后利用skill进行数据分析，尽可能深度洞察出情况
```

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

# 通过主 Skill (支持高级分析模式)
@data-analysis 对 olist数据进行 statistical 分析
```

---

## 🌐 互联网分析模式 (Internet Analysis Mode)

**重要功能**：专为互联网业务场景设计的专业技能体系，提供针对电商、内容、增长等场景的深度分析能力。

### 为什么使用互联网分析模式？

- **场景聚焦**：针对互联网业务特点设计
- **专业深度**：提供 LTV 预测、归因分析、AB 测试等专业能力
- **灵活组合**：可单独使用，也可通过入口技能协调组合
- **开箱即用**：提供 templates/、guide/、examples/ 完整配套

### 互联网分析技能列表

| Skill | 核心能力 | 适用场景 |
|-------|---------|----------|
| **internet-data-analysis** | 入口协调技能 | 需要综合分析时使用 |
| **ltv-predictor** | 客户生命周期价值预测 | 客户价值分层、获客成本优化 |
| **content-analysis** | 文本内容分析 | 评论情感分析、主题挖掘 |
| **funnel-analysis** | 转化漏斗分析 | 流失节点识别、转化率优化 |
| **growth-model-analyzer** | 增长策略分析 | AARRR 框架、Uplift 建模 |
| **ab-testing-analyzer** | A/B测试设计与分析 | 实验设计、统计检验 |
| **attribution-analysis-modeling** | 渠道归因分析 | 营销渠道评估、预算优化 |
| **data-exploration-visualization** | 数据探索与可视化 | 基础分析、数据概览 |

### 互联网分析技能使用示例

```bash
# 综合电商分析（入口技能自动协调）
@internet-data-analysis 对 olist 数据进行全面分析

# 专项分析 - 只做 LTV
@ltv-predictor 分析 olist 客户生命周期价值

# 专项分析 - 只做评论分析
@content-analysis 分析 olist 用户评论

# 专项分析 - 只做增长策略
@growth-model-analyzer 制定 olist 增长策略
```

---

## ⚡ 高级分析模式 (Advanced Mode)

**重要功能**：系统支持高级分析模式，用户可以指定详细的分析类型，控制分析的深度、范围和输出形式。

### 为什么要使用高级分析模式？

- **精细控制**：根据需求选择合适的分析深度
- **效率提升**：避免过度分析或分析不足
- **场景匹配**：不同场景使用不同的分析策略

### 高级模式支持的类型

| Skill | 类型数量 | 可用类型 |
|-------|---------|----------|
| data-analysis | 4种 | exploratory, statistical, predictive, complete |
| visualization-specialist | 6种 | all, trends, distribution, correlation, comparison, custom |
| report-writer | 5种 | summary, complete, executive, technical, custom |
| code-generator | 5种 | data-cleaning, statistical, visualization, machine-learning, custom |

---

### 📊 data-analysis 高级模式

**用途**：控制数据分析的深度

| 类型 | 英文 | 深度 | 分析内容 | 适用场景 |
|------|------|------|----------|----------|
| 探索性 | exploratory | ⭐ | 数据结构、描述性统计、质量检查、基础可视化 | 快速了解数据 |
| 统计性 | statistical | ⭐⭐ | +假设检验、相关性分析、回归分析、异常值检测 | 需要深度统计 |
| 预测性 | predictive | ⭐⭐⭐ | +特征工程、机器学习模型准备、客户分群 | 需要预测建模 |
| 完整性 | complete | ⭐⭐⭐⭐ | 以上全部 + 综合报告、可视化仪表板 | 全面分析 |

**使用示例**：

```bash
# 快速了解数据结构
@data-analysis 分析 house.csv 类型=exploratory

# 需要统计检验
@data-analysis 分析 olist 数据 类型=statistical

# 需要预测分析
@data-analysis 分析 customer.csv 类型=predictive

# 完整分析
@data-analysis 分析 sales.csv 类型=complete
```

---

### 📈 visualization-specialist 高级模式

**用途**：控制图表的类型和用途

| 类型 | 英文 | 图表内容 | 适用场景 |
|------|------|----------|----------|
| 完整仪表板 | all | 综合仪表板、多图表组合 | 数据概览 |
| 趋势分析 | trends | 折线图、移动平均、季节性分析 | 时间序列数据 |
| 分布分析 | distribution | 直方图、密度图、箱线图 | 了解数据分布 |
| 相关性分析 | correlation | 散点图、热力图、配对图 | 变量关系 |
| 对比分析 | comparison | 分组条形图、堆叠图 | 多组对比 |
| 自定义 | custom | 根据需求定制 | 特殊需求 |

**使用示例**：

```bash
# 创建销售趋势图
@visualization-specialist 创建 olist 销售趋势图 类型=trends

# 创建分布图
@visualization-specialist 分析价格分布 类型=distribution

# 创建相关性热力图
@visualization-specialist 分析变量相关 类型=correlation
```

---

### 📝 report-writer 高级模式

**用途**：控制报告的形式和受众

| 类型 | 英文 | 内容特点 | 适用对象 |
|------|------|----------|----------|
| 简要 | summary | 关键发现、核心指标 | 快速查阅 |
| 完整 | complete | 执行摘要、详细分析、可视化 | 全面了解 |
| 高管 | executive | 战略摘要、关键指标、行动项 | 决策层 |
| 技术 | technical | 方法论、统计细节、代码 | 技术团队 |
| 自定义 | custom | 根据需求定制 | 特殊需求 |

**使用示例**：

```bash
# 生成简要报告
@report-writer 生成 olist 报告 类型=summary

# 生成高管报告
@report-writer 生成 olist 报告 类型=executive

# 生成技术报告
@report-writer 生成 olist 报告 类型=technical
```

---

### 💻 code-generator 高级模式

**用途**：控制生成代码的类型

| 类型 | 英文 | 代码内容 | 适用场景 |
|------|------|----------|----------|
| 数据清洗 | data-cleaning | 缺失值处理、标准化、去重 | 数据预处理 |
| 统计分析 | statistical | 描述统计、假设检验、相关分析 | 统计分析 |
| 可视化 | visualization | Matplotlib/Seaborn/Plotly 图表 | 数据展示 |
| 机器学习 | machine-learning | 特征工程、模型训练、评估 | 预测建模 |
| 自定义 | custom | 根据需求定制 | 特殊需求 |

**使用示例**：

```bash
# 生成数据清洗代码
@code-generator 生成 python 数据清洗 代码类型=data-cleaning

# 生成统计代码
@code-generator 生成 r 统计分析 代码类型=statistical

# 生成可视化代码
@code-generator 生成 python 可视化 代码类型=visualization

# 生成机器学习代码
@code-generator 生成 python 预测模型 代码类型=machine-learning
```

---

## 📦 Skills 架构

项目包含两套完整的技能体系：

### 体系 1: 互联网分析技能（新增）

```
.claude/skills/
├── internet-data-analysis/       # 入口技能（协调者）
├── ab-testing-analyzer/         # AB测试分析
├── attribution-analysis-modeling/# 归因分析建模
├── content-analysis/            # 内容分析（NLP）
├── data-exploration-visualization/# 数据探索与可视化
├── funnel-analysis/             # 漏斗分析
├── growth-model-analyzer/       # 增长模型分析
└── ltv-predictor/               # LTV预测
```

#### 各互联网分析 Skill 功能

| Skill | 核心功能 | 配套资源 |
|-------|---------|---------|
| **internet-data-analysis** | 统筹全部互联网分析技能 | SKILL.md |
| **ab-testing-analyzer** | 实验设计、样本量计算、统计检验 | templates/ + guide/ + examples/ |
| **attribution-analysis-modeling** | 首次/末次/线性/马尔可夫/Shapley值归因 | templates/ + guide/ + examples/ |
| **content-analysis** | 情感分析、关键词提取、主题建模 | templates/ + guide/ + examples/ |
| **data-exploration-visualization** | 数据概览、分布分析、可视化 | templates/ + guide/ + examples/ |
| **funnel-analysis** | 漏斗分析、流失节点识别、转化优化 | templates/ + guide/ + examples/ |
| **growth-model-analyzer** | AARRR框架、Uplift建模、增长策略 | templates/ + guide/ + examples/ |
| **ltv-predictor** | RFM分群、LTV预测、价值分层 | templates/ + guide/ + examples/ |

---

### 体系 2: 通用数据分析技能（保留）

```
.Claude/skills/
├── data-analysis/           # 主 Skill（协调者）
├── data-explorer/         # 数据分析
├── visualization-specialist/  # 数据可视化
├── report-writer/        # 报告生成
├── code-generator/        # 代码生成
├── hypothesis-generator  # 假设生成
└── quality-assurance     # 质量保证
```

### 各通用分析 Skill 功能说明

### 1. data-analysis (主 Skill)

入口 Skill，协调完整分析流程。

**功能**：
- 完整工作流：质量验证 → 数据分析 → 假设生成 → 可视化 → 报告
- 快速分析：EDA → 可视化 → 简要报告
- **支持 analysis_type**: exploratory, statistical, predictive, complete

**使用场景**：
- 需要全面分析时调用
- 需要协调多个子 Skill 时使用

### 2. data-explorer (数据分析) ⭐ 核心

**功能**：
- 描述性统计（均值、中位数、标准差、四分位数）
- 深度统计分析（假设检验、p 值、置信区间）
- 分布分析（偏度、峰度、正态性检验）
- 异常值检测（IQR、Z-score）
- 相关性分析（Pearson、Spearman）
- 客户分群（RFM 分析、K-means）
- 模式发现（聚类、趋势分析）

**关键处理规则**（必须遵守）：

| 数据类型 | 正确处理方式 |
|----------|--------------|
| 订单金额 | **必须按 order_id 汇总** (price + freight_value)，不能直接平均 order_items |
| 评分 | 按 order_id 取平均值 |
| 配送时间 | delivered_date - purchase_date |

**使用场景**：
- 需要探索性数据分析（EDA）
- 需要统计检验和推断
- 需要发现数据模式和客户分群

### 3. visualization-specialist (数据可视化)

**功能**：
- 统计图表（直方图、箱线图、散点图）
- 分布图、热力图
- 时间序列图
- 交互式仪表板
- **支持 chart_type**: all, trends, distribution, correlation, comparison, custom

**使用场景**：
- 需要创建图表
- 需要数据可视化
- 需要交互式仪表板

### 4. report-writer (报告生成)

**功能**：
- 执行摘要
- 技术报告
- 业务洞察文档
- **支持 report_type**: summary, complete, executive, technical, custom

**使用场景**：
- 需要生成书面报告
- 需要文档化分析结果
- 需要业务建议文档

### 5. code-generator (代码生成)

**功能**：
- Python/R/SQL 代码
- 数据处理管道
- 机器学习代码
- **支持 code_type**: data-cleaning, statistical, visualization, machine-learning, custom

**使用场景**：
- 需要可重用代码
- 需要自动化分析脚本
- 需要生产级代码

### 6. hypothesis-generator (假设生成)

**功能**：
- 可检验假设形成
- A/B 测试设计
- 样本量计算
- 实验方法论

**使用场景**：
- 需要从数据洞察生成假设
- 需要设计实验
- 需要研究方法论

### 7. quality-assurance (质量保证)

**功能**：
- 数据质量验证
- 缺失值检查
- 重复检测
- 一致性验证

**使用场景**：
- 需要数据质量检查
- 需要结果验证
- 需要交叉验证

---

## 📊 分析工作流

### 完整工作流

```
1. quality-assurance: 数据质量验证
   ↓
2. data-explorer: 探索性数据分析 + 深度统计分析
   ↓
3. hypothesis-generator: 基于发现生成假设
   ↓
4. visualization-specialist: 创建可视化图表
   ↓
5. code-generator: 生成可重用代码
   ↓
6. report-writer: 生成完整分析报告
```

### 快速工作流

```
1. data-explorer: 快速 EDA
2. visualization-specialist: 关键可视化
3. report-writer: 简要报告
```

---

## 📁 目录结构

```
.
├── data_storage/           # 输入数据目录
├── analysis_reports/       # 分析报告输出
├── visualizations/         # 可视化图表输出
├── generated_code/        # 生成的代码输出
├── .claude/               # 原 Claude Agent 配置（保留）
├── .Claude/                 # 通用数据分析 Skills 配置
│   └── skills/
│       ├── data-analysis/
│       ├── data-explorer/
│       ├── visualization-specialist/
│       ├── report-writer/
│       ├── code-generator/
│       ├── hypothesis-generator/
│       └── quality-assurance/
├── .claude/                  # 互联网分析 Skills 配置（新增）
│   └── skills/
│       ├── internet-data-analysis/
│       ├── ab-testing-analyzer/
│       │   ├── templates/      # 报告模板
│       │   ├── guide/          # 操作指南
│       │   └── examples/       # 示例代码
│       ├── attribution-analysis-modeling/
│       │   ├── templates/
│       │   ├── guide/
│       │   └── examples/
│       ├── content-analysis/
│       │   ├── templates/
│       │   ├── guide/
│       │   └── examples/
│       ├── data-exploration-visualization/
│       │   ├── templates/
│       │   ├── guide/
│       │   └── examples/
│       ├── funnel-analysis/
│       │   ├── templates/
│       │   ├── guide/
│       │   └── examples/
│       ├── growth-model-analyzer/
│       │   ├── templates/
│       │   ├── guide/
│       │   └── examples/
│       └── ltv-predictor/
│           ├── templates/
│           ├── guide/
│           └── examples/
├── CLAUDE.md               # 项目详细文档
└── README.md               # 本文件
```

---

## 🔧 数据处理规范

### Pandas vs Pure Python

系统会自动检测环境，优先使用 Pandas（性能更好）：

| 情况 | 推荐方式 | 原因 |
|------|----------|------|
| 数据量 > 10,000 行 | Pandas | 向量化操作，C底层优化 |
| 数据量 < 10,000 行 | Pandas | 代码简洁易维护 |
| 环境无 pandas | Pure Python | 降级兼容 |

**自动检测代码**:
```python
try:
    import pandas as pd
    import numpy as np
    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False
```

### 电商数据处理（必须遵守）

```python
# ❌ 错误：直接平均 order_items
prices = [float(item['price']) for item in order_items]
avg_price = sum(prices) / len(prices)  # 错误！

# ✅ 正确：按订单汇总
from collections import defaultdict
order_amounts = defaultdict(float)
for item in order_items:
    order_amounts[item['order_id']] += float(item['price']) + float(item.get('freight_value', 0))

amounts = list(order_amounts.values())
avg_amount = sum(amounts) / len(amounts)  # 正确！
```

### 样本量要求

- **始终使用全量数据**（除非数据量 > 100万行）
- 在报告中明确标注样本量
- 避免因样本限制导致统计偏差

---

## 📖 完整使用示例

### 示例 A: 互联网数据分析 - 电商综合分析

```
用户: 对 olist 电商数据进行全面互联网分析

调用: @internet-data-analysis
输出:
1. 数据探索 (@data-exploration-visualization)
2. 漏斗分析 (@funnel-analysis)
3. LTV 预测 (@ltv-predictor)
4. 内容分析 (@content-analysis)
5. 增长策略 (@growth-model-analyzer)
```

### 示例 B: 互联网数据分析 - 专项分析

```
用户: 分析 olist 客户生命周期价值

调用: @ltv-predictor
输出: RFM 分群、LTV 分层、价值预测、策略建议
```

```
用户: 分析 olist 用户评论

调用: @content-analysis
输出: 情感分析、关键词、主题建模、用户洞察
```

---

### 示例 1: 基础 EDA（通用分析）

```
用户: 分析 house.csv 数据

调用: data-explorer
输出: 描述性统计、分布分析、相关性矩阵
```

### 示例 2: 指定分析类型

```
用户: 对 olist 电商数据进行 statistical 分析

调用: data-analysis (analysis_type=statistical)
输出: 假设检验、相关性分析、回归分析
```

### 示例 3: 完整分析

```
用户: 对 olist 电商数据进行 complete 分析

调用流程:
1. quality-assurance → 数据质量报告
2. data-explorer → 深度统计分析
3. hypothesis-generator → 业务假设
4. visualization-specialist → 可视化仪表板
5. report-writer → 完整分析报告
```

### 示例 4: 生成可视化图表

```
用户: 创建 olist 销售趋势图

调用: visualization-specialist (chart_type=trends)
输出: 时间序列趋势图
```

### 示例 5: 生成报告

```
用户: 生成 olist 高管报告

调用: report-writer (report_type=executive)
输出: 高管摘要报告
```

---

## 🎯 Skill 调用场景对照表

| 场景 | 调用 Skill | 可选类型 |
|------|------------|----------|
| 探索数据、理解结构 | data-explorer | - |
| 深度统计分析 | data-analysis | exploratory/statistical/predictive/complete |
| 创建图表、可视化 | visualization-specialist | all/trends/distribution/correlation/comparison |
| 生成书面报告 | report-writer | summary/complete/executive/technical |
| 编写可重用代码 | code-generator | data-cleaning/statistical/visualization/machine-learning |
| 生成研究假设 | hypothesis-generator | - |
| 验证数据质量 | quality-assurance | - |
| 完整分析流程 | data-analysis | complete |

---

## 📝 输出规范

所有输出必须使用**中文**：
- 报告内容：中文
- 可视化标签：中文
- 代码注释：中文
- 文件命名：英文（保持一致）

---

## 🔄 与原 Claude Agent 的关系

Skills 版本保留了原 Agent 的核心能力：

| 原 Agent 能力 | Skill 实现 |
|--------------|-----------|
| data-explorer | data-explorer |
| visualization-specialist | visualization-specialist |
| report-writer | report-writer |
| code-generator | code-generator |
| hypothesis-generator | hypothesis-generator (新增) |
| quality-assurance | quality-assurance (新增) |
| 分析类型 (analysis types) | 全部支持 ✅ |

**主要改进**：
1. ✅ 更清晰的数据处理规范（特别是订单金额计算）
2. ✅ 强制使用全量数据
3. ✅ 更详细的统计分析能力
4. ✅ 更完整的分析工作流
5. ✅ **高级分析模式 (Advanced Mode)** 支持多种分析类型

---

## 📞 支持

- 数据格式：CSV、Excel、JSON、Parquet
- 分析类型：exploratory, statistical, predictive, complete
- 图表类型：all, trends, distribution, correlation, comparison
- 报告类型：summary, complete, executive, technical
- 代码类型：data-cleaning, statistical, visualization, machine-learning
- 输出格式：Markdown、HTML、PNG、交互式图表

---

## 📊 项目产出记录

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


## 特别感谢

https://linux.do 感谢佬友们支持