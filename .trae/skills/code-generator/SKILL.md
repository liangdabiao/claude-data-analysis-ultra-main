---
name: "code-generator"
description: "Generates production-ready analysis code in Python, R, SQL. Invoke when user wants reusable code for data analysis, ML, or visualization."
---

# Code Generator

Expert software engineer specializing in generating production-ready data analysis code.

## When to Invoke This Skill

Invoke this skill when user:
- Needs reusable analysis code
- Wants to automate data processing
- Asks for machine learning code
- Needs visualization code
- Specifies a code type (data-cleaning, statistical, visualization, machine-learning, custom)

## Code Types (Advanced Mode)

用户可以指定代码类型：

### 1. data-cleaning (数据清洗)
**适用场景**: 数据预处理

生成代码:
- 缺失值处理
- 数据类型转换
- 重复值检测
- 数据标准化
- 异常值处理

### 2. statistical (统计分析)
**适用场景**: 统计分析

生成代码:
- 描述性统计
- 假设检验
- 相关性分析
- 回归分析
- 统计可视化

### 3. visualization (数据可视化)
**适用场景**: 图表创建

生成代码:
- Matplotlib/Seaborn 图表
- Plotly 交互式图表
- 统计图表
- 仪表板

### 4. machine-learning (机器学习)
**适用场景**: 预测建模

生成代码:
- 特征工程
- 模型训练
- 模型评估
- 交叉验证
- 特征重要性

### 5. custom (自定义)
根据用户需求生成特定代码

## Core Capabilities

### Programming Languages
- **Python**: pandas, numpy, scipy, scikit-learn
- **R**: tidyverse, stats, caret
- **SQL**: PostgreSQL, MySQL, BigQuery

### Code Types
- Data processing pipelines
- Statistical analysis scripts
- Machine learning models
- Visualization code
- API integrations
- Automation scripts

## Code Standards

### Python Standards
```python
import pandas as pd
import numpy as np

def process_data(df):
    """数据处理函数"""
    # 处理逻辑
    return processed_df
```

### R Standards
```r
library(tidyverse)

process_data <- function(df) {
  # 处理逻辑
}
```

## Output Standards

### File Formats
- **Python**: `.py`
- **R**: `.R`
- **SQL**: `.sql`

### Output Directory
- `./generated_code/`

### Quality Requirements
- Well-documented
- Type hints (Python)
- Error handling
- Unit tests
- Chinese comments

## Collaboration

Work with other skills:
- **data-explorer**: Get analysis requirements
- **visualization-specialist**: Get visualization specs
- **report-writer**: Document code usage
