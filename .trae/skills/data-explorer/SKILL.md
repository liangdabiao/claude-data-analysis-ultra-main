---
name: "data-explorer"
description: "Performs exploratory data analysis, statistical analysis, and pattern discovery. Invoke when user wants to analyze data, find patterns, statistical testing, or get deep insights."
---

# Data Explorer

Expert data scientist specializing in exploratory data analysis (EDA) and statistical analysis. Helps discover meaningful patterns, insights, and relationships in data.

## When to Invoke This Skill

Invoke this skill when user:
- Wants to explore and understand a dataset
- Needs deep statistical analysis (inference, hypothesis testing, p-values)
- Wants distribution analysis (skewness, kurtosis, normality tests)
- Needs outlier detection (IQR, Z-score)
- Asks for clustering or segmentation (K-means)
- Needs correlation analysis with significance testing
- Wants RFM analysis or customer segmentation
- Needs business intelligence insights from data

## Core Capabilities

### 1. Basic Statistical Analysis
- Descriptive statistics (mean, median, std, quartiles, percentiles)
- Summary statistics for all variables
- Data type identification

### 2. Deep Statistical Analysis (Advanced)
- **Inferential Statistics**: Hypothesis testing, confidence intervals, p-values
- **Distribution Analysis**: Skewness, kurtosis, normality tests (Shapiro-Wilk)
- **Correlation Analysis**: Pearson, Spearman with significance levels
- **ANOVA**: Analysis of variance for group comparisons
- **Chi-square Test**: Categorical variable independence testing

### 3. Outlier Detection
- **IQR Method**: Interquartile range based detection
- **Z-score Method**: Standard deviation based detection
- **Treatment Strategies**: Remove, cap, or transform outliers

### 4. Pattern Discovery
- **Clustering**: K-means, hierarchical clustering for segmentation
- **Trend Analysis**: Time series decomposition
- **Association Rules**: Market basket analysis
- **Dimensionality Reduction**: PCA for feature importance

### 5. Customer Analysis (E-commerce)
- **RFM Analysis**: Recency, Frequency, Monetary value
- **Customer Segmentation**: High-value, at-risk, churned
- **Customer Lifetime Value**: CLV calculation

### 6. Data Quality Assessment
- Missing value patterns and imputation
- Duplicate detection
- Data consistency checking
- Data profiling

## CRITICAL: Data Processing Rules

### 1. Pandas vs Pure Python - When to Use Which?

**使用 Pandas 的情况**:
- 数据量较大 (>10,000 行)
- 需要复杂的数据操作 (merge, groupby, pivot)
- 需要高效统计分析
- 追求代码简洁和可维护性

**使用 Pure Python 的情况**:
- 数据量较小 (<10,000 行)
- 简单统计计算
- 环境没有安装 pandas

**自动检测并使用 Pandas**:
```python
# 自动检测是否有 pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if HAS_PANDAS:
    # 使用 Pandas (推荐，数据量大时性能更好)
    df = pd.read_csv('data.csv')
    result = df.groupby('category')['value'].sum()
else:
    # 降级到纯 Python
    from collections import defaultdict
    result = defaultdict(float)
    # ... 手动实现
```

### 2. Order Amount Calculation (IMPORTANT!)
For e-commerce datasets, **MUST** calculate order amounts correctly:
- **WRONG**: Just average all order_items (double counts multi-item orders)
- **RIGHT**: Aggregate by order_id first, then calculate statistics

```python
# 正确的订单金额计算
from collections import defaultdict

# 按订单汇总 (order_items -> order)
order_total = defaultdict(float)
for item in order_items:
    order_total[item['order_id']] += float(item['price']) + float(item.get('freight_value', 0))

# 然后计算统计量
all_order_amounts = list(order_total.values())
mean_amount = sum(all_order_amounts) / len(all_order_amounts)
```

### Sample Size Requirements
- **ALWAYS** use full dataset for analysis (no limit)
- If dataset is too large (>1M rows), sample with appropriate method
- Report sample size in results

### Data Aggregation Rules
| 数据类型 | 聚合方式 |
|----------|----------|
| 订单金额 | 按 order_id 汇总 (price + freight_value) |
| 评分 | 按 order_id 取平均值或最新值 |
| 支付金额 | 按 order_id 汇总 |
| 配送时间 | 按 order_id 计算 (delivered - purchase) |

## Analysis Workflow

### Phase 1: Data Understanding
1. Load ALL data (no sampling unless necessary)
2. Examine dataset structure and relationships
3. Identify data types and key variables
4. Check for data quality issues
5. Report actual record counts

### Phase 2: Correct Data Processing
1. **Aggregate data properly** (especially order amounts)
2. Generate summary statistics on aggregated data
3. Distribution analysis (skewness, kurtosis)
4. Correlation matrix with p-values
5. Hypothesis testing where appropriate
6. Outlier detection and treatment

### Phase 3: Advanced Pattern Discovery
1. Clustering analysis for segmentation
2. Trend and seasonality detection
3. Feature importance analysis

### Phase 4: Insight Generation
1. Translate findings into business insights
2. Provide actionable recommendations
3. Suggest visualization approaches

## Usage Examples

### Statistical Analysis Code (推荐使用 Pandas)

```python
# 自动检测 pandas
try:
    import pandas as pd
    import numpy as np
    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False

if USE_PANDAS:
    # ============ Pandas 版本 (推荐，数据量大时使用) ============
    # 读取数据
    orders = pd.read_csv('./data_storage/olist_orders_dataset.csv')
    order_items = pd.read_csv('./data_storage/olist_order_items_dataset.csv')
    
    # 正确的订单金额统计 (按order_id聚合)
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    
    amounts = order_amounts.values
    
    # 描述性统计
    mean_amount = amounts.mean()
    median_amount = np.median(amounts)
    std_amount = amounts.std()
    q1, q2, q3 = np.percentile(amounts, [25, 50, 75])
    
    # 偏度和峰度
    skewness = pd.Series(amounts).skew()
    kurtosis = pd.Series(amounts).kurtosis()
    
    # 异常值检测 (IQR)
    q1 = np.percentile(amounts, 25)
    q3 = np.percentile(amounts, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = amounts[(amounts < lower) | (amounts > upper)]
    
    # RFM 分析
    latest_date = orders['order_purchase_timestamp'].max()
    rfm = orders.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (pd.to_datetime(latest_date) - pd.to_datetime(x).max()).days,
        'order_id': 'count',
        'revenue': 'sum'
    })
    
else:
    # ============ Pure Python 版本 (备用) ============
    from collections import defaultdict
    
    # 按订单聚合 (重要!)
    order_amounts = defaultdict(float)
    for item in order_items:
        order_amounts[item['order_id']] += float(item['price']) + float(item.get('freight_value', 0))
    
    amounts = list(order_amounts.values())
    
    # 描述性统计
    mean_amount = sum(amounts) / len(amounts)
    sorted_amounts = sorted(amounts)
    n = len(sorted_amounts)
    median_amount = (sorted_amounts[n//2-1] + sorted_amounts[n//2]) / 2 if n % 2 == 0 else sorted_amounts[n//2]
    
    # 异常值检测 (IQR)
    q1 = sorted_amounts[n//4]
    q3 = sorted_amounts[3*n//4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [x for x in amounts if x < lower or x > upper]
```

## Output Standards

### Analysis Report Should Include
1. **Data Overview** - Actual record counts (no sampling)
2. **Correct Aggregation** - Order-level statistics
3. **Executive Summary** - Key findings in plain language
4. **Statistical Findings** - Detailed statistical analysis
5. **Advanced Analysis** - Hypothesis testing, clustering results
6. **Key Insights** - Actionable discoveries
7. **Recommendations** - Next steps for deeper analysis

### Quality Assurance
- Validate all statistical calculations on aggregated data
- Cross-check important findings
- Document assumptions and limitations
- Ensure reproducible analysis

## Collaboration

Work with other skills:
- **visualization-specialist**: Provide insights for visualization
- **report-writer**: Supply findings for reports
- **code-generator**: Generate analysis code
- **hypothesis-generator**: Create testable hypotheses
- **quality-assurance**: Validate data quality

## Language

All outputs should be in **Chinese** unless user specifies otherwise. Use Chinese for:
- Report content and summaries
- Visualization labels and titles
- Code comments and documentation

## Data Location

- Input data: `./data_storage/`
- Output reports: `./analysis_reports/`
- Generated code: `./generated_code/`
