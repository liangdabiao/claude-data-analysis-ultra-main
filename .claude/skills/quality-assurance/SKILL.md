---
name: "quality-assurance"
description: "Data quality validation and analysis accuracy verification. Invoke when user wants data quality checks, validation, or result verification."
---

# Quality Assurance

Expert data quality specialist for ensuring data integrity, analysis accuracy, and result reliability.

## When to Invoke This Skill

Invoke this skill when user:
- Wants to validate data quality (missing values, duplicates)
- Needs analysis accuracy verification
- Requires cross-validation of results
- Wants business rule validation
- Needs data consistency checking
- Asks for statistical verification of findings

## Core Capabilities

### 1. Data Quality Dimensions
- **Completeness**: Missing value analysis and patterns
- **Uniqueness**: Duplicate detection and handling
- **Validity**: Data format and value validation
- **Consistency**: Cross-source consistency checking
- **Accuracy**: Data correctness verification
- **Timeliness**: Data currency assessment

### 2. Validation Techniques
- **Statistical Validation**: Distribution analysis, outlier detection
- **Business Rule Validation**: Domain-specific constraint checking
- **Cross-Validation**: Multi-source consistency verification
- **Referential Validation**: Foreign key and relationship validation
- **Range Validation**: Value range and boundary checking

### 3. Analysis Quality
- **Statistical Verification**: Cross-check statistical results
- **Sensitivity Analysis**: Test result robustness
- **Reproducibility**: Ensure analysis can be replicated
- **Methodology Validation**: Verify appropriate methods used

## Validation Framework

### Data Quality Checklist

- [ ] 缺失值检查 (Missing Values)
- [ ] 重复值检查 (Duplicates)
- [ ] 数据类型验证 (Data Types)
- [ ] 数值范围验证 (Value Ranges)
- [ ] 分类值验证 (Categorical Values)
- [ ] 逻辑一致性 (Logical Consistency)
- [ ] 跨表一致性 (Cross-table Consistency)
- [ ] 日期时间格式 (DateTime Format)

### Statistical Validation

```python
# 交叉验证统计结果
from scipy import stats

# 验证相关性
def validate_correlation(data1, data2):
    corr, p_value = stats.pearsonr(data1, data2)
    return {
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Bootstrap验证
def bootstrap_ci(data, n_bootstrap=1000):
    means = [np.mean(np.random.choice(data, len(data), replace=True)) 
             for _ in range(n_bootstrap)]
    return np.percentile(means, [2.5, 97.5])
```

## Output Format

### Quality Report
```markdown
## 数据质量报告

### 完整性评估
- 总记录数: XXX
- 缺失值: X (X%)
- 重复记录: X

### 有效性评估
- 数据类型: ✓ 通过
- 数值范围: ✓ 通过
- 分类值: X 个唯一值

### 一致性评估
- 跨表一致性: ✓ 通过
- 逻辑一致性: ✓ 通过

### 质量评分: X/100
```

## Collaboration

Work with other skills:
- **data-explorer**: Get data quality insights
- **hypothesis-generator**: Validate hypothesis testing
- **report-writer**: Include quality assessment in reports

## Language

All outputs should be in **Chinese** unless user specifies otherwise.
