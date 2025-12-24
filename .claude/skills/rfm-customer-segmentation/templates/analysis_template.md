# RFM客户分群分析报告模板
# RFM Customer Segmentation Analysis Report Template

---

**报告标题:** RFM客户分群分析报告
**分析日期:** {{ANALYSIS_DATE}}
**数据时间范围:** {{DATA_RANGE_START}} 至 {{DATA_RANGE_END}}
**分析师:** {{ANALYST_NAME}}
**报告版本:** v1.0

---

## 执行摘要 | Executive Summary

### 核心发现 | Key Findings
- **总客户数:** {{TOTAL_CUSTOMERS}} 人
- **总收入:** ￥{{TOTAL_REVENUE}}
- **平均客户价值:** ￥{{AVERAGE_CUSTOMER_VALUE}}

### 分群分布 | Segment Distribution
- **高价值客户:** {{HIGH_VALUE_COUNT}} 人 ({{HIGH_VALUE_PERCENTAGE}}%)
- **中等价值客户:** {{MEDIUM_VALUE_COUNT}} 人 ({{MEDIUM_VALUE_PERCENTAGE}}%)
- **低价值客户:** {{LOW_VALUE_COUNT}} 人 ({{LOW_VALUE_PERCENTAGE}}%)

### 主要建议 | Key Recommendations
1. {{RECOMMENDATION_1}}
2. {{RECOMMENDATION_2}}
3. {{RECOMMENDATION_3}}

---

## 数据概览 | Data Overview

### 数据质量评估 | Data Quality Assessment
- **数据完整性:** {{DATA_COMPLETENESS}}
- **分析期间:** {{ANALYSIS_PERIOD}} 天
- **数据覆盖范围:** {{DATA_COVERAGE}}

### RFM指标统计 | RFM Metrics Statistics

| 指标 | 平均值 | 中位数 | 标准差 | 最小值 | 最大值 |
|------|--------|--------|--------|--------|--------|
| 最近性(Recency) | {{RECENCY_MEAN}} | {{RECENCY_MEDIAN}} | {{RECENCY_STD}} | {{RECENCY_MIN}} | {{RECENCY_MAX}} |
| 频率性(Frequency) | {{FREQUENCY_MEAN}} | {{FREQUENCY_MEDIAN}} | {{FREQUENCY_STD}} | {{FREQUENCY_MIN}} | {{FREQUENCY_MAX}} |
| 金额性(Monetary) | ￥{{MONETARY_MEAN}} | ￥{{MONETARY_MEDIAN}} | ￥{{MONETARY_STD}} | ￥{{MONETARY_MIN}} | ￥{{MONETARY_MAX}} |

---

## 客户分群分析 | Customer Segment Analysis

### 高价值客户 (High Value) - {{HIGH_VALUE_COUNT}} 人

**特征描述:**
- 平均最近购买: {{HIGH_AVG_RECENCY}} 天前
- 平均购买频次: {{HIGH_AVG_FREQUENCY}} 次
- 平均消费金额: ￥{{HIGH_AVG_MONETARY}}
- 收入贡献占比: {{HIGH_REVENUE_CONTRIBUTION}}%

**营销策略:**
- {{HIGH_STRATEGY_1}}
- {{HIGH_STRATEGY_2}}
- {{HIGH_STRATEGY_3}}

**预期效果:**
- {{HIGH_EXPECTED_RESULT_1}}
- {{HIGH_EXPECTED_RESULT_2}}

### 中等价值客户 (Medium Value) - {{MEDIUM_VALUE_COUNT}} 人

**特征描述:**
- 平均最近购买: {{MEDIUM_AVG_RECENCY}} 天前
- 平均购买频次: {{MEDIUM_AVG_FREQUENCY}} 次
- 平均消费金额: ￥{{MEDIUM_AVG_MONETARY}}
- 收入贡献占比: {{MEDIUM_REVENUE_CONTRIBUTION}}%

**营销策略:**
- {{MEDIUM_STRATEGY_1}}
- {{MEDIUM_STRATEGY_2}}
- {{MEDIUM_STRATEGY_3}}

**预期效果:**
- {{MEDIUM_EXPECTED_RESULT_1}}
- {{MEDIUM_EXPECTED_RESULT_2}}

### 低价值客户 (Low Value) - {{LOW_VALUE_COUNT}} 人

**特征描述:**
- 平均最近购买: {{LOW_AVG_RECENCY}} 天前
- 平均购买频次: {{LOW_AVG_FREQUENCY}} 次
- 平均消费金额: ￥{{LOW_AVG_MONETARY}}
- 收入贡献占比: {{LOW_REVENUE_CONTRIBUTION}}%

**营销策略:**
- {{LOW_STRATEGY_1}}
- {{LOW_STRATEGY_2}}
- {{LOW_STRATEGY_3}}

**预期效果:**
- {{LOW_EXPECTED_RESULT_1}}
- {{LOW_EXPECTED_RESULT_2}}

---

## VIP客户清单 | VIP Customer List

### Top 10 高价值客户 | Top 10 High-Value Customers

| 排名 | 客户ID | RFM总分 | 最近购买天数 | 购买频次 | 总消费金额 | 营销建议 |
|------|--------|---------|--------------|----------|------------|----------|
{{TOP_VIP_TABLE}}

---

## 营销活动建议 | Marketing Campaign Recommendations

### 即即行动 | Immediate Actions (本周)
- [ ] {{IMMEDIATE_ACTION_1}}
- [ ] {{IMMEDIATE_ACTION_2}}
- [ ] {{IMMEDIATE_ACTION_3}}

### 短期计划 | Short-term Plans (本月)
- [ ] {{SHORT_TERM_PLAN_1}}
- [ ] {{SHORT_TERM_PLAN_2}}
- [ ] {{SHORT_TERM_PLAN_3}}

### 长期战略 | Long-term Strategy (季度)
- [ ] {{LONG_TERM_STRATEGY_1}}
- [ ] {{LONG_TERM_STRATEGY_2}}
- [ ] {{LONG_TERM_STRATEGY_3}}

---

## KPI监控指标 | KPI Monitoring Metrics

### 核心指标 | Key Metrics
- **客户留存率:** {{RETENTION_RATE_TARGET}}%
- **客户价值提升:** {{CUSTOMER_VALUE_INCREASE_TARGET}}%
- **营销ROI:** {{MARKETING_ROI_TARGET}}%
- **转化率:** {{CONVERSION_RATE_TARGET}}%

### 监控频率 | Monitoring Frequency
- **日报:** {{DAILY_MONITORING_ITEMS}}
- **周报:** {{WEEKLY_MONITORING_ITEMS}}
- **月报:** {{MONTHLY_MONITORING_ITEMS}}
- **季报:** {{QUARTERLY_MONITORING_ITEMS}}

---

## 风险评估与应对 | Risk Assessment & Mitigation

### 潜在风险 | Potential Risks
1. **{{RISK_1}}**
   - 风险等级: {{RISK_1_LEVEL}}
   - 应对措施: {{RISK_1_MITIGATION}}

2. **{{RISK_2}}**
   - 风险等级: {{RISK_2_LEVEL}}
   - 应对措施: {{RISK_2_MITIGATION}}

3. **{{RISK_3}}**
   - 风险等级: {{RISK_3_LEVEL}}
   - 应对措施: {{RISK_3_MITIGATION}}

---

## 技术说明 | Technical Notes

### 分析方法 | Analysis Methodology
- **RFM模型:** Recency, Frequency, Monetary分析
- **聚类算法:** K-means聚类
- **评分标准:** 1-3分制，基于33%和66%分位数
- **数据处理:** 异常值清理、标准化处理

### 工具和版本 | Tools and Versions
- **Python版本:** {{PYTHON_VERSION}}
- **pandas版本:** {{PANDAS_VERSION}}
- **scikit-learn版本:** {{SKLEARN_VERSION}}
- **分析工具:** RFM Customer Segmentation Engine v1.0

---

## 附录 | Appendix

### A. 详细数据表 | Detailed Data Tables
{{DETAILED_DATA_TABLES}}

### B. 可视化图表 | Visualization Charts
{{VISUALIZATION_CHARTS}}

### C. 术语表 | Glossary
- **RFM:** Recency(最近性), Frequency(频率性), Monetary(金额性)
- **K-means:** K均值聚类算法
- **客户生命周期价值:** Customer Lifetime Value (CLV)
- **帕累托法则:** 80/20法则，少数客户贡献大部分收入

---

**报告结束**

*本报告由RFM客户分群分析系统自动生成*
*如有疑问，请联系数据分析团队*

---
**备注:** {{NOTES}}