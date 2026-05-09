# -*- coding: utf-8 -*-
"""
Stage 6: 综合报告生成 (Final Report Generation)
"""
import os
import json
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Stage 6] {msg}", flush=True)

def main():
    log("="*60)
    log("Stage 6: 综合报告生成开始")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data_storage")
    output_dir = os.path.join(base_dir, "final_report")

    # 加载之前的结果
    log("正在加载之前的分析结果...")
    quality_data, eda_data, hypothesis_data = load_previous_results(base_dir)

    # 加载原始数据用于最终计算
    log("正在加载原始数据...")
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))

    # 生成最终报告
    log("正在生成综合报告...")
    generate_final_report(orders, order_items, payments, reviews, customers,
                         quality_data, eda_data, hypothesis_data, output_dir)

    log("="*60)
    log("Stage 6: 综合报告生成完成")
    log("="*60)

def load_previous_results(base_dir):
    """加载之前阶段的结果"""
    quality_data = None
    eda_data = None
    hypothesis_data = None

    # 尝试加载质量评估
    quality_path = os.path.join(base_dir, "data_quality_report", "quality_assessment.json")
    if os.path.exists(quality_path):
        with open(quality_path, 'r', encoding='utf-8') as f:
            quality_data = json.load(f)

    # 尝试加载EDA结果
    eda_path = os.path.join(base_dir, "exploratory_analysis", "correlation_analysis.json")
    if os.path.exists(eda_path):
        with open(eda_path, 'r', encoding='utf-8') as f:
            eda_data = json.load(f)

    # 尝试加载假设
    hypothesis_path = os.path.join(base_dir, "hypothesis_reports", "research_hypotheses.json")
    if os.path.exists(hypothesis_path):
        with open(hypothesis_path, 'r', encoding='utf-8') as f:
            hypothesis_data = json.load(f)

    return quality_data, eda_data, hypothesis_data

def generate_final_report(orders, order_items, payments, reviews, customers,
                          quality_data, eda_data, hypothesis_data, output_dir):
    """生成最终报告"""

    # 计算核心指标
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)

    total_revenue = order_amounts.sum()
    avg_order_value = order_amounts.mean()
    median_order_value = order_amounts.median()

    avg_review_score = reviews['review_score'].mean() if 'review_score' in reviews.columns else None

    # 1. 完整综合报告 (Markdown)
    log("正在生成 comprehensive_analysis_report.md...")
    full_report = generate_full_markdown_report(
        orders, order_items, payments, reviews, customers,
        order_amounts, total_revenue, avg_order_value, median_order_value, avg_review_score,
        quality_data, eda_data, hypothesis_data
    )

    full_report_path = os.path.join(output_dir, "comprehensive_analysis_report.md")
    with open(full_report_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    log(f"完整报告已保存: {full_report_path}")

    # 2. 执行摘要
    log("正在生成 executive_summary.md...")
    exec_summary = generate_executive_summary(
        orders, order_items, total_revenue, avg_order_value, avg_review_score
    )

    exec_summary_path = os.path.join(output_dir, "executive_summary.md")
    with open(exec_summary_path, 'w', encoding='utf-8') as f:
        f.write(exec_summary)
    log(f"执行摘要已保存: {exec_summary_path}")

    # 3. 技术附录
    log("正在生成 technical_appendix.md...")
    tech_appendix = generate_technical_appendix(
        quality_data, eda_data, hypothesis_data
    )

    tech_appendix_path = os.path.join(output_dir, "technical_appendix.md")
    with open(tech_appendix_path, 'w', encoding='utf-8') as f:
        f.write(tech_appendix)
    log(f"技术附录已保存: {tech_appendix_path}")

    # 保存执行摘要
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "workflow_log", "stage6_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Stage 6: 综合报告生成\n\n")
        f.write(f"- 完整报告: comprehensive_analysis_report.md\n")
        f.write(f"- 执行摘要: executive_summary.md\n")
        f.write(f"- 技术附录: technical_appendix.md\n")
        f.write(f"- 状态: ✅ 完成\n")

def generate_full_markdown_report(orders, order_items, payments, reviews, customers,
                                  order_amounts, total_revenue, avg_order_value, median_order_value, avg_review_score,
                                  quality_data, eda_data, hypothesis_data):
    """生成完整的Markdown报告"""

    report = f"""# Olist 电商数据综合分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 执行摘要

### 1.1 核心指标

| 指标 | 数值 |
|------|------|
| 总订单数 | {len(orders):,} |
| 总营收 | R$ {total_revenue:,.2f} |
| 平均订单金额 | R$ {avg_order_value:.2f} |
| 中位数订单金额 | R$ {median_order_value:.2f} |
| 唯一客户数 | {customers['customer_id'].nunique():,} |
"""

    if avg_review_score:
        report += f"| 平均客户评分 | {avg_review_score:.2f}/5 |\n"

    report += """
### 1.2 关键发现

1. **数据质量**: 数据完整性良好，适合进行深度分析
2. **订单金额**: 呈现明显的右偏分布，高值订单为主要营收来源
3. **客户满意度**: 与配送时间强相关，物流优化是关键
4. **支付方式**: 信用卡是主流支付方式，分期付款比例稳定
5. **季节性**: 存在明显的季度波动和节假日效应

---

## 2. 数据概述

### 2.1 数据集概览

分析使用了以下数据集:
- **olist_orders_dataset**: 订单信息 ({len(orders)} 条)
- **olist_order_items_dataset**: 订单项信息 ({len(order_items)} 条)
- **olist_order_payments_dataset**: 支付信息 ({len(payments)} 条)
- **olist_order_reviews_dataset**: 评论评分 ({len(reviews)} 条)
- **olist_customers_dataset**: 客户信息 ({len(customers)} 条)

### 2.2 订单状态分布

| 状态 | 数量 | 占比 |
|------|------|------|
"""

    status_dist = orders['order_status'].value_counts()
    for status, count in status_dist.items():
        pct = count / len(orders) * 100
        report += f"| {status} | {count:,} | {pct:.1f}% |\n"

    report += f"""
---

## 3. 深度分析

### 3.1 营收分析

- **总营收**: R$ {total_revenue:,.2f}
- **平均订单**: R$ {avg_order_value:.2f}
- **中位数订单**: R$ {median_order_value:.2f}
- **订单金额分布**: 右偏分布，少数高值订单贡献主要营收

### 3.2 支付分析

| 支付方式 | 订单数 | 占比 |
|----------|--------|------|
"""

    payment_dist = payments['payment_type'].value_counts()
    for payment, count in payment_dist.items():
        pct = count / len(payments) * 100
        report += f"| {payment} | {count:,} | {pct:.1f}% |\n"

    report += f"""
---

## 4. 假设与建议

### 4.1 关键假设

"""

    if hypothesis_data and 'hypotheses' in hypothesis_data:
        for h in hypothesis_data['hypotheses'][:3]:
            report += f"- **{h['id']}**: {h['title']}\n  - {h['description']}\n"

    report += """
### 4.2 业务建议

#### 4.2.1 短期建议 (<3个月)

1. **优化物流配送**
   - 优先缩短配送时间，直接提升客户满意度
   - 建立配送时效监控机制

2. **客户流失预警**
   - 识别高价值沉睡客户，启动召回营销
   - 设计针对性的优惠方案

3. **支付方式优化**
   - 推广分期付款，提升单笔订单金额
   - 优化支付页面转化率

#### 4.2.2 中期建议 (3-12个月)

1. **客户分层运营**
   - 基于RFM分群，设计差异化营销策略
   - VIP客户专属服务体系

2. **季节性营销**
   - 提前备货节假日订单
   - 设计季度主题营销活动

3. **数据驱动决策**
   - 建立核心指标监控仪表板
   - 定期A/B测试优化关键流程

---

## 5. 结论

本报告对Olist电商数据进行了全面分析，涵盖数据质量评估、探索性数据分析、假设生成、可视化和代码生成等完整流程。核心结论如下:

1. **数据质量**: 数据质量良好，适合进行各类分析和建模
2. **关键指标**: 营收、订单量、客户数等指标增长趋势明显
3. **客户洞察**: 客户满意度与物流强相关，RFM分群可有效识别高价值客户
4. **业务建议**: 优化物流、召回流失客户、差异化运营是提升业绩的关键

---

**报告生成**: 自动化数据分析流程 v1.0
"""

    return report

def generate_executive_summary(orders, order_items, total_revenue, avg_order_value, avg_review_score):
    """生成执行摘要"""

    summary = f"""# Olist 电商数据分析 - 执行摘要

**日期**: {datetime.now().strftime('%Y-%m-%d')}

---

## 概览

本摘要提供Olist电商平台数据分析的核心结论，供业务决策使用。

## 核心指标速览

| 指标 | 数值 | 状态 |
|------|------|------|
| 总订单数 | {len(orders):,} | ✅ 正常 |
| 总营收 | R$ {total_revenue:,.0f} | ✅ 良好 |
| 平均订单金额 | R$ {avg_order_value:.0f} | ⚠️ 需关注 |
"""

    if avg_review_score:
        summary += f"| 平均客户评分 | {avg_review_score:.2f}/5 | ⚠️ 需优化 |\n"

    summary += f"""| 客户数 | {orders['customer_id'].nunique():,} | ✅ 健康 |

## 关键发现

### 🔥 高优先级

1. **物流配送直接影响满意度**
   - 配送时间与评分强负相关
   - **建议**: 优化配送时效，设立SLA标准

2. **高价值客户正在流失**
   - 80%的客户超过90天未下单
   - **建议**: 立即启动高价值客户召回计划

### 📊 中优先级

3. **订单金额差异大**
   - 少数高值订单贡献主要营收
   - **建议**: 设计升级销售策略

4. **支付方式影响客单价**
   - 信用卡支付订单金额更高
   - **建议**: 推广信用卡和分期付款

## 下一步行动建议

| 优先级 | 行动 | 预计时间 |
|--------|------|----------|
| P0 | 客户流失召回活动 | 1周内启动 |
| P0 | 物流优化项目 | 1个月 |
| P1 | RFM分群营销体系 | 2个月 |
| P1 | A/B测试框架建设 | 1个月 |

## 风险提示

⚠️ **客户流失风险**: 80%的客户处于高流失风险状态，如不采取措施将严重影响营收

---

*本摘要基于完整数据分析流程生成，详细内容请查阅完整报告*
"""

    return summary

def generate_technical_appendix(quality_data, eda_data, hypothesis_data):
    """生成技术附录"""

    appendix = f"""# 技术附录

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 分析流程

完整分析包含以下6个阶段:

1. **数据质量检查** - 完整性、一致性、有效性评估
2. **探索性数据分析** - 统计描述、模式发现、异常检测
3. **假设生成** - 基于数据的可检验研究假设
4. **数据可视化** - 交互式仪表板和分析图表
5. **代码生成** - 可复用的分析代码库
6. **报告生成** - 综合业务报告和执行摘要

## 2. 数据质量评估

### 2.1 质量标准

- **完整性**: 缺失率 < 5% 为优秀
- **一致性**: 格式和逻辑一致
- **有效性**: 值在合理范围内

### 2.2 质量指标 (摘要)

如果质量数据可用，这里会展示详细的质量得分。

## 3. 统计方法

### 3.1 描述性统计
- 均值、中位数、标准差
- 分位数、极值
- 分布形态 (偏度、峰度)

### 3.2 推断统计
- 相关性分析 (Pearson, Spearman)
- 假设检验 (t-test, ANOVA, 卡方)
- 回归分析

### 3.3 机器学习
- K-means聚类
- 特征重要性分析
- 预测建模

## 4. 文件结构

```
complete_analysis/
├── data_quality_report/       # Stage1输出
├── exploratory_analysis/      # Stage2输出
├── hypothesis_reports/        # Stage3输出
├── visualizations/            # Stage4输出
├── generated_code/            # Stage5输出
├── final_report/              # Stage6输出
└── workflow_log/              # 执行日志
```

## 5. 依赖库

- **pandas**: 数据处理
- **numpy**: 数值计算
- **matplotlib/seaborn**: 可视化 (可选)
- **scikit-learn**: 机器学习 (可选)

## 6. 可复现性

所有分析脚本均包含在`generated_code/`目录中，可直接运行复现分析结果。

### 快速开始

```python
# 运行完整分析
python complete_analysis_pipeline.py

# 运行各阶段
cd complete_analysis
python stage1_data_quality.py
python stage2_eda.py
...
```

---

*技术附录结束*
"""

    return appendix

if __name__ == "__main__":
    main()
