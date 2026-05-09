# 数据质量评估报告

**生成时间**: 2026-05-09 20:38:44

## 整体质量评分: 97.9/100

### 各数据集质量

| 数据集 | 行数 | 列数 | 质量分数 | 问题数 |
|--------|------|------|----------|--------|
| olist_orders_dataset.csv | 99441 | 8 | 99.6 | 0 |
| olist_order_items_dataset.csv | 112650 | 7 | 100.0 | 0 |
| olist_order_payments_dataset.csv | 103886 | 5 | 100.0 | 0 |
| olist_order_reviews_dataset.csv | 99224 | 7 | 86.0 | 2 |
| olist_customers_dataset.csv | 99441 | 5 | 100.0 | 0 |
| olist_products_dataset.csv | 32951 | 9 | 99.4 | 0 |
| olist_sellers_dataset.csv | 3095 | 4 | 100.0 | 0 |
| olist_geolocation_dataset.csv | 1000163 | 5 | 95.6 | 1 |
| product_category_name_translation.csv | 71 | 2 | 100.0 | 0 |

## 详细分析

### olist_orders_dataset.csv

- 形状: 99441 行 × 8 列
- 质量分数: 99.6/100
- 完整性分数: 98.8
- 唯一性分数: 100.0
- 有效性分数: 100.0

### olist_order_items_dataset.csv

- 形状: 112650 行 × 7 列
- 质量分数: 100.0/100
- 完整性分数: 100.0
- 唯一性分数: 100.0
- 有效性分数: 100.0

### olist_order_payments_dataset.csv

- 形状: 103886 行 × 5 列
- 质量分数: 100.0/100
- 完整性分数: 100.0
- 唯一性分数: 100.0
- 有效性分数: 100.0

### olist_order_reviews_dataset.csv

- 形状: 99224 行 × 7 列
- 质量分数: 86.0/100
- 完整性分数: 58.0
- 唯一性分数: 100.0
- 有效性分数: 100.0

#### 发现的问题:
- 高缺失值: review_comment_title (88.3%)
- 高缺失值: review_comment_message (58.7%)

### olist_customers_dataset.csv

- 形状: 99441 行 × 5 列
- 质量分数: 100.0/100
- 完整性分数: 100.0
- 唯一性分数: 100.0
- 有效性分数: 100.0

### olist_products_dataset.csv

- 形状: 32951 行 × 9 列
- 质量分数: 99.4/100
- 完整性分数: 98.3
- 唯一性分数: 100.0
- 有效性分数: 100.0

### olist_sellers_dataset.csv

- 形状: 3095 行 × 4 列
- 质量分数: 100.0/100
- 完整性分数: 100.0
- 唯一性分数: 100.0
- 有效性分数: 100.0

### olist_geolocation_dataset.csv

- 形状: 1000163 行 × 5 列
- 质量分数: 95.6/100
- 完整性分数: 100.0
- 唯一性分数: 86.9
- 有效性分数: 100.0

#### 发现的问题:
- 重复行: 26.2%

### product_category_name_translation.csv

- 形状: 71 行 × 2 列
- 质量分数: 100.0/100
- 完整性分数: 100.0
- 唯一性分数: 100.0
- 有效性分数: 100.0

## 改进建议

1. **处理缺失值**: 对于缺失率>10%的字段，考虑删除、填充或标记
2. **删除重复**: 检查并删除重复行
3. **数据验证**: 确保数据类型和格式正确
4. **持续监控**: 建立数据质量监控机制
