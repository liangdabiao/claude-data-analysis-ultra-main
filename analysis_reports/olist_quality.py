# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import re
from datetime import datetime

def read_csv(filename):
    data = []
    with open(f'./data_storage/{filename}', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None

print('=' * 70)
print('Olist 数据集质量验证报告')
print('=' * 70)

# 加载所有数据集
datasets = {
    'customers': read_csv('olist_customers_dataset.csv'),
    'orders': read_csv('olist_orders_dataset.csv'),
    'order_items': read_csv('olist_order_items_dataset.csv'),
    'products': read_csv('olist_products_dataset.csv'),
    'payments': read_csv('olist_order_payments_dataset.csv'),
    'sellers': read_csv('olist_sellers_dataset.csv'),
    'reviews': read_csv('olist_order_reviews_dataset.csv')
}

print('\n【1. 完整性评估】')
total_records = 0
total_fields = 0
total_empty = 0

for name, data in datasets.items():
    total_records += len(data)
    if len(data) > 0:
        total_fields += len(data[0])
        for row in data:
            for key, value in row.items():
                if not value or value.strip() == '':
                    total_empty += 1

print(f'总记录数: {total_records}')
print(f'总字段数: {total_fields}')
print(f'空值总数: {total_empty}')

# 各数据集缺失值检查
print('\n【2. 各数据集缺失值检查】')
for name, data in datasets.items():
    empty_count = 0
    for row in data:
        for key, value in row.items():
            if not value or value.strip() == '':
                empty_count += 1
    total_cells = len(data) * len(data[0]) if data else 0
    pct = (empty_count / total_cells * 100) if total_cells > 0 else 0
    print(f'{name}: {empty_count} 空值 ({pct:.2f}%)')

# 3. 重复值检查
print('\n【3. 重复值检查】')
for name, data in datasets.items():
    if data and len(data) > 0:
        ids = []
        id_field = 'customer_id' if 'customer' in name else 'order_id' if 'order' in name else 'product_id' if 'product' in name else 'seller_id'
        if id_field in data[0]:
            ids = [row[id_field] for row in data]
            unique_ids = set(ids)
            print(f'{name}: 总记录={len(ids)}, 唯一ID={len(unique_ids)}, 重复={len(ids)-len(unique_ids)}')

# 4. 订单状态验证
print('\n【4. 订单状态有效性】')
valid_statuses = {'delivered', 'shipped', 'canceled', 'unavailable', 'invoiced', 'processing', 'created', 'approved'}
orders = datasets['orders']
invalid_status = 0
for order in orders:
    if order['order_status'] not in valid_statuses:
        invalid_status += 1
print(f'有效订单状态: {len(valid_statuses)} 种')
print(f'无效状态记录: {invalid_status}')

# 5. 数值范围验证
print('\n【5. 数值范围验证】')
# 订单金额
order_items = datasets['order_items']
prices = [float(item['price']) for item in order_items if item['price']]
print(f'价格: 最小={min(prices):.2f}, 最大={max(prices):.2f}, 异常值={sum(1 for p in prices if p < 0)}')

# 评分范围
reviews = datasets['reviews']
scores = [int(r['review_score']) for r in reviews if r.get('review_score')]
print(f'评分: 范围=[{min(scores)}, {max(scores)}], 异常值={sum(1 for s in scores if s < 1 or s > 5)}')

# 6. 跨表一致性
print('\n【6. 跨表一致性检查】')
# 客户ID一致性
customer_ids = set(c['customer_id'] for c in datasets['customers'])
order_customer_ids = set(o['customer_id'] for o in orders)
missing_customers = order_customer_ids - customer_ids
print(f'订单引用的客户ID: {len(order_customer_ids)}')
print(f'缺失的客户ID: {len(missing_customers)}')

# 订单ID一致性
order_ids = set(o['order_id'] for o in orders)
payment_order_ids = set(p['order_id'] for p in datasets['payments'])
order_item_order_ids = set(o['order_id'] for o in order_items)
print(f'支付记录引用的订单ID: {len(payment_order_ids)}')
print(f'订单商品引用的订单ID: {len(order_item_order_ids)}')
print(f'支付与订单匹配: {len(payment_order_ids & order_ids)}/{len(payment_order_ids)}')
print(f'订单商品与订单匹配: {len(order_item_order_ids & order_ids)}/{len(order_item_order_ids)}')

# 7. 日期时间验证
print('\n【7. 日期时间格式验证】')
invalid_dates = 0
for order in orders[:1000]:  # 检查前1000条
    for date_field in ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']:
        if order.get(date_field):
            if parse_date(order[date_field]) is None:
                invalid_dates += 1
print(f'无效日期格式: {invalid_dates}')

# 8. 数据质量评分
print('\n【8. 数据质量评分】')
score = 100
# 扣分项
if total_empty > 0:
    score -= min(20, total_empty / total_records * 100)
if invalid_status > 0:
    score -= min(10, invalid_status / len(orders) * 100)
if len(missing_customers) > 0:
    score -= min(10, len(missing_customers) / len(order_customer_ids) * 100)

print(f'数据质量评分: {score:.1f}/100')

print('\n' + '=' * 70)
print('质量验证完成!')
print('=' * 70)
