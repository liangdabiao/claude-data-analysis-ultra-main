# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import os

def read_csv(filename, limit=None):
    data = []
    with open(f'./data_storage/{filename}', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            data.append(row)
    return data

print('='*60)
print('全面检查 Olist 真实数据')
print('='*60)

files = [f for f in os.listdir('./data_storage') if f.endswith('.csv')]
print(f'\n数据文件: {files}')

# orders
print('\n--- orders ---')
orders = read_csv('olist_orders_dataset.csv')
print(f'总订单数: {len(orders)}')
status_count = defaultdict(int)
for o in orders:
    status_count[o['order_status']] += 1
print('订单状态:', dict(status_count))

# order_items
print('\n--- order_items ---')
items = read_csv('olist_order_items_dataset.csv')
print(f'总订单项: {len(items)}')
prices = [float(i['price']) for i in items if i.get('price')]
print(f'价格范围: {min(prices):.2f} - {max(prices):.2f}')
print(f'价格均值: {sum(prices)/len(prices):.2f}')

order_total = defaultdict(float)
for i in items:
    order_total[i['order_id']] += float(i['price']) + float(i.get('freight_value', 0))

all_prices = list(order_total.values())
sorted_prices = sorted(all_prices)
n = len(sorted_prices)
median_price = (sorted_prices[n//2-1] + sorted_prices[n//2]) / 2 if n % 2 == 0 else sorted_prices[n//2]
print(f'订单金额: 均值={sum(all_prices)/len(all_prices):.2f}, 中位数={median_price:.2f}')

# payments
print('\n--- payments ---')
payments = read_csv('olist_order_payments_dataset.csv')
print(f'总支付记录: {len(payments)}')
pay_type = defaultdict(int)
for p in payments:
    pay_type[p['payment_type']] += 1
print('支付方式:', dict(pay_type))

# reviews
print('\n--- reviews ---')
reviews = read_csv('olist_order_reviews_dataset.csv')
print(f'总评价: {len(reviews)}')
scores = [int(r['review_score']) for r in reviews if r.get('review_score')]
score_count = defaultdict(int)
for s in scores:
    score_count[s] += 1
print('评分分布:', dict(score_count))
print(f'评分均值: {sum(scores)/len(scores):.2f}')

# customers
print('\n--- customers ---')
customers = read_csv('olist_customers_dataset.csv')
print(f'总客户: {len(customers)}')

# 检查配送时间
print('\n--- 配送时间 ---')
from datetime import datetime
delivery_days = []
for o in orders:
    if o['order_status'] == 'delivered':
        try:
            purchase = datetime.strptime(o['order_purchase_timestamp'], '%Y-%m-%d %H:%M:%S')
            delivered = datetime.strptime(o['order_delivered_customer_date'], '%Y-%m-%d %H:%M:%S')
            days = (delivered - purchase).days
            if 0 <= days <= 100:
                delivery_days.append(days)
        except:
            pass
if delivery_days:
    print(f'配送时间: 均值={sum(delivery_days)/len(delivery_days):.1f}天, 中位数={sorted(delivery_days)[len(delivery_days)//2]}天')

print('\n' + '='*60)
