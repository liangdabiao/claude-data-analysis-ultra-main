# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import math
from datetime import datetime

def read_csv_limited(filename, limit=10000):
    data = []
    with open(f'./data_storage/{filename}', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            data.append(row)
    return data

def mean(data):
    return sum(data) / len(data)

def std(data):
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

def median(data):
    s = sorted(data)
    n = len(s)
    return (s[n//2-1] + s[n//2]) / 2 if n % 2 == 0 else s[n//2]

print('=' * 60)
print('Olist 深度统计分析 (优化版)')
print('=' * 60)

# 加载数据
print('加载数据...')
orders = read_csv_limited('olist_orders_dataset.csv', 15000)
order_items = read_csv_limited('olist_order_items_dataset.csv', 15000)
payments = read_csv_limited('olist_order_payments_dataset.csv', 15000)
reviews = read_csv_limited('olist_order_reviews_dataset.csv', 15000)

# 1. 订单金额分析
print('\n【1. 订单金额分析】')
prices = [float(item['price']) + float(item['freight_value']) for item in order_items]
print(f'均值: R$ {mean(prices):.2f}')
print(f'标准差: R$ {std(prices):.2f}')
print(f'中位数: R$ {median(prices):.2f}')

# 2. 异常值检测
print('\n【2. 异常值检测 (IQR)】')
s = sorted(prices)
q1, q3 = s[len(s)//4], s[3*len(s)//4]
iqr = q3 - q1
outliers = [p for p in prices if p < q1-1.5*iqr or p > q3+1.5*iqr]
print(f'Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}')
print(f'异常值: {len(outliers)} 个 ({len(outliers)/len(prices)*100:.1f}%)')

# 3. 评分分布
print('\n【3. 评分分布】')
scores = [int(r['review_score']) for r in reviews if r.get('review_score')]
print(f'均值: {mean(scores):.2f}')
print(f'标准差: {std(scores):.2f}')
print('评分分布:', {i: scores.count(i) for i in range(1,6)})

# 4. 支付方式
print('\n【4. 支付方式】')
payment_types = defaultdict(int)
for p in payments:
    payment_types[p['payment_type']] += 1
for pt, cnt in sorted(payment_types.items(), key=lambda x: -x[1]):
    print(f'{pt}: {cnt} ({cnt/len(payments)*100:.1f}%)')

# 5. 配送时间
print('\n【5. 配送时间】')
delivery_days = []
for order in orders:
    if order['order_status'] == 'delivered':
        try:
            purchase = datetime.strptime(order['order_purchase_timestamp'], '%Y-%m-%d %H:%M:%S')
            delivered = datetime.strptime(order['order_delivered_customer_date'], '%Y-%m-%d %H:%M:%S')
            days = (delivered - purchase).days
            if 0 <= days <= 60:
                delivery_days.append(days)
        except:
            pass
print(f'样本: {len(delivery_days)}')
print(f'平均: {mean(delivery_days):.1f} 天')
print(f'中位数: {median(delivery_days):.1f} 天')

print('\n' + '=' * 60)
print('分析完成!')
print('=' * 60)
