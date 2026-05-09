# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import statistics
from datetime import datetime
import re

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
print('Olist 巴西电商数据分析报告')
print('=' * 70)

# 加载主要数据集
print('\n【1. 数据集概览】')
customers = read_csv('olist_customers_dataset.csv')
orders = read_csv('olist_orders_dataset.csv')
order_items = read_csv('olist_order_items_dataset.csv')
products = read_csv('olist_products_dataset.csv')
payments = read_csv('olist_order_payments_dataset.csv')
sellers = read_csv('olist_sellers_dataset.csv')
reviews = read_csv('olist_order_reviews_dataset.csv')

print(f'客户数量: {len(customers)}')
print(f'订单数量: {len(orders)}')
print(f'订单商品数量: {len(order_items)}')
print(f'产品数量: {len(products)}')
print(f'支付记录: {len(payments)}')
print(f'卖家数量: {len(sellers)}')
print(f'评价数量: {len(reviews)}')

# 2. 订单分析
print('\n【2. 订单状态分析】')
order_status = defaultdict(int)
for order in orders:
    order_status[order['order_status']] += 1
for status, count in sorted(order_status.items(), key=lambda x: -x[1]):
    print(f'{status}: {count} ({count/len(orders)*100:.1f}%)')

# 3. 订单时间分析
print('\n【3. 订单时间趋势】')
order_dates = []
for order in orders:
    date = parse_date(order['order_purchase_timestamp'])
    if date:
        order_dates.append(date)

if order_dates:
    order_dates.sort()
    print(f'最早订单: {order_dates[0].strftime("%Y-%m-%d")}')
    print(f'最新订单: {order_dates[-1].strftime("%Y-%m-%d")}')
    
    # 按月统计
    monthly = defaultdict(int)
    for d in order_dates:
        key = d.strftime('%Y-%m')
        monthly[key] += 1
    
    print('\n月度订单量 (前10个月):')
    for month, count in sorted(monthly.items())[:10]:
        print(f'{month}: {count} 订单')

# 4. 客户分布
print('\n【4. 客户地理分布】')
customer_state = defaultdict(int)
customer_city = defaultdict(int)
for c in customers:
    customer_state[c['customer_state']] += 1
    customer_city[c['customer_city'].title()] += 1

print('\n各州客户数量 (前10):')
for state, count in sorted(customer_state.items(), key=lambda x: -x[1])[:10]:
    print(f'{state}: {count}')

print('\n城市客户数量 (前10):')
for city, count in sorted(customer_city.items(), key=lambda x: -x[1])[:10]:
    print(f'{city}: {count}')

# 5. 销售分析
print('\n【5. 销售分析】')
total_revenue = 0
order_revenues = defaultdict(float)
for item in order_items:
    price = float(item['price'])
    freight = float(item['freight_value'])
    total_revenue += price + freight
    order_revenues[item['order_id']] += price + freight

print(f'总收入: R$ {total_revenue:,.2f}')
print(f'平均订单金额: R$ {total_revenue/len(orders):,.2f}')

# 客单价分布
order_values = list(order_revenues.values())
print(f'\n订单金额统计:')
print(f'最低: R$ {min(order_values):.2f}')
print(f'最高: R$ {max(order_values):.2f}')
print(f'平均: R$ {statistics.mean(order_values):.2f}')
print(f'中位数: R$ {statistics.median(order_values):.2f}')

# 6. 支付分析
print('\n【6. 支付方式分析】')
payment_types = defaultdict(int)
payment_installments = defaultdict(int)
for p in payments:
    payment_types[p['payment_type']] += 1
    payment_installments[int(p['payment_installments'])] += 1

print('支付方式:')
for ptype, count in sorted(payment_types.items(), key=lambda x: -x[1]):
    print(f'{ptype}: {count} ({count/len(payments)*100:.1f}%)')

print('\n分期期数分布:')
for inst, count in sorted(payment_installments.items())[:6]:
    print(f'{inst}期: {count}')

# 7. 产品分析
print('\n【7. 产品类别分析】')
product_categories = defaultdict(int)
for p in products:
    if p['product_category_name']:
        product_categories[p['product_category_name']] += 1

print('产品类别数量 (前15):')
for cat, count in sorted(product_categories.items(), key=lambda x: -x[1])[:15]:
    print(f'{cat}: {count}')

# 8. 配送分析
print('\n【8. 配送时间分析】')
delivery_times = []
for order in orders:
    if order['order_status'] == 'delivered':
        purchase = parse_date(order['order_purchase_timestamp'])
        delivered = parse_date(order['order_delivered_customer_date'])
        if purchase and delivered:
            days = (delivered - purchase).days
            if days >= 0:
                delivery_times.append(days)

if delivery_times:
    print(f'平均配送天数: {statistics.mean(delivery_times):.1f} 天')
    print(f'中位数配送天数: {statistics.median(delivery_times):.1f} 天')
    print(f'最快: {min(delivery_times)} 天')
    print(f'最慢: {max(delivery_times)} 天')

# 9. 评价分析
print('\n【9. 评价分析】')
review_scores = defaultdict(int)
for r in reviews:
    if r['review_score']:
        review_scores[int(r['review_score'])] += 1

print('评分分布:')
for score in range(1, 6):
    count = review_scores[score]
    pct = count/len(reviews)*100
    bar = '█' * int(pct/2)
    print(f'{score}星: {count:5d} ({pct:5.1f}%) {bar}')

avg_score = sum(k*v for k,v in review_scores.items()) / sum(review_scores.values())
print(f'\n平均评分: {avg_score:.2f}')

print('\n' + '=' * 70)
print('分析完成!')
print('=' * 70)
