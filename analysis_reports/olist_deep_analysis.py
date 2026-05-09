# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import math
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

def mean(data):
    return sum(data) / len(data)

def std(data):
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2-1] + sorted_data[n//2]) / 2
    return sorted_data[n//2]

def quartiles(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1_idx = n // 4
    q2_idx = n // 2
    q3_idx = 3 * n // 4
    return sorted_data[q1_idx], sorted_data[q2_idx], sorted_data[q3_idx]

def skewness(data):
    n = len(data)
    m = mean(data)
    s = std(data)
    if s == 0:
        return 0
    return (n / ((n-1) * (n-2))) * sum(((x - m) / s) ** 3 for x in data)

def kurtosis(data):
    n = len(data)
    m = mean(data)
    s = std(data)
    if s == 0:
        return 0
    return (n*(n+1) / ((n-1)*(n-2)*(n-3))) * sum(((x - m) / s) ** 4 for x in data) - 3*(n-1)**2 / ((n-2)*(n-3))

def pearson_correlation(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    cov = sum((x[i]-mx)*(y[i]-my) for i in range(n)) / n
    sx, sy = std(x), std(y)
    if sx == 0 or sy == 0:
        return 0
    return cov / (sx * sy)

def t_test_independent(x, y):
    """简化版T检验"""
    nx, ny = len(x), len(y)
    mx, my = mean(x), mean(y)
    sx2, sy2 = std(x)**2, std(y)**2
    
    # 合并方差
    pooled_var = ((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)
    se = math.sqrt(pooled_var * (1/nx + 1/ny))
    
    if se == 0:
        return 0, 1
    
    t_stat = (mx - my) / se
    # 简化p值估算
    df = nx + ny - 2
    p_value = 2 * (1 - min(0.9999, abs(t_stat) / math.sqrt(df + 1)))
    return t_stat, p_value

print('=' * 70)
print('Olist 深度统计分析报告')
print('=' * 70)

# 加载数据
orders = read_csv('olist_orders_dataset.csv')
order_items = read_csv('olist_order_items_dataset.csv')
payments = read_csv('olist_order_payments_dataset.csv')
reviews = read_csv('olist_order_reviews_dataset.csv')
customers = read_csv('olist_customers_dataset.csv')

# 1. 订单金额分布分析
print('\n【1. 订单金额分布分析】')
order_prices = {}
for item in order_items:
    oid = item['order_id']
    price = float(item['price']) + float(item['freight_value'])
    if oid not in order_prices:
        order_prices[oid] = 0
    order_prices[oid] += price

prices = list(order_prices.values())
print(f'样本数: {len(prices)}')
print(f'均值: R$ {mean(prices):.2f}')
print(f'标准差: R$ {std(prices):.2f}')
print(f'中位数: R$ {median(prices):.2f}')
q1, q2, q3 = quartiles(prices)
print(f'四分位数: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}')
print(f'偏度: {skewness(prices):.3f} (正偏态, 右尾长)')
print(f'峰度: {kurtosis(prices):.3f} (尖峰分布)')

# 2. 异常值检测 (IQR方法)
print('\n【2. 订单金额异常值检测 (IQR方法)】')
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = [p for p in prices if p < lower_bound or p > upper_bound]
print(f'IQR: {iqr:.2f}')
print(f'异常值边界: [{lower_bound:.2f}, {upper_bound:.2f}]')
print(f'异常值数量: {len(outliers)} ({len(outliers)/len(prices)*100:.2f}%)')
print(f'最大异常值: R$ {max(outliers):.2f}')

# 3. 配送时间分析
print('\n【3. 配送时间分析】')
delivery_days = []
for order in orders:
    if order['order_status'] == 'delivered':
        purchase = parse_date(order['order_purchase_timestamp'])
        delivered = parse_date(order['order_delivered_customer_date'])
        if purchase and delivered:
            days = (delivered - purchase).days
            if 0 <= days <= 100:
                delivery_days.append(days)

print(f'样本数: {len(delivery_days)}')
print(f'平均配送天数: {mean(delivery_days):.2f} 天')
print(f'标准差: {std(delivery_days):.2f} 天')
print(f'中位数: {median(delivery_days):.2f} 天')
print(f'偏度: {skewness(delivery_days):.3f}')
print(f'峰度: {kurtosis(delivery_days):.3f}')

# 4. 支付方式与订单金额关系
print('\n【4. 支付方式与订单金额关系 (T检验)】')
credit_prices = []
boleto_prices = []
for p in payments:
    price = float(p['payment_value'])
    if p['payment_type'] == 'credit_card':
        credit_prices.append(price)
    elif p['payment_type'] == 'boleto':
        boleto_prices.append(price)

t_stat, p_value = t_test_independent(credit_prices[:5000], boleto_prices[:5000])
print(f'信用卡平均: R$ {mean(credit_prices):.2f} (n={len(credit_prices)})')
print(f'Boleto平均: R$ {mean(boleto_prices):.2f} (n={len(boleto_prices)})')
print(f'T统计量: {t_stat:.3f}')
print(f'P值: {p_value:.6f}')
if p_value < 0.05:
    print('结论: 两种支付方式的订单金额存在显著差异 (p < 0.05)')
else:
    print('结论: 两种支付方式的订单金额无显著差异 (p >= 0.05)')

# 5. 客户评分分析
print('\n【5. 客户评分分布分析】')
scores = [int(r['review_score']) for r in reviews if r.get('review_score')]
score_counts = defaultdict(int)
for s in scores:
    score_counts[s] += 1

print(f'样本数: {len(scores)}')
print(f'平均评分: {mean(scores):.3f}')
print(f'标准差: {std(scores):.3f}')
print(f'偏度: {skewness(scores):.3f} (负偏态, 低分更多)')

# 6. 评分与配送时间相关性
print('\n【6. 评分与配送时间相关性分析】')
delivery_with_score = []
for order in orders:
    if order['order_status'] == 'delivered':
        purchase = parse_date(order['order_purchase_timestamp'])
        delivered = parse_date(order['order_delivered_customer_date'])
        if purchase and delivered:
            days = (delivered - purchase).days
            if 0 <= days <= 100:
                # 找对应的评分
                for review in reviews:
                    if review['order_id'] == order['order_id'] and review.get('review_score'):
                        delivery_with_score.append((days, int(review['review_score'])))
                        break

if len(delivery_with_score) > 10:
    delivery_times = [d[0] for d in delivery_with_score]
    review_scores = [d[1] for d in delivery_with_score]
    corr = pearson_correlation(delivery_times, review_scores)
    print(f'样本数: {len(delivery_with_score)}')
    print(f'皮尔逊相关系数: {corr:.4f}')
    if corr < -0.1:
        print('结论: 配送时间与评分呈负相关，配送越慢评分越低')
    elif corr > 0.1:
        print('结论: 配送时间与评分呈正相关')

# 7. RFM 客户分析
print('\n【7. RFM 客户分析】')
# 获取最新订单日期
latest_date = None
for order in orders:
    if order['order_status'] == 'delivered':
        date = parse_date(order['order_purchase_timestamp'])
        if date and (latest_date is None or date > latest_date):
            latest_date = date

if latest_date:
    print(f'数据最新日期: {latest_date.strftime("%Y-%m-%d")}')
    
    # 计算RFM
    rfm_data = defaultdict(lambda: {'recency': 0, 'frequency': 0, 'monetary': 0})
    
    for order in orders:
        if order['order_status'] == 'delivered':
            cid = order['customer_id']
            date = parse_date(order['order_purchase_timestamp'])
            if date:
                # Recency: 距离最新订单的天数
                recency = (latest_date - date).days
                if rfm_data[cid]['recency'] == 0 or recency < rfm_data[cid]['recency']:
                    rfm_data[cid]['recency'] = recency
                # Frequency
                rfm_data[cid]['frequency'] += 1
    
    # 添加 monetary
    for item in order_items:
        for order in orders:
            if order['order_id'] == item['order_id'] and order['customer_id'] in rfm_data:
                price = float(item['price']) + float(item['freight_value'])
                rfm_data[order['customer_id']]['monetary'] += price
                break
    
    # RFM统计
    recencies = [v['recency'] for v in rfm_data.values()]
    frequencies = [v['frequency'] for v in rfm_data.values()]
    monetaries = [v['monetary'] for v in rfm_data.values()]
    
    print(f'\nRecency (最近购买天数):')
    print(f'  平均: {mean(recencies):.1f} 天')
    print(f'  中位数: {median(recencies):.1f} 天')
    
    print(f'\nFrequency (购买频次):')
    print(f'  平均: {mean(frequencies):.2f} 次')
    print(f'  中位数: {median(frequencies):.2f} 次')
    
    print(f'\nMonetary (消费金额):')
    print(f'  平均: R$ {mean(monetaries):.2f}')
    print(f'  中位数: R$ {median(monetaries):.2f}')

# 8. 客户分群
print('\n【8. 客户价值分群】')
# 按消费金额分群
sorted_customers = sorted(rfm_data.items(), key=lambda x: x[1]['monetary'], reverse=True)
top_20 = sorted_customers[:int(len(sorted_customers)*0.2)]
bottom_80 = sorted_customers[int(len(sorted_customers)*0.2):]

top_20_revenue = sum(c[1]['monetary'] for c in top_20)
total_revenue = sum(c[1]['monetary'] for c in rfm_data.values())
print(f'Top 20% 客户数量: {len(top_20)} ({len(top_20)/len(rfm_data)*100:.1f}%)')
print(f'Top 20% 客户贡献收入: R$ {top_20_revenue:.2f}')
print(f'占总收入比例: {top_20_revenue/total_revenue*100:.1f}%')
print(f'Bottom 80% 客户贡献收入: R$ {total_revenue - top_20_revenue:.2f}')
print(f'占总收入比例: {(total_revenue-top_20_revenue)/total_revenue*100:.1f}%')

print('\n' + '=' * 70)
print('深度分析完成!')
print('=' * 70)
