# -*- coding: utf-8 -*-
"""
Olist 电商数据可视化参考代码
使用 Matplotlib 和 Seaborn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import seaborn as sns

# 设置中文支持
rcParams['font.family'] = 'SimHei, DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# 加载数据
orders = pd.read_csv('../data_storage/olist_orders_dataset.csv')
order_items = pd.read_csv('../data_storage/olist_order_items_dataset.csv')
payments = pd.read_csv('../data_storage/olist_order_payments_dataset.csv')
reviews = pd.read_csv('../data_storage/olist_order_reviews_dataset.csv')
customers = pd.read_csv('../data_storage/olist_customers_dataset.csv')

# 时间处理
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# 订单金额计算
order_amounts = order_items.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
}).sum(axis=1)

# 1. 月度订单趋势图
plt.figure(figsize=(14, 6))
orders['month'] = orders['order_purchase_timestamp'].dt.to_period('M')
monthly_data = orders.groupby('month').size()
monthly_dates = [pd.to_datetime(str(m)) for m in monthly_data.index]
plt.plot(monthly_dates, monthly_data.values, marker='o', linewidth=2, color='#3498db')
plt.title('月度订单趋势', fontsize=16)
plt.xlabel('月份', fontsize=12)
plt.ylabel('订单数', fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_orders.png', dpi=150)

# 2. 订单金额分布图
plt.figure(figsize=(14, 6))
plt.hist(order_amounts, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
plt.title('订单金额分布', fontsize=16)
plt.xlabel('订单金额 (R$)', fontsize=12)
plt.ylabel('订单数', fontsize=12)
plt.xscale('log')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('order_amount_distribution.png', dpi=150)

# 3. 支付方式分布图
plt.figure(figsize=(12, 6))
payment_counts = payments['payment_type'].value_counts()
plt.bar(range(len(payment_counts)), payment_counts.values, color='#9b59b6', alpha=0.7)
plt.xticks(range(len(payment_counts)), payment_counts.index, rotation=45)
plt.title('支付方式分布', fontsize=16)
plt.xlabel('支付方式', fontsize=12)
plt.ylabel('订单数', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('payment_types.png', dpi=150)

# 4. 评分分布图
if 'review_score' in reviews.columns:
    plt.figure(figsize=(12, 6))
    score_counts = reviews['review_score'].value_counts().sort_index()
    plt.bar(score_counts.index, score_counts.values, color='#f39c12', alpha=0.7)
    plt.title('客户评分分布', fontsize=16)
    plt.xlabel('评分', fontsize=12)
    plt.ylabel('评论数', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('review_scores.png', dpi=150)

# 5. 州分布Top10
plt.figure(figsize=(14, 6))
state_counts = customers['customer_state'].value_counts().head(10)
plt.bar(range(len(state_counts)), state_counts.values, color='#e74c3c', alpha=0.7)
plt.xticks(range(len(state_counts)), state_counts.index)
plt.title('客户地理位置 Top 10', fontsize=16)
plt.xlabel('州', fontsize=12)
plt.ylabel('客户数', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('customer_states.png', dpi=150)

print("所有图表已生成!")
