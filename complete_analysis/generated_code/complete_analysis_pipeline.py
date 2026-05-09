# -*- coding: utf-8 -*-
"""
完整数据分析管道
一站式运行所有分析
"""
import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """加载所有数据集"""
    print("正在加载数据...")
    orders = pd.read_csv('../data_storage/olist_orders_dataset.csv')
    order_items = pd.read_csv('../data_storage/olist_order_items_dataset.csv')
    payments = pd.read_csv('../data_storage/olist_order_payments_dataset.csv')
    reviews = pd.read_csv('../data_storage/olist_order_reviews_dataset.csv')
    customers = pd.read_csv('../data_storage/olist_customers_dataset.csv')
    products = pd.read_csv('../data_storage/olist_products_dataset.csv')
    sellers = pd.read_csv('../data_storage/olist_sellers_dataset.csv')
    print(f"数据加载完成: 订单={len(orders)}, 订单项={len(order_items)}")
    return orders, order_items, payments, reviews, customers, products, sellers

def preprocess_data(orders, order_items, payments):
    """预处理数据"""
    print("正在预处理数据...")
    
    # 时间转换
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    # 计算订单金额 (按order_id聚合)
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    
    orders = orders.merge(order_amounts.reset_index().rename(columns={0: 'total_amount'}),
                         on='order_id', how='left')
    
    return orders

def calculate_kpis(orders, order_items, payments, reviews):
    """计算KPI指标"""
    print("正在计算KPI...")
    
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    
    kpis = {
        'total_orders': len(orders),
        'total_revenue': order_amounts.sum(),
        'avg_order_value': order_amounts.mean(),
        'median_order_value': order_amounts.median(),
        'unique_customers': orders['customer_id'].nunique(),
    }
    
    if 'review_score' in reviews.columns:
        kpis['avg_review_score'] = reviews['review_score'].mean()
    
    return kpis

def main():
    print("="*60)
    print("完整数据分析管道")
    print("="*60)
    
    # 加载数据
    orders, order_items, payments, reviews, customers, products, sellers = load_data()
    
    # 预处理
    orders = preprocess_data(orders, order_items, payments)
    
    # 计算KPI
    kpis = calculate_kpis(orders, order_items, payments, reviews)
    
    # 输出结果
    print("\n=== KPI 指标 ===")
    for key, value in kpis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()
