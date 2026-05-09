# -*- coding: utf-8 -*-
"""
数据预处理模块
数据清洗、格式转换、特征工程
"""
import pandas as pd
import numpy as np

def clean_orders(orders):
    """清洗订单数据"""
    orders_clean = orders.copy()
    
    # 时间格式转换
    time_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    
    for col in time_columns:
        if col in orders_clean.columns:
            orders_clean[col] = pd.to_datetime(orders_clean[col])
    
    # 计算配送时间
    if ('order_delivered_customer_date' in orders_clean.columns and
        'order_purchase_timestamp' in orders_clean.columns):
        orders_clean['delivery_days'] = (
            orders_clean['order_delivered_customer_date'] -
            orders_clean['order_purchase_timestamp']
        ).dt.total_seconds() / (60 * 60 * 24)
    
    return orders_clean

def calculate_order_amounts(order_items):
    """计算订单金额 (正确聚合方法!)"""
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    })
    order_amounts['total_amount'] = order_amounts['price'] + order_amounts['freight_value']
    return order_amounts

def create_customer_features(orders, order_items, customers):
    """创建客户特征"""
    order_amounts = calculate_order_amounts(order_items)
    
    customer_features = orders.groupby('customer_id').agg({
        'order_id': 'count',
        'order_purchase_timestamp': ['min', 'max']
    }).reset_index()
    
    customer_features.columns = ['customer_id', 'order_count', 'first_order', 'last_order']
    
    customer_features = customer_features.merge(
        orders.merge(order_amounts[['total_amount']], on='order_id')
        .groupby('customer_id')['total_amount'].sum()
        .reset_index().rename(columns={'total_amount': 'total_spent'}),
        on='customer_id'
    )
    
    customer_features['avg_order_value'] = (
        customer_features['total_spent'] / customer_features['order_count']
    )
    
    return customer_features

def main():
    print("数据预处理模块")
    print("可用函数:")
    print("- clean_orders(orders): 清洗订单数据")
    print("- calculate_order_amounts(order_items): 计算订单金额")
    print("- create_customer_features(...): 创建客户特征")

if __name__ == "__main__":
    main()
