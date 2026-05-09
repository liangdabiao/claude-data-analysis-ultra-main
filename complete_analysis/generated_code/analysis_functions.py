# -*- coding: utf-8 -*-
"""
分析函数库
统计分析、RFM分析、客户分群等
"""
import pandas as pd
import numpy as np

def rfm_analysis(orders, order_items, latest_date=None):
    """RFM客户价值分析"""
    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    
    orders = orders.merge(
        order_amounts.reset_index().rename(columns={0: 'monetary'}),
        on='order_id', how='left'
    )
    
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    if latest_date is None:
        latest_date = orders['order_purchase_timestamp'].max()
    
    rfm = orders.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,
        'order_id': 'count',
        'monetary': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'recency',
        'order_id': 'frequency'
    })
    
    # 评分
    rfm['R_score'] = pd.qcut(rfm['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
    rfm['M_score'] = pd.qcut(rfm['monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
    
    rfm['RFM_score'] = (
        rfm['R_score'].astype(str) +
        rfm['F_score'].astype(str) +
        rfm['M_score'].astype(str)
    )
    
    return rfm

def segment_customers(rfm):
    """客户分群"""
    def get_segment(row):
        r = int(row['R_score'])
        f = int(row['F_score'])
        m = int(row['M_score'])
        
        if r >= 3 and f >= 3 and m >= 3:
            return 'VIP客户'
        elif r >= 3 and m >= 3:
            return '高价值客户'
        elif r >= 3 and f >= 3:
            return '潜力客户'
        elif r <= 2 and f <= 2:
            return '流失风险客户'
        elif r <= 2:
            return '沉睡客户'
        else:
            return '普通客户'
    
    rfm['segment'] = rfm.apply(get_segment, axis=1)
    return rfm

def cohort_analysis(orders):
    """群组分析"""
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['cohort_month'] = orders.groupby('customer_id')['order_purchase_timestamp'].transform('min').dt.to_period('M')
    orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
    
    cohort = orders.groupby(['cohort_month', 'order_month']).agg({
        'customer_id': 'nunique'
    }).reset_index()
    
    cohort['cohort_index'] = (cohort['order_month'] - cohort['cohort_month']).apply(lambda x: x.n)
    
    return cohort

def main():
    print("分析函数库")
    print("可用函数:")
    print("- rfm_analysis(): RFM分析")
    print("- segment_customers(): 客户分群")
    print("- cohort_analysis(): 群组分析")

if __name__ == "__main__":
    main()
