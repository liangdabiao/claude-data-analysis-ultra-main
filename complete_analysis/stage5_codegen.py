# -*- coding: utf-8 -*-
"""
Stage 5: 代码生成 (Code Generation)
"""
import os
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Stage 5] {msg}", flush=True)

def main():
    log("="*60)
    log("Stage 5: 代码生成开始")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data_storage")
    output_dir = os.path.join(base_dir, "generated_code")

    # 生成代码
    log("正在生成分析代码...")
    generate_all_code(data_dir, output_dir)

    log("="*60)
    log("Stage 5: 代码生成完成")
    log("="*60)

def generate_all_code(data_dir, output_dir):
    """生成所有代码"""

    # 1. 完整分析管道
    log("正在生成 complete_analysis_pipeline.py...")
    pipeline_code = generate_pipeline_code()
    pipeline_path = os.path.join(output_dir, "complete_analysis_pipeline.py")
    with open(pipeline_path, 'w', encoding='utf-8') as f:
        f.write(pipeline_code)
    log(f"分析管道已保存: {pipeline_path}")

    # 2. 数据预处理
    log("正在生成 data_preprocessing.py...")
    preprocessing_code = generate_preprocessing_code()
    preprocessing_path = os.path.join(output_dir, "data_preprocessing.py")
    with open(preprocessing_path, 'w', encoding='utf-8') as f:
        f.write(preprocessing_code)
    log(f"预处理代码已保存: {preprocessing_path}")

    # 3. 质量检查
    log("正在生成 quality_checks.py...")
    quality_code = generate_quality_check_code()
    quality_path = os.path.join(output_dir, "quality_checks.py")
    with open(quality_path, 'w', encoding='utf-8') as f:
        f.write(quality_code)
    log(f"质量检查代码已保存: {quality_path}")

    # 4. 分析函数
    log("正在生成 analysis_functions.py...")
    analysis_code = generate_analysis_functions()
    analysis_path = os.path.join(output_dir, "analysis_functions.py")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis_code)
    log(f"分析函数已保存: {analysis_path}")

    # 保存执行摘要
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "workflow_log", "stage5_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Stage 5: 代码生成\n\n")
        f.write(f"- 生成文件数: 4\n")
        f.write(f"- 文件: complete_analysis_pipeline.py, data_preprocessing.py, quality_checks.py, analysis_functions.py\n")
        f.write(f"- 状态: ✅ 完成\n")

def generate_pipeline_code():
    """生成完整分析管道代码"""
    return '''# -*- coding: utf-8 -*-
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
    print("\\n=== KPI 指标 ===")
    for key, value in kpis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\\n分析完成!")

if __name__ == "__main__":
    main()
'''

def generate_preprocessing_code():
    """生成数据预处理代码"""
    return '''# -*- coding: utf-8 -*-
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
'''

def generate_quality_check_code():
    """生成质量检查代码"""
    return '''# -*- coding: utf-8 -*-
"""
数据质量检查模块
完整性、一致性、有效性检查
"""
import pandas as pd
import numpy as np

def check_completeness(df):
    """检查数据完整性"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    result = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)
    
    return result

def check_duplicates(df):
    """检查重复"""
    duplicates = df.duplicated().sum()
    return {
        'duplicate_count': duplicates,
        'duplicate_pct': (duplicates / len(df)) * 100
    }

def check_data_types(df):
    """检查数据类型"""
    return df.dtypes.astype(str).to_dict()

def check_outliers_series(series, method='iqr'):
    """检查异常值"""
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (series < lower) | (series > upper)
        return {
            'count': outliers.sum(),
            'pct': outliers.sum() / len(series) * 100,
            'lower_bound': lower,
            'upper_bound': upper
        }
    return {}

def run_full_quality_check(df, name='dataset'):
    """运行完整质量检查"""
    print(f"=== 质量检查: {name} ===")
    print(f"形状: {df.shape}")
    
    print("\\n完整性检查:")
    completeness = check_completeness(df)
    print(completeness[completeness['missing_pct'] > 0].head(10))
    
    print("\\n重复检查:")
    duplicates = check_duplicates(df)
    print(f"重复行数: {duplicates['duplicate_count']} ({duplicates['duplicate_pct']:.2f}%)")
    
    return {
        'completeness': completeness,
        'duplicates': duplicates
    }

def main():
    print("数据质量检查模块")

if __name__ == "__main__":
    main()
'''

def generate_analysis_functions():
    """生成分析函数代码"""
    return '''# -*- coding: utf-8 -*-
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
'''

if __name__ == "__main__":
    main()
