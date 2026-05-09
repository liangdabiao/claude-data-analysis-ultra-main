# -*- coding: utf-8 -*-
"""
Stage 4: 数据可视化 (Data Visualization)
"""
import os
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Stage 4] {msg}", flush=True)

def main():
    log("="*60)
    log("Stage 4: 数据可视化开始")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data_storage")
    output_dir = os.path.join(base_dir, "visualizations")

    # 加载数据
    log("正在加载数据...")
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))

    # 生成可视化
    log("正在生成可视化文件...")
    generate_visualizations(orders, order_items, payments, reviews, customers, output_dir)

    log("="*60)
    log("Stage 4: 数据可视化完成")
    log("="*60)

def generate_visualizations(orders, order_items, payments, reviews, customers, output_dir):
    """生成所有可视化"""

    # 1. 交互式仪表板 (HTML)
    log("正在生成交互式仪表板...")
    dashboard_path = os.path.join(output_dir, "interactive_dashboard.html")
    generate_html_dashboard(orders, order_items, payments, reviews, customers, dashboard_path)
    log(f"交互式仪表板已保存: {dashboard_path}")

    # 2. 可视化代码
    log("正在生成可视化代码...")
    code_path = os.path.join(output_dir, "visualization_code.py")
    generate_visualization_code(code_path)
    log(f"可视化代码已保存: {code_path}")

    # 保存执行摘要
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "workflow_log", "stage4_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Stage 4: 数据可视化\n\n")
        f.write(f"- 交互式仪表板: interactive_dashboard.html\n")
        f.write(f"- 可视化代码: visualization_code.py\n")
        f.write(f"- 状态: ✅ 完成\n")

def generate_html_dashboard(orders, order_items, payments, reviews, customers, output_path):
    """生成HTML交互式仪表板"""

    # 准备数据
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['month'] = orders['order_purchase_timestamp'].dt.to_period('M')
    monthly_orders = orders.groupby('month').size()
    monthly_order_str = {str(k): int(v) for k, v in monthly_orders.items()}

    order_amounts = order_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).sum(axis=1)
    avg_amount = order_amounts.mean()

    payment_types = payments['payment_type'].value_counts().to_dict()

    if 'review_score' in reviews.columns:
        score_dist = reviews['review_score'].value_counts().sort_index().to_dict()
    else:
        score_dist = {}

    state_dist = customers['customer_state'].value_counts().head(10).to_dict()

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Olist 电商数据分析仪表板</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }}
        body {{ background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; margin-bottom: 30px; text-align: center; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .kpi-card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .kpi-value {{ font-size: 28px; font-weight: bold; color: #3498db; }}
        .kpi-label {{ color: #7f8c8d; margin-top: 8px; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .chart-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chart-title {{ font-size: 18px; color: #2c3e50; margin-bottom: 15px; }}
        .chart-container {{ height: 300px; display: flex; align-items: flex-end; gap: 5px; }}
        .bar {{ flex: 1; background: linear-gradient(to top, #3498db, #2980b9); border-radius: 3px 3px 0 0; position: relative; min-height: 20px; }}
        .bar-label {{ position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #7f8c8d; white-space: nowrap; }}
        .pie-legend {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 20px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 15px; height: 15px; border-radius: 3px; }}
        .single-chart {{ grid-column: 1 / -1; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Olist 电商数据分析仪表板</h1>

        <!-- KPI指标 -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">{len(orders):,}</div>
                <div class="kpi-label">总订单数</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">R$ {order_amounts.sum():,.0f}</div>
                <div class="kpi-label">总营收</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">R$ {avg_amount:.0f}</div>
                <div class="kpi-label">平均订单金额</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{customers['customer_id'].nunique():,}</div>
                <div class="kpi-label">客户数</div>
            </div>
        </div>

        <!-- 图表 -->
        <div class="charts-grid">
            <!-- 月度订单趋势 -->
            <div class="chart-card">
                <div class="chart-title">📈 月度订单趋势</div>
                <div class="chart-container">
                    {generate_bar_chart(monthly_order_str, '#3498db')}
                </div>
            </div>

            <!-- 支付方式分布 -->
            <div class="chart-card">
                <div class="chart-title">💳 支付方式分布</div>
                <div class="chart-container">
                    {generate_bar_chart(payment_types, '#2ecc71')}
                </div>
            </div>

            <!-- 评分分布 -->
            <div class="chart-card">
                <div class="chart-title">⭐ 评分分布</div>
                <div class="chart-container">
                    {generate_bar_chart(score_dist, '#f39c12')}
                </div>
            </div>

            <!-- 州分布 -->
            <div class="chart-card">
                <div class="chart-title">📍 客户地理位置 (Top 10)</div>
                <div class="chart-container">
                    {generate_bar_chart(state_dist, '#9b59b6')}
                </div>
            </div>
        </div>

        <!-- 订单状态分布 -->
        <div class="charts-grid">
            <div class="chart-card single-chart">
                <div class="chart-title">📋 订单状态分布</div>
                <div class="chart-container">
                    {generate_bar_chart(orders['order_status'].value_counts().to_dict(), '#e74c3c')}
                </div>
            </div>
        </div>

    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_bar_chart(data, color):
    """生成简单的HTML条形图"""
    if not data:
        return "<div style='padding:20px;color:#999;'>无数据</div>"

    max_val = max(data.values()) if data else 1
    bars_html = []

    for key, value in list(data.items())[:12]:
        height_pct = (value / max_val * 100) if max_val > 0 else 0
        height = max(height_pct, 5)
        key_str = str(key)[:8]
        bars_html.append(f'''
            <div class="bar" style="height: {height}%; background: {color};">
                <div class="bar-label">{key_str}</div>
            </div>
        ''')

    return ''.join(bars_html)

def generate_visualization_code(output_path):
    """生成可视化参考代码"""

    code = '''# -*- coding: utf-8 -*-
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
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)

if __name__ == "__main__":
    main()
