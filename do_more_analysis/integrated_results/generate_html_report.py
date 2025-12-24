"""
生成HTML综合报告
创建图文并茂的交互式HTML报告
"""

import pandas as pd
import json
import os
import base64
from datetime import datetime
from io import BytesIO

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def load_json_results(filepath):
    """加载JSON结果文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def main():
    output_dir = 'do_more_analysis/integrated_results'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Generating HTML Comprehensive Report")
    print("="*80)

    # 加载所有分析结果
    print("\nLoading analysis results...")

    results = {}
    skills = [
        ('eda', 'data-exploration-visualization'),
        ('rfm', 'rfm-customer-segmentation'),
        ('ltv', 'ltv-predictor'),
        ('retention', 'retention-analysis'),
        ('funnel', 'funnel-analysis'),
        ('growth', 'growth-model-analyzer'),
        ('content', 'content-analysis')
    ]

    for key, folder in skills:
        json_path = f'do_more_analysis/skill_execution/{folder}/'
        # 查找JSON文件
        for file in os.listdir(json_path):
            if file.endswith('.json') and 'summary' in file.lower():
                data = load_json_results(os.path.join(json_path, file))
                if data:
                    results[key] = data
                    break

    print(f"Loaded results from {len(results)} skills")

    # 收集所有图片
    print("\nEncoding images...")
    images = {}

    image_mappings = {
        'eda': [
            ('data-exploration-visualization/order_status_distribution.png', 'Order Status Distribution'),
            ('data-exploration-visualization/order_time_trend.png', 'Order Time Trend'),
            ('data-exploration-visualization/delivery_performance.png', 'Delivery Performance'),
            ('data-exploration-visualization/weekly_order_distribution.png', 'Weekly Order Distribution'),
            ('data-exploration-visualization/hourly_order_distribution.png', 'Hourly Order Distribution')
        ],
        'rfm': [
            ('rfm-customer-segmentation/segment_distribution.png', 'Customer Segment Distribution'),
            ('rfm-customer-segmentation/rfm_dashboard.png', 'RFM Analysis Dashboard'),
            ('rfm-customer-segmentation/rfm_boxplots.png', 'RFM Boxplots')
        ],
        'ltv': [
            ('ltv-predictor/model_comparison.png', 'Model Performance Comparison'),
            ('ltv-predictor/feature_importance.png', 'Feature Importance'),
            ('ltv-predictor/ltv_distribution.png', 'LTV Distribution')
        ],
        'retention': [
            ('retention-analysis/retention_heatmap.png', 'Customer Retention Heatmap'),
            ('retention-analysis/purchase_distribution.png', 'Purchase Frequency Distribution'),
            ('retention-analysis/retention_trend.png', 'Retention Rate Trend')
        ],
        'funnel': [
            ('funnel-analysis/conversion_funnel.png', 'Conversion Funnel'),
            ('funnel-analysis/funnel_dashboard.png', 'Funnel Analysis Dashboard'),
            ('funnel-analysis/stage_conversion_rates.png', 'Stage Conversion Rates')
        ],
        'growth': [
            ('growth-model-analyzer/growth_dashboard.png', 'Growth Analysis Dashboard'),
            ('growth-model-analyzer/growth_trends.png', 'Growth Trends'),
            ('growth-model-analyzer/yearly_comparison.png', 'Yearly Comparison')
        ],
        'content': [
            ('content-analysis/review_distributions.png', 'Review Score Distribution'),
            ('content-analysis/keyword_analysis.png', 'Keyword Analysis'),
            ('content-analysis/sentiment_dashboard.png', 'Sentiment Dashboard')
        ]
    }

    for skill, img_list in image_mappings.items():
        images[skill] = []
        for img_path, img_title in img_list:
            full_path = f'do_more_analysis/skill_execution/{img_path}'
            if os.path.exists(full_path):
                encoded = encode_image_to_base64(full_path)
                if encoded:
                    images[skill].append({
                        'title': img_title,
                        'data': encoded
                    })

    # 生成HTML报告
    print("\nGenerating HTML report...")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Olist E-commerce Comprehensive Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }}

        .header .subtitle {{
            font-size: 1.3em;
            opacity: 0.95;
            margin-top: 15px;
        }}

        .header .date {{
            font-size: 1em;
            opacity: 0.85;
            margin-top: 20px;
        }}

        .nav {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 2px solid #e9ecef;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}

        .nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }}

        .nav li {{
            margin: 0;
        }}

        .nav a {{
            display: block;
            padding: 10px 20px;
            color: #495057;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s ease;
            font-weight: 500;
        }}

        .nav a:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 60px;
            scroll-margin-top: 100px;
        }}

        .section h2 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
        }}

        .section h2::before {{
            content: '▶';
            margin-right: 15px;
            font-size: 0.7em;
        }}

        .section h3 {{
            color: #495057;
            font-size: 1.8em;
            margin: 30px 0 20px 0;
            padding-left: 20px;
            border-left: 4px solid #667eea;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}

        .kpi-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }}

        .kpi-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .kpi-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .kpi-card .unit {{
            font-size: 0.6em;
            opacity: 0.8;
        }}

        .kpi-card.green {{
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        }}

        .kpi-card.blue {{
            background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%);
        }}

        .kpi-card.orange {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}

        .kpi-card.purple {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}

        .chart-container {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}

        .chart-container h4 {{
            color: #495057;
            font-size: 1.3em;
            margin-bottom: 20px;
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-item {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .stat-item:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }}

        .stat-item .label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}

        .stat-item .value {{
            color: #667eea;
            font-size: 2em;
            font-weight: bold;
        }}

        .insight-box {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            border-left: 5px solid #f39c12;
            padding: 25px;
            border-radius: 10px;
            margin: 25px 0;
        }}

        .insight-box h4 {{
            color: #d35400;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}

        .insight-box ul {{
            margin-left: 20px;
        }}

        .insight-box li {{
            margin-bottom: 10px;
            color: #2c3e50;
        }}

        .recommendation {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 10px;
            padding: 25px;
            margin: 25px 0;
        }}

        .recommendation h4 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}

        .recommendation ul {{
            margin-left: 20px;
        }}

        .recommendation li {{
            margin-bottom: 10px;
            color: #34495e;
        }}

        .table-wrapper {{
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}

        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}

        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .footer p {{
            opacity: 0.8;
            margin-bottom: 10px;
        }}

        .scroll-top {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            text-decoration: none;
        }}

        .scroll-top:hover {{
            background: #764ba2;
            transform: translateY(-3px);
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2em;
            }}

            .kpi-grid {{
                grid-template-columns: 1fr;
            }}

            .content {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Olist E-commerce Data Analysis</h1>
            <p class="subtitle">Comprehensive Multi-Dimensional Business Intelligence Report</p>
            <p class="date">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>

        <nav class="nav">
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#eda">Data Exploration</a></li>
                <li><a href="#rfm">Customer Segments</a></li>
                <li><a href="#ltv">LTV Prediction</a></li>
                <li><a href="#retention">Retention</a></li>
                <li><a href="#funnel">Funnel</a></li>
                <li><a href="#growth">Growth</a></li>
                <li><a href="#content">Reviews</a></li>
                <li><a href="#insights">Insights</a></li>
            </ul>
        </nav>

        <div class="content">
            <!-- Overview Section -->
            <section id="overview" class="section">
                <h2>Executive Overview</h2>

                <div class="kpi-grid">
"""

    # 添加KPI卡片
    if 'eda' in results:
        total_orders = results['eda'].get('total_orders', 0)
        html += f"""
                    <div class="kpi-card blue">
                        <div class="label">Total Orders</div>
                        <div class="value">{total_orders:,}<span class="unit"> orders</span></div>
                    </div>
        """

    if 'growth' in results:
        total_revenue = results['growth'].get('total_revenue', 0)
        cagr = results['growth'].get('revenue_cagr_monthly', 0)
        html += f"""
                    <div class="kpi-card green">
                        <div class="label">Total Revenue</div>
                        <div class="value">${total_revenue:,.0f}<span class="unit"> BRL</span></div>
                    </div>

                    <div class="kpi-card purple">
                        <div class="label">Monthly Growth Rate</div>
                        <div class="value">{cagr:.1f}%<span class="unit"> CAGR</span></div>
                    </div>
        """

    if 'rfm' in results:
        total_customers = results['rfm'].get('total_customers', 0)
        html += f"""
                    <div class="kpi-card orange">
                        <div class="label">Total Customers</div>
                        <div class="value">{total_customers:,}<span class="unit"> customers</span></div>
                    </div>
        """

    if 'funnel' in results:
        conversion_rate = results['funnel'].get('overall_conversion_rate', 0)
        html += f"""
                    <div class="kpi-card">
                        <div class="label">Conversion Rate</div>
                        <div class="value">{conversion_rate:.1f}%<span class="unit"> efficiency</span></div>
                    </div>
        """

    if 'retention' in results:
        repeat_rate = results['retention'].get('repeat_purchase_rate', 0)
        html += f"""
                    <div class="kpi-card">
                        <div class="label">Repeat Purchase Rate</div>
                        <div class="value">{repeat_rate:.1f}%<span class="unit"> retention</span></div>
                    </div>
        """

    if 'content' in results:
        avg_score = results['content'].get('avg_score', 0)
        html += f"""
                    <div class="kpi-card green">
                        <div class="label">Average Review Score</div>
                        <div class="value">{avg_score:.1f}<span class="unit"> / 5.0</span></div>
                    </div>
        """

    html += """
                </div>
            </section>

            <!-- Data Exploration Section -->
            <section id="eda" class="section">
                <h2>1. Data Exploration & Visualization</h2>

                <h3>Key Findings</h3>
                <div class="stats-grid">
"""

    if 'eda' in results:
        status_dist = results['eda'].get('order_status_distribution', {})
        if isinstance(status_dist, dict) and 'delivered' in status_dist:
            if isinstance(status_dist['delivered'], dict):
                for status, data in list(status_dist.items())[:4]:
                    if isinstance(data, dict):
                        count = data.get('counts', 0)
                        pct = data.get('percentages', 0)
                        badge_class = 'badge-success' if pct > 90 else 'badge-warning' if pct > 50 else 'badge-danger'
                        html += f"""
                    <div class="stat-item">
                        <div class="label">{status.title()}</div>
                        <div class="value">{count:,}</div>
                        <span class="badge {badge_class}">{pct:.1f}%</span>
                    </div>
                        """

    html += """
                </div>
"""

    # 添加EDA图表
    for img in images.get('eda', [])[:3]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- RFM Section -->
            <section id="rfm" class="section">
                <h2>2. RFM Customer Segmentation</h2>

                <h3>Customer Value Segments</h3>
"""

    if 'rfm' in results:
        html += '<div class="stats-grid">'
        for segment, data in list(results['rfm'].get('segments', {}).items())[:5]:
            count = data.get('count', 0)
            revenue = data.get('total_revenue', 0)
            html += f"""
                    <div class="stat-item">
                        <div class="label">{segment.replace('_', ' ').title()}</div>
                        <div class="value">{count:,}</div>
                        <small>${revenue:,.0f}</small>
                    </div>
            """
        html += '</div>'

    # 添加RFM图表
    for img in images.get('rfm', [])[:2]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- LTV Section -->
            <section id="ltv" class="section">
                <h2>3. Customer Lifetime Value Prediction</h2>

                <h3>Model Performance</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Test R²</th>
                                <th>Test RMSE</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    if 'ltv' in results:
        for model, perf in results['ltv'].get('models', {}).items():
            r2 = perf.get('test_r2', 0)
            rmse = perf.get('test_rmse', 0)
            html += f"""
                            <tr>
                                <td>{model}</td>
                                <td>{r2:.4f}</td>
                                <td>${rmse:.2f}</td>
                            </tr>
            """

    html += """
                        </tbody>
                    </table>
                </div>
"""

    # 添加LTV图表
    for img in images.get('ltv', [])[:2]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- Retention Section -->
            <section id="retention" class="section">
                <h2>4. Customer Retention Analysis</h2>

                <div class="insight-box">
                    <h4>Key Retention Metrics</h4>
                    <ul>
"""

    if 'retention' in results:
        html += f"""
                        <li><strong>Repeat Purchase Rate:</strong> {results['retention'].get('repeat_purchase_rate', 0):.1f}%</li>
                        <li><strong>Average Purchases per Customer:</strong> {results['retention'].get('avg_purchases_per_customer', 0):.2f}</li>
                        <li><strong>Average Customer Lifespan:</strong> {results['retention'].get('avg_customer_lifespan_days', 0):.0f} days</li>
"""

    html += """
                    </ul>
                </div>
"""

    # 添加留存图表
    for img in images.get('retention', [])[:2]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- Funnel Section -->
            <section id="funnel" class="section">
                <h2>5. Conversion Funnel Analysis</h2>

                <h3>Funnel Stages Performance</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Stage</th>
                                <th>Orders</th>
                                <th>Conversion %</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    if 'funnel' in results:
        funnel_stages = results['funnel'].get('funnel_stages', {})
        stage_names = {
            'order_created': 'Order Created',
            'order_approved': 'Order Approved',
            'shipped_to_carrier': 'Shipped to Carrier',
            'delivered_to_customer': 'Delivered'
        }

        for stage_key, stage_name in list(stage_names.items())[:4]:
            count = funnel_stages.get(stage_key.replace(' ', '_').lower(), funnel_stages.get(stage_key, 0))
            if count == 0:
                # Try other keys
                for k in funnel_stages.keys():
                    if stage_key.lower() in k.lower():
                        count = funnel_stages[k]
                        break
            html += f"""
                            <tr>
                                <td>{stage_name}</td>
                                <td>{int(count):,}</td>
                                <td>{(count/99441*100):.1f}%</td>
                            </tr>
            """

    html += """
                        </tbody>
                    </table>
                </div>
"""

    # 添加漏斗图表
    for img in images.get('funnel', [])[:2]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- Growth Section -->
            <section id="growth" class="section">
                <h2>6. Growth Model Analysis</h2>

                <div class="insight-box">
                    <h4>Growth Performance Summary</h4>
                    <ul>
"""

    if 'growth' in results:
        html += f"""
                        <li><strong>Total Revenue:</strong> ${results['growth'].get('total_revenue', 0):,.2f}</li>
                        <li><strong>Average Monthly Revenue:</strong> ${results['growth'].get('avg_monthly_revenue', 0):,.2f}</li>
                        <li><strong>Revenue CAGR:</strong> {results['growth'].get('revenue_cagr_monthly', 0):.2f}% per month</li>
                        <li><strong>Analysis Period:</strong> {results['growth'].get('analysis_period_months', 0)} months</li>
"""

    html += """
                    </ul>
                </div>
"""

    # 添加增长图表
    for img in images.get('growth', [])[:2]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- Content Analysis Section -->
            <section id="content" class="section">
                <h2>7. Review Content Analysis</h2>

                <h3>Customer Sentiment Distribution</h3>
                <div class="stats-grid">
"""

    if 'content' in results:
        sentiment_dist = results['content'].get('sentiment_distribution', {})
        total = results['content'].get('total_reviews', 1)

        for sentiment, label in [('positive', 'Positive'), ('neutral', 'Neutral'), ('negative', 'Negative')]:
            count = sentiment_dist.get(sentiment, 0)
            pct = count / total * 100
            badge_class = 'badge-success' if sentiment == 'positive' else 'badge-warning' if sentiment == 'neutral' else 'badge-danger'
            html += f"""
                    <div class="stat-item">
                        <div class="label">{label}</div>
                        <div class="value">{count:,}</div>
                        <span class="badge {badge_class}">{pct:.1f}%</span>
                    </div>
            """

    html += """
                </div>
"""

    # 添加评论分析图表
    for img in images.get('content', [])[:2]:
        html += f"""
                <div class="chart-container">
                    <h4>{img['title']}</h4>
                    <img src="data:image/png;base64,{img['data']}" alt="{img['title']}">
                </div>
"""

    html += """
            </section>

            <!-- Strategic Insights Section -->
            <section id="insights" class="section">
                <h2>Strategic Insights & Recommendations</h2>

                <div class="insight-box">
                    <h4>Key Strengths</h4>
                    <ul>
                        <li><strong>Exceptional Growth Trajectory:</strong> {results['growth'].get('revenue_cagr_monthly', 0):.1f}% monthly CAGR demonstrates strong market expansion</li>
                        <li><strong>High Operational Efficiency:</strong> {results['funnel'].get('overall_conversion_rate', 0):.1f}% conversion rate indicates streamlined processes</li>
                        <li><strong>Positive Customer Sentiment:</strong> {results['content'].get('sentiment_distribution', {}).get('positive', 0)/results['content'].get('total_reviews', 1)*100:.1f}% positive reviews reflect quality service</li>
                        <li><strong>Strong Delivery Performance:</strong> {100-results['eda'].get('delivery_performance', {}).get('late_delivery_rate', 0):.1f}% on-time delivery rate</li>
                    </ul>
                </div>

                <div class="recommendation">
                    <h4>Priority Recommendations</h4>
                    <ul>
                        <li><strong>Boost Customer Retention:</strong> Current {results['retention'].get('repeat_purchase_rate', 0):.1f}% repeat purchase rate presents significant opportunity. Implement loyalty programs, personalized communication, and win-back campaigns.</li>
                        <li><strong>Increase Average Order Value:</strong> Leverage cross-selling and upselling strategies to maximize revenue per transaction.</li>
                        <li><strong>Enhance Customer Engagement:</strong> Reduce response times from current {results['content'].get('avg_response_time_days', 0):.0f} days to under 48 hours for improved satisfaction.</li>
                        <li><strong>Segment-Based Marketing:</strong> Utilize RFM segmentation insights for targeted campaigns across different customer value tiers.</li>
                        <li><strong>Continuous Monitoring:</strong> Establish real-time dashboards for tracking KPIs across all business dimensions.</li>
                    </ul>
                </div>

                <div class="insight-box">
                    <h4>Growth Opportunities</h4>
                    <ul>
                        <li>Convert one-time buyers to repeat customers through retention strategies</li>
                        <li>Optimize marketing spend based on customer lifetime value predictions</li>
                        <li>Expand product categories based on positive review themes</li>
                        <li>Improve operational efficiency in weakest funnel stage</li>
                        <li>Leverage high customer satisfaction for referral programs</li>
                    </ul>
                </div>
            </section>

        </div>

        <footer class="footer">
            <p><strong>Comprehensive Data Analysis Report</strong></p>
            <p>Generated by Claude Data Analysis Assistant - do-more Automation</p>
            <p>Analysis Date: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
            <p>Skills Executed: {len(results)} | Data Source: Olist Brazilian E-commerce Dataset</p>
        </footer>
    </div>

    <a href="#" class="scroll-top">↑</a>

    <script>
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});

        // Animate elements on scroll
        const observerOptions = {{
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        }};

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }}
            }});
        }}, observerOptions);

        document.querySelectorAll('.kpi-card, .stat-item, .chart-container').forEach(el => {{
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(el);
        }});
    </script>
</body>
</html>
"""

    # 保存HTML文件
    html_path = os.path.join(output_dir, 'Comprehensive_Analysis_Report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n[OK] HTML report saved to: {html_path}")
    print(f"     File size: {len(html)/1024:.1f} KB")

    # 同时保存一个简化版本（不包含图片base64）
    html_light = html.replace(r'<img src="data:image/png;base64,[^"]*"', '<img src="CHART_PLACEHOLDER"')
    light_path = os.path.join(output_dir, 'Comprehensive_Analysis_Report_Light.html')
    with open(light_path, 'w', encoding='utf-8') as f:
        f.write(html_light)

    print(f"[OK] Light version saved to: {light_path}")

    print("\n" + "="*80)
    print("HTML Report Generation Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - Comprehensive_Analysis_Report.html (with embedded charts)")
    print("  - Comprehensive_Analysis_Report_Light.html (lighter version)")
    print("\nOpen the HTML file in your browser to view the interactive report!")

if __name__ == "__main__":
    main()
