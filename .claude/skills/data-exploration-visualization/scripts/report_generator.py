#!/usr/bin/env python3
"""
报告生成器 (Report Generator) - 专业分析报告自动生成模块

提供全面的报告生成功能，包括：
- HTML交互式报告生成
- PDF高质量报告导出
- Markdown轻量级报告
- 自定义报告模板
- 多图表集成展示
- 数据洞察和建议生成
- 医疗数据报告特化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Template, Environment, FileSystemLoader
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """报告生成器 - 专业数据分析报告自动生成引擎"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化报告生成器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.templates = {}
        self.report_data = {}
        self.chart_cache = {}

        # 默认配置
        self.default_config = {
            'report_title': '数据分析报告',
            'author': '数据分析助手',
            'company': '',
            'theme': 'modern',  # 主题样式
            'language': 'zh',  # 报告语言
            'include_toc': True,  # 是否包含目录
            'include_summary': True,  # 是否包含摘要
            'include_recommendations': True,  # 是否包含建议
            'max_charts_per_page': 6,  # 每页最大图表数
            'chart_format': 'html',  # 图表格式 (html, png, svg)
            'page_size': 'A4',  # 页面大小
            'margin': '2cm',  # 页边距
            'medical_specialization': False  # 是否启用医疗数据特化
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

        # 初始化模板环境
        self._initialize_templates()

    def _initialize_templates(self):
        """初始化报告模板"""
        # HTML报告模板
        self.html_template = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }
        .header .meta {
            color: #666;
            margin-top: 10px;
        }
        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .toc h2 {
            margin-top: 0;
            color: #007acc;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            margin: 5px 0;
        }
        .toc a {
            text-decoration: none;
            color: #007acc;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #007acc;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section h3 {
            color: #495057;
            margin-top: 25px;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
        }
        .chart-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #495057;
        }
        .insight-box {
            background-color: #e7f3ff;
            border-left: 4px solid #007acc;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }
        .recommendation-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }
        .warning-box {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid #007acc;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #666;
        }
        @media print {
            body { background-color: white; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ report_title }}</h1>
            <div class="meta">
                <p>生成时间: {{ generation_time }}</p>
                <p>分析师: {{ author }}</p>
                {% if company %}<p>机构: {{ company }}</p>{% endif %}
            </div>
        </div>

        {% if include_toc %}
        <div class="toc">
            <h2>目录</h2>
            <ul>
                {% if include_summary %}<li><a href="#summary">1. 分析摘要</a></li>{% endif %}
                <li><a href="#data-overview">2. 数据概览</a></li>
                <li><a href="#eda-analysis">3. 探索性数据分析</a></li>
                {% if model_results %}<li><a href="#modeling">4. 模型分析</a></li>{% endif %}
                <li><a href="#insights">5. 关键洞察</a></li>
                {% if include_recommendations %}<li><a href="#recommendations">6. 建议</a></li>{% endif %}
            </ul>
        </div>
        {% endif %}

        {% if include_summary %}
        <div class="section" id="summary">
            <h2>1. 分析摘要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ data_summary.total_rows | format_number }}</div>
                    <div class="stat-label">数据行数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ data_summary.total_columns }}</div>
                    <div class="stat-label">特征数量</div>
                </div>
                {% if data_summary.quality_score %}
                <div class="stat-card">
                    <div class="stat-value">{{ "%.1f"|format(data_summary.quality_score) }}%</div>
                    <div class="stat-label">数据质量分数</div>
                </div>
                {% endif %}
                {% if model_results and model_results.best_model %}
                <div class="stat-card">
                    <div class="stat-value">{{ "%.3f"|format(model_results.best_model.accuracy) }}</div>
                    <div class="stat-label">最佳模型准确率</div>
                </div>
                {% endif %}
            </div>
            <p>{{ summary_text }}</p>
        </div>
        {% endif %}

        <div class="section" id="data-overview">
            <h2>2. 数据概览</h2>
            <h3>基本信息</h3>
            <div class="table-container">
                <table>
                    <tr><th>项目</th><th>值</th></tr>
                    <tr><td>数据集大小</td><td>{{ data_summary.total_rows }} 行 × {{ data_summary.total_columns }} 列</td></tr>
                    <tr><td>缺失值</td><td>{{ data_summary.missing_count }} ({{ "%.1f"|format(data_summary.missing_percentage) }}%)</td></tr>
                    <tr><td>重复行数</td><td>{{ data_summary.duplicate_count }}</td></tr>
                    <tr><td>内存使用</td><td>{{ "%.2f"|format(data_summary.memory_usage) }} MB</td></tr>
                </table>
            </div>

            <h3>变量类型分布</h3>
            <div class="chart-container">
                <div class="chart-title">变量类型分布</div>
                {{ data_type_chart | safe }}
            </div>
        </div>

        <div class="section" id="eda-analysis">
            <h2>3. 探索性数据分析</h2>
            {% for chart_name, chart_data in eda_charts.items() %}
            <div class="chart-container">
                <div class="chart-title">{{ chart_name }}</div>
                {{ chart_data | safe }}
            </div>
            {% endfor %}
        </div>

        {% if model_results %}
        <div class="section" id="modeling">
            <h2>4. 模型分析</h2>
            <h3>模型性能比较</h3>
            <div class="chart-container">
                <div class="chart-title">模型性能比较</div>
                {{ model_results.model_comparison_chart | safe }}
            </div>

            <h3>最佳模型详情</h3>
            <div class="insight-box">
                <h4>最佳模型: {{ model_results.best_model.name }}</h4>
                <p>准确率: {{ "%.3f"|format(model_results.best_model.accuracy) }}</p>
                <p>精确率: {{ "%.3f"|format(model_results.best_model.precision) }}</p>
                <p>召回率: {{ "%.3f"|format(model_results.best_model.recall) }}</p>
                <p>F1分数: {{ "%.3f"|format(model_results.best_model.f1) }}</p>
            </div>

            {% if model_results.feature_importance %}
            <h3>特征重要性</h3>
            <div class="chart-container">
                <div class="chart-title">Top 10 特征重要性</div>
                {{ model_results.feature_importance_chart | safe }}
            </div>
            {% endif %}
        </div>
        {% endif %}

        <div class="section" id="insights">
            <h2>5. 关键洞察</h2>
            {% for insight in insights %}
            <div class="insight-box">
                <h4>{{ insight.title }}</h4>
                <p>{{ insight.description }}</p>
                {% if insight.impact %}<p><strong>影响:</strong> {{ insight.impact }}</p>{% endif %}
            </div>
            {% endfor %}
        </div>

        {% if include_recommendations %}
        <div class="section" id="recommendations">
            <h2>6. 建议</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation-box">
                <h4>{{ recommendation.title }}</h4>
                <p>{{ recommendation.description }}</p>
                {% if recommendation.priority %}<p><strong>优先级:</strong> {{ recommendation.priority }}</p>{% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="footer">
            <p>报告由数据分析助手自动生成 | 生成时间: {{ generation_time }}</p>
        </div>
    </div>
</body>
</html>
        """

        # Markdown报告模板
        self.markdown_template = """
# {{ report_title }}

**生成时间:** {{ generation_time }}
**分析师:** {{ author }}{% if company %}
**机构:** {{ company }}{% endif %}

{% if include_summary %}

## 1. 分析摘要

### 关键指标
- 数据行数: {{ data_summary.total_rows | format_number }}
- 特征数量: {{ data_summary.total_columns }}
{% if data_summary.quality_score %}- 数据质量分数: {{ "%.1f"|format(data_summary.quality_score) }}%{% endif %}
{% if model_results and model_results.best_model %}- 最佳模型准确率: {{ "%.3f"|format(model_results.best_model.accuracy) }}{% endif %}

{{ summary_text }}

{% endif %}

## 2. 数据概览

### 基本信息
| 项目 | 值 |
|------|-----|
| 数据集大小 | {{ data_summary.total_rows }} 行 × {{ data_summary.total_columns }} 列 |
| 缺失值 | {{ data_summary.missing_count }} ({{ "%.1f"|format(data_summary.missing_percentage) }}%) |
| 重复行数 | {{ data_summary.duplicate_count }} |
| 内存使用 | {{ "%.2f"|format(data_summary.memory_usage) }} MB |

### 变量类型分布
{% for type_name, count in data_summary.type_distribution.items() %}
- {{ type_name }}: {{ count }}
{% endfor %}

## 3. 探索性数据分析

{% for chart_name, chart_desc in eda_charts_description.items() %}
### {{ chart_name }}
{{ chart_desc }}

{% endfor %}

{% if model_results %}

## 4. 模型分析

### 模型性能比较
{% for model_name, performance in model_results.model_performances.items() %}
- **{{ model_name }}**: {{ "%.3f"|format(performance.accuracy) }}
{% endfor %}

### 最佳模型详情
**模型名称:** {{ model_results.best_model.name }}

**性能指标:**
- 准确率: {{ "%.3f"|format(model_results.best_model.accuracy) }}
- 精确率: {{ "%.3f"|format(model_results.best_model.precision) }}
- 召回率: {{ "%.3f"|format(model_results.best_model.recall) }}
- F1分数: {{ "%.3f"|format(model_results.best_model.f1) }}

{% if model_results.feature_importance %}

### 特征重要性
{% for feature in model_results.feature_importance[:10] %}
{{ loop.index }}. {{ feature.name }}: {{ "%.3f"|format(feature.importance) }}
{% endfor %}

{% endif %}
{% endif %}

## 5. 关键洞察

{% for insight in insights %}
### {{ loop.index }}. {{ insight.title }}
{{ insight.description }}
{% if insight.impact %}
**影响:** {{ insight.impact }}
{% endif %}

{% endfor %}

{% if include_recommendations %}

## 6. 建议

{% for recommendation in recommendations %}
### {{ loop.index }}. {{ recommendation.title }}
{{ recommendation.description }}
{% if recommendation.priority %}
**优先级:** {{ recommendation.priority }}
{% endif %}

{% endfor %}
{% endif %}

---
*报告由数据分析助手自动生成*
        """

    def generate_comprehensive_report(self, data: pd.DataFrame,
                                    eda_results: Optional[Dict] = None,
                                    model_results: Optional[Dict] = None,
                                    output_path: str = "data_analysis_report.html",
                                    format: str = "html") -> str:
        """
        生成综合分析报告

        Parameters:
        - data: 原始数据
        - eda_results: EDA分析结果
        - model_results: 模型分析结果
        - output_path: 输出路径
        - format: 报告格式 (html, markdown, pdf)

        Returns:
        - 生成的报告路径
        """
        print("📝 生成综合分析报告...")

        # 准备报告数据
        self.report_data = {
            'report_title': self.config['report_title'],
            'author': self.config['author'],
            'company': self.config['company'],
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'include_toc': self.config['include_toc'],
            'include_summary': self.config['include_summary'],
            'include_recommendations': self.config['include_recommendations']
        }

        # 数据摘要
        data_summary = self._create_data_summary(data)
        self.report_data['data_summary'] = data_summary

        # EDA结果处理
        eda_charts = {}
        eda_charts_description = {}
        if eda_results:
            eda_charts, eda_charts_description = self._process_eda_results(eda_results, data)

        self.report_data['eda_charts'] = eda_charts
        self.report_data['eda_charts_description'] = eda_charts_description

        # 模型结果处理
        processed_model_results = {}
        if model_results:
            processed_model_results = self._process_model_results(model_results)

        self.report_data['model_results'] = processed_model_results

        # 生成洞察
        insights = self._generate_insights(data, eda_results, model_results)
        self.report_data['insights'] = insights

        # 生成建议
        recommendations = self._generate_recommendations(data, eda_results, model_results)
        self.report_data['recommendations'] = recommendations

        # 生成摘要文本
        summary_text = self._generate_summary_text(data_summary, eda_results, model_results)
        self.report_data['summary_text'] = summary_text

        # 生成报告
        if format.lower() == 'html':
            output_file = self._generate_html_report(output_path)
        elif format.lower() == 'markdown':
            output_file = self._generate_markdown_report(output_path)
        elif format.lower() == 'pdf':
            output_file = self._generate_pdf_report(output_path)
        else:
            raise ValueError(f"不支持的报告格式: {format}")

        print(f"✅ 报告已生成: {output_file}")
        return output_file

    def _create_data_summary(self, data: pd.DataFrame) -> Dict:
        """创建数据摘要"""
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_count': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_count': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'type_distribution': data.dtypes.value_counts().to_dict(),
            'quality_score': None  # 可以从EDA结果中获取
        }

    def _process_eda_results(self, eda_results: Dict, data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """处理EDA结果，生成图表"""
        eda_charts = {}
        eda_charts_description = {}

        # 处理相关性热图
        if 'correlation' in eda_results:
            correlation_data = eda_results['correlation']
            if 'correlation_matrix' in correlation_data:
                fig = self._create_correlation_heatmap(correlation_data['correlation_matrix'])
                eda_charts['相关性分析'] = fig.to_html(include_plotlyjs=False, div_id="correlation_chart")
                eda_charts_description['相关性分析'] = '变量间相关性热图，红色表示正相关，蓝色表示负相关。'

        # 处理分布图
        if 'distribution' in eda_results:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                fig = self._create_distribution_chart(data, numeric_cols[:6])
                eda_charts['数值变量分布'] = fig.to_html(include_plotlyjs=False, div_id="distribution_chart")
                eda_charts_description['数值变量分布'] = '主要数值变量的分布情况，包括直方图和密度曲线。'

        # 处理分类变量
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            fig = self._create_categorical_chart(data, categorical_cols[:4])
            eda_charts['分类变量分析'] = fig.to_html(include_plotlyjs=False, div_id="categorical_chart")
            eda_charts_description['分类变量分析'] = '主要分类变量的类别分布情况。'

        return eda_charts, eda_charts_description

    def _process_model_results(self, model_results: Dict) -> Dict:
        """处理模型结果"""
        processed = {}

        # 模型性能比较
        model_performances = {}
        model_names = []
        accuracies = []

        for model_name, result in model_results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                model_performances[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0)
                }
                model_names.append(model_name)
                accuracies.append(metrics.get('accuracy', 0))

        # 模型比较图表
        if model_names:
            fig = self._create_model_comparison_chart(model_names, accuracies)
            processed['model_comparison_chart'] = fig.to_html(include_plotlyjs=False, div_id="model_comparison")
            processed['model_performances'] = model_performances

        # 最佳模型
        if model_performances:
            best_model_name = max(model_performances.keys(), key=lambda x: model_performances[x]['accuracy'])
            processed['best_model'] = {
                'name': best_model_name,
                **model_performances[best_model_name]
            }

        # 特征重要性
        if 'feature_importance' in model_results:
            importance_data = model_results['feature_importance']
            if isinstance(importance_data, dict) and importance_data:
                # 转换特征重要性格式
                feature_importance = []
                for feature_name, importance in importance_data.items():
                    if isinstance(importance, (int, float)):
                        feature_importance.append({
                            'name': feature_name,
                            'importance': importance
                        })

                if feature_importance:
                    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                    processed['feature_importance'] = feature_importance

                    # 特征重要性图表
                    top_features = feature_importance[:10]
                    fig = self._create_feature_importance_chart(top_features)
                    processed['feature_importance_chart'] = fig.to_html(include_plotlyjs=False, div_id="feature_importance")

        return processed

    def _create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """创建相关性热图"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))

        fig.update_layout(
            title='变量相关性热图',
            width=600,
            height=500
        )

        return fig

    def _create_distribution_chart(self, data: pd.DataFrame, columns: List[str]) -> go.Figure:
        """创建分布图"""
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns
        )

        for i, col in enumerate(columns):
            row = (i // n_cols) + 1
            col_idx = (i % n_cols) + 1

            fig.add_trace(
                go.Histogram(x=data[col], name=col),
                row=row, col=col_idx
            )

        fig.update_layout(
            title_text="数值变量分布",
            showlegend=False,
            height=300 * n_rows
        )

        return fig

    def _create_categorical_chart(self, data: pd.DataFrame, columns: List[str]) -> go.Figure:
        """创建分类变量图表"""
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns,
            specs=[[{"type": "domain"}] * n_cols] * n_rows
        )

        for i, col in enumerate(columns):
            if i >= n_rows * n_cols:
                break

            row = (i // n_cols) + 1
            col_idx = (i % n_cols) + 1

            value_counts = data[col].value_counts().head(10)

            fig.add_trace(
                go.Pie(labels=value_counts.index, values=value_counts.values, name=col),
                row=row, col=col_idx
            )

        fig.update_layout(
            title_text="分类变量分布",
            height=300 * n_rows
        )

        return fig

    def _create_model_comparison_chart(self, model_names: List[str], accuracies: List[float]) -> go.Figure:
        """创建模型比较图表"""
        fig = go.Figure(data=[
            go.Bar(x=model_names, y=accuracies, text=[f"{acc:.3f}" for acc in accuracies])
        ])

        fig.update_layout(
            title='模型性能比较',
            xaxis_title='模型',
            yaxis_title='准确率',
            yaxis=dict(range=[0, 1])
        )

        return fig

    def _create_feature_importance_chart(self, feature_importance: List[Dict]) -> go.Figure:
        """创建特征重要性图表"""
        features = [f['name'] for f in feature_importance]
        importances = [f['importance'] for f in feature_importance]

        fig = go.Figure(data=[
            go.Bar(x=importances, y=features, orientation='h')
        ])

        fig.update_layout(
            title='特征重要性 (Top 10)',
            xaxis_title='重要性分数',
            yaxis_title='特征',
            height=max(400, len(features) * 30)
        )

        return fig

    def _generate_insights(self, data: pd.DataFrame, eda_results: Optional[Dict],
                         model_results: Optional[Dict]) -> List[Dict]:
        """生成洞察"""
        insights = []

        # 数据质量洞察
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_percentage > 20:
            insights.append({
                'title': '数据质量需要关注',
                'description': f'数据集中有 {missing_percentage:.1f}% 的缺失值，可能影响分析结果的准确性。',
                'impact': '高'
            })
        elif missing_percentage > 5:
            insights.append({
                'title': '存在少量缺失值',
                'description': f'数据集中有 {missing_percentage:.1f}% 的缺失值，建议进行适当处理。',
                'impact': '中'
            })

        # 数据规模洞察
        if len(data) < 1000:
            insights.append({
                'title': '数据规模较小',
                'description': '数据集样本量较少，可能影响模型的泛化能力。建议收集更多数据或使用交叉验证。',
                'impact': '中'
            })

        # 特征洞察
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > len(data.columns) * 0.7:
            insights.append({
                'title': '数值特征占主导',
                'description': f'数据集中 {len(numeric_cols)}/{len(data.columns)} 个特征为数值型，适合进行统计分析。',
                'impact': '低'
            })

        # 模型洞察
        if model_results:
            best_accuracy = max([result.get('metrics', {}).get('accuracy', 0) for result in model_results.values()])
            if best_accuracy > 0.9:
                insights.append({
                    'title': '模型性能优异',
                    'description': f'最佳模型准确率达到 {best_accuracy:.3f}，具有很好的预测能力。',
                    'impact': '高'
                })
            elif best_accuracy > 0.8:
                insights.append({
                    'title': '模型性能良好',
                    'description': f'最佳模型准确率为 {best_accuracy:.3f}，具有较好的预测能力。',
                    'impact': '中'
                })
            elif best_accuracy < 0.7:
                insights.append({
                    'title': '模型性能有待提升',
                    'description': f'最佳模型准确率仅为 {best_accuracy:.3f}，建议进行特征工程或尝试其他算法。',
                    'impact': '高'
                })

        # 医疗数据特化洞察
        if self.config['medical_specialization']:
            # 检查是否有医疗相关特征
            medical_keywords = ['diagnosis', 'symptom', 'treatment', 'patient', 'medical', 'health']
            medical_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in medical_keywords)]

            if medical_cols:
                insights.append({
                    'title': '医疗数据特征',
                    'description': f'检测到 {len(medical_cols)} 个医疗相关特征，建议关注患者隐私和数据合规性。',
                    'impact': '高'
                })

        if not insights:
            insights.append({
                'title': '数据质量良好',
                'description': '数据集整体质量较好，适合进行进一步分析。',
                'impact': '低'
            })

        return insights

    def _generate_recommendations(self, data: pd.DataFrame, eda_results: Optional[Dict],
                                model_results: Optional[Dict]) -> List[Dict]:
        """生成建议"""
        recommendations = []

        # 数据清洗建议
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_percentage > 10:
            recommendations.append({
                'title': '处理缺失值',
                'description': '建议使用适当的填充策略（如均值、中位数或众数填充）来处理缺失值。',
                'priority': '高'
            })

        # 特征工程建议
        if len(data.columns) > 50:
            recommendations.append({
                'title': '特征选择',
                'description': '特征数量较多，建议使用特征选择技术来减少维度，提高模型效率。',
                'priority': '中'
            })

        # 数据平衡建议
        if model_results:
            # 检查类别平衡（如果是分类问题）
            for result in model_results.values():
                if 'classification_report' in result.get('metrics', {}):
                    # 这里可以进一步分析分类报告中的类别平衡情况
                    break

        # 模型改进建议
        if model_results:
            best_accuracy = max([result.get('metrics', {}).get('accuracy', 0) for result in model_results.values()])
            if best_accuracy < 0.8:
                recommendations.append({
                    'title': '模型优化',
                    'description': '建议尝试更多的特征工程、超参数调优或集成方法来提升模型性能。',
                    'priority': '高'
                })

        # 数据收集建议
        if len(data) < 5000:
            recommendations.append({
                'title': '增加数据量',
                'description': '当前数据量可能不足以训练复杂模型，建议收集更多数据。',
                'priority': '中'
            })

        # 验证建议
        recommendations.append({
            'title': '模型验证',
            'description': '建议在独立的测试集上验证模型性能，确保模型的泛化能力。',
            'priority': '高'
        })

        return recommendations

    def _generate_summary_text(self, data_summary: Dict, eda_results: Optional[Dict],
                             model_results: Optional[Dict]) -> str:
        """生成摘要文本"""
        summary_parts = []

        # 数据概况
        summary_parts.append(
            f"本次分析处理了包含 {data_summary['total_rows']:,} 行和 "
            f"{data_summary['total_columns']} 个特征的数据集。"
        )

        # 数据质量
        if data_summary['missing_percentage'] > 10:
            summary_parts.append(
                f"数据集中存在 {data_summary['missing_percentage']:.1f}% 的缺失值，"
                "需要在分析前进行适当处理。"
            )

        # 分析发现
        if eda_results:
            summary_parts.append(
                "探索性数据分析揭示了数据的主要特征和变量间的关系。"
            )

        # 模型性能
        if model_results:
            best_acc = max([result.get('metrics', {}).get('accuracy', 0) for result in model_results.values()])
            summary_parts.append(
                f"机器学习建模显示最佳模型可达到 {best_acc:.1%} 的预测准确率。"
            )

        # 总体评价
        if data_summary['missing_percentage'] < 5 and (not model_results or best_acc > 0.8):
            summary_parts.append("整体而言，数据质量良好，分析结果具有较好的可靠性。")
        else:
            summary_parts.append("建议进一步进行数据清洗和特征工程以提升分析质量。")

        return " ".join(summary_parts)

    def _generate_html_report(self, output_path: str) -> str:
        """生成HTML报告"""
        # 创建Jinja2模板
        template = Template(self.html_template)

        # 添加自定义过滤器
        template.globals['format_number'] = lambda x: f"{x:,}"

        # 渲染模板
        html_content = template.render(**self.report_data)

        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_markdown_report(self, output_path: str) -> str:
        """生成Markdown报告"""
        # 创建Jinja2模板
        template = Template(self.markdown_template)

        # 添加自定义过滤器
        template.globals['format_number'] = lambda x: f"{x:,}"

        # 渲染模板
        markdown_content = template.render(**self.report_data)

        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return output_path

    def _generate_pdf_report(self, output_path: str) -> str:
        """生成PDF报告"""
        try:
            # 先生成HTML
            html_path = output_path.replace('.pdf', '.html')
            self._generate_html_report(html_path)

            # 转换为PDF（需要安装weasyprint）
            import weasyprint
            weasyprint.HTML(filename=html_path).write_pdf(output_path)

            # 删除临时HTML文件
            Path(html_path).unlink()

            return output_path

        except ImportError:
            print("⚠️ 需要安装 weasyprint 来生成PDF报告: pip install weasyprint")
            # 回退到HTML格式
            return self._generate_html_report(output_path.replace('.pdf', '.html'))
        except Exception as e:
            print(f"⚠️ PDF生成失败: {str(e)}")
            # 回退到HTML格式
            return self._generate_html_report(output_path.replace('.pdf', '.html'))

    def generate_quick_report(self, data: pd.DataFrame, target_col: Optional[str] = None,
                            output_path: str = "quick_report.html") -> str:
        """
        生成快速报告（简化版）

        Parameters:
        - data: 数据DataFrame
        - target_col: 目标列名（可选）
        - output_path: 输出路径

        Returns:
        - 生成的报告路径
        """
        print("📝 生成快速分析报告...")

        # 快速EDA
        eda_analyzer = None
        try:
            from scripts.eda_analyzer import EDAAnalyzer
            eda_analyzer = EDAAnalyzer()
            eda_results = eda_analyzer.auto_eda(data)
        except:
            eda_results = None

        # 快速建模（如果有目标列）
        model_results = None
        if target_col:
            try:
                from scripts.modeling_evaluator import ModelingEvaluator
                modeler = ModelingEvaluator()
                model_results = modeler.auto_modeling(data, target_col)
            except:
                pass

        # 生成报告
        return self.generate_comprehensive_report(
            data=data,
            eda_results=eda_results,
            model_results=model_results,
            output_path=output_path,
            format="html"
        )

    def export_report_data(self, output_path: str):
        """
        导出报告数据为JSON格式

        Parameters:
        - output_path: 输出路径
        """
        # 准备可序列化的数据
        export_data = {
            'config': self.config,
            'report_data': self.report_data,
            'generation_time': datetime.now().isoformat()
        }

        # 处理不可序列化的对象
        for key, value in export_data['report_data'].items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                export_data['report_data'][key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)

        # 保存JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"✅ 报告数据已导出到 {output_path}")

    def get_report_summary(self) -> Dict:
        """
        获取报告摘要信息

        Returns:
        - 报告摘要
        """
        return {
            'config': self.config,
            'report_data_keys': list(self.report_data.keys()),
            'chart_cache_size': len(self.chart_cache),
            'template_count': len(self.templates)
        }