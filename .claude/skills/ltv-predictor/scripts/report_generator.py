#!/usr/bin/env python3
"""
报告生成器
基于第3课理论实现的专业LTV分析报告生成功能
支持HTML、Markdown和Excel格式的报告输出
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from datetime import datetime
import json
from pathlib import Path

class ReportGenerator:
    """
    报告生成器

    生成专业的LTV分析报告，支持多种输出格式
    整合数据洞察、模型评估和业务建议
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化报告生成器

        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            'report_title': '客户生命周期价值(LTV)分析报告',
            'author': 'LTV预测系统',
            'company': '数据分析部门',
            'include_summary': True,
            'include_recommendations': True,
            'include_technical_details': True,
            'language': 'zh-CN'
        }

        # 更新配置
        if config:
            self.config.update(config)

        print("📋 报告生成器初始化完成")

    def generate_html_report(self,
                           rfm_data: pd.DataFrame,
                           model_results: Dict[str, Dict],
                           feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                           summary_report: Optional[Dict[str, Any]] = None,
                           output_path: str = 'ltv_analysis_report.html') -> str:
        """
        生成HTML格式的分析报告

        Args:
            rfm_data: RFM特征数据
            model_results: 模型训练结果
            feature_importance: 特征重要性
            summary_report: 摘要报告
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        print("📄 生成HTML分析报告...")

        # 生成报告内容
        html_content = self._generate_html_content(
            rfm_data, model_results, feature_importance, summary_report
        )

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ HTML报告已生成: {output_path}")
        return output_path

    def _generate_html_content(self,
                             rfm_data: pd.DataFrame,
                             model_results: Dict[str, Dict],
                             feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                             summary_report: Optional[Dict[str, Any]] = None) -> str:
        """生成HTML报告内容"""
        # 计算关键指标
        total_customers = len(rfm_data)
        avg_ltv = rfm_data['年度LTV'].mean() if '年度LTV' in rfm_data.columns else 0

        # 获取最佳模型信息
        best_model_name = None
        best_model_r2 = 0
        for model_name, result in model_results.items():
            if 'error' not in result and result.get('r2_score', 0) > best_model_r2:
                best_model_name = model_name
                best_model_r2 = result.get('r2_score', 0)

        # 生成HTML
        html_template = f"""
<!DOCTYPE html>
<html lang="{self.config['language']}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config['report_title']}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #007acc;
            border-left: 5px solid #007acc;
            padding-left: 15px;
            font-size: 1.8em;
        }}
        .section h3 {{
            color: #333;
            font-size: 1.4em;
            margin-top: 25px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #007acc, #005f99);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 1em;
            opacity: 0.9;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .table th {{
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }}
        .table tr:hover {{
            background-color: #f5f5f5;
        }}
        .recommendations {{
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .recommendations h3 {{
            color: #4caf50;
            margin-top: 0;
        }}
        .recommendation-item {{
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 3px;
            border-left: 3px solid #4caf50;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 5px solid #ffc107;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .model-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .model-card.best {{
            border: 2px solid #4caf50;
            background-color: #f1f8e9;
        }}
        .model-card h4 {{
            margin-top: 0;
            color: #333;
        }}
        .best-model {{
            color: #4caf50;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 报告头部 -->
        <div class="header">
            <h1>{self.config['report_title']}</h1>
            <div class="subtitle">
                基于RFM模型和回归算法的客户生命周期价值预测分析
            </div>
            <div class="subtitle">
                生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}
            </div>
        </div>

        <!-- 核心指标 -->
        <div class="section">
            <h2>📊 核心分析指标</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-label">分析客户总数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_ltv:,.0f}</div>
                    <div class="metric-label">平均客户LTV</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_model_r2:.4f}</div>
                    <div class="metric-label">最佳模型R²分数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_model_name or 'Unknown'}</div>
                    <div class="metric-label">最佳预测模型</div>
                </div>
            </div>
        </div>

        <!-- RFM分析结果 -->
        <div class="section">
            <h2>🔍 RFM特征分析</h2>
            <p>RFM模型通过分析客户的最近消费时间(R)、消费频率(F)和消费金额(M)来评估客户价值。</p>

            {self._generate_rfm_summary_table(rfm_data)}

            {self._generate_customer_segments_section(rfm_data)}
        </div>

        <!-- 模型性能对比 -->
        <div class="section">
            <h2>🤖 预测模型性能对比</h2>
            <p>我们训练了多种机器学习模型来预测客户生命周期价值，以下是各模型的性能表现：</p>

            {self._generate_model_comparison_section(model_results)}

            {self._generate_feature_importance_section(feature_importance)}
        </div>

        <!-- 业务洞察和建议 -->
        {self._generate_business_insights_section(summary_report, rfm_data, model_results)}

        <!-- 技术详情 -->
        {self._generate_technical_details_section(summary_report)}

        <!-- 报告尾部 -->
        <div class="footer">
            <p>本报告由 {self.config['author']} 自动生成</p>
            <p>如有疑问，请联系 {self.config['company']}</p>
        </div>
    </div>
</body>
</html>
        """
        return html_template

    def _generate_rfm_summary_table(self, rfm_data: pd.DataFrame) -> str:
        """生成RFM摘要表格"""
        if not all(col in rfm_data.columns for col in ['R值', 'F值', 'M值']):
            return "<p>⚠️ RFM特征数据不完整</p>"

        rfm_stats = {
            '指标': ['R值 (最近消费间隔)', 'F值 (消费频率)', 'M值 (消费金额)', '年度LTV'],
            '平均值': [
                f"{rfm_data['R值'].mean():.1f}天",
                f"{rfm_data['F值'].mean():.1f}次",
                f"¥{rfm_data['M值'].mean():.2f}",
                f"¥{rfm_data['年度LTV'].mean():.2f}"
            ],
            '中位数': [
                f"{rfm_data['R值'].median():.1f}天",
                f"{rfm_data['F值'].median():.1f}次",
                f"¥{rfm_data['M值'].median():.2f}",
                f"¥{rfm_data['年度LTV'].median():.2f}"
            ],
            '标准差': [
                f"{rfm_data['R值'].std():.1f}",
                f"{rfm_data['F值'].std():.1f}",
                f"¥{rfm_data['M值'].std():.2f}",
                f"¥{rfm_data['年度LTV'].std():.2f}"
            ]
        }

        table_html = '<table class="table">'
        table_html += '<thead><tr>' + ''.join(f'<th>{col}</th>' for col in rfm_stats['指标']) + '</tr></thead>'
        table_html += '<tbody>'

        for i in range(len(rfm_stats['指标'])):
            table_html += f'<tr>'
            table_html += f'<td><strong>{rfm_stats["指标"][i]}</strong></td>'
            table_html += f'<td>{rfm_stats["平均值"][i]}</td>'
            table_html += f'<td>{rfm_stats["中位数"][i]}</td>'
            table_html += f'<td>{rfm_stats["标准差"][i]}</td>'
            table_html += f'</tr>'

        table_html += '</tbody></table>'
        return table_html

    def _generate_customer_segments_section(self, rfm_data: pd.DataFrame) -> str:
        """生成客户分群部分"""
        if '客户价值分层' not in rfm_data.columns:
            return ""

        segment_counts = rfm_data['客户价值分层'].value_counts()
        segment_ltv = rfm_data.groupby('客户价值分层')['年度LTV'].mean()

        section_html = """
        <h3>客户价值分层</h3>
        <p>基于RFM特征对客户进行价值分层，识别不同价值的客户群体：</p>
        """

        section_html += '<table class="table">'
        section_html += '<thead><tr><th>客户层级</th><th>客户数量</th><th>占比</th><th>平均LTV</th></tr></thead>'
        section_html += '<tbody>'

        total_customers = len(rfm_data)
        for segment in ['钻石客户', '白金客户', '金牌客户', '银牌客户', '铜牌客户']:
            if segment in segment_counts:
                count = segment_counts[segment]
                percentage = (count / total_customers) * 100
                avg_ltv = segment_ltv.get(segment, 0)

                section_html += f'''
                <tr>
                    <td><strong>{segment}</strong></td>
                    <td>{count:,}</td>
                    <td>{percentage:.1f}%</td>
                    <td>¥{avg_ltv:,.0f}</td>
                </tr>
                '''

        section_html += '</tbody></table>'
        return section_html

    def _generate_model_comparison_section(self, model_results: Dict[str, Dict]) -> str:
        """生成模型对比部分"""
        # 获取最佳模型
        best_model_name = None
        best_model_r2 = 0
        for model_name, result in model_results.items():
            if 'error' not in result and result.get('r2_score', 0) > best_model_r2:
                best_model_name = model_name
                best_model_r2 = result.get('r2_score', 0)

        section_html = '<div class="model-comparison">'

        for model_name, result in model_results.items():
            if 'error' in result:
                continue

            is_best = (model_name == best_model_name)
            card_class = 'model-card best' if is_best else 'model-card'

            section_html += f'''
            <div class="{card_class}">
                <h4>{model_name} {'🏆 最佳模型' if is_best else ''}</h4>
                <p><strong>R² 分数:</strong> {result.get('r2_score', 0):.4f}</p>
                <p><strong>平均绝对误差 (MAE):</strong> {result.get('mae', 0):.2f}</p>
                <p><strong>均方根误差 (RMSE):</strong> {result.get('rmse', 0):.2f}</p>
                <p><strong>平均绝对百分比误差 (MAPE):</strong> {result.get('mape', 0):.2f}%</p>
            </div>
            '''

        section_html += '</div>'

        if best_model_r2 > 0.7:
            section_html += '''
            <div class="highlight">
                <strong>🎯 模型性能评估:</strong> 最佳模型的R²分数超过0.7，表明模型具有优秀的预测能力，
                可用于精准的客户价值评估和营销决策。
            </div>
            '''
        elif best_model_r2 > 0.5:
            section_html += '''
            <div class="highlight">
                <strong>📈 模型性能评估:</strong> 最佳模型的R²分数在0.5-0.7之间，表明模型具有较好的预测能力，
                建议结合业务规则进行客户价值管理。
            </div>
            '''
        else:
            section_html += '''
            <div class="highlight">
                <strong>⚠️ 模型性能评估:</strong> 最佳模型的R²分数低于0.5，预测能力有限，
                建议增加更多特征数据或尝试其他算法以提高预测准确性。
            </div>
            '''

        return section_html

    def _generate_feature_importance_section(self, feature_importance: Optional[Dict[str, Dict[str, float]]]) -> str:
        """生成特征重要性部分"""
        if not feature_importance:
            return ""

        # 选择最佳模型的特征重要性
        best_model_importance = None
        for model_name, importance in feature_importance.items():
            if importance and (best_model_importance is None or model_name == 'random_forest'):
                best_model_importance = importance

        if not best_model_importance:
            return ""

        section_html = '<h3>特征重要性分析</h3>'
        section_html += '<p>以下是影响LTV预测最重要的特征及其贡献度：</p>'

        section_html += '<table class="table">'
        section_html += '<thead><tr><th>特征名称</th><th>重要性分数</th><th>贡献度</th><th>业务解释</th></tr></thead>'
        section_html += '<tbody>'

        total_importance = sum(best_model_importance.values())
        feature_descriptions = {
            'R值': '客户最近一次消费时间间隔，反映客户活跃度',
            'F值': '客户消费频率，反映客户忠诚度',
            'M值': '客户消费金额，反映客户价值贡献'
        }

        for feature, importance in sorted(best_model_importance.items(), key=lambda x: x[1], reverse=True):
            contribution = (importance / total_importance) * 100
            description = feature_descriptions.get(feature, '自定义特征')

            section_html += f'''
            <tr>
                <td><strong>{feature}</strong></td>
                <td>{importance:.4f}</td>
                <td>{contribution:.1f}%</td>
                <td>{description}</td>
            </tr>
            '''

        section_html += '</tbody></table>'
        return section_html

    def _generate_business_insights_section(self, summary_report: Optional[Dict[str, Any]],
                                          rfm_data: pd.DataFrame,
                                          model_results: Dict[str, Dict]) -> str:
        """生成业务洞察部分"""
        if not self.config['include_recommendations']:
            return ""

        section_html = '''
        <div class="section">
            <h2>💡 业务洞察与建议</h2>
        '''

        # 获取建议
        recommendations = []
        if summary_report and 'recommendations' in summary_report:
            recommendations = summary_report['recommendations']

        # 添加数据分析洞察
        total_customers = len(rfm_data)
        avg_ltv = rfm_data['年度LTV'].mean() if '年度LTV' in rfm_data.columns else 0

        # 客户价值洞察
        if '客户价值分层' in rfm_data.columns:
            high_value_ratio = (rfm_data['客户价值分层'].isin(['白金客户', '钻石客户'])).sum() / total_customers
            if high_value_ratio > 0.2:
                recommendations.append("高价值客户占比较高(>20%)，建议重点维护这些核心客户群体")

        # 模型性能洞察
        best_r2 = max([result.get('r2_score', 0) for result in model_results.values() if 'error' not in result])
        if best_r2 > 0.6:
            recommendations.append("模型预测性能优秀，可用于制定精准营销策略和客户价值管理")

        # 添加默认建议
        if not recommendations:
            recommendations = [
                "定期监控客户RFM特征变化，及时调整客户管理策略",
                "基于LTV预测结果制定差异化的客户服务方案",
                "持续收集客户行为数据，提高模型预测准确性"
            ]

        section_html += '<div class="recommendations">'
        section_html += '<h3>📋 业务建议</h3>'

        for i, recommendation in enumerate(recommendations, 1):
            section_html += f'<div class="recommendation-item">{i}. {recommendation}</div>'

        section_html += '</div>'
        section_html += '</div>'

        return section_html

    def _generate_technical_details_section(self, summary_report: Optional[Dict[str, Any]]) -> str:
        """生成技术详情部分"""
        if not self.config['include_technical_details']:
            return ""

        section_html = '''
        <div class="section">
            <h2>🔧 技术实现详情</h2>
            <h3>分析方法</h3>
            <ul>
                <li><strong>RFM模型:</strong> 通过最近消费时间(R)、消费频率(F)、消费金额(M)三个维度评估客户价值</li>
                <li><strong>时间窗口分析:</strong> 使用3个月的历史数据预测12个月的客户生命周期价值</li>
                <li><strong>机器学习算法:</strong> 集成线性回归、随机森林等多种算法进行预测建模</li>
                <li><strong>交叉验证:</strong> 使用5折交叉验证确保模型性能评估的可靠性</li>
            </ul>

            <h3>数据处理流程</h3>
            <ul>
                <li><strong>数据清洗:</strong> 移除异常值和缺失值处理</li>
                <li><strong>特征工程:</strong> RFM特征计算和时间窗口划分</li>
                <li><strong>模型训练:</strong> 多算法并行训练和超参数优化</li>
                <li><strong>模型评估:</strong> 使用R²、MAE、RMSE、MAPE等多种评估指标</li>
            </ul>

            <h3>应用场景</h3>
            <ul>
                <li><strong>客户分层管理:</strong> 基于LTV预测结果进行客户价值分层</li>
                <li><strong>营销预算分配:</strong> 根据客户价值预测优化营销投入</li>
                <li><strong>客户流失预警:</strong> 识别低价值和高流失风险客户</li>
                <li><strong>个性化服务:</strong> 为不同价值客户提供差异化服务</li>
            </ul>
        </div>
        '''

        return section_html

    def generate_markdown_report(self,
                               rfm_data: pd.DataFrame,
                               model_results: Dict[str, Dict],
                               feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                               output_path: str = 'ltv_analysis_report.md') -> str:
        """
        生成Markdown格式的分析报告

        Args:
            rfm_data: RFM特征数据
            model_results: 模型训练结果
            feature_importance: 特征重要性
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        print("📝 生成Markdown分析报告...")

        # 生成报告内容
        md_content = self._generate_markdown_content(rfm_data, model_results, feature_importance)

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✓ Markdown报告已生成: {output_path}")
        return output_path

    def _generate_markdown_content(self,
                                 rfm_data: pd.DataFrame,
                                 model_results: Dict[str, Dict],
                                 feature_importance: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """生成Markdown报告内容"""
        # 计算关键指标
        total_customers = len(rfm_data)
        avg_ltv = rfm_data['年度LTV'].mean() if '年度LTV' in rfm_data.columns else 0

        # 获取最佳模型信息
        best_model_name = None
        best_model_r2 = 0
        for model_name, result in model_results.items():
            if 'error' not in result and result.get('r2_score', 0) > best_model_r2:
                best_model_name = model_name
                best_model_r2 = result.get('r2_score', 0)

        # 生成Markdown内容
        md_content = f"""
# {self.config['report_title']}

**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}
**分析工具**: {self.config['author']}
**数据来源**: {self.config['company']}

---

## 📊 核心分析指标

| 指标 | 数值 |
|------|------|
| 分析客户总数 | {total_customers:,} |
| 平均客户LTV | ¥{avg_ltv:,.0f} |
| 最佳模型R²分数 | {best_model_r2:.4f} |
| 最佳预测模型 | {best_model_name or 'Unknown'} |

---

## 🔍 RFM特征分析

RFM模型通过分析客户的最近消费时间(R)、消费频率(F)和消费金额(M)来评估客户价值。

### RFM特征统计

| 指标 | 平均值 | 中位数 | 标准差 |
|------|--------|--------|--------|
"""

        # 添加RFM统计表格
        if all(col in rfm_data.columns for col in ['R值', 'F值', 'M值', '年度LTV']):
            rfmdict = {
                'R值 (最近消费间隔)': f"{rfm_data['R值'].mean():.1f}天",
                'F值 (消费频率)': f"{rfm_data['F值'].mean():.1f}次",
                'M值 (消费金额)': f"¥{rfm_data['M值'].mean():.2f}",
                '年度LTV': f"¥{rfm_data['年度LTV'].mean():.2f}"
            }

            for feature, avg_val in rfmdict.items():
                md_content += f"| {feature} | {avg_val} | ...\n"

        # 添加模型性能对比
        md_content += f"""

---

## 🤖 预测模型性能对比

我们训练了多种机器学习模型来预测客户生命周期价值：

| 模型名称 | R² 分数 | MAE | RMSE | MAPE |
|----------|---------|-----|------|------|
"""

        for model_name, result in model_results.items():
            if 'error' in result:
                continue
            md_content += f"| {model_name} | {result.get('r2_score', 0):.4f} | {result.get('mae', 0):.2f} | {result.get('rmse', 0):.2f} | {result.get('mape', 0):.2f}% |\n"

        # 添加特征重要性
        if feature_importance:
            md_content += "\n---\n\n## 📈 特征重要性分析\n\n"

            # 选择最佳模型的特征重要性
            best_model_importance = None
            for model_name, importance in feature_importance.items():
                if importance and (best_model_importance is None or model_name == 'random_forest'):
                    best_model_importance = importance

            if best_model_importance:
                md_content += "| 特征名称 | 重要性分数 | 贡献度 |\n|----------|------------|--------|\n"
                total_importance = sum(best_model_importance.values())

                for feature, importance in sorted(best_model_importance.items(), key=lambda x: x[1], reverse=True):
                    contribution = (importance / total_importance) * 100
                    md_content += f"| {feature} | {importance:.4f} | {contribution:.1f}% |\n"

        # 添加业务建议
        md_content += """

---

## 💡 业务洞察与建议

### 主要发现

1. **客户价值分布**: 分析了客户的生命周期价值分布特征
2. **模型性能**: 建立的预测模型具有良好的解释能力和预测准确性
3. **关键特征**: 识别了影响客户价值的关键驱动因素

### 业务建议

1. **客户分层管理**: 基于LTV预测结果对客户进行分层，制定差异化的服务策略
2. **营销优化**: 将营销资源重点投向高价值客户群体
3. **客户维系**: 针对高流失风险客户制定挽留策略
4. **数据驱动**: 持续收集客户数据，优化预测模型

---

## 🔧 技术实现

- **分析方法**: RFM模型 + 机器学习回归算法
- **时间窗口**: 3个月历史数据预测12个月LTV
- **评估指标**: R²、MAE、RMSE、MAPE
- **交叉验证**: 5折交叉验证确保模型可靠性

---

*本报告由 {self.config['author']} 自动生成，如有疑问请联系 {self.config['company']}*
        """

        return md_content

    def export_summary_to_excel(self,
                              rfm_data: pd.DataFrame,
                              model_results: Dict[str, Dict],
                              feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                              output_path: str = 'ltv_analysis_summary.xlsx'):
        """
        导出分析摘要到Excel文件

        Args:
            rfm_data: RFM特征数据
            model_results: 模型训练结果
            feature_importance: 特征重要性
            output_path: 输出路径
        """
        print("📊 导出Excel分析摘要...")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. RFM数据
            rfm_data.to_excel(writer, sheet_name='RFM特征数据', index=False)

            # 2. 模型性能对比
            model_comparison = []
            for model_name, result in model_results.items():
                if 'error' not in result:
                    comparison_row = {
                        '模型名称': model_name,
                        'R²分数': result.get('r2_score', 0),
                        '平均绝对误差(MAE)': result.get('mae', 0),
                        '均方根误差(RMSE)': result.get('rmse', 0),
                        '平均绝对百分比误差(MAPE)': result.get('mape', 0)
                    }
                    model_comparison.append(comparison_row)

            if model_comparison:
                pd.DataFrame(model_comparison).to_excel(writer, sheet_name='模型性能对比', index=False)

            # 3. 特征重要性
            if feature_importance:
                for model_name, importance in feature_importance.items():
                    if importance:
                        importance_df = pd.DataFrame(list(importance.items()),
                                                   columns=['特征名称', '重要性分数'])
                        importance_df = importance_df.sort_values('重要性分数', ascending=False)
                        importance_df.to_excel(writer, sheet_name=f'特征重要性_{model_name}', index=False)

            # 4. 摘要统计
            summary_stats = {
                '指标': ['总客户数', '平均LTV', '中位数LTV', 'LTV标准差'],
                '数值': [
                    len(rfm_data),
                    rfm_data['年度LTV'].mean() if '年度LTV' in rfm_data.columns else 0,
                    rfm_data['年度LTV'].median() if '年度LTV' in rfm_data.columns else 0,
                    rfm_data['年度LTV'].std() if '年度LTV' in rfm_data.columns else 0
                ]
            }
            pd.DataFrame(summary_stats).to_excel(writer, sheet_name='摘要统计', index=False)

        print(f"✓ Excel摘要已导出: {output_path}")

# 便利函数
def generate_comprehensive_report(rfm_data: pd.DataFrame,
                                model_results: Dict[str, Dict],
                                feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                                output_dir: str = './reports',
                                report_title: str = '客户生命周期价值分析报告') -> Dict[str, str]:
    """
    生成完整的多格式分析报告

    Args:
        rfm_data: RFM特征数据
        model_results: 模型训练结果
        feature_importance: 特征重要性
        output_dir: 输出目录
        report_title: 报告标题

    Returns:
        生成的报告文件路径字典
    """
    from pathlib import Path

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化报告生成器
    generator = ReportGenerator({'report_title': report_title})

    report_paths = {}

    print("📋 生成完整分析报告...")

    # 1. 生成HTML报告
    try:
        html_path = os.path.join(output_dir, 'ltv_analysis_report.html')
        generator.generate_html_report(rfm_data, model_results, feature_importance, output_path=html_path)
        report_paths['html'] = html_path
    except Exception as e:
        print(f"⚠️ HTML报告生成失败: {str(e)}")

    # 2. 生成Markdown报告
    try:
        md_path = os.path.join(output_dir, 'ltv_analysis_report.md')
        generator.generate_markdown_report(rfm_data, model_results, feature_importance, output_path=md_path)
        report_paths['markdown'] = md_path
    except Exception as e:
        print(f"⚠️ Markdown报告生成失败: {str(e)}")

    # 3. 导出Excel摘要
    try:
        excel_path = os.path.join(output_dir, 'ltv_analysis_summary.xlsx')
        generator.export_summary_to_excel(rfm_data, model_results, feature_importance, output_path=excel_path)
        report_paths['excel'] = excel_path
    except Exception as e:
        print(f"⚠️ Excel摘要导出失败: {str(e)}")

    print(f"✓ 完整分析报告生成完成，共生成 {len(report_paths)} 个文件")
    print(f"  - 输出目录: {output_dir}")

    return report_paths

if __name__ == "__main__":
    # 示例使用
    print("📋 报告生成器测试")

    # 创建示例数据
    np.random.seed(42)
    sample_rfm = pd.DataFrame({
        'R值': np.random.randint(1, 90, 100),
        'F值': np.random.randint(1, 100, 100),
        'M值': np.random.uniform(100, 10000, 100),
        '年度LTV': np.random.uniform(500, 20000, 100),
        '客户价值分层': np.random.choice(['铜牌客户', '银牌客户', '金牌客户', '白金客户'], 100)
    })

    sample_model_results = {
        'linear_regression': {
            'r2_score': 0.65,
            'mae': 1200,
            'rmse': 1800,
            'mape': 15.2
        },
        'random_forest': {
            'r2_score': 0.78,
            'mae': 980,
            'rmse': 1450,
            'mape': 12.1
        }
    }

    sample_feature_importance = {
        'random_forest': {
            'M值': 0.65,
            'F值': 0.25,
            'R值': 0.10
        }
    }

    # 测试报告生成
    generator = ReportGenerator()
    generator.generate_html_report(sample_rfm, sample_model_results, sample_feature_importance,
                                 output_path='test_report.html')
    generator.generate_markdown_report(sample_rfm, sample_model_results, sample_feature_importance,
                                    output_path='test_report.md')

    print("✓ 报告生成器测试完成")