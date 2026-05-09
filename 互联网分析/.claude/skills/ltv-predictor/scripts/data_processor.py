#!/usr/bin/env python3
"""
数据预处理器和RFM特征工程模块
基于第3课核心算法实现RFM分析和数据预处理功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    数据预处理器

    专门处理电商订单数据，进行RFM特征工程和数据预处理
    支持多种数据格式和时间窗口配置
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据处理器

        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            'date_column': '消费日期',
            'customer_column': '用户码',
            'quantity_column': '数量',
            'price_column': '单价',
            'order_id_column': '订单号',
            'product_column': '产品码',
            'city_column': '城市',
            'feature_period_months': 3,
            'prediction_period_months': 12,
            'min_orders_per_customer': 1,
            'remove_outliers': True,
            'outlier_threshold': 3.0
        }

        # 更新配置
        if config:
            self.config.update(config)

        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.rfm_data = None
        self.data_quality_report = {}

    def load_order_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        加载订单数据

        Args:
            file_path: 文件路径
            **kwargs: pandas.read_csv的额外参数

        Returns:
            加载的订单数据
        """
        try:
            # 尝试不同的编码格式
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']

            for encoding in encodings:
                try:
                    self.raw_data = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    print(f"✓ 数据加载成功: {self.raw_data.shape}")
                    print(f"  - 使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法解码文件，请检查文件编码")

            # 数据质量检查
            self._validate_data()
            self._generate_data_quality_report()

            return self.raw_data

        except Exception as e:
            raise ValueError(f"数据加载失败: {str(e)}")

    def _validate_data(self):
        """验证数据格式和必需字段"""
        required_columns = [
            self.config['date_column'],
            self.config['customer_column'],
            self.config['quantity_column'],
            self.config['price_column']
        ]

        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]

        if missing_columns:
            raise ValueError(f"缺少必需字段: {missing_columns}")

        print(f"✓ 数据验证通过，包含必需字段: {required_columns}")

    def _generate_data_quality_report(self):
        """生成数据质量报告"""
        df = self.raw_data

        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_orders': df[self.config['order_id_column']].nunique() if self.config['order_id_column'] in df.columns else 'Unknown',
            'total_customers': df[self.config['customer_column']].nunique(),
            'total_products': df[self.config['product_column']].nunique() if self.config['product_column'] in df.columns else 'Unknown',
            'date_range': self._get_date_range(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_orders': df.duplicated(subset=[self.config['order_id_column']]).sum() if self.config['order_id_column'] in df.columns else 'Unknown'
        }

        self.data_quality_report = report

        # 打印质量报告
        print(f"📊 数据质量报告:")
        print(f"  - 总记录数: {report['total_rows']:,}")
        print(f"  - 总客户数: {report['total_customers']:,}")
        print(f"  - 时间范围: {report['date_range'][0]} ~ {report['date_range'][1]}")
        print(f"  - 缺失值: {sum(val for val in report['missing_values'].values())}")

    def _get_date_range(self) -> Tuple[str, str]:
        """获取数据时间范围"""
        try:
            dates = pd.to_datetime(self.raw_data[self.config['date_column']])
            return (dates.min().strftime('%Y-%m-%d'),
                   dates.max().strftime('%Y-%m-%d'))
        except:
            return ('Unknown', 'Unknown')

    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        预处理订单数据

        Args:
            data: 输入数据，如果为None则使用self.raw_data

        Returns:
            预处理后的数据
        """
        if data is None:
            data = self.raw_data.copy()

        print("🧹 开始数据预处理...")

        # 1. 创建总价字段
        if '总价' not in data.columns:
            data['总价'] = data[self.config['quantity_column']] * data[self.config['price_column']]
            print("  ✓ 计算总价")

        # 2. 转换日期格式
        data[self.config['date_column']] = pd.to_datetime(data[self.config['date_column']])
        print("  ✓ 转换日期格式")

        # 3. 过滤异常数据
        if self.config['remove_outliers']:
            data = self._remove_outliers(data)
            print("  ✓ 移除异常值")

        # 4. 筛选活跃客户
        min_orders = self.config['min_orders_per_customer']
        customer_order_counts = data[self.config['customer_column']].value_counts()
        active_customers = customer_order_counts[customer_order_counts >= min_orders].index
        data = data[data[self.config['customer_column']].isin(active_customers)]
        print(f"  ✓ 筛选活跃客户 (≥{min_orders}订单): {len(active_customers)}个客户")

        # 5. 数据排序
        data = data.sort_values([self.config['customer_column'], self.config['date_column']])

        self.processed_data = data
        print(f"✓ 数据预处理完成: {data.shape}")

        return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """移除异常值"""
        threshold = self.config['outlier_threshold']

        # 移除价格异常值
        price_mean = data[self.config['price_column']].mean()
        price_std = data[self.config['price_column']].std()
        price_outliers = np.abs(data[self.config['price_column']] - price_mean) > threshold * price_std

        # 移除数量异常值
        qty_mean = data[self.config['quantity_column']].mean()
        qty_std = data[self.config['quantity_column']].std()
        qty_outliers = np.abs(data[self.config['quantity_column']] - qty_mean) > threshold * qty_std

        # 移除总价异常值
        total_mean = data['总价'].mean()
        total_std = data['总价'].std()
        total_outliers = np.abs(data['总价'] - total_mean) > threshold * total_std

        # 组合异常值条件
        outlier_mask = price_outliers | qty_outliers | total_outliers
        clean_data = data[~outlier_mask]

        removed_count = len(data) - len(clean_data)
        if removed_count > 0:
            print(f"    移除异常值: {removed_count} 条记录")

        return clean_data

    def calculate_rfm_features(self,
                             data: Optional[pd.DataFrame] = None,
                             feature_period_months: Optional[int] = None,
                             prediction_period_months: Optional[int] = None) -> pd.DataFrame:
        """
        计算RFM特征

        Args:
            data: 输入数据
            feature_period_months: 特征计算时间窗口（月）
            prediction_period_months: 预测时间窗口（月）

        Returns:
            包含RFM特征和LTV标签的数据
        """
        if data is None:
            data = self.processed_data
        else:
            # 如果传入的是原始数据，需要先预处理
            if self.processed_data is None or not data.equals(self.processed_data):
                data = self.preprocess_data(data)

        if feature_period_months is None:
            feature_period_months = self.config['feature_period_months']
        if prediction_period_months is None:
            prediction_period_months = self.config['prediction_period_months']

        print(f"🔍 开始RFM特征计算...")
        print(f"  - 特征计算期: {feature_period_months}个月")
        print(f"  - 预测期: {prediction_period_months}个月")

        # 确定数据时间范围
        data_sorted = data.sort_values(self.config['date_column'])
        start_date = data_sorted[self.config['date_column']].min()
        # 确保start_date是Timestamp类型
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        feature_end_date = start_date + pd.DateOffset(months=feature_period_months)
        prediction_end_date = start_date + pd.DateOffset(months=prediction_period_months)

        print(f"  - 特征计算期: {start_date.strftime('%Y-%m-%d')} ~ {feature_end_date.strftime('%Y-%m-%d')}")
        print(f"  - 完整预测期: {start_date.strftime('%Y-%m-%d')} ~ {prediction_end_date.strftime('%Y-%m-%d')}")

        # 特征计算期数据
        feature_data = data[
            (data[self.config['date_column']] > start_date) &
            (data[self.config['date_column']] <= feature_end_date)
        ].copy()

        # 完整数据用于计算LTV
        full_data = data[
            (data[self.config['date_column']] > start_date) &
            (data[self.config['date_column']] <= prediction_end_date)
        ].copy()

        # 获取独立客户列表
        unique_customers = feature_data[self.config['customer_column']].unique()
        print(f"  - 活跃客户数: {len(unique_customers)}")

        # 初始化RFM数据框
        rfm_data = pd.DataFrame({
            self.config['customer_column']: unique_customers
        })

        # 计算R值 (Recency - 最近一次消费距期末天数)
        print("  计算R值 (最近消费时间间隔)...")
        r_data = feature_data.groupby(self.config['customer_column'])[self.config['date_column']].max().reset_index()
        r_data.columns = [self.config['customer_column'], '最近购买日期']
        r_data['R值'] = (feature_end_date - r_data['最近购买日期']).dt.days
        rfm_data = rfm_data.merge(r_data[[self.config['customer_column'], 'R值']],
                                 on=self.config['customer_column'], how='left')

        # 计算F值 (Frequency - 消费频率)
        print("  计算F值 (消费频率)...")
        f_data = feature_data.groupby(self.config['customer_column'])[self.config['date_column']].count().reset_index()
        f_data.columns = [self.config['customer_column'], 'F值']
        rfm_data = rfm_data.merge(f_data, on=self.config['customer_column'], how='left')

        # 计算M值 (Monetary - 消费金额)
        print("  计算M值 (消费金额)...")
        m_data = feature_data.groupby(self.config['customer_column'])['总价'].sum().reset_index()
        m_data.columns = [self.config['customer_column'], 'M值']
        rfm_data = rfm_data.merge(m_data, on=self.config['customer_column'], how='left')

        # 计算年度LTV (目标变量)
        print("  计算年度LTV (目标变量)...")
        ltv_data = full_data.groupby(self.config['customer_column'])['总价'].sum().reset_index()
        ltv_data.columns = [self.config['customer_column'], '年度LTV']
        rfm_data = rfm_data.merge(ltv_data, on=self.config['customer_column'], how='left')

        # 处理缺失值
        rfm_data['年度LTV'] = rfm_data['年度LTV'].fillna(0)

        # 添加RFM分析信息
        self._add_rfm_insights(rfm_data)

        self.rfm_data = rfm_data
        print(f"✓ RFM特征计算完成: {rfm_data.shape}")

        return rfm_data

    def _add_rfm_insights(self, rfm_data: pd.DataFrame):
        """添加RFM分析洞察"""
        # RFM分位数分析 - 处理边界情况
        try:
            rfm_data['R_分位数'] = pd.qcut(rfm_data['R值'], q=4, labels=['D', 'C', 'B', 'A'], duplicates='drop')
        except ValueError:
            # 如果qcut失败，使用cut作为备选方案
            rfm_data['R_分位数'] = pd.cut(rfm_data['R值'], bins=4, labels=['D', 'C', 'B', 'A'], include_lowest=True)

        try:
            rfm_data['F_分位数'] = pd.qcut(rfm_data['F值'], q=4, labels=['A', 'B', 'C', 'D'], duplicates='drop')
        except ValueError:
            rfm_data['F_分位数'] = pd.cut(rfm_data['F值'], bins=4, labels=['A', 'B', 'C', 'D'], include_lowest=True)

        try:
            rfm_data['M_分位数'] = pd.qcut(rfm_data['M值'], q=4, labels=['A', 'B', 'C', 'D'], duplicates='drop')
        except ValueError:
            rfm_data['M_分位数'] = pd.cut(rfm_data['M值'], bins=4, labels=['A', 'B', 'C', 'D'], include_lowest=True)

        # RFM组合分群
        rfm_data['RFM_分群'] = rfm_data['R_分位数'].astype(str) + rfm_data['F_分位数'].astype(str) + rfm_data['M_分位数'].astype(str)

        # 计算RFM得分
        rfm_data['RFM_得分'] = (
            rfm_data['R值'].rank(ascending=False) * 0.2 +
            rfm_data['F值'].rank() * 0.3 +
            rfm_data['M值'].rank() * 0.5
        )

    def segment_customers(self, rfm_data: Optional[pd.DataFrame] = None, n_segments: int = 5) -> pd.DataFrame:
        """
        客户分群

        Args:
            rfm_data: RFM数据
            n_segments: 分群数量

        Returns:
            包含客户分群的数据
        """
        if rfm_data is None:
            rfm_data = self.rfm_data

        # 基于RFM得分进行分群
        rfm_data['客户价值分层'] = pd.qcut(
            rfm_data['RFM_得分'],
            q=n_segments,
            labels=['铜牌客户', '银牌客户', '金牌客户', '白金客户', '钻石客户']
        )

        # 计算各层级统计信息
        segment_stats = rfm_data.groupby('客户价值分层').agg({
            self.config['customer_column']: 'count',
            '年度LTV': ['mean', 'sum'],
            'R值': 'mean',
            'F值': 'mean',
            'M值': 'mean'
        }).round(2)

        print("📊 客户价值分层统计:")
        print(segment_stats)

        return rfm_data

    def get_rfm_summary(self, rfm_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        获取RFM分析摘要

        Args:
            rfm_data: RFM数据

        Returns:
            RFM分析摘要字典
        """
        if rfm_data is None:
            rfm_data = self.rfm_data

        if rfm_data is None:
            return {"error": "RFM数据未计算，请先运行calculate_rfm_features"}

        summary = {
            'total_customers': len(rfm_data),
            'date_range': self._get_date_range(),
            'rfm_statistics': {
                'R值': {
                    'mean': rfm_data['R值'].mean(),
                    'median': rfm_data['R值'].median(),
                    'std': rfm_data['R值'].std(),
                    'min': rfm_data['R值'].min(),
                    'max': rfm_data['R值'].max()
                },
                'F值': {
                    'mean': rfm_data['F值'].mean(),
                    'median': rfm_data['F值'].median(),
                    'std': rfm_data['F值'].std(),
                    'min': rfm_data['F值'].min(),
                    'max': rfm_data['F值'].max()
                },
                'M值': {
                    'mean': rfm_data['M值'].mean(),
                    'median': rfm_data['M值'].median(),
                    'std': rfm_data['M值'].std(),
                    'min': rfm_data['M值'].min(),
                    'max': rfm_data['M值'].max()
                },
                '年度LTV': {
                    'mean': rfm_data['年度LTV'].mean(),
                    'median': rfm_data['年度LTV'].median(),
                    'std': rfm_data['年度LTV'].std(),
                    'min': rfm_data['年度LTV'].min(),
                    'max': rfm_data['年度LTV'].max()
                }
            },
            'high_value_customers': {
                'top_10_percent_threshold': rfm_data['年度LTV'].quantile(0.9),
                'count': len(rfm_data[rfm_data['年度LTV'] >= rfm_data['年度LTV'].quantile(0.9)])
            }
        }

        return summary

    def export_rfm_data(self, rfm_data: pd.DataFrame, output_path: str, format: str = 'csv'):
        """
        导出RFM数据

        Args:
            rfm_data: RFM数据
            output_path: 输出路径
            format: 输出格式 ('csv', 'excel')
        """
        try:
            if format.lower() == 'csv':
                rfm_data.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif format.lower() == 'excel':
                rfm_data.to_excel(output_path, index=False)
            else:
                raise ValueError("不支持的格式，请使用 'csv' 或 'excel'")

            print(f"✓ RFM数据已导出: {output_path}")

        except Exception as e:
            print(f"❌ 导出失败: {str(e)}")

# 便利函数
def quick_rfm_analysis(file_path: str,
                      feature_period_months: int = 3,
                      prediction_period_months: int = 12,
                      output_dir: str = './rfm_results') -> Dict:
    """
    快速RFM分析

    Args:
        file_path: 订单数据文件路径
        feature_period_months: 特征计算期（月）
        prediction_period_months: 预测期（月）
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    import os
    from pathlib import Path

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化处理器
    processor = DataProcessor({
        'feature_period_months': feature_period_months,
        'prediction_period_months': prediction_period_months
    })

    # 加载和预处理数据
    data = processor.load_order_data(file_path)
    processed_data = processor.preprocess_data(data)

    # 计算RFM特征
    rfm_data = processor.calculate_rfm_features(processed_data)

    # 客户分群
    segmented_data = processor.segment_customers(rfm_data)

    # 获取摘要
    summary = processor.get_rfm_summary(segmented_data)

    # 导出结果
    rfm_output_path = os.path.join(output_dir, 'rfm_features.csv')
    processor.export_rfm_data(segmented_data, rfm_output_path)

    return {
        'rfm_data': segmented_data,
        'summary': summary,
        'processor': processor,
        'output_paths': {
            'rfm_features': rfm_output_path
        }
    }

if __name__ == "__main__":
    # 示例使用
    print("🔧 数据处理器测试")

    # 如果有示例数据文件，可以进行测试
    sample_file = '../data/sample_orders.csv'
    if os.path.exists(sample_file):
        results = quick_rfm_analysis(sample_file)
        print("✓ 快速RFM分析完成")
    else:
        print("⚠️ 示例数据文件不存在，跳过测试")