#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFM Customer Segmentation Core Analysis Engine
Core algorithms for RFM analysis and customer segmentation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RFMAnalyzer:
    """Core RFM analysis engine for customer segmentation"""

    def __init__(self, chinese_font='SimHei'):
        """Initialize the analyzer with Chinese font support"""
        self.chinese_font = chinese_font
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False
        self.scaler = StandardScaler()

    def load_and_clean_data(self, file_path):
        """
        Load and clean e-commerce transaction data

        Args:
            file_path (str): Path to CSV file containing order data

        Returns:
            tuple: (original_df, cleaned_df) with original and cleaned data
        """
        # Load data
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk')
            except:
                df = pd.read_csv(file_path, encoding='latin-1')

        original_df = df.copy()

        # Data cleaning
        print(f"原始数据: {len(df)} 条订单")

        # Remove invalid orders
        cleaned_df = df[(df['数量'] > 0) & (df['单价'] > 0)].copy()
        removed_count = len(df) - len(cleaned_df)

        if removed_count > 0:
            print(f"已清理 {removed_count} 条无效订单 (负数量或零价格)")

        # Add calculated fields
        cleaned_df['订单金额'] = cleaned_df['数量'] * cleaned_df['单价']
        cleaned_df['消费日期'] = pd.to_datetime(cleaned_df['消费日期'])

        print(f"清理后数据: {len(cleaned_df)} 条订单")
        print(f"独立客户数: {cleaned_df['用户码'].nunique()}")

        return original_df, cleaned_df

    def calculate_rfm_metrics(self, df):
        """
        Calculate RFM metrics for each customer

        Args:
            df (pd.DataFrame): Cleaned transaction data

        Returns:
            pd.DataFrame: RFM metrics for each customer
        """
        print("计算RFM指标...")

        # Calculate RFM metrics
        rfm_df = df.groupby('用户码').agg({
            '消费日期': 'max',     # Recency: most recent purchase
            '订单号': 'count',     # Frequency: number of orders
            '订单金额': 'sum'      # Monetary: total spending
        }).reset_index()

        rfm_df.columns = ['用户码', '最近购买日期', '购买频率', '消费金额']

        # Calculate Recency (days since last purchase)
        max_date = rfm_df['最近购买日期'].max()
        rfm_df['最近性'] = (max_date - rfm_df['最近购买日期']).dt.days

        # Rename for consistency
        rfm_df = rfm_df.rename(columns={
            '购买频率': '频率性',
            '消费金额': '金额性'
        })

        # Select RFM columns
        rfm_metrics = rfm_df[['用户码', '最近性', '频率性', '金额性']].copy()

        print(f"RFM指标计算完成，覆盖 {len(rfm_metrics)} 位客户")

        return rfm_metrics

    def find_optimal_clusters(self, data, max_k=10):
        """
        Find optimal number of clusters using elbow method

        Args:
            data (pd.DataFrame): RFM metrics data
            max_k (int): Maximum number of clusters to test

        Returns:
            int: Optimal number of clusters
        """
        print("寻找最优聚类数...")

        # Prepare data for clustering
        rfm_values = data[['最近性', '频率性', '金额性']].values
        rfm_scaled = self.scaler.fit_transform(rfm_values)

        # Calculate inertia for different k values
        inertias = []
        k_range = range(1, min(max_k + 1, len(data) // 2))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(rfm_scaled)
            inertias.append(kmeans.inertia_)

        # Find elbow point (simplified method)
        if len(inertias) >= 3:
            # Calculate percentage decrease
            decreases = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            percentages = [decreases[i] / inertias[i] * 100 for i in range(len(decreases))]

            # Find point where decrease becomes minimal (< 20%)
            optimal_k = 3  # default
            for i, pct in enumerate(percentages):
                if pct < 20:
                    optimal_k = i + 2  # +2 because indices start from k=2
                    break
        else:
            optimal_k = 3

        print(f"选择最优聚类数: {optimal_k}")
        return optimal_k

    def apply_clustering(self, rfm_data, n_clusters=None):
        """
        Apply K-means clustering to RFM data

        Args:
            rfm_data (pd.DataFrame): RFM metrics data
            n_clusters (int, optional): Number of clusters. If None, auto-determine

        Returns:
            pd.DataFrame: RFM data with cluster assignments
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(rfm_data)

        print(f"应用K-means聚类 (k={n_clusters})...")

        # Prepare data
        rfm_values = rfm_data[['最近性', '频率性', '金额性']].values
        rfm_scaled = self.scaler.fit_transform(rfm_values)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(rfm_scaled)

        # Add cluster labels to data
        result_df = rfm_data.copy()
        result_df['cluster'] = cluster_labels

        # Order clusters by monetary value (descending)
        cluster_monetary = result_df.groupby('cluster')['金额性'].mean().sort_values(ascending=False)
        cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(cluster_monetary.index)}

        result_df['cluster'] = result_df['cluster'].map(cluster_mapping)

        print(f"聚类完成，生成 {n_clusters} 个客户分群")

        return result_df

    def calculate_rfm_scores(self, rfm_df):
        """
        Calculate RFM scores (1-3 scale) for each customer

        Args:
            rfm_df (pd.DataFrame): RFM metrics with clusters

        Returns:
            pd.DataFrame: RFM data with scores and value segments
        """
        print("计算RFM评分...")

        result_df = rfm_df.copy()

        # Calculate quartiles for scoring (using 33/66 percentiles for 1-3 scale)
        def calculate_scores(series):
            """Calculate 1-3 scale scores based on 33/66 percentiles"""
            p33, p66 = series.quantile([0.33, 0.66])
            scores = pd.cut(series,
                          bins=[-np.inf, p33, p66, np.inf],
                          labels=[1, 2, 3],
                          include_lowest=True)
            return scores.astype(int)

        # Calculate individual scores
        result_df['R_评分'] = calculate_scores(result_df['最近性'])  # Lower recency = higher score
        result_df['R_评分'] = 4 - result_df['R_评分']  # Reverse for proper scoring

        result_df['F_评分'] = calculate_scores(result_df['频率性'])
        result_df['M_评分'] = calculate_scores(result_df['金额性'])

        # Calculate total score
        result_df['RFM总分'] = result_df['R_评分'] + result_df['F_评分'] + result_df['M_评分']

        # Assign customer value segments
        def assign_value_segment(total_score):
            if total_score >= 7:
                return 'High'
            elif total_score >= 5:
                return 'Medium'
            else:
                return 'Low'

        result_df['客户价值'] = result_df['RFM总分'].apply(assign_value_segment)

        print("RFM评分计算完成")
        return result_df

    def generate_segment_summary(self, final_df):
        """
        Generate summary statistics for each customer segment

        Args:
            final_df (pd.DataFrame): Complete RFM analysis results

        Returns:
            dict: Summary statistics for each segment
        """
        print("生成分群摘要...")

        summary = {}

        for segment in ['High', 'Medium', 'Low']:
            segment_data = final_df[final_df['客户价值'] == segment]

            if len(segment_data) > 0:
                summary[segment] = {
                    '客户数量': len(segment_data),
                    '客户占比': f"{len(segment_data) / len(final_df) * 100:.1f}%",
                    '平均最近购买天数': segment_data['最近性'].mean(),
                    '平均购买频次': segment_data['频率性'].mean(),
                    '平均消费金额': segment_data['金额性'].mean(),
                    '总消费金额': segment_data['金额性'].sum(),
                    '收入贡献占比': f"{segment_data['金额性'].sum() / final_df['金额性'].sum() * 100:.1f}%",
                    '平均RFM总分': segment_data['RFM总分'].mean()
                }

        return summary

    def run_complete_analysis(self, file_path, n_clusters=None):
        """
        Run complete RFM analysis pipeline

        Args:
            file_path (str): Path to transaction data file
            n_clusters (int, optional): Number of clusters. Auto-determined if None

        Returns:
            pd.DataFrame: Complete analysis results
        """
        print("开始RFM客户分群分析")
        print("=" * 50)

        # 1. Load and clean data
        original_df, cleaned_df = self.load_and_clean_data(file_path)

        # 2. Calculate RFM metrics
        rfm_metrics = self.calculate_rfm_metrics(cleaned_df)

        # 3. Apply clustering
        clustered_df = self.apply_clustering(rfm_metrics, n_clusters)

        # 4. Calculate RFM scores and segments
        final_df = self.calculate_rfm_scores(clustered_df)

        # 5. Generate summary
        summary = self.generate_segment_summary(final_df)

        # 6. Display results
        print("\n分析完成！")
        print("\n客户分群摘要:")
        for segment, stats in summary.items():
            print(f"\n{segment} 价值客户:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        return final_df, summary

def main():
    """Example usage of RFM analyzer"""
    analyzer = RFMAnalyzer()

    # Example file path - would be provided by user
    file_path = "电商历史订单.csv"

    if pd.io.common.file_exists(file_path):
        results, summary = analyzer.run_complete_analysis(file_path)

        # Save results
        results.to_csv('rfm_analysis_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: rfm_analysis_results.csv")
    else:
        print(f"数据文件未找到: {file_path}")
        print("请确保提供正确的电商订单数据文件路径")

if __name__ == "__main__":
    main()