"""
用户分群分析模块

提供用户分群和交互效应分析功能，包括：
- 基于价值的用户分群
- RFM分群分析
- 行为模式分群
- 交互效应分析
- 自定义分群策略
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SegmentAnalyzer:
    """用户分群分析器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.segment_labels = {}
        self.clustering_model = None

    def value_based_segmentation(self,
                                data: pd.DataFrame,
                                value_col: str,
                                n_tiers: int = 3,
                                method: str = 'quantile',
                                custom_boundaries: Optional[List[float]] = None) -> Dict:
        """
        基于价值的用户分群

        Parameters:
        - n_tiers: 分层数量
        - method: 'quantile', 'equal_width', 'custom'
        - custom_boundaries: 自定义分界值列表
        """
        df = data.copy()

        # 确保价值列是数值类型
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

        if method == 'custom' and custom_boundaries:
            # 使用自定义边界
            boundaries = [float('-inf')] + custom_boundaries + [float('inf')]
            labels = [f'Tier_{i+1}' for i in range(len(boundaries)-1)]
            df['value_segment'] = pd.cut(df[value_col], bins=boundaries, labels=labels)

        elif method == 'quantile':
            # 使用分位数
            quantiles = np.linspace(0, 1, n_tiers + 1)
            boundaries = df[value_col].quantile(quantiles).tolist()
            labels = [f'Value_Tier_{i+1}' for i in range(n_tiers)]
            df['value_segment'] = pd.qcut(df[value_col], q=n_tiers, labels=labels, duplicates='drop')

        elif method == 'equal_width':
            # 等宽分箱
            labels = [f'Value_Tier_{i+1}' for i in range(n_tiers)]
            df['value_segment'] = pd.cut(df[value_col], bins=n_tiers, labels=labels)

        else:
            raise ValueError("method必须是'quantile', 'equal_width'或'custom'")

        # 计算分层统计
        segment_stats = {}
        for segment in df['value_segment'].cat.categories:
            segment_data = df[df['value_segment'] == segment]
            segment_stats[segment] = {
                'count': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100,
                'mean_value': segment_data[value_col].mean(),
                'median_value': segment_data[value_col].median(),
                'std_value': segment_data[value_col].std(),
                'min_value': segment_data[value_col].min(),
                'max_value': segment_data[value_col].max()
            }

        return {
            'segmented_data': df,
            'segment_statistics': segment_stats,
            'segmentation_method': method,
            'n_segments': len(df['value_segment'].cat.categories),
            'value_column': value_col
        }

    def rfm_segmentation(self,
                        data: pd.DataFrame,
                        customer_id_col: str,
                        recency_col: str,
                        frequency_col: str,
                        monetary_col: str,
                        n_clusters: int = 5) -> Dict:
        """
        RFM分群分析

        Parameters:
        - recency_col: 最近消费时间距今天数 (越小越好)
        - frequency_col: 消费频率
        - monetary_col: 消费金额
        """
        df = data.copy()

        # 确保数值列是数值类型
        for col in [recency_col, frequency_col, monetary_col]:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算RFM分数
        rf_data = df.groupby(customer_id_col)[[recency_col, frequency_col, monetary_col]].agg({
            recency_col: 'min',  # 最近消费时间
            frequency_col: 'sum',  # 总消费次数
            monetary_col: 'sum'   # 总消费金额
        }).reset_index()

        # 重命名列
        rf_data.columns = [customer_id_col, 'Recency', 'Frequency', 'Monetary']

        # RFM评分 (1-5分制，分数越高越好)
        rf_data['R_Score'] = pd.qcut(rf_data['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rf_data['F_Score'] = pd.qcut(rf_data['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rf_data['M_Score'] = pd.qcut(rf_data['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

        # 计算RFM总分
        rf_data['RFM_Score'] = rf_data['R_Score'].astype(str) + rf_data['F_Score'].astype(str) + rf_data['M_Score'].astype(str)
        rf_data['RFM_Total'] = rf_data['R_Score'].astype(int) + rf_data['F_Score'].astype(int) + rf_data['M_Score'].astype(int)

        # 聚类分析
        features = rf_data[['Recency', 'Frequency', 'Monetary']]
        features_scaled = self.scaler.fit_transform(features)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rf_data['Cluster'] = kmeans.fit_predict(features_scaled)

        # 计算轮廓系数
        silhouette_avg = silhouette_score(features_scaled, kmeans.labels_)

        # 聚类统计
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = rf_data[rf_data['Cluster'] == cluster]
            cluster_stats[f'Cluster_{cluster}'] = {
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(rf_data) * 100,
                'avg_recency': cluster_data['Recency'].mean(),
                'avg_frequency': cluster_data['Frequency'].mean(),
                'avg_monetary': cluster_data['Monetary'].mean(),
                'avg_rfm_total': cluster_data['RFM_Total'].mean()
            }

        # RFM客户类型定义
        def classify_rfm(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                return 'Champions'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Loyal Customers'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 2:
                return 'Potential Loyalists'
            elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
                return 'New Customers'
            elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
                return 'At Risk'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 3:
                return 'Cannot Lose Them'
            else:
                return 'Lost'

        rf_data['RFM_Segment'] = rf_data.apply(classify_rfm, axis=1)

        # 保存模型
        self.clustering_model = kmeans

        return {
            'rfm_data': rf_data,
            'cluster_statistics': cluster_stats,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'rfm_segments': rf_data['RFM_Segment'].value_counts().to_dict()
        }

    def behavioral_segmentation(self,
                               data: pd.DataFrame,
                               customer_id_col: str,
                               behavior_cols: List[str],
                               n_clusters: int = 4) -> Dict:
        """基于行为的用户分群"""
        df = data.copy()

        # 聚合用户行为数据
        behavior_data = df.groupby(customer_id_col)[behavior_cols].agg(['mean', 'sum', 'count']).reset_index()

        # 扁平化列名
        behavior_data.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in behavior_data.columns]

        # 确保数值列是数值类型
        numeric_cols = behavior_data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if behavior_data[col].dtype == 'object':
                behavior_data[col] = pd.to_numeric(behavior_data[col], errors='coerce')

        # 填充缺失值
        behavior_data[numeric_cols] = behavior_data[numeric_cols].fillna(0)

        # 标准化
        features = behavior_data[numeric_cols]
        features_scaled = self.scaler.fit_transform(features)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        behavior_data['Behavior_Cluster'] = kmeans.fit_predict(features_scaled)

        # 计算轮廓系数
        silhouette_avg = silhouette_score(features_scaled, kmeans.labels_)

        # 聚类统计
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = behavior_data[behavior_data['Behavior_Cluster'] == cluster]
            cluster_stats[f'Behavior_Cluster_{cluster}'] = {
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(behavior_data) * 100
            }

            # 为每个行为特征添加统计
            for col in numeric_cols:
                if col in cluster_data.columns:
                    cluster_stats[f'Behavior_Cluster_{cluster}'][f'avg_{col}'] = cluster_data[col].mean()

        # 保存模型
        self.clustering_model = kmeans

        return {
            'behavior_segmented_data': behavior_data,
            'cluster_statistics': cluster_stats,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'behavior_features': behavior_cols
        }

    def segment_conversion_analysis(self,
                                  data: pd.DataFrame,
                                  segment_col: str,
                                  group_col: str,
                                  conversion_col: str) -> Dict:
        """分群转化率分析"""
        df = data.copy()

        # 确保转化列是数值类型
        if df[conversion_col].dtype == 'object':
            df[conversion_col] = df[conversion_col].apply(lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0)

        # 计算每个分群的转化率
        results = {}
        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]
            segment_results = {}

            for group in segment_data[group_col].unique():
                group_data = segment_data[segment_data[group_col] == group]
                conversion_rate = group_data[conversion_col].mean()
                sample_size = len(group_data)
                conversions = group_data[conversion_col].sum()

                # 计算置信区间
                se = np.sqrt(conversion_rate * (1 - conversion_rate) / sample_size)
                ci_lower = max(0, conversion_rate - 1.96 * se)
                ci_upper = min(1, conversion_rate + 1.96 * se)

                segment_results[group] = {
                    'conversion_rate': conversion_rate,
                    'sample_size': sample_size,
                    'conversions': int(conversions),
                    'standard_error': se,
                    'confidence_interval': (ci_lower, ci_upper)
                }

            # 计算分群内提升率
            lift_results = {}
            groups = list(segment_results.keys())
            if len(groups) == 2:
                control_rate = segment_results[groups[0]]['conversion_rate']
                test_rate = segment_results[groups[1]]['conversion_rate']
                absolute_lift = test_rate - control_rate
                relative_lift = absolute_lift / control_rate if control_rate > 0 else 0

                lift_results = {
                    'absolute_lift': absolute_lift,
                    'relative_lift': relative_lift,
                    'lift_percentage': relative_lift * 100
                }

            results[segment] = {
                'group_results': segment_results,
                'lift_analysis': lift_results
            }

        # 总体统计
        overall_stats = {}
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            overall_stats[group] = {
                'conversion_rate': group_data[conversion_col].mean(),
                'sample_size': len(group_data),
                'conversions': int(group_data[conversion_col].sum())
            }

        return {
            'segment_conversion_analysis': results,
            'overall_statistics': overall_stats,
            'segment_column': segment_col,
            'group_column': group_col,
            'conversion_column': conversion_col
        }

    def interaction_analysis(self,
                            data: pd.DataFrame,
                            group_col: str,
                            segment_col: str,
                            outcome_col: str,
                            method: str = 'logistic') -> Dict:
        """交互效应分析"""
        df = data.copy()

        # 确保结果列是数值类型
        if df[outcome_col].dtype == 'object':
            df[outcome_col] = df[outcome_col].apply(lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0)

        results = {}

        # 分析每个分群内的组间差异
        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]

            # 检查组别数量
            groups = segment_data[group_col].unique()
            if len(groups) == 2:
                # 两组比较
                group1_data = segment_data[segment_data[group_col] == groups[0]][outcome_col]
                group2_data = segment_data[segment_data[group_col] == groups[1]][outcome_col]

                # t检验
                t_stat, t_p = stats.ttest_ind(group1_data, group2_data)

                # 卡方检验
                contingency = pd.crosstab(segment_data[group_col], segment_data[outcome_col])
                chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

                results[segment] = {
                    'group1': groups[0],
                    'group2': groups[1],
                    'group1_conversion': group1_data.mean(),
                    'group2_conversion': group2_data.mean(),
                    'absolute_difference': group2_data.mean() - group1_data.mean(),
                    'relative_difference': (group2_data.mean() - group1_data.mean()) / group1_data.mean() if group1_data.mean() > 0 else 0,
                    't_test_statistic': t_stat,
                    't_test_p_value': t_p,
                    'chi2_statistic': chi2_stat,
                    'chi2_p_value': chi2_p,
                    'is_significant_t': t_p < 0.05,
                    'is_significant_chi2': chi2_p < 0.05
                }
            else:
                # 多组比较 (ANOVA)
                group_outcomes = [segment_data[segment_data[group_col] == group][outcome_col] for group in groups]
                f_stat, f_p = stats.f_oneway(*group_outcomes)

                # 各组转化率
                conversion_rates = {}
                for group in groups:
                    group_data = segment_data[segment_data[group_col] == group]
                    conversion_rates[group] = group_data[outcome_col].mean()

                results[segment] = {
                    'groups': list(groups),
                    'conversion_rates': conversion_rates,
                    'f_statistic': f_stat,
                    'f_p_value': f_p,
                    'is_significant': f_p < 0.05
                }

        # 整体交互效应检验
        # 创建交互项并进行方差分析
        if method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder

            # 编码分类变量
            le_group = LabelEncoder()
            le_segment = LabelEncoder()

            df_encoded = df.copy()
            df_encoded[f'{group_col}_encoded'] = le_group.fit_transform(df[group_col])
            df_encoded[f'{segment_col}_encoded'] = le_segment.fit_transform(df[segment_col])

            # 创建交互项
            df_encoded['interaction'] = df_encoded[f'{group_col}_encoded'] * df_encoded[f'{segment_col}_encoded']

            # 逻辑回归
            X = df_encoded[[f'{group_col}_encoded', f'{segment_col}_encoded', 'interaction']]
            y = df_encoded[outcome_col]

            logreg = LogisticRegression()
            logreg.fit(X, y)

            # 计算交互效应的统计显著性
            interaction_coef = logreg.coef_[0][2]
            interaction_se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)))[2] * np.std(y) / np.sqrt(len(y))
            interaction_z = interaction_coef / interaction_se
            interaction_p = 2 * (1 - stats.norm.cdf(abs(interaction_z)))

            results['overall_interaction'] = {
                'interaction_coefficient': interaction_coef,
                'interaction_z_score': interaction_z,
                'interaction_p_value': interaction_p,
                'is_significant_interaction': interaction_p < 0.05
            }

        return {
            'segment_interactions': results,
            'group_column': group_col,
            'segment_column': segment_col,
            'outcome_column': outcome_col,
            'analysis_method': method
        }

    def apply_custom_segments(self,
                            data: pd.DataFrame,
                            segment_rules: Dict[str, Dict[str, Tuple[float, float]]]) -> pd.DataFrame:
        """
        应用自定义分群规则

        Parameters:
        - segment_rules: 分群规则字典
          {
              'segment_name': {'column_name': (min_value, max_value)}
          }
        """
        df = data.copy()
        df['custom_segment'] = 'Uncategorized'

        for segment_name, rules in segment_rules.items():
            mask = pd.Series(True, index=df.index)

            for column, (min_val, max_val) in rules.items():
                if column in df.columns:
                    # 确保数值类型
                    if df[column].dtype == 'object':
                        df[column] = pd.to_numeric(df[column], errors='coerce')

                    if min_val is not None and max_val is not None:
                        mask &= (df[column] >= min_val) & (df[column] < max_val)
                    elif min_val is not None:
                        mask &= df[column] >= min_val
                    elif max_val is not None:
                        mask &= df[column] < max_val

            df.loc[mask, 'custom_segment'] = segment_name

        return df

    def calculate_segment_insights(self,
                                  segmented_data: pd.DataFrame,
                                  segment_col: str,
                                  metrics_cols: List[str]) -> Dict:
        """计算分群洞察"""
        df = segmented_data.copy()

        # 确保数值列是数值类型
        for col in metrics_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        insights = {}

        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]
            insights[segment] = {
                'count': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100
            }

            # 为每个指标计算统计
            for metric in metrics_cols:
                if metric in segment_data.columns:
                    insights[segment][f'{metric}_mean'] = segment_data[metric].mean()
                    insights[segment][f'{metric}_median'] = segment_data[metric].median()
                    insights[segment][f'{metric}_std'] = segment_data[metric].std()
                    insights[segment][f'{metric}_min'] = segment_data[metric].min()
                    insights[segment][f'{metric}_max'] = segment_data[metric].max()

        return {
            'segment_insights': insights,
            'segment_column': segment_col,
            'metrics_analyzed': metrics_cols,
            'total_segments': len(df[segment_col].unique())
        }