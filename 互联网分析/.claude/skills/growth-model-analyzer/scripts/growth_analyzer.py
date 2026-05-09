"""
增长模型分析器 - 核心增长分析模块

提供全面的增长黑客分析功能，包括：
- 裂变策略效果评估
- 用户细分和RFM分析
- 转化率分析和统计检验
- 增长策略效果对比
- 用户画像和行为分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class GrowthModelAnalyzer:
    """增长模型分析器 - 核心分析引擎"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化增长分析器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.data = None
        self.results = {}

    def load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        加载增长分析数据

        Parameters:
        - data_path: 数据文件路径
        - **kwargs: pandas.read_csv的额外参数

        Returns:
        - 加载的数据DataFrame
        """
        try:
            if data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path, **kwargs)
            elif data_path.endswith('.xlsx'):
                self.data = pd.read_excel(data_path, **kwargs)
            else:
                raise ValueError("支持的文件格式: .csv, .xlsx")

            print(f"✅ 数据加载成功: {len(self.data)} 条记录")
            return self.data

        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return None

    def data_quality_check(self, data: pd.DataFrame = None,
                          required_cols: List[str] = None) -> Dict:
        """
        数据质量检查

        Parameters:
        - data: 待检查的数据
        - required_cols: 必需的列名列表

        Returns:
        - 质量检查报告
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'duplicate_rows': data.duplicated().sum(),
            'missing_values': {},
            'data_types': data.dtypes.to_dict(),
            'required_columns_check': True
        }

        # 检查缺失值
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(data) * 100
                }

        # 检查必需列
        if required_cols:
            missing_required = [col for col in required_cols if col not in data.columns]
            if missing_required:
                quality_report['required_columns_check'] = False
                quality_report['missing_required_columns'] = missing_required

        return quality_report

    def analyze_campaign_effectiveness(self,
                                     data: pd.DataFrame = None,
                                     campaign_col: str = '裂变类型',
                                     conversion_col: str = '是否转化',
                                     control_group: Optional[str] = None) -> Dict:
        """
        分析营销活动效果

        Parameters:
        - data: 分析数据
        - campaign_col: 活动类型列名
        - conversion_col: 转化结果列名
        - control_group: 对照组名称

        Returns:
        - 效果分析结果
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        results = {}

        # 1. 基础统计
        campaign_stats = data.groupby(campaign_col)[conversion_col].agg([
            'count', 'sum', 'mean', 'std'
        ]).round(4)

        campaign_stats.columns = ['用户数', '转化数', '转化率', '标准差']
        campaign_stats['转化数'] = campaign_stats['转化数'].astype(int)

        results['campaign_statistics'] = campaign_stats.to_dict()

        # 2. 转化率提升分析
        if control_group and control_group in campaign_stats.index:
            control_rate = campaign_stats.loc[control_group, '转化率']

            lift_analysis = {}
            for campaign in campaign_stats.index:
                if campaign != control_group:
                    campaign_rate = campaign_stats.loc[campaign, '转化率']
                    absolute_lift = campaign_rate - control_rate
                    relative_lift = absolute_lift / control_rate if control_rate > 0 else 0

                    lift_analysis[campaign] = {
                        'absolute_lift': absolute_lift,
                        'relative_lift': relative_lift,
                        'lift_percentage': relative_lift * 100,
                        'control_rate': control_rate,
                        'campaign_rate': campaign_rate
                    }

            results['lift_analysis'] = lift_analysis

        # 3. 统计显著性检验
        contingency_table = pd.crosstab(data[campaign_col], data[conversion_col])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        results['statistical_test'] = {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < 0.05
        }

        # 4. 效应量计算 (Cramer's V)
        n = contingency_table.sum().sum()
        cramer_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

        results['effect_size'] = {
            'cramers_v': cramer_v,
            'interpretation': self._interpret_cramers_v(cramer_v)
        }

        self.results['campaign_effectiveness'] = results
        return results

    def rfm_segmentation(self,
                        data: pd.DataFrame = None,
                        user_col: str = '用户码',
                        recency_col: str = 'R值',
                        frequency_col: str = 'F值',
                        monetary_col: str = 'M值',
                        n_clusters: int = 5) -> Dict:
        """
        RFM用户分群分析

        Parameters:
        - data: 分析数据
        - user_col: 用户标识列名
        - recency_col: 近度指标列名
        - frequency_col: 频度指标列名
        - monetary_col: 金额指标列名
        - n_clusters: 聚类数量

        Returns:
        - RFM分群结果
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        # 准备RFM数据
        rfm_data = data.groupby(user_col)[[recency_col, frequency_col, monetary_col]].agg({
            recency_col: 'min',      # R值越小越好 (最近消费时间)
            frequency_col: 'sum',     # F值越大越好 (消费频次)
            monetary_col: 'sum'       # M值越大越好 (消费金额)
        }).reset_index()

        rfm_data.columns = [user_col, 'Recency', 'Frequency', 'Monetary']

        # RFM评分 (1-5分制)
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

        # RFM总分
        rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)
        rfm_data['RFM_Total'] = rfm_data['R_Score'].astype(int) + rfm_data['F_Score'].astype(int) + rfm_data['M_Score'].astype(int)

        # K-means聚类
        features = rfm_data[['Recency', 'Frequency', 'Monetary']]
        features_scaled = self.scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rfm_data['Cluster'] = kmeans.fit_predict(features_scaled)

        # 计算轮廓系数
        silhouette_avg = silhouette_score(features_scaled, kmeans.labels_)

        # 聚类统计
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
            cluster_stats[f'Cluster_{cluster}'] = {
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(rfm_data) * 100,
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

        rfm_data['RFM_Segment'] = rfm_data.apply(classify_rfm, axis=1)

        results = {
            'rfm_data': rfm_data,
            'cluster_statistics': cluster_stats,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'rfm_segments': rfm_data['RFM_Segment'].value_counts().to_dict(),
            'segment_analysis': rfm_data.groupby('RFM_Segment')[['Recency', 'Frequency', 'Monetary']].mean().to_dict()
        }

        self.results['rfm_segmentation'] = results
        return results

    def user_behavior_analysis(self,
                             data: pd.DataFrame = None,
                             user_col: str = '用户码',
                             behavior_cols: List[str] = None) -> Dict:
        """
        用户行为分析

        Parameters:
        - data: 分析数据
        - user_col: 用户标识列名
        - behavior_cols: 行为指标列名列表

        Returns:
        - 用户行为分析结果
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        if behavior_cols is None:
            behavior_cols = ['曾助力', '曾拼团', '曾推荐']

        # 过滤存在的列
        existing_cols = [col for col in behavior_cols if col in data.columns]

        # 用户行为统计
        behavior_stats = {}
        for col in existing_cols:
            behavior_stats[col] = {
                'total_users': len(data),
                'behavior_users': data[col].sum(),
                'behavior_rate': data[col].sum() / len(data),
                'non_behavior_users': len(data) - data[col].sum()
            }

        # 行为组合分析
        if len(existing_cols) >= 2:
            behavior_combinations = data.groupby(existing_cols)[user_col].count().reset_index()
            behavior_combinations.columns = existing_cols + ['user_count']
            behavior_combinations['percentage'] = behavior_combinations['user_count'] / len(data)

            results = {
                'individual_behavior': behavior_stats,
                'behavior_combinations': behavior_combinations.to_dict(),
                'behavior_insights': self._analyze_behavior_insights(behavior_stats, behavior_combinations)
            }
        else:
            results = {
                'individual_behavior': behavior_stats
            }

        self.results['user_behavior_analysis'] = results
        return results

    def conversion_funnel_analysis(self,
                                 data: pd.DataFrame = None,
                                 funnel_stages: List[str] = None) -> Dict:
        """
        转化漏斗分析

        Parameters:
        - data: 分析数据
        - funnel_stages: 漏斗阶段列名列表

        Returns:
        - 转化漏斗分析结果
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        if funnel_stages is None:
            funnel_stages = ['曾助力', '曾拼团', '是否转化']

        # 过滤存在的列
        existing_stages = [stage for stage in funnel_stages if stage in data.columns]

        funnel_data = []
        for i, stage in enumerate(existing_stages):
            stage_name = self._get_stage_name(stage)
            users_at_stage = data[stage].sum() if i > 0 else len(data)
            conversion_rate = users_at_stage / len(data)

            funnel_data.append({
                'stage': stage_name,
                'stage_column': stage,
                'users': users_at_stage,
                'conversion_rate': conversion_rate,
                'dropoff_rate': 1 - conversion_rate if i > 0 else 0
            })

        # 计算阶段间转化率
        for i in range(1, len(funnel_data)):
            prev_users = funnel_data[i-1]['users']
            current_users = funnel_data[i]['users']
            stage_conversion = current_users / prev_users if prev_users > 0 else 0
            funnel_data[i]['stage_conversion_rate'] = stage_conversion
            funnel_data[i]['stage_dropoff_rate'] = 1 - stage_conversion

        results = {
            'funnel_data': funnel_data,
            'overall_conversion_rate': funnel_data[-1]['conversion_rate'] if funnel_data else 0,
            'total_users': len(data),
            'insights': self._analyze_funnel_insights(funnel_data)
        }

        self.results['conversion_funnel'] = results
        return results

    def cohort_analysis(self,
                       data: pd.DataFrame = None,
                       user_col: str = '用户码',
                       time_col: str = 'R值',
                       conversion_col: str = '是否转化') -> Dict:
        """
        队列分析

        Parameters:
        - data: 分析数据
        - user_col: 用户标识列名
        - time_col: 时间指标列名
        - conversion_col: 转化结果列名

        Returns:
        - 队列分析结果
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        # 创建时间队列 (使用R值作为时间分组)
        data_copy = data.copy()
        data_copy['time_group'] = pd.qcut(data_copy[time_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # 计算各队列的转化率
        cohort_conversion = data_copy.groupby('time_group')[conversion_col].agg([
            'count', 'sum', 'mean'
        ]).round(4)

        cohort_conversion.columns = ['total_users', 'conversions', 'conversion_rate']
        cohort_conversion['conversions'] = cohort_conversion['conversions'].astype(int)

        results = {
            'cohort_conversion': cohort_conversion.to_dict(),
            'overall_conversion_rate': data[conversion_col].mean(),
            'cohort_insights': self._analyze_cohort_insights(cohort_conversion)
        }

        self.results['cohort_analysis'] = results
        return results

    def generate_insights_report(self) -> Dict:
        """
        生成洞察报告

        Returns:
        - 综合洞察报告
        """
        if not self.results:
            return {"error": "请先进行分析以生成结果"}

        insights = {
            "summary": {},
            "key_findings": [],
            "recommendations": [],
            "data_quality": "良好"
        }

        # 策略效果洞察
        if 'campaign_effectiveness' in self.results:
            campaign_results = self.results['campaign_effectiveness']
            if campaign_results['statistical_test']['is_significant']:
                insights["key_findings"].append("营销策略对用户转化有显著影响")

                # 找到最佳策略
                if 'lift_analysis' in campaign_results:
                    best_campaign = max(
                        campaign_results['lift_analysis'].items(),
                        key=lambda x: x[1]['lift_percentage']
                    )
                    insights["key_findings"].append(
                        f"{best_campaign[0]}策略效果最佳，提升{best_campaign[1]['lift_percentage']:.1f}%"
                    )
                    insights["recommendations"].append(
                        f"建议优先推广{best_campaign[0]}策略"
                    )

        # RFM分群洞察
        if 'rfm_segmentation' in self.results:
            rfm_results = self.results['rfm_segmentation']
            top_segment = max(rfm_results['segment_analysis'].items(),
                            key=lambda x: x[1]['Monetary'])
            insights["key_findings"].append(
                f"最有价值用户群体是{top_segment[0]}，平均消费金额为{top_segment[1]['Monetary']:.2f}"
            )
            insights["recommendations"].append(
                "针对高价值用户群体提供个性化服务和权益"
            )

        # 转化漏斗洞察
        if 'conversion_funnel' in self.results:
            funnel_results = self.results['conversion_funnel']
            if funnel_results['overall_conversion_rate'] < 0.1:
                insights["recommendations"].append("整体转化率偏低，建议优化用户旅程")

            # 找到最大流失环节
            funnel_data = funnel_results['funnel_data']
            if len(funnel_data) > 1:
                max_dropoff = max(funnel_data[1:], key=lambda x: x.get('stage_dropoff_rate', 0))
                insights["key_findings"].append(
                    f"最大流失环节是{max_dropoff['stage']}，流失率{max_dropoff.get('stage_dropoff_rate', 0)*100:.1f}%"
                )

        return insights

    def _interpret_cramers_v(self, v: float) -> str:
        """解释Cramer's V效应量"""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"

    def _get_stage_name(self, stage_col: str) -> str:
        """获取漏斗阶段的中文名称"""
        stage_names = {
            '曾助力': '助力行为',
            '曾拼团': '拼团行为',
            '曾推荐': '推荐行为',
            '是否转化': '最终转化'
        }
        return stage_names.get(stage_col, stage_col)

    def _analyze_behavior_insights(self, behavior_stats: Dict, behavior_combinations: pd.DataFrame) -> List[str]:
        """分析用户行为洞察"""
        insights = []

        # 找到最常见的行为
        most_common = max(behavior_stats.items(), key=lambda x: x[1]['behavior_rate'])
        insights.append(f"最常见的行为是{most_common[0]}，参与率{most_common[1]['behavior_rate']:.1%}")

        # 分析行为重叠
        if len(behavior_combinations) > 0:
            multi_behavior = behavior_combinations[
                behavior_combinations[[col for col in behavior_combinations.columns if col in behavior_stats]].sum(axis=1) > 1
            ]
            if len(multi_behavior) > 0:
                multi_behavior_rate = multi_behavior['user_count'].sum() / len(behavior_combinations)
                insights.append(f"多行为用户占比{multi_behavior_rate:.1%}")

        return insights

    def _analyze_funnel_insights(self, funnel_data: List[Dict]) -> List[str]:
        """分析转化漏斗洞察"""
        insights = []

        if len(funnel_data) > 1:
            # 找到最大流失环节
            max_dropoff_stage = max(funnel_data[1:], key=lambda x: x.get('stage_dropoff_rate', 0))
            dropoff_rate = max_dropoff_stage.get('stage_dropoff_rate', 0)
            insights.append(f"最大流失环节是{max_dropoff_stage['stage']}，流失率{dropoff_rate*100:.1f}%")

        # 整体转化率评估
        overall_rate = funnel_data[-1]['conversion_rate'] if funnel_data else 0
        if overall_rate > 0.2:
            insights.append("整体转化率表现良好")
        elif overall_rate > 0.1:
            insights.append("整体转化率有提升空间")
        else:
            insights.append("整体转化率偏低，需要重点优化")

        return insights

    def _analyze_cohort_insights(self, cohort_conversion: pd.DataFrame) -> List[str]:
        """分析队列洞察"""
        insights = []

        # 找到表现最好的队列
        best_cohort = cohort_conversion.loc[cohort_conversion['conversion_rate'].idxmax()]
        insights.append(f"表现最好的队列是{best_cohort.name}，转化率{best_cohort['conversion_rate']:.1%}")

        # 分析队列间差异
        conversion_rates = cohort_conversion['conversion_rate']
        if conversion_rates.max() - conversion_rates.min() > 0.1:
            insights.append("不同队列间转化率差异较大，建议针对性优化")

        return insights