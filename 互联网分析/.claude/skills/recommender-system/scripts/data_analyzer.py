"""
推荐系统数据分析器

提供全面的数据分析功能：
- 用户行为分析
- 商品热度分析
- 数据质量检查
- 冷启动问题检测
- 用户画像分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class DataAnalyzer:
    """推荐系统数据分析器类"""

    def __init__(self):
        self.user_data = None
        self.item_data = None
        self.user_item_matrix = None
        self.analysis_results = {}

    def load_data(self, user_behavior_path: str, item_info_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        加载用户行为数据和商品信息数据

        Args:
            user_behavior_path: 用户行为数据文件路径
            item_info_path: 商品信息数据文件路径（可选）

        Returns:
            用户行为数据和商品信息数据的元组
        """
        try:
            # 加载用户行为数据
            self.user_data = pd.read_csv(user_behavior_path, encoding='utf-8')

            # 数据预处理
            if '评分' in self.user_data.columns:
                self.user_data['评分'] = pd.to_numeric(self.user_data['评分'], errors='coerce')
                self.user_data.dropna(subset=['评分'], inplace=True)

            # 处理时间戳
            if '时间戳' in self.user_data.columns:
                self.user_data['时间戳'] = pd.to_datetime(self.user_data['时间戳'])

            print(f"用户行为数据加载成功：{len(self.user_data)} 条记录")

            # 加载商品信息数据（如果提供）
            self.item_data = None
            if item_info_path:
                self.item_data = pd.read_csv(item_info_path, encoding='utf-8')
                print(f"商品信息数据加载成功：{len(self.item_data)} 个商品")

            return self.user_data, self.item_data

        except Exception as e:
            print(f"数据加载失败：{str(e)}")
            return None, None

    def analyze_user_behavior(self, user_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        分析用户行为

        Args:
            user_data: 用户行为数据（可选，默认使用self.user_data）

        Returns:
            用户行为分析结果字典
        """
        if user_data is None:
            user_data = self.user_data

        if user_data is None:
            print("没有可用的用户数据")
            return {}

        print("开始分析用户行为...")

        analysis = {}

        # 基本统计
        analysis['total_users'] = user_data['用户ID'].nunique()
        analysis['total_items'] = user_data['商品ID'].nunique()
        analysis['total_interactions'] = len(user_data)

        # 用户活跃度分析
        user_activity = user_data.groupby('用户ID').size().describe()
        analysis['user_activity'] = {
            'mean': float(user_activity['mean']),
            'std': float(user_activity['std']),
            'min': int(user_activity['min']),
            'max': int(user_activity['max']),
            'median': float(user_activity['50%'])
        }

        # 活跃用户分布
        activity_quantiles = user_activity.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        analysis['activity_distribution'] = {
            f'{int(p*100)}%_quantile': float(val) for p, val in activity_quantiles.items()
        }

        # 评分分析（如果有评分数据）
        if '评分' in user_data.columns:
            rating_stats = user_data['评分'].describe()
            analysis['rating_stats'] = {
                'mean': float(rating_stats['mean']),
                'std': float(rating_stats['std']),
                'min': float(rating_stats['min']),
                'max': float(rating_stats['max']),
                'median': float(rating_stats['50%'])
            }

            # 评分分布
            rating_dist = user_data['评分'].value_counts().to_dict()
            analysis['rating_distribution'] = {str(k): int(v) for k, v in rating_dist.items()}

        # 时间分析（如果有时间戳数据）
        if '时间戳' in user_data.columns:
            # 确保时间戳是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(user_data['时间戳']):
                user_data['时间戳'] = pd.to_datetime(user_data['时间戳'])

            time_span = user_data['时间戳'].max() - user_data['时间戳'].min()
            analysis['time_analysis'] = {
                'time_span_days': time_span.days,
                'start_date': user_data['时间戳'].min().strftime('%Y-%m-%d'),
                'end_date': user_data['时间戳'].max().strftime('%Y-%m-%d'),
                'avg_interactions_per_day': float(len(user_data) / max(1, time_span.days))
            }

            # 每日活跃用户数
            daily_active_users = user_data.groupby(user_data['时间戳'].dt.date)['用户ID'].nunique()
            analysis['daily_active_users'] = {
                'mean': float(daily_active_users.mean()),
                'std': float(daily_active_users.std()),
                'min': int(daily_active_users.min()),
                'max': int(daily_active_users.max())
            }

        # 商品类别分析
        if '商品类别' in user_data.columns:
            category_stats = user_data['商品类别'].value_counts().describe()
            analysis['category_stats'] = {
                'total_categories': int(len(user_data['商品类别'].unique())),
                'most_popular_category': user_data['商品类别'].value_counts().index[0],
                'most_popular_category_count': int(user_data['商品类别'].value_counts().iloc[0])
            }

        self.analysis_results['user_behavior'] = analysis
        print("用户行为分析完成")

        return analysis

    def analyze_item_popularity(self, item_data: Optional[pd.DataFrame] = None,
                              user_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        分析商品热度

        Args:
            item_data: 商品信息数据（可选）
            user_data: 用户行为数据（可选）

        Returns:
            商品热度分析结果字典
        """
        if user_data is None:
            user_data = self.user_data
        if item_data is None:
            item_data = self.item_data

        if user_data is None:
            print("没有可用的用户行为数据")
            return {}

        print("开始分析商品热度...")

        analysis = {}

        # 商品交互统计
        item_interactions = user_data.groupby('商品ID').size().describe()
        analysis['item_interactions'] = {
            'mean': float(item_interactions['mean']),
            'std': float(item_interactions['std']),
            'min': int(item_interactions['min']),
            'max': int(item_interactions['max']),
            'median': float(item_interactions['50%'])
        }

        # 热门商品
        item_popularity = user_data.groupby('商品ID').size().sort_values(ascending=False)
        top_items = item_popularity.head(20)
        analysis['top_items'] = {
            str(item_id): int(count) for item_id, count in top_items.items()
        }

        # 长尾分析
        interaction_quantiles = item_popularity.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        analysis['popularity_distribution'] = {
            f'{int(p*100)}%_quantile': float(val) for p, val in interaction_quantiles.items()
        }

        # 冷门商品（交互次数少于5次）
        cold_items = (item_popularity < 5).sum()
        analysis['cold_start_items'] = {
            'count': int(cold_items),
            'percentage': float(cold_items / len(item_popularity) * 100)
        }

        # 评分分析（如果有评分数据）
        if '评分' in user_data.columns:
            item_ratings = user_data.groupby('商品ID')['评分'].agg(['mean', 'count', 'std'])
            analysis['item_ratings'] = {
                'avg_rating': float(item_ratings['mean'].mean()),
                'avg_rating_std': float(item_ratings['mean'].std()),
                'high_rated_items': int((item_ratings['mean'] >= 4).sum()),
                'low_rated_items': int((item_ratings['mean'] <= 2).sum())
            }

        # 商品类别分析
        if '商品类别' in user_data.columns:
            category_popularity = user_data.groupby('商品类别').agg({
                '商品ID': 'nunique',
                '用户ID': 'nunique'
            }).rename(columns={'商品ID': 'unique_items', '用户ID': 'unique_users'})

            # 添加总交互次数
            category_interactions = user_data.groupby('商品类别').size().rename('total_interactions')
            category_popularity = category_popularity.join(category_interactions)

            analysis['category_popularity'] = category_popularity.to_dict('index')

        self.analysis_results['item_popularity'] = analysis
        print("商品热度分析完成")

        return analysis

    def calculate_sparsity(self, user_item_matrix: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        计算用户-商品矩阵的稀疏度

        Args:
            user_item_matrix: 用户-商品交互矩阵（可选）

        Returns:
            稀疏度分析结果字典
        """
        if user_item_matrix is None:
            if self.user_item_matrix is None:
                # 从用户数据构建矩阵
                if self.user_data is None:
                    print("没有可用的数据")
                    return {}
                self.user_item_matrix = self.user_data.pivot_table(
                    index='用户ID', columns='商品ID', values='评分', fill_value=0
                )
            user_item_matrix = self.user_item_matrix

        total_entries = user_item_matrix.shape[0] * user_item_matrix.shape[1]
        zero_entries = (user_item_matrix == 0).sum().sum()
        non_zero_entries = total_entries - zero_entries

        sparsity = {
            'sparsity_ratio': float(zero_entries / total_entries),
            'density_ratio': float(non_zero_entries / total_entries),
            'total_entries': int(total_entries),
            'zero_entries': int(zero_entries),
            'non_zero_entries': int(non_zero_entries),
            'avg_interactions_per_user': float(non_zero_entries / user_item_matrix.shape[0]),
            'avg_interactions_per_item': float(non_zero_entries / user_item_matrix.shape[1])
        }

        self.analysis_results['sparsity'] = sparsity
        print(f"数据稀疏度分析完成：稀疏度 {sparsity['sparsity_ratio']:.2%}")

        return sparsity

    def detect_cold_start(self, user_data: Optional[pd.DataFrame] = None,
                         item_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        检测冷启动问题

        Args:
            user_data: 用户行为数据（可选）
            item_data: 商品信息数据（可选）

        Returns:
            冷启动问题分析结果字典
        """
        if user_data is None:
            user_data = self.user_data
        if item_data is None:
            item_data = self.item_data

        if user_data is None:
            print("没有可用的用户数据")
            return {}

        print("开始检测冷启动问题...")

        analysis = {}

        # 用户冷启动分析
        user_interactions = user_data.groupby('用户ID').size()
        new_users = (user_interactions <= 5).sum()
        total_users = len(user_interactions)

        analysis['user_cold_start'] = {
            'new_users_count': int(new_users),
            'new_users_percentage': float(new_users / total_users * 100),
            'total_users': int(total_users),
            'avg_interactions_per_new_user': float(user_interactions[user_interactions <= 5].mean())
        }

        # 商品冷启动分析
        item_interactions = user_data.groupby('商品ID').size()
        new_items = (item_interactions <= 5).sum()
        total_items = len(item_interactions)

        analysis['item_cold_start'] = {
            'new_items_count': int(new_items),
            'new_items_percentage': float(new_items / total_items * 100),
            'total_items': int(total_items),
            'avg_interactions_per_new_item': float(item_interactions[item_interactions <= 5].mean())
        }

        # 时间维度冷启动分析（如果有时间戳）
        if '时间戳' in user_data.columns:
            # 最近30天的新用户和新商品
            recent_date = user_data['时间戳'].max() - timedelta(days=30)
            recent_data = user_data[user_data['时间戳'] >= recent_date]

            recent_users = recent_data['用户ID'].nunique()
            recent_items = recent_data['商品ID'].nunique()

            # 计算这些新用户和新商品的历史交互次数
            new_user_historical = user_interactions[user_interactions.index.isin(recent_data['用户ID'])]
            new_item_historical = item_interactions[item_interactions.index.isin(recent_data['商品ID'])]

            analysis['temporal_cold_start'] = {
                'recent_new_users': int(recent_users),
                'recent_new_items': int(recent_items),
                'recent_new_users_with_few_interactions': int((new_user_historical <= 3).sum()),
                'recent_new_items_with_few_interactions': int((new_item_historical <= 3).sum())
            }

        # 冷启动严重程度评估
        cold_start_severity = "低"
        if analysis['user_cold_start']['new_users_percentage'] > 30:
            cold_start_severity = "高"
        elif analysis['user_cold_start']['new_users_percentage'] > 15:
            cold_start_severity = "中"

        analysis['cold_start_severity'] = cold_start_severity

        self.analysis_results['cold_start'] = analysis
        print(f"冷启动问题检测完成：严重程度 {cold_start_severity}")

        return analysis

    def analyze_user_profiling(self, user_data: Optional[pd.DataFrame] = None,
                             item_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        用户画像分析

        Args:
            user_data: 用户行为数据（可选）
            item_data: 商品信息数据（可选）

        Returns:
            用户画像分析结果字典
        """
        if user_data is None:
            user_data = self.user_data
        if item_data is None:
            item_data = self.item_data

        if user_data is None:
            print("没有可用的用户数据")
            return {}

        print("开始用户画像分析...")

        analysis = {}

        # 用户价值分析（RFM）
        if '时间戳' in user_data.columns and '评分' in user_data.columns:
            # Recency: 最近一次交互时间
            recency = user_data.groupby('用户ID')['时间戳'].max().apply(lambda x: (user_data['时间戳'].max() - x).days)

            # Frequency: 交互频率
            frequency = user_data.groupby('用户ID').size()

            # Monetary: 平均评分（作为价值指标）
            monetary = user_data.groupby('用户ID')['评分'].mean()

            # RFM分层
            analysis['rfm_analysis'] = {
                'recency_stats': {
                    'mean_days': float(recency.mean()),
                    'median_days': float(recency.median()),
                    'min_days': int(recency.min()),
                    'max_days': int(recency.max())
                },
                'frequency_stats': {
                    'mean_interactions': float(frequency.mean()),
                    'median_interactions': float(frequency.median()),
                    'min_interactions': int(frequency.min()),
                    'max_interactions': int(frequency.max())
                },
                'monetary_stats': {
                    'mean_rating': float(monetary.mean()),
                    'median_rating': float(monetary.median()),
                    'min_rating': float(monetary.min()),
                    'max_rating': float(monetary.max())
                }
            }

            # 用户分层（基于RFM）
            def classify_user(r, f, m):
                if r <= recency.quantile(0.33) and f >= frequency.quantile(0.67) and m >= monetary.quantile(0.67):
                    return "高价值用户"
                elif r <= recency.quantile(0.67) and f >= frequency.quantile(0.33):
                    return "价值用户"
                elif r <= recency.quantile(0.67):
                    return "活跃用户"
                elif f <= frequency.quantile(0.33) and r > recency.quantile(0.67):
                    return "流失风险用户"
                else:
                    return "普通用户"

            user_segments = pd.concat([recency, frequency, monetary], axis=1)
            user_segments.columns = ['recency', 'frequency', 'monetary']
            user_segments['segment'] = user_segments.apply(lambda x: classify_user(x['recency'], x['frequency'], x['monetary']), axis=1)

            segment_counts = user_segments['segment'].value_counts()
            analysis['user_segments'] = {
                str(segment): int(count) for segment, count in segment_counts.items()
            }

        # 偏好分析
        if '商品类别' in user_data.columns:
            # 用户类别偏好
            user_category_preference = user_data.groupby(['用户ID', '商品类别']).size().unstack(fill_value=0)
            analysis['category_preference'] = {
                'total_categories': int(len(user_data['商品类别'].unique())),
                'avg_categories_per_user': float((user_category_preference > 0).sum(axis=1).mean()),
                'most_diverse_user': int((user_category_preference > 0).sum(axis=1).max()),
                'least_diverse_user': int((user_category_preference > 0).sum(axis=1).min())
            }

            # 用户忠诚度分析（重复购买同一类别）
            category_loyalty = user_data.groupby('用户ID')['商品类别'].apply(lambda x: x.value_counts().iloc[0] / len(x))
            analysis['loyalty_analysis'] = {
                'avg_loyalty_score': float(category_loyalty.mean()),
                'high_loyalty_users': int((category_loyalty >= 0.8).sum()),
                'medium_loyalty_users': int((category_loyalty >= 0.5).sum() - (category_loyalty >= 0.8).sum()),
                'low_loyalty_users': int((category_loyalty < 0.5).sum())
            }

        # 人口统计学分析（如果有相关信息）
        demographic_fields = ['用户年龄', '用户性别', '用户城市']
        available_demographics = [field for field in demographic_fields if field in user_data.columns]

        if available_demographics:
            analysis['demographics'] = {}

            if '用户年龄' in user_data.columns:
                age_stats = user_data['用户年龄'].describe()
                analysis['demographics']['age'] = {
                    'mean': float(age_stats['mean']),
                    'median': float(age_stats['50%']),
                    'min': int(age_stats['min']),
                    'max': int(age_stats['max'])
                }

            if '用户性别' in user_data.columns:
                gender_dist = user_data['用户性别'].value_counts().to_dict()
                analysis['demographics']['gender'] = {str(k): int(v) for k, v in gender_dist.items()}

            if '用户城市' in user_data.columns:
                city_stats = user_data['用户城市'].value_counts().describe()
                analysis['demographics']['cities'] = {
                    'total_cities': int(len(user_data['用户城市'].unique())),
                    'most_active_city': user_data['用户城市'].value_counts().index[0],
                    'most_active_city_users': int(user_data['用户城市'].value_counts().iloc[0])
                }

        self.analysis_results['user_profiling'] = analysis
        print("用户画像分析完成")

        return analysis

    def generate_data_quality_report(self, user_data: Optional[pd.DataFrame] = None,
                                   item_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        生成数据质量报告

        Args:
            user_data: 用户行为数据（可选）
            item_data: 商品信息数据（可选）

        Returns:
            数据质量报告字典
        """
        if user_data is None:
            user_data = self.user_data
        if item_data is None:
            item_data = self.item_data

        if user_data is None:
            print("没有可用的用户数据")
            return {}

        print("开始生成数据质量报告...")

        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_data_quality': {},
            'item_data_quality': {}
        }

        # 用户数据质量检查
        if user_data is not None:
            user_quality = {
                'total_records': int(len(user_data)),
                'missing_values': user_data.isnull().sum().to_dict(),
                'duplicate_records': int(user_data.duplicated().sum()),
                'data_types': user_data.dtypes.astype(str).to_dict()
            }

            # 检查异常值
            if '评分' in user_data.columns:
                rating_range = user_data['评分'].describe()
                user_quality['rating_anomalies'] = {
                    'min_rating': float(rating_range['min']),
                    'max_rating': float(rating_range['max']),
                    'out_of_range_count': int(((user_data['评分'] < 1) | (user_data['评分'] > 5)).sum())
                }

            report['user_data_quality'] = user_quality

        # 商品数据质量检查
        if item_data is not None:
            item_quality = {
                'total_items': int(len(item_data)),
                'missing_values': item_data.isnull().sum().to_dict(),
                'duplicate_items': int(item_data.duplicated().sum()),
                'data_types': item_data.dtypes.astype(str).to_dict()
            }

            report['item_data_quality'] = item_quality

        # 数据一致性检查
        if user_data is not None and item_data is not None:
            user_items = set(user_data['商品ID'].unique())
            catalog_items = set(item_data['商品ID'].unique())

            consistency = {
                'items_in_behavior_not_in_catalog': int(len(user_items - catalog_items)),
                'items_in_catalog_not_in_behavior': int(len(catalog_items - user_items)),
                'common_items': int(len(user_items & catalog_items))
            }

            report['data_consistency'] = consistency

        # 数据覆盖度分析
        if user_data is not None:
            coverage = {
                'user_coverage': float(len(user_data['用户ID'].unique())),
                'item_coverage': float(len(user_data['商品ID'].unique())),
                'interaction_coverage': float(len(user_data))
            }

            report['data_coverage'] = coverage

        self.analysis_results['data_quality'] = report
        print("数据质量报告生成完成")

        return report

    def save_analysis_results(self, file_path: str, format: str = 'json') -> None:
        """
        保存分析结果

        Args:
            file_path: 保存路径
            format: 保存格式 ('json', 'csv', 'excel')
        """
        if format == 'json':
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)

        elif format == 'csv':
            # 将结果扁平化并保存为CSV
            flat_data = {}
            for category, results in self.analysis_results.items():
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                flat_data[f"{category}_{key}_{sub_key}"] = sub_value
                        else:
                            flat_data[f"{category}_{key}"] = value
                else:
                    flat_data[category] = results

            df = pd.DataFrame([flat_data])
            df.to_csv(file_path, index=False, encoding='utf-8')

        print(f"分析结果已保存到: {file_path}")

    def get_analysis_summary(self) -> str:
        """
        获取分析结果摘要

        Returns:
            分析结果摘要文本
        """
        if not self.analysis_results:
            return "暂无分析结果"

        summary = "# 数据分析结果摘要\n\n"

        # 用户行为摘要
        if 'user_behavior' in self.analysis_results:
            ub = self.analysis_results['user_behavior']
            summary += f"## 用户行为分析\n"
            summary += f"- 总用户数: {ub.get('total_users', 'N/A'):,}\n"
            summary += f"- 总商品数: {ub.get('total_items', 'N/A'):,}\n"
            summary += f"- 总交互次数: {ub.get('total_interactions', 'N/A'):,}\n"
            if 'user_activity' in ub:
                summary += f"- 平均用户活跃度: {ub['user_activity'].get('mean', 'N/A'):.1f}\n"
            summary += "\n"

        # 商品热度摘要
        if 'item_popularity' in self.analysis_results:
            ip = self.analysis_results['item_popularity']
            summary += f"## 商品热度分析\n"
            if 'cold_start_items' in ip:
                summary += f"- 冷门商品比例: {ip['cold_start_items'].get('percentage', 'N/A'):.1f}%\n"
            summary += "\n"

        # 数据稀疏度摘要
        if 'sparsity' in self.analysis_results:
            sp = self.analysis_results['sparsity']
            summary += f"## 数据稀疏度\n"
            summary += f"- 稀疏度: {sp.get('sparsity_ratio', 'N/A'):.2%}\n"
            summary += f"- 密度: {sp.get('density_ratio', 'N/A'):.2%}\n"
            summary += "\n"

        # 冷启动问题摘要
        if 'cold_start' in self.analysis_results:
            cs = self.analysis_results['cold_start']
            summary += f"## 冷启动问题\n"
            summary += f"- 严重程度: {cs.get('cold_start_severity', 'N/A')}\n"
            if 'user_cold_start' in cs:
                summary += f"- 新用户比例: {cs['user_cold_start'].get('new_users_percentage', 'N/A'):.1f}%\n"
            if 'item_cold_start' in cs:
                summary += f"- 新商品比例: {cs['item_cold_start'].get('new_items_percentage', 'N/A'):.1f}%\n"
            summary += "\n"

        # 用户画像摘要
        if 'user_profiling' in self.analysis_results:
            up = self.analysis_results['user_profiling']
            summary += f"## 用户画像分析\n"
            if 'user_segments' in up:
                summary += "- 用户分层:\n"
                for segment, count in up['user_segments'].items():
                    summary += f"  - {segment}: {count:,} 用户\n"
            summary += "\n"

        return summary