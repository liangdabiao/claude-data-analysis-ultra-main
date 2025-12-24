"""
数据探索分析器 (EDA Analyzer) - 自动化探索性数据分析核心模块

提供全面的EDA功能，包括：
- 自动数据质量检查
- 统计描述分析
- 异常值检测
- 相关性分析
- 数据分布分析
- 智能洞察生成
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson, jarque_bera
import json
from datetime import datetime

warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """数据探索分析器 - 自动化EDA核心引擎"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化EDA分析器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.data = None
        self.results = {}
        self.report = None

        # 默认配置
        self.default_config = {
            'missing_threshold': 0.05,  # 缺失值阈值
            'outlier_method': 'iqr',    # 异常值检测方法
            'correlation_method': 'pearson',  # 相关性计算方法
            'significance_level': 0.05,  # 显著性水平
            'sample_size_threshold': 10000,  # 大数据集阈值
            'encoding_detection': True,  # 自动编码检测
            'chinese_support': True     # 中文支持
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

    def load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        加载数据文件

        Parameters:
        - data_path: 数据文件路径
        - **kwargs: pandas.read_csv的额外参数

        Returns:
        - 加载的DataFrame
        """
        try:
            # 自动检测文件格式
            if data_path.endswith('.csv'):
                # 自动检测编码
                if self.config['encoding_detection']:
                    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
                    for encoding in encodings:
                        try:
                            data = pd.read_csv(data_path, encoding=encoding, **kwargs)
                            print(f"✅ 数据加载成功，使用编码: {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("无法检测到正确的文件编码")
                else:
                    data = pd.read_csv(data_path, **kwargs)

            elif data_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(data_path, **kwargs)
                print("✅ Excel文件加载成功")

            elif data_path.endswith('.json'):
                data = pd.read_json(data_path, **kwargs)
                print("✅ JSON文件加载成功")

            else:
                raise ValueError("不支持的文件格式，请使用CSV、Excel或JSON格式")

            self.data = data
            self._log_basic_info()

            return data

        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            raise

    def _log_basic_info(self):
        """记录数据基本信息"""
        if self.data is not None:
            print(f"📊 数据基本信息:")
            print(f"   - 数据形状: {self.data.shape}")
            print(f"   - 内存使用: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"   - 列数: {len(self.data.columns)}")
            print(f"   - 数据类型分布:\n{self.data.dtypes.value_counts()}")

    def data_quality_check(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        数据质量检查

        Parameters:
        - data: 待检查的数据

        Returns:
        - 数据质量报告
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        quality_report = {
            'basic_info': {
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
                'total_cells': data.shape[0] * data.shape[1]
            },
            'missing_values': {},
            'data_types': {},
            'duplicates': {},
            'outliers': {},
            'data_quality_score': 0,
            'issues': [],
            'recommendations': []
        }

        # 1. 缺失值分析
        missing_analysis = self._analyze_missing_values(data)
        quality_report['missing_values'] = missing_analysis

        # 2. 数据类型分析
        dtype_analysis = self._analyze_data_types(data)
        quality_report['data_types'] = dtype_analysis

        # 3. 重复值分析
        duplicate_analysis = self._analyze_duplicates(data)
        quality_report['duplicates'] = duplicate_analysis

        # 4. 异常值分析（仅数值列）
        outlier_analysis = self._analyze_outliers(data)
        quality_report['outliers'] = outlier_analysis

        # 5. 计算数据质量分数
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['data_quality_score'] = quality_score

        # 6. 生成问题和建议
        issues, recommendations = self._generate_quality_recommendations(quality_report)
        quality_report['issues'] = issues
        quality_report['recommendations'] = recommendations

        self.results['data_quality'] = quality_report
        return quality_report

    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        """分析缺失值"""
        missing_info = {}

        # 总体缺失值情况
        total_missing = data.isnull().sum().sum()
        total_cells = data.shape[0] * data.shape[1]
        overall_missing_rate = total_missing / total_cells

        missing_info['overall'] = {
            'total_missing': total_missing,
            'total_cells': total_cells,
            'missing_rate': overall_missing_rate,
            'quality_rating': self._rate_missing_quality(overall_missing_rate)
        }

        # 各列缺失值详情
        missing_columns = {}
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_rate = missing_count / len(data)
                missing_columns[col] = {
                    'missing_count': missing_count,
                    'missing_rate': missing_rate,
                    'severity': self._classify_missing_severity(missing_rate)
                }

        missing_info['columns'] = missing_columns
        missing_info['missing_columns_count'] = len(missing_columns)

        # 缺失值模式分析
        missing_patterns = self._analyze_missing_patterns(data)
        missing_info['patterns'] = missing_patterns

        return missing_info

    def _analyze_data_types(self, data: pd.DataFrame) -> Dict:
        """分析数据类型"""
        dtype_info = {
            'type_distribution': data.dtypes.value_counts().to_dict(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(data.select_dtypes(include=['datetime64']).columns),
            'type_issues': []
        }

        # 检测可能的类型问题
        for col in data.columns:
            if data[col].dtype == 'object':
                # 检查是否应该是数值类型
                try:
                    pd.to_numeric(data[col], errors='raise')
                    dtype_info['type_issues'].append({
                        'column': col,
                        'issue': '可能应该是数值类型',
                        'suggestion': '尝试转换为数值类型'
                    })
                except:
                    pass

                # 检查是否应该是日期类型
                if data[col].dtype == 'object':
                    sample_values = data[col].dropna().head(5).astype(str)
                    if any('/' in val or '-' in val for val in sample_values if val):
                        try:
                            pd.to_datetime(data[col], errors='raise')
                            dtype_info['type_issues'].append({
                                'column': col,
                                'issue': '可能应该是日期类型',
                                'suggestion': '尝试转换为日期类型'
                            })
                        except:
                            pass

        return dtype_info

    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict:
        """分析重复值"""
        duplicate_info = {}

        # 完全重复行
        duplicate_rows = data.duplicated().sum()
        duplicate_info['exact_duplicates'] = {
            'count': duplicate_rows,
            'rate': duplicate_rows / len(data),
            'severity': self._classify_duplicate_severity(duplicate_rows / len(data))
        }

        # 基于关键字段的重复（如果存在ID列）
        id_columns = [col for col in data.columns if any(keyword in col.lower()
                      for keyword in ['id', '编号', 'code', '标识'])]

        if id_columns:
            for id_col in id_columns[:3]:  # 最多检查3个ID列
                id_duplicates = data[id_col].duplicated().sum()
                duplicate_info[f'{id_col}_duplicates'] = {
                    'count': id_duplicates,
                    'rate': id_duplicates / len(data),
                    'severity': self._classify_duplicate_severity(id_duplicates / len(data))
                }

        return duplicate_info

    def _analyze_outliers(self, data: pd.DataFrame) -> Dict:
        """分析异常值"""
        outlier_info = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        method = self.config['outlier_method']

        for col in numeric_columns:
            outliers = self._detect_outliers(data[col], method)
            if len(outliers) > 0:
                outlier_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_rate': len(outliers) / len(data),
                    'outlier_indices': outliers.tolist(),
                    'severity': self._classify_outlier_severity(len(outliers) / len(data))
                }

        outlier_info['total_outlier_columns'] = len(outlier_info)
        outlier_info['detection_method'] = method

        return outlier_info

    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> np.ndarray:
        """检测异常值"""
        series_clean = series.dropna()

        if method == 'iqr':
            Q1 = series_clean.quantile(0.25)
            Q3 = series_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series_clean))
            outliers = series_clean[z_scores > 3]

        elif method == 'modified_zscore':
            median = series_clean.median()
            mad = np.median(np.abs(series_clean - median))
            modified_z_scores = 0.6745 * (series_clean - median) / mad
            outliers = series_clean[np.abs(modified_z_scores) > 3.5]

        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")

        return outliers.index.values

    def generate_statistical_summary(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        生成统计描述摘要

        Parameters:
        - data: 待分析的数据

        Returns:
        - 统计摘要
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        summary = {
            'dataset_info': {
                'shape': data.shape,
                'columns_count': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
            },
            'numeric_summary': {},
            'categorical_summary': {},
            'distribution_tests': {},
            'correlation_analysis': {}
        }

        # 数值列统计摘要
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = data[numeric_cols].describe()

            # 添加额外的统计指标
            for col in numeric_cols:
                if data[col].dtype in [np.number]:
                    extra_stats = {
                        'skewness': stats.skew(data[col].dropna()),
                        'kurtosis': stats.kurtosis(data[col].dropna()),
                        'cv': stats.variation(data[col].dropna()),  # 变异系数
                        'missing_rate': data[col].isnull().sum() / len(data),
                        'unique_count': data[col].nunique()
                    }

                    # 正态性检验（如果样本量合适）
                    if len(data[col].dropna()) >= 8 and len(data[col].dropna()) <= 5000:
                        try:
                            # Shapiro-Wilk检验（小样本）
                            if len(data[col].dropna()) <= 50:
                                stat, p_value = shapiro(data[col].dropna())
                                test_name = 'Shapiro-Wilk'
                            else:
                                # D'Agostino's K-squared检验（大样本）
                                stat, p_value = normaltest(data[col].dropna())
                                test_name = "D'Agostino's K-squared"

                            summary['distribution_tests'][col] = {
                                'test_name': test_name,
                                'statistic': stat,
                                'p_value': p_value,
                                'is_normal': p_value > self.config['significance_level'],
                                'interpretation': '正态分布' if p_value > self.config['significance_level'] else '非正态分布'
                            }
                        except:
                            summary['distribution_tests'][col] = {
                                'test_name': 'Failed',
                                'reason': '无法执行正态性检验'
                            }

                    summary['numeric_summary'][col] = {
                        **numeric_stats[col].to_dict(),
                        **extra_stats
                    }

        # 分类列统计摘要
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                cat_stats = {
                    'unique_count': data[col].nunique(),
                    'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'most_frequent_count': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0,
                    'missing_rate': data[col].isnull().sum() / len(data),
                    'value_counts': data[col].value_counts().head(10).to_dict()  # 前10个值
                }

                # 分类变量的均匀性检验
                if data[col].nunique() <= 10:  # 只对类别数较少的变量进行检验
                    try:
                        observed = data[col].value_counts().values
                        expected = [len(data) / len(observed)] * len(observed)
                        chi2_stat, p_value = stats.chisquare(observed, expected)

                        cat_stats['uniformity_test'] = {
                            'chi2_statistic': chi2_stat,
                            'p_value': p_value,
                            'is_uniform': p_value > self.config['significance_level'],
                            'interpretation': '分布均匀' if p_value > self.config['significance_level'] else '分布不均匀'
                        }
                    except:
                        cat_stats['uniformity_test'] = {
                            'test_name': 'Failed',
                            'reason': '无法执行均匀性检验'
                        }

                summary['categorical_summary'][col] = cat_stats

        # 相关性分析
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr(method=self.config['correlation_method'])

            # 找出高相关性的变量对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # 高相关性阈值
                        high_corr_pairs.append({
                            'variable1': correlation_matrix.columns[i],
                            'variable2': correlation_matrix.columns[j],
                            'correlation': corr_val,
                            'strength': self._interpret_correlation_strength(abs(corr_val))
                        })

            summary['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'method': self.config['correlation_method'],
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation': correlation_matrix.abs().max().max(),
                'avg_correlation': correlation_matrix.abs().mean().mean()
            }

        self.results['statistical_summary'] = summary
        return summary

    def auto_eda(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        自动化EDA分析

        Parameters:
        - data: 待分析的数据

        Returns:
        - 完整的EDA分析结果
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("请先加载数据")

        print("🚀 开始自动化EDA分析...")

        # 1. 数据质量检查
        print("   1. 数据质量检查...")
        quality_report = self.data_quality_check(data)

        # 2. 统计摘要分析
        print("   2. 统计摘要分析...")
        stats_summary = self.generate_statistical_summary(data)

        # 3. 生成洞察
        print("   3. 生成数据洞察...")
        insights = self._generate_insights(quality_report, stats_summary)

        # 4. 数据建议
        print("   4. 生成处理建议...")
        recommendations = self._generate_recommendations(quality_report, stats_summary)

        # 综合结果
        eda_results = {
            'analysis_time': datetime.now().isoformat(),
            'data_quality': quality_report,
            'statistical_summary': stats_summary,
            'insights': insights,
            'recommendations': recommendations,
            'next_steps': self._suggest_next_steps(quality_report, stats_summary)
        }

        self.results['auto_eda'] = eda_results
        self._print_eda_summary(eda_results)

        return eda_results

    def _generate_insights(self, quality_report: Dict, stats_summary: Dict) -> List[str]:
        """生成数据洞察"""
        insights = []

        # 数据质量洞察
        quality_score = quality_report['data_quality_score']
        if quality_score >= 0.8:
            insights.append("数据质量优秀，适合直接进行分析")
        elif quality_score >= 0.6:
            insights.append("数据质量良好，但存在一些需要关注的问题")
        else:
            insights.append("数据质量需要改善，建议进行数据清洗")

        # 缺失值洞察
        missing_rate = quality_report['missing_values']['overall']['missing_rate']
        if missing_rate > 0.1:
            insights.append(f"数据存在较高缺失率({missing_rate:.1%})，需要针对性处理")
        elif missing_rate > 0.05:
            insights.append(f"数据存在少量缺失值({missing_rate:.1%})，可考虑填充或删除")

        # 数据分布洞察
        normal_distributions = sum(1 for test in stats_summary['distribution_tests'].values()
                                 if test.get('is_normal', False))
        total_tests = len(stats_summary['distribution_tests'])

        if total_tests > 0:
            normal_ratio = normal_distributions / total_tests
            if normal_ratio > 0.7:
                insights.append("大部分数值变量呈正态分布，适合参数统计方法")
            elif normal_ratio < 0.3:
                insights.append("大部分数值变量呈非正态分布，建议使用非参数方法")

        # 相关性洞察
        if 'correlation_analysis' in stats_summary:
            high_corr_count = len(stats_summary['correlation_analysis']['high_correlation_pairs'])
            if high_corr_count > 0:
                insights.append(f"发现{high_corr_count}对高度相关的变量，可能存在多重共线性问题")

        return insights

    def _generate_recommendations(self, quality_report: Dict, stats_summary: Dict) -> List[str]:
        """生成处理建议"""
        recommendations = []

        # 基于质量报告的建议
        recommendations.extend(quality_report.get('recommendations', []))

        # 基于统计摘要的建议
        # 异常值处理建议
        outlier_columns = len(quality_report['outliers'])
        if outlier_columns > 0:
            recommendations.append(f"发现{outlier_columns}个变量存在异常值，建议进行异常值处理")

        # 数据类型建议
        type_issues = quality_report['data_types']['type_issues']
        if type_issues:
            recommendations.append("检测到数据类型问题，建议检查并修正数据类型")

        # 正态性建议
        for col, test in stats_summary['distribution_tests'].items():
            if not test.get('is_normal', False) and test.get('p_value') is not None:
                recommendations.append(f"变量'{col}'非正态分布，考虑数据变换或使用非参数方法")

        return recommendations

    def _suggest_next_steps(self, quality_report: Dict, stats_summary: Dict) -> List[str]:
        """建议下一步操作"""
        next_steps = []

        # 基于数据质量确定优先级
        quality_score = quality_report['data_quality_score']

        if quality_score < 0.6:
            next_steps.extend([
                "1. 优先进行数据清洗和质量改善",
                "2. 处理缺失值和异常值",
                "3. 修正数据类型问题"
            ])
        else:
            next_steps.extend([
                "1. 进行数据可视化分析",
                "2. 探索变量间关系",
                "3. 考虑特征工程"
            ])

        # 基于数据特征建议分析方法
        numeric_cols = len(stats_summary['numeric_summary'])
        categorical_cols = len(stats_summary['categorical_summary'])

        if numeric_cols > 5:
            next_steps.append("建议进行降维分析（如PCA）")

        if categorical_cols > 3:
            next_steps.append("建议进行编码转换处理")

        # 相关性分析建议
        if 'correlation_analysis' in stats_summary:
            high_corr_pairs = stats_summary['correlation_analysis']['high_correlation_pairs']
            if high_corr_pairs:
                next_steps.append("处理高相关性变量以避免多重共线性")

        return next_steps

    def _print_eda_summary(self, eda_results: Dict):
        """打印EDA分析摘要"""
        print("\n" + "="*60)
        print("🎉 自动化EDA分析完成!")
        print("="*60)

        quality_score = eda_results['data_quality']['data_quality_score']
        print(f"\n📊 数据质量评分: {quality_score:.2f}/1.00")

        print(f"\n💡 主要洞察:")
        for insight in eda_results['insights']:
            print(f"   • {insight}")

        print(f"\n📋 关键建议:")
        for rec in eda_results['recommendations'][:5]:  # 显示前5个建议
            print(f"   • {rec}")

        print(f"\n🔄 下一步操作:")
        for step in eda_results['next_steps']:
            print(f"   • {step}")

    # 辅助方法
    def _rate_missing_quality(self, missing_rate: float) -> str:
        """评估缺失值质量"""
        if missing_rate == 0:
            return "完美"
        elif missing_rate < 0.01:
            return "优秀"
        elif missing_rate < 0.05:
            return "良好"
        elif missing_rate < 0.15:
            return "一般"
        else:
            return "较差"

    def _classify_missing_severity(self, missing_rate: float) -> str:
        """分类缺失值严重程度"""
        if missing_rate < 0.01:
            return "很低"
        elif missing_rate < 0.05:
            return "低"
        elif missing_rate < 0.15:
            return "中等"
        elif missing_rate < 0.30:
            return "高"
        else:
            return "极高"

    def _classify_duplicate_severity(self, duplicate_rate: float) -> str:
        """分类重复值严重程度"""
        if duplicate_rate < 0.01:
            return "很低"
        elif duplicate_rate < 0.05:
            return "低"
        elif duplicate_rate < 0.10:
            return "中等"
        else:
            return "高"

    def _classify_outlier_severity(self, outlier_rate: float) -> str:
        """分类异常值严重程度"""
        if outlier_rate < 0.01:
            return "很低"
        elif outlier_rate < 0.05:
            return "低"
        elif outlier_rate < 0.10:
            return "中等"
        else:
            return "高"

    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict:
        """分析缺失值模式"""
        # 计算缺失值模式
        missing_matrix = data.isnull()

        # 找出常见的缺失值组合
        missing_combinations = {}

        # 如果数据太大，进行采样
        if len(data) > self.config['sample_size_threshold']:
            sample_data = data.sample(min(self.config['sample_size_threshold'], len(data)))
        else:
            sample_data = data

        missing_combinations = sample_data.isnull().value_counts().head(10).to_dict()

        return {
            'common_patterns': {str(k): v for k, v in missing_combinations.items()},
            'total_patterns': len(missing_combinations)
        }

    def _calculate_quality_score(self, quality_report: Dict) -> float:
        """计算数据质量分数"""
        score = 1.0

        # 缺失值扣分
        missing_rate = quality_report['missing_values']['overall']['missing_rate']
        score -= missing_rate * 0.5

        # 重复值扣分
        duplicate_rate = quality_report['duplicates']['exact_duplicates']['rate']
        score -= duplicate_rate * 0.3

        # 异常值扣分
        outlier_cols = quality_report['outliers']['total_outlier_columns']
        if outlier_cols > 0:
            avg_outlier_rate = np.mean([
                info['outlier_rate'] for info in quality_report['outliers'].values()
            ])
            score -= avg_outlier_rate * 0.2

        # 数据类型问题扣分
        type_issues = len(quality_report['data_types']['type_issues'])
        score -= type_issues * 0.1

        return max(0, score)

    def _generate_quality_recommendations(self, quality_report: Dict) -> Tuple[List[str], List[str]]:
        """生成质量相关的建议"""
        issues = []
        recommendations = []

        # 缺失值问题
        missing_rate = quality_report['missing_values']['overall']['missing_rate']
        if missing_rate > 0.05:
            issues.append(f"数据缺失率较高({missing_rate:.1%})")
            recommendations.append("建议使用适当的填充策略处理缺失值")

        # 重复值问题
        duplicate_rate = quality_report['duplicates']['exact_duplicates']['rate']
        if duplicate_rate > 0.01:
            issues.append(f"存在重复数据({duplicate_rate:.1%})")
            recommendations.append("建议检查并处理重复数据")

        # 异常值问题
        outlier_cols = quality_report['outliers']['total_outlier_columns']
        if outlier_cols > 0:
            issues.append(f"发现{outlier_cols}个变量存在异常值")
            recommendations.append("建议检查异常值并决定处理策略")

        # 数据类型问题
        type_issues = quality_report['data_types']['type_issues']
        if type_issues:
            issues.append("检测到数据类型问题")
            recommendations.append("建议修正数据类型以获得更好的分析结果")

        return issues, recommendations

    def _interpret_correlation_strength(self, corr_value: float) -> str:
        """解释相关性强度"""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.9:
            return "极强"
        elif abs_corr >= 0.7:
            return "强"
        elif abs_corr >= 0.5:
            return "中等"
        elif abs_corr >= 0.3:
            return "弱"
        else:
            return "极弱"

    def export_results(self, output_path: str, format: str = 'json'):
        """
        导出分析结果

        Parameters:
        - output_path: 输出路径
        - format: 输出格式 ('json', 'csv')
        """
        if not self.results:
            print("❌ 没有可导出的分析结果")
            return

        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
            print(f"✅ 分析结果已导出到 {output_path}")

        elif format == 'csv':
            # 导出为CSV格式（主要用于统计摘要）
            if 'statistical_summary' in self.results:
                # 数值统计
                if self.results['statistical_summary']['numeric_summary']:
                    numeric_df = pd.DataFrame(self.results['statistical_summary']['numeric_summary']).T
                    numeric_df.to_csv(output_path.replace('.csv', '_numeric.csv'), encoding='utf-8-sig')

                # 分类统计
                if self.results['statistical_summary']['categorical_summary']:
                    categorical_df = pd.DataFrame(self.results['statistical_summary']['categorical_summary']).T
                    categorical_df.to_csv(output_path.replace('.csv', '_categorical.csv'), encoding='utf-8-sig')

                print(f"✅ 统计摘要已导出到CSV文件")
        else:
            raise ValueError("不支持的输出格式，请使用 'json' 或 'csv'")