"""
AB测试核心分析模块

提供完整的AB测试分析功能，包括：
- 基础转化率和留存率分析
- 样本量计算和功效分析
- 提升率计算和置信区间
- 贝叶斯AB测试方法
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ABTestAnalyzer:
    """AB测试分析器主类"""

    def __init__(self):
        self.data = None
        self.significance_level = 0.05
        self.statistical_power = 0.8
        self.multiple_comparison_correction = None

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载AB测试数据"""
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            print(f"数据加载成功: {len(self.data)} 条记录")
            return self.data
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return None

    def set_significance_level(self, alpha: float = 0.05):
        """设置显著性水平"""
        if not 0 < alpha < 1:
            raise ValueError("显著性水平必须在0和1之间")
        self.significance_level = alpha
        print(f"显著性水平设置为: {alpha}")

    def set_statistical_power(self, power: float = 0.8):
        """设置统计功效"""
        if not 0 < power < 1:
            raise ValueError("统计功效必须在0和1之间")
        self.statistical_power = power
        print(f"统计功效设置为: {power}")

    def set_multiple_comparison_correction(self, method: str = None):
        """设置多重比较校正方法"""
        valid_methods = ['bonferroni', 'fdr_bh', 'holm', 'sidak']
        if method and method not in valid_methods:
            raise ValueError(f"校正方法必须是: {valid_methods}")
        self.multiple_comparison_correction = method
        print(f"多重比较校正方法设置为: {method}")

    def calculate_conversion_rates(self,
                                 data: pd.DataFrame,
                                 group_col: str,
                                 conversion_col: str,
                                 value_mapping: Optional[Dict] = None) -> Dict:
        """计算各组转化率"""
        df = data.copy()

        # 处理转化状态映射
        if value_mapping:
            df[conversion_col] = df[conversion_col].map(value_mapping)

        # 确保转化列是数值类型
        if df[conversion_col].dtype == 'object':
            df[conversion_col] = df[conversion_col].apply(lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0)

        # 计算转化率
        results = {}
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            conversion_rate = group_data[conversion_col].mean()
            sample_size = len(group_data)
            conversions = group_data[conversion_col].sum()

            # 计算置信区间
            se = np.sqrt(conversion_rate * (1 - conversion_rate) / sample_size)
            ci_lower = max(0, conversion_rate - 1.96 * se)
            ci_upper = min(1, conversion_rate + 1.96 * se)

            results[group] = {
                'conversion_rate': conversion_rate,
                'sample_size': sample_size,
                'conversions': int(conversions),
                'standard_error': se,
                'confidence_interval': (ci_lower, ci_upper)
            }

        return results

    def calculate_lift(self,
                      conversion_results: Dict,
                      control_group: str,
                      test_group: str) -> Dict:
        """计算提升率和相关指标"""
        if control_group not in conversion_results or test_group not in conversion_results:
            raise ValueError("指定的组别在结果中不存在")

        control_rate = conversion_results[control_group]['conversion_rate']
        test_rate = conversion_results[test_group]['conversion_rate']

        # 绝对提升
        absolute_lift = test_rate - control_rate

        # 相对提升
        relative_lift = absolute_lift / control_rate if control_rate > 0 else 0

        # 提升率的置信区间 (使用Delta方法)
        control_se = conversion_results[control_group]['standard_error']
        test_se = conversion_results[test_group]['standard_error']

        lift_se = np.sqrt(test_se**2 + control_se**2)
        lift_ci_lower = absolute_lift - 1.96 * lift_se
        lift_ci_upper = absolute_lift + 1.96 * lift_se

        return {
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'lift_percentage': relative_lift * 100,
            'lift_standard_error': lift_se,
            'lift_confidence_interval': (lift_ci_lower, lift_ci_upper)
        }

    def calculate_retention_rates(self,
                                data: pd.DataFrame,
                                group_col: str,
                                retention_col: str) -> Dict:
        """计算留存率"""
        df = data.copy()

        # 确保留存列是布尔类型
        if df[retention_col].dtype == 'object':
            df[retention_col] = df[retention_col].apply(lambda x: True if str(x).strip() in ['TRUE', 'true', '1', '是'] else False)

        results = {}
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            retention_rate = group_data[retention_col].mean()
            sample_size = len(group_data)
            retained_users = group_data[retention_col].sum()

            # 计算置信区间
            se = np.sqrt(retention_rate * (1 - retention_rate) / sample_size)
            ci_lower = max(0, retention_rate - 1.96 * se)
            ci_upper = min(1, retention_rate + 1.96 * se)

            results[group] = {
                'retention_rate': retention_rate,
                'sample_size': sample_size,
                'retained_users': int(retained_users),
                'standard_error': se,
                'confidence_interval': (ci_lower, ci_upper)
            }

        return results

    def calculate_sample_size(self,
                            baseline_rate: float,
                            expected_lift: float,
                            alpha: float = None,
                            power: float = None,
                            two_tailed: bool = True) -> int:
        """计算AB测试所需样本量"""
        alpha = alpha or self.significance_level
        power = power or self.statistical_power

        # 获取Z值
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # 计算效应量
        p1 = baseline_rate
        p2 = p1 * (1 + expected_lift)

        # 使用Cohen's h计算样本量
        h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

        if h == 0:
            raise ValueError("效应量为0，无法计算样本量")

        # 计算每组所需样本量
        n_per_group = 2 * (z_alpha + z_beta)**2 / (h**2)

        return int(np.ceil(n_per_group))

    def check_randomization(self,
                           data: pd.DataFrame,
                           group_col: str,
                           feature_cols: List[str]) -> Dict:
        """检查随机分组是否均衡"""
        df = data.copy()
        results = {}

        for feature in feature_cols:
            if df[feature].dtype == 'object':
                # 分类变量：使用卡方检验
                contingency_table = pd.crosstab(df[group_col], df[feature])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                results[feature] = {
                    'test': 'Chi-square Test',
                    'statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'is_balanced': p_value > self.significance_level
                }
            else:
                # 数值变量：使用t检验或方差分析
                groups = [df[df[group_col] == group][feature].dropna() for group in df[group_col].unique()]
                if len(groups) == 2:
                    statistic, p_value = stats.ttest_ind(*groups)
                    test_name = 'Independent t-test'
                else:
                    statistic, p_value = stats.f_oneway(*groups)
                    test_name = 'ANOVA'

                results[feature] = {
                    'test': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_balanced': p_value > self.significance_level
                }

        return results

    def data_quality_check(self,
                          data: pd.DataFrame,
                          required_cols: List[str]) -> Dict:
        """数据质量检查"""
        df = data.copy()
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'duplicate_rows': df.duplicated().sum(),
            'required_columns_check': {}
        }

        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                results['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }

        # 检查数据类型
        for col in df.columns:
            results['data_types'][col] = str(df[col].dtype)

        # 检查必需列
        for col in required_cols:
            results['required_columns_check'][col] = col in df.columns

        return results

    def analyze_conversion(self,
                          data: pd.DataFrame,
                          group_col: str,
                          conversion_col: str,
                          control_group: Optional[str] = None) -> Dict:
        """综合转化率分析"""
        # 计算转化率
        conversion_results = self.calculate_conversion_rates(data, group_col, conversion_col)

        # 如果指定了对照组，计算提升率
        lift_results = {}
        if control_group and len(conversion_results) > 1:
            for group in conversion_results:
                if group != control_group:
                    lift_results[group] = self.calculate_lift(
                        conversion_results, control_group, group
                    )

        return {
            'conversion_analysis': conversion_results,
            'lift_analysis': lift_results,
            'summary': {
                'total_users': len(data),
                'groups': list(conversion_results.keys()),
                'overall_conversion_rate': data[conversion_col].mean() if data[conversion_col].dtype != 'object'
                else data[conversion_col].apply(lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0).mean()
            }
        }

    def analyze_retention(self,
                         data: pd.DataFrame,
                         group_col: str,
                         retention_col: str) -> Dict:
        """综合留存率分析"""
        retention_results = self.calculate_retention_rates(data, group_col, retention_col)

        return {
            'retention_analysis': retention_results,
            'summary': {
                'total_users': len(data),
                'groups': list(retention_results.keys()),
                'overall_retention_rate': data[retention_col].mean() if data[retention_col].dtype != 'object'
                else data[retention_col].apply(lambda x: 1 if str(x).strip() in ['TRUE', 'true', '1', '是'] else 0).mean()
            }
        }

    def bayesian_ab_test(self,
                        data: pd.DataFrame,
                        group_col: str,
                        conversion_col: str,
                        prior_type: str = 'jeffreys',
                        n_simulations: int = 10000) -> Dict:
        """贝叶斯AB测试分析"""
        df = data.copy()

        # 确保转化列是数值类型
        if df[conversion_col].dtype == 'object':
            df[conversion_col] = df[conversion_col].apply(lambda x: 1 if str(x).strip() in ['是', 'True', 'true', '1', 'yes'] else 0)

        results = {}

        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            successes = int(group_data[conversion_col].sum())
            trials = len(group_data)

            # 设置先验分布
            if prior_type == 'jeffreys':
                alpha_prior = 0.5
                beta_prior = 0.5
            elif prior_type == 'uniform':
                alpha_prior = 1
                beta_prior = 1
            else:  # 使用经验贝叶斯
                alpha_prior = successes + 1
                beta_prior = trials - successes + 1

            # 后验分布参数
            alpha_post = alpha_prior + successes
            beta_post = beta_prior + trials - successes

            # 从后验分布采样
            samples = np.random.beta(alpha_post, beta_post, n_simulations)

            results[group] = {
                'successes': successes,
                'trials': trials,
                'posterior_alpha': alpha_post,
                'posterior_beta': beta_post,
                'posterior_mean': alpha_post / (alpha_post + beta_post),
                'posterior_samples': samples,
                'credible_interval': (
                    np.percentile(samples, 2.5),
                    np.percentile(samples, 97.5)
                )
            }

        # 计算组间比较
        if len(results) == 2:
            groups = list(results.keys())
            group1_samples = results[groups[0]]['posterior_samples']
            group2_samples = results[groups[1]]['posterior_samples']

            # 计算概率
            prob_group1_better = np.mean(group1_samples > group2_samples)
            prob_group2_better = 1 - prob_group1_better

            # 计算差异分布
            diff_samples = group1_samples - group2_samples

            results['comparison'] = {
                'prob_' + groups[0] + '_better': prob_group1_better,
                'prob_' + groups[1] + '_better': prob_group2_better,
                'difference_distribution': diff_samples,
                'difference_mean': np.mean(diff_samples),
                'difference_ci': (
                    np.percentile(diff_samples, 2.5),
                    np.percentile(diff_samples, 97.5)
                )
            }

        return results

    def calculate_win_probability(self, bayesian_results: Dict) -> Dict:
        """计算贝叶斯AB测试的获胜概率"""
        if 'comparison' not in bayesian_results:
            return {"error": "需要先运行贝叶斯AB测试"}

        comparison = bayesian_results['comparison']

        # 找出获胜概率最大的组
        win_probabilities = {}
        for key, value in comparison.items():
            if key.startswith('prob_') and key.endswith('_better'):
                group_name = key.replace('prob_', '').replace('_better', '')
                win_probabilities[group_name] = value

        # 找出获胜组
        winner = max(win_probabilities, key=win_probabilities.get)

        return {
            'win_probabilities': win_probabilities,
            'winner': winner,
            'winner_probability': win_probabilities[winner]
        }