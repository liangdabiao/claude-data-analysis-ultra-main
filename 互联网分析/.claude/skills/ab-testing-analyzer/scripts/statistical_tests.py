"""
统计检验模块

提供全面的统计显著性检验功能，包括：
- t检验 (独立样本、配对样本)
- 卡方检验 (拟合优度、独立性)
- 方差分析 (ANOVA)
- 非参数检验
- 效应量计算
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class StatisticalTests:
    """统计检验类"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.correction_method = None

    def set_alpha(self, alpha: float):
        """设置显著性水平"""
        if not 0 < alpha < 1:
            raise ValueError("显著性水平必须在0和1之间")
        self.alpha = alpha

    def set_correction_method(self, method: str):
        """设置多重比较校正方法"""
        valid_methods = ['bonferroni', 'fdr_bh', 'holm', 'sidak']
        if method not in valid_methods:
            raise ValueError(f"校正方法必须是: {valid_methods}")
        self.correction_method = method

    def t_test(self,
               data: pd.DataFrame,
               group_col: str,
               metric_col: str,
               test_type: str = 'independent',
               equal_var: bool = True,
               alternative: str = 'two-sided') -> Dict:
        """
        t检验

        Parameters:
        - test_type: 'independent' 独立样本t检验, 'paired' 配对样本t检验
        - equal_var: 是否假设方差相等 (仅用于独立样本t检验)
        - alternative: 'two-sided', 'less', 'greater'
        """
        df = data.copy()

        # 确保数值列是数值类型
        if df[metric_col].dtype == 'object':
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

        if test_type == 'independent':
            groups = df[group_col].unique()
            if len(groups) != 2:
                raise ValueError("独立样本t检验需要恰好2个组")

            group1_data = df[df[group_col] == groups[0]][metric_col].dropna()
            group2_data = df[df[group_col] == groups[1]][metric_col].dropna()

            # 执行t检验
            statistic, p_value = stats.ttest_ind(
                group1_data, group2_data,
                equal_var=equal_var,
                alternative=alternative
            )

            # 计算效应量 (Cohen's d)
            pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() +
                                 (len(group2_data) - 1) * group2_data.var()) /
                                (len(group1_data) + len(group2_data) - 2))
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

            # 计算置信区间
            se = pooled_std * np.sqrt(1/len(group1_data) + 1/len(group2_data))
            mean_diff = group1_data.mean() - group2_data.mean()
            ci_lower = mean_diff - 1.96 * se
            ci_upper = mean_diff + 1.96 * se

            result = {
                'test_type': 'Independent t-test',
                'statistic': statistic,
                'p_value': p_value,
                'degrees_of_freedom': len(group1_data) + len(group2_data) - 2,
                'effect_size': cohens_d,
                'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                'mean_difference': mean_diff,
                'confidence_interval': (ci_lower, ci_upper),
                'group1_stats': {
                    'name': groups[0],
                    'n': len(group1_data),
                    'mean': group1_data.mean(),
                    'std': group1_data.std()
                },
                'group2_stats': {
                    'name': groups[1],
                    'n': len(group2_data),
                    'mean': group2_data.mean(),
                    'std': group2_data.std()
                },
                'is_significant': p_value < self.alpha
            }

        elif test_type == 'paired':
            if len(df[group_col].unique()) != 2:
                raise ValueError("配对样本t检验需要恰好2个组")

            # 确保配对数据
            group1_data = df[df[group_col] == df[group_col].unique()[0]][metric_col].dropna()
            group2_data = df[df[group_col] == df[group_col].unique()[1]][metric_col].dropna()

            if len(group1_data) != len(group2_data):
                raise ValueError("配对样本t检验需要两组数据长度相同")

            # 执行配对t检验
            statistic, p_value = stats.ttest_rel(group1_data, group2_data, alternative=alternative)

            # 计算效应量
            differences = group1_data - group2_data
            cohens_d = differences.mean() / differences.std()

            # 计算置信区间
            se = differences.std() / np.sqrt(len(differences))
            mean_diff = differences.mean()
            ci_lower = mean_diff - 1.96 * se
            ci_upper = mean_diff + 1.96 * se

            result = {
                'test_type': 'Paired t-test',
                'statistic': statistic,
                'p_value': p_value,
                'degrees_of_freedom': len(differences) - 1,
                'effect_size': cohens_d,
                'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                'mean_difference': mean_diff,
                'confidence_interval': (ci_lower, ci_upper),
                'paired_differences_stats': {
                    'n': len(differences),
                    'mean': differences.mean(),
                    'std': differences.std()
                },
                'is_significant': p_value < self.alpha
            }

        else:
            raise ValueError("test_type必须是'independent'或'paired'")

        return result

    def chi_square_test(self,
                        data: pd.DataFrame,
                        group_col: str = None,
                        outcome_col: str = None,
                        contingency_table: pd.DataFrame = None,
                        test_type: str = 'independence') -> Dict:
        """
        卡方检验

        Parameters:
        - test_type: 'independence' 独立性检验, 'goodness_of_fit' 拟合优度检验
        """
        if test_type == 'independence':
            if not group_col or not outcome_col:
                raise ValueError("独立性检验需要group_col和outcome_col参数")

            # 创建列联表
            observed = pd.crosstab(data[group_col], data[outcome_col])

        elif test_type == 'goodness_of_fit':
            if contingency_table is None:
                raise ValueError("拟合优度检验需要contingency_table参数")
            observed = contingency_table

        else:
            raise ValueError("test_type必须是'independence'或'goodness_of_fit'")

        # 执行卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        # 计算效应量 (Cramer's V)
        n = observed.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))

        # 计算标准化残差
        std_residuals = (observed - expected) / np.sqrt(expected)

        result = {
            'test_type': f'Chi-square {test_type.replace("_", " ").title()} Test',
            'statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size': cramers_v,
            'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
            'observed_frequencies': observed.to_dict(),
            'expected_frequencies': pd.DataFrame(expected,
                                               index=observed.index,
                                               columns=observed.columns).to_dict(),
            'standardized_residuals': std_residuals.to_dict(),
            'is_significant': p_value < self.alpha
        }

        return result

    def one_way_anova(self,
                      data: pd.DataFrame,
                      group_col: str,
                      metric_col: str) -> Dict:
        """单因素方差分析"""
        df = data.copy()

        # 确保数值列是数值类型
        if df[metric_col].dtype == 'object':
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

        # 获取各组数据
        groups = [df[df[group_col] == group][metric_col].dropna()
                 for group in df[group_col].unique()]

        # 执行ANOVA
        statistic, p_value = stats.f_oneway(*groups)

        # 计算效应量 (Eta-squared)
        total_var = np.var(df[metric_col].dropna())
        group_means = [group.mean() for group in groups]
        group_sizes = [len(group) for group in groups]
        grand_mean = np.mean(df[metric_col].dropna())

        ss_between = sum(group_sizes[i] * (group_means[i] - grand_mean)**2 for i in range(len(groups)))
        ss_total = (len(df[metric_col].dropna()) - 1) * total_var
        eta_squared = ss_between / ss_total

        # 计算组间统计
        group_stats = {}
        for i, group_name in enumerate(df[group_col].unique()):
            group_stats[group_name] = {
                'n': group_sizes[i],
                'mean': group_means[i],
                'std': groups[i].std(),
                'var': groups[i].var()
            }

        result = {
            'test_type': 'One-way ANOVA',
            'statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom_between': len(groups) - 1,
            'degrees_of_freedom_within': len(df[metric_col].dropna()) - len(groups),
            'effect_size': eta_squared,
            'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
            'group_statistics': group_stats,
            'is_significant': p_value < self.alpha
        }

        return result

    def mann_whitney_u_test(self,
                           data: pd.DataFrame,
                           group_col: str,
                           metric_col: str,
                           alternative: str = 'two-sided') -> Dict:
        """Mann-Whitney U检验 (非参数检验)"""
        df = data.copy()

        # 确保数值列是数值类型
        if df[metric_col].dtype == 'object':
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

        groups = df[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Mann-Whitney U检验需要恰好2个组")

        group1_data = df[df[group_col] == groups[0]][metric_col].dropna()
        group2_data = df[df[group_col] == groups[1]][metric_col].dropna()

        # 执行Mann-Whitney U检验
        statistic, p_value = stats.mannwhitneyu(
            group1_data, group2_data,
            alternative=alternative
        )

        # 计算效应量 (Rank-biserial correlation)
        n1, n2 = len(group1_data), len(group2_data)
        z_score = (statistic - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        rank_biserial = z_score / np.sqrt((n1 + n2))

        result = {
            'test_type': 'Mann-Whitney U Test',
            'statistic': statistic,
            'p_value': p_value,
            'z_score': z_score,
            'effect_size': abs(rank_biserial),
            'effect_size_interpretation': self._interpret_rank_biserial(abs(rank_biserial)),
            'group1_stats': {
                'name': groups[0],
                'n': len(group1_data),
                'median': group1_data.median(),
                'mean_rank': group1_data.rank().mean()
            },
            'group2_stats': {
                'name': groups[1],
                'n': len(group2_data),
                'median': group2_data.median(),
                'mean_rank': group2_data.rank().mean()
            },
            'is_significant': p_value < self.alpha
        }

        return result

    def wilcoxon_signed_rank_test(self,
                                 data: pd.DataFrame,
                                 group_col: str,
                                 metric_col: str) -> Dict:
        """Wilcoxon符号秩检验 (配对样本非参数检验)"""
        df = data.copy()

        # 确保数值列是数值类型
        if df[metric_col].dtype == 'object':
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

        if len(df[group_col].unique()) != 2:
            raise ValueError("Wilcoxon符号秩检验需要恰好2个组")

        groups = df[group_col].unique()
        group1_data = df[df[group_col] == groups[0]][metric_col].dropna()
        group2_data = df[df[group_col] == groups[1]][metric_col].dropna()

        if len(group1_data) != len(group2_data):
            raise ValueError("配对检验需要两组数据长度相同")

        # 执行Wilcoxon符号秩检验
        statistic, p_value = stats.wilcoxon(group1_data, group2_data)

        # 计算效应量
        differences = group1_data - group2_data
        non_zero_diffs = differences[differences != 0]
        z_score = (statistic - len(non_zero_diffs) * (len(non_zero_diffs) + 1) / 4) / np.sqrt(len(non_zero_diffs) * (len(non_zero_diffs) + 1) * (2 * len(non_zero_diffs) + 1) / 24)
        rank_biserial = z_score / np.sqrt(len(differences))

        result = {
            'test_type': 'Wilcoxon Signed-Rank Test',
            'statistic': statistic,
            'p_value': p_value,
            'z_score': z_score,
            'effect_size': abs(rank_biserial),
            'effect_size_interpretation': self._interpret_rank_biserial(abs(rank_biserial)),
            'sample_size': len(differences),
            'zero_differences_removed': len(differences) - len(non_zero_diffs),
            'is_significant': p_value < self.alpha
        }

        return result

    def cohens_d(self, data: pd.DataFrame, group_col: str, metric_col: str) -> float:
        """计算Cohen's d效应量"""
        groups = data[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Cohen's d计算需要恰好2个组")

        group1_data = data[data[group_col] == groups[0]][metric_col].dropna()
        group2_data = data[data[group_col] == groups[1]][metric_col].dropna()

        # 计算合并标准差
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() +
                             (len(group2_data) - 1) * group2_data.var()) /
                            (len(group1_data) + len(group2_data) - 2))

        # 计算Cohen's d
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

        return cohens_d

    def cramers_v(self, data: pd.DataFrame, col1: str, col2: str) -> float:
        """计算Cramer's V效应量"""
        # 创建列联表
        contingency_table = pd.crosstab(data[col1], data[col2])

        # 执行卡方检验
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)

        # 计算Cramer's V
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

        return cramers_v

    def multiple_comparison_correction(self, p_values: List[float], method: str = None) -> List[float]:
        """多重比较校正"""
        method = method or self.correction_method

        if method is None:
            return p_values

        p_values = np.array(p_values)

        if method == 'bonferroni':
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)

        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR校正
            ranked_p_values = stats.rankdata(p_values)
            corrected_p = p_values * len(p_values) / ranked_p_values
            corrected_p = np.minimum.accumulate(corrected_p[::-1])[::-1]
            corrected_p = np.minimum(corrected_p, 1.0)

        elif method == 'holm':
            # Holm-Bonferroni校正
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * (len(p_values) - i)
            corrected_p = np.minimum(corrected_p, 1.0)

        elif method == 'sidak':
            # Šidák校正
            corrected_p = 1 - (1 - p_values) ** len(p_values)

        else:
            raise ValueError(f"未知的校正方法: {method}")

        return corrected_p.tolist()

    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d效应量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

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

    def _interpret_eta_squared(self, eta2: float) -> str:
        """解释Eta-squared效应量"""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_rank_biserial(self, r: float) -> str:
        """解释Rank-biserial相关效应量"""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"