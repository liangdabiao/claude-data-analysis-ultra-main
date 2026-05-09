"""
ROI分析器模块 - 增长模型成本效益分析

提供全面的ROI分析功能，包括：
- 营销活动ROI计算
- 成本效益分析
- 预算分配优化
- LTV预测
- 增长策略投资回报评估
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ROIAnalyzer:
    """ROI分析器 - 成本效益与预算优化核心"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化ROI分析器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.results = {}

    def calculate_campaign_roi(self,
                             data: pd.DataFrame,
                             campaign_col: str = '裂变类型',
                             conversion_col: str = '是否转化',
                             cost_col: str = '成本',
                             revenue_col: str = '收入',
                             user_col: str = '用户码') -> Dict:
        """
        计算营销活动ROI

        Parameters:
        - data: 营销数据
        - campaign_col: 活动类型列名
        - conversion_col: 转化结果列名
        - cost_col: 成本列名
        - revenue_col: 收入列名
        - user_col: 用户标识列名

        Returns:
        - ROI分析结果
        """
        results = {}

        # 按活动分组计算指标
        campaign_metrics = data.groupby(campaign_col).agg({
            user_col: 'nunique',        # 触达用户数
            conversion_col: 'sum',       # 转化数
            cost_col: 'sum',            # 总成本
            revenue_col: 'sum'          # 总收入
        }).round(4)

        campaign_metrics.columns = ['触达用户数', '转化数', '总成本', '总收入']
        campaign_metrics['转化率'] = campaign_metrics['转化数'] / campaign_metrics['触达用户数']
        campaign_metrics['ARPU'] = campaign_metrics['总收入'] / campaign_metrics['触达用户数']  # 每用户平均收入
        campaign_metrics['CPA'] = campaign_metrics['总成本'] / campaign_metrics['转化数']  # 获客成本
        campaign_metrics['ROI'] = (campaign_metrics['总收入'] - campaign_metrics['总成本']) / campaign_metrics['总成本']  # ROI

        # 计算投资回收期
        campaign_metrics['回收期(天)'] = campaign_metrics['总成本'] / (campaign_metrics['总收入'] / 30)  # 假设按30天计算

        # 计算利润率
        campaign_metrics['利润率'] = (campaign_metrics['总收入'] - campaign_metrics['总成本']) / campaign_metrics['总收入']

        # LTV/CPA比率
        campaign_metrics['LTV_CPA_Ratio'] = campaign_metrics['ARPU'] * 12 / campaign_metrics['CPA']  # 假设年度LTV

        results['campaign_metrics'] = campaign_metrics.to_dict()
        results['overall_roi'] = (campaign_metrics['总收入'].sum() - campaign_metrics['总成本'].sum()) / campaign_metrics['总成本'].sum()
        results['best_campaign'] = campaign_metrics['ROI'].idxmax()
        results['worst_campaign'] = campaign_metrics['ROI'].idxmin()

        self.results['campaign_roi'] = results
        return results

    def calculate_ltv(self,
                     data: pd.DataFrame,
                     user_col: str = '用户码',
                     revenue_col: str = '收入',
                     frequency_col: str = 'F值',
                     recency_col: str = 'R值') -> pd.DataFrame:
        """
        计算用户生命周期价值(LTV)

        Parameters:
        - data: 用户数据
        - user_col: 用户标识列名
        - revenue_col: 收入列名
        - frequency_col: 频次指标列名
        - recency_col: 近度指标列名

        Returns:
        - 包含LTV的DataFrame
        """
        # 用户历史价值分析
        user_metrics = data.groupby(user_col).agg({
            revenue_col: ['sum', 'mean'],
            frequency_col: 'sum',
            recency_col: 'min'
        }).round(4)

        user_metrics.columns = ['总收入', '平均收入', '总频次', '最近消费']

        # 计算基础LTV指标
        user_metrics['平均订单价值'] = user_metrics['总收入'] / user_metrics['总频次']
        user_metrics['购买频率'] = user_metrics['总频次']

        # 简单LTV预测 (基于历史数据)
        user_metrics['预测LTV'] = user_metrics['平均订单价值'] * user_metrics['购买频率'] * 12  # 年度预测

        # 考虑客户流失的LTV调整
        churn_rate = 1 / (user_metrics['最近消费'].mean() / 30 + 1)  # 简化的流失率估算
        user_metrics['调整LTV'] = user_metrics['预测LTV'] * (1 - churn_rate)

        # LTV分群
        ltv_quantiles = user_metrics['调整LTV'].quantile([0.2, 0.4, 0.6, 0.8])

        def classify_ltv(ltv):
            if ltv <= ltv_quantiles[0.2]:
                return '低价值'
            elif ltv <= ltv_quantiles[0.4]:
                return '中低价值'
            elif ltv <= ltv_quantiles[0.6]:
                return '中等价值'
            elif ltv <= ltv_quantiles[0.8]:
                return '中高价值'
            else:
                return '高价值'

        user_metrics['LTV分群'] = user_metrics['调整LTV'].apply(classify_ltv)

        self.results['ltv_analysis'] = user_metrics
        return user_metrics

    def optimize_budget_allocation(self,
                                  campaigns_data: Dict[str, Dict],
                                  total_budget: float,
                                  min_allocation: float = 0.05) -> Dict:
        """
        优化预算分配

        Parameters:
        - campaigns_data: 活动数据字典 {活动名: {roi, reach, cost, capacity}}
        - total_budget: 总预算
        - min_allocation: 最小分配比例

        Returns:
        - 预算优化结果
        """
        campaigns = list(campaigns_data.keys())
        n_campaigns = len(campaigns)

        # 目标函数：最大化总回报
        def objective(allocations):
            total_return = 0
            for i, campaign in enumerate(campaigns):
                allocation = allocations[i]
                campaign_data = campaigns_data[campaign]

                # 使用边际效益递减模型
                base_roi = campaign_data.get('roi', 0.1)
                capacity = campaign_data.get('capacity', total_budget)

                # 边际效益递减函数
                marginal_return = base_roi * allocation * (1 - allocation / (2 * capacity))
                total_return += marginal_return

            return -total_return  # 负号因为要最大化

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: sum(x) - total_budget},  # 总预算约束
        ]

        # 边界约束
        bounds = [(total_budget * min_allocation, total_budget * 0.6) for _ in range(n_campaigns)]

        # 初始猜测 (均匀分配)
        initial_guess = [total_budget / n_campaigns] * n_campaigns

        # 优化
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            optimized_allocations = result.x
        else:
            # 如果优化失败，使用均匀分配
            optimized_allocations = initial_guess

        # 生成结果
        allocation_results = {}
        for i, campaign in enumerate(campaigns):
            allocation = optimized_allocations[i]
            campaign_data = campaigns_data[campaign]

            allocation_results[campaign] = {
                '分配预算': allocation,
                '分配比例': allocation / total_budget,
                '预期回报': allocation * campaign_data.get('roi', 0.1),
                '预期触达': allocation / campaign_data.get('cost', 1),
                '分配比例提升': self._calculate_allocation_improvement(campaign, allocation, campaigns_data)
            }

        results = {
            'allocations': allocation_results,
            'total_expected_return': sum(r['预期回报'] for r in allocation_results.values()),
            'optimization_success': result.success,
            'optimization_message': result.message if not result.success else 'Optimization successful'
        }

        self.results['budget_optimization'] = results
        return results

    def calculate_marginal_roi(self,
                              data: pd.DataFrame,
                              campaign_col: str = '裂变类型',
                              spend_col: str = '成本',
                              return_col: str = '收入') -> Dict:
        """
        计算边际ROI

        Parameters:
        - data: 营销数据
        - campaign_col: 活动类型列名
        - spend_col: 投入列名
        - return_col: 回报列名

        Returns:
        - 边际ROI分析结果
        """
        marginal_results = {}

        for campaign in data[campaign_col].unique():
            campaign_data = data[data[campaign_col] == campaign].copy()

            if len(campaign_data) < 10:
                continue

            # 按投入排序，计算边际效益
            campaign_data = campaign_data.sort_values(spend_col)
            campaign_data['累计投入'] = campaign_data[spend_col].cumsum()
            campaign_data['累计回报'] = campaign_data[return_col].cumsum()
            campaign_data['边际投入'] = campaign_data[spend_col].diff()
            campaign_data['边际回报'] = campaign_data[return_col].diff()

            # 计算边际ROI
            campaign_data['边际ROI'] = campaign_data['边际回报'] / campaign_data['边际投入']

            # 计算平均边际ROI (排除异常值)
            valid_marginal_roi = campaign_data['边际ROI'].dropna()
            valid_marginal_roi = valid_marginal_roi[valid_marginal_roi.abs() < 10]  # 过滤极端值

            if len(valid_marginal_roi) > 0:
                marginal_results[campaign] = {
                    '平均边际ROI': valid_marginal_roi.mean(),
                    '边际ROI标准差': valid_marginal_roi.std(),
                    '边际ROI中位数': valid_marginal_roi.median(),
                    '最佳投入点': campaign_data.loc[valid_marginal_roi.idxmax(), spend_col] if len(valid_marginal_roi) > 0 else 0,
                    '投入回报弹性': self._calculate_roi_elasticity(campaign_data)
                }

        self.results['marginal_roi'] = marginal_results
        return marginal_results

    def simulate_roi_scenario(self,
                             base_data: Dict,
                             scenarios: List[Dict]) -> Dict:
        """
        ROI情景模拟

        Parameters:
        - base_data: 基础数据
        - scenarios: 情景列表

        Returns:
        - 情景模拟结果
        """
        simulation_results = {}

        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')

            # 应用情景参数
            modified_data = self._apply_scenario_parameters(base_data, scenario)

            # 计算情景ROI
            scenario_roi = self._calculate_scenario_roi(modified_data)

            simulation_results[scenario_name] = {
                'scenario_parameters': scenario,
                'roi_result': scenario_roi,
                'roi_vs_base': scenario_roi - base_data.get('base_roi', 0),
                'improvement_percentage': (scenario_roi - base_data.get('base_roi', 0)) / base_data.get('base_roi', 1) * 100
            }

        self.results['roi_simulation'] = simulation_results
        return simulation_results

    def analyze_roi_trend(self,
                         data: pd.DataFrame,
                         time_col: str = '时间',
                         revenue_col: str = '收入',
                         cost_col: str = '成本') -> Dict:
        """
        分析ROI趋势

        Parameters:
        - data: 时间序列数据
        - time_col: 时间列名
        - revenue_col: 收入列名
        - cost_col: 成本列名

        Returns:
        - ROI趋势分析结果
        """
        # 确保时间列格式正确
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])

        # 按时间分组计算ROI
        time_series = data.groupby(data[time_col].dt.to_period('M')).agg({
            revenue_col: 'sum',
            cost_col: 'sum'
        })

        time_series['ROI'] = (time_series[revenue_col] - time_series[cost_col]) / time_series[cost_col]
        time_series['累计ROI'] = ((time_series[revenue_col].cumsum() - time_series[cost_col].cumsum()) /
                                time_series[cost_col].cumsum())

        # 计算趋势
        time_series['ROI_MoM'] = time_series['ROI'].pct_change() * 100  # 月环比
        time_series['ROI_3MAvg'] = time_series['ROI'].rolling(window=3).mean()  # 3月移动平均

        # 趋势分析
        trend_analysis = {
            'overall_trend': '上升' if time_series['ROI'].iloc[-1] > time_series['ROI'].iloc[0] else '下降',
            'average_roi': time_series['ROI'].mean(),
            'roi_volatility': time_series['ROI'].std(),
            'best_period': time_series['ROI'].idxmax(),
            'worst_period': time_series['ROI'].idxmin(),
            'trend_strength': self._calculate_trend_strength(time_series['ROI'])
        }

        results = {
            'time_series': time_series.to_dict(),
            'trend_analysis': trend_analysis
        }

        self.results['roi_trend'] = results
        return results

    def generate_roi_report(self) -> Dict:
        """
        生成ROI分析报告

        Returns:
        - 综合ROI分析报告
        """
        if not self.results:
            return {"error": "请先进行分析以生成结果"}

        report = {
            "summary": {},
            "key_findings": [],
            "recommendations": [],
            "budget_optimization": {}
        }

        # 营销活动ROI总结
        if 'campaign_roi' in self.results:
            roi_results = self.results['campaign_roi']
            report["summary"]["overall_roi"] = roi_results['overall_roi']
            report["summary"]["best_campaign"] = roi_results['best_campaign']
            report["summary"]["campaign_count"] = len(roi_results['campaign_metrics'])

            # 关键发现
            if roi_results['overall_roi'] > 0.5:
                report["key_findings"].append("整体营销ROI表现优秀")
            elif roi_results['overall_roi'] > 0.2:
                report["key_findings"].append("整体营销ROI表现良好")
            else:
                report["key_findings"].append("营销ROI需要改善")

            report["recommendations"].append(f"重点关注{roi_results['best_campaign']}活动的成功经验")

        # 预算优化建议
        if 'budget_optimization' in self.results:
            optimization = self.results['budget_optimization']
            report["budget_optimization"] = optimization['allocations']
            report["recommendations"].append("根据ROI优化结果调整预算分配")

        # LTV分析洞察
        if 'ltv_analysis' in self.results:
            ltv_data = self.results['ltv_analysis']
            high_value_users = len(ltv_data[ltv_data['LTV分群'] == '高价值'])
            total_users = len(ltv_data)
            high_value_ratio = high_value_users / total_users

            report["key_findings"].append(f"高价值用户占比: {high_value_ratio:.1%}")
            if high_value_ratio < 0.2:
                report["recommendations"].append("加强高价值用户识别和维护策略")

        return report

    def plot_roi_comparison(self, roi_data: Dict, title: str = "ROI对比分析"):
        """
        绘制ROI对比图

        Parameters:
        - roi_data: ROI数据
        - title: 图表标题
        """
        if isinstance(roi_data, dict) and 'campaign_metrics' in roi_data:
            metrics_df = pd.DataFrame(roi_data['campaign_metrics']).T
        else:
            metrics_df = pd.DataFrame(roi_data).T

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)

        # ROI对比
        metrics_df['ROI'].sort_values().plot(kind='barh', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('各活动ROI对比')
        axes[0,0].set_xlabel('ROI')

        # 转化率对比
        metrics_df['转化率'].sort_values().plot(kind='barh', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('各活动转化率对比')
        axes[0,1].set_xlabel('转化率')

        # CPA对比
        metrics_df['CPA'].sort_values().plot(kind='barh', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('各活动获客成本对比')
        axes[1,0].set_xlabel('CPA')

        # 触达用户数对比
        metrics_df['触达用户数'].sort_values().plot(kind='barh', ax=axes[1,1], color='gold')
        axes[1,1].set_title('各活动触达用户数对比')
        axes[1,1].set_xlabel('触达用户数')

        plt.tight_layout()
        return fig

    def _calculate_allocation_improvement(self, campaign: str, allocation: float, campaigns_data: Dict) -> float:
        """计算分配比例改善情况"""
        current_roi = campaigns_data[campaign].get('roi', 0.1)
        total_current_roi = sum(c.get('roi', 0.1) for c in campaigns_data.values())

        if total_current_roi > 0:
            proportional_allocation = allocation / sum(campaigns_data[c].get('cost', 1) for c in campaigns_data)
            roi_based_allocation = current_roi / total_current_roi

            return (proportional_allocation - roi_based_allocation) / roi_based_allocation
        return 0

    def _calculate_roi_elasticity(self, campaign_data: pd.DataFrame) -> float:
        """计算ROI弹性"""
        if len(campaign_data) < 10:
            return 0

        # 简化的弹性计算
        spend_change = campaign_data['累计投入'].pct_change().dropna()
        return_change = campaign_data['累计回报'].pct_change().dropna()

        if len(spend_change) > 1 and len(return_change) > 1:
            correlation = np.corrcoef(spend_change[1:], return_change[1:])[0, 1]
            return correlation if not np.isnan(correlation) else 0
        return 0

    def _apply_scenario_parameters(self, base_data: Dict, scenario: Dict) -> Dict:
        """应用情景参数到基础数据"""
        modified_data = base_data.copy()

        for key, value in scenario.items():
            if key in ['cost_reduction', 'revenue_increase', 'conversion_improvement']:
                if key == 'cost_reduction':
                    modified_data['total_cost'] = modified_data.get('total_cost', 0) * (1 - value)
                elif key == 'revenue_increase':
                    modified_data['total_revenue'] = modified_data.get('total_revenue', 0) * (1 + value)
                elif key == 'conversion_improvement':
                    modified_data['conversion_rate'] = modified_data.get('conversion_rate', 0) * (1 + value)

        return modified_data

    def _calculate_scenario_roi(self, data: Dict) -> float:
        """计算情景ROI"""
        revenue = data.get('total_revenue', 0)
        cost = data.get('total_cost', 1)
        return (revenue - cost) / cost if cost > 0 else 0

    def _calculate_trend_strength(self, roi_series: pd.Series) -> str:
        """计算趋势强度"""
        if len(roi_series) < 3:
            return "数据不足"

        # 简单线性回归计算趋势
        x = np.arange(len(roi_series))
        slope, _ = np.polyfit(x, roi_series, 1)

        if abs(slope) < 0.001:
            return "平稳"
        elif abs(slope) < 0.01:
            return "温和"
        else:
            return "强劲"