#!/usr/bin/env python3
"""
高级分析模块
提供客户行为分析、流失预测、聚类分析等高级功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedAnalytics:
    """高级分析类"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.anomaly_detector = None

    def customer_behavior_analysis(self, rfm_data, output_dir='./analysis_results'):
        """
        客户行为深度分析

        Args:
            rfm_data: RFM特征数据
            output_dir: 输出目录

        Returns:
            分析结果字典
        """
        print("🔍 开始客户行为分析...")

        results = {}

        # 1. 购买模式分析
        results['purchase_patterns'] = self._analyze_purchase_patterns(rfm_data)

        # 2. 客户生命周期阶段分析
        results['lifecycle_stages'] = self._analyze_lifecycle_stages(rfm_data)

        # 3. 价值分布分析
        results['value_distribution'] = self._analyze_value_distribution(rfm_data)

        # 4. 活跃度分析
        results['activity_analysis'] = self._analyze_activity_patterns(rfm_data)

        # 生成可视化
        self._plot_behavior_analysis(rfm_data, results, output_dir)

        print("✅ 客户行为分析完成")
        return results

    def _analyze_purchase_patterns(self, rfm_data):
        """分析购买模式"""
        print("  分析购买模式...")

        patterns = {}

        # 频率分布
        freq_dist = rfm_data['F值'].value_counts().sort_index()
        patterns['frequency_distribution'] = freq_dist.to_dict()

        # 金额分布
        amount_stats = {
            'mean': rfm_data['M值'].mean(),
            'median': rfm_data['M值'].median(),
            'std': rfm_data['M值'].std(),
            'min': rfm_data['M值'].min(),
            'max': rfm_data['M值'].max()
        }
        patterns['amount_statistics'] = amount_stats

        # 最近购买时间分布
        recency_stats = {
            'mean_days': rfm_data['R值'].mean(),
            'median_days': rfm_data['R值'].median(),
            'recent_customers': (rfm_data['R值'] <= 30).sum(),
            'inactive_customers': (rfm_data['R值'] > 90).sum()
        }
        patterns['recency_analysis'] = recency_stats

        return patterns

    def _analyze_lifecycle_stages(self, rfm_data):
        """分析客户生命周期阶段"""
        print("  分析生命周期阶段...")

        # 定义生命周期阶段
        def classify_lifecycle_stage(row):
            r, f, m = row['R值'], row['F值'], row['M值']

            if r <= 30 and f >= 3 and m >= m * 0.8:  # 高R值、高F值、高M值
                return "成熟期"
            elif r <= 60 and f >= 2:
                return "成长期"
            elif r <= 30 and f == 1:
                return "新客户期"
            elif r > 90 and f <= 2:
                return "衰退期"
            elif r > 180:
                return "流失期"
            else:
                return "稳定期"

        rfm_data['生命周期阶段'] = rfm_data.apply(classify_lifecycle_stage, axis=1)

        stage_counts = rfm_data['生命周期阶段'].value_counts()
        stage_percentages = (stage_counts / len(rfm_data) * 100).round(2)

        return {
            'stage_counts': stage_counts.to_dict(),
            'stage_percentages': stage_percentages.to_dict(),
            'stage_distribution': rfm_data.groupby('生命周期阶段')['年度LTV'].mean().to_dict()
        }

    def _analyze_value_distribution(self, rfm_data):
        """分析价值分布"""
        print("  分析价值分布...")

        ltv_stats = {
            'mean': rfm_data['年度LTV'].mean(),
            'median': rfm_data['年度LTV'].median(),
            'std': rfm_data['年度LTV'].std(),
            'percentiles': {
                '25%': rfm_data['年度LTV'].quantile(0.25),
                '50%': rfm_data['年度LTV'].quantile(0.50),
                '75%': rfm_data['年度LTV'].quantile(0.75),
                '90%': rfm_data['年度LTV'].quantile(0.90),
                '95%': rfm_data['年度LTV'].quantile(0.95)
            }
        }

        # 帕累托分析（80/20法则）
        sorted_ltv = rfm_data['年度LTV'].sort_values(ascending=False)
        total_ltv = sorted_ltv.sum()
        cumulative_ltv = sorted_ltv.cumsum() / total_ltv

        # 找到贡献80%价值的客户比例
        eighty_percent_idx = (cumulative_ltv >= 0.8).idxmax()
        top_customer_percentage = (eighty_percent_idx + 1) / len(rfm_data) * 100

        ltv_stats['pareto_analysis'] = {
            'top_20_percent_customers_contribute': cumulative_ltv.iloc[int(len(rfm_data) * 0.2) - 1] * 100,
            'customers_for_80_percent_value': top_customer_percentage
        }

        return ltv_stats

    def _analyze_activity_patterns(self, rfm_data):
        """分析活跃度模式"""
        print("  分析活跃度模式...")

        # 活跃度分类
        def classify_activity(row):
            r, f = row['R值'], row['F值']

            if r <= 30 and f >= 3:
                return "高活跃"
            elif r <= 60 and f >= 2:
                return "中等活跃"
            elif r <= 90 and f >= 1:
                return "低活跃"
            else:
                return "非活跃"

        rfm_data['活跃度'] = rfm_data.apply(classify_activity, axis=1)

        activity_counts = rfm_data['活跃度'].value_counts()
        activity_stats = rfm_data.groupby('活跃度')['年度LTV'].agg(['mean', 'count'])

        return {
            'activity_distribution': activity_counts.to_dict(),
            'activity_value_stats': activity_stats.to_dict()
        }

    def _plot_behavior_analysis(self, rfm_data, results, output_dir):
        """绘制行为分析图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("  生成行为分析图表...")

        # 1. RFM分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # R值分布
        axes[0, 0].hist(rfm_data['R值'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('最近购买时间分布 (R值)')
        axes[0, 0].set_xlabel('距离上次购买天数')
        axes[0, 0].set_ylabel('客户数')

        # F值分布
        axes[0, 1].hist(rfm_data['F值'], bins=10, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('购买频率分布 (F值)')
        axes[0, 1].set_xlabel('购买次数')
        axes[0, 1].set_ylabel('客户数')

        # M值分布
        axes[1, 0].hist(rfm_data['M值'], bins=15, alpha=0.7, color='salmon')
        axes[1, 0].set_title('消费金额分布 (M值)')
        axes[1, 0].set_xlabel('总消费金额')
        axes[1, 0].set_ylabel('客户数')

        # LTV分布
        axes[1, 1].hist(rfm_data['年度LTV'], bins=15, alpha=0.7, color='gold')
        axes[1, 1].set_title('年度LTV分布')
        axes[1, 1].set_xlabel('年度LTV')
        axes[1, 1].set_ylabel('客户数')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/behavior_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 生命周期阶段分布
        if '生命周期阶段' in rfm_data.columns:
            plt.figure(figsize=(10, 6))
            stage_counts = rfm_data['生命周期阶段'].value_counts()
            plt.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%')
            plt.title('客户生命周期阶段分布')
            plt.savefig(f'{output_dir}/lifecycle_stages.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 活跃度 vs LTV
        if '活跃度' in rfm_data.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=rfm_data, x='活跃度', y='年度LTV')
            plt.title('不同活跃度客户的LTV分布')
            plt.ylabel('年度LTV')
            plt.savefig(f'{output_dir}/activity_vs_ltv.png', dpi=300, bbox_inches='tight')
            plt.close()

    def advanced_customer_segmentation(self, rfm_data, n_clusters=5, output_dir='./analysis_results'):
        """
        高级客户细分（基于聚类算法）

        Args:
            rfm_data: RFM特征数据
            n_clusters: 聚类数量
            output_dir: 输出目录

        Returns:
            聚类结果
        """
        print("🎯 开始高级客户细分...")

        # 准备特征数据
        features = ['R值', 'F值', 'M值']
        X = rfm_data[features].copy()

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 寻找最佳聚类数量
        best_k = self._find_optimal_clusters(X_scaled, max_k=10)

        print(f"  最佳聚类数量: {best_k}")

        # 执行K-means聚类
        self.cluster_model = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(X_scaled)

        # 添加聚类标签到数据
        rfm_data_copy = rfm_data.copy()
        rfm_data_copy['聚类标签'] = cluster_labels

        # 分析聚类结果
        cluster_analysis = self._analyze_clusters(rfm_data_copy, features)

        # 生成可视化
        self._plot_clustering_results(rfm_data_copy, features, output_dir)

        print(f"✅ 客户细分完成，共识别{best_k}个客户群体")
        return {
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'optimal_clusters': best_k,
            'data_with_clusters': rfm_data_copy
        }

    def _find_optimal_clusters(self, X_scaled, max_k=10):
        """使用肘部法则和轮廓系数找最佳聚类数量"""
        print("  寻找最佳聚类数量...")

        inertias = []
        silhouette_scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))

        # 综合考虑肘部法则和轮廓系数
        # 选择轮廓系数最高的k值
        best_k = np.argmax(silhouette_scores) + 2

        return best_k

    def _analyze_clusters(self, data, features):
        """分析聚类特征"""
        print("  分析聚类特征...")

        cluster_analysis = {}

        for cluster_id in sorted(data['聚类标签'].unique()):
            cluster_data = data[data['聚类标签'] == cluster_id]

            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'feature_means': {},
                'ltv_stats': {
                    'mean': cluster_data['年度LTV'].mean(),
                    'median': cluster_data['年度LTV'].median(),
                    'std': cluster_data['年度LTV'].std()
                }
            }

            for feature in features:
                analysis['feature_means'][feature] = cluster_data[feature].mean()

            # 为聚类命名
            r_mean = analysis['feature_means']['R值']
            f_mean = analysis['feature_means']['F值']
            m_mean = analysis['feature_means']['M值']

            if r_mean <= 30 and f_mean >= 3 and m_mean >= 500:
                cluster_name = "高价值活跃客户"
            elif r_mean <= 60 and f_mean >= 2:
                cluster_name = "中价值潜力客户"
            elif r_mean > 90 and f_mean <= 2:
                cluster_name = "低价值流失风险客户"
            elif r_mean <= 30 and f_mean == 1:
                cluster_name = "新客户"
            else:
                cluster_name = f"客户群体{cluster_id + 1}"

            analysis['cluster_name'] = cluster_name
            cluster_analysis[cluster_id] = analysis

        return cluster_analysis

    def _plot_clustering_results(self, data, features, output_dir):
        """绘制聚类结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("  生成聚类可视化...")

        # 1. 聚类散点图 (使用PCA降维)
        X = data[features]
        X_scaled = self.scaler.transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['聚类标签'],
                            cmap='viridis', alpha=0.6, s=100)
        plt.colorbar(scatter)
        plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('客户聚类结果 (PCA降维)')
        plt.savefig(f'{output_dir}/customer_clusters_pca.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 聚类特征雷达图
        self._plot_cluster_radar_chart(data, features, output_dir)

        # 3. 聚类大小分布
        plt.figure(figsize=(10, 6))
        cluster_counts = data['聚类标签'].value_counts().sort_index()
        plt.bar(range(len(cluster_counts)), cluster_counts.values)
        plt.xlabel('聚类标签')
        plt.ylabel('客户数')
        plt.title('各聚类客户数量分布')
        plt.xticks(range(len(cluster_counts)), [f'聚类{i}' for i in cluster_counts.index])
        plt.savefig(f'{output_dir}/cluster_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cluster_radar_chart(self, data, features, output_dir):
        """绘制聚类特征雷达图"""
        from math import pi

        # 计算各聚类的平均特征值
        cluster_means = data.groupby('聚类标签')[features].mean()

        # 归一化到0-1范围
        normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

        # 雷达图设置
        angles = [n / float(len(features)) * 2 * pi for n in range(len(features))]
        angles += angles[:1]  # 闭合图形

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)

        # 为每个聚类绘制雷达图
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_means)))

        for i, (cluster_id, row) in enumerate(normalized_means.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, 'o-', linewidth=2, label=f'聚类{cluster_id}', color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # 设置标签
        plt.xticks(angles[:-1], features)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.title('客户聚类特征雷达图', size=16, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.savefig(f'{output_dir}/cluster_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

    def churn_prediction(self, rfm_data, output_dir='./analysis_results'):
        """
        客户流失预测分析

        Args:
            rfm_data: RFM特征数据
            output_dir: 输出目录

        Returns:
            流失预测结果
        """
        print("⚠️ 开始流失预测分析...")

        # 定义流失标签
        def define_churn(row):
            # 基于R值定义流失：90天未购买为流失
            return 1 if row['R值'] > 90 else 0

        rfm_data_copy = rfm_data.copy()
        rfm_data_copy['流失标签'] = rfm_data_copy.apply(define_churn, axis=1)

        # 流失统计分析
        churn_rate = rfm_data_copy['流失标签'].mean()
        print(f"  当前流失率: {churn_rate:.2%}")

        # 流失特征分析
        churn_features = self._analyze_churn_features(rfm_data_copy)

        # 高风险流失客户识别
        high_risk_customers = self._identify_high_risk_customers(rfm_data_copy)

        # 生成可视化
        self._plot_churn_analysis(rfm_data_copy, output_dir)

        print("✅ 流失预测分析完成")
        return {
            'churn_rate': churn_rate,
            'churn_features': churn_features,
            'high_risk_customers': high_risk_customers,
            'data_with_churn_labels': rfm_data_copy
        }

    def _analyze_churn_features(self, data):
        """分析流失相关特征"""
        churn_group = data.groupby('流失标签')[['R值', 'F值', 'M值']].mean()
        return churn_group.to_dict()

    def _identify_high_risk_customers(self, data, risk_threshold=0.7):
        """识别高风险流失客户"""
        # 基于R值和F值识别高风险客户
        high_risk = data[
            (data['R值'] > 60) |  # 60天未购买
            (data['F值'] <= 1)     # 只购买过一次
        ].sort_values('R值', ascending=False)

        return high_risk.head(20)  # 返回前20个高风险客户

    def _plot_churn_analysis(self, data, output_dir):
        """绘制流失分析图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. 流失vs非流失客户特征对比
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        features = ['R值', 'F值', 'M值']
        feature_names = ['最近购买', '购买频率', '消费金额']

        for i, (feature, name) in enumerate(zip(features, feature_names)):
            churn_0 = data[data['流失标签'] == 0][feature]
            churn_1 = data[data['流失标签'] == 1][feature]

            axes[i].hist([churn_0, churn_1], bins=20, alpha=0.7,
                        label=['非流失', '流失'], color=['green', 'red'])
            axes[i].set_title(f'{name}分布')
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('客户数')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/churn_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 流失风险散点图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data['R值'], data['F值'], c=data['流失标签'],
                            cmap='RdYlGn', alpha=0.6, s=100)
        plt.colorbar(scatter)
        plt.xlabel('最近购买天数 (R值)')
        plt.ylabel('购买频率 (F值)')
        plt.title('客户流失风险分布')
        plt.savefig(f'{output_dir}/churn_risk_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_insights(self, rfm_data, behavior_results,
                                      segmentation_results, churn_results,
                                      output_path='comprehensive_insights.md'):
        """生成综合洞察报告"""
        print("📋 生成综合洞察报告...")

        report = "# 客户生命周期价值综合洞察报告\n\n"
        report += f"生成时间: {pd.Timestamp.now()}\n\n"

        # 客户概览
        report += "## 客户概览\n\n"
        report += f"- 总客户数: {len(rfm_data):,}\n"
        report += f"- 平均年度LTV: {rfm_data['年度LTV'].mean():,.0f}\n"
        report += f"- LTV中位数: {rfm_data['年度LTV'].median():,.0f}\n"
        report += f"- 平均购买频率: {rfm_data['F值'].mean():.1f}次\n"
        report += f"- 平均最近购买: {rfm_data['R值'].mean():.0f}天前\n\n"

        # 行为分析洞察
        if behavior_results:
            report += "## 客户行为洞察\n\n"

            # 生命周期阶段
            if 'lifecycle_stages' in behavior_results:
                stages = behavior_results['lifecycle_stages']
                report += "### 生命周期阶段分布\n\n"
                for stage, percentage in stages['stage_percentages'].items():
                    report += f"- {stage}: {percentage}%\n"
                report += "\n"

        # 细分分析洞察
        if segmentation_results:
            report += "## 客户细分洞察\n\n"
            clusters = segmentation_results['cluster_analysis']

            for cluster_id, analysis in clusters.items():
                report += f"### {analysis['cluster_name']}\n\n"
                report += f"- 客户数量: {analysis['size']} ({analysis['percentage']:.1f}%)\n"
                report += f"- 平均LTV: {analysis['ltv_stats']['mean']:,.0f}\n"
                report += f"- 特征特征: R值{analysis['feature_means']['R值']:.1f}, "
                report += f"F值{analysis['feature_means']['F值']:.1f}, "
                report += f"M值{analysis['feature_means']['M值']:.0f}\n\n"

        # 流失分析洞察
        if churn_results:
            report += "## 流失风险洞察\n\n"
            report += f"当前流失率: {churn_results['churn_rate']:.1%}\n\n"

            high_risk = churn_results['high_risk_customers']
            if not high_risk.empty:
                report += f"高风险客户数: {len(high_risk)}\n"
                report += "主要风险因素:\n"
                report += "- 长时间未购买 (R值 > 60天)\n"
                report += "- 购买频率低 (F值 ≤ 1)\n\n"

        # 策略建议
        report += "## 营销策略建议\n\n"
        report += "基于上述分析，建议采取以下策略:\n\n"

        if segmentation_results:
            # 基于聚类结果的建议
            clusters = segmentation_results['cluster_analysis']
            for cluster_id, analysis in clusters.items():
                cluster_name = analysis['cluster_name']

                if '高价值' in cluster_name:
                    report += f"### {cluster_name}\n"
                    report += "- 提供VIP专属服务\n"
                    report += "- 安排客户经理专人对接\n"
                    report += "- 定制个性化营销方案\n\n"
                elif '潜力' in cluster_name:
                    report += f"### {cluster_name}\n"
                    report += "- 推荐升级产品和套餐\n"
                    report += "- 提供会员激励计划\n"
                    report += "- 增加互动频率\n\n"
                elif '流失风险' in cluster_name:
                    report += f"### {cluster_name}\n"
                    report += "- 发放召回优惠券\n"
                    report += "- 推荐性价比高的产品\n"
                    report += "- 主动关怀和沟通\n\n"

        if churn_results and churn_results['churn_rate'] > 0.2:
            report += "### 流失预防策略\n"
            report += "- 建立流失预警机制\n"
            report += "- 实施客户关怀计划\n"
            report += "- 优化产品和服务体验\n\n"

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 综合洞察报告已保存: {output_path}")
        return output_path