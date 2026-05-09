"""
Uplift建模模块 - 增长模型核心机器学习组件

提供先进的Uplift建模功能，包括：
- XGBoost Uplift建模
- 增量分数计算
- Qini曲线分析
- 模型效果评估
- 策略效果预测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


class UpliftModeler:
    """Uplift建模器 - 增长模型机器学习核心"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化Uplift建模器

        Parameters:
        - config: 配置参数字典
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}

    def prepare_uplift_data(self,
                           data: pd.DataFrame,
                           treatment_col: str = '裂变类型',
                           outcome_col: str = '是否转化',
                           control_value: str = '无裂变页面',
                           treatment_value: str = None) -> pd.DataFrame:
        """
        准备Uplift建模数据

        Parameters:
        - data: 原始数据
        - treatment_col: 处理组列名
        - outcome_col: 结果列名
        - control_value: 对照组值
        - treatment_value: 处理组值 (如果为None，使用所有非对照组)

        Returns:
        - 准备好的Uplift数据
        """
        # 筛选相关数据
        if treatment_value:
            uplift_data = data[data[treatment_col].isin([control_value, treatment_value])].copy()
        else:
            # 使用所有非对照组作为处理组
            uplift_data = data[data[treatment_col].isin([control_value]) | ~data[treatment_col].isin([control_value])].copy()

        # 构造Uplift标签
        def create_uplift_label(row):
            if row[treatment_col] == control_value:
                # 对照组
                return 2 if row[outcome_col] == 1 else 3  # CR, CN
            else:
                # 处理组
                return 0 if row[outcome_col] == 1 else 1  # TR, TN

        uplift_data['uplift_label'] = uplift_data.apply(create_uplift_label, axis=1)

        print(f"✅ Uplift数据准备完成: {len(uplift_data)} 条记录")
        print(f"   - 处理组: {len(uplift_data[uplift_data[treatment_col] != control_value])} 条")
        print(f"   - 对照组: {len(uplift_data[uplift_data[treatment_col] == control_value])} 条")

        return uplift_data

    def build_uplift_model(self,
                          data: pd.DataFrame,
                          feature_cols: List[str] = None,
                          treatment_col: str = '裂变_type',
                          outcome_col: str = '是否转化',
                          model_type: str = 'xgboost',
                          test_size: float = 0.2,
                          random_state: int = 42) -> Dict:
        """
        构建Uplift模型

        Parameters:
        - data: 准备好的Uplift数据
        - feature_cols: 特征列名列表
        - treatment_col: 处理组编码列名
        - outcome_col: 结果列名
        - model_type: 模型类型 ('xgboost', 'lightgbm')
        - test_size: 测试集比例
        - random_state: 随机种子

        Returns:
        - 模型训练结果
        """
        # 确定特征列
        if feature_cols is None:
            # 排除标识符和结果列
            exclude_cols = [treatment_col, outcome_col, 'uplift_label', '用户码', '裂变类型']
            feature_cols = [col for col in data.columns if col not in exclude_cols]

        self.feature_names = feature_cols

        # 准备训练数据
        X = data[feature_cols]
        y = data['uplift_label']

        # 处理分类变量
        X = pd.get_dummies(X, drop_first=True)

        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=4,
                random_state=random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        print(f"🚀 开始训练{model_type.upper()} Uplift模型...")
        model.fit(X_train_scaled, y_train)
        print("✅ 模型训练完成")

        # 保存模型和标准化器
        self.models['uplift'] = model
        self.scalers['uplift'] = scaler

        # 模型评估
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'accuracy': np.mean(y_pred == y_test)
        }

        self.results['uplift_model'] = results
        return results

    def calculate_uplift_scores(self,
                              data: pd.DataFrame,
                              treatment_col: str = '裂变类型',
                              outcome_col: str = '是否转化',
                              model_name: str = 'uplift') -> pd.DataFrame:
        """
        计算增量分数

        Parameters:
        - data: 测试数据
        - treatment_col: 处理组列名
        - outcome_col: 结果列名
        - model_name: 模型名称

        Returns:
        - 包含增量分数的DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在，请先训练模型")

        model = self.models[model_name]
        scaler = self.scalers[model_name]

        # 准备特征数据
        feature_data = data[self.feature_names]
        feature_data = pd.get_dummies(feature_data, drop_first=True)

        # 确保特征列匹配
        for col in self.feature_names:
            if col not in feature_data.columns:
                feature_data[col] = 0

        feature_data = feature_data[self.feature_names]

        # 标准化
        X_scaled = scaler.transform(feature_data)

        # 预测概率
        probas = model.predict_proba(X_scaled)

        # P_TR: Treatment Responders (处理组响应者)
        # P_TN: Treatment Non-responders (处理组非响应者)
        # P_CR: Control Responders (对照组响应者)
        # P_CN: Control Non-responders (对照组非响应者)
        P_TR, P_TN, P_CR, P_CN = probas[:, 0], probas[:, 1], probas[:, 2], probas[:, 3]

        # 计算增量分数 (Uplift Score)
        # 增量分数 = (P_TR - P_TN) + (P_CN - P_CR)
        epsilon = 1e-15  # 避免除零
        uplift_score = (P_TR - P_TN) / (P_TR + P_TN + epsilon) + (P_CN - P_CR) / (P_CN + P_CR + epsilon)

        # 构建结果DataFrame
        results_df = data[['用户码', treatment_col, outcome_col]].copy()
        results_df['P_TR'] = P_TR
        results_df['P_TN'] = P_TN
        results_df['P_CR'] = P_CR
        results_df['P_CN'] = P_CN
        results_df['uplift_score'] = uplift_score

        print(f"✅ 增量分数计算完成")
        print(f"   - 平均增量分数: {uplift_score.mean():.4f}")
        print(f"   - 标准差: {uplift_score.std():.4f}")

        return results_df

    def analyze_qini_curve(self,
                          uplift_scores_df: pd.DataFrame,
                          treatment_col: str = '裂变类型',
                          outcome_col: str = '是否转化',
                          control_value: str = '无裂变页面') -> Dict:
        """
        分析Qini曲线

        Parameters:
        - uplift_scores_df: 包含增量分数的DataFrame
        - treatment_col: 处理组列名
        - outcome_col: 结果列名
        - control_value: 对照组值

        Returns:
        - Qini曲线分析结果
        """
        # 按增量分数降序排序
        df_sorted = uplift_scores_df.sort_values('uplift_score', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1
        df_sorted['fraction'] = df_sorted['rank'] / len(df_sorted)

        # 计算TR和CR总数
        tr_total = len(df_sorted[(df_sorted[treatment_col] != control_value) & (df_sorted[outcome_col] == 1)])
        cr_total = len(df_sorted[(df_sorted[treatment_col] == control_value) & (df_sorted[outcome_col] == 1)])

        if tr_total == 0 or cr_total == 0:
            raise ValueError("数据中缺少足够的转化样本")

        # 计算累积指标
        df_sorted['is_tr'] = ((df_sorted[treatment_col] != control_value) & (df_sorted[outcome_col] == 1)).astype(int)
        df_sorted['is_cr'] = ((df_sorted[treatment_col] == control_value) & (df_sorted[outcome_col] == 1)).astype(int)

        df_sorted['cum_tr'] = df_sorted['is_tr'].cumsum()
        df_sorted['cum_cr'] = df_sorted['is_cr'].cumsum()

        # 计算Qini增益
        df_sorted['qini_gain'] = (df_sorted['cum_tr'] / tr_total) - (df_sorted['cum_cr'] / cr_total)

        # 计算随机模型的Qini增益
        df_sorted['random_qini'] = df_sorted['fraction'] * df_sorted['qini_gain'].iloc[-1]

        # 计算AUC
        qini_auc = np.trapz(df_sorted['qini_gain'], df_sorted['fraction'])
        random_auc = np.trapz(df_sorted['random_qini'], df_sorted['fraction'])
        auqc = qini_auc - random_auc  # Adjusted Uplift Qini Curve

        results = {
            'qini_data': df_sorted[['fraction', 'qini_gain', 'random_qini']].to_dict(),
            'qini_auc': qini_auc,
            'random_auc': random_auc,
            'auqc': auqc,
            'total_tr': tr_total,
            'total_cr': cr_total,
            'model_performance': self._evaluate_qini_performance(auqc)
        }

        print(f"✅ Qini曲线分析完成")
        print(f"   - Qini AUC: {qini_auc:.4f}")
        print(f"   - Random AUC: {random_auc:.4f}")
        print(f"   - AUQC: {auqc:.4f}")
        print(f"   - 模型表现: {results['model_performance']}")

        return results

    def uplift_segmentation(self,
                           uplift_scores_df: pd.DataFrame,
                           n_segments: int = 5) -> pd.DataFrame:
        """
        基于增量分数的用户分群

        Parameters:
        - uplift_scores_df: 包含增量分数的DataFrame
        - n_segments: 分群数量

        Returns:
        - 包含分群结果的DataFrame
        """
        # 按增量分数分群
        df_segmented = uplift_scores_df.copy()

        # 创建分群标签
        quantiles = np.linspace(0, 1, n_segments + 1)
        cutoffs = df_segmented['uplift_score'].quantile(quantiles).tolist()

        segment_labels = []
        for i in range(n_segments):
            if i == 0:
                label = f"Bottom {100//n_segments}%"
            elif i == n_segments - 1:
                label = f"Top {100//n_segments}%"
            else:
                label = f"Q{i+1}"
            segment_labels.append(label)

        # 分配分群
        df_segmented['uplift_segment'] = pd.cut(
            df_segmented['uplift_score'],
            bins=cutoffs,
            labels=segment_labels,
            include_lowest=True
        )

        # 计算分群统计
        segment_stats = df_segmented.groupby('uplift_segment').agg({
            'uplift_score': ['count', 'mean', 'std'],
            '用户码': 'nunique'
        }).round(4)

        segment_stats.columns = ['用户数', '平均增量分数', '标准差', '独立用户数']

        print(f"✅ 用户分群完成")
        for segment, stats in segment_stats.iterrows():
            print(f"   - {segment}: {stats['用户数']} 用户, 平均增量分数 {stats['平均增量分数']:.4f}")

        return df_segmented

    def save_model(self, model_name: str = 'uplift', filepath: str = None) -> str:
        """
        保存模型

        Parameters:
        - model_name: 模型名称
        - filepath: 保存路径

        Returns:
        - 保存的文件路径
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")

        if filepath is None:
            filepath = f'uplift_model_{model_name}.pkl'

        # 保存模型和标准化器
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers[model_name],
            'feature_names': self.feature_names,
            'config': self.config
        }

        joblib.dump(model_data, filepath)
        print(f"✅ 模型已保存到: {filepath}")

        return filepath

    def load_model(self, filepath: str, model_name: str = 'uplift') -> Dict:
        """
        加载模型

        Parameters:
        - filepath: 模型文件路径
        - model_name: 模型名称

        Returns:
        - 加载的模型数据
        """
        model_data = joblib.load(filepath)

        self.models[model_name] = model_data['model']
        self.scalers[model_name] = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', {})

        print(f"✅ 模型已从 {filepath} 加载")
        return model_data

    def plot_qini_curve(self, qini_results: Dict, title: str = "Qini Curve Analysis"):
        """
        绘制Qini曲线

        Parameters:
        - qini_results: Qini曲线分析结果
        - title: 图表标题
        """
        plt.figure(figsize=(10, 6))

        qini_data = qini_results['qini_data']
        fractions = qini_data['fraction']
        qini_gains = qini_data['qini_gain']
        random_qini = qini_data['random_qini']

        plt.plot(fractions, qini_gains, label='Uplift Model', linewidth=2, color='blue')
        plt.plot(fractions, random_qini, label='Random', linewidth=2, color='gray', linestyle='--')

        # 填充区域
        plt.fill_between(fractions, qini_gains, random_qini, alpha=0.3, color='lightblue')

        plt.xlabel('Population Fraction', fontsize=12)
        plt.ylabel('Cumulative Uplift', fontsize=12)
        plt.title(f'{title}\nAUQC: {qini_results["auqc"]:.4f}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gcf()

    def _evaluate_qini_performance(self, auqc: float) -> str:
        """评估Qini曲线表现"""
        if auqc > 0.1:
            return "优秀"
        elif auqc > 0.05:
            return "良好"
        elif auqc > 0:
            return "一般"
        else:
            return "较差"

    def generate_uplift_report(self, uplift_scores_df: pd.DataFrame) -> Dict:
        """
        生成Uplift分析报告

        Parameters:
        - uplift_scores_df: 包含增量分数的DataFrame

        Returns:
        - 分析报告
        """
        report = {
            "summary": {},
            "insights": [],
            "recommendations": []
        }

        # 基础统计
        report["summary"] = {
            "total_users": len(uplift_scores_df),
            "avg_uplift_score": uplift_scores_df['uplift_score'].mean(),
            "uplift_score_std": uplift_scores_df['uplift_score'].std(),
            "positive_uplift_ratio": (uplift_scores_df['uplift_score'] > 0).mean()
        }

        # 生成洞察
        positive_ratio = report["summary"]["positive_uplift_ratio"]
        if positive_ratio > 0.7:
            report["insights"].append("大部分用户对策略有积极响应")
            report["recommendations"].append("建议扩大策略覆盖范围")
        elif positive_ratio > 0.4:
            report["insights"].append("约半数用户对策略有响应")
            report["recommendations"].append("建议优化目标用户选择")
        else:
            report["insights"].append("策略响应率偏低")
            report["recommendations"].append("建议重新设计策略或调整目标用户")

        # 增量分数分布分析
        high_uplift_threshold = uplift_scores_df['uplift_score'].quantile(0.8)
        high_uplift_users = uplift_scores_df[uplift_scores_df['uplift_score'] >= high_uplift_threshold]

        report["insights"].append(
            f"前20%高增量用户平均增量分数: {high_uplift_users['uplift_score'].mean():.4f}"
        )
        report["recommendations"].append(
            "重点针对高增量分数用户进行营销投放"
        )

        return report