#!/usr/bin/env python3
"""
LTV预测引擎
整合数据预处理、RFM分析和回归建模的完整LTV预测流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from regression_models import RegressionModels

class LTVPredictor:
    """
    LTV预测引擎

    整合数据预处理、RFM分析和回归建模的完整流程
    基于第3课理论实现的客户生命周期价值预测系统
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化LTV预测器

        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            'data_processor_config': {
                'feature_period_months': 3,
                'prediction_period_months': 12,
                'remove_outliers': True,
                'min_orders_per_customer': 1
            },
            'regression_config': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scoring_metric': 'r2',
                'enable_hyperparameter_tuning': False
            },
            'feature_columns': ['R值', 'F值', 'M值'],
            'target_column': '年度LTV',
            'models_to_train': ['linear_regression', 'random_forest'],
            'customer_column': '用户码'
        }

        # 更新配置
        if config:
            self.config.update(config)
            if 'data_processor_config' in config:
                self.config['data_processor_config'].update(config['data_processor_config'])
            if 'regression_config' in config:
                self.config['regression_config'].update(config['regression_config'])

        # 初始化组件
        self.data_processor = DataProcessor(self.config['data_processor_config'])
        self.regression_models = RegressionModels(self.config['regression_config'])

        # 结果存储
        self.training_results = None
        self.model_results = None
        self.predictions = None
        self.feature_importance = None
        self.summary_report = None

        print("🚀 LTV预测引擎初始化完成")

    def load_and_preprocess_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        加载和预处理数据

        Args:
            file_path: 数据文件路径
            **kwargs: 加载参数

        Returns:
            预处理后的数据
        """
        print("📁 开始数据加载和预处理...")

        # 加载数据
        raw_data = self.data_processor.load_order_data(file_path, **kwargs)

        # 预处理数据
        processed_data = self.data_processor.preprocess_data(raw_data)

        return processed_data

    def calculate_rfm_and_prepare_training_data(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算RFM特征并准备训练数据

        Args:
            processed_data: 预处理后的订单数据

        Returns:
            包含RFM特征和LTV标签的训练数据
        """
        print("🔍 开始RFM特征计算和训练数据准备...")

        # 计算RFM特征
        rfm_data = self.data_processor.calculate_rfm_features(processed_data)

        # 客户分群
        segmented_data = self.data_processor.segment_customers(rfm_data)

        # 获取RFM摘要
        rfm_summary = self.data_processor.get_rfm_summary(segmented_data)
        self.rfm_summary = rfm_summary

        print(f"✓ RFM分析完成:")
        print(f"  - 总客户数: {rfm_summary['total_customers']}")
        print(f"  - R值均值: {rfm_summary['rfm_statistics']['R值']['mean']:.2f}")
        print(f"  - F值均值: {rfm_summary['rfm_statistics']['F值']['mean']:.2f}")
        print(f"  - M值均值: {rfm_summary['rfm_statistics']['M值']['mean']:.2f}")
        print(f"  - 年度LTV均值: {rfm_summary['rfm_statistics']['年度LTV']['mean']:.2f}")

        return segmented_data

    def train_models(self, rfm_data: pd.DataFrame, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练LTV预测模型

        Args:
            rfm_data: RFM特征数据
            model_names: 要训练的模型列表

        Returns:
            训练结果字典
        """
        print("🤖 开始模型训练...")

        if model_names is None:
            model_names = self.config['models_to_train']

        # 准备训练数据
        feature_columns = self.config['feature_columns']
        target_column = self.config['target_column']

        X, y = self.regression_models.prepare_data(rfm_data, feature_columns, target_column)
        X_train, X_test, y_train, y_test = self.regression_models.split_data(X, y)

        # 训练多个模型
        model_results = self.regression_models.train_multiple_models(
            model_names, X_train, y_train, X_test, y_test
        )

        self.model_results = model_results
        self.training_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }

        # 提取特征重要性
        self.feature_importance = self.regression_models.feature_importance

        return {
            'model_results': model_results,
            'training_data': self.training_data,
            'feature_importance': self.feature_importance,
            'best_model_name': self.regression_models.best_model_name,
            'rfm_summary': self.rfm_summary
        }

    def predict_ltv(self, data: Union[pd.DataFrame, Dict[str, float]],
                    model_name: Optional[str] = None) -> Union[np.ndarray, float]:
        """
        预测客户LTV

        Args:
            data: 客户数据（DataFrame或单个客户字典）
            model_name: 使用的模型名称

        Returns:
            LTV预测结果
        """
        if self.model_results is None:
            raise ValueError("模型未训练，请先调用train_models")

        if isinstance(data, dict):
            # 单个客户预测
            return self._predict_single_customer(data, model_name)
        else:
            # 批量预测
            return self._predict_batch(data, model_name)

    def _predict_single_customer(self, customer_data: Dict[str, float], model_name: Optional[str] = None) -> float:
        """
        预测单个客户的LTV

        Args:
            customer_data: 客户RFM特征字典
            model_name: 模型名称

        Returns:
            LTV预测值
        """
        # 准备特征数据
        feature_columns = self.config['feature_columns']
        features = [customer_data.get(col, 0) for col in feature_columns]

        # 转换为DataFrame
        X = pd.DataFrame([features], columns=feature_columns)

        # 预测
        prediction = self.regression_models.predict(model_name, X)

        return float(prediction[0])

    def _predict_batch(self, customer_data: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        批量预测客户LTV

        Args:
            customer_data: 客户RFM特征DataFrame
            model_name: 模型名称

        Returns:
            LTV预测数组
        """
        # 确保包含所需特征列
        feature_columns = self.config['feature_columns']
        missing_cols = [col for col in feature_columns if col not in customer_data.columns]

        if missing_cols:
            raise ValueError(f"数据缺少特征列: {missing_cols}")

        X = customer_data[feature_columns]

        # 预测
        predictions = self.regression_models.predict(model_name, X)

        return predictions

    def predict_new_customers(self, customer_orders: pd.DataFrame,
                            model_name: Optional[str] = None) -> pd.DataFrame:
        """
        为新客户预测LTV

        Args:
            customer_orders: 新客户的订单数据
            model_name: 使用的模型名称

        Returns:
            包含LTV预测的客户数据
        """
        print("🔮 为新客户预测LTV...")

        # 使用相同的配置计算RFM特征
        temp_processor = DataProcessor(self.config['data_processor_config'])

        # 预处理新客户数据
        processed_new_data = temp_processor.preprocess_data(customer_orders)

        # 计算RFM特征（仅特征期）
        feature_config = {
            'feature_period_months': self.config['data_processor_config']['feature_period_months'],
            'prediction_period_months': 0  # 不需要LTV标签
        }

        # 临时修改配置
        original_config = temp_processor.config.copy()
        temp_processor.config.update(feature_config)

        try:
            # 计算RFM特征
            new_rfm = temp_processor.calculate_rfm_features(processed_new_data)
        except:
            # 如果失败，使用简化的RFM计算
            print("  使用简化RFM计算...")
            new_rfm = self._simple_rfm_calculation(processed_new_data)
        finally:
            # 恢复配置
            temp_processor.config = original_config

        # 预测LTV
        feature_columns = self.config['feature_columns']
        X_new = new_rfm[feature_columns]
        predictions = self.regression_models.predict(model_name, X_new)

        # 添加预测结果
        new_rfm['预测LTV'] = predictions
        new_rfm['预测时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"✓ 完成对{len(new_rfm)}个新客户的LTV预测")
        print(f"  - 平均预测LTV: {predictions.mean():.2f}")
        print(f"  - 预测范围: {predictions.min():.2f} ~ {predictions.max():.2f}")

        return new_rfm

    def _simple_rfm_calculation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        简化的RFM计算（用于新客户）

        Args:
            data: 订单数据

        Returns:
            RFM特征数据
        """
        customer_col = self.config['customer_column']
        date_col = self.data_processor.config['date_column']

        # 获取唯一客户
        unique_customers = data[customer_col].unique()
        rfm_data = pd.DataFrame({customer_col: unique_customers})

        # 计算R值（距今天数）
        latest_date = data[date_col].max()
        r_data = data.groupby(customer_col)[date_col].max().reset_index()
        r_data['R值'] = (latest_date - r_data[date_col]).dt.days
        rfm_data = rfm_data.merge(r_data[[customer_col, 'R值']], on=customer_col, how='left')

        # 计算F值
        f_data = data.groupby(customer_col)[date_col].count().reset_index()
        f_data.columns = [customer_col, 'F值']
        rfm_data = rfm_data.merge(f_data, on=customer_col, how='left')

        # 计算M值
        if '总价' not in data.columns:
            data['总价'] = data[self.data_processor.config['quantity_column']] * data[self.data_processor.config['price_column']]

        m_data = data.groupby(customer_col)['总价'].sum().reset_index()
        m_data.columns = [customer_col, 'M值']
        rfm_data = rfm_data.merge(m_data, on=customer_col, how='left')

        return rfm_data

    def evaluate_prediction_accuracy(self, test_data: pd.DataFrame,
                                   actual_ltv_column: str = '年度LTV') -> Dict[str, float]:
        """
        评估预测准确性

        Args:
            test_data: 测试数据
            actual_ltv_column: 实际LTV列名

        Returns:
            评估指标字典
        """
        if self.model_results is None:
            raise ValueError("模型未训练，请先调用train_models")

        # 预测测试集LTV
        feature_columns = self.config['feature_columns']
        X_test = test_data[feature_columns]
        y_true = test_data[actual_ltv_column]

        # 使用最佳模型预测
        y_pred = self.regression_models.predict(self.regression_models.best_model_name, X_test)

        # 计算评估指标
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        evaluation_metrics = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mean_actual_ltv': y_true.mean(),
            'mean_predicted_ltv': y_pred.mean(),
            'total_customers': len(test_data)
        }

        print(f"📊 预测准确性评估:")
        print(f"  - R² 分数: {r2:.4f}")
        print(f"  - 平均绝对误差: {mae:.2f}")
        print(f"  - 均方根误差: {rmse:.2f}")
        print(f"  - 平均绝对百分比误差: {mape:.2f}%")
        print(f"  - 实际平均LTV: {y_true.mean():.2f}")
        print(f"  - 预测平均LTV: {y_pred.mean():.2f}")

        return evaluation_metrics

    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        获取特征分析报告

        Returns:
            特征分析字典
        """
        if self.feature_importance is None:
            return {"error": "特征重要性未计算，请先训练模型"}

        analysis = {
            'feature_importance': self.feature_importance,
            'best_model_features': {},
            'feature_insights': {}
        }

        # 分析最佳模型的特征重要性
        if self.regression_models.best_model_name in self.feature_importance:
            best_features = self.feature_importance[self.regression_models.best_model_name]
            analysis['best_model_features'] = best_features

            # 生成特征洞察
            if best_features:
                top_feature = max(best_features.keys(), key=lambda x: best_features[x])
                analysis['feature_insights'] = {
                    'most_important_feature': top_feature,
                    'most_important_score': best_features[top_feature],
                    'feature_ranking': dict(sorted(best_features.items(), key=lambda x: x[1], reverse=True)),
                    'feature_contribution_analysis': self._analyze_feature_contributions(best_features)
                }

        return analysis

    def _analyze_feature_contributions(self, feature_importance: Dict[str, float]) -> Dict[str, str]:
        """
        分析特征贡献

        Args:
            feature_importance: 特征重要性字典

        Returns:
            特征贡献分析字典
        """
        total_importance = sum(feature_importance.values())
        contributions = {}

        for feature, importance in feature_importance.items():
            contribution_pct = (importance / total_importance) * 100

            if feature == 'R值':
                contributions[feature] = f"最近消费时间贡献了{contribution_pct:.1f}%的预测信息，客户活跃度对LTV影响显著"
            elif feature == 'F值':
                contributions[feature] = f"消费频率贡献了{contribution_pct:.1f}%的预测信息，频繁购买客户更有价值"
            elif feature == 'M值':
                contributions[feature] = f"消费金额贡献了{contribution_pct:.1f}%的预测信息，历史消费金额是LTV的关键指标"
            else:
                contributions[feature] = f"该特征贡献了{contribution_pct:.1f}%的预测信息"

        return contributions

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        生成综合摘要报告

        Returns:
            摘要报告字典
        """
        if self.model_results is None:
            return {"error": "模型未训练，请先运行完整流程"}

        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'data_summary': self.rfm_summary,
            'model_summary': self.regression_models.get_model_summary(),
            'best_model_performance': {},
            'feature_analysis': self.get_feature_analysis(),
            'recommendations': []
        }

        # 最佳模型性能
        if self.regression_models.best_model_name and self.regression_models.best_model_name in self.model_results:
            best_result = self.model_results[self.regression_models.best_model_name]
            if 'r2_score' in best_result:
                report['best_model_performance'] = {
                    'model_name': self.regression_models.best_model_name,
                    'r2_score': best_result['r2_score'],
                    'mae': best_result.get('mae'),
                    'rmse': best_result.get('rmse'),
                    'mape': best_result.get('mape')
                }

        # 生成业务建议
        report['recommendations'] = self._generate_business_recommendations(report)

        self.summary_report = report
        return report

    def _generate_business_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        生成业务建议

        Args:
            report: 分析报告

        Returns:
            建议列表
        """
        recommendations = []

        try:
            # 基于模型性能的建议
            best_performance = report.get('best_model_performance', {})
            r2_score = best_performance.get('r2_score', 0)

            if r2_score > 0.7:
                recommendations.append("模型预测性能优秀（R² > 0.7），可用于精准营销和客户价值管理")
            elif r2_score > 0.5:
                recommendations.append("模型预测性能良好（R² > 0.5），建议结合业务规则进行客户分层")
            else:
                recommendations.append("模型预测性能一般，建议增加更多特征数据或尝试其他算法")

            # 基于特征重要性的建议
            feature_analysis = report.get('feature_analysis', {})
            if 'most_important_feature' in feature_analysis:
                top_feature = feature_analysis['most_important_feature']
                if top_feature == 'M值':
                    recommendations.append("消费金额是影响LTV的最重要因素，建议重点提升客单价")
                elif top_feature == 'F值':
                    recommendations.append("消费频率是影响LTV的最重要因素，建议重点提高复购率")
                elif top_feature == 'R值':
                    recommendations.append("客户活跃度是影响LTV的最重要因素，建议加强客户互动和唤醒")

            # 基于数据的建议
            data_summary = report.get('data_summary', {})
            if data_summary and 'total_customers' in data_summary:
                customer_count = data_summary['total_customers']
                if customer_count < 100:
                    recommendations.append("样本量较少，建议积累更多数据以提高模型稳定性")
                elif customer_count > 10000:
                    recommendations.append("样本量充足，可考虑更复杂的机器学习算法")

        except Exception as e:
            print(f"生成建议时出错: {str(e)}")

        if not recommendations:
            recommendations.append("模型训练完成，建议结合具体业务场景进行应用")

        return recommendations

    def save_results(self, directory: str):
        """
        保存分析结果

        Args:
            directory: 保存目录
        """
        import os
        from pathlib import Path

        Path(directory).mkdir(parents=True, exist_ok=True)

        # 保存模型
        if self.model_results:
            model_dir = os.path.join(directory, "models")
            self.regression_models.save_models(model_dir)

        # 保存摘要报告
        if self.summary_report:
            report_path = os.path.join(directory, "summary_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.summary_report, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ 分析结果已保存到: {directory}")

    def load_results(self, directory: str):
        """
        加载分析结果

        Args:
            directory: 结果目录
        """
        import os

        # 加载模型
        model_dir = os.path.join(directory, "models")
        if os.path.exists(model_dir):
            self.regression_models.load_models(model_dir)

        # 加载摘要报告
        report_path = os.path.join(directory, "summary_report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                self.summary_report = json.load(f)

        print(f"✓ 分析结果已从{directory}加载")

# 便利函数
def complete_ltv_analysis(file_path: str,
                         output_dir: str = './ltv_results',
                         config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    完整的LTV分析流程

    Args:
        file_path: 订单数据文件路径
        output_dir: 输出目录
        config: 配置参数

    Returns:
        完整分析结果
    """
    from pathlib import Path

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化预测器
    predictor = LTVPredictor(config)

    print("🚀 开始完整LTV分析流程...")
    print(f"  - 输入数据: {file_path}")
    print(f"  - 输出目录: {output_dir}")

    try:
        # 1. 加载和预处理数据
        processed_data = predictor.load_and_preprocess_data(file_path)

        # 2. 计算RFM特征
        rfm_data = predictor.calculate_rfm_and_prepare_training_data(processed_data)

        # 3. 训练模型
        training_results = predictor.train_models(rfm_data)

        # 4. 生成摘要报告
        summary_report = predictor.generate_summary_report()

        # 5. 保存结果
        predictor.save_results(output_dir)

        # 6. 导出RFM数据
        rfm_output_path = os.path.join(output_dir, 'rfm_features.csv')
        predictor.data_processor.export_rfm_data(rfm_data, rfm_output_path)

        results = {
            'predictor': predictor,
            'rfm_data': rfm_data,
            'training_results': training_results,
            'summary_report': summary_report,
            'output_paths': {
                'rfm_features': rfm_output_path,
                'models': os.path.join(output_dir, 'models'),
                'summary_report': os.path.join(output_dir, 'summary_report.json')
            }
        }

        print("🎉 完整LTV分析流程完成！")

        # 打印关键结果
        if summary_report and 'best_model_performance' in summary_report:
            best_perf = summary_report['best_model_performance']
            print(f"\n📊 关键结果:")
            print(f"  - 最佳模型: {best_perf.get('model_name', 'Unknown')}")
            print(f"  - 模型R²: {best_perf.get('r2_score', 0):.4f}")
            print(f"  - 分析客户数: {training_results.get('rfm_summary', {}).get('total_customers', 0)}")

        return results

    except Exception as e:
        print(f"❌ LTV分析流程失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 示例使用
    print("🎯 LTV预测引擎测试")

    # 如果有示例数据文件，可以进行测试
    sample_file = '../data/sample_orders.csv'
    if os.path.exists(sample_file):
        results = complete_ltv_analysis(sample_file, output_dir='./ltv_test_results')
        print("✓ 完整测试完成")
    else:
        print("⚠️ 示例数据文件不存在，跳过测试")