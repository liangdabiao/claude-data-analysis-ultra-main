"""
推荐系统评估器

提供全面的推荐系统评估功能：
- 离线评估指标 (Precision@K, Recall@K, MAE, RMSE)
- 多种评估方法 (留一法, K折交叉验证, 时间序列验证)
- 综合评估报告生成
- 算法性能对比分析
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class RecommenderEvaluator:
    """推荐系统评估器类"""

    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = {}

    def precision_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """
        计算Precision@K指标

        Args:
            recommendations: 推荐商品列表
            ground_truth: 真实相关商品列表
            k: 推荐列表长度

        Returns:
            Precision@K值
        """
        if k <= 0 or not recommendations:
            return 0.0

        # 取前k个推荐
        top_k_recommendations = set(recommendations[:k])
        relevant_items = set(ground_truth)

        # 计算交集
        hits = len(top_k_recommendations.intersection(relevant_items))

        return hits / min(k, len(recommendations))

    def recall_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """
        计算Recall@K指标

        Args:
            recommendations: 推荐商品列表
            ground_truth: 真实相关商品列表
            k: 推荐列表长度

        Returns:
            Recall@K值
        """
        if not ground_truth:
            return 0.0

        if k <= 0 or not recommendations:
            return 0.0

        # 取前k个推荐
        top_k_recommendations = set(recommendations[:k])
        relevant_items = set(ground_truth)

        # 计算交集
        hits = len(top_k_recommendations.intersection(relevant_items))

        return hits / len(relevant_items)

    def f1_score_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """
        计算F1@K指标

        Args:
            recommendations: 推荐商品列表
            ground_truth: 真实相关商品列表
            k: 推荐列表长度

        Returns:
            F1@K值
        """
        precision = self.precision_at_k(recommendations, ground_truth, k)
        recall = self.recall_at_k(recommendations, ground_truth, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def mean_absolute_error(self, predictions: List[float], actual_ratings: List[float]) -> float:
        """
        计算平均绝对误差 (MAE)

        Args:
            predictions: 预测评分列表
            actual_ratings: 实际评分列表

        Returns:
            MAE值
        """
        return mean_absolute_error(actual_ratings, predictions)

    def root_mean_square_error(self, predictions: List[float], actual_ratings: List[float]) -> float:
        """
        计算均方根误差 (RMSE)

        Args:
            predictions: 预测评分列表
            actual_ratings: 实际评分列表

        Returns:
            RMSE值
        """
        return np.sqrt(mean_squared_error(actual_ratings, predictions))

    def coverage(self, all_recommendations: List[List[str]], total_items: List[str]) -> float:
        """
        计算覆盖率指标

        Args:
            all_recommendations: 所有用户的推荐结果列表
            total_items: 总商品列表

        Returns:
            覆盖率值
        """
        # 获取所有被推荐的商品
        recommended_items = set()
        for recommendations in all_recommendations:
            recommended_items.update(recommendations)

        # 计算覆盖率
        coverage = len(recommended_items) / len(total_items)

        return coverage

    def diversity(self, recommendations: List[str], item_similarity_matrix: pd.DataFrame) -> float:
        """
        计算推荐列表的多样性

        Args:
            recommendations: 推荐商品列表
            item_similarity_matrix: 商品相似度矩阵

        Returns:
            多样性值
        """
        if len(recommendations) < 2:
            return 0.0

        # 计算推荐商品之间的平均不相似度
        total_similarity = 0.0
        count = 0

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item1, item2 = recommendations[i], recommendations[j]
                if item1 in item_similarity_matrix.index and item2 in item_similarity_matrix.columns:
                    similarity = item_similarity_matrix.loc[item1, item2]
                    total_similarity += similarity
                    count += 1

        if count == 0:
            return 0.0

        average_similarity = total_similarity / count
        diversity = 1 - average_similarity

        return diversity

    def novelty(self, recommendations: List[str], item_popularity: Dict[str, float]) -> float:
        """
        计算新颖性指标

        Args:
            recommendations: 推荐商品列表
            item_popularity: 商品流行度字典

        Returns:
            新颖性值
        """
        if not recommendations:
            return 0.0

        # 计算推荐商品的平均负对数流行度
        total_novelty = 0.0
        count = 0

        for item in recommendations:
            if item in item_popularity and item_popularity[item] > 0:
                novelty = -np.log2(item_popularity[item])
                total_novelty += novelty
                count += 1

        if count == 0:
            return 0.0

        return total_novelty / count

    def leave_one_out_evaluation(self, recommendation_engine, user_item_matrix: pd.DataFrame,
                               k_values: List[int] = [5, 10, 20], num_users: int = 50) -> Dict[str, Any]:
        """
        留一法评估

        Args:
            recommendation_engine: 推荐引擎实例
            user_item_matrix: 用户-商品交互矩阵
            k_values: 评估的K值列表
            num_users: 参与评估的用户数量

        Returns:
            评估结果字典
        """
        print(f"开始留一法评估，评估用户数: {num_users}")

        # 选择要评估的用户
        users_to_evaluate = user_item_matrix.index[:num_users]
        results = {f'precision@{k}': [] for k in k_values}
        results.update({f'recall@{k}': [] for k in k_values})
        results.update({f'f1@{k}': [] for k in k_values})

        evaluated_users = 0

        for user_id in users_to_evaluate:
            if user_id not in user_item_matrix.index:
                continue

            user_ratings = user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0]

            if len(rated_items) < 2:
                continue

            # 选择一个商品作为测试集（选择评分最高的商品）
            test_item = rated_items.idxmax()
            train_items = rated_items.drop(test_item)

            if len(train_items) == 0:
                continue

            # 构造训练数据
            train_matrix = user_item_matrix.copy()
            train_matrix.loc[user_id, test_item] = 0

            # 重新训练模型（简化版，实际应用中可能需要增量学习）
            # 这里直接使用现有的相似度矩阵进行评估

            # 生成推荐
            try:
                if hasattr(recommendation_engine, 'user_similarity_matrix') and recommendation_engine.user_similarity_matrix is not None:
                    recommendations = recommendation_engine.recommend_user_based_cf(user_id, top_k=max(k_values) * 2)
                    recommended_items = [item[0] for item in recommendations]
                elif hasattr(recommendation_engine, 'item_similarity_matrix') and recommendation_engine.item_similarity_matrix is not None:
                    recommendations = recommendation_engine.recommend_item_based_cf(user_id, top_k=max(k_values) * 2)
                    recommended_items = [item[0] for item in recommendations]
                else:
                    continue

                # 评估不同K值的指标
                ground_truth = [test_item]

                for k in k_values:
                    precision = self.precision_at_k(recommended_items, ground_truth, k)
                    recall = self.recall_at_k(recommended_items, ground_truth, k)
                    f1 = self.f1_score_at_k(recommended_items, ground_truth, k)

                    results[f'precision@{k}'].append(precision)
                    results[f'recall@{k}'].append(recall)
                    results[f'f1@{k}'].append(f1)

                evaluated_users += 1

                # 显示进度
                if evaluated_users % 10 == 0:
                    print(f"已评估 {evaluated_users} 个用户")

            except Exception as e:
                print(f"用户 {user_id} 评估失败: {str(e)}")
                continue

        # 计算平均指标
        final_results = {
            'evaluated_users': evaluated_users,
            'evaluation_method': 'leave_one_out'
        }

        for key, values in results.items():
            if values:
                final_results[key] = np.mean(values)
                final_results[f'{key}_std'] = np.std(values)
            else:
                final_results[key] = 0.0
                final_results[f'{key}_std'] = 0.0

        print(f"留一法评估完成，评估用户数: {evaluated_users}")

        return final_results

    def cross_validation_evaluation(self, recommendation_engine, user_item_matrix: pd.DataFrame,
                                  cv_folds: int = 5, k_values: List[int] = [5, 10]) -> Dict[str, Any]:
        """
        K折交叉验证评估

        Args:
            recommendation_engine: 推荐引擎实例
            user_item_matrix: 用户-商品交互矩阵
            cv_folds: 交叉验证折数
            k_values: 评估的K值列表

        Returns:
            评估结果字典
        """
        print(f"开始K折交叉验证评估，折数: {cv_folds}")

        # 将用户分成K组
        users = user_item_matrix.index.tolist()
        np.random.shuffle(users)

        fold_size = len(users) // cv_folds
        all_results = []

        for fold in range(cv_folds):
            print(f"正在处理第 {fold + 1} 折")

            # 分割训练集和测试集
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else len(users)

            test_users = users[start_idx:end_idx]
            train_users = users[:start_idx] + users[end_idx:]

            # 构建训练矩阵
            train_matrix = user_item_matrix.loc[train_users]

            fold_results = {f'precision@{k}': [] for k in k_values}
            fold_results.update({f'recall@{k}': [] for k in k_values})
            fold_results.update({f'f1@{k}': [] for k in k_values})

            # 在测试集上评估
            for user_id in test_users:
                if user_id not in user_item_matrix.index:
                    continue

                user_ratings = user_item_matrix.loc[user_id]
                rated_items = user_ratings[user_ratings > 0]

                if len(rated_items) < 2:
                    continue

                # 简化的评估：使用训练集的相似度进行推荐
                try:
                    # 这里简化处理，实际应用中需要重新训练模型
                    # 生成推荐
                    if hasattr(recommendation_engine, 'user_similarity_matrix') and recommendation_engine.user_similarity_matrix is not None:
                        recommendations = recommendation_engine.recommend_user_based_cf(user_id, top_k=max(k_values) * 2)
                        recommended_items = [item[0] for item in recommendations]
                    else:
                        continue

                    # 评估（使用用户评分最高的几个商品作为ground truth）
                    ground_truth = rated_items.nlargest(3).index.tolist()

                    for k in k_values:
                        precision = self.precision_at_k(recommended_items, ground_truth, k)
                        recall = self.recall_at_k(recommended_items, ground_truth, k)
                        f1 = self.f1_score_at_k(recommended_items, ground_truth, k)

                        fold_results[f'precision@{k}'].append(precision)
                        fold_results[f'recall@{k}'].append(recall)
                        fold_results[f'f1@{k}'].append(f1)

                except Exception as e:
                    continue

            # 计算fold平均指标
            fold_avg = {}
            for key, values in fold_results.items():
                if values:
                    fold_avg[key] = np.mean(values)
                else:
                    fold_avg[key] = 0.0

            all_results.append(fold_avg)

        # 计算跨fold的平均指标
        final_results = {
            'evaluated_users': len(test_users) * cv_folds,
            'evaluation_method': 'cross_validation',
            'cv_folds': cv_folds
        }

        # 聚合所有fold的结果
        metric_keys = [f'precision@{k}' for k in k_values] + [f'recall@{k}' for k in k_values] + [f'f1@{k}' for k in k_values]

        for key in metric_keys:
            values = [fold.get(key, 0.0) for fold in all_results]
            final_results[key] = np.mean(values)
            final_results[f'{key}_std'] = np.std(values)

        print(f"K折交叉验证评估完成")

        return final_results

    def temporal_evaluation(self, recommendation_engine, user_behavior_data: pd.DataFrame,
                           split_date: str, k_values: List[int] = [5, 10]) -> Dict[str, Any]:
        """
        时间序列评估

        Args:
            recommendation_engine: 推荐引擎实例
            user_behavior_data: 用户行为数据（包含时间戳）
            split_date: 分割日期
            k_values: 评估的K值列表

        Returns:
            评估结果字典
        """
        print(f"开始时间序列评估，分割日期: {split_date}")

        # 转换时间戳
        user_behavior_data['时间戳'] = pd.to_datetime(user_behavior_data['时间戳'])
        split_datetime = pd.to_datetime(split_date)

        # 分割训练集和测试集
        train_data = user_behavior_data[user_behavior_data['时间戳'] < split_datetime]
        test_data = user_behavior_data[user_behavior_data['时间戳'] >= split_datetime]

        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

        # 构建训练矩阵
        train_matrix = train_data.pivot_table(
            index='用户ID', columns='商品ID', values='评分', fill_value=0
        )

        # 评估结果
        results = {f'precision@{k}': [] for k in k_values}
        results.update({f'recall@{k}': [] for k in k_values})
        results.update({f'f1@{k}': [] for k in k_values})

        evaluated_users = 0

        # 对测试集中的用户进行评估
        test_users = test_data['用户ID'].unique()

        for user_id in test_users:
            if user_id not in train_matrix.index:
                continue

            # 获取用户在测试集中的商品
            user_test_items = test_data[test_data['用户ID'] == user_id]['商品ID'].tolist()

            if not user_test_items:
                continue

            try:
                # 生成推荐（基于训练数据）
                # 这里简化处理，实际应用中需要使用训练数据重新训练模型
                if hasattr(recommendation_engine, 'user_similarity_matrix') and recommendation_engine.user_similarity_matrix is not None:
                    recommendations = recommendation_engine.recommend_user_based_cf(user_id, top_k=max(k_values) * 2)
                    recommended_items = [item[0] for item in recommendations]
                else:
                    continue

                # 评估
                ground_truth = user_test_items

                for k in k_values:
                    precision = self.precision_at_k(recommended_items, ground_truth, k)
                    recall = self.recall_at_k(recommended_items, ground_truth, k)
                    f1 = self.f1_score_at_k(recommended_items, ground_truth, k)

                    results[f'precision@{k}'].append(precision)
                    results[f'recall@{k}'].append(recall)
                    results[f'f1@{k}'].append(f1)

                evaluated_users += 1

            except Exception as e:
                continue

        # 计算平均指标
        final_results = {
            'evaluated_users': evaluated_users,
            'evaluation_method': 'temporal',
            'train_size': len(train_data),
            'test_size': len(test_data),
            'split_date': split_date
        }

        for key, values in results.items():
            if values:
                final_results[key] = np.mean(values)
                final_results[f'{key}_std'] = np.std(values)
            else:
                final_results[key] = 0.0
                final_results[f'{key}_std'] = 0.0

        print(f"时间序列评估完成，评估用户数: {evaluated_users}")

        return final_results

    def compare_algorithms(self, algorithm_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        比较不同算法的性能

        Args:
            algorithm_results: 算法结果字典，格式为 {算法名: 评估结果字典}

        Returns:
            算法比较结果DataFrame
        """
        comparison_data = []

        for algorithm_name, results in algorithm_results.items():
            row = {'Algorithm': algorithm_name}

            # 提取关键指标
            for key, value in results.items():
                if isinstance(value, (int, float)) and not key.endswith('_std'):
                    row[key] = value

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # 计算排名
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Algorithm':
                # 假设指标越大越好（如precision, recall, f1）
                comparison_df[f'{col}_rank'] = comparison_df[col].rank(ascending=False)

        self.comparison_results = comparison_df
        return comparison_df

    def generate_evaluation_report(self, results: Dict[str, Any], algorithm_name: str = "Algorithm") -> str:
        """
        生成评估报告

        Args:
            results: 评估结果字典
            algorithm_name: 算法名称

        Returns:
            评估报告文本
        """
        report = f"""
# {algorithm_name} 推荐系统评估报告

## 评估概况
- 评估方法: {results.get('evaluation_method', 'Unknown')}
- 评估用户数: {results.get('evaluated_users', 0)}
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估指标

### 准确率 (Precision)
"""

        # 添加precision指标
        for key, value in results.items():
            if 'precision' in key and not key.endswith('_std'):
                std_key = f'{key}_std'
                std_value = results.get(std_key, 0)
                k_value = key.split('@')[-1] if '@' in key else 'N/A'
                report += f"- Precision@{k_value}: {value:.4f} (±{std_value:.4f})\n"

        report += "\n### 召回率 (Recall)\n"

        # 添加recall指标
        for key, value in results.items():
            if 'recall' in key and not key.endswith('_std'):
                std_key = f'{key}_std'
                std_value = results.get(std_key, 0)
                k_value = key.split('@')[-1] if '@' in key else 'N/A'
                report += f"- Recall@{k_value}: {value:.4f} (±{std_value:.4f})\n"

        report += "\n### F1分数 (F1 Score)\n"

        # 添加F1指标
        for key, value in results.items():
            if 'f1' in key and not key.endswith('_std'):
                std_key = f'{key}_std'
                std_value = results.get(std_key, 0)
                k_value = key.split('@')[-1] if '@' in key else 'N/A'
                report += f"- F1@{k_value}: {value:.4f} (±{std_value:.4f})\n"

        # 添加其他信息
        if 'cv_folds' in results:
            report += f"\n- 交叉验证折数: {results['cv_folds']}\n"

        if 'train_size' in results and 'test_size' in results:
            report += f"\n- 训练集大小: {results['train_size']}\n"
            report += f"- 测试集大小: {results['test_size']}\n"

        report += "\n## 结论\n"

        # 根据指标给出结论
        precision_5 = results.get('precision@5', 0)
        recall_5 = results.get('recall@5', 0)
        f1_5 = results.get('f1@5', 0)

        if precision_5 > 0.1 and recall_5 > 0.1:
            report += "该推荐算法表现良好，具有较高的准确率和召回率。\n"
        elif precision_5 > 0.05:
            report += "该推荐算法具有较好的准确率，但召回率有待提高。\n"
        elif recall_5 > 0.05:
            report += "该推荐算法具有较好的召回率，但准确率有待提高。\n"
        else:
            report += "该推荐算法性能需要进一步优化。\n"

        report += f"\n推荐系统最佳表现: F1@5 = {f1_5:.4f}\n"

        return report

    def save_evaluation_results(self, results: Dict[str, Any], file_path: str, format: str = 'json') -> None:
        """
        保存评估结果

        Args:
            results: 评估结果字典
            file_path: 保存路径
            format: 保存格式 ('json', 'csv', 'excel')
        """
        if format == 'json':
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        elif format == 'csv':
            # 将结果转换为扁平字典
            flat_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_results[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_results[key] = value

            df = pd.DataFrame([flat_results])
            df.to_csv(file_path, index=False, encoding='utf-8')

        elif format == 'excel':
            # 将结果转换为扁平字典
            flat_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_results[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_results[key] = value

            df = pd.DataFrame([flat_results])
            df.to_excel(file_path, index=False)

        print(f"评估结果已保存到: {file_path}")

    def plot_evaluation_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        绘制算法比较图表

        Args:
            comparison_df: 算法比较结果DataFrame
            save_path: 保存路径（可选）
        """
        # 选择要绘制的指标
        metrics = [col for col in comparison_df.columns if any(x in col for x in ['precision', 'recall', 'f1']) and not col.endswith('_rank')]

        if not metrics:
            print("没有可绘制的指标")
            return

        # 创建子图
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # 提取数据
            algorithms = comparison_df['Algorithm'].tolist()
            values = comparison_df[metric].tolist()

            # 绘制柱状图
            bars = ax.bar(algorithms, values, alpha=0.7)
            ax.set_title(f'{metric.replace("@", "@").upper()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图表已保存到: {save_path}")

        plt.show()