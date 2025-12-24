"""
推荐系统算法引擎

提供多种推荐算法实现：
- 基于用户的协同过滤 (User-Based Collaborative Filtering)
- 基于物品的协同过滤 (Item-Based Collaborative Filtering)
- SVD矩阵分解推荐 (Singular Value Decomposition)
- 混合推荐策略 (Hybrid Recommendation)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class RecommendationEngine:
    """推荐系统算法引擎类"""

    def __init__(self):
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.is_trained = False

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

            # 构建用户-商品矩阵
            self._build_interaction_matrices()

            # 加载商品信息数据（如果提供）
            item_data = None
            if item_info_path:
                item_data = pd.read_csv(item_info_path, encoding='utf-8')

            print(f"数据加载成功：用户数 {len(self.user_item_matrix.index)}, 商品数 {len(self.user_item_matrix.columns)}")
            return self.user_data, item_data

        except Exception as e:
            print(f"数据加载失败：{str(e)}")
            return None, None

    def _build_interaction_matrices(self) -> None:
        """构建用户-商品交互矩阵和商品-用户交互矩阵"""
        if '评分' in self.user_data.columns:
            # 基于评分构建矩阵
            self.user_item_matrix = self.user_data.pivot_table(
                index='用户ID', columns='商品ID', values='评分',
                fill_value=0, aggfunc='mean'
            )
        else:
            # 基于行为次数构建矩阵
            self.user_item_matrix = self.user_data.pivot_table(
                index='用户ID', columns='商品ID', values='行为类型',
                fill_value=0, aggfunc='count'
            )

        # 构建商品-用户矩阵（用于物品协同过滤）
        self.item_user_matrix = self.user_item_matrix.T

    def train_user_based_cf(self, similarity_metric: str = 'cosine', normalize: bool = True) -> None:
        """
        训练基于用户的协同过滤模型

        Args:
            similarity_metric: 相似度计算方法 ('cosine', 'pearson')
            normalize: 是否对用户评分进行标准化
        """
        print("开始训练基于用户的协同过滤模型...")

        # 数据标准化
        if normalize:
            scaler = StandardScaler()
            matrix_scaled = scaler.fit_transform(self.user_item_matrix)
            user_item_matrix_scaled = pd.DataFrame(
                matrix_scaled,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.columns
            )
        else:
            user_item_matrix_scaled = self.user_item_matrix

        # 计算用户相似度矩阵
        if similarity_metric == 'cosine':
            similarities = cosine_similarity(user_item_matrix_scaled)
        elif similarity_metric == 'pearson':
            # 皮尔逊相关系数
            similarities = np.corrcoef(user_item_matrix_scaled)
        else:
            raise ValueError(f"不支持的相似度计算方法: {similarity_metric}")

        self.user_similarity_matrix = pd.DataFrame(
            similarities,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

        print("基于用户的协同过滤模型训练完成")

    def train_item_based_cf(self, similarity_metric: str = 'cosine') -> None:
        """
        训练基于物品的协同过滤模型

        Args:
            similarity_metric: 相似度计算方法 ('cosine', 'pearson')
        """
        print("开始训练基于物品的协同过滤模型...")

        # 计算物品相似度矩阵
        if similarity_metric == 'cosine':
            similarities = cosine_similarity(self.item_user_matrix)
        elif similarity_metric == 'pearson':
            similarities = np.corrcoef(self.item_user_matrix)
        else:
            raise ValueError(f"不支持的相似度计算方法: {similarity_metric}")

        self.item_similarity_matrix = pd.DataFrame(
            similarities,
            index=self.item_user_matrix.index,
            columns=self.item_user_matrix.index
        )

        print("基于物品的协同过滤模型训练完成")

    def train_svd(self, n_components: int = 50, random_state: int = 42) -> None:
        """
        训练SVD矩阵分解模型

        Args:
            n_components: SVD组件数量
            random_state: 随机种子
        """
        print("开始训练SVD矩阵分解模型...")

        # 创建索引映射
        self.user_to_idx = {user: i for i, user in enumerate(self.user_item_matrix.index)}
        self.idx_to_user = {i: user for user, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(self.user_item_matrix.columns)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        # 训练SVD模型
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T

        print(f"SVD矩阵分解模型训练完成，组件数量: {n_components}")
        print(f"用户因子维度: {self.user_factors.shape}, 商品因子维度: {self.item_factors.shape}")

    def recommend_user_based_cf(self, user_id: str, top_k: int = 5,
                               n_neighbors: int = 50) -> List[Tuple[str, float]]:
        """
        基于用户的协同过滤推荐

        Args:
            user_id: 目标用户ID
            top_k: 推荐商品数量
            n_neighbors: 相似邻居数量

        Returns:
            推荐商品列表，格式为 [(商品ID, 预测评分), ...]
        """
        if self.user_similarity_matrix is None:
            raise ValueError("请先训练基于用户的协同过滤模型")

        if user_id not in self.user_item_matrix.index:
            print(f"警告: 用户 {user_id} 不在训练数据中，返回热门商品推荐")
            return self._recommend_popular_items(top_k)

        # 获取相似用户
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(user_id, errors='ignore')[:n_neighbors]

        # 获取用户已评分的商品
        user_rated_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)

        # 计算推荐分数
        recommendations = {}
        for similar_user, similarity in similar_users.items():
            if similarity <= 0:
                continue

            similar_user_ratings = self.user_item_matrix.loc[similar_user]
            for item_id, rating in similar_user_ratings.items():
                if item_id not in user_rated_items and rating > 0:
                    if item_id not in recommendations:
                        recommendations[item_id] = 0
                    recommendations[item_id] += similarity * rating

        # 归一化推荐分数
        for item_id in recommendations:
            recommendations[item_id] /= len(similar_users)

        # 返回top-k推荐
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_k]

    def recommend_item_based_cf(self, user_id: str, top_k: int = 5,
                               n_neighbors: int = 50) -> List[Tuple[str, float]]:
        """
        基于物品的协同过滤推荐

        Args:
            user_id: 目标用户ID
            top_k: 推荐商品数量
            n_neighbors: 相似邻居数量

        Returns:
            推荐商品列表，格式为 [(商品ID, 预测评分), ...]
        """
        if self.item_similarity_matrix is None:
            raise ValueError("请先训练基于物品的协同过滤模型")

        if user_id not in self.user_item_matrix.index:
            print(f"警告: 用户 {user_id} 不在训练数据中，返回热门商品推荐")
            return self._recommend_popular_items(top_k)

        # 获取用户已评分的商品
        user_ratings = self.user_item_matrix.loc[user_id]
        user_rated_items = user_ratings[user_ratings > 0]

        if len(user_rated_items) == 0:
            print(f"警告: 用户 {user_id} 没有评分记录，返回热门商品推荐")
            return self._recommend_popular_items(top_k)

        # 计算推荐分数
        recommendations = {}
        for rated_item, rating in user_rated_items.items():
            # 获取与该商品相似的商品
            similar_items = self.item_similarity_matrix[rated_item].sort_values(ascending=False)
            similar_items = similar_items.drop(rated_item, errors='ignore')[:n_neighbors]

            for similar_item, similarity in similar_items.items():
                if similarity <= 0 or similar_item in user_rated_items.index:
                    continue

                if similar_item not in recommendations:
                    recommendations[similar_item] = 0
                recommendations[similar_item] += similarity * rating

        # 归一化推荐分数
        for item_id in recommendations:
            recommendations[item_id] /= len(user_rated_items)

        # 返回top-k推荐
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_k]

    def recommend_svd(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        SVD矩阵分解推荐

        Args:
            user_id: 目标用户ID
            top_k: 推荐商品数量

        Returns:
            推荐商品列表，格式为 [(商品ID, 预测评分), ...]
        """
        if self.svd_model is None:
            raise ValueError("请先训练SVD矩阵分解模型")

        if user_id not in self.user_to_idx:
            print(f"警告: 用户 {user_id} 不在训练数据中，返回热门商品推荐")
            return self._recommend_popular_items(top_k)

        user_idx = self.user_to_idx[user_id]

        # 获取用户已评分的商品
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()

        # 计算该用户对所有商品的预测评分
        predicted_ratings = np.dot(self.user_factors[user_idx], self.item_factors.T)

        # 创建预测评分Series
        predictions = pd.Series(predicted_ratings, index=self.user_item_matrix.columns)

        # 过滤掉用户已评分的商品
        recommendations = predictions.drop(rated_items, errors='ignore').sort_values(ascending=False).head(top_k)

        # 返回推荐列表
        return [(item_id, float(score)) for item_id, score in recommendations.items()]

    def recommend_hybrid(self, user_id: str, top_k: int = 5,
                        weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        混合推荐策略

        Args:
            user_id: 目标用户ID
            top_k: 推荐商品数量
            weights: 各算法权重，格式为 {'user_cf': 0.3, 'item_cf': 0.3, 'svd': 0.4}

        Returns:
            推荐商品列表，格式为 [(商品ID, 预测评分), ...]
        """
        # 默认权重
        if weights is None:
            weights = {'user_cf': 0.3, 'item_cf': 0.3, 'svd': 0.4}

        # 检查权重总和
        if abs(sum(weights.values()) - 1.0) > 0.001:
            raise ValueError("权重总和必须为1.0")

        recommendations = {}

        # 基于用户的协同过滤
        if 'user_cf' in weights and self.user_similarity_matrix is not None:
            user_cf_recs = self.recommend_user_based_cf(user_id, top_k * 2)
            for item_id, score in user_cf_recs:
                if item_id not in recommendations:
                    recommendations[item_id] = 0
                recommendations[item_id] += weights['user_cf'] * score

        # 基于物品的协同过滤
        if 'item_cf' in weights and self.item_similarity_matrix is not None:
            item_cf_recs = self.recommend_item_based_cf(user_id, top_k * 2)
            for item_id, score in item_cf_recs:
                if item_id not in recommendations:
                    recommendations[item_id] = 0
                recommendations[item_id] += weights['item_cf'] * score

        # SVD矩阵分解
        if 'svd' in weights and self.svd_model is not None:
            svd_recs = self.recommend_svd(user_id, top_k * 2)
            for item_id, score in svd_recs:
                if item_id not in recommendations:
                    recommendations[item_id] = 0
                recommendations[item_id] += weights['svd'] * score

        # 如果没有可用的算法，返回热门推荐
        if not recommendations:
            return self._recommend_popular_items(top_k)

        # 返回top-k推荐
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_k]

    def _recommend_popular_items(self, top_k: int) -> List[Tuple[str, float]]:
        """
        推荐热门商品（冷启动策略）

        Args:
            top_k: 推荐商品数量

        Returns:
            推荐商品列表，格式为 [(商品ID, 热度分数), ...]
        """
        # 计算商品热度（评分用户数和平均评分的组合）
        item_popularity = {}
        for item_id in self.user_item_matrix.columns:
            ratings = self.user_item_matrix[item_id]
            num_ratings = len(ratings[ratings > 0])
            avg_rating = ratings[ratings > 0].mean() if num_ratings > 0 else 0

            # 热度计算：评分用户数 * 平均评分
            popularity = num_ratings * avg_rating
            item_popularity[item_id] = popularity

        # 返回热门商品
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        获取模型信息

        Returns:
            包含模型基本信息的字典
        """
        info = {
            'user_count': len(self.user_item_matrix.index) if self.user_item_matrix is not None else 0,
            'item_count': len(self.user_item_matrix.columns) if self.user_item_matrix is not None else 0,
            'matrix_sparsity': self._calculate_sparsity() if self.user_item_matrix is not None else 0,
            'user_cf_trained': self.user_similarity_matrix is not None,
            'item_cf_trained': self.item_similarity_matrix is not None,
            'svd_trained': self.svd_model is not None,
            'svd_components': self.svd_model.n_components if self.svd_model else 0
        }
        return info

    def _calculate_sparsity(self) -> float:
        """计算用户-商品矩阵的稀疏度"""
        if self.user_item_matrix is None:
            return 0

        total_entries = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        zero_entries = (self.user_item_matrix == 0).sum().sum()
        sparsity = zero_entries / total_entries

        return float(sparsity)

    def save_model(self, file_path: str) -> None:
        """
        保存训练好的模型

        Args:
            file_path: 模型保存路径
        """
        import pickle

        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'item_user_matrix': self.item_user_matrix,
            'user_similarity_matrix': self.user_similarity_matrix,
            'item_similarity_matrix': self.item_similarity_matrix,
            'svd_model': self.svd_model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        加载预训练模型

        Args:
            file_path: 模型文件路径
        """
        import pickle

        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.user_item_matrix = model_data.get('user_item_matrix')
        self.item_user_matrix = model_data.get('item_user_matrix')
        self.user_similarity_matrix = model_data.get('user_similarity_matrix')
        self.item_similarity_matrix = model_data.get('item_similarity_matrix')
        self.svd_model = model_data.get('svd_model')
        self.user_factors = model_data.get('user_factors')
        self.item_factors = model_data.get('item_factors')
        self.user_to_idx = model_data.get('user_to_idx', {})
        self.idx_to_user = model_data.get('idx_to_user', {})
        self.item_to_idx = model_data.get('item_to_idx', {})
        self.idx_to_item = model_data.get('idx_to_item', {})

        self.is_trained = True
        print(f"模型已从 {file_path} 加载")