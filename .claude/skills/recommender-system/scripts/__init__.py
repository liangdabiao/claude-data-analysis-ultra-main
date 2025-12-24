"""
推荐系统分析技能模块

提供完整的推荐系统实现，包括：
- 协同过滤算法 (用户/物品基于)
- 矩阵分解算法 (SVD)
- 推荐系统评估框架
- 数据分析和可视化
"""

from .recommendation_engine import RecommendationEngine
from .recommender_evaluator import RecommenderEvaluator
from .data_analyzer import DataAnalyzer
from .recommender_visualizer import RecommenderVisualizer

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = [
    "RecommendationEngine",
    "RecommenderEvaluator",
    "DataAnalyzer",
    "RecommenderVisualizer"
]